"""Hand-written agent loop: the LLM chooses tools, the controller executes them,
feeds results back, enforces budgets and validates findings."""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, NoReturn

from auditor import prompts
from auditor.llm_client import LLM, LLMBudgetExhausted, LLMError, LLMResponse
from auditor.redaction import redact, redact_data
from auditor.report import FindingRejected, FindingStore
from tools import RepoTools, dispatch, tool_schemas

log = logging.getLogger("audit.agent")


@dataclass
class AuditConfig:
    max_steps: int = 45  # LLM turns spent investigating
    max_tool_calls: int = 120  # total inspection tool executions
    wrapup_steps: int = 4  # extra turns allowed after the budget is exhausted
    max_findings: int = 50
    context_char_budget: int = 220_000
    keep_recent_tool_results: int = 12
    wall_clock_seconds: int = 1500  # exceeding it ends the run as incomplete (LLMBudgetExhausted), with no wrap-up
    self_review: bool = True
    cache_inspection_results: bool = True


@dataclass
class Trace:
    events: list[dict[str, Any]] = field(default_factory=list)
    started: float = field(default_factory=time.time)

    def add(self, kind: str, **data: Any) -> None:
        # Every event passes through redaction here, so no call site can write a secret into the trace.
        self.events.append({"t": round(time.time() - self.started, 2), "type": kind, **redact_data(data)})

    def to_dict(self) -> dict[str, Any]:
        tool_counts = Counter(e["name"] for e in self.events if e["type"] == "tool_call")
        return {
            "duration_seconds": round(time.time() - self.started, 1),
            "tool_call_counts": dict(tool_counts),
            "events": self.events,
        }


class AuditAgent:
    _CACHEABLE_INSPECTION_TOOLS = frozenset(
        {
            "list_files",
            "read_file",
            "search_code",
            "git_log",
            "git_show_commit",
            "inspect_tracked_artifacts",
            "inspect_workflows",
            "scan_secrets",
            "run_ruff",
        }
    )

    def __init__(self, llm: LLM, tools: RepoTools, repo_url: str, config: AuditConfig | None = None) -> None:
        self.llm = llm
        self.tools = tools
        self.repo_url = repo_url
        self.config = config or AuditConfig()
        self.store = FindingStore(tools, max_findings=self.config.max_findings)
        self.trace = Trace()
        self.messages: list[dict[str, Any]] = []
        self.reviewed: dict[str, dict[str, str]] = {}
        self.inspection_calls = 0
        self.finished = False
        self.in_wrapup = False
        self.stop_reason: str | None = None  # set when the run ended early for a budget reason
        self._coverage_nudged = False
        self._self_reviewed = False
        self.tokens = {"input": 0, "output": 0}
        self._inspection_cache: dict[str, str] = {}

    # ---------------------------------------------------------------- overview
    def _overview(self) -> str:
        files = self.tools.tracked_files()
        top = sorted({PurePosixPath(f).parts[0] + ("/" if len(PurePosixPath(f).parts) > 1 else "") for f in files})
        exts = Counter(PurePosixPath(f).suffix or PurePosixPath(f).name for f in files)
        return (
            f"- {len(files)} tracked files\n"
            f"- top-level entries: {', '.join(top[:60])}{' ...' if len(top) > 60 else ''}\n"
            f"- most common extensions: {', '.join(f'{e}({n})' for e, n in exts.most_common(12))}"
        )

    # ---------------------------------------------------------------- main loop
    def run(self) -> list[dict[str, Any]]:
        overview = self._overview()
        self.messages = [{"role": "user", "content": prompts.initial_message(self.repo_url, overview)}]
        self.trace.add("observation", content=overview)

        all_tools = tool_schemas(self.tools) + prompts.AGENT_TOOL_SCHEMAS
        wrapup_tools = [
            t for t in prompts.AGENT_TOOL_SCHEMAS if t["name"] in ("record_finding", "retract_finding", "finish_audit")
        ]
        idle_turns = 0
        step = 0
        wrapup_left = self.config.wrapup_steps
        self._set_llm_deadline(self.config.wall_clock_seconds - (time.time() - self.trace.started))

        while not self.finished:
            if time.time() - self.trace.started > self.config.wall_clock_seconds:
                self._abort(
                    step,
                    "wall_clock_budget_exhausted",
                    LLMBudgetExhausted(
                        f"the {self.config.wall_clock_seconds}s wall-clock budget ran out before the next model turn"
                    ),
                )
            over_budget = step >= self.config.max_steps or self.inspection_calls >= self.config.max_tool_calls
            if over_budget and not self.in_wrapup:
                self.in_wrapup = True
                self.messages.append({"role": "user", "content": prompts.BUDGET_EXHAUSTED})
                self.trace.add("control", message="budget exhausted; entering wrap-up")
            if self.in_wrapup:
                if wrapup_left <= 0:
                    log.warning("Wrap-up turns exhausted; finalizing with recorded findings")
                    break
                wrapup_left -= 1

            step += 1
            compacted = self._trim_context()
            system_chars = len(prompts.SYSTEM_PROMPT)
            schema_chars = len(json.dumps(all_tools, separators=(",", ":"), ensure_ascii=False))
            history_chars = len(json.dumps(self.messages, separators=(",", ":"), ensure_ascii=False))
            request_started = time.monotonic()
            try:
                response = self.llm.chat(prompts.SYSTEM_PROMPT, self.messages, all_tools)
            except LLMBudgetExhausted as exc:
                self._abort(step, "wall_clock_budget_exhausted", exc)
            except LLMError as exc:
                self._abort(step, "llm_error", exc)
            self._account(response)
            self.trace.add(
                "llm_turn",
                step=step,
                reasoning=redact(response.text)[:1500],
                tool_calls=[{"name": c.name, "arguments": c.arguments} for c in response.tool_calls],
            )
            if response.text:
                log.info("[step %d] %s", step, redact(response.text)[:300].replace("\n", " "))

            self.messages.append(
                {
                    "role": "assistant",
                    "content": response.text,
                    "tool_calls": [
                        {"id": c.id, "name": c.name, "arguments": c.arguments, "raw": c.raw} for c in response.tool_calls
                    ],
                }
            )
            tool_result_chars = 0

            if not response.tool_calls:
                self._trace_llm_telemetry(
                    step,
                    response,
                    system_chars,
                    schema_chars,
                    history_chars,
                    request_started,
                    compacted,
                    tool_result_chars,
                )
                idle_turns += 1
                if idle_turns >= 3:
                    log.warning("Model stopped calling tools; finalizing")
                    break
                self.messages.append({"role": "user", "content": prompts.NO_TOOL_NUDGE})
                continue
            idle_turns = 0

            for call in response.tool_calls:
                if self.in_wrapup and call.name not in {t["name"] for t in wrapup_tools}:
                    result = "ERROR: inspection tools are disabled during wrap-up. Record verified findings or finish_audit."
                elif call.parse_error:
                    result = f"ERROR: {call.parse_error}"
                else:
                    result = self._execute(call.name, call.arguments) or "(empty output)"
                tool_result_chars += len(result)
                self.messages.append({"role": "tool", "tool_call_id": call.id, "name": call.name, "content": result})
            self._trace_llm_telemetry(
                step,
                response,
                system_chars,
                schema_chars,
                history_chars,
                request_started,
                compacted,
                tool_result_chars,
            )

        self._trace_finished(step)
        return self.store.as_report_list()

    def _trace_finished(self, step: int) -> None:
        self.trace.add(
            "finished",
            findings=len(self.store.findings),
            retracted=self.store.retracted,
            reviewed=self.reviewed,
            tokens=self.tokens,
            steps=step,
            stop_reason=self.stop_reason,
        )

    def _trace_llm_telemetry(
        self,
        step: int,
        response: LLMResponse,
        system_chars: int,
        schema_chars: int,
        history_chars: int,
        request_started: float,
        compacted: bool,
        tool_result_chars: int,
    ) -> None:
        self.trace.add(
            "llm_telemetry",
            step=step,
            model=response.model or getattr(self.llm, "model", None),
            system_prompt_chars=system_chars,
            tool_schema_chars=schema_chars,
            conversation_history_chars=history_chars,
            approximate_request_chars=system_chars + schema_chars + history_chars,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            tool_call_count=len(response.tool_calls),
            tool_result_chars=tool_result_chars,
            request_duration_seconds=(
                response.request_duration_seconds
                if response.request_duration_seconds is not None
                else round(time.monotonic() - request_started, 3)
            ),
            retry_count=response.retry_count,
            throttle_wait_seconds=response.throttle_wait_seconds,
            context_compacted=compacted,
        )

    def _abort(self, step: int, reason: str, exc: LLMError) -> NoReturn:
        """End the run as incomplete: keep the findings and the trace, then re-raise so the caller exits nonzero."""
        self.stop_reason = reason
        log.error("Investigation incomplete (%s): %s", reason, redact(str(exc))[:300])
        self.trace.add(
            "control",
            message=f"investigation incomplete: {reason}",
            detail=str(exc)[:500],
            elapsed_seconds=round(time.time() - self.trace.started, 1),
            wall_clock_budget_seconds=self.config.wall_clock_seconds,
            findings_kept=len(self.store.findings),
        )
        self._trace_finished(step)
        raise exc

    # ------------------------------------------------------------------ budget
    def _set_llm_deadline(self, seconds: float) -> None:
        """Tell the LLM adapter how long it may keep waiting and retrying (adapters without a deadline are left alone)."""
        if hasattr(self.llm, "deadline"):
            self.llm.deadline = time.monotonic() + max(0.0, seconds)

    # --------------------------------------------------------------- execution
    def _execute(self, name: str, args: dict[str, Any]) -> str:
        started = time.time()
        if name == "record_finding":
            try:
                fid = self.store.add(args)
                result = f"Recorded as {fid}."
            except FindingRejected as exc:
                result = f"REJECTED: {exc}"
            self.trace.add("finding", arguments=args, result=result)
            log.info("record_finding %s %s:%s -> %s", args.get("rule_id"), args.get("file"), args.get("line"), result)
            return result
        if name == "retract_finding":
            try:
                self.store.retract(str(args.get("finding_id", "")), str(args.get("reason", "")))
                result = f"Retracted {args.get('finding_id')}."
            except FindingRejected as exc:
                result = f"ERROR: {exc}"
            self.trace.add("retraction", arguments=args, result=result)
            return result
        if name == "mark_reviewed":
            cat = str(args.get("category", ""))
            if cat not in prompts.CATEGORIES:
                return f"ERROR: unknown category; use one of {', '.join(prompts.CATEGORIES)}"
            self.reviewed[cat] = {"status": str(args.get("status")), "note": redact(str(args.get("note", "")))[:500]}
            self.trace.add("reviewed", category=cat, **self.reviewed[cat])
            remaining = [c for c in prompts.CATEGORIES if c not in self.reviewed]
            return f"Marked {cat} reviewed. Remaining: {', '.join(remaining) or 'none'}."
        if name == "finish_audit":
            return self._handle_finish(args)

        # Inspection tool.
        self.inspection_calls += 1
        cache_key = json.dumps({"name": name, "arguments": args}, sort_keys=True, separators=(",", ":"))
        cached = (
            self._inspection_cache.get(cache_key)
            if self.config.cache_inspection_results and name in self._CACHEABLE_INSPECTION_TOOLS
            else None
        )
        if cached is not None and any(
            message.get("role") == "tool"
            and message.get("content") == cached
            and not str(message.get("content", "")).startswith("[elided")
            for message in self.messages
        ):
            result = "See earlier identical inspection result."
            result = redact(result)
            self.trace.add(
                "tool_call",
                name=name,
                arguments=args,
                duration=round(time.time() - started, 2),
                output_chars=len(result),
                output_preview=result[:400],
                cache_hit=True,
            )
            return result
        result = dispatch(self.tools, name, args)
        result = redact(result)
        if self.config.cache_inspection_results and name in self._CACHEABLE_INSPECTION_TOOLS:
            self._inspection_cache[cache_key] = result
        self.trace.add(
            "tool_call",
            name=name,
            arguments=args,
            duration=round(time.time() - started, 2),
            output_chars=len(result),
            output_preview=result[:400],
            cache_hit=False,
        )
        log.info("tool %s(%s) -> %d chars", name, json.dumps(args)[:120], len(result))
        return result

    def _handle_finish(self, args: dict[str, Any]) -> str:
        if self.in_wrapup:
            self.finished = True
            self.trace.add("finish", summary=redact(str(args.get("summary", "")))[:2000])
            return "Audit finished."
        missing = [c for c in prompts.CATEGORIES if c not in self.reviewed]
        if missing and not self._coverage_nudged:
            self._coverage_nudged = True
            self.trace.add("control", message="coverage nudge", missing=missing)
            return prompts.coverage_nudge(missing)
        if self.config.self_review and not self._self_reviewed and self.store.findings:
            self._self_reviewed = True
            self.trace.add("control", message="self-review requested")
            return prompts.review_nudge(self.store.summary_text())
        self.finished = True
        self.trace.add("finish", summary=redact(str(args.get("summary", "")))[:2000])
        return "Audit finished."

    # ----------------------------------------------------------------- context
    def _account(self, response: LLMResponse) -> None:
        if response.input_tokens is not None:
            self.tokens["input"] += response.input_tokens
        if response.output_tokens is not None:
            self.tokens["output"] += response.output_tokens

    def _trim_context(self) -> bool:
        """Elide old tool outputs when the conversation grows too large (pairing stays intact)."""
        size = sum(len(m.get("content") or "") for m in self.messages)
        if size <= self.config.context_char_budget:
            return False
        compacted = False
        tool_idx = [i for i, m in enumerate(self.messages) if m["role"] == "tool"]
        for i in tool_idx[: max(0, len(tool_idx) - self.config.keep_recent_tool_results)]:
            content = self.messages[i]["content"]
            if len(content) > 300 and not content.startswith("[elided"):
                self.messages[i]["content"] = "[elided to save context; first 200 chars] " + content[:200]
                compacted = True
                size -= len(content) - 250
                if size <= self.config.context_char_budget * 0.7:
                    break
        self.trace.add("control", message="context trimmed", approx_chars=size)
        return compacted
