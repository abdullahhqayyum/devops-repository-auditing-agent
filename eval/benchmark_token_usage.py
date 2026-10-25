"""Run deterministic token-usage experiments without contacting an LLM provider.

    python eval/benchmark_token_usage.py

The fixture repositories and all intermediate data are created in a temporary
directory and removed automatically. The alternative strategies are simulated
from recorded scripted requests; they do not change production behavior.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from auditor import prompts
from auditor.agent_loop import AuditAgent, AuditConfig
from auditor.llm_client import LLMResponse, ToolCall
from tools import RepoTools, tool_schemas


@dataclass
class BenchmarkResult:
    name: str
    turns: int
    request_chars: int
    inspection_calls: int
    tool_output_chars: int
    findings: int
    completed: bool
    evidence_valid: bool
    risks: str


class ScriptedBenchmarkLLM:
    model = "offline-scripted-model"
    context_chars = 100_000

    def __init__(self, turns: list[list[ToolCall]]) -> None:
        self.turns = turns
        self.requests: list[list[dict[str, Any]]] = []

    def chat(self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        self.requests.append(json.loads(json.dumps(messages)))
        calls = self.turns.pop(0) if self.turns else []
        return LLMResponse("offline scripted reasoning", calls)


def _call(identifier: str, name: str, **arguments: Any) -> ToolCall:
    return ToolCall(identifier, name, arguments)


def _mark_all(start: int) -> list[ToolCall]:
    return [
        _call(
            f"call-{start + index}",
            "mark_reviewed",
            category=category,
            status="no_issues",
            note="offline benchmark coverage",
        )
        for index, category in enumerate(prompts.CATEGORIES)
    ]


def _run(
    repo: Path, turns: list[list[ToolCall]], name: str, cache_inspection_results: bool = True
) -> tuple[BenchmarkResult, ScriptedBenchmarkLLM]:
    llm = ScriptedBenchmarkLLM(turns)
    config_kwargs = {"self_review": False}
    if "cache_inspection_results" in {item.name for item in fields(AuditConfig)}:
        config_kwargs["cache_inspection_results"] = cache_inspection_results
    agent = AuditAgent(llm, RepoTools(repo), str(repo), AuditConfig(**config_kwargs))
    findings = agent.run()
    telemetry = [event for event in agent.trace.events if event["type"] == "llm_telemetry"]
    schema_chars = len(
        json.dumps(tool_schemas(RepoTools(repo)) + prompts.AGENT_TOOL_SCHEMAS, separators=(",", ":"), ensure_ascii=False)
    )
    request_chars = (
        sum(event["approximate_request_chars"] for event in telemetry)
        if telemetry
        else sum(len(prompts.SYSTEM_PROMPT) + schema_chars + _serialized_size(messages) for messages in llm.requests)
    )
    tool_events = [event for event in agent.trace.events if event["type"] == "tool_call"]
    evidence_valid = all(item.get("evidence") and item.get("file") in agent.tools.observed_files for item in findings)
    return (
        BenchmarkResult(
            name,
            len(telemetry) or len(llm.requests),
            request_chars,
            agent.inspection_calls,
            sum(event.get("tool_result_chars", event.get("output_chars", 0)) for event in tool_events),
            len(findings),
            agent.finished,
            evidence_valid,
            "none",
        ),
        llm,
    )


def _serialized_size(messages: list[dict[str, Any]]) -> int:
    return len(json.dumps(messages, separators=(",", ":"), ensure_ascii=False))


def _simulate_compaction(requests: list[list[dict[str, Any]]], repo: Path, keep_recent: int, excerpt: int) -> int:
    total = 0
    schema_chars = len(
        json.dumps(tool_schemas(RepoTools(repo)) + prompts.AGENT_TOOL_SCHEMAS, separators=(",", ":"), ensure_ascii=False)
    )
    for original in requests:
        messages = json.loads(json.dumps(original))
        tool_indexes = [index for index, item in enumerate(messages) if item.get("role") == "tool"]
        for index in tool_indexes[:-keep_recent]:
            content = messages[index]["content"]
            if len(content) > excerpt + 50:
                messages[index]["content"] = "[summary: prior tool output retained] " + content[:excerpt]
        total += len(prompts.SYSTEM_PROMPT) + schema_chars + _serialized_size(messages)
    return total


def _make_turns(defective: bool, batched: bool = False) -> list[list[ToolCall]]:
    first = (
        [_call("1", "list_files"), _call("2", "inspect_workflows"), _call("3", "scan_secrets"), _call("7", "git_log")]
        if batched
        else [_call("1", "list_files")]
    )
    inspected_path = "app/config.py" if defective else "README.md"
    turns = [
        first,
        [_call("4", "read_file", path=inspected_path)],
        [_call("6", "search_code", pattern=".", path=inspected_path)],
    ]
    if not batched:
        turns.insert(3, [_call("7", "git_log")])
    if defective:
        turns.append(
            [
                _call(
                    "5",
                    "record_finding",
                    rule_id="SEC-001",
                    category="security",
                    severity="critical",
                    file="app/config.py",
                    line=5,
                    description="Credential in source",
                    evidence="Observed credential assignment.",
                    recommendation="Use a secret store.",
                )
            ]
        )
    turns.extend([_mark_all(10), [_call("30", "finish_audit", summary="offline benchmark")]])
    return turns


def _duplicate_turns(defective: bool) -> list[list[ToolCall]]:
    turns = _make_turns(defective)
    turns[0].insert(1, _call("duplicate-list", "list_files"))
    return turns


def _print(results: list[BenchmarkResult]) -> None:
    print("strategy                 turns  request_chars  inspections  tool_chars  findings  complete  evidence")
    for result in results:
        print(
            f"{result.name:24} {result.turns:>5} {result.request_chars:>14} {result.inspection_calls:>12} {result.tool_output_chars:>10} "
            f"{result.findings:>9} {str(result.completed):>9} {str(result.evidence_valid):>8}"
        )
    print("\nNo network calls were made; provider token counts are unavailable for this benchmark.")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="devops-audit-token-benchmark-") as directory:
        fixture_root = Path(directory) / "fixtures"
        subprocess.run(
            [sys.executable, "eval/make_demo_repo.py", "--out", str(fixture_root)],
            check=True,
            capture_output=True,
            text=True,
        )
        results: list[BenchmarkResult] = []
        for fixture, defective in (("healthy", False), ("defective", True)):
            repo = fixture_root / fixture
            baseline, baseline_llm = _run(repo, _duplicate_turns(defective), f"{fixture}:main-baseline", False)
            results.append(baseline)
            compact, compact_llm = _run(repo, _make_turns(defective), f"{fixture}:compact")
            compact.request_chars = _simulate_compaction(compact_llm.requests, repo, keep_recent=2, excerpt=240)
            compact.risks = "old evidence is summarized"
            results.append(compact)
            optimized, _ = _run(repo, _duplicate_turns(defective), f"{fixture}:optimized-cache")
            optimized.risks = "only deterministic read-only calls are cacheable"
            results.append(optimized)
            batched, _ = _run(repo, _make_turns(defective, batched=True), f"{fixture}:batched")
            batched.risks = "dependent calls must not be batched"
            results.append(batched)
        _print(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
