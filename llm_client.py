"""Minimal LLM client layer (plain provider SDKs, no agent framework).

Internal message format, shared by all providers:
    {"role": "user", "content": str}
    {"role": "assistant", "content": str, "tool_calls": [{"id", "name", "arguments": dict}]}
    {"role": "tool", "tool_call_id": str, "name": str, "content": str}
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Protocol

log = logging.getLogger("audit.llm")

DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-5-5",
    "openai": "gpt-4.1",
    "gemini": "gemini-3.5-flash",
    "ollama": "qwen3:8b",
}
# Tried in order when the default model has been retired (HTTP 404). Never used for an explicitly chosen model.
FALLBACK_MODELS = {
    "anthropic": ["claude-sonnet-4-5"],
    "openai": ["gpt-4o", "gpt-5-mini"],
    "gemini": ["gemini-flash-latest", "gemini-2.5-flash"],
    "ollama": [],
}
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
# Free-tier Gemini allows roughly 10 requests/minute, so space calls out by default.
DEFAULT_MIN_INTERVAL = {"anthropic": 0.0, "openai": 0.0, "gemini": 6.5, "ollama": 0.0}
OLLAMA_DEFAULT_HOST = "http://localhost:11434"
OLLAMA_DEFAULT_NUM_CTX = 32768
# Conversation size (characters) the agent keeps before eliding old tool output; small local contexts need less.
DEFAULT_CONTEXT_CHARS = 220_000
# Retry policy. The SDKs are built with max_retries=0, so these are the only retries.
RATE_LIMIT_ATTEMPTS = 3  # total attempts for an HTTP 429 that is not known to be quota exhaustion
TRANSIENT_ATTEMPTS = 6  # total attempts for timeouts, connection errors and 5xx
RATE_LIMIT_BASE_DELAY = 5.0
TRANSIENT_BASE_DELAY = 3.0
MAX_BACKOFF = 60.0
MAX_RETRY_AFTER = 60.0  # longest server-requested wait we are willing to sit through


class LLMError(RuntimeError):
    """Unrecoverable LLM failure (auth, quota, repeated network errors)."""


class LLMBudgetExhausted(LLMError):
    """The agent's wall-clock budget ran out while waiting for, or retrying, a model request."""


class _TransientError(RuntimeError):
    """Retryable failure raised by adapters that do not use a provider SDK."""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    parse_error: str | None = None
    raw: dict[str, Any] | None = None  # provider payload echoed back verbatim (e.g. Gemini thought signatures)


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0


class LLM(Protocol):
    model: str
    context_chars: int

    def chat(self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse: ...


@dataclass
class _ErrorInfo:
    """What a provider exception tells us about whether and when to retry."""

    status: int | None
    kind: str  # "quota_exhausted" | "rate_limited" | "transient" | "fatal"
    retry_after: float | None = None  # seconds, from Retry-After, RetryInfo or the message text
    window: str | None = None  # "minute" | "day" | None when the provider did not say
    quota_id: str | None = None
    reason: str = ""  # why "quota_exhausted" was concluded


TRANSIENT_STATUSES = (408, 409, 500, 502, 503, 504, 529)
TRANSIENT_NAMES = ("APIConnectionError", "APITimeoutError", "InternalServerError", "OverloadedError")

# Quota windows are only inferred from a structured quotaId, or from message wording that names the window.
# Generic text such as "check your plan and billing details" appears on per-minute limits too and proves nothing.
_DAY_ID = re.compile(r"per[_ -]?day|daily", re.IGNORECASE)
_MINUTE_ID = re.compile(r"per[_ -]?min", re.IGNORECASE)
_DAY_MESSAGE = re.compile(r"\bper[ -]day\b|\bdaily (?:quota|limit|request)|\(RPD\)|\bRequestsPerDay", re.IGNORECASE)
_MINUTE_MESSAGE = re.compile(r"\bper[ -]min(?:ute)?\b|\(RPM\)|\(TPM\)|\bRequestsPerMinute", re.IGNORECASE)
_RETRY_IN_MESSAGE = re.compile(r"retry in (\d+(?:\.\d+)?)\s*(ms|s)\b", re.IGNORECASE)
_DURATION = re.compile(r"\s*(\d+(?:\.\d+)?)\s*(ms|s)?\s*")


def _error_objects(body: Any) -> list[dict[str, Any]]:
    """Normalise an error body (dict, list of dicts, or missing) to a list of ``error`` dictionaries."""
    items = body if isinstance(body, list) else [body]
    found: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, dict):
            inner = item.get("error", item)
            if isinstance(inner, dict):
                found.append(inner)
    return found


def _parse_duration(value: Any) -> float | None:
    """Parse ``"34s"``, ``"500ms"``, ``"7"`` or a number into seconds; anything else is no hint."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value) if math.isfinite(value) and value >= 0 else None
    match = _DURATION.fullmatch(value) if isinstance(value, str) else None
    if not match:
        return None
    seconds = float(match.group(1))
    return seconds / 1000 if match.group(2) and match.group(2).lower() == "ms" else seconds


def _retry_after_from_headers(headers: Any) -> float | None:
    """Seconds from ``retry-after-ms`` or ``Retry-After`` (delta-seconds or HTTP date); None when absent or invalid."""
    if headers is None:
        return None
    try:
        milliseconds, retry_after = headers.get("retry-after-ms"), headers.get("retry-after")
    except AttributeError:
        return None
    if milliseconds is not None:
        seconds = _parse_duration(str(milliseconds).strip() + "ms")
        if seconds is not None:
            return seconds
    if retry_after is None:
        return None
    seconds = _parse_duration(str(retry_after).strip())
    if seconds is not None:
        return seconds
    try:
        moment = parsedate_to_datetime(str(retry_after))
    except (TypeError, ValueError):
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return max(0.0, (moment - datetime.now(timezone.utc)).total_seconds())


def _quota_violations(errors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    for error in errors:
        for detail in error.get("details") or []:
            if isinstance(detail, dict) and "QuotaFailure" in str(detail.get("@type", "")):
                violations.extend(v for v in detail.get("violations") or [] if isinstance(v, dict))
    return violations


def _retry_info_delay(errors: list[dict[str, Any]]) -> float | None:
    for error in errors:
        for detail in error.get("details") or []:
            if isinstance(detail, dict) and "RetryInfo" in str(detail.get("@type", "")):
                seconds = _parse_duration(detail.get("retryDelay"))
                if seconds is not None:
                    return seconds
    return None


def _classify(exc: Exception) -> _ErrorInfo:
    """Decide how a provider exception should be handled, from structured details first and wording second."""
    status = getattr(exc, "status_code", None)
    if isinstance(exc, _TransientError):
        return _ErrorInfo(status, "transient")
    errors = _error_objects(getattr(exc, "body", None))
    message = " ".join([str(exc), *(str(e.get("message", "")) for e in errors)])
    retry_after = _retry_after_from_headers(getattr(getattr(exc, "response", None), "headers", None))
    if retry_after is None:
        retry_after = _retry_info_delay(errors)
    if retry_after is None and (hint := _RETRY_IN_MESSAGE.search(message)):
        retry_after = _parse_duration(hint.group(1) + hint.group(2))

    if status == 429 or type(exc).__name__ == "RateLimitError":
        quota_ids = [str(v["quotaId"]) for v in _quota_violations(errors) if v.get("quotaId")]
        window: str | None = None
        if any(_DAY_ID.search(q) for q in quota_ids):
            window = "day"
        elif any(_MINUTE_ID.search(q) for q in quota_ids):
            window = "minute"
        elif not quota_ids:
            window = "day" if _DAY_MESSAGE.search(message) else "minute" if _MINUTE_MESSAGE.search(message) else None
        quota_id = next((q for q in quota_ids if _DAY_ID.search(q)), quota_ids[0] if quota_ids else None)
        if any("insufficient_quota" in (str(e.get("code")), str(e.get("type"))) for e in errors):
            return _ErrorInfo(status, "quota_exhausted", retry_after, window, quota_id, "provider reported insufficient_quota")
        if window == "day":
            return _ErrorInfo(status, "quota_exhausted", retry_after, window, quota_id, "a per-day quota window was reported")
        return _ErrorInfo(status, "rate_limited", retry_after, window, quota_id)
    if status in TRANSIENT_STATUSES or type(exc).__name__ in TRANSIENT_NAMES:
        return _ErrorInfo(status, "transient", retry_after)
    return _ErrorInfo(status, "fatal")


def _describe(info: _ErrorInfo) -> str:
    parts = [f"HTTP {info.status}" if info.status else "no HTTP status"]
    if info.kind in ("quota_exhausted", "rate_limited"):
        parts.append(f"{info.window}-window quota" if info.window else "quota window not identified")
        if info.quota_id:
            parts.append(f"quotaId {info.quota_id}")
    return ", ".join(parts)


def _delay(info: _ErrorInfo, attempt: int) -> float:
    """Seconds to wait after failed ``attempt`` (1-based): the server's hint if usable, else jittered exponential backoff."""
    if info.retry_after is not None:
        return max(1.0, info.retry_after) + random.uniform(0, 1.0)
    base = RATE_LIMIT_BASE_DELAY if info.kind == "rate_limited" else TRANSIENT_BASE_DELAY
    return min(MAX_BACKOFF, base * 2 ** (attempt - 1)) + random.uniform(0, 1.5)


def _retrying(call, attempts: int = TRANSIENT_ATTEMPTS, deadline: float | None = None):
    """Run ``call``, retrying failures worth retrying.

    * Rate limits (HTTP 429) get at most RATE_LIMIT_ATTEMPTS attempts; timeouts, connection errors and 5xx get ``attempts``.
    * Explicit quota exhaustion (see _classify) stops at once, since waiting seconds will not clear a daily quota.
    * A server hint (Retry-After, RetryInfo) is honoured up to MAX_RETRY_AFTER; a longer one ends the run.
    * Never sleeps after the final attempt, and never past ``deadline`` (an absolute ``time.monotonic()`` value).
    """
    attempt = 0
    while True:
        attempt += 1
        try:
            return call()
        except LLMError:
            raise
        except Exception as exc:  # noqa: BLE001 - SDK exception types differ per provider
            name = type(exc).__name__
            info = _classify(exc)
            if info.kind == "fatal":
                raise LLMError(f"{name}: {_safe_message(exc)}") from exc
            if info.kind == "quota_exhausted":
                log.error("LLM quota exhausted (%s); not retrying", _describe(info))
                raise LLMError(
                    f"LLM quota exhausted ({_describe(info)}; {info.reason}). Retrying will not help until the quota resets: "
                    f"wait for the reset, use another project or key, or choose another provider or model. "
                    f"Provider said: {_safe_message(exc, 300)}"
                ) from exc
            limit = min(attempts, RATE_LIMIT_ATTEMPTS) if info.kind == "rate_limited" else attempts
            if attempt >= limit:
                log.error("LLM request failed after %d attempt(s) (%s)", attempt, _describe(info))
                raise LLMError(
                    f"LLM request failed after {attempt} attempt(s) ({_describe(info)}): {name}: {_safe_message(exc)}"
                ) from exc
            if info.retry_after is not None and info.retry_after > MAX_RETRY_AFTER:
                raise LLMError(
                    f"LLM asked for a {info.retry_after:.0f}s wait ({_describe(info)}), longer than the {MAX_RETRY_AFTER:.0f}s "
                    f"this agent will wait. Provider said: {_safe_message(exc, 300)}"
                ) from exc
            delay = _delay(info, attempt)
            if deadline is not None and time.monotonic() + delay >= deadline:
                log.error("LLM retry in %.1fs would pass the wall-clock budget; stopping (%s)", delay, _describe(info))
                raise LLMBudgetExhausted(
                    f"wall-clock budget exhausted: a {delay:.0f}s retry delay would pass the deadline ({_describe(info)})"
                ) from exc
            first = attempt == 1
            log.warning(
                "LLM %s (%s), attempt %d/%d; retrying in %.1fs%s",
                info.kind.replace("_", " "),
                _describe(info),
                attempt,
                limit,
                delay,
                f": {_safe_message(exc, 200)}" if first else "",
            )
            time.sleep(delay)


_KEY_ENV_VARS = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY")


def _safe_message(exc: Exception | None, limit: int = 400) -> str:
    """Error text safe to log or show: pattern-redacted, with any configured API key removed, and truncated."""
    from redaction import redact

    if exc is None:
        return ""
    text = str(exc)
    for var in _KEY_ENV_VARS:
        secret = os.environ.get(var, "")
        if len(secret) >= 8:
            text = text.replace(secret, "[REDACTED:api_key]")
    return redact(text)[:limit]


def _is_model_not_found(exc: LLMError) -> bool:
    cause = exc.__cause__
    return getattr(cause, "status_code", None) == 404 or type(cause).__name__ == "NotFoundError"


class _Budgeted:
    """Wall-clock budget shared by the adapters.

    ``deadline`` is an absolute ``time.monotonic()`` value the agent sets before it starts. It bounds retry sleeps, the
    request throttle and each request's timeout. SDK timeouts apply per phase (connect, read, write), not to the whole
    request, so one call can still run somewhat past the deadline when the network stalls.
    """

    deadline: float | None = None
    timeout: float = 120.0

    def _request_timeout(self) -> float:
        if self.deadline is None:
            return self.timeout
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise LLMBudgetExhausted("wall-clock budget exhausted before the model request")
        return max(1.0, min(self.timeout, remaining))


class _ModelFallback(_Budgeted):
    """Switches to the next fallback model when the provider reports the current one does not exist."""

    model: str
    fallbacks: list[str]

    def _with_fallback(self, call):
        while True:
            try:
                return _retrying(call, deadline=self.deadline)
            except LLMError as exc:
                if not (self.fallbacks and _is_model_not_found(exc)):
                    raise
                log.warning("Model %s is unavailable; falling back to %s", self.model, self.fallbacks[0])
                self.model = self.fallbacks.pop(0)


class _Throttle:
    """Keeps a minimum gap between requests (for low per-minute quotas such as free tiers)."""

    def __init__(self, min_interval: float) -> None:
        self.min_interval = max(0.0, min_interval)
        self._last = 0.0

    def wait(self, deadline: float | None = None) -> None:
        gap = time.monotonic() - self._last
        if self._last and gap < self.min_interval:
            pause = self.min_interval - gap
            if deadline is not None and time.monotonic() + pause >= deadline:
                raise LLMBudgetExhausted("wall-clock budget exhausted while waiting between model requests")
            time.sleep(pause)
        self._last = time.monotonic()


class AnthropicLLM(_ModelFallback):
    def __init__(
        self, model: str | None = None, max_tokens: int = 4096, timeout: float = 120.0, min_interval: float = 0.0
    ) -> None:
        import anthropic

        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise LLMError("ANTHROPIC_API_KEY is not set")
        self.client = anthropic.Anthropic(timeout=timeout, max_retries=0)
        self.timeout = timeout
        self.model = model or DEFAULT_MODELS["anthropic"]
        self.fallbacks = [] if model else list(FALLBACK_MODELS["anthropic"])
        self.context_chars = DEFAULT_CONTEXT_CHARS
        self.max_tokens = max_tokens
        self.throttle = _Throttle(min_interval)

    def _convert(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for m in messages:
            if m["role"] == "user":
                out.append({"role": "user", "content": m["content"]})
            elif m["role"] == "assistant":
                blocks: list[dict[str, Any]] = []
                if m.get("content"):
                    blocks.append({"type": "text", "text": m["content"]})
                for tc in m.get("tool_calls", []):
                    blocks.append({"type": "tool_use", "id": tc["id"], "name": tc["name"], "input": tc["arguments"]})
                out.append({"role": "assistant", "content": blocks or [{"type": "text", "text": "(no content)"}]})
            elif m["role"] == "tool":
                block = {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}
                if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                    out[-1]["content"].append(block)
                else:
                    out.append({"role": "user", "content": [block]})
        return out

    def chat(self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        api_tools = [{"name": t["name"], "description": t["description"], "input_schema": t["parameters"]} for t in tools]
        if api_tools:
            api_tools[-1] = {**api_tools[-1], "cache_control": {"type": "ephemeral"}}
        system_blocks = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]

        def call():
            self.throttle.wait(self.deadline)
            return self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system_blocks,
                messages=self._convert(messages),
                tools=api_tools,
                timeout=self._request_timeout(),
            )

        resp = self._with_fallback(call)
        text_parts, calls = [], []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                calls.append(ToolCall(block.id, block.name, dict(block.input or {})))
        usage = getattr(resp, "usage", None)
        return LLMResponse(
            "\n".join(text_parts).strip(),
            calls,
            resp.stop_reason,
            getattr(usage, "input_tokens", 0) or 0,
            getattr(usage, "output_tokens", 0) or 0,
        )


class OpenAILLM(_ModelFallback):
    """OpenAI Chat Completions. Also serves OpenAI-compatible endpoints (Gemini, local servers)."""

    def __init__(
        self,
        model: str | None = None,
        max_tokens: int = 4096,
        timeout: float = 120.0,
        api_key: str | None = None,
        base_url: str | None = None,
        min_interval: float = 0.0,
        provider: str = "openai",
    ) -> None:
        import openai

        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise LLMError("OPENAI_API_KEY is not set")
        kwargs: dict[str, Any] = {"api_key": api_key, "timeout": timeout, "max_retries": 0}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = openai.OpenAI(**kwargs)
        self.timeout = timeout
        self.model = model or DEFAULT_MODELS[provider]
        # A custom OPENAI_BASE_URL serves its own model names, so OpenAI fallbacks would be meaningless there.
        self.fallbacks = [] if model or (base_url and provider == "openai") else list(FALLBACK_MODELS[provider])
        self.context_chars = DEFAULT_CONTEXT_CHARS
        self.max_tokens = max_tokens
        self.throttle = _Throttle(min_interval)

    def _convert(self, system: str, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = [{"role": "system", "content": system}]
        for m in messages:
            if m["role"] == "user":
                out.append({"role": "user", "content": m["content"]})
            elif m["role"] == "assistant":
                msg: dict[str, Any] = {"role": "assistant", "content": m.get("content") or None}
                if m.get("tool_calls"):
                    msg["tool_calls"] = [
                        tc.get("raw")
                        or {
                            "id": tc["id"],
                            "type": "function",
                            "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"])},
                        }
                        for tc in m["tool_calls"]
                    ]
                out.append(msg)
            elif m["role"] == "tool":
                out.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"]})
        return out

    def chat(self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        api_tools = [
            {"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["parameters"]}}
            for t in tools
        ]

        token_param = {"max_tokens": self.max_tokens}

        def call():
            self.throttle.wait(self.deadline)
            timeout = self._request_timeout()
            try:
                return self.client.chat.completions.create(
                    model=self.model,
                    messages=self._convert(system, messages),
                    tools=api_tools,
                    timeout=timeout,
                    **token_param,
                )
            except Exception as exc:  # newer models reject max_tokens
                if "max_completion_tokens" in str(exc) and "max_tokens" in token_param:
                    token_param.clear()
                    token_param["max_completion_tokens"] = self.max_tokens
                    return call()
                raise

        resp = self._with_fallback(call)
        choice = resp.choices[0]
        calls = []
        for tc in choice.message.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
                err = None if isinstance(args, dict) else "arguments must be a JSON object"
                args = args if isinstance(args, dict) else {}
            except json.JSONDecodeError as exc:
                args, err = {}, f"invalid JSON arguments: {exc}"
            try:
                raw = tc.model_dump(exclude_none=True)  # keeps provider extras such as thought signatures
            except AttributeError:
                raw = None
            calls.append(ToolCall(tc.id, tc.function.name, args, err, raw))
        usage = getattr(resp, "usage", None)
        return LLMResponse(
            (choice.message.content or "").strip(),
            calls,
            choice.finish_reason,
            getattr(usage, "prompt_tokens", 0) or 0,
            getattr(usage, "completion_tokens", 0) or 0,
        )


class OllamaLLM(_Budgeted):
    """Local models through Ollama's native /api/chat endpoint.

    The native API is used (not Ollama's OpenAI-compatible one) so the context window can be set per request via
    options.num_ctx; otherwise Ollama may silently truncate the long system prompt to its small default context.
    """

    def __init__(
        self,
        model: str | None = None,
        host: str | None = None,
        num_ctx: int = OLLAMA_DEFAULT_NUM_CTX,
        timeout: float = 900.0,
        max_tokens: int = 4096,
    ) -> None:
        self.model = model or DEFAULT_MODELS["ollama"]
        self.host = (host or OLLAMA_DEFAULT_HOST).rstrip("/")
        if not self.host.startswith(("http://", "https://")):
            self.host = "http://" + self.host
        self.num_ctx = num_ctx
        self.timeout = timeout
        self.max_tokens = max_tokens
        # ~3 characters per token, minus room for the system prompt, tool schemas and the reply.
        self.context_chars = max(20_000, int(num_ctx * 3) - 40_000)
        self._ids = 0

    def _convert(self, system: str, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = [{"role": "system", "content": system}]
        for m in messages:
            if m["role"] == "user":
                out.append({"role": "user", "content": m["content"]})
            elif m["role"] == "assistant":
                msg: dict[str, Any] = {"role": "assistant", "content": m.get("content") or ""}
                if m.get("tool_calls"):
                    msg["tool_calls"] = [
                        {"function": {"name": tc["name"], "arguments": tc["arguments"]}} for tc in m["tool_calls"]
                    ]
                out.append(msg)
            elif m["role"] == "tool":
                out.append({"role": "tool", "content": m["content"], "tool_name": m.get("name", "")})
        return out

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        import urllib.error
        import urllib.request

        req = urllib.request.Request(
            f"{self.host}/api/chat",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        request_timeout = self._request_timeout()
        try:
            with urllib.request.urlopen(req, timeout=request_timeout) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace")[:300]
            if exc.code == 404 and "not found" in body.lower():
                raise LLMError(f"Ollama model '{self.model}' is not installed; run: ollama pull {self.model}") from exc
            if exc.code in (500, 502, 503):
                raise _TransientError(f"Ollama HTTP {exc.code}: {body}") from exc
            raise LLMError(f"Ollama HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise LLMError(f"cannot reach Ollama at {self.host} ({exc.reason}); is `ollama serve` running?") from exc
        except TimeoutError as exc:
            raise LLMError(f"Ollama did not answer within {self.timeout:.0f}s; try a smaller model") from exc

    def chat(self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": self._convert(system, messages),
            "tools": [
                {
                    "type": "function",
                    "function": {"name": t["name"], "description": t["description"], "parameters": t["parameters"]},
                }
                for t in tools
            ],
            "stream": False,
            "options": {"num_ctx": self.num_ctx, "num_predict": self.max_tokens, "temperature": 0.2},
        }
        data = _retrying(lambda: self._post(payload), attempts=3, deadline=self.deadline)
        msg = data.get("message") or {}
        calls = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            args = fn.get("arguments")
            err = None
            if isinstance(args, str):  # some models return a JSON string
                try:
                    args = json.loads(args or "{}")
                except json.JSONDecodeError as exc:
                    args, err = {}, f"invalid JSON arguments: {exc}"
            if not isinstance(args, dict):
                args, err = {}, err or "arguments must be a JSON object"
            self._ids += 1
            calls.append(ToolCall(tc.get("id") or f"ollama_{self._ids}", fn.get("name", ""), args, err))
        text = (msg.get("content") or "").strip()
        if "</think>" in text:  # drop inline reasoning blocks some local models emit
            text = text.split("</think>", 1)[1].strip()
        return LLMResponse(text, calls, data.get("done_reason"), data.get("prompt_eval_count") or 0, data.get("eval_count") or 0)


def build_llm(provider: str | None = None, model: str | None = None) -> LLM:
    provider = (provider or os.environ.get("LLM_PROVIDER") or "").strip().lower()
    model = model or os.environ.get("LLM_MODEL") or None
    if not provider:
        if os.environ.get("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        elif os.environ.get("OPENAI_API_KEY"):
            provider = "openai"
        elif os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
            provider = "gemini"
        else:
            raise LLMError("No LLM credentials found: set ANTHROPIC_API_KEY, OPENAI_API_KEY or GEMINI_API_KEY (see .env.example)")
    if provider not in DEFAULT_MODELS:
        raise LLMError(f"Unsupported LLM_PROVIDER '{provider}' (use anthropic, openai, gemini or ollama)")
    interval_env = os.environ.get("LLM_MIN_SECONDS_BETWEEN_CALLS", "").strip()
    try:
        min_interval = float(interval_env) if interval_env else DEFAULT_MIN_INTERVAL[provider]
    except ValueError as exc:
        raise LLMError("LLM_MIN_SECONDS_BETWEEN_CALLS must be a number") from exc
    if provider == "ollama":
        try:
            num_ctx = int(os.environ.get("OLLAMA_NUM_CTX") or OLLAMA_DEFAULT_NUM_CTX)
        except ValueError as exc:
            raise LLMError("OLLAMA_NUM_CTX must be an integer") from exc
        return OllamaLLM(model, host=os.environ.get("OLLAMA_HOST") or None, num_ctx=num_ctx)
    if provider == "anthropic":
        return AnthropicLLM(model, min_interval=min_interval)
    if provider == "gemini":
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise LLMError("GEMINI_API_KEY is not set")
        return OpenAILLM(model, api_key=key, base_url=GEMINI_BASE_URL, min_interval=min_interval, provider="gemini")
    return OpenAILLM(model, base_url=os.environ.get("OPENAI_BASE_URL") or None, min_interval=min_interval)
