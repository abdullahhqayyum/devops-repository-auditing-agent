"""OpenRouter client through the OpenAI-compatible SDK, without an agent framework.

Internal message format, shared with the hand-written agent loop:
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
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Protocol

log = logging.getLogger("audit.llm")

# OpenRouter's free router filters for the request's capabilities, including tools. It can select a different model
# each turn; users may pin a tool-capable slug through --model or LLM_MODEL. No fallback can select a paid model.
DEFAULT_MODEL = "openrouter/free"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# The free tier is documented at 20 requests/minute; 3.5 s stays below that account-wide limit.
DEFAULT_MIN_INTERVAL = 3.5
# OpenRouter serves models with different context windows, so trim history conservatively.
CONTEXT_CHARS = 100_000
# The SDK is built with max_retries=0, so these are the only retries.
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
    """Retryable failure when a completion response contains no choices."""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    parse_error: str | None = None
    raw: dict[str, Any] | None = None  # preserve any routed model extras when echoing tool calls


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    request_duration_seconds: float | None = None
    retry_count: int | None = None
    throttle_wait_seconds: float | None = None
    model: str | None = None


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
TRANSIENT_NAMES = ("APIConnectionError", "APITimeoutError", "InternalServerError")

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
    """Decide how an OpenRouter exception should be handled, from structured details first and wording second."""
    status = getattr(exc, "status_code", None)
    if isinstance(exc, _TransientError) and status is None:
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


class _EmptyCompletion(_TransientError):
    """A provider answered without any completion. OpenRouter can return an error object inside a 200 response; its
    code and body are exposed like an HTTP error so the usual classification (429 quota, 402 credits, 5xx) applies."""

    def __init__(self, resp: Any) -> None:
        error = getattr(resp, "error", None)
        error = error if isinstance(error, dict) else {}
        super().__init__(str(error.get("message") or "the provider returned no completion"))
        code = error.get("code")
        self.status_code = int(code) if str(code).isdigit() else None
        self.body = {"error": error} if error else None


def _fatal_hint(info: _ErrorInfo, exc: Exception) -> str:
    """Actionable advice for errors that are never retried: credentials, credits, unknown model, missing tool support."""
    text = " ".join([str(exc), *(str(e.get("message", "")) for e in _error_objects(getattr(exc, "body", None)))]).lower()
    if info.status in (400, 404, 422) and "tool" in text and any(w in text for w in ("support", "endpoint", "unsupported")):
        return (
            " [hint: this model does not appear to support tool calling, which the agent requires; choose a tool-capable model]"
        )
    hints = {
        401: " [hint: authentication failed; check OPENROUTER_API_KEY]",
        402: " [hint: payment required or insufficient credits, so the request is not retried; add credits or choose a free model]",
        403: " [hint: access denied; the key or project may lack permission for this model]",
        404: " [hint: model not found; check the model name or slug]",
    }
    return hints.get(info.status, "")


def _reported_reset(exc: Exception) -> str:
    """When the provider says a rate limit resets (OpenRouter reports ``X-RateLimit-Reset`` in the error metadata)."""
    for error in _error_objects(getattr(exc, "body", None)):
        metadata = error.get("metadata")
        headers = metadata.get("headers") if isinstance(metadata, dict) else None
        if not isinstance(headers, dict):
            continue
        for key, value in headers.items():
            if str(key).lower() != "x-ratelimit-reset":
                continue
            try:
                stamp = float(value)
                moment = datetime.fromtimestamp(stamp / 1000 if stamp > 1e11 else stamp, timezone.utc)
            except (TypeError, ValueError, OverflowError, OSError):
                continue
            now = datetime.now(timezone.utc)
            if now <= moment <= now + timedelta(days=2):
                return f" The provider reports the limit resets around {moment:%Y-%m-%d %H:%M} UTC."
    return ""


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


def _retrying(call, attempts: int = TRANSIENT_ATTEMPTS, deadline: float | None = None, stats: dict[str, Any] | None = None):
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
        except Exception as exc:  # noqa: BLE001 - OpenRouter can relay different upstream error shapes
            name = type(exc).__name__
            info = _classify(exc)
            if info.kind == "fatal":
                raise LLMError(f"{name}: {_safe_message(exc)}{_fatal_hint(info, exc)}") from exc
            if info.kind == "quota_exhausted":
                log.error("LLM quota exhausted (%s); not retrying", _describe(info))
                raise LLMError(
                    f"LLM quota exhausted ({_describe(info)}; {info.reason}).{_reported_reset(exc)} Retrying will not help until the quota resets: "
                    f"wait for the reset or use an OpenRouter key with available allowance. "
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
            if stats is not None:
                stats["count"] = stats.get("count", 0) + 1
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


# Also scrub any old provider keys inherited from the shell if an error happens to echo one. These variables never
# select a provider; auditor.redaction separately detects credentials inside repositories being audited.
_KEY_ENV_VARS = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENROUTER_API_KEY")


def _safe_message(exc: Exception | None, limit: int = 400) -> str:
    """Error text safe to log or show: pattern-redacted, with any configured API key removed, and truncated."""
    from auditor.redaction import redact

    if exc is None:
        return ""
    text = str(exc)
    for var in _KEY_ENV_VARS:
        secret = os.environ.get(var, "")
        if len(secret) >= 8:
            text = text.replace(secret, "[REDACTED:api_key]")
    return redact(text)[:limit]


class _Budgeted:
    """Wall-clock budget for model requests.

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


class _Throttle:
    """Keeps a minimum gap between requests (for low per-minute quotas such as free tiers)."""

    def __init__(self, min_interval: float) -> None:
        self.min_interval = max(0.0, min_interval)
        self._last = 0.0

    def wait(self, deadline: float | None = None) -> float:
        gap = time.monotonic() - self._last
        waited = 0.0
        if self._last and gap < self.min_interval:
            pause = self.min_interval - gap
            if deadline is not None and time.monotonic() + pause >= deadline:
                raise LLMBudgetExhausted("wall-clock budget exhausted while waiting between model requests")
            time.sleep(pause)
            waited = pause
        self._last = time.monotonic()
        return waited


class OpenRouterLLM(_Budgeted):
    """OpenRouter Chat Completions through the OpenAI-compatible SDK."""

    def __init__(
        self,
        model: str | None = None,
        max_tokens: int = 4096,
        timeout: float = 120.0,
        min_interval: float = DEFAULT_MIN_INTERVAL,
    ) -> None:
        import openai

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise LLMError("OPENROUTER_API_KEY is not set")
        self.client = openai.OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL, timeout=timeout, max_retries=0)
        self.timeout = timeout
        self.model = model or DEFAULT_MODEL
        self.context_chars = CONTEXT_CHARS
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

        retry_stats = {"count": 0, "throttle_wait": 0.0}

        def call():
            retry_stats["throttle_wait"] += self.throttle.wait(self.deadline)
            timeout = self._request_timeout()
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=self._convert(system, messages),
                    tools=api_tools,
                    timeout=timeout,
                    **token_param,
                )
                if not getattr(resp, "choices", None) or not getattr(resp.choices[0], "message", None):
                    raise _EmptyCompletion(resp)
                return resp
            except Exception as exc:  # newer models reject max_tokens
                if "max_completion_tokens" in str(exc) and "max_tokens" in token_param:
                    token_param.clear()
                    token_param["max_completion_tokens"] = self.max_tokens
                    return call()
                raise

        started = time.monotonic()
        resp = _retrying(call, deadline=self.deadline, stats=retry_stats)
        choice = resp.choices[0]
        calls = []
        for tc in choice.message.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
                err = None if isinstance(args, dict) else "arguments must be a JSON object"
                args = args if isinstance(args, dict) else {}
            except (json.JSONDecodeError, TypeError) as exc:
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
            getattr(usage, "prompt_tokens", None),
            getattr(usage, "completion_tokens", None),
            request_duration_seconds=round(time.monotonic() - started, 3),
            retry_count=retry_stats["count"],
            throttle_wait_seconds=round(retry_stats["throttle_wait"], 3),
            model=self.model,
        )


def build_llm(provider: str | None = None, model: str | None = None) -> LLM:
    """Build the only supported provider. ``provider`` remains for CLI compatibility."""
    if provider is not None and provider.strip().lower() != "openrouter":
        raise LLMError(f"Unsupported provider '{provider}' (use openrouter)")
    # An obsolete LLM_PROVIDER from an older shell cannot redirect this client. If it also carried a provider-specific
    # LLM_MODEL, ignore that inherited value; an explicit --model still wins.
    legacy_provider = os.environ.get("LLM_PROVIDER", "").strip().lower()
    env_model = os.environ.get("LLM_MODEL") if legacy_provider in ("", "openrouter") else None
    selected_model = model or env_model or DEFAULT_MODEL
    interval_env = os.environ.get("LLM_MIN_SECONDS_BETWEEN_CALLS", "").strip()
    try:
        min_interval = float(interval_env) if interval_env else DEFAULT_MIN_INTERVAL
    except ValueError as exc:
        raise LLMError("LLM_MIN_SECONDS_BETWEEN_CALLS must be a number") from exc
    return OpenRouterLLM(selected_model, min_interval=min_interval)
