"""Minimal LLM client layer (plain provider SDKs, no agent framework).

Internal message format, shared by all providers:
    {"role": "user", "content": str}
    {"role": "assistant", "content": str, "tool_calls": [{"id", "name", "arguments": dict}]}
    {"role": "tool", "tool_call_id": str, "name": str, "content": str}
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from dataclasses import dataclass, field
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


class LLMError(RuntimeError):
    """Unrecoverable LLM failure (auth, quota, repeated network errors)."""


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


def _retrying(call, attempts: int = 6):
    """Retry transient failures with jittered exponential backoff."""
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return call()
        except Exception as exc:  # noqa: BLE001 - SDK exception types differ per provider
            status = getattr(exc, "status_code", None)
            name = type(exc).__name__
            if isinstance(exc, LLMError):
                raise
            transient = (
                isinstance(exc, _TransientError)
                or status in (408, 409, 429, 500, 502, 503, 504, 529)
                or name
                in (
                    "APIConnectionError",
                    "APITimeoutError",
                    "RateLimitError",
                    "InternalServerError",
                    "OverloadedError",
                )
            )
            if not transient:
                raise LLMError(f"{name}: {_safe_message(exc)}") from exc
            last = exc
            delay = min(60, 2**attempt * 3) + random.uniform(0, 1.5)
            log.warning("LLM transient error (%s); retrying in %.1fs", name, delay)
            time.sleep(delay)
    raise LLMError(f"LLM request failed after {attempts} attempts: {type(last).__name__}: {_safe_message(last)}")


def _safe_message(exc: Exception | None) -> str:
    from redaction import redact

    return redact(str(exc))[:400] if exc else ""


def _is_model_not_found(exc: LLMError) -> bool:
    cause = exc.__cause__
    return getattr(cause, "status_code", None) == 404 or type(cause).__name__ == "NotFoundError"


class _ModelFallback:
    """Switches to the next fallback model when the provider reports the current one does not exist."""

    model: str
    fallbacks: list[str]

    def _with_fallback(self, call):
        while True:
            try:
                return _retrying(call)
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

    def wait(self) -> None:
        gap = time.monotonic() - self._last
        if self._last and gap < self.min_interval:
            time.sleep(self.min_interval - gap)
        self._last = time.monotonic()


class AnthropicLLM(_ModelFallback):
    def __init__(
        self, model: str | None = None, max_tokens: int = 4096, timeout: float = 120.0, min_interval: float = 0.0
    ) -> None:
        import anthropic

        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise LLMError("ANTHROPIC_API_KEY is not set")
        self.client = anthropic.Anthropic(timeout=timeout, max_retries=2)
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
            self.throttle.wait()
            return self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system_blocks,
                messages=self._convert(messages),
                tools=api_tools,
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
        kwargs: dict[str, Any] = {"api_key": api_key, "timeout": timeout, "max_retries": 2}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = openai.OpenAI(**kwargs)
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
            self.throttle.wait()
            try:
                return self.client.chat.completions.create(
                    model=self.model,
                    messages=self._convert(system, messages),
                    tools=api_tools,
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


class OllamaLLM:
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
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
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
        data = _retrying(lambda: self._post(payload), attempts=3)
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
