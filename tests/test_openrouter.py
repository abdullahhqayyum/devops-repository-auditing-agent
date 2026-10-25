"""OpenRouter configuration, request/response compatibility and errors. Everything is mocked."""

import json
import logging
import time as real_time
from types import SimpleNamespace

import pytest

import agent as cli
import auditor.llm_client as llm_client
from auditor.llm_client import (
    DEFAULT_MODEL,
    OPENROUTER_BASE_URL,
    LLMError,
    OpenRouterLLM,
    _retrying,
    _safe_message,
    _Throttle,
    build_llm,
)
from auditor.redaction import redact

OR_KEY = "sk-or-v1-" + "a1b2c3d4" * 8  # synthetic, shaped like an OpenRouter key
SLUG = "nvidia/nemotron-3-super-120b-a12b:free"
KEY_VARS = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "OPENROUTER_API_KEY",
    "LLM_PROVIDER",
    "LLM_MODEL",
    "OPENAI_BASE_URL",
    "LLM_MIN_SECONDS_BETWEEN_CALLS",
)


@pytest.fixture
def env(monkeypatch):
    for var in KEY_VARS:
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


class Clock:
    """Stands in for ``time`` inside llm_client: sleeping only advances the clock."""

    def __init__(self):
        self.now = 1000.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    fake = Clock()
    monkeypatch.setattr(llm_client, "time", fake)
    monkeypatch.setattr(llm_client.random, "uniform", lambda _a, _b: 0.0)
    return fake


# ---------------------------------------------------------------- fake OpenAI-compatible client
class Completions:
    """Returns (or raises) the given results in order; the last one repeats. Records every request."""

    def __init__(self, *results):
        self.results = list(results)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        result = self.results.pop(0) if len(self.results) > 1 else self.results[0]
        if isinstance(result, Exception):
            raise result
        return result


def tool_call(call_id, name, arguments):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
        model_dump=lambda exclude_none=True: {
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        },
    )


def completion(text="", calls=(), finish="stop"):
    message = SimpleNamespace(content=text, tool_calls=list(calls) or None)
    usage = SimpleNamespace(prompt_tokens=120, completion_tokens=15)
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=finish)], usage=usage)


def make_llm(env, completions, model=None):
    env.setenv("OPENROUTER_API_KEY", OR_KEY)
    llm = build_llm("openrouter", model)
    llm.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    llm.throttle = _Throttle(0)
    return llm


def http_error(status, message="boom", body=None, headers=None):
    exc = Exception(message)
    exc.status_code = status
    exc.body = body
    exc.response = SimpleNamespace(headers=headers or {})
    return exc


USER = [{"role": "user", "content": "audit"}]
TOOLS = [
    {"name": "list_files", "description": "List files", "parameters": {"type": "object", "properties": {}}},
    {
        "name": "read_file",
        "description": "Read a file",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    },
]


# ----------------------------------------------------------------- selection, credentials, endpoint
def test_openrouter_is_the_only_provider(env):
    env.setenv("OPENROUTER_API_KEY", OR_KEY)
    llm = build_llm("openrouter")
    assert isinstance(llm, OpenRouterLLM)
    assert str(llm.client.base_url).rstrip("/") == OPENROUTER_BASE_URL == "https://openrouter.ai/api/v1"
    assert llm.model == DEFAULT_MODEL == "openrouter/free"
    assert llm.client.max_retries == 0  # the application's retry policy is the only retry layer
    assert llm.throttle.min_interval == 3.5 and llm.context_chars == 100_000
    assert not hasattr(llm, "fallbacks")  # never silently swap to another (possibly paid) model


def test_cli_defaults_to_openrouter_and_accepts_legacy_flag(env):
    env.setenv("OPENROUTER_API_KEY", OR_KEY)
    env.setenv("LLM_PROVIDER", "gemini")
    assert build_llm().client.base_url.host == "openrouter.ai"
    assert cli.parse_args(["--repo", "https://github.com/a/b", "--output", "r.json"]).provider == "openrouter"
    args = cli.parse_args(["--repo", "https://github.com/a/b", "--output", "r.json", "--provider", "openrouter"])
    assert args.provider == "openrouter"
    with pytest.raises(SystemExit):
        cli.parse_args(["--repo", "x", "--output", "y", "--provider", "not-a-provider"])


def test_openrouter_reads_its_own_key_and_never_the_openai_one(env):
    env.setenv("OPENAI_API_KEY", "sk-this-is-an-openai-key-0123456789")
    with pytest.raises(LLMError, match="OPENROUTER_API_KEY is not set"):
        build_llm("openrouter")
    env.setenv("OPENROUTER_API_KEY", OR_KEY)
    assert build_llm("openrouter").client.api_key == OR_KEY


def test_missing_key_errors_name_the_right_variable(env):
    with pytest.raises(LLMError, match="OPENROUTER_API_KEY is not set"):
        build_llm("openrouter")
    with pytest.raises(LLMError, match="OPENROUTER_API_KEY"):  # also listed when nothing at all is configured
        build_llm()
    with pytest.raises(LLMError, match="openrouter"):
        build_llm("mistral")  # the unsupported-provider message lists the choices


def test_other_provider_keys_do_not_change_selection(env):
    env.setenv("OPENROUTER_API_KEY", OR_KEY)
    env.setenv("GEMINI_API_KEY", "g-key")
    env.setenv("OPENAI_API_KEY", "o-key")
    env.setenv("ANTHROPIC_API_KEY", "a-key")
    env.setenv("LLM_PROVIDER", "anthropic")
    assert isinstance(build_llm(), OpenRouterLLM)
    assert build_llm().model == DEFAULT_MODEL


@pytest.mark.parametrize("slug", [SLUG, "openrouter/free", "google/gemma-4-31b-it:free", "openai/gpt-4.1"])
def test_model_slugs_are_passed_through_verbatim_and_pinned(env, slug):
    completions = Completions(completion("ok"))
    llm = make_llm(env, completions, model=slug)
    assert llm.model == slug
    llm.chat("sys", USER, TOOLS)
    assert completions.requests[0]["model"] == slug


def test_model_slug_can_come_from_llm_model(env):
    env.setenv("OPENROUTER_API_KEY", OR_KEY)
    env.setenv("LLM_MODEL", SLUG)
    assert build_llm("openrouter").model == SLUG


# ----------------------------------------------------------- tool-call request and response format
def test_request_carries_tools_in_the_openai_function_format(env):
    completions = Completions(completion("ok"))
    make_llm(env, completions).chat("system prompt", USER, TOOLS)
    sent = completions.requests[0]
    assert sent["tools"][0] == {
        "type": "function",
        "function": {"name": "list_files", "description": "List files", "parameters": {"type": "object", "properties": {}}},
    }
    assert sent["tools"][1]["function"]["parameters"]["required"] == ["path"]
    assert sent["messages"][0] == {"role": "system", "content": "system prompt"}
    assert sent["max_tokens"] == 4096 and sent["timeout"] == 120.0


def test_multiple_tool_calls_in_one_turn_are_parsed_in_order_with_their_ids(env):
    reply = completion(
        "Checking two files.",
        [tool_call("call_a", "read_file", {"path": "README.md"}), tool_call("call_b", "read_file", {"path": "app.py"})],
        finish="tool_calls",
    )
    response = make_llm(env, Completions(reply)).chat("s", USER, TOOLS)
    assert [(c.id, c.name, c.arguments) for c in response.tool_calls] == [
        ("call_a", "read_file", {"path": "README.md"}),
        ("call_b", "read_file", {"path": "app.py"}),
    ]
    assert response.text == "Checking two files." and response.stop_reason == "tool_calls"
    assert (response.input_tokens, response.output_tokens) == (120, 15)


def test_missing_usage_metadata_is_null(env):
    message = SimpleNamespace(content="ok", tool_calls=None)
    response = SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")], usage=None)
    result = make_llm(env, Completions(response)).chat("s", USER, TOOLS)
    assert result.input_tokens is None and result.output_tokens is None


def test_request_telemetry_reports_retries_throttle_and_model(env, clock):
    completions = Completions(http_error(503), completion("done"))
    llm = make_llm(env, completions)
    llm.throttle = _Throttle(10)
    response = llm.chat("s", USER, TOOLS)
    assert response.model == DEFAULT_MODEL
    assert response.retry_count == 1
    assert response.throttle_wait_seconds == 7
    assert response.request_duration_seconds is not None


def test_follow_up_turn_returns_every_tool_result_under_the_matching_call_id(env):
    first = completion("", [tool_call("call_a", "list_files", {}), tool_call("call_b", "read_file", {"path": "README.md"})])
    completions = Completions(first, completion("done"))
    llm = make_llm(env, completions)
    turn = llm.chat("s", USER, TOOLS)
    history = [
        *USER,
        {
            "role": "assistant",
            "content": turn.text,
            "tool_calls": [{"id": c.id, "name": c.name, "arguments": c.arguments, "raw": c.raw} for c in turn.tool_calls],
        },
        {"role": "tool", "tool_call_id": "call_a", "name": "list_files", "content": "README.md"},
        {"role": "tool", "tool_call_id": "call_b", "name": "read_file", "content": "1| # demo"},
    ]
    llm.chat("s", history, TOOLS)
    sent = completions.requests[1]["messages"]
    assistant = next(m for m in sent if m["role"] == "assistant")
    assert [tc["id"] for tc in assistant["tool_calls"]] == ["call_a", "call_b"]
    assert [(m["tool_call_id"], m["content"]) for m in sent if m["role"] == "tool"] == [
        ("call_a", "README.md"),
        ("call_b", "1| # demo"),
    ]


def test_malformed_tool_arguments_are_reported_not_fatal(env):
    bad = SimpleNamespace(
        id="call_x",
        function=SimpleNamespace(name="read_file", arguments="{not json"),
        model_dump=lambda exclude_none=True: {},
    )
    response = make_llm(env, Completions(completion("", [bad]))).chat("s", USER, TOOLS)
    assert response.tool_calls[0].parse_error and "invalid JSON" in response.tool_calls[0].parse_error


def test_non_string_tool_arguments_are_reported_not_fatal(env):
    bad = SimpleNamespace(
        id="call_x",
        function=SimpleNamespace(name="read_file", arguments=123),
        model_dump=lambda exclude_none=True: {},
    )
    response = make_llm(env, Completions(completion("", [bad]))).chat("s", USER, TOOLS)
    assert response.tool_calls[0].arguments == {}
    assert "invalid JSON arguments" in response.tool_calls[0].parse_error


# --------------------------------------------------------------------------- errors and quotas
def test_insufficient_credits_402_is_not_retried_and_explains_what_to_do(env, clock):
    body = {"error": {"code": 402, "message": "Insufficient credits. Add more using https://openrouter.ai/credits"}}
    completions = Completions(http_error(402, "Error code: 402 - Insufficient credits", body))
    with pytest.raises(LLMError, match="insufficient credits") as info:
        make_llm(env, completions).chat("s", USER, TOOLS)
    assert len(completions.requests) == 1 and clock.sleeps == []
    assert "add credits or choose a free model" in str(info.value)


def test_authentication_failure_401_is_not_retried(env, clock):
    completions = Completions(http_error(401, "Error code: 401 - No auth credentials found"))
    with pytest.raises(LLMError, match="authentication failed; check OPENROUTER_API_KEY"):
        make_llm(env, completions).chat("s", USER, TOOLS)
    assert len(completions.requests) == 1 and clock.sleeps == []


def test_a_model_without_tool_support_is_reported_clearly(env, clock):
    message = "No endpoints found that support tool use. To learn more about provider routing, visit: https://openrouter.ai/docs"
    completions = Completions(http_error(404, f"Error code: 404 - {message}", {"error": {"code": 404, "message": message}}))
    with pytest.raises(LLMError, match="does not appear to support tool calling"):
        make_llm(env, completions, model="some/model-without-tools:free").chat("s", USER, TOOLS)
    assert len(completions.requests) == 1


def test_an_unknown_model_slug_is_reported_as_not_found_without_swapping(env, clock):
    completions = Completions(http_error(404, "Error code: 404 - No endpoints found for nope/nope"))
    llm = make_llm(env, completions, model="nope/nope")
    with pytest.raises(LLMError, match="model not found; check the model name or slug"):
        llm.chat("s", USER, TOOLS)
    assert llm.model == "nope/nope" and len(completions.requests) == 1


def daily_limit_error(reset_in_seconds=3 * 3600):
    reset_ms = int((real_time.time() + reset_in_seconds) * 1000)
    message = "Rate limit exceeded: free-models-per-day. Add 10 credits to unlock 1000 free model requests per day"
    body = {
        "error": {
            "code": 429,
            "message": message,
            "metadata": {"headers": {"X-RateLimit-Limit": "50", "X-RateLimit-Reset": str(reset_ms)}},
        }
    }
    return http_error(429, f"Error code: 429 - {message}", body), reset_ms


def test_free_daily_limit_stops_at_once_and_reports_the_reset_time(env, clock):
    error, reset_ms = daily_limit_error()
    completions = Completions(error)
    with pytest.raises(LLMError, match="quota exhausted") as info:
        make_llm(env, completions).chat("s", USER, TOOLS)
    assert len(completions.requests) == 1 and clock.sleeps == []
    reset = real_time.strftime("%Y-%m-%d %H:%M", real_time.gmtime(reset_ms / 1000))
    assert f"resets around {reset} UTC" in str(info.value)


def test_free_per_minute_limit_is_retried_a_bounded_number_of_times(env, clock):
    message = "Rate limit exceeded: free-models-per-min."
    error = http_error(429, message, {"error": {"code": 429, "message": message}}, {"retry-after": "4"})
    completions = Completions(error, error, completion("ok"))
    assert make_llm(env, completions).chat("s", USER, TOOLS).text == "ok"
    assert len(completions.requests) == 3 and clock.sleeps == [4.0, 4.0]
    clock.sleeps.clear()
    completions = Completions(error)
    with pytest.raises(LLMError, match="after 3 attempt"):
        make_llm(env, completions).chat("s", USER, TOOLS)
    assert len(completions.requests) == 3 and len(clock.sleeps) == 2  # no sleep after the final attempt


def test_a_completion_with_no_choices_but_a_daily_limit_error_does_not_crash(env, clock):
    error, _ = daily_limit_error()
    completions = Completions(SimpleNamespace(choices=None, error=error.body["error"]))
    with pytest.raises(LLMError, match="quota exhausted"):
        make_llm(env, completions).chat("s", USER, TOOLS)
    assert len(completions.requests) == 1


def test_a_completion_with_no_choices_and_an_upstream_5xx_is_retried(env, clock):
    broken = SimpleNamespace(choices=None, error={"code": 502, "message": "Provider returned error"})
    completions = Completions(broken, completion("recovered"))
    assert make_llm(env, completions).chat("s", USER, TOOLS).text == "recovered"
    assert len(completions.requests) == 2 and clock.sleeps == [3.0]


def test_an_empty_completion_without_any_error_is_treated_as_transient(env, clock):
    completions = Completions(SimpleNamespace(choices=[], error=None), completion("ok"))
    assert make_llm(env, completions).chat("s", USER, TOOLS).text == "ok"


def test_a_completion_without_a_message_is_retried_as_transient(env, clock):
    broken = SimpleNamespace(choices=[SimpleNamespace(message=None)], error=None)
    completions = Completions(broken, completion("ok"))
    assert make_llm(env, completions).chat("s", USER, TOOLS).text == "ok"
    assert len(completions.requests) == 2 and clock.sleeps == [3.0]


def test_a_completion_carrying_a_402_error_is_fatal_with_the_credit_hint(env, clock):
    completions = Completions(SimpleNamespace(choices=None, error={"code": 402, "message": "Insufficient credits"}))
    with pytest.raises(LLMError, match="insufficient credits"):
        make_llm(env, completions).chat("s", USER, TOOLS)
    assert len(completions.requests) == 1


def test_transient_upstream_errors_still_recover(env, clock):
    completions = Completions(http_error(503, "overloaded"), http_error(502, "bad gateway"), completion("ok"))
    assert make_llm(env, completions).chat("s", USER, TOOLS).text == "ok"
    assert clock.sleeps == [3.0, 6.0]


# --------------------------------------------------------------------- request spacing
def test_request_spacing_can_be_overridden(env):
    env.setenv("LLM_MIN_SECONDS_BETWEEN_CALLS", "0")
    env.setenv("OPENROUTER_API_KEY", OR_KEY)
    assert build_llm("openrouter").throttle.min_interval == 0.0  # the override still works


# --------------------------------------------------------------------------------- no secret leakage
def test_the_openrouter_key_is_scrubbed_from_errors_and_logs(env, clock, caplog):
    env.setenv("OPENROUTER_API_KEY", OR_KEY)
    nasty = f"upstream rejected Authorization: Bearer {OR_KEY} for key {OR_KEY}"
    assert OR_KEY not in _safe_message(Exception(nasty))
    completions = Completions(http_error(503, nasty), http_error(401, nasty))
    llm = make_llm(env, completions)
    with caplog.at_level(logging.INFO, logger="audit.llm"), pytest.raises(LLMError) as info:
        llm.chat("s", USER, TOOLS)
    assert OR_KEY not in caplog.text and OR_KEY not in str(info.value)


def test_openrouter_style_keys_are_recognised_by_the_redactor():
    assert redact(f"key = {OR_KEY}") == "key = [REDACTED:openrouter_key]"
    assert OR_KEY not in redact(f"Authorization: Bearer {OR_KEY}")


def test_retry_helper_still_gives_up_on_credit_errors_for_any_status_carrier(clock):
    calls = []

    def fail():
        calls.append(1)
        raise http_error(402, "payment required")

    with pytest.raises(LLMError, match="insufficient credits"):
        _retrying(fail)
    assert len(calls) == 1 and clock.sleeps == []


def test_direct_construction_uses_only_the_openrouter_variable(env):
    env.setenv("OPENAI_API_KEY", "sk-this-is-an-openai-key-0123456789")
    with pytest.raises(LLMError, match="OPENROUTER_API_KEY is not set"):
        OpenRouterLLM()
    env.setenv("OPENROUTER_API_KEY", OR_KEY)
    assert OpenRouterLLM().client.api_key == OR_KEY


def test_a_configured_key_of_unknown_shape_is_scrubbed_from_errors_and_logs(env, clock, caplog):
    odd_key = "orkey-unknown-format-0123456789abcdef"  # matches no known token pattern, so only the scrubber can catch it
    env.setenv("OPENROUTER_API_KEY", odd_key)
    assert redact(odd_key) == odd_key  # the pattern-based redactor alone would let it through
    message = f"401 from upstream, request used key {odd_key}"
    assert odd_key not in _safe_message(Exception(message))
    completions = Completions(http_error(503, message), http_error(401, message))
    llm = build_llm("openrouter")
    llm.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    llm.throttle = _Throttle(0)
    with caplog.at_level(logging.INFO, logger="audit.llm"), pytest.raises(LLMError) as info:
        llm.chat("s", USER, TOOLS)
    assert odd_key not in caplog.text and odd_key not in str(info.value)
