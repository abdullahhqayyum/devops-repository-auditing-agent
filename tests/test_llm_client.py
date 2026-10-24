import pytest

from llm_client import DEFAULT_MODELS, FALLBACK_MODELS, GEMINI_BASE_URL, AnthropicLLM, LLMError, OpenAILLM, build_llm

HISTORY = [
    {"role": "user", "content": "audit"},
    {
        "role": "assistant",
        "content": "look",
        "tool_calls": [
            {"id": "a", "name": "list_files", "arguments": {}},
            {"id": "b", "name": "read_file", "arguments": {"path": "x"}},
        ],
    },
    {"role": "tool", "tool_call_id": "a", "name": "list_files", "content": "x"},
    {"role": "tool", "tool_call_id": "b", "name": "read_file", "content": "1| y"},
]


def test_anthropic_conversion_groups_tool_results():
    llm = AnthropicLLM.__new__(AnthropicLLM)
    msgs = llm._convert(HISTORY)
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert [b["type"] for b in msgs[1]["content"]] == ["text", "tool_use", "tool_use"]
    assert [b["tool_use_id"] for b in msgs[2]["content"]] == ["a", "b"]


def test_openai_conversion():
    llm = OpenAILLM.__new__(OpenAILLM)
    msgs = llm._convert("sys", HISTORY)
    assert msgs[0] == {"role": "system", "content": "sys"}
    assert msgs[2]["tool_calls"][1]["function"]["arguments"] == '{"path": "x"}'
    assert [m["role"] for m in msgs[3:]] == ["tool", "tool"]


def test_openai_conversion_echoes_raw_tool_calls():
    """Provider extras (e.g. Gemini thought signatures) must be sent back unchanged."""
    raw = {
        "id": "a",
        "type": "function",
        "function": {"name": "list_files", "arguments": "{}"},
        "extra_content": {"google": {"thought_signature": "sig"}},
    }
    history = [
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "a", "name": "list_files", "arguments": {}, "raw": raw}]},
    ]
    msgs = OpenAILLM.__new__(OpenAILLM)._convert("sys", history)
    assert msgs[2]["tool_calls"][0] == raw


@pytest.fixture
def clean_env(monkeypatch):
    for var in (
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "LLM_PROVIDER",
        "LLM_MODEL",
        "OPENAI_BASE_URL",
        "LLM_MIN_SECONDS_BETWEEN_CALLS",
    ):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def test_build_llm_gemini_autodetect(clean_env):
    clean_env.setenv("GEMINI_API_KEY", "test-key")
    llm = build_llm()
    assert llm.model == DEFAULT_MODELS["gemini"]
    assert llm.fallbacks == FALLBACK_MODELS["gemini"]
    assert str(llm.client.base_url) == GEMINI_BASE_URL
    assert llm.throttle.min_interval > 0


def test_build_llm_respects_overrides(clean_env):
    clean_env.setenv("GEMINI_API_KEY", "k")
    clean_env.setenv("LLM_MODEL", "gemini-other")
    clean_env.setenv("LLM_MIN_SECONDS_BETWEEN_CALLS", "0")
    llm = build_llm("gemini")
    assert llm.model == "gemini-other" and llm.throttle.min_interval == 0
    assert llm.fallbacks == []  # an explicitly chosen model is never swapped


def test_build_llm_errors(clean_env):
    with pytest.raises(LLMError, match="GEMINI_API_KEY"):
        build_llm()
    with pytest.raises(LLMError, match="Unsupported"):
        build_llm("mistral")


class _NotFound(Exception):
    status_code = 404


class _FakeCompletions:
    """Rejects every model except ``available`` the way a retired model is rejected (HTTP 404)."""

    def __init__(self, available: str) -> None:
        self.available = available
        self.models: list[str] = []

    def create(self, model, **_kwargs):
        from types import SimpleNamespace

        self.models.append(model)
        if model != self.available:
            raise _NotFound(f"model {model} not found")
        message = SimpleNamespace(content="ok", tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")], usage=None)


def _fake_openai(model: str, fallbacks: list[str], available: str) -> OpenAILLM:
    from types import SimpleNamespace

    from llm_client import _Throttle

    llm = OpenAILLM.__new__(OpenAILLM)
    llm.model, llm.fallbacks, llm.max_tokens, llm.throttle = model, list(fallbacks), 100, _Throttle(0)
    llm.client = SimpleNamespace(chat=SimpleNamespace(completions=_FakeCompletions(available)))
    return llm


def test_retired_default_model_falls_back():
    llm = _fake_openai("retired-model", ["also-retired", "live-model"], available="live-model")
    assert llm.chat("sys", [{"role": "user", "content": "hi"}], []).text == "ok"
    assert llm.model == "live-model"
    assert llm.client.chat.completions.models == ["retired-model", "also-retired", "live-model"]


def test_missing_model_without_fallback_is_fatal():
    llm = _fake_openai("retired-model", [], available="live-model")
    with pytest.raises(LLMError, match="not found"):
        llm.chat("sys", [{"role": "user", "content": "hi"}], [])


def test_gemini_default_is_the_live_verified_model_and_keeps_older_names_as_fallbacks():
    assert DEFAULT_MODELS["gemini"] == "gemini-3.6-flash"
    assert FALLBACK_MODELS["gemini"][0] == "gemini-3.5-flash"  # tried only if the default is reported missing (404)
    assert DEFAULT_MODELS["gemini"] not in FALLBACK_MODELS["gemini"]
