"""Provider-neutral message history and OpenRouter-only configuration."""

import pytest

from auditor.llm_client import DEFAULT_MODEL, LLMError, OpenRouterLLM, build_llm

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


@pytest.fixture
def clean_env(monkeypatch):
    for var in (
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "OPENROUTER_API_KEY",
        "LLM_PROVIDER",
        "LLM_MODEL",
        "OPENAI_BASE_URL",
        "LLM_MIN_SECONDS_BETWEEN_CALLS",
    ):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def test_openrouter_conversion_keeps_parallel_call_ids_and_results():
    messages = OpenRouterLLM.__new__(OpenRouterLLM)._convert("sys", HISTORY)
    assert messages[0] == {"role": "system", "content": "sys"}
    assert [call["id"] for call in messages[2]["tool_calls"]] == ["a", "b"]
    assert messages[2]["tool_calls"][1]["function"]["arguments"] == '{"path": "x"}'
    assert [(message["tool_call_id"], message["content"]) for message in messages[3:]] == [
        ("a", "x"),
        ("b", "1| y"),
    ]


def test_openrouter_conversion_echoes_raw_tool_payload():
    raw = {
        "id": "a",
        "type": "function",
        "function": {"name": "list_files", "arguments": "{}"},
        "extra_content": {"router": {"signature": "sig"}},
    }
    history = [
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "a", "name": "list_files", "arguments": {}, "raw": raw}]},
    ]
    messages = OpenRouterLLM.__new__(OpenRouterLLM)._convert("sys", history)
    assert messages[2]["tool_calls"] == [raw]


def test_only_openrouter_key_can_build_client(clean_env):
    clean_env.setenv("GEMINI_API_KEY", "other-key")
    clean_env.setenv("OPENAI_API_KEY", "another-key")
    with pytest.raises(LLMError, match="OPENROUTER_API_KEY is not set"):
        build_llm()
    clean_env.setenv("OPENROUTER_API_KEY", "router-test-key")
    client = build_llm()
    assert isinstance(client, OpenRouterLLM)
    assert client.model == DEFAULT_MODEL == "openrouter/free"
    assert client.client.api_key == "router-test-key"
    assert client.client.base_url.host == "openrouter.ai"


def test_model_selection_and_legacy_provider_environment(clean_env):
    clean_env.setenv("OPENROUTER_API_KEY", "router-test-key")
    clean_env.setenv("LLM_MODEL", "cohere/north-mini-code:free")
    assert build_llm().model == "cohere/north-mini-code:free"
    assert build_llm(model="google/gemma-4-31b-it:free").model == "google/gemma-4-31b-it:free"

    # An old provider setting paired with its model must not select that model through OpenRouter.
    clean_env.setenv("LLM_PROVIDER", "openai")
    clean_env.setenv("LLM_MODEL", "openai/gpt-4.1")
    assert build_llm().model == DEFAULT_MODEL
    assert build_llm(model="cohere/north-mini-code:free").model == "cohere/north-mini-code:free"
    clean_env.setenv("LLM_PROVIDER", "openrouter")
    assert build_llm().model == "openai/gpt-4.1"  # an intentional OpenRouter slug is passed through


def test_explicit_non_openrouter_provider_is_rejected(clean_env):
    clean_env.setenv("OPENROUTER_API_KEY", "router-test-key")
    with pytest.raises(LLMError, match="Unsupported provider"):
        build_llm("gemini")
