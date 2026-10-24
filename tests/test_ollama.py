import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from llm_client import LLMError, OllamaLLM, build_llm
from prompts import AGENT_TOOL_SCHEMAS


class _Handler(BaseHTTPRequestHandler):
    requests: list = []
    reply: dict = {}
    status = 200

    def do_POST(self):  # noqa: N802 - http.server API
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Handler.requests.append((self.path, body))
        data = json.dumps(_Handler.reply if _Handler.status == 200 else {"error": "model 'x' not found"}).encode()
        self.send_response(_Handler.status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    _Handler.requests, _Handler.status = [], 200
    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()


def test_chat_parses_tool_calls_and_sends_num_ctx(server):
    _Handler.reply = {
        "message": {
            "role": "assistant",
            "content": "<think>plan</think>Checking files.",
            "tool_calls": [
                {"function": {"name": "list_files", "arguments": {"path": "."}}},
                {"function": {"name": "read_file", "arguments": '{"path": "README.md"}'}},
            ],
        },
        "done_reason": "stop",
        "prompt_eval_count": 100,
        "eval_count": 20,
    }
    llm = OllamaLLM("qwen3:8b", host=server, num_ctx=16384)
    history = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "1", "name": "list_files", "arguments": {}}]},
        {"role": "tool", "tool_call_id": "1", "name": "list_files", "content": "README.md"},
    ]
    r = llm.chat("sys", history, AGENT_TOOL_SCHEMAS)
    path, body = _Handler.requests[0]
    assert path == "/api/chat" and body["stream"] is False
    assert body["options"]["num_ctx"] == 16384
    assert body["messages"][0] == {"role": "system", "content": "sys"}
    assert body["messages"][3] == {"role": "tool", "content": "README.md", "tool_name": "list_files"}
    assert [c.name for c in r.tool_calls] == ["list_files", "read_file"]
    assert r.tool_calls[1].arguments == {"path": "README.md"}
    assert len({c.id for c in r.tool_calls}) == 2
    assert r.text == "Checking files."


def test_missing_model_gives_pull_hint(server):
    _Handler.status = 404
    with pytest.raises(LLMError, match="ollama pull"):
        OllamaLLM("x", host=server).chat("s", [{"role": "user", "content": "hi"}], [])


def test_unreachable_server_is_clear_error():
    with pytest.raises(LLMError, match="ollama serve"):
        OllamaLLM("x", host="http://127.0.0.1:9", timeout=2).chat("s", [{"role": "user", "content": "hi"}], [])


def test_build_llm_ollama(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_NUM_CTX", "65536")
    monkeypatch.setenv("LLM_MODEL", "gpt-oss:20b")
    llm = build_llm()
    assert isinstance(llm, OllamaLLM) and llm.num_ctx == 65536 and llm.model == "gpt-oss:20b"
    assert llm.context_chars < 220_000
