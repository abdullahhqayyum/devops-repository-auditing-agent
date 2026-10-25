import json
from types import SimpleNamespace

from fakes import ScriptedLLM, call

import agent as cli
from auditor import prompts


def test_invalid_url_exits_nonzero(tmp_path, capsys):
    code = cli.main(["--repo", "https://gitlab.com/a/b", "--output", str(tmp_path / "r.json")])
    assert code == cli.EXIT_USAGE
    assert "not a GitHub repository URL" in capsys.readouterr().err
    assert not (tmp_path / "r.json").exists()


def test_missing_credentials_exit_nonzero(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda **_: False)
    code = cli.main(["--repo", "https://github.com/a/b", "--output", str(tmp_path / "r.json")])
    assert code == cli.EXIT_USAGE
    assert "OPENROUTER_API_KEY" in capsys.readouterr().err


def test_end_to_end_with_scripted_llm(sample_repo, tmp_path, monkeypatch):
    finding = {
        "rule_id": "CI-003",
        "category": "ci",
        "severity": "high",
        "file": ".github/workflows/ci.yml",
        "line": 12,
        "description": "Test failures are ignored",
        "evidence": "pytest || true with continue-on-error",
        "recommendation": "Remove || true and continue-on-error so failing tests block merges",
    }
    turns = [
        [call("inspect_workflows")],
        [call("record_finding", **finding)],
        [call("mark_reviewed", category=c, status="no_issues", note="ok") for c in prompts.CATEGORIES],
        [call("finish_audit")],
        [call("finish_audit")],
    ]
    monkeypatch.setattr(cli, "build_llm", lambda *a, **k: ScriptedLLM(turns))
    out = tmp_path / "report.json"
    code = cli.main(["--repo", str(sample_repo), "--allow-local", "--output", str(out)])
    assert code == cli.EXIT_OK
    report = json.loads(out.read_text())
    assert set(report) == {"repository", "violations"}
    assert report["violations"][0]["rule_id"] == "CI-003"
    trace = json.loads((tmp_path / "report.trace.json").read_text())
    assert trace["tool_call_counts"] == {"inspect_workflows": 1}
    # The original repository is untouched.
    import subprocess

    st = subprocess.run(["git", "status", "--porcelain"], cwd=sample_repo, capture_output=True, text=True)
    assert st.stdout == ""


def test_cli_with_only_openrouter_key_and_mocked_sdk(sample_repo, tmp_path, monkeypatch):
    """The required command runs through the real client and loop without a network request."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-test-key")
    monkeypatch.setattr("auditor.llm_client.DEFAULT_MIN_INTERVAL", 0.0)
    requests = []

    def tool_call(call_id, name, arguments):
        raw = {
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }
        return SimpleNamespace(
            id=call_id,
            function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
            model_dump=lambda exclude_none=True: raw,
        )

    def fake_create(_self, **kwargs):
        requests.append(kwargs)
        turn = len(requests)
        if turn == 1:
            calls = [tool_call("list-1", "list_files", {})]
        elif turn == 2:
            calls = [
                tool_call(f"review-{i}", "mark_reviewed", {"category": category, "status": "no_issues", "note": "ok"})
                for i, category in enumerate(prompts.CATEGORIES)
            ]
        else:
            calls = [tool_call("finish-1", "finish_audit", {})]
        message = SimpleNamespace(content="", tool_calls=calls)
        choice = SimpleNamespace(message=message, finish_reason="tool_calls")
        return SimpleNamespace(choices=[choice], usage=None)

    monkeypatch.setattr("openai.resources.chat.completions.Completions.create", fake_create)
    output = tmp_path / "openrouter-report.json"
    code = cli.main(["--repo", str(sample_repo), "--allow-local", "--output", str(output)])
    assert code == cli.EXIT_OK
    assert json.loads(output.read_text()) == {"repository": str(sample_repo.resolve()), "violations": []}
    assert [request["model"] for request in requests] == ["openrouter/free"] * 4  # includes empty-finding self-review
    assert requests[3]["messages"][-1]["content"] == prompts.review_nudge("(none)")
    trace = json.loads((tmp_path / "openrouter-report.trace.json").read_text())
    assert trace["status"] == "complete" and trace["tool_call_counts"] == {"list_files": 1}


def test_local_path_requires_flag(sample_repo, tmp_path):
    code = cli.main(["--repo", str(sample_repo), "--output", str(tmp_path / "r.json")])
    assert code == cli.EXIT_USAGE


def test_the_test_environment_is_isolated_from_the_developers_keys_and_dotenv():
    import os

    from conftest import LLM_ENV_VARS

    assert [v for v in LLM_ENV_VARS if v in os.environ] == []
    import dotenv

    assert dotenv.load_dotenv() is False  # the real .env is never loaded during tests
    assert [v for v in LLM_ENV_VARS if v in os.environ] == []
