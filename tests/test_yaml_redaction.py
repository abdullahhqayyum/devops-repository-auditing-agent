"""Regression tests: unquoted YAML-style secrets must not reach the model, the report or the trace."""

import json
import subprocess

import pytest
from conftest import make_git_repo
from fakes import ScriptedLLM, call

import prompts
from agent_loop import AuditAgent, AuditConfig
from redaction import redact
from report import FindingStore
from tools import RepoTools

YAML_PW = "hunter2"  # synthetic
YAML_KEY = "s3cr3t-Value!9"  # synthetic


@pytest.mark.parametrize(
    "line",
    [
        "password: hunter2",
        "    db_password: Zq9xLm2pRt7vWn   # inline comment",
        "- token: s3cr3t-Value!9",
        "MYSQL_ROOT_PASSWORD: rootpass123",
        "api-key: abc123def456",
        '"client_secret": abc123def456',
        "   6|     password: hunter2",
        "docker-compose.yml:6:    password: hunter2",
        "+  password: hunter2",
    ],
)
def test_unquoted_yaml_secret_values_are_redacted_with_or_without_a_prefix(line):
    out = redact(line)
    assert "[REDACTED:value]" in out
    for secret in ("hunter2", "Zq9xLm2pRt7vWn", "s3cr3t-Value!9", "rootpass123", "abc123def456"):
        assert secret not in out


@pytest.mark.parametrize(
    "line",
    [
        "password: ${{ secrets.DB_PW }}",
        "password: $DB_PASSWORD",
        "password: <your-password>",
        "token_url: https://auth.example.com/oauth/token",
        "password_min_length: 12",
        "secrets: inherit",
        "password: required",
        "secret_path: /etc/secrets/db",
        "token_version: 2.1.0",
        "def f(password: str, token: Optional[str] = None, secret: SecretStr = None): ...",
        "api_key: Optional[str] = None",
        "name: my-service",
    ],
)
def test_benign_yaml_and_source_code_are_not_redacted(line):
    assert redact(line) == line


def test_yaml_redaction_is_idempotent():
    once = redact("password: hunter2")
    assert redact(once) == once


def _yaml_repo(tmp_path):
    return make_git_repo(
        tmp_path / "yamlrepo",
        {
            "docker-compose.yml": (
                f"services:\n  db:\n    password: {YAML_PW}\n    api_key: {YAML_KEY}\n"
                "    token_url: https://auth.example.com/token\n    secrets: inherit\n"
            ),
            "app.py": "def f(password: str, token: Optional[str] = None, secret: SecretStr = None): ...\n",
        },
    )


def test_yaml_secrets_are_redacted_by_read_file_and_search_code(tmp_path):
    t = RepoTools(_yaml_repo(tmp_path))
    read = t.read_file("docker-compose.yml")
    found = t.search_code("password|api_key")
    for out in (read, found):
        assert YAML_PW not in out and YAML_KEY not in out
    assert "    3|     password: [REDACTED:value]" in read
    assert "docker-compose.yml:3:    password: [REDACTED:value]" in found
    assert "token_url: https://auth.example.com/token" in read and "secrets: inherit" in read


def test_yaml_secrets_are_redacted_in_commit_diffs(tmp_path):
    repo = _yaml_repo(tmp_path)
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()
    out = RepoTools(repo).git_show_commit(sha)
    assert YAML_PW not in out and YAML_KEY not in out
    assert "+    password: [REDACTED:value]" in out


def test_source_code_annotations_survive_yaml_redaction(tmp_path):
    t = RepoTools(_yaml_repo(tmp_path))
    out = t.read_file("app.py") + t.search_code("SecretStr")
    assert "password: str, token: Optional[str] = None, secret: SecretStr = None" in out


def test_yaml_secret_in_report_evidence_is_redacted(tmp_path):
    tools = RepoTools(_yaml_repo(tmp_path))
    tools.read_file("docker-compose.yml")
    store = FindingStore(tools)
    store.add(
        {
            "rule_id": "SEC-001",
            "category": "security",
            "severity": "high",
            "file": "docker-compose.yml",
            "line": 3,
            "description": "Hardcoded database password",
            "evidence": f"compose file has password: {YAML_PW} and api_key: {YAML_KEY} in plain text",
            "recommendation": "Use a secrets manager",
        }
    )
    text = json.dumps(store.as_report_list())
    assert YAML_PW not in text and YAML_KEY not in text
    assert "password: [REDACTED:value]" in text


def test_trace_redacts_yaml_style_secrets_in_arguments(sample_repo):
    turns = [
        [call("search_code", pattern=f"password: {YAML_PW}")],
        [call("mark_reviewed", category=c, status="no_issues", note="ok") for c in prompts.CATEGORIES],
        [call("finish_audit")],
    ]
    agent = AuditAgent(ScriptedLLM(turns), RepoTools(sample_repo), "u", AuditConfig(self_review=False))
    agent.run()
    assert YAML_PW not in json.dumps(agent.trace.to_dict())
