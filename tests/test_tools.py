import pytest
from conftest import FAKE_KEY

from tools import RepoTools, dispatch, tool_schemas


def test_list_files_and_filter(sample_repo):
    t = RepoTools(sample_repo)
    out = t.list_files()
    assert "app/config.py" in out and ".git/" not in out
    assert "ci.yml" in t.list_files(pattern="*.yml")
    assert "No tracked files" in t.list_files(path="nope")


def test_read_file_numbers_lines_and_marks_observed(sample_repo):
    t = RepoTools(sample_repo)
    out = t.read_file("app/config.py", start_line=2, end_line=3)
    assert "    2| DEBUG = True" in out
    assert FAKE_KEY not in out  # redacted
    assert "app/config.py" in t.observed_files


def test_read_file_rejects_traversal_and_git_dir(sample_repo):
    t = RepoTools(sample_repo)
    assert "escapes" in dispatch(t, "read_file", {"path": "../../etc/passwd"})
    assert "ERROR" in dispatch(t, "read_file", {"path": ".git/config"})
    assert "File not found" in t.read_file("missing.py")


def test_search_code(sample_repo):
    t = RepoTools(sample_repo)
    out = t.search_code("verify=False")
    assert "app/main.py:6:" in out
    assert "No matches" in t.search_code("definitely_not_here_123")
    assert "app/main.py" in t.observed_files


def test_search_code_bad_regex_is_reported(sample_repo):
    out = RepoTools(sample_repo).search_code("([unclosed")
    assert "failed" in out.lower() or "no matches" in out.lower()


def test_git_log(sample_repo):
    out = RepoTools(sample_repo).git_log(max_count=5)
    assert "init" in out and "Commits reachable: 1" in out


def test_tracked_artifacts(sample_repo):
    out = RepoTools(sample_repo).inspect_tracked_artifacts()
    assert "python_bytecode" in out and "debug.log" in out
    assert ".env.example" not in out  # templates are fine


def test_scan_secrets_redacts(sample_repo):
    out = RepoTools(sample_repo).scan_secrets()
    assert "app/config.py:3" in out and FAKE_KEY not in out
    assert ".env.example" not in out.split("ignored")[0]


def test_inspect_workflows_markers(sample_repo):
    out = RepoTools(sample_repo).inspect_workflows()
    assert "write-all" in out
    assert "continue-on-error" in out
    assert "mutable branch ref" in out
    assert "|| true" in out


def test_inspect_workflows_absent(tmp_path):
    from conftest import make_git_repo

    repo = make_git_repo(tmp_path / "r", {"README.md": "x", ".gitlab-ci.yml": "test: {script: [pytest]}"})
    out = RepoTools(repo).inspect_workflows()
    assert "No GitHub Actions" in out and ".gitlab-ci.yml" in out


def test_run_ruff_static(sample_repo):
    out = RepoTools(sample_repo).run_ruff()
    assert "ruff" in out


def test_dispatch_validation(sample_repo):
    t = RepoTools(sample_repo)
    assert "unknown tool" in dispatch(t, "rm_rf", {})
    assert "missing required" in dispatch(t, "read_file", {})
    assert "run_tests" not in {s["name"] for s in tool_schemas(t)}  # no sandbox => no code execution
    assert "disabled" in t.run_tests()


def test_tools_do_not_modify_repo(sample_repo):
    import subprocess

    t = RepoTools(sample_repo)
    for name in ("list_files", "inspect_tracked_artifacts", "scan_secrets", "inspect_workflows", "run_ruff", "git_log"):
        dispatch(t, name, {})
    status = subprocess.run(["git", "status", "--porcelain", "--ignored"], cwd=sample_repo, capture_output=True, text=True)
    assert status.stdout.strip() == ""


ENV_HEX = "a8f5f167f44f4964e6c998dee827110c"  # synthetic
ENV_PW = "Zq9xLm2pRt7vWn"  # synthetic


def _env_repo(tmp_path):
    from conftest import make_git_repo

    return make_git_repo(
        tmp_path / "envrepo",
        {
            ".env": f"SECRET_KEY={ENV_HEX}\nDB_PASSWORD={ENV_PW}\nDEBUG=1\n",
            "cfg/app.ini": f"api_key = {ENV_HEX}\n",
            "app.py": "self.api_key = api_key\ntoken = get_token()\n",
        },
    )


def test_read_file_redacts_unquoted_env_values(tmp_path):
    t = RepoTools(_env_repo(tmp_path))
    out = t.read_file(".env") + t.read_file("cfg/app.ini")
    assert ENV_HEX not in out and ENV_PW not in out
    assert "SECRET_KEY=[REDACTED:value]" in out and "DEBUG=1" in out  # still useful for debugging


def test_search_code_redacts_unquoted_env_values(tmp_path):
    t = RepoTools(_env_repo(tmp_path))
    out = t.search_code("SECRET_KEY|DB_PASSWORD|api_key")
    assert ENV_HEX not in out and ENV_PW not in out
    assert ".env:1:SECRET_KEY=[REDACTED:value]" in out


def test_git_show_commit_redacts_added_and_removed_env_values(tmp_path):
    import subprocess

    repo = _env_repo(tmp_path)
    git = ["git", "-c", "user.name=T", "-c", "user.email=t@example.com"]
    first = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()
    subprocess.run([*git, "rm", "-q", ".env"], cwd=repo, check=True)
    subprocess.run([*git, "commit", "-q", "-m", "remove env"], cwd=repo, check=True)
    removed = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()
    t = RepoTools(repo)
    for sha in (first, removed):
        out = t.git_show_commit(sha)
        assert ENV_HEX not in out and ENV_PW not in out


def test_ordinary_source_code_is_not_over_redacted(tmp_path):
    t = RepoTools(_env_repo(tmp_path))
    out = t.read_file("app.py") + t.search_code("get_token")
    assert "self.api_key = api_key" in out and "token = get_token()" in out


MALFORMED_ARGUMENTS = [
    ("list_files", {"path": {"x": 1}}, "path"),
    ("list_files", {"path": ["a"]}, "path"),
    ("list_files", {"pattern": 5}, "pattern"),
    ("list_files", {"max_entries": "many"}, "max_entries"),
    ("read_file", {"path": 5}, "path"),
    ("read_file", {"path": "README.md", "start_line": {"a": 1}}, "start_line"),
    ("read_file", {"path": "README.md", "end_line": "abc"}, "end_line"),
    ("read_file", {"path": "README.md", "start_line": True}, "start_line"),
    ("search_code", {"pattern": 5}, "pattern"),
    ("search_code", {"pattern": "x", "path": 5}, "path"),
    ("search_code", {"pattern": "x", "max_results": [1]}, "max_results"),
    ("search_code", {"pattern": "x", "case_insensitive": "maybe"}, "case_insensitive"),
    ("git_log", {"path": 5}, "path"),
    ("git_log", {"max_count": "many"}, "max_count"),
    ("git_show_commit", {"commit": 123}, "commit"),
    ("scan_secrets", {"path": ["a"]}, "path"),
    ("inspect_workflows", {"path": 7}, "path"),
    ("run_ruff", {"path": 7}, "path"),
]


@pytest.mark.parametrize(("name", "args", "bad_key"), MALFORMED_ARGUMENTS)
def test_malformed_arguments_return_a_tool_error_instead_of_raising(sample_repo, name, args, bad_key):
    out = dispatch(RepoTools(sample_repo), name, args)
    assert out.startswith("ERROR: invalid argument")
    assert f"'{bad_key}'" in out and name in out


def test_all_problems_are_reported_in_one_error(sample_repo):
    out = dispatch(RepoTools(sample_repo), "read_file", {"path": 5, "start_line": {}})
    assert "'path'" in out and "'start_line'" in out


def test_non_object_arguments_are_rejected(sample_repo):
    t = RepoTools(sample_repo)
    assert "must be a JSON object" in dispatch(t, "list_files", ["app"])
    assert "must be a JSON object" in dispatch(t, "list_files", "app")


def test_lenient_coercions_still_work(sample_repo):
    t = RepoTools(sample_repo)
    assert "    2| DEBUG = True" in dispatch(t, "read_file", {"path": "app/config.py", "start_line": "2", "end_line": 3.0})
    assert "showing first 5 files" in dispatch(t, "list_files", {"max_entries": "5"})
    assert "No matches" not in dispatch(t, "search_code", {"pattern": "DEBUG", "case_insensitive": "true"})


def test_none_and_unknown_arguments_are_ignored_with_a_note(sample_repo):
    t = RepoTools(sample_repo)
    out = dispatch(t, "list_files", {"path": None, "colour": "red"})
    assert "app/config.py" in out and "ignored unknown argument(s): colour" in out
    assert "ignored" not in dispatch(t, "list_files", {"path": None})


def test_unrelated_programming_errors_are_not_swallowed(sample_repo, monkeypatch):
    t = RepoTools(sample_repo)
    monkeypatch.setattr(t, "list_files", lambda **_: {}["missing"])
    with pytest.raises(KeyError):
        dispatch(t, "list_files", {})
