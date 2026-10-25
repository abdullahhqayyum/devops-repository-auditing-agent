import sys
from types import SimpleNamespace

import pytest
from conftest import FAKE_KEY

import tools as tools_module
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


# ---------------------------------------------------------------- run_ruff detection
def _no_ruff_anywhere(monkeypatch, tmp_path):
    """Pretend ruff is installed nowhere: not importable, not beside the interpreter, not on PATH."""
    monkeypatch.setattr(tools_module, "importlib", SimpleNamespace(util=SimpleNamespace(find_spec=lambda *_a, **_k: None)))
    monkeypatch.setattr(tools_module, "sys", SimpleNamespace(executable=str(tmp_path / "empty-bin" / "python")))
    monkeypatch.setattr(tools_module.shutil, "which", lambda *_a, **_k: None)


def test_run_ruff_works_when_ruff_is_not_on_path(sample_repo, monkeypatch):
    """Regression: with the virtual environment not activated ruff is not on PATH, but the running interpreter has it."""
    monkeypatch.setattr(tools_module.shutil, "which", lambda *_a, **_k: None)
    out = RepoTools(sample_repo).run_ruff()
    assert "not installed" not in out and "unavailable" not in out
    assert "issue(s) across" in out and "E722" in out


def test_ruff_command_prefers_the_running_interpreter_over_path(monkeypatch):
    monkeypatch.setattr(tools_module.shutil, "which", lambda *_a, **_k: "/somewhere/else/ruff")
    assert tools_module._ruff_command() == [sys.executable, "-I", "-m", "ruff"]


def test_ruff_command_falls_back_to_the_executable_beside_the_interpreter(monkeypatch, tmp_path):
    _no_ruff_anywhere(monkeypatch, tmp_path)
    bin_dir = tmp_path / "empty-bin"
    bin_dir.mkdir()
    exe = bin_dir / ("ruff.exe" if tools_module.os.name == "nt" else "ruff")
    exe.write_text("")
    assert tools_module._ruff_command() == [str(exe)]


def test_ruff_command_falls_back_to_path_last(monkeypatch, tmp_path):
    _no_ruff_anywhere(monkeypatch, tmp_path)
    monkeypatch.setattr(tools_module.shutil, "which", lambda *_a, **_k: "/opt/tools/ruff")
    assert tools_module._ruff_command() == ["/opt/tools/ruff"]


def test_run_ruff_reports_unavailable_when_ruff_is_really_missing(sample_repo, monkeypatch, tmp_path):
    _no_ruff_anywhere(monkeypatch, tmp_path)
    assert tools_module._ruff_command() is None
    for out in (RepoTools(sample_repo).run_ruff(), dispatch(RepoTools(sample_repo), "run_ruff", {})):
        assert "not installed" in out and "unavailable" in out and "do not infer" in out
        assert "issue(s)" not in out  # an unavailable check is not a finding


def test_run_ruff_never_executes_repository_code(tmp_path):
    """The audited repository is the working directory, so a ruff.py in it must not shadow the real package."""
    from conftest import make_git_repo

    repo = make_git_repo(
        tmp_path / "evil",
        {"ruff.py": "import pathlib\npathlib.Path('PWNED.txt').write_text('repository code ran')\n", "ok.py": "import os\n"},
    )
    out = RepoTools(repo).run_ruff()
    assert not (repo / "PWNED.txt").exists()
    assert "F401" in out  # the real ruff ran and linted the repository


def test_run_ruff_registers_reported_files_as_inspected_with_posix_paths(tmp_path):
    from conftest import make_git_repo

    repo = make_git_repo(tmp_path / "r", {"app/main.py": "import os\n", ".tools/helper.py": "import sys\n"})
    tools = RepoTools(repo)
    tools.run_ruff()
    assert {"app/main.py", ".tools/helper.py"} <= tools.observed_files
    assert not any("\\" in f for f in tools.observed_files)


# ------------------------------------------- findings about tracked binary artifacts (e.g. a committed .pyc)
PYC = "app/__pycache__/m.cpython-311.pyc"


def _binary_repo(tmp_path, extra=None):
    from conftest import make_git_repo

    files = {"README.md": "# x\n", ".gitignore": "*.log\n", PYC: "\x00compiled\x00", "app/main.py": "x = 1\n", **(extra or {})}
    return make_git_repo(tmp_path / "bin", files)


def _hyg003(file):
    return {
        "rule_id": "HYG-003",
        "category": "repository_hygiene",
        "severity": "medium",
        "file": file,
        "line": 1,
        "description": "Tracked Python bytecode",
        "evidence": "The bytecode file is tracked in git",
        "recommendation": "Remove it from git and ignore __pycache__/",
    }


def test_reading_a_binary_tracked_file_registers_it_as_seen(tmp_path):
    t = RepoTools(_binary_repo(tmp_path))
    assert "binary file; not displayed" in t.read_file(PYC)
    assert PYC in t.observed_files


def test_inspect_tracked_artifacts_registers_only_the_files_it_names(tmp_path):
    t = RepoTools(_binary_repo(tmp_path))
    out = t.inspect_tracked_artifacts()
    assert PYC in out and PYC in t.observed_files
    assert "app/main.py" not in t.observed_files and "README.md" not in t.observed_files


def test_a_finding_about_a_tracked_binary_artifact_needs_the_agent_to_have_seen_it(tmp_path):
    from report import FindingRejected, FindingStore

    repo = _binary_repo(tmp_path)
    unseen = RepoTools(repo)
    with pytest.raises(FindingRejected, match="not inspected"):
        FindingStore(unseen).add(_hyg003(PYC))  # grounding is intact: nothing has looked at the file yet
    seen = RepoTools(repo)
    seen.inspect_tracked_artifacts()
    assert FindingStore(seen).add(_hyg003(PYC)) == "F1"  # once the tool has named it, the finding can cite it
    read_only = RepoTools(repo)
    read_only.read_file(PYC)
    assert FindingStore(read_only).add(_hyg003(PYC)) == "F1"


def test_listing_a_file_is_still_not_inspecting_it(tmp_path):
    from report import FindingRejected, FindingStore

    t = RepoTools(_binary_repo(tmp_path))
    assert PYC in t.list_files()
    assert PYC not in t.observed_files
    with pytest.raises(FindingRejected, match="not inspected"):
        FindingStore(t).add(_hyg003(PYC))


def test_artifacts_beyond_the_display_limit_are_not_registered(tmp_path):
    many = {f"pkg{i}/__pycache__/m.cpython-311.pyc": "\x00x\x00" for i in range(20)}
    t = RepoTools(_binary_repo(tmp_path, many))
    out = t.inspect_tracked_artifacts()
    assert "21 tracked" in out and "..." in out
    shown = {f for f in t.observed_files if f.endswith(".pyc")}
    assert len(shown) == 15 and all(f in out for f in shown)


def test_files_tracked_despite_gitignore_are_registered_when_named(tmp_path):
    # *.dat matches none of the artifact categories, so only the tracked-but-gitignored branch can register it.
    t = RepoTools(_binary_repo(tmp_path, {".gitignore": "*.dat\n", "data/blob.dat": "x\n"}))
    out = t.inspect_tracked_artifacts()
    assert "tracked_but_gitignored" in out and "data/blob.dat" in out
    assert "data/blob.dat" in t.observed_files
