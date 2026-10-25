"""Read-only inspection tools exposed to the LLM.

Design rules:
* Tools never modify the target repository (the checkout is also made read-only).
* Tools never execute repository code. ``run_tests`` is the single exception and is
  only registered when the operator explicitly enables the Docker sandbox.
* Every tool returns plain text that has been redacted and truncated, and records
  which files it actually showed to the model (used to verify evidence later).
"""

from __future__ import annotations

import fnmatch
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from auditor.redaction import find_secrets, is_env_style, redact, redact_env_values

MAX_READ_LINES = 400
MAX_FILE_BYTES = 2_000_000
DEFAULT_OUTPUT_CHARS = 12_000

# Files that are usually generated, temporary, machine-local or secret and should not be tracked.
SUSPICIOUS_TRACKED = [
    ("python_bytecode", ["*.pyc", "*.pyo", "*/__pycache__/*", "__pycache__/*"]),
    ("virtualenv", ["venv/*", ".venv/*", "env/*", "*/site-packages/*"]),
    ("node_modules", ["node_modules/*", "*/node_modules/*"]),
    ("build_output", ["dist/*", "build/*", "*.egg-info/*", "target/*", "out/*", "*.whl"]),
    (
        "test_and_tool_caches",
        [
            ".pytest_cache/*",
            ".mypy_cache/*",
            ".ruff_cache/*",
            ".tox/*",
            ".coverage",
            "htmlcov/*",
            "coverage.xml",
            ".nyc_output/*",
        ],
    ),
    ("os_and_editor_junk", [".DS_Store", "*/.DS_Store", "Thumbs.db", "*.swp", "*.swo", "*~", "*.bak", "*.tmp"]),
    ("ide_settings", [".idea/*", ".vscode/settings.json", "*.code-workspace", "*.iml"]),
    ("logs", ["*.log", "logs/*", "npm-debug.log*", "yarn-error.log*"]),
    ("local_databases", ["*.sqlite", "*.sqlite3", "*.db"]),
    (
        "env_files",
        [".env", "*/.env", ".env.local", ".env.*.local", ".env.production", ".env.development", ".env.prod", ".env.dev"],
    ),
    (
        "keys_and_credentials",
        [
            "*.pem",
            "*.key",
            "*.p12",
            "*.pfx",
            "id_rsa",
            "id_dsa",
            "id_ed25519",
            "*/id_rsa",
            "credentials.json",
            "service-account*.json",
            "*.keystore",
            ".npmrc",
            ".pypirc",
            ".netrc",
        ],
    ),
    ("notebook_checkpoints", [".ipynb_checkpoints/*", "*/.ipynb_checkpoints/*"]),
]
ENV_TEMPLATE_NAMES = {".env.example", ".env.sample", ".env.template", ".env.dist", ".env.defaults"}
SKIP_SECRET_SCAN_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".ico",
    ".pdf",
    ".zip",
    ".gz",
    ".lock",
    ".svg",
    ".woff",
    ".woff2",
    ".ttf",
    ".min.js",
    ".map",
}
UNTRUSTED_CONTEXTS = re.compile(
    r"\$\{\{\s*(github\.event\.(?:issue\.(?:title|body)|pull_request\.(?:title|body|head\.ref|head\.label)|"
    r"comment\.body|review\.body|review_comment\.body|pages\.[^}]*|commits\.[^}]*|head_commit\.(?:message|author)[^}]*|"
    r"workflow_run\.(?:head_branch|head_commit)[^}]*)|github\.head_ref)\s*\}\}"
)


class ToolError(Exception):
    """Raised for invalid tool input; the message is shown to the LLM."""


def _ruff_command() -> list[str] | None:
    """Command that runs the ruff installed for this interpreter, or None when ruff is not installed.

    Tried in order: ``python -I -m ruff`` for the running interpreter (works when a virtual environment's
    Scripts/bin directory is not on PATH), the ``ruff`` executable next to the interpreter, then ``ruff`` on PATH.

    ``-I`` (isolated mode) is essential: the command runs with the audited repository as its working directory, and
    without it a ``ruff.py`` or ``ruff/`` inside that repository would shadow the real package and execute
    repository code.
    """
    if sys.executable and importlib.util.find_spec("ruff") is not None:
        return [sys.executable, "-I", "-m", "ruff"]
    if sys.executable:
        sibling = Path(sys.executable).parent / ("ruff.exe" if os.name == "nt" else "ruff")
        if sibling.is_file():
            return [str(sibling)]
    on_path = shutil.which("ruff")
    return [on_path] if on_path else None


class RepoTools:
    def __init__(
        self,
        root: Path | str,
        max_output_chars: int = DEFAULT_OUTPUT_CHARS,
        command_timeout: int = 60,
        sandbox: str | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        if not (self.root / ".git").exists():
            raise ValueError(f"{self.root} is not a git checkout")
        self.max_output_chars = max_output_chars
        self.command_timeout = command_timeout
        self.sandbox = sandbox
        self.observed_files: set[str] = set()
        self._tracked: list[str] | None = None
        self._line_counts: dict[str, int] = {}

    # ------------------------------------------------------------------ helpers
    def _run(self, args: list[str], timeout: int | None = None) -> subprocess.CompletedProcess[str]:
        env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_OPTIONAL_LOCKS": "0", "LC_ALL": "C"}
        return subprocess.run(
            args,
            cwd=self.root,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout or self.command_timeout,
            env=env,
            check=False,
        )

    def _git(self, *args: str, timeout: int | None = None) -> str:
        proc = self._run(["git", "-c", "core.quotepath=false", *args], timeout=timeout)
        if proc.returncode not in (0, 1):  # grep returns 1 for "no match"
            raise ToolError(f"git {args[0]} failed: {proc.stderr.strip()[:300]}")
        return proc.stdout

    def _finish(self, text: str) -> str:
        text = redact(text)
        if len(text) > self.max_output_chars:
            cut = self.max_output_chars
            text = text[:cut] + f"\n... [output truncated at {cut} chars; narrow the request to see more]"
        return text

    def tracked_files(self) -> list[str]:
        if self._tracked is None:
            out = self._git("ls-files", "-z")
            self._tracked = sorted(p for p in out.split("\0") if p)
        return self._tracked

    def _safe_path(self, path: str) -> Path:
        if not isinstance(path, str) or not path.strip():
            raise ToolError("path must be a non-empty string")
        rel = path.strip().lstrip("/")
        if rel.startswith("./"):
            rel = rel[2:]
        target = (self.root / rel).resolve()
        if target != self.root and self.root not in target.parents:
            raise ToolError("path escapes the repository root")
        if ".git" in PurePosixPath(rel).parts:
            raise ToolError("the .git directory is not inspectable; use git_log instead")
        return target

    def normalize_path(self, path: str | None) -> str | None:
        if path is None:
            return None
        p = path.strip().replace("\\", "/")
        while p.startswith("./"):
            p = p[2:]
        return p.lstrip("/") or None

    def line_count(self, rel: str) -> int | None:
        if rel in self._line_counts:
            return self._line_counts[rel]
        try:
            target = self._safe_path(rel)
            if not target.is_file():
                return None
            with target.open("rb") as fh:
                n = sum(1 for _ in fh)
        except (ToolError, OSError):
            return None
        self._line_counts[rel] = n
        return n

    def _read_text(self, target: Path) -> str:
        data = target.read_bytes()[:MAX_FILE_BYTES]
        if b"\0" in data[:8000]:
            raise ToolError("binary file; not displayed")
        return data.decode("utf-8", errors="replace")

    # -------------------------------------------------------------------- tools
    def list_files(self, path: str = ".", pattern: str | None = None, max_entries: int = 300) -> str:
        """List tracked files (optionally under ``path`` and matching a glob)."""
        prefix = "" if path in (".", "", "/") else self.normalize_path(path).rstrip("/") + "/"
        files = [f for f in self.tracked_files() if f.startswith(prefix)]
        if pattern:
            files = [f for f in files if fnmatch.fnmatch(f, pattern) or fnmatch.fnmatch(PurePosixPath(f).name, pattern)]
        total = len(files)
        if not files:
            return f"No tracked files under '{path}'" + (f" matching '{pattern}'" if pattern else "") + "."
        max_entries = max(1, min(int(max_entries), 1000))
        header = f"{total} tracked file(s) under '{path or '.'}'" + (f" matching '{pattern}'" if pattern else "")
        lines = [header]
        if total > max_entries:
            # Summarize by directory so huge repos stay readable.
            dirs = Counter(str(PurePosixPath(f).parent) for f in files)
            lines.append(f"(showing first {max_entries} files; directory summary below)")
            lines.extend(files[:max_entries])
            lines.append("\nFiles per directory (top 40):")
            lines.extend(f"  {d}/: {n}" for d, n in dirs.most_common(40))
        else:
            lines.extend(files)
        return self._finish("\n".join(lines))

    def read_file(self, path: str, start_line: int = 1, end_line: int | None = None) -> str:
        """Return numbered lines of a file (max 400 lines per call)."""
        target = self._safe_path(path)
        rel = target.relative_to(self.root).as_posix()
        if not target.exists():
            close = [f for f in self.tracked_files() if PurePosixPath(f).name.lower() == PurePosixPath(rel).name.lower()]
            hint = f" Similar tracked files: {', '.join(close[:5])}" if close else ""
            return f"File not found: {rel}.{hint}"
        if target.is_dir():
            return f"{rel} is a directory; use list_files(path='{rel}')."
        try:
            text = self._read_text(target)
        except ToolError as exc:
            # The model has now seen that the file exists and is binary, which is all a finding about it
            # (for example a tracked .pyc) can rest on.
            self.observed_files.add(rel)
            return f"{rel}: {exc}"
        lines = text.splitlines()
        if lines and is_env_style(rel):
            # Redact before line numbers are prefixed: the env pattern is anchored to the start of a line.
            lines = redact_env_values("\n".join(lines)).split("\n")
        total = len(lines)
        start = max(1, int(start_line or 1))
        end = min(total, int(end_line) if end_line else start + MAX_READ_LINES - 1, start + MAX_READ_LINES - 1)
        self.observed_files.add(rel)
        tracked = rel in set(self.tracked_files())
        header = f"{rel} ({total} lines{'' if tracked else ', NOT tracked by git'}) showing {start}-{end}:"
        body = "\n".join(f"{i:>5}| {lines[i - 1][:500]}" for i in range(start, end + 1))
        more = f"\n... {total - end} more line(s); call read_file with start_line={end + 1}" if end < total else ""
        return self._finish(f"{header}\n{body}{more}")

    def search_code(self, pattern: str, path: str | None = None, case_insensitive: bool = False, max_results: int = 80) -> str:
        """Extended-regex search over tracked text files (git grep)."""
        if not pattern or len(pattern) > 300:
            raise ToolError("pattern must be 1-300 characters")
        args = ["grep", "-n", "-I", "-E", "--no-color", "--full-name"]
        if case_insensitive:
            args.append("-i")
        args += ["-e", pattern, "--"]
        if path:
            self._safe_path(path)
            args.append(self.normalize_path(path))
        try:
            out = self._git(*args, timeout=30)
        except subprocess.TimeoutExpired:
            return "Search timed out; use a more specific pattern or path."
        except ToolError as exc:
            return f"Search failed (check regex syntax, POSIX ERE): {exc}"
        hits = [h for h in out.splitlines() if h]
        if not hits:
            return f"No matches for /{pattern}/" + (f" in {path}" if path else "") + "."
        max_results = max(1, min(int(max_results), 300))
        shown = []
        for h in hits[:max_results]:
            file_part = h.split(":", 1)[0]
            self.observed_files.add(file_part)
            hit = re.match(r"([^:]+:\d+:)(.*)$", h)
            if hit and is_env_style(file_part):
                h = hit.group(1) + redact_env_values(hit.group(2))
            shown.append(h[:400])
        files = Counter(h.split(":", 1)[0] for h in hits)
        summary = f"{len(hits)} match(es) in {len(files)} file(s)"
        if len(hits) > max_results:
            summary += f" (showing {max_results})"
        return self._finish(summary + ":\n" + "\n".join(shown))

    def git_log(self, max_count: int = 30, path: str | None = None, show_files: bool = False) -> str:
        """Recent commit history (the clone is shallow, so very old history may be missing)."""
        max_count = max(1, min(int(max_count), 200))
        args = ["log", f"-n{max_count}", "--date=short", "--format=%h %ad %an | %s"]
        if show_files:
            args.append("--name-status")
        if path:
            args += ["--", self.normalize_path(path)]
        out = self._git(*args)
        total = self._git("rev-list", "--count", "HEAD").strip()
        shallow = (self.root / ".git" / "shallow").exists()
        note = f"Commits reachable: {total}{' (shallow clone; older history not fetched)' if shallow else ''}\n"
        return self._finish(note + (out or "No commits found."))

    def git_show_commit(self, commit: str) -> str:
        """Show the file-level diff summary and patch of one commit (truncated)."""
        if not re.fullmatch(r"[0-9a-fA-F]{4,40}", commit or ""):
            raise ToolError("commit must be a hex commit hash")
        out = self._git("show", "--stat", "--patch", "--format=%h %ad %an%n%s%n%b", "--date=short", commit)
        return self._finish(self._redact_diff(out))

    @staticmethod
    def _redact_diff(patch: str) -> str:
        """Redact env-style assignments on added/removed/context lines; the diff marker hides them from the line anchor."""
        lines = patch.split("\n")
        current = old = ""
        in_header = False  # file headers sit between "diff --git" and the first "@@"; content lines never do
        for i, line in enumerate(lines):
            if line.startswith("diff --git "):
                in_header, current, old = True, "", ""
            elif in_header:
                if line.startswith("--- "):
                    old = line[4:].removeprefix("a/")
                elif line.startswith("+++ "):
                    new = line[4:].removeprefix("b/")
                    current = old if new == "/dev/null" else new
                elif line.startswith("@@"):
                    in_header = False
            elif line[:1] in ("+", "-", " ") and is_env_style(current):
                lines[i] = line[:1] + redact_env_values(line[1:])
        return "\n".join(lines)

    def inspect_tracked_artifacts(self) -> str:
        """Find tracked files that are typically generated, local, temporary or secret,
        plus tracked files that the repository's own .gitignore says should be ignored."""
        tracked = self.tracked_files()
        report: list[str] = []
        for label, globs in SUSPICIOUS_TRACKED:
            matches = [
                f
                for f in tracked
                if any(fnmatch.fnmatch(f, g) or fnmatch.fnmatch(PurePosixPath(f).name, g) for g in globs)
                and PurePosixPath(f).name not in ENV_TEMPLATE_NAMES
            ]
            if matches:
                report.append(
                    f"[{label}] {len(matches)} tracked: " + ", ".join(matches[:15]) + (" ..." if len(matches) > 15 else "")
                )
                self.observed_files.update(matches[:15])  # only the files this output actually names
        ignored = [p for p in self._git("ls-files", "-ci", "--exclude-standard").splitlines() if p]
        if ignored:
            report.append(
                f"[tracked_but_gitignored] {len(ignored)} file(s) tracked despite matching .gitignore: " + ", ".join(ignored[:15])
            )
            self.observed_files.update(ignored[:15])
        gi = self.root / ".gitignore"
        report.append(".gitignore present at root: " + ("yes" if gi.is_file() else "NO"))
        large = []
        for f in tracked:
            try:
                size = (self.root / f).stat().st_size
            except OSError:
                continue
            if size > 5_000_000:
                large.append(f"{f} ({size // 1_000_000} MB)")
        if large:
            report.append("[large_files>5MB] " + ", ".join(large[:10]))
            self.observed_files.update(entry.rsplit(" (", 1)[0] for entry in large[:10])
        if len(report) == 1:
            report.insert(0, "No suspicious tracked artifacts found.")
        report.append("Note: verify context before reporting (e.g. fixtures, intentionally vendored files, sample data).")
        return self._finish("\n".join(report))

    def scan_secrets(self, path: str | None = None) -> str:
        """Heuristic credential scan over tracked text files. Values are always redacted."""
        scope = self._safe_path(self.normalize_path(path) or ".") if path else self.root
        prefix = scope.relative_to(self.root).as_posix()
        if prefix == ".":
            prefix = ""
        results: list[str] = []
        placeholder_count = 0
        for f in self.tracked_files():
            if prefix and f != prefix and not f.startswith(prefix + "/"):
                continue
            if any(f.lower().endswith(s) for s in SKIP_SECRET_SCAN_SUFFIXES):
                continue
            try:
                target = self._safe_path(f)
                if not target.is_file() or target.stat().st_size > MAX_FILE_BYTES:
                    continue
                text = self._read_text(target)
            except (ToolError, OSError):
                continue
            for hit in find_secrets(text, env_style=is_env_style(f)):
                if hit.likely_placeholder:
                    placeholder_count += 1
                    continue
                self.observed_files.add(f)
                low = f.lower()
                context = (
                    " (test/docs/example path: likely fixture)"
                    if re.search(r"(^|/)(tests?|testing|docs?|examples?|samples?|fixtures?)/|(^|/)test_|_test\.|\.(md|rst)$", low)
                    else ""
                )
                results.append(f"{f}:{hit.line} [{hit.kind}]{context} {hit.redacted_line}")
        if not results:
            msg = "No likely real credentials found."
        else:
            msg = f"{len(results)} candidate credential(s) (values redacted):\n" + "\n".join(results[:100])
        if placeholder_count:
            msg += f"\n({placeholder_count} placeholder/example-looking value(s) ignored)"
        msg += "\nHeuristic only: confirm with read_file; test fixtures and documentation examples are usually not violations."
        return self._finish(msg)

    def inspect_workflows(self, path: str | None = None) -> str:
        """Structured summary of GitHub Actions workflows with notable risk markers."""
        wf_files = [f for f in self.tracked_files() if f.startswith(".github/workflows/") and f.endswith((".yml", ".yaml"))]
        if path:
            wf_files = [f for f in wf_files if f == self.normalize_path(path)]
        others = [
            f
            for f in self.tracked_files()
            if f
            in (
                ".gitlab-ci.yml",
                "Jenkinsfile",
                ".travis.yml",
                "azure-pipelines.yml",
                "bitbucket-pipelines.yml",
                ".circleci/config.yml",
            )
        ]
        if not wf_files:
            msg = "No GitHub Actions workflows found in .github/workflows/."
            if others:
                msg += f" Other CI configuration present (equivalent CI): {', '.join(others)}"
            return msg
        out: list[str] = []
        for f in wf_files:
            self.observed_files.add(f)
            raw = self._read_text(self.root / f)
            lines = raw.splitlines()
            out.append(f"=== {f} ({len(lines)} lines)")
            try:
                doc = yaml.safe_load(raw) or {}
            except yaml.YAMLError as exc:
                out.append(f"  YAML PARSE ERROR: {str(exc)[:200]}")
                continue
            if not isinstance(doc, dict):
                out.append("  Unexpected YAML structure")
                continue
            triggers = doc.get("on", doc.get(True))  # PyYAML parses bare `on` as True
            out.append(f"  name: {doc.get('name')}")
            out.append(f"  triggers: {json.dumps(triggers, default=str)[:600]}")
            out.append(
                f"  top-level permissions: {json.dumps(doc.get('permissions', 'NOT SET (repository default applies)'), default=str)}"
            )
            jobs = doc.get("jobs") or {}
            if not isinstance(jobs, dict):
                jobs = {}
            for job_id, job in jobs.items():
                if not isinstance(job, dict):
                    continue
                meta = []
                for key in ("if", "needs", "permissions", "environment", "continue-on-error", "runs-on", "strategy"):
                    if key in job:
                        meta.append(f"{key}={json.dumps(job[key], default=str)[:200]}")
                if "uses" in job:
                    meta.append(f"reusable_workflow={job['uses']}")
                out.append(f"  job {job_id}: " + "; ".join(meta))
                for step in job.get("steps") or []:
                    if not isinstance(step, dict):
                        continue
                    desc = step.get("name") or step.get("uses") or (str(step.get("run", ""))[:60])
                    parts = [f"    - {desc}"]
                    if "uses" in step:
                        parts.append(f"uses={step['uses']}")
                    if "run" in step:
                        run = " ".join(str(step["run"]).split())
                        parts.append(f"run=`{run[:220]}`")
                    for key in ("if", "continue-on-error", "with", "env", "working-directory"):
                        if key in step:
                            parts.append(f"{key}={json.dumps(step[key], default=str)[:200]}")
                    out.append(" ".join(parts))
            markers = self._workflow_markers(raw, lines)
            if markers:
                out.append("  markers (verify each before reporting):")
                out.extend(f"    * {m}" for m in markers)
        if others:
            out.append(f"Other CI configuration present: {', '.join(others)}")
        out.append("Note: branch protection / required reviews are remote settings and cannot be seen here.")
        return self._finish("\n".join(out))

    @staticmethod
    def _workflow_markers(raw: str, lines: list[str]) -> list[str]:
        markers = []
        checks = [
            (r"pull_request_target", "pull_request_target trigger (dangerous if PR code is checked out/executed)"),
            (r"workflow_run", "workflow_run trigger (runs with base-repo privileges)"),
            (r"continue-on-error:\s*true", "continue-on-error: true"),
            (r"\|\|\s*(true|exit 0|:)\b", "failure suppressed with `|| true`/`|| exit 0`"),
            (r"set \+e", "`set +e` disables fail-fast"),
            (r"\bif:\s*(false|\$\{\{\s*false\s*\}\})", "step/job disabled with if: false"),
            (r"write-all", "permissions: write-all"),
            (
                r"(contents|pull-requests|actions|packages|id-token|deployments|issues|checks|statuses):\s*write",
                "write permission granted",
            ),
            (r"uses:\s*[^\s@]+@(main|master|dev|develop|latest|HEAD)\b", "action pinned to a mutable branch ref"),
            (r"uses:\s*[^\s@./][^\s@]*\s*$", "action used without any version ref"),
            (r"github\.event\.pull_request\.head\.(sha|ref)", "PR head ref/sha referenced (check with pull_request_target)"),
            (r"echo .*\$\{\{\s*secrets\.", "secret echoed in a run step"),
            (
                r"(--exit-zero|--no-verify|-k\s+['\"]?not\b|--deselect|pytest.*--co\b|\|\|\s*echo)",
                "flag that may neutralize validation",
            ),
            (r"(automerge|auto-merge|merge --auto|gh pr merge|gh pr review --approve)", "automatic PR merge/approval"),
            (r"git push( --force| -f)?", "workflow pushes to the repository"),
            (r"(curl|wget)[^|\n]*\|\s*(ba|z)?sh", "pipes a downloaded script into a shell"),
            (r"secrets:\s*inherit", "secrets: inherit to reusable workflow"),
            (
                r"(ubuntu-(16|18|20)\.04|python-version:\s*['\"]?(2\.7|3\.[5-8])\b|node-version:\s*['\"]?(1[0-6]|[0-9])\b)",
                "end-of-life runner/runtime version",
            ),
            (
                r"actions/(checkout|setup-python|setup-node|cache|upload-artifact|download-artifact)@v[12]\b",
                "deprecated major version of an official action",
            ),
        ]
        for pattern, label in checks:
            for i, line in enumerate(lines, start=1):
                if re.search(pattern, line, flags=re.IGNORECASE):
                    markers.append(f"line {i}: {label}: {line.strip()[:160]}")
        for i, line in enumerate(lines, start=1):
            if UNTRUSTED_CONTEXTS.search(line) and "run:" in raw:
                markers.append(
                    f"line {i}: untrusted event data interpolated with ${{{{ }}}} (script injection if inside run:): {line.strip()[:160]}"
                )
        return markers[:60]

    def run_ruff(self, path: str | None = None) -> str:
        """Run the ruff linter (static analysis only; does not execute repository code)."""
        ruff = _ruff_command()
        if not ruff:
            return "ruff is not installed in the agent environment; this check is unavailable (do not infer results)."
        py_files = [f for f in self.tracked_files() if f.endswith(".py")]
        if not py_files:
            return "No Python files tracked; ruff not applicable."
        target = self.normalize_path(path) if path else "."
        if path:
            self._safe_path(path)
        has_config = any((self.root / c).is_file() for c in ("ruff.toml", ".ruff.toml")) or "[tool.ruff" in self._safe_read(
            "pyproject.toml"
        )
        args = [*ruff, "check", "--no-cache", "--exit-zero", "--output-format", "concise", "--force-exclude"]
        if not has_config:
            args.append("--isolated")  # ruff defaults only; ignore any user-level config on the agent host
        args.append(target)
        try:
            proc = self._run(args, timeout=120)
        except subprocess.TimeoutExpired:
            return "ruff timed out; check unavailable."
        lines = [ln for ln in proc.stdout.splitlines() if re.match(r".+:\d+:\d+: ", ln)]
        codes = Counter(ln.split(": ", 1)[1].split(" ", 1)[0] for ln in lines if ": " in ln)
        head = (
            f"ruff ({'repository config' if has_config else 'ruff default rules E4,E7,E9,F; repo has no ruff config'}): "
            f"{len(lines)} issue(s) across {len(py_files)} Python file(s)."
        )
        if proc.returncode != 0 and not lines:
            return self._finish(head + f"\nruff error: {proc.stderr.strip()[:500]}")
        summary = ", ".join(f"{c}x{n}" for c, n in codes.most_common(15))
        for ln in lines[:60]:
            reported = self.normalize_path(ln.split(":", 1)[0])  # ruff prints backslashes on Windows
            if reported:
                self.observed_files.add(reported)
        return self._finish(
            f"{head}\nBy rule: {summary or 'none'}\n"
            + "\n".join(lines[:60])
            + "\nNote: F821/F401/E9 (undefined names, syntax errors) are strong signals; style-only codes are weak."
        )

    def _safe_read(self, rel: str) -> str:
        try:
            p = self._safe_path(rel)
            return self._read_text(p) if p.is_file() else ""
        except (ToolError, OSError):
            return ""

    def run_tests(self, command: str = "pytest") -> str:
        """Run the Python test suite inside an isolated Docker container (opt-in only)."""
        if self.sandbox != "docker":
            return "Test execution is disabled (no sandbox configured). Do not claim tests pass or fail."
        if not shutil.which("docker"):
            return "Docker is not available; test execution unavailable."
        if command not in ("pytest", "unittest"):
            raise ToolError("command must be 'pytest' or 'unittest'")
        runner = (
            "python -m pytest -q -p no:cacheprovider --no-header -x 2>&1 | tail -n 60"
            if command == "pytest"
            else "python -m unittest discover -v 2>&1 | tail -n 60"
        )
        script = (
            "set -u; cp -r /src /work && cd /work && "
            "(if [ -f requirements.txt ]; then pip install -q -r requirements.txt; "
            "elif [ -f pyproject.toml ] || [ -f setup.py ]; then pip install -q -e .; fi) >/tmp/install.log 2>&1 "
            "|| { echo INSTALL_FAILED; tail -n 20 /tmp/install.log; }; "
            "pip install -q pytest >/dev/null 2>&1; timeout 300 sh -c '" + runner + "'; echo EXIT_CODE=$?"
        )
        args = [
            "docker",
            "run",
            "--rm",
            "--memory",
            "2g",
            "--cpus",
            "2",
            "--pids-limit",
            "512",
            "--security-opt",
            "no-new-privileges",
            "--cap-drop",
            "ALL",
            "-v",
            f"{self.root}:/src:ro",
            "python:3.12-slim",
            "sh",
            "-c",
            script,
        ]
        try:
            proc = subprocess.run(args, capture_output=True, text=True, timeout=900, check=False)
        except subprocess.TimeoutExpired:
            return "Sandboxed test run timed out."
        return self._finish((proc.stdout + proc.stderr)[-6000:])


# ---------------------------------------------------------------------------
# LLM-facing schemas (provider-neutral JSON Schema) and dispatch.
# ---------------------------------------------------------------------------
TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "list_files",
        "description": "List git-tracked files. Use path to focus on a directory and pattern (glob, e.g. '*.yml', 'requirements*') to filter.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory relative to repo root, default '.'"},
                "pattern": {"type": "string", "description": "Optional glob matched against path or file name"},
                "max_entries": {"type": "integer", "description": "Default 300"},
            },
        },
    },
    {
        "name": "read_file",
        "description": "Read a file with line numbers (max 400 lines per call). Always read a file before reporting a finding in it.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "start_line": {"type": "integer"},
                "end_line": {"type": "integer"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "search_code",
        "description": "Search tracked text files with a POSIX extended regex (git grep -E). Returns file:line:content.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string", "description": "Optional file or directory to restrict the search"},
                "case_insensitive": {"type": "boolean"},
                "max_results": {"type": "integer"},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "git_log",
        "description": "Show recent commit history, optionally for one path and with changed files per commit.",
        "parameters": {
            "type": "object",
            "properties": {
                "max_count": {"type": "integer"},
                "path": {"type": "string"},
                "show_files": {"type": "boolean"},
            },
        },
    },
    {
        "name": "git_show_commit",
        "description": "Show the patch of a single commit (e.g. to see whether a secret was added or removed).",
        "parameters": {"type": "object", "properties": {"commit": {"type": "string"}}, "required": ["commit"]},
    },
    {
        "name": "inspect_tracked_artifacts",
        "description": "Detect tracked generated/temporary/local/secret files (bytecode, venvs, node_modules, build output, caches, IDE settings, .env, keys) and files tracked despite .gitignore.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "scan_secrets",
        "description": "Heuristic scan of tracked files for credentials (values are redacted). Confirm hits with read_file.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
    },
    {
        "name": "inspect_workflows",
        "description": "Parse GitHub Actions workflows: triggers, permissions, jobs, steps and risk markers. Also reports other CI systems.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Optional single workflow file"}},
        },
    },
    {
        "name": "run_ruff",
        "description": "Run the ruff Python linter statically (no code execution) over the repo or a path.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
    },
]

RUN_TESTS_SCHEMA = {
    "name": "run_tests",
    "description": "Run the Python test suite inside an isolated Docker container. Slow; use once at most.",
    "parameters": {"type": "object", "properties": {"command": {"type": "string", "enum": ["pytest", "unittest"]}}},
}


def tool_schemas(tools: RepoTools) -> list[dict[str, Any]]:
    schemas = list(TOOL_SCHEMAS)
    if tools.sandbox == "docker":
        schemas.append(RUN_TESTS_SCHEMA)
    return schemas


def _json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int | float):
        return "number"
    return {str: "string", dict: "object", list: "array"}.get(type(value), type(value).__name__)


def _coerce_argument(value: Any, spec: dict[str, Any]) -> tuple[Any, str | None]:
    """Check one LLM-provided argument against its schema entry; returns (value, error).

    Accepts what model clients commonly send for numbers and flags (``"5"`` or ``5.0`` for an integer,
    ``"true"`` for a boolean) and rejects everything else, so tools only ever see the declared types.
    """
    expected = spec.get("type", "string")
    if expected == "string" and isinstance(value, str):
        if "\0" in value:
            return value, "must not contain NUL characters"
        if "enum" in spec and value not in spec["enum"]:
            return value, f"must be one of: {', '.join(spec['enum'])}"
        return value, None
    if expected == "integer" and not isinstance(value, bool):
        if isinstance(value, int):
            return value, None
        if isinstance(value, float) and value.is_integer():
            return int(value), None
        if isinstance(value, str) and re.fullmatch(r"-?\d+", value.strip()):
            return int(value), None
    if expected == "boolean":
        if isinstance(value, bool):
            return value, None
        if isinstance(value, str) and value.strip().lower() in ("true", "false"):
            return value.strip().lower() == "true", None
    if expected not in ("string", "integer", "boolean"):
        return value, None  # no other parameter types are declared today; leave them to the tool
    return value, f"expected {expected}, got {_json_type(value)}"


def dispatch(tools: RepoTools, name: str, arguments: dict[str, Any]) -> str:
    """Execute a tool by name with LLM-provided arguments; errors become text for the LLM.

    Arguments are validated against the tool's schema first, so a malformed call yields a descriptive
    ``ERROR:`` result the model can correct. Unexpected exceptions inside a tool are deliberately not caught here.
    """
    schemas = tool_schemas(tools)
    allowed = {s["name"] for s in schemas}
    if name not in allowed:
        return f"ERROR: unknown tool '{name}'. Available: {', '.join(sorted(allowed))}"
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        return f"ERROR: arguments for {name} must be a JSON object, got {_json_type(arguments)}"
    fn: Callable[..., str] = getattr(tools, name)
    schema = next(s for s in schemas if s["name"] == name)
    props = schema["parameters"].get("properties", {})
    provided = {k: v for k, v in arguments.items() if v is not None}
    missing = [r for r in schema["parameters"].get("required", []) if r not in provided]
    if missing:
        return f"ERROR: missing required argument(s): {', '.join(missing)}"
    clean: dict[str, Any] = {}
    problems: list[str] = []
    for key, value in provided.items():
        if key not in props:
            continue
        coerced, error = _coerce_argument(value, props[key])
        if error:
            problems.append(f"'{key}' {error}")
        else:
            clean[key] = coerced
    if problems:
        return f"ERROR: invalid argument(s) for {name}: " + "; ".join(problems)
    unknown = sorted(str(k)[:40] for k in provided if k not in props)
    try:
        result = fn(**clean)
    except ToolError as exc:
        return f"ERROR: {exc}"
    except subprocess.TimeoutExpired:
        return f"ERROR: {name} timed out"
    except OSError as exc:
        return f"ERROR: {name} failed: {exc.strerror or exc}"
    if unknown:
        result += f"\n(note: ignored unknown argument(s): {', '.join(unknown)}; valid: {', '.join(props) or 'none'})"
    return result
