"""Validate GitHub URLs and create a read-only, isolated checkout of the target repository."""

from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

GITHUB_URL = re.compile(
    r"^(?:https?://)?(?:www\.)?github\.com/(?P<owner>[A-Za-z0-9](?:[A-Za-z0-9-]{0,38}))/"
    r"(?P<repo>[A-Za-z0-9._-]{1,100}?)(?:\.git)?(?:/tree/(?P<ref>[^?#\s]+))?/?(?:[?#].*)?$"
)


class RepoError(RuntimeError):
    """Invalid repository input or clone failure."""


def parse_github_url(url: str) -> tuple[str, str | None]:
    """Return (canonical https URL, optional ref from a /tree/<ref> URL)."""
    if not isinstance(url, str) or not url.strip():
        raise RepoError("repository URL is empty")
    m = GITHUB_URL.match(url.strip())
    if not m or m.group("repo") in (".", ".."):
        raise RepoError(f"not a GitHub repository URL: {url!r} (expected https://github.com/<owner>/<repo>)")
    canonical = f"https://github.com/{m.group('owner')}/{m.group('repo')}"
    return canonical, m.group("ref")


def _git_env() -> dict[str, str]:
    return {
        **os.environ,
        "GIT_TERMINAL_PROMPT": "0",  # never prompt for credentials (private repos fail fast)
        "GIT_LFS_SKIP_SMUDGE": "1",  # do not download LFS blobs
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_ASKPASS": "echo",
    }


def clone_repository(url: str, dest_parent: Path | None = None, depth: int = 200, timeout: int = 300) -> Path:
    """Clone ``url`` into a fresh temporary directory and make the worktree read-only."""
    canonical, ref = parse_github_url(url)
    workdir = Path(tempfile.mkdtemp(prefix="devops-audit-", dir=dest_parent))
    target = workdir / "repo"
    args = [
        "git",
        "-c",
        "core.symlinks=false",
        "-c",
        "protocol.file.allow=never",
        "-c",
        "core.hooksPath=/dev/null",
        "clone",
        "--quiet",
        "--no-tags",
        f"--depth={depth}",
    ]
    if ref:
        args += ["--branch", ref]
    args += [canonical + ".git", str(target)]
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout, env=_git_env(), check=False)
    except subprocess.TimeoutExpired as exc:
        cleanup(workdir)
        raise RepoError(f"git clone timed out after {timeout}s") from exc
    except FileNotFoundError as exc:
        cleanup(workdir)
        raise RepoError("git is not installed or not on PATH") from exc
    if proc.returncode != 0:
        cleanup(workdir)
        err = proc.stderr.strip().splitlines()[-1:] or ["unknown error"]
        raise RepoError(f"could not clone {canonical}: {err[0]} (is the repository public and the URL correct?)")
    make_read_only(target)
    return target


def prepare_local_repository(path: str) -> Path:
    """Copy a local git repository (for development/evaluation only) into a temp dir, read-only."""
    src = Path(path).expanduser().resolve()
    if not (src / ".git").exists():
        raise RepoError(f"{src} is not a git repository")
    workdir = Path(tempfile.mkdtemp(prefix="devops-audit-"))
    target = workdir / "repo"
    proc = subprocess.run(
        ["git", "clone", "--quiet", "--no-hardlinks", str(src), str(target)],
        capture_output=True,
        text=True,
        timeout=120,
        env=_git_env(),
        check=False,
    )
    if proc.returncode != 0:
        cleanup(workdir)
        raise RepoError(f"could not copy local repository: {proc.stderr.strip()[:300]}")
    make_read_only(target)
    return target


def make_read_only(root: Path) -> None:
    """Remove write permission from every worktree file (the .git directory stays writable for git's own use)."""
    for dirpath, dirnames, filenames in os.walk(root):
        if ".git" in dirnames:
            dirnames.remove(".git")
        for name in filenames:
            p = Path(dirpath) / name
            try:
                if not p.is_symlink():
                    p.chmod(p.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
            except OSError:
                pass


def cleanup(path: Path) -> None:
    """Delete a temporary checkout, restoring write permission first."""
    path = Path(path)
    workdir = path.parent if path.name == "repo" else path
    if not workdir.name.startswith("devops-audit-"):
        return  # refuse to delete anything we did not create

    def _onerror(func, p, _exc):
        try:
            os.chmod(p, stat.S_IWUSR | stat.S_IRUSR | stat.S_IXUSR)
            parent = os.path.dirname(p)
            os.chmod(parent, stat.S_IWUSR | stat.S_IRUSR | stat.S_IXUSR)
            func(p)
        except OSError:
            pass

    if sys.version_info >= (3, 12):
        shutil.rmtree(workdir, onexc=_onerror)
    else:  # pragma: no cover
        shutil.rmtree(workdir, onerror=_onerror)
