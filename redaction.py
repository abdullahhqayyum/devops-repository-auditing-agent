"""Secret detection and redaction.

Every string that leaves a tool (and therefore reaches the LLM, the trace,
the logs or the final report) passes through :func:`redact`, so secret values
found in a target repository are never reproduced anywhere.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

# Well-known token formats. Each match is replaced entirely.
TOKEN_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("aws_access_key_id", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("github_token", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}\b")),
    ("github_fine_grained_pat", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{22,}\b")),
    ("gitlab_token", re.compile(r"\bglpat-[A-Za-z0-9_-]{20,}\b")),
    ("slack_token", re.compile(r"\bxox[abposr]-[A-Za-z0-9-]{10,}\b")),
    ("slack_webhook", re.compile(r"https://hooks\.slack\.com/services/[A-Za-z0-9/]{20,}")),
    ("stripe_key", re.compile(r"\b(?:sk|rk)_(?:live|test)_[A-Za-z0-9]{16,}\b")),
    ("anthropic_key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}")),
    ("openai_key", re.compile(r"\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{20,}")),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("sendgrid_key", re.compile(r"\bSG\.[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{16,}\b")),
    ("twilio_key", re.compile(r"\bSK[0-9a-fA-F]{32}\b")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}")),
    (
        "private_key_block",
        re.compile(
            r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP |ENCRYPTED )?PRIVATE KEY(?: BLOCK)?-----"
            r"(?:[\s\S]{0,8000}?-----END [A-Z ]*PRIVATE KEY(?: BLOCK)?-----|[\s\S]{0,4000})"
        ),
    ),
]

PRIVATE_KEY_HEADER = re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP |ENCRYPTED )?PRIVATE KEY(?: BLOCK)?-----")

# user:password@host inside connection strings / URLs (only the password is replaced).
URL_CREDENTIALS = re.compile(r"(?P<prefix>\b[a-zA-Z][a-zA-Z0-9+.-]*://[^\s:/@'\"]+:)(?P<value>[^\s@/'\"]{3,})(?P<suffix>@)")

# NAME = "literal" where NAME looks secret-ish (only the literal is replaced).
SECRET_NAME = (
    r"[A-Za-z0-9_.-]*(?:password|passwd|pwd|secret|token|api[_-]?key|apikey|access[_-]?key|"
    r"private[_-]?key|client[_-]?secret|auth[_-]?key|credential|signing[_-]?key)[A-Za-z0-9_.-]*"
)
QUOTED_ASSIGNMENT = re.compile(
    rf"""(?ix)(?P<name>\b{SECRET_NAME})(?P<op>["']?\s*(?:=|:=|:|=>)\s*)(?P<q>\\?["'])(?P<value>[^"'\n]{{4,}}?)(?P=q)"""
)
# NAME=value in .env-style files (unquoted).
ENV_ASSIGNMENT = re.compile(rf"""(?im)^(?P<name>\s*(?:export\s+)?{SECRET_NAME})(?P<op>\s*=\s*)(?P<value>[^\s#'"]{{4,}})""")

PLACEHOLDER_HINTS = (
    "changeme",
    "change_me",
    "change-me",
    "example",
    "placeholder",
    "your",
    "dummy",
    "fake",
    "sample",
    "redacted",
    "xxxx",
    "****",
    "todo",
    "replace",
    "insert",
    "<",
    "${",
    "{{",
    "%(",
    "os.environ",
    "getenv",
    "env(",
    "secrets.",
    "none",
    "null",
    "test",
    "mock",
    "foobar",
    "password123",
    "secret123",
)


@dataclass
class SecretHit:
    kind: str
    line: int
    likely_placeholder: bool
    redacted_line: str


def _entropy(value: str) -> float:
    if not value:
        return 0.0
    counts = {c: value.count(c) for c in set(value)}
    return -sum((n / len(value)) * math.log2(n / len(value)) for n in counts.values())


def looks_like_placeholder(value: str) -> bool:
    low = value.lower()
    # Known placeholder words, or trivially repetitive values such as "aaaa" / "xxxxxx" / "1111".
    return any(h in low for h in PLACEHOLDER_HINTS) or len(set(value)) <= 3


REFERENCE_MARKERS = ("${", "{{", "$(", "%(", "secrets.", "os.environ", "getenv", "process.env", "env(", "vars.")


def is_reference(value: str) -> bool:
    """True for values that reference a secret store/variable instead of containing a secret."""
    v = value.strip()
    return v.startswith(("$", "<")) or any(marker in v for marker in REFERENCE_MARKERS)


def redact(text: str) -> str:
    """Replace secret-looking values in ``text`` with ``[REDACTED:<kind>]``."""
    if not text:
        return text
    for kind, pattern in TOKEN_PATTERNS:
        text = pattern.sub(f"[REDACTED:{kind}]", text)
    text = URL_CREDENTIALS.sub(lambda m: f"{m.group('prefix')}[REDACTED:password]{m.group('suffix')}", text)

    def _quoted(m: re.Match[str]) -> str:
        value = m.group("value")
        if value.startswith("[REDACTED") or is_reference(value):
            return m.group(0)
        return f"{m.group('name')}{m.group('op')}{m.group('q')}[REDACTED:literal]{m.group('q')}"

    text = QUOTED_ASSIGNMENT.sub(_quoted, text)

    return ENV_ASSIGNMENT.sub(_redact_env_match, text)


def _redact_env_match(m: re.Match[str]) -> str:
    value = m.group("value")
    if value.startswith("[REDACTED") or is_reference(value):
        return m.group(0)
    return f"{m.group('name')}{m.group('op')}[REDACTED:value]"


def is_env_style(path: str) -> bool:
    """True for files whose unquoted ``NAME=value`` lines are configuration values (.env, .ini, .cfg, ...)."""
    name = path.replace("\\", "/").rsplit("/", 1)[-1]
    return name.startswith(".env") or name.endswith((".env", ".ini", ".cfg", ".properties", ".conf"))


def redact_env_values(text: str) -> str:
    """Redact unquoted ``NAME=value`` assignments in raw file content.

    ENV_ASSIGNMENT is anchored to the start of a line, so this must run on the file's own lines before a prefix
    such as a line number, ``file:line:`` or a diff marker is added. Apply it only to env-style files
    (:func:`is_env_style`): in source code the same pattern would hide ordinary assignments like ``token = get_token()``.
    """
    return ENV_ASSIGNMENT.sub(_redact_env_match, text)


def find_secrets(text: str, env_style: bool = False) -> list[SecretHit]:
    """Return candidate secrets per line. Values are never returned, only redacted lines."""
    hits: list[SecretHit] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if len(line) > 2000:
            line = line[:2000]
        kinds: list[tuple[str, bool]] = []
        for kind, pattern in TOKEN_PATTERNS:
            for m in pattern.finditer(line):
                kinds.append((kind, looks_like_placeholder(m.group(0))))
        if PRIVATE_KEY_HEADER.search(line):
            kinds.append(("private_key_block", False))
        for m in URL_CREDENTIALS.finditer(line):
            kinds.append(("url_embedded_password", looks_like_placeholder(m.group("value"))))
        for m in QUOTED_ASSIGNMENT.finditer(line):
            value = m.group("value")
            if is_reference(value):
                continue
            placeholder = (
                looks_like_placeholder(value)
                or (len(value) < 8 and _entropy(value) < 2.5)
                or bool(re.fullmatch(r"[a-z_]{1,24}", value))  # plain identifiers such as "security"
            )
            kinds.append(("hardcoded_secret_literal", placeholder))
        if env_style:
            for m in ENV_ASSIGNMENT.finditer(line):
                value = m.group("value")
                if is_reference(value):
                    continue
                kinds.append(("env_file_secret_value", looks_like_placeholder(value)))
        if kinds:
            # Collapse duplicates on the same line; a specific token type beats the generic heuristics.
            generic = {"hardcoded_secret_literal", "env_file_secret_value"}
            if any(k not in generic for k, _ in kinds):
                kinds = [(k, p) for k, p in kinds if k not in generic]
            seen: dict[str, bool] = {}
            for kind, placeholder in kinds:
                seen[kind] = seen.get(kind, True) and placeholder
            for kind, placeholder in seen.items():
                hits.append(SecretHit(kind, lineno, placeholder, redact(line.strip())[:300]))
    return hits


def redact_data(value: Any) -> Any:
    """Return a copy of ``value`` with every string, including those nested in dicts, lists and tuples, redacted."""
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        return {k: redact_data(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [redact_data(v) for v in value]
    return value
