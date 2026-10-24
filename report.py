"""Finding validation, de-duplication and JSON report writing."""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from prompts import CATEGORIES, CATEGORY_ALIASES, RULE_PREFIX_TO_CATEGORY
from redaction import redact

SEVERITIES = ("critical", "high", "medium", "low")
REPORT_FIELDS = ("rule_id", "category", "severity", "file", "line", "description", "evidence", "recommendation")
RULE_ID_RE = re.compile(r"^[A-Z]{2,5}-\d{3}$")
SEVERITY_ORDER = {s: i for i, s in enumerate(SEVERITIES)}


class FindingRejected(ValueError):
    """Raised when a proposed finding fails validation; the message goes back to the LLM."""


@dataclass
class FindingStore:
    """Holds verified findings. ``tools`` must expose tracked_files(), observed_files,
    normalize_path() and line_count()."""

    tools: Any
    max_findings: int = 50
    findings: dict[str, dict[str, Any]] = field(default_factory=dict)
    retracted: dict[str, str] = field(default_factory=dict)
    _counter: int = 0

    def add(self, raw: dict[str, Any]) -> str:
        if len(self.findings) >= self.max_findings:
            raise FindingRejected(f"finding limit ({self.max_findings}) reached; record only the most important issues")
        f = self._validate(raw)
        dup = self._duplicate_of(f)
        if dup:
            raise FindingRejected(f"duplicate of {dup} (same rule/location or same defect); not recorded")
        self._counter += 1
        fid = f"F{self._counter}"
        self.findings[fid] = f
        return fid

    def retract(self, finding_id: str, reason: str) -> None:
        if finding_id not in self.findings:
            raise FindingRejected(f"no active finding with id {finding_id}; active ids: {', '.join(self.findings) or 'none'}")
        self.findings.pop(finding_id)
        self.retracted[finding_id] = reason

    def _validate(self, raw: dict[str, Any]) -> dict[str, Any]:
        missing = [k for k in REPORT_FIELDS if k not in raw]
        if missing:
            raise FindingRejected(f"missing field(s): {', '.join(missing)}")

        rule_id = str(raw["rule_id"]).strip().upper()
        if not RULE_ID_RE.match(rule_id):
            raise FindingRejected("rule_id must look like 'SEC-001'")

        category = str(raw["category"]).strip().lower().replace(" ", "_").replace("-", "_").replace("/", "_")
        category = CATEGORY_ALIASES.get(category, category)
        prefix_cat = RULE_PREFIX_TO_CATEGORY.get(rule_id.split("-")[0])
        if category not in CATEGORIES:
            if prefix_cat:
                category = prefix_cat
            else:
                raise FindingRejected(f"category must be one of: {', '.join(CATEGORIES)}")

        severity = str(raw["severity"]).strip().lower()
        if severity not in SEVERITIES:
            raise FindingRejected("severity must be critical, high, medium or low")

        file = raw["file"]
        if isinstance(file, str) and file.strip().lower() in ("", "null", "none", "n/a"):
            file = None
        line = raw["line"]
        if isinstance(line, str):
            line = int(line) if line.strip().isdigit() else None
        if isinstance(line, float) and line.is_integer():
            line = int(line)
        if isinstance(line, bool) or (line is not None and not isinstance(line, int)):
            raise FindingRejected("line must be an integer or null")
        if line == 0:
            line = None

        if file is not None:
            file = self.tools.normalize_path(str(file))
            tracked = set(self.tools.tracked_files())
            if file not in tracked:
                if any(t.startswith(file.rstrip("/") + "/") for t in tracked):
                    pass  # directory-level finding (e.g. tracked venv/); allowed, line must be null
                else:
                    raise FindingRejected(
                        f"'{file}' is not a tracked file. For something that is absent, use file=null and line=null."
                    )
            elif file not in self.tools.observed_files:
                raise FindingRejected(
                    f"you have not inspected '{file}' yet. Read or search it to verify the evidence, then record again."
                )
            if line is not None:
                n = self.tools.line_count(file)
                if n is None or line < 1 or line > max(n, 1):
                    raise FindingRejected(f"line {line} is outside '{file}' ({n or 0} lines); use a valid line or null")
        elif line is not None:
            line = None  # a line number without a file is meaningless

        out: dict[str, Any] = {
            "rule_id": rule_id,
            "category": category,
            "severity": severity,
            "file": file,
            "line": line,
        }
        for key in ("description", "evidence", "recommendation"):
            value = redact(" ".join(str(raw[key]).split()))
            if len(value) < 5:
                raise FindingRejected(f"{key} is too short to be useful")
            out[key] = value[:800]
        return out

    def _duplicate_of(self, f: dict[str, Any]) -> str | None:
        for fid, g in self.findings.items():
            if (
                g["file"] == f["file"]
                and g["line"] == f["line"]
                and (g["rule_id"] == f["rule_id"] or (f["line"] is not None and g["category"] == f["category"]))
            ):
                return fid
            if g["rule_id"] == f["rule_id"] and g["file"] == f["file"] and (g["line"] is None or f["line"] is None):
                return fid
            if f["file"] is None and g["file"] is None and g["rule_id"] == f["rule_id"]:
                return fid
        return None

    def as_report_list(self) -> list[dict[str, Any]]:
        items = sorted(
            self.findings.values(),
            key=lambda f: (SEVERITY_ORDER[f["severity"]], f["category"], f["file"] or "", f["line"] or 0),
        )
        return [{k: f[k] for k in REPORT_FIELDS} for f in items]

    def summary_text(self) -> str:
        if not self.findings:
            return "(none)"
        return "\n".join(
            f"{fid}: {f['rule_id']} [{f['severity']}] {f['file']}:{f['line']} - {f['description']}"
            for fid, f in self.findings.items()
        )


def build_report(repository: str, violations: list[dict[str, Any]]) -> dict[str, Any]:
    return {"repository": repository, "violations": violations}


def validate_report(report: dict[str, Any]) -> None:
    """Raise ValueError if ``report`` does not satisfy the output contract."""
    if not isinstance(report, dict) or set(report) != {"repository", "violations"}:
        raise ValueError("report must have exactly 'repository' and 'violations'")
    if not isinstance(report["repository"], str) or not isinstance(report["violations"], list):
        raise ValueError("bad top-level types")
    for v in report["violations"]:
        if set(v) != set(REPORT_FIELDS):
            raise ValueError(f"violation has wrong fields: {sorted(v)}")
        if v["severity"] not in SEVERITIES:
            raise ValueError(f"bad severity {v['severity']}")
        if v["file"] is not None and not isinstance(v["file"], str):
            raise ValueError("file must be string or null")
        if v["line"] is not None and (not isinstance(v["line"], int) or isinstance(v["line"], bool)):
            raise ValueError("line must be int or null")
        for k in ("rule_id", "category", "description", "evidence", "recommendation"):
            if not isinstance(v[k], str) or not v[k]:
                raise ValueError(f"{k} must be a non-empty string")


def write_json_atomic(path: Path | str, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
