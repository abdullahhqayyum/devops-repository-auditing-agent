import json

import pytest

from auditor.report import FindingRejected, FindingStore, build_report, validate_report, write_json_atomic
from tools import RepoTools


def finding(**kw):
    base = {
        "rule_id": "SEC-001",
        "category": "security",
        "severity": "critical",
        "file": "app/config.py",
        "line": 3,
        "description": "Hardcoded AWS key",
        "evidence": "AWS key literal assigned",
        "recommendation": "Use env vars",
    }
    base.update(kw)
    return base


@pytest.fixture
def store(sample_repo):
    tools = RepoTools(sample_repo)
    tools.read_file("app/config.py")
    return FindingStore(tools)


def test_accepts_valid_finding(store):
    assert store.add(finding()) == "F1"
    assert store.as_report_list()[0]["line"] == 3


def test_rejects_unobserved_file(store):
    with pytest.raises(FindingRejected, match="not inspected"):
        store.add(finding(file="app/main.py", line=1))


def test_rejects_untracked_file(store):
    with pytest.raises(FindingRejected, match="not a tracked file"):
        store.add(finding(file="nope.py"))


def test_rejects_out_of_range_line_and_bad_severity(store):
    with pytest.raises(FindingRejected, match="outside"):
        store.add(finding(line=999))
    with pytest.raises(FindingRejected, match="severity"):
        store.add(finding(severity="urgent"))


def test_null_file_and_line_allowed(store):
    store.add(finding(rule_id="TEST-001", category="testing", severity="medium", file=None, line=None))
    assert store.as_report_list()[0]["file"] is None


def test_duplicates_rejected(store):
    store.add(finding())
    with pytest.raises(FindingRejected, match="duplicate"):
        store.add(finding(description="Same thing again"))


def test_category_alias_and_rule_prefix(store):
    store.add(finding(category="Secrets", line=4, rule_id="CFG-001"))
    assert store.as_report_list()[0]["category"] == "security"


def test_retract(store):
    fid = store.add(finding())
    store.retract(fid, "SEC-001", "app/config.py", 3, "fixture")
    assert store.as_report_list() == []


def test_retract_requires_matching_identity(store):
    fid = store.add(finding())
    with pytest.raises(FindingRejected, match="does not match"):
        store.retract(fid, "HYG-001", "README.md", 1, "wrong target")
    assert list(store.findings) == [fid]
    store.retract(fid, "SEC-001", "app/config.py", 3, "correct target")
    assert store.as_report_list() == []


def test_retract_handles_null_identity(store):
    raw = finding(rule_id="TEST-001", category="testing", severity="medium", file=None, line=None)
    fid = store.add(raw)
    store.retract(fid, "TEST-001", None, None, "not applicable")
    assert store.as_report_list() == []


def test_retract_unknown_and_retracted_ids_are_safe(store):
    with pytest.raises(FindingRejected, match="no active finding"):
        store.retract("F9", "SEC-001", "app/config.py", 3, "unknown")
    fid = store.add(finding())
    store.retract(fid, "SEC-001", "app/config.py", 3, "first")
    with pytest.raises(FindingRejected, match="no active finding"):
        store.retract(fid, "SEC-001", "app/config.py", 3, "again")


def test_redacts_evidence(store):
    store.add(finding(evidence="value AKIA" + "Z3XQ9W7LMN2B4VCD found"))
    assert "Z3XQ9W7LMN2B4VCD" not in json.dumps(store.as_report_list())


def test_report_contract_and_atomic_write(tmp_path, store):
    store.add(finding())
    report = build_report("https://github.com/x/y", store.as_report_list())
    validate_report(report)
    out = tmp_path / "r.json"
    write_json_atomic(out, report)
    assert json.loads(out.read_text()) == report
    validate_report(build_report("https://github.com/x/y", []))
    with pytest.raises(ValueError):
        validate_report({"repository": "x", "violations": [{"rule_id": "x"}]})


def test_zero_line_and_empty_file_mean_null(store):
    store.add(finding(rule_id="TEST-001", category="testing", severity="medium", file="", line=0))
    v = store.as_report_list()[0]
    assert v["file"] is None and v["line"] is None
