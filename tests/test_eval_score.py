"""Deterministic tests for eval/score.py (loaded by path: eval/ is not a package)."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location("eval_score", Path(__file__).resolve().parents[1] / "eval" / "score.py")
eval_score = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = eval_score  # dataclasses look the defining module up here
_spec.loader.exec_module(eval_score)

KEY = [
    {"rule_id": "SEC-001", "file": "app/config.py"},
    {"rule_id": "CI-003", "file": ".github/workflows/ci.yml"},
    {"rule_id": "TEST-001", "file": None},
    {"rule_id": "DEP-002", "file": "requirements.txt"},
]


def violation(rule_id, file):
    return {
        "rule_id": rule_id,
        "category": "x",
        "severity": "low",
        "file": file,
        "line": None,
        "description": "d",
        "evidence": "e",
        "recommendation": "r",
    }


def report(*pairs, **extra):
    return {"repository": "https://github.com/x/y", "violations": [violation(*p) for p in pairs], **extra}


def test_perfect_detection():
    s = eval_score.score(report(*[(k["rule_id"], k["file"]) for k in KEY]), KEY, lenient=False)
    assert (s.tp, s.fp, s.fn) == (4, 0, 0)
    assert (s.precision, s.recall, s.f1) == (1.0, 1.0, 1.0)


def test_partial_detection_with_one_false_positive():
    s = eval_score.score(
        report(("SEC-001", "app/config.py"), ("CI-003", ".github/workflows/ci.yml"), ("CQ-002", "x.py")), KEY, False
    )
    assert (s.tp, s.fp, s.fn) == (2, 1, 2)
    assert s.precision == pytest.approx(2 / 3) and s.recall == 0.5
    assert s.f1 == pytest.approx(2 * (2 / 3) * 0.5 / (2 / 3 + 0.5))
    assert [v["rule_id"] for v in s.false_positives] == ["CQ-002"]
    assert [k["rule_id"] for k in s.missed] == ["TEST-001", "DEP-002"]


def test_only_false_positives_gives_zero_not_na():
    s = eval_score.score(report(("SEC-009", "a.py"), ("CQ-001", "b.py")), KEY, False)
    assert (s.tp, s.fp, s.fn) == (0, 2, 4)
    assert (s.precision, s.recall, s.f1) == (0.0, 0.0, 0.0)


def test_empty_report_has_undefined_precision_and_f1_but_zero_recall():
    s = eval_score.score(report(), KEY, False)
    assert (s.tp, s.fp, s.fn) == (0, 0, 4)
    assert s.precision is None and s.f1 is None  # never a misleading 1.00
    assert s.recall == 0.0


def test_healthy_repository_with_no_ground_truth():
    clean = eval_score.score(report(), [], False)
    assert (clean.tp, clean.fp, clean.fn) == (0, 0, 0)
    assert (clean.precision, clean.recall, clean.f1) == (None, None, None)
    noisy = eval_score.score(report(("SEC-001", "a.py"), ("CI-001", None)), [], False)
    assert (noisy.tp, noisy.fp, noisy.fn) == (0, 2, 0)  # the false-positive count stays meaningful
    assert noisy.precision == 0.0 and noisy.recall is None and noisy.f1 is None


def test_duplicate_findings_earn_no_credit_but_count_as_reported():
    s = eval_score.score(report(("SEC-001", "app/config.py"), ("SEC-001", "app/config.py")), KEY[:1], False)
    assert (s.tp, s.fp, s.fn) == (1, 1, 0)
    assert s.precision == 0.5 and s.recall == 1.0
    assert len(s.duplicates) == 1 and s.false_positives == []


def test_null_file_in_the_key_matches_null_or_empty_file():
    assert eval_score.score(report(("TEST-001", None)), [KEY[2]], False).tp == 1
    assert eval_score.score(report(("TEST-001", "")), [KEY[2]], False).tp == 1
    assert eval_score.score(report(("TEST-001", "tests/")), [KEY[2]], False).tp == 0


def test_strict_versus_lenient_matching():
    r = report(("SEC-002", "app/config.py"))  # same file and category, different rule id
    assert eval_score.score(r, KEY, lenient=False).tp == 0
    assert eval_score.score(r, KEY, lenient=True).tp == 1


def _run(tmp_path, capsys, rep, key=KEY):
    rp, kp = tmp_path / "r.json", tmp_path / "k.json"
    rp.write_text(json.dumps(rep))
    kp.write_text(json.dumps(key))
    code = eval_score.main([str(rp), str(kp)])
    return code, capsys.readouterr().out


def test_incomplete_report_is_labeled_and_gets_no_metrics(tmp_path, capsys):
    rep = report(("SEC-001", "app/config.py"), incomplete=True, stop_reason="LLM failure: quota exhausted")
    code, out = _run(tmp_path, capsys, rep)
    assert code == 2
    assert "INCOMPLETE" in out and "NOT a completed accuracy evaluation" in out and "quota exhausted" in out
    assert "precision=N/A recall=N/A f1=N/A" in out
    assert "precision=1.00" not in out and "TP=" not in out


def test_incomplete_empty_report_is_not_reported_as_perfect_precision(tmp_path, capsys):
    code, out = _run(tmp_path, capsys, report(incomplete=True))
    assert code == 2 and "INCOMPLETE" in out and "1.00" not in out


def test_complete_empty_report_prints_na_not_one(tmp_path, capsys):
    code, out = _run(tmp_path, capsys, report())
    assert code == 0 and "status: COMPLETE" in out
    assert "precision=N/A" in out and "precision=1.00" not in out
    assert "recall=0.00" in out and "f1=N/A" in out
    assert "missed known:     SEC-001 app/config.py" in out


def test_complete_report_output_lists_fp_duplicates_and_misses(tmp_path, capsys):
    rep = report(("SEC-001", "app/config.py"), ("SEC-001", "app/config.py"), ("CQ-002", "z.py"))
    code, out = _run(tmp_path, capsys, rep)
    assert code == 0
    assert "TP=1 FP=2 FN=3" in out
    assert "false positive:   CQ-002 z.py" in out and "duplicate:        SEC-001 app/config.py" in out


def test_healthy_run_output_reports_false_positive_count(tmp_path, capsys):
    code, out = _run(tmp_path, capsys, report(("SEC-001", "a.py")), key=[])
    assert code == 0 and "TP=0 FP=1 FN=0" in out and "recall=N/A" in out


def test_unreadable_inputs_exit_nonzero(tmp_path, capsys):
    assert eval_score.main([str(tmp_path / "missing.json"), str(tmp_path / "k.json")]) == 2
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"repository": "x"}))
    key = tmp_path / "k.json"
    key.write_text("[]")
    assert eval_score.main([str(bad), str(key)]) == 2
    assert "cannot read inputs" in capsys.readouterr().out


def test_answer_keys_produced_by_the_fixture_generator_are_untouched_by_scoring(tmp_path):
    spec = importlib.util.spec_from_file_location("make_demo", Path(__file__).resolve().parents[1] / "eval" / "make_demo_repo.py")
    demo = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = demo
    spec.loader.exec_module(demo)
    assert len(demo.DEFECTIVE_KEY) == 20
    before = json.dumps(demo.DEFECTIVE_KEY)
    eval_score.score(report(("SEC-001", "app/config.py")), demo.DEFECTIVE_KEY, lenient=True)
    assert json.dumps(demo.DEFECTIVE_KEY) == before
