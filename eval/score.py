"""Score an agent report against an answer key (TP / FP / FN, precision, recall, F1).

    python eval/score.py report.json answer_key.json

A reported violation matches a key entry when the file is the same and either the rule_id is identical (strict)
or the rule prefix/category agrees (lenient). Each key entry can be matched at most once, so a duplicate finding earns no
extra credit; it still counts as reported, so it lowers precision.

Metrics are never invented:
* A report marked ``"incomplete": true`` (an interrupted run) is labelled INCOMPLETE and gets no accuracy metrics.
* Precision is N/A when nothing was reported, recall is N/A when the answer key is empty, and F1 is N/A unless both
  are defined.
* With an empty key (a healthy repository) every reported violation is a false positive; the FP count stays meaningful.

Exit status: 0 when the report was scored, 2 when it is INCOMPLETE or unreadable.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field


def _prefix(rule_id: str) -> str:
    return rule_id.split("-")[0].upper()


def _matches(entry: dict, known: dict, lenient: bool) -> bool:
    same_file = (known.get("file") or None) == (entry.get("file") or None)
    same_rule = known["rule_id"] == entry["rule_id"] or (lenient and _prefix(known["rule_id"]) == _prefix(entry["rule_id"]))
    return same_file and same_rule


@dataclass
class Score:
    mode: str
    reported: int
    known: int
    tp: int
    fp: int  # everything reported that earned no credit, duplicates included
    fn: int
    precision: float | None
    recall: float | None
    f1: float | None
    false_positives: list = field(default_factory=list)  # reported entries with no matching known defect
    duplicates: list = field(default_factory=list)  # reported entries repeating an already-matched defect
    missed: list = field(default_factory=list)  # known defects nobody reported


def score(report: dict, key: list[dict], lenient: bool) -> Score:
    violations = report["violations"]
    remaining = list(key)
    matched: list[dict] = []
    unmatched: list[dict] = []
    for v in violations:
        hit = next((k for k in remaining if _matches(v, k, lenient)), None)
        if hit:
            remaining.remove(hit)
            matched.append(hit)
        else:
            unmatched.append(v)
    duplicates = [v for v in unmatched if any(_matches(v, k, lenient) for k in matched)]
    false_positives = [v for v in unmatched if v not in duplicates]

    tp = len(matched)
    precision = tp / len(violations) if violations else None
    recall = tp / len(key) if key else None
    if precision is None or recall is None:
        f1 = None
    elif precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return Score(
        mode="lenient" if lenient else "strict",
        reported=len(violations),
        known=len(key),
        tp=tp,
        fp=len(violations) - tp,
        fn=len(remaining),
        precision=precision,
        recall=recall,
        f1=f1,
        false_positives=false_positives,
        duplicates=duplicates,
        missed=remaining,
    )


def _fmt(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2f}"


def format_score(s: Score) -> str:
    return (
        f"{s.mode:8s} TP={s.tp} FP={s.fp} FN={s.fn} precision={_fmt(s.precision)} recall={_fmt(s.recall)} "
        f"f1={_fmt(s.f1)} reported={s.reported} known={s.known}"
    )


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        print(__doc__)
        return 2
    try:
        with open(argv[0], encoding="utf-8") as fh:
            report = json.load(fh)
        with open(argv[1], encoding="utf-8") as fh:
            key = json.load(fh)
        if not isinstance(report.get("violations"), list):
            raise ValueError("report has no 'violations' list")
    except (OSError, ValueError) as exc:
        print(f"error: cannot read inputs: {exc}")
        return 2

    if report.get("incomplete") is True:
        print("status: INCOMPLETE - the investigation was interrupted, so this is NOT a completed accuracy evaluation.")
        print(f"        stop reason: {str(report.get('stop_reason', 'not recorded'))[:200]}")
        print(f"        findings recorded before the stop: {len(report['violations'])}; known defects: {len(key)}")
        print("        precision=N/A recall=N/A f1=N/A (not computed)")
        return 2

    print("status: COMPLETE")
    for lenient in (False, True):
        print(format_score(score(report, key, lenient)))
    lenient_score = score(report, key, lenient=True)
    for v in lenient_score.false_positives:
        print(f"  false positive:   {v['rule_id']} {v.get('file')}:{v.get('line')} {v.get('description')}")
    for v in lenient_score.duplicates:
        print(f"  duplicate:        {v['rule_id']} {v.get('file')}:{v.get('line')} {v.get('description')}")
    for k in lenient_score.missed:
        print(f"  missed known:     {k['rule_id']} {k.get('file')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
