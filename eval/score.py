"""Score an agent report against an answer key (precision / recall).

    python eval/score.py report.json answer_key.json

A reported violation matches a key entry when the file is the same and either the
rule_id is identical (strict) or the rule prefix/category agrees (lenient).
Each key entry can be matched at most once, so duplicates do not earn credit.
"""

from __future__ import annotations

import json
import sys


def _prefix(rule_id: str) -> str:
    return rule_id.split("-")[0].upper()


def score(report: dict, key: list[dict], lenient: bool) -> tuple[float, float, list, list]:
    remaining = list(key)
    tp, fp = [], []
    for v in report["violations"]:
        match = None
        for k in remaining:
            same_file = (k.get("file") or None) == (v.get("file") or None)
            same_rule = k["rule_id"] == v["rule_id"] or (lenient and _prefix(k["rule_id"]) == _prefix(v["rule_id"]))
            if same_file and same_rule:
                match = k
                break
        if match:
            remaining.remove(match)
            tp.append(v)
        else:
            fp.append(v)
    precision = len(tp) / len(report["violations"]) if report["violations"] else 1.0
    recall = len(tp) / len(key) if key else 1.0
    return precision, recall, fp, remaining


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    with open(sys.argv[1]) as fh:
        report = json.load(fh)
    with open(sys.argv[2]) as fh:
        key = json.load(fh)
    for mode in ("strict", "lenient"):
        p, r, fp, missed = score(report, key, lenient=mode == "lenient")
        print(f"{mode:8s} precision={p:.2f} recall={r:.2f} reported={len(report['violations'])} known={len(key)}")
    _, _, fp, missed = score(report, key, lenient=True)
    for v in fp:
        print(f"  unmatched report: {v['rule_id']} {v['file']}:{v['line']} {v['description']}")
    for k in missed:
        print(f"  missed known:     {k['rule_id']} {k['file']}")


if __name__ == "__main__":
    main()
