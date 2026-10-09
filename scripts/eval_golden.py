#!/usr/bin/env python3
"""Evaluate deadline + priority heuristics on VI golden set."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from intent_task_ai.deadline.extract import extract_deadline
from intent_task_ai.priority.heuristic import infer_priority

TZ = ZoneInfo("Asia/Ho_Chi_Minh")


def _deadline_match(expected: str | None, got: str | None) -> bool:
    if expected is None and got is None:
        return True
    if expected is None or got is None:
        return False
    # Compare date + hour (ignore tz offset formatting)
    e = datetime.fromisoformat(expected)
    g = datetime.fromisoformat(got)
    return e.year == g.year and e.month == g.month and e.day == g.day and e.hour == g.hour


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", type=Path, default=Path("data/golden/vi_golden.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("reports/golden_eval.json"))
    parser.add_argument(
        "--reference-time",
        type=str,
        default="2026-09-19T10:00:00+07:00",
        help="Fixed anchor so relative dates are deterministic",
    )
    args = parser.parse_args()
    ref = datetime.fromisoformat(args.reference_time)

    n = 0
    dl_ok = 0
    pr_ok = 0
    rows_out = []
    with args.golden.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            n += 1
            dl = extract_deadline(row["text"], reference_time=ref)
            pr = infer_priority(row["text"])
            d_match = _deadline_match(row.get("deadline"), dl.deadline)
            p_match = row.get("priority") == pr.priority
            dl_ok += int(d_match)
            pr_ok += int(p_match)
            rows_out.append(
                {
                    "id": row.get("id"),
                    "deadline_ok": d_match,
                    "priority_ok": p_match,
                    "expected_deadline": row.get("deadline"),
                    "got_deadline": dl.deadline,
                    "expected_priority": row.get("priority"),
                    "got_priority": pr.priority,
                }
            )

    report = {
        "n": n,
        "deadline_exact_or_null_accuracy": round(dl_ok / n, 4) if n else 0,
        "priority_exact_accuracy": round(pr_ok / n, 4) if n else 0,
        "thresholds": {"deadline": 0.70, "priority": 0.60},
        "deadline_pass": (dl_ok / n >= 0.70) if n else False,
        "priority_pass": (pr_ok / n >= 0.60) if n else False,
        "reference_time": args.reference_time,
        "samples": rows_out,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "n": n,
                "deadline_acc": report["deadline_exact_or_null_accuracy"],
                "priority_acc": report["priority_exact_accuracy"],
                "deadline_pass": report["deadline_pass"],
                "priority_pass": report["priority_pass"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
