#!/usr/bin/env python3
"""Map MASSIVE intents → taxonomy and write train/validation/test + label report."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from intent_task_ai.category.map_intents import load_mapping, map_intent_to_category


def _intent_name(row: dict) -> str:
    # MASSIVE rows may expose intent as int id or string depending on export.
    for key in ("intent_str", "intent"):
        val = row.get(key)
        if isinstance(val, str) and val and not val.isdigit():
            return val
    # Fallback: datasets ClassLabel often stores name in scenario+intent combo;
    # download script tries to preserve intent_str.
    return str(row.get("intent", "unknown"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=Path("data/raw/massive"))
    parser.add_argument("--out", type=Path, default=Path("data/processed"))
    parser.add_argument("--report", type=Path, default=Path("reports/label_distribution.json"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    mapping = load_mapping()
    by_split: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
    label_counts: Counter[str] = Counter()
    intent_counts: Counter[str] = Counter()
    locale_label: dict[str, Counter[str]] = {}

    files = sorted(args.raw.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"No raw jsonl under {args.raw}; run scripts/download_massive.py first")

    for path in files:
        with path.open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                utt = (row.get("utt") or "").strip()
                if not utt:
                    continue
                intent = _intent_name(row)
                category = map_intent_to_category(intent, mapping)
                split = row.get("split") or "train"
                if split == "dev":
                    split = "validation"
                if split not in by_split:
                    by_split[split] = []
                locale = row.get("locale", "unknown")
                item = {
                    "text": utt,
                    "intent": intent,
                    "category": category,
                    "locale": locale,
                    "split": split,
                }
                by_split[split].append(item)
                label_counts[category] += 1
                intent_counts[intent] += 1
                locale_label.setdefault(locale, Counter())[category] += 1

    for split, rows in by_split.items():
        out_path = args.out / f"{split}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for item in rows:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"{split}: {len(rows)} → {out_path}")

    report = {
        "total": sum(label_counts.values()),
        "by_category": dict(label_counts.most_common()),
        "by_locale_category": {
            loc: dict(c.most_common()) for loc, c in locale_label.items()
        },
        "top_intents": dict(intent_counts.most_common(40)),
        "taxonomy": mapping.get("_meta", {}).get("taxonomy"),
    }
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote report {args.report}")
    print("Label distribution:", report["by_category"])


if __name__ == "__main__":
    main()
