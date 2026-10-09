#!/usr/bin/env python3
"""Evaluate category model on holdout split; write metrics JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from intent_task_ai.category.model import CategoryModel, evaluate_predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=Path, default=Path("data/processed/test.jsonl"))
    parser.add_argument("--model", type=Path, default=Path("artifacts/category/latest"))
    parser.add_argument("--out", type=Path, default=Path("reports/category_eval.json"))
    args = parser.parse_args()

    model = CategoryModel.load(args.model)
    y_true, y_pred, locales = [], [], []
    with args.test.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            y_true.append(row["category"])
            y_pred.append(model.predict(row["text"]).category)
            locales.append(row.get("locale"))

    metrics = evaluate_predictions(y_true, y_pred)
    metrics["model_version"] = model.version
    metrics["split"] = str(args.test)
    # per-locale accuracy
    by_locale: dict[str, dict] = {}
    for loc in sorted(set(locales)):
        idx = [i for i, l in enumerate(locales) if l == loc]
        if not idx:
            continue
        by_locale[loc] = evaluate_predictions(
            [y_true[i] for i in idx], [y_pred[i] for i in idx]
        )
    metrics["by_locale"] = by_locale

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: metrics[k] for k in ("n", "accuracy", "majority_baseline_accuracy", "lift_over_majority", "model_version")}, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
