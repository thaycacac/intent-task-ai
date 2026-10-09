#!/usr/bin/env python3
"""Train category TF-IDF + LogisticRegression; write versioned artifacts."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from intent_task_ai.category.model import CategoryModel


def _load_jsonl(path: Path) -> tuple[list[str], list[str]]:
    texts, labels = [], []
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            texts.append(row["text"])
            labels.append(row["category"])
    return texts, labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("data/processed/train.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/category"))
    parser.add_argument("--version", type=str, default=None)
    args = parser.parse_args()

    if not args.train.exists():
        raise SystemExit(f"Missing {args.train}; run prepare_dataset.py first")

    version = args.version or datetime.now(timezone.utc).strftime("v%Y%m%dT%H%M%SZ")
    texts, labels = _load_jsonl(args.train)
    print(f"Training on {len(texts)} examples, version={version}")
    model = CategoryModel.build(version=version).fit(texts, labels)

    version_dir = args.out / version
    model.save(version_dir)
    latest = args.out / "latest"
    if latest.exists() or latest.is_symlink():
        if latest.is_symlink() or latest.is_file():
            latest.unlink()
        else:
            shutil.rmtree(latest)
    shutil.copytree(version_dir, latest)
    print(f"Saved {version_dir} and copied to {latest}")


if __name__ == "__main__":
    main()
