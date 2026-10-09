#!/usr/bin/env python3
"""Download MASSIVE (vi-VN + en-US) and write raw jsonl under data/raw."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def _intent_str(ds_split, row) -> str:
    intent_val = row.get("intent")
    feat = ds_split.features.get("intent")
    if feat is not None and hasattr(feat, "int2str") and isinstance(intent_val, int):
        return feat.int2str(intent_val)
    if isinstance(intent_val, str):
        return intent_val
    return str(intent_val)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/raw/massive"))
    parser.add_argument("--locales", nargs="+", default=["vi-VN", "en-US"])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    for locale in args.locales:
        print(f"Loading MASSIVE locale={locale} …")
        # MASSIVE still ships a dataset script; pin datasets<3 and omit trust_remote_code.
        ds = load_dataset("AmazonScience/massive", locale)
        for split_name, split in ds.items():
            # HF uses train/validation/test; some dumps use "dev"
            out_split = "validation" if split_name == "dev" else split_name
            out_path = args.out / f"{locale}_{out_split}.jsonl"
            n = 0
            with out_path.open("w", encoding="utf-8") as f:
                for row in split:
                    f.write(
                        json.dumps(
                            {
                                "locale": locale,
                                "split": out_split,
                                "id": row.get("id"),
                                "utt": row.get("utt"),
                                "intent": _intent_str(split, row),
                                "scenario": row.get("scenario"),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    n += 1
            print(f"  wrote {out_path} ({n} rows)")


if __name__ == "__main__":
    main()
