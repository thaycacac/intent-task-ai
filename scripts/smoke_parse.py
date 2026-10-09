#!/usr/bin/env python3
"""Smoke-call local parse-task API."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request


SAMPLES = [
    {"text": "Mai 9h họp sprint với team, ưu tiên cao", "locale": "vi-VN"},
    {"text": "Buy groceries tomorrow evening", "locale": "en-US"},
    {"text": "Nhắc tôi uống thuốc lúc 20h hôm nay", "locale": "vi-VN"},
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    args = parser.parse_args()

    health = urllib.request.urlopen(f"{args.base}/health", timeout=10).read()
    print("health:", health.decode())

    for sample in SAMPLES:
        req = urllib.request.Request(
            f"{args.base}/v1/parse-task",
            data=json.dumps(sample).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode())
        print("---")
        print("in:", sample)
        print("out:", json.dumps(body, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"smoke failed: {exc}", file=sys.stderr)
        sys.exit(1)
