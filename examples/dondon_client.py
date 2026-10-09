#!/usr/bin/env python3
"""Minimal DonDon-side example client for Intent Task AI (contract only)."""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from typing import Any


def parse_task(
    text: str,
    *,
    base_url: str,
    locale: str = "vi-VN",
    timeout: float = 15.0,
) -> dict[str, Any]:
    """Call POST /v1/parse-task and return JSON fields for DonDon quick-create."""
    payload = json.dumps({"text": text, "locale": locale}).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/parse-task",
        data=payload,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def to_dondon_draft(parsed: dict[str, Any]) -> dict[str, Any]:
    """Map API response → example DonDon draft task shape (illustrative)."""
    return {
        "title": parsed.get("task_detail"),
        "category": parsed.get("category"),
        "due_at": parsed.get("deadline"),
        "priority": parsed.get("priority"),
        "source": "intent-task-ai",
        "nlp_confidence": parsed.get("confidence"),
        "nlp_explanations": parsed.get("explanations"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="DonDon example client")
    parser.add_argument(
        "--base",
        default=os.getenv("INTENT_TASK_AI_URL", "http://127.0.0.1:8000"),
    )
    parser.add_argument(
        "--text",
        default="Mai 9h họp sprint với team, ưu tiên cao",
    )
    parser.add_argument("--locale", default="vi-VN")
    args = parser.parse_args()

    try:
        parsed = parse_task(args.text, base_url=args.base, locale=args.locale)
    except urllib.error.URLError as exc:
        raise SystemExit(f"Cannot reach API at {args.base}: {exc}") from exc

    draft = to_dondon_draft(parsed)
    print("=== parse-task response ===")
    print(json.dumps(parsed, ensure_ascii=False, indent=2))
    print("=== DonDon draft (example) ===")
    print(json.dumps(draft, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
