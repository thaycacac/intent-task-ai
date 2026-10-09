from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

CATEGORIES = ("work", "personal", "errand", "learning", "health", "other")

_DEFAULT_MAPPING = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "mappings"
    / "massive_intent_to_category.json"
)


@lru_cache(maxsize=1)
def load_mapping(path: str | None = None) -> dict[str, Any]:
    mapping_path = Path(path) if path else _DEFAULT_MAPPING
    with mapping_path.open(encoding="utf-8") as f:
        return json.load(f)


def map_intent_to_category(intent: str, mapping: dict[str, Any] | None = None) -> str:
    """Map a MASSIVE intent id to DonDon taxonomy category."""
    data = mapping or load_mapping()
    by_intent = data.get("by_intent") or {}
    if intent in by_intent:
        return by_intent[intent]
    by_prefix = data.get("by_prefix") or {}
    for prefix, category in sorted(by_prefix.items(), key=lambda x: -len(x[0])):
        if intent.startswith(prefix):
            return category
    return "other"
