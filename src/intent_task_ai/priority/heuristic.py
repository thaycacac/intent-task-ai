from __future__ import annotations

import re
from dataclasses import dataclass

Priority = str  # low | medium | high | urgent


@dataclass
class PriorityResult:
    priority: Priority
    confidence: float
    explanation: str


_URGENT = re.compile(
    r"\b(khẩn\s*cấp|gấp\s*lắm|asap|urgent|immediately|ngay\s*lập\s*tức)\b",
    re.IGNORECASE,
)
_HIGH = re.compile(
    r"\b(ưu\s*tiên\s*cao|high\s*priority|quan\s*trọng|critical|deadline|"
    r"phải\s*xong|must)\b",
    re.IGNORECASE,
)
_LOW = re.compile(
    r"\b(khi\s*nào\s*rảnh|low\s*priority|không\s*gấp|không\s*vội|optional|"
    r"nếu\s*có\s*thời\s*gian|someday|no\s*rush)\b",
    re.IGNORECASE,
)


def infer_priority(text: str) -> PriorityResult:
    """Heuristic priority from urgency cues (VI/EN)."""
    if _URGENT.search(text):
        return PriorityResult("urgent", 0.9, "matched:urgent_keywords")
    if _HIGH.search(text):
        return PriorityResult("high", 0.8, "matched:high_keywords")
    if _LOW.search(text):
        return PriorityResult("low", 0.75, "matched:low_keywords")
    return PriorityResult("medium", 0.55, "default:medium")
