from __future__ import annotations

import re
from dataclasses import dataclass

_FILLERS = re.compile(
    r"\b(ơi|à|ừ|uh|um|please|làm\s*ơn|giúp\s*mình|nhé|nha|with\s+me)\b",
    re.IGNORECASE,
)
_LEADING = re.compile(
    r"^(?:tôi\s+muốn\s+|mình\s+cần\s+|hãy\s+|please\s+|i\s+(?:want|need)\s+to\s+"
    r"|can\s+you\s+)\s*",
    re.IGNORECASE,
)
_SPACE = re.compile(r"\s+")


@dataclass
class TaskDetailResult:
    task_detail: str
    confidence: float
    explanation: str


def normalize_task_detail(text: str) -> TaskDetailResult:
    """Strip fillers and leading politeness; keep action/object."""
    cleaned = text.strip()
    cleaned = _LEADING.sub("", cleaned)
    cleaned = _FILLERS.sub(" ", cleaned)
    cleaned = _SPACE.sub(" ", cleaned).strip(" .,!?;:")
    if cleaned:
        cleaned = cleaned[0].upper() + cleaned[1:]
    else:
        cleaned = text.strip()
    return TaskDetailResult(
        task_detail=cleaned[:200],
        confidence=0.8 if cleaned != text.strip() else 0.6,
        explanation="normalizer:strip_fillers_leading",
    )
