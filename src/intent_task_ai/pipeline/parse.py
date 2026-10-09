from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from intent_task_ai.category.model import CategoryModel, normalize_text
from intent_task_ai.deadline.extract import extract_deadline
from intent_task_ai.priority.heuristic import infer_priority
from intent_task_ai.task_detail.normalize import normalize_task_detail


@dataclass
class ParseResult:
    category: str
    deadline: Optional[str]
    task_detail: str
    priority: str
    confidence: dict[str, float]
    explanations: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class HybridParser:
    def __init__(self, category_model: CategoryModel):
        self.category_model = category_model

    @classmethod
    def from_artifacts(cls, artifacts_dir: Path) -> HybridParser:
        model = CategoryModel.load(artifacts_dir)
        return cls(category_model=model)

    def parse(
        self,
        text: str,
        locale: str = "vi-VN",
        reference_time: datetime | None = None,
    ) -> ParseResult:
        cleaned = normalize_text(text)
        cat = self.category_model.predict(cleaned)
        dl = extract_deadline(cleaned, reference_time=reference_time)
        pr = infer_priority(cleaned)
        td = normalize_task_detail(cleaned)
        return ParseResult(
            category=cat.category,
            deadline=dl.deadline,
            task_detail=td.task_detail,
            priority=pr.priority,
            confidence={
                "category": cat.confidence,
                "deadline": dl.confidence,
                "priority": pr.confidence,
                "task_detail": td.confidence,
            },
            explanations={
                "category": cat.explanation,
                "deadline": dl.explanation,
                "priority": pr.explanation,
                "task_detail": td.explanation,
                "locale": locale,
            },
        )
