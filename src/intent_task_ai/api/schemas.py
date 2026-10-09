from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


Category = Literal["work", "personal", "errand", "learning", "health", "other"]
Priority = Literal["low", "medium", "high", "urgent"]
Engine = Literal["hybrid", "claude", "gemini"]


class ParseTaskRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    locale: str = "vi-VN"
    reference_time: Optional[datetime] = None
    engine: Optional[Engine] = None  # None -> server DEFAULT_ENGINE (hybrid unless configured)


class FieldConfidence(BaseModel):
    category: float
    deadline: float
    priority: float
    task_detail: float


class ParseTaskResponse(BaseModel):
    category: Category
    deadline: Optional[datetime]
    task_detail: str
    priority: Priority
    confidence: FieldConfidence
    explanations: dict[str, str]
    engine: Engine = "hybrid"


class ParseOkrRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=8000)
    locale: str = "vi-VN"
    reference_time: Optional[datetime] = None
    engine: Optional[Engine] = None


class OkrItemResponse(BaseModel):
    kind: Literal["objective", "key_result", "task"]
    label: str
    source_text: str
    result: Optional[ParseTaskResponse] = None
    error: Optional[str] = None
    latency_ms: int = 0


class ParseOkrResponse(BaseModel):
    engine: Engine
    items: list[OkrItemResponse]


class EnginesResponse(BaseModel):
    hybrid: bool
    claude: bool
    gemini: bool
    default: Engine


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    category_model: str
