from datetime import datetime
from zoneinfo import ZoneInfo

from intent_task_ai.category.map_intents import map_intent_to_category
from intent_task_ai.deadline.extract import extract_deadline
from intent_task_ai.priority.heuristic import infer_priority
from intent_task_ai.task_detail.normalize import normalize_task_detail

REF = datetime(2026, 9, 19, 10, 0, tzinfo=ZoneInfo("Asia/Ho_Chi_Minh"))


def test_map_intent_prefix():
    assert map_intent_to_category("calendar_set") == "work"
    assert map_intent_to_category("cooking_recipe") == "errand"
    assert map_intent_to_category("qa_factoid") == "learning"
    assert map_intent_to_category("totally_unknown_xyz") == "other"


def test_deadline_tomorrow():
    r = extract_deadline("Mai 9h họp sprint", reference_time=REF)
    assert r.deadline is not None
    assert r.deadline.startswith("2026-09-20T09:00:00")


def test_deadline_null():
    r = extract_deadline("Viết tài liệu API", reference_time=REF)
    assert r.deadline is None


def test_priority_urgent():
    assert infer_priority("Nộp bài ASAP").priority == "urgent"
    assert infer_priority("Không gấp đâu").priority == "low"
    assert infer_priority("Làm việc bình thường").priority == "medium"


def test_task_detail_strips_leading():
    r = normalize_task_detail("Tôi muốn mua sữa")
    assert r.task_detail.lower().startswith("mua")
