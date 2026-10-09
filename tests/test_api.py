import importlib
import json
import subprocess

import pytest
from fastapi.testclient import TestClient

app_module = importlib.import_module("intent_task_ai.api.app")
app = app_module.app
from intent_task_ai.llm import claude_local


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def _fake_run(stdout: str, returncode: int = 0):
    def run(cmd, **kwargs):
        assert "--tools" in cmd and cmd[cmd.index("--tools") + 1] == ""
        return subprocess.CompletedProcess(cmd, returncode, stdout=stdout, stderr="")
    return run


CLAUDE_DATA = {
    "category": "work",
    "deadline": "2026-09-20T09:00:00+07:00",
    "task_detail": "Họp sprint với team",
    "priority": "high",
    "confidence": {"category": 0.9, "deadline": 0.95, "priority": 0.8, "task_detail": 0.9},
    "reasoning": "meeting",
}


def test_ui_served(client):
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (302, 307)
    assert "Intent Task AI" in client.get("/ui/").text


def test_engines_endpoint(client, monkeypatch):
    monkeypatch.setattr(app_module, "claude_available", lambda: False)
    assert client.get("/v1/engines").json() == {"hybrid": True, "claude": False}


def test_hybrid_default_unchanged(client):
    r = client.post("/v1/parse-task", json={"text": "Mai 9h họp sprint, ưu tiên cao"})
    assert r.status_code == 200
    body = r.json()
    assert body["engine"] == "hybrid" and body["priority"] == "high"


def test_claude_engine(client, monkeypatch):
    envelope = {"structured_output": CLAUDE_DATA, "modelUsage": {"claude-haiku-5-5": {}}}
    monkeypatch.setattr(claude_local, "claude_available", lambda: True)
    monkeypatch.setattr(claude_local.subprocess, "run", _fake_run(json.dumps(envelope)))
    r = client.post("/v1/parse-task", json={"text": "Mai 9h họp sprint", "engine": "claude"})
    assert r.status_code == 200
    body = r.json()
    assert body["engine"] == "claude" and body["category"] == "work"
    assert body["deadline"].startswith("2026-09-20T09:00:00")
    assert body["explanations"]["category"] == "claude_local:claude-haiku-5-5"


def test_claude_engine_unavailable(client, monkeypatch):
    monkeypatch.setattr(claude_local, "claude_available", lambda: False)
    r = client.post("/v1/parse-task", json={"text": "x", "engine": "claude"})
    assert r.status_code == 503


def test_claude_invalid_values_fall_back():
    res = claude_local.to_parse_result(
        {"category": "bogus", "priority": "??", "deadline": "not-a-date", "task_detail": "", "confidence": {"category": 7}},
        "hello", "vi-VN", "m",
    )
    assert (res.category, res.priority, res.deadline, res.task_detail) == ("other", "medium", None, "hello")
    assert res.confidence["category"] == 1.0


def test_extract_json_tolerates_prose_and_fences():
    assert claude_local.extract_json('Sure:\n```json\n{"a": 1}\n```\nbye') == {"a": 1}
    with pytest.raises(ValueError):
        claude_local.extract_json("no json here")


OKR = """Objective: Tăng trưởng người dùng Q4
KR1: Hoàn thành onboarding mới vào thứ 6 tuần sau
- Gấp: sửa lỗi thanh toán
O2: Giảm churn
Key Results:
- Gọi 50 khách hàng cũ trước mai
"""


def test_split_okr_markers_and_bullets():
    from intent_task_ai.okr import split_okr

    got = [(i.kind, i.label) for i in split_okr(OKR)]
    assert got == [
        ("objective", "O1"), ("key_result", "KR1"), ("key_result", "KR2"),
        ("objective", "O2"), ("key_result", "KR1"),
    ]


def test_split_okr_plain_cases():
    from intent_task_ai.okr import split_okr

    assert [i.kind for i in split_okr("Mai 9h họp sprint")] == ["task"]
    assert [i.kind for i in split_okr("Tăng doanh thu\nBán 100 đơn\nGọi 20 khách")] == [
        "objective", "key_result", "key_result",
    ]
    assert [i.kind for i in split_okr("- a\n- b")] == ["task", "task"]


def test_parse_okr_hybrid(client):
    r = client.post("/v1/parse-okr", json={"text": OKR, "reference_time": "2026-10-09T10:00:00+07:00"})
    assert r.status_code == 200
    items = r.json()["items"]
    assert [i["kind"] for i in items][:2] == ["objective", "key_result"]
    assert all(i["result"] and i["result"]["engine"] == "hybrid" for i in items)
    assert items[-1]["result"]["deadline"].startswith("2026-10-10")  # "trước mai"


def test_parse_okr_single_line_is_task(client):
    items = client.post("/v1/parse-okr", json={"text": "Mai 9h họp sprint"}).json()["items"]
    assert len(items) == 1 and items[0]["kind"] == "task"


def test_parse_okr_claude(client, monkeypatch):
    envelope = {"structured_output": CLAUDE_DATA, "modelUsage": {"claude-haiku-5-5": {}}}
    monkeypatch.setattr(claude_local, "claude_available", lambda: True)
    monkeypatch.setattr(app_module, "claude_available", lambda: True)
    monkeypatch.setattr(claude_local.subprocess, "run", _fake_run(json.dumps(envelope)))
    r = client.post("/v1/parse-okr", json={"text": OKR, "engine": "claude"})
    assert r.status_code == 200
    assert len(r.json()["items"]) == 5
    assert all(i["result"]["engine"] == "claude" for i in r.json()["items"])


def test_parse_okr_claude_unavailable(client, monkeypatch):
    monkeypatch.setattr(app_module, "claude_available", lambda: False)
    assert client.post("/v1/parse-okr", json={"text": OKR, "engine": "claude"}).status_code == 503
