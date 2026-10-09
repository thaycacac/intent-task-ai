"""Optional LLM engine: parse a task via the locally installed Claude Code CLI.

Runs `claude -p` headless with all tools disabled and a JSON schema, so it uses
the same login as the developer's Claude Code session (no ANTHROPIC_API_KEY).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from intent_task_ai.pipeline.parse import ParseResult

TZ = ZoneInfo("Asia/Ho_Chi_Minh")
CATEGORIES = ("work", "personal", "errand", "learning", "health", "other")
PRIORITIES = ("low", "medium", "high", "urgent")
FIELDS = ("category", "deadline", "priority", "task_detail")

OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "category": {"type": "string", "enum": list(CATEGORIES)},
        "deadline": {"type": ["string", "null"]},
        "task_detail": {"type": "string"},
        "priority": {"type": "string", "enum": list(PRIORITIES)},
        "confidence": {
            "type": "object",
            "properties": {f: {"type": "number"} for f in FIELDS},
            "required": list(FIELDS),
        },
        "reasoning": {"type": "string"},
    },
    "required": ["category", "deadline", "task_detail", "priority", "confidence"],
}

PROMPT = """You turn a free-text to-do utterance (Vietnamese or English) into structured task fields for a task app.

Fields:
- category: one of work | personal | errand | learning | health | other
  (work = job/meetings/clients; personal = family/friends/home life; errand = shopping, bills, chores, pickups;
   learning = study/courses/reading; health = doctor, medicine, exercise; other = anything else)
- deadline: ISO-8601 datetime with +07:00 offset (timezone Asia/Ho_Chi_Minh) resolved against the reference time,
  or null when the text has no explicit date/time. Do not invent a deadline. If only a day is given, use 09:00.
  Vietnamese hints: "mai"/"ngày mai" = tomorrow, "mốt"/"ngày kia" = day after tomorrow, "chiều" = afternoon (+12h),
  "tối" = evening, "sáng" = morning, "thứ 2..thứ 7" = Monday..Saturday, "chủ nhật" = Sunday, "tuần sau" = next week.
- task_detail: short task title in the original language, without date/time and priority words.
- priority: low | medium | high | urgent (urgent = "gấp", "khẩn", "ngay", ASAP; high = "quan trọng", "ưu tiên cao";
  low = "khi rảnh", "không gấp"; otherwise medium).
- confidence: 0..1 per field. reasoning: one short sentence.

Reference time: {reference_time} ({weekday})
Locale hint: {locale}
Text: {text}"""


class ClaudeUnavailable(RuntimeError):
    """Raised when the local Claude CLI cannot produce a usable parse."""


def _cli_path() -> str:
    return os.getenv("CLAUDE_CLI_PATH", "claude")


def claude_available() -> bool:
    return shutil.which(_cli_path()) is not None


def _build_command(prompt: str) -> list[str]:
    cmd = [
        _cli_path(),
        "-p",
        prompt,
        "--output-format",
        "json",
        "--json-schema",
        json.dumps(OUTPUT_SCHEMA),
        "--tools",
        "",
        "--no-session-persistence",
        "--strict-mcp-config",
        "--setting-sources",
        "",
    ]
    model = os.getenv("CLAUDE_MODEL", "").strip()
    if model:
        cmd += ["--model", model]
    return cmd


def extract_json(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of free text (tolerates prose / code fences)."""
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object in output")
    obj, _ = json.JSONDecoder().raw_decode(text[start:])
    if not isinstance(obj, dict):
        raise ValueError("JSON output is not an object")
    return obj


def _clamp(value: Any, default: float) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _normalize_deadline(value: Any) -> str | None:
    if not value or not isinstance(value, str):
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=TZ)
    return dt.astimezone(TZ).isoformat()


def to_parse_result(data: dict[str, Any], text: str, locale: str, model: str) -> ParseResult:
    category = data.get("category")
    priority = data.get("priority")
    conf = data.get("confidence") or {}
    tag = f"claude_local:{model}"
    reasoning = str(data.get("reasoning") or "").strip()
    return ParseResult(
        category=category if category in CATEGORIES else "other",
        deadline=_normalize_deadline(data.get("deadline")),
        task_detail=str(data.get("task_detail") or text).strip(),
        priority=priority if priority in PRIORITIES else "medium",
        confidence={f: _clamp(conf.get(f), 0.5) for f in FIELDS},
        explanations={
            "category": tag,
            "deadline": tag,
            "priority": tag,
            "task_detail": tag,
            "reasoning": reasoning,
            "locale": locale,
        },
    )


def parse_with_claude(
    text: str,
    locale: str = "vi-VN",
    reference_time: datetime | None = None,
) -> ParseResult:
    if not claude_available():
        raise ClaudeUnavailable(f"Claude CLI not found ({_cli_path()!r}); install Claude Code and log in")

    ref = reference_time or datetime.now(TZ)
    if ref.tzinfo is None:
        ref = ref.replace(tzinfo=TZ)
    ref = ref.astimezone(TZ)
    prompt = PROMPT.format(
        reference_time=ref.isoformat(timespec="minutes"),
        weekday=ref.strftime("%A"),
        locale=locale,
        text=text,
    )
    timeout = float(os.getenv("CLAUDE_TIMEOUT_S", "60"))
    try:
        # Neutral cwd so no project CLAUDE.md / settings leak into the call.
        proc = subprocess.run(
            _build_command(prompt),
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=tempfile.gettempdir(),
        )
    except subprocess.TimeoutExpired as exc:
        raise ClaudeUnavailable(f"Claude CLI timed out after {timeout:.0f}s") from exc
    except OSError as exc:
        raise ClaudeUnavailable(f"Claude CLI failed to start: {exc}") from exc

    try:
        envelope = extract_json(proc.stdout)
    except ValueError as exc:
        detail = (proc.stderr or proc.stdout).strip()[:300]
        raise ClaudeUnavailable(f"Claude CLI returned no JSON (exit {proc.returncode}): {detail}") from exc
    if envelope.get("is_error"):
        raise ClaudeUnavailable(f"Claude CLI error: {str(envelope.get('result'))[:300]}")

    data = envelope.get("structured_output")
    if not isinstance(data, dict):
        try:
            data = extract_json(str(envelope.get("result") or ""))
        except ValueError as exc:
            raise ClaudeUnavailable("Claude output did not contain task JSON") from exc

    model = next(iter(envelope.get("modelUsage") or {}), None) or os.getenv("CLAUDE_MODEL") or "default"
    return to_parse_result(data, text, locale, model)
