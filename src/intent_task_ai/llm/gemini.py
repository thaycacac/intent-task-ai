"""Optional LLM engine: parse a task with the Gemini API (REST, no extra dependency).

Config via env: GEMINI_API_KEY (required), GEMINI_MODEL, GEMINI_TIMEOUT_S.
"""
from __future__ import annotations

import json
import os
import ssl
import urllib.error
import urllib.request
from datetime import datetime

import certifi

from intent_task_ai.llm.claude_local import (
    PROMPT,
    TZ,
    ClaudeUnavailable,
    extract_json,
    to_parse_result,
)
from intent_task_ai.pipeline.parse import ParseResult

DEFAULT_MODEL = "gemini-3.1-flash-lite"
ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

JSON_SUFFIX = """

Reply with ONLY a JSON object with exactly these keys:
{"category": "...", "deadline": "ISO-8601 or null", "task_detail": "...", "priority": "...",
 "confidence": {"category": 0.0, "deadline": 0.0, "priority": 0.0, "task_detail": 0.0}, "reasoning": "..."}"""


class GeminiUnavailable(ClaudeUnavailable):
    """Raised when the Gemini API cannot produce a usable parse."""


def _ssl_context() -> ssl.SSLContext:
    # python.org / venv builds on macOS ship without a CA bundle; use certifi's.
    return ssl.create_default_context(cafile=certifi.where())


def gemini_model() -> str:
    return os.getenv("GEMINI_MODEL", "").strip() or DEFAULT_MODEL


def gemini_available() -> bool:
    return bool(os.getenv("GEMINI_API_KEY", "").strip())


def parse_with_gemini(
    text: str,
    locale: str = "vi-VN",
    reference_time: datetime | None = None,
) -> ParseResult:
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        raise GeminiUnavailable("GEMINI_API_KEY is not set")

    ref = reference_time or datetime.now(TZ)
    if ref.tzinfo is None:
        ref = ref.replace(tzinfo=TZ)
    ref = ref.astimezone(TZ)
    prompt = PROMPT.format(
        reference_time=ref.isoformat(timespec="minutes"),
        weekday=ref.strftime("%A"),
        locale=locale,
        text=text,
    ) + JSON_SUFFIX
    model = gemini_model()
    body = json.dumps(
        {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"responseMimeType": "application/json", "temperature": 0},
        }
    ).encode()
    req = urllib.request.Request(
        ENDPOINT.format(model=model),
        data=body,
        headers={"Content-Type": "application/json", "x-goog-api-key": key},
    )
    timeout = float(os.getenv("GEMINI_TIMEOUT_S", "30"))
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=_ssl_context()) as resp:
            payload = json.load(resp)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:300].replace(key, "***")
        raise GeminiUnavailable(f"Gemini API HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        raise GeminiUnavailable(f"Gemini API request failed: {type(exc).__name__}") from exc

    try:
        parts = payload["candidates"][0]["content"]["parts"]
        out = "".join(p.get("text", "") for p in parts)
        data = extract_json(out)
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        reason = (payload.get("promptFeedback") or {}).get("blockReason") or "no JSON in response"
        raise GeminiUnavailable(f"Gemini output unusable: {reason}") from exc
    return to_parse_result(data, text, locale, model, source="gemini")
