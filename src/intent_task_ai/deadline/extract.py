from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

TZ = ZoneInfo("Asia/Ho_Chi_Minh")

# Relative VI/EN patterns — prefer null over wrong guess.
_TOMORROW = re.compile(
    r"\b(ngày\s*mai|mai(?:\s+này)?|tomorrow)\b", re.IGNORECASE
)
_TODAY = re.compile(r"\b(hôm\s*nay|today)\b", re.IGNORECASE)
_DAY_AFTER = re.compile(
    r"\b(ngày\s*kia|ngày\s*mốt|day\s+after\s+tomorrow)\b", re.IGNORECASE
)
_NEXT_WEEK = re.compile(r"\b(tuần\s*sau|next\s+week)\b", re.IGNORECASE)
_TIME = re.compile(
    r"(?:lúc\s*|vào\s*|at\s+|@\s*)?(\d{1,2})(?::(\d{2}))?\s*(h|giờ|am|pm)?",
    re.IGNORECASE,
)
_WEEKDAY_VI = {
    "thứ hai": 0,
    "thứ 2": 0,
    "thứ ba": 1,
    "thứ 3": 1,
    "thứ tư": 2,
    "thứ 4": 2,
    "thứ năm": 3,
    "thứ 5": 3,
    "thứ sáu": 4,
    "thứ 6": 4,
    "thứ bảy": 5,
    "thứ 7": 5,
    "chủ nhật": 6,
    "cn": 6,
}
_WEEKDAY_EN = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}
_DATE_DMY = re.compile(
    r"\b(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?\b"
)
_IN_N_DAYS = re.compile(
    r"\b(?:trong|sau)\s+(\d+)\s+ngày\b|\bin\s+(\d+)\s+days?\b",
    re.IGNORECASE,
)


@dataclass
class DeadlineResult:
    deadline: Optional[str]
    confidence: float
    explanation: str


def _parse_time(text: str, base: datetime) -> tuple[int, int, float, str]:
    """Return hour, minute, confidence bump, rule name."""
    m = _TIME.search(text)
    if not m:
        lower = text.lower()
        if re.search(r"\b(tối|evening|tonight)\b", lower):
            return 19, 0, 0.05, "default_evening_19:00"
        if re.search(r"\b(chiều|afternoon)\b", lower):
            return 15, 0, 0.05, "default_afternoon_15:00"
        if re.search(r"\b(sáng|morning)\b", lower):
            return 8, 0, 0.05, "default_morning_08:00"
        return 9, 0, 0.0, "default_09:00"
    hour = int(m.group(1))
    minute = int(m.group(2) or 0)
    suffix = (m.group(3) or "").lower()
    lower = text.lower()
    if suffix == "pm" and hour < 12:
        hour += 12
    if suffix == "am" and hour == 12:
        hour = 0
    # Vietnamese day-part without am/pm
    if suffix in ("", "h", "giờ") and hour <= 12:
        if re.search(r"\b(tối|chiều|pm)\b", lower) and hour < 12:
            hour += 12
        elif re.search(r"\b(sáng|am)\b", lower) and hour == 12:
            hour = 0
    if hour > 23 or minute > 59:
        return 9, 0, 0.0, "invalid_time_fallback"
    return hour, minute, 0.15, f"time={hour:02d}:{minute:02d}"


def _next_weekday(base: datetime, target: int) -> datetime:
    days_ahead = (target - base.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7
    return base + timedelta(days=days_ahead)


def extract_deadline(
    text: str, reference_time: datetime | None = None
) -> DeadlineResult:
    """Extract deadline as ISO-8601 in Asia/Ho_Chi_Minh, or null."""
    base = reference_time.astimezone(TZ) if reference_time else datetime.now(TZ)
    rules: list[str] = []
    target_date: datetime | None = None
    confidence = 0.0

    if _DAY_AFTER.search(text):
        target_date = base + timedelta(days=2)
        rules.append("day_after_tomorrow")
        confidence = 0.85
    elif _TOMORROW.search(text):
        target_date = base + timedelta(days=1)
        rules.append("tomorrow")
        confidence = 0.9
    elif _TODAY.search(text):
        target_date = base
        rules.append("today")
        confidence = 0.85
    elif _NEXT_WEEK.search(text):
        target_date = base + timedelta(days=7)
        rules.append("next_week")
        confidence = 0.7

    if target_date is None:
        m = _IN_N_DAYS.search(text)
        if m:
            n = int(m.group(1) or m.group(2))
            target_date = base + timedelta(days=n)
            rules.append(f"in_{n}_days")
            confidence = 0.8

    if target_date is None:
        lower = text.lower()
        for name, wd in {**_WEEKDAY_VI, **_WEEKDAY_EN}.items():
            if name in lower:
                target_date = _next_weekday(base, wd)
                rules.append(f"weekday={name}")
                confidence = 0.75
                break

    if target_date is None:
        m = _DATE_DMY.search(text)
        if m:
            day, month = int(m.group(1)), int(m.group(2))
            year_raw = m.group(3)
            year = int(year_raw) if year_raw else base.year
            if year < 100:
                year += 2000
            try:
                target_date = datetime(year, month, day, tzinfo=TZ)
                if target_date.date() < base.date() and not year_raw:
                    target_date = datetime(year + 1, month, day, tzinfo=TZ)
                rules.append("date_dmy")
                confidence = 0.85
            except ValueError:
                pass

    if target_date is None:
        return DeadlineResult(
            deadline=None,
            confidence=1.0,
            explanation="no_temporal_cue→null",
        )

    hour, minute, time_bump, time_rule = _parse_time(text, base)
    rules.append(time_rule)
    confidence = min(1.0, confidence + time_bump)
    result = target_date.replace(hour=hour, minute=minute, second=0, microsecond=0)
    return DeadlineResult(
        deadline=result.isoformat(),
        confidence=confidence,
        explanation="rules:" + ",".join(rules),
    )
