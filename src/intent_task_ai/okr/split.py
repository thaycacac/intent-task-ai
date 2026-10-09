"""Split a pasted OKR (Objective + Key Results) into one parseable item per line.

Understands English and Vietnamese markers (`O1:`, `Objective:`, `Mục tiêu:`,
`KR1.2:`, `Key Result 2:`, `Kết quả then chốt:`), bullets and numbered lists.
Plain single-line input stays a single `task`, so old behaviour is unchanged.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

MAX_ITEMS = 30

_SEP = r"\s*[:.)\-–—]\s*"
_OBJ = re.compile(rf"^(?:O\s*\d*|Obj(?:ectives?)?\s*\d*|Mục\s*tiêu\s*\d*){_SEP}(?P<body>.*)$", re.I)
_KR = re.compile(
    rf"^(?:KRs?\s*(?P<n1>\d+(?:\.\d+)*)?|Key\s*Results?\s*(?P<n2>\d+(?:\.\d+)*)?"
    rf"|Kết\s*quả(?:\s*(?:then\s*chốt|chính|chủ\s*chốt))?\s*(?P<n3>\d+(?:\.\d+)*)?){_SEP}(?P<body>.*)$",
    re.I,
)
_BULLET = re.compile(r"^(?:[-*•▪◦]|\d+[.)])\s+(?P<body>.+)$")
_KR_HEADER = re.compile(r"^(?:key\s*results?|krs?|kết\s*quả(?:\s*then\s*chốt|\s*chính)?)\s*:?$", re.I)


@dataclass
class OkrItem:
    kind: str  # objective | key_result | task
    label: str  # "O1", "KR1.2", "" for task
    text: str


def split_okr(text: str) -> list[OkrItem]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    items: list[OkrItem] = []
    marked = False  # saw any explicit O / KR marker or "Key Results:" header
    n_obj = n_kr = 0
    in_kr = False

    for ln in lines:
        if _KR_HEADER.match(ln):
            marked = in_kr = True
            continue
        m = _OBJ.match(ln)
        if m and m["body"].strip():
            marked = True
            n_obj += 1
            n_kr = 0
            in_kr = False
            items.append(OkrItem("objective", f"O{n_obj}", m["body"].strip()))
            continue
        m = _KR.match(ln)
        if m and m["body"].strip():
            marked = in_kr = True
            n_kr += 1
            num = m["n1"] or m["n2"] or m["n3"] or str(n_kr)
            items.append(OkrItem("key_result", f"KR{num}", m["body"].strip()))
            continue
        m = _BULLET.match(ln)
        if m:
            body = m["body"].strip()
            if in_kr or n_obj:
                n_kr += 1
                items.append(OkrItem("key_result", f"KR{n_kr}", body))
            else:
                items.append(OkrItem("task", "", body))
            continue
        items.append(OkrItem("task", "", ln))

    # No markers at all, several lines: first plain line is the objective, the rest are KRs.
    if not marked and len(lines) > 1 and items and items[0].kind == "task" and not _BULLET.match(lines[0]):
        items[0] = OkrItem("objective", "O1", items[0].text)
        for i, it in enumerate(items[1:], start=1):
            items[i] = OkrItem("key_result", f"KR{i}", it.text)

    return items[:MAX_ITEMS]
