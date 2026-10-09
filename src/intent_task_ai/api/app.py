from __future__ import annotations

import logging
import os
import time
from collections import Counter
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from intent_task_ai import __version__
from intent_task_ai.api.schemas import (
    EnginesResponse,
    FieldConfidence,
    HealthResponse,
    OkrItemResponse,
    ParseOkrRequest,
    ParseOkrResponse,
    ParseTaskRequest,
    ParseTaskResponse,
)
from intent_task_ai.llm import ClaudeUnavailable, claude_available, parse_with_claude
from intent_task_ai.okr import split_okr
from intent_task_ai.pipeline.parse import HybridParser, ParseResult

logger = logging.getLogger("intent_task_ai")
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='{"ts":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":%(message)s}',
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", "artifacts/category/latest"))

_counters: Counter[str] = Counter()
_parser: HybridParser | None = None


def _resolve_artifacts() -> Path:
    if ARTIFACTS_DIR.exists():
        return ARTIFACTS_DIR
    # Repo-relative fallback when running from project root
    root = Path(__file__).resolve().parents[3]
    candidate = root / "artifacts" / "category" / "latest"
    return candidate


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    global _parser
    path = _resolve_artifacts()
    if not (path / "category_model.joblib").exists():
        logger.warning('"category_model_missing path=%s"' % path)
        _parser = None
    else:
        _parser = HybridParser.from_artifacts(path)
        logger.info('"category_model_loaded version=%s"' % _parser.category_model.version)
    yield


app = FastAPI(
    title="Intent Task AI",
    version=__version__,
    lifespan=lifespan,
)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    _counters["http_requests_total"] += 1
    _counters[f"http_requests_status_{response.status_code}"] += 1
    if request.url.path == "/v1/parse-task":
        _counters["parse_task_total"] += 1
        _counters["parse_task_latency_ms_sum"] += int(elapsed_ms)
    logger.info(
        '"http_request method=%s path=%s status=%s latency_ms=%.1f"'
        % (request.method, request.url.path, response.status_code, elapsed_ms)
    )
    return response


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    if _parser is None:
        return HealthResponse(
            status="degraded",
            version=__version__,
            category_model="missing",
        )
    return HealthResponse(
        status="ok",
        version=__version__,
        category_model=_parser.category_model.version,
    )


@app.get("/metrics")
def metrics() -> Response:
    lines = [
        "# HELP http_requests_total Total HTTP requests",
        "# TYPE http_requests_total counter",
        f"http_requests_total {_counters['http_requests_total']}",
        "# HELP parse_task_total Total parse-task calls",
        "# TYPE parse_task_total counter",
        f"parse_task_total {_counters['parse_task_total']}",
        "# HELP parse_task_latency_ms_sum Sum of parse-task latency in ms",
        "# TYPE parse_task_latency_ms_sum counter",
        f"parse_task_latency_ms_sum {_counters['parse_task_latency_ms_sum']}",
    ]
    for key, value in sorted(_counters.items()):
        if key.startswith("http_requests_status_"):
            code = key.rsplit("_", 1)[-1]
            lines.append(f'http_requests_status{{code="{code}"}} {value}')
    return PlainTextResponse("\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse("/ui/")


@app.get("/v1/engines", response_model=EnginesResponse)
def engines() -> EnginesResponse:
    return EnginesResponse(hybrid=True, claude=claude_available())


def _to_response(result: ParseResult, engine: str) -> ParseTaskResponse:
    return ParseTaskResponse(
        category=result.category,  # type: ignore[arg-type]
        deadline=datetime.fromisoformat(result.deadline) if result.deadline else None,
        task_detail=result.task_detail,
        priority=result.priority,  # type: ignore[arg-type]
        confidence=FieldConfidence(**result.confidence),
        explanations=result.explanations,
        engine=engine,  # type: ignore[arg-type]
    )


def _parse_one(text: str, engine: str, locale: str, reference_time: datetime | None) -> ParseTaskResponse:
    """Parse one utterance with the chosen engine. Raises ClaudeUnavailable for engine=claude."""
    if engine == "claude":
        _counters["parse_task_engine_claude_total"] += 1
        try:
            return _to_response(
                parse_with_claude(text, locale=locale, reference_time=reference_time), "claude"
            )
        except ClaudeUnavailable:
            _counters["parse_task_engine_claude_errors"] += 1
            raise

    if _parser is None:
        # Soft fallback: heuristics only, category=other
        from intent_task_ai.deadline.extract import extract_deadline
        from intent_task_ai.priority.heuristic import infer_priority
        from intent_task_ai.task_detail.normalize import normalize_task_detail

        dl = extract_deadline(text, reference_time)
        pr = infer_priority(text)
        td = normalize_task_detail(text)
        _counters["parse_task_degraded"] += 1
        return ParseTaskResponse(
            category="other",
            deadline=datetime.fromisoformat(dl.deadline) if dl.deadline else None,
            task_detail=td.task_detail,
            priority=pr.priority,  # type: ignore[arg-type]
            confidence=FieldConfidence(
                category=0.0,
                deadline=dl.confidence,
                priority=pr.confidence,
                task_detail=td.confidence,
            ),
            explanations={
                "category": "model_missing→other",
                "deadline": dl.explanation,
                "priority": pr.explanation,
                "task_detail": td.explanation,
            },
        )

    result = _parser.parse(text, locale=locale, reference_time=reference_time)
    return _to_response(result, "hybrid")


@app.post("/v1/parse-task", response_model=ParseTaskResponse)
def parse_task(body: ParseTaskRequest) -> ParseTaskResponse:
    try:
        return _parse_one(body.text, body.engine, body.locale, body.reference_time)
    except ClaudeUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/v1/parse-okr", response_model=ParseOkrResponse)
def parse_okr(body: ParseOkrRequest) -> ParseOkrResponse:
    """Split a pasted OKR (objective + key results) and parse every line as a task."""
    items = split_okr(body.text)
    if not items:
        raise HTTPException(status_code=422, detail="no parseable lines")
    if body.engine == "claude" and not claude_available():
        raise HTTPException(status_code=503, detail="Claude CLI not found; install Claude Code and log in")
    _counters["parse_okr_total"] += 1

    def work(item) -> OkrItemResponse:
        start = time.perf_counter()
        out = OkrItemResponse(kind=item.kind, label=item.label, source_text=item.text)  # type: ignore[arg-type]
        try:
            out.result = _parse_one(item.text, body.engine, body.locale, body.reference_time)
        except ClaudeUnavailable as exc:
            out.error = str(exc)
        out.latency_ms = int((time.perf_counter() - start) * 1000)
        return out

    # Claude calls take seconds each, so fan out; hybrid is instant.
    workers = min(4, len(items)) if body.engine == "claude" else 1
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(work, items))
    return ParseOkrResponse(engine=body.engine, items=results)


if STATIC_DIR.exists():
    app.mount("/ui", StaticFiles(directory=STATIC_DIR, html=True), name="ui")
