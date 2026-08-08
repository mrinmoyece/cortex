"""
Cortex FastAPI application.

Endpoints:
  POST /api/v1/runs          — Start an agent run (returns run_id)
  GET  /api/v1/runs/{id}     — Poll run status
  POST /api/v1/runs/{id}/stream  — SSE stream of a run's step outputs
  POST /api/v1/ingest        — Ingest a document into the knowledge base
  GET  /api/v1/memory/{user_id}  — Query user memory
  POST /api/v1/mcp/call      — Direct MCP tool invocation
  GET  /metrics              — Prometheus metrics scrape endpoint
  GET  /health               — Liveness + readiness probe
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field
from starlette.responses import Response

from cortex.api.auth import TokenPayload, get_current_user
from cortex.api.ratelimit import RateLimitMiddleware
from cortex.config import settings
from cortex.exceptions import CortexError
from cortex.graph.cortex_graph import run_cortex
from cortex.graph.state import CortexState, RunStatus
from cortex.logging_config import configure_logging, get_logger
from cortex.mcp.server import ingest_document
from cortex.obs.metrics import (
    api_request_duration,
    api_requests_total,
    configure_observability,
)
from cortex.safety.middleware import SafetyMiddleware

configure_logging()
logger = get_logger(__name__)

# ── In-memory run store (replace with Redis/Postgres in production) ───────────
_runs: dict[str, CortexState] = {}


# ── Lifespan ──────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("cortex.startup", environment=settings.environment.value)
    configure_observability()
    yield
    logger.info("cortex.shutdown")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Cortex",
    description="Enterprise agentic AI platform",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if not settings.is_production else None,
    redoc_url=None,
)

# Rate limiting goes on FIRST so it runs LAST in the onion - i.e. before
# routing and body validation, which is the only placement that also meters
# requests destined to 404 or 422. Those are cheaper for an attacker to
# generate than valid ones.
app.add_middleware(RateLimitMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[str(o) for o in settings.api_cors_origins],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Middleware ────────────────────────────────────────────────────────────────


@app.middleware("http")
async def telemetry_middleware(request: Request, call_next: Callable) -> Response:
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start

    labels = {
        "method": request.method,
        "endpoint": request.url.path,
        "status_code": str(response.status_code),
    }
    api_request_duration.labels(**labels).observe(elapsed)
    api_requests_total.labels(**labels).inc()
    return response


@app.exception_handler(CortexError)
async def cortex_exception_handler(request: Request, exc: CortexError) -> JSONResponse:
    logger.warning("api.cortex_error", code=exc.code, message=exc.message, path=request.url.path)
    return JSONResponse(
        status_code=exc.http_status.value,
        content={"error": {"code": exc.code, "message": exc.message, "details": exc.details}},
    )


# ── Request / Response models ─────────────────────────────────────────────────


class RunRequest(BaseModel):
    goal: str = Field(..., min_length=1, max_length=4096, description="User goal for the agent run")
    context: dict[str, Any] = Field(default_factory=dict)
    session_id: str | None = None


class RunResponse(BaseModel):
    run_id: str
    status: str
    created_at: str


class RunStatusResponse(BaseModel):
    run_id: str
    status: str
    final_output: str | None
    total_cost_usd: float
    iteration_count: int
    task_count: int
    completed_tasks: int
    low_confidence: bool
    error: str | None


class IngestRequest(BaseModel):
    text: str = Field(..., min_length=10)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MCPCallRequest(BaseModel):
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)


# ── Background task ───────────────────────────────────────────────────────────


async def _execute_run(run_id: str, request: RunRequest, user: TokenPayload) -> None:
    """Execute an agent run in the background and store result.

    The safety layer is applied HERE, on the goal going in and the answer
    coming out. It was fully implemented - injection detection, PII
    redaction, NeMo rails - and referenced by nothing: `grep -rn safety
    src/` outside the safety package itself returned a metric name and an
    exception class. Every request went straight to the graph.

    Third occurrence of the same defect class in this codebase (the MCP
    client, the rate-limit setting, and now the whole guardrail layer), so
    it is worth naming the smell: a module with a clean interface, thorough
    tests, and no inbound call edge from production code.
    """
    safety = SafetyMiddleware()
    try:
        goal = await safety.check_input(request.goal, user_id=user.sub)
        state = await run_cortex(
            user_goal=goal,
            user_id=user.sub,
            session_id=request.session_id,
            context=request.context,
        )
        if state.final_output:
            state = state.model_copy(
                update={"final_output": await safety.check_output(state.final_output)}
            )
        _runs[run_id] = state
    except Exception as exc:
        logger.error("run.background_error", run_id=run_id, error=str(exc))
        # Store a failed state so callers don't hang
        _runs[run_id] = CortexState(  # type: ignore[call-arg]
            run_id=run_id,
            session_id=request.session_id or run_id,
            user_id=user.sub,
            user_goal=request.goal,
            status=RunStatus.FAILED,
            error=str(exc),
        )


# ── Routes ────────────────────────────────────────────────────────────────────


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "cortex", "environment": settings.environment.value}


@app.get("/metrics")
async def metrics() -> Response:
    """Prometheus scrape endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/api/v1/runs", response_model=RunResponse, status_code=202)
async def create_run(
    request: RunRequest,
    background_tasks: BackgroundTasks,
    user: TokenPayload = Depends(get_current_user),
) -> RunResponse:
    """Start an agent run asynchronously. Poll GET /runs/{id} for status."""
    run_id = str(uuid.uuid4())
    # Initialise as pending so callers can poll immediately
    _runs[run_id] = CortexState(  # type: ignore[call-arg]
        run_id=run_id,
        session_id=request.session_id or run_id,
        user_id=user.sub,
        user_goal=request.goal,
        status=RunStatus.PENDING,
    )
    background_tasks.add_task(_execute_run, run_id, request, user)
    logger.info("run.created", run_id=run_id, user_id=user.sub)
    return RunResponse(
        run_id=run_id,
        status=RunStatus.PENDING.value,
        created_at=_runs[run_id].created_at.isoformat(),
    )


@app.get("/api/v1/runs/{run_id}", response_model=RunStatusResponse)
async def get_run(
    run_id: str,
    user: TokenPayload = Depends(get_current_user),
) -> RunStatusResponse:
    """Poll the status and result of an agent run."""
    state = _runs.get(run_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found")
    if state.user_id != user.sub:
        raise HTTPException(status_code=403, detail="Access denied")

    return RunStatusResponse(
        run_id=run_id,
        status=state.status.value,
        final_output=state.final_output,
        total_cost_usd=state.total_cost_usd,
        iteration_count=state.iteration_count,
        task_count=len(state.tasks),
        completed_tasks=len(state.completed_tasks()),
        low_confidence=state.output_metadata.get("low_confidence", False),
        error=state.error,
    )


@app.post("/api/v1/runs/{run_id}/stream")
async def stream_run(
    run_id: str,
    user: TokenPayload = Depends(get_current_user),
) -> StreamingResponse:
    """
    Server-Sent Events stream of run progress.
    Polls internal state every 500ms and emits status change events.
    """
    state = _runs.get(run_id)
    if not state:
        raise HTTPException(status_code=404, detail="Run not found")
    if state.user_id != user.sub:
        raise HTTPException(status_code=403, detail="Access denied")

    async def event_stream() -> AsyncIterator[str]:
        import json

        last_status = None
        last_task_count = 0
        timeout_at = time.time() + 300  # 5-minute stream timeout

        while time.time() < timeout_at:
            current = _runs.get(run_id)
            if not current:
                break

            # Emit on status change
            if current.status != last_status:
                last_status = current.status
                data = {"event": "status", "status": current.status.value, "run_id": run_id}
                yield f"data: {json.dumps(data)}\n\n"

            # Emit on new completed task
            completed = len(current.completed_tasks())
            if completed > last_task_count:
                new_tasks = current.completed_tasks()[last_task_count:]
                for task in new_tasks:
                    data = {
                        "event": "task_complete",
                        "task_id": task.id,
                        "description": task.description,
                    }
                    yield f"data: {json.dumps(data)}\n\n"
                last_task_count = completed

            # Terminal states
            if current.status in (RunStatus.COMPLETED, RunStatus.FAILED):
                final_data = {
                    "event": "done",
                    "status": current.status.value,
                    "output": current.final_output,
                    "cost_usd": current.total_cost_usd,
                    "error": current.error,
                }
                yield f"data: {json.dumps(final_data)}\n\n"
                break

            await asyncio.sleep(0.5)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/api/v1/ingest", status_code=201)
async def ingest(
    request: IngestRequest,
    user: TokenPayload = Depends(get_current_user),
) -> dict:
    """Ingest a document into the RAG knowledge base."""
    metadata = {**request.metadata, "ingested_by": user.sub}
    chunk_count = await ingest_document(request.text, metadata)
    return {"chunks_created": chunk_count, "status": "ingested"}


@app.post("/api/v1/mcp/call")
async def call_mcp_tool(
    request: MCPCallRequest,
    user: TokenPayload = Depends(get_current_user),
) -> dict:
    """Directly invoke an MCP tool. Useful for testing and debugging."""
    from cortex.mcp.server import mcp as mcp_app

    # Route to the appropriate tool function
    tool_fn = {
        "search_knowledge": mcp_app.get_tool("search_knowledge"),
        "query_memory": mcp_app.get_tool("query_memory"),
        "execute_code": mcp_app.get_tool("execute_code"),
        "synthesise": mcp_app.get_tool("synthesise"),
    }.get(request.tool)

    if not tool_fn:
        raise HTTPException(status_code=404, detail=f"Tool '{request.tool}' not found")

    result = await tool_fn.fn(**request.arguments)
    return {"tool": request.tool, "result": result}


def run() -> None:
    import uvicorn

    uvicorn.run(
        "cortex.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers if settings.is_production else 1,
        reload=not settings.is_production,
        log_config=None,  # Use structlog, not uvicorn's logger
    )


if __name__ == "__main__":
    run()
