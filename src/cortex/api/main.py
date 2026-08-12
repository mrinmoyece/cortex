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
import hmac
import json
import time
import uuid
from collections import OrderedDict
from collections.abc import AsyncIterator, Awaitable, Callable
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
from cortex.mcp.client import get_mcp_client
from cortex.mcp.server import Principal, ingest_document
from cortex.obs.metrics import (
    api_request_duration,
    api_requests_total,
    configure_observability,
)
from cortex.safety.middleware import SafetyMiddleware

configure_logging()
logger = get_logger(__name__)

# ── In-memory run store ───────────────────────────────────────────────────────
#
# Bounded and TTL'd, but still a *development* store. It is per-process, so
# with `api_workers > 1` a poll can land on a worker that never saw the run,
# and it does not survive a restart. Replace with Redis/Postgres for real
# deployments - see docs/LIMITATIONS.md. What it must not be is an unbounded
# dict: every run added an entry that was never removed, so a long-lived API
# process grew until the OOM killer resolved it.


class _RunStore:
    """A bounded, TTL'd map of run_id -> state."""

    def __init__(self) -> None:
        self._runs: OrderedDict[str, tuple[float, CortexState]] = OrderedDict()

    @property
    def _max_entries(self) -> int:
        return int(settings.api_max_tracked_runs)

    @property
    def _ttl_seconds(self) -> float:
        return float(settings.api_run_retention_seconds)

    def _evict(self) -> None:
        cutoff = time.time() - self._ttl_seconds
        for run_id in [rid for rid, (ts, _) in self._runs.items() if ts < cutoff]:
            self._runs.pop(run_id, None)
        while len(self._runs) > self._max_entries:
            self._runs.popitem(last=False)

    def put(self, run_id: str, state: CortexState) -> None:
        self._runs[run_id] = (time.time(), state)
        self._runs.move_to_end(run_id)
        self._evict()

    def get(self, run_id: str) -> CortexState | None:
        entry = self._runs.get(run_id)
        if entry is None:
            return None
        created_at, state = entry
        if time.time() - created_at > self._ttl_seconds:
            self._runs.pop(run_id, None)
            return None
        # Capacity eviction is LRU, not FIFO: a run a client is still polling
        # must not be the one dropped to make room. The timestamp is *not*
        # refreshed - the TTL bounds age, this bounds count.
        self._runs.move_to_end(run_id)
        return state

    def clear(self) -> None:
        self._runs.clear()

    def __len__(self) -> int:
        return len(self._runs)


_runs = _RunStore()


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
async def telemetry_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start

    labels = {
        "method": request.method,
        "endpoint": getattr(request.scope.get("route"), "path", "unmatched"),
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
            tenant_id=user.tenant,
        )
        if state.final_output:
            state = state.model_copy(
                update={"final_output": await safety.check_output(state.final_output)}
            )
        _runs.put(run_id, state)
    except Exception as exc:
        logger.error("run.background_error", run_id=run_id, error=str(exc))
        # Store a failed state so callers don't hang
        _runs.put(
            run_id,
            CortexState(
                run_id=run_id,
                session_id=request.session_id or run_id,
                user_id=user.sub,
                user_goal=request.goal,
                status=RunStatus.FAILED,
                error=str(exc),
            ),
        )


# ── Routes ────────────────────────────────────────────────────────────────────


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok", "service": "cortex", "environment": settings.environment.value}


@app.get("/metrics")
async def metrics(request: Request) -> Response:
    """Prometheus scrape endpoint.

    Served on the public API port, so it is optionally token-protected.
    Cortex metrics carry model names, cost totals, endpoint paths and run
    volumes - a useful reconnaissance surface, and a direct read on what a
    business is spending. `METRICS_TOKEN` is unset by default because the
    endpoint is harmless when the port is not routable; where it is
    routable, set it and give Prometheus the same bearer token.
    """
    expected = settings.metrics_token
    if expected is not None:
        header = request.headers.get("authorization", "")
        scheme, _, presented = header.partition(" ")
        if scheme.lower() != "bearer" or not hmac.compare_digest(
            presented, expected.get_secret_value()
        ):
            return Response(
                status_code=401,
                content="Unauthorized",
                media_type="text/plain",
                headers={"WWW-Authenticate": "Bearer"},
            )
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
    pending = CortexState(
        run_id=run_id,
        session_id=request.session_id or run_id,
        user_id=user.sub,
        user_goal=request.goal,
        status=RunStatus.PENDING,
    )
    _runs.put(run_id, pending)
    background_tasks.add_task(_execute_run, run_id, request, user)
    logger.info("run.created", run_id=run_id, user_id=user.sub)
    return RunResponse(
        run_id=run_id,
        status=RunStatus.PENDING.value,
        created_at=pending.created_at.isoformat(),
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
) -> dict[str, Any]:
    """Ingest a document into the RAG knowledge base."""
    metadata = {**request.metadata, "ingested_by": user.sub}
    chunk_count = await ingest_document(request.text, metadata)
    return {"chunks_created": chunk_count, "status": "ingested"}


@app.post("/api/v1/mcp/call")
async def call_mcp_tool(
    request: MCPCallRequest,
    user: TokenPayload = Depends(get_current_user),
) -> dict[str, Any]:
    """Directly invoke an MCP tool.

    Two things changed here and both were bugs.

    The dispatch table called `mcp_app.get_tool(...)` synchronously and then
    `.fn(...)` on the result. In FastMCP v3 `get_tool` is a coroutine, so
    this built a dict of coroutine objects and raised AttributeError on
    every single call - the endpoint could not work at all.

    Second, the tool set is an explicit allowlist from configuration rather
    than whatever happens to be registered on the MCP server. Adding a tool
    for an MCP client should not silently publish it on an HTTP endpoint.
    """
    if request.tool not in set(settings.mcp_http_tool_allowlist):
        raise HTTPException(status_code=404, detail=f"Tool '{request.tool}' not available")

    # The principal is bound out of band. Tools that act on a user's data
    # read it from here, never from `arguments`, which the caller controls.
    principal = Principal(user_id=user.sub, tenant_id=user.tenant)
    # Argument and authorisation failures arrive as `MCPToolArgumentError`
    # (422) and `MCPPermissionError` (403), and the CortexError handler maps
    # them. There is deliberately no `except TypeError` here: it could never
    # fire, because the client wrapped every exception into MCPToolError, so
    # a mistyped argument came back as a 500.
    result = await get_mcp_client().call_tool(request.tool, request.arguments, principal=principal)
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
