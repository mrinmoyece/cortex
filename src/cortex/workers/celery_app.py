"""
Cortex Celery workers.

Background tasks:
  - run_agent_task      — Execute an agent run asynchronously
  - ingest_document     — Ingest a document into RAG (can be slow for large docs)
  - run_eval_regression — Scheduled Ragas eval suite (runs on deploy + every 6h)
  - consolidate_memory  — Scheduled memory consolidation for all active users

Beat schedule:
  - eval regression: every 6 hours
  - memory consolidation: every hour
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any, TypeVar

from celery import Celery, Task
from celery.schedules import crontab
from celery.utils.log import get_task_logger

from cortex.config import settings
from cortex.logging_config import configure_logging

T = TypeVar("T")

configure_logging()
logger = get_task_logger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────

celery_app = Celery(
    "cortex",
    broker=str(settings.redis_url).replace("/0", f"/{settings.redis_celery_db}"),
    backend=str(settings.redis_url).replace("/0", f"/{settings.redis_celery_db}"),
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_soft_time_limit=settings.celery_task_timeout_seconds,
    task_time_limit=settings.celery_task_timeout_seconds + 30,
    task_acks_late=True,  # Only ack after successful completion
    worker_prefetch_multiplier=1,  # One task at a time per worker — prevents memory issues
    task_reject_on_worker_lost=True,  # Re-queue if worker crashes
    result_expires=3600,  # Results expire after 1 hour
)

# ── Beat schedule ─────────────────────────────────────────────────────────────

celery_app.conf.beat_schedule = {
    "eval-regression-every-6h": {
        "task": "cortex.workers.celery_app.run_eval_regression",
        "schedule": crontab(minute=0, hour="*/6"),
        "options": {"queue": "eval"},
    },
    "memory-consolidation-hourly": {
        "task": "cortex.workers.celery_app.consolidate_stale_memory",
        "schedule": crontab(minute=30),
        "options": {"queue": "maintenance"},
    },
}


# ── Tasks ─────────────────────────────────────────────────────────────────────


def _run_async(coro: Coroutine[Any, Any, T]) -> T:
    """Run an async coroutine from a synchronous Celery task."""
    return asyncio.get_event_loop().run_until_complete(coro)


@celery_app.task(
    bind=True,
    max_retries=settings.celery_max_retries,
    default_retry_delay=10,
    queue="agent",
    name="cortex.workers.celery_app.run_agent_task",
)
def run_agent_task(
    self: Task, run_id: str, goal: str, user_id: str, session_id: str, context: dict
) -> dict[str, Any]:
    """Execute an Cortex agent run. Called when the API receives a run request."""
    logger.info(f"Starting agent run {run_id}")
    try:
        from cortex.graph.cortex_graph import run_cortex

        state = _run_async(
            run_cortex(
                user_goal=goal,
                user_id=user_id,
                session_id=session_id,
                context=context,
            )
        )
        logger.info(f"Run {run_id} completed: {state.status.value}")
        return {"run_id": run_id, "status": state.status.value, "cost_usd": state.total_cost_usd}

    except Exception as exc:
        logger.error(f"Run {run_id} failed: {exc}")
        try:
            raise self.retry(exc=exc, countdown=10 * (self.request.retries + 1))
        except self.MaxRetriesExceededError:
            return {"run_id": run_id, "status": "failed", "error": str(exc)}


@celery_app.task(
    queue="ingest",
    name="cortex.workers.celery_app.ingest_document_task",
)
def ingest_document_task(text: str, metadata: dict) -> dict[str, Any]:
    """Ingest a document into the RAG pipeline in the background."""
    from cortex.rag.pipeline import RAGPipeline

    rag = RAGPipeline()
    chunk_count = _run_async(rag.ingest(text, metadata))
    logger.info(f"Ingested {chunk_count} chunks from document")
    return {"chunks": chunk_count}


@celery_app.task(
    queue="eval",
    name="cortex.workers.celery_app.run_eval_regression",
)
def run_eval_regression() -> dict[str, Any]:
    """Run the Ragas regression test suite. Alerts if quality drops below threshold."""
    from cortex.eval.ragas_runner import run_regression_suite

    result = _run_async(run_regression_suite())
    passed = result.passes_threshold()
    logger.info(f"Eval regression: composite={result.composite:.3f} passed={passed}")

    if not passed:
        logger.error(
            f"REGRESSION FAILURE: faithfulness={result.faithfulness:.3f} "
            f"relevancy={result.answer_relevancy:.3f} composite={result.composite:.3f}"
        )
        # In production: trigger PagerDuty / Slack alert here

    return result.to_dict()


@celery_app.task(
    queue="maintenance",
    name="cortex.workers.celery_app.consolidate_stale_memory",
)
def consolidate_stale_memory() -> dict[str, Any]:
    """
    Placeholder for memory consolidation job.
    In production: scan episodic store for episodes ready for semantic extraction.
    """
    logger.info("Memory consolidation tick — no stale sessions in this cycle")
    return {"consolidated": 0}


def run() -> None:
    celery_app.worker_main(["worker", "--loglevel=info", "-Q", "agent,ingest,eval,maintenance"])
