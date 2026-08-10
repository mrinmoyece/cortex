"""
Cortex observability stack.

Three layers:
  1. OpenTelemetry tracing — every LLM call, agent step, and tool invocation
     is a span. Exported to Arize Phoenix for LLM-specific analysis.
  2. Prometheus metrics — cost, latency, token usage, error rates.
     Scraped by Prometheus, visualised in Grafana.
  3. Structured logs (via structlog) — correlated with trace IDs.

Call configure_observability() once at application startup, before
any other imports that touch LLM or agent code.
"""

from __future__ import annotations

from prometheus_client import Counter, Histogram

from cortex.config import settings
from cortex.logging_config import get_logger

logger = get_logger(__name__)


# ── Prometheus Metrics ────────────────────────────────────────────────────────
# All metrics follow Prometheus naming conventions:
# - Counters end in _total
# - Histograms end in _duration_seconds or _bytes
# - Labels are lowercase_snake_case

# LLM
llm_request_duration = Histogram(
    "cortex_llm_request_duration_seconds",
    "Time from LLM request to first token",
    labelnames=["model"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
)

llm_tokens_total = Counter(
    "cortex_llm_tokens_total",
    "Total tokens consumed",
    labelnames=["model", "token_type"],  # token_type: input | output
)

llm_cost_total = Counter(
    "cortex_llm_cost_usd_total",
    "Total USD spent on LLM calls",
    labelnames=["model"],
)

llm_cache_hits_total = Counter(
    "cortex_llm_cache_hits_total",
    "Number of LLM calls served from semantic cache",
    labelnames=["model"],
)

llm_errors_total = Counter(
    "cortex_llm_errors_total",
    "LLM call failures by error type",
    labelnames=["model", "error_type"],
)

# Agent runs
agent_runs_total = Counter(
    "cortex_agent_runs_total",
    "Agent runs by final status",
    labelnames=["status"],
)

agent_run_duration = Histogram(
    "cortex_agent_run_duration_seconds",
    "End-to-end agent run duration",
    buckets=[1.0, 5.0, 15.0, 30.0, 60.0, 120.0, 300.0],
)

agent_tasks_per_run = Histogram(
    "cortex_agent_tasks_per_run",
    "Number of tasks executed per agent run",
    buckets=[1, 2, 3, 5, 8, 13],
)

agent_iterations_per_run = Histogram(
    "cortex_agent_iterations_per_run",
    "Graph iterations per run",
    buckets=[1, 2, 3, 5, 10, 20, 25],
)

# Critic
critic_rejection_total = Counter(
    "cortex_critic_rejections_total",
    "Number of critic rejections requiring replanning",
)

critic_score_histogram = Histogram(
    "cortex_critic_score",
    "Critic quality score distribution",
    buckets=[0.0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0],
)

# RAG
rag_retrieval_duration = Histogram(
    "cortex_rag_retrieval_duration_seconds",
    "RAG retrieval latency including reranking",
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)

rag_documents_ingested = Counter(
    "cortex_rag_documents_ingested_total",
    "Number of document chunks ingested into RAG pipeline",
)

# Memory
memory_operations_total = Counter(
    "cortex_memory_operations_total",
    "Memory read/write operations",
    labelnames=["tier", "operation"],  # tier: episodic|semantic|working, operation: read|write
)

# API
api_request_duration = Histogram(
    "cortex_api_request_duration_seconds",
    "FastAPI request duration",
    labelnames=["method", "endpoint", "status_code"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

api_requests_total = Counter(
    "cortex_api_requests_total",
    "Total API requests",
    labelnames=["method", "endpoint", "status_code"],
)

# Safety
safety_violations_total = Counter(
    "cortex_safety_violations_total",
    "Safety guardrail violations by type",
    labelnames=["violation_type"],
)

hallucination_score = Histogram(
    "cortex_hallucination_score",
    "Hallucination detection score (0=hallucinated, 1=faithful)",
    buckets=[0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0],
)

# Cost per run.
#
# A Histogram, NOT a Gauge labelled by run_id. `run_id` is unbounded: every
# run mints a fresh label value, each becomes a permanent time series, and
# Prometheus falls over some weeks later during an unrelated incident. It is
# the classic cardinality mistake and it is invisible until it is expensive.
#
# Nothing is lost by removing the label. Per-run cost belongs in logs and
# traces, where you can pivot on run_id; what a metric should answer is
# "what does a run cost, typically and at the tail", which a histogram
# answers directly and a per-run gauge cannot.
agent_eval_score = Histogram(
    "cortex_agent_eval_score",
    "Per-axis agent evaluation scores. Labelled by metric, never by run - "
    "the distribution is the signal, an individual run is a trace.",
    labelnames=["metric"],
    buckets=(0.0, 0.25, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0),
)

agent_step_duration = Histogram(
    "cortex_agent_step_duration_seconds",
    "Wall-clock duration of one instrumented agent step.",
    labelnames=["step"],
    buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
)

agent_step_failures = Counter(
    "cortex_agent_step_failures_total",
    "Agent steps that raised. Labelled by step and exception type - never "
    "by run or user, which are unbounded.",
    labelnames=["step", "error_type"],
)

rate_limit_rejections = Counter(
    "cortex_rate_limit_rejections_total",
    "Requests rejected by the API rate limiter. A sustained rate means "
    "either an abusive client or a limit set below legitimate demand.",
)

memory_consolidation_failures = Counter(
    "cortex_memory_consolidation_failures_total",
    "Runs whose memory write failed after the answer was produced. Degraded, "
    "not failed - but a rising rate means the agent stops learning.",
)

memory_retrieval_failures = Counter(
    "cortex_memory_retrieval_failures_total",
    "Runs that started with no memory context because a memory store was "
    "unavailable. Degraded, not failed - the agent answers without history.",
)

run_cost_usd = Histogram(
    "cortex_run_cost_usd",
    "Total cost of a completed agent run, in USD",
    buckets=(0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0),
)


# ── OpenTelemetry Setup ───────────────────────────────────────────────────────


def configure_tracing() -> None:
    """
    Configure OpenTelemetry with Arize Phoenix + OTLP export.
    Call once at startup before any LLM calls.
    """
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import SERVICE_NAME, Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        resource = Resource(attributes={SERVICE_NAME: "cortex"})
        provider = TracerProvider(resource=resource)

        if settings.otel_exporter_endpoint:
            otlp_exporter = OTLPSpanExporter(
                endpoint=str(settings.otel_exporter_endpoint),
                insecure=not settings.is_production,
            )
            provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

        trace.set_tracer_provider(provider)
        logger.info("tracing.configured", endpoint=str(settings.otel_exporter_endpoint))

    except ImportError:
        logger.warning("tracing.otel_not_installed")
    except Exception as exc:
        logger.warning("tracing.setup_failed", error=str(exc))


def configure_phoenix() -> None:
    """Register Arize Phoenix as an OTEL trace collector for LLM-specific analysis."""
    try:
        from phoenix.otel import register

        # `register()` installs itself as the global tracer provider; the
        # return value is only needed if you want to create tracers from it
        # directly, which nothing here does.
        register(
            project_name="cortex",
            endpoint=f"{settings.phoenix_endpoint}/v1/traces",
        )
        logger.info("phoenix.configured", endpoint=str(settings.phoenix_endpoint))
    except ImportError:
        logger.warning("phoenix.not_installed — pip install arize-phoenix-otel")
    except Exception as exc:
        logger.warning("phoenix.setup_failed", error=str(exc))


def configure_observability() -> None:
    """Call once at application startup."""
    configure_tracing()
    configure_phoenix()
    logger.info("observability.ready")
