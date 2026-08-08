# LLMOps & Observability

Cortex is instrumented at three levels: distributed traces (OpenTelemetry + Arize Phoenix), application metrics (Prometheus + Grafana), and structured logs (structlog).

## Quick Start

```bash
docker compose up -d phoenix prometheus grafana
```

| Tool | URL | Purpose |
|------|-----|---------|
| Arize Phoenix | http://localhost:6006 | LLM trace analysis — prompt/response pairs, token counts, span timelines |
| Grafana | http://localhost:3000 (admin/cortex) | Dashboards — cost, latency, quality, safety |
| Prometheus | http://localhost:9090 | Raw metrics, alert rules |

## OpenTelemetry Tracing

Every component is instrumented:

```
Agent Run (root span)
├── load_memory (span)
│   ├── episodic_retrieve (span)
│   └── semantic_retrieve (span)
├── planner (span)
│   └── LLM: gpt-4o (span) — prompt, response, tokens, cost
├── executor (span, one per task)
│   ├── MCP: search_knowledge (span)
│   └── LLM: gpt-4o-mini (span)
├── critic (span)
│   └── LLM: gpt-4o (span)
└── save_memory (span)
```

Configure:
```
OTEL_EXPORTER_ENDPOINT=http://localhost:4317  # OTLP gRPC
PHOENIX_ENDPOINT=http://localhost:6006
```

In Phoenix, you can:
- See every prompt and response for debugging
- Filter by model, cost, latency
- Identify which agent step is the bottleneck
- Compare before/after a model change

## Prometheus Metrics Catalogue

All metrics are prefixed `cortex_`.

### LLM Metrics

| Metric | Type | Labels | What to watch |
|--------|------|--------|--------------|
| `cortex_llm_request_duration_seconds` | Histogram | `model` | p95 > 10s → slow model or network issue |
| `cortex_llm_tokens_total` | Counter | `model`, `token_type` | Sudden spike → runaway agent or bad prompt |
| `cortex_llm_cost_usd_total` | Counter | `model` | Monitor rate — alert if $10+/hour |
| `cortex_llm_cache_hits_total` | Counter | — | Low hit rate → optimise prompts for reuse |
| `cortex_llm_errors_total` | Counter | `model`, `error_type` | Any value → investigate provider issues |

### Agent Metrics

| Metric | Type | What to watch |
|--------|------|--------------|
| `cortex_agent_runs_total` | Counter (`status` label) | High `failed` rate → systemic problem |
| `cortex_agent_run_duration_seconds` | Histogram | p95 > 2min → run timeout risk |
| `cortex_agent_tasks_per_run` | Histogram | Very high → planner generating too many tasks |
| `cortex_agent_iterations_per_run` | Histogram | Hitting max → critic-replan loop cycling |

### Quality Metrics

| Metric | Type | What to watch |
|--------|------|--------------|
| `cortex_hallucination_score` | Histogram | p10 < 0.70 → RAG quality problem or prompt drift |
| `cortex_critic_score` | Histogram | Median < 0.75 → output quality degraded |
| `cortex_critic_rejections_total` | Counter | Spike → planner producing bad plans |

### Safety Metrics

| Metric | Type | Labels | What to watch |
|--------|------|--------|--------------|
| `cortex_safety_violations_total` | Counter | `violation_type` | Spike in `prompt_injection` → active attack |

### API Metrics

| Metric | Type | Labels | What to watch |
|--------|------|--------|--------------|
| `cortex_api_request_duration_seconds` | Histogram | `method`, `endpoint`, `status_code` | p95 > 500ms → bottleneck |
| `cortex_api_requests_total` | Counter | `method`, `endpoint`, `status_code` | 5xx rate > 1% → error |

## Grafana Dashboard

Pre-built dashboard at `obs/grafana/dashboards/cortex.json`. Auto-provisioned by Docker Compose.

Panels are organised in 4 rows:
1. **💰 Cost & Usage** — Spend per hour, token consumption, cache hit rate, cost by model over time
2. **⚡ Latency** — LLM p50/p95/p99 by model, agent run duration distribution
3. **🎯 Quality & Evaluation** — Faithfulness score, critic acceptance rate, quality over time
4. **🛡️ Safety** — Violation counts by type, agent run success/failure ratio

## Alert Rules

Defined in `obs/alerts.yaml`. Key alerts:

| Alert | Condition | Severity |
|-------|-----------|----------|
| `CortexHighCostPerRun` | > $10/hour LLM spend | Warning |
| `CortexHighLLMLatency` | p95 > 15s | Warning |
| `CortexLowHallucinationScore` | Median faithfulness < 0.75 | Critical |
| `CortexHighCriticRejectionRate` | > 50% rejection rate | Warning |
| `CortexHighAgentFailureRate` | > 10% failure rate | Critical |
| `CortexSafetyViolationSpike` | > 10 violations in 5 min | Warning |

## Structured Logs

All logs use structlog with consistent fields:

```json
{
  "timestamp": "2025-01-15T10:30:00Z",
  "level": "info",
  "event": "llm.complete",
  "service": "cortex",
  "environment": "production",
  "run_id": "abc-123",
  "model": "gpt-4o",
  "latency_ms": 1243,
  "prompt_tokens": 512,
  "completion_tokens": 128
}
```

Key events to monitor in log aggregator (DataDog, CloudWatch, etc.):
- `run.completed` — successful runs with cost and latency
- `critic.rejected_replanning` — quality control triggered
- `safety.injection_detected` — security events
- `llm.rate_limit_falling_back` — provider issues
- `eval.regression_FAILED` — quality regression

## Cost Analysis Queries

Useful Prometheus queries for cost analysis:

```promql
# Hourly spend by model
rate(cortex_llm_cost_usd_total[1h]) * 3600 by (model)

# Estimated monthly spend at current rate
rate(cortex_llm_cost_usd_total[24h]) * 86400 * 30

# Cache savings (tokens not sent to LLM due to cache hits)
rate(cortex_llm_cache_hits_total[1h]) * avg(cortex_llm_tokens_total) * 0.0025 / 1000

# Average cost per successful run
rate(cortex_llm_cost_usd_total[1h]) / rate(cortex_agent_runs_total{status="completed"}[1h])
```
