# ADR 003: Centralize model calls behind LiteLLM

- **Status:** Accepted
- **Date:** 2026-01
- **Owner:** `@mrinmoyece`

## Context

Agents, tools, moderation, and evaluation need a provider-neutral completion
interface with consistent retry, fallback, cost, cache, telemetry, and error
behavior.

## Decision

Use LiteLLM behind one Cortex router. The router:

1. checks the per-run Redis cost ledger;
2. attempts a tenant/model-scoped semantic-cache read when scope exists;
3. calls the configured primary model with retry behavior;
4. uses the configured fallback after retry exhaustion;
5. calculates and records provider cost; and
6. emits metrics and stores an eligible scoped cache entry.

Runtime model selection comes from validated settings. The current
`config/models.yaml` file is not loaded and is not an architectural authority.

## Consequences

- Provider switching does not require changes in each agent.
- Cost, cache, retry, telemetry, and failure mapping are applied consistently.
- LiteLLM model metadata and compatibility become runtime dependencies.
- Missing pricing metadata must be surfaced without inventing cost.
- Cache failures degrade to provider calls, while an unscoped call skips the
  cache to preserve isolation.
- Provider-specific features remain limited to LiteLLM's common interface or
  require explicit router work.

## Evidence

- [`src/cortex/llm/router.py`](../../src/cortex/llm/router.py)
- [`src/cortex/llm/cost_tracker.py`](../../src/cortex/llm/cost_tracker.py)
- [`src/cortex/llm/cache.py`](../../src/cortex/llm/cache.py)
- [`tests/test_llm`](../../tests/test_llm)
- [Architecture: LLM routing, cache, and cost](../ARCHITECTURE.md#llm-routing-cache-and-cost)
