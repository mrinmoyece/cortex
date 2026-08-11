# ADR 004: Separate working, episodic, and semantic memory

- **Status:** Accepted
- **Date:** 2026-01
- **Owner:** `@mrinmoyece`

## Context

Current-run prompt context, recent run history, and content-based long-term
facts have different access and retention needs. Treating all three as prompt
history or one vector collection would obscure token, recency, and isolation
policy.

## Decision

Use three tiers:

| Tier | Store | Scope | Purpose |
|---|---|---|---|
| Working | Process memory | Run | Token-budgeted active context |
| Episodic | Redis sorted set | Tenant and user | Recent run summaries with TTL |
| Semantic | Qdrant | Tenant and user | Similarity retrieval over extracted facts |

Load episodic and semantic context before planning. After graph completion,
attempt to persist the episode and heuristic facts; failure is logged but does
not discard an otherwise completed answer.

**Implementation status:** episodic and semantic tiers are integrated.
`WorkingMemory` exists and is unit tested but is not instantiated by the graph,
so its token budget is not currently enforced in prompts.

## Consequences

- Each retrieval pattern uses a fitting store and can degrade independently.
- Tenant/user scoping is mandatory on shared episodic and semantic stores.
- Working state and its budget remain process-local.
- Semantic facts have no repository-defined expiry or deletion workflow.
- Heuristic fact extraction can retain noisy or incomplete information.
- Deployments own Redis/Qdrant durability, backup, and lifecycle policy.

## Evidence

- [`src/cortex/agents/memory_agent.py`](../../src/cortex/agents/memory_agent.py)
- [`tests/test_memory`](../../tests/test_memory)
- [Architecture: Memory model](../ARCHITECTURE.md#memory-model)
- [Limitations: Retrieval and memory](../LIMITATIONS.md#retrieval-and-memory-do-not-have-production-lifecycle-controls)
