# ADR 004: Three-Tier Memory Architecture

**Status:** Accepted  
**Date:** 2026-01  

---

## Context

Agentic systems need memory to be useful across sessions. The naive approach (include everything in the context window) doesn't scale — context windows fill up, costs increase, and retrieval quality degrades.

We needed a memory architecture that:
- Keeps the most relevant context in-context (working memory)
- Preserves session history cheaply (episodic memory)
- Stores extracted knowledge permanently for retrieval (semantic memory)
- Degrades gracefully when stores are unavailable

---

## Decision

Implement **three-tier memory** with distinct stores per tier:

| Tier | Store | Lifetime | What lives here |
|------|-------|----------|-----------------|
| Working | In-process dict | Single run | Current context, token-budget aware |
| Episodic | Redis sorted set | 7 days (TTL) | Run summaries, what was done |
| Semantic | Qdrant vectors | Permanent | Extracted facts, entity knowledge |

---

## Rationale

**Separation by temporal scope.** Working memory is hot (in-process, zero latency), episodic is warm (Redis, <1ms), semantic is cold (vector search, 50-200ms). The right tier for each query type avoids unnecessary overhead.

**Token budget in working memory.** LLM context windows are finite and expensive. Working memory tracks tokens used and rejects additions that would overflow the budget. This prevents silent context truncation, which is a common source of subtle agent failures.

**Redis for episodic over a database.** Episodic memory is accessed by recency (last N sessions), not by content. Redis sorted sets give O(log N) recency retrieval and automatic TTL without a database query. The operational simplicity outweighs the lack of durability for week-old session summaries.

**Qdrant for semantic over Redis.** Semantic facts must be retrieved by content similarity, not by key. Qdrant's HNSW index gives us sub-100ms approximate nearest neighbour search at the scale we need. A Redis vector search would work but Qdrant is purpose-built and gives better filtering support.

---

## Memory consolidation

At run end, the Memory Agent:
1. Writes a run summary (goal, status, output snippet, cost) to episodic store.
2. Extracts up to 5 facts from task results and upserts them to semantic store.
3. Does NOT copy everything — only information likely to be useful in future sessions.

Consolidation is asynchronous (the main run doesn't wait for it) and failures are non-fatal.

---

## Consequences

**Positive:**
- Agents have context from past sessions without bloating every prompt.
- Cost scales sublinearly: semantic retrieval finds the 5 most relevant facts, not all 500.
- Memory tiers can be upgraded independently (e.g., swap Redis for DynamoDB without touching semantic store).

**Negative:**
- Three stores = three potential failure points. We handle this with defensive fallbacks — each tier failure is logged but non-fatal.
- Fact extraction quality is limited (currently heuristic, not LLM-based). This means not all useful information is captured.

---

## Future Work

- LLM-based fact extraction at consolidation time (currently heuristic)
- Memory eviction policy for semantic store (currently grows indefinitely per user)
- Cross-user shared knowledge base for tenant-level memory
