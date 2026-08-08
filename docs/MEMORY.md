# Memory System

Cortex uses a three-tier memory architecture. Each tier serves a different temporal scope and access pattern.

## Why Three Tiers

A single context window can't hold everything a user has ever done. A single vector store can't answer "what did I work on last Tuesday?" efficiently. The three-tier design puts the right data in the right store.

| Tier | Store | Access Time | Data | Lifetime |
|------|-------|-------------|------|----------|
| Working | In-process dict | <1ms | Current run context | Single run |
| Episodic | Redis sorted set | <2ms | Past run summaries | 7 days (configurable) |
| Semantic | Qdrant HNSW index | 20-100ms | Extracted facts, knowledge | Permanent |

## Working Memory

**Purpose:** Hold the current run's context within the token budget.

**Token budget:** Configurable via `MEMORY_WORKING_TOKEN_BUDGET` (default 8,192). Uses `tiktoken` to count tokens before adding each item. Refuses additions that would overflow.

**Why this matters:** Without a token budget, a long conversation history could fill the LLM context window silently, causing truncation and unexpected agent behaviour. The working memory makes the budget explicit.

## Episodic Memory

**Purpose:** "What have I done for this user recently?"

**Implementation:** Redis sorted set, scored by Unix timestamp. The most recent N episodes are retrieved in O(log N).

**What's stored:**
```json
{
  "run_id": "abc-123",
  "goal": "Summarise Q3 sales report",
  "status": "completed",
  "output_summary": "Q3 revenue grew 12% YoY driven by...",
  "task_count": 3,
  "cost_usd": 0.018,
  "timestamp": 1737014400
}
```

**TTL:** 7 days by default (`MEMORY_EPISODIC_TTL_SECONDS`). Redis auto-evicts expired entries.

**At retrieval:** Last 5 episodes injected into planner prompt as context about what was previously done.

## Semantic Memory

**Purpose:** "What facts about this user and their domain have we learned?"

**Implementation:** Qdrant collection with embedding-based similarity search. Facts persist indefinitely.

**What's stored:** Extracted facts from task results:
```json
{
  "content": "fetch_data: Q3 revenue was $4.2M, up 12% from Q2's $3.75M",
  "source": "run:abc-123",
  "entity_type": "task_result",
  "user_id": "user-456",
  "stored_at": 1737014450.0
}
```

**At retrieval:** Top 10 semantically similar facts fetched and injected into planner. Filter by `user_id` ensures isolation between users.

**Improving extraction:** The default `_extract_facts()` method is intentionally simple (task description + result text). For production, replace it with an LLM call:

```python
async def _extract_facts_llm(self, state: CortexState) -> list[dict]:
    prompt = f"""
Extract factual statements from this run output. One fact per line, as JSON.
Only extract objective, durable facts — not temporary states.

Output: {state.final_output[:2000]}
Task results: {[t.result for t in state.completed_tasks()]}

Return JSON array: [{{"content": "...", "entity_type": "..."}}]
"""
    # Call LLM, parse response
    ...
```

## Memory Flow in a Run

```
load_memory_node:
  ├── episodic.retrieve_recent(user_id, limit=5)     # "I summarised Q3 for this user last week"
  └── semantic.retrieve(user_goal, user_id, top_k=10) # "Q3 revenue was $4.2M, EBITDA margin 18%"
           │
           ▼
   MemoryContext → injected into planner prompt
           │
   [run executes]
           │
           ▼
save_memory_node:
  ├── episodic.store(summary)                         # "Completed: summarise Q4 report, cost $0.02"
  └── semantic.store_facts([extracted_facts])          # "Q4 revenue $4.8M up 14% QoQ"
```

## Querying Memory via MCP

```python
# Claude Desktop can query memory directly
memory = await query_memory(
    query="what revenue figures do you know about this company",
    user_id="user-123",
    memory_type="semantic",
    limit=10
)
```

## Configuration

```env
MEMORY_EPISODIC_TTL_SECONDS=604800    # 7 days
MEMORY_WORKING_TOKEN_BUDGET=8192      # ~6K words
MEMORY_CONSOLIDATION_THRESHOLD=10    # consolidate after 10 episodes
```

## Multi-tenancy

Episodic memory is isolated by `user_id` (Redis key includes user_id).  
Semantic memory is isolated by Qdrant filter (`user_id` field in payload).  
Tenants never see each other's memories.
