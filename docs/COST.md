# Cost Management

Cortex treats LLM cost as a first-class concern. Every token spent is tracked, cached, and bounded.

## Budget Enforcement

Every agent run has a hard cost ceiling (`MAX_COST_PER_RUN_USD`, default $2.00).

```python
# In LLMRouter.complete(), before every LLM call:
current_cost = await self._cost_tracker.get_run_cost(run_id)
if current_cost >= settings.max_cost_per_run_usd:
    raise LLMBudgetExceededError(...)
```

The check happens in Redis — it's consistent across parallel task execution. If two tasks execute concurrently and one pushes the run over budget, the other will see the updated total and stop.

**Per-organisation limits:** Extend the `CostTracker` with an organisation-level ledger if you need to enforce monthly budgets across multiple users.

## Semantic Cache

Before an LLM call, the router embeds the **prompt text** and searches Qdrant
for a near-identical previous response, within the caller's own scope:

```python
cached = await self._cache.get(prompt, model=model, scope=cache_scope)
if cached is not None:
    return cached  # Free — no LLM call
```

Two things here were wrong and are worth stating plainly, because both
produced *wrong answers* rather than slow ones:

- **It embedded a SHA-256 digest, not the prompt.** Cosine similarity between
  hex digests is noise — two semantically identical prompts hash to unrelated
  strings, and two unrelated prompts can embed closer than two paraphrases.
  The "semantic" cache was neither semantic nor sound. The prompt text is now
  what gets embedded; only its digest is stored in the payload, so the cache
  can be checked for an exact match without keeping prompt text at rest.
- **There was no scope.** With a 0.95 threshold and no tenant partition, a
  near-collision served one tenant's answer to another. Reads and writes are
  now filtered on `scope` *and* `model`, and **the cache is skipped entirely
  when no scope is supplied** — an unscoped call is a caller that has not
  thought about isolation, and the safe response to that is a cache miss, not
  a shared bucket. `cache_scope=state.tenant_id` is threaded through the
  planner, critic and executor.

**Cache hit threshold:** `SEMANTIC_CACHE_SIMILARITY_THRESHOLD=0.95` (default). Tune down to 0.90 for more hits at the cost of occasional stale answers — and note that lowering it widens the near-collision window the scope filter now contains.

**Cache TTL:** 24 hours. After that, the response is re-fetched from the LLM.

**Failure mode:** the cache fails **open**. A Qdrant outage produces a miss and
a real LLM call, because a cache is an optimisation. (The moderation layer
fails *closed*, for the opposite reason — see [GUARDRAILS.md](GUARDRAILS.md).)

**When it helps most:**
- High-traffic deployments where many users ask equivalent questions
- Scheduled workflows that run the same analysis daily
- Agent reruns during development/debugging

**When to disable:** `use_cache=False` on the `router.complete()` call. Already disabled for streaming.

## Cost Tracking

Every LLM call is logged to a Redis list:

```
cortex:cost:{run_id} → [
  {"model": "gpt-4o", "cost_usd": 0.003, "prompt_tokens": 512, "completion_tokens": 128, ...},
  {"model": "gpt-4o-mini", "cost_usd": 0.0002, ...},
  ...
]
```

Get a run's cost breakdown:
```python
from cortex.llm.cost_tracker import CostTracker
tracker = CostTracker()
summary = await tracker.get_run_summary("run-abc-123")
# {
#   "total_cost_usd": 0.0182,
#   "total_prompt_tokens": 4230,
#   "total_completion_tokens": 1150,
#   "calls": 5,
#   "by_model": {"gpt-4o": 0.015, "gpt-4o-mini": 0.003}
# }
```

## Model Routing for Cost Optimisation

Configure `config/models.yaml` to route cheaper tasks to cheaper models:

```yaml
routing:
  planning:    { primary: gpt-4o }          # $0.0025/1K input — quality matters
  execution:   { primary: gpt-4o-mini }     # $0.00015/1K input — 16× cheaper
  critic:      { primary: gpt-4o }          # Quality gate — use good model
  synthesis:   { primary: gpt-4o-mini }     # Cheap for summaries
```

A typical run with this config:
- Planning: 1 call × $0.003 = $0.003
- Execution: 3 tasks × 2 tool-call rounds × $0.0003 = $0.002
- Critic: 1 call × $0.004 = $0.004
- Total: ~$0.009 per run

## Monitoring Cost in Grafana

Key panels in the **💰 Cost & Usage** row:
- **LLM Spend (1h)** — current hourly burn rate with colour-coded threshold
- **Cost by Model** — time-series showing which model is spending most
- **Total Tokens (1h)** — input vs output split

Useful Prometheus queries:
```promql
# Projected monthly cost
rate(cortex_llm_cost_usd_total[24h]) * 86400 * 30

# Cost saved by cache
rate(cortex_llm_cache_hits_total[1h]) * 0.003  # approx $0.003 saved per cache hit
```

## Reducing Costs Without Sacrificing Quality

1. **Tune chunk retrieval** — `RAG_TOP_K_RERANK=5` means the LLM only sees 5 chunks, not 20
2. **Shorten system prompts** — every 1K tokens of system prompt adds ~$0.0025 per call
3. **Increase cache threshold** — lower to 0.90 for more cache hits on similar queries
4. **Use gpt-4o-mini for execution** — saves ~14× on tool-call rounds with minimal quality loss
5. **Limit max_critique_iterations** — set to 1 in cost-sensitive deployments
