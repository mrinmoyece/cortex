# ADR 003: Multi-LLM Routing via LiteLLM

**Status:** Accepted  
**Date:** 2026-01  

---

## Context

Cortex needs to call LLMs from multiple providers (OpenAI, Anthropic, Azure OpenAI, AWS Bedrock, Google Vertex). Users deploy Cortex in environments that may only have access to one provider. The application code should not need to change based on which provider is available.

Options considered:
1. Direct SDK calls per provider with an abstraction layer we build
2. LiteLLM as the abstraction layer
3. LangChain's LLM abstraction
4. OpenAI SDK only (drop non-OpenAI providers)

---

## Decision

Use **LiteLLM** as the single interface for all LLM calls.

---

## Rationale

**Universal interface.** LiteLLM provides a single `completion()` / `acompletion()` call that works with 100+ models across all major providers. We do not need to write and maintain per-provider adapters.

**Cost calculation.** LiteLLM's `completion_cost()` function provides accurate per-request cost estimates for all supported models. This is essential for our budget enforcement system.

**Fallback routing.** LiteLLM supports fallback model lists natively. We layer our own fallback logic on top (primary → fallback model) because we want explicit control over fallback behaviour and logging.

**Not LangChain.** LangChain's LLM abstraction is higher-level and pulls in a larger dependency footprint. We use LangChain for message schema compatibility (`BaseMessage`) but not for LLM calls — LiteLLM gives us a thinner, faster layer.

---

## Routing Strategy

```
Request
  │
  ▼
Budget check (Redis)
  │ over budget → raise LLMBudgetExceededError
  ▼
Semantic cache lookup (Qdrant)
  │ hit → return cached response
  ▼
Primary model (DEFAULT_MODEL)
  │ rate limit → retry with exponential backoff (3 attempts)
  │ persistent failure → fallback model
  ▼
Fallback model (FALLBACK_MODEL)
  │ failure → raise LLMProviderUnavailableError
  ▼
Record cost to Redis
Emit Prometheus metrics
Store in semantic cache
Return response
```

---

## Consequences

**Positive:**
- Single place to add cost tracking, caching, retry logic.
- Provider switching requires only a config change, no code changes.
- Fallback logic is transparent and logged.

**Negative:**
- LiteLLM adds ~50ms cold-start overhead (model config loading). Mitigated by using a module-level singleton.
- LiteLLM is a third-party dependency with its own release cadence. We pin major versions and test on upgrades.

---

## Cost Model Config

Provider routing can be extended via `config/models.yaml` to route different task types to different models:

```yaml
routing:
  planning: gpt-4o          # High-quality planning
  execution: gpt-4o-mini    # Cheap tool calls
  critic: gpt-4o            # High-quality evaluation
  embedding: text-embedding-3-large
```
