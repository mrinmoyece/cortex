# Cortex Architecture

## System Overview

Cortex is a production MCP-native agentic platform. The design follows one rule: every component should be replaceable independently without rewriting the rest.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           External Clients                               │
│  Claude Desktop  │  VS Code Copilot  │  Custom Agents  │  REST API      │
└────────┬─────────────────┬──────────────────┬──────────────────────────┘
         │ MCP / SSE        │ MCP              │ HTTPS
┌────────▼─────────────────▼──────────────────▼──────────────────────────┐
│                      Cortex API Gateway (FastAPI)                         │
│  POST /api/v1/runs   GET /runs/{id}   SSE /stream   GET /metrics        │
│  JWT Auth  │  Rate Limiting  │  CORS  │  OTEL Middleware                 │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────────────┐
│                    LangGraph Orchestration Graph                          │
│                                                                           │
│   load_memory ──► planner ──► executor ──► critic ──► save_memory       │
│        │                          │           │                           │
│   (async)                     (MCP tools)  (reject? replan)              │
└──────────────────┬──────────────────────────────────────────────────────┘
                   │
    ┌──────────────┼──────────────────┬──────────────────────┐
    │              │                  │                       │
┌───▼───┐    ┌────▼────┐    ┌────────▼──────┐    ┌─────────▼────────┐
│  MCP  │    │   RAG   │    │    Memory     │    │    Safety        │
│Server │    │Pipeline │    │   3-Tier      │    │  Middleware      │
│       │    │         │    │               │    │                  │
│search │    │ingest   │    │working: dict  │    │NeMo Guardrails   │
│memory │    │chunk    │    │episodic: Redis│    │Presidio PII      │
│code   │    │embed    │    │semantic: Qdrt │    │Injection Detect  │
│data   │    │BM25     │    │               │    │                  │
│synth  │    │rerank   │    └───────────────┘    └──────────────────┘
└───────┘    └─────────┘
    │              │
┌───▼──────────────▼──────────────────────────────────────────────────────┐
│                        LiteLLM Router                                     │
│  Budget Gate → Semantic Cache → Primary Model → Retry → Fallback Model   │
│  OpenAI  │  Anthropic  │  Azure OpenAI  │  Bedrock  │  Vertex  │  Ollama │
└────────────────────────────────────────────────────────────────────────-─┘
         │                   │                      │
    ┌────▼────┐        ┌─────▼─────┐    ┌──────────▼─────────────────────┐
    │ Qdrant  │        │   Redis   │    │  Observability Stack            │
    │         │        │           │    │  OTEL Traces → Arize Phoenix    │
    │RAG vecs │        │episodic   │    │  Prometheus Metrics → Grafana   │
    │sem mem  │        │cost ledger│    │  Structlog → stdout / JSON      │
    │llm cache│        │celery     │    │                                  │
    └─────────┘        └───────────┘    └──────────────────────────────────┘
```

## Component Responsibilities

### API Gateway (`src/cortex/api/`)
- Receives all external requests
- JWT authentication and scope checking
- Rate limiting per user
- Delegates run execution to background Celery tasks
- Exposes SSE streaming endpoint for real-time progress
- Serves Prometheus metrics at `/metrics`

### LangGraph Orchestration (`src/cortex/graph/`)
- Stateful graph where every node receives and returns the full `CortexState`
- `load_memory` — retrieves episodic + semantic context before planning
- `planner` — LLM call that produces an ordered task list
- `executor` — calls MCP tools to complete each task; handles retries
- `critic` — scores output quality; triggers replanning if below threshold
- `save_memory` — persists run summary and extracted facts
- `interrupt_before=["critic"]` enables human-in-the-loop review

### MCP Server (`src/cortex/mcp/`)
- FastMCP server exposing 5 tools: search_knowledge, query_memory, execute_code, query_data, synthesise
- Tools are the same functions called by the API and agents — no duplication
- Runs as a separate process; connects to agent graph via MCPClient

### RAG Pipeline (`src/cortex/rag/`)
- Ingestion: text → semantic chunks → embeddings → Qdrant + BM25 index
- Retrieval: query → parallel dense (Qdrant) + sparse (BM25) → RRF fusion → Cohere rerank
- Evaluation: Ragas pipeline measuring faithfulness, context precision, answer relevancy

### Memory System (`src/cortex/agents/memory_agent.py`)
- WorkingMemory: in-process, token-budget aware
- EpisodicMemory: Redis sorted set, TTL-based, recency retrieval
- SemanticMemory: Qdrant collection, embedding-based similarity retrieval
- Consolidation at run end: episodic write + fact extraction to semantic

### LLM Router (`src/cortex/llm/`)
- LiteLLM wrapping all provider calls through a single interface
- Per-run cost ledger in Redis — enforces MAX_COST_PER_RUN_USD before each call
- Semantic cache in Qdrant — skips the LLM for near-identical prompts
- Exponential backoff retry (3 attempts) + fallback model on persistent failure

### Safety Layer (`src/cortex/safety/`)
- Input: injection detection (regex patterns) → PII detection + redaction
- Output: PII redaction before returning to users
- NeMo Guardrails: policy engine for topic scoping, harmful content, jailbreak blocking

### Observability (`src/cortex/obs/`)
- OpenTelemetry: every LLM call and agent step is a span
- Arize Phoenix: LLM-specific trace analysis (prompt/response pairs, token usage)
- Prometheus: 20+ custom metrics (cost, latency, quality scores, safety violations)
- Grafana: 4-section dashboard (Cost, Latency, Quality, Safety)

### Evaluation (`src/cortex/eval/`)
- RagasEvaluator: faithfulness, context precision, answer relevancy
- LLM-as-judge fallback when Ragas isn't available
- Regression suite runs on every deployment and every 6 hours via Celery beat
- Results emitted as Prometheus metrics for Grafana trending

## Data Flow: A Complete Agent Run

```
1. POST /api/v1/runs {"goal": "Summarise Q3 sales report"}
2. API creates run_id, stores pending state, starts background task
3. Returns 202 {"run_id": "abc", "status": "pending"}

4. [Background] load_memory_node:
   - Fetch 5 recent episodes from Redis for this user
   - Fetch 10 semantically similar facts from Qdrant
   - Inject into state.memory_context

5. planner_node:
   - Build prompt with goal + memory context
   - LLM call → JSON task list (3 tasks: fetch_data, analyse, summarise)
   - Budget check: $0.003 spent

6. executor_node (×3, one per task):
   - For each task: call MCP tool (search_knowledge returns 5 chunks)
   - LLM synthesises result using retrieved context
   - Mark task completed with result string

7. executor_node detects all tasks done → compiles final output

8. critic_node:
   - Evaluates output: faithfulness=0.91, completeness=0.88, coherence=0.94
   - overall=0.91 ≥ 0.80 → accepted=True

9. save_memory_node:
   - Write run summary to Redis (episodic)
   - Extract 3 facts to Qdrant (semantic)

10. GET /api/v1/runs/abc → {"status": "completed", "final_output": "...", "cost_usd": 0.018}
```

## Deployment Topology

**Local development:** `docker compose up` starts all 10 services.

**Production (cloud):** 
- API and Workers run as Kubernetes Deployments (HPA configured)
- Qdrant: managed cloud or self-hosted StatefulSet
- Redis: AWS ElastiCache / Azure Cache for Redis
- PostgreSQL: AWS RDS / Azure Database for PostgreSQL
- Observability: self-hosted Phoenix + Grafana, or Arize Cloud + Grafana Cloud

## Security Model

- **Authentication:** JWT tokens (HS256, 1-hour expiry) + API keys for service-to-service
- **Authorisation:** User-scoped runs — users can only read their own runs
- **Network:** All inter-service traffic within the `cortex-net` Docker network; no external ports except API (8000) and observability UIs
- **Container:** Non-root user (`cortex:cortex`, UID 1000), read-only root filesystem
- **Secrets:** Never in environment variables directly in production — use AWS Secrets Manager / Azure Key Vault, injected as K8s secrets
