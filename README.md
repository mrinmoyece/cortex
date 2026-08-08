# Cortex

**An MCP-native agentic platform: planner/executor/critic over hybrid RAG, with the budget, safety and observability layers a production deployment needs.**

241 tests · 82% coverage, gate-enforced · runs offline against fakes · no service required to develop

Cortex combines MCP, multi-agent orchestration, hybrid RAG, three-tier memory, LLMOps, automated evaluation and safety guardrails in one codebase. What it is *not* is a system that has been run at scale: it has never served production traffic, and the load-testing and scale-out work is listed in [LIMITATIONS.md](docs/LIMITATIONS.md) rather than implied here.

---

## Why Cortex

Most AI projects are tutorials. Cortex is built the way a senior AI engineer would build it for production:

- **MCP-native** — all capabilities exposed as Model Context Protocol tools. Connect Claude Desktop, VS Code, or any MCP client to your running Cortex server in minutes.
- **Multi-LLM via LiteLLM** — swap between OpenAI, Anthropic, Azure, Bedrock, or Vertex by changing one config value.
- **Full observability** — every LLM call, agent step, and tool invocation is an OTEL span, visible in Arize Phoenix and Grafana.
- **Automated evaluation** — Ragas metrics with a regression suite on a 6-hourly schedule. Faithfulness is a tracked metric with a threshold, not a hope.
- **Budget enforcement** — a per-run cost ceiling checked *before* every model call, against a Redis ledger shared across async tasks and workers.
- **Rate limiting that runs before routing** — so requests that 404 or 422 are metered too. Those are cheaper for an attacker to generate than valid ones.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         MCP Clients                                  │
│         Claude Desktop │ VS Code │ Custom Agents │ REST API          │
└─────────────────┬───────────────────────────────────────────────────┘
                  │ MCP / HTTP
┌─────────────────▼───────────────────────────────────────────────────┐
│                    Cortex API Gateway (FastAPI)                        │
│              Auth │ Rate Limiting │ SSE Streaming │ /metrics          │
└─────────────────┬───────────────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────────────┐
│                 LangGraph Agent Orchestration                         │
│        load_memory → planner → executor → critic → save_memory       │
└──────┬──────────┬──────────────┬──────────────┬───────────────────-─┘
       │          │              │              │
  ┌────▼───┐ ┌───▼────┐ ┌──────▼────┐ ┌──────▼──────┐
  │  MCP   │ │  RAG   │ │  Memory  │ │   Safety    │
  │ Server │ │Pipeline│ │  3-Tier  │ │ Guardrails  │
  └────┬───┘ └───┬────┘ └──────┬────┘ └─────────────┘
       │          │              │
┌──────▼──────────▼──────────────▼──────────────────────────────────-─┐
│                      LiteLLM Router                                   │
│         OpenAI │ Anthropic │ Azure │ Bedrock │ Vertex │ Ollama        │
└───────────────────────────────────────────────────────────────────-──┘
       │          │              │
  ┌────▼──┐  ┌───▼───┐  ┌──────▼──────────────────────────────────┐
  │Qdrant │  │ Redis │  │  Observability: OTEL + Phoenix + Grafana │
  └───────┘  └───────┘  └─────────────────────────────────────────┘
```

---

## Quickstart (5 minutes)

### Prerequisites

- Python 3.11+
- Docker + Docker Compose
- At least one LLM API key (OpenAI recommended for quickstart)

### 1. Clone and configure

```bash
git clone https://github.com/your-org/cortex
cd cortex
cp .env.example .env
# Edit .env — set OPENAI_API_KEY and SECRET_KEY at minimum
```

### 2. Start all services

```bash
docker compose up -d
```

Services will be available at:
| Service | URL |
|---------|-----|
| Cortex API | http://localhost:8000/docs |
| Cortex MCP | stdio (for Claude Desktop) |
| Arize Phoenix | http://localhost:6006 |
| Grafana | http://localhost:3000 (admin / cortex) |
| Qdrant UI | http://localhost:6333/dashboard |
| Prometheus | http://localhost:9090 |

### 3. Ingest a document

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Authorization: Bearer $(python scripts/gen_token.py)" \
  -H "Content-Type: application/json" \
  -d '{"text": "Your document content here", "metadata": {"source": "quickstart"}}'
```

### 4. Run an agent

```bash
curl -X POST http://localhost:8000/api/v1/runs \
  -H "Authorization: Bearer $(python scripts/gen_token.py)" \
  -H "Content-Type: application/json" \
  -d '{"goal": "Summarise the key points from the ingested documents"}'
```

### 5. Connect Claude Desktop

Add to your Claude Desktop `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "cortex": {
      "command": "python",
      "args": ["-m", "cortex.mcp.server"],
      "cwd": "/path/to/cortex"
    }
  }
}
```

---

## Project Structure

```
cortex/
├── src/cortex/
│   ├── config.py              # Pydantic Settings — all configuration
│   ├── exceptions.py          # Domain exception hierarchy
│   ├── logging_config.py      # Structlog structured logging
│   ├── mcp/
│   │   ├── server.py          # FastMCP server — MCP tool definitions
│   │   └── client.py          # MCP client for agent tool calls
│   ├── llm/
│   │   ├── router.py          # LiteLLM wrapper with retry, fallback, budget
│   │   ├── cost_tracker.py    # Per-run token cost accounting (Redis)
│   │   └── cache.py           # Semantic cache (Qdrant)
│   ├── graph/
│   │   ├── state.py           # LangGraph state schema (CortexState)
│   │   └── cortex_graph.py     # Graph topology, nodes, routing functions
│   ├── agents/
│   │   ├── planner.py         # Goal → task list decomposition
│   │   ├── executor.py        # Task execution via MCP tools
│   │   ├── critic.py          # Output quality evaluation
│   │   └── memory_agent.py    # Memory retrieval and consolidation
│   ├── rag/
│   │   └── pipeline.py        # Ingest, chunk, embed, hybrid search, rerank
│   ├── memory/                # Three-tier memory (working / episodic / semantic)
│   ├── safety/
│   │   └── middleware.py      # Guardrails, PII scanner, injection detector
│   ├── obs/
│   │   └── metrics.py         # Prometheus metrics + OTEL + Phoenix setup
│   ├── eval/
│   │   └── ragas_runner.py    # Automated Ragas evaluation suite
│   └── api/
│       ├── main.py            # FastAPI app — all HTTP endpoints
│       └── auth.py            # JWT authentication
├── config/
│   └── rails/                 # NeMo Guardrails Colang policies
├── obs/
│   ├── prometheus.yml         # Prometheus scrape config
│   └── grafana/               # Grafana dashboard JSON + provisioning
├── tests/                     # Pytest test suite (80%+ coverage required)
├── docs/adr/                  # Architecture Decision Records
├── docker-compose.yml         # Full local stack
├── Dockerfile                 # Multi-stage production image
├── pyproject.toml             # Dependencies and tooling config
└── .env.example               # All required environment variables
```

---

## Documentation

| Document | What it covers |
|----------|---------------|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | Full system design, data flows, design decisions |
| [MCP.md](docs/MCP.md) | MCP server, tool definitions, connecting clients |
| [AGENTS.md](docs/AGENTS.md) | Agent design, prompts, tool access, failure modes |
| [RAG.md](docs/RAG.md) | Ingestion, chunking strategy, hybrid search, evaluation |
| [MEMORY.md](docs/MEMORY.md) | Three-tier memory design and retrieval strategy |
| [LLMOPS.md](docs/LLMOPS.md) | Observability setup, metrics catalogue, dashboards |
| [GUARDRAILS.md](docs/GUARDRAILS.md) | Safety policies, PII handling, injection defence |
| [EVALUATION.md](docs/EVALUATION.md) | Ragas metrics, regression testing, baseline scores |
| [COST.md](docs/COST.md) | Token tracking, caching, budget enforcement |
| [DEPLOYMENT.md](docs/DEPLOYMENT.md) | Docker Compose, Azure/AWS cloud deployment |
| [TESTING.md](docs/TESTING.md) | Test strategy, coverage targets, eval tests |
| [ADR/001](docs/adr/001-mcp-over-rest.md) | Why MCP over custom REST for tool exposure |
| [ADR/002](docs/adr/002-langgraph-state.md) | State schema design decisions |
| [ADR/003](docs/adr/003-litellm-routing.md) | Multi-LLM routing strategy |
| [ADR/004](docs/adr/004-three-tier-memory.md) | Memory architecture |

---

## Running Tests

```bash
pip install -e ".[dev]"
pytest                    # Full suite with coverage
pytest tests/test_rag/    # RAG tests only
pytest -k "test_planner"  # Single test
```

Coverage target: **80% minimum** (enforced in CI).

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). All PRs require:
- Tests for new code
- Updated ADR if architecture changes
- Eval regression suite must still pass (`pytest tests/eval/`)


---

## What was wrong with this codebase, and how it was found

Cortex was generated complete — 77 files, eight layers, an 80% coverage
gate — and had never been executed. Not once. Every defect below was found
by making it run.

| Severity | Defect | Why nothing caught it |
|---|---|---|
| Blocking | **The test suite could not be collected.** Sixteen modules called `get_settings()` at import time; `SECRET_KEY` is unset on a clean machine, so `import cortex.obs.metrics` raised `ValidationError` before any test started. | `--cov-fail-under=80` was gating a suite that had never run. |
| Blocking | **`FastMCP(version=...)` is not a valid API.** Two different classes share the name; the code imported the one from the `mcp` SDK and passed it the other one's arguments. | Both packages are declared in `pyproject.toml`, which made the mistake look plausible. |
| Blocking | **The executor never passed its tool schemas to the model.** It fetched them into a local variable and dropped them. | Under a mocked router the loop still "worked" — the mock returns tool calls whether or not any tools were offered. Ruff found it, as an unused variable. |
| Blocking | **Four tests required a live Redis.** The `mock_router` fixture patched `cortex.llm.router.get_router`, but agents bind that name at import, so the patch rebound a name nobody read. | The fixture looked correct and was a no-op. |
| High | **Rate limiting did not exist.** `api_rate_limit_per_minute` was configured, documented, and read by nothing. | A control an operator believes they have. |
| High | **`query_data` was a stub advertising itself as a working tool.** It returned an empty result set with a note, so an agent would "query the data", get zero rows, and confidently report no matching records. | A tool that fakes success is worse than an absent one — the absent one can be planned around. |
| High | **`tenant_id` was carried in state, described in the docs, and used by neither memory tier.** | Safe only while user ids are globally unique, which they are not when they come from tenant-local identity providers. |
| High | **A `Gauge` labelled by `run_id`.** Every run minted a permanent time series. | The classic cardinality mistake: invisible until Prometheus falls over weeks later, during an unrelated incident. |
| Medium | **The graph's cost guard read `>= 2.0`** instead of the configured ceiling — which happens to equal the default, so it looked right and ignored its own configuration. | |
| Medium | **A memory-write failure crashed a run that had already produced its answer.** | Losing the memory write is a degradation. Losing the answer is a bug. |
| Medium | **`Document.id` was `uuid4()`** next to an unused `doc_hash` property, so re-ingesting an unchanged file duplicated the corpus instead of upserting over it. | |
| Medium | **An unexpected exception type skipped the LLM fallback entirely** and propagated raw, so the documented failure mode was not the one callers got. | |
| Low | Six dependencies declared and never imported; a Postgres service nothing connected to; `MemoryError` shadowing the builtin. | |

The reason this table is here rather than quietly fixed: the gap between
"generated" and "working" is where all the engineering lives, and every row
is a question I can answer in depth.
