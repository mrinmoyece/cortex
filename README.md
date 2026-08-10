# Cortex

**An MCP-native agentic platform: planner/executor/critic over hybrid RAG, with the budget, safety and observability layers a production deployment needs.**

384 tests · 84% coverage, gate-enforced · runs offline against fakes · no service required to develop

Cortex combines MCP, multi-agent orchestration, hybrid RAG, three-tier memory, LLMOps, automated evaluation and safety guardrails in one codebase. What it is *not* is a system that has been run at scale: it has never served production traffic, and the load-testing and scale-out work is listed in [LIMITATIONS.md](docs/LIMITATIONS.md) rather than implied here.

---

## Why Cortex

Most AI projects are tutorials. Cortex is built the way a senior AI engineer would build it for production:

- **MCP-native** — all capabilities exposed as Model Context Protocol tools. Connect Claude Desktop, VS Code, or any MCP client to your running Cortex server in minutes.
- **Multi-LLM via LiteLLM** — swap between OpenAI, Anthropic, Azure, Bedrock, or Vertex by changing one config value.
- **Full observability** — every LLM call, agent step, and tool invocation is an OTEL span, visible in Arize Phoenix and Grafana.
- **Automated evaluation** — a regression suite on a 6-hourly schedule with a faithfulness threshold. Ragas is an optional extra and currently fails to import against the released langchain-community, so the default install scores with an LLM-as-judge fallback — [LIMITATIONS.md](docs/LIMITATIONS.md) says which you are getting.
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

- Python 3.10+ (CI tests 3.10 and 3.13)
- Docker + Docker Compose
- At least one LLM API key (OpenAI recommended for quickstart)

### 1. Clone and configure

```bash
git clone https://github.com/mrinmoyece/cortex
cd cortex
cp .env.example .env
# Edit .env — set OPENAI_API_KEY and SECRET_KEY at minimum.
# SECRET_KEY is what signs API tokens: leave it unset and one is generated
# per process, so every restart invalidates every token in flight.
```

### 2. Start all services

```bash
docker compose up -d
```

Services will be available at:
| Service | URL |
|---------|-----|
| Cortex API | http://localhost:8000/docs |
| Cortex MCP | http://localhost:8001 (compose sets `MCP_TRANSPORT=http`; stdio is the default elsewhere) |
| Arize Phoenix | http://localhost:6006 |
| Grafana | http://localhost:3000 (admin / `$GRAFANA_ADMIN_PASSWORD`, default `cortex`) |
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

### 5. Check on the run, or call a tool directly

```bash
export TOKEN=$(python scripts/gen_token.py)

# Poll the run you just created
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/api/v1/runs/<run_id>

# Call one MCP tool directly. The endpoint authenticates you and binds your
# identity to the call, so tools that read your memory work here and nowhere
# else over the network.
curl -X POST http://localhost:8000/api/v1/mcp/call \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"tool": "search_knowledge", "arguments": {"query": "quickstart", "top_k": 3}}'
```

Reachable tools are `MCP_HTTP_TOOL_ALLOWLIST`. Anything else is a 404;
arguments a tool cannot accept are a 422; a tool needing an identity it does
not have is a 403.

### 6. Connect Claude Desktop

Add to your Claude Desktop `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "cortex": {
      "command": "/path/to/your/venv/bin/cortex-mcp",
      "env": {
        "MCP_TRANSPORT": "stdio",
        "OPENAI_API_KEY": "sk-...",
        "SECRET_KEY": "your-32-plus-char-key"
      }
    }
  }
}
```

`cortex-mcp` is installed by `pip install -e .`; use its absolute path,
because Claude Desktop does not inherit your shell's `PATH` or your `.env`.
The transport must be `stdio` here — that is the default, and it is what
Claude Desktop speaks.

`query_memory` reads memory belonging to a specific person, and an MCP
transport carries no authentication to derive that from. Add
`"MCP_PRINCIPAL_USER_ID": "your-user-id"` to the `env` block above to declare
whose memory this single-user process may read; without it the tool refuses.
The declaration is honoured on stdio only — on the HTTP transport it is
ignored, because one identity shared across every caller on a port is not an
identity. See [docs/MCP.md](docs/MCP.md).

### Running without Docker

The stack needs Redis and Qdrant; everything else runs in-process.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env            # set OPENAI_API_KEY and SECRET_KEY

docker run -d -p 6379:6379 redis:7-alpine
docker run -d -p 6333:6333 qdrant/qdrant:latest

cortex-api                      # http://localhost:8000/docs
cortex-mcp                      # separate shell; MCP_TRANSPORT=http to reach it over HTTP
cortex-worker                   # separate shell; only if you use the queues
```

Each of those is a console script installed by `pip install -e .`. Verify the
API is up before going further:

```bash
curl http://localhost:8000/health
# {"status":"ok","service":"cortex","environment":"local", ...}
```

Then mint a token for the curl examples above:

```bash
export TOKEN=$(python scripts/gen_token.py)
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/api/v1/runs ...
```

The test suite needs none of this — Redis is faked and every network client
is mocked, so `pytest` runs on a laptop with no services at all.

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
│   │   └── memory_agent.py    # Three-tier memory: working, episodic, semantic
│   ├── rag/
│   │   └── pipeline.py        # Ingest, chunk, embed, hybrid search, rerank
│   ├── safety/
│   │   ├── middleware.py      # PII scanner, injection detector, guardrails
│   │   └── moderation.py      # Layered input/output moderation, fails closed
│   ├── obs/
│   │   ├── metrics.py         # Prometheus metrics
│   │   └── tracing.py         # OTEL spans + Phoenix setup
│   ├── eval/
│   │   ├── ragas_runner.py    # RAG quality, with an LLM-judge fallback
│   │   └── agent_eval.py      # Agent-level metrics from run structure
│   ├── workers/
│   │   └── celery_app.py      # Background tasks and the beat schedule
│   └── api/
│       ├── main.py            # FastAPI app — all HTTP endpoints
│       ├── auth.py            # JWT authentication
│       └── ratelimit.py       # Token-bucket limiter, before routing
├── config/
│   └── rails/                 # NeMo Guardrails Colang policies
├── obs/
│   ├── prometheus.yml         # Prometheus scrape config
│   └── grafana/               # Grafana dashboard JSON + provisioning
├── perf/
│   ├── benchmark.py           # Edge latency gate — exits non-zero on breach
│   └── locustfile.py          # Load profile for a deployed instance
├── scripts/
│   ├── audit.py               # Project-scoped dependency audit
│   └── gen_token.py           # Local API token for the curl examples
├── tests/                     # Pytest test suite (80%+ coverage required)
├── deploy/k8s/                # Kubernetes manifests (unapplied — see LIMITATIONS)
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
| [PERFORMANCE.md](docs/PERFORMANCE.md) | What the latency gate measures, and what it does not |
| [LIMITATIONS.md](docs/LIMITATIONS.md) | Where this stops being production-ready, in detail |
| [SECURITY.md](SECURITY.md) | Reporting a vulnerability, and what is in scope |
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

Coverage target: **80% minimum** (enforced in CI). No service is required:
Redis is faked, Qdrant and every provider are mocked.

Everything CI runs, in the same order:

```bash
make gate                 # lint, types, tests, bandit, dependency audit, perf
```

Or individually:

```bash
make lint                 # ruff check + ruff format --check
make types                # mypy (strict, and blocking in CI)
make test                 # pytest with the coverage gate
make security             # bandit -r src -ll
make audit                # pip-audit, scoped to Cortex's dependency closure
make perf                 # edge latency budgets
```

`make audit` audits the versions installed in your environment. On a
long-lived machine those can be far older than a clean install would pick,
which reports advisories you cannot act on from this repository;
`make audit-fresh` resolves the tree from the index instead, which is what
CI installs.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). All PRs require:
- Tests for new code
- Updated ADR if architecture changes
- The full gate to pass (`make gate`)

`tests/eval/` holds the regression *cases* (`regression_cases.json`) consumed
by `cortex.eval.ragas_runner.run_regression_suite`, not pytest tests. The
evaluation harness's own tests are in `tests/test_eval/`.


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

### Second pass: what a full audit found after it ran

The first pass made Cortex run. A second, repository-wide review asked a
different question — what breaks the first time this meets a second replica,
a real client, or an attacker — and found these.

| Severity | Defect | Why nothing caught it |
|---|---|---|
| Blocking | **`docker-compose.yml` was not valid YAML.** The `depends_on` block under `x-cortex-common` was missing its first mapping key, so `docker compose up` failed before starting anything. The documented five-minute quickstart could never have worked. | Nothing in CI parses the compose file, and a human reads the shape rather than the indentation. |
| Blocking | **`POST /api/v1/mcp/call` could not succeed.** It awaited `get_tool()` — a coroutine in FastMCP 3 — and then reached for `.fn` on it. | The endpoint had no test. |
| Blocking | **The graph interrupted before the critic unconditionally**, with no resume endpoint anywhere, so every run stopped half-done and was reported as `COMPLETED`. | The test asserted the interrupt was configured, which it was. It never ran a graph to the end. |
| Blocking | **The semantic cache called `AsyncQdrantClient.search`**, removed in qdrant-client 1.13. Every cache read raised `AttributeError` against a real server. | Tests mocked `.search`, so the mock kept an API alive that the library had deleted. |
| Critical | **The cache embedded a SHA-256 digest, not the prompt.** Cosine similarity between hex digests is noise, so the "semantic" cache was neither semantic nor safe — and it had no tenant scope, so a near-collision served one tenant's answer to another. | A cache that returns wrong answers looks exactly like a cache that returns right ones until someone reads a hit. |
| Critical | **`query_memory` took `user_id` as a tool argument.** The model chose whose memory to read. | The parameter looked like plumbing rather than an authorisation decision. |
| High | **`execute_code` ran arbitrary Python in the API container, enabled, advertised to the planner, and documented as "sandboxed".** | It has a timeout and an output cap, which reads like isolation and is not. |
| High | **The run store was an unbounded dict.** Every run ever accepted stayed in memory for the life of the process — a memory leak with a public trigger. | Nothing evicts, so nothing fails, until it does. |
| High | **Rate-limit buckets were keyed by `hash(token) & 0xFFFFFFFF`.** 32 bits is ~77k tokens to a coin-flip collision, and colliding principals share a bucket, so one caller can exhaust another's allowance. `hash()` is also per-process randomised. | Both bugs are invisible at the scale a test runs at. |
| High | **Sparse retrieval ignored the filters dense retrieval enforced**, so a filtered hybrid search returned unfiltered BM25 results — including, in a multi-tenant deployment, other tenants'. | The dense half was filtered, so spot checks looked right. |
| High | **`Document.id` was a SHA-256 hex digest.** Qdrant only accepts UUIDs or unsigned integers as point IDs, so ingestion rejected every point. | The RAG tests never reached a real Qdrant. |
| Medium | **Presidio and the regex fallback were mutually exclusive**, so installing Presidio *removed* credit-card and UK NINO detection — its English config ships neither recogniser. Its default entity set also classified "annual" as a date and redacted it out of user goals. | Three safety tests failed only when Presidio happened to be installed. |
| Medium | **`agent_runs_total.labels(status=RunStatus.COMPLETED)`** produced the label `RunStatus.COMPLETED` rather than `completed`, so the dashboards queried a series that did not exist. | Prometheus accepts any string as a label value. |
| Medium | **`/metrics` was public**, exposing model names, spend totals and run volumes on the same port as the API. | Metrics endpoints are conventionally open, so nobody asks what is in them. |
| Medium | **A memory-tier failure aborted the run** before the planner had a chance to work without memory. | Degradation was written as an exception path. |
| Medium | **`completion_cost` raising discarded a completion that had already been paid for** — and it raises for any model litellm has no pricing for, which is every model newer than the pinned release. | Pricing lookups are assumed infallible because they are local. |
| Medium | **The worker and beat pods ran as root with a writable root filesystem and no probes**, next to an API pod that was fully hardened. | The hardening was written once, for the deployment someone was looking at. |
| Medium | **Prometheus scraped a Redis exporter that does not exist** and loaded a rule file that was never mounted, so every alert in `obs/alerts.yaml` was inert. | An alert that never fires and an alert that cannot fire look identical. |
| Medium | **The image had no `CMD`.** `docker run cortex` exited immediately. | Every compose service named a command explicitly. |
| Low | `python-jose` pulled in `ecdsa` and its unfixed timing-attack advisory; `presidio-anonymizer` was declared, unused, and pinned `cryptography<49`; `ragas` — which does not import at all against the released langchain-community — sat in the runtime closure dragging `diskcache` and `pillow` advisories with it. | `pip-audit --strict` audited the whole machine, so its output was noise and got read as noise. |
| Low | The performance gate's `rate_limited` row called `RateLimiter.check()` in-process and reported 0.00ms against a 20ms budget. | A budget that cannot be breached is decoration. |

`mypy --strict` went from 83 errors in 21 files, marked advisory in CI, to
zero and blocking.
