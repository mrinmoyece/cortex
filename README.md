# Cortex

Cortex is a reference implementation of an MCP-enabled agent service. It
combines a planner/executor/critic graph, hybrid retrieval, scoped memory,
multi-provider LLM routing, safety checks, and operational instrumentation
behind a FastAPI service.

The repository is tested and has CI quality gates, but it has **not** served
production traffic. The Kubernetes manifests, load profile, evaluation
thresholds, and alert thresholds are starting points rather than evidence of
production readiness. See [Limitations and roadmap](docs/LIMITATIONS.md).

## What is implemented

- Authenticated HTTP APIs for asynchronous runs, polling, SSE status updates,
  document ingestion, and allowlisted MCP tool calls.
- A LangGraph flow:
  `load_memory -> planner -> executor -> critic -> save_memory`, including
  replanning, per-run cost limits, and optional suspension before critique.
- Five MCP tools: `search_knowledge`, `query_memory`, `query_data`,
  `synthesise`, and disabled-by-default `execute_code`.
- Dense Qdrant retrieval plus an in-process BM25 index, reciprocal-rank fusion,
  and optional Cohere reranking.
- Episodic (Redis) and semantic (Qdrant) memory scoped by tenant and user.
- LiteLLM routing with retry, fallback, Redis cost accounting, and a
  tenant-scoped Qdrant semantic cache.
- HTTP-run input/output safety checks, Prometheus metrics, OpenTelemetry
  tracing, structured logs, and Celery tasks.

The authoritative component and flow description is
[Architecture](docs/ARCHITECTURE.md).

## Local quickstart

### Run the hermetic test suite

Python 3.10 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

The default suite uses fakes and mocks; it does not require Redis, Qdrant, an
LLM key, or network access. See [Contributing](CONTRIBUTING.md) for every
development gate.

### Run the local service stack

Docker Compose requires at least one configured LLM provider for real agent
runs.

```bash
cp .env.example .env
# Set SECRET_KEY and at least one provider key in .env.
docker compose up -d
docker compose ps
curl http://localhost:8000/health
```

The local stack exposes the API documentation at
<http://localhost:8000/docs>, MCP over HTTP at
<http://localhost:8001/mcp>, Phoenix at <http://localhost:6006>, Grafana at
<http://localhost:3000>, Prometheus at <http://localhost:9090>, and Qdrant at
<http://localhost:6333/dashboard>. These ports are development defaults, not
a hardened deployment.

Create a local token and submit a document and run:

```bash
export TOKEN="$(python3 scripts/gen_token.py --user-id dev-user --tenant default)"

curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{"text":"Cortex uses a planner, executor, and critic.","metadata":{"source":"quickstart"}}'

curl -X POST http://localhost:8000/api/v1/runs \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{"goal":"Summarise the quickstart document"}'
```

The run response contains a `run_id`; poll
`GET /api/v1/runs/<run_id>` with the same bearer token. Complete deployment,
configuration, metrics, and incident procedures are in
[Operations](docs/OPERATIONS.md).

## Documentation map and ownership

`README.md` is the entry point. Each concern below has one canonical document;
other documents link to it rather than restating it. Repository ownership is
defined by [CODEOWNERS](.github/CODEOWNERS); the current owner for every entry
is `@mrinmoyece`. Changes to behavior must update the corresponding canonical
document in the same pull request.

| Concern | Canonical document | Primary implementation evidence |
|---|---|---|
| Product scope and quickstart | This README | [`src/cortex/api/main.py`](src/cortex/api/main.py), [`docker-compose.yml`](docker-compose.yml) |
| System and AI architecture, flows, state, memory, RAG, tools, safety, cost | [Architecture](docs/ARCHITECTURE.md) | [`src/cortex`](src/cortex), [`tests`](tests) |
| Deployment, configuration, SLI/SLO status, observability, and runbooks | [Operations](docs/OPERATIONS.md) | [`deploy`](deploy), [`obs`](obs), [`.env.example`](.env.example) |
| Threats, trust boundaries, and security controls | [Threat model](docs/THREAT_MODEL.md) | [`src/cortex/api/auth.py`](src/cortex/api/auth.py), [`tests/test_safety`](tests/test_safety) |
| Vulnerability reporting | [Security policy](SECURITY.md) | [GitHub private advisories](https://github.com/mrinmoyece/cortex/security/advisories/new) |
| Evaluation methodology and evidence | [Evaluation](docs/EVALUATION.md) | [`src/cortex/eval`](src/cortex/eval), [`tests/eval/regression_cases.json`](tests/eval/regression_cases.json) |
| Performance methodology and latest local report | [Performance](docs/PERFORMANCE.md) | [`perf/benchmark.py`](perf/benchmark.py), [`perf/locustfile.py`](perf/locustfile.py) |
| Known gaps and planned work | [Limitations and roadmap](docs/LIMITATIONS.md) | Source and test links within that document |
| Development, testing, and contribution | [Contributing](CONTRIBUTING.md) | [`pyproject.toml`](pyproject.toml), [CI](.github/workflows/ci.yml) |
| Major design decisions | [ADRs](docs/adr) | The four accepted records in that directory |

## Supported entry points

| Entry point | Authentication and identity |
|---|---|
| FastAPI on port 8000 | JWT bearer token for protected APIs; tenant and user come from token claims |
| `POST /api/v1/mcp/call` | JWT-authenticated and restricted by `MCP_HTTP_TOOL_ALLOWLIST` |
| Standalone MCP over stdio | Process-local identity may be declared with `MCP_PRINCIPAL_USER_ID` |
| Standalone MCP over HTTP/SSE | No authentication or per-caller identity; do not expose publicly |
| Direct Python calls to `run_cortex()` | Library caller is responsible for safety checks and identity correctness |

## Important boundaries

- Run state, rate-limit buckets, graph checkpoints, and BM25 state are
  process-local.
- Human review can suspend a run, but this repository has no approval/resume
  endpoint.
- `execute_code` is not a sandbox and is disabled by default.
- Memory and cache isolation rely on application-level payload filters, not
  separate infrastructure; the RAG knowledge corpus is shared across tenants.
- The `WorkingMemory` token-budget helper exists but is not wired into graph
  prompt construction.
- A graph-level budget/iteration guard can end execution without setting a
  terminal failed status; the LLM router's pre-call budget error is the
  dependable spend control.
- Evaluation contains harnesses and target thresholds, not a validated model
  quality baseline.
- Edge latency is measured in CI; end-to-end load testing has not been run
  against a production deployment.

Read [Limitations and roadmap](docs/LIMITATIONS.md) before using Cortex outside
local evaluation.

## License and security

Cortex is licensed under the [MIT License](LICENSE). Report vulnerabilities
privately according to the [Security policy](SECURITY.md).
