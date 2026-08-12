# Threat Model

This threat model covers the repository's implemented entry points and
deployment artifacts. It is not a certification or a claim that heuristic AI
safety controls prevent all abuse. Vulnerability reporting is defined by the
[Security policy](../SECURITY.md).

## Scope and assets

Assets include:

- JWT signing keys and model/provider credentials;
- tenant documents, retrieved chunks, prompts, outputs, and memory;
- run state, graph checkpoints, cost records, and evaluation data;
- integrity of tool calls, model instructions, and safety policy;
- service availability and provider spend; and
- operational telemetry, which can reveal model names, volumes, and cost.

The API, graph, agents, tool registry, stores, worker, container manifests, and
CI automation are in scope. Model providers, managed Redis/Qdrant, ingress
controllers, secret managers, and operator identity systems are external
dependencies whose controls are not implemented here.

## Trust boundaries

```mermaid
flowchart LR
    User[Untrusted user/client] -->|JWT boundary| API[FastAPI]
    API -->|principal binding| Tool[Tool registry]
    API --> Safety[Input/output safety]
    Desktop[Local desktop client] -->|process boundary + declared identity| Stdio[MCP stdio]
    Network[Untrusted network caller] -->|no built-in auth| MCPNet[MCP HTTP/SSE]
    Tool -->|untrusted retrieved content| Agent[Agent prompts]
    Agent -->|provider credentials + prompts| Provider[External LLM]
    Agent --> Redis[(Redis)]
    Agent --> Qdrant[(Qdrant)]
    API --> Metrics[/metrics]
    Operator[Cluster/host operator] --> API
    Operator --> Redis
    Operator --> Qdrant
```

1. **Client to API:** protected endpoints verify HS256 JWTs and derive user and
   tenant identity from claims.
2. **API to tool:** authenticated MCP API calls bind a `Principal`; the model
   does not choose the user ID for memory access.
3. **Standalone MCP:** stdio trusts the process launcher. HTTP/SSE has no
   repository-provided authentication and cannot bind a safe per-caller
   principal.
4. **Retrieved content to agent:** documents and tool results are untrusted
   input even when the user trusts the corpus.
5. **Service to provider/store:** prompts, embeddings, memory, credentials, and
   metadata leave the process according to provider configuration.
6. **Metrics boundary:** `/metrics` is public unless a bearer token or network
   restriction protects it.
7. **Tenant boundary:** memory and semantic cache isolation use
   application-level keys and payload filters in shared stores. The RAG
   knowledge corpus has no principal-derived tenant boundary.

## Threats, controls, and residual risk

| Threat | Implemented controls | Evidence | Residual risk |
|---|---|---|---|
| Forged, expired, or malformed API identity | Required `sub`/`exp`, signature and expiry validation, protected routes | [`api/auth.py`](../src/cortex/api/auth.py), [`test_api.py`](../tests/test_api/test_api.py) | HS256 shares one signing secret; key distribution/rotation is external |
| Cross-user memory access through model arguments | Principal bound outside the model; memory tool accepts no caller-selected user | [`mcp/server.py`](../src/cortex/mcp/server.py), [`test_principal_binding.py`](../tests/test_mcp/test_principal_binding.py) | Standalone network MCP cannot derive identity; filter bugs remain possible |
| Cross-tenant memory/cache leakage | Tenant/user filters on memory; scoped cache; unscoped cache bypass | [`memory_agent.py`](../src/cortex/agents/memory_agent.py), [`llm/cache.py`](../src/cortex/llm/cache.py), [`test_principal_binding.py`](../tests/test_mcp/test_principal_binding.py) | Shared collections provide no second isolation layer |
| Cross-tenant RAG disclosure | No repository control currently derives an ingestion/retrieval filter from the authenticated tenant | [`api/main.py`](../src/cortex/api/main.py), [`mcp/server.py`](../src/cortex/mcp/server.py) | Any authenticated caller with knowledge-search access can search the shared corpus |
| Direct prompt injection | Scored patterns, normalization, optional rails, refusal before graph execution for HTTP runs | [`safety/middleware.py`](../src/cortex/safety/middleware.py), [`test_safety.py`](../tests/test_safety/test_safety.py) | Novel attacks and direct `run_cortex()` calls can bypass these checks |
| Indirect injection from documents/tools | Untrusted results fenced and labelled in executor prompts | [`agents/executor.py`](../src/cortex/agents/executor.py), [`test_executor.py`](../tests/test_agents/test_executor.py) | Spotlighting reduces instruction confusion; it is not an isolation boundary |
| PII disclosure | Presidio and regex union, typed redaction on HTTP input/output, safe tracing capture policy | [`safety/middleware.py`](../src/cortex/safety/middleware.py), [`obs/tracing.py`](../src/cortex/obs/tracing.py) | Statistical and regex detection have false negatives; provider handling remains external |
| Harmful model output | Local output moderation, optional classifier interface, fail-closed classifier errors | [`safety/moderation.py`](../src/cortex/safety/moderation.py), [`test_moderation.py`](../tests/test_safety/test_moderation.py) | No external classifier configured by default; local patterns are incomplete |
| Arbitrary code execution | Tool disabled by default, API allowlist excludes it, timeout and returned-output truncation, tests for default refusal | [`mcp/server.py`](../src/cortex/mcp/server.py), [`test_mcp.py`](../tests/test_mcp/test_mcp.py) | When enabled, subprocess has service privileges and buffers output before truncating it, so output can exhaust memory |
| SQL modification or stacked queries | Read-only connection and single-SELECT validation | [`mcp/server.py`](../src/cortex/mcp/server.py), [`test_query_data.py`](../tests/test_mcp/test_query_data.py) | Database permissions and dialect behavior must enforce defense in depth |
| Resource exhaustion | Per-process token bucket, bounded bucket/run maps, task/iteration/token/cost limits, BM25 cap | [`api/ratelimit.py`](../src/cortex/api/ratelimit.py), [`config.py`](../src/cortex/config.py) | Limits multiply across replicas; external provider/store exhaustion remains |
| Cost abuse | Budget checked before provider calls, Redis ledger, spend metrics and alert | [`llm/router.py`](../src/cortex/llm/router.py), [`llm/cost_tracker.py`](../src/cortex/llm/cost_tracker.py) | Concurrent calls can cross a threshold; provider pricing metadata can be incomplete |
| Metrics information disclosure | Optional bearer token; shipped ingress-nginx snippet intends to deny external `/metrics` | [`api/main.py`](../src/cortex/api/main.py), [`deploy/k8s/service.yaml`](../deploy/k8s/service.yaml) | The manifest does not enforce an ingress class; incompatible/disabled snippets can leave `/metrics` exposed through the catch-all route |
| Dependency or build compromise | Read-only CI permissions, pinned action majors, scoped `pip-audit`, Bandit, Dependabot | [CI](../.github/workflows/ci.yml), [`scripts/audit.py`](../scripts/audit.py) | Dependencies are not hash-locked; published-advisory scans do not find unknown flaws |
| Container privilege escalation | Non-root image; Kubernetes security contexts, dropped capabilities, read-only filesystems | [`Dockerfile`](../Dockerfile), [`deploy/k8s/deployment.yaml`](../deploy/k8s/deployment.yaml) | Manifests are unapplied; host/cluster security is external |

## Data handling

- HTTP goals and final outputs pass through PII redaction, but direct library
  callers and standalone tools may not.
- The trace decorator intentionally excludes common content-bearing parameter
  names; adding captured fields requires a privacy review.
- Semantic cache payloads retain a prompt digest and response, not the prompt
  text, and are scope filtered.
- Episodic and semantic memory contain user-derived summaries/facts and require
  access controls, retention decisions, and deletion procedures in a real
  deployment.
- Prometheus and logs are operational data that may still identify activity
  patterns.

The repository has no end-user deletion API, legal retention policy, backup
encryption configuration, or data residency control.

## Security defaults and deployment requirements

- Keep `CODE_EXECUTION_ENABLED=false` unless code runs in an external sandbox.
- Set a stable, randomly generated `SECRET_KEY` in production and rotate it
  through the deployment's secret-management process.
- Do not expose standalone MCP HTTP/SSE to untrusted networks. Route network
  tool calls through the authenticated FastAPI adapter.
- Protect `/metrics` with either an authenticated Prometheus scrape or network
  controls; configure both ends together.
- Use TLS and an authenticating ingress; neither is provided by the application
  process itself.
- Give Redis, Qdrant, and model credentials least privilege and separate them
  by environment.
- Do not ingest cross-tenant material into one deployment until RAG tenant
  scoping is implemented. Consider separate collections or clusters where
  isolation has a strong contractual requirement.
- Restrict outbound network access if model/store destinations are known.
- Add backup, restore, retention, and deletion procedures before storing
  important data.

## Security validation

Repository gates are documented in [Contributing](../CONTRIBUTING.md). Relevant
test areas include [`tests/test_safety`](../tests/test_safety),
[`tests/test_mcp`](../tests/test_mcp),
[`tests/test_api/test_ratelimit.py`](../tests/test_api/test_ratelimit.py), and
[`tests/test_scripts/test_audit.py`](../tests/test_scripts/test_audit.py).

For a security-sensitive change, demonstrate both the blocked attack and a
nearby benign case, then review whether telemetry can detect a bypass or
failure. Static analysis and dependency scans complement rather than replace
those behavior tests.

## Incident priorities

1. Disable or isolate the affected entry point.
2. Rotate potentially exposed signing/provider/store credentials.
3. Preserve sanitized logs, metrics, traces, image digest, and configuration.
4. Determine affected tenants/users without widening data access during
   investigation.
5. Follow the operational process in
   [Operations](OPERATIONS.md#incident-response).
6. Report repository vulnerabilities privately under the
   [Security policy](../SECURITY.md).
