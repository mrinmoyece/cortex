# Limitations and Roadmap

Cortex is a tested reference implementation, not an operated production
service. This document is the canonical list of material gaps and planned
directions. Roadmap items are intentions without delivery dates.

## Current limitations

### Process-local control and run state

Run polling, rate-limit buckets, LangGraph checkpoints, and the BM25 sparse
index are process-local.

- A run accepted by one replica may be a 404 on another.
- Restarting a process loses pending runs and checkpoints.
- Effective rate limits multiply by replica count.
- Sparse search results can differ until each replica has indexed the same
  documents.

The run and bucket maps are bounded by settings in
[`config.py`](../src/cortex/config.py), preventing unbounded growth but allowing
TTL/LRU eviction. A shared run store, rate limiter, durable checkpointer, and
shared sparse index are not implemented.

### Human review cannot resume

`HUMAN_REVIEW_BEFORE_CRITIC=true` suspends before critique and reports
`awaiting_human`, but no HTTP or worker flow approves, updates, or resumes the
checkpoint. Leave it disabled unless an external caller uses LangGraph APIs
inside the same process.

### Budget and iteration exits can remain nonterminal

The graph-level guard ends execution when state cost or iteration count reaches
its ceiling, but does not first change an `executing` state to `failed`.
Polling/SSE can therefore observe a nonterminal status after graph execution.
The LLM router separately checks the Redis ledger before provider calls and
remains the effective spend ceiling. The graph exit needs an explicit failure
node and regression test.

### Standalone MCP network transports are unauthenticated

The MCP HTTP/SSE server does not authenticate callers or derive per-request
identity. `query_memory` therefore refuses without a safely bound principal.
Use stdio for one local process/user or the JWT-authenticated
`POST /api/v1/mcp/call` adapter for network calls.

### Code execution is not isolated

`execute_code` starts Python with the service account's filesystem, network,
environment, and container privileges. A timeout and post-collection response
truncation constrain some accidents, but `communicate()` buffers subprocess
output before slicing it and therefore does not cap memory use. Blocked
patterns are not an attacker boundary. The tool is disabled by default and
requires an external sandbox before multi-tenant use.

### RAG is not tenant-isolated

Memory and semantic cache isolation depend on keys and Qdrant payload filters
in shared infrastructure. RAG ingestion stores the authenticated user as
`ingested_by`, but not the tenant, and knowledge search does not derive a
tenant filter from the principal. The shipped knowledge corpus is shared
across authenticated tenants. Use separate deployments/corpora until
principal-derived RAG scoping is implemented and tested.

### Working-memory budgeting is not integrated

`WorkingMemory` implements a token-budgeted in-process store and has unit tests,
but no runtime graph path instantiates it. Planner and executor prompts do not
receive this control, so `MEMORY_WORKING_TOKEN_BUDGET` is not an active
context-window limit.

### Safety is heuristic and entry-point dependent

HTTP-created runs receive input/output safety checks. Direct calls to
`run_cortex()` are library calls and bypass that wrapper. Regex/scored
injection checks, PII recognition, spotlighting, and local moderation have
false positives and false negatives. No external moderation classifier is
configured by default.

The Colang files under `config/rails` are not mounted by shipped deployments,
so they are not active policy evidence. None of these controls is a GDPR,
HIPAA, or other compliance certification.

### Retrieval and memory do not have production lifecycle controls

- BM25 is capped at `RAG_MAX_INDEXED_CHUNKS` and rebuilt in process.
- Semantic facts use heuristic extraction and have no repository-defined
  expiry.
- There is no user deletion API, retention workflow, collection migration, or
  reconciliation between replicas.
- Redis cost and episodic records have TTLs and are not durable accounting.

### Evaluation has no validated baseline

The evaluator implementations and fallbacks are tested, but no versioned
live-model report is checked in. Optional Ragas/DeepEval compatibility depends
on their external release set. Target thresholds, scheduled tasks, metrics,
and dashboards are not evidence that a model/corpus combination meets them.
See [Evaluation](EVALUATION.md).

### Production operations are unproven

- Kubernetes manifests are statically tested but have not been applied by this
  project.
- The Locust profile has not been run against a production deployment.
- No production SLO, error budget, capacity model, or availability baseline
  exists.
- No Alertmanager receiver, on-call schedule, backup/restore automation,
  disaster-recovery test, or data migration process is shipped.
- `/metrics` is public unless bearer authentication or network controls are
  configured; the shipped Kubernetes scrape relies on in-cluster access.

### Configuration artifacts are partially illustrative

[`config/models.yaml`](../config/models.yaml) describes routing and pricing but
is not loaded or mounted. Runtime model settings remain authoritative. The
same distinction applies to unmounted Colang files.

## Roadmap

### Reliability and scale

1. Introduce a shared run store and durable LangGraph checkpointer with
   ownership-safe polling and resume semantics.
2. Move rate limiting to an atomic shared backend.
3. Mark budget/iteration exits with an explicit terminal status.
4. Replace or synchronize the in-process sparse index and add ingestion
   reconciliation.
5. Exercise the Kubernetes topology under failure, rollout, and restore tests.

### Security and governance

1. Add a real approval/resume API with authorization and audit history.
2. Put code execution behind an isolated service or remove the tool from
   deployable profiles.
3. Add deployment-tested MCP authentication only if it maps each request to a
   verified principal.
4. Add principal-derived tenant isolation to RAG ingestion and retrieval.
5. Add retention/deletion workflows and evaluate stronger tenant storage
   isolation.
6. Wire and test policy configuration rather than shipping inactive examples.

### Evaluation and operations

1. Produce a versioned evaluation report with model, corpus, configuration,
   dataset hash, and per-sample evidence.
2. Run and record representative load and failure tests.
3. Establish traffic-derived SLOs and error budgets, then tune actionable
   alerts and paging.
4. Define backup, restore, disaster-recovery, and incident ownership for a
   concrete deployment.

### Product behavior

1. Replace heuristic semantic-memory extraction with evaluated structured fact
   extraction.
2. Integrate and evaluate working-memory token budgeting in prompt construction.
3. Add durable progress events instead of SSE polling over process memory.
4. Version stored payload schemas and define migrations before compatibility
   depends on them.

Roadmap changes should cite an issue or pull request once scheduled. Until an
item is implemented and tested, documentation must continue to describe the
limitation rather than the intended capability.
