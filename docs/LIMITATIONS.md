# Limitations

Cortex has never served production traffic. It runs, it is tested, and the
gates in CI are real — but "works and is gated" is not the same claim as
"operated at scale", and this file exists so the difference is written down
somewhere other than in a footnote.

Everything below is a property of the code as it stands, not a to-do list
disguised as one. Where a limitation has a setting attached, the setting is
named.

---

## State that does not survive a process

**Run state is per-process and in-memory.** `POST /api/v1/runs` returns a
`run_id` that only the replica which accepted it can resolve. Behind two
replicas, roughly half of all `GET /api/v1/runs/{id}` polls hit a process
that has never heard of the run and get a 404. A restart loses every
in-flight run.

The store is bounded — it was an unbounded dict, which is a memory leak with
a public trigger — and evicts by TTL and then LRU:

| Setting | Default | Effect |
|---|---|---|
| `API_MAX_TRACKED_RUNS` | 1000 | Hard cap on retained runs |
| `API_RUN_RETENTION_SECONDS` | 3600 | Age after which a run is dropped |

A run can therefore be evicted before the client polls for it. The fix is a
shared store (Redis or Postgres) behind the same `_RunStore` interface; that
work has not been done and is not claimed.

**Rate limiting is per-process.** `RateLimiter` holds its buckets in memory,
so behind N replicas the effective limit is N × `API_RATE_LIMIT_PER_MINUTE`.
The bucket map is bounded by `API_RATE_LIMIT_MAX_BUCKETS` (default 10,000)
because its key is caller-controlled; eviction is LRU and deliberately fails
open for the evicted key, since refusing new principals would let an
attacker deny service to everyone else.

**The LangGraph checkpointer is `MemorySaver`.** Checkpoints are per-process
and lost on restart, so "runs can resume" is true within a process and false
across one. A durable checkpointer (Redis/Postgres) is a drop-in change that
has not been made.

**Working memory is per-process.** The episodic (Redis) and semantic
(Qdrant) tiers are shared; the working tier is not.

---

## Human-in-the-loop is opt-in and has no resume path

The graph can suspend before the critic, but this is **off by default**
(`HUMAN_REVIEW_BEFORE_CRITIC=false`). It used to be unconditional, which
meant every run stopped half-finished and was reported as completed.

With it enabled, `run_cortex` detects the suspension and returns
`RunStatus.AWAITING_HUMAN` rather than lying about completion — but there is
**no endpoint to approve and resume**. Enabling the flag today gives you a
run that stops honestly and stays stopped. Do not enable it expecting an
approval workflow.

---

## `execute_code` is not a sandbox

The MCP `execute_code` tool runs a Python subprocess in the same container as
the caller, with a timeout and an output cap and nothing else. There is no
seccomp profile, no namespace isolation, no filesystem restriction and no
network restriction. It can read the environment — including API keys — and
reach anything the container can reach.

It is **disabled by default** (`CODE_EXECUTION_ENABLED=false`), its docstring
says exactly this, and it is not advertised to the planner or in the HTTP
tool allowlist when disabled. Enabling it in a multi-tenant deployment
without an external sandbox (gVisor, Firecracker, a separate execution
service) is a decision to run untrusted code as your service account.

---

## Safety layers are heuristics, not compliance controls

**PII detection** runs a regex pass and, when installed, Presidio, and takes
the union of both. Neither is exhaustive. Presidio is restricted to a
12-entity allowlist with a score threshold (`PII_ENTITIES`,
`PII_SCORE_THRESHOLD`) because its default entity set redacted ordinary
words out of user goals — "annual" was being classified as a date and
destroyed the request. Narrowing the entity set trades recall for a system
that does not corrupt its own input. Treat the output as best-effort
redaction, not as a GDPR or HIPAA control.

**Prompt-injection detection** is pattern matching plus spotlighting of
untrusted tool output. It will catch the documented phrasings and will not
catch a novel one. `Moderator` accepts a pluggable classifier
(`safety/moderation.py`) and none is wired in by default, so what runs is
the local layers only. The moderation path **fails closed**: if a configured
classifier errors, the request is refused rather than passed through.

**NeMo Guardrails** is imported lazily and its Colang policies live in
`config/rails/`. If the import or the config fails, the local layers still
run; the rails do not.

---

## Evaluation

`import ragas` currently fails against the installed langchain-community
(`langchain_community.chat_models.vertexai` was removed), so the Ragas path
is dead on a default install and `RagasEvaluator` falls back to
LLM-as-judge. The fallback is real — it scores faithfulness and relevancy
with a model — but it is a weaker instrument and it produces no context
precision or recall.

Both `ragas` and `deepeval` moved to an optional `eval` extra
(`pip install -e ".[eval]"`). They are not needed to serve a request, they
are heavy, and ragas pulls in transitive packages with open advisories that
would otherwise sit in the API image for no runtime benefit.

The scores in `docs/EVALUATION.md` illustrate the harness. They are not a
published benchmark result.

---

## Multi-tenancy

Tenant isolation is enforced by **payload filters at query time**, not by
separate collections, databases or credentials. Every memory and RAG query
filters on `tenant_id`, and the semantic LLM cache is scoped by tenant (an
unscoped call bypasses the cache entirely rather than sharing it) — but a
bug in a filter is a cross-tenant read, and there is no second layer behind
it. A deployment with a genuine isolation requirement should use a
collection or cluster per tenant.

`Principal` binding on the MCP server means tool calls take their identity
from the authenticated caller rather than from a model-supplied argument,
and `current_principal()` fails closed when no principal is bound. That
closes the obvious IDOR, not the general class.

**The standalone MCP server is not authenticated, and this repository does
not pretend to fix that.** FastMCP serves an HTTP/SSE transport to every
caller on the port identically, and a stdio server is whatever process the
client spawned. There is no per-request identity to derive, so:

* Over **stdio**, an operator may declare the single identity the process
  acts as with `MCP_PRINCIPAL_USER_ID`. That is honest for the case it
  covers — one process, one desktop client, one person — and it is a
  configuration statement, not a login. Anyone who can run the process can
  set it.
* Over **http** and **sse**, the declaration is refused and `query_memory`
  is unavailable. There is no safe way to serve a per-user tool from an
  unauthenticated port, and a token check bolted onto FastMCP here would be
  a second, weaker authentication system beside the API's.

Reaching memory tools over a network goes through `POST /api/v1/mcp/call`,
which authenticates first. Putting the MCP HTTP port behind an
authenticating proxy is possible but out of scope here: nothing in this
repository maps a proxy-asserted identity onto a `Principal`, and inventing
that mapping without a deployment to test it against is how the original
`user_id`-as-an-argument bug happened.

---

## Operational gaps

* **`/metrics` is unauthenticated unless `METRICS_TOKEN` is set.** Cortex
  metrics carry model names, spend totals and run volumes. The shipped
  Kubernetes manifests leave the token unset on purpose, because they scrape
  by pod annotation and annotation scraping sends no credential — a token
  there breaks the scrape rather than securing it. The ingress refuses
  `/metrics` from outside the cluster; anything inside it can still scrape.
  Requiring a token means switching to a scrape config with a
  `credentials_file` and mounting the Secret into Prometheus, which is
  documented in `docs/DEPLOYMENT.md` and not shipped.
* **No backups, no persistence guarantees.** `docker-compose.yml` runs
  single-node Redis and Qdrant with named volumes. That is a development
  stack.
* **No database.** There is no relational store and no migrations; anything
  described as a "ledger" is a Redis key with a TTL, so cost history is not
  durable accounting.
* **The Kubernetes manifests are unapplied.** They set non-root users,
  read-only root filesystems, dropped capabilities, resource limits, probes
  and a PDB, and no cluster has run them. Treat them as a starting point.
* **BM25 sparse retrieval is in-process and bounded** by
  `RAG_MAX_INDEXED_CHUNKS` (default 50,000). Beyond that the corpus stops
  growing rather than growing without limit. It is also per-process, so
  replicas disagree about the sparse half of a hybrid search until they have
  all seen the same ingests.

---

## Performance

`docs/PERFORMANCE.md` reports the **request edge only**: health, metrics,
auth rejection, validation rejection and the rate limiter's 429 branch,
measured in-process over ASGI on one machine. Those numbers say nothing
about throughput, about agent-run latency (which is dominated by the model
provider), or about behaviour under sustained load. `perf/locustfile.py`
exists for load testing against a deployed instance and has not been run
against one.

---

## Dependency surface

The dependency closure is audited by `scripts/audit.py`, scoped to Cortex's
own tree rather than to whatever else is installed on the machine. As of the
last run, a clean install of the runtime and `[dev]` closures reports no
known vulnerabilities. That is a statement about published advisories, not
about the code in those 234 packages.
