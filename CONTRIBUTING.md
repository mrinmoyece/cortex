# Contributing to Cortex

## Development setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Python support and all dependency groups are defined in
[`pyproject.toml`](pyproject.toml). The default tests are hermetic: use
`fakeredis` or injected doubles for stores, patch provider clients, and do not
require developer credentials or running services.

## Required gates

Run the same aggregate gate exposed by the repository:

```bash
make gate
```

Its commands are defined in the [`Makefile`](Makefile):

| Gate | Command | Purpose |
|---|---|---|
| Documentation | `make docs` | Relative link and anchor integrity |
| Lint/format | `make lint` | Ruff checks for source, tests, performance code, and scripts |
| Types | `make types` | Strict mypy checks for `src` |
| Tests | `make test` | Pytest with the 80% coverage floor from `pyproject.toml` |
| Static security | `make security` | Bandit over `src` |
| Dependency audit | `make audit-fresh` | Project-scoped dependency closure resolved from the index |
| Edge performance | `make perf` | Deterministic ASGI edge budgets |

CI runs Python 3.10 and 3.13 for quality checks, plus independent security and
performance jobs; the workflow is
[`.github/workflows/ci.yml`](.github/workflows/ci.yml).

`make audit` is useful for inspecting the versions already installed in a
development environment. The aggregate gate uses `make audit-fresh` so stale
unrelated state in a long-lived environment does not determine whether the
repository's current dependency constraints pass.

Use targeted commands while iterating:

```bash
pytest tests/test_api tests/test_graph --no-cov
pytest tests/test_agents/test_agents.py -k planner --no-cov
ruff check src tests perf scripts
mypy src
python3 scripts/check_docs.py
```

## Test strategy

- Unit tests isolate agents, routing, safety, retrieval, and storage behavior.
- API tests exercise FastAPI with real JWTs and mocked external work.
- Graph tests execute state transitions with a mocked LLM router.
- Deployment tests parse Compose/Kubernetes configuration and assert manifest
  consistency.
- Evaluation harness tests validate scoring and fallback behavior; the JSON
  cases under [`tests/eval`](tests/eval) are evaluation inputs, not live-model
  pytest cases.
- Performance gate tests validate the benchmark itself before CI trusts its
  result.

The test tree is the executable index:
[`tests/test_api`](tests/test_api),
[`tests/test_agents`](tests/test_agents),
[`tests/test_graph`](tests/test_graph),
[`tests/test_mcp`](tests/test_mcp),
[`tests/test_rag`](tests/test_rag),
[`tests/test_memory`](tests/test_memory),
[`tests/test_safety`](tests/test_safety),
[`tests/test_eval`](tests/test_eval),
[`tests/test_obs`](tests/test_obs),
[`tests/test_perf`](tests/test_perf), and
[`tests/test_deploy`](tests/test_deploy).

## Change requirements

- Add a test that fails without the behavior being introduced or fixed.
- Update the canonical document from the
  [documentation map](README.md#documentation-map-and-ownership) when behavior,
  configuration, operations, or limitations change.
- Add or supersede an ADR when changing a major architectural decision.
- Keep external calls mocked in the default suite.
- Do not add a setting, dependency, dashboard, alert, or security control that
  no runtime path reads.
- Do not report generated or aspirational evaluation numbers as measured
  evidence.

Security-sensitive changes to authentication, principal binding, tenant
filters, code execution, rate limiting, safety checks, or budget enforcement
need attack-oriented regression tests. Review the
[threat model](docs/THREAT_MODEL.md) before making those changes.

## Commit and pull request conventions

Use an imperative subject under roughly 70 characters. Explain the defect or
decision in the body when the subject is not enough. Complete the repository's
[pull request template](.github/PULL_REQUEST_TEMPLATE.md), listing only gates
actually run and any unverified operational behavior.
