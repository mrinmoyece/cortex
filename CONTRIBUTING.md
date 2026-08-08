# Contributing to Cortex

The README linked to this file for some time before it existed. That is the
kind of small dishonesty this document is about.

## Getting a working checkout

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                      # 233 tests, no services required
```

The suite is **hermetic**. It needs no Redis, no Qdrant, no API key, and no
environment variables. If a change makes a test require infrastructure, the
change is wrong, not the test — see [Hermetic tests](#hermetic-tests) below.

## The bar for a change

**Every behavioural claim needs a test that would fail without it.** Not a
test that passes either way. Before opening a PR, break your own fix and
confirm the test goes red; if it stays green, the test is decoration.

**The coverage gate is 80% and it is enforced** (`--cov-fail-under=80` in
`pyproject.toml`). Do not lower it to make a branch pass. Coverage is a
floor, not a target — 100% coverage of code that asserts nothing is worth
less than 60% of code that asserts precisely.

**Comments explain *why*, never *what*.** `# increment the counter` above
`counter += 1` is noise. `# Checked before the call, not after, because the
spend that breaches the limit must never happen` is the reason someone will
need in six months.

## Hermetic tests

A test that needs a live service is a test that does not run — and a suite
that does not run is how this project ended up with 77 files, an 80%
coverage gate and zero executions.

- External stores: use `fakeredis`, or inject a double.
- LLM calls: patch the **router singleton** (`cortex.llm.router._router`),
  not the `get_router` name. Agents bind `get_router` at import time, so
  patching the source module rebinds a name nobody reads — a mistake that
  silently sent four tests at a real Redis.
- Anything with a clock: inject the time, or assert a bound rather than a
  value.

## Security-sensitive areas

Changes to these need a test demonstrating the attack is still blocked:

| Area | The property |
|---|---|
| `mcp/server.py` `execute_code` | The block list is an allowlist of safe constructs, not a denylist of scary words |
| `mcp/server.py` `query_data` | A single bare `SELECT`, executed on a read-only connection |
| `safety/middleware.py` | Injection patterns and PII redaction |
| `api/ratelimit.py` | Limiting happens before routing, so 404s and 422s are metered too |
| `llm/router.py` | The budget gate runs *before* the call it authorises |

## Commit messages

Subject line in the imperative, under ~70 characters. If the change fixes a
defect, the body should say what the defect was and how it was found — those
messages are the most useful documentation in the repository.

```
fix: bind tools to the model before invoking it

`bind_tools()` was defined and never called, so no model was ever told
which tools existed. Invisible under a scripted model whose tool calls
are authored into the fixture; fatal against a real provider.
```

## What gets rejected

- A test that would pass against the unfixed code.
- A dependency added to `pyproject.toml` and not imported. Unused
  dependencies inflate the image and the CVE surface, and mislead a reader
  about the architecture — several were removed for exactly this reason.
- A setting that nothing reads. `api_rate_limit_per_minute` was configured,
  documented and unenforced for the project's whole life; a control an
  operator believes they have is worse than one they know they lack.
- A claim in a doc that the code does not support.
