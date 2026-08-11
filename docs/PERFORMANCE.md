# Performance

Regenerate with `python3 -m perf.benchmark --write`. Enforced in CI by
`make perf`, which exits non-zero on a breached budget.

## Measured

Generated at `2026-08-11T23:49:21Z` on `Darwin 25.5.0 (arm64)` with Python `3.13.2`.
Measured source: `cc1a83259103 (dirty working tree)`. A dirty result includes uncommitted code and is
not attributable to the named commit alone.

5 independent rounds, each 6 concurrent clients running
25 iterations, ASGI in-process. Latencies in milliseconds.
`p50`, `p95` and `p99` are the **median across rounds**; `samples` and `max`
are pooled over all of them.

| path | samples | p50 | p95 | p99 | max | budget p95/p99 |
|---|---|---|---|---|---|---|
| `health` | 750 | 1.94 | 2.37 | 2.46 | 2.73 | 15 / 40 |
| `metrics` | 750 | 6.29 | 7.05 | 7.41 | 8.02 | 40 / 90 |
| `unauthorised` | 750 | 4.17 | 4.64 | 4.85 | 5.10 | 20 / 50 |
| `invalid_body` | 750 | 4.57 | 5.20 | 5.39 | 5.84 | 25 / 60 |
| `rate_limited` | 750 | 1.10 | 1.51 | 1.63 | 1.81 | 20 / 50 |

Per-round p99, so the agreement between rounds is visible rather than
smoothed away by the median:

| path | p99 per round |
|---|---|
| `health` | 2.59, 2.46, 2.31, 2.72, 2.43 |
| `metrics` | 8.00, 7.41, 7.07, 7.44, 7.38 |
| `unauthorised` | 4.99, 5.09, 4.56, 4.70, 4.85 |
| `invalid_body` | 5.39, 5.72, 5.33, 5.51, 5.25 |
| `rate_limited` | 1.68, 1.67, 1.63, 1.57, 1.56 |

## How it is measured, and why the harness is this careful

The first version of this gate was red on every run, including on `main`,
always on `unauthorised p99 ~230ms > 50ms` while p95 sat near 8ms. None of
it was the code. Each of the following is a fix for a specific way the
harness was measuring the runtime instead of the request.

**The import-time heap is frozen before measuring.** Cortex's dependency
closure leaves roughly 600k objects resident. A generation-2 collection
walks all of them — 120ms on a developer machine, 230-285ms on a CI runner
— and because the event loop is single-threaded, that pause is charged in
full to whatever requests are in flight. Cyclic GC is triggered by
allocation counts, and this workload is deterministic, so it fired at the
same iteration in every CI run and hit the same path every time.
`gc.freeze()` after warm-up moves that heap into the permanent generation.
Collection stays **enabled** — objects allocated during measurement are
still collected — so a change that starts producing garbage per request
still shows up.

**Every measured path is warmed, not just `/health`.** Warm-up and
measurement iterate the same table of calls, so the two cannot drift apart
again.

**Log records go to memory for the duration.** The rate-limited path emits
a warning per request; on CI stdout is a pipe owned by the runner agent, so
each of those writes is a syscall that can block on another process inside
the timed window. The records are still emitted, formatted and counted —
the count is printed after every run — so logging added to a hot path is
still paid for in latency and still visible.

**The app under test is un-throttled explicitly, not by import order.**
`RateLimitMiddleware` builds its limiter when the app is imported, so
setting `API_RATE_LIMIT_PER_MINUTE` first only works when this module wins
the import race. Where it does not, warm-up drains the bucket and every
metered path answers 429 — the rate limiter measured under five other
names. The harness installs its own limiter for the duration and puts the
app's back afterwards.

**The gate reads the median of several rounds.** Nearest-rank p99 over 150
samples is the second-worst observation, so two contaminated samples decide
the build. Collecting more samples in a single run does not help: stalls
arrive at a rate, so a longer run collects proportionally more of them and
contamination stays above the 99th percentile. Independent rounds do help —
a stall must hit a majority of them to move the median, while a real
regression slows every round. Budgets are unchanged and p99 is still
enforced.

The deliberate limit: a regression appearing in a minority of rounds is
treated as noise. `max` is reported per path so a genuine rare stall is
non-gating rather than invisible.

## Scope, and why it stops where it does

These are **edge** paths: health, metrics, auth rejection, validation
rejection, and the rate-limiter's 429 branch. Agent runs are deliberately
absent.

A Cortex run calls an LLM, so its latency is somebody else's p99 plus
network. Gating CI on that produces a number either too loose to catch
anything or too flaky to keep — and in both cases it measures a provider
rather than this codebase.

What a commit here *can* regress is the edge: a middleware that starts doing
IO, an auth path that stops being constant-time, a validator compiling a
regex per request, a limiter holding its lock across an await. Those are on
every request, including the ones that fail, and they are stable enough to
budget.

End-to-end run latency belongs in `perf/locustfile.py`, run against a
deployed instance.

## Reading it

p50 is the typical request. **p95 and p99 are the worst-served users**, and
they are the ones who open tickets. No mean is reported: a mean hides the
tail that matters.
