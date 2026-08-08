# Performance

Regenerate with `python -m perf.benchmark --write`. Enforced in CI by
`make perf`, which exits non-zero on a breached budget.

## Measured

6 concurrent clients x 25 iterations, ASGI in-process.
Latencies in milliseconds.

| path | samples | p50 | p95 | p99 | max | budget p95/p99 |
|---|---|---|---|---|---|---|
| `health` | 150 | 2.16 | 2.29 | 2.44 | 2.46 | 15 / 40 |
| `metrics` | 150 | 7.22 | 7.93 | 8.00 | 8.01 | 40 / 90 |
| `unauthorised` | 150 | 4.00 | 4.44 | 4.69 | 4.71 | 20 / 50 |
| `invalid_body` | 150 | 4.49 | 4.93 | 5.25 | 5.26 | 25 / 60 |
| `rate_limited` | 150 | 0.00 | 0.00 | 0.01 | 0.01 | 20 / 50 |

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
