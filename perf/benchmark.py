"""Latency benchmark and performance gate for the Cortex API edge.

    python -m perf.benchmark              # measure, print, enforce budgets
    python -m perf.benchmark --write      # ...and regenerate docs/PERFORMANCE.md

Exit code 1 on a breached budget, so CI gates on it like a test.

## What is measured, and the reason for the scope

Only the **request edge**: health, metrics, authentication, validation and
rate limiting. Not agent runs.

That is deliberate, and the reasoning is worth stating because the opposite
choice is the tempting one. A Cortex run calls an LLM. Its latency is
therefore somebody else's p99, plus network, and it varies by an order of
magnitude between runs. Putting that in a CI gate produces a number that is
either so loose it catches nothing or so flaky it gets disabled within a
month — and either way it tells you nothing about *your* code.

What a code change can actually regress is the edge: a middleware that
starts doing IO, an auth path that stops being constant-time, a validator
that compiles a regex per request, a rate limiter that holds its lock too
long. Those are cheap to measure, stable enough to gate, and they are on
every single request including the ones that fail.

For end-to-end run latency, `perf/locustfile.py` against a deployed
instance is the right instrument. It is not this one.

## Reading the numbers

p50 is the typical request. **p95 and p99 are the worst-served users** —
the ones who open tickets. No mean is reported, because a mean hides the
tail that matters.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path

import httpx

# Set BEFORE importing the app: `RateLimitMiddleware` builds its limiter at
# import time, so a later change has no effect. Without this the benchmark
# throttles itself and the "unauthorised" and "invalid body" figures become
# a measurement of the rate limiter rather than of the paths named.
os.environ.setdefault("API_RATE_LIMIT_PER_MINUTE", "1000000")

from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route

from cortex.api.auth import create_access_token
from cortex.api.main import app
from cortex.api.ratelimit import RateLimiter, RateLimitMiddleware


def _rejecting_app() -> Starlette:
    """A minimal app whose limiter has no allowance left.

    The 429 branch used to be timed by calling `RateLimiter.check()` in
    process and recording the result under a name that implied an HTTP
    path. That measured a dictionary lookup - it reported 0.00ms against a
    20ms budget, so the budget could never fail and the row was decoration.
    Rejecting through the middleware, over ASGI, measures what a throttled
    client actually waits for: principal derivation, the bucket, and the
    JSON response.
    """

    async def unreachable(request: object) -> PlainTextResponse:  # pragma: no cover
        return PlainTextResponse("should never be reached")

    limited = Starlette(routes=[Route("/api/v1/runs", unreachable, methods=["POST"])])
    limited.add_middleware(RateLimitMiddleware, limiter=RateLimiter(per_minute=1, burst=0))
    return limited


DOC = Path(__file__).resolve().parents[1] / "docs" / "PERFORMANCE.md"

#: Budgets in milliseconds, derived from measured runs with ~3x headroom on
#: the worst observation. Wider headroom than a normal service budget for a
#: deliberate reason: these paths are sub-millisecond, so absolute variance
#: on a shared CI runner is a large *relative* fraction. A 3x band still
#: catches the regressions that matter here — an accidental disk read or a
#: per-request regex compile costs far more than 3x.
BUDGETS: dict[str, dict[str, float]] = {
    "health": {"p95": 15, "p99": 40},
    "metrics": {"p95": 40, "p99": 90},
    "unauthorised": {"p95": 20, "p99": 50},
    "invalid_body": {"p95": 25, "p99": 60},
    "rate_limited": {"p95": 20, "p99": 50},
}


@dataclass
class Measurement:
    name: str
    samples: list[float] = field(default_factory=list)
    errors: int = 0

    def percentile(self, q: float) -> float:
        if not self.samples:
            return 0.0
        ordered = sorted(self.samples)
        # Nearest-rank: with a few hundred samples, interpolating invents a
        # latency nobody experienced.
        return ordered[min(len(ordered) - 1, int(q * len(ordered)))]

    def summary(self) -> dict[str, float]:
        return {
            "n": len(self.samples),
            "errors": self.errors,
            "p50": round(self.percentile(0.50), 3),
            "p95": round(self.percentile(0.95), 3),
            "p99": round(self.percentile(0.99), 3),
            "max": round(max(self.samples), 3) if self.samples else 0.0,
        }


async def _timed(
    m: Measurement, call: Callable[[], Awaitable[httpx.Response]], ok: tuple[int, ...]
) -> None:
    start = time.perf_counter()
    try:
        response = await call()
        elapsed = (time.perf_counter() - start) * 1000
        if response.status_code in ok:
            m.samples.append(elapsed)
        else:
            m.errors += 1
    except Exception:
        m.errors += 1


async def run(concurrency: int = 8, iterations: int = 40) -> dict[str, Measurement]:
    token = create_access_token(user_id="bench", tenant="bench", scopes=["runs:read"])
    auth = {"authorization": f"Bearer {token}"}
    names = ("health", "metrics", "unauthorised", "invalid_body", "rate_limited")
    results = {n: Measurement(n) for n in names}

    # The 429 branch is on every request once a client misbehaves, so it
    # deserves a budget as much as the happy path does.
    rejecting = _rejecting_app()

    async with (
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://bench", timeout=30.0
        ) as client,
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=rejecting),
            base_url="http://bench",
            timeout=30.0,
        ) as throttled,
    ):
        await client.get("/health")  # warm-up: lazy imports, not latency
        await throttled.post("/api/v1/runs", json={"goal": "x"})

        async def one_user() -> None:
            for _ in range(iterations):
                await _timed(results["health"], lambda: client.get("/health"), (200,))
                await _timed(results["metrics"], lambda: client.get("/metrics"), (200,))
                await _timed(
                    results["unauthorised"],
                    lambda: client.post("/api/v1/runs", json={"goal": "x"}),
                    (401, 403),
                )
                await _timed(
                    results["invalid_body"],
                    lambda: client.post("/api/v1/runs", json={}, headers=auth),
                    (422,),
                )
                await _timed(
                    results["rate_limited"],
                    lambda: throttled.post("/api/v1/runs", json={"goal": "x"}),
                    (429,),
                )

        await asyncio.gather(*(one_user() for _ in range(concurrency)))

    return results


def check(results: dict[str, Measurement]) -> list[str]:
    breaches = []
    for name, budget in BUDGETS.items():
        m = results.get(name)
        if not m or not m.samples:
            breaches.append(f"{name}: no successful samples")
            continue
        for label, q in (("p95", 0.95), ("p99", 0.99)):
            observed = m.percentile(q)
            if observed > budget[label]:
                breaches.append(f"{name} {label} {observed:.1f}ms > {budget[label]:.0f}ms")
    return breaches


def render(results: dict[str, Measurement], concurrency: int, iterations: int) -> str:
    rows = "\n".join(
        f"| `{name}` | {s['n']} | {s['p50']:.2f} | {s['p95']:.2f} | {s['p99']:.2f} | "
        f"{s['max']:.2f} | {BUDGETS[name]['p95']:.0f} / {BUDGETS[name]['p99']:.0f} |"
        for name, m in results.items()
        for s in [m.summary()]
    )
    return f"""# Performance

Regenerate with `python -m perf.benchmark --write`. Enforced in CI by
`make perf`, which exits non-zero on a breached budget.

## Measured

{concurrency} concurrent clients x {iterations} iterations, ASGI in-process.
Latencies in milliseconds.

| path | samples | p50 | p95 | p99 | max | budget p95/p99 |
|---|---|---|---|---|---|---|
{rows}

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
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Cortex edge latency benchmark and gate")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    results = asyncio.run(run(args.concurrency, args.iterations))

    if args.json:
        print(json.dumps({k: v.summary() for k, v in results.items()}, indent=2))
    else:
        print(f"\n{'path':<15}{'n':>6}{'p50':>9}{'p95':>9}{'p99':>9}{'max':>9}   budget p95")
        for name, m in results.items():
            s = m.summary()
            print(
                f"{name:<15}{s['n']:>6}{s['p50']:>9.2f}{s['p95']:>9.2f}"
                f"{s['p99']:>9.2f}{s['max']:>9.2f}{BUDGETS[name]['p95']:>13.0f}"
            )

    if args.write:
        DOC.parent.mkdir(parents=True, exist_ok=True)
        DOC.write_text(render(results, args.concurrency, args.iterations))
        print(f"\nwritten to {DOC}")

    breaches = check(results)
    if breaches:
        print("\nPERFORMANCE BUDGET BREACHED")
        for b in breaches:
            print(f"  - {b}")
        return 1
    print("\nAll performance budgets met.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
