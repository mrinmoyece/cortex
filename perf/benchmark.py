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

## Why the harness controls the process, not just the requests

This gate was red on every run, on `main` as well as on branches, always
on the same line: `unauthorised p99 ~230ms > 50ms`, while p95 stayed near
8ms. The tail was not the code. Three properties of the harness were
measuring the runtime instead:

1. **A full cyclic-GC collection fired inside the measured window.** Cortex's
   import closure leaves ~600k tracked objects resident; a generation-2
   collection walks all of them and costs 120ms locally and 230-285ms on a
   CI runner. It stops the single event loop, so its entire cost is charged
   to whichever requests happen to be in flight. It landed on the same
   iteration in every CI run, because cyclic GC is triggered by allocation
   counts and the workload is deterministic - which is why one path was
   always the casualty. `gc.freeze()` after warm-up moves the import-time
   heap to the permanent generation, so a collection during measurement
   walks only what the benchmark itself allocated. GC stays *enabled*: a
   change that churns objects per request is still visible.

2. **Only `/health` was warmed.** Every other measured path paid its
   first-touch cost inside the window, and with clients running in lockstep
   that contaminates one sample per client per path.

3. **The rate-limited path logged 150 lines to stdout mid-measurement.** On
   CI stdout is a pipe owned by the runner agent, so a log write is a
   syscall that can block on a reader outside this process. During
   measurement those records are still *formatted* - so a hot-path log line
   added by a future commit still costs latency, and the count is reported -
   but they are written to memory rather than to somebody else's pipe.

## Why the gate is the median of several rounds

Nearest-rank p99 over 150 samples is the second-worst observation. Two
contaminated samples therefore *define* the number the build gates on,
which is how a single 230ms pause turned into a permanently red required
check while p95 sat at 8ms.

Collecting more samples in one run does not fix it. Environmental stalls
arrive at a roughly constant *rate*, so a longer run collects proportionally
more of them: at ~2 contaminated samples per 150, contamination is ~1.3% of
the distribution and sits above the 99th percentile no matter how long the
run is.

Repetition does fix it. Each round is an independent estimate of the same
quantity, and the gate takes the **median** of the per-round p99s. A stall
has to hit a majority of rounds to move the median, while a genuine
regression - which by definition slows every round - moves it immediately.
Budgets are unchanged, p99 is still enforced, and each round's p99 is
printed so a widening spread is visible rather than silently smoothed.

The honest limit: a regression that affects fewer than ~1% of requests is
below the resolution of a p99 gate, and one that appears in a minority of
rounds is deliberately treated as noise. `max` is reported per path so that
a genuine one-in-a-thousand stall is not invisible, only non-gating.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import gc
import io
import json
import logging
import os
import statistics
import sys
import time
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

# Set BEFORE importing the app: `RateLimitMiddleware` builds its limiter at
# import time, so a later change has no effect. Without this the benchmark
# throttles itself and the "unauthorised" and "invalid body" figures become
# a measurement of the rate limiter rather than of the paths named.
#
# It is belt and braces with `unthrottled()` below, which does not depend on
# this module being the first to import the app.
os.environ.setdefault("API_RATE_LIMIT_PER_MINUTE", "1000000")

from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route

from cortex.api.auth import create_access_token
from cortex.api.main import app
from cortex.api.ratelimit import RateLimiter, RateLimitMiddleware

#: Requests per minute granted to the app under test. Warm-up alone issues a
#: hundred, and the measured rounds thousands.
BENCH_RATE_LIMIT = 1_000_000.0


@contextlib.contextmanager
def unthrottled(target: Starlette) -> Iterator[None]:
    """Give the app under test an allowance the benchmark cannot exhaust.

    The environment variable above only works when this module is the first
    thing to import `cortex.api.main` - true for `python -m perf.benchmark`,
    false for a test process that imported the app earlier. When it does not
    hold, warm-up drains the bucket and every subsequent request on a
    metered path answers 429, so the benchmark measures the rate limiter
    under five other names.

    Installing the limiter explicitly and rebuilding the middleware stack
    removes the dependency on import order entirely. The `rate_limited` path
    is unaffected: it is measured against its own deliberately-exhausted app.
    """
    swapped = [mw for mw in target.user_middleware if mw.cls is RateLimitMiddleware]
    originals = [dict(mw.kwargs) for mw in swapped]
    for mw in swapped:
        mw.kwargs["limiter"] = RateLimiter(
            per_minute=BENCH_RATE_LIMIT, burst=BENCH_RATE_LIMIT, max_buckets=64
        )
    previous, target.middleware_stack = target.middleware_stack, target.build_middleware_stack()
    try:
        yield
    finally:
        for mw, kwargs in zip(swapped, originals, strict=True):
            mw.kwargs.clear()
            mw.kwargs.update(kwargs)
        target.middleware_stack = previous


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

#: Independent measurement rounds. Odd, so the median is an observation
#: rather than a midpoint between two. Five gives a breakdown point of two
#: fully-corrupted rounds, which is comfortably more than the one stall a
#: run of this length actually sees.
ROUNDS = 5

#: Requests issued per path before anything is timed. Enough to force every
#: lazy import, every pydantic-core schema build, every response-class
#: construction and every metric label child into existence, on every path
#: that carries a budget - not just on `/health`.
WARMUP_REQUESTS = 20

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


@dataclass
class Aggregate:
    """One path's result across independent rounds — what the gate reads.

    Every percentile is the **median of the per-round percentiles**, not a
    percentile of the pooled samples. Pooling would defeat the purpose: a
    process-wide stall contaminates a roughly constant *fraction* of each
    round (about 1.3% of samples here), so it stays above the 99th
    percentile however many rounds are pooled. Taking the median across
    rounds instead requires a stall to hit a majority of them.

    `samples` and `max` remain pooled, because for those two the honest
    answer is the whole population: how many observations were taken, and
    what the worst one was.

    Duck-types `Measurement` deliberately, so `check()` reads one round or
    fifty through the same interface.
    """

    name: str
    rounds: list[Measurement] = field(default_factory=list)

    @property
    def samples(self) -> list[float]:
        return [s for r in self.rounds for s in r.samples]

    @property
    def errors(self) -> int:
        return sum(r.errors for r in self.rounds)

    def percentile(self, q: float) -> float:
        per_round = [r.percentile(q) for r in self.rounds if r.samples]
        if not per_round:
            return 0.0
        return statistics.median(per_round)

    def spread(self, q: float) -> list[float]:
        """Each round's estimate, in order. Printed, never gated on.

        A gate that reports only its aggregate hides the thing an engineer
        needs to judge it: whether the rounds agreed.
        """
        return [round(r.percentile(q), 3) for r in self.rounds if r.samples]

    def summary(self) -> dict[str, float]:
        pooled = self.samples
        return {
            "n": len(pooled),
            "rounds": len(self.rounds),
            "errors": self.errors,
            "p50": round(self.percentile(0.50), 3),
            "p95": round(self.percentile(0.95), 3),
            "p99": round(self.percentile(0.99), 3),
            "max": round(max(pooled), 3) if pooled else 0.0,
        }


#: A measured call: how to issue it, and which status codes count as a
#: successful observation rather than an error.
Call = tuple[Callable[[], Awaitable[httpx.Response]], tuple[int, ...]]


def calls(
    client: httpx.AsyncClient, throttled: httpx.AsyncClient, auth: dict[str, str]
) -> dict[str, Call]:
    """The measured paths, defined once.

    Warm-up and measurement both iterate this table. When they were written
    out separately, warm-up covered `/health` and measurement covered five
    paths, so four of them paid their first-touch cost inside the timed
    window - and a lockstep client fleet turns that into one contaminated
    sample per client per path. Sharing the table makes that class of drift
    impossible rather than merely unlikely.
    """
    return {
        "health": (lambda: client.get("/health"), (200,)),
        "metrics": (lambda: client.get("/metrics"), (200,)),
        "unauthorised": (lambda: client.post("/api/v1/runs", json={"goal": "x"}), (401, 403)),
        "invalid_body": (lambda: client.post("/api/v1/runs", json={}, headers=auth), (422,)),
        "rate_limited": (lambda: throttled.post("/api/v1/runs", json={"goal": "x"}), (429,)),
    }


@contextlib.contextmanager
def quiet_logs() -> Iterator[io.StringIO]:
    """Send log output to memory for the duration of the measurement.

    The rate-limited path logs a warning per request: 150 records, each
    rendered with ANSI colour and written with its own `flush()`. On a CI
    runner stdout is a pipe owned by the runner agent, so every one of those
    is a syscall whose duration depends on a process this one does not
    control, inside the window being timed.

    Note what is *not* done here. The records are still emitted, still
    formatted, still rendered - only the destination changes. The cost of
    logging on a hot path therefore still appears in the latency it causes,
    and the record count is reported at the end of the run, so a commit that
    starts logging per request is visible rather than absorbed.
    """
    sink = io.StringIO()
    root = logging.getLogger()
    swapped: list[tuple[logging.StreamHandler[Any], Any]] = [
        (h, h.stream) for h in root.handlers if isinstance(h, logging.StreamHandler)
    ]
    for handler, _ in swapped:
        handler.setStream(sink)
    try:
        yield sink
    finally:
        for handler, original in swapped:
            handler.setStream(original)


@contextlib.contextmanager
def frozen_heap() -> Iterator[int]:
    """Exclude the import-time heap from cyclic collection while measuring.

    This is the fix for the failure that made this gate useless: a
    generation-2 collection firing mid-run, walking the ~600k objects that
    Cortex's dependency closure leaves resident, and stopping the event loop
    for 120ms locally or 230-285ms on a CI runner. Because the loop is
    single-threaded, that pause is charged in full to whichever requests are
    in flight - so a runtime property of the interpreter was being reported
    as the latency of an HTTP path.

    `gc.freeze()` moves everything currently tracked into a permanent
    generation the collector never visits again. Collection is *not*
    disabled: objects allocated after this point are tracked and collected
    as usual, so a change that starts producing cyclic garbage per request
    still shows up as the pause it causes.

    Yields the number of frozen objects, which is worth printing - it is a
    direct measure of how much import-time heap a full collection would
    otherwise have had to walk.
    """
    gc.collect()
    gc.freeze()
    try:
        yield gc.get_freeze_count()
    finally:
        gc.unfreeze()


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


async def warm_up(table: dict[str, Call], requests: int = WARMUP_REQUESTS) -> None:
    """Exercise every measured path before anything is timed.

    Serial and single-client on purpose: the point is to reach steady state,
    not to reproduce the load profile. Returns once each path has been
    walked `requests` times, by which point its lazy imports, schema builds,
    exception handlers and metric label children all exist.
    """
    for _ in range(requests):
        for call, _ok in table.values():
            await call()


async def run_round(
    table: dict[str, Call], concurrency: int, iterations: int
) -> dict[str, Measurement]:
    """One independent estimate: `concurrency` clients x `iterations` passes."""
    results = {name: Measurement(name) for name in table}

    async def one_user() -> None:
        for _ in range(iterations):
            for name, (call, ok) in table.items():
                await _timed(results[name], call, ok)

    await asyncio.gather(*(one_user() for _ in range(concurrency)))
    return results


@dataclass
class Run:
    """A completed measurement, and the conditions it was taken under."""

    paths: dict[str, Aggregate]
    concurrency: int
    iterations: int
    frozen_objects: int = 0
    log_records: int = 0


async def run(concurrency: int = 8, iterations: int = 40, rounds: int = ROUNDS) -> Run:
    token = create_access_token(user_id="bench", tenant="bench", scopes=["runs:read"])
    auth = {"authorization": f"Bearer {token}"}
    # The 429 branch is on every request once a client misbehaves, so it
    # deserves a budget as much as the happy path does.
    rejecting = _rejecting_app()

    with quiet_logs() as sink, unthrottled(app):
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
            table = calls(client, throttled, auth)
            await warm_up(table)

            # Freeze *after* warm-up, so the objects those first requests
            # create are permanent too and no collection ever walks them.
            with frozen_heap() as frozen:
                measured = [await run_round(table, concurrency, iterations) for _ in range(rounds)]

        records = len(sink.getvalue().splitlines())

    paths = {name: Aggregate(name, [r[name] for r in measured]) for name in table}
    return Run(
        paths=paths,
        concurrency=concurrency,
        iterations=iterations,
        frozen_objects=frozen,
        log_records=records,
    )


def check(results: dict[str, Aggregate] | dict[str, Measurement]) -> list[str]:
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


def render(result: Run) -> str:
    rows = "\n".join(
        f"| `{name}` | {s['n']} | {s['p50']:.2f} | {s['p95']:.2f} | {s['p99']:.2f} | "
        f"{s['max']:.2f} | {BUDGETS[name]['p95']:.0f} / {BUDGETS[name]['p99']:.0f} |"
        for name, m in result.paths.items()
        for s in [m.summary()]
    )
    spread = "\n".join(
        f"| `{name}` | {', '.join(f'{v:.2f}' for v in m.spread(0.99))} |"
        for name, m in result.paths.items()
    )
    rounds = len(next(iter(result.paths.values())).rounds) if result.paths else 0
    return f"""# Performance

Regenerate with `python -m perf.benchmark --write`. Enforced in CI by
`make perf`, which exits non-zero on a breached budget.

## Measured

{rounds} independent rounds, each {result.concurrency} concurrent clients running
{result.iterations} iterations, ASGI in-process. Latencies in milliseconds.
`p50`, `p95` and `p99` are the **median across rounds**; `samples` and `max`
are pooled over all of them.

| path | samples | p50 | p95 | p99 | max | budget p95/p99 |
|---|---|---|---|---|---|---|
{rows}

Per-round p99, so the agreement between rounds is visible rather than
smoothed away by the median:

| path | p99 per round |
|---|---|
{spread}

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
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Cortex edge latency benchmark and gate")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument(
        "--rounds",
        type=int,
        default=ROUNDS,
        help="Independent measurement rounds; the gate reads the median of their p99s",
    )
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    result = asyncio.run(run(args.concurrency, args.iterations, args.rounds))

    if args.json:
        print(json.dumps({k: v.summary() for k, v in result.paths.items()}, indent=2))
    else:
        print(f"\n{'path':<15}{'n':>6}{'p50':>9}{'p95':>9}{'p99':>9}{'max':>9}   budget p95")
        for name, m in result.paths.items():
            s = m.summary()
            print(
                f"{name:<15}{s['n']:>6}{s['p50']:>9.2f}{s['p95']:>9.2f}"
                f"{s['p99']:>9.2f}{s['max']:>9.2f}{BUDGETS[name]['p95']:>13.0f}"
            )
        print("\np99 per round (the gate takes the median of each row):")
        for name, m in result.paths.items():
            print(f"  {name:<15}{'  '.join(f'{v:7.2f}' for v in m.spread(0.99))}")
        # Printed, not asserted on. Both numbers are how you notice the
        # harness has stopped being isolated: a collapsed freeze count means
        # the heap is being walked again, and a jump in log records means
        # something started logging on a measured path.
        print(
            f"\n{result.frozen_objects} objects frozen out of cyclic GC; "
            f"{result.log_records} log records emitted during measurement."
        )

    if args.write:
        DOC.parent.mkdir(parents=True, exist_ok=True)
        DOC.write_text(render(result))
        print(f"\nwritten to {DOC}")

    breaches = check(result.paths)
    if breaches:
        print("\nPERFORMANCE BUDGET BREACHED")
        for b in breaches:
            print(f"  - {b}")
        return 1
    print("\nAll performance budgets met.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
