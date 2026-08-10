"""The performance gate itself.

A gate that cannot fail is a badge. This project already shipped an 80%
coverage gate over a suite that had never run, so the gate gets tested
before it is trusted.
"""

from __future__ import annotations

import asyncio
import gc
import logging
from typing import ClassVar

import pytest
from httpx import ASGITransport, AsyncClient
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route

from cortex.api.ratelimit import RateLimiter, RateLimitMiddleware
from cortex.logging_config import get_logger
from perf.benchmark import (
    BUDGETS,
    WARMUP_REQUESTS,
    Aggregate,
    Measurement,
    calls,
    check,
    frozen_heap,
    quiet_logs,
    run_round,
    warm_up,
)


def _fast(name: str) -> Measurement:
    return Measurement(name=name, samples=[0.5] * 100)


def _slow(name: str) -> Measurement:
    limit = max(BUDGETS[name].values())
    return Measurement(name=name, samples=[limit * 10] * 100)


def _all_fast() -> dict[str, Measurement]:
    return {name: _fast(name) for name in BUDGETS}


class TestPercentiles:
    def test_percentiles_are_ordered(self):
        m = Measurement(name="x", samples=[float(i) for i in range(100)])
        assert m.percentile(0.50) <= m.percentile(0.95) <= m.percentile(0.99)

    def test_the_tail_is_not_averaged_away(self):
        """One 10-second request among 99 fast ones must move p99 but not
        p50 - which is the entire reason percentiles are reported instead
        of a mean."""
        m = Measurement(name="x", samples=[1.0] * 99 + [10_000.0])
        assert m.percentile(0.50) == 1.0
        assert m.percentile(0.99) == 10_000.0

    def test_no_samples_reports_zero_rather_than_dividing_by_zero(self):
        assert Measurement(name="x").percentile(0.95) == 0.0


class TestGate:
    def test_a_healthy_run_passes(self):
        assert check(_all_fast()) == []

    @pytest.mark.parametrize("name", list(BUDGETS))
    def test_each_budget_can_fail_on_its_own(self, name):
        """One path regressing must fail the build even when everything
        else is fast - otherwise the gate is really an average."""
        results = _all_fast()
        results[name] = _slow(name)
        breaches = check(results)
        assert breaches, f"{name} blew its budget 10x and the gate stayed green"
        assert any(name in b for b in breaches)

    def test_a_path_with_no_samples_is_a_failure_not_a_pass(self):
        """Silently skipping an unmeasured path is how a gate quietly stops
        covering the endpoint someone removed from the harness.

        The path name is taken from BUDGETS rather than hardcoded - the two
        projects budget different endpoints, and a test that names one of
        them passes vacuously in the other."""
        name = next(iter(BUDGETS))
        results = _all_fast()
        results[name] = Measurement(name=name)
        assert any(name in b for b in check(results))

    def test_every_budget_has_both_percentiles(self):
        for name, budget in BUDGETS.items():
            assert {"p95", "p99"} <= set(budget), f"{name} is missing a percentile"
            assert budget["p99"] >= budget["p95"], f"{name}: p99 budget below p95"


class TestAggregate:
    """The gate statistic: the median of independent rounds.

    The reason this type exists is a failure, so the failure is what it is
    tested against. The `performance` check was red on every run - on `main`
    too - always `unauthorised p99 ~230ms > 50ms` with p95 near 8ms. A
    generation-2 GC pass over Cortex's ~600k-object import heap stopped the
    event loop for a fifth of a second, and nearest-rank p99 over 150
    samples is the second-worst observation, so the two requests in flight
    at that moment *were* the number the build gated on.
    """

    @staticmethod
    def _round(name: str, ms: float, n: int = 100) -> Measurement:
        return Measurement(name=name, samples=[ms] * n)

    @staticmethod
    def _stalled_round(name: str, ms: float, stall: float, n: int = 100) -> Measurement:
        """A round in which two samples absorbed a process-wide pause."""
        return Measurement(name=name, samples=[ms] * (n - 2) + [stall, stall])

    def test_percentiles_are_the_median_across_rounds(self):
        agg = Aggregate("x", [self._round("x", 1.0), self._round("x", 5.0), self._round("x", 3.0)])
        assert agg.percentile(0.99) == 3.0

    def test_samples_and_max_stay_pooled(self):
        """Medianing the *count* would misreport how much evidence there is,
        and medianing the max would hide the worst thing that happened."""
        agg = Aggregate(
            "x", [self._round("x", 1.0, n=10), self._stalled_round("x", 1.0, 200.0, 10)]
        )
        assert len(agg.samples) == 20
        assert agg.summary()["max"] == 200.0
        assert agg.summary()["n"] == 20

    def test_one_stalled_round_does_not_decide_the_gate(self):
        """The exact CI failure: one round contaminated by a 230ms pause,
        every other round healthy."""
        name = "unauthorised"
        rounds = [self._round(name, 8.0) for _ in range(5)]
        rounds[2] = self._stalled_round(name, 8.0, 230.0)
        results = _all_fast()
        results[name] = Aggregate(name, rounds)

        assert rounds[2].percentile(0.99) == 230.0, "the round itself is still contaminated"
        assert check(results) == []

    def test_a_regression_in_a_majority_of_rounds_still_fails(self):
        """Robustness is not immunity. Median has a breakdown point of half
        the rounds, and past it the gate fires - which is what stops this
        being a way to launder a real regression."""
        name = "unauthorised"
        over = BUDGETS[name]["p99"] * 2
        rounds = [self._round(name, over) for _ in range(3)] + [self._round(name, 1.0)] * 2
        results = _all_fast()
        results[name] = Aggregate(name, rounds)
        assert any(name in b for b in check(results))

    def test_pooling_would_not_have_worked(self):
        """The justification for medianing rather than simply collecting
        more samples, asserted rather than claimed.

        Stalls arrive at a rate, so every round contributes its own couple
        of contaminated samples: pooling more rounds keeps contamination at
        the same ~2% of the distribution, which is above the 99th
        percentile no matter how large the pool gets."""
        name = "unauthorised"
        rounds = [self._stalled_round(name, 8.0, 230.0) for _ in range(5)]
        pooled = Measurement(name=name, samples=[s for r in rounds for s in r.samples])
        assert pooled.percentile(0.99) == 230.0, "pooling does not dilute a per-round stall"
        assert Aggregate(name, rounds).percentile(0.99) == 230.0, "and the median agrees here"

    def test_spread_exposes_each_round_estimate(self):
        agg = Aggregate("x", [self._round("x", 1.0), self._round("x", 9.0)])
        assert agg.spread(0.99) == [1.0, 9.0]

    def test_errors_are_summed_not_medianed(self):
        rounds = [Measurement("x", samples=[1.0], errors=2), Measurement("x", samples=[1.0])]
        assert Aggregate("x", rounds).errors == 2

    def test_a_path_measured_in_no_round_is_a_failure(self):
        name = next(iter(BUDGETS))
        results = _all_fast()
        results[name] = Aggregate(name, [Measurement(name), Measurement(name)])
        assert any(name in b for b in check(results))


async def _ok(request):
    return PlainTextResponse("ok")


class _Response:
    def __init__(self, status_code: int = 200) -> None:
        self.status_code = status_code


def _table(**delays: float):
    """A stub call table: path name -> how long that path takes.

    Lets the harness be tested end to end - warm-up, rounds, aggregation,
    budgets - without an HTTP stack in the way, and with latency chosen
    rather than hoped for.
    """
    issued: dict[str, int] = dict.fromkeys(delays, 0)

    def make(name: str, delay: float):
        async def call():
            issued[name] += 1
            if delay:
                await asyncio.sleep(delay)
            return _Response(200)

        return call

    return {name: (make(name, d), (200,)) for name, d in delays.items()}, issued


class TestWarmUp:
    """Every measured path is warmed, because for a long time only one was.

    `/health` was warmed and the other four paths paid their first-touch
    cost - lazy imports, pydantic-core schema builds, metric label children
    - inside the timed window. With clients running in lockstep that is one
    contaminated sample per client per path, on a statistic decided by the
    two worst samples.
    """

    def test_warm_up_covers_exactly_the_budgeted_paths(self):
        """Warm-up and measurement read the same table, so they cannot drift
        apart the way they had."""
        table = calls(client=None, throttled=None, auth={})
        assert set(table) == set(BUDGETS)

    async def test_every_path_is_warmed_the_same_number_of_times(self):
        table, issued = _table(a=0.0, b=0.0, c=0.0)
        await warm_up(table, requests=7)
        assert issued == {"a": 7, "b": 7, "c": 7}

    async def test_warm_up_requests_are_not_measured(self):
        table, issued = _table(a=0.0)
        await warm_up(table, requests=WARMUP_REQUESTS)
        results = await run_round(table, concurrency=2, iterations=3)
        assert len(results["a"].samples) == 6, "warm-up leaked into the samples"
        assert issued["a"] == WARMUP_REQUESTS + 6


class TestFrozenHeap:
    """The root cause. A generation-2 collection walks every object the
    dependency closure left resident - 120ms on a laptop, 230-285ms on a CI
    runner - and stops the single event loop, so the pause is charged to
    whichever requests are in flight."""

    def test_the_resident_heap_is_excluded_from_collection(self):
        before = len(gc.get_objects())
        with frozen_heap() as frozen:
            assert frozen >= before * 0.9, "the resident heap was not frozen"
            # `gc.get_objects()` does not report the permanent generation,
            # so this is a direct measure of what a collection would walk.
            assert len(gc.get_objects()) < before / 2

    def test_collection_is_frozen_not_disabled(self):
        """Freezing the *existing* heap is isolation. Turning the collector
        off would be hiding a regression: a change that starts producing
        cyclic garbage per request has to still pay for it."""
        with frozen_heap():
            assert gc.isenabled()
            garbage = {}
            garbage["self"] = garbage
            del garbage
            assert gc.collect() >= 0

    def test_the_heap_is_unfrozen_afterwards(self):
        """A benchmark that permanently froze the interpreter's heap would
        leak that decision into every test that runs after it."""
        with frozen_heap():
            pass
        assert gc.get_freeze_count() == 0

    def test_unfrozen_even_when_the_run_raises(self):
        with pytest.raises(RuntimeError), frozen_heap():
            raise RuntimeError("round failed")
        assert gc.get_freeze_count() == 0


class TestQuietLogs:
    """The rate-limited path logs a warning per request. On CI stdout is a
    pipe owned by the runner agent, so each of those is a syscall that can
    block on a process this one does not control - inside the timed window.
    """

    def test_records_do_not_reach_stdout_during_measurement(self, capsys):
        logger = get_logger("cortex.test")
        with quiet_logs() as sink:
            logger.warning("api.rate_limited", path="/api/v1/runs")
        assert "api.rate_limited" not in capsys.readouterr().out
        assert "api.rate_limited" in sink.getvalue(), "the record was dropped, not redirected"

    def test_records_are_still_rendered_in_full(self):
        """Not silenced, and not cheapened. The cost of logging on a hot
        path still lands in the latency it causes, and the count is reported
        after every run, so a commit that starts logging per request cannot
        hide in here."""
        logger = get_logger("cortex.test")
        with quiet_logs() as sink:
            for _ in range(5):
                logger.warning("api.rate_limited", path="/api/v1/runs")
        written = sink.getvalue()
        assert written.count("api.rate_limited") >= 5
        assert "/api/v1/runs" in written, "structured fields were not rendered"

    def test_the_original_stream_is_restored(self):
        streams = {
            h: h.stream
            for h in logging.getLogger().handlers
            if isinstance(h, logging.StreamHandler)
        }
        with quiet_logs():
            pass
        for handler, stream in streams.items():
            assert handler.stream is stream

    def test_the_stream_is_restored_when_the_run_raises(self):
        streams = {
            h: h.stream
            for h in logging.getLogger().handlers
            if isinstance(h, logging.StreamHandler)
        }
        with pytest.raises(RuntimeError), quiet_logs():
            raise RuntimeError("round failed")
        for handler, stream in streams.items():
            assert handler.stream is stream


class TestGateEndToEnd:
    """Through the whole harness - rounds, aggregation, budgets - because
    the point of the change is what the *combination* does."""

    BUDGET: ClassVar[dict[str, dict[str, float]]] = {"slow": {"p95": 10.0, "p99": 10.0}}

    async def _rounds(self, table, n: int, concurrency: int = 2, iterations: int = 5):
        return {
            "slow": Aggregate(
                "slow",
                [(await run_round(table, concurrency, iterations))["slow"] for _ in range(n)],
            )
        }

    async def test_a_real_regression_still_fails_the_gate(self, monkeypatch):
        """40ms on every request of a measured path, against a 10ms budget.
        If this ever goes green the gate has been neutralised."""
        monkeypatch.setattr("perf.benchmark.BUDGETS", self.BUDGET)
        table, _ = _table(slow=0.04)
        breaches = check(await self._rounds(table, n=3))
        assert any("slow p99" in b for b in breaches), breaches

    async def test_a_single_stalled_round_does_not_fail_the_gate(self, monkeypatch):
        """The CI failure, reproduced through the real code path: one round
        in five hits a 200ms process-wide pause, the rest are healthy."""
        monkeypatch.setattr("perf.benchmark.BUDGETS", self.BUDGET)

        stalls = iter([False, False, True, False, False])
        rounds = []
        for stalled in stalls:
            table, _ = _table(slow=0.2 if stalled else 0.0)
            rounds.append((await run_round(table, 2, 5))["slow"])

        results = {"slow": Aggregate("slow", rounds)}
        assert rounds[2].percentile(0.99) > 10.0, "the stalled round is genuinely over budget"
        assert check(results) == []


class TestAgainstTheRealApp:
    """One run of the actual harness against the actual ASGI app, small
    enough for the unit suite. Cheap insurance that the isolation described
    above survives contact with the real middleware stack."""

    async def test_a_real_run_is_isolated_and_complete(self, capsys):
        from perf.benchmark import run

        result = await run(concurrency=2, iterations=3, rounds=2)

        assert set(result.paths) == set(BUDGETS)
        for name, agg in result.paths.items():
            assert len(agg.rounds) == 2
            assert len(agg.samples) == 12, f"{name} lost samples to errors"
            assert agg.errors == 0, f"{name} saw unexpected status codes"

        assert result.frozen_objects > 0, "the resident heap was never frozen"
        assert result.log_records > 0, "the rate-limited path stopped logging, or the sink broke"
        assert "api.rate_limited" not in capsys.readouterr().out

    async def test_the_heap_and_log_stream_are_left_as_they_were(self):
        from perf.benchmark import run

        streams = {
            h: h.stream
            for h in logging.getLogger().handlers
            if isinstance(h, logging.StreamHandler)
        }
        await run(concurrency=1, iterations=2, rounds=1)
        assert gc.get_freeze_count() == 0
        for handler, stream in streams.items():
            assert handler.stream is stream


class TestUnthrottled:
    """The benchmark's own traffic must not be rate limited, and that must
    not depend on which module imported the app first.

    `RateLimitMiddleware` builds its limiter at import time, so the
    environment variable this module sets before importing the app only
    works when it wins the race. When it loses - a test process, an
    interactive session - warm-up drains the bucket and every metered path
    answers 429, which is the rate limiter being measured under five other
    names.
    """

    async def test_a_metered_path_is_not_throttled_by_the_benchmarks_own_load(self):
        from cortex.api.main import app
        from perf.benchmark import unthrottled

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://bench") as client:
            with unthrottled(app):
                codes = set()
                for _ in range(200):
                    codes.add((await client.post("/api/v1/runs", json={"goal": "x"})).status_code)
        assert codes == {401}, f"the benchmark throttled itself: {codes}"

    async def test_the_apps_own_limit_is_restored_afterwards(self):
        """A harness that permanently lifted the limit on the imported app
        would silently disarm the rate-limit tests that run after it."""
        from perf.benchmark import unthrottled

        metered = Starlette(routes=[Route("/api/v1/runs", _ok, methods=["POST"])])
        metered.add_middleware(RateLimitMiddleware, limiter=RateLimiter(per_minute=2, burst=2))

        async def codes(n: int) -> set[int]:
            transport = ASGITransport(app=metered)
            async with AsyncClient(transport=transport, base_url="http://bench") as client:
                return {(await client.post("/api/v1/runs", json={})).status_code for _ in range(n)}

        assert 429 in await codes(10), "the fixture app was not actually metered"
        with unthrottled(metered):
            assert await codes(200) == {200}
        assert 429 in await codes(10), "the limit did not come back"
