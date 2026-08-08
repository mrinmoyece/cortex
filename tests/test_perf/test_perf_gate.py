"""The performance gate itself.

A gate that cannot fail is a badge. This project already shipped an 80%
coverage gate over a suite that had never run, so the gate gets tested
before it is trusted.
"""

from __future__ import annotations

import pytest

from perf.benchmark import BUDGETS, Measurement, check


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
