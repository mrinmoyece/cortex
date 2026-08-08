"""The evaluation layer, which was at 0% coverage.

An eval harness nobody runs is the same failure as a coverage gate nobody
meets: the number it produces is decoration. These tests exercise the
thresholds, the Ragas-absent fallback, and the regression loader.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from cortex.eval.ragas_runner import EvalResult, EvalSample, RagasEvaluator, run_regression_suite


def _result(faith=0.9, rel=0.9, prec=0.9, comp=0.9, n=5, failed=0):
    return EvalResult(
        faithfulness=faith,
        answer_relevancy=rel,
        context_precision=prec,
        context_recall=None,
        composite=comp,
        sample_count=n,
        failed_samples=failed,
    )


class TestThresholds:
    def test_a_good_result_passes(self):
        assert _result().passes_threshold() is True

    @pytest.mark.parametrize(
        ("field", "value"),
        [("faith", 0.79), ("rel", 0.74), ("comp", 0.77)],
        ids=["faithfulness-just-under", "relevancy-just-under", "composite-just-under"],
    )
    def test_each_threshold_can_fail_on_its_own(self, field, value):
        """One dimension collapsing must fail the gate even when the others
        are perfect - otherwise the gate is really just the average."""
        assert _result(**{field: value}).passes_threshold() is False

    def test_thresholds_are_the_documented_values(self):
        """docs/EVALUATION.md quotes 0.80 / 0.75 / 0.78. A doc that drifts
        from the default is a doc that misleads."""
        assert _result(faith=0.80, rel=0.75, comp=0.78).passes_threshold() is True
        assert _result(faith=0.7999).passes_threshold() is False

    def test_to_dict_rounds_and_keeps_every_field(self):
        d = _result(faith=0.123456).to_dict()
        assert d["faithfulness"] == 0.1235
        assert set(d) >= {
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "composite",
            "sample_count",
            "failed_samples",
        }


class TestRagasFallback:
    def test_reports_whether_ragas_is_available(self):
        ev = RagasEvaluator()
        assert isinstance(ev._ragas_available, bool)

    def test_missing_ragas_degrades_instead_of_raising(self):
        """Ragas is a heavy optional dependency. Constructing the evaluator
        without it must not explode - the LLM-judge path exists precisely
        so evaluation still happens."""
        import builtins

        real_import = builtins.__import__

        def no_ragas(name, *args, **kwargs):
            if name == "ragas":
                raise ImportError("not installed")
            return real_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", no_ragas):
            ev = RagasEvaluator()
        assert ev._ragas_available is False

    @pytest.mark.asyncio
    async def test_evaluating_nothing_is_an_error_not_a_zero_score(self):
        """Deliberate asymmetry, worth pinning down: `evaluate([])` raises
        because asking for a score over no samples is a caller bug, while
        `run_regression_suite` on a missing file returns zero because CI
        needs a number rather than a stack trace. Both are defensible; only
        a test stops them drifting into each other."""
        ev = RagasEvaluator()
        ev._ragas_available = False
        with pytest.raises(ValueError, match="No samples"):
            await ev.evaluate([])


class TestRegressionSuite:
    @pytest.mark.asyncio
    async def test_missing_file_returns_a_zero_result_not_an_exception(self, tmp_path):
        """CI should report 'no cases' as a score of zero, not a stack trace
        that looks like the harness itself is broken."""
        out = await run_regression_suite(str(tmp_path / "nope.json"))
        assert out.sample_count == 0
        assert out.composite == 0.0

    @pytest.mark.asyncio
    async def test_loads_the_shipped_regression_cases(self, tmp_path):
        cases = [
            {"question": f"q{i}", "answer": f"a{i}", "contexts": [f"c{i}"], "ground_truth": f"g{i}"}
            for i in range(5)
        ]
        path = tmp_path / "cases.json"
        path.write_text(json.dumps(cases))

        with patch(
            "cortex.eval.ragas_runner.RagasEvaluator.evaluate",
            AsyncMock(return_value=_result(n=5)),
        ):
            out = await run_regression_suite(str(path))
        assert out.sample_count == 5

    def test_the_shipped_cases_file_is_valid_and_complete(self):
        """The file CI points at, checked directly - a regression suite with
        a malformed case file fails in a way that looks like a code bug."""
        cases = json.loads(open("tests/eval/regression_cases.json").read())
        assert len(cases) >= 5
        for c in cases:
            assert {"question", "answer", "contexts", "ground_truth"} <= set(c)
            assert isinstance(c["contexts"], list) and c["contexts"]


class TestLLMJudgeFallback:
    """The path taken whenever Ragas is not installed - which, given it is a
    heavy optional dependency, is most deployments."""

    @pytest.mark.asyncio
    async def test_the_judge_scores_every_dimension(self):
        from types import SimpleNamespace

        ev = RagasEvaluator()
        ev._ragas_available = False

        judged = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(
                            {
                                "faithfulness": 0.9,
                                "relevancy": 0.85,
                                "reasoning": "grounded",
                            }
                        )
                    )
                )
            ]
        )
        # `get_router` is imported inside the method, so the patch target is
        # the source module - the mirror image of the agents, which import it
        # at module scope and therefore need the singleton patched instead.
        with patch("cortex.llm.router.get_router") as router:
            router.return_value.complete = AsyncMock(return_value=judged)
            out = await ev.evaluate(
                [EvalSample(question="q", answer="a", contexts=["c"], ground_truth="g")]
            )

        assert 0.0 <= out.faithfulness <= 1.0
        assert out.sample_count == 1

    @pytest.mark.asyncio
    async def test_a_judge_that_returns_junk_does_not_crash_the_suite(self):
        """An unparseable judge response must count as a failed sample, not
        take down a scheduled regression run."""
        from types import SimpleNamespace

        ev = RagasEvaluator()
        ev._ragas_available = False
        junk = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="not json at all"))]
        )
        with patch("cortex.llm.router.get_router") as router:
            router.return_value.complete = AsyncMock(return_value=junk)
            out = await ev.evaluate(
                [EvalSample(question="q", answer="a", contexts=["c"], ground_truth="g")]
            )
        assert out.failed_samples >= 1
