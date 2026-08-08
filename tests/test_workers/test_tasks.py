"""Celery task bodies: retry behaviour and the eval alarm.

A background task that swallows a failure silently is worse than one that
crashes — the work is lost and nothing says so.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from cortex.workers import celery_app as w


class TestRunAgentTask:
    def test_a_successful_run_reports_status_and_cost(self):
        state = MagicMock()
        state.status.value = "completed"
        state.total_cost_usd = 0.42

        with (
            patch("cortex.graph.cortex_graph.run_cortex"),
            patch.object(w, "_run_async", return_value=state),
        ):
            out = w.run_agent_task.run(
                run_id="r1", goal="g", user_id="u", session_id="s", context={}
            )
        assert out == {"run_id": "r1", "status": "completed", "cost_usd": 0.42}

    def test_a_failing_run_is_retried(self):
        """Losing an agent run because a provider blipped is the failure
        mode `acks_late` plus retries exists to prevent."""
        task = w.run_agent_task
        with (
            patch("cortex.graph.cortex_graph.run_cortex"),
            patch.object(w, "_run_async", side_effect=RuntimeError("provider down")),
            patch.object(task, "retry", side_effect=RuntimeError("retried")) as retry,
        ):
            with pytest.raises(RuntimeError):
                task.run(run_id="r1", goal="g", user_id="u", session_id="s", context={})
        assert retry.called, "a failed run must be retried, not dropped"

    def test_exhausted_retries_return_a_failure_record_not_an_exception(self):
        """After the last retry the result must still land in the backend,
        so the API can tell the user the run failed rather than leaving it
        pending forever."""
        task = w.run_agent_task
        with (
            patch("cortex.graph.cortex_graph.run_cortex"),
            patch.object(w, "_run_async", side_effect=RuntimeError("provider down")),
            patch.object(task, "retry", side_effect=task.MaxRetriesExceededError()),
        ):
            out = task.run(run_id="r1", goal="g", user_id="u", session_id="s", context={})
        assert out["status"] == "failed"
        assert "provider down" in out["error"]


class TestEvalRegressionTask:
    def test_a_passing_suite_returns_its_scores(self):
        result = MagicMock()
        result.passes_threshold.return_value = True
        result.composite, result.faithfulness, result.answer_relevancy = 0.9, 0.9, 0.9
        result.to_dict.return_value = {"composite": 0.9}

        with patch.object(w, "_run_async", return_value=result):
            assert w.run_eval_regression.run() == {"composite": 0.9}

    def test_a_failing_suite_still_returns_rather_than_raising(self):
        """The scheduled job must record the bad score. Raising would lose
        the number that proves quality regressed."""
        result = MagicMock()
        result.passes_threshold.return_value = False
        result.composite, result.faithfulness, result.answer_relevancy = 0.3, 0.2, 0.3
        result.to_dict.return_value = {"composite": 0.3}

        with patch.object(w, "_run_async", return_value=result):
            out = w.run_eval_regression.run()
        assert out["composite"] == 0.3
