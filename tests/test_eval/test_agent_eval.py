"""Agent-level evaluation.

Ragas scores retrieval and generation. Nothing scored the agent loop except
the critic — which is part of the loop, so the system was grading its own
homework and producing a number that was stable, plausible and worthless.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from cortex.eval.agent_eval import (
    AgentEvaluator,
    AgentScore,
    score_plan_efficiency,
    score_termination,
    score_tool_correctness,
)
from cortex.graph.state import CortexState, RunStatus, Task, TaskStatus


def _state(**kw) -> CortexState:
    base = {"run_id": "r", "session_id": "s", "user_id": "u", "tenant_id": "t", "user_goal": "g"}
    base.update(kw)
    return CortexState(**base)


def _task(tid, tool=None, status=TaskStatus.COMPLETED) -> Task:
    return Task(id=tid, description="d", tool=tool, depends_on=[], status=status)


class TestTermination:
    def test_a_completed_run_scores_full_marks(self):
        assert score_termination(_state(status=RunStatus.COMPLETED))[0] == 1.0

    def test_hitting_the_iteration_ceiling_scores_zero(self):
        """The loop failing to converge and being stopped from outside is
        the worst outcome: the user pays for the run and gets nothing."""
        score, reason = score_termination(
            _state(status=RunStatus.FAILED, iteration_count=25, max_iterations=25)
        )
        assert score == 0.0
        assert "converge" in reason

    def test_a_validated_failure_scores_partial(self):
        """Refusing a bad input or detecting a deadlock is correct
        behaviour that did not deliver - distinct from failing to stop."""
        score, _ = score_termination(_state(status=RunStatus.FAILED, error="dependency deadlock"))
        assert score == 0.5


class TestToolCorrectness:
    def test_calling_every_expected_tool_scores_full(self):
        state = _state(tasks=[_task("a", "search_knowledge"), _task("b", "query_memory")])
        assert score_tool_correctness(state, {"search_knowledge", "query_memory"})[0] == 1.0

    def test_missing_a_tool_costs_proportionally(self):
        state = _state(tasks=[_task("a", "search_knowledge")])
        assert score_tool_correctness(state, {"search_knowledge", "query_data"})[0] == 0.5

    def test_extra_tools_are_not_penalised(self):
        """An agent that calls an extra tool to check its work is behaving
        well. Penalising precision here trains the plan to under-verify."""
        state = _state(tasks=[_task("a", "search_knowledge"), _task("b", "query_memory")])
        assert score_tool_correctness(state, {"search_knowledge"})[0] == 1.0

    def test_no_expectation_means_no_opinion(self):
        assert score_tool_correctness(_state(tasks=[]), None)[0] == 1.0


class TestPlanEfficiency:
    def test_a_clean_single_pass_scores_full(self):
        assert score_plan_efficiency(_state(tasks=[_task("a")]))[0] == 1.0

    def test_replans_are_charged_for(self):
        """Each replan repeats the whole plan-execute cycle, so a run that
        needed three attempts cost roughly three times what it should -
        whatever the final answer looked like."""
        one = score_plan_efficiency(_state(tasks=[_task("a")]))[0]
        three = score_plan_efficiency(_state(tasks=[_task("a")], critique_iteration=2))[0]
        assert three < one
        assert three == pytest.approx(1 / 3)

    def test_incomplete_tasks_lower_the_score(self):
        state = _state(tasks=[_task("a"), _task("b", status=TaskStatus.PENDING)])
        assert score_plan_efficiency(state)[0] == 0.5

    def test_an_empty_plan_scores_zero_rather_than_dividing_by_zero(self):
        assert score_plan_efficiency(_state(tasks=[]))[0] == 0.0


class TestThresholds:
    def _good(self, **kw) -> AgentScore:
        base = {
            "task_completion": 0.9,
            "tool_correctness": 0.9,
            "plan_efficiency": 0.9,
            "termination": 1.0,
        }
        base.update(kw)
        return AgentScore(**base)

    def test_a_healthy_run_passes(self):
        assert self._good().passes()

    @pytest.mark.parametrize(
        ("axis", "value"),
        [("task_completion", 0.5), ("tool_correctness", 0.5), ("termination", 0.5)],
    )
    def test_one_collapsed_axis_fails_despite_the_others(self, axis, value):
        """Thresholds are per-axis, not on the average. 'Terminated
        correctly 40% of the time' must not hide behind an overall 0.8."""
        assert not self._good(**{axis: value}).passes()

    def test_plan_efficiency_is_reported_but_not_gated(self):
        """An inefficient plan that reaches the right answer is a cost
        problem, not a correctness one - it belongs on a dashboard, not in
        a gate that blocks a release."""
        assert self._good(plan_efficiency=0.1).passes()


class TestEvaluator:
    @pytest.mark.asyncio
    async def test_deterministic_axes_need_no_model(self):
        """The fallback is not a stub: three of four axes are computed from
        the run's own structure. A harness whose fallback measures nothing
        measures nothing in CI, where the heavy dependency is exactly the
        one nobody installs."""
        state = _state(
            status=RunStatus.COMPLETED,
            final_output="an answer",
            tasks=[_task("a", "search_knowledge")],
        )
        with patch("cortex.llm.router.get_router") as router:
            router.return_value.complete = AsyncMock(side_effect=ConnectionError("no model"))
            score = await AgentEvaluator().score(state, expected_tools={"search_knowledge"})

        assert score.tool_correctness == 1.0
        assert score.plan_efficiency == 1.0
        assert score.termination == 1.0

    @pytest.mark.asyncio
    async def test_an_unavailable_judge_scores_neither_pass_nor_fail(self):
        """0.5 deliberately. Scoring an outage as 0.0 makes it look like a
        quality regression; scoring it 1.0 lets an outage hide one."""
        state = _state(status=RunStatus.COMPLETED, final_output="an answer", tasks=[_task("a")])
        with patch("cortex.llm.router.get_router") as router:
            router.return_value.complete = AsyncMock(side_effect=ConnectionError("down"))
            score = await AgentEvaluator().score(state)

        assert score.task_completion == 0.5
        assert "not evidence" in score.details["task_completion"]

    @pytest.mark.asyncio
    async def test_no_output_scores_zero_completion(self):
        state = _state(status=RunStatus.FAILED, error="boom", tasks=[_task("a")])
        score = await AgentEvaluator().score(state)
        assert score.task_completion == 0.0

    @pytest.mark.asyncio
    async def test_the_backend_is_reported_so_numbers_are_comparable(self):
        """A DeepEval score and a judge score are not the same measurement;
        recording which produced a number is what stops them being averaged
        together in a report."""
        state = _state(status=RunStatus.COMPLETED, final_output="x", tasks=[_task("a")])
        with patch("cortex.llm.router.get_router") as router:
            router.return_value.complete = AsyncMock(side_effect=ConnectionError("down"))
            score = await AgentEvaluator().score(state)
        assert score.backend in ("deepeval", "deterministic")
        assert score.to_dict()["backend"] == score.backend
