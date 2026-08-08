"""Tests for Cortex agents."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from cortex.agents.critic import CriticAgent
from cortex.agents.planner import PlannerAgent
from cortex.exceptions import AgentPlanningError
from cortex.graph.state import CritiqueResult, Task, TaskStatus


class TestPlannerAgent:
    @pytest.mark.asyncio
    async def test_plan_returns_tasks(self, base_state, mock_router):
        """Planner should return a list of Task objects from a valid LLM response."""
        valid_plan = json.dumps(
            {
                "reasoning": "Break goal into three steps",
                "tasks": [
                    {
                        "id": "step1",
                        "description": "Fetch data",
                        "tool": "search_knowledge",
                        "depends_on": [],
                    },
                    {
                        "id": "step2",
                        "description": "Analyse",
                        "tool": None,
                        "depends_on": ["step1"],
                    },
                ],
            }
        )
        mock_router.complete.return_value.choices[0].message.content = valid_plan

        agent = PlannerAgent()
        with patch("cortex.agents.planner.get_router", return_value=mock_router):
            tasks = await agent.plan(base_state)

        assert len(tasks) == 2
        assert tasks[0].id == "step1"
        assert tasks[1].depends_on == ["step1"]

    @pytest.mark.asyncio
    async def test_plan_raises_on_invalid_json(self, base_state, mock_router):
        """Planner should raise AgentPlanningError when LLM returns malformed JSON."""
        mock_router.complete.return_value.choices[0].message.content = "not json"

        agent = PlannerAgent()
        with patch("cortex.agents.planner.get_router", return_value=mock_router):
            with pytest.raises(AgentPlanningError, match="invalid JSON"):
                await agent.plan(base_state)

    @pytest.mark.asyncio
    async def test_plan_raises_on_missing_tasks_key(self, base_state, mock_router):
        """Planner should raise AgentPlanningError when 'tasks' key is absent."""
        mock_router.complete.return_value.choices[0].message.content = json.dumps(
            {"reasoning": "no tasks here"}
        )

        agent = PlannerAgent()
        with patch("cortex.agents.planner.get_router", return_value=mock_router):
            with pytest.raises(AgentPlanningError, match="missing 'tasks'"):
                await agent.plan(base_state)

    @pytest.mark.asyncio
    async def test_plan_includes_critique_context(self, base_state, mock_router):
        """Planner prompt should include critique suggestions when replanning."""
        state_with_critique = base_state.model_copy(
            update={
                "critique_results": [
                    CritiqueResult(
                        accepted=False,
                        score=0.6,
                        reasoning="Missing data validation step",
                        suggestions=["Add a data validation task"],
                    )
                ]
            }
        )
        valid_plan = json.dumps(
            {
                "reasoning": "Added validation",
                "tasks": [
                    {
                        "id": "validate",
                        "description": "Validate data",
                        "tool": None,
                        "depends_on": [],
                    }
                ],
            }
        )
        mock_router.complete.return_value.choices[0].message.content = valid_plan

        agent = PlannerAgent()
        with patch("cortex.agents.planner.get_router", return_value=mock_router):
            await agent.plan(state_with_critique)

        # Verify the prompt included critique context
        call_args = mock_router.complete.call_args
        messages = call_args.kwargs["messages"]
        user_content = messages[1]["content"]
        assert "Missing data validation step" in user_content
        assert "Add a data validation task" in user_content


class TestCriticAgent:
    @pytest.mark.asyncio
    async def test_critic_accepts_high_quality_output(self, completed_state, mock_router):
        """Critic should accept output with score >= 0.80."""
        high_score_response = json.dumps(
            {
                "faithfulness": 0.95,
                "completeness": 0.90,
                "coherence": 0.92,
                "overall": 0.92,
                "accepted": True,
                "reasoning": "All claims supported by evidence",
                "suggestions": [],
            }
        )
        mock_router.complete.return_value.choices[0].message.content = high_score_response

        agent = CriticAgent()
        with patch("cortex.agents.critic.get_router", return_value=mock_router):
            result = await agent.critique(completed_state)

        assert result.accepted is True
        assert result.score >= 0.80

    @pytest.mark.asyncio
    async def test_critic_rejects_low_faithfulness(self, completed_state, mock_router):
        """Critic should reject output with faithfulness < 0.85."""
        low_score_response = json.dumps(
            {
                "faithfulness": 0.60,
                "completeness": 0.85,
                "coherence": 0.88,
                "overall": 0.78,
                "accepted": False,
                "reasoning": "Claims not fully supported by retrieved context",
                "suggestions": ["Add citations for revenue figures", "Verify growth percentage"],
            }
        )
        mock_router.complete.return_value.choices[0].message.content = low_score_response

        agent = CriticAgent()
        with patch("cortex.agents.critic.get_router", return_value=mock_router):
            result = await agent.critique(completed_state)

        assert result.accepted is False
        assert len(result.suggestions) > 0

    @pytest.mark.asyncio
    async def test_critic_handles_parse_failure_gracefully(self, completed_state, mock_router):
        """Critic should not raise on malformed LLM response — fail open with low score."""
        mock_router.complete.return_value.choices[0].message.content = "malformed"

        agent = CriticAgent()
        with patch("cortex.agents.critic.get_router", return_value=mock_router):
            result = await agent.critique(completed_state)

        # Should not raise — should return low-confidence acceptance
        assert isinstance(result, CritiqueResult)
        assert result.accepted is True
        assert result.score <= 0.7


class TestTaskModel:
    """Unit tests for the Task state model."""

    def test_task_dependency_chain(self):
        tasks = [
            Task(id="a", description="First"),
            Task(id="b", description="Second", depends_on=["a"]),
            Task(id="c", description="Third", depends_on=["b"]),
        ]
        assert tasks[1].depends_on == ["a"]
        assert tasks[2].depends_on == ["b"]

    def test_mark_completed(self):
        task = Task(id="t1", description="Test")
        completed = task.mark_completed("done")
        assert completed.status == TaskStatus.COMPLETED
        assert completed.result == "done"
        assert completed.completed_at is not None

    def test_mark_failed(self):
        task = Task(id="t1", description="Test")
        failed = task.mark_failed("connection error")
        assert failed.status == TaskStatus.FAILED
        assert failed.error == "connection error"
