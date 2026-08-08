"""Graph routing: the brakes that stop a run looping or overspending.

Routing is pure and cheap to test, and it is where an agent platform
actually fails - not in a clever prompt, but in a conditional edge that
sends a failed run back round the loop.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from cortex.graph.cortex_graph import build_graph, route_after_critic, route_after_executor
from cortex.graph.state import CortexState, RunStatus


def _state(**kw) -> CortexState:
    base = {
        "run_id": "r",
        "session_id": "s",
        "user_id": "u",
        "tenant_id": "t",
        "user_goal": "do a thing",
    }
    base.update(kw)
    return CortexState(**base)


class TestRouteAfterExecutor:
    def test_a_failed_run_ends_rather_than_looping(self):
        assert route_after_executor(_state(status=RunStatus.FAILED)) == "end_failed"

    def test_work_continues_while_under_every_limit(self):
        s = _state(status=RunStatus.EXECUTING, total_cost_usd=0.01, iteration_count=1)
        assert route_after_executor(s) == "executor"

    def test_critiquing_moves_to_the_critic(self):
        assert route_after_executor(_state(status=RunStatus.CRITIQUING)) == "critic"

    def test_the_cost_ceiling_halts_the_loop(self):
        s = _state(status=RunStatus.EXECUTING, total_cost_usd=999.0, iteration_count=1)
        assert route_after_executor(s) == "end_failed"

    def test_the_ceiling_is_the_configured_one_not_a_literal(self):
        """This guard read `>= 2.0`, which equals the default - so it looked
        correct and ignored the setting entirely. Raising the configured
        ceiling must actually raise it."""
        s = _state(status=RunStatus.EXECUTING, total_cost_usd=5.0, iteration_count=1)
        assert route_after_executor(s) == "end_failed", "5.0 should exceed the $2 default"

        with patch("cortex.graph.cortex_graph.settings") as cfg:
            cfg.max_cost_per_run_usd = 50.0
            assert route_after_executor(s) == "executor", "a raised ceiling must be honoured"

    def test_the_iteration_ceiling_halts_the_loop(self):
        """Without this, a task that never satisfies the executor spins
        until the cost ceiling catches it - burning the whole budget to
        discover a bug the iteration count would have caught in seconds."""
        s = _state(status=RunStatus.EXECUTING, total_cost_usd=0.0, iteration_count=25)
        assert route_after_executor(s) == "end_failed"

    def test_the_iteration_ceiling_is_per_run_not_global(self):
        s = _state(status=RunStatus.EXECUTING, iteration_count=30, max_iterations=100)
        assert route_after_executor(s) == "executor"


class TestRouteAfterCritic:
    def test_accepted_output_is_saved(self):
        assert route_after_critic(_state(status=RunStatus.COMPLETED)) == "save_memory"

    def test_a_failed_run_ends(self):
        assert route_after_critic(_state(status=RunStatus.FAILED)) == "end_failed"

    def test_rejection_sends_the_work_back_to_the_planner(self):
        s = _state(status=RunStatus.PLANNING, critique_iteration=1, max_critique_iterations=3)
        assert route_after_critic(s) == "planner"

    @pytest.mark.asyncio
    async def test_critique_attempts_are_bounded_in_the_node_not_the_router(self):
        """A critic that never accepts would replan forever. The bound is
        what makes "self-improving" terminate - and it lives in
        `critic_node`, not in `route_after_critic`, which only reads the
        status the node already decided. Testing the router for this would
        have asserted nothing."""
        from cortex.agents.critic import CritiqueResult
        from cortex.graph.cortex_graph import critic_node

        rejection = CritiqueResult(accepted=False, score=0.1, reasoning="no", suggestions=[])

        with patch("cortex.graph.cortex_graph.CriticAgent") as agent_cls:
            agent_cls.return_value.critique = AsyncMock(return_value=rejection)

            # Under the bound: rejected, so replan.
            mid = await critic_node(
                _state(final_output="draft", critique_iteration=0, max_critique_iterations=3)
            )
            assert mid["status"] == RunStatus.PLANNING
            assert mid["critique_iteration"] == 1

            # At the bound: stop, and mark the output low-confidence rather
            # than looping or silently passing it off as accepted.
            last = await critic_node(
                _state(final_output="draft", critique_iteration=2, max_critique_iterations=3)
            )
            assert last["status"] == RunStatus.COMPLETED
            assert last["output_metadata"]["low_confidence"] is True
            assert route_after_critic(_state(status=last["status"])) == "save_memory"

    @pytest.mark.asyncio
    async def test_nothing_to_critique_fails_rather_than_scoring_emptiness(self):
        from cortex.graph.cortex_graph import critic_node

        out = await critic_node(_state(final_output=None))
        assert out["status"] == RunStatus.FAILED


class TestGraphAssembly:
    def test_the_graph_compiles(self):
        assert build_graph() is not None

    def test_every_declared_node_is_present(self):
        g = build_graph()
        nodes = set(g.get_graph().nodes)
        for expected in ("load_memory", "planner", "executor", "critic", "save_memory"):
            assert expected in nodes, f"{expected} missing from {sorted(nodes)}"

    def test_it_interrupts_before_the_critic_for_human_review(self):
        """Human-in-the-loop is a headline claim; without the interrupt it
        is just a comment."""
        g = build_graph()
        assert "critic" in g.interrupt_before_nodes

    def test_a_checkpointer_is_attached_so_runs_can_resume(self):
        g = build_graph()
        assert g.checkpointer is not None
