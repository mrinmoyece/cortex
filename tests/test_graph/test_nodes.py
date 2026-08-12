"""Graph node behaviour — dependency ordering, deadlock, and failure isolation.

The executor node is where a plan meets reality: dependencies that cannot
be satisfied, tasks that throw, and the decision about when a run is
actually finished.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from cortex.graph.cortex_graph import executor_node, load_memory_node, save_memory_node
from cortex.graph.state import CortexState, RunStatus, Task, TaskStatus


def _state(**kw) -> CortexState:
    base = {"run_id": "r", "session_id": "s", "user_id": "u", "tenant_id": "t", "user_goal": "goal"}
    base.update(kw)
    return CortexState(**base)


def _task(tid, deps=None, status=TaskStatus.PENDING) -> Task:
    return Task(id=tid, description=f"task {tid}", tool=None, depends_on=deps or [], status=status)


class TestLoadMemoryNode:
    @pytest.mark.asyncio
    async def test_memory_is_loaded_before_planning_starts(self):
        with patch("cortex.graph.cortex_graph.MemoryAgent") as agent_cls:
            agent_cls.return_value.retrieve = AsyncMock(return_value="prior context")
            out = await load_memory_node(_state())
        assert out["memory_context"] == "prior context"
        assert out["status"] == RunStatus.PLANNING


class TestExecutorNode:
    @pytest.mark.asyncio
    async def test_no_pending_work_moves_to_critique(self):
        assert (await executor_node(_state(tasks=[])))["status"] == RunStatus.CRITIQUING

    @pytest.mark.asyncio
    async def test_a_task_waits_for_its_dependencies(self):
        """Running a dependent task early gives the model a prompt with a
        placeholder where its input should be, and it will answer anyway."""
        done = _task("a", status=TaskStatus.COMPLETED)
        blocked = _task("b", deps=["a"])
        state = _state(tasks=[done, blocked])

        with patch("cortex.graph.cortex_graph.ExecutorAgent") as agent_cls:
            agent = agent_cls.return_value
            agent.execute_task = AsyncMock(
                return_value=(blocked.mark_started().mark_completed("ok"), 0.01)
            )
            agent.compile_output = AsyncMock(return_value="final")
            await executor_node(state)

        assert agent.execute_task.await_args.args[0].id == "b"

    @pytest.mark.asyncio
    async def test_an_unsatisfiable_dependency_is_reported_as_deadlock(self):
        """Not silently skipped, and not looped on forever. A plan that
        cannot make progress is a planner bug, and it should say so."""
        out = await executor_node(_state(tasks=[_task("b", deps=["never-existed"])]))
        assert out["status"] == RunStatus.FAILED
        assert "deadlock" in out["error"].lower()

    @pytest.mark.asyncio
    async def test_cost_accumulates_onto_the_run_total(self):
        t = _task("a")
        state = _state(tasks=[t], total_cost_usd=0.10)
        with patch("cortex.graph.cortex_graph.ExecutorAgent") as agent_cls:
            agent = agent_cls.return_value
            agent.execute_task = AsyncMock(
                return_value=(t.mark_started().mark_completed("ok"), 0.05)
            )
            agent.compile_output = AsyncMock(return_value="final")
            out = await executor_node(state)
        assert out["total_cost_usd"] == pytest.approx(0.15)

    @pytest.mark.asyncio
    async def test_output_is_compiled_only_once_every_task_is_done(self):
        pending, running = _task("a"), _task("b")
        state = _state(tasks=[pending, running])
        with patch("cortex.graph.cortex_graph.ExecutorAgent") as agent_cls:
            agent = agent_cls.return_value
            agent.execute_task = AsyncMock(
                return_value=(pending.mark_started().mark_completed("ok"), 0.0)
            )
            agent.compile_output = AsyncMock(return_value="final")
            out = await executor_node(state)

        assert out["final_output"] is None, "one task still pending - nothing to compile yet"
        assert out["status"] == RunStatus.EXECUTING
        agent.compile_output.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_failed_task_does_not_abort_the_whole_run(self):
        """One task throwing must degrade to a failed task, not a failed
        run - the other tasks may still produce a usable answer."""
        t = _task("a")
        with patch("cortex.graph.cortex_graph.ExecutorAgent") as agent_cls:
            agent_cls.return_value.execute_task = AsyncMock(side_effect=RuntimeError("boom"))
            out = await executor_node(_state(tasks=[t]))

        assert "status" not in out or out.get("status") != RunStatus.FAILED
        assert out["tasks"][0].status == TaskStatus.FAILED
        assert "boom" in out["tasks"][0].error

    @pytest.mark.asyncio
    async def test_the_ledger_total_is_preferred_when_it_is_available(self):
        """Compilation makes an extra LLM call after the task, so the
        per-task delta alone under-reports what the run actually spent."""
        t = _task("a")
        state = _state(tasks=[t], total_cost_usd=0.10)
        with patch("cortex.graph.cortex_graph.ExecutorAgent") as agent_cls:
            agent = agent_cls.return_value
            agent.execute_task = AsyncMock(
                return_value=(t.mark_started().mark_completed("ok"), 0.05)
            )
            agent.compile_output = AsyncMock(return_value="final")
            agent.get_run_cost = AsyncMock(return_value=0.22)
            out = await executor_node(state)

        assert out["total_cost_usd"] == pytest.approx(0.22)

    @pytest.mark.asyncio
    async def test_an_unreachable_cost_ledger_does_not_discard_a_finished_task(self):
        """The ledger lookup shares a `try` with task execution. If Redis is
        down, an accounting failure must not be able to mark a task that
        succeeded as failed and throw its result away."""
        t = _task("a")
        state = _state(tasks=[t], total_cost_usd=0.10)
        with patch("cortex.graph.cortex_graph.ExecutorAgent") as agent_cls:
            agent = agent_cls.return_value
            agent.execute_task = AsyncMock(
                return_value=(t.mark_started().mark_completed("ok"), 0.05)
            )
            agent.compile_output = AsyncMock(return_value="final")
            agent.get_run_cost = AsyncMock(side_effect=ConnectionError("redis down"))
            out = await executor_node(state)

        assert out["tasks"][0].status == TaskStatus.COMPLETED
        assert out["tasks"][0].result == "ok"
        assert out["final_output"] == "final"
        assert out["status"] == RunStatus.CRITIQUING
        assert out["total_cost_usd"] == pytest.approx(0.15), "degrades to the accumulated total"


class TestSaveMemoryNode:
    @pytest.mark.asyncio
    async def test_the_run_is_consolidated_into_memory(self):
        with patch("cortex.graph.cortex_graph.MemoryAgent") as agent_cls:
            agent_cls.return_value.consolidate = AsyncMock(return_value=None)
            out = await save_memory_node(_state(final_output="answer"))
        assert out["status"] == RunStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_a_memory_write_failure_does_not_fail_a_finished_run(self):
        """The user already has their answer. Losing the memory write is a
        degradation; losing the answer is a bug."""
        with patch("cortex.graph.cortex_graph.MemoryAgent") as agent_cls:
            agent_cls.return_value.consolidate = AsyncMock(
                side_effect=ConnectionError("redis down")
            )
            out = await save_memory_node(_state(final_output="answer"))
        assert out["status"] == RunStatus.COMPLETED
