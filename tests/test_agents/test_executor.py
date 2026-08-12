"""The executor loop: tool rounds, failure signalling, and its bound.

At 26% coverage this was the least-tested agent, and it is the one that
actually spends money — a loop with a broken exit condition here burns the
run budget before any other control notices.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from cortex.agents.executor import ExecutorAgent
from cortex.graph.state import CortexState, Task, TaskStatus


def _state(**kw) -> CortexState:
    base = {"run_id": "r", "session_id": "s", "user_id": "u", "tenant_id": "t", "user_goal": "goal"}
    base.update(kw)
    return CortexState(**base)


def _task(tid="t1") -> Task:
    return Task(id=tid, description="do the thing", tool=None, depends_on=[])


def _msg(content=None, tool_calls=None):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=tool_calls))]
    )


def _tool_call(name="search_knowledge", args=None, cid="c1"):
    return SimpleNamespace(
        id=cid,
        function=SimpleNamespace(name=name, arguments=json.dumps(args or {"query": "x"})),
    )


@pytest.fixture
def agent():
    with patch("cortex.agents.executor.get_router"), patch("cortex.agents.executor.get_mcp_client"):
        a = ExecutorAgent()
    a._router = AsyncMock()
    a._router.get_run_cost = AsyncMock(return_value=0.0)
    a._mcp = AsyncMock()
    a._mcp.get_tool_schemas = AsyncMock(return_value=[])
    return a


class TestExecuteTask:
    @pytest.mark.asyncio
    async def test_a_plain_answer_completes_the_task(self, agent):
        agent._router.complete = AsyncMock(return_value=_msg(content="the answer"))
        task, _cost = await agent.execute_task(_task(), _state())
        assert task.status == TaskStatus.COMPLETED
        assert task.result == "the answer"

    @pytest.mark.asyncio
    async def test_cost_delta_comes_from_the_router_ledger(self, agent):
        agent._router.get_run_cost = AsyncMock(side_effect=[0.1, 0.4])
        agent._router.complete = AsyncMock(return_value=_msg(content="the answer"))

        _, cost = await agent.execute_task(_task(), _state(total_cost_usd=0.0))

        assert cost == pytest.approx(0.3)

    @pytest.mark.asyncio
    async def test_the_failure_sentinel_marks_the_task_failed(self, agent):
        """The model signals failure in-band with `TASK_FAILED:`. If that
        prefix is ever mishandled, a failed task is recorded as a successful
        one whose result happens to be an apology."""
        agent._router.complete = AsyncMock(return_value=_msg(content="TASK_FAILED: no data source"))
        task, _ = await agent.execute_task(_task(), _state())
        assert task.status == TaskStatus.FAILED
        assert "no data source" in task.error

    @pytest.mark.asyncio
    async def test_a_tool_call_is_executed_then_the_answer_returned(self, agent):
        agent._router.complete = AsyncMock(
            side_effect=[_msg(content=None, tool_calls=[_tool_call()]), _msg(content="done")]
        )
        agent._call_mcp_tool = AsyncMock(return_value={"result": "found it"})
        task, _ = await agent.execute_task(_task(), _state())
        assert task.status == TaskStatus.COMPLETED
        assert agent._call_mcp_tool.await_count == 1

    @pytest.mark.asyncio
    async def test_tool_rounds_are_bounded_at_three(self, agent):
        """A model that calls a tool forever must be stopped by the loop,
        not by the budget. Three rounds is the documented bound."""
        agent._router.complete = AsyncMock(
            return_value=_msg(content=None, tool_calls=[_tool_call()])
        )
        agent._call_mcp_tool = AsyncMock(return_value={"result": "again"})

        task, _ = await agent.execute_task(_task(), _state())
        assert task.status == TaskStatus.FAILED
        assert "maximum tool-call rounds" in task.error
        assert agent._router.complete.await_count == 3, "the bound is 3 rounds, not 3 tool calls"

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_in_one_round_all_run(self, agent):
        agent._router.complete = AsyncMock(
            side_effect=[
                _msg(content=None, tool_calls=[_tool_call(cid="a"), _tool_call(cid="b")]),
                _msg(content="done"),
            ]
        )
        agent._call_mcp_tool = AsyncMock(return_value={"result": "ok"})
        await agent.execute_task(_task(), _state())
        assert agent._call_mcp_tool.await_count == 2

    @pytest.mark.asyncio
    async def test_the_task_is_marked_started_before_any_model_call(self, agent):
        agent._router.complete = AsyncMock(return_value=_msg(content="x"))
        task, _ = await agent.execute_task(_task(), _state())
        assert task.started_at is not None
        assert task.completed_at is not None


class TestCompileOutput:
    @pytest.mark.asyncio
    async def test_every_task_result_reaches_the_synthesis_prompt(self, agent):
        """A compile step that drops a task silently produces a confident
        answer missing a third of its evidence."""
        agent._router.complete = AsyncMock(return_value=_msg(content="final answer"))
        tasks = [
            _task("t1").mark_started().mark_completed("first finding"),
            _task("t2").mark_started().mark_failed("second blew up"),
        ]
        out = await agent.compile_output(tasks, _state())

        assert out == "final answer"
        prompt = agent._router.complete.await_args.kwargs["messages"][1]["content"]
        assert "first finding" in prompt
        assert "second blew up" in prompt, "a failed task must still inform the synthesis"

    @pytest.mark.asyncio
    async def test_an_empty_response_becomes_an_empty_string_not_none(self, agent):
        agent._router.complete = AsyncMock(return_value=_msg(content=None))
        assert await agent.compile_output([], _state()) == ""


class TestToolSchemasReachTheModel:
    """The executor fetched its MCP tool schemas into a local variable and
    never passed them to the model.

    Under a mocked router the loop still "worked" - the mock returns
    whatever it was told to, tool calls included - so every existing test
    passed. Against a real provider the model is offered no tools,
    `response.tool_calls` is always empty, and the agent degrades to
    single-shot prose while the MCP server, the sandbox and the whole tool
    layer sit unreachable behind it.

    Ruff found it, as an unused variable. Nothing else did.
    """

    @pytest.mark.asyncio
    async def test_the_schemas_are_passed_on_every_round(self, agent):
        schemas = [{"type": "function", "function": {"name": "search_knowledge"}}]
        agent._mcp.get_tool_schemas = AsyncMock(return_value=schemas)
        agent._router.complete = AsyncMock(
            side_effect=[_msg(content=None, tool_calls=[_tool_call()]), _msg(content="done")]
        )
        agent._call_mcp_tool = AsyncMock(return_value={"ok": True})

        await agent.execute_task(_task(), _state())

        assert agent._router.complete.await_count == 2
        for call in agent._router.complete.await_args_list:
            assert call.kwargs.get("tools") == schemas, "a round was run with no tools offered"

    @pytest.mark.asyncio
    async def test_a_server_offering_no_tools_still_completes_the_task(self, agent):
        """An empty tool list is a valid state - the model answers directly."""
        agent._mcp.get_tool_schemas = AsyncMock(return_value=[])
        agent._router.complete = AsyncMock(return_value=_msg(content="answered directly"))
        task, _ = await agent.execute_task(_task(), _state())
        assert task.result == "answered directly"
