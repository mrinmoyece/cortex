"""Which cost ledger a tool's own LLM calls bill to.

`query_data` and `synthesise` call the router themselves. Both minted a fresh
UUID per call, so a tool invoked from an agent run billed itself to a run id
nothing else knew about: the run's budget gate never saw that spend, and the
`total_cost_usd` reported back to the user under-counted it.

Standalone MCP calls still get an isolated ledger - unrelated callers must not
consume each other's budget.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cortex.graph.state import CortexState, RunStatus
from cortex.mcp.client import MCPClient
from cortex.mcp.server import Principal, current_run_id, use_run_id


def _state(run_id: str = "run-1") -> CortexState:
    return CortexState(
        run_id=run_id,
        session_id="sess-1",
        user_id="alice",
        tenant_id="acme",
        user_goal="summarise the report",
        status=RunStatus.EXECUTING,
    )


def _completion(content: str) -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=content))]
    return response


class TestRunIdBinding:
    def test_nothing_bound_yields_a_fresh_ledger_per_call(self):
        """A standalone MCP client has no enclosing run. A shared constant
        would make unrelated callers consume each other's budget."""
        assert current_run_id() != current_run_id()

    def test_a_bound_run_id_is_returned(self):
        with use_run_id("run-42"):
            assert current_run_id() == "run-42"

    def test_the_binding_is_released_after_the_call(self):
        with use_run_id("run-42"):
            pass
        assert current_run_id() != "run-42"

    def test_bindings_nest(self):
        with use_run_id("outer"):
            with use_run_id("inner"):
                assert current_run_id() == "inner"
            assert current_run_id() == "outer"


class TestTheClientBindsTheSuppliedRunId:
    @pytest.mark.asyncio
    async def test_the_tool_sees_the_run_id_passed_to_call_tool(self):
        seen: list[str] = []

        async def fake_tool(query: str) -> dict[str, str]:
            seen.append(current_run_id())
            return {"ok": query}

        client = MCPClient()
        client._tools = {"synthesise": fake_tool}
        await client.call_tool("synthesise", {"query": "q"}, run_id="run-7")

        assert seen == ["run-7"]

    @pytest.mark.asyncio
    async def test_omitting_it_leaves_the_tool_on_its_own_ledger(self):
        """Compatibility for standalone MCP callers: `POST /api/v1/mcp/call`
        and stdio clients have no run to bill to."""
        seen: list[str] = []

        async def fake_tool(query: str) -> dict[str, str]:
            seen.append(current_run_id())
            return {"ok": query}

        client = MCPClient()
        client._tools = {"synthesise": fake_tool}
        await client.call_tool("synthesise", {"query": "q"})
        await client.call_tool("synthesise", {"query": "q"})

        assert len(set(seen)) == 2, "each unbound call gets its own ledger"

    @pytest.mark.asyncio
    async def test_identity_and_ledger_are_bound_together(self):
        seen: list[tuple[Principal, str]] = []

        async def fake_tool(query: str) -> dict[str, str]:
            from cortex.mcp.server import current_principal

            seen.append((current_principal(), current_run_id()))
            return {"ok": query}

        client = MCPClient()
        client._tools = {"synthesise": fake_tool}
        await client.call_tool(
            "synthesise", {"query": "q"}, principal=Principal("alice", "acme"), run_id="run-7"
        )

        assert seen == [(Principal("alice", "acme"), "run-7")]


class TestTheExecutorBillsToolCallsToItsOwnRun:
    @pytest.mark.asyncio
    async def test_the_run_id_reaches_the_client(self):
        from cortex.agents.executor import ExecutorAgent

        with (
            patch("cortex.agents.executor.get_router"),
            patch("cortex.agents.executor.get_mcp_client") as get_client,
        ):
            client = MagicMock()
            client.call_tool = AsyncMock(return_value={"result": "ok"})
            get_client.return_value = client
            agent = ExecutorAgent()
            await agent._call_mcp_tool("synthesise", {"content": "c"}, _state("run-99"))

        assert client.call_tool.await_args.kwargs["run_id"] == "run-99"


class TestToolLlmCallsUseTheBoundLedger:
    @pytest.mark.asyncio
    async def test_synthesise_bills_the_enclosing_run(self):
        from cortex.mcp.server import synthesise

        router = MagicMock()
        router.complete = AsyncMock(return_value=_completion("a summary"))
        with patch("cortex.llm.router.get_router", return_value=router), use_run_id("run-5"):
            result = await synthesise(content="c", instruction="summarise")

        assert result["result"] == "a summary"
        assert router.complete.await_args.kwargs["run_id"] == "run-5"

    @pytest.mark.asyncio
    async def test_synthesise_standalone_still_gets_its_own_ledger(self):
        from cortex.mcp.server import synthesise

        router = MagicMock()
        router.complete = AsyncMock(return_value=_completion("a summary"))
        with patch("cortex.llm.router.get_router", return_value=router):
            await synthesise(content="c", instruction="summarise")
            first = router.complete.await_args.kwargs["run_id"]
            await synthesise(content="c", instruction="summarise")
            second = router.complete.await_args.kwargs["run_id"]

        assert first != second

    @pytest.mark.asyncio
    async def test_natural_language_to_sql_bills_the_enclosing_run(self):
        from cortex.mcp.server import _to_sql

        router = MagicMock()
        router.complete = AsyncMock(return_value=_completion("SELECT 1"))
        with patch("cortex.llm.router.get_router", return_value=router), use_run_id("run-5"):
            sql = await _to_sql("how many rows", "t(a)")

        assert sql == "SELECT 1"
        assert router.complete.await_args.kwargs["run_id"] == "run-5"

    @pytest.mark.asyncio
    async def test_a_tool_call_that_blows_the_budget_is_not_hidden_by_a_private_ledger(self):
        """The point of binding: the router's budget gate is per-run, so a
        run that has already spent its limit must be refused inside the tool
        too, not handed a clean ledger to spend against."""
        from cortex.exceptions import LLMBudgetExceededError
        from cortex.mcp.server import synthesise

        spent: set[str] = {"run-5"}
        router = MagicMock()

        async def complete(**kwargs):
            if kwargs["run_id"] in spent:
                raise LLMBudgetExceededError("run out of budget")
            return _completion("a summary")

        router.complete = AsyncMock(side_effect=complete)
        with patch("cortex.llm.router.get_router", return_value=router), use_run_id("run-5"):
            with pytest.raises(LLMBudgetExceededError):
                await synthesise(content="c", instruction="summarise")
