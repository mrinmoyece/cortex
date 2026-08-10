"""Where the identity for a tool call comes from, on every path that has one.

Binding existed only on `POST /api/v1/mcp/call`. A graph run - the way tools
are actually used - reached `query_memory` with nothing bound and got a flat
refusal, so the fix for one hole left the main path broken.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cortex.exceptions import MCPPermissionError, MCPToolArgumentError, MCPToolError
from cortex.graph.state import CortexState, RunStatus
from cortex.mcp.client import MCPClient
from cortex.mcp.server import Principal, configure_default_principal, current_principal


def _state(user_id: str = "alice", tenant_id: str = "acme") -> CortexState:
    return CortexState(
        run_id="run-1",
        session_id="sess-1",
        user_id=user_id,
        tenant_id=tenant_id,
        user_goal="summarise the report",
        status=RunStatus.EXECUTING,
    )


@pytest.fixture(autouse=True)
def _no_default_principal():
    """Every test starts with nothing bound, and leaves nothing behind."""
    import cortex.mcp.server as server

    server._default_principal = None
    yield
    server._default_principal = None


class TestClientBindsTheSuppliedPrincipal:
    @pytest.mark.asyncio
    async def test_the_tool_sees_the_principal_passed_to_call_tool(self):
        seen: list[Principal] = []

        async def fake_tool(query: str) -> dict[str, str]:
            seen.append(current_principal())
            return {"ok": query}

        client = MCPClient()
        client._tools = {"query_memory": fake_tool}
        await client.call_tool("query_memory", {"query": "q"}, principal=Principal("alice", "acme"))

        assert seen == [Principal("alice", "acme")]

    @pytest.mark.asyncio
    async def test_the_binding_is_released_after_the_call(self):
        async def fake_tool(query: str) -> str:
            return query

        client = MCPClient()
        client._tools = {"query_memory": fake_tool}
        await client.call_tool("query_memory", {"query": "q"}, principal=Principal("alice"))

        with pytest.raises(PermissionError):
            current_principal()

    @pytest.mark.asyncio
    async def test_no_principal_means_no_binding(self):
        """Passing nothing must not quietly reuse whatever was bound last."""

        async def fake_tool(query: str) -> str:
            current_principal()
            return query

        client = MCPClient()
        client._tools = {"query_memory": fake_tool}
        with pytest.raises(MCPPermissionError):
            await client.call_tool("query_memory", {"query": "q"})


class TestExecutorBindsTheRunIdentity:
    @pytest.mark.asyncio
    async def test_the_principal_comes_from_state(self):
        from cortex.agents.executor import ExecutorAgent

        with (
            patch("cortex.agents.executor.get_router"),
            patch("cortex.agents.executor.get_mcp_client") as get_client,
        ):
            client = MagicMock()
            client.call_tool = AsyncMock(return_value={"episodic": []})
            get_client.return_value = client
            agent = ExecutorAgent()

            result = await agent._call_mcp_tool("query_memory", {"query": "q"}, _state())

        assert result["success"] is True
        assert client.call_tool.await_args.kwargs["principal"] == Principal("alice", "acme")

    @pytest.mark.asyncio
    async def test_a_tenant_scoped_run_does_not_reach_another_tenant(self):
        from cortex.agents.executor import ExecutorAgent

        with (
            patch("cortex.agents.executor.get_router"),
            patch("cortex.agents.executor.get_mcp_client") as get_client,
        ):
            client = MagicMock()
            client.call_tool = AsyncMock(return_value={})
            get_client.return_value = client
            agent = ExecutorAgent()
            await agent._call_mcp_tool("query_memory", {"query": "q"}, _state("bob", "globex"))

        assert client.call_tool.await_args.kwargs["principal"].tenant_id == "globex"

    @pytest.mark.asyncio
    async def test_the_model_cannot_override_the_identity(self):
        """`arguments` is model-written. If an injected instruction adds a
        `user_id`, the tool must reject it rather than honour it."""
        from cortex.agents.executor import ExecutorAgent

        with (
            patch("cortex.agents.executor.get_router"),
            patch("cortex.agents.executor.get_mcp_client") as get_client,
        ):
            get_client.return_value = MCPClient()
            agent = ExecutorAgent()
            result = await agent._call_mcp_tool(
                "query_memory", {"query": "q", "user_id": "victim"}, _state()
            )

        assert result["success"] is False
        assert "user_id" in result["error"]

    @pytest.mark.asyncio
    async def test_a_tool_failure_is_reported_not_raised(self):
        """One bad tool call must not abandon a task with rounds left."""
        from cortex.agents.executor import ExecutorAgent

        with (
            patch("cortex.agents.executor.get_router"),
            patch("cortex.agents.executor.get_mcp_client") as get_client,
        ):
            client = MagicMock()
            client.call_tool = AsyncMock(side_effect=MCPToolError("qdrant is down"))
            get_client.return_value = client
            agent = ExecutorAgent()
            result = await agent._call_mcp_tool("search_knowledge", {"query": "q"}, _state())

        assert result == {"success": False, "error": "qdrant is down"}


class TestStandaloneServerPrincipal:
    """A standalone MCP transport has no authentication. Rather than invent
    one, the operator may declare the single identity a stdio process acts
    as - and that declaration is refused on the network transports, where it
    would hand one user's memory to every caller."""

    def test_nothing_is_bound_by_default(self, override_settings):
        override_settings(mcp_principal_user_id=None)
        assert configure_default_principal("stdio") is None
        with pytest.raises(PermissionError):
            current_principal()

    def test_stdio_honours_an_explicitly_declared_identity(self, override_settings):
        override_settings(mcp_principal_user_id="desktop-user", mcp_principal_tenant_id="acme")
        assert configure_default_principal("stdio") == Principal("desktop-user", "acme")
        assert current_principal() == Principal("desktop-user", "acme")

    @pytest.mark.parametrize("transport", ["http", "sse"])
    def test_network_transports_refuse_a_declared_identity(self, override_settings, transport: str):
        override_settings(mcp_principal_user_id="desktop-user")
        assert configure_default_principal(transport) is None
        with pytest.raises(PermissionError):
            current_principal()

    def test_a_bound_call_still_wins_over_the_process_default(self, override_settings):
        """The default is a fallback for a single-user process, not an
        override - a request that carries an identity is never served under
        someone else's."""
        from cortex.mcp.server import use_principal

        override_settings(mcp_principal_user_id="desktop-user")
        configure_default_principal("stdio")
        with use_principal(Principal("real-caller", "globex")):
            assert current_principal() == Principal("real-caller", "globex")

    def test_the_refusal_explains_the_supported_options(self, override_settings):
        override_settings(mcp_principal_user_id=None)
        configure_default_principal("http")
        with pytest.raises(PermissionError) as excinfo:
            current_principal()
        message = str(excinfo.value)
        assert "/api/v1/mcp/call" in message
        assert "MCP_PRINCIPAL_USER_ID" in message


class TestArgumentAndPermissionClassification:
    """`MCPClient` wrapped every exception into `MCPToolError`, so the API's
    422 branch was unreachable and a mistyped argument returned 500."""

    @pytest.mark.asyncio
    async def test_an_unknown_argument_is_an_argument_error(self):
        client = MCPClient()
        with pytest.raises(MCPToolArgumentError):
            await client.call_tool("query_memory", {"query": "q", "user_id": "victim"})

    @pytest.mark.asyncio
    async def test_a_missing_required_argument_is_an_argument_error(self):
        client = MCPClient()
        with pytest.raises(MCPToolArgumentError):
            await client.call_tool("query_memory", {})

    @pytest.mark.asyncio
    async def test_the_argument_error_maps_to_422(self):
        from http import HTTPStatus

        assert MCPToolArgumentError("x").http_status == HTTPStatus.UNPROCESSABLE_ENTITY

    @pytest.mark.asyncio
    async def test_a_missing_principal_is_a_permission_error(self):
        client = MCPClient()
        memory = MagicMock()
        memory.episodic.retrieve_recent = AsyncMock(return_value=[])
        memory.semantic.retrieve = AsyncMock(return_value=[])
        with patch("cortex.mcp.server._get_memory", return_value=memory):
            with pytest.raises(MCPPermissionError):
                await client.call_tool("query_memory", {"query": "q"})

    def test_the_permission_error_maps_to_403(self):
        from http import HTTPStatus

        assert MCPPermissionError("x").http_status == HTTPStatus.FORBIDDEN

    @pytest.mark.asyncio
    async def test_a_typeerror_inside_the_tool_stays_a_tool_error(self):
        """The distinction has to survive the obvious confusion: a bug in the
        tool body is a 500, not a hint to the caller that they typed the
        arguments wrong."""

        async def buggy(query: str) -> None:
            raise TypeError("unsupported operand type(s) for +: 'int' and 'str'")

        client = MCPClient()
        client._tools = {"search_knowledge": buggy}
        with pytest.raises(MCPToolError) as excinfo:
            await client.call_tool("search_knowledge", {"query": "q"})
        assert not isinstance(excinfo.value, MCPToolArgumentError)

    @pytest.mark.asyncio
    async def test_an_unknown_tool_is_still_a_tool_error(self):
        client = MCPClient()
        with pytest.raises(MCPToolError):
            await client.call_tool("no_such_tool", {})
