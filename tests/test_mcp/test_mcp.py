"""Tests for the Cortex MCP server tools and client."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestSearchKnowledgeTool:
    @pytest.mark.asyncio
    async def test_returns_list_of_chunks(self):
        from cortex.mcp.server import search_knowledge

        mock_chunk = MagicMock()
        mock_chunk.content = "Cortex revenue grew 20% in Q3"
        mock_chunk.score = 0.92
        mock_chunk.source = "reranked"
        mock_chunk.metadata = {"source": "annual-report", "chunk_index": 0}

        with patch("cortex.mcp.server._get_rag") as mock_rag_factory:
            mock_rag = AsyncMock()
            mock_rag.retrieve = AsyncMock(return_value=[mock_chunk])
            mock_rag_factory.return_value = mock_rag

            results = await search_knowledge(query="revenue Q3", top_k=1)

        assert isinstance(results, list)
        assert len(results) == 1
        assert results[0]["content"] == "Cortex revenue grew 20% in Q3"
        assert results[0]["score"] == 0.92

    @pytest.mark.asyncio
    async def test_top_k_capped_at_20(self):
        from cortex.mcp.server import search_knowledge

        with patch("cortex.mcp.server._get_rag") as mock_rag_factory:
            mock_rag = AsyncMock()
            mock_rag.retrieve = AsyncMock(return_value=[])
            mock_rag_factory.return_value = mock_rag

            await search_knowledge(query="test", top_k=999)

            call_kwargs = mock_rag.retrieve.call_args.kwargs
            assert call_kwargs["top_k"] <= 20

    @pytest.mark.asyncio
    async def test_top_k_floor_at_1(self):
        from cortex.mcp.server import search_knowledge

        with patch("cortex.mcp.server._get_rag") as mock_rag_factory:
            mock_rag = AsyncMock()
            mock_rag.retrieve = AsyncMock(return_value=[])
            mock_rag_factory.return_value = mock_rag

            await search_knowledge(query="test", top_k=-5)

            call_kwargs = mock_rag.retrieve.call_args.kwargs
            assert call_kwargs["top_k"] >= 1


class TestExecuteCodeTool:
    @pytest.mark.asyncio
    async def test_executes_simple_python(self):
        from cortex.mcp.server import execute_code

        result = await execute_code(code="print('hello world')", language="python")
        assert result["exit_code"] == 0
        assert "hello world" in result["stdout"]
        assert result["timed_out"] is False

    @pytest.mark.asyncio
    async def test_captures_stdout_and_stderr(self):
        """The original version of this test used
        `import sys; print('err', file=sys.stderr)` - and `import sys` is on
        the block list, by requirement. So the tool correctly refused the
        code, returned `{"error": ..., "exit_code": -1}`, and the test died
        on `KeyError: 'stdout'`. The test was asserting against a security
        control it was simultaneously tripping.

        stderr is produced here by an uncaught exception instead, which
        needs no forbidden import."""
        from cortex.mcp.server import execute_code

        result = await execute_code(code="print('out')\nraise ValueError('err')")
        assert "out" in result["stdout"]
        assert "err" in result["stderr"]
        assert result["exit_code"] != 0
        assert result["timed_out"] is False

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "code",
        [
            "import os",
            "import subprocess",
            "import sys",
            "exec('print(1)')",
            "eval('1+1')",
        ],
    )
    async def test_dangerous_constructs_are_blocked(self, code):
        """The block list is a security requirement, so it gets a test that
        fails if any entry is quietly dropped."""
        from cortex.mcp.server import execute_code

        result = await execute_code(code=code)
        assert result["exit_code"] == -1
        assert "Blocked" in result.get("error", "")
        assert "stdout" not in result

    @pytest.mark.asyncio
    async def test_non_python_rejected(self):
        from cortex.mcp.server import execute_code

        result = await execute_code(code="console.log('hi')", language="javascript")
        assert "error" in result
        assert result["exit_code"] == -1

    @pytest.mark.asyncio
    async def test_os_import_blocked(self):
        from cortex.mcp.server import execute_code

        result = await execute_code(code="import os; print(os.getcwd())")
        assert result["exit_code"] == -1
        assert "Blocked" in result["error"]

    @pytest.mark.asyncio
    async def test_subprocess_blocked(self):
        from cortex.mcp.server import execute_code

        result = await execute_code(code="import subprocess; subprocess.run(['ls'])")
        assert result["exit_code"] == -1

    @pytest.mark.asyncio
    async def test_handles_syntax_error(self):
        from cortex.mcp.server import execute_code

        result = await execute_code(code="def broken(: pass")
        assert result["exit_code"] != 0

    @pytest.mark.asyncio
    async def test_timeout_enforced(self):
        from cortex.mcp.server import execute_code

        code = "import time; time.sleep(100)"
        result = await execute_code(code=code, timeout_seconds=1)
        assert result["timed_out"] is True

    @pytest.mark.asyncio
    async def test_math_computation(self):
        from cortex.mcp.server import execute_code

        code = "result = sum(range(1, 101)); print(result)"
        result = await execute_code(code=code)
        assert result["exit_code"] == 0
        assert "5050" in result["stdout"]


class TestMCPClient:
    @pytest.mark.asyncio
    async def test_tool_schemas_have_required_fields(self):
        from cortex.mcp.client import MCPClient

        client = MCPClient()
        schemas = await client.get_tool_schemas()
        assert len(schemas) >= 3
        for schema in schemas:
            assert schema["type"] == "function"
            assert "name" in schema["function"]
            assert "description" in schema["function"]
            assert "parameters" in schema["function"]

    @pytest.mark.asyncio
    async def test_unknown_tool_raises_mcp_error(self):
        from cortex.exceptions import MCPToolError
        from cortex.mcp.client import MCPClient

        client = MCPClient()
        with pytest.raises(MCPToolError, match="not found"):
            await client.call_tool("nonexistent_tool", {})

    @pytest.mark.asyncio
    async def test_tool_error_wrapped_as_mcp_error(self):
        from cortex.exceptions import MCPToolError
        from cortex.mcp.client import MCPClient

        client = MCPClient()

        # Register a tool that raises
        async def bad_tool(**kwargs):
            raise ValueError("tool exploded")

        client._tools = {"bad_tool": bad_tool}

        with pytest.raises(MCPToolError, match="bad_tool"):
            await client.call_tool("bad_tool", {})
