"""
Cortex MCP client.

Used by the ExecutorAgent to call tools exposed by the MCP server.
Maintains a persistent connection to avoid per-call handshake overhead.

In production, this connects to the Cortex MCP server over SSE transport.
Locally, it can also call tool functions in-process for faster dev iteration.
"""

from __future__ import annotations

from typing import Any

from cortex.exceptions import MCPToolError
from cortex.logging_config import get_logger

logger = get_logger(__name__)


class MCPClient:
    """
    Thin client for invoking Cortex MCP tools.
    Routes calls either in-process (local) or over SSE (production).
    """

    def __init__(self) -> None:
        # In-process tool registry — populated on first use
        self._tools: dict[str, Any] = {}
        self._schemas: list[dict] | None = None

    def _register_local_tools(self) -> None:
        """Register tool functions directly for in-process calls."""
        from cortex.mcp.server import (
            execute_code,
            query_memory,
            search_knowledge,
            synthesise,
        )

        self._tools = {
            "search_knowledge": search_knowledge,
            "query_memory": query_memory,
            "execute_code": execute_code,
            "synthesise": synthesise,
        }

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Invoke an MCP tool by name with the given arguments."""
        if not self._tools:
            self._register_local_tools()

        tool_fn = self._tools.get(tool_name)
        if not tool_fn:
            raise MCPToolError(
                f"Tool '{tool_name}' not found",
                details={"available": list(self._tools.keys())},
            )

        try:
            result = await tool_fn(**arguments)
            logger.debug("mcp.tool_success", tool=tool_name)
            return result
        except MCPToolError:
            raise
        except Exception as exc:
            raise MCPToolError(
                f"Tool '{tool_name}' raised {type(exc).__name__}: {exc}",
                details={
                    "tool": tool_name,
                    "arguments": {k: str(v)[:100] for k, v in arguments.items()},
                },
            ) from exc

    async def get_tool_schemas(self) -> list[dict]:
        """
        Return OpenAI-compatible function schemas for all available tools.
        Used by the ExecutorAgent to populate the `tools` parameter in LLM calls.
        """
        if self._schemas is not None:
            return self._schemas

        self._schemas = [
            {
                "type": "function",
                "function": {
                    "name": "search_knowledge",
                    "description": "Search the knowledge base using hybrid retrieval",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Natural language search query",
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "Number of results (1-20)",
                                "default": 5,
                            },
                            "source_filter": {
                                "type": "string",
                                "description": "Filter by document source",
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "query_memory",
                    "description": "Retrieve relevant memories for a user",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "user_id": {"type": "string"},
                            "memory_type": {
                                "type": "string",
                                "enum": ["all", "episodic", "semantic"],
                            },
                            "limit": {"type": "integer", "default": 5},
                        },
                        "required": ["query", "user_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "execute_code",
                    "description": "Execute Python code in a sandboxed environment",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {"type": "string", "description": "Python code to execute"},
                            "timeout_seconds": {"type": "integer", "default": 30},
                        },
                        "required": ["code"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "synthesise",
                    "description": "Synthesise, summarise or transform content",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string"},
                            "instruction": {"type": "string"},
                            "output_format": {
                                "type": "string",
                                "enum": ["markdown", "json", "plain"],
                            },
                        },
                        "required": ["content", "instruction"],
                    },
                },
            },
        ]
        return self._schemas


_client: MCPClient | None = None


def get_mcp_client() -> MCPClient:
    global _client
    if _client is None:
        _client = MCPClient()
    return _client
