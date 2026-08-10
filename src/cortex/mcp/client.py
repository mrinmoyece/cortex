"""
Cortex MCP client.

Used by the ExecutorAgent to call tools exposed by the MCP server.
Maintains a persistent connection to avoid per-call handshake overhead.

In production, this connects to the Cortex MCP server over SSE transport.
Locally, it can also call tool functions in-process for faster dev iteration.
"""

from __future__ import annotations

import contextlib
import inspect
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from cortex.config import settings
from cortex.exceptions import MCPPermissionError, MCPToolArgumentError, MCPToolError
from cortex.logging_config import get_logger
from cortex.mcp.catalog import TOOL_NAMES

if TYPE_CHECKING:
    from cortex.mcp.server import Principal

logger = get_logger(__name__)


class MCPClient:
    """
    Thin client for invoking Cortex MCP tools.
    Routes calls either in-process (local) or over SSE (production).
    """

    def __init__(self) -> None:
        # In-process tool registry — populated on first use
        self._tools: dict[str, Any] = {}
        self._schemas: list[dict[str, Any]] | None = None

    def _register_local_tools(self) -> None:
        """Register tool functions directly for in-process calls."""
        from cortex.mcp.server import (
            execute_code,
            query_data,
            query_memory,
            search_knowledge,
            synthesise,
        )

        self._tools = {
            "search_knowledge": search_knowledge,
            "query_memory": query_memory,
            "execute_code": execute_code,
            # `query_data` is a real MCP tool and was advertised to the
            # planner, but was missing from both the registry and the
            # schemas - so the model was told to plan around a tool that
            # could never be called.
            "query_data": query_data,
            "synthesise": synthesise,
        }
        # The registry, the server and the HTTP allowlist have to name the
        # same tools. When they drifted, the symptom was a 404 for a working
        # tool and a 500 for a near-miss.
        missing = sorted(TOOL_NAMES - set(self._tools))
        if missing:
            raise MCPToolError(
                f"MCP tool registry is missing catalogued tools: {missing}",
                details={"registered": sorted(self._tools)},
            )

    @staticmethod
    @contextlib.contextmanager
    def _bound(principal: Principal | None) -> Iterator[None]:
        """Bind `principal` for the call, if one was supplied."""
        if principal is None:
            yield
            return
        from cortex.mcp.server import use_principal

        with use_principal(principal):
            yield

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        principal: Principal | None = None,
    ) -> Any:
        """Invoke an MCP tool by name with the given arguments.

        `principal` is the authenticated identity the call runs as. It is
        passed here rather than in `arguments` on purpose: `arguments` comes
        from the model or from an HTTP body, and identity must not be
        something either of those can choose.
        """
        if not self._tools:
            self._register_local_tools()

        tool_fn = self._tools.get(tool_name)
        if not tool_fn:
            raise MCPToolError(
                f"Tool '{tool_name}' not found",
                details={"available": sorted(self._tools)},
            )

        # Bind the arguments against the signature *before* calling, so a
        # caller passing `user_id=` to a tool that has no such parameter is
        # told that, rather than getting a generic tool failure. Doing it
        # afterwards is impossible to get right: a `TypeError` raised inside
        # the tool body is indistinguishable from one raised by the call
        # itself once it has propagated.
        try:
            inspect.signature(tool_fn).bind(**arguments)
        except TypeError as exc:
            raise MCPToolArgumentError(
                f"Tool '{tool_name}' cannot accept these arguments: {exc}",
                details={"tool": tool_name, "supplied": sorted(arguments)},
            ) from exc

        try:
            with self._bound(principal):
                result = await tool_fn(**arguments)
            logger.debug("mcp.tool_success", tool=tool_name)
            return result
        except MCPToolError:
            raise
        except PermissionError as exc:
            # A tool refusing for want of an identity is an authorisation
            # failure, not an internal error, and flattening it into one
            # loses the only thing the caller can act on.
            raise MCPPermissionError(str(exc), details={"tool": tool_name}) from exc
        except Exception as exc:
            raise MCPToolError(
                f"Tool '{tool_name}' raised {type(exc).__name__}: {exc}",
                details={
                    "tool": tool_name,
                    "arguments": {k: str(v)[:100] for k, v in arguments.items()},
                },
            ) from exc

    async def get_tool_schemas(self) -> list[dict[str, Any]]:
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
                    "description": (
                        "Retrieve memories belonging to the calling user. The "
                        "user is taken from the authenticated session and "
                        "cannot be specified."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "memory_type": {
                                "type": "string",
                                "enum": ["all", "episodic", "semantic"],
                            },
                            "limit": {"type": "integer", "default": 5},
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "query_data",
                    "description": (
                        "Answer a question against a configured read-only SQL "
                        "database. Only SELECT statements are executed."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "natural_language_query": {"type": "string"},
                            "database_alias": {"type": "string", "default": "default"},
                            "max_rows": {"type": "integer", "default": 100},
                        },
                        "required": ["natural_language_query"],
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

        # Only advertise code execution when it is actually enabled.
        # Offering a tool that always returns "disabled" wastes a planning
        # step and teaches the model to plan around a capability that
        # does not exist here.
        if settings.code_execution_enabled:
            self._schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": "execute_code",
                        "description": (
                            "Execute Python in a subprocess. NOT sandboxed - it runs "
                            "with the platform's own privileges."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "code": {
                                    "type": "string",
                                    "description": "Python code to execute",
                                },
                                "timeout_seconds": {"type": "integer", "default": 30},
                            },
                            "required": ["code"],
                        },
                    },
                }
            )
        return self._schemas


_client: MCPClient | None = None


def get_mcp_client() -> MCPClient:
    global _client
    if _client is None:
        _client = MCPClient()
    return _client
