"""The canonical set of MCP tool names, and what each one needs.

This module exists so three places cannot disagree: the MCP server that
registers the tools, the client registry that dispatches them in-process,
and the HTTP allowlist in configuration that decides which are reachable
over the API.

The default allowlist used to name `summarize_document`, which does not
exist and never has - the real tool is `synthesise`. Nothing caught it,
because the allowlist was checked against configuration and the registry
was checked against the server, and no one checked the two against each
other. The result was a 404 for a tool that works and, had the name been
one letter off instead of entirely wrong, a 500 from a dispatch table miss.

It deliberately imports nothing from `cortex`, so `cortex.config` can
validate against it without an import cycle.
"""

from __future__ import annotations

from typing import Final

#: Every tool registered on the Cortex MCP server.
TOOL_NAMES: Final[frozenset[str]] = frozenset(
    {
        "search_knowledge",
        "query_memory",
        "execute_code",
        "query_data",
        "synthesise",
    }
)

#: Tools that read or write data belonging to a specific user. These refuse
#: to run without an authenticated principal - see
#: `cortex.mcp.server.current_principal`.
PRINCIPAL_REQUIRED: Final[frozenset[str]] = frozenset({"query_memory"})

#: Tools reachable over `POST /api/v1/mcp/call` unless configuration says
#: otherwise. `execute_code` is absent on purpose: it is not a sandbox, and
#: publishing it on an authenticated-but-public HTTP endpoint is a different
#: risk decision from enabling it for a local MCP client.
DEFAULT_HTTP_ALLOWLIST: Final[tuple[str, ...]] = (
    "search_knowledge",
    "query_memory",
    "query_data",
    "synthesise",
)
