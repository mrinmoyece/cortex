"""The three places that name MCP tools must agree.

The server registers them, the client dispatches them, and configuration
decides which are reachable over HTTP. Each was checked against something,
and none against each other, so the shipped default allowlist named
`summarize_document` - a tool that has never existed.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from cortex.config import Settings
from cortex.mcp.catalog import DEFAULT_HTTP_ALLOWLIST, PRINCIPAL_REQUIRED, TOOL_NAMES
from cortex.mcp.client import MCPClient


class TestCatalogMatchesReality:
    def test_the_client_registry_is_exactly_the_catalogue(self):
        client = MCPClient()
        client._register_local_tools()
        assert set(client._tools) == set(TOOL_NAMES)

    def test_every_catalogued_tool_exists_on_the_server(self):
        import cortex.mcp.server as server

        for name in TOOL_NAMES:
            assert callable(getattr(server, name, None)), f"{name} is not defined on the server"

    def test_advertised_schemas_are_catalogued_tools(self):
        """A schema for a tool the registry cannot dispatch teaches the model
        to plan around a call that will always fail."""
        import asyncio

        schemas = asyncio.run(MCPClient().get_tool_schemas())
        advertised = {s["function"]["name"] for s in schemas}
        assert advertised <= set(TOOL_NAMES)

    def test_the_principal_required_set_names_real_tools(self):
        assert PRINCIPAL_REQUIRED <= TOOL_NAMES


class TestDefaultAllowlist:
    def test_the_default_names_only_real_tools(self):
        """The regression: `summarize_document` was in this list, and the real
        tool is `synthesise`. Every call to it 404'd."""
        assert set(DEFAULT_HTTP_ALLOWLIST) <= set(TOOL_NAMES)
        assert "summarize_document" not in DEFAULT_HTTP_ALLOWLIST

    def test_synthesise_is_reachable_over_http(self):
        assert "synthesise" in Settings().mcp_http_tool_allowlist

    def test_execute_code_is_not_published_by_default(self):
        """Enabling it for a local MCP client and publishing it on an HTTP
        endpoint are different risk decisions."""
        assert "execute_code" not in DEFAULT_HTTP_ALLOWLIST


class TestAllowlistValidation:
    def test_an_unknown_tool_is_rejected_at_config_load(self):
        with pytest.raises(ValidationError, match="unknown tools"):
            Settings(mcp_http_tool_allowlist=["search_knowledge", "summarize_document"])

    def test_the_error_names_the_offender_and_the_alternatives(self):
        with pytest.raises(ValidationError) as excinfo:
            Settings(mcp_http_tool_allowlist=["nope"])
        message = str(excinfo.value)
        assert "nope" in message
        assert "synthesise" in message

    def test_a_valid_subset_is_accepted(self):
        settings = Settings(mcp_http_tool_allowlist=["search_knowledge"])
        assert settings.mcp_http_tool_allowlist == ["search_knowledge"]

    def test_an_empty_allowlist_is_allowed(self):
        """Refusing every tool over HTTP is a legitimate deployment choice."""
        assert Settings(mcp_http_tool_allowlist=[]).mcp_http_tool_allowlist == []
