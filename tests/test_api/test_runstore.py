"""The API's bounded state and the endpoints that guard it.

Both things tested here were unbounded or unguarded until an audit asked
what happens on the ten-thousandth request rather than the first.
"""

from __future__ import annotations

import time

import pytest
from httpx import ASGITransport, AsyncClient

from cortex.api.main import _RunStore, app
from cortex.graph.state import CortexState, RunStatus


@pytest.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


def _state(run_id: str) -> CortexState:
    return CortexState(
        run_id=run_id,
        session_id=run_id,
        user_id="u",
        user_goal="anything",
        status=RunStatus.PENDING,
    )


class TestRunStoreIsBounded:
    """`_runs` was a plain dict that nothing ever removed from. Every run the
    process had ever accepted stayed resident for its lifetime - a memory
    leak whose trigger is an unauthenticated-shaped request loop."""

    def test_entries_beyond_the_cap_are_evicted_oldest_first(self, override_settings):
        override_settings(api_max_tracked_runs=3, api_run_retention_seconds=3600)
        store = _RunStore()

        for i in range(5):
            store.put(f"run-{i}", _state(f"run-{i}"))

        assert len(store) == 3
        assert store.get("run-0") is None
        assert store.get("run-1") is None
        assert store.get("run-4") is not None

    def test_reading_a_run_keeps_it_alive(self, override_settings):
        """LRU, not FIFO: a run a client is actively polling must not be the
        one thrown away."""
        override_settings(api_max_tracked_runs=2, api_run_retention_seconds=3600)
        store = _RunStore()
        store.put("old", _state("old"))
        store.put("new", _state("new"))

        assert store.get("old") is not None
        store.put("newest", _state("newest"))

        assert store.get("old") is not None
        assert store.get("new") is None

    def test_an_expired_run_is_not_served(self, override_settings):
        override_settings(api_max_tracked_runs=100, api_run_retention_seconds=0.01)
        store = _RunStore()
        store.put("stale", _state("stale"))
        time.sleep(0.02)
        assert store.get("stale") is None

    def test_a_fresh_run_is_served(self, override_settings):
        override_settings(api_max_tracked_runs=100, api_run_retention_seconds=3600)
        store = _RunStore()
        store.put("fresh", _state("fresh"))
        assert store.get("fresh") is not None


class TestMetricsEndpointAuth:
    """Cortex metrics carry model names, spend totals and run volumes, and
    they were served unauthenticated on the public API port."""

    @pytest.mark.asyncio
    async def test_metrics_is_open_when_no_token_is_configured(self, client, override_settings):
        override_settings(metrics_token=None)
        response = await client.get("/metrics")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_a_configured_token_is_required(self, client, override_settings):
        from pydantic import SecretStr

        override_settings(metrics_token=SecretStr("scrape-me"))
        response = await client.get("/metrics")
        assert response.status_code == 401
        assert response.headers["WWW-Authenticate"] == "Bearer"

    @pytest.mark.asyncio
    async def test_the_right_token_is_accepted(self, client, override_settings):
        from pydantic import SecretStr

        override_settings(metrics_token=SecretStr("scrape-me"))
        response = await client.get("/metrics", headers={"Authorization": "Bearer scrape-me"})
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_a_wrong_token_is_rejected(self, client, override_settings):
        from pydantic import SecretStr

        override_settings(metrics_token=SecretStr("scrape-me"))
        response = await client.get("/metrics", headers={"Authorization": "Bearer guess"})
        assert response.status_code == 401


class TestMetricsStaysExemptFromRateLimiting:
    def test_scrape_paths_are_not_metered(self):
        """A throttled /metrics makes a busy service look unobservable, and a
        throttled /health makes Kubernetes restart a pod that was merely
        busy."""
        from cortex.api.ratelimit import RateLimitMiddleware

        assert "/metrics" in RateLimitMiddleware.EXEMPT_PATHS
        assert "/health" in RateLimitMiddleware.EXEMPT_PATHS


class TestRoutesAreRegistered:
    def test_the_documented_endpoints_exist(self):
        paths = {route.path for route in app.routes}  # type: ignore[attr-defined]
        for expected in (
            "/health",
            "/metrics",
            "/api/v1/runs",
            "/api/v1/runs/{run_id}",
            "/api/v1/ingest",
            "/api/v1/mcp/call",
        ):
            assert expected in paths, f"{expected} missing from {sorted(paths)}"


class TestMCPCallEndpointStatusMapping:
    """`POST /api/v1/mcp/call` had one visible failure mode: 500. The client
    flattened every exception into `MCPToolError`, so the endpoint's own 422
    branch was unreachable code and a caller who mistyped an argument was told
    the server had broken."""

    @pytest.fixture
    def auth_headers(self) -> dict[str, str]:
        from cortex.api.auth import create_access_token

        return {"Authorization": f"Bearer {create_access_token('alice', tenant='acme')}"}

    @pytest.mark.asyncio
    async def test_a_tool_outside_the_allowlist_is_404(self, client, auth_headers):
        response = await client.post(
            "/api/v1/mcp/call",
            json={"tool": "execute_code", "arguments": {"code": "print(1)"}},
            headers=auth_headers,
        )
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_the_renamed_tool_is_reachable(self, client, auth_headers):
        """The allowlist named `summarize_document`, which does not exist, so
        the real tool `synthesise` was refused with a 404."""
        from unittest.mock import patch

        async def fake_synthesise(
            content: str, instruction: str, output_format: str = "markdown"
        ) -> dict[str, str]:
            return {"result": "ok"}

        with patch("cortex.mcp.server.synthesise", new=fake_synthesise):
            from cortex.mcp.client import get_mcp_client

            get_mcp_client()._tools = {}
            response = await client.post(
                "/api/v1/mcp/call",
                json={
                    "tool": "synthesise",
                    "arguments": {"content": "text", "instruction": "shorten"},
                },
                headers=auth_headers,
            )
            get_mcp_client()._tools = {}

        assert response.status_code == 200
        assert response.json()["tool"] == "synthesise"

    @pytest.mark.asyncio
    async def test_an_unknown_argument_is_422(self, client, auth_headers):
        response = await client.post(
            "/api/v1/mcp/call",
            json={"tool": "query_memory", "arguments": {"query": "q", "user_id": "victim"}},
            headers=auth_headers,
        )
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "MCP_TOOL_BAD_ARGUMENTS"

    @pytest.mark.asyncio
    async def test_a_missing_required_argument_is_422(self, client, auth_headers):
        response = await client.post(
            "/api/v1/mcp/call",
            json={"tool": "query_memory", "arguments": {}},
            headers=auth_headers,
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_an_unauthorised_tool_call_is_403(self, client, auth_headers):
        """A tool refusing for want of an identity is an authorisation
        failure. The caller is authenticated to the API; that is a different
        thing from being entitled to what the tool was asked for."""

        async def refuse(query: str, memory_type: str = "all", limit: int = 5) -> dict:
            raise PermissionError("No authenticated principal is bound")

        from cortex.mcp.client import get_mcp_client

        client_obj = get_mcp_client()
        client_obj._register_local_tools()
        original = client_obj._tools["query_memory"]
        client_obj._tools["query_memory"] = refuse
        try:
            response = await client.post(
                "/api/v1/mcp/call",
                json={"tool": "query_memory", "arguments": {"query": "q"}},
                headers=auth_headers,
            )
        finally:
            client_obj._tools["query_memory"] = original

        assert response.status_code == 403
        assert response.json()["error"]["code"] == "MCP_FORBIDDEN"

    @pytest.mark.asyncio
    async def test_a_tool_crash_is_still_500(self, client, auth_headers):
        """Classification must not turn every failure into a client error."""

        async def boom(query: str, memory_type: str = "all", limit: int = 5) -> dict:
            raise RuntimeError("qdrant is unreachable")

        from cortex.mcp.client import get_mcp_client

        client_obj = get_mcp_client()
        client_obj._register_local_tools()
        original = client_obj._tools["query_memory"]
        client_obj._tools["query_memory"] = boom
        try:
            response = await client.post(
                "/api/v1/mcp/call",
                json={"tool": "query_memory", "arguments": {"query": "q"}},
                headers=auth_headers,
            )
        finally:
            client_obj._tools["query_memory"] = original

        assert response.status_code == 500
        assert response.json()["error"]["code"] == "MCP_TOOL_FAILED"

    @pytest.mark.asyncio
    async def test_the_endpoint_still_requires_authentication(self, client):
        response = await client.post(
            "/api/v1/mcp/call", json={"tool": "query_memory", "arguments": {"query": "q"}}
        )
        assert response.status_code in (401, 403)
