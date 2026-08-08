"""Tests for the Cortex FastAPI application."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from cortex.api.auth import create_access_token


@pytest.fixture
def auth_token() -> str:
    return create_access_token(user_id="test-user", tenant="test")


@pytest.fixture
def auth_headers(auth_token: str) -> dict:
    return {"Authorization": f"Bearer {auth_token}"}


@pytest.fixture
async def client():
    """Async test client for the Cortex FastAPI app."""
    from cortex.api.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health_returns_ok(self, client):
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["service"] == "cortex"

    @pytest.mark.asyncio
    async def test_health_no_auth_required(self, client):
        response = await client.get("/health")
        assert response.status_code == 200


class TestMetricsEndpoint:
    @pytest.mark.asyncio
    async def test_metrics_returns_prometheus_format(self, client):
        response = await client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
        assert "cortex_" in response.text


class TestRunEndpoints:
    @pytest.mark.asyncio
    async def test_create_run_requires_auth(self, client):
        response = await client.post(
            "/api/v1/runs",
            json={"goal": "test goal"},
        )
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_create_run_returns_202(self, client, auth_headers):
        response = await client.post(
            "/api/v1/runs",
            headers=auth_headers,
            json={"goal": "Summarise the quarterly report"},
        )
        assert response.status_code == 202
        data = response.json()
        assert "run_id" in data
        assert data["status"] == "pending"

    @pytest.mark.asyncio
    async def test_get_run_404_for_unknown(self, client, auth_headers):
        response = await client.get("/api/v1/runs/nonexistent-id", headers=auth_headers)
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_get_run_403_for_other_user(self, client, auth_headers):
        # Create a run as user A
        create_resp = await client.post(
            "/api/v1/runs",
            headers=auth_headers,
            json={"goal": "test goal"},
        )
        run_id = create_resp.json()["run_id"]

        # Try to read it as user B
        other_token = create_access_token(user_id="different-user")
        other_headers = {"Authorization": f"Bearer {other_token}"}
        get_resp = await client.get(f"/api/v1/runs/{run_id}", headers=other_headers)
        assert get_resp.status_code == 403

    @pytest.mark.asyncio
    async def test_create_run_goal_cannot_be_empty(self, client, auth_headers):
        response = await client.post(
            "/api/v1/runs",
            headers=auth_headers,
            json={"goal": ""},
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_run_status_structure(self, client, auth_headers):
        create_resp = await client.post(
            "/api/v1/runs",
            headers=auth_headers,
            json={"goal": "test"},
        )
        run_id = create_resp.json()["run_id"]
        get_resp = await client.get(f"/api/v1/runs/{run_id}", headers=auth_headers)
        assert get_resp.status_code == 200
        data = get_resp.json()
        required_fields = {"run_id", "status", "total_cost_usd", "iteration_count", "task_count"}
        assert required_fields.issubset(data.keys())


class TestIngestEndpoint:
    @pytest.mark.asyncio
    async def test_ingest_requires_auth(self, client):
        response = await client.post("/api/v1/ingest", json={"text": "test document content here"})
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_ingest_success(self, client, auth_headers):
        with patch("cortex.api.main.ingest_document", new=AsyncMock(return_value=3)):
            response = await client.post(
                "/api/v1/ingest",
                headers=auth_headers,
                json={"text": "This is a test document with enough content to be meaningful."},
            )
        assert response.status_code == 201
        data = response.json()
        assert data["chunks_created"] == 3
        assert data["status"] == "ingested"

    @pytest.mark.asyncio
    async def test_ingest_text_too_short_rejected(self, client, auth_headers):
        response = await client.post(
            "/api/v1/ingest",
            headers=auth_headers,
            json={"text": "short"},
        )
        assert response.status_code == 422


class TestAuthSystem:
    def test_create_token_valid_for_user(self):
        token = create_access_token(user_id="user-123")
        from jose import jwt

        from cortex.config import get_settings

        settings = get_settings()
        payload = jwt.decode(token, settings.secret_key.get_secret_value(), algorithms=["HS256"])
        assert payload["sub"] == "user-123"

    @pytest.mark.asyncio
    async def test_invalid_token_rejected(self, client):
        bad_headers = {"Authorization": "Bearer not-a-valid-token"}
        response = await client.post(
            "/api/v1/runs",
            headers=bad_headers,
            json={"goal": "test"},
        )
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_missing_auth_header_rejected(self, client):
        response = await client.post("/api/v1/runs", json={"goal": "test"})
        assert response.status_code == 401
