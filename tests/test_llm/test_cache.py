"""The semantic cache.

A cache is a correctness hazard dressed as an optimisation: too loose a
similarity threshold and one user's answer is served to another's question.
Every test here is about *when not to serve a hit*.
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from cortex.llm.cache import SemanticCache


def _hit(score=0.99, age_seconds=0, content="cached answer"):
    payload = {
        "created_at": time.time() - age_seconds,
        "response_json": json.dumps(
            {"choices": [{"message": {"role": "assistant", "content": content}}]}
        ),
    }
    return SimpleNamespace(score=score, payload=payload)


def _points(*hits):
    """`query_points` returns a response object, not a bare list.

    `AsyncQdrantClient.search` was removed in qdrant-client 1.13 and is gone
    in the pinned 1.19, so every cache read raised AttributeError against a
    real client while the tests - which mocked `.search` - stayed green.
    Mocking `query_points` is what keeps the test honest about the API that
    actually exists.
    """
    return SimpleNamespace(points=list(hits))


@pytest.fixture
def cache():
    c = SemanticCache()
    c._ensure_collection = AsyncMock(return_value=AsyncMock())
    c._embed = AsyncMock(return_value=[0.1] * 8)
    return c


def _get(cache, prompt="what is our refund policy", model="gpt-4o", scope="acme"):
    return cache.get(prompt, model=model, scope=scope)


class TestCacheReads:
    @pytest.mark.asyncio
    async def test_a_fresh_similar_hit_is_returned(self, cache):
        client = await cache._ensure_collection()
        client.query_points = AsyncMock(return_value=_points(_hit()))
        assert await _get(cache) is not None

    @pytest.mark.asyncio
    async def test_no_match_returns_none_rather_than_the_nearest_thing(self, cache):
        """Qdrant's score_threshold does the filtering; returning a
        best-effort nearest neighbour would answer one question with
        another's answer."""
        client = await cache._ensure_collection()
        client.query_points = AsyncMock(return_value=_points())
        assert await _get(cache, "something nobody has asked") is None

    @pytest.mark.asyncio
    async def test_an_expired_entry_is_not_served(self, cache):
        """Similarity says nothing about freshness. A 3-day-old answer to
        'what is our current pricing' is similar and wrong."""
        client = await cache._ensure_collection()
        client.query_points = AsyncMock(return_value=_points(_hit(age_seconds=48 * 3600)))
        assert await _get(cache, "pricing") is None

    @pytest.mark.asyncio
    async def test_an_entry_just_inside_the_ttl_is_still_served(self, cache):
        client = await cache._ensure_collection()
        client.query_points = AsyncMock(return_value=_points(_hit(age_seconds=60)))
        assert await _get(cache, "pricing") is not None


class TestCacheFailuresAreNonFatal:
    @pytest.mark.asyncio
    async def test_a_dead_vector_store_degrades_to_a_miss(self, cache):
        """The cache is an optimisation. If Qdrant is down the platform must
        get slower, not stop - so a failure here returns None and the real
        LLM call proceeds."""
        cache._ensure_collection = AsyncMock(side_effect=ConnectionError("qdrant is down"))
        assert await _get(cache, "anything") is None

    @pytest.mark.asyncio
    async def test_a_corrupt_payload_degrades_to_a_miss(self, cache):
        client = await cache._ensure_collection()
        bad = SimpleNamespace(
            score=0.99, payload={"created_at": time.time(), "response_json": "{{{"}
        )
        client.query_points = AsyncMock(return_value=_points(bad))
        assert await _get(cache, "anything") is None

    @pytest.mark.asyncio
    async def test_a_failed_write_does_not_break_the_caller(self, cache):
        """A cache write failing must never fail the request whose answer
        was being cached - the user already has their answer."""
        cache._ensure_collection = AsyncMock(side_effect=ConnectionError("down"))
        await cache.set(
            "k",
            SimpleNamespace(model_dump_json=lambda: "{}"),
            model="gpt-4o",
            scope="acme",
        )


class TestCacheIsolation:
    """Similarity is not authorisation. A near-identical prompt from another
    tenant, or answered by a different model, is not a valid hit."""

    @pytest.mark.asyncio
    async def test_a_lookup_is_filtered_by_model_and_scope(self, cache):
        client = await cache._ensure_collection()
        client.query_points = AsyncMock(return_value=_points())
        await cache.get("refund policy", model="gpt-4o", scope="tenant-a")

        query_filter = client.query_points.call_args.kwargs["query_filter"]
        conditions = {c.key: c.match.value for c in query_filter.must}
        assert conditions == {"model": "gpt-4o", "scope": "tenant-a"}

    @pytest.mark.asyncio
    async def test_a_write_records_model_and_scope(self, cache):
        client = await cache._ensure_collection()
        client.upsert = AsyncMock()
        await cache.set(
            "refund policy",
            SimpleNamespace(model_dump_json=lambda: "{}"),
            model="gpt-4o",
            scope="tenant-a",
        )
        payload = client.upsert.call_args.kwargs["points"][0].payload
        assert payload["model"] == "gpt-4o"
        assert payload["scope"] == "tenant-a"

    @pytest.mark.asyncio
    async def test_the_raw_prompt_is_never_stored(self, cache):
        """The cache would otherwise become an unbounded, unaudited copy of
        every prompt users send."""
        client = await cache._ensure_collection()
        client.upsert = AsyncMock()
        await cache.set(
            "my national insurance number is AB123456C",
            SimpleNamespace(model_dump_json=lambda: "{}"),
            model="gpt-4o",
            scope="acme",
        )
        payload = client.upsert.call_args.kwargs["points"][0].payload
        assert "AB123456C" not in json.dumps(payload)
