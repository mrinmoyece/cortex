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


@pytest.fixture
def cache():
    c = SemanticCache()
    c._ensure_collection = AsyncMock(return_value=AsyncMock())
    c._embed = AsyncMock(return_value=[0.1] * 8)
    return c


class TestCacheReads:
    @pytest.mark.asyncio
    async def test_a_fresh_similar_hit_is_returned(self, cache):
        client = await cache._ensure_collection()
        client.search = AsyncMock(return_value=[_hit()])
        assert await cache.get("what is our refund policy") is not None

    @pytest.mark.asyncio
    async def test_no_match_returns_none_rather_than_the_nearest_thing(self, cache):
        """Qdrant's score_threshold does the filtering; returning a
        best-effort nearest neighbour would answer one question with
        another's answer."""
        client = await cache._ensure_collection()
        client.search = AsyncMock(return_value=[])
        assert await cache.get("something nobody has asked") is None

    @pytest.mark.asyncio
    async def test_an_expired_entry_is_not_served(self, cache):
        """Similarity says nothing about freshness. A 3-day-old answer to
        'what is our current pricing' is similar and wrong."""
        client = await cache._ensure_collection()
        client.search = AsyncMock(return_value=[_hit(age_seconds=48 * 3600)])
        assert await cache.get("pricing") is None

    @pytest.mark.asyncio
    async def test_an_entry_just_inside_the_ttl_is_still_served(self, cache):
        client = await cache._ensure_collection()
        client.search = AsyncMock(return_value=[_hit(age_seconds=60)])
        assert await cache.get("pricing") is not None


class TestCacheFailuresAreNonFatal:
    @pytest.mark.asyncio
    async def test_a_dead_vector_store_degrades_to_a_miss(self, cache):
        """The cache is an optimisation. If Qdrant is down the platform must
        get slower, not stop - so a failure here returns None and the real
        LLM call proceeds."""
        cache._ensure_collection = AsyncMock(side_effect=ConnectionError("qdrant is down"))
        assert await cache.get("anything") is None

    @pytest.mark.asyncio
    async def test_a_corrupt_payload_degrades_to_a_miss(self, cache):
        client = await cache._ensure_collection()
        bad = SimpleNamespace(
            score=0.99, payload={"created_at": time.time(), "response_json": "{{{"}
        )
        client.search = AsyncMock(return_value=[bad])
        assert await cache.get("anything") is None

    @pytest.mark.asyncio
    async def test_a_failed_write_does_not_break_the_caller(self, cache):
        """A cache write failing must never fail the request whose answer
        was being cached - the user already has their answer."""
        cache._ensure_collection = AsyncMock(side_effect=ConnectionError("down"))
        await cache.set("k", SimpleNamespace(model_dump=lambda: {"choices": []}))
