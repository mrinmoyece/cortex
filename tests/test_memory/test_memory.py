"""The three memory tiers.

Working memory is the one with a hard invariant worth pinning: it must
refuse content that would blow the token budget rather than silently
truncating it, because a silently truncated context produces a confident
answer built on half the evidence.
"""

from __future__ import annotations

import json

import fakeredis.aioredis
import pytest

from cortex.agents.memory_agent import EpisodicMemory, WorkingMemory


class TestWorkingMemory:
    def test_items_are_stored_and_returned_in_order(self):
        m = WorkingMemory(token_budget=1000)
        assert m.add("a", "first") is True
        assert m.add("b", "second") is True
        assert [i["key"] for i in m.get_all()] == ["a", "b"]

    def test_content_over_budget_is_refused_not_truncated(self):
        """Returning False lets the caller decide. Truncating silently is
        how a summary loses its most important paragraph without anyone
        finding out."""
        m = WorkingMemory(token_budget=10)
        assert m.add("small", "hi") is True
        assert m.add("huge", "word " * 500) is False
        assert [i["key"] for i in m.get_all()] == ["small"]

    def test_the_budget_accumulates_across_items(self):
        """A per-item check would let a hundred small items overflow a
        window that no single one could."""
        m = WorkingMemory(token_budget=25)
        added = [m.add(f"k{i}", "ten tokens of text here roughly") for i in range(10)]
        assert added[0] is True
        assert False in added, "the budget must apply cumulatively, not per item"

    def test_metadata_is_preserved(self):
        m = WorkingMemory(token_budget=1000)
        m.add("k", "v", {"source": "rag"})
        assert m.get_all()[0]["metadata"] == {"source": "rag"}

    def test_default_budget_comes_from_settings_at_call_time(self):
        """Not from a default argument - that would rebuild Settings at
        import time, which is the coupling this codebase had to remove."""
        assert WorkingMemory()._token_budget > 0


class TestEpisodicMemory:
    @pytest.fixture
    def memory(self):
        m = EpisodicMemory()
        m._redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
        return m

    @pytest.mark.asyncio
    async def test_episodes_come_back_newest_first(self, memory):
        """Ordering matters: the most recent run is the most relevant
        context, and a sorted set scored by timestamp is what guarantees it."""
        for i in range(3):
            await memory.store(user_id="u", session_id=f"s{i}", summary={"episode": i})
        out = await memory.retrieve_recent(user_id="u", limit=3)
        assert len(out) == 3
        assert out[0]["episode"] == 2, f"expected newest first, got {out}"

    @pytest.mark.asyncio
    async def test_users_cannot_see_each_others_episodes(self, memory):
        """Multi-tenancy at the storage layer. A leak here is a data breach,
        not a bug report."""
        await memory.store(user_id="alice", session_id="s1", summary={"note": "alice private"})
        await memory.store(user_id="bob", session_id="s2", summary={"note": "bob private"})

        alice = json.dumps(await memory.retrieve_recent(user_id="alice", limit=10))
        assert "bob private" not in alice

    @pytest.mark.asyncio
    async def test_episodes_expire(self, memory):
        await memory.store(user_id="u", session_id="s", summary={"note": "s"})
        keys = await memory._redis.keys("*")
        assert keys, "nothing was written"
        ttls = [await memory._redis.ttl(k) for k in keys]
        assert any(t > 0 for t in ttls), (
            f"no TTL set - memory grows forever: {dict(zip(keys, ttls, strict=False))}"
        )

    @pytest.mark.asyncio
    async def test_a_user_with_no_history_gets_an_empty_list(self, memory):
        assert await memory.retrieve_recent(user_id="nobody", limit=5) == []

    @pytest.mark.asyncio
    async def test_history_is_capped_per_user(self, memory):
        """Unbounded history is a slow memory leak with a per-user blast
        radius; the cap is what keeps recall cost predictable."""
        for i in range(60):
            await memory.store(user_id="u", session_id=f"s{i}", summary={"i": i})
        out = await memory.retrieve_recent(user_id="u", limit=100)
        assert len(out) <= 50, f"expected at most 50 retained episodes, got {len(out)}"


class TestTenantIsolation:
    """`tenant_id` was carried in state, described in the docs, and used by
    neither memory tier. Safe only while user ids are globally unique - and
    they are not, when they come from tenant-local identity providers, which
    is the normal enterprise case."""

    @pytest.fixture
    def memory(self):
        m = EpisodicMemory()
        m._redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
        return m

    @pytest.mark.asyncio
    async def test_the_same_user_id_in_two_tenants_stays_separate(self, memory):
        await memory.store(
            user_id="user-1", session_id="s", summary={"note": "acme secret"}, tenant_id="acme"
        )
        await memory.store(
            user_id="user-1", session_id="s", summary={"note": "globex secret"}, tenant_id="globex"
        )

        acme = json.dumps(await memory.retrieve_recent("user-1", limit=10, tenant_id="acme"))
        assert "acme secret" in acme
        assert "globex secret" not in acme, "cross-tenant leak on a colliding user id"

    @pytest.mark.asyncio
    async def test_the_tenant_appears_in_the_storage_key(self, memory):
        await memory.store(user_id="u", session_id="s", summary={"x": 1}, tenant_id="acme")
        keys = await memory._redis.keys("*")
        assert any("acme" in k for k in keys), f"tenant absent from keys: {keys}"

    @pytest.mark.asyncio
    async def test_a_tenant_with_no_history_sees_nothing(self, memory):
        await memory.store(user_id="u", session_id="s", summary={"x": 1}, tenant_id="acme")
        assert await memory.retrieve_recent("u", limit=10, tenant_id="other") == []
