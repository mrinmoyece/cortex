"""
Cortex three-tier memory system.

┌──────────────┬────────────────────┬─────────────────────────┬──────────────────┐
│ Tier         │ Store              │ Lifetime                │ What lives here  │
├──────────────┼────────────────────┼─────────────────────────┼──────────────────┤
│ Working      │ In-process dict    │ Single run              │ Current context  │
│ Episodic     │ Redis (list)       │ 7 days (configurable)   │ Past run summaries│
│ Semantic     │ Qdrant (vectors)   │ Permanent (until evict) │ Extracted facts  │
└──────────────┴────────────────────┴─────────────────────────┴──────────────────┘

At run start: retrieve relevant episodic + semantic context → inject into agents.
At run end: consolidate run summary to episodic; extract new facts to semantic.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

import litellm
import redis.asyncio as aioredis
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from cortex.config import settings
from cortex.graph.state import CortexState, MemoryContext
from cortex.logging_config import get_logger

logger = get_logger(__name__)

# Tenant FIRST in the key, then user.
#
# `CortexState` carries `tenant_id` and the docs describe Cortex as
# multi-tenant, but memory keyed on `user_id` alone. That is safe only while
# user ids are globally unique - and they are not, when they come from
# tenant-local identity providers, which is the normal enterprise case.
# Two tenants with a "user-1" would have shared a memory store.
#
# Defence in depth, not paranoia: the isolation now holds even if two
# tenants issue the same user id, and a key prefix is a cheap place to
# enforce it.
_EPISODIC_KEY = "cortex:episodic:{tenant_id}:{user_id}"
_EPISODIC_MEMBER_KEY = "cortex:episodic:{tenant_id}:{user_id}:{session_id}"
DEFAULT_TENANT = "default"


# ── Working Memory ─────────────────────────────────────────────────────────────


class WorkingMemory:
    """
    In-context memory for a single run.
    Token-budget aware — automatically truncates to fit the LLM window.
    """

    def __init__(self, token_budget: int | None = None) -> None:
        # Resolved here, not in the signature: a default argument is
        # evaluated at import time, so `= settings.x` reconstructs
        # Settings the moment the module is imported - the exact
        # import-time coupling this refactor removed everywhere else.
        token_budget = (
            token_budget if token_budget is not None else settings.memory_working_token_budget
        )
        self._store: list[dict[str, Any]] = []
        self._token_budget = token_budget
        self._tokens_used = 0

    def add(self, key: str, content: str, metadata: dict | None = None) -> bool:
        """Add item. Returns False if budget would be exceeded."""
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        tokens = len(enc.encode(content))

        if self._tokens_used + tokens > self._token_budget:
            logger.warning("working_memory.budget_exceeded", key=key, tokens_needed=tokens)
            return False

        self._store.append({"key": key, "content": content, "metadata": metadata or {}})
        self._tokens_used += tokens
        return True

    def get_all(self) -> list[dict[str, Any]]:
        return list(self._store)

    def to_prompt_text(self) -> str:
        if not self._store:
            return ""
        lines = [f"[{item['key']}] {item['content']}" for item in self._store]
        return "Current context:\n" + "\n".join(lines)


# ── Episodic Memory ────────────────────────────────────────────────────────────


class EpisodicMemory:
    """
    Redis-backed store of past run summaries for a user/session.
    Stored as a sorted set keyed by timestamp for efficient recency queries.
    """

    def __init__(self) -> None:
        self._redis: aioredis.Redis | None = None

    async def _get_redis(self) -> aioredis.Redis:
        if self._redis is None:
            db_url = str(settings.redis_url).rsplit("/", 1)[0] + f"/{settings.redis_episodic_db}"
            self._redis = await aioredis.from_url(db_url, encoding="utf-8", decode_responses=True)
        return self._redis

    async def store(
        self, user_id: str, session_id: str, summary: dict, tenant_id: str = DEFAULT_TENANT
    ) -> None:
        redis = await self._get_redis()
        key = _EPISODIC_KEY.format(tenant_id=tenant_id, user_id=user_id)
        member_key = _EPISODIC_MEMBER_KEY.format(
            tenant_id=tenant_id, user_id=user_id, session_id=session_id
        )
        score = time.time()

        pipe = redis.pipeline()
        pipe.set(member_key, json.dumps(summary), ex=settings.memory_episodic_ttl_seconds)
        pipe.zadd(key, {member_key: score})
        pipe.zremrangebyrank(key, 0, -51)  # Keep last 50 episodes
        await pipe.execute()
        logger.debug("episodic.stored", user_id=user_id, session_id=session_id)

    async def retrieve_recent(
        self, user_id: str, limit: int = 5, tenant_id: str = DEFAULT_TENANT
    ) -> list[dict]:
        redis = await self._get_redis()
        key = _EPISODIC_KEY.format(tenant_id=tenant_id, user_id=user_id)
        # Fetch most recent member keys
        member_keys = await redis.zrevrange(key, 0, limit - 1)
        if not member_keys:
            return []

        pipe = redis.pipeline()
        for mk in member_keys:
            pipe.get(mk)
        results = await pipe.execute()

        episodes = []
        for raw in results:
            if raw:
                try:
                    episodes.append(json.loads(raw))
                except json.JSONDecodeError:
                    pass
        return episodes


# ── Semantic Memory ────────────────────────────────────────────────────────────


class SemanticMemory:
    """
    Qdrant-backed store of extracted facts and knowledge.
    Facts are embedded and stored permanently (until explicit deletion).
    Retrieval is semantic — finds relevant facts by meaning, not exact match.
    """

    def __init__(self) -> None:
        self._client: AsyncQdrantClient | None = None
        self._initialized = False

    async def _get_client(self) -> AsyncQdrantClient:
        if self._client is None:
            self._client = AsyncQdrantClient(
                url=str(settings.qdrant_url),
                api_key=settings.qdrant_api_key.get_secret_value()
                if settings.qdrant_api_key
                else None,
            )
        if not self._initialized:
            collections = await self._client.get_collections()
            names = {c.name for c in collections.collections}
            if settings.qdrant_collection_memory not in names:
                await self._client.create_collection(
                    collection_name=settings.qdrant_collection_memory,
                    vectors_config=VectorParams(
                        size=settings.embedding_dimensions,
                        distance=Distance.COSINE,
                    ),
                )
            self._initialized = True
        return self._client

    async def _embed(self, text: str) -> list[float]:
        response = await litellm.aembedding(model=settings.embedding_model, input=[text])
        return response.data[0]["embedding"]

    async def store_facts(
        self, facts: list[dict], user_id: str, tenant_id: str = DEFAULT_TENANT
    ) -> None:
        """Store a list of extracted facts. Each fact: {content, source, entity_type}"""
        if not facts:
            return
        client = await self._get_client()
        points = []
        for fact in facts:
            vector = await self._embed(fact["content"])
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    # tenant_id is written here so the retrieval filter has
                    # something to match on. A filter on an absent payload key
                    # matches nothing, which would have been a silent, total
                    # retrieval failure rather than a leak - but still wrong.
                    payload={
                        **fact,
                        "user_id": user_id,
                        "tenant_id": tenant_id,
                        "stored_at": time.time(),
                    },
                )
            )
        await client.upsert(collection_name=settings.qdrant_collection_memory, points=points)
        logger.debug("semantic.facts_stored", count=len(facts), user_id=user_id)

    async def retrieve(
        self, query: str, user_id: str, top_k: int = 10, tenant_id: str = DEFAULT_TENANT
    ) -> list[dict]:
        """Retrieve relevant facts, filtered to this tenant AND this user."""
        client = await self._get_client()
        vector = await self._embed(query)
        results = await client.search(
            collection_name=settings.qdrant_collection_memory,
            query_vector=vector,
            limit=top_k,
            query_filter={
                "must": [
                    {"key": "tenant_id", "match": {"value": tenant_id}},
                    {"key": "user_id", "match": {"value": user_id}},
                ]
            },
            score_threshold=0.72,
        )
        return [{"content": r.payload["content"], "score": r.score, **r.payload} for r in results]


# ── Memory Agent ───────────────────────────────────────────────────────────────


class MemoryAgent:
    """
    Orchestrates all three memory tiers.
    Called by the graph's load_memory_node and save_memory_node.
    """

    def __init__(self) -> None:
        self.episodic = EpisodicMemory()
        self.semantic = SemanticMemory()

    async def retrieve(
        self, user_goal: str, session_id: str, user_id: str, tenant_id: str = DEFAULT_TENANT
    ) -> MemoryContext:
        """Gather relevant context from episodic and semantic stores."""
        episodic_task = self.episodic.retrieve_recent(user_id, limit=5, tenant_id=tenant_id)
        semantic_task = self.semantic.retrieve(user_goal, user_id, top_k=10, tenant_id=tenant_id)

        import asyncio

        episodic, semantic = await asyncio.gather(episodic_task, semantic_task)

        logger.info(
            "memory.retrieved",
            user_id=user_id,
            episodic_count=len(episodic),
            semantic_count=len(semantic),
        )
        return MemoryContext(episodic=episodic, semantic=semantic)

    async def consolidate(self, state: CortexState) -> None:
        """Persist run summary to episodic and extract facts to semantic."""
        summary = {
            "run_id": state.run_id,
            "goal": state.user_goal,
            "status": state.status.value,
            "output_summary": (state.final_output or "")[:500],
            "task_count": len(state.tasks),
            "cost_usd": state.total_cost_usd,
            "timestamp": time.time(),
        }

        await self.episodic.store(
            user_id=state.user_id,
            session_id=state.session_id,
            summary=summary,
            tenant_id=state.tenant_id,
        )

        # Extract facts from task results and output
        facts = self._extract_facts(state)
        if facts:
            await self.semantic.store_facts(facts, state.user_id, tenant_id=state.tenant_id)

        logger.info(
            "memory.consolidated",
            run_id=state.run_id,
            facts_stored=len(facts),
        )

    def _extract_facts(self, state: CortexState) -> list[dict]:
        """Simple heuristic fact extraction — replace with LLM extraction in prod."""
        facts = []
        for task in state.completed_tasks():
            if task.result and len(task.result) > 20:
                facts.append(
                    {
                        "content": f"{task.description}: {task.result[:300]}",
                        "source": f"run:{state.run_id}",
                        "entity_type": "task_result",
                    }
                )
        return facts[:5]  # Limit to 5 facts per run to control growth
