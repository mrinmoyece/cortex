"""
Semantic cache for LLM responses.

Before calling the LLM, we embed the prompt and check Qdrant for a
near-identical previous response. If similarity >= threshold, we return
the cached response without touching the LLM.

This cuts costs significantly for high-traffic deployments where users
ask semantically equivalent questions.
"""

from __future__ import annotations

import json
import time

import litellm
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from cortex.config import settings
from cortex.logging_config import get_logger

logger = get_logger(__name__)

_CACHE_COLLECTION = "cortex_llm_cache"
_CACHE_TTL_SECONDS = 3600 * 24  # 24 hours


class SemanticCache:
    def __init__(self) -> None:
        self._client: AsyncQdrantClient | None = None
        self._initialized = False

    async def _ensure_collection(self) -> AsyncQdrantClient:
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
            if _CACHE_COLLECTION not in names:
                await self._client.create_collection(
                    collection_name=_CACHE_COLLECTION,
                    vectors_config=VectorParams(
                        size=settings.embedding_dimensions,
                        distance=Distance.COSINE,
                    ),
                )
            self._initialized = True

        return self._client

    async def _embed(self, text: str) -> list[float]:
        response = await litellm.aembedding(
            model=settings.embedding_model,
            input=[text],
        )
        return response.data[0]["embedding"]

    async def get(self, cache_key: str) -> litellm.ModelResponse | None:
        """Return cached response if a semantically similar prompt exists."""
        try:
            client = await self._ensure_collection()
            vector = await self._embed(cache_key)

            results = await client.search(
                collection_name=_CACHE_COLLECTION,
                query_vector=vector,
                limit=1,
                score_threshold=settings.semantic_cache_similarity_threshold,
            )

            if not results:
                return None

            hit = results[0]
            payload = hit.payload or {}

            # Check TTL
            if time.time() - payload.get("created_at", 0) > _CACHE_TTL_SECONDS:
                logger.debug("cache.expired", score=hit.score)
                return None

            logger.info("cache.hit", score=f"{hit.score:.3f}")
            return litellm.ModelResponse(**json.loads(payload["response_json"]))

        except Exception as exc:
            # Cache failures are non-fatal — log and proceed
            logger.warning("cache.get_failed", error=str(exc))
            return None

    async def set(self, cache_key: str, response: litellm.ModelResponse) -> None:
        """Store an LLM response in the semantic cache."""
        try:
            client = await self._ensure_collection()
            vector = await self._embed(cache_key)

            import uuid

            point_id = str(uuid.uuid4())
            await client.upsert(
                collection_name=_CACHE_COLLECTION,
                points=[
                    PointStruct(
                        id=point_id,
                        vector=vector,
                        payload={
                            "response_json": response.model_dump_json(),
                            "created_at": time.time(),
                            "cache_key_prefix": cache_key[:64],
                        },
                    )
                ],
            )
            logger.debug("cache.stored", point_id=point_id)

        except Exception as exc:
            logger.warning("cache.set_failed", error=str(exc))
