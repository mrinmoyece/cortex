"""
Semantic cache for LLM responses.

Before calling the LLM, we embed the prompt and check Qdrant for a
near-identical previous response. If similarity >= threshold, we return
the cached response without touching the LLM.

This cuts costs significantly for high-traffic deployments where users
ask semantically equivalent questions.

Two properties matter more than the hit rate:

1. **The embedded text is the prompt.** The original implementation
   embedded `sha256(f"{model}:{messages}")` - a 64-character hex digest.
   Cosine similarity between two hex digests is noise, so the cache was
   neither semantic nor safe: near-identical prompts missed, and unrelated
   prompts could clear a 0.95 threshold and return someone else's answer.
2. **Every lookup is scoped.** A hit is only served if it was produced by
   the same model *and* the same tenant. Similarity alone is not
   authorisation.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from typing import Any

import litellm
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from cortex.config import settings
from cortex.logging_config import get_logger
from cortex.obs.metrics import llm_cache_hits_total

logger = get_logger(__name__)

_CACHE_COLLECTION = "cortex_llm_cache"
_CACHE_TTL_SECONDS = 3600 * 24  # 24 hours
#: Embedding providers reject oversized inputs, and a prompt long enough to
#: hit this is not one whose exact tail decides semantic equivalence.
_MAX_EMBED_CHARS = 8_000


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
            input=[text[:_MAX_EMBED_CHARS]],
        )
        embedding: list[float] = response.data[0]["embedding"]
        return embedding

    @staticmethod
    def _scope_filter(*, model: str, scope: str) -> Filter:
        """Similarity is not authorisation - a hit must match model and tenant."""
        return Filter(
            must=[
                FieldCondition(key="model", match=MatchValue(value=model)),
                FieldCondition(key="scope", match=MatchValue(value=scope)),
            ]
        )

    async def get(self, prompt: str, *, model: str, scope: str) -> litellm.ModelResponse | None:
        """Return a cached response for a semantically similar *scoped* prompt."""
        try:
            client = await self._ensure_collection()
            vector = await self._embed(prompt)

            response = await client.query_points(
                collection_name=_CACHE_COLLECTION,
                query=vector,
                query_filter=self._scope_filter(model=model, scope=scope),
                limit=1,
                score_threshold=settings.semantic_cache_similarity_threshold,
                with_payload=True,
            )
            results = response.points

            if not results:
                return None

            hit = results[0]
            payload: dict[str, Any] = hit.payload or {}

            # Check TTL
            if time.time() - payload.get("created_at", 0) > _CACHE_TTL_SECONDS:
                logger.debug("cache.expired", score=hit.score)
                return None

            logger.info("cache.hit", score=f"{hit.score:.3f}", model=model)
            llm_cache_hits_total.labels(model=model).inc()
            return litellm.ModelResponse(**json.loads(payload["response_json"]))

        except Exception as exc:
            # Cache failures are non-fatal — log and proceed
            logger.warning("cache.get_failed", error=str(exc))
            return None

    async def set(
        self, prompt: str, response: litellm.ModelResponse, *, model: str, scope: str
    ) -> None:
        """Store an LLM response in the semantic cache."""
        try:
            client = await self._ensure_collection()
            vector = await self._embed(prompt)

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
                            "model": model,
                            "scope": scope,
                            # The prompt itself is deliberately not stored:
                            # the cache would otherwise become an unbounded,
                            # unaudited copy of every prompt users send.
                            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                        },
                    )
                ],
            )
            logger.debug("cache.stored", point_id=point_id)

        except Exception as exc:
            logger.warning("cache.set_failed", error=str(exc))
