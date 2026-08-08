"""
Per-run cost tracker.

Stores cumulative token costs in Redis so budget enforcement works
across multiple LLM calls within the same agent run — even when those
calls happen across different async tasks or workers.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import redis.asyncio as aioredis
from litellm import ModelResponse

from cortex.config import settings
from cortex.logging_config import get_logger

#: `datetime.UTC` is 3.11+; this package supports 3.10.
UTC = timezone.utc

logger = get_logger(__name__)

_COST_KEY_PREFIX = "cortex:cost:"
_COST_TTL_SECONDS = 3600  # 1 hour — runs shouldn't last longer


class CostEntry:
    __slots__ = ("completion_tokens", "cost_usd", "model", "prompt_tokens", "recorded_at")

    def __init__(
        self,
        model: str,
        cost_usd: float,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        self.model = model
        self.cost_usd = cost_usd
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.recorded_at = datetime.now(UTC).isoformat()

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "cost_usd": self.cost_usd,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "recorded_at": self.recorded_at,
        }


class CostTracker:
    def __init__(self) -> None:
        self._redis: aioredis.Redis | None = None

    async def _get_redis(self) -> aioredis.Redis:
        if self._redis is None:
            self._redis = await aioredis.from_url(
                str(settings.redis_url).replace("/0", f"/{settings.redis_cache_db}"),
                encoding="utf-8",
                decode_responses=True,
            )
        return self._redis

    async def record(
        self,
        *,
        run_id: str,
        model: str,
        cost_usd: float,
        response: ModelResponse,
    ) -> None:
        """Append a cost entry to this run's ledger."""
        redis = await self._get_redis()
        entry = CostEntry(
            model=model,
            cost_usd=cost_usd,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
        )
        key = f"{_COST_KEY_PREFIX}{run_id}"
        pipe = redis.pipeline()
        pipe.rpush(key, json.dumps(entry.to_dict()))
        pipe.expire(key, _COST_TTL_SECONDS)
        await pipe.execute()
        logger.debug("cost.recorded", run_id=run_id, model=model, cost_usd=f"{cost_usd:.6f}")

    async def get_run_cost(self, run_id: str) -> float:
        """Return total USD spent in this run so far."""
        redis = await self._get_redis()
        key = f"{_COST_KEY_PREFIX}{run_id}"
        entries = await redis.lrange(key, 0, -1)
        return sum(json.loads(e)["cost_usd"] for e in entries)

    async def get_run_summary(self, run_id: str) -> dict:
        """Return a full cost breakdown for this run."""
        redis = await self._get_redis()
        key = f"{_COST_KEY_PREFIX}{run_id}"
        entries = [json.loads(e) for e in await redis.lrange(key, 0, -1)]
        total_cost = sum(e["cost_usd"] for e in entries)
        total_prompt = sum(e["prompt_tokens"] for e in entries)
        total_completion = sum(e["completion_tokens"] for e in entries)

        by_model: dict[str, float] = {}
        for e in entries:
            by_model[e["model"]] = by_model.get(e["model"], 0.0) + e["cost_usd"]

        return {
            "run_id": run_id,
            "total_cost_usd": round(total_cost, 6),
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
            "calls": len(entries),
            "by_model": {m: round(c, 6) for m, c in by_model.items()},
            "entries": entries,
        }
