"""The LLM layer: budget enforcement, retry, fallback, cost ledger, cache.

These modules were at 31-40% coverage and hold the controls that stop a
runaway agent spending real money, so they are the last place to accept
"it looks right". Everything here runs against a fake Redis rather than a
live one, because a test that needs infrastructure is a test that does not
run - which is exactly how this suite ended up never having been executed.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import fakeredis.aioredis
import litellm
import pytest

from cortex.exceptions import LLMBudgetExceededError, LLMProviderUnavailableError
from cortex.llm.cost_tracker import CostTracker
from cortex.llm.router import LLMRouter, get_router


@pytest.fixture
def fake_redis():
    return fakeredis.aioredis.FakeRedis(decode_responses=True)


@pytest.fixture
def tracker(fake_redis):
    t = CostTracker()
    t._redis = fake_redis
    return t


def _response(cost_model="gpt-4o", prompt=100, completion=50, content="ok"):
    r = AsyncMock()
    r.usage.prompt_tokens = prompt
    r.usage.completion_tokens = completion
    r.choices = [AsyncMock()]
    r.choices[0].message.content = content
    return r


# ── cost ledger ───────────────────────────────────────────────────────────────


class TestCostTracker:
    @pytest.mark.asyncio
    async def test_record_then_read_back_the_total(self, tracker):
        await tracker.record(run_id="r1", model="gpt-4o", cost_usd=0.012, response=_response())
        await tracker.record(run_id="r1", model="gpt-4o", cost_usd=0.008, response=_response())
        assert await tracker.get_run_cost("r1") == pytest.approx(0.020)

    @pytest.mark.asyncio
    async def test_runs_are_isolated_from_each_other(self, tracker):
        """A ledger keyed loosely would let one run's spend halt another."""
        await tracker.record(run_id="r1", model="gpt-4o", cost_usd=1.0, response=_response())
        await tracker.record(run_id="r2", model="gpt-4o", cost_usd=0.1, response=_response())
        assert await tracker.get_run_cost("r1") == pytest.approx(1.0)
        assert await tracker.get_run_cost("r2") == pytest.approx(0.1)

    @pytest.mark.asyncio
    async def test_unknown_run_costs_nothing_rather_than_raising(self, tracker):
        assert await tracker.get_run_cost("never-seen") == 0.0

    @pytest.mark.asyncio
    async def test_summary_breaks_cost_down_by_model(self, tracker):
        await tracker.record(run_id="r", model="gpt-4o", cost_usd=0.10, response=_response())
        await tracker.record(run_id="r", model="gpt-4o-mini", cost_usd=0.01, response=_response())
        await tracker.record(run_id="r", model="gpt-4o", cost_usd=0.05, response=_response())
        s = await tracker.get_run_summary("r")
        assert s["calls"] == 3
        assert s["total_cost_usd"] == pytest.approx(0.16)
        assert s["by_model"]["gpt-4o"] == pytest.approx(0.15)
        assert s["by_model"]["gpt-4o-mini"] == pytest.approx(0.01)

    @pytest.mark.asyncio
    async def test_ledger_entries_expire(self, tracker, fake_redis):
        """Without a TTL the ledger grows forever; runs last minutes, not days."""
        await tracker.record(run_id="r", model="gpt-4o", cost_usd=0.1, response=_response())
        assert 0 < await fake_redis.ttl("cortex:cost:r") <= 3600


# ── budget gate ───────────────────────────────────────────────────────────────


class TestBudgetEnforcement:
    @pytest.mark.asyncio
    async def test_call_is_refused_once_the_run_ceiling_is_reached(self, tracker):
        """The whole point of the ledger. Checked BEFORE the call, so the
        spend that breaches the limit never happens."""
        router = LLMRouter()
        router._cost_tracker = tracker
        await tracker.record(run_id="r", model="gpt-4o", cost_usd=99.0, response=_response())

        with pytest.raises(LLMBudgetExceededError) as excinfo:
            await router.complete(messages=[{"role": "user", "content": "hi"}], run_id="r")
        assert "r" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_a_run_under_budget_is_allowed_through(self, tracker):
        router = LLMRouter()
        router._cost_tracker = tracker
        router._cache = AsyncMock()
        router._cache.get = AsyncMock(return_value=None)
        router._cache.set = AsyncMock()
        router._call_with_retry = AsyncMock(return_value=_response())

        with patch("cortex.llm.router.completion_cost", return_value=0.001):
            out = await router.complete(messages=[{"role": "user", "content": "hi"}], run_id="r")
        assert out is not None
        assert await tracker.get_run_cost("r") == pytest.approx(0.001)


# ── retry and fallback ────────────────────────────────────────────────────────


class TestRetryAndFallback:
    @pytest.mark.asyncio
    async def test_falls_back_to_the_cheaper_model_when_the_primary_fails(self, tracker):
        router = LLMRouter()
        router._cost_tracker = tracker
        seen: list[str] = []

        async def flaky(**kwargs):
            seen.append(kwargs["model"])
            if kwargs["model"] == "gpt-4o":
                raise litellm.exceptions.APIError(
                    status_code=500,
                    message="primary is down",
                    llm_provider="openai",
                    model="gpt-4o",
                )
            return _response()

        with patch("cortex.llm.router.acompletion", side_effect=flaky):
            out = await router._call_with_retry(
                {"model": "gpt-4o", "messages": [{"role": "user", "content": "x"}]}, run_id="r"
            )
        assert out is not None
        assert seen[0] == "gpt-4o", "should try the primary first"
        assert seen[-1] != "gpt-4o", "should have moved to the fallback"

    @pytest.mark.asyncio
    async def test_raises_when_primary_and_fallback_both_fail(self, tracker):
        """Silently returning nothing here would surface as an empty answer
        much further downstream, with no clue where it came from."""
        router = LLMRouter()
        router._cost_tracker = tracker

        with patch("cortex.llm.router.acompletion", side_effect=RuntimeError("everything is down")):
            with pytest.raises(LLMProviderUnavailableError):
                await router._call_with_retry(
                    {"model": "gpt-4o", "messages": [{"role": "user", "content": "x"}]},
                    run_id="r",
                )

    @pytest.mark.asyncio
    async def test_a_transient_failure_is_retried_not_escalated(self, tracker):
        router = LLMRouter()
        router._cost_tracker = tracker
        calls = {"n": 0}

        async def once_flaky(**kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                # A type the retry policy actually retries on. Anything else
                # is a fallback event, not a retry event - a distinction the
                # first draft of this test got wrong.
                raise litellm.exceptions.Timeout(
                    message="transient", model="gpt-4o", llm_provider="openai"
                )
            return _response()

        with patch("cortex.llm.router.acompletion", side_effect=once_flaky):
            out = await router._call_with_retry(
                {"model": "gpt-4o", "messages": [{"role": "user", "content": "x"}]}, run_id="r"
            )
        assert out is not None
        assert calls["n"] == 2, "should have retried the same model, not fallen back immediately"


def test_get_router_returns_one_shared_instance():
    """Agents resolve the router through this singleton, which is also why
    tests patch the singleton rather than the imported name."""
    assert get_router() is get_router()

    @pytest.mark.asyncio
    async def test_an_unexpected_exception_type_still_reaches_the_fallback(self, tracker):
        """Providers, proxies and SDK upgrades raise types litellm does not
        document. Before this, such an exception skipped the fallback and
        propagated raw - so the documented failure mode was not the one
        callers got, and the fallback was never tried for exactly the
        failures most likely to need it."""
        router = LLMRouter()
        router._cost_tracker = tracker
        seen: list[str] = []

        async def weird(**kwargs):
            seen.append(kwargs["model"])
            if kwargs["model"] == "gpt-4o":
                raise ValueError("something nobody anticipated")
            return _response()

        with patch("cortex.llm.router.acompletion", side_effect=weird):
            out = await router._call_with_retry(
                {"model": "gpt-4o", "messages": [{"role": "user", "content": "x"}]}, run_id="r"
            )
        assert out is not None
        assert seen[-1] != "gpt-4o", "an unexpected error must still fall back"

    @pytest.mark.asyncio
    async def test_unexpected_exception_on_both_models_is_wrapped_not_raw(self, tracker):
        router = LLMRouter()
        router._cost_tracker = tracker
        with patch("cortex.llm.router.acompletion", side_effect=ValueError("boom")):
            with pytest.raises(LLMProviderUnavailableError) as excinfo:
                await router._call_with_retry(
                    {"model": "gpt-4o", "messages": [{"role": "user", "content": "x"}]},
                    run_id="r",
                )
        assert "ValueError" in str(excinfo.value)


class TestToolForwarding:
    @pytest.mark.asyncio
    async def test_tool_schemas_are_forwarded_to_the_provider(self, tracker):
        router = LLMRouter()
        router._cost_tracker = tracker
        router._cache = AsyncMock(get=AsyncMock(return_value=None), set=AsyncMock())
        schemas = [{"type": "function", "function": {"name": "search_knowledge"}}]

        with patch("cortex.llm.router.acompletion", AsyncMock(return_value=_response())) as call:
            with patch("cortex.llm.router.completion_cost", return_value=0.001):
                await router.complete(
                    messages=[{"role": "user", "content": "hi"}], run_id="r", tools=schemas
                )
        assert call.await_args.kwargs["tools"] == schemas

    @pytest.mark.asyncio
    async def test_a_tool_bearing_call_bypasses_the_semantic_cache(self, tracker):
        """A cached response was produced against a different tool set.
        Replaying its tool_calls would invoke tools this caller never
        offered - a cache hit turning into an unauthorised action."""
        router = LLMRouter()
        router._cost_tracker = tracker
        router._cache = AsyncMock(get=AsyncMock(return_value=_response()), set=AsyncMock())

        with patch("cortex.llm.router.acompletion", AsyncMock(return_value=_response())):
            with patch("cortex.llm.router.completion_cost", return_value=0.001):
                await router.complete(
                    messages=[{"role": "user", "content": "hi"}],
                    run_id="r",
                    tools=[{"type": "function", "function": {"name": "x"}}],
                )
        router._cache.get.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_plain_scoped_call_still_uses_the_cache(self, tracker):
        router = LLMRouter()
        router._cost_tracker = tracker
        router._cache = AsyncMock(get=AsyncMock(return_value=None), set=AsyncMock())

        with patch("cortex.llm.router.acompletion", AsyncMock(return_value=_response())):
            with patch("cortex.llm.router.completion_cost", return_value=0.001):
                await router.complete(
                    messages=[{"role": "user", "content": "hi"}],
                    run_id="r",
                    cache_scope="acme",
                )
        router._cache.get.assert_awaited()
        assert router._cache.get.await_args.kwargs["scope"] == "acme"


class TestCacheScoping:
    """A semantic cache with no isolation boundary answers one tenant's
    question with another tenant's answer. Absence of a scope must disable
    the cache, not silently share it."""

    @pytest.mark.asyncio
    async def test_an_unscoped_call_does_not_read_the_cache(self, tracker):
        router = LLMRouter()
        router._cost_tracker = tracker
        router._cache = AsyncMock(get=AsyncMock(return_value=None), set=AsyncMock())

        with patch("cortex.llm.router.acompletion", AsyncMock(return_value=_response())):
            with patch("cortex.llm.router.completion_cost", return_value=0.001):
                await router.complete(messages=[{"role": "user", "content": "hi"}], run_id="r")
        router._cache.get.assert_not_awaited()
        router._cache.set.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_the_cache_embeds_the_prompt_not_a_digest(self):
        """The key used to be sha256(model + messages). Cosine similarity
        between hex digests is noise, so the cache was neither semantic nor
        safe."""
        text = LLMRouter._cache_text(
            [
                {"role": "system", "content": "be terse"},
                {"role": "user", "content": "refund policy"},
            ]
        )
        assert "refund policy" in text
        assert "be terse" in text

    @pytest.mark.asyncio
    async def test_a_pricing_failure_does_not_fail_a_successful_completion(self, tracker):
        """`completion_cost` raises for any model litellm has no pricing for,
        including anything newer than the pinned release - which used to
        discard a completion that had already been paid for."""
        router = LLMRouter()
        router._cost_tracker = tracker
        router._cache = AsyncMock(get=AsyncMock(return_value=None), set=AsyncMock())

        with patch("cortex.llm.router.acompletion", AsyncMock(return_value=_response())):
            with patch(
                "cortex.llm.router.completion_cost", side_effect=Exception("model not mapped")
            ):
                response = await router.complete(
                    messages=[{"role": "user", "content": "hi"}], run_id="r"
                )
        assert response is not None
        assert await tracker.get_run_cost("r") == pytest.approx(0.0)
