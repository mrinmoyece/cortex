"""
Cortex LLM Router.

Single entry point for all LLM calls across the platform.
Wraps LiteLLM to provide:
  - Provider-agnostic interface (OpenAI / Anthropic / Azure / Bedrock / Vertex / Ollama)
  - Per-request cost tracking with run-level budget enforcement
  - Semantic cache (skip the LLM if a near-identical prompt was seen recently)
  - Exponential-backoff retry on transient errors
  - Automatic fallback to a cheaper model when the primary fails
  - Structured logging of every call with latency, tokens, and cost
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, cast

import litellm
from litellm import acompletion, completion_cost
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from cortex.config import settings
from cortex.exceptions import LLMBudgetExceededError, LLMProviderUnavailableError, LLMRateLimitError
from cortex.llm.cache import SemanticCache
from cortex.llm.cost_tracker import CostTracker
from cortex.logging_config import get_logger
from cortex.obs.metrics import (
    llm_cost_total,
    llm_errors_total,
    llm_request_duration,
    llm_tokens_total,
)

logger = get_logger(__name__)

# Disable LiteLLM's own verbose logging; we handle it ourselves.
# `litellm.set_verbose` is deprecated and removed in newer releases, so it is
# set only if it still exists rather than being assumed.
litellm.suppress_debug_info = True
if hasattr(litellm, "set_verbose"):
    litellm.set_verbose = False
litellm.drop_params = True  # Silently ignore unsupported params per provider


class LLMRouter:
    """
    Central LLM router. Instantiate once and share across the application.

    Usage:
        router = LLMRouter()
        response = await router.complete(
            messages=[{"role": "user", "content": "Hello"}],
            model="gpt-4o",
            run_id="run-abc123",
        )
    """

    def __init__(self) -> None:
        self._cache = SemanticCache()
        self._cost_tracker = CostTracker()
        self._configure_litellm()

    def _configure_litellm(self) -> None:
        """Wire up provider credentials from settings."""
        if settings.openai_api_key:
            litellm.openai_key = settings.openai_api_key.get_secret_value()
        if settings.anthropic_api_key:
            litellm.anthropic_key = settings.anthropic_api_key.get_secret_value()
        if settings.azure_openai_api_key:
            litellm.azure_key = settings.azure_openai_api_key.get_secret_value()
        if settings.cohere_api_key:
            litellm.cohere_key = settings.cohere_api_key.get_secret_value()

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        run_id: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        use_cache: bool = True,
        cache_scope: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> litellm.ModelResponse:
        """
        Execute an LLM completion with full production guardrails.

        Args:
            cache_scope: Tenant (or other isolation boundary) the cached
                response may be served back to. Required for caching: a
                semantic cache with no scope will happily answer one
                tenant's question with another tenant's answer.

        Raises:
            LLMBudgetExceededError: If this run has already hit its cost limit.
            LLMProviderUnavailableError: If primary + fallback both fail.
        """
        model = model or settings.default_model
        max_tokens = max_tokens or settings.max_tokens_per_request

        # Budget gate — checked before every call
        current_cost = await self._cost_tracker.get_run_cost(run_id)
        if current_cost >= settings.max_cost_per_run_usd:
            raise LLMBudgetExceededError(
                f"Run {run_id} has spent ${current_cost:.4f}, limit is ${settings.max_cost_per_run_usd}",
                details={"run_id": run_id, "cost_usd": current_cost},
            )

        # A tool-bearing call is never cache-eligible: the cached response
        # was produced against a different tool set, and replaying its
        # tool_calls would invoke tools the caller did not offer.
        if tools:
            use_cache = False

        # An unscoped cache is a cross-tenant leak waiting to happen, so a
        # missing scope disables the cache rather than defaulting to global.
        if use_cache and not cache_scope:
            logger.debug("llm.cache_skipped_no_scope", run_id=run_id)
            use_cache = False

        cache_text = self._cache_text(messages)

        # Semantic cache lookup
        if use_cache:
            assert cache_scope is not None
            cached = await self._cache.get(cache_text, model=model, scope=cache_scope)
            if cached:
                logger.info("llm.cache_hit", run_id=run_id, model=model)
                return cached

        # Build call kwargs
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "metadata": {"run_id": run_id, **(metadata or {})},
        }
        if response_format:
            kwargs["response_format"] = response_format
        if tools:
            # Without this the model is never told which tools exist, so
            # `response.tool_calls` is always empty and every agent loop
            # silently degrades to single-shot prose. The executor fetched
            # its MCP schemas and dropped them on the floor; the linter
            # found it as an unused variable, which is the only reason it
            # was found at all.
            kwargs["tools"] = tools
            if tool_choice:
                kwargs["tool_choice"] = tool_choice

        # Execute with retry + fallback
        response = await self._call_with_retry(kwargs, run_id=run_id)

        # Accounting. `completion_cost` raises for models it has no pricing
        # for - including any model released after the pinned litellm - and
        # that used to fail a call whose completion had already succeeded and
        # already been paid for.
        try:
            cost = float(completion_cost(completion_response=response))
        except Exception as exc:
            logger.warning("llm.cost_unavailable", model=model, error=str(exc))
            cost = 0.0

        await self._cost_tracker.record(
            run_id=run_id, model=model, cost_usd=cost, response=response
        )

        # Metrics. Usage counts are optional in the OpenAI schema and are
        # absent from some providers and from streamed responses.
        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        # `token_type` here means LLM tokens, not auth tokens - the linter
        # cannot tell, and the label values are constants either way.
        llm_tokens_total.labels(model=model, token_type="input").inc(  # noqa: S106
            prompt_tokens
        )
        llm_tokens_total.labels(model=model, token_type="output").inc(  # noqa: S106
            completion_tokens
        )
        llm_cost_total.labels(model=model).inc(cost)

        # Cache store
        if use_cache:
            assert cache_scope is not None
            await self._cache.set(cache_text, response, model=model, scope=cache_scope)

        return response

    async def _call_with_retry(
        self, kwargs: dict[str, Any], *, run_id: str
    ) -> litellm.ModelResponse:
        """Retry on transient errors; fall back to cheaper model on persistent failure."""
        primary_model = kwargs["model"]
        fallback_model = settings.fallback_model

        for attempt_model in [primary_model, fallback_model]:
            kwargs["model"] = attempt_model
            try:
                async for attempt in AsyncRetrying(
                    stop=stop_after_attempt(3),
                    wait=wait_exponential(multiplier=1, min=2, max=30),
                    retry=retry_if_exception_type(
                        (litellm.exceptions.RateLimitError, litellm.exceptions.Timeout)
                    ),
                    reraise=True,
                ):
                    with attempt:
                        start = time.perf_counter()
                        response = await acompletion(**kwargs)
                        elapsed = time.perf_counter() - start

                        llm_request_duration.labels(model=attempt_model).observe(elapsed)
                        usage = getattr(response, "usage", None)
                        logger.info(
                            "llm.complete",
                            run_id=run_id,
                            model=attempt_model,
                            latency_ms=round(elapsed * 1000),
                            prompt_tokens=getattr(usage, "prompt_tokens", None),
                            completion_tokens=getattr(usage, "completion_tokens", None),
                        )
                        return cast(litellm.ModelResponse, response)

            except litellm.exceptions.RateLimitError as exc:
                llm_errors_total.labels(model=attempt_model, error_type="rate_limit").inc()
                if attempt_model == fallback_model:
                    raise LLMRateLimitError(str(exc)) from exc
                logger.warning("llm.rate_limit_falling_back", model=attempt_model)

            except litellm.exceptions.APIError as exc:
                llm_errors_total.labels(model=attempt_model, error_type="api_error").inc()
                if attempt_model == fallback_model:
                    raise LLMProviderUnavailableError(
                        f"All providers failed. Last error: {exc}"
                    ) from exc
                logger.warning(
                    "llm.provider_error_falling_back", model=attempt_model, error=str(exc)
                )

            except Exception as exc:
                # Deliberately last, and deliberately broad. The two handlers
                # above cover the exceptions litellm documents; providers,
                # proxies and SDK upgrades raise things it does not. Without
                # this, an unexpected type skipped the fallback entirely and
                # propagated raw out of `complete()` - so the documented
                # failure mode (LLMProviderUnavailableError) was not the one
                # callers actually got, and the fallback model was never
                # tried for the failures most likely to need it.
                llm_errors_total.labels(model=attempt_model, error_type=type(exc).__name__).inc()
                if attempt_model == fallback_model:
                    raise LLMProviderUnavailableError(
                        f"All providers failed. Last error: {type(exc).__name__}: {exc}"
                    ) from exc
                logger.warning(
                    "llm.unexpected_error_falling_back",
                    model=attempt_model,
                    error_type=type(exc).__name__,
                    error=str(exc),
                )

        # Should never reach here
        raise LLMProviderUnavailableError("Exhausted all models and retries")

    @asynccontextmanager
    async def stream(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        run_id: str,
    ) -> AsyncIterator[AsyncIterator[str]]:
        """
        Async context manager for streaming completions.

        Usage:
            async with router.stream(messages, run_id=run_id) as stream:
                async for chunk in stream:
                    yield chunk
        """
        model = model or settings.default_model
        response = await acompletion(
            model=model,
            messages=messages,
            stream=True,
            metadata={"run_id": run_id},
        )

        async def _iter() -> AsyncIterator[str]:
            async for chunk in response:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield delta.content

        yield _iter()

    @staticmethod
    def _cache_text(messages: list[dict[str, str]]) -> str:
        """The text the semantic cache embeds and compares.

        The message *content*, in order, with roles - not a hash of it.
        Two prompts that differ only in wording must land close together in
        embedding space, which is the entire premise of a semantic cache.
        """
        return "\n".join(
            f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages if m.get("content")
        )


# Module-level singleton — import this everywhere
_router: LLMRouter | None = None


def get_router() -> LLMRouter:
    global _router
    if _router is None:
        _router = LLMRouter()
    return _router
