"""`@observe` — decorator tracing for agent steps.

The observability layer configured OTLP export and registered Phoenix, and
then nothing emitted a span. Traces existed as *configuration*, which is the
same defect as a metric nobody increments: the dashboard is wired up
and permanently empty, and you discover that during the incident you built it for.

## Why a decorator rather than manual spans

An agent run is a tree — plan, then N tasks, each with M tool calls, then a
critique, possibly looping. That shape is exactly what a trace is for, and
exactly what manual `with tracer.start_span(...)` blocks get wrong: someone
forgets one, the tree loses a level, and the flame graph silently lies about
where the time went.

A decorator makes instrumenting a step a one-line decision at the point the
step is defined, which is the only place anyone will remember to make it.

## What gets recorded, and what deliberately does not

Recorded: the step name, duration, success or failure, the exception type on
failure, and any explicitly-listed argument.

**Not** recorded by default: arguments and return values. Both routinely
contain the user's prompt, retrieved documents and model output — so
tracing everything means shipping customer data to whichever collector is
configured, usually a third-party SaaS. That is a privacy incident committed
by a debugging tool. Opt in per-argument with `capture=("run_id",)` when
the value is known-safe.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import time
from collections.abc import Callable, Sequence
from typing import Any, TypeVar

from cortex.logging_config import get_logger
from cortex.obs.metrics import agent_step_duration, agent_step_failures

logger = get_logger(__name__)

T = TypeVar("T")

#: Argument names that are never captured even when explicitly requested.
#: A denylist as well as the opt-in default, because "capture=('prompt',)"
#: is exactly the mistake someone makes at 2am while debugging.
_NEVER_CAPTURE = frozenset(
    {
        "prompt",
        "prompts",
        "messages",
        "text",
        "content",
        "query",
        "answer",
        "document",
        "documents",
        "context",
        "api_key",
        "token",
        "secret",
        "password",
    }
)


def _tracer() -> Any | None:
    """The OTel tracer, or None when tracing is not configured.

    Resolved per call rather than cached at import: `configure_tracing()`
    runs at startup, and a module-level tracer captured before it would be
    a no-op provider forever.
    """
    try:
        from opentelemetry import trace

        return trace.get_tracer("cortex")
    except Exception:
        return None


def _attributes(
    func: Callable[..., Any], args: tuple, kwargs: dict, capture: Sequence[str]
) -> dict[str, Any]:
    if not capture:
        return {}
    try:
        bound = inspect.signature(func).bind_partial(*args, **kwargs)
        bound.apply_defaults()
    except TypeError:
        return {}
    out: dict[str, Any] = {}
    for name in capture:
        if name in _NEVER_CAPTURE:
            logger.warning("tracing.refused_capture", argument=name, step=func.__name__)
            continue
        if name in bound.arguments:
            value = bound.arguments[name]
            if isinstance(value, (str, int, float, bool)):
                out[f"cortex.{name}"] = value
    return out


def observe(
    name: str | None = None,
    *,
    kind: str = "step",
    capture: Sequence[str] = (),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Trace a step: one span, one duration metric, one failure counter.

    Works on sync and async callables. The wrapper is chosen at decoration
    time rather than by inspecting at call time, because wrapping an async
    function in a sync wrapper returns the coroutine unawaited and the
    "duration" recorded is the time taken to *create* it — a nonsense
    number that looks plausible on a dashboard.

        @observe("planner.plan", capture=("run_id",))
        async def plan(self, state): ...
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        step_name = name or f"{func.__module__.rsplit('.', 1)[-1]}.{func.__name__}"

        def _record(start: float, error: BaseException | None) -> None:
            agent_step_duration.labels(step=step_name).observe(time.perf_counter() - start)
            if error is not None:
                agent_step_failures.labels(step=step_name, error_type=type(error).__name__).inc()

        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                tracer = _tracer()
                start = time.perf_counter()
                if tracer is None:
                    # Metrics still record without a tracer: Prometheus and
                    # OTel are independent, and losing step timings because
                    # a collector is unconfigured would be silly.
                    try:
                        result = await func(*args, **kwargs)
                    except BaseException as exc:
                        _record(start, exc)
                        raise
                    _record(start, None)
                    return result
                with tracer.start_as_current_span(step_name) as span:
                    span.set_attribute("cortex.kind", kind)
                    for key, value in _attributes(func, args, kwargs, capture).items():
                        span.set_attribute(key, value)
                    try:
                        result = await func(*args, **kwargs)
                    except BaseException as exc:
                        span.record_exception(exc)
                        span.set_attribute("cortex.ok", False)
                        _record(start, exc)
                        raise
                    span.set_attribute("cortex.ok", True)
                    _record(start, None)
                    return result

            return async_wrapper  # type: ignore[return-value]

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = _tracer()
            start = time.perf_counter()
            if tracer is None:
                try:
                    result = func(*args, **kwargs)
                except BaseException as exc:
                    _record(start, exc)
                    raise
                _record(start, None)
                return result
            with tracer.start_as_current_span(step_name) as span:
                span.set_attribute("cortex.kind", kind)
                for key, value in _attributes(func, args, kwargs, capture).items():
                    span.set_attribute(key, value)
                try:
                    result = func(*args, **kwargs)
                except BaseException as exc:
                    span.record_exception(exc)
                    span.set_attribute("cortex.ok", False)
                    _record(start, exc)
                    raise
                span.set_attribute("cortex.ok", True)
                _record(start, None)
                return result

        return sync_wrapper  # type: ignore[return-value]

    return decorator
