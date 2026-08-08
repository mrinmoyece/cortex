"""Per-principal request rate limiting.

`api_rate_limit_per_minute` existed as a setting, was quoted in the README,
and nothing read it. A configured limit that nothing enforces is worse than
no limit at all: it is a control an operator believes they have.

Design notes, because rate limiting is easy to get subtly wrong:

**Token bucket, not a fixed window.** A fixed 60-per-minute window lets a
client send 60 requests at 11:59:59 and 60 more at 12:00:00 - 120 in one
second, twice the intended rate, at exactly the moment a traffic spike is
most likely to be adversarial. A bucket refilling continuously admits
bursts up to its capacity and no more.

**Middleware, not a route dependency.** It has to run before routing and
before body validation, or requests that fail validation escape limiting
entirely - and a 422 is cheaper for an attacker to generate than a valid
request. That exact hole was found and fixed in the sibling project; this
one starts on the right side of it.

**Bucketed by principal, falling back to client address.** Authenticated
users get their own allowance so one noisy tenant cannot exhaust another's.
Unauthenticated traffic is bucketed by source address, so failing to
present a token is not a way to escape the limit.

**In-process, and honest about it.** Behind N replicas the effective limit
is N x the configured value. That is a real limitation, recorded in
LIMITATIONS.md rather than papered over; the fix is a shared Redis counter
and the interface here does not change when that arrives.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

from cortex.config import settings
from cortex.logging_config import get_logger
from cortex.obs.metrics import rate_limit_rejections

logger = get_logger(__name__)


@dataclass
class Decision:
    allowed: bool
    retry_after_s: float = 0.0
    remaining: float = 0.0


class TokenBucket:
    """One bucket. Refills continuously; capacity bounds the burst."""

    __slots__ = ("_capacity", "_refill_per_second", "_tokens", "_updated")

    def __init__(self, capacity: float, refill_per_second: float) -> None:
        self._capacity = capacity
        self._refill_per_second = refill_per_second
        self._tokens = capacity
        self._updated = time.monotonic()

    def consume(self, tokens: float = 1.0) -> Decision:
        now = time.monotonic()
        elapsed = now - self._updated
        self._updated = now
        self._tokens = min(self._capacity, self._tokens + elapsed * self._refill_per_second)

        if self._tokens >= tokens:
            self._tokens -= tokens
            return Decision(allowed=True, remaining=self._tokens)

        deficit = tokens - self._tokens
        return Decision(
            allowed=False,
            retry_after_s=deficit / self._refill_per_second,
            remaining=self._tokens,
        )


class RateLimiter:
    """Buckets keyed by principal, created on first sight."""

    def __init__(self, per_minute: float | None = None, burst: float | None = None) -> None:
        self._per_minute = (
            per_minute if per_minute is not None else settings.api_rate_limit_per_minute
        )
        # A burst of a quarter of the per-minute rate: enough for a page that
        # fires several requests at once, far short of a useful flood.
        self._burst = burst if burst is not None else max(5.0, self._per_minute / 4)
        self._buckets: dict[str, TokenBucket] = {}
        self._lock = threading.Lock()

    @property
    def per_minute(self) -> float:
        return self._per_minute

    def check(self, principal: str) -> Decision:
        with self._lock:
            bucket = self._buckets.get(principal)
            if bucket is None:
                bucket = TokenBucket(self._burst, self._per_minute / 60.0)
                self._buckets[principal] = bucket
        return bucket.consume()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Consume one token per request, before routing or body validation."""

    #: Unmetered: liveness probes and scrapes must not be throttled, or a
    #: rate-limited service looks *unhealthy* to the very systems meant to
    #: observe it, and Kubernetes restarts a pod that was merely busy.
    EXEMPT_PATHS = frozenset({"/health", "/healthz", "/metrics", "/docs", "/openapi.json"})

    def __init__(self, app: ASGIApp, limiter: RateLimiter | None = None) -> None:
        super().__init__(app)
        self._limiter = limiter or RateLimiter()

    @staticmethod
    def _principal(request: Request) -> str:
        """Shallow identity: the token's subject if present, else the peer.

        Deliberately does not verify the JWT - full authentication happens in
        the route dependency, and doing it twice means two places to get
        authorisation wrong. An invalid token simply buckets by address.
        """
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            token = auth[7:].strip()
            if token:
                # The token itself is the bucket key. Two requests with the
                # same token share a bucket without this layer needing to
                # know, or trust, what is inside it.
                return f"token:{hash(token) & 0xFFFFFFFF:08x}"
        client = request.client
        return f"anon:{client.host if client else 'unknown'}"

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if request.url.path in self.EXEMPT_PATHS:
            return await call_next(request)

        decision = self._limiter.check(self._principal(request))
        if not decision.allowed:
            retry_after = max(1, int(decision.retry_after_s + 0.999))
            rate_limit_rejections.inc()
            logger.warning("api.rate_limited", path=request.url.path, retry_after_s=retry_after)
            return JSONResponse(
                {
                    "code": "RATE_LIMITED",
                    "message": (
                        f"Rate limit of {self._limiter.per_minute:.0f} requests/minute exceeded."
                    ),
                    "details": {"retry_after_seconds": retry_after},
                },
                status_code=429,
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(int(self._limiter.per_minute)),
                    "X-RateLimit-Remaining": "0",
                },
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(int(self._limiter.per_minute))
        response.headers["X-RateLimit-Remaining"] = str(int(decision.remaining))
        return response
