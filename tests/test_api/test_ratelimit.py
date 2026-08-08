"""Rate limiting — the control that existed only as a setting.

`api_rate_limit_per_minute` was configured, documented in the README, and
read by nothing. Every test here would have failed against that.
"""

from __future__ import annotations

import time

from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from cortex.api.ratelimit import RateLimiter, RateLimitMiddleware, TokenBucket


class TestTokenBucket:
    def test_a_burst_up_to_capacity_is_allowed(self):
        b = TokenBucket(capacity=5, refill_per_second=1)
        assert all(b.consume().allowed for _ in range(5))

    def test_the_bucket_empties(self):
        b = TokenBucket(capacity=3, refill_per_second=1)
        for _ in range(3):
            b.consume()
        assert b.consume().allowed is False

    def test_it_refills_over_time(self):
        b = TokenBucket(capacity=2, refill_per_second=100)
        b.consume()
        b.consume()
        assert b.consume().allowed is False
        time.sleep(0.05)
        assert b.consume().allowed is True

    def test_refill_is_capped_at_capacity(self):
        """Otherwise an idle client accrues an unbounded allowance and can
        dump the whole backlog at once — the exact burst the limiter exists
        to prevent."""
        b = TokenBucket(capacity=2, refill_per_second=1000)
        time.sleep(0.05)
        assert b.consume().allowed and b.consume().allowed
        assert b.consume().allowed is False

    def test_rejection_says_how_long_to_wait(self):
        b = TokenBucket(capacity=1, refill_per_second=2)
        b.consume()
        d = b.consume()
        assert d.allowed is False
        assert 0 < d.retry_after_s <= 1.0


class TestRateLimiter:
    def test_principals_get_independent_allowances(self):
        """One noisy tenant must not exhaust another's limit."""
        limiter = RateLimiter(per_minute=60, burst=2)
        assert limiter.check("alice").allowed and limiter.check("alice").allowed
        assert limiter.check("alice").allowed is False
        assert limiter.check("bob").allowed is True


def _app(limiter: RateLimiter) -> TestClient:
    routes = [
        Route("/v1/thing", lambda r: JSONResponse({"ok": True})),
        Route("/health", lambda r: JSONResponse({"status": "ok"})),
        Route("/metrics", lambda r: JSONResponse({"m": 1})),
    ]
    app = Starlette(routes=routes)
    app.add_middleware(RateLimitMiddleware, limiter=limiter)
    return TestClient(app)


class TestMiddleware:
    def test_requests_over_the_limit_get_429_with_retry_after(self):
        client = _app(RateLimiter(per_minute=60, burst=2))
        assert client.get("/v1/thing").status_code == 200
        assert client.get("/v1/thing").status_code == 200

        blocked = client.get("/v1/thing")
        assert blocked.status_code == 429
        assert int(blocked.headers["Retry-After"]) >= 1
        assert blocked.json()["code"] == "RATE_LIMITED"

    def test_health_and_metrics_are_never_throttled(self):
        """A throttled /health makes Kubernetes restart a pod that was
        merely busy, turning a load spike into an outage."""
        client = _app(RateLimiter(per_minute=60, burst=1))
        client.get("/v1/thing")
        client.get("/v1/thing")
        assert client.get("/health").status_code == 200
        assert client.get("/metrics").status_code == 200

    def test_remaining_allowance_is_advertised(self):
        client = _app(RateLimiter(per_minute=60, burst=5))
        r = client.get("/v1/thing")
        assert r.headers["X-RateLimit-Limit"] == "60"
        assert int(r.headers["X-RateLimit-Remaining"]) < 5

    def test_different_tokens_are_metered_separately(self):
        client = _app(RateLimiter(per_minute=60, burst=1))
        assert client.get("/v1/thing", headers={"authorization": "Bearer aaa"}).status_code == 200
        assert client.get("/v1/thing", headers={"authorization": "Bearer aaa"}).status_code == 429
        assert client.get("/v1/thing", headers={"authorization": "Bearer bbb"}).status_code == 200

    def test_dropping_the_token_does_not_escape_the_limit(self):
        """Anonymous traffic buckets by address, so presenting no
        credentials is not a bypass."""
        client = _app(RateLimiter(per_minute=60, burst=1))
        assert client.get("/v1/thing").status_code == 200
        assert client.get("/v1/thing").status_code == 429

    def test_unknown_routes_are_metered_too(self):
        """404s and 422s are cheaper to generate than valid requests, which
        makes them the better flood. Limiting must happen before routing."""
        client = _app(RateLimiter(per_minute=60, burst=1))
        client.get("/v1/nope")
        assert client.get("/v1/nope").status_code == 429
