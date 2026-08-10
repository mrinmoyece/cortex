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


class TestBucketMapIsBounded:
    """The bucket key is caller-controlled: an unauthenticated attacker picks
    it by changing source address or token, so an unbounded map here is a
    memory-exhaustion primitive that needs no credential."""

    def test_bucket_count_never_exceeds_the_cap(self):
        limiter = RateLimiter(per_minute=600, max_buckets=8)
        for i in range(200):
            limiter.check(f"anon:10.0.0.{i}")
        assert limiter.bucket_count == 8

    def test_eviction_is_least_recently_seen(self):
        limiter = RateLimiter(per_minute=600, max_buckets=2)
        limiter.check("a")
        limiter.check("b")
        limiter.check("a")  # a is now the most recent
        limiter.check("c")  # evicts b

        # A surviving bucket has been consumed from; an evicted one is fresh.
        assert limiter.check("a").remaining < limiter.check("b").remaining

    def test_an_evicted_principal_fails_open(self):
        """Refusing new principals once full would let an attacker deny
        service to every legitimate caller - the opposite of the goal."""
        limiter = RateLimiter(per_minute=600, max_buckets=1)
        limiter.check("victim")
        limiter.check("attacker")
        assert limiter.check("victim").allowed


class TestPrincipalDerivation:
    """`hash(token) & 0xFFFFFFFF` was the old key: 32 bits is ~77k tokens to a
    coin-flip collision, colliding principals share an allowance, and `hash()`
    on str is randomised per process."""

    @staticmethod
    def _request(headers: dict[str, str], host: str = "203.0.113.7"):
        from starlette.datastructures import Headers
        from starlette.requests import Request

        scope = {
            "type": "http",
            "method": "GET",
            "path": "/api/v1/runs",
            "headers": Headers(headers).raw,
            "client": (host, 12345),
            "query_string": b"",
        }
        return Request(scope)

    def test_the_raw_token_is_never_the_key(self):
        secret = "eyJhbGciOi.super-secret-token.sig"
        key = RateLimitMiddleware._principal(self._request({"authorization": f"Bearer {secret}"}))
        assert secret not in key
        assert key.startswith("token:")

    def test_distinct_tokens_get_distinct_buckets(self):
        keys = {
            RateLimitMiddleware._principal(self._request({"authorization": f"Bearer t{i}"}))
            for i in range(500)
        }
        assert len(keys) == 500

    def test_the_key_is_stable_across_processes(self):
        """blake2b, not `hash()`: with PYTHONHASHSEED randomisation the same
        token keyed differently in every worker, so a caller got one
        allowance per replica."""
        import hashlib

        token = "a-token"
        key = RateLimitMiddleware._principal(self._request({"authorization": f"Bearer {token}"}))
        assert key == f"token:{hashlib.blake2b(token.encode(), digest_size=16).hexdigest()}"

    def test_no_token_buckets_by_peer_address(self):
        key = RateLimitMiddleware._principal(self._request({}, host="198.51.100.4"))
        assert key == "anon:198.51.100.4"

    def test_an_empty_bearer_falls_back_to_the_peer(self):
        key = RateLimitMiddleware._principal(
            self._request({"authorization": "Bearer   "}, host="198.51.100.4")
        )
        assert key == "anon:198.51.100.4"
