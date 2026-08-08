"""Load profile for the Cortex API.

    locust -f perf/locustfile.py --host http://localhost:8000

Why Locust and not k6: the workload here is not "hammer one endpoint". It
is a mix of cheap reads and expensive analyses under a spend ceiling, with
auth, and it needs to assert on the *server's own* metrics afterwards.
Expressing that in Python next to the code it tests beats a second language
and a second dependency.

## What this measures, and what it does not

**Does:** end-to-end request latency under concurrency, error-rate under
saturation, and whether the rate limiter and spend ceiling degrade the
service politely (429s with `Retry-After`) rather than by falling over.

**Does not:** model quality, or anything about a real LLM provider. Cortex
runs a deterministic scripted model, so these numbers describe *the
platform* - routing, serialisation, streaming, locking, the graph fan-out -
with provider latency removed. That is the honest thing to load-test here,
and it makes the results reproducible on any machine, which a
provider-bound benchmark never is.

A run against `CORTEX_PROVIDER=anthropic` would measure the provider. That is
a different, useful, and much noisier experiment; it is not this one.
"""

from __future__ import annotations

import os
import random

from locust import HttpUser, between, events, task

API_KEY = os.environ.get("CORTEX_LOAD_TEST_KEY", "")
GOALS = (
    "summarise our refund policy",
    "list the top three delivery risks",
    "what changed in pricing last quarter",
)

#: Budgets asserted at the end of a run. Set from a measured baseline with
#: roughly 2x headroom - tight enough that a real regression trips them,
#: loose enough to survive a noisy laptop. See docs/PERFORMANCE.md.
BUDGETS = {
    "/health": {"p95_ms": 50, "p99_ms": 150},
    "/api/v1/runs": {"p95_ms": 2_000, "p99_ms": 4_000},
    "/api/v1/runs/stream": {"p95_ms": 3_000, "p99_ms": 6_000},
}
#: Above this, the service is not degrading politely - it is failing.
MAX_ERROR_RATE = 0.01


class CortexUser(HttpUser):
    """One analyst. Reads are frequent, analyses are not."""

    wait_time = between(0.5, 2.0)

    def on_start(self) -> None:
        self.client.headers.update(
            {"authorization": f"Bearer {API_KEY}", "content-type": "application/json"}
        )

    @task(20)
    def health(self) -> None:
        """The cheapest possible path. If this degrades under load, the
        problem is the server itself and not any handler."""
        with self.client.get("/health", name="/health", catch_response=True) as r:
            if r.status_code != 200:
                r.failure(f"health returned {r.status_code}")

    @task(10)
    def metrics(self) -> None:
        """Prometheus scrapes this every 15s in production. A metrics
        endpoint that slows down under load takes your observability out at
        exactly the moment you need it."""
        self.client.get("/metrics", name="/metrics")

    @task(5)
    def analyse(self) -> None:
        with self.client.post(
            "/api/v1/runs",
            json={"goal": random.choice(GOALS)},
            name="/api/v1/runs",
            catch_response=True,
        ) as r:
            # 429 is a SUCCESS for this test. The rate limiter and the spend
            # ceiling doing their job is the system behaving correctly under
            # load; counting it as an error would mean the load test rewards
            # removing the controls.
            if r.status_code in (200, 429):
                r.success()
            else:
                r.failure(f"unexpected {r.status_code}")

    @task(2)
    def stream(self) -> None:
        with self.client.post(
            "/api/v1/runs/stream",
            json={"goal": random.choice(GOALS)},
            name="/api/v1/runs/stream",
            stream=True,
            catch_response=True,
        ) as r:
            if r.status_code not in (200, 429):
                r.failure(f"unexpected {r.status_code}")
                return
            # Drain it. Not draining leaves the connection open and measures
            # time-to-first-byte while calling it total latency.
            for _ in r.iter_lines():
                pass
            r.success()

    @task(1)
    def invalid_request(self) -> None:
        """404s and 422s are cheaper for an attacker to generate than valid
        requests, so they belong in the load profile."""
        with self.client.post(
            "/api/v1/runs",
            json={"goal": ""},  # fails validation
            name="/api/v1/runs [422]",
            catch_response=True,
        ) as r:
            if r.status_code in (422, 429):
                r.success()
            else:
                r.failure(f"expected 422/429, got {r.status_code}")


@events.quitting.add_listener
def enforce_budgets(environment, **_kwargs) -> None:
    """Fail the process if a budget was breached.

    This is what makes the load test a *gate* rather than a report. A
    performance number nobody enforces drifts, and the drift is only
    noticed when a user complains.
    """
    stats = environment.stats
    failures: list[str] = []

    total = stats.total
    if total.num_requests:
        error_rate = total.num_failures / total.num_requests
        if error_rate > MAX_ERROR_RATE:
            failures.append(f"error rate {error_rate:.2%} > {MAX_ERROR_RATE:.2%}")

    for name, budget in BUDGETS.items():
        entry = (
            stats.get(name, "GET") if name in ("/health", "/metrics") else stats.get(name, "POST")
        )
        if not entry or not entry.num_requests:
            continue
        for percentile, limit in (("p95_ms", 0.95), ("p99_ms", 0.99)):
            observed = entry.get_response_time_percentile(limit)
            allowed = budget[percentile]
            if observed > allowed:
                failures.append(f"{name} {percentile} {observed:.0f}ms > {allowed}ms")

    if failures:
        environment.process_exit_code = 1
        print("\nPERFORMANCE BUDGET BREACHED")
        for f in failures:
            print(f"  - {f}")
    else:
        environment.process_exit_code = 0
        print("\nAll performance budgets met.")
