"""Settings that other modules derive infrastructure addresses from.

`redis_url_for_db` exists because four call sites did the same string
surgery, two different ways, and both were wrong for URLs that were not the
localhost default.
"""

from __future__ import annotations

import pytest

from cortex.config import Settings


def _settings(url: str) -> Settings:
    s = Settings()
    object.__setattr__(s, "redis_url", url)
    return s


class TestRedisUrlForDb:
    def test_replaces_the_database_on_the_default_url(self):
        assert _settings("redis://localhost:6379/0").redis_url_for_db(3) == (
            "redis://localhost:6379/3"
        )

    def test_appends_a_database_when_the_url_has_no_path(self):
        """`rsplit("/", 1)` turned `redis://host:6379` into
        `redis://host:63791` - a valid-looking URL pointing at a port that
        does not exist."""
        assert _settings("redis://host:6379").redis_url_for_db(1) == "redis://host:6379/1"

    def test_a_zero_in_the_password_is_not_rewritten(self):
        """`.replace("/0", "/2")` rewrote any "/0" in the URL, including one
        inside a credential - silently corrupting the password."""
        url = "redis://user:p%2F0ss@redis.internal:6379/0"
        result = _settings(url).redis_url_for_db(2)
        assert "p%2F0ss" in result
        assert result.endswith("/2")

    def test_host_and_scheme_survive(self):
        result = _settings("rediss://cache.example.com:6380/0").redis_url_for_db(5)
        assert result == "rediss://cache.example.com:6380/5"

    @pytest.mark.parametrize("db", [0, 1, 15])
    def test_round_trips_for_every_standard_database(self, db: int):
        assert _settings("redis://h:6379/0").redis_url_for_db(db).endswith(f"/{db}")


class TestSecurityDefaults:
    def test_code_execution_is_off_by_default(self):
        """It runs arbitrary Python in the API container. Off unless someone
        deliberately turns it on, in an environment they chose."""
        assert Settings().code_execution_enabled is False

    def test_human_review_is_off_by_default(self):
        """It suspends the graph, and there is no resume endpoint. On by
        default meant every run stopped half-done."""
        assert Settings().human_review_before_critic is False

    def test_state_bounds_are_finite(self):
        s = Settings()
        assert s.api_max_tracked_runs > 0
        assert s.api_run_retention_seconds > 0
        assert s.api_rate_limit_max_buckets > 0
        assert s.rag_max_indexed_chunks > 0

    def test_secret_key_is_not_a_plain_string(self):
        """A `SecretStr` does not land in a repr, a log line, or an error
        page by accident."""
        from pydantic import SecretStr

        assert isinstance(Settings().secret_key, SecretStr)


class TestNoSettingsAtImportTime:
    def test_the_module_exposes_a_lazy_proxy(self):
        """Sixteen modules called `get_settings()` at import scope, so
        `import cortex.obs.metrics` raised `ValidationError` on any machine
        without SECRET_KEY - which is every clean machine, including CI."""
        import cortex.config as config

        assert type(config.settings).__name__ == "_LazySettings"
