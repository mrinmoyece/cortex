"""
Cortex platform configuration.

All settings are loaded from environment variables or .env file.
Sensitive values are never logged. Defaults are safe for local dev only.
"""

from __future__ import annotations

import secrets
import warnings
from enum import Enum
from functools import lru_cache
from typing import Any

from pydantic import (
    AnyHttpUrl,
    Field,
    RedisDsn,
    SecretStr,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    LOCAL = "local"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE = "azure"
    BEDROCK = "bedrock"
    VERTEX = "vertex"
    OLLAMA = "ollama"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Core ─────────────────────────────────────────────────────────────────
    environment: Environment = Environment.LOCAL
    log_level: LogLevel = LogLevel.INFO
    debug: bool = False
    secret_key: SecretStr | None = Field(
        default=None,
        description="JWT signing key. Required in production; ephemeral in dev.",
    )

    # ── API ───────────────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"  # noqa: S104 - a container binds all interfaces  # noqa: S104 - a container must bind all interfaces to be reachable
    api_port: int = 8000
    api_workers: int = 4
    #: Empty by default, and that means CORS is OFF - no browser origin is
    #: allowed. Deliberate: an API whose clients are servers needs no CORS,
    #: and `["*"]` on a JWT-bearing API is how a token gets read by any page
    #: the user happens to visit. Set explicit origins to enable it.
    api_cors_origins: list[AnyHttpUrl] = []
    api_rate_limit_per_minute: int = 60
    #: Read-only SQLite databases the `query_data` MCP tool may query, as
    #: "alias=/path/to.db,other=/path/other.db". Empty by default: a data
    #: tool with no configured source should refuse, not invent.
    sql_database_aliases: str = ""

    # ── MCP Server ────────────────────────────────────────────────────────────
    mcp_host: str = "0.0.0.0"  # noqa: S104 - a container binds all interfaces
    mcp_port: int = 8001

    # ── Database ─────────────────────────────────────────────────────────────
    database_pool_size: int = 10
    database_max_overflow: int = 20

    # ── Redis ─────────────────────────────────────────────────────────────────
    redis_url: RedisDsn = Field(default="redis://localhost:6379/0")
    redis_episodic_db: int = 1
    redis_cache_db: int = 2
    redis_celery_db: int = 3

    # ── Qdrant ────────────────────────────────────────────────────────────────
    qdrant_url: AnyHttpUrl = Field(default="http://localhost:6333")
    qdrant_api_key: SecretStr | None = None
    qdrant_collection_rag: str = "cortex_rag"
    qdrant_collection_memory: str = "cortex_memory"

    # ── LLM Providers ─────────────────────────────────────────────────────────
    default_llm_provider: LLMProvider = LLMProvider.OPENAI
    default_model: str = "gpt-4o"
    fallback_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-large"
    embedding_dimensions: int = 3072

    # Provider API keys (all optional — use only what you have)
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    azure_openai_api_key: SecretStr | None = None
    azure_openai_endpoint: AnyHttpUrl | None = None
    azure_openai_api_version: str = "2024-08-01-preview"
    cohere_api_key: SecretStr | None = None
    aws_access_key_id: SecretStr | None = None
    aws_secret_access_key: SecretStr | None = None
    aws_region: str = "us-east-1"

    # ── LLM Budgets & Limits ──────────────────────────────────────────────────
    max_tokens_per_request: int = 4096
    max_cost_per_run_usd: float = 2.00
    semantic_cache_similarity_threshold: float = 0.95

    # ── RAG ───────────────────────────────────────────────────────────────────
    rag_chunk_size: int = 512
    rag_chunk_overlap: int = 64
    rag_top_k_retrieve: int = 20
    rag_top_k_rerank: int = 5
    rag_bm25_weight: float = 0.3
    rag_dense_weight: float = 0.7

    # ── Memory ────────────────────────────────────────────────────────────────
    memory_episodic_ttl_seconds: int = 86_400 * 7  # 7 days
    memory_working_token_budget: int = 8_192
    memory_consolidation_threshold: int = 10  # episodes before consolidation

    # ── Observability ─────────────────────────────────────────────────────────
    otel_exporter_endpoint: AnyHttpUrl | None = Field(
        default="http://localhost:4317", description="OTLP gRPC endpoint"
    )
    phoenix_endpoint: AnyHttpUrl = Field(default="http://localhost:6006")
    prometheus_port: int = 9090

    # ── Celery ────────────────────────────────────────────────────────────────
    celery_task_timeout_seconds: int = 300
    celery_max_retries: int = 3

    # ── Safety ────────────────────────────────────────────────────────────────
    guardrails_enabled: bool = True
    pii_detection_enabled: bool = True
    injection_detection_enabled: bool = True

    @field_validator("secret_key")
    @classmethod
    def validate_secret_key(cls, v: SecretStr | None) -> SecretStr | None:
        if v is not None and len(v.get_secret_value()) < 32:
            raise ValueError("secret_key must be at least 32 characters")
        return v

    @model_validator(mode="after")
    def require_secret_key_in_production(self) -> Settings:
        """Fail closed in production; generate an ephemeral key elsewhere.

        `secret_key` used to be `Field(...)` - required unconditionally. That
        is the right instinct and the wrong mechanism: it meant *reading any
        setting at all* raised ValidationError without it. A module that only
        wanted `redis_url` could not be imported, the whole test suite failed
        to collect, and `--cov-fail-under=80` sat on top of a suite that had
        never run.

        Production must still fail closed - a service signing JWTs with a key
        it invented on boot would invalidate every token on restart and, far
        worse, might do so silently. Outside production the key is generated
        per process and announced loudly, so local development works with no
        setup and nobody can mistake it for a configured secret.
        """
        if self.secret_key is None:
            if self.is_production:
                raise ValueError(
                    "SECRET_KEY is required when ENVIRONMENT=production. Generate one "
                    'with: python -c "import secrets; print(secrets.token_urlsafe(48))"'
                )
            object.__setattr__(self, "secret_key", SecretStr(secrets.token_urlsafe(48)))
            warnings.warn(
                "SECRET_KEY is unset - generated an ephemeral one for this process. "
                "Tokens will not survive a restart. Set SECRET_KEY for anything real.",
                RuntimeWarning,
                stacklevel=2,
            )
        return self

    @property
    def is_production(self) -> bool:
        return self.environment == Environment.PRODUCTION

    @property
    def redis_url_str(self) -> str:
        return str(self.redis_url)


@lru_cache(maxsize=1)
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings, constructed once.

    Cached deliberately: sixteen modules ask for settings, and building a
    `BaseSettings` re-reads the environment and `.env` every time.
    """
    return Settings()  # type: ignore[call-arg]


class _LazySettings:
    """Attribute access that resolves settings on first *use*, not on import.

    Every module used to do `settings = get_settings()` at module scope,
    which meant importing anything - including a pure-metrics module -
    constructed Settings, which validates SECRET_KEY, which is unset on a
    clean machine. So `import cortex.obs.metrics` raised ValidationError
    before a single line of application code ran, and the test suite could
    not even be collected: `--cov-fail-under=80` was gating a suite that
    could never start.

    A proxy keeps every existing `settings.foo` call site working while
    moving the construction to first attribute access, which is inside a
    function, at runtime, after the environment exists.
    """

    def __getattr__(self, name: str) -> Any:
        return getattr(get_settings(), name)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<LazySettings {get_settings()!r}>"


#: Import this, do not call `get_settings()` at module scope.
settings = _LazySettings()
