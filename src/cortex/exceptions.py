"""
Cortex exception hierarchy.

All domain exceptions inherit from CortexError so callers can catch
at the right granularity. Every exception carries a machine-readable
`code` for API responses and structured logging.
"""

from __future__ import annotations

from http import HTTPStatus
from typing import Any


class CortexError(Exception):
    """Base for all Cortex exceptions."""

    http_status: HTTPStatus = HTTPStatus.INTERNAL_SERVER_ERROR
    code: str = "CORTEX_ERROR"

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(code={self.code!r}, message={self.message!r})"


# ── Configuration ─────────────────────────────────────────────────────────────


class ConfigurationError(CortexError):
    """Raised when the platform is misconfigured."""

    http_status = HTTPStatus.INTERNAL_SERVER_ERROR
    code = "CONFIGURATION_ERROR"


# ── LLM / Model Layer ─────────────────────────────────────────────────────────


class LLMError(CortexError):
    """Base for LLM-related failures."""

    http_status = HTTPStatus.BAD_GATEWAY
    code = "LLM_ERROR"


class LLMRateLimitError(LLMError):
    """Provider rate limit hit — retry after backoff."""

    http_status = HTTPStatus.TOO_MANY_REQUESTS
    code = "LLM_RATE_LIMIT"


class LLMBudgetExceededError(LLMError):
    """Run exceeded the configured cost budget."""

    http_status = HTTPStatus.PAYMENT_REQUIRED
    code = "LLM_BUDGET_EXCEEDED"


class LLMProviderUnavailableError(LLMError):
    """All providers failed including fallbacks."""

    http_status = HTTPStatus.SERVICE_UNAVAILABLE
    code = "LLM_PROVIDER_UNAVAILABLE"


# ── Agent / Graph ─────────────────────────────────────────────────────────────


class AgentError(CortexError):
    """Base for agent execution failures."""

    http_status = HTTPStatus.UNPROCESSABLE_ENTITY
    code = "AGENT_ERROR"


class AgentTimeoutError(AgentError):
    """Agent exceeded its execution time budget."""

    http_status = HTTPStatus.GATEWAY_TIMEOUT
    code = "AGENT_TIMEOUT"


class AgentPlanningError(AgentError):
    """Planner failed to produce a valid task list."""

    code = "AGENT_PLANNING_FAILED"


class AgentMaxIterationsError(AgentError):
    """Graph hit the maximum iteration limit."""

    code = "AGENT_MAX_ITERATIONS"


class CriticRejectionError(AgentError):
    """Critic rejected output after max retries — escalate to human."""

    code = "CRITIC_MAX_REJECTIONS"


# ── RAG ───────────────────────────────────────────────────────────────────────


class RAGError(CortexError):
    """Base for retrieval-augmented generation failures."""

    http_status = HTTPStatus.INTERNAL_SERVER_ERROR
    code = "RAG_ERROR"


class IngestionError(RAGError):
    """Document ingestion failed."""

    code = "RAG_INGESTION_FAILED"


class RetrievalError(RAGError):
    """Vector store retrieval failed."""

    code = "RAG_RETRIEVAL_FAILED"


class EmbeddingError(RAGError):
    """Embedding generation failed."""

    code = "RAG_EMBEDDING_FAILED"


# ── Memory ────────────────────────────────────────────────────────────────────


class MemoryStoreError(CortexError):
    """Base for memory system failures.

    Named `MemoryStoreError` originally, which shadows the Python builtin. An
    `except MemoryStoreError:` anywhere downstream would then catch a Redis
    timeout and an actual out-of-memory condition with the same handler -
    and the reader has no way to tell which one was meant.
    """

    http_status = HTTPStatus.INTERNAL_SERVER_ERROR
    code = "MEMORY_ERROR"


class MemoryReadError(MemoryError):
    code = "MEMORY_READ_FAILED"


class MemoryWriteError(MemoryError):
    code = "MEMORY_WRITE_FAILED"


# ── Safety / Guardrails ───────────────────────────────────────────────────────


class SafetyError(CortexError):
    """Base for safety violations."""

    http_status = HTTPStatus.UNPROCESSABLE_ENTITY
    code = "SAFETY_VIOLATION"


class PromptInjectionError(SafetyError):
    """Input contains suspected prompt injection."""

    code = "PROMPT_INJECTION_DETECTED"


class PIIDetectedError(SafetyError):
    """Input or output contains personally identifiable information."""

    code = "PII_DETECTED"


class GuardrailViolationError(SafetyError):
    """NeMo guardrail policy triggered."""

    code = "GUARDRAIL_VIOLATION"


class HallucinationError(SafetyError):
    """Output failed faithfulness check against retrieved context."""

    code = "HALLUCINATION_DETECTED"


# ── MCP ───────────────────────────────────────────────────────────────────────


class MCPError(CortexError):
    """Base for MCP tool failures."""

    http_status = HTTPStatus.INTERNAL_SERVER_ERROR
    code = "MCP_ERROR"


class MCPToolError(MCPError):
    """A specific MCP tool call failed."""

    code = "MCP_TOOL_FAILED"


class MCPToolArgumentError(MCPError):
    """The caller supplied arguments the tool cannot accept.

    Distinct from `MCPToolError` because the blame is different, and so is
    the status code. A tool that raises while doing its job is a 500; a
    caller that passes an argument the tool does not have is a 422, and
    telling them so is the difference between "try again with `query`" and
    "something went wrong".

    The distinction used to be impossible to make. The client wrapped every
    exception - including the `TypeError` Python raises when arguments do
    not match the signature - into `MCPToolError`, so the API's 422 branch
    was unreachable code and a typo in a tool argument returned 500.
    """

    http_status = HTTPStatus.UNPROCESSABLE_ENTITY
    code = "MCP_TOOL_BAD_ARGUMENTS"


class MCPPermissionError(MCPError):
    """The tool refused because the caller is not entitled to what it asked for.

    Raised when a tool that acts on user-owned data has no authenticated
    principal bound. It is a 403 rather than a 401: the caller may well be
    authenticated to the API, and still have no identity bound *for this
    tool call*, which is a different failure and needs a different message.
    """

    http_status = HTTPStatus.FORBIDDEN
    code = "MCP_FORBIDDEN"


# ── Auth ──────────────────────────────────────────────────────────────────────


class AuthError(CortexError):
    """Base for authentication / authorisation failures."""

    http_status = HTTPStatus.UNAUTHORIZED
    code = "AUTH_ERROR"


class TokenExpiredError(AuthError):
    code = "TOKEN_EXPIRED"


class InsufficientPermissionsError(AuthError):
    http_status = HTTPStatus.FORBIDDEN
    code = "INSUFFICIENT_PERMISSIONS"
