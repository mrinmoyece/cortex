"""
Cortex structured logging.

Uses structlog for machine-readable JSON logs in production and
pretty console output locally. Every log entry carries trace_id,
span_id, and service context for correlation with OTEL traces.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog
from structlog.types import EventDict, WrappedLogger

from cortex.config import Environment, get_settings


def _add_service_context(logger: WrappedLogger, method: str, event_dict: EventDict) -> EventDict:
    event_dict["service"] = "cortex"
    event_dict["environment"] = get_settings().environment.value
    return event_dict


def _drop_color_message_key(logger: WrappedLogger, method: str, event_dict: EventDict) -> EventDict:
    """Uvicorn adds `color_message` for terminal; strip it from structured logs."""
    event_dict.pop("color_message", None)
    return event_dict


def configure_logging() -> None:
    """Call once at application startup."""
    settings = get_settings()
    is_local = settings.environment == Environment.LOCAL

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        _add_service_context,
        _drop_color_message_key,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    renderer: structlog.typing.Processor = (
        structlog.dev.ConsoleRenderer(colors=True)
        if is_local
        else structlog.processors.JSONRenderer()
    )

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(settings.log_level.value)
        ),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            *shared_processors,
            renderer,
        ]
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(settings.log_level.value)

    # Silence noisy libraries
    for noisy in ("httpx", "httpcore", "uvicorn.access", "LiteLLM"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.BoundLogger:
    logger: structlog.BoundLogger = structlog.get_logger(name)
    return logger
