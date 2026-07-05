"""Structured logging built on ``structlog``.

All Sentinel services log structured, contextual events. In production we emit JSON (one object
per line) suitable for log shippers; locally we render a colourised console view. A contextvar
context lets an agent bind ``intersection_id``/``correlation_id`` once so every subsequent log
line -- including those from deep in a call stack -- carries them automatically.
"""

from __future__ import annotations

import logging
from typing import Any, cast

import structlog
from structlog.contextvars import (
    bind_contextvars,
    clear_contextvars,
    unbind_contextvars,
)
from structlog.typing import Processor

_configured = False


def configure_logging(*, level: str = "INFO", json_output: bool = True) -> None:
    """Configure structlog and the stdlib logging bridge.

    Idempotent: safe to call from every service entrypoint. The final renderer is JSON when
    ``json_output`` is true, otherwise a colourised console renderer for local development.
    """
    global _configured

    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    renderer: Processor = (
        structlog.processors.JSONRenderer()
        if json_output
        else structlog.dev.ConsoleRenderer(colors=True)
    )

    structlog.reset_defaults()
    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelNamesMapping()[level.upper()]
        ),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Route stdlib logging (uvicorn, aio_pika, redis, ...) through the same renderer.
    logging.basicConfig(
        format="%(message)s",
        level=logging.getLevelNamesMapping()[level.upper()],
        force=True,
    )
    _configured = True


def get_logger(name: str | None = None, **initial_context: Any) -> structlog.stdlib.BoundLogger:
    """Return a bound logger, configuring logging with defaults on first use.

    Args:
        name: Logical logger name (usually the module or agent name).
        **initial_context: Key/value pairs bound to every line from this logger.
    """
    if not _configured:
        configure_logging()
    logger = structlog.get_logger(name)
    if initial_context:
        logger = logger.bind(**initial_context)
    return cast("structlog.stdlib.BoundLogger", logger)


__all__ = [
    "bind_contextvars",
    "clear_contextvars",
    "configure_logging",
    "get_logger",
    "unbind_contextvars",
]
