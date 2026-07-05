"""Cross-cutting observability: structured logging, Prometheus metrics and tracing."""

from __future__ import annotations

from sentinel.observability.logging import (
    bind_contextvars,
    clear_contextvars,
    configure_logging,
    get_logger,
    unbind_contextvars,
)
from sentinel.observability.metrics import (
    DECISION_LATENCY_SECONDS,
    EVENTS_CONSUMED,
    EVENTS_PUBLISHED,
    SAFETY_VIOLATIONS,
    SENTINEL_REGISTRY,
    observe_duration,
    start_metrics_server,
)
from sentinel.observability.tracing import configure_tracing, is_available, start_span

__all__ = [
    "DECISION_LATENCY_SECONDS",
    "EVENTS_CONSUMED",
    "EVENTS_PUBLISHED",
    "SAFETY_VIOLATIONS",
    "SENTINEL_REGISTRY",
    "bind_contextvars",
    "clear_contextvars",
    "configure_logging",
    "configure_tracing",
    "get_logger",
    "is_available",
    "observe_duration",
    "start_metrics_server",
    "start_span",
    "unbind_contextvars",
]
