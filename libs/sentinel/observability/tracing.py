"""OpenTelemetry-optional distributed tracing.

Tracing is a first-class part of a multi-agent system -- a single decision fans out across
perception, prediction and control -- but OpenTelemetry must never be a hard dependency of the
foundation. This module degrades gracefully: if ``opentelemetry`` is not installed or tracing is
disabled in config, :func:`start_span` becomes a zero-overhead no-op and the code path is
identical, so callers never branch on availability.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

try:  # pragma: no cover - exercised indirectly depending on install extras
    from opentelemetry import trace as _otel_trace

    _OTEL_AVAILABLE = True
except ImportError:  # pragma: no cover
    _otel_trace = None
    _OTEL_AVAILABLE = False

_enabled = False


def is_available() -> bool:
    """Whether the OpenTelemetry API is importable in this environment."""
    return _OTEL_AVAILABLE


def configure_tracing(
    *, service_name: str, enabled: bool, otlp_endpoint: str | None = None
) -> bool:
    """Configure the global tracer provider.

    Returns ``True`` if tracing was actually enabled (requires both ``enabled=True`` and the
    optional ``tracing`` extra installed), else ``False``. Wiring the concrete OTLP exporter is
    intentionally deferred to the deployment module; enabling here activates the API-level spans.
    """
    global _enabled
    if not enabled or not _OTEL_AVAILABLE:
        _enabled = False
        return False
    _enabled = True
    return True


@contextmanager
def start_span(name: str, **attributes: Any) -> Iterator[Any]:
    """Start a span named ``name`` with ``attributes``; a no-op if tracing is disabled.

    Yields the span (or ``None`` when disabled) so callers can add events without caring whether
    tracing is active::

        with start_span("decision.evaluate", intersection="i-1") as span:
            ...
    """
    if not (_enabled and _OTEL_AVAILABLE):
        yield None
        return

    tracer = _otel_trace.get_tracer("sentinel")
    with tracer.start_as_current_span(name) as span:
        for key, value in attributes.items():
            span.set_attribute(key, value)
        yield span


__all__ = ["configure_tracing", "is_available", "start_span"]
