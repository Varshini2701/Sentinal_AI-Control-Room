"""Tests for logging, metrics and tracing helpers."""

from __future__ import annotations

import time

from sentinel.observability import (
    SENTINEL_REGISTRY,
    configure_logging,
    get_logger,
    observe_duration,
    start_span,
)
from sentinel.observability.metrics import EVENTS_PUBLISHED, SAFETY_VIOLATIONS
from sentinel.observability.tracing import configure_tracing, is_available


class TestLogging:
    def test_get_logger_returns_bound_logger(self) -> None:
        logger = get_logger("test", intersection="i-1")
        # Bound loggers expose bind(); calling a log method must not raise.
        logger.info("hello", extra_field=1)
        assert hasattr(logger, "bind")

    def test_configure_logging_idempotent(self) -> None:
        configure_logging(level="DEBUG", json_output=False)
        configure_logging(level="INFO", json_output=True)  # second call must not raise


class TestMetrics:
    def test_counter_increments(self) -> None:
        labels = {"event_type": "state.updated", "source": "perception"}
        before = SENTINEL_REGISTRY.get_sample_value(
            "sentinel_events_published_total", labels
        ) or 0.0
        EVENTS_PUBLISHED.labels(**labels).inc()
        after = SENTINEL_REGISTRY.get_sample_value("sentinel_events_published_total", labels)
        assert after == before + 1

    def test_safety_violations_metric_exists(self) -> None:
        # The SLO metric must be registered even before any violation occurs.
        SAFETY_VIOLATIONS.labels(intersection="i-1", constraint="min_green")
        value = SENTINEL_REGISTRY.get_sample_value(
            "sentinel_safety_violations_total",
            {"intersection": "i-1", "constraint": "min_green"},
        )
        assert value == 0.0

    def test_observe_duration_records_sample(self) -> None:
        from sentinel.observability.metrics import EVENT_PROCESSING_SECONDS

        labels = {"event_type": "decision.made", "consumer": "unit-test"}
        with observe_duration(EVENT_PROCESSING_SECONDS, **labels):
            time.sleep(0.001)
        count = SENTINEL_REGISTRY.get_sample_value(
            "sentinel_event_processing_seconds_count", labels
        )
        assert count == 1


class TestTracing:
    def test_start_span_is_noop_when_disabled(self) -> None:
        configure_tracing(service_name="sentinel", enabled=False)
        with start_span("noop.span", intersection="i-1") as span:
            assert span is None  # disabled -> no span object, but no error

    def test_configure_returns_availability(self) -> None:
        # When disabled, configure returns False regardless of install state.
        assert configure_tracing(service_name="s", enabled=False) is False
        # is_available reflects whether the optional dependency is importable.
        assert isinstance(is_available(), bool)
