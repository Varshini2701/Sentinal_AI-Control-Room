"""Prometheus metrics for Sentinel AI.

All metrics are registered on a dedicated :data:`SENTINEL_REGISTRY` rather than the global
default registry. This keeps Sentinel metrics isolated (important for embedding and for tests
that create/discard registries) and prevents ``Duplicated timeseries`` errors on module reload.

The metric names encode the two most important SLOs of the system:

* ``sentinel_safety_violations_total`` -- **must remain 0**; a Prometheus alert fires otherwise.
* ``sentinel_decision_latency_seconds`` -- the control-loop latency budget.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, start_http_server

SENTINEL_REGISTRY = CollectorRegistry()
"""The registry all Sentinel metrics are attached to."""

# --- Event bus ---
EVENTS_PUBLISHED = Counter(
    "sentinel_events_published_total",
    "Total domain events published to the bus.",
    labelnames=("event_type", "source"),
    registry=SENTINEL_REGISTRY,
)
EVENTS_CONSUMED = Counter(
    "sentinel_events_consumed_total",
    "Total domain events handled by a consumer, by outcome.",
    labelnames=("event_type", "consumer", "outcome"),  # outcome: success|retry|dlq
    registry=SENTINEL_REGISTRY,
)
EVENT_PROCESSING_SECONDS = Histogram(
    "sentinel_event_processing_seconds",
    "Wall-clock time a consumer spent handling one event.",
    labelnames=("event_type", "consumer"),
    registry=SENTINEL_REGISTRY,
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)
DLQ_MESSAGES = Counter(
    "sentinel_dlq_messages_total",
    "Messages routed to a dead-letter queue after exhausting retries.",
    labelnames=("event_type",),
    registry=SENTINEL_REGISTRY,
)

# --- Control loop ---
DECISION_LATENCY_SECONDS = Histogram(
    "sentinel_decision_latency_seconds",
    "Time from receiving a state update to emitting a signal command.",
    labelnames=("intersection",),
    registry=SENTINEL_REGISTRY,
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)
SAFETY_VIOLATIONS = Counter(
    "sentinel_safety_violations_total",
    "Times the safety envelope had to clamp or reject an unsafe proposed command. "
    "This MUST stay at zero in a correct system.",
    labelnames=("intersection", "constraint"),
    registry=SENTINEL_REGISTRY,
)

# --- Perception ---
PERCEPTION_FPS = Gauge(
    "sentinel_perception_fps",
    "Frames per second processed by the perception pipeline.",
    labelnames=("intersection",),
    registry=SENTINEL_REGISTRY,
)
PERCEPTION_DROPPED_FRAMES = Counter(
    "sentinel_perception_dropped_frames_total",
    "Frames skipped due to inference errors or backpressure.",
    labelnames=("intersection",),
    registry=SENTINEL_REGISTRY,
)
ACTIVE_TRACKS = Gauge(
    "sentinel_active_tracks",
    "Currently active object tracks.",
    labelnames=("intersection",),
    registry=SENTINEL_REGISTRY,
)

# --- Agent health ---
AGENT_UP = Gauge(
    "sentinel_agent_up",
    "1 if the agent reported healthy on its last heartbeat, else 0.",
    labelnames=("agent",),
    registry=SENTINEL_REGISTRY,
)


@contextmanager
def observe_duration(histogram: Histogram, **labels: str) -> Iterator[None]:
    """Context manager that records the wall-clock duration of a block into ``histogram``.

    Uses a monotonic clock so it is immune to system-clock adjustments::

        with observe_duration(DECISION_LATENCY_SECONDS, intersection="i-1"):
            command = policy.decide(state)
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        target = histogram.labels(**labels) if labels else histogram
        target.observe(time.perf_counter() - start)


def start_metrics_server(port: int) -> None:
    """Expose the Sentinel registry over HTTP for Prometheus to scrape at ``/metrics``."""
    start_http_server(port, registry=SENTINEL_REGISTRY)


__all__ = [
    "ACTIVE_TRACKS",
    "AGENT_UP",
    "DECISION_LATENCY_SECONDS",
    "DLQ_MESSAGES",
    "EVENTS_CONSUMED",
    "EVENTS_PUBLISHED",
    "EVENT_PROCESSING_SECONDS",
    "PERCEPTION_DROPPED_FRAMES",
    "PERCEPTION_FPS",
    "SAFETY_VIOLATIONS",
    "SENTINEL_REGISTRY",
    "observe_duration",
    "start_metrics_server",
]
