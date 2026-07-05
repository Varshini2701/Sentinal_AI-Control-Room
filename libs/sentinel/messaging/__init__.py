"""Event-bus abstractions and transports (in-memory, RabbitMQ, Redis Streams).

The RabbitMQ and Redis implementations require the optional ``rabbitmq`` / ``redis`` extras and
import their client libraries lazily; importing this package never requires them.
"""

from __future__ import annotations

from sentinel.messaging.bus import (
    WILDCARD,
    EventBus,
    EventEnvelope,
    EventHandler,
    IdempotencyCache,
    RetryPolicy,
    Subscription,
)
from sentinel.messaging.inmemory import DeadLetterHandler, InMemoryEventBus

__all__ = [
    "WILDCARD",
    "DeadLetterHandler",
    "EventBus",
    "EventEnvelope",
    "EventHandler",
    "IdempotencyCache",
    "InMemoryEventBus",
    "RetryPolicy",
    "Subscription",
]
