"""The event-bus port and shared transport-agnostic building blocks.

Sentinel uses two transports behind one :class:`EventBus` interface:

* **RabbitMQ** for inter-agent events (moderate rate, needs topic routing + DLQ).
* **Redis Streams** for high-rate perception state deltas (needs consumer groups + trimming).

Both -- and the :class:`~sentinel.messaging.inmemory.InMemoryEventBus` used in tests -- share the
same delivery contract defined here: **at-least-once** delivery, **idempotent** consumption via an
``idempotency_key`` (defaulting to the event id), and **retry-with-backoff** that dead-letters a
message after a bounded number of attempts.
"""

from __future__ import annotations

import abc
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from sentinel.contracts.events import DomainEvent, deserialize_event

EventHandler = Callable[[DomainEvent], Awaitable[None]]
"""An async callable that processes one event. Raising signals a processing failure (retry)."""

WILDCARD = "*"
"""Subscribe to every event type."""


@dataclass(slots=True)
class EventEnvelope:
    """Transport wrapper around a :class:`DomainEvent` carrying delivery metadata."""

    event: DomainEvent
    idempotency_key: str = ""
    retry_count: int = 0
    headers: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.idempotency_key:
            self.idempotency_key = self.event.event_id

    @property
    def event_type(self) -> str:
        return self.event.event_type

    @property
    def routing_key(self) -> str:
        return self.event.routing_key

    def serialize(self) -> dict[str, Any]:
        """Produce a JSON-serialisable dict for wire transport."""
        return {
            "idempotency_key": self.idempotency_key,
            "retry_count": self.retry_count,
            "headers": self.headers,
            **self.event.to_envelope_dict(),
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> EventEnvelope:
        """Reconstruct an envelope (and its concrete event) from a wire dict."""
        return cls(
            event=deserialize_event(data),
            idempotency_key=data.get("idempotency_key", ""),
            retry_count=int(data.get("retry_count", 0)),
            headers=dict(data.get("headers", {})),
        )


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Bounded exponential-backoff policy for failed event processing."""

    max_retries: int = 3
    base_delay_s: float = 0.1
    backoff_factor: float = 2.0
    max_delay_s: float = 5.0

    def delay_for(self, attempt: int) -> float:
        """Return the backoff delay before the given (1-based) retry ``attempt``."""
        delay = self.base_delay_s * (self.backoff_factor ** max(0, attempt - 1))
        return min(delay, self.max_delay_s)

    def should_retry(self, retry_count: int) -> bool:
        """Whether a message that has already been retried ``retry_count`` times may retry again."""
        return retry_count < self.max_retries


class IdempotencyCache:
    """A bounded LRU set of recently-seen idempotency keys, per consumer.

    Guarantees that a redelivered message (at-least-once transports redeliver on ack loss) is
    processed at most once within the cache window. Not a distributed dedup store -- for
    multi-replica consumers a shared store is wired in the deployment module -- but correct and
    sufficient for a single consumer instance and for tests.
    """

    def __init__(self, max_size: int = 10_000) -> None:
        if max_size < 1:
            raise ValueError("max_size must be >= 1")
        self._max_size = max_size
        self._keys: OrderedDict[str, None] = OrderedDict()

    def seen(self, key: str) -> bool:
        """Return ``True`` if ``key`` was already recorded (marking it as recently used)."""
        if key in self._keys:
            self._keys.move_to_end(key)
            return True
        return False

    def add(self, key: str) -> None:
        """Record ``key`` as processed, evicting the oldest entry if at capacity."""
        self._keys[key] = None
        self._keys.move_to_end(key)
        if len(self._keys) > self._max_size:
            self._keys.popitem(last=False)

    def __len__(self) -> int:
        return len(self._keys)


@dataclass(slots=True)
class Subscription:
    """A registered interest in one or more event types by a named consumer."""

    consumer_name: str
    event_types: frozenset[str]
    handler: EventHandler

    def matches(self, event_type: str) -> bool:
        return WILDCARD in self.event_types or event_type in self.event_types


class EventBus(abc.ABC):
    """Abstract event-bus port. Implementations provide the concrete transport.

    Lifecycle: register interest with :meth:`subscribe`, then :meth:`start` to begin consuming;
    :meth:`publish` may be called any time after construction. Always :meth:`stop` to drain and
    release resources -- or use the async context-manager form.
    """

    @abc.abstractmethod
    async def publish(self, event: DomainEvent, *, idempotency_key: str | None = None) -> None:
        """Publish ``event`` to all interested subscribers (at-least-once)."""

    @abc.abstractmethod
    def subscribe(
        self,
        event_types: Iterable[str] | str,
        handler: EventHandler,
        *,
        consumer_name: str,
    ) -> Subscription:
        """Register ``handler`` for ``event_types`` under the given ``consumer_name``.

        ``event_types`` may be a single type, an iterable of types, or :data:`WILDCARD`.
        """

    @abc.abstractmethod
    async def start(self) -> None:
        """Begin consuming for all registered subscriptions."""

    @abc.abstractmethod
    async def stop(self) -> None:
        """Stop consuming and release transport resources."""

    @staticmethod
    def _normalise_types(event_types: Iterable[str] | str) -> frozenset[str]:
        if isinstance(event_types, str):
            return frozenset({event_types})
        return frozenset(event_types)

    async def __aenter__(self) -> EventBus:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.stop()


__all__ = [
    "WILDCARD",
    "EventBus",
    "EventEnvelope",
    "EventHandler",
    "IdempotencyCache",
    "RetryPolicy",
    "Subscription",
]
