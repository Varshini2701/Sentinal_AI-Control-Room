"""RabbitMQ :class:`EventBus` transport for the inter-agent event plane.

Uses a durable **topic exchange**; events route on ``<event_type>.<intersection_id>`` so a
consumer can subscribe to an event type across all intersections (``state.updated.#``) or scope
to one. Delivery is at-least-once with manual acks; failed handling is retried with backoff
(republish with an incremented ``retry_count``) and finally dead-lettered to a dedicated DLQ.

``aio_pika`` is imported lazily so the foundation package imports cleanly without the optional
``rabbitmq`` extra. Install with ``pip install "sentinel-core[rabbitmq]"``.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from sentinel.config.settings import RabbitMqSettings
from sentinel.contracts.events import DomainEvent
from sentinel.messaging.bus import (
    WILDCARD,
    EventBus,
    EventEnvelope,
    EventHandler,
    IdempotencyCache,
    RetryPolicy,
    Subscription,
)
from sentinel.observability.logging import get_logger
from sentinel.observability.metrics import (
    DLQ_MESSAGES,
    EVENT_PROCESSING_SECONDS,
    EVENTS_CONSUMED,
    EVENTS_PUBLISHED,
    observe_duration,
)

if TYPE_CHECKING:  # pragma: no cover
    import aio_pika

_log = get_logger("sentinel.messaging.rabbitmq")

_DLQ_SUFFIX = ".dlq"


@dataclass(slots=True)
class _PendingConsumer:
    subscription: Subscription
    cache: IdempotencyCache


class RabbitMqEventBus(EventBus):
    """Topic-exchange event bus over RabbitMQ (aio-pika)."""

    def __init__(
        self,
        settings: RabbitMqSettings,
        *,
        retry_policy: RetryPolicy | None = None,
        idempotency_cache_size: int = 10_000,
    ) -> None:
        self._settings = settings
        self._retry = retry_policy or RetryPolicy()
        self._cache_size = idempotency_cache_size
        self._consumers: list[_PendingConsumer] = []
        self._connection: aio_pika.abc.AbstractRobustConnection | None = None
        self._channel: aio_pika.abc.AbstractChannel | None = None
        self._exchange: aio_pika.abc.AbstractExchange | None = None
        self._dlx: aio_pika.abc.AbstractExchange | None = None
        self._running = False

    # -- registration -----------------------------------------------------
    def subscribe(
        self,
        event_types: Iterable[str] | str,
        handler: EventHandler,
        *,
        consumer_name: str,
    ) -> Subscription:
        subscription = Subscription(
            consumer_name=consumer_name,
            event_types=self._normalise_types(event_types),
            handler=handler,
        )
        self._consumers.append(
            _PendingConsumer(subscription=subscription, cache=IdempotencyCache(self._cache_size))
        )
        return subscription

    # -- lifecycle ---------------------------------------------------------
    async def start(self) -> None:
        if self._running:
            return
        import aio_pika  # lazy: only required when RabbitMQ transport is actually used

        self._connection = await aio_pika.connect_robust(
            self._settings.url, timeout=self._settings.connection_timeout_s
        )
        self._channel = await self._connection.channel()
        await self._channel.set_qos(prefetch_count=self._settings.prefetch_count)

        self._exchange = await self._channel.declare_exchange(
            self._settings.exchange, aio_pika.ExchangeType.TOPIC, durable=True
        )
        self._dlx = await self._channel.declare_exchange(
            f"{self._settings.exchange}{_DLQ_SUFFIX}", aio_pika.ExchangeType.TOPIC, durable=True
        )

        for pending in self._consumers:
            await self._bind_consumer(pending)

        self._running = True
        _log.info(
            "event_bus_started",
            transport="rabbitmq",
            exchange=self._settings.exchange,
            consumers=len(self._consumers),
        )

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        if self._connection is not None:
            await self._connection.close()
        self._connection = self._channel = self._exchange = self._dlx = None
        _log.info("event_bus_stopped", transport="rabbitmq")

    # -- publishing --------------------------------------------------------
    async def publish(self, event: DomainEvent, *, idempotency_key: str | None = None) -> None:
        if self._exchange is None:
            raise RuntimeError("RabbitMqEventBus.publish called before start()")
        import aio_pika

        envelope = EventEnvelope(event=event, idempotency_key=idempotency_key or event.event_id)
        message = aio_pika.Message(
            body=json.dumps(envelope.serialize()).encode("utf-8"),
            content_type="application/json",
            message_id=event.event_id,
            correlation_id=event.correlation_id,
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            headers={"idempotency_key": envelope.idempotency_key, "event_type": event.event_type},
        )
        await self._exchange.publish(message, routing_key=event.routing_key)
        EVENTS_PUBLISHED.labels(event_type=event.event_type, source=event.source).inc()
        _log.debug("event_published", event_type=event.event_type, event_id=event.event_id)

    # -- internals ---------------------------------------------------------
    async def _bind_consumer(self, pending: _PendingConsumer) -> None:
        assert self._channel is not None and self._exchange is not None and self._dlx is not None
        sub = pending.subscription
        dlq_name = f"{sub.consumer_name}{_DLQ_SUFFIX}"
        dlq = await self._channel.declare_queue(dlq_name, durable=True)
        await dlq.bind(self._dlx, routing_key=f"{sub.consumer_name}.#")

        queue = await self._channel.declare_queue(
            sub.consumer_name,
            durable=True,
            arguments={
                "x-dead-letter-exchange": self._dlx.name,
                "x-dead-letter-routing-key": f"{sub.consumer_name}.dead",
            },
        )
        for pattern in self._binding_patterns(sub.event_types):
            await queue.bind(self._exchange, routing_key=pattern)

        await queue.consume(self._make_callback(pending))
        _log.info("consumer_bound", consumer=sub.consumer_name, types=sorted(sub.event_types))

    @staticmethod
    def _binding_patterns(event_types: frozenset[str]) -> list[str]:
        if WILDCARD in event_types:
            return ["#"]
        return [f"{event_type}.#" for event_type in sorted(event_types)]

    def _make_callback(
        self, pending: _PendingConsumer
    ) -> Any:  # returns an aio_pika message handler
        async def _on_message(message: aio_pika.abc.AbstractIncomingMessage) -> None:
            async with message.process(requeue=False, ignore_processed=True):
                await self._handle_message(pending, message)

        return _on_message

    async def _handle_message(
        self, pending: _PendingConsumer, message: aio_pika.abc.AbstractIncomingMessage
    ) -> None:
        sub = pending.subscription
        try:
            envelope = EventEnvelope.deserialize(json.loads(message.body.decode("utf-8")))
        except Exception as exc:  # malformed payload -> straight to DLQ, cannot retry
            _log.error("event_deserialize_failed", consumer=sub.consumer_name, error=str(exc))
            await self._to_dlq(sub.consumer_name, message.body, "deserialize_failed")
            return

        event = envelope.event
        if pending.cache.seen(envelope.idempotency_key):
            _log.debug("event_deduplicated", consumer=sub.consumer_name, event_id=event.event_id)
            return

        try:
            with observe_duration(
                EVENT_PROCESSING_SECONDS, event_type=event.event_type, consumer=sub.consumer_name
            ):
                await sub.handler(event)
        except Exception as exc:  # bus owns handler failure policy
            await self._on_handler_failure(pending, envelope, exc)
            return

        pending.cache.add(envelope.idempotency_key)
        EVENTS_CONSUMED.labels(
            event_type=event.event_type, consumer=sub.consumer_name, outcome="success"
        ).inc()

    async def _on_handler_failure(
        self, pending: _PendingConsumer, envelope: EventEnvelope, exc: Exception
    ) -> None:
        sub = pending.subscription
        if self._retry.should_retry(envelope.retry_count):
            envelope.retry_count += 1
            EVENTS_CONSUMED.labels(
                event_type=envelope.event_type, consumer=sub.consumer_name, outcome="retry"
            ).inc()
            _log.warning(
                "event_handler_retry",
                consumer=sub.consumer_name,
                event_id=envelope.event.event_id,
                attempt=envelope.retry_count,
                error=str(exc),
            )
            await asyncio.sleep(self._retry.delay_for(envelope.retry_count))
            await self._republish(envelope)
        else:
            EVENTS_CONSUMED.labels(
                event_type=envelope.event_type, consumer=sub.consumer_name, outcome="dlq"
            ).inc()
            await self._to_dlq(
                sub.consumer_name, json.dumps(envelope.serialize()).encode("utf-8"), str(exc)
            )

    async def _republish(self, envelope: EventEnvelope) -> None:
        assert self._exchange is not None
        import aio_pika

        message = aio_pika.Message(
            body=json.dumps(envelope.serialize()).encode("utf-8"),
            content_type="application/json",
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            headers={"idempotency_key": envelope.idempotency_key, "retry": envelope.retry_count},
        )
        await self._exchange.publish(message, routing_key=envelope.routing_key)

    async def _to_dlq(self, consumer_name: str, body: bytes, reason: str) -> None:
        assert self._dlx is not None
        import aio_pika

        await self._dlx.publish(
            aio_pika.Message(body=body, headers={"reason": reason}),
            routing_key=f"{consumer_name}.dead",
        )
        DLQ_MESSAGES.labels(event_type="unknown").inc()
        _log.error("event_dead_lettered", consumer=consumer_name, reason=reason)


__all__ = ["RabbitMqEventBus"]
