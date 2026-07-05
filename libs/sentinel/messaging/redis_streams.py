"""Redis Streams :class:`EventBus` transport for the high-rate perception state plane.

Redis Streams give us consumer groups (load balancing + at-least-once) and native length
trimming (``XADD MAXLEN``) -- ideal for the ~5-10 Hz ``state.updated`` firehose where RabbitMQ's
per-message overhead would be wasteful. One stream per event type; each consumer joins a named
consumer group so multiple replicas share the load without duplicating work.

``redis`` is imported lazily so the foundation package imports without the optional ``redis``
extra. Install with ``pip install "sentinel-core[redis]"``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from sentinel.config.settings import RedisSettings
from sentinel.contracts.events import EVENT_REGISTRY, DomainEvent
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
    import redis.asyncio as aioredis

_log = get_logger("sentinel.messaging.redis_streams")

_STREAM_PREFIX = "sentinel:stream:"
_DLQ_STREAM = "sentinel:stream:dlq"
_BLOCK_MS = 2000


def _stream_key(event_type: str) -> str:
    return f"{_STREAM_PREFIX}{event_type}"


@dataclass(slots=True)
class _ConsumerBinding:
    subscription: Subscription
    cache: IdempotencyCache
    tasks: list[asyncio.Task[None]] = field(default_factory=list)


class RedisStreamBus(EventBus):
    """Consumer-group event bus over Redis Streams."""

    def __init__(
        self,
        settings: RedisSettings,
        *,
        consumer_id: str = "worker-1",
        retry_policy: RetryPolicy | None = None,
        idempotency_cache_size: int = 10_000,
    ) -> None:
        self._settings = settings
        self._consumer_id = consumer_id
        self._retry = retry_policy or RetryPolicy()
        self._cache_size = idempotency_cache_size
        self._bindings: list[_ConsumerBinding] = []
        self._redis: aioredis.Redis | None = None
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
        self._bindings.append(
            _ConsumerBinding(subscription=subscription, cache=IdempotencyCache(self._cache_size))
        )
        return subscription

    # -- lifecycle ---------------------------------------------------------
    async def start(self) -> None:
        if self._running:
            return
        import redis.asyncio as aioredis

        self._redis = aioredis.from_url(self._settings.url, decode_responses=True)
        self._running = True

        for binding in self._bindings:
            for event_type in self._resolve_types(binding.subscription.event_types):
                await self._ensure_group(event_type, binding.subscription.consumer_name)
                binding.tasks.append(
                    asyncio.create_task(
                        self._read_loop(binding, event_type),
                        name=f"redis-{binding.subscription.consumer_name}-{event_type}",
                    )
                )
        _log.info("event_bus_started", transport="redis_streams", consumers=len(self._bindings))

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        for binding in self._bindings:
            for task in binding.tasks:
                task.cancel()
            for task in binding.tasks:
                with contextlib.suppress(asyncio.CancelledError):
                    await task  # cancellation is expected on shutdown
            binding.tasks.clear()
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None
        _log.info("event_bus_stopped", transport="redis_streams")

    # -- publishing --------------------------------------------------------
    async def publish(self, event: DomainEvent, *, idempotency_key: str | None = None) -> None:
        if self._redis is None:
            raise RuntimeError("RedisStreamBus.publish called before start()")
        envelope = EventEnvelope(event=event, idempotency_key=idempotency_key or event.event_id)
        await self._redis.xadd(
            _stream_key(event.event_type),
            {"data": json.dumps(envelope.serialize())},
            maxlen=self._settings.stream_maxlen,
            approximate=True,
        )
        EVENTS_PUBLISHED.labels(event_type=event.event_type, source=event.source).inc()
        _log.debug("event_published", event_type=event.event_type, event_id=event.event_id)

    # -- internals ---------------------------------------------------------
    @staticmethod
    def _resolve_types(event_types: frozenset[str]) -> list[str]:
        if WILDCARD in event_types:
            return sorted(EVENT_REGISTRY.keys())
        return sorted(event_types)

    async def _ensure_group(self, event_type: str, group: str) -> None:
        assert self._redis is not None
        try:
            await self._redis.xgroup_create(
                _stream_key(event_type), group, id="$", mkstream=True
            )
        except Exception as exc:  # BUSYGROUP -> group already exists, which is fine
            if "BUSYGROUP" not in str(exc):
                raise

    async def _read_loop(self, binding: _ConsumerBinding, event_type: str) -> None:
        assert self._redis is not None
        group = binding.subscription.consumer_name
        stream = _stream_key(event_type)
        while self._running:
            try:
                response = await self._redis.xreadgroup(
                    group, self._consumer_id, {stream: ">"}, count=16, block=_BLOCK_MS
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # transient read errors must not kill the loop
                _log.warning("stream_read_error", stream=stream, error=str(exc))
                await asyncio.sleep(0.5)
                continue
            if not response:
                continue
            for _stream_name, messages in response:
                for message_id, fields in messages:
                    await self._handle(binding, stream, group, message_id, fields)

    async def _handle(
        self,
        binding: _ConsumerBinding,
        stream: str,
        group: str,
        message_id: str,
        fields: dict[str, Any],
    ) -> None:
        assert self._redis is not None
        sub = binding.subscription
        try:
            envelope = EventEnvelope.deserialize(json.loads(fields["data"]))
        except Exception as exc:  # malformed -> DLQ + ack so it is not redelivered
            _log.error("event_deserialize_failed", consumer=group, error=str(exc))
            await self._to_dlq(fields.get("data", ""), "deserialize_failed")
            await self._redis.xack(stream, group, message_id)
            return

        event = envelope.event
        if binding.cache.seen(envelope.idempotency_key):
            await self._redis.xack(stream, group, message_id)
            return

        attempt = 0
        while True:
            try:
                with observe_duration(
                    EVENT_PROCESSING_SECONDS, event_type=event.event_type, consumer=group
                ):
                    await sub.handler(event)
            except Exception as exc:  # bus owns handler failure policy
                if self._retry.should_retry(attempt):
                    attempt += 1
                    EVENTS_CONSUMED.labels(
                        event_type=event.event_type, consumer=group, outcome="retry"
                    ).inc()
                    _log.warning(
                        "event_handler_retry",
                        consumer=group,
                        event_id=event.event_id,
                        attempt=attempt,
                        error=str(exc),
                    )
                    await asyncio.sleep(self._retry.delay_for(attempt))
                    continue
                EVENTS_CONSUMED.labels(
                    event_type=event.event_type, consumer=group, outcome="dlq"
                ).inc()
                await self._to_dlq(json.dumps(envelope.serialize()), str(exc))
                break
            else:
                binding.cache.add(envelope.idempotency_key)
                EVENTS_CONSUMED.labels(
                    event_type=event.event_type, consumer=group, outcome="success"
                ).inc()
                break
        await self._redis.xack(stream, group, message_id)

    async def _to_dlq(self, data: str, reason: str) -> None:
        assert self._redis is not None
        await self._redis.xadd(
            _DLQ_STREAM,
            {"data": data, "reason": reason},
            maxlen=self._settings.stream_maxlen,
            approximate=True,
        )
        DLQ_MESSAGES.labels(event_type="unknown").inc()
        _log.error("event_dead_lettered", reason=reason)


__all__ = ["RedisStreamBus"]
