"""In-process :class:`EventBus` implementation.

Used for unit/integration tests and for single-process local runs. It implements the *full*
delivery contract -- at-least-once dispatch, per-consumer idempotent dedup, retry-with-backoff and
dead-lettering -- so behaviour verified here matches the RabbitMQ/Redis transports.

Each subscription owns an :class:`asyncio.Queue` and a worker task; publishing fans an envelope
out to every matching subscription's queue. Messages published before :meth:`start` are buffered.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field

from sentinel.contracts.events import DomainEvent
from sentinel.messaging.bus import (
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

_log = get_logger("sentinel.messaging.inmemory")

DeadLetterHandler = Callable[[EventEnvelope, BaseException], Awaitable[None]]


@dataclass(slots=True)
class _Worker:
    subscription: Subscription
    queue: asyncio.Queue[EventEnvelope | None]
    cache: IdempotencyCache
    task: asyncio.Task[None] | None = None
    dead_letters: list[EventEnvelope] = field(default_factory=list)


class InMemoryEventBus(EventBus):
    """An asyncio-based event bus with no external dependencies."""

    def __init__(
        self,
        *,
        retry_policy: RetryPolicy | None = None,
        dead_letter_handler: DeadLetterHandler | None = None,
        idempotency_cache_size: int = 10_000,
    ) -> None:
        self._retry = retry_policy or RetryPolicy()
        self._dlq_handler = dead_letter_handler
        self._cache_size = idempotency_cache_size
        self._workers: list[_Worker] = []
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
        worker = _Worker(
            subscription=subscription,
            queue=asyncio.Queue(),
            cache=IdempotencyCache(self._cache_size),
        )
        self._workers.append(worker)
        if self._running:
            worker.task = asyncio.create_task(
                self._run_worker(worker), name=f"bus-{consumer_name}"
            )
        return subscription

    # -- lifecycle ---------------------------------------------------------
    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        for worker in self._workers:
            if worker.task is None:
                worker.task = asyncio.create_task(
                    self._run_worker(worker), name=f"bus-{worker.subscription.consumer_name}"
                )
        _log.info("event_bus_started", transport="in_memory", subscriptions=len(self._workers))

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        for worker in self._workers:
            await worker.queue.put(None)  # sentinel to unblock the worker
        for worker in self._workers:
            if worker.task is not None:
                await worker.task
                worker.task = None
        _log.info("event_bus_stopped", transport="in_memory")

    # -- publishing --------------------------------------------------------
    async def publish(self, event: DomainEvent, *, idempotency_key: str | None = None) -> None:
        envelope = EventEnvelope(event=event, idempotency_key=idempotency_key or event.event_id)
        EVENTS_PUBLISHED.labels(event_type=event.event_type, source=event.source).inc()
        delivered = 0
        for worker in self._workers:
            if worker.subscription.matches(event.event_type):
                # Fresh envelope per consumer so retry_count is independent.
                await worker.queue.put(
                    EventEnvelope(event=event, idempotency_key=envelope.idempotency_key)
                )
                delivered += 1
        _log.debug(
            "event_published",
            event_type=event.event_type,
            event_id=event.event_id,
            consumers=delivered,
        )

    # -- test / drain helpers ---------------------------------------------
    async def join(self) -> None:
        """Block until every queued message has been fully processed (test helper)."""
        for worker in self._workers:
            await worker.queue.join()

    def dead_letters(self, consumer_name: str) -> list[EventEnvelope]:
        """Return the dead-lettered envelopes for a given consumer (test/inspection helper)."""
        return [
            env
            for worker in self._workers
            if worker.subscription.consumer_name == consumer_name
            for env in worker.dead_letters
        ]

    # -- internals ---------------------------------------------------------
    async def _run_worker(self, worker: _Worker) -> None:
        while True:
            envelope = await worker.queue.get()
            if envelope is None:  # shutdown sentinel
                worker.queue.task_done()
                return
            try:
                await self._process(worker, envelope)
            finally:
                worker.queue.task_done()

    async def _process(self, worker: _Worker, envelope: EventEnvelope) -> None:
        consumer = worker.subscription.consumer_name
        event = envelope.event
        if worker.cache.seen(envelope.idempotency_key):
            _log.debug("event_deduplicated", consumer=consumer, event_id=event.event_id)
            return

        attempt = 0
        while True:
            try:
                with observe_duration(
                    EVENT_PROCESSING_SECONDS, event_type=event.event_type, consumer=consumer
                ):
                    await worker.subscription.handler(event)
            except Exception as exc:  # the bus deliberately catches all handler errors
                if self._retry.should_retry(attempt):
                    attempt += 1
                    EVENTS_CONSUMED.labels(
                        event_type=event.event_type, consumer=consumer, outcome="retry"
                    ).inc()
                    _log.warning(
                        "event_handler_retry",
                        consumer=consumer,
                        event_id=event.event_id,
                        attempt=attempt,
                        error=str(exc),
                    )
                    await asyncio.sleep(self._retry.delay_for(attempt))
                    continue
                await self._dead_letter(worker, envelope, exc)
                return
            else:
                worker.cache.add(envelope.idempotency_key)
                EVENTS_CONSUMED.labels(
                    event_type=event.event_type, consumer=consumer, outcome="success"
                ).inc()
                return

    async def _dead_letter(
        self, worker: _Worker, envelope: EventEnvelope, exc: BaseException
    ) -> None:
        worker.dead_letters.append(envelope)
        EVENTS_CONSUMED.labels(
            event_type=envelope.event_type,
            consumer=worker.subscription.consumer_name,
            outcome="dlq",
        ).inc()
        DLQ_MESSAGES.labels(event_type=envelope.event_type).inc()
        _log.error(
            "event_dead_lettered",
            consumer=worker.subscription.consumer_name,
            event_id=envelope.event.event_id,
            event_type=envelope.event_type,
            error=str(exc),
        )
        if self._dlq_handler is not None:
            await self._dlq_handler(envelope, exc)


__all__ = ["DeadLetterHandler", "InMemoryEventBus"]
