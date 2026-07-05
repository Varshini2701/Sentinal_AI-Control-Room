"""Tests for the event-bus contract, exercised via the in-memory transport.

Behaviour verified here (at-least-once, idempotent dedup, retry-with-backoff, dead-lettering) is
the contract every transport implements, so these tests guard the RabbitMQ and Redis transports
by proxy.
"""

from __future__ import annotations

import asyncio

import pytest
from tests.conftest import make_state_event

from sentinel.contracts import StateUpdated, deserialize_event
from sentinel.messaging import (
    WILDCARD,
    EventEnvelope,
    IdempotencyCache,
    InMemoryEventBus,
    RetryPolicy,
)


class TestEventEnvelope:
    def test_default_idempotency_key_is_event_id(self) -> None:
        event = make_state_event()
        env = EventEnvelope(event=event)
        assert env.idempotency_key == event.event_id
        assert env.event_type == "state.updated"

    def test_serialize_deserialize_roundtrip(self) -> None:
        event = make_state_event()
        env = EventEnvelope(event=event, retry_count=2, headers={"k": "v"})
        restored = EventEnvelope.deserialize(env.serialize())
        assert restored.idempotency_key == env.idempotency_key
        assert restored.retry_count == 2
        assert restored.headers == {"k": "v"}
        assert isinstance(restored.event, StateUpdated)
        assert restored.event.event_id == event.event_id

    def test_serialized_payload_is_deserializable_as_event(self) -> None:
        env = EventEnvelope(event=make_state_event())
        restored_event = deserialize_event(env.serialize())
        assert isinstance(restored_event, StateUpdated)


class TestIdempotencyCache:
    def test_seen_and_add(self) -> None:
        cache = IdempotencyCache(max_size=3)
        assert cache.seen("a") is False
        cache.add("a")
        assert cache.seen("a") is True

    def test_lru_eviction(self) -> None:
        cache = IdempotencyCache(max_size=2)
        cache.add("a")
        cache.add("b")
        assert cache.seen("a") is True  # touch 'a' -> most recently used
        cache.add("c")  # evicts least-recently-used, which is 'b'
        assert cache.seen("b") is False
        assert cache.seen("a") is True
        assert cache.seen("c") is True

    def test_rejects_bad_size(self) -> None:
        with pytest.raises(ValueError):
            IdempotencyCache(max_size=0)


class TestRetryPolicy:
    def test_backoff_grows_and_caps(self) -> None:
        policy = RetryPolicy(max_retries=5, base_delay_s=0.1, backoff_factor=2.0, max_delay_s=0.5)
        assert policy.delay_for(1) == pytest.approx(0.1)
        assert policy.delay_for(2) == pytest.approx(0.2)
        assert policy.delay_for(3) == pytest.approx(0.4)
        assert policy.delay_for(4) == pytest.approx(0.5)  # capped

    def test_should_retry_boundary(self) -> None:
        policy = RetryPolicy(max_retries=2)
        assert policy.should_retry(0) is True
        assert policy.should_retry(1) is True
        assert policy.should_retry(2) is False


class TestInMemoryEventBus:
    async def test_publish_delivers_to_subscriber(self) -> None:
        received: list[StateUpdated] = []

        async def handler(event: StateUpdated) -> None:  # type: ignore[override]
            received.append(event)

        bus = InMemoryEventBus()
        bus.subscribe("state.updated", handler, consumer_name="c1")
        async with bus:
            event = make_state_event()
            await bus.publish(event)
            await bus.join()
        assert len(received) == 1
        assert received[0].event_id == event.event_id

    async def test_only_matching_types_delivered(self) -> None:
        got: list[str] = []

        async def handler(event: StateUpdated) -> None:  # type: ignore[override]
            got.append(event.event_type)

        bus = InMemoryEventBus()
        bus.subscribe("decision.made", handler, consumer_name="c1")  # different type
        async with bus:
            await bus.publish(make_state_event())
            await bus.join()
        assert got == []

    async def test_wildcard_subscription(self) -> None:
        count = 0

        async def handler(_event: StateUpdated) -> None:  # type: ignore[override]
            nonlocal count
            count += 1

        bus = InMemoryEventBus()
        bus.subscribe(WILDCARD, handler, consumer_name="c1")
        async with bus:
            await bus.publish(make_state_event())
            await bus.join()
        assert count == 1

    async def test_idempotent_dedup(self) -> None:
        calls = 0

        async def handler(_event: StateUpdated) -> None:  # type: ignore[override]
            nonlocal calls
            calls += 1

        bus = InMemoryEventBus()
        bus.subscribe("state.updated", handler, consumer_name="c1")
        async with bus:
            event = make_state_event()
            key = "dup-key"
            await bus.publish(event, idempotency_key=key)
            await bus.publish(event, idempotency_key=key)  # duplicate
            await bus.join()
        assert calls == 1

    async def test_retry_then_success(self) -> None:
        attempts = 0

        async def flaky(_event: StateUpdated) -> None:  # type: ignore[override]
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise RuntimeError("transient")

        bus = InMemoryEventBus(retry_policy=RetryPolicy(max_retries=3, base_delay_s=0.0))
        bus.subscribe("state.updated", flaky, consumer_name="c1")
        async with bus:
            await bus.publish(make_state_event())
            await bus.join()
        assert attempts == 3
        assert bus.dead_letters("c1") == []

    async def test_dead_letter_after_exhaustion(self) -> None:
        dlq: list[EventEnvelope] = []

        async def always_fail(_event: StateUpdated) -> None:  # type: ignore[override]
            raise RuntimeError("permanent")

        async def dlq_handler(env: EventEnvelope, _exc: BaseException) -> None:
            dlq.append(env)

        bus = InMemoryEventBus(
            retry_policy=RetryPolicy(max_retries=2, base_delay_s=0.0),
            dead_letter_handler=dlq_handler,
        )
        bus.subscribe("state.updated", always_fail, consumer_name="c1")
        async with bus:
            event = make_state_event()
            await bus.publish(event)
            await bus.join()
        assert len(dlq) == 1
        assert dlq[0].event.event_id == event.event_id
        assert len(bus.dead_letters("c1")) == 1

    async def test_multiple_consumers_each_receive(self) -> None:
        a: list[str] = []
        b: list[str] = []

        async def ha(e: StateUpdated) -> None:  # type: ignore[override]
            a.append(e.event_id)

        async def hb(e: StateUpdated) -> None:  # type: ignore[override]
            b.append(e.event_id)

        bus = InMemoryEventBus()
        bus.subscribe("state.updated", ha, consumer_name="c1")
        bus.subscribe("state.updated", hb, consumer_name="c2")
        async with bus:
            await bus.publish(make_state_event())
            await bus.join()
        assert len(a) == 1 and len(b) == 1

    async def test_subscribe_after_start(self) -> None:
        received: list[str] = []

        async def handler(e: StateUpdated) -> None:  # type: ignore[override]
            received.append(e.event_id)

        bus = InMemoryEventBus()
        await bus.start()
        try:
            bus.subscribe("state.updated", handler, consumer_name="late")
            await bus.publish(make_state_event())
            await bus.join()
        finally:
            await bus.stop()
        assert len(received) == 1

    async def test_publish_before_start_is_buffered(self) -> None:
        received: list[str] = []

        async def handler(e: StateUpdated) -> None:  # type: ignore[override]
            received.append(e.event_id)

        bus = InMemoryEventBus()
        bus.subscribe("state.updated", handler, consumer_name="c1")
        await bus.publish(make_state_event())  # buffered before start
        await bus.start()
        await bus.join()
        await bus.stop()
        assert len(received) == 1

    async def test_stop_is_idempotent(self) -> None:
        bus = InMemoryEventBus()
        await bus.start()
        await bus.stop()
        await bus.stop()  # must not raise

    async def test_concurrent_publish(self) -> None:
        received: list[str] = []

        async def handler(e: StateUpdated) -> None:  # type: ignore[override]
            received.append(e.event_id)

        bus = InMemoryEventBus()
        bus.subscribe("state.updated", handler, consumer_name="c1")
        async with bus:
            await asyncio.gather(*(bus.publish(make_state_event()) for _ in range(20)))
            await bus.join()
        assert len(received) == 20
