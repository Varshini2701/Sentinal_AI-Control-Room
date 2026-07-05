"""Tests for the Traffic Memory Agent and its history repository."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from sentinel.config.settings import MemorySettings
from sentinel.contracts.enums import DensityLevel, Direction, SignalPhase
from sentinel.contracts.events import BaselineUpdated, StateUpdated
from sentinel.contracts.value_objects import IntersectionState, LaneState
from sentinel.memory import InMemoryStateHistoryRepository, TrafficMemoryAgent
from sentinel.messaging import InMemoryEventBus

_T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _lane(direction: Direction, count: int, wait: float = 0.0) -> LaneState:
    return LaneState(
        direction=direction,
        vehicle_count=count,
        moving_count=0,
        stopped_count=count,
        queue_length_m=count * 7.0,
        avg_wait_s=wait,
        occupancy_pct=min(100.0, count * 3.0),
        density=DensityLevel.MODERATE,
    )


def _state(
    intersection_id: str, count: int, *, wait: float = 0.0, ts: datetime = _T0
) -> IntersectionState:
    return IntersectionState(
        intersection_id=intersection_id,
        timestamp=ts,
        lanes={d: _lane(d, count, wait=wait) for d in Direction},
        current_phase=SignalPhase.NS_GREEN,
        phase_elapsed_s=1.0,
    )


class TestInMemoryStateHistoryRepository:
    def test_append_and_recent(self) -> None:
        repo = InMemoryStateHistoryRepository(window_size=10)
        repo.append(_state("i-1", 1))
        repo.append(_state("i-1", 2))
        recent = repo.recent("i-1")
        assert len(recent) == 2
        assert recent[-1].lanes[Direction.NORTH].vehicle_count == 2

    def test_window_is_bounded(self) -> None:
        repo = InMemoryStateHistoryRepository(window_size=3)
        for i in range(10):
            repo.append(_state("i-1", i))
        assert repo.count("i-1") == 3
        assert repo.recent("i-1")[0].lanes[Direction.NORTH].vehicle_count == 7  # oldest retained

    def test_baseline_is_rolling_average(self) -> None:
        repo = InMemoryStateHistoryRepository()
        repo.append(_state("i-1", 2, wait=4.0))
        repo.append(_state("i-1", 4, wait=8.0))
        baseline = repo.baseline("i-1")
        assert baseline is not None
        north = baseline.baseline_for(Direction.NORTH)
        assert north is not None
        assert north.avg_queue_veh == 3.0
        assert north.avg_wait_s == 6.0
        assert north.sample_count == 2

    def test_baseline_empty_returns_none(self) -> None:
        repo = InMemoryStateHistoryRepository()
        assert repo.baseline("nope") is None

    def test_isolated_per_intersection(self) -> None:
        repo = InMemoryStateHistoryRepository()
        repo.append(_state("i-1", 1))
        repo.append(_state("i-2", 9))
        assert repo.count("i-1") == 1
        assert repo.count("i-2") == 1

    def test_rejects_bad_window_size(self) -> None:
        with pytest.raises(ValueError, match="window_size"):
            InMemoryStateHistoryRepository(window_size=0)


class TestTrafficMemoryAgent:
    async def test_persists_state_updates(self) -> None:
        bus = InMemoryEventBus()
        agent = TrafficMemoryAgent(
            event_bus=bus, intersection_id="i-1",
            settings=MemorySettings(baseline_publish_every=100), heartbeat_interval_s=0.0,
        )
        async with bus:
            await bus.publish(
                StateUpdated(source="perception", intersection_id="i-1", state=_state("i-1", 5))
            )
            await bus.join()
        assert agent.repository.count("i-1") == 1
        assert agent.current_baseline() is not None

    async def test_publishes_baseline_on_interval(self) -> None:
        published: list[BaselineUpdated] = []

        async def capture(event: BaselineUpdated) -> None:  # type: ignore[override]
            published.append(event)

        bus = InMemoryEventBus()
        bus.subscribe("history.baseline.updated", capture, consumer_name="cap")
        TrafficMemoryAgent(
            event_bus=bus, intersection_id="i-1",
            settings=MemorySettings(baseline_publish_every=3), heartbeat_interval_s=0.0,
        )

        async with bus:
            for i in range(3):
                await bus.publish(
                    StateUpdated(
                        source="perception",
                        intersection_id="i-1",
                        state=_state("i-1", i, ts=_T0 + timedelta(seconds=i)),
                    )
                )
                await bus.join()

        assert len(published) == 1
        assert published[0].baseline.window_size == 3

    async def test_does_not_publish_before_interval(self) -> None:
        published: list[BaselineUpdated] = []

        async def capture(event: BaselineUpdated) -> None:  # type: ignore[override]
            published.append(event)

        bus = InMemoryEventBus()
        bus.subscribe("history.baseline.updated", capture, consumer_name="cap")
        TrafficMemoryAgent(
            event_bus=bus, intersection_id="i-1",
            settings=MemorySettings(baseline_publish_every=5), heartbeat_interval_s=0.0,
        )
        async with bus:
            await bus.publish(
                StateUpdated(source="perception", intersection_id="i-1", state=_state("i-1", 1))
            )
            await bus.join()
        assert published == []
