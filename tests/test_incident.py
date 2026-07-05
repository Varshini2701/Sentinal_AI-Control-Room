"""Tests for the incident rules and the Incident Detection Agent."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sentinel.config.settings import IncidentSettings
from sentinel.contracts.enums import DensityLevel, Direction, IncidentType, SignalPhase
from sentinel.contracts.events import IncidentDetected, StateUpdated
from sentinel.contracts.value_objects import (
    HistoricalContext,
    IntersectionState,
    LaneBaseline,
    LaneState,
)
from sentinel.incident import AbnormalCongestionRule, IncidentDetectionAgent, StalledVehicleRule
from sentinel.messaging import InMemoryEventBus

_T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _lane(direction: Direction, *, count: int, moving: int = 0) -> LaneState:
    return LaneState(
        direction=direction,
        vehicle_count=count,
        moving_count=moving,
        stopped_count=count - moving,
        queue_length_m=count * 7.0,
        avg_wait_s=0.0,
        occupancy_pct=min(100.0, count * 3.0),
        density=DensityLevel.MODERATE,
    )


def _state(
    *, ns: int, ns_moving: int = 0, phase: SignalPhase = SignalPhase.NS_GREEN,
    elapsed: float = 0.0, ts: datetime = _T0,
) -> IntersectionState:
    return IntersectionState(
        intersection_id="i-1",
        timestamp=ts,
        lanes={
            Direction.NORTH: _lane(Direction.NORTH, count=ns, moving=ns_moving),
            Direction.SOUTH: _lane(Direction.SOUTH, count=0),
            Direction.EAST: _lane(Direction.EAST, count=0),
            Direction.WEST: _lane(Direction.WEST, count=0),
        },
        current_phase=phase,
        phase_elapsed_s=elapsed,
    )


_SETTINGS = IncidentSettings(
    stalled_wait_s=90.0, congestion_ratio=2.5, congestion_min_baseline_samples=5
)


class TestStalledVehicleRule:
    def test_flags_stopped_lane_with_long_green_and_no_discharge(self) -> None:
        state = _state(ns=3, ns_moving=0, phase=SignalPhase.NS_GREEN, elapsed=100.0)
        incidents = StalledVehicleRule(_SETTINGS).evaluate(state, None)
        assert len(incidents) == 1
        assert incidents[0].incident_type is IncidentType.STALLED_VEHICLE
        assert incidents[0].direction is Direction.NORTH

    def test_silent_when_lane_is_discharging(self) -> None:
        state = _state(ns=3, ns_moving=1, phase=SignalPhase.NS_GREEN, elapsed=100.0)
        assert StalledVehicleRule(_SETTINGS).evaluate(state, None) == []

    def test_silent_before_wait_threshold(self) -> None:
        state = _state(ns=3, ns_moving=0, phase=SignalPhase.NS_GREEN, elapsed=10.0)
        assert StalledVehicleRule(_SETTINGS).evaluate(state, None) == []

    def test_silent_during_clearance(self) -> None:
        state = _state(ns=3, ns_moving=0, phase=SignalPhase.ALL_RED, elapsed=100.0)
        assert StalledVehicleRule(_SETTINGS).evaluate(state, None) == []

    def test_silent_for_empty_lane(self) -> None:
        state = _state(ns=0, phase=SignalPhase.NS_GREEN, elapsed=100.0)
        assert StalledVehicleRule(_SETTINGS).evaluate(state, None) == []


class TestAbnormalCongestionRule:
    def _baseline(self, avg: float, samples: int) -> HistoricalContext:
        return HistoricalContext(
            intersection_id="i-1",
            window_size=samples,
            lanes={
                Direction.NORTH: LaneBaseline(
                    direction=Direction.NORTH,
                    avg_queue_veh=avg,
                    avg_wait_s=0.0,
                    sample_count=samples,
                )
            },
        )

    def test_flags_queue_far_above_baseline(self) -> None:
        state = _state(ns=20)  # 20 vs baseline avg 5 -> ratio 4.0 >= 2.5
        baseline = self._baseline(avg=5.0, samples=10)
        incidents = AbnormalCongestionRule(_SETTINGS).evaluate(state, baseline)
        assert len(incidents) == 1
        assert incidents[0].incident_type is IncidentType.ABNORMAL_CONGESTION

    def test_silent_within_normal_range(self) -> None:
        state = _state(ns=6)  # ratio 1.2
        baseline = self._baseline(avg=5.0, samples=10)
        assert AbnormalCongestionRule(_SETTINGS).evaluate(state, baseline) == []

    def test_silent_without_baseline(self) -> None:
        state = _state(ns=100)
        assert AbnormalCongestionRule(_SETTINGS).evaluate(state, None) == []

    def test_silent_with_insufficient_samples(self) -> None:
        state = _state(ns=100)
        baseline = self._baseline(avg=5.0, samples=2)  # below congestion_min_baseline_samples
        assert AbnormalCongestionRule(_SETTINGS).evaluate(state, baseline) == []


class TestIncidentDetectionAgent:
    async def test_emits_incident_for_stalled_vehicle(self) -> None:
        incidents: list[IncidentDetected] = []

        async def capture(event: IncidentDetected) -> None:  # type: ignore[override]
            incidents.append(event)

        bus = InMemoryEventBus()
        bus.subscribe("incident.detected", capture, consumer_name="cap")
        IncidentDetectionAgent(
            event_bus=bus, intersection_id="i-1", settings=_SETTINGS, heartbeat_interval_s=0.0
        )

        async with bus:
            await bus.publish(
                StateUpdated(
                    source="perception", intersection_id="i-1",
                    state=_state(ns=3, ns_moving=0, elapsed=100.0),
                )
            )
            await bus.join()

        assert len(incidents) == 1
        assert incidents[0].incident.incident_type is IncidentType.STALLED_VEHICLE

    async def test_debounces_repeated_incident(self) -> None:
        incidents: list[IncidentDetected] = []

        async def capture(event: IncidentDetected) -> None:  # type: ignore[override]
            incidents.append(event)

        bus = InMemoryEventBus()
        bus.subscribe("incident.detected", capture, consumer_name="cap")
        settings = IncidentSettings(stalled_wait_s=90.0, debounce_s=60.0)
        IncidentDetectionAgent(
            event_bus=bus, intersection_id="i-1", settings=settings, heartbeat_interval_s=0.0
        )

        async with bus:
            for i in range(3):  # same ongoing incident, timestamps 10s apart < debounce
                await bus.publish(
                    StateUpdated(
                        source="perception", intersection_id="i-1",
                        state=_state(
                            ns=3, ns_moving=0, elapsed=100.0, ts=_T0 + timedelta(seconds=i * 10)
                        ),
                    )
                )
                await bus.join()

        assert len(incidents) == 1  # only the first raise, rest debounced

    async def test_baseline_updates_are_used_by_rules(self) -> None:
        incidents: list[IncidentDetected] = []

        async def capture(event: IncidentDetected) -> None:  # type: ignore[override]
            incidents.append(event)

        bus = InMemoryEventBus()
        bus.subscribe("incident.detected", capture, consumer_name="cap")
        settings = IncidentSettings(congestion_ratio=2.5, congestion_min_baseline_samples=1)
        agent = IncidentDetectionAgent(
            event_bus=bus, intersection_id="i-1", settings=settings, heartbeat_interval_s=0.0
        )
        from sentinel.contracts.events import BaselineUpdated

        async with bus:
            await bus.publish(
                BaselineUpdated(
                    source="memory", intersection_id="i-1",
                    baseline=HistoricalContext(
                        intersection_id="i-1", window_size=5,
                        lanes={
                            Direction.NORTH: LaneBaseline(
                                direction=Direction.NORTH, avg_queue_veh=2.0,
                                avg_wait_s=0.0, sample_count=5,
                            )
                        },
                    ),
                )
            )
            await bus.join()
            await bus.publish(
                StateUpdated(
                    source="perception", intersection_id="i-1", state=_state(ns=20, elapsed=0.0)
                )
            )
            await bus.join()

        assert any(
            e.incident.incident_type is IncidentType.ABNORMAL_CONGESTION for e in incidents
        )
        assert agent.name == "incident-detection"
