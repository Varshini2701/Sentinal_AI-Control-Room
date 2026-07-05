"""Tests for the Decision Agent: event wiring, intent, fairness accrual over time."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sentinel.contracts.enums import DecisionAction, DensityLevel, Direction, SignalPhase
from sentinel.contracts.events import DecisionMade, StateUpdated
from sentinel.contracts.value_objects import IntersectionState, LaneState
from sentinel.decision import DecisionAgent
from sentinel.messaging import InMemoryEventBus

_T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _lane(direction: Direction, count: int, *, emergency: bool = False) -> LaneState:
    return LaneState(
        direction=direction,
        vehicle_count=count,
        moving_count=0,
        stopped_count=count,
        queue_length_m=count * 7.0,
        avg_wait_s=0.0,
        occupancy_pct=min(100.0, count * 3.0),
        density=DensityLevel.MODERATE,
        has_emergency_vehicle=emergency,
    )


def _state(
    *, ns: int, ew: int, ts: datetime, emergency_dir: Direction | None = None
) -> IntersectionState:
    return IntersectionState(
        intersection_id="i-1",
        timestamp=ts,
        lanes={
            Direction.NORTH: _lane(Direction.NORTH, ns, emergency=emergency_dir is Direction.NORTH),
            Direction.SOUTH: _lane(Direction.SOUTH, 0),
            Direction.EAST: _lane(Direction.EAST, ew, emergency=emergency_dir is Direction.EAST),
            Direction.WEST: _lane(Direction.WEST, 0),
        },
        current_phase=SignalPhase.NS_GREEN,
        phase_elapsed_s=10.0,
    )


class TestDecisionAgent:
    async def test_emits_decision_for_state(self) -> None:
        decisions: list[DecisionMade] = []

        async def capture(event: DecisionMade) -> None:  # type: ignore[override]
            decisions.append(event)

        bus = InMemoryEventBus()
        bus.subscribe("decision.made", capture, consumer_name="cap")
        DecisionAgent(event_bus=bus, intersection_id="i-1", heartbeat_interval_s=0.0)

        async with bus:
            evt = StateUpdated(
                source="perception", intersection_id="i-1", state=_state(ns=8, ew=1, ts=_T0)
            )
            await bus.publish(evt)
            await bus.join()

        assert len(decisions) == 1
        command = decisions[0].command
        assert command.target_phase.is_green
        assert decisions[0].causation_id == evt.event_id
        assert "score_ns" in command.feature_snapshot

    async def test_emergency_produces_override(self) -> None:
        decisions: list[DecisionMade] = []

        async def capture(event: DecisionMade) -> None:  # type: ignore[override]
            decisions.append(event)

        bus = InMemoryEventBus()
        bus.subscribe("decision.made", capture, consumer_name="cap")
        DecisionAgent(event_bus=bus, intersection_id="i-1", heartbeat_interval_s=0.0)

        async with bus:
            await bus.publish(
                StateUpdated(
                    source="perception",
                    intersection_id="i-1",
                    state=_state(ns=20, ew=0, ts=_T0, emergency_dir=Direction.EAST),
                )
            )
            await bus.join()

        assert decisions[-1].command.action is DecisionAction.EMERGENCY_OVERRIDE
        assert decisions[-1].command.target_phase is SignalPhase.EW_GREEN

    async def test_fairness_accrues_over_time_and_switches(self) -> None:
        decisions: list[DecisionMade] = []

        async def capture(event: DecisionMade) -> None:  # type: ignore[override]
            decisions.append(event)

        bus = InMemoryEventBus()
        bus.subscribe("decision.made", capture, consumer_name="cap")
        DecisionAgent(event_bus=bus, intersection_id="i-1", heartbeat_interval_s=0.0)

        async with bus:
            # Equal light demand; NS is green. First tick keeps NS.
            await bus.publish(
                StateUpdated(source="p", intersection_id="i-1", state=_state(ns=3, ew=3, ts=_T0))
            )
            await bus.join()
            # 200 s later, EW has waited long for green -> fairness flips the decision.
            await bus.publish(
                StateUpdated(
                    source="p",
                    intersection_id="i-1",
                    state=_state(ns=3, ew=3, ts=_T0 + timedelta(seconds=200)),
                )
            )
            await bus.join()

        assert decisions[0].command.target_phase is SignalPhase.NS_GREEN
        assert decisions[-1].command.target_phase is SignalPhase.EW_GREEN
        assert decisions[-1].command.action is DecisionAction.SWITCH_PHASE
