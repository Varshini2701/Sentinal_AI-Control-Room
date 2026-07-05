"""Tests for the multi-objective utility decision policy (pure, tier 1)."""

from __future__ import annotations

from sentinel.config.settings import DecisionSettings
from sentinel.contracts.enums import (
    Axis,
    DecisionAction,
    DensityLevel,
    Direction,
    SignalPhase,
)
from sentinel.contracts.value_objects import IntersectionState, LaneState
from sentinel.decision.policy import DecisionContext, UtilityPolicy


def _lane(
    direction: Direction,
    count: int,
    *,
    wait: float = 0.0,
    emergency: bool = False,
    ped: bool = False,
) -> LaneState:
    return LaneState(
        direction=direction,
        vehicle_count=count,
        moving_count=0,
        stopped_count=count,
        queue_length_m=count * 7.0,
        avg_wait_s=wait,
        occupancy_pct=min(100.0, count * 3.0),
        density=DensityLevel.MODERATE,
        has_emergency_vehicle=emergency,
        pedestrian_waiting=ped,
    )


def _state(
    *,
    ns: int,
    ew: int,
    phase: SignalPhase = SignalPhase.NS_GREEN,
    ns_wait: float = 0.0,
    ew_wait: float = 0.0,
    emergency_dir: Direction | None = None,
    ped_dir: Direction | None = None,
) -> IntersectionState:
    return IntersectionState(
        intersection_id="i-test",
        lanes={
            Direction.NORTH: _lane(
                Direction.NORTH, ns, wait=ns_wait,
                emergency=emergency_dir is Direction.NORTH, ped=ped_dir is Direction.NORTH,
            ),
            Direction.SOUTH: _lane(Direction.SOUTH, 0),
            Direction.EAST: _lane(
                Direction.EAST, ew, wait=ew_wait,
                emergency=emergency_dir is Direction.EAST, ped=ped_dir is Direction.EAST,
            ),
            Direction.WEST: _lane(Direction.WEST, 0),
        },
        current_phase=phase,
        phase_elapsed_s=10.0,
    )


def _ctx(state: IntersectionState, **since: float) -> DecisionContext:
    return DecisionContext(
        state=state,
        since_green_s={
            Axis.NORTH_SOUTH: since.get("ns", 0.0),
            Axis.EAST_WEST: since.get("ew", 0.0),
        },
    )


_SETTINGS = DecisionSettings()


class TestUtilityPolicy:
    def test_emergency_preempts(self) -> None:
        state = _state(ns=20, ew=0, emergency_dir=Direction.EAST)
        outcome = UtilityPolicy(_SETTINGS).decide(_ctx(state))
        assert outcome.action is DecisionAction.EMERGENCY_OVERRIDE
        assert outcome.desired_axis is Axis.EAST_WEST

    def test_switch_when_opposing_much_higher(self) -> None:
        state = _state(ns=0, ew=20, phase=SignalPhase.NS_GREEN)
        outcome = UtilityPolicy(_SETTINGS).decide(_ctx(state))
        assert outcome.action is DecisionAction.SWITCH_PHASE
        assert outcome.desired_axis is Axis.EAST_WEST

    def test_extend_when_current_congested(self) -> None:
        state = _state(ns=20, ew=0, phase=SignalPhase.NS_GREEN)  # 20 >= congestion threshold
        outcome = UtilityPolicy(_SETTINGS).decide(_ctx(state))
        assert outcome.action is DecisionAction.EXTEND_GREEN
        assert outcome.desired_axis is Axis.NORTH_SOUTH

    def test_keep_when_serving_moderately(self) -> None:
        state = _state(ns=5, ew=0, phase=SignalPhase.NS_GREEN)
        outcome = UtilityPolicy(_SETTINGS).decide(_ctx(state))
        assert outcome.action is DecisionAction.KEEP_GREEN

    def test_reduce_when_current_empty_and_no_reason_to_switch(self) -> None:
        state = _state(ns=0, ew=0, phase=SignalPhase.NS_GREEN)
        outcome = UtilityPolicy(_SETTINGS).decide(_ctx(state))
        assert outcome.action is DecisionAction.REDUCE_GREEN
        assert outcome.desired_axis is Axis.NORTH_SOUTH

    def test_switch_penalty_prevents_thrashing(self) -> None:
        # EW only marginally busier than NS (within the switch penalty) -> stay put.
        state = _state(ns=5, ew=10, phase=SignalPhase.NS_GREEN)
        outcome = UtilityPolicy(_SETTINGS).decide(_ctx(state))
        assert outcome.desired_axis is Axis.NORTH_SOUTH

    def test_fairness_forces_switch_to_starved_axis(self) -> None:
        # Equal queues, but EW has waited a long time for green.
        state = _state(ns=5, ew=5, phase=SignalPhase.NS_GREEN)
        outcome = UtilityPolicy(_SETTINGS).decide(_ctx(state, ew=100.0))
        assert outcome.action is DecisionAction.SWITCH_PHASE
        assert outcome.desired_axis is Axis.EAST_WEST

    def test_pedestrian_bumps_utility(self) -> None:
        # NS green with light demand; a pedestrian waiting E-W adds a big utility bump -> switch.
        state = _state(ns=1, ew=1, phase=SignalPhase.NS_GREEN, ped_dir=Direction.EAST)
        outcome = UtilityPolicy(_SETTINGS).decide(_ctx(state))
        assert outcome.desired_axis is Axis.EAST_WEST

    def test_clearance_interval_holds(self) -> None:
        state = _state(ns=5, ew=5, phase=SignalPhase.ALL_RED)
        outcome = UtilityPolicy(_SETTINGS).decide(_ctx(state))
        assert outcome.reason_code == "clearance_interval"
        assert outcome.action is DecisionAction.KEEP_GREEN

    def test_features_and_scores_populated(self) -> None:
        state = _state(ns=8, ew=3, phase=SignalPhase.NS_GREEN)
        outcome = UtilityPolicy(_SETTINGS).decide(_ctx(state))
        assert outcome.features["queue_ns"] == 8.0
        assert outcome.features["queue_ew"] == 3.0
        assert set(outcome.scores) == {"north_south", "east_west"}
