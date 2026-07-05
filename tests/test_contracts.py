"""Tests for the domain contracts: enums, value objects and versioned events."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from tests.conftest import make_lane, make_state, make_state_event

from sentinel.contracts import (
    Axis,
    BoundingBox,
    DecisionAction,
    Direction,
    IntersectionState,
    LaneState,
    SignalPhase,
    StateUpdated,
    VehicleClass,
    deserialize_event,
)
from sentinel.contracts.enums import DensityLevel
from sentinel.contracts.events import EVENT_REGISTRY, UnknownEventTypeError


class TestEnums:
    def test_direction_opposite_and_axis(self) -> None:
        assert Direction.NORTH.opposite is Direction.SOUTH
        assert Direction.EAST.opposite is Direction.WEST
        assert Direction.NORTH.axis is Axis.NORTH_SOUTH
        assert Direction.WEST.axis is Axis.EAST_WEST

    def test_axis_directions(self) -> None:
        assert Axis.NORTH_SOUTH.directions == (Direction.NORTH, Direction.SOUTH)
        assert Axis.EAST_WEST.directions == (Direction.EAST, Direction.WEST)

    def test_phase_axis_and_green(self) -> None:
        assert SignalPhase.NS_GREEN.axis is Axis.NORTH_SOUTH
        assert SignalPhase.NS_GREEN.is_green is True
        assert SignalPhase.ALL_RED.axis is None
        assert SignalPhase.EW_YELLOW.is_green is False

    def test_vehicle_class_flags(self) -> None:
        assert VehicleClass.AMBULANCE.is_emergency is True
        assert VehicleClass.CAR.is_emergency is False
        assert VehicleClass.PEDESTRIAN.is_vehicle is False
        assert VehicleClass.BUS.is_vehicle is True

    def test_enum_serialises_to_string(self) -> None:
        assert DecisionAction.SWITCH_PHASE.value == "switch_phase"
        assert Direction.NORTH == "north"


class TestValueObjects:
    def test_bounding_box_geometry(self) -> None:
        box = BoundingBox(x1=0, y1=0, x2=10, y2=20)
        assert box.width == 10
        assert box.height == 20
        assert box.area == 200
        assert box.centroid == (5.0, 10.0)

    def test_bounding_box_rejects_inverted(self) -> None:
        with pytest.raises(ValidationError):
            BoundingBox(x1=10, y1=0, x2=5, y2=20)

    def test_lane_state_count_consistency(self) -> None:
        with pytest.raises(ValidationError):
            LaneState(
                direction=Direction.NORTH,
                vehicle_count=5,
                moving_count=3,
                stopped_count=1,  # 3 + 1 != 5
                queue_length_m=1,
                avg_wait_s=1,
                occupancy_pct=1,
                density=DensityLevel.FREE,
            )

    def test_value_object_is_frozen(self) -> None:
        lane = make_lane(Direction.NORTH)
        with pytest.raises(ValidationError):
            lane.vehicle_count = 99  # type: ignore[misc]

    def test_intersection_state_helpers(self) -> None:
        state = make_state()
        assert state.total_vehicles == 4 * 5
        assert state.total_stopped == 4 * 2
        assert state.busiest_lane() is not None
        assert state.lane(Direction.NORTH).direction is Direction.NORTH

    def test_intersection_rejects_mismatched_lane_key(self) -> None:
        with pytest.raises(ValidationError):
            IntersectionState(
                intersection_id="x",
                lanes={Direction.NORTH: make_lane(Direction.SOUTH)},
                current_phase=SignalPhase.NS_GREEN,
                phase_elapsed_s=0,
            )

    def test_emergency_lanes(self) -> None:
        lanes = {d: make_lane(d) for d in Direction}
        lanes[Direction.EAST] = LaneState(
            direction=Direction.EAST,
            vehicle_count=1,
            moving_count=1,
            stopped_count=0,
            queue_length_m=0,
            avg_wait_s=0,
            occupancy_pct=0,
            density=DensityLevel.FREE,
            has_emergency_vehicle=True,
        )
        state = IntersectionState(
            intersection_id="x",
            lanes=lanes,
            current_phase=SignalPhase.NS_GREEN,
            phase_elapsed_s=0,
        )
        assert state.emergency_lanes() == (Direction.EAST,)


class TestEvents:
    def test_all_concrete_events_registered(self) -> None:
        for event_type in ("state.updated", "decision.made", "signal.changed", "incident.detected"):
            assert event_type in EVENT_REGISTRY

    def test_routing_key(self) -> None:
        event = make_state_event(intersection_id="i-42")
        assert event.routing_key == "state.updated.i-42"

    def test_event_roundtrip_serialisation(self) -> None:
        original = make_state_event()
        wire = original.to_envelope_dict()
        restored = deserialize_event(wire)
        assert isinstance(restored, StateUpdated)
        assert restored.event_id == original.event_id
        assert restored.state == original.state

    def test_deserialize_unknown_type_raises(self) -> None:
        with pytest.raises(UnknownEventTypeError):
            deserialize_event({"event_type": "does.not.exist", "payload": {}})

    def test_event_is_immutable(self) -> None:
        event = make_state_event()
        with pytest.raises(ValidationError):
            event.source = "other"  # type: ignore[misc]

    def test_provenance_fields_optional(self) -> None:
        event = make_state_event()
        assert event.correlation_id is None
        assert event.causation_id is None
        assert len(event.event_id) == 32
