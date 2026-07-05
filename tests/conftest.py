"""Shared pytest fixtures and domain object factories for the foundation test suite."""

from __future__ import annotations

import pytest

from sentinel.contracts import (
    BoundingBox,
    DensityLevel,
    Detection,
    Direction,
    IntersectionState,
    LaneState,
    SignalPhase,
    StateUpdated,
    Track,
    VehicleClass,
    Velocity,
)


def make_lane(
    direction: Direction,
    *,
    moving: int = 3,
    stopped: int = 2,
    queue_m: float = 10.0,
) -> LaneState:
    """Build a valid :class:`LaneState` with consistent counts."""
    return LaneState(
        direction=direction,
        vehicle_count=moving + stopped,
        moving_count=moving,
        stopped_count=stopped,
        queue_length_m=queue_m,
        avg_wait_s=queue_m * 1.5,
        occupancy_pct=min(100.0, queue_m * 4.0),
        density=DensityLevel.MODERATE,
    )


def make_state(intersection_id: str = "intersection-1") -> IntersectionState:
    """Build a valid four-approach :class:`IntersectionState`."""
    return IntersectionState(
        intersection_id=intersection_id,
        lanes={d: make_lane(d) for d in Direction},
        current_phase=SignalPhase.NS_GREEN,
        phase_elapsed_s=12.0,
    )


def make_state_event(
    source: str = "perception", intersection_id: str = "intersection-1"
) -> StateUpdated:
    """Build a :class:`StateUpdated` event wrapping a sample state."""
    return StateUpdated(
        source=source,
        intersection_id=intersection_id,
        state=make_state(intersection_id),
    )


def make_box(x1: float = 0.0, y1: float = 0.0, x2: float = 10.0, y2: float = 20.0) -> BoundingBox:
    """Build a bounding box; defaults to a 10x20 box at the origin."""
    return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)


def make_detection(
    *,
    vehicle_class: VehicleClass = VehicleClass.CAR,
    confidence: float = 0.9,
    box: BoundingBox | None = None,
) -> Detection:
    return Detection(
        vehicle_class=vehicle_class, confidence=confidence, box=box or make_box()
    )


def make_track(
    track_id: int,
    *,
    box: BoundingBox | None = None,
    vx: float = 0.0,
    vy: float = 0.0,
    lane: Direction = Direction.NORTH,
    vehicle_class: VehicleClass = VehicleClass.CAR,
    is_moving: bool = False,
) -> Track:
    return Track(
        track_id=track_id,
        vehicle_class=vehicle_class,
        box=box or make_box(),
        velocity=Velocity(vx=vx, vy=vy),
        lane=lane,
        is_moving=is_moving,
    )


@pytest.fixture
def state() -> IntersectionState:
    return make_state()


@pytest.fixture
def state_event() -> StateUpdated:
    return make_state_event()
