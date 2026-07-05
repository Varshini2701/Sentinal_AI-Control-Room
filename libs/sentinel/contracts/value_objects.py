"""Immutable domain value objects shared across the Perception and Cognition planes.

The most important type here is :class:`IntersectionState` -- the compact snapshot the
Perception plane writes to Redis and every cognition agent reasons over. It is deliberately
free of pixels: it carries derived, decision-relevant quantities only.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from pydantic import Field, model_validator

from sentinel.contracts.base import FrozenModel, utcnow
from sentinel.contracts.enums import (
    AgentStatus,
    DecisionAction,
    DensityLevel,
    Direction,
    IncidentSeverity,
    IncidentType,
    SignalPhase,
    SystemMode,
    VehicleClass,
)

# ---------------------------------------------------------------------------
# Reusable constrained field types
# ---------------------------------------------------------------------------
NonNegInt = Annotated[int, Field(ge=0)]
NonNegFloat = Annotated[float, Field(ge=0.0)]
Percent = Annotated[float, Field(ge=0.0, le=100.0)]
Confidence = Annotated[float, Field(ge=0.0, le=1.0)]


# ---------------------------------------------------------------------------
# Perception primitives
# ---------------------------------------------------------------------------
class BoundingBox(FrozenModel):
    """Axis-aligned bounding box in pixel coordinates (top-left origin)."""

    x1: NonNegFloat
    y1: NonNegFloat
    x2: float
    y2: float

    @model_validator(mode="after")
    def _check_ordering(self) -> BoundingBox:
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("bounding box requires x2 > x1 and y2 > y1")
        return self

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def centroid(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


class Detection(FrozenModel):
    """A single object detected by the Vision Agent in one frame."""

    vehicle_class: VehicleClass
    confidence: Confidence
    box: BoundingBox


class Velocity(FrozenModel):
    """A 2-D velocity vector in pixels/second (or metres/second post-homography)."""

    vx: float
    vy: float

    @property
    def speed(self) -> float:
        """Euclidean magnitude of the velocity vector."""
        return float((self.vx**2 + self.vy**2) ** 0.5)


class Track(FrozenModel):
    """A tracked object with a stable identity across frames (ByteTrack output)."""

    track_id: int = Field(ge=0)
    vehicle_class: VehicleClass
    box: BoundingBox
    velocity: Velocity
    lane: Direction | None = None
    is_moving: bool
    age_frames: NonNegInt = 0


# ---------------------------------------------------------------------------
# Aggregated lane / intersection state
# ---------------------------------------------------------------------------
class LaneState(FrozenModel):
    """Derived traffic state for a single approach at one instant.

    Produced by the Density/Occupancy Agent from tracked, movement-classified objects.
    """

    direction: Direction
    vehicle_count: NonNegInt
    moving_count: NonNegInt
    stopped_count: NonNegInt
    queue_length_m: NonNegFloat
    avg_wait_s: NonNegFloat
    occupancy_pct: Percent
    density: DensityLevel
    has_emergency_vehicle: bool = False
    pedestrian_waiting: bool = False

    @model_validator(mode="after")
    def _check_counts(self) -> LaneState:
        if self.moving_count + self.stopped_count != self.vehicle_count:
            raise ValueError(
                "moving_count + stopped_count must equal vehicle_count "
                f"({self.moving_count} + {self.stopped_count} != {self.vehicle_count})"
            )
        return self


class IntersectionState(FrozenModel):
    """The canonical snapshot of an intersection the Perception plane publishes.

    This is the contract between the two planes: perception writes it to Redis on every
    aggregation tick, and cognition agents consume the latest value (last-write-wins).
    """

    intersection_id: str = Field(min_length=1)
    timestamp: datetime = Field(default_factory=utcnow)
    lanes: dict[Direction, LaneState]
    current_phase: SignalPhase
    phase_elapsed_s: NonNegFloat
    mode: SystemMode = SystemMode.AI
    perception_confidence: Confidence = 1.0

    @model_validator(mode="after")
    def _check_lane_consistency(self) -> IntersectionState:
        for direction, lane in self.lanes.items():
            if lane.direction != direction:
                raise ValueError(
                    f"lane keyed under {direction} reports direction {lane.direction}"
                )
        return self

    @property
    def total_vehicles(self) -> int:
        return sum(lane.vehicle_count for lane in self.lanes.values())

    @property
    def total_stopped(self) -> int:
        return sum(lane.stopped_count for lane in self.lanes.values())

    def lane(self, direction: Direction) -> LaneState | None:
        """Return the state for ``direction`` if present."""
        return self.lanes.get(direction)

    def busiest_lane(self) -> LaneState | None:
        """Return the lane with the greatest queue length, if any lanes are present."""
        if not self.lanes:
            return None
        return max(self.lanes.values(), key=lambda lane: lane.queue_length_m)

    def emergency_lanes(self) -> tuple[Direction, ...]:
        """Return the directions currently carrying an emergency vehicle."""
        return tuple(d for d, lane in self.lanes.items() if lane.has_emergency_vehicle)


# ---------------------------------------------------------------------------
# Memory / history
# ---------------------------------------------------------------------------
class LaneBaseline(FrozenModel):
    """Rolling-average historical demand for one approach, as seen by the Memory Agent."""

    direction: Direction
    avg_queue_veh: NonNegFloat
    avg_wait_s: NonNegFloat
    sample_count: NonNegInt


class HistoricalContext(FrozenModel):
    """A refreshed baseline snapshot published by the Traffic Memory Agent."""

    intersection_id: str = Field(min_length=1)
    computed_at: datetime = Field(default_factory=utcnow)
    window_size: NonNegInt
    lanes: dict[Direction, LaneBaseline]

    def baseline_for(self, direction: Direction) -> LaneBaseline | None:
        return self.lanes.get(direction)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
class LaneForecast(FrozenModel):
    """A short-horizon forecast for one lane, with a confidence interval."""

    direction: Direction
    horizon_s: float = Field(gt=0)
    predicted_queue_length_m: NonNegFloat
    predicted_wait_s: NonNegFloat
    confidence: Confidence
    lower_bound_m: NonNegFloat
    upper_bound_m: NonNegFloat

    @model_validator(mode="after")
    def _check_bounds(self) -> LaneForecast:
        if not (self.lower_bound_m <= self.predicted_queue_length_m <= self.upper_bound_m):
            raise ValueError("predicted_queue_length_m must lie within [lower_bound, upper_bound]")
        return self


class Forecast(FrozenModel):
    """A set of per-lane forecasts produced by the Prediction Agent for one intersection."""

    intersection_id: str = Field(min_length=1)
    generated_at: datetime = Field(default_factory=utcnow)
    horizon_s: float = Field(gt=0)
    lanes: dict[Direction, LaneForecast]
    model_version: str = "persistence-v0"


# ---------------------------------------------------------------------------
# Decision & control
# ---------------------------------------------------------------------------
class SignalCommand(FrozenModel):
    """The output of the Decision Agent, consumed by the Signal Controller.

    ``feature_snapshot`` captures the exact inputs that produced the command so the
    Explainability Agent can narrate the decision without re-reading state, and so the
    audit log is fully reconstructable.
    """

    intersection_id: str = Field(min_length=1)
    action: DecisionAction
    target_phase: SignalPhase
    duration_s: float = Field(ge=0)
    reason_code: str = Field(min_length=1)
    issued_at: datetime = Field(default_factory=utcnow)
    feature_snapshot: dict[str, float] = Field(default_factory=dict)
    rejected_alternatives: tuple[DecisionAction, ...] = ()
    constraints_applied: tuple[str, ...] = ()
    policy_version: str = "utility-v0"


class SignalState(FrozenModel):
    """The Signal Controller's authoritative view of the current signal."""

    intersection_id: str = Field(min_length=1)
    phase: SignalPhase
    phase_elapsed_s: NonNegFloat
    phase_remaining_s: NonNegFloat
    updated_at: datetime = Field(default_factory=utcnow)


# ---------------------------------------------------------------------------
# Incidents & explanations
# ---------------------------------------------------------------------------
class Incident(FrozenModel):
    """An abnormal event raised by the Incident Detection Agent."""

    intersection_id: str = Field(min_length=1)
    incident_type: IncidentType
    severity: IncidentSeverity
    direction: Direction | None = None
    confidence: Confidence
    description: str = Field(min_length=1)
    detected_at: datetime = Field(default_factory=utcnow)


class Explanation(FrozenModel):
    """A natural-language explanation of a decision, produced out of the control loop."""

    intersection_id: str = Field(min_length=1)
    decision_reason_code: str = Field(min_length=1)
    text: str = Field(min_length=1)
    counterfactual: str | None = None
    generator: str = "template"
    model_id: str | None = None
    generated_at: datetime = Field(default_factory=utcnow)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
class AgentHealth(FrozenModel):
    """A heartbeat payload published by every agent for the Orchestrator and Prometheus."""

    agent_name: str = Field(min_length=1)
    status: AgentStatus
    last_heartbeat: datetime = Field(default_factory=utcnow)
    details: dict[str, str] = Field(default_factory=dict)
