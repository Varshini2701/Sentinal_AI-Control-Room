"""Enumerations shared across the Sentinel AI domain.

All enums subclass :class:`str` so they serialise to plain strings in JSON payloads,
database columns and message headers without custom encoders.
"""

from __future__ import annotations

from enum import StrEnum


class Direction(StrEnum):
    """A cardinal approach to the intersection."""

    NORTH = "north"
    SOUTH = "south"
    EAST = "east"
    WEST = "west"

    @property
    def opposite(self) -> Direction:
        """Return the approach directly opposite this one."""
        return _OPPOSITE[self]

    @property
    def axis(self) -> Axis:
        """Return the axis (NS or EW) this approach belongs to."""
        return Axis.NORTH_SOUTH if self in (Direction.NORTH, Direction.SOUTH) else Axis.EAST_WEST


class Axis(StrEnum):
    """The two movement axes of a standard two-phase intersection."""

    NORTH_SOUTH = "north_south"
    EAST_WEST = "east_west"

    @property
    def directions(self) -> tuple[Direction, Direction]:
        """Return the two approaches served by this axis."""
        if self is Axis.NORTH_SOUTH:
            return (Direction.NORTH, Direction.SOUTH)
        return (Direction.EAST, Direction.WEST)


_OPPOSITE: dict[Direction, Direction] = {
    Direction.NORTH: Direction.SOUTH,
    Direction.SOUTH: Direction.NORTH,
    Direction.EAST: Direction.WEST,
    Direction.WEST: Direction.EAST,
}


class SignalColor(StrEnum):
    """The aspect shown to a single approach."""

    RED = "red"
    GREEN = "green"
    YELLOW = "yellow"


class SignalPhase(StrEnum):
    """A phase in the intersection signal state machine.

    Sentinel models a canonical two-phase intersection. Green/​yellow phases serve one axis;
    ``ALL_RED`` is the mandatory clearance interval between conflicting greens.
    """

    NS_GREEN = "ns_green"
    NS_YELLOW = "ns_yellow"
    EW_GREEN = "ew_green"
    EW_YELLOW = "ew_yellow"
    ALL_RED = "all_red"

    @property
    def axis(self) -> Axis | None:
        """Return the axis this phase serves, or ``None`` for the all-red clearance."""
        return _PHASE_AXIS.get(self)

    @property
    def is_green(self) -> bool:
        """Whether this phase grants right of way to an axis."""
        return self in (SignalPhase.NS_GREEN, SignalPhase.EW_GREEN)


_PHASE_AXIS: dict[SignalPhase, Axis] = {
    SignalPhase.NS_GREEN: Axis.NORTH_SOUTH,
    SignalPhase.NS_YELLOW: Axis.NORTH_SOUTH,
    SignalPhase.EW_GREEN: Axis.EAST_WEST,
    SignalPhase.EW_YELLOW: Axis.EAST_WEST,
}


class DensityLevel(StrEnum):
    """Qualitative traffic density for a lane, derived from occupancy and queue length."""

    FREE = "free"
    MODERATE = "moderate"
    HEAVY = "heavy"
    JAM = "jam"

    @classmethod
    def from_occupancy(cls, occupancy_pct: float) -> DensityLevel:
        """Classify a lane by its occupancy percentage (single source of truth for the thresholds).

        Used by every producer of :class:`~sentinel.contracts.value_objects.LaneState` -- the
        perception pipeline and both simulation environments -- so they agree by construction.
        """
        if occupancy_pct >= 80.0:
            return cls.JAM
        if occupancy_pct >= 50.0:
            return cls.HEAVY
        if occupancy_pct >= 20.0:
            return cls.MODERATE
        return cls.FREE


class DecisionAction(StrEnum):
    """The set of actions the Decision Agent may choose from.

    These are *intents*; the Signal Controller translates them into safe phase transitions.
    """

    KEEP_GREEN = "keep_green"
    EXTEND_GREEN = "extend_green"
    REDUCE_GREEN = "reduce_green"
    SWITCH_PHASE = "switch_phase"
    EMERGENCY_OVERRIDE = "emergency_override"


class VehicleClass(StrEnum):
    """Object classes produced by the Vision Agent."""

    CAR = "car"
    TRUCK = "truck"
    BUS = "bus"
    MOTORCYCLE = "motorcycle"
    BICYCLE = "bicycle"
    PEDESTRIAN = "pedestrian"
    AMBULANCE = "ambulance"
    FIRE_TRUCK = "fire_truck"
    POLICE = "police"

    @property
    def is_emergency(self) -> bool:
        """Whether this class should trigger emergency preemption."""
        return self in (VehicleClass.AMBULANCE, VehicleClass.FIRE_TRUCK, VehicleClass.POLICE)

    @property
    def is_vehicle(self) -> bool:
        """Whether this class occupies a traffic lane (excludes pedestrians)."""
        return self is not VehicleClass.PEDESTRIAN


class IncidentType(StrEnum):
    """Categories of abnormal events raised by the Incident Detection Agent."""

    COLLISION = "collision"
    STALLED_VEHICLE = "stalled_vehicle"
    WRONG_WAY = "wrong_way"
    ABNORMAL_CONGESTION = "abnormal_congestion"
    PEDESTRIAN_ON_ROAD = "pedestrian_on_road"


class IncidentSeverity(StrEnum):
    """Severity ranking for an incident, ordered from least to most urgent."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        """Return a numeric rank (0=LOW .. 3=CRITICAL) for comparisons."""
        return _SEVERITY_RANK[self]


_SEVERITY_RANK: dict[IncidentSeverity, int] = {
    IncidentSeverity.LOW: 0,
    IncidentSeverity.MEDIUM: 1,
    IncidentSeverity.HIGH: 2,
    IncidentSeverity.CRITICAL: 3,
}


class SystemMode(StrEnum):
    """Operating mode of an intersection, owned by the Orchestrator Agent."""

    AI = "ai"
    """Full autonomous control by the Decision Agent."""

    FIXED_TIMER = "fixed_timer"
    """Baseline fixed-timer control (used for A/B benchmarking)."""

    DEGRADED = "degraded"
    """Conservative fallback triggered by low perception confidence or agent failure."""

    MANUAL = "manual"
    """Operator override; the AI defers to explicit human commands."""


class AgentStatus(StrEnum):
    """Health status reported by each agent's heartbeat."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    STOPPED = "stopped"
