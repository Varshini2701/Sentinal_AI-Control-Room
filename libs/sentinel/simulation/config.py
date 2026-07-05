"""Scenario configuration for the traffic simulation environments.

A :class:`SimConfig` fully and deterministically describes a benchmark run: per-approach demand,
discharge capacity, geometry, duration and RNG seed. Running two controllers against the same
``SimConfig`` (same seed) guarantees identical arrival streams, which is what makes the fixed-timer
vs. Sentinel-AI comparison a fair A/B test.
"""

from __future__ import annotations

from pydantic import Field, model_validator

from sentinel.contracts.base import FrozenModel
from sentinel.contracts.enums import Direction


class LaneDemand(FrozenModel):
    """Traffic demand for a single approach."""

    direction: Direction
    arrival_rate_vps: float = Field(ge=0.0, description="Mean vehicle arrivals per second")


class EmergencyEvent(FrozenModel):
    """A scheduled emergency-vehicle presence on one approach, for preemption demos."""

    direction: Direction
    start_s: float = Field(ge=0.0)
    duration_s: float = Field(gt=0.0)

    def active_at(self, t: float) -> bool:
        """Whether the emergency vehicle is present at simulation time ``t``."""
        return self.start_s <= t < self.start_s + self.duration_s


class SimConfig(FrozenModel):
    """A complete, reproducible simulation scenario."""

    intersection_id: str = Field(default="intersection-1", min_length=1)
    dt_s: float = Field(default=1.0, gt=0.0, description="Simulation/control tick length")
    horizon_s: float = Field(default=1200.0, gt=0.0, description="Total simulated duration")
    seed: int = Field(default=42, ge=0)

    demand: dict[Direction, LaneDemand]
    saturation_flow_vps: float = Field(
        default=0.5, gt=0.0, description="Max discharge per approach during green (veh/s)"
    )
    vehicle_length_m: float = Field(
        default=7.0, gt=0.0, description="Average headway per queued vehicle (metres)"
    )
    approach_length_m: float = Field(
        default=200.0, gt=0.0, description="Usable approach length for occupancy (metres)"
    )
    emergencies: tuple[EmergencyEvent, ...] = ()

    @property
    def total_steps(self) -> int:
        """Number of discrete ticks in a full run."""
        return round(self.horizon_s / self.dt_s)

    @model_validator(mode="after")
    def _check_demand_directions(self) -> SimConfig:
        for direction, lane in self.demand.items():
            if lane.direction != direction:
                raise ValueError(
                    f"demand keyed under {direction} declares direction {lane.direction}"
                )
        return self


def symmetric_demand(rate_vps: float) -> dict[Direction, LaneDemand]:
    """Build equal demand on all four approaches."""
    return {d: LaneDemand(direction=d, arrival_rate_vps=rate_vps) for d in Direction}


def asymmetric_demand(ns_rate_vps: float, ew_rate_vps: float) -> dict[Direction, LaneDemand]:
    """Build heavy North/South demand and light East/West demand.

    This is the canonical benchmark scenario: a fixed timer wastes green on the light axis, while
    an adaptive controller shifts green to the busy axis -- so the improvement is real and visible.
    """
    return {
        Direction.NORTH: LaneDemand(direction=Direction.NORTH, arrival_rate_vps=ns_rate_vps),
        Direction.SOUTH: LaneDemand(direction=Direction.SOUTH, arrival_rate_vps=ns_rate_vps),
        Direction.EAST: LaneDemand(direction=Direction.EAST, arrival_rate_vps=ew_rate_vps),
        Direction.WEST: LaneDemand(direction=Direction.WEST, arrival_rate_vps=ew_rate_vps),
    }


__all__ = [
    "EmergencyEvent",
    "LaneDemand",
    "SimConfig",
    "asymmetric_demand",
    "symmetric_demand",
]
