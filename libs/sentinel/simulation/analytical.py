"""A deterministic point-queue traffic environment ("the analytical twin").

This is a vertical/point-queue model: each approach holds a queue of vehicles that grows with
Poisson arrivals and drains at the saturation flow rate while its axis has a green light. It is
intentionally simple physics, but it captures exactly the effect that matters for signal control --
**green time allocated to an approach with no demand is wasted, and queues on a starved approach
accrue delay** -- so a smarter controller measurably wins.

It has no external dependencies, is fully deterministic given a seed, and runs thousands of
simulated seconds in milliseconds, which makes it the backbone of the test suite and CI benchmark.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from sentinel.contracts.enums import Axis, DensityLevel, Direction, SignalPhase, SystemMode
from sentinel.contracts.value_objects import IntersectionState, LaneState, SignalCommand
from sentinel.simulation.config import SimConfig
from sentinel.simulation.environment import TrafficEnvironment
from sentinel.simulation.kpi import KpiSummary, LaneKpi

_SIM_EPOCH = datetime(2020, 1, 1, tzinfo=UTC)
"""Fixed epoch for simulated timestamps -- deterministic and independent of wall-clock time.

Consumers (Decision Agent fairness accrual, Prediction Agent trend fitting) treat
``IntersectionState.timestamp`` as the authoritative clock to compute elapsed time between
samples. Stamping with wall-clock ``utcnow()`` would make a fast, sleep-free simulation loop
report near-zero elapsed time between ticks, corrupting both fairness and trend calculations.
"""


def _poisson(rng: random.Random, lam: float) -> int:
    """Sample a Poisson-distributed count with mean ``lam`` using Knuth's algorithm.

    Deterministic for a given seeded ``rng``. Adequate for the modest rates used in traffic demand
    (a handful of arrivals per tick); no third-party dependency required.
    """
    if lam <= 0.0:
        return 0
    target = pow(2.718281828459045, -lam)
    count = 0
    product = 1.0
    while True:
        product *= rng.random()
        if product <= target:
            return count
        count += 1


@dataclass(slots=True)
class _LaneSim:
    """Mutable per-approach simulation state and accumulators."""

    direction: Direction
    arrival_rate_vps: float
    queue: int = 0
    last_discharge: int = 0
    emergency: bool = False
    # accumulators
    arrived: int = 0
    served: int = 0
    total_delay_veh_s: float = 0.0
    max_queue: int = 0
    _capacity: float = field(default=0.0)  # fractional discharge capacity carried between ticks

    def to_kpi(self) -> LaneKpi:
        return LaneKpi(
            direction=self.direction,
            arrived=self.arrived,
            served=self.served,
            total_delay_veh_s=self.total_delay_veh_s,
            max_queue_veh=self.max_queue,
        )


class AnalyticalTrafficEnvironment(TrafficEnvironment):
    """A zero-dependency, deterministic point-queue traffic environment."""

    def __init__(self, config: SimConfig) -> None:
        self._config = config
        self._rng = random.Random(config.seed)
        self._time = 0.0
        self._phase = SignalPhase.NS_GREEN
        self._phase_elapsed = 0.0
        self._lanes: dict[Direction, _LaneSim] = {}
        self.reset()

    # -- TrafficEnvironment ------------------------------------------------
    def reset(self) -> IntersectionState:
        self._rng = random.Random(self._config.seed)
        self._time = 0.0
        self._phase = SignalPhase.NS_GREEN
        self._phase_elapsed = 0.0
        self._lanes = {
            d: _LaneSim(direction=d, arrival_rate_vps=lane.arrival_rate_vps)
            for d, lane in self._config.demand.items()
        }
        return self._observe()

    def step(self, command: SignalCommand) -> IntersectionState:
        dt = self._config.dt_s
        if command.target_phase != self._phase:
            self._phase = command.target_phase
            self._phase_elapsed = 0.0
        else:
            self._phase_elapsed += dt

        green_axis = self._phase.axis if self._phase.is_green else None
        self._update_emergencies()

        for lane in self._lanes.values():
            self._advance_lane(lane, dt, green_axis)

        self._time += dt
        return self._observe()

    def metrics(self) -> KpiSummary:
        lanes = {d: lane.to_kpi() for d, lane in self._lanes.items()}
        return KpiSummary(
            controller="unknown",
            sim_duration_s=max(self._time, self._config.dt_s),
            total_arrived=sum(lane.arrived for lane in self._lanes.values()),
            total_served=sum(lane.served for lane in self._lanes.values()),
            total_delay_veh_s=sum(lane.total_delay_veh_s for lane in self._lanes.values()),
            max_queue_veh=max((lane.max_queue for lane in self._lanes.values()), default=0),
            lanes=lanes,
        )

    @property
    def time_s(self) -> float:
        return self._time

    # -- physics -----------------------------------------------------------
    def _advance_lane(self, lane: _LaneSim, dt: float, green_axis: Axis | None) -> None:
        arrivals = _poisson(self._rng, lane.arrival_rate_vps * dt)
        lane.queue += arrivals
        lane.arrived += arrivals

        is_green = green_axis is not None and lane.direction.axis is green_axis
        if is_green:
            lane._capacity += self._config.saturation_flow_vps * dt
            discharge = min(lane.queue, int(lane._capacity))
            lane.queue -= discharge
            lane._capacity -= discharge
            lane.served += discharge
            lane.last_discharge = discharge
        else:
            lane._capacity = 0.0  # capacity does not bank across a red interval
            lane.last_discharge = 0

        lane.total_delay_veh_s += lane.queue * dt
        lane.max_queue = max(lane.max_queue, lane.queue)

    def _update_emergencies(self) -> None:
        for lane in self._lanes.values():
            lane.emergency = False
        for event in self._config.emergencies:
            if event.active_at(self._time):
                self._lanes[event.direction].emergency = True

    # -- observation -------------------------------------------------------
    def _observe(self) -> IntersectionState:
        lanes = {d: self._lane_state(lane) for d, lane in self._lanes.items()}
        return IntersectionState(
            intersection_id=self._config.intersection_id,
            timestamp=_SIM_EPOCH + timedelta(seconds=self._time),
            lanes=lanes,
            current_phase=self._phase,
            phase_elapsed_s=self._phase_elapsed,
            mode=SystemMode.AI,
        )

    def _lane_state(self, lane: _LaneSim) -> LaneState:
        queue = lane.queue
        moving = min(lane.last_discharge, queue)
        stopped = queue - moving
        queue_len_m = queue * self._config.vehicle_length_m
        occupancy = min(100.0, queue_len_m / self._config.approach_length_m * 100.0)
        sat = self._config.saturation_flow_vps
        wait_s = queue / sat if sat else 0.0
        return LaneState(
            direction=lane.direction,
            vehicle_count=queue,
            moving_count=moving,
            stopped_count=stopped,
            queue_length_m=queue_len_m,
            avg_wait_s=wait_s,
            occupancy_pct=occupancy,
            density=DensityLevel.from_occupancy(occupancy),
            has_emergency_vehicle=lane.emergency,
        )


__all__ = ["AnalyticalTrafficEnvironment"]
