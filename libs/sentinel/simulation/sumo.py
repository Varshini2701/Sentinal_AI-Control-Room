"""SUMO (TraCI) traffic environment -- the high-fidelity closed-loop demo.

Implements the same :class:`TrafficEnvironment` port as the analytical twin, so the harness, the
Decision Agent and the dashboard drive it unchanged. It maps Sentinel's :class:`SignalPhase` onto a
SUMO traffic-light program and reads per-approach queues back from the microsimulation.

``traci`` and the SUMO binary are required only to *use* this class; they are imported lazily so
that importing the module (and the rest of Sentinel) never depends on a SUMO install. Build the
network with ``sim/build_network.py`` first -- see ``sim/README.md``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sentinel.contracts.enums import DensityLevel, Direction, SignalPhase, SystemMode
from sentinel.contracts.value_objects import IntersectionState, LaneState, SignalCommand
from sentinel.observability.logging import get_logger
from sentinel.simulation.environment import TrafficEnvironment
from sentinel.simulation.kpi import KpiSummary, LaneKpi

_log = get_logger("sentinel.simulation.sumo")

_SIM_EPOCH = datetime(2020, 1, 1, tzinfo=UTC)
"""Fixed epoch for simulated timestamps; see the analytical twin for why this matters."""

#: Maps a Sentinel phase onto the phase index of the TLS program emitted by ``sim/build_network``.
_PHASE_TO_INDEX: dict[SignalPhase, int] = {
    SignalPhase.NS_GREEN: 0,
    SignalPhase.NS_YELLOW: 1,
    SignalPhase.ALL_RED: 2,
    SignalPhase.EW_GREEN: 3,
    SignalPhase.EW_YELLOW: 4,
}


class SumoTrafficEnvironment(TrafficEnvironment):
    """A TraCI-backed traffic environment for a single four-approach intersection.

    Args:
        sumo_cfg: Path to the ``.sumocfg`` scenario file.
        tls_id: The traffic-light system id in the network.
        approach_lanes: Maps each :class:`Direction` to the list of incoming SUMO lane ids.
        step_length_s: SUMO simulation step length (must match the scenario's ``--step-length``).
        use_gui: Launch ``sumo-gui`` instead of headless ``sumo`` for the visual demo.
    """

    def __init__(
        self,
        *,
        sumo_cfg: str,
        tls_id: str,
        approach_lanes: dict[Direction, list[str]],
        intersection_id: str = "intersection-1",
        step_length_s: float = 1.0,
        approach_length_m: float = 200.0,
        vehicle_length_m: float = 7.0,
        use_gui: bool = False,
    ) -> None:
        self._cfg = sumo_cfg
        self._tls_id = tls_id
        self._approach_lanes = approach_lanes
        self._intersection_id = intersection_id
        self._step_length = step_length_s
        self._approach_length = approach_length_m
        self._vehicle_length = vehicle_length_m
        self._use_gui = use_gui

        self._traci = None  # set on reset()
        self._time = 0.0
        self._phase = SignalPhase.NS_GREEN
        self._phase_elapsed = 0.0
        self._served: dict[Direction, int] = dict.fromkeys(approach_lanes, 0)
        self._seen_ids: dict[Direction, set[str]] = {d: set() for d in approach_lanes}
        self._delay: dict[Direction, float] = dict.fromkeys(approach_lanes, 0.0)
        self._max_queue: dict[Direction, int] = dict.fromkeys(approach_lanes, 0)
        self._arrived: dict[Direction, int] = dict.fromkeys(approach_lanes, 0)

    # -- TrafficEnvironment ------------------------------------------------
    def reset(self) -> IntersectionState:
        import traci  # lazy: only needed to actually run SUMO

        if self._traci is not None:
            traci.close()
        binary = "sumo-gui" if self._use_gui else "sumo"
        traci.start([binary, "-c", self._cfg, "--step-length", str(self._step_length)])
        self._traci = traci
        self._time = 0.0
        self._phase = SignalPhase.NS_GREEN
        self._phase_elapsed = 0.0
        self._served = dict.fromkeys(self._approach_lanes, 0)
        self._seen_ids = {d: set() for d in self._approach_lanes}
        self._delay = dict.fromkeys(self._approach_lanes, 0.0)
        self._max_queue = dict.fromkeys(self._approach_lanes, 0)
        self._arrived = dict.fromkeys(self._approach_lanes, 0)
        traci.trafficlight.setPhase(self._tls_id, _PHASE_TO_INDEX[self._phase])
        return self._observe()

    def step(self, command: SignalCommand) -> IntersectionState:
        assert self._traci is not None, "step() called before reset()"
        if command.target_phase != self._phase:
            self._phase = command.target_phase
            self._phase_elapsed = 0.0
            self._traci.trafficlight.setPhase(self._tls_id, _PHASE_TO_INDEX[self._phase])
        else:
            self._phase_elapsed += self._step_length

        self._traci.simulationStep()
        self._time += self._step_length
        self._accumulate()
        return self._observe()

    def metrics(self) -> KpiSummary:
        lanes = {
            d: LaneKpi(
                direction=d,
                arrived=self._arrived[d],
                served=self._served[d],
                total_delay_veh_s=self._delay[d],
                max_queue_veh=self._max_queue[d],
            )
            for d in self._approach_lanes
        }
        return KpiSummary(
            controller="unknown",
            sim_duration_s=max(self._time, self._step_length),
            total_arrived=sum(self._arrived.values()),
            total_served=sum(self._served.values()),
            total_delay_veh_s=sum(self._delay.values()),
            max_queue_veh=max(self._max_queue.values(), default=0),
            lanes=lanes,
        )

    @property
    def time_s(self) -> float:
        return self._time

    def close(self) -> None:
        if self._traci is not None:
            self._traci.close()
            self._traci = None

    # -- internals ---------------------------------------------------------
    def _halting(self, direction: Direction) -> int:
        assert self._traci is not None
        return sum(
            self._traci.lane.getLastStepHaltingNumber(lane)
            for lane in self._approach_lanes[direction]
        )

    def _accumulate(self) -> None:
        """Update per-approach KPI accumulators after a simulation step."""
        assert self._traci is not None
        for direction, lanes in self._approach_lanes.items():
            halting = self._halting(direction)
            self._delay[direction] += halting * self._step_length
            self._max_queue[direction] = max(self._max_queue[direction], halting)
            # Count newly-seen vehicles as arrivals; vehicles that left as served.
            current_ids: set[str] = set()
            for lane in lanes:
                current_ids.update(self._traci.lane.getLastStepVehicleIDs(lane))
            new_ids = current_ids - self._seen_ids[direction]
            self._arrived[direction] += len(new_ids)
            departed = self._seen_ids[direction] - current_ids
            self._served[direction] += len(departed)
            self._seen_ids[direction] = current_ids

    def _observe(self) -> IntersectionState:
        lanes = {d: self._lane_state(d) for d in self._approach_lanes}
        return IntersectionState(
            intersection_id=self._intersection_id,
            timestamp=_SIM_EPOCH + timedelta(seconds=self._time),
            lanes=lanes,
            current_phase=self._phase,
            phase_elapsed_s=self._phase_elapsed,
            mode=SystemMode.AI,
        )

    def _lane_state(self, direction: Direction) -> LaneState:
        assert self._traci is not None
        lanes = self._approach_lanes[direction]
        total = sum(self._traci.lane.getLastStepVehicleNumber(lane) for lane in lanes)
        halting = self._halting(direction)
        moving = max(0, total - halting)
        queue_len_m = halting * self._vehicle_length
        occupancy = min(100.0, queue_len_m / self._approach_length * 100.0)
        return LaneState(
            direction=direction,
            vehicle_count=total,
            moving_count=moving,
            stopped_count=halting,
            queue_length_m=queue_len_m,
            avg_wait_s=max(
                self._traci.lane.getWaitingTime(lane) for lane in lanes
            )
            if lanes
            else 0.0,
            occupancy_pct=occupancy,
            density=DensityLevel.from_occupancy(occupancy),
        )


__all__ = ["SumoTrafficEnvironment"]
