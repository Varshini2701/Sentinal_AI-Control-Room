"""Incident detection rules: pure functions from state (+ baseline) to an optional Incident.

Each rule inspects one :class:`IntersectionState` (and, where useful, the latest
:class:`HistoricalContext` baseline from the Traffic Memory Agent) and returns an
:class:`Incident` if its condition holds, else ``None``. Rules are deliberately conservative
(favouring false negatives over false positives) since they drive an audit trail, not an
automatic actuator response.

**Scope note:** collision and wrong-way detection require per-vehicle trajectory data (track
convergence, heading) that lives in the Perception plane's :class:`Track` objects, not in the
aggregated :class:`IntersectionState`. Wiring those in is a straightforward extension once a
Track-level event is added to the bus; the two rules implemented here (stalled vehicle, abnormal
congestion) are the ones expressible purely from aggregated lane state, which is what every
consumer of this agent already has.
"""

from __future__ import annotations

import abc

from sentinel.config.settings import IncidentSettings
from sentinel.contracts.enums import IncidentSeverity, IncidentType
from sentinel.contracts.value_objects import HistoricalContext, Incident, IntersectionState


class IncidentRule(abc.ABC):
    """A single incident-detection heuristic."""

    @abc.abstractmethod
    def evaluate(
        self, state: IntersectionState, baseline: HistoricalContext | None
    ) -> list[Incident]:
        """Return zero or more incidents this rule detects in ``state``."""


class StalledVehicleRule(IncidentRule):
    """Flags a lane that has right-of-way but is not discharging any vehicles.

    If an axis has been green for at least ``stalled_wait_s`` and its lanes have stopped vehicles
    but zero moving vehicles, something is blocking the lane despite having the signal in its
    favour -- the classic signature of a stalled vehicle or minor collision.
    """

    def __init__(self, settings: IncidentSettings) -> None:
        self._settings = settings

    def evaluate(
        self, state: IntersectionState, baseline: HistoricalContext | None
    ) -> list[Incident]:
        phase = state.current_phase
        if not phase.is_green or phase.axis is None:
            return []
        if state.phase_elapsed_s < self._settings.stalled_wait_s:
            return []

        incidents: list[Incident] = []
        for direction, lane in state.lanes.items():
            if direction.axis is not phase.axis:
                continue
            if lane.stopped_count > 0 and lane.moving_count == 0:
                incidents.append(
                    Incident(
                        intersection_id=state.intersection_id,
                        incident_type=IncidentType.STALLED_VEHICLE,
                        severity=IncidentSeverity.HIGH,
                        direction=direction,
                        confidence=0.7,
                        description=(
                            f"{direction.value} has had right-of-way for "
                            f"{state.phase_elapsed_s:.0f}s but {lane.stopped_count} vehicle(s) "
                            "remain stopped with none discharging."
                        ),
                        detected_at=state.timestamp,
                    )
                )
        return incidents


class AbnormalCongestionRule(IncidentRule):
    """Flags a lane whose queue is far above its historical baseline.

    Requires a baseline with at least ``congestion_min_baseline_samples`` -- with too little
    history the comparison is unreliable, so the rule stays silent rather than guessing.
    """

    def __init__(self, settings: IncidentSettings) -> None:
        self._settings = settings

    def evaluate(
        self, state: IntersectionState, baseline: HistoricalContext | None
    ) -> list[Incident]:
        if baseline is None:
            return []

        incidents: list[Incident] = []
        for direction, lane in state.lanes.items():
            lane_baseline = baseline.baseline_for(direction)
            if lane_baseline is None:
                continue
            if lane_baseline.sample_count < self._settings.congestion_min_baseline_samples:
                continue
            if lane_baseline.avg_queue_veh <= 0:
                continue
            ratio = lane.vehicle_count / lane_baseline.avg_queue_veh
            if ratio >= self._settings.congestion_ratio:
                incidents.append(
                    Incident(
                        intersection_id=state.intersection_id,
                        incident_type=IncidentType.ABNORMAL_CONGESTION,
                        severity=IncidentSeverity.MEDIUM,
                        direction=direction,
                        confidence=min(0.95, 0.5 + (ratio - self._settings.congestion_ratio) * 0.1),
                        description=(
                            f"{direction.value} queue ({lane.vehicle_count} veh) is "
                            f"{ratio:.1f}x its baseline average "
                            f"({lane_baseline.avg_queue_veh:.1f} veh)."
                        ),
                        detected_at=state.timestamp,
                    )
                )
        return incidents


__all__ = ["AbnormalCongestionRule", "IncidentRule", "StalledVehicleRule"]
