"""The decision policy: the reasoning core of the Decision Agent.

This is *tier 1* of the two-tier controller. It reasons about the whole intersection and picks an
**intent** -- which axis to serve, and whether to keep/extend/reduce/switch -- by scoring each axis
with a multi-objective utility function (queue, wait, fairness, pedestrians, predicted congestion),
with a switch penalty that prevents thrashing. Emergency vehicles override the score.

It emits intent only; *tier 2* (the Signal Controller's phase state machine) enforces the hard
safety constraints. The policy is a **pure function** of its context, which makes it exhaustively
testable and lets a trained RL policy drop in behind the same :class:`DecisionPolicy` port later.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field

from sentinel.config.settings import DecisionSettings
from sentinel.contracts.enums import Axis, DecisionAction
from sentinel.contracts.value_objects import Forecast, IntersectionState


@dataclass(frozen=True, slots=True)
class AxisMetrics:
    """Aggregated demand on one axis, derived from the intersection state (+ forecast)."""

    queue_veh: int
    wait_s: float
    pedestrian: bool
    predicted_queue_veh: float
    since_green_s: float


@dataclass(slots=True)
class DecisionContext:
    """Everything the policy needs for one decision (a pure input)."""

    state: IntersectionState
    since_green_s: dict[Axis, float]
    forecast: Forecast | None = None

    @property
    def active_axis(self) -> Axis | None:
        """The axis currently green, or ``None`` during a yellow/all-red clearance."""
        phase = self.state.current_phase
        return phase.axis if phase.is_green else None


@dataclass(frozen=True, slots=True)
class DecisionOutcome:
    """The policy's chosen intent plus the evidence behind it (for explainability + audit)."""

    desired_axis: Axis
    action: DecisionAction
    reason_code: str
    scores: dict[str, float]
    features: dict[str, float] = field(default_factory=dict)
    rejected: tuple[DecisionAction, ...] = ()


class DecisionPolicy(abc.ABC):
    """Maps a :class:`DecisionContext` to a :class:`DecisionOutcome` (intent only)."""

    @abc.abstractmethod
    def decide(self, context: DecisionContext) -> DecisionOutcome:
        ...


class UtilityPolicy(DecisionPolicy):
    """Multi-objective, explainable, deterministic decision policy."""

    def __init__(self, settings: DecisionSettings) -> None:
        self._s = settings

    def decide(self, context: DecisionContext) -> DecisionOutcome:
        state = context.state
        metrics = {axis: self._metrics(context, axis) for axis in _AXES}
        scores = {axis: self._score(m) for axis, m in metrics.items()}

        # 1) Emergency preemption overrides the utility score entirely.
        emergency = self._emergency_axis(state)
        if emergency is not None:
            return self._outcome(
                emergency, DecisionAction.EMERGENCY_OVERRIDE, "emergency_preemption",
                scores, metrics,
            )

        active = context.active_axis
        # 2) Mid-clearance: do not fight the state machine; hold intent toward the active side.
        if active is None:
            desired = self._higher(scores)
            return self._outcome(
                desired, DecisionAction.KEEP_GREEN, "clearance_interval", scores, metrics
            )

        # 3) Green: switch only if the opposing axis beats the current one by the switch penalty.
        opposing = _other(active)
        if scores[opposing] > scores[active] + self._s.switch_penalty:
            return self._outcome(
                opposing, DecisionAction.SWITCH_PHASE, "opposing_demand_higher", scores, metrics,
                rejected=(DecisionAction.KEEP_GREEN,),
            )

        # 4) Otherwise keep the current axis; label extend/reduce by its congestion.
        action, reason = self._hold_action(metrics[active])
        return self._outcome(
            active, action, reason, scores, metrics, rejected=(DecisionAction.SWITCH_PHASE,)
        )

    # -- scoring -----------------------------------------------------------
    def _metrics(self, context: DecisionContext, axis: Axis) -> AxisMetrics:
        lanes = [lane for d, lane in context.state.lanes.items() if d.axis is axis]
        predicted = 0.0
        if context.forecast is not None:
            predicted = sum(
                f.predicted_queue_length_m
                for d, f in context.forecast.lanes.items()
                if d.axis is axis
            )
        return AxisMetrics(
            queue_veh=sum(lane.vehicle_count for lane in lanes),
            wait_s=sum(lane.avg_wait_s for lane in lanes),
            pedestrian=any(lane.pedestrian_waiting for lane in lanes),
            predicted_queue_veh=predicted,
            since_green_s=context.since_green_s.get(axis, 0.0),
        )

    def _score(self, m: AxisMetrics) -> float:
        return (
            self._s.weight_queue * m.queue_veh
            + self._s.weight_wait * m.wait_s
            + self._s.weight_fairness * m.since_green_s
            + self._s.weight_pedestrian * (1.0 if m.pedestrian else 0.0)
            + self._s.weight_prediction * m.predicted_queue_veh
        )

    def _hold_action(self, m: AxisMetrics) -> tuple[DecisionAction, str]:
        if m.queue_veh == 0:
            return DecisionAction.REDUCE_GREEN, "current_axis_clearing"
        if m.queue_veh >= self._s.queue_congestion_threshold_m:
            return DecisionAction.EXTEND_GREEN, "current_axis_congested"
        return DecisionAction.KEEP_GREEN, "current_axis_serving"

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _emergency_axis(state: IntersectionState) -> Axis | None:
        for direction in state.emergency_lanes():
            return direction.axis
        return None

    @staticmethod
    def _higher(scores: dict[Axis, float]) -> Axis:
        return max(scores, key=lambda a: scores[a])

    def _outcome(
        self,
        desired: Axis,
        action: DecisionAction,
        reason: str,
        scores: dict[Axis, float],
        metrics: dict[Axis, AxisMetrics],
        *,
        rejected: tuple[DecisionAction, ...] = (),
    ) -> DecisionOutcome:
        return DecisionOutcome(
            desired_axis=desired,
            action=action,
            reason_code=reason,
            scores={axis.value: round(score, 3) for axis, score in scores.items()},
            features={
                "queue_ns": float(metrics[Axis.NORTH_SOUTH].queue_veh),
                "queue_ew": float(metrics[Axis.EAST_WEST].queue_veh),
                "wait_ns": round(metrics[Axis.NORTH_SOUTH].wait_s, 2),
                "wait_ew": round(metrics[Axis.EAST_WEST].wait_s, 2),
                "since_green_ns": round(metrics[Axis.NORTH_SOUTH].since_green_s, 1),
                "since_green_ew": round(metrics[Axis.EAST_WEST].since_green_s, 1),
            },
            rejected=rejected,
        )


_AXES = (Axis.NORTH_SOUTH, Axis.EAST_WEST)


def _other(axis: Axis) -> Axis:
    return Axis.EAST_WEST if axis is Axis.NORTH_SOUTH else Axis.NORTH_SOUTH


__all__ = [
    "AxisMetrics",
    "DecisionContext",
    "DecisionOutcome",
    "DecisionPolicy",
    "UtilityPolicy",
]
