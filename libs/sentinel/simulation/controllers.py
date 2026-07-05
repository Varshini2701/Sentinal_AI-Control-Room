"""Signal controllers used by the benchmark harness.

Two controllers implement the same :class:`Controller` interface:

* :class:`FixedTimerController` -- the baseline: fixed green split, blind to demand. This is what
  most real intersections do today.
* :class:`AdaptiveController` -- a transparent, safety-clamped policy: longest-queue-first with a
  fairness (anti-starvation) guarantee and emergency-vehicle preemption.

Both delegate *all* timing safety to :class:`~sentinel.control.phase.PhaseStateMachine`; they only
express *intent* (``request_switch``). The adaptive policy here is deliberately explainable and
provisional -- the Decision Agent module later swaps in the multi-objective / RL policy behind this
same interface, but the phase machine and safety envelope are unchanged.
"""

from __future__ import annotations

import abc

from sentinel.config.settings import DecisionSettings
from sentinel.contracts.enums import Axis, DecisionAction, SignalPhase
from sentinel.contracts.value_objects import IntersectionState, SignalCommand
from sentinel.control.phase import PhaseStateMachine, other_axis


def _axis_queue(state: IntersectionState, axis: Axis) -> float:
    """Total queued vehicles on both approaches of ``axis``."""
    return sum(
        lane.vehicle_count for d, lane in state.lanes.items() if d.axis is axis
    )


def _emergency_axis(state: IntersectionState) -> Axis | None:
    """The axis of the first approach carrying an emergency vehicle, if any."""
    for direction in state.emergency_lanes():
        return direction.axis
    return None


class Controller(abc.ABC):
    """A signal controller: maps an observation to a safe signal command each tick."""

    name: str

    @abc.abstractmethod
    def decide(self, state: IntersectionState, dt: float) -> SignalCommand:
        """Return the command to apply for the next ``dt`` seconds."""

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset internal state for a fresh run."""

    @staticmethod
    def _action_for(previous: SignalPhase, nxt: SignalPhase) -> DecisionAction:
        return DecisionAction.KEEP_GREEN if nxt == previous else DecisionAction.SWITCH_PHASE


class FixedTimerController(Controller):
    """Fixed-split baseline: each green lasts exactly ``green_s`` before yielding."""

    def __init__(
        self,
        settings: DecisionSettings,
        *,
        green_s: float = 30.0,
        initial_axis: Axis = Axis.NORTH_SOUTH,
    ) -> None:
        self.name = "fixed_timer"
        self._settings = settings
        self._green_s = green_s
        self._initial_axis = initial_axis
        self._fsm = PhaseStateMachine(settings, initial_axis=initial_axis)

    def reset(self) -> None:
        self._fsm = PhaseStateMachine(self._settings, initial_axis=self._initial_axis)

    def decide(self, state: IntersectionState, dt: float) -> SignalCommand:
        request = self._fsm.phase.is_green and self._fsm.elapsed_s >= self._green_s
        previous = self._fsm.phase
        phase = self._fsm.step(dt, request_switch=request)
        return SignalCommand(
            intersection_id=state.intersection_id,
            action=self._action_for(previous, phase),
            target_phase=phase,
            duration_s=self._green_s,
            reason_code="fixed_timer_cycle",
            policy_version="fixed-timer-v1",
        )


class AdaptiveController(Controller):
    """Demand-responsive controller: longest-queue-first, fair, emergency-aware."""

    def __init__(
        self,
        settings: DecisionSettings,
        *,
        initial_axis: Axis = Axis.NORTH_SOUTH,
    ) -> None:
        self.name = "adaptive"
        self._settings = settings
        self._initial_axis = initial_axis
        self._fsm = PhaseStateMachine(settings, initial_axis=initial_axis)
        self._since_green: dict[Axis, float] = {Axis.NORTH_SOUTH: 0.0, Axis.EAST_WEST: 0.0}

    def reset(self) -> None:
        self._fsm = PhaseStateMachine(self._settings, initial_axis=self._initial_axis)
        self._since_green = {Axis.NORTH_SOUTH: 0.0, Axis.EAST_WEST: 0.0}

    def decide(self, state: IntersectionState, dt: float) -> SignalCommand:
        active = self._fsm.active_axis
        self._track_fairness(active, dt)

        request, action, reason = self._evaluate(state, active)
        previous = self._fsm.phase
        phase = self._fsm.step(dt, request_switch=request)
        if phase == previous and phase.is_green:
            action = DecisionAction.KEEP_GREEN

        return SignalCommand(
            intersection_id=state.intersection_id,
            action=action,
            target_phase=phase,
            duration_s=self._settings.min_green_s,
            reason_code=reason,
            feature_snapshot={
                "queue_ns": _axis_queue(state, Axis.NORTH_SOUTH),
                "queue_ew": _axis_queue(state, Axis.EAST_WEST),
                "phase_elapsed_s": self._fsm.elapsed_s,
            },
            policy_version="adaptive-lqf-v1",
        )

    # -- policy ------------------------------------------------------------
    def _evaluate(
        self, state: IntersectionState, active: Axis | None
    ) -> tuple[bool, DecisionAction, str]:
        """Return ``(request_switch, action, reason_code)`` for the current observation."""
        emergency = _emergency_axis(state)
        if emergency is not None:
            if active is not None and active is not emergency:
                return True, DecisionAction.EMERGENCY_OVERRIDE, "emergency_preemption"
            return False, DecisionAction.EMERGENCY_OVERRIDE, "emergency_hold_green"

        # Not in a green interval (yellow / all-red): let the machine finish clearance.
        if active is None:
            return False, DecisionAction.KEEP_GREEN, "clearance_interval"

        opposing = other_axis(active)
        if self._since_green[opposing] >= self._settings.max_starvation_s:
            return True, DecisionAction.SWITCH_PHASE, "fairness_anti_starvation"

        current_q = _axis_queue(state, active)
        opposing_q = _axis_queue(state, opposing)
        if current_q == 0 and opposing_q > 0:
            return True, DecisionAction.SWITCH_PHASE, "current_lane_empty"
        if opposing_q > current_q:
            return True, DecisionAction.SWITCH_PHASE, "opposing_queue_longer"
        return False, DecisionAction.KEEP_GREEN, "current_lane_busiest"

    def _track_fairness(self, active: Axis | None, dt: float) -> None:
        for axis in self._since_green:
            self._since_green[axis] += dt
        if active is not None:
            self._since_green[active] = 0.0


__all__ = ["AdaptiveController", "Controller", "FixedTimerController"]
