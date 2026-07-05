"""The signal phase state machine and safety envelope.

This is the **provably-safe core** of Sentinel's control system. Whatever a policy (fixed-timer,
adaptive, or the trained RL policy in a later module) *wants* to do, it can only express intent by
asking the :class:`PhaseStateMachine` to switch. The machine guarantees, by construction, that:

* a green interval is never shorter than ``min_green_s`` (unless preempted -- see below),
* a green interval never exceeds ``max_green_s`` (it force-switches),
* every switch between conflicting greens passes through ``yellow`` **and** ``all_red`` clearance,
* two conflicting greens are never shown back to back.

The :class:`SafetyEnvelope` independently validates the transitions the machine emits and increments
``sentinel_safety_violations_total`` if an illegal one ever slips through -- a belt-and-suspenders
check whose counter must remain zero in a correct system. Both classes are consumed here by the
simulation controllers and, unchanged, by the Signal Controller Agent in a later module.
"""

from __future__ import annotations

from sentinel.config.settings import DecisionSettings
from sentinel.contracts.enums import Axis, SignalPhase
from sentinel.observability.logging import get_logger
from sentinel.observability.metrics import SAFETY_VIOLATIONS

_log = get_logger("sentinel.control.phase")

_GREEN_FOR_AXIS: dict[Axis, SignalPhase] = {
    Axis.NORTH_SOUTH: SignalPhase.NS_GREEN,
    Axis.EAST_WEST: SignalPhase.EW_GREEN,
}
_YELLOW_FOR_AXIS: dict[Axis, SignalPhase] = {
    Axis.NORTH_SOUTH: SignalPhase.NS_YELLOW,
    Axis.EAST_WEST: SignalPhase.EW_YELLOW,
}


def other_axis(axis: Axis) -> Axis:
    """Return the conflicting axis."""
    return Axis.EAST_WEST if axis is Axis.NORTH_SOUTH else Axis.NORTH_SOUTH


def green_phase_for(axis: Axis) -> SignalPhase:
    """Return the green phase that serves ``axis``."""
    return _GREEN_FOR_AXIS[axis]


class PhaseStateMachine:
    """Deterministic, safety-enforcing signal sequencer.

    Call :meth:`step` once per control tick with the elapsed ``dt`` and whether the active policy
    *requests* a switch. The machine returns the phase that must be displayed for the next
    interval, having enforced all timing constraints.
    """

    def __init__(
        self,
        settings: DecisionSettings,
        *,
        initial_axis: Axis = Axis.NORTH_SOUTH,
    ) -> None:
        self._settings = settings
        self._phase = _GREEN_FOR_AXIS[initial_axis]
        self._elapsed = 0.0
        # The axis to serve after the current clearance completes.
        self._next_axis = other_axis(initial_axis)

    @property
    def phase(self) -> SignalPhase:
        return self._phase

    @property
    def elapsed_s(self) -> float:
        return self._elapsed

    @property
    def active_axis(self) -> Axis | None:
        """The axis currently receiving green/yellow, or ``None`` during all-red."""
        return self._phase.axis

    def can_switch(self) -> bool:
        """Whether a policy-requested switch would be honoured right now (min-green satisfied)."""
        return self._phase.is_green and self._elapsed >= self._settings.min_green_s

    def must_switch(self) -> bool:
        """Whether the machine will force a switch on the next tick (max-green reached)."""
        return self._phase.is_green and self._elapsed >= self._settings.max_green_s

    def step(self, dt: float, *, request_switch: bool) -> SignalPhase:
        """Advance the machine by ``dt`` seconds and return the resulting phase.

        Args:
            dt: Elapsed time since the previous tick, in seconds (> 0).
            request_switch: The active policy's intent to end the current green early. Ignored
                unless ``min_green_s`` has elapsed; overridden by the ``max_green_s`` force-switch.
        """
        if dt <= 0:
            raise ValueError("dt must be positive")
        self._elapsed += dt
        phase = self._phase

        if phase.is_green:
            reached_max = self._elapsed >= self._settings.max_green_s
            allowed_early = self._elapsed >= self._settings.min_green_s
            if reached_max or (request_switch and allowed_early):
                axis = phase.axis
                assert axis is not None  # green phases always have an axis
                self._next_axis = other_axis(axis)
                self._enter(_YELLOW_FOR_AXIS[axis])
        elif phase in (SignalPhase.NS_YELLOW, SignalPhase.EW_YELLOW):
            if self._elapsed >= self._settings.yellow_s:
                self._enter(SignalPhase.ALL_RED)
        elif phase is SignalPhase.ALL_RED:
            if self._elapsed >= self._settings.all_red_s:
                self._enter(_GREEN_FOR_AXIS[self._next_axis])

        return self._phase

    def _enter(self, new_phase: SignalPhase) -> None:
        _log.debug("phase_transition", from_phase=self._phase, to_phase=new_phase)
        self._phase = new_phase
        self._elapsed = 0.0


class SafetyEnvelope:
    """Independent validator for phase transitions.

    The :class:`PhaseStateMachine` is safe by construction; this class re-checks its output so any
    future policy that bypasses the machine, or any regression, is caught and counted. It never
    mutates state -- it only reports (and meters) violations.
    """

    #: Transitions that are always legal, expressed as ``(from, to)`` pairs.
    _LEGAL: frozenset[tuple[SignalPhase, SignalPhase]] = frozenset(
        {
            (SignalPhase.NS_GREEN, SignalPhase.NS_GREEN),
            (SignalPhase.NS_GREEN, SignalPhase.NS_YELLOW),
            (SignalPhase.NS_YELLOW, SignalPhase.NS_YELLOW),
            (SignalPhase.NS_YELLOW, SignalPhase.ALL_RED),
            (SignalPhase.EW_GREEN, SignalPhase.EW_GREEN),
            (SignalPhase.EW_GREEN, SignalPhase.EW_YELLOW),
            (SignalPhase.EW_YELLOW, SignalPhase.EW_YELLOW),
            (SignalPhase.EW_YELLOW, SignalPhase.ALL_RED),
            (SignalPhase.ALL_RED, SignalPhase.ALL_RED),
            (SignalPhase.ALL_RED, SignalPhase.NS_GREEN),
            (SignalPhase.ALL_RED, SignalPhase.EW_GREEN),
        }
    )

    @classmethod
    def is_legal_transition(cls, previous: SignalPhase, nxt: SignalPhase) -> bool:
        """Return whether moving from ``previous`` to ``nxt`` is a permitted transition."""
        return (previous, nxt) in cls._LEGAL

    @classmethod
    def validate_transition(
        cls, previous: SignalPhase, nxt: SignalPhase, *, intersection_id: str
    ) -> bool:
        """Validate a transition, metering a violation if it is illegal.

        Returns ``True`` if legal. Returns ``False`` (and increments
        ``sentinel_safety_violations_total``) if illegal -- notably any conflicting green-to-green
        transition that skips the yellow/all-red clearance.
        """
        if cls.is_legal_transition(previous, nxt):
            return True
        SAFETY_VIOLATIONS.labels(
            intersection=intersection_id, constraint="illegal_phase_transition"
        ).inc()
        _log.error(
            "safety_violation",
            intersection=intersection_id,
            from_phase=previous,
            to_phase=nxt,
            constraint="illegal_phase_transition",
        )
        return False


__all__ = [
    "PhaseStateMachine",
    "SafetyEnvelope",
    "green_phase_for",
    "other_axis",
]
