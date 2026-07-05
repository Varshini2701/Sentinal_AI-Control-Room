"""Safety-critical control primitives shared by the simulator and the Signal Controller Agent."""

from __future__ import annotations

from sentinel.control.phase import (
    PhaseStateMachine,
    SafetyEnvelope,
    green_phase_for,
    other_axis,
)

__all__ = [
    "PhaseStateMachine",
    "SafetyEnvelope",
    "green_phase_for",
    "other_axis",
]
