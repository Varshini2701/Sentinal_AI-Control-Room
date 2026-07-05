"""Signal actuator port and lightweight implementations.

The :class:`SignalActuator` is the seam between the Signal Controller and the physical (or
simulated) traffic light. In production a hardware/NTCIP adapter or the SUMO bridge implements it;
:class:`RecordingActuator` and :class:`CallableActuator` are used for tests and for wiring the
controller to the analytical twin in the closed-loop demo.
"""

from __future__ import annotations

import abc
from collections.abc import Callable

from sentinel.contracts.value_objects import SignalState


class SignalActuator(abc.ABC):
    """Applies a signal state to the underlying traffic light."""

    @abc.abstractmethod
    def apply(self, signal: SignalState) -> None:
        """Drive the light to the phase described by ``signal``."""


class RecordingActuator(SignalActuator):
    """Records every applied signal state -- for tests and inspection."""

    def __init__(self) -> None:
        self.applied: list[SignalState] = []

    def apply(self, signal: SignalState) -> None:
        self.applied.append(signal)

    @property
    def current(self) -> SignalState | None:
        return self.applied[-1] if self.applied else None


class CallableActuator(SignalActuator):
    """Adapts a plain callable into an actuator (e.g. to push a phase into a simulator)."""

    def __init__(self, sink: Callable[[SignalState], None]) -> None:
        self._sink = sink

    def apply(self, signal: SignalState) -> None:
        self._sink(signal)


__all__ = ["CallableActuator", "RecordingActuator", "SignalActuator"]
