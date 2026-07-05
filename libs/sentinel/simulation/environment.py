"""The :class:`TrafficEnvironment` port -- the boundary between control and the world.

A controller neither knows nor cares whether it is driving a lightweight analytical model or a
full SUMO microsimulation: both implement this port. That is the whole point of the abstraction --
the closed-loop benchmark, the Decision Agent and the demo all run unchanged against either.
"""

from __future__ import annotations

import abc

from sentinel.contracts.value_objects import IntersectionState, SignalCommand
from sentinel.simulation.kpi import KpiSummary


class TrafficEnvironment(abc.ABC):
    """Abstract, steppable traffic environment.

    Lifecycle mirrors a reinforcement-learning environment: :meth:`reset` to obtain the initial
    observation, then repeatedly :meth:`step` a :class:`SignalCommand` to advance one control tick
    and receive the next :class:`IntersectionState`. :meth:`metrics` returns physics-truth KPIs at
    any point; :meth:`close` releases resources.
    """

    @abc.abstractmethod
    def reset(self) -> IntersectionState:
        """Reset to the initial state and return the first observation."""

    @abc.abstractmethod
    def step(self, command: SignalCommand) -> IntersectionState:
        """Apply ``command`` for one tick, advance the world, and return the new observation."""

    @abc.abstractmethod
    def metrics(self) -> KpiSummary:
        """Return aggregate KPIs accumulated since the last :meth:`reset`."""

    @property
    @abc.abstractmethod
    def time_s(self) -> float:
        """Current simulation time in seconds since reset."""

    def close(self) -> None:  # noqa: B027 - intentional concrete no-op default; subclasses override
        """Release any resources held by the environment (no-op by default)."""

    def __enter__(self) -> TrafficEnvironment:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


__all__ = ["TrafficEnvironment"]
