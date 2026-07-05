"""The state-history repository port and its in-process implementation.

:class:`StateHistoryRepository` is the boundary between the Traffic Memory Agent and durable
storage. :class:`InMemoryStateHistoryRepository` keeps a bounded rolling window per intersection --
enough for baselines, short-horizon prediction and tests -- with no external dependency. A
PostgreSQL-backed implementation (for long-term analytics/replay) is added when the deployment
module wires a real database; it implements this same port so the agent above it never changes.
"""

from __future__ import annotations

import abc
from collections import deque

from sentinel.contracts.enums import Direction
from sentinel.contracts.value_objects import HistoricalContext, IntersectionState, LaneBaseline


class StateHistoryRepository(abc.ABC):
    """Persists and serves historical :class:`IntersectionState` snapshots."""

    @abc.abstractmethod
    def append(self, state: IntersectionState) -> None:
        """Record one snapshot for its intersection."""

    @abc.abstractmethod
    def recent(self, intersection_id: str, limit: int | None = None) -> list[IntersectionState]:
        """Return the most recent snapshots for an intersection, oldest first."""

    @abc.abstractmethod
    def baseline(self, intersection_id: str) -> HistoricalContext | None:
        """Compute a rolling-average baseline from retained history, or ``None`` if empty."""

    @abc.abstractmethod
    def count(self, intersection_id: str) -> int:
        """Number of snapshots currently retained for an intersection."""


class InMemoryStateHistoryRepository(StateHistoryRepository):
    """Bounded per-intersection rolling window, held entirely in process memory.

    The baseline is a simple rolling average over the retained window -- not a time-of-day model --
    which is an intentional simplification: it captures "typical recent demand" well enough to
    seed predictions and fairness heuristics, and a richer time-bucketed baseline can be layered on
    top by a PostgreSQL-backed repository without changing this interface.
    """

    def __init__(self, window_size: int = 3600) -> None:
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        self._window_size = window_size
        self._history: dict[str, deque[IntersectionState]] = {}

    def append(self, state: IntersectionState) -> None:
        window = self._history.setdefault(
            state.intersection_id, deque(maxlen=self._window_size)
        )
        window.append(state)

    def recent(self, intersection_id: str, limit: int | None = None) -> list[IntersectionState]:
        window = self._history.get(intersection_id)
        if not window:
            return []
        items = list(window)
        return items[-limit:] if limit is not None else items

    def count(self, intersection_id: str) -> int:
        return len(self._history.get(intersection_id, ()))

    def baseline(self, intersection_id: str) -> HistoricalContext | None:
        window = self._history.get(intersection_id)
        if not window:
            return None

        totals: dict[Direction, tuple[float, float, int]] = {}
        for state in window:
            for direction, lane in state.lanes.items():
                queue_sum, wait_sum, n = totals.get(direction, (0.0, 0.0, 0))
                totals[direction] = (
                    queue_sum + lane.vehicle_count,
                    wait_sum + lane.avg_wait_s,
                    n + 1,
                )

        lanes = {
            direction: LaneBaseline(
                direction=direction,
                avg_queue_veh=queue_sum / n,
                avg_wait_s=wait_sum / n,
                sample_count=n,
            )
            for direction, (queue_sum, wait_sum, n) in totals.items()
        }
        return HistoricalContext(
            intersection_id=intersection_id, window_size=len(window), lanes=lanes
        )


__all__ = ["InMemoryStateHistoryRepository", "StateHistoryRepository"]
