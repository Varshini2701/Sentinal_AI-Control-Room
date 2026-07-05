"""The Traffic Memory Agent: persists history and serves rolling baselines.

Consumes every ``state.updated`` event and appends it to the :class:`StateHistoryRepository`.
Every ``baseline_publish_every`` updates it recomputes the rolling baseline and emits
``history.baseline.updated`` for the Prediction Agent (and any dashboard/analytics consumer) --
this is deliberately throttled since the baseline changes slowly relative to per-tick state.
"""

from __future__ import annotations

from sentinel.agents.base import BaseAgent
from sentinel.config.settings import MemorySettings
from sentinel.contracts.events import BaselineUpdated, DomainEvent, StateUpdated
from sentinel.contracts.value_objects import HistoricalContext
from sentinel.memory.repository import InMemoryStateHistoryRepository, StateHistoryRepository
from sentinel.messaging.bus import EventBus


class TrafficMemoryAgent(BaseAgent):
    """Event-driven agent that persists intersection state and serves historical baselines."""

    def __init__(
        self,
        *,
        event_bus: EventBus,
        intersection_id: str,
        settings: MemorySettings | None = None,
        repository: StateHistoryRepository | None = None,
        heartbeat_interval_s: float = 5.0,
    ) -> None:
        self._settings = settings or MemorySettings()
        self._repository = repository or InMemoryStateHistoryRepository(
            window_size=self._settings.window_size
        )
        self._updates_since_publish = 0
        super().__init__(
            name="traffic-memory",
            event_bus=event_bus,
            intersection_id=intersection_id,
            heartbeat_interval_s=heartbeat_interval_s,
        )

    def _register(self) -> None:
        self._subscribe("state.updated", self._on_state)

    @property
    def repository(self) -> StateHistoryRepository:
        """Expose the repository read-side for co-located queries (e.g. by the Prediction Agent)."""
        return self._repository

    def current_baseline(self) -> HistoricalContext | None:
        return self._repository.baseline(self._intersection_id)

    async def _on_state(self, event: DomainEvent) -> None:
        if not isinstance(event, StateUpdated):
            return
        self._repository.append(event.state)
        self._updates_since_publish += 1
        if self._updates_since_publish >= self._settings.baseline_publish_every:
            self._updates_since_publish = 0
            await self._publish_baseline()

    async def _publish_baseline(self) -> None:
        baseline = self.current_baseline()
        if baseline is None:
            return
        await self._publish(
            BaselineUpdated(
                source=self.name, intersection_id=self._intersection_id, baseline=baseline
            )
        )
        self._log.debug("baseline_published", samples=baseline.window_size)


__all__ = ["TrafficMemoryAgent"]
