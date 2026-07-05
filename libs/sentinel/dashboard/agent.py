"""The Dashboard Agent: a CQRS read side that never writes, only observes.

Subscribes to every event type a live dashboard needs, keeps a :class:`LiveSnapshot` up to date,
and maintains a bounded, human-readable log buffer (the "system logs" panel). It publishes
nothing -- the API gateway (or any other consumer) polls :meth:`snapshot` / :meth:`logs` directly,
or a thin adapter forwards them over a WebSocket, as done in ``services/api_gateway``.
"""

from __future__ import annotations

from collections import deque
from datetime import UTC, datetime

from sentinel.agents.base import BaseAgent
from sentinel.contracts.enums import SystemMode
from sentinel.contracts.events import (
    AgentHeartbeat,
    DecisionMade,
    DomainEvent,
    ExplanationGenerated,
    IncidentDetected,
    PredictionUpdated,
    SignalChanged,
    StateUpdated,
    SystemModeChanged,
)
from sentinel.dashboard.readmodel import LiveSnapshot
from sentinel.messaging.bus import EventBus

_SUBSCRIBED_TYPES = (
    "state.updated",
    "decision.made",
    "signal.changed",
    "prediction.updated",
    "incident.detected",
    "explanation.generated",
    "agent.heartbeat",
    "system.mode.changed",
)


class DashboardAgent(BaseAgent):
    """Maintains a live, queryable projection of the whole intersection for the UI."""

    def __init__(
        self,
        *,
        event_bus: EventBus,
        intersection_id: str,
        log_capacity: int = 200,
        heartbeat_interval_s: float = 5.0,
    ) -> None:
        self._snapshot = LiveSnapshot(intersection_id=intersection_id)
        self._logs: deque[str] = deque(maxlen=log_capacity)
        super().__init__(
            name="dashboard",
            event_bus=event_bus,
            intersection_id=intersection_id,
            heartbeat_interval_s=heartbeat_interval_s,
        )

    def _register(self) -> None:
        for event_type in _SUBSCRIBED_TYPES:
            self._subscribe(event_type, self._on_event)

    def snapshot(self) -> LiveSnapshot:
        return self._snapshot

    def logs(self) -> list[str]:
        return list(self._logs)

    async def _on_event(self, event: DomainEvent) -> None:
        self._snapshot.updated_at = datetime.now(tz=UTC)
        if isinstance(event, StateUpdated):
            self._snapshot.state = event.state
        elif isinstance(event, DecisionMade):
            self._snapshot.decision = event.command
            self._record(f"decision: {event.command.action.value} ({event.command.reason_code})")
        elif isinstance(event, SignalChanged):
            self._snapshot.signal = event.signal
            self._record(f"signal: -> {event.signal.phase.value}")
        elif isinstance(event, PredictionUpdated):
            self._snapshot.forecast = event.forecast
        elif isinstance(event, IncidentDetected):
            self._snapshot.latest_incident = event.incident
            self._record(
                f"INCIDENT [{event.incident.severity.value}] "
                f"{event.incident.incident_type.value}: {event.incident.description}"
            )
        elif isinstance(event, ExplanationGenerated):
            self._snapshot.latest_explanation = event.explanation
            self._record(f"explanation: {event.explanation.text}")
        elif isinstance(event, AgentHeartbeat):
            self._snapshot.agent_health[event.health.agent_name] = event.health
        elif isinstance(event, SystemModeChanged):
            self._snapshot.mode = SystemMode(event.new_mode)
            self._snapshot.degraded_reason = event.reason if event.new_mode != "ai" else None
            self._record(f"mode: {event.previous_mode} -> {event.new_mode} ({event.reason})")

    def _record(self, message: str) -> None:
        timestamp = datetime.now(tz=UTC).strftime("%H:%M:%S")
        self._logs.append(f"[{timestamp}] {message}")


__all__ = ["DashboardAgent"]
