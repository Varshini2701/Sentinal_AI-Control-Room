"""The Incident Detection Agent: applies incident rules with debounce and publishes findings.

Consumes ``state.updated`` and ``history.baseline.updated`` (to keep the latest baseline for
:class:`AbnormalCongestionRule`), runs every configured rule, and emits ``incident.detected`` for
each new finding. A per ``(direction, incident_type)`` debounce window prevents re-raising the same
ongoing incident every tick -- the audit trail records onset, not every subsequent frame.
"""

from __future__ import annotations

from sentinel.agents.base import BaseAgent
from sentinel.config.settings import IncidentSettings
from sentinel.contracts.enums import Direction, IncidentType
from sentinel.contracts.events import BaselineUpdated, DomainEvent, IncidentDetected, StateUpdated
from sentinel.contracts.value_objects import HistoricalContext
from sentinel.incident.rules import AbnormalCongestionRule, IncidentRule, StalledVehicleRule
from sentinel.messaging.bus import EventBus


class IncidentDetectionAgent(BaseAgent):
    """Event-driven agent that watches for abnormal traffic behaviour."""

    def __init__(
        self,
        *,
        event_bus: EventBus,
        intersection_id: str,
        settings: IncidentSettings | None = None,
        rules: list[IncidentRule] | None = None,
        heartbeat_interval_s: float = 5.0,
    ) -> None:
        self._settings = settings or IncidentSettings()
        self._rules = rules or [
            StalledVehicleRule(self._settings),
            AbnormalCongestionRule(self._settings),
        ]
        self._baseline: HistoricalContext | None = None
        self._last_raised: dict[tuple[Direction, IncidentType], float] = {}
        super().__init__(
            name="incident-detection",
            event_bus=event_bus,
            intersection_id=intersection_id,
            heartbeat_interval_s=heartbeat_interval_s,
        )

    def _register(self) -> None:
        self._subscribe("state.updated", self._on_state)
        self._subscribe("history.baseline.updated", self._on_baseline)

    async def _on_baseline(self, event: DomainEvent) -> None:
        if not isinstance(event, BaselineUpdated):
            return
        self._baseline = event.baseline

    async def _on_state(self, event: DomainEvent) -> None:
        if not isinstance(event, StateUpdated):
            return
        state = event.state
        now = state.timestamp.timestamp()

        for rule in self._rules:
            for incident in rule.evaluate(state, self._baseline):
                key = (incident.direction or Direction.NORTH, incident.incident_type)
                last = self._last_raised.get(key)
                if last is not None and (now - last) < self._settings.debounce_s:
                    continue
                self._last_raised[key] = now
                await self._publish(
                    IncidentDetected(
                        source=self.name,
                        intersection_id=self._intersection_id,
                        incident=incident,
                        correlation_id=event.correlation_id or event.event_id,
                        causation_id=event.event_id,
                    )
                )
                self._log.warning(
                    "incident_detected",
                    incident_type=incident.incident_type,
                    direction=incident.direction,
                    severity=incident.severity,
                )


__all__ = ["IncidentDetectionAgent"]
