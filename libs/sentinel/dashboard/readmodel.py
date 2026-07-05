"""The dashboard read-model: a CQRS read side aggregating the latest state of everything.

:class:`LiveSnapshot` is a plain, JSON-serialisable-by-convention snapshot of "what's true right
now" for one intersection -- the exact shape a dashboard needs to render in a single glance,
without joining across events itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from sentinel.contracts.enums import SystemMode
from sentinel.contracts.value_objects import (
    AgentHealth,
    Explanation,
    Forecast,
    Incident,
    IntersectionState,
    SignalCommand,
    SignalState,
)


@dataclass(slots=True)
class LiveSnapshot:
    """Mutable, in-process read-model updated as events arrive; read-only to consumers."""

    intersection_id: str
    mode: SystemMode = SystemMode.AI
    degraded_reason: str | None = None
    state: IntersectionState | None = None
    signal: SignalState | None = None
    decision: SignalCommand | None = None
    forecast: Forecast | None = None
    latest_incident: Incident | None = None
    latest_explanation: Explanation | None = None
    agent_health: dict[str, AgentHealth] = field(default_factory=dict)
    updated_at: datetime | None = None

    def to_dict(self) -> dict[str, object]:
        """A plain-dict projection convenient for JSON APIs / WebSocket payloads."""
        return {
            "intersection_id": self.intersection_id,
            "mode": self.mode.value,
            "degraded_reason": self.degraded_reason,
            "state": self.state.model_dump(mode="json") if self.state else None,
            "signal": self.signal.model_dump(mode="json") if self.signal else None,
            "decision": self.decision.model_dump(mode="json") if self.decision else None,
            "forecast": self.forecast.model_dump(mode="json") if self.forecast else None,
            "latest_incident": (
                self.latest_incident.model_dump(mode="json") if self.latest_incident else None
            ),
            "latest_explanation": (
                self.latest_explanation.model_dump(mode="json")
                if self.latest_explanation
                else None
            ),
            "agent_health": {
                name: health.model_dump(mode="json") for name, health in self.agent_health.items()
            },
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


__all__ = ["LiveSnapshot"]
