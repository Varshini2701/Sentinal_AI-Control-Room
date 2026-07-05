"""Versioned domain events exchanged on the Sentinel AI event bus.

Every event carries provenance (``source``, ``correlation_id``, ``causation_id``) so that a
full decision can be reconstructed from the audit log, and a ``schema_version`` so consumers
can evolve independently. Concrete events register themselves in :data:`EVENT_REGISTRY` at
import time, enabling :func:`deserialize_event` to reconstruct the right subclass from a wire
payload without the consumer knowing the type in advance.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from sentinel.contracts.base import utcnow
from sentinel.contracts.value_objects import (
    AgentHealth,
    Explanation,
    Forecast,
    HistoricalContext,
    Incident,
    IntersectionState,
    SignalCommand,
    SignalState,
)


def _new_event_id() -> str:
    return uuid.uuid4().hex


class DomainEvent(BaseModel):
    """Base class for all events on the bus.

    Subclasses must set the class variables :attr:`event_type` and :attr:`schema_version`,
    and are auto-registered in :data:`EVENT_REGISTRY`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # --- class-level metadata (not serialised as fields) ---
    event_type: ClassVar[str]
    schema_version: ClassVar[int] = 1

    # --- instance fields (serialised) ---
    event_id: str = Field(default_factory=_new_event_id)
    occurred_at: datetime = Field(default_factory=utcnow)
    source: str = Field(min_length=1, description="Name of the agent that emitted the event")
    intersection_id: str = Field(min_length=1)
    correlation_id: str | None = Field(
        default=None, description="Groups events belonging to one logical flow"
    )
    causation_id: str | None = Field(
        default=None, description="event_id of the event that directly caused this one"
    )

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Only register concrete events that declare an event_type.
        event_type = cls.__dict__.get("event_type")
        if event_type is not None:
            if event_type in EVENT_REGISTRY:
                raise RuntimeError(
                    f"duplicate event_type {event_type!r} "
                    f"({cls.__name__} vs {EVENT_REGISTRY[event_type].__name__})"
                )
            EVENT_REGISTRY[event_type] = cls

    @property
    def routing_key(self) -> str:
        """The bus routing key for this event (``<event_type>.<intersection_id>``)."""
        return f"{self.event_type}.{self.intersection_id}"

    def to_envelope_dict(self) -> dict[str, Any]:
        """Serialise the event plus its type metadata for transport."""
        return {
            "event_type": self.event_type,
            "schema_version": self.schema_version,
            "payload": self.model_dump(mode="json"),
        }


EVENT_REGISTRY: dict[str, type[DomainEvent]] = {}
"""Maps ``event_type`` -> concrete :class:`DomainEvent` subclass. Populated at import time."""


class UnknownEventTypeError(LookupError):
    """Raised when a wire payload references an event type not present in the registry."""


def deserialize_event(data: dict[str, Any]) -> DomainEvent:
    """Reconstruct a concrete :class:`DomainEvent` from a transport dict.

    Args:
        data: A dict produced by :meth:`DomainEvent.to_envelope_dict`.

    Raises:
        UnknownEventTypeError: if ``event_type`` is not registered.
    """
    event_type = data.get("event_type")
    if event_type not in EVENT_REGISTRY:
        raise UnknownEventTypeError(f"unregistered event_type: {event_type!r}")
    cls = EVENT_REGISTRY[event_type]
    return cls.model_validate(data["payload"])


# ---------------------------------------------------------------------------
# Concrete events -- one per cross-plane / inter-agent message
# ---------------------------------------------------------------------------
class StateUpdated(DomainEvent):
    """Emitted by the Perception plane whenever a new intersection snapshot is available."""

    event_type: ClassVar[str] = "state.updated"
    state: IntersectionState


class BaselineUpdated(DomainEvent):
    """Emitted by the Traffic Memory Agent with a refreshed historical baseline."""

    event_type: ClassVar[str] = "history.baseline.updated"
    baseline: HistoricalContext


class PredictionUpdated(DomainEvent):
    """Emitted by the Prediction Agent with a refreshed short-horizon forecast."""

    event_type: ClassVar[str] = "prediction.updated"
    forecast: Forecast


class DecisionMade(DomainEvent):
    """Emitted by the Decision Agent after selecting (and safety-clamping) an action."""

    event_type: ClassVar[str] = "decision.made"
    command: SignalCommand


class SignalChanged(DomainEvent):
    """Emitted by the Signal Controller after a phase transition is actuated."""

    event_type: ClassVar[str] = "signal.changed"
    previous_phase: SignalState | None = None
    signal: SignalState


class IncidentDetected(DomainEvent):
    """Emitted by the Incident Detection Agent."""

    event_type: ClassVar[str] = "incident.detected"
    incident: Incident


class ExplanationGenerated(DomainEvent):
    """Emitted by the Explainability Agent for a prior :class:`DecisionMade`."""

    event_type: ClassVar[str] = "explanation.generated"
    explanation: Explanation


class SystemModeChanged(DomainEvent):
    """Emitted by the Orchestrator when an intersection changes operating mode."""

    event_type: ClassVar[str] = "system.mode.changed"
    previous_mode: str
    new_mode: str
    reason: str


class AgentHeartbeat(DomainEvent):
    """Periodic health signal emitted by every agent."""

    event_type: ClassVar[str] = "agent.heartbeat"
    health: AgentHealth


__all__ = [
    "EVENT_REGISTRY",
    "AgentHeartbeat",
    "BaselineUpdated",
    "DecisionMade",
    "DomainEvent",
    "ExplanationGenerated",
    "IncidentDetected",
    "PredictionUpdated",
    "SignalChanged",
    "StateUpdated",
    "SystemModeChanged",
    "UnknownEventTypeError",
    "deserialize_event",
]
