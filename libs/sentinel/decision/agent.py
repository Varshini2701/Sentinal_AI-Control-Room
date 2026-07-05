"""The Decision Agent -- tier 1 of the safe controller.

Consumes ``state.updated`` (and ``prediction.updated`` when available), runs the
:class:`~sentinel.decision.policy.DecisionPolicy`, and emits a ``decision.made`` event carrying a
:class:`SignalCommand` of *intent* (which axis to serve, and keep/extend/reduce/switch). It tracks
per-axis time-since-green for the fairness term, and records decision latency for the control-loop
SLO. It never actuates -- the Signal Controller does that, safely.
"""

from __future__ import annotations

from datetime import datetime

from sentinel.agents.base import BaseAgent
from sentinel.config.settings import DecisionSettings
from sentinel.contracts.enums import Axis
from sentinel.contracts.events import (
    DecisionMade,
    DomainEvent,
    PredictionUpdated,
    StateUpdated,
)
from sentinel.contracts.value_objects import Forecast, IntersectionState, SignalCommand
from sentinel.control.phase import green_phase_for
from sentinel.decision.policy import DecisionContext, DecisionPolicy, UtilityPolicy
from sentinel.messaging.bus import EventBus
from sentinel.observability.metrics import DECISION_LATENCY_SECONDS, observe_duration


class DecisionAgent(BaseAgent):
    """Event-driven agent that turns intersection state into signal-command intent."""

    def __init__(
        self,
        *,
        event_bus: EventBus,
        intersection_id: str,
        settings: DecisionSettings | None = None,
        policy: DecisionPolicy | None = None,
        heartbeat_interval_s: float = 5.0,
    ) -> None:
        self._settings = settings or DecisionSettings()
        self._policy = policy or UtilityPolicy(self._settings)
        self._since_green: dict[Axis, float] = {Axis.NORTH_SOUTH: 0.0, Axis.EAST_WEST: 0.0}
        self._prev_ts: datetime | None = None
        self._forecast: Forecast | None = None
        super().__init__(
            name="decision-agent",
            event_bus=event_bus,
            intersection_id=intersection_id,
            heartbeat_interval_s=heartbeat_interval_s,
        )

    def _register(self) -> None:
        self._subscribe("state.updated", self._on_state)
        self._subscribe("prediction.updated", self._on_prediction)

    async def _on_prediction(self, event: DomainEvent) -> None:
        if not isinstance(event, PredictionUpdated):
            return
        self._forecast = event.forecast

    async def _on_state(self, event: DomainEvent) -> None:
        if not isinstance(event, StateUpdated):
            return
        state = event.state
        self._advance_fairness(state)

        context = DecisionContext(
            state=state, since_green_s=dict(self._since_green), forecast=self._forecast
        )
        with observe_duration(DECISION_LATENCY_SECONDS, intersection=self._intersection_id):
            outcome = self._policy.decide(context)

        command = SignalCommand(
            intersection_id=self._intersection_id,
            action=outcome.action,
            target_phase=green_phase_for(outcome.desired_axis),
            duration_s=self._settings.min_green_s,
            reason_code=outcome.reason_code,
            feature_snapshot={
                **outcome.features,
                "score_ns": outcome.scores.get(Axis.NORTH_SOUTH.value, 0.0),
                "score_ew": outcome.scores.get(Axis.EAST_WEST.value, 0.0),
            },
            rejected_alternatives=outcome.rejected,
            constraints_applied=(),
            policy_version="utility-v1",
        )
        await self._publish(
            DecisionMade(
                source=self.name,
                intersection_id=self._intersection_id,
                command=command,
                correlation_id=event.correlation_id or event.event_id,
                causation_id=event.event_id,
            )
        )
        self._log.debug(
            "decision_made", action=command.action, reason=command.reason_code,
            target=command.target_phase,
        )

    def _advance_fairness(self, state: IntersectionState) -> None:
        """Accrue time-since-green per axis from consecutive state timestamps."""
        dt = (state.timestamp - self._prev_ts).total_seconds() if self._prev_ts is not None else 0.0
        self._prev_ts = state.timestamp
        for axis in self._since_green:
            self._since_green[axis] += max(0.0, dt)
        phase = state.current_phase
        if phase.is_green and phase.axis is not None:
            self._since_green[phase.axis] = 0.0


__all__ = ["DecisionAgent"]
