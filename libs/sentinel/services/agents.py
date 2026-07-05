"""Event-driven agents that wire Sentinel's domain logic into runnable services."""

from __future__ import annotations

import abc
from collections.abc import Iterable
from dataclasses import dataclass, field

from sentinel.config.settings import DecisionSettings
from sentinel.contracts.enums import (
    AgentStatus,
    DensityLevel,
    Direction,
    IncidentSeverity,
    IncidentType,
    SignalPhase,
    SystemMode,
)
from sentinel.contracts.events import (
    AgentHeartbeat,
    DecisionMade,
    DomainEvent,
    ExplanationGenerated,
    IncidentDetected,
    SignalChanged,
    StateUpdated,
)
from sentinel.contracts.value_objects import (
    AgentHealth,
    Explanation,
    Incident,
    IntersectionState,
    SignalCommand,
    SignalState,
)
from sentinel.messaging.bus import EventBus
from sentinel.services.ports import SignalActuator
from sentinel.simulation.controllers import AdaptiveController, Controller, FixedTimerController


class AgentService(abc.ABC):
    """Base class for a named, event-driven service."""

    event_types: tuple[str, ...]

    def __init__(self, bus: EventBus, *, name: str, intersection_id: str) -> None:
        self.bus = bus
        self.name = name
        self.intersection_id = intersection_id
        self._subscribed = False

    def subscribe(self) -> None:
        """Register this agent's handler with the event bus."""
        if self._subscribed:
            return
        self.bus.subscribe(self.event_types, self.handle, consumer_name=self.name)
        self._subscribed = True

    async def handle(self, event: DomainEvent) -> None:
        """Process one subscribed event."""
        if event.intersection_id != self.intersection_id:
            return
        await self._handle(event)

    async def heartbeat(self, status: AgentStatus = AgentStatus.HEALTHY) -> None:
        """Publish a health heartbeat for this agent."""
        await self.bus.publish(
            AgentHeartbeat(
                source=self.name,
                intersection_id=self.intersection_id,
                health=AgentHealth(agent_name=self.name, status=status),
            )
        )

    @abc.abstractmethod
    async def _handle(self, event: DomainEvent) -> None:
        """Handle an event already filtered to this intersection."""


class DecisionAgent(AgentService):
    """Consumes perception states and emits safe signal-control decisions."""

    event_types = (StateUpdated.event_type,)

    def __init__(
        self,
        bus: EventBus,
        *,
        intersection_id: str,
        settings: DecisionSettings | None = None,
        ai_controller: Controller | None = None,
        fallback_controller: Controller | None = None,
        name: str = "decision-agent",
    ) -> None:
        super().__init__(bus, name=name, intersection_id=intersection_id)
        self.settings = settings or DecisionSettings()
        self.ai_controller = ai_controller or AdaptiveController(self.settings)
        self.fallback_controller = fallback_controller or FixedTimerController(self.settings)

    async def _handle(self, event: DomainEvent) -> None:
        if not isinstance(event, StateUpdated):
            return
        controller = self._controller_for(event.state)
        command = controller.decide(event.state, dt=1.0)
        if controller is self.fallback_controller:
            command = command.model_copy(
                update={
                    "constraints_applied": (*command.constraints_applied, "degraded_fallback"),
                    "policy_version": "degraded-fixed-timer-v1",
                }
            )
        await self.bus.publish(
            DecisionMade(
                source=self.name,
                intersection_id=event.intersection_id,
                correlation_id=event.correlation_id or event.event_id,
                causation_id=event.event_id,
                command=command,
            )
        )

    def _controller_for(self, state: IntersectionState) -> Controller:
        if state.mode in (SystemMode.DEGRADED, SystemMode.FIXED_TIMER):
            return self.fallback_controller
        if state.mode is SystemMode.MANUAL:
            return self.fallback_controller
        if state.perception_confidence < self.settings.min_perception_confidence:
            return self.fallback_controller
        return self.ai_controller


class SignalControllerAgent(AgentService):
    """Applies decisions through an actuator and emits authoritative signal changes."""

    event_types = (DecisionMade.event_type,)

    def __init__(
        self,
        bus: EventBus,
        *,
        intersection_id: str,
        actuator: SignalActuator,
        name: str = "signal-controller",
    ) -> None:
        super().__init__(bus, name=name, intersection_id=intersection_id)
        self.actuator = actuator
        self._last_signal: SignalState | None = None

    async def _handle(self, event: DomainEvent) -> None:
        if not isinstance(event, DecisionMade):
            return
        previous = self._last_signal
        signal = await self.actuator.apply(event.command)
        self._last_signal = signal
        await self.bus.publish(
            SignalChanged(
                source=self.name,
                intersection_id=event.intersection_id,
                correlation_id=event.correlation_id or event.event_id,
                causation_id=event.event_id,
                previous_phase=previous,
                signal=signal,
            )
        )


class ExplanationAgent(AgentService):
    """Generates deterministic, operator-facing explanations for decisions."""

    event_types = (DecisionMade.event_type,)

    async def _handle(self, event: DomainEvent) -> None:
        if not isinstance(event, DecisionMade):
            return
        explanation = Explanation(
            intersection_id=event.intersection_id,
            decision_reason_code=event.command.reason_code,
            text=_explain_command(event.command),
            counterfactual=_counterfactual(event.command),
            generator="template-v1",
        )
        await self.bus.publish(
            ExplanationGenerated(
                source=self.name,
                intersection_id=event.intersection_id,
                correlation_id=event.correlation_id or event.event_id,
                causation_id=event.event_id,
                explanation=explanation,
            )
        )


@dataclass(slots=True)
class IncidentRule:
    """Thresholds used by the rule-based incident detector."""

    jam_wait_s: float = 90.0
    jam_queue_m: float = 50.0
    min_confidence: float = 0.75


class IncidentDetectionAgent(AgentService):
    """Raises incidents for severe congestion and emergency preemption signals."""

    event_types = (StateUpdated.event_type,)

    def __init__(
        self,
        bus: EventBus,
        *,
        intersection_id: str,
        rule: IncidentRule | None = None,
        name: str = "incident-agent",
    ) -> None:
        super().__init__(bus, name=name, intersection_id=intersection_id)
        self.rule = rule or IncidentRule()
        self._active_keys: set[tuple[IncidentType, Direction | None]] = set()

    async def _handle(self, event: DomainEvent) -> None:
        if not isinstance(event, StateUpdated):
            return
        for incident in self._detect(event.state):
            key = (incident.incident_type, incident.direction)
            if key in self._active_keys:
                continue
            self._active_keys.add(key)
            await self.bus.publish(
                IncidentDetected(
                    source=self.name,
                    intersection_id=event.intersection_id,
                    correlation_id=event.correlation_id or event.event_id,
                    causation_id=event.event_id,
                    incident=incident,
                )
            )
        self._forget_resolved(event.state)

    def _detect(self, state: IntersectionState) -> Iterable[Incident]:
        for lane in state.lanes.values():
            if lane.has_emergency_vehicle:
                yield Incident(
                    intersection_id=state.intersection_id,
                    incident_type=IncidentType.STALLED_VEHICLE,
                    severity=IncidentSeverity.HIGH,
                    direction=lane.direction,
                    confidence=0.8,
                    description=f"Emergency vehicle detected on {lane.direction.value} approach",
                )
            if (
                lane.density is DensityLevel.JAM
                and lane.avg_wait_s >= self.rule.jam_wait_s
                and lane.queue_length_m >= self.rule.jam_queue_m
            ):
                yield Incident(
                    intersection_id=state.intersection_id,
                    incident_type=IncidentType.ABNORMAL_CONGESTION,
                    severity=IncidentSeverity.MEDIUM,
                    direction=lane.direction,
                    confidence=self.rule.min_confidence,
                    description=f"Persistent jam on {lane.direction.value} approach",
                )

    def _forget_resolved(self, state: IntersectionState) -> None:
        active: set[tuple[IncidentType, Direction | None]] = set()
        for lane in state.lanes.values():
            if lane.has_emergency_vehicle:
                active.add((IncidentType.STALLED_VEHICLE, lane.direction))
            if (
                lane.density is DensityLevel.JAM
                and lane.avg_wait_s >= self.rule.jam_wait_s
                and lane.queue_length_m >= self.rule.jam_queue_m
            ):
                active.add((IncidentType.ABNORMAL_CONGESTION, lane.direction))
        self._active_keys.intersection_update(active)


@dataclass(slots=True)
class InMemorySignalActuator(SignalActuator):
    """Tiny actuator for local services and tests."""

    intersection_id: str
    phase_elapsed_s: float = 0.0
    phase_remaining_s: float = 0.0
    applied: list[SignalCommand] = field(default_factory=list)

    async def apply(self, command: SignalCommand) -> SignalState:
        self.applied.append(command)
        self.phase_elapsed_s = (
            0.0 if _starts_new_phase(command.target_phase) else self.phase_elapsed_s
        )
        self.phase_remaining_s = command.duration_s
        return SignalState(
            intersection_id=self.intersection_id,
            phase=command.target_phase,
            phase_elapsed_s=self.phase_elapsed_s,
            phase_remaining_s=self.phase_remaining_s,
        )


def _starts_new_phase(phase: SignalPhase) -> bool:
    return phase in {
        SignalPhase.NS_GREEN,
        SignalPhase.NS_YELLOW,
        SignalPhase.EW_GREEN,
        SignalPhase.EW_YELLOW,
        SignalPhase.ALL_RED,
    }


def _explain_command(command: SignalCommand) -> str:
    queues = command.feature_snapshot
    if command.reason_code == "emergency_preemption":
        return "Switching priority to the approach with an emergency vehicle."
    if command.reason_code == "fairness_anti_starvation":
        return "Switching phases because the opposing approach has waited too long."
    if command.reason_code == "opposing_queue_longer":
        return (
            "Switching because the opposing queue is larger "
            f"(NS={queues.get('queue_ns', 0):.0f}, EW={queues.get('queue_ew', 0):.0f})."
        )
    if command.reason_code == "current_lane_busiest":
        return "Holding green because the current approach remains the busiest."
    if "degraded_fallback" in command.constraints_applied:
        return "Using fixed-timer fallback because the intersection is in degraded control."
    return f"Applied {command.action.value} for reason {command.reason_code}."


def _counterfactual(command: SignalCommand) -> str | None:
    if command.action.value == "keep_green":
        return "Switching now would interrupt the currently served traffic stream."
    if command.action.value == "switch_phase":
        return "Keeping green would increase delay on the opposing approach."
    return None


__all__ = [
    "AgentService",
    "DecisionAgent",
    "ExplanationAgent",
    "InMemorySignalActuator",
    "IncidentDetectionAgent",
    "IncidentRule",
    "SignalControllerAgent",
]
