"""Tests for the event-driven cognition/control service agents."""

from __future__ import annotations

from typing import cast

from sentinel.config.settings import DecisionSettings
from sentinel.contracts.enums import AgentStatus, DensityLevel, Direction, SystemMode
from sentinel.contracts.events import (
    AgentHeartbeat,
    DecisionMade,
    DomainEvent,
    ExplanationGenerated,
    IncidentDetected,
    SignalChanged,
    StateUpdated,
)
from sentinel.contracts.value_objects import IntersectionState, LaneState
from sentinel.messaging import InMemoryEventBus
from sentinel.services import (
    DecisionAgent,
    ExplanationAgent,
    InMemorySignalActuator,
    SignalControllerAgent,
)
from sentinel.services.agents import IncidentDetectionAgent, IncidentRule


async def test_decision_agent_publishes_decision(state: IntersectionState) -> None:
    bus = InMemoryEventBus()
    decisions: list[DecisionMade] = []

    async def collect(event: DomainEvent) -> None:
        decisions.append(cast(DecisionMade, event))

    DecisionAgent(bus, intersection_id=state.intersection_id).subscribe()
    bus.subscribe(DecisionMade.event_type, collect, consumer_name="collector")

    await bus.start()
    await bus.publish(
        StateUpdated(source="perception", intersection_id=state.intersection_id, state=state)
    )
    await bus.join()
    await bus.stop()

    assert len(decisions) == 1
    assert decisions[0].command.intersection_id == state.intersection_id
    assert decisions[0].causation_id is not None


async def test_decision_agent_uses_fallback_when_confidence_is_low(
    state: IntersectionState,
) -> None:
    bus = InMemoryEventBus()
    decisions: list[DecisionMade] = []
    low_confidence = state.model_copy(update={"perception_confidence": 0.1})

    async def collect(event: DomainEvent) -> None:
        decisions.append(cast(DecisionMade, event))

    DecisionAgent(
        bus,
        intersection_id=state.intersection_id,
        settings=DecisionSettings(min_perception_confidence=0.5),
    ).subscribe()
    bus.subscribe(DecisionMade.event_type, collect, consumer_name="collector")

    await bus.start()
    await bus.publish(
        StateUpdated(
            source="perception", intersection_id=state.intersection_id, state=low_confidence
        )
    )
    await bus.join()
    await bus.stop()

    assert decisions[0].command.policy_version == "degraded-fixed-timer-v1"
    assert "degraded_fallback" in decisions[0].command.constraints_applied


async def test_signal_controller_applies_command_and_emits_signal(
    state: IntersectionState,
) -> None:
    bus = InMemoryEventBus()
    signals: list[SignalChanged] = []
    actuator = InMemorySignalActuator(intersection_id=state.intersection_id)

    async def collect(event: DomainEvent) -> None:
        signals.append(cast(SignalChanged, event))

    decision_agent = DecisionAgent(bus, intersection_id=state.intersection_id)
    signal_agent = SignalControllerAgent(
        bus, intersection_id=state.intersection_id, actuator=actuator
    )
    decision_agent.subscribe()
    signal_agent.subscribe()
    bus.subscribe(SignalChanged.event_type, collect, consumer_name="collector")

    await bus.start()
    await bus.publish(
        StateUpdated(source="perception", intersection_id=state.intersection_id, state=state)
    )
    await bus.join()
    await bus.stop()

    assert len(actuator.applied) == 1
    assert len(signals) == 1
    assert signals[0].signal.phase == actuator.applied[0].target_phase


async def test_explanation_agent_generates_template_explanation(
    state: IntersectionState,
) -> None:
    bus = InMemoryEventBus()
    explanations: list[ExplanationGenerated] = []

    async def collect(event: DomainEvent) -> None:
        explanations.append(cast(ExplanationGenerated, event))

    DecisionAgent(bus, intersection_id=state.intersection_id).subscribe()
    ExplanationAgent(
        bus, name="explanation-agent", intersection_id=state.intersection_id
    ).subscribe()
    bus.subscribe(ExplanationGenerated.event_type, collect, consumer_name="collector")

    await bus.start()
    await bus.publish(
        StateUpdated(source="perception", intersection_id=state.intersection_id, state=state)
    )
    await bus.join()
    await bus.stop()

    assert len(explanations) == 1
    assert explanations[0].explanation.text
    assert explanations[0].causation_id is not None


async def test_incident_detection_publishes_and_deduplicates_congestion(
    state: IntersectionState,
) -> None:
    bus = InMemoryEventBus()
    incidents: list[IncidentDetected] = []
    jammed_lane = LaneState(
        direction=Direction.EAST,
        vehicle_count=20,
        moving_count=0,
        stopped_count=20,
        queue_length_m=80.0,
        avg_wait_s=120.0,
        occupancy_pct=95.0,
        density=DensityLevel.JAM,
    )
    jammed_state = state.model_copy(
        update={"lanes": {**state.lanes, Direction.EAST: jammed_lane}}
    )

    async def collect(event: DomainEvent) -> None:
        incidents.append(cast(IncidentDetected, event))

    IncidentDetectionAgent(
        bus,
        intersection_id=state.intersection_id,
        rule=IncidentRule(jam_wait_s=60.0, jam_queue_m=40.0),
    ).subscribe()
    bus.subscribe(IncidentDetected.event_type, collect, consumer_name="collector")

    await bus.start()
    for _ in range(2):
        await bus.publish(
            StateUpdated(
                source="perception", intersection_id=state.intersection_id, state=jammed_state
            )
        )
    await bus.join()
    await bus.stop()

    assert len(incidents) == 1
    assert incidents[0].incident.direction is Direction.EAST


async def test_agent_heartbeat(state: IntersectionState) -> None:
    bus = InMemoryEventBus()
    heartbeats: list[AgentHeartbeat] = []
    agent = DecisionAgent(bus, intersection_id=state.intersection_id)

    async def collect(event: DomainEvent) -> None:
        heartbeats.append(cast(AgentHeartbeat, event))

    bus.subscribe(AgentHeartbeat.event_type, collect, consumer_name="collector")
    await bus.start()
    await agent.heartbeat(AgentStatus.DEGRADED)
    await bus.join()
    await bus.stop()

    assert heartbeats[0].health.agent_name == "decision-agent"
    assert heartbeats[0].health.status is AgentStatus.DEGRADED


async def test_decision_agent_uses_fallback_in_degraded_mode(
    state: IntersectionState,
) -> None:
    bus = InMemoryEventBus()
    decisions: list[DecisionMade] = []
    degraded = state.model_copy(update={"mode": SystemMode.DEGRADED})

    async def collect(event: DomainEvent) -> None:
        decisions.append(cast(DecisionMade, event))

    DecisionAgent(bus, intersection_id=state.intersection_id).subscribe()
    bus.subscribe(DecisionMade.event_type, collect, consumer_name="collector")

    await bus.start()
    await bus.publish(
        StateUpdated(source="perception", intersection_id=state.intersection_id, state=degraded)
    )
    await bus.join()
    await bus.stop()

    assert decisions[0].command.policy_version == "degraded-fixed-timer-v1"
