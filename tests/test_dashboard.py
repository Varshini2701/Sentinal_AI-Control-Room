"""Tests for the Dashboard Agent's live read-model."""

from __future__ import annotations

from datetime import UTC, datetime

from sentinel.contracts.enums import (
    AgentStatus,
    DecisionAction,
    DensityLevel,
    Direction,
    IncidentSeverity,
    IncidentType,
    SignalPhase,
)
from sentinel.contracts.events import (
    AgentHeartbeat,
    DecisionMade,
    ExplanationGenerated,
    IncidentDetected,
    SignalChanged,
    StateUpdated,
    SystemModeChanged,
)
from sentinel.contracts.value_objects import (
    AgentHealth,
    Explanation,
    Incident,
    IntersectionState,
    LaneState,
    SignalCommand,
    SignalState,
)
from sentinel.dashboard import DashboardAgent
from sentinel.messaging import InMemoryEventBus

_T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _lane(direction: Direction) -> LaneState:
    return LaneState(
        direction=direction, vehicle_count=2, moving_count=1, stopped_count=1,
        queue_length_m=7.0, avg_wait_s=3.0, occupancy_pct=10.0, density=DensityLevel.FREE,
    )


def _state() -> IntersectionState:
    return IntersectionState(
        intersection_id="i-1", timestamp=_T0, lanes={d: _lane(d) for d in Direction},
        current_phase=SignalPhase.NS_GREEN, phase_elapsed_s=1.0,
    )


class TestDashboardAgent:
    async def test_tracks_latest_state(self) -> None:
        bus = InMemoryEventBus()
        dashboard = DashboardAgent(event_bus=bus, intersection_id="i-1", heartbeat_interval_s=0.0)
        async with bus:
            await bus.publish(
                StateUpdated(source="perception", intersection_id="i-1", state=_state())
            )
            await bus.join()
        assert dashboard.snapshot().state is not None
        assert dashboard.snapshot().state.intersection_id == "i-1"

    async def test_tracks_decision_and_logs_it(self) -> None:
        bus = InMemoryEventBus()
        dashboard = DashboardAgent(event_bus=bus, intersection_id="i-1", heartbeat_interval_s=0.0)
        command = SignalCommand(
            intersection_id="i-1", action=DecisionAction.SWITCH_PHASE,
            target_phase=SignalPhase.EW_GREEN, duration_s=10.0,
            reason_code="opposing_demand_higher",
        )
        async with bus:
            await bus.publish(
                DecisionMade(source="decision", intersection_id="i-1", command=command)
            )
            await bus.join()
        assert dashboard.snapshot().decision == command
        assert any("decision" in line for line in dashboard.logs())

    async def test_tracks_signal_change(self) -> None:
        bus = InMemoryEventBus()
        dashboard = DashboardAgent(event_bus=bus, intersection_id="i-1", heartbeat_interval_s=0.0)
        signal = SignalState(
            intersection_id="i-1", phase=SignalPhase.EW_GREEN,
            phase_elapsed_s=0.0, phase_remaining_s=30.0,
        )
        async with bus:
            await bus.publish(
                SignalChanged(source="signal-controller", intersection_id="i-1", signal=signal)
            )
            await bus.join()
        assert dashboard.snapshot().signal == signal

    async def test_tracks_incident_and_explanation(self) -> None:
        bus = InMemoryEventBus()
        dashboard = DashboardAgent(event_bus=bus, intersection_id="i-1", heartbeat_interval_s=0.0)
        incident = Incident(
            intersection_id="i-1", incident_type=IncidentType.STALLED_VEHICLE,
            severity=IncidentSeverity.HIGH, confidence=0.7, description="test incident",
        )
        explanation = Explanation(
            intersection_id="i-1", decision_reason_code="x", text="because reasons",
        )
        async with bus:
            await bus.publish(
                IncidentDetected(source="incident", intersection_id="i-1", incident=incident)
            )
            await bus.publish(
                ExplanationGenerated(
                    source="explainability", intersection_id="i-1", explanation=explanation
                )
            )
            await bus.join()
        assert dashboard.snapshot().latest_incident == incident
        assert dashboard.snapshot().latest_explanation == explanation
        logs = dashboard.logs()
        assert any("INCIDENT" in line for line in logs)
        assert any("because reasons" in line for line in logs)

    async def test_tracks_agent_health_and_mode(self) -> None:
        bus = InMemoryEventBus()
        dashboard = DashboardAgent(event_bus=bus, intersection_id="i-1", heartbeat_interval_s=0.0)
        async with bus:
            await bus.publish(
                AgentHeartbeat(
                    source="decision-agent", intersection_id="i-1",
                    health=AgentHealth(agent_name="decision-agent", status=AgentStatus.HEALTHY),
                )
            )
            await bus.publish(
                SystemModeChanged(
                    source="orchestrator", intersection_id="i-1",
                    previous_mode="ai", new_mode="degraded", reason="low confidence",
                )
            )
            await bus.join()

        snap = dashboard.snapshot()
        assert snap.agent_health["decision-agent"].status is AgentStatus.HEALTHY
        assert snap.mode.value == "degraded"
        assert snap.degraded_reason == "low confidence"

    async def test_to_dict_is_json_serialisable(self) -> None:
        import json

        bus = InMemoryEventBus()
        dashboard = DashboardAgent(event_bus=bus, intersection_id="i-1", heartbeat_interval_s=0.0)
        async with bus:
            await bus.publish(
                StateUpdated(source="perception", intersection_id="i-1", state=_state())
            )
            await bus.join()
        payload = dashboard.snapshot().to_dict()
        json.dumps(payload)  # must not raise

    async def test_log_buffer_is_bounded(self) -> None:
        bus = InMemoryEventBus()
        dashboard = DashboardAgent(
            event_bus=bus, intersection_id="i-1", log_capacity=3, heartbeat_interval_s=0.0
        )
        command = SignalCommand(
            intersection_id="i-1", action=DecisionAction.KEEP_GREEN,
            target_phase=SignalPhase.NS_GREEN, duration_s=1.0, reason_code="r",
        )
        async with bus:
            for _ in range(10):
                await bus.publish(
                    DecisionMade(source="decision", intersection_id="i-1", command=command)
                )
                await bus.join()
        assert len(dashboard.logs()) == 3
