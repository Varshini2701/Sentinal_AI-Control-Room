"""Tests for the Orchestrator Agent: health aggregation and mode switching."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sentinel.config.settings import DecisionSettings, OrchestratorSettings
from sentinel.contracts.enums import AgentStatus, DensityLevel, Direction, SignalPhase, SystemMode
from sentinel.contracts.events import AgentHeartbeat, StateUpdated, SystemModeChanged
from sentinel.contracts.value_objects import AgentHealth, IntersectionState, LaneState
from sentinel.messaging import InMemoryEventBus
from sentinel.orchestrator import OrchestratorAgent

_T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _lane(direction: Direction) -> LaneState:
    return LaneState(
        direction=direction, vehicle_count=0, moving_count=0, stopped_count=0,
        queue_length_m=0.0, avg_wait_s=0.0, occupancy_pct=0.0, density=DensityLevel.FREE,
    )


def _state(confidence: float, ts: datetime = _T0) -> IntersectionState:
    return IntersectionState(
        intersection_id="i-1",
        timestamp=ts,
        lanes={d: _lane(d) for d in Direction},
        current_phase=SignalPhase.NS_GREEN,
        phase_elapsed_s=1.0,
        perception_confidence=confidence,
    )


class TestOrchestratorHealth:
    async def test_tracks_agent_health_from_heartbeats(self) -> None:
        bus = InMemoryEventBus()
        orchestrator = OrchestratorAgent(
            event_bus=bus, intersection_id="i-1",
            settings=OrchestratorSettings(check_interval_s=0.0), heartbeat_interval_s=0.0,
        )
        async with bus:
            await bus.publish(
                AgentHeartbeat(
                    source="decision-agent", intersection_id="i-1",
                    health=AgentHealth(agent_name="decision-agent", status=AgentStatus.HEALTHY),
                )
            )
            await bus.join()

        snap = orchestrator.snapshot()
        assert snap.agents["decision-agent"].status is AgentStatus.HEALTHY

    async def test_watchdog_marks_stale_agent_unhealthy(self) -> None:
        clock_time = [_T0]

        def fake_clock() -> datetime:
            return clock_time[0]

        bus = InMemoryEventBus()
        orchestrator = OrchestratorAgent(
            event_bus=bus, intersection_id="i-1",
            settings=OrchestratorSettings(stale_after_s=10.0, check_interval_s=0.01),
            heartbeat_interval_s=0.0, clock=fake_clock,
        )
        import asyncio

        async with bus, orchestrator:
            await bus.publish(
                AgentHeartbeat(
                    source="perception", intersection_id="i-1",
                    health=AgentHealth(
                        agent_name="perception", status=AgentStatus.HEALTHY, last_heartbeat=_T0
                    ),
                )
            )
            await bus.join()
            clock_time[0] = _T0 + timedelta(seconds=20)  # advance past stale_after_s
            await asyncio.sleep(0.05)  # let the watchdog loop tick at least once

        assert orchestrator.snapshot().agents["perception"].status is AgentStatus.UNHEALTHY


class TestOrchestratorModeSwitching:
    async def test_low_confidence_triggers_degraded(self) -> None:
        mode_changes: list[SystemModeChanged] = []

        async def capture(event: SystemModeChanged) -> None:  # type: ignore[override]
            mode_changes.append(event)

        bus = InMemoryEventBus()
        bus.subscribe("system.mode.changed", capture, consumer_name="cap")
        decision_settings = DecisionSettings(min_perception_confidence=0.5)
        orchestrator = OrchestratorAgent(
            event_bus=bus, intersection_id="i-1", decision_settings=decision_settings,
            settings=OrchestratorSettings(check_interval_s=0.0), heartbeat_interval_s=0.0,
        )

        async with bus:
            await bus.publish(
                StateUpdated(source="perception", intersection_id="i-1", state=_state(0.3))
            )
            await bus.join()

        assert orchestrator.snapshot().mode is SystemMode.DEGRADED
        assert len(mode_changes) == 1
        assert mode_changes[0].new_mode == "degraded"

    async def test_recovers_to_ai_when_confidence_restored(self) -> None:
        mode_changes: list[SystemModeChanged] = []

        async def capture(event: SystemModeChanged) -> None:  # type: ignore[override]
            mode_changes.append(event)

        bus = InMemoryEventBus()
        bus.subscribe("system.mode.changed", capture, consumer_name="cap")
        decision_settings = DecisionSettings(min_perception_confidence=0.5)
        orchestrator = OrchestratorAgent(
            event_bus=bus, intersection_id="i-1", decision_settings=decision_settings,
            settings=OrchestratorSettings(check_interval_s=0.0), heartbeat_interval_s=0.0,
        )

        async with bus:
            await bus.publish(
                StateUpdated(source="perception", intersection_id="i-1", state=_state(0.3))
            )
            await bus.join()
            await bus.publish(
                StateUpdated(source="perception", intersection_id="i-1", state=_state(0.9))
            )
            await bus.join()

        assert orchestrator.snapshot().mode is SystemMode.AI
        assert [e.new_mode for e in mode_changes] == ["degraded", "ai"]

    async def test_no_duplicate_mode_events_while_still_degraded(self) -> None:
        mode_changes: list[SystemModeChanged] = []

        async def capture(event: SystemModeChanged) -> None:  # type: ignore[override]
            mode_changes.append(event)

        bus = InMemoryEventBus()
        bus.subscribe("system.mode.changed", capture, consumer_name="cap")
        decision_settings = DecisionSettings(min_perception_confidence=0.5)
        OrchestratorAgent(
            event_bus=bus, intersection_id="i-1", decision_settings=decision_settings,
            settings=OrchestratorSettings(check_interval_s=0.0), heartbeat_interval_s=0.0,
        )

        async with bus:
            for _ in range(3):
                await bus.publish(
                    StateUpdated(source="perception", intersection_id="i-1", state=_state(0.2))
                )
                await bus.join()

        assert len(mode_changes) == 1  # only the initial transition, not every low-confidence tick
