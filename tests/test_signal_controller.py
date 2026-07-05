"""Tests for the Signal Controller Agent: safe sequencing and actuation."""

from __future__ import annotations

from sentinel.config.settings import DecisionSettings
from sentinel.contracts.enums import DecisionAction, SignalPhase
from sentinel.contracts.events import DecisionMade, SignalChanged
from sentinel.contracts.value_objects import SignalCommand
from sentinel.messaging import InMemoryEventBus
from sentinel.observability.metrics import SENTINEL_REGISTRY
from sentinel.signal_control import RecordingActuator, SignalControllerAgent

_SETTINGS = DecisionSettings(min_green_s=5, max_green_s=40, yellow_s=3, all_red_s=2)


def _switch_to(phase: SignalPhase) -> SignalCommand:
    return SignalCommand(
        intersection_id="i-1",
        action=DecisionAction.SWITCH_PHASE,
        target_phase=phase,
        duration_s=5.0,
        reason_code="test",
    )


class TestSignalController:
    async def test_actuates_initial_phase(self) -> None:
        bus = InMemoryEventBus()
        actuator = RecordingActuator()
        SignalControllerAgent(
            event_bus=bus, intersection_id="i-1", actuator=actuator, settings=_SETTINGS
        )
        assert actuator.current is not None
        assert actuator.current.phase is SignalPhase.NS_GREEN

    async def test_switches_through_full_clearance(self) -> None:
        changes: list[SignalChanged] = []

        async def capture(event: SignalChanged) -> None:  # type: ignore[override]
            changes.append(event)

        bus = InMemoryEventBus()
        bus.subscribe("signal.changed", capture, consumer_name="cap")
        agent = SignalControllerAgent(
            event_bus=bus, intersection_id="i-1", actuator=RecordingActuator(), settings=_SETTINGS
        )

        seen: list[SignalPhase] = []
        async with bus:
            await bus.publish(_decision(SignalPhase.EW_GREEN))
            await bus.join()
            for _ in range(20):
                sig = await agent.tick(1.0)
                if not seen or seen[-1] != sig.phase:
                    seen.append(sig.phase)
                if sig.phase is SignalPhase.EW_GREEN:
                    break
            await bus.join()

        assert seen == [
            SignalPhase.NS_GREEN,
            SignalPhase.NS_YELLOW,
            SignalPhase.ALL_RED,
            SignalPhase.EW_GREEN,
        ]
        assert len(changes) == 3  # three transitions announced

    async def test_min_green_respected_before_switch(self) -> None:
        bus = InMemoryEventBus()
        agent = SignalControllerAgent(
            event_bus=bus, intersection_id="i-1", actuator=RecordingActuator(), settings=_SETTINGS
        )
        async with bus:
            await bus.publish(_decision(SignalPhase.EW_GREEN))
            await bus.join()
            left_green_at = None
            for i in range(1, 11):
                sig = await agent.tick(1.0)
                if sig.phase is not SignalPhase.NS_GREEN:
                    left_green_at = i
                    break
        # Despite an immediate switch request, green is held for exactly min_green seconds.
        assert left_green_at == _SETTINGS.min_green_s

    async def test_no_safety_violations(self) -> None:
        labels = {"intersection": "i-1", "constraint": "illegal_phase_transition"}
        before = SENTINEL_REGISTRY.get_sample_value(
            "sentinel_safety_violations_total", labels
        ) or 0.0

        bus = InMemoryEventBus()
        agent = SignalControllerAgent(
            event_bus=bus, intersection_id="i-1", actuator=RecordingActuator(), settings=_SETTINGS
        )
        async with bus:
            # Flip the desired axis aggressively; the FSM must still never emit an illegal jump.
            for i in range(60):
                target = SignalPhase.EW_GREEN if i % 2 == 0 else SignalPhase.NS_GREEN
                await bus.publish(_decision(target))
                await bus.join()
                await agent.tick(1.0)

        after = (
            SENTINEL_REGISTRY.get_sample_value("sentinel_safety_violations_total", labels) or 0.0
        )
        assert after == before


def _decision(phase: SignalPhase) -> DecisionMade:
    return DecisionMade(source="decision", intersection_id="i-1", command=_switch_to(phase))
