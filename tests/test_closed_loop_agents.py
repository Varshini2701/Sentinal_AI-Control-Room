"""Integration test: the async agent architecture closes the control loop safely.

Wires the analytical traffic twin -> event bus -> Decision Agent -> Signal Controller -> back into
the twin, and verifies the whole thing behaves sensibly (serves the busy axis, controls the loop)
without ever tripping the safety envelope.
"""

from __future__ import annotations

from sentinel.config.settings import DecisionSettings
from sentinel.contracts.enums import DecisionAction, Direction, SignalPhase
from sentinel.contracts.events import DecisionMade, StateUpdated
from sentinel.contracts.value_objects import SignalCommand
from sentinel.decision import DecisionAgent
from sentinel.messaging import InMemoryEventBus
from sentinel.observability.metrics import SENTINEL_REGISTRY
from sentinel.signal_control import RecordingActuator, SignalControllerAgent
from sentinel.simulation import AnalyticalTrafficEnvironment, SimConfig, asymmetric_demand


def _apply(phase: SignalPhase, intersection_id: str) -> SignalCommand:
    return SignalCommand(
        intersection_id=intersection_id,
        action=DecisionAction.KEEP_GREEN,
        target_phase=phase,
        duration_s=1.0,
        reason_code="apply",
    )


class TestClosedLoopAgents:
    async def test_agent_loop_serves_busy_axis_safely(self) -> None:
        settings = DecisionSettings()
        config = SimConfig(horizon_s=600, seed=42, demand=asymmetric_demand(0.22, 0.05))

        decisions: list[DecisionMade] = []

        async def count_decisions(event: DecisionMade) -> None:  # type: ignore[override]
            decisions.append(event)

        bus = InMemoryEventBus()
        bus.subscribe("decision.made", count_decisions, consumer_name="probe")
        env = AnalyticalTrafficEnvironment(config)
        DecisionAgent(event_bus=bus, intersection_id=config.intersection_id, settings=settings,
                      heartbeat_interval_s=0.0)
        controller = SignalControllerAgent(
            event_bus=bus,
            intersection_id=config.intersection_id,
            actuator=RecordingActuator(),
            settings=settings,
            heartbeat_interval_s=0.0,
        )

        labels = {"intersection": config.intersection_id, "constraint": "illegal_phase_transition"}
        violations_before = SENTINEL_REGISTRY.get_sample_value(
            "sentinel_safety_violations_total", labels
        ) or 0.0

        async with bus:
            state = env.reset()
            for _ in range(config.total_steps):
                await bus.publish(
                    StateUpdated(
                        source="perception", intersection_id=config.intersection_id, state=state
                    )
                )
                await bus.join()  # decision.made emitted...
                await bus.join()  # ...and consumed by the signal controller
                signal = await controller.tick(config.dt_s)
                state = env.step(_apply(signal.phase, config.intersection_id))

        metrics = env.metrics()
        violations_after = SENTINEL_REGISTRY.get_sample_value(
            "sentinel_safety_violations_total", labels
        ) or 0.0

        ns_served = metrics.lanes[Direction.NORTH].served + metrics.lanes[Direction.SOUTH].served
        ew_served = metrics.lanes[Direction.EAST].served + metrics.lanes[Direction.WEST].served

        assert len(decisions) == config.total_steps  # a decision per state tick
        assert metrics.total_served > 0
        assert ns_served > ew_served  # busy axis prioritised through the full agent chain
        assert violations_after == violations_before  # safety envelope never tripped
