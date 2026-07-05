"""Wires the full cognition-plane agent fleet together for a single intersection.

This is the composition root every runnable entrypoint shares: the API gateway, the terminal
demos, and integration tests all build the same fleet from :func:`build_fleet` so "the app" is
defined in exactly one place. It intentionally drives the analytical traffic twin rather than a
camera feed -- swapping in :class:`~sentinel.perception.PerceptionWorker` publishing to the same
bus is the only change needed to go from simulated to live video, since every agent below only
ever consumes ``state.updated`` off the bus.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from sentinel.agents.base import BaseAgent
from sentinel.config.settings import Settings
from sentinel.contracts.enums import DecisionAction, SignalPhase
from sentinel.contracts.events import StateUpdated
from sentinel.contracts.value_objects import SignalCommand
from sentinel.dashboard import DashboardAgent
from sentinel.decision import DecisionAgent
from sentinel.explainability import ExplainabilityAgent
from sentinel.incident import IncidentDetectionAgent
from sentinel.memory import TrafficMemoryAgent
from sentinel.messaging import InMemoryEventBus
from sentinel.orchestrator import OrchestratorAgent
from sentinel.prediction import PredictionAgent
from sentinel.signal_control import RecordingActuator, SignalControllerAgent
from sentinel.simulation import AnalyticalTrafficEnvironment, SimConfig, asymmetric_demand


@dataclass(slots=True)
class Fleet:
    """Every cognition-plane agent for one intersection, plus the bus that connects them."""

    bus: InMemoryEventBus
    intersection_id: str
    config: SimConfig
    env: AnalyticalTrafficEnvironment
    memory: TrafficMemoryAgent
    prediction: PredictionAgent
    decision: DecisionAgent
    signal_controller: SignalControllerAgent
    incident: IncidentDetectionAgent
    explainability: ExplainabilityAgent
    orchestrator: OrchestratorAgent
    dashboard: DashboardAgent

    @property
    def agents(self) -> tuple[BaseAgent, ...]:
        return (
            self.memory,
            self.prediction,
            self.decision,
            self.signal_controller,
            self.incident,
            self.explainability,
            self.orchestrator,
            self.dashboard,
        )

    async def start(self) -> None:
        await self.bus.start()
        for agent in self.agents:
            await agent.start()

    async def stop(self) -> None:
        for agent in self.agents:
            await agent.stop()
        await self.bus.stop()


def build_fleet(
    intersection_id: str = "intersection-1",
    *,
    settings: Settings | None = None,
    sim_config: SimConfig | None = None,
) -> Fleet:
    """Construct (but do not start) a full agent fleet for ``intersection_id``."""
    settings = settings or Settings()
    config = sim_config or SimConfig(
        intersection_id=intersection_id,
        horizon_s=86_400.0,  # effectively unbounded for a long-running server
        demand=asymmetric_demand(0.22, 0.05),
    )
    bus = InMemoryEventBus()
    env = AnalyticalTrafficEnvironment(config)

    return Fleet(
        bus=bus,
        intersection_id=intersection_id,
        config=config,
        env=env,
        memory=TrafficMemoryAgent(
            event_bus=bus, intersection_id=intersection_id, settings=settings.memory
        ),
        prediction=PredictionAgent(
            event_bus=bus, intersection_id=intersection_id, settings=settings.prediction
        ),
        decision=DecisionAgent(
            event_bus=bus, intersection_id=intersection_id, settings=settings.decision
        ),
        signal_controller=SignalControllerAgent(
            event_bus=bus,
            intersection_id=intersection_id,
            actuator=RecordingActuator(),
            settings=settings.decision,
        ),
        incident=IncidentDetectionAgent(
            event_bus=bus, intersection_id=intersection_id, settings=settings.incident
        ),
        explainability=ExplainabilityAgent(
            event_bus=bus, intersection_id=intersection_id, settings=settings.explainability
        ),
        orchestrator=OrchestratorAgent(
            event_bus=bus,
            intersection_id=intersection_id,
            settings=settings.orchestrator,
            decision_settings=settings.decision,
        ),
        dashboard=DashboardAgent(event_bus=bus, intersection_id=intersection_id),
    )


def _apply(intersection_id: str, phase: SignalPhase) -> SignalCommand:
    return SignalCommand(
        intersection_id=intersection_id,
        action=DecisionAction.KEEP_GREEN,
        target_phase=phase,
        duration_s=1.0,
        reason_code="apply",
    )


async def run_simulated_loop(
    fleet: Fleet, config: SimConfig, *, tick_sleep_s: float = 1.0
) -> None:
    """Drive the analytical twin through the fleet forever (until cancelled).

    Publishes ``state.updated``, drains the bus so every agent reacts, ticks the Signal
    Controller's phase machine, and steps the twin -- looping (resetting the twin) once the
    configured horizon elapses, so a long-running server never runs out of simulated road.
    ``tick_sleep_s`` paces the loop to feel "live" for a dashboard; set to ``0`` for tests.
    """
    state = fleet.env.reset()
    step = 0
    while True:
        await fleet.bus.publish(
            StateUpdated(source="perception", intersection_id=fleet.intersection_id, state=state)
        )
        # Drain state.updated -> prediction.updated -> decision.made -> signal.changed in order.
        for _ in range(3):
            await fleet.bus.join()
        signal = await fleet.signal_controller.tick(config.dt_s)
        state = fleet.env.step(_apply(fleet.intersection_id, signal.phase))

        step += 1
        if step >= config.total_steps:
            state = fleet.env.reset()
            step = 0
        if tick_sleep_s > 0:
            await asyncio.sleep(tick_sleep_s)


__all__ = ["Fleet", "build_fleet", "run_simulated_loop"]
