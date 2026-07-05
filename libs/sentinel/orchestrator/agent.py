"""The Orchestrator Agent: aggregates fleet health and owns the operating-mode state machine.

Consumes every agent's ``agent.heartbeat`` to build a live health snapshot, and consumes
``state.updated`` to watch perception confidence -- dropping the intersection to
:attr:`SystemMode.DEGRADED` when confidence falls below
:attr:`~sentinel.config.settings.DecisionSettings.min_perception_confidence` (and recovering to
:attr:`SystemMode.AI` automatically), emitting ``system.mode.changed`` on every transition. A
lightweight watchdog loop marks an agent unhealthy in the snapshot if it stops heartbeating,
without waiting for it to crash something else first.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

from sentinel.agents.base import BaseAgent
from sentinel.config.settings import DecisionSettings, OrchestratorSettings
from sentinel.contracts.enums import AgentStatus, SystemMode
from sentinel.contracts.events import AgentHeartbeat, DomainEvent, StateUpdated, SystemModeChanged
from sentinel.contracts.value_objects import AgentHealth
from sentinel.messaging.bus import EventBus

Clock = Callable[[], datetime]


def _utcnow() -> datetime:
    return datetime.now(tz=UTC)


@dataclass(frozen=True, slots=True)
class OrchestratorSnapshot:
    """A point-in-time view of system mode and fleet health, for the Dashboard/API."""

    mode: SystemMode
    degraded_reason: str | None
    agents: dict[str, AgentHealth] = field(default_factory=dict)


class OrchestratorAgent(BaseAgent):
    """Coordinates agent health and the intersection's operating mode."""

    def __init__(
        self,
        *,
        event_bus: EventBus,
        intersection_id: str,
        settings: OrchestratorSettings | None = None,
        decision_settings: DecisionSettings | None = None,
        heartbeat_interval_s: float = 5.0,
        clock: Clock = _utcnow,
    ) -> None:
        self._settings = settings or OrchestratorSettings()
        self._decision_settings = decision_settings or DecisionSettings()
        self._clock = clock
        self._mode = SystemMode.AI
        self._degraded_reason: str | None = None
        self._health: dict[str, AgentHealth] = {}
        self._watchdog_task: asyncio.Task[None] | None = None
        super().__init__(
            name="orchestrator",
            event_bus=event_bus,
            intersection_id=intersection_id,
            heartbeat_interval_s=heartbeat_interval_s,
        )

    def _register(self) -> None:
        self._subscribe("agent.heartbeat", self._on_heartbeat)
        self._subscribe("state.updated", self._on_state)

    async def start(self) -> None:
        await super().start()
        if self._settings.check_interval_s > 0 and self._watchdog_task is None:
            self._watchdog_task = asyncio.create_task(
                self._watchdog_loop(), name=f"{self.name}-watchdog"
            )

    async def stop(self) -> None:
        if self._watchdog_task is not None:
            self._watchdog_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._watchdog_task
            self._watchdog_task = None
        await super().stop()

    def snapshot(self) -> OrchestratorSnapshot:
        return OrchestratorSnapshot(
            mode=self._mode, degraded_reason=self._degraded_reason, agents=dict(self._health)
        )

    async def _on_heartbeat(self, event: DomainEvent) -> None:
        if not isinstance(event, AgentHeartbeat):
            return
        self._health[event.health.agent_name] = event.health

    async def _on_state(self, event: DomainEvent) -> None:
        if not isinstance(event, StateUpdated):
            return
        confidence = event.state.perception_confidence
        threshold = self._decision_settings.min_perception_confidence

        if confidence < threshold and self._mode is not SystemMode.DEGRADED:
            await self._set_mode(
                SystemMode.DEGRADED,
                f"perception_confidence {confidence:.2f} below threshold {threshold:.2f}",
            )
        elif (
            confidence >= threshold
            and self._mode is SystemMode.DEGRADED
            and self._degraded_reason is not None
            and self._degraded_reason.startswith("perception_confidence")
        ):
            await self._set_mode(SystemMode.AI, "perception_confidence recovered")

    async def _set_mode(self, new_mode: SystemMode, reason: str) -> None:
        if new_mode is self._mode:
            return
        previous = self._mode
        self._mode = new_mode
        self._degraded_reason = reason if new_mode is SystemMode.DEGRADED else None
        await self._publish(
            SystemModeChanged(
                source=self.name,
                intersection_id=self._intersection_id,
                previous_mode=previous.value,
                new_mode=new_mode.value,
                reason=reason,
            )
        )
        self._log.info("system_mode_changed", previous=previous, new=new_mode, reason=reason)

    async def _watchdog_loop(self) -> None:
        while True:
            await asyncio.sleep(self._settings.check_interval_s)
            self._mark_stale_agents()

    def _mark_stale_agents(self) -> None:
        now = self._clock()
        for agent_name, health in list(self._health.items()):
            if health.status is AgentStatus.STOPPED:
                continue
            age = (now - health.last_heartbeat).total_seconds()
            if age > self._settings.stale_after_s and health.status is not AgentStatus.UNHEALTHY:
                self._health[agent_name] = health.model_copy(
                    update={"status": AgentStatus.UNHEALTHY}
                )
                self._log.warning("agent_marked_stale", agent=agent_name, age_s=round(age, 1))


__all__ = ["OrchestratorAgent", "OrchestratorSnapshot"]
