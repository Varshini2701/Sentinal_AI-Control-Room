"""Base class for event-driven Cognition-plane agents.

Every cognition agent (Decision, Signal Controller, Prediction, Incident, Explainability, ...)
shares the same skeleton: it registers its bus subscriptions, reports health, and periodically
emits an :class:`AgentHeartbeat` so the Orchestrator and Prometheus can see it is alive. This base
class captures that skeleton so each agent only writes its own handlers and logic.

Subscriptions are registered in ``__init__`` (via :meth:`_register`) so they exist before the bus
starts consuming; :meth:`start`/:meth:`stop` manage the optional heartbeat task and any agent loop.
"""

from __future__ import annotations

import abc
import asyncio
import contextlib
from collections.abc import Iterable

from sentinel.contracts.enums import AgentStatus
from sentinel.contracts.events import AgentHeartbeat, DomainEvent
from sentinel.contracts.value_objects import AgentHealth
from sentinel.messaging.bus import EventBus, EventHandler
from sentinel.observability.logging import get_logger
from sentinel.observability.metrics import AGENT_UP


class BaseAgent(abc.ABC):
    """Common lifecycle, subscription and health machinery for cognition agents."""

    def __init__(
        self,
        *,
        name: str,
        event_bus: EventBus,
        intersection_id: str,
        heartbeat_interval_s: float = 5.0,
    ) -> None:
        self.name = name
        self._bus = event_bus
        self._intersection_id = intersection_id
        self._heartbeat_interval_s = heartbeat_interval_s
        self._status = AgentStatus.STOPPED
        self._details: dict[str, str] = {}
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._log = get_logger(name, agent=name, intersection=intersection_id)
        self._register()

    # -- subclass hooks ----------------------------------------------------
    @abc.abstractmethod
    def _register(self) -> None:
        """Register bus subscriptions. Called once from ``__init__``."""

    # -- lifecycle ---------------------------------------------------------
    async def start(self) -> None:
        self.set_status(AgentStatus.HEALTHY)
        if self._heartbeat_interval_s > 0 and self._heartbeat_task is None:
            self._heartbeat_task = asyncio.create_task(
                self._heartbeat_loop(), name=f"{self.name}-heartbeat"
            )
        self._log.info("agent_started")

    async def stop(self) -> None:
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._heartbeat_task
            self._heartbeat_task = None
        self.set_status(AgentStatus.STOPPED)
        self._log.info("agent_stopped")

    async def __aenter__(self) -> BaseAgent:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.stop()

    # -- health ------------------------------------------------------------
    def set_status(self, status: AgentStatus, **details: str) -> None:
        self._status = status
        self._details.update(details)
        AGENT_UP.labels(agent=self.name).set(1.0 if status is AgentStatus.HEALTHY else 0.0)

    def health(self) -> AgentHealth:
        return AgentHealth(agent_name=self.name, status=self._status, details=dict(self._details))

    async def emit_heartbeat(self) -> None:
        await self._bus.publish(
            AgentHeartbeat(
                source=self.name, intersection_id=self._intersection_id, health=self.health()
            )
        )

    async def _heartbeat_loop(self) -> None:
        await self.emit_heartbeat()  # announce presence immediately, don't wait a full interval
        while True:
            await asyncio.sleep(self._heartbeat_interval_s)
            await self.emit_heartbeat()

    # -- helpers -----------------------------------------------------------
    def _subscribe(self, event_types: Iterable[str] | str, handler: EventHandler) -> None:
        self._bus.subscribe(event_types, handler, consumer_name=self.name)

    async def _publish(self, event: DomainEvent) -> None:
        await self._bus.publish(event)


__all__ = ["BaseAgent"]
