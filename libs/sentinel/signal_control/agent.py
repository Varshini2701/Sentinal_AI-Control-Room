"""The Signal Controller Agent -- tier 2 of the safe controller (the actuator authority).

Consumes ``decision.made`` and treats the command only as a *desired axis*. It owns the
:class:`~sentinel.control.phase.PhaseStateMachine`, so whatever the Decision Agent (or a future RL
policy, or a buggy caller) asks for, this agent guarantees min/max green, full yellow + all-red
clearance and legal transitions before driving the actuator. It emits ``signal.changed`` on every
real phase transition and re-validates each transition through the :class:`SafetyEnvelope`, keeping
``sentinel_safety_violations_total`` at zero.

Timing is real: :meth:`tick` advances the machine by ``dt``. In production :meth:`start` runs a
fixed-cadence loop; tests and the closed-loop driver call :meth:`tick` explicitly.
"""

from __future__ import annotations

import asyncio
import contextlib

from sentinel.agents.base import BaseAgent
from sentinel.config.settings import DecisionSettings
from sentinel.contracts.enums import Axis, SignalPhase
from sentinel.contracts.events import DecisionMade, DomainEvent, SignalChanged
from sentinel.contracts.value_objects import SignalState
from sentinel.control.phase import PhaseStateMachine, SafetyEnvelope
from sentinel.messaging.bus import EventBus
from sentinel.signal_control.actuator import SignalActuator


class SignalControllerAgent(BaseAgent):
    """Safely sequences and actuates the traffic light from decision intent."""

    def __init__(
        self,
        *,
        event_bus: EventBus,
        intersection_id: str,
        actuator: SignalActuator,
        settings: DecisionSettings | None = None,
        initial_axis: Axis = Axis.NORTH_SOUTH,
        tick_interval_s: float = 1.0,
        auto_run: bool = False,
        heartbeat_interval_s: float = 5.0,
    ) -> None:
        self._settings = settings or DecisionSettings()
        self._fsm = PhaseStateMachine(self._settings, initial_axis=initial_axis)
        self._desired_axis = initial_axis
        self._actuator = actuator
        self._tick_interval_s = tick_interval_s
        self._auto_run = auto_run
        self._loop_task: asyncio.Task[None] | None = None
        super().__init__(
            name="signal-controller",
            event_bus=event_bus,
            intersection_id=intersection_id,
            heartbeat_interval_s=heartbeat_interval_s,
        )
        self._signal = self._build_signal()
        self._actuator.apply(self._signal)  # actuate the initial phase

    def _register(self) -> None:
        self._subscribe("decision.made", self._on_decision)

    @property
    def current_signal(self) -> SignalState:
        return self._signal

    async def start(self) -> None:
        await super().start()
        # Announce the current signal immediately so a late-joining consumer (e.g. a dashboard
        # subscribing after startup) learns the live phase without waiting for the next transition.
        await self._publish(
            SignalChanged(
                source=self.name, intersection_id=self._intersection_id, signal=self._signal
            )
        )
        if self._auto_run and self._loop_task is None:
            self._loop_task = asyncio.create_task(self._run_loop(), name=f"{self.name}-loop")

    async def stop(self) -> None:
        if self._loop_task is not None:
            self._loop_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._loop_task
            self._loop_task = None
        await super().stop()

    async def _on_decision(self, event: DomainEvent) -> None:
        if not isinstance(event, DecisionMade):
            return
        axis = event.command.target_phase.axis
        if axis is not None:
            self._desired_axis = axis

    async def tick(self, dt: float) -> SignalState:
        """Advance the signal by ``dt`` seconds; actuate and announce any phase change."""
        request_switch = (
            self._fsm.active_axis is not None and self._fsm.active_axis is not self._desired_axis
        )
        previous = self._fsm.phase
        new_phase = self._fsm.step(dt, request_switch=request_switch)
        self._signal = self._build_signal()

        if new_phase != previous:
            SafetyEnvelope.validate_transition(
                previous, new_phase, intersection_id=self._intersection_id
            )
            previous_signal = SignalState(
                intersection_id=self._intersection_id,
                phase=previous,
                phase_elapsed_s=0.0,
                phase_remaining_s=0.0,
            )
            self._actuator.apply(self._signal)
            await self._publish(
                SignalChanged(
                    source=self.name,
                    intersection_id=self._intersection_id,
                    previous_phase=previous_signal,
                    signal=self._signal,
                )
            )
            self._log.debug("signal_changed", from_phase=previous, to_phase=new_phase)
        return self._signal

    async def _run_loop(self) -> None:
        while True:
            await self.tick(self._tick_interval_s)
            await asyncio.sleep(self._tick_interval_s)

    def _build_signal(self) -> SignalState:
        return SignalState(
            intersection_id=self._intersection_id,
            phase=self._fsm.phase,
            phase_elapsed_s=self._fsm.elapsed_s,
            phase_remaining_s=self._remaining_s(),
        )

    def _remaining_s(self) -> float:
        phase = self._fsm.phase
        elapsed = self._fsm.elapsed_s
        if phase.is_green:
            budget = self._settings.max_green_s
        elif phase in (SignalPhase.NS_YELLOW, SignalPhase.EW_YELLOW):
            budget = self._settings.yellow_s
        else:
            budget = self._settings.all_red_s
        return max(0.0, budget - elapsed)


__all__ = ["SignalControllerAgent"]
