"""The Prediction Agent: turns recent state history into a short-horizon forecast.

Consumes ``state.updated``, maintains a small per-lane rolling window of ``(elapsed_s, state)``
samples (independent of the Traffic Memory Agent's larger retention window -- this one only needs
enough points to fit a trend), runs the configured :class:`TrafficPredictor`, and emits
``prediction.updated``. The Decision Agent consumes this forecast directly via its
``weight_prediction`` term.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime

from sentinel.agents.base import BaseAgent
from sentinel.config.settings import PredictionSettings
from sentinel.contracts.enums import Direction
from sentinel.contracts.events import DomainEvent, PredictionUpdated, StateUpdated
from sentinel.contracts.value_objects import IntersectionState
from sentinel.messaging.bus import EventBus
from sentinel.prediction.forecaster import LinearTrendForecaster, TrafficPredictor


class PredictionAgent(BaseAgent):
    """Event-driven agent that forecasts near-future per-lane congestion."""

    def __init__(
        self,
        *,
        event_bus: EventBus,
        intersection_id: str,
        settings: PredictionSettings | None = None,
        predictor: TrafficPredictor | None = None,
        heartbeat_interval_s: float = 5.0,
    ) -> None:
        self._settings = settings or PredictionSettings()
        self._predictor = predictor or LinearTrendForecaster(
            min_samples=self._settings.min_samples_for_trend
        )
        self._history: dict[Direction, deque[tuple[float, IntersectionState]]] = {
            d: deque(maxlen=self._settings.trend_window) for d in Direction
        }
        self._elapsed_s = 0.0
        self._prev_ts: datetime | None = None
        super().__init__(
            name="prediction-agent",
            event_bus=event_bus,
            intersection_id=intersection_id,
            heartbeat_interval_s=heartbeat_interval_s,
        )

    def _register(self) -> None:
        self._subscribe("state.updated", self._on_state)

    async def _on_state(self, event: DomainEvent) -> None:
        if not isinstance(event, StateUpdated):
            return
        state = event.state
        self._advance_clock(state)
        for direction in state.lanes:
            self._history[direction].append((self._elapsed_s, state))

        forecast = self._predictor.predict(
            self._intersection_id,
            {d: list(window) for d, window in self._history.items()},
            self._settings.horizon_s,
        )
        await self._publish(
            PredictionUpdated(
                source=self.name,
                intersection_id=self._intersection_id,
                forecast=forecast,
                correlation_id=event.correlation_id or event.event_id,
                causation_id=event.event_id,
            )
        )

    def _advance_clock(self, state: IntersectionState) -> None:
        dt = (state.timestamp - self._prev_ts).total_seconds() if self._prev_ts is not None else 0.0
        self._prev_ts = state.timestamp
        self._elapsed_s += max(0.0, dt)


__all__ = ["PredictionAgent"]
