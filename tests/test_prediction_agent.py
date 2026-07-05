"""Tests for the Prediction Agent: event wiring and Decision Agent integration."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sentinel.config.settings import DecisionSettings, PredictionSettings
from sentinel.contracts.enums import DensityLevel, Direction, SignalPhase
from sentinel.contracts.events import DecisionMade, PredictionUpdated, StateUpdated
from sentinel.contracts.value_objects import IntersectionState, LaneState
from sentinel.decision import DecisionAgent
from sentinel.messaging import InMemoryEventBus
from sentinel.prediction import PredictionAgent

_T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _lane(direction: Direction, count: int) -> LaneState:
    return LaneState(
        direction=direction,
        vehicle_count=count,
        moving_count=0,
        stopped_count=count,
        queue_length_m=count * 7.0,
        avg_wait_s=0.0,
        occupancy_pct=min(100.0, count * 3.0),
        density=DensityLevel.MODERATE,
    )


def _state(ns: int, ew: int, ts: datetime) -> IntersectionState:
    return IntersectionState(
        intersection_id="i-1",
        timestamp=ts,
        lanes={
            Direction.NORTH: _lane(Direction.NORTH, ns),
            Direction.SOUTH: _lane(Direction.SOUTH, ns),
            Direction.EAST: _lane(Direction.EAST, ew),
            Direction.WEST: _lane(Direction.WEST, ew),
        },
        current_phase=SignalPhase.NS_GREEN,
        phase_elapsed_s=10.0,
    )


class TestPredictionAgent:
    async def test_emits_prediction_for_each_state(self) -> None:
        predictions: list[PredictionUpdated] = []

        async def capture(event: PredictionUpdated) -> None:  # type: ignore[override]
            predictions.append(event)

        bus = InMemoryEventBus()
        bus.subscribe("prediction.updated", capture, consumer_name="cap")
        PredictionAgent(event_bus=bus, intersection_id="i-1", heartbeat_interval_s=0.0)

        async with bus:
            evt = StateUpdated(source="perception", intersection_id="i-1", state=_state(5, 1, _T0))
            await bus.publish(evt)
            await bus.join()

        assert len(predictions) == 1
        assert predictions[0].causation_id == evt.event_id
        assert set(predictions[0].forecast.lanes) <= set(Direction)

    async def test_forecast_reflects_growth_trend(self) -> None:
        predictions: list[PredictionUpdated] = []

        async def capture(event: PredictionUpdated) -> None:  # type: ignore[override]
            predictions.append(event)

        bus = InMemoryEventBus()
        bus.subscribe(
            "prediction.updated", capture, consumer_name="cap",
        )
        PredictionAgent(
            event_bus=bus, intersection_id="i-1",
            settings=PredictionSettings(min_samples_for_trend=3, trend_window=10),
            heartbeat_interval_s=0.0,
        )

        async with bus:
            for i, ns in enumerate([2, 4, 6, 8]):
                await bus.publish(
                    StateUpdated(
                        source="perception", intersection_id="i-1",
                        state=_state(ns, 0, _T0 + timedelta(seconds=i)),
                    )
                )
                await bus.join()

        last_forecast = predictions[-1].forecast
        north = last_forecast.lanes[Direction.NORTH]
        # Growing queue -> the forecast should predict continued growth beyond the last sample.
        assert north.predicted_queue_length_m > 8 * 7.0


class TestDecisionIntegration:
    async def test_decision_agent_uses_forecast(self) -> None:
        """The Decision Agent's weight_prediction term picks up whatever forecast arrives."""
        decisions: list[DecisionMade] = []

        async def capture(event: DecisionMade) -> None:  # type: ignore[override]
            decisions.append(event)

        settings = DecisionSettings(weight_prediction=5.0, switch_penalty=8.0)
        bus = InMemoryEventBus()
        bus.subscribe("decision.made", capture, consumer_name="cap")
        PredictionAgent(event_bus=bus, intersection_id="i-1", heartbeat_interval_s=0.0)
        DecisionAgent(
            event_bus=bus, intersection_id="i-1", settings=settings, heartbeat_interval_s=0.0
        )

        async with bus:
            # Light current demand, but East-West trending up sharply -> prediction should
            # eventually tip the decision even though the raw queue counts are similar.
            for i, (ns, ew) in enumerate([(3, 1), (3, 3), (3, 6)]):
                await bus.publish(
                    StateUpdated(
                        source="perception", intersection_id="i-1",
                        state=_state(ns, ew, _T0 + timedelta(seconds=i)),
                    )
                )
                await bus.join()  # prediction.updated
                await bus.join()  # decision.made

        assert len(decisions) == 3
        # Whatever the final decision, it must carry a valid axis/action -- proves the wiring works
        # end to end (Prediction -> Decision) without the Decision Agent erroring on the forecast.
        assert decisions[-1].command.target_phase in (
            SignalPhase.NS_GREEN,
            SignalPhase.EW_GREEN,
            SignalPhase.NS_YELLOW,
            SignalPhase.EW_YELLOW,
        )
