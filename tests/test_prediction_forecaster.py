"""Tests for the prediction forecasters: persistence, linear trend, counterfactual."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sentinel.contracts.enums import DensityLevel, Direction, SignalPhase
from sentinel.contracts.value_objects import IntersectionState, LaneState
from sentinel.prediction.forecaster import (
    LinearTrendForecaster,
    PersistenceForecaster,
    counterfactual_wait,
)

_T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _lane(direction: Direction, queue_m: float, wait: float = 5.0) -> LaneState:
    return LaneState(
        direction=direction,
        vehicle_count=int(queue_m // 7) or 0,
        moving_count=0,
        stopped_count=int(queue_m // 7) or 0,
        queue_length_m=queue_m,
        avg_wait_s=wait,
        occupancy_pct=min(100.0, queue_m),
        density=DensityLevel.MODERATE,
    )


def _state(queue_m: float, ts: datetime = _T0) -> IntersectionState:
    return IntersectionState(
        intersection_id="i-1",
        timestamp=ts,
        lanes={d: _lane(d, queue_m) for d in Direction},
        current_phase=SignalPhase.NS_GREEN,
        phase_elapsed_s=1.0,
    )


def _history(queues: list[float]) -> dict[Direction, list[tuple[float, IntersectionState]]]:
    points = [(float(i), _state(q, _T0 + timedelta(seconds=i))) for i, q in enumerate(queues)]
    return {d: points for d in Direction}


class TestPersistenceForecaster:
    def test_predicts_latest_value_unchanged(self) -> None:
        forecast = PersistenceForecaster().predict("i-1", _history([10.0, 20.0, 30.0]), 60.0)
        lane = forecast.lanes[Direction.NORTH]
        assert lane.predicted_queue_length_m == 30.0
        assert lane.lower_bound_m == lane.upper_bound_m == 30.0

    def test_empty_history_yields_no_lanes(self) -> None:
        forecast = PersistenceForecaster().predict("i-1", {d: [] for d in Direction}, 60.0)
        assert forecast.lanes == {}


class TestLinearTrendForecaster:
    def test_growing_queue_projects_upward(self) -> None:
        # Perfectly linear growth: 0, 7, 14, 21, 28 (one car every second).
        queues = [0.0, 7.0, 14.0, 21.0, 28.0]
        forecast = LinearTrendForecaster(min_samples=4).predict(
            "i-1", _history(queues), horizon_s=10.0
        )
        lane = forecast.lanes[Direction.NORTH]
        # Slope is 7/s; from t=4 (last sample) + 10s horizon => predicted ~= 28 + 70 = 98.
        assert lane.predicted_queue_length_m > 60.0
        assert lane.confidence > 0.8  # perfectly linear -> near-zero residual -> high confidence

    def test_falls_back_below_min_samples(self) -> None:
        forecast = LinearTrendForecaster(min_samples=5).predict(
            "i-1", _history([0.0, 7.0, 14.0]), horizon_s=10.0
        )
        lane = forecast.lanes[Direction.NORTH]
        assert lane.predicted_queue_length_m == 14.0  # persistence fallback
        assert forecast.model_version == "linear-trend-v1"

    def test_flat_queue_projects_flat(self) -> None:
        queues = [10.0, 10.0, 10.0, 10.0, 10.0]
        forecast = LinearTrendForecaster(min_samples=4).predict(
            "i-1", _history(queues), horizon_s=30.0
        )
        lane = forecast.lanes[Direction.NORTH]
        assert lane.predicted_queue_length_m == 10.0
        assert lane.confidence > 0.9

    def test_prediction_never_negative(self) -> None:
        # Declining queue projected far enough forward should clamp at zero, not go negative.
        queues = [50.0, 40.0, 30.0, 20.0, 10.0]
        forecast = LinearTrendForecaster(min_samples=4).predict(
            "i-1", _history(queues), horizon_s=100.0
        )
        lane = forecast.lanes[Direction.NORTH]
        assert lane.predicted_queue_length_m >= 0.0

    def test_bounds_are_consistent(self) -> None:
        queues = [1.0, 3.0, 2.0, 5.0, 4.0, 6.0]  # noisy but trending up
        forecast = LinearTrendForecaster(min_samples=4).predict(
            "i-1", _history(queues), horizon_s=20.0
        )
        lane = forecast.lanes[Direction.NORTH]
        assert lane.lower_bound_m <= lane.predicted_queue_length_m <= lane.upper_bound_m


class TestCounterfactual:
    def test_accumulates_new_arrivals(self) -> None:
        result = counterfactual_wait(
            current_queue_len_m=10.0, arrival_rate_veh_s=0.2, horizon_s=30.0, vehicle_length_m=7.0
        )
        # 0.2 veh/s * 30s = 6 vehicles * 7m = 42m added.
        assert result == 10.0 + 42.0

    def test_no_arrivals_holds_steady(self) -> None:
        result = counterfactual_wait(
            current_queue_len_m=15.0, arrival_rate_veh_s=0.0, horizon_s=60.0, vehicle_length_m=7.0
        )
        assert result == 15.0
