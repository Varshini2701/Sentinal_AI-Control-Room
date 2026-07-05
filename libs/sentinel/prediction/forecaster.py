"""Short-horizon traffic forecasting: the reasoning core of the Prediction Agent.

:class:`TrafficPredictor` is the port; two deterministic, dependency-free implementations satisfy
it. :class:`PersistenceForecaster` is the naive "nothing changes" baseline and the failure-recovery
fallback. :class:`LinearTrendForecaster` fits a least-squares line to the recent queue-length
samples per lane and projects it forward, with a confidence that degrades as the fit's residual
error grows -- a fast, explainable stand-in for the LSTM/TFT model the architecture allows swapping
in later behind this same port.

:func:`counterfactual_wait` answers "what would happen if we didn't switch?" -- extrapolating the
queue that never gets served -- which is what lets the Explainability Agent (M6) produce statements
like *"predicted West wait if switched: 12s."*
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

from sentinel.contracts.enums import Direction
from sentinel.contracts.value_objects import Forecast, IntersectionState, LaneForecast


@dataclass(frozen=True, slots=True)
class _Sample:
    t: float
    queue_len_m: float
    wait_s: float


class TrafficPredictor(abc.ABC):
    """Produces a :class:`Forecast` from a per-lane history of recent samples."""

    @abc.abstractmethod
    def predict(
        self,
        intersection_id: str,
        history: dict[Direction, list[tuple[float, IntersectionState]]],
        horizon_s: float,
    ) -> Forecast:
        """Forecast each lane ``horizon_s`` seconds ahead from ``(timestamp, state)`` history."""


class PersistenceForecaster(TrafficPredictor):
    """Naive forecaster: predicts the most recent observed value, unchanged.

    This is both a legitimate baseline (traffic often *is* persistent over a 30-60s horizon) and
    the safe fallback when the trend model has too few samples or fails.
    """

    def predict(
        self,
        intersection_id: str,
        history: dict[Direction, list[tuple[float, IntersectionState]]],
        horizon_s: float,
    ) -> Forecast:
        lanes: dict[Direction, LaneForecast] = {}
        for direction, points in history.items():
            if not points:
                continue
            _, latest_state = points[-1]
            lane = latest_state.lanes.get(direction)
            if lane is None:
                continue
            lanes[direction] = LaneForecast(
                direction=direction,
                horizon_s=horizon_s,
                predicted_queue_length_m=lane.queue_length_m,
                predicted_wait_s=lane.avg_wait_s,
                confidence=0.5,
                lower_bound_m=lane.queue_length_m,
                upper_bound_m=lane.queue_length_m,
            )
        return Forecast(
            intersection_id=intersection_id,
            horizon_s=horizon_s,
            lanes=lanes,
            model_version="persistence-v1",
        )


class LinearTrendForecaster(TrafficPredictor):
    """Least-squares linear extrapolation of queue length per lane.

    Args:
        min_samples: Below this many points for a lane, delegate to ``fallback``.
        fallback: The forecaster used when there is not enough history to fit a trend.
    """

    def __init__(
        self, *, min_samples: int = 4, fallback: TrafficPredictor | None = None
    ) -> None:
        self._min_samples = min_samples
        self._fallback = fallback or PersistenceForecaster()

    def predict(
        self,
        intersection_id: str,
        history: dict[Direction, list[tuple[float, IntersectionState]]],
        horizon_s: float,
    ) -> Forecast:
        fallback_lanes = self._fallback.predict(intersection_id, history, horizon_s).lanes
        lanes: dict[Direction, LaneForecast] = {}

        for direction, points in history.items():
            samples = [
                _Sample(
                    t=t,
                    queue_len_m=s.lanes[direction].queue_length_m,
                    wait_s=s.lanes[direction].avg_wait_s,
                )
                for t, s in points
                if direction in s.lanes
            ]
            if len(samples) < self._min_samples:
                if direction in fallback_lanes:
                    lanes[direction] = fallback_lanes[direction]
                continue
            lanes[direction] = self._fit_and_project(direction, samples, horizon_s)

        return Forecast(
            intersection_id=intersection_id,
            horizon_s=horizon_s,
            lanes=lanes,
            model_version="linear-trend-v1",
        )

    def _fit_and_project(
        self, direction: Direction, samples: list[_Sample], horizon_s: float
    ) -> LaneForecast:
        ts = [s.t for s in samples]
        queues = [s.queue_len_m for s in samples]
        slope, intercept, residual_std = _least_squares(ts, queues)

        t_target = ts[-1] + horizon_s
        predicted = max(0.0, slope * t_target + intercept)

        # Confidence shrinks as residual noise grows relative to the current queue's scale.
        scale = max(1.0, queues[-1])
        confidence = max(0.05, min(0.95, 1.0 - residual_std / scale))
        margin = residual_std * 1.96  # ~95% band under a normal-residual assumption
        lower = max(0.0, predicted - margin)
        upper = predicted + margin
        if predicted < lower:
            predicted = lower
        if predicted > upper:
            predicted = upper

        latest_wait = samples[-1].wait_s
        wait_scale = predicted / queues[-1] if queues[-1] > 0 else 1.0
        predicted_wait = max(0.0, latest_wait * wait_scale)

        return LaneForecast(
            direction=direction,
            horizon_s=horizon_s,
            predicted_queue_length_m=predicted,
            predicted_wait_s=predicted_wait,
            confidence=confidence,
            lower_bound_m=lower,
            upper_bound_m=upper,
        )


def _least_squares(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Fit ``y = slope*x + intercept`` and return ``(slope, intercept, residual_std)``."""
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    var_x = sum((x - mean_x) ** 2 for x in xs)
    if var_x == 0.0:
        slope = 0.0
    else:
        cov_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
        slope = cov_xy / var_x
    intercept = mean_y - slope * mean_x
    residuals = [y - (slope * x + intercept) for x, y in zip(xs, ys, strict=True)]
    residual_std = (sum(r**2 for r in residuals) / n) ** 0.5
    return slope, intercept, residual_std


def counterfactual_wait(
    current_queue_len_m: float, arrival_rate_veh_s: float, horizon_s: float, vehicle_length_m: float
) -> float:
    """Estimate the queue length after ``horizon_s`` if the lane is **not** served at all.

    Used for counterfactual explanations ("if we had switched, West would have kept growing to
    ~X m"). A pure, deterministic projection: the queue simply accumulates new arrivals.
    """
    added_length = arrival_rate_veh_s * horizon_s * vehicle_length_m
    return current_queue_len_m + added_length


__all__ = [
    "LinearTrendForecaster",
    "PersistenceForecaster",
    "TrafficPredictor",
    "counterfactual_wait",
]
