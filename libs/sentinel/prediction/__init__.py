"""The Prediction Agent and its pluggable forecaster."""

from __future__ import annotations

from sentinel.prediction.agent import PredictionAgent
from sentinel.prediction.forecaster import (
    LinearTrendForecaster,
    PersistenceForecaster,
    TrafficPredictor,
    counterfactual_wait,
)

__all__ = [
    "LinearTrendForecaster",
    "PersistenceForecaster",
    "PredictionAgent",
    "TrafficPredictor",
    "counterfactual_wait",
]
