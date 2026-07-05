"""Traffic simulation environments, controllers and the closed-loop benchmark harness.

The :class:`AnalyticalTrafficEnvironment` is dependency-free and always available. The
:class:`~sentinel.simulation.sumo.SumoTrafficEnvironment` requires the SUMO binary and the ``traci``
package and is imported from its own module on demand.
"""

from __future__ import annotations

from sentinel.simulation.analytical import AnalyticalTrafficEnvironment
from sentinel.simulation.config import (
    EmergencyEvent,
    LaneDemand,
    SimConfig,
    asymmetric_demand,
    symmetric_demand,
)
from sentinel.simulation.controllers import (
    AdaptiveController,
    Controller,
    FixedTimerController,
)
from sentinel.simulation.environment import TrafficEnvironment
from sentinel.simulation.harness import run_comparison, run_controller
from sentinel.simulation.kpi import ComparisonResult, KpiSummary, LaneKpi

__all__ = [
    "AdaptiveController",
    "AnalyticalTrafficEnvironment",
    "ComparisonResult",
    "Controller",
    "EmergencyEvent",
    "FixedTimerController",
    "KpiSummary",
    "LaneDemand",
    "LaneKpi",
    "SimConfig",
    "TrafficEnvironment",
    "asymmetric_demand",
    "run_comparison",
    "run_controller",
    "symmetric_demand",
]
