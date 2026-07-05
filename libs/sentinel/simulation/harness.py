"""The closed-loop benchmark harness.

Runs one or more controllers against fresh instances of the *same* scenario and returns a
:class:`ComparisonResult`. Because each controller gets a freshly seeded environment, the arrival
streams are identical -- so any difference in delay is attributable to the control policy alone.
This is the harness that produces the headline "Sentinel AI reduces average wait by N%" number.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from sentinel.observability.logging import get_logger
from sentinel.simulation.config import SimConfig
from sentinel.simulation.controllers import Controller
from sentinel.simulation.environment import TrafficEnvironment
from sentinel.simulation.kpi import ComparisonResult, KpiSummary

_log = get_logger("sentinel.simulation.harness")

EnvFactory = Callable[[], TrafficEnvironment]


def run_controller(
    env_factory: EnvFactory, controller: Controller, config: SimConfig
) -> KpiSummary:
    """Run a single controller for a full horizon and return its KPIs."""
    env = env_factory()
    try:
        controller.reset()
        state = env.reset()
        for _ in range(config.total_steps):
            command = controller.decide(state, config.dt_s)
            state = env.step(command)
        summary = env.metrics().model_copy(update={"controller": controller.name})
    finally:
        env.close()
    _log.info(
        "controller_run_complete",
        controller=controller.name,
        avg_delay_s=round(summary.avg_delay_s, 2),
        throughput_vph=round(summary.throughput_vph, 1),
        served=summary.total_served,
    )
    return summary


def run_comparison(
    env_factory: EnvFactory,
    controllers: Sequence[Controller],
    config: SimConfig,
    *,
    baseline: str,
) -> ComparisonResult:
    """Run every controller on identical traffic and compare them against ``baseline``."""
    if not any(c.name == baseline for c in controllers):
        raise ValueError(f"baseline {baseline!r} is not among the provided controllers")
    summaries = {c.name: run_controller(env_factory, c, config) for c in controllers}
    return ComparisonResult(baseline=baseline, summaries=summaries)


__all__ = ["EnvFactory", "run_comparison", "run_controller"]
