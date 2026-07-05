"""Key performance indicators and A/B comparison results for benchmark runs.

The headline metric is **average delay per vehicle** (total vehicle-seconds of delay divided by
vehicles served). ``ComparisonResult`` turns two runs into the one number that sells the project:
the percentage reduction in wait time Sentinel AI achieves over a fixed timer on identical traffic.
"""

from __future__ import annotations

from pydantic import Field

from sentinel.contracts.base import FrozenModel
from sentinel.contracts.enums import Direction


class LaneKpi(FrozenModel):
    """Per-approach performance over a run."""

    direction: Direction
    arrived: int
    served: int
    total_delay_veh_s: float
    max_queue_veh: int

    @property
    def avg_delay_s(self) -> float:
        """Mean delay per served vehicle on this approach."""
        return self.total_delay_veh_s / self.served if self.served else 0.0


class KpiSummary(FrozenModel):
    """Aggregate performance of a single controller over a full run."""

    controller: str = Field(min_length=1)
    sim_duration_s: float = Field(gt=0.0)
    total_arrived: int
    total_served: int
    total_delay_veh_s: float
    max_queue_veh: int
    lanes: dict[Direction, LaneKpi]

    @property
    def avg_delay_s(self) -> float:
        """Mean delay per served vehicle across the intersection (the headline metric)."""
        return self.total_delay_veh_s / self.total_served if self.total_served else 0.0

    @property
    def throughput_vph(self) -> float:
        """Vehicles served per hour."""
        return self.total_served / self.sim_duration_s * 3600.0

    @property
    def clearance_rate(self) -> float:
        """Fraction of arriving vehicles that were served (1.0 == no residual queue growth)."""
        return self.total_served / self.total_arrived if self.total_arrived else 1.0


class ComparisonResult(FrozenModel):
    """The outcome of running several controllers on one scenario, versus a baseline."""

    baseline: str = Field(min_length=1)
    summaries: dict[str, KpiSummary]

    def _pct_change(self, candidate: str, attr: str, *, lower_is_better: bool) -> float:
        base = float(getattr(self.summaries[self.baseline], attr))
        cand = float(getattr(self.summaries[candidate], attr))
        if base == 0:
            return 0.0
        change = (cand - base) / base * 100.0
        return -change if lower_is_better else change

    def wait_reduction_pct(self, candidate: str) -> float:
        """Percentage cut in average delay of ``candidate`` vs. the baseline (higher is better)."""
        return self._pct_change(candidate, "avg_delay_s", lower_is_better=True)

    def throughput_gain_pct(self, candidate: str) -> float:
        """Percentage increase in throughput of ``candidate`` vs. the baseline."""
        return self._pct_change(candidate, "throughput_vph", lower_is_better=False)


__all__ = ["ComparisonResult", "KpiSummary", "LaneKpi"]
