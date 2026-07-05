"""CLI: run the fixed-timer vs. Sentinel-AI adaptive benchmark and print the result.

    python -m sentinel.simulation.run_baseline
    python -m sentinel.simulation.run_baseline --ns-rate 0.22 --ew-rate 0.05 --horizon 1800

Runs both controllers on an identical, seeded, asymmetric-demand scenario using the analytical
traffic twin, then reports the average-wait reduction Sentinel AI achieves over the fixed timer.
"""

from __future__ import annotations

import argparse

from sentinel.config.settings import DecisionSettings
from sentinel.observability.logging import configure_logging
from sentinel.simulation.analytical import AnalyticalTrafficEnvironment
from sentinel.simulation.config import SimConfig, asymmetric_demand
from sentinel.simulation.controllers import AdaptiveController, FixedTimerController
from sentinel.simulation.harness import run_comparison
from sentinel.simulation.kpi import ComparisonResult


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sentinel AI baseline benchmark")
    parser.add_argument("--ns-rate", type=float, default=0.22, help="North/South arrivals (veh/s)")
    parser.add_argument("--ew-rate", type=float, default=0.05, help="East/West arrivals (veh/s)")
    parser.add_argument("--horizon", type=float, default=1800.0, help="Simulated seconds")
    parser.add_argument("--fixed-green", type=float, default=30.0, help="Fixed-timer green (s)")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def run(args: argparse.Namespace) -> ComparisonResult:
    settings = DecisionSettings()
    config = SimConfig(
        horizon_s=args.horizon,
        seed=args.seed,
        demand=asymmetric_demand(args.ns_rate, args.ew_rate),
    )
    controllers = [
        FixedTimerController(settings, green_s=args.fixed_green),
        AdaptiveController(settings),
    ]
    return run_comparison(
        lambda: AnalyticalTrafficEnvironment(config),
        controllers,
        config,
        baseline="fixed_timer",
    )


def format_report(result: ComparisonResult) -> str:
    lines = [
        "",
        "Sentinel AI - Baseline Benchmark (analytical traffic twin)",
        "=" * 62,
        f"{'controller':<16}{'avg wait (s)':>14}{'throughput/h':>16}{'max queue':>12}",
        "-" * 62,
    ]
    for name, summary in result.summaries.items():
        lines.append(
            f"{name:<16}{summary.avg_delay_s:>14.2f}"
            f"{summary.throughput_vph:>16.1f}{summary.max_queue_veh:>12}"
        )
    lines.append("-" * 62)
    for name in result.summaries:
        if name == result.baseline:
            continue
        wait = result.wait_reduction_pct(name)
        thru = result.throughput_gain_pct(name)
        lines.append(
            f"{name} vs {result.baseline}: "
            f"wait {wait:+.1f}%  |  throughput {thru:+.1f}%"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    configure_logging(level="WARNING", json_output=False)
    args = build_parser().parse_args()
    result = run(args)
    print(format_report(result))


if __name__ == "__main__":
    main()
