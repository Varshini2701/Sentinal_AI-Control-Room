"""Terminal visualiser for the Sentinel AI closed loop (Modules 1 + 2).

Runs the adaptive controller against the analytical traffic twin and renders periodic frames
showing per-lane queues, the live signal phase, and the *reason* behind each decision -- a preview
of the Explainability Agent. Ends with the fixed-timer vs. Sentinel-AI benchmark.

    python scripts/demo_intersection.py
    python scripts/demo_intersection.py --frames 10 --horizon 600

This is a demo script, not shipped library code; it depends only on the tested `sentinel` package.
"""

from __future__ import annotations

import argparse

from sentinel.config.settings import DecisionSettings
from sentinel.contracts.enums import Direction, SignalPhase
from sentinel.contracts.value_objects import IntersectionState, SignalCommand
from sentinel.observability.logging import configure_logging
from sentinel.simulation import (
    AdaptiveController,
    AnalyticalTrafficEnvironment,
    FixedTimerController,
    SimConfig,
    asymmetric_demand,
    run_comparison,
)

_PHASE_LABEL: dict[SignalPhase, str] = {
    SignalPhase.NS_GREEN: "NORTH-SOUTH: GREEN   east-west: red",
    SignalPhase.NS_YELLOW: "NORTH-SOUTH: yellow  east-west: red",
    SignalPhase.EW_GREEN: "north-south: red   EAST-WEST: GREEN",
    SignalPhase.EW_YELLOW: "north-south: red   EAST-WEST: yellow",
    SignalPhase.ALL_RED: "ALL RED (clearance)",
}

_REASON_TEMPLATES: dict[str, str] = {
    "current_lane_busiest": "Hold green: the active axis still has the longest queue.",
    "opposing_queue_longer": "Switch: the waiting axis queue ({queue_ew:.0f} EW / {queue_ns:.0f} NS) now exceeds the active one.",
    "current_lane_empty": "Switch: the active axis is empty while traffic waits on the other.",
    "fairness_anti_starvation": "Switch: fairness guarantee -- the waiting axis has been red too long.",
    "emergency_preemption": "EMERGENCY OVERRIDE: preempting to clear the emergency vehicle's approach.",
    "emergency_hold_green": "EMERGENCY: holding green for the emergency vehicle's approach.",
    "clearance_interval": "Clearance interval in progress (yellow / all-red).",
    "fixed_timer_cycle": "Fixed timer: advancing on schedule (demand-blind).",
}


def _explain(command: SignalCommand) -> str:
    template = _REASON_TEMPLATES.get(command.reason_code, command.reason_code)
    return template.format(**command.feature_snapshot) if command.feature_snapshot else template


def _bar(count: int, scale: int = 2, width: int = 24) -> str:
    filled = min(width, count // scale)
    return "#" * filled + "." * (width - filled)


def _render(state: IntersectionState, command: SignalCommand) -> str:
    lines = [
        f"  t = {state.phase_elapsed_s:>4.0f}s into phase   |   signal: {_PHASE_LABEL[state.current_phase]}",
        "  " + "-" * 58,
    ]
    for direction in Direction:
        lane = state.lanes[direction]
        marker = " *EMERGENCY*" if lane.has_emergency_vehicle else ""
        lines.append(
            f"  {direction.value:<5} |{_bar(lane.vehicle_count)}| "
            f"{lane.vehicle_count:>3} veh  wait {lane.avg_wait_s:>4.0f}s{marker}"
        )
    lines.append("  " + "-" * 58)
    lines.append(f"  DECISION: {command.action.value.upper()}")
    lines.append(f"  WHY:      {_explain(command)}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sentinel AI closed-loop terminal demo")
    parser.add_argument("--horizon", type=float, default=600.0, help="Simulated seconds")
    parser.add_argument("--frames", type=int, default=8, help="Number of snapshot frames to print")
    parser.add_argument("--ns-rate", type=float, default=0.22)
    parser.add_argument("--ew-rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    configure_logging(level="WARNING", json_output=False)  # keep the demo output clean
    args = build_parser().parse_args()
    settings = DecisionSettings()
    config = SimConfig(
        horizon_s=args.horizon, seed=args.seed, demand=asymmetric_demand(args.ns_rate, args.ew_rate)
    )

    env = AnalyticalTrafficEnvironment(config)
    controller = AdaptiveController(settings)
    state = env.reset()
    frame_every = max(1, config.total_steps // args.frames)

    print("\n" + "=" * 62)
    print("  SENTINEL AI - live closed loop (adaptive control on the twin)")
    print("  Heavy North-South demand, light East-West. Watch green follow demand.")
    print("=" * 62)

    for step in range(config.total_steps):
        command = controller.decide(state, config.dt_s)
        state = env.step(command)
        if step % frame_every == 0 or step == config.total_steps - 1:
            print(f"\n[ sim time {env.time_s:>4.0f}s ]")
            print(_render(state, command))

    # Headline A/B comparison.
    result = run_comparison(
        lambda: AnalyticalTrafficEnvironment(config),
        [FixedTimerController(settings, green_s=30), AdaptiveController(settings)],
        config,
        baseline="fixed_timer",
    )
    fixed = result.summaries["fixed_timer"]
    adaptive = result.summaries["adaptive"]
    print("\n" + "=" * 62)
    print("  RESULT vs a conventional fixed timer (same traffic):")
    print(f"    fixed timer : avg wait {fixed.avg_delay_s:>6.1f}s   max queue {fixed.max_queue_veh:>3}")
    print(f"    Sentinel AI : avg wait {adaptive.avg_delay_s:>6.1f}s   max queue {adaptive.max_queue_veh:>3}")
    print(f"    >> {result.wait_reduction_pct('adaptive'):.0f}% less waiting for drivers <<")
    print("=" * 62 + "\n")


if __name__ == "__main__":
    main()
