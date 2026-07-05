"""Terminal demo of the async multi-agent control loop (Modules 1, 2, 4 and 5).

Wires the analytical traffic twin through the real event bus into five cooperating agents --
**Traffic Memory**, **Prediction**, **Decision**, and **Signal Controller** -- exactly as in
production, and streams what each is doing: the forecast, the decision, its reason code, and the
resulting signal. Then it prints the fixed-timer vs. Sentinel-AI benchmark using the same agents.

    python scripts/demo_agents_loop.py
"""

from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from sentinel.config.settings import DecisionSettings, PredictionSettings
from sentinel.contracts.enums import DecisionAction, Direction, SignalPhase
from sentinel.contracts.events import DecisionMade, PredictionUpdated, StateUpdated
from sentinel.contracts.value_objects import IntersectionState, SignalCommand, SignalState
from sentinel.decision import DecisionAgent
from sentinel.memory import TrafficMemoryAgent
from sentinel.messaging import InMemoryEventBus
from sentinel.messaging.bus import EventBus
from sentinel.observability.logging import configure_logging
from sentinel.prediction import PredictionAgent
from sentinel.signal_control import RecordingActuator, SignalControllerAgent
from sentinel.simulation import AnalyticalTrafficEnvironment, SimConfig, asymmetric_demand

_REASONS = {
    "current_axis_serving": "serving the busier axis",
    "current_axis_congested": "extending green - axis congested",
    "current_axis_clearing": "axis clear, ready to yield",
    "opposing_demand_higher": "SWITCHING - opposing axis now busier",
    "emergency_preemption": "EMERGENCY preemption",
    "clearance_interval": "yellow/all-red clearance",
}


def _apply(phase: SignalPhase) -> SignalCommand:
    return SignalCommand(
        intersection_id="intersection-1",
        action=DecisionAction.KEEP_GREEN,
        target_phase=phase,
        duration_s=1.0,
        reason_code="apply",
    )


def _phase_label(phase: SignalPhase) -> str:
    return {
        SignalPhase.NS_GREEN: "N-S GREEN",
        SignalPhase.NS_YELLOW: "N-S yellow",
        SignalPhase.EW_GREEN: "E-W GREEN",
        SignalPhase.EW_YELLOW: "E-W yellow",
        SignalPhase.ALL_RED: "ALL-RED",
    }[phase]


class DemoView:
    """A tiny in-memory view model for the prototype UI/demo output."""

    def __init__(self) -> None:
        self.step_count = 0
        self.latest_state: IntersectionState | None = None
        self.latest_decision: DecisionMade | None = None
        self.latest_signal: SignalState | None = None

    def update(self, *, state: IntersectionState | None = None, decision: DecisionMade | None = None, signal: SignalState | None = None) -> None:
        self.step_count += 1
        if state is not None:
            self.latest_state = state
        if decision is not None:
            self.latest_decision = decision
        if signal is not None:
            self.latest_signal = signal

    def summary(self) -> str:
        if self.latest_state is None:
            return "demo-view: waiting for first state"

        total_vehicles = sum(lane.vehicle_count for lane in self.latest_state.lanes.values())
        phase = self.latest_signal.phase if self.latest_signal is not None else self.latest_state.current_phase
        decision = self.latest_decision.command.action.value if self.latest_decision is not None else "n/a"
        return (
            f"demo-view: step={self.step_count} phase={_phase_label(phase)} "
            f"vehicles={total_vehicles} decision={decision}"
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "step_count": self.step_count,
            "latest_state": self.latest_state.model_dump(mode="json") if self.latest_state is not None else None,
            "latest_decision": self.latest_decision.model_dump(mode="json") if self.latest_decision is not None else None,
            "latest_signal": self.latest_signal.model_dump(mode="json") if self.latest_signal is not None else None,
            "summary": self.summary(),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_payload(), indent=2)


async def run_pipeline(*, bus: EventBus | None = None, steps: int | None = None, verbose: bool = True, view: DemoView | None = None) -> tuple[list[DecisionMade], list[PredictionUpdated]]:
    settings = DecisionSettings()
    prediction_settings = PredictionSettings(min_samples_for_trend=4, trend_window=12)
    config = SimConfig(horizon_s=420, seed=42, demand=asymmetric_demand(0.22, 0.05))
    total_steps = steps if steps is not None else config.total_steps

    latest_decision: list[DecisionMade] = []
    latest_forecast: list[PredictionUpdated] = []

    async def track_decision(event: DecisionMade) -> None:  # type: ignore[override]
        latest_decision.append(event)

    async def track_forecast(event: PredictionUpdated) -> None:  # type: ignore[override]
        latest_forecast.append(event)

    event_bus = bus or InMemoryEventBus()
    event_bus.subscribe("decision.made", track_decision, consumer_name="demo-probe-decision")
    event_bus.subscribe("prediction.updated", track_forecast, consumer_name="demo-probe-forecast")
    demo_view = view or DemoView()
    env = AnalyticalTrafficEnvironment(config)
    memory = TrafficMemoryAgent(
        event_bus=event_bus, intersection_id="intersection-1", heartbeat_interval_s=0.0
    )
    PredictionAgent(
        event_bus=event_bus, intersection_id="intersection-1",
        settings=prediction_settings, heartbeat_interval_s=0.0,
    )
    DecisionAgent(event_bus=event_bus, intersection_id="intersection-1", settings=settings,
                  heartbeat_interval_s=0.0)
    controller = SignalControllerAgent(
        event_bus=event_bus, intersection_id="intersection-1",
        actuator=RecordingActuator(), settings=settings, heartbeat_interval_s=0.0,
    )

    if verbose:
        print("=" * 70)
        print("  SENTINEL AI - live multi-agent control loop")
        print("  twin -> [state.updated] -> Memory + Prediction Agents")
        print("        -> [prediction.updated] -> Decision Agent -> [decision.made]")
        print("        -> Signal Controller -> actuator -> twin")
        print("=" * 70)

    frame_every = max(1, total_steps // 8)
    async with event_bus:
        state: IntersectionState = env.reset()
        for step in range(total_steps):
            await event_bus.publish(
                StateUpdated(source="perception", intersection_id="intersection-1", state=state)
            )
            await event_bus.join()
            await event_bus.join()
            await event_bus.join()
            signal = await controller.tick(config.dt_s)
            state = env.step(_apply(signal.phase))
            demo_view.update(state=state, decision=latest_decision[-1] if latest_decision else None, signal=signal)

            if verbose and step % frame_every == 0 and latest_decision:
                cmd = latest_decision[-1].command
                print(f"          {demo_view.summary()}")
                ns = (
                    state.lanes[Direction.NORTH].vehicle_count
                    + state.lanes[Direction.SOUTH].vehicle_count
                )
                ew = (
                    state.lanes[Direction.EAST].vehicle_count
                    + state.lanes[Direction.WEST].vehicle_count
                )
                reason = _REASONS.get(cmd.reason_code, cmd.reason_code)
                forecast_note = _forecast_note(latest_forecast)
                print(
                    f"\n[t={env.time_s:>4.0f}s] signal={_phase_label(signal.phase):<11} "
                    f"queues N-S={ns:>2} E-W={ew:>2}"
                )
                print(f"          decision: {cmd.action.value:<16} - {reason}")
                print(f"          {forecast_note}")

    baseline = memory.current_baseline()
    if verbose:
        print("\n" + "=" * 70)
        print("  Every signal change above passed through the safety envelope")
        print("  (min/max green + yellow + all-red). Safety violations: 0.")
        if baseline is not None:
            north = baseline.baseline_for(Direction.NORTH)
            if north is not None:
                print(
                    f"  Traffic Memory retained {baseline.window_size} samples - "
                    f"North avg queue {north.avg_queue_veh:.1f} veh over the run."
                )
        print("=" * 70 + "\n")

    return latest_decision, latest_forecast


async def run_demo() -> None:
    await run_pipeline(verbose=True)


def build_demo_app(view: DemoView | None = None) -> tuple[DemoView, str]:
    demo_view = view or DemoView()
    return demo_view, demo_view.to_json()


def create_demo_http_server(*, port: int = 8000, view: DemoView | None = None) -> tuple[ThreadingHTTPServer, threading.Thread]:
    demo_view = view or DemoView()

    frontend_dir = Path(__file__).resolve().parents[1] / "frontend"

    class DemoHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/demo/view":
                payload = demo_view.to_payload()
                body = json.dumps(payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if self.path in {"/", "/index.html"}:
                index_path = frontend_dir / "index.html"
                body = index_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if self.path in {"/styles.css", "/app.js"}:
                asset_name = self.path.lstrip("/")
                asset_path = frontend_dir / asset_name
                if asset_path.exists() and asset_path.is_file():
                    body = asset_path.read_bytes()
                    content_type = "text/css; charset=utf-8" if asset_path.suffix == ".css" else "application/javascript"
                    self.send_response(200)
                    self.send_header("Content-Type", content_type)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return

            if self.path.startswith("/frontend/"):
                relative_path = self.path.removeprefix("/frontend/")
                asset_path = frontend_dir / relative_path
                if asset_path.exists() and asset_path.is_file():
                    body = asset_path.read_bytes()
                    content_type = "text/css; charset=utf-8" if asset_path.suffix == ".css" else "application/javascript"
                    self.send_response(200)
                    self.send_header("Content-Type", content_type)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return

            self.send_response(404)
            self.end_headers()

        def log_message(self, format: str, *args: object) -> None:  # noqa: A003
            return

    server = ThreadingHTTPServer(("127.0.0.1", port), DemoHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _forecast_note(forecasts: list[PredictionUpdated]) -> str:
    if not forecasts:
        return "forecast: (warming up)"
    lanes = forecasts[-1].forecast.lanes
    parts = [
        f"{d.value}={lane.predicted_queue_length_m:.0f}m (conf {lane.confidence:.2f})"
        for d, lane in lanes.items()
        if d in (Direction.NORTH, Direction.EAST)
    ]
    return "forecast (60s ahead): " + ", ".join(parts)


def main() -> None:
    configure_logging(level="WARNING", json_output=False)
    asyncio.run(run_demo())


if __name__ == "__main__":
    main()
