"""Sentinel AI — Streamlit dashboard.

Unlike Vercel's serverless model, Streamlit (self-hosted or Streamlit Community Cloud) runs one
persistent Python process per app -- which is exactly what this system needs: a long-running
agent fleet with in-memory state and a continuously-ticking control loop. This file *is* the
deployable app: Streamlit Community Cloud auto-detects `streamlit_app.py` at the repo root.

Run locally:
    pip install -e ".[streamlit]"
    streamlit run streamlit_app.py

The fleet (all 8 cognition agents + the analytical traffic twin) is started exactly once per
server process via `st.cache_resource`, on a dedicated background thread running its own asyncio
event loop -- Streamlit's script-rerun model is synchronous, so the agent loop must live outside
of it. Every page render just reads the Dashboard Agent's live snapshot; it never blocks on I/O.
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass

import streamlit as st

from sentinel.config.settings import DecisionSettings
from sentinel.contracts.enums import AgentStatus, Direction, SignalPhase
from sentinel.fleet import Fleet, build_fleet, run_simulated_loop
from sentinel.observability.logging import configure_logging
from sentinel.simulation import (
    AdaptiveController,
    AnalyticalTrafficEnvironment,
    FixedTimerController,
    SimConfig,
    asymmetric_demand,
    run_comparison,
)

st.set_page_config(page_title="Sentinel AI — Control Room", page_icon="🚦", layout="wide")

_PHASE_LABEL = {
    SignalPhase.NS_GREEN: "🟢 NORTH-SOUTH GREEN",
    SignalPhase.NS_YELLOW: "🟡 North-South Yellow",
    SignalPhase.EW_GREEN: "🟢 EAST-WEST GREEN",
    SignalPhase.EW_YELLOW: "🟡 East-West Yellow",
    SignalPhase.ALL_RED: "🔴 ALL RED (clearance)",
}
_STATUS_EMOJI = {
    AgentStatus.HEALTHY: "🟢",
    AgentStatus.DEGRADED: "🟡",
    AgentStatus.UNHEALTHY: "🔴",
    AgentStatus.STOPPED: "⚪",
}


@dataclass
class _FleetHandle:
    fleet: Fleet
    loop: asyncio.AbstractEventLoop


def _run_fleet_forever(fleet: Fleet, config: SimConfig, loop: asyncio.AbstractEventLoop) -> None:
    asyncio.set_event_loop(loop)

    async def _main() -> None:
        await fleet.start()
        await run_simulated_loop(fleet, config, tick_sleep_s=1.0)

    loop.run_until_complete(_main())


@st.cache_resource(show_spinner="Starting the Sentinel AI agent fleet…")
def get_fleet_handle() -> _FleetHandle:
    """Build and start the fleet exactly once per server process."""
    configure_logging(level="WARNING", json_output=False)
    fleet = build_fleet("intersection-1")
    loop = asyncio.new_event_loop()
    thread = threading.Thread(
        target=_run_fleet_forever,
        args=(fleet, fleet.config, loop),
        daemon=True,
        name="sentinel-fleet",
    )
    thread.start()
    time.sleep(0.3)  # let the fleet finish start() before the first render reads it
    return _FleetHandle(fleet=fleet, loop=loop)


@st.cache_data(ttl=3600)
def run_baseline_benchmark() -> dict[str, float]:
    """Fixed-timer vs. Sentinel-AI on identical seeded traffic (cached: it's deterministic)."""
    settings = DecisionSettings()
    config = SimConfig(horizon_s=1500, seed=42, demand=asymmetric_demand(0.22, 0.05))
    result = run_comparison(
        lambda: AnalyticalTrafficEnvironment(config),
        [FixedTimerController(settings, green_s=30), AdaptiveController(settings)],
        config,
        baseline="fixed_timer",
    )
    fixed, adaptive = result.summaries["fixed_timer"], result.summaries["adaptive"]
    return {
        "fixed_wait": fixed.avg_delay_s,
        "adaptive_wait": adaptive.avg_delay_s,
        "reduction_pct": result.wait_reduction_pct("adaptive"),
        "fixed_max_queue": fixed.max_queue_veh,
        "adaptive_max_queue": adaptive.max_queue_veh,
    }


def render_header(mode: str, degraded_reason: str | None) -> None:
    left, right = st.columns([3, 1])
    with left:
        st.title("🚦 Sentinel AI — Control Room")
        st.caption("Autonomous Traffic Intelligence · live agent fleet · intersection-1")
    with right:
        badge = "🟢 AI" if mode == "ai" else f"🟡 {mode.upper()}"
        st.metric("System Mode", badge)
        if degraded_reason:
            st.caption(degraded_reason)


def render_lanes(state) -> None:  # noqa: ANN001
    st.subheader("Lane Queues")
    cols = st.columns(4)
    for col, direction in zip(cols, Direction, strict=True):
        lane = state.lanes.get(direction) if state else None
        with col:
            emergency = " 🚨" if lane and lane.has_emergency_vehicle else ""
            st.markdown(f"**{direction.value.upper()}**{emergency}")
            if lane is None:
                st.progress(0, text="waiting…")
                continue
            pct = min(100, int(lane.occupancy_pct))
            st.progress(pct / 100, text=f"{lane.vehicle_count} veh")
            st.caption(f"wait {lane.avg_wait_s:.0f}s · {lane.density.value}")


def render_signal_and_decision(snapshot) -> None:  # noqa: ANN001
    left, right = st.columns(2)
    with left:
        st.subheader("Signal")
        phase = snapshot.signal.phase if snapshot.signal else (
            snapshot.state.current_phase if snapshot.state else None
        )
        st.markdown(f"### {_PHASE_LABEL.get(phase, '—')}" if phase else "### —")
        elapsed = snapshot.signal.phase_elapsed_s if snapshot.signal else None
        if elapsed is not None:
            st.caption(f"elapsed {elapsed:.0f}s")
    with right:
        st.subheader("Decision")
        if snapshot.decision:
            action = snapshot.decision.action.value.upper()
            st.markdown(f"**{action}** — {snapshot.decision.reason_code}")
        else:
            st.markdown("—")
        if snapshot.latest_explanation:
            st.info(snapshot.latest_explanation.text)
        else:
            st.caption("Waiting for the first decision…")


def render_agents_and_logs(snapshot, logs: list[str]) -> None:  # noqa: ANN001
    left, right = st.columns(2)
    with left:
        st.subheader("Agent Health")
        if not snapshot.agent_health:
            st.caption("no heartbeats yet")
        for name, health in sorted(snapshot.agent_health.items()):
            st.markdown(f"{_STATUS_EMOJI.get(health.status, '⚪')} {name}")
    with right:
        st.subheader("System Log")
        st.code("\n".join(reversed(logs[-30:])) or "(no events yet)", language=None)


def render_benchmark() -> None:
    st.divider()
    st.subheader("Proof: Sentinel AI vs. a Fixed Timer")
    st.caption("Same seeded traffic, same duration — deterministic, reproducible.")
    result = run_baseline_benchmark()
    c1, c2, c3 = st.columns(3)
    c1.metric("Fixed timer — avg wait", f"{result['fixed_wait']:.1f}s")
    c2.metric(
        "Sentinel AI — avg wait",
        f"{result['adaptive_wait']:.1f}s",
        delta=f"-{result['reduction_pct']:.0f}%",
        delta_color="inverse",
    )
    c3.metric(
        "Max queue reduction",
        f"{result['fixed_max_queue']:.0f} → {result['adaptive_max_queue']:.0f} veh",
    )


def main() -> None:
    handle = get_fleet_handle()
    snapshot = handle.fleet.dashboard.snapshot()
    logs = handle.fleet.dashboard.logs()

    render_header(snapshot.mode.value, snapshot.degraded_reason)
    st.divider()
    render_lanes(snapshot.state)
    st.divider()
    render_signal_and_decision(snapshot)
    st.divider()
    render_agents_and_logs(snapshot, logs)
    render_benchmark()

    # Auto-refresh so the dashboard feels live. Skipped under automated testing (AppTest expects
    # a script to settle; the infinite rerun loop is intentional in the real deployed app).
    if not st.session_state.get("_disable_autorefresh"):
        time.sleep(1.0)
        st.rerun()


if __name__ == "__main__":
    main()
