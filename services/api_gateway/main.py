"""The Sentinel AI API gateway: a thin FastAPI read-API + WebSocket stream over a live fleet.

On startup this builds and starts a full :class:`~sentinel.fleet.Fleet` (every cognition agent,
driving the analytical traffic twin) and a background task that keeps it ticking. Every HTTP/WS
endpoint is **read-only** against the Dashboard and Orchestrator agents' live snapshots -- the
gateway itself never touches the control loop, matching the hybrid deployment split (this process
is what runs behind Vercel's thin API layer; the perception/agents/SUMO stay on the GPU host).

Run locally with::

    pip install -e ".[api]"
    uvicorn services.api_gateway.main:app --reload

Requires the optional ``api`` extra (``fastapi``, ``uvicorn``, ``websockets``).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import make_asgi_app

from sentinel.fleet import Fleet, build_fleet, run_simulated_loop
from sentinel.observability.logging import configure_logging, get_logger
from sentinel.observability.metrics import SENTINEL_REGISTRY

_log = get_logger("sentinel.api_gateway")
_FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"

_fleets: dict[str, Fleet] = {}
_loop_tasks: dict[str, asyncio.Task[None]] = {}
_DEFAULT_INTERSECTION = "intersection-1"


async def _startup() -> None:
    configure_logging(level="INFO", json_output=False)
    fleet = build_fleet(_DEFAULT_INTERSECTION)
    await fleet.start()
    _fleets[_DEFAULT_INTERSECTION] = fleet
    _loop_tasks[_DEFAULT_INTERSECTION] = asyncio.create_task(
        run_simulated_loop(fleet, fleet.config, tick_sleep_s=1.0),
        name="sim-loop",
    )
    _log.info("fleet_started", intersection_id=_DEFAULT_INTERSECTION)


async def _shutdown() -> None:
    for task in _loop_tasks.values():
        task.cancel()
    for task in _loop_tasks.values():
        try:
            await task
        except asyncio.CancelledError:
            pass
    for fleet in _fleets.values():
        await fleet.stop()
    _log.info("fleet_stopped")


async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    await _startup()
    yield
    await _shutdown()


app = FastAPI(title="Sentinel AI API Gateway", version="0.1.0", lifespan=_lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/metrics", make_asgi_app(registry=SENTINEL_REGISTRY))


def _get_fleet(intersection_id: str) -> Fleet:
    fleet = _fleets.get(intersection_id)
    if fleet is None:
        raise HTTPException(status_code=404, detail=f"unknown intersection {intersection_id!r}")
    return fleet


@app.get("/api/v1/system/health")
def system_health() -> dict[str, Any]:
    """Aggregate mode + per-agent health across every running intersection."""
    return {
        intersection_id: {
            "mode": fleet.orchestrator.snapshot().mode.value,
            "degraded_reason": fleet.orchestrator.snapshot().degraded_reason,
            "agents": {
                name: health.model_dump(mode="json")
                for name, health in fleet.orchestrator.snapshot().agents.items()
            },
        }
        for intersection_id, fleet in _fleets.items()
    }


@app.get("/api/v1/intersections/{intersection_id}/state")
def intersection_state(intersection_id: str) -> dict[str, Any]:
    """The full live snapshot: state, signal, decision, forecast, incident, explanation."""
    fleet = _get_fleet(intersection_id)
    return fleet.dashboard.snapshot().to_dict()


@app.get("/api/v1/intersections/{intersection_id}/logs")
def intersection_logs(intersection_id: str) -> list[str]:
    """Recent human-readable system log lines (decisions, signals, incidents, mode changes)."""
    fleet = _get_fleet(intersection_id)
    return fleet.dashboard.logs()


@app.websocket("/api/v1/ws/live/{intersection_id}")
async def live_stream(websocket: WebSocket, intersection_id: str) -> None:
    """Push the live snapshot + logs to the client roughly once per second."""
    if intersection_id not in _fleets:
        await websocket.close(code=4004)
        return
    fleet = _fleets[intersection_id]
    await websocket.accept()
    try:
        while True:
            await websocket.send_json(
                {"snapshot": fleet.dashboard.snapshot().to_dict(), "logs": fleet.dashboard.logs()}
            )
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        _log.debug("ws_client_disconnected", intersection_id=intersection_id)


if _FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=_FRONTEND_DIR, html=True), name="frontend")
else:

    @app.get("/")
    def root() -> FileResponse | dict[str, str]:
        return {"detail": "frontend not built; see services/api_gateway/main.py"}


__all__ = ["app"]
