"""Tests for the FastAPI gateway: read-only endpoints over a live fleet."""

from __future__ import annotations

import time

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient
from services.api_gateway.main import app


class TestApiGateway:
    def test_health_reports_running_intersection(self) -> None:
        with TestClient(app) as client:
            response = client.get("/api/v1/system/health")
            assert response.status_code == 200
            body = response.json()
            assert "intersection-1" in body
            assert body["intersection-1"]["mode"] in ("ai", "degraded")
            assert "decision-agent" in body["intersection-1"]["agents"]

    def test_state_endpoint_returns_snapshot_shape(self) -> None:
        with TestClient(app) as client:
            response = client.get("/api/v1/intersections/intersection-1/state")
            assert response.status_code == 200
            body = response.json()
            assert body["intersection_id"] == "intersection-1"
            assert "mode" in body
            assert "agent_health" in body

    def test_state_becomes_populated_once_the_loop_ticks(self) -> None:
        with TestClient(app) as client:
            time.sleep(1.5)  # allow the background simulated loop to publish at least one state
            response = client.get("/api/v1/intersections/intersection-1/state")
            body = response.json()
            assert body["state"] is not None
            assert body["signal"] is not None

    def test_unknown_intersection_returns_404(self) -> None:
        with TestClient(app) as client:
            response = client.get("/api/v1/intersections/does-not-exist/state")
            assert response.status_code == 404

    def test_logs_endpoint_returns_a_list(self) -> None:
        with TestClient(app) as client:
            response = client.get("/api/v1/intersections/intersection-1/logs")
            assert response.status_code == 200
            assert isinstance(response.json(), list)

    def test_metrics_endpoint_exposes_prometheus_format(self) -> None:
        with TestClient(app) as client:
            # Mounted ASGI apps require the trailing slash; see deploy/prometheus.yml.
            response = client.get("/metrics/")
            assert response.status_code == 200
            assert "sentinel_agent_up" in response.text

    def test_websocket_streams_a_snapshot(self) -> None:
        with TestClient(app) as client, client.websocket_connect(
            "/api/v1/ws/live/intersection-1"
        ) as ws:
            message = ws.receive_json()
            assert "snapshot" in message
            assert "logs" in message
