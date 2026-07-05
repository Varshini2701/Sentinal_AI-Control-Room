from __future__ import annotations

import json
import urllib.request

from scripts.demo_agents_loop import DemoView, create_demo_http_server, run_pipeline

from sentinel.contracts.events import DecisionMade, SignalChanged
from sentinel.messaging import InMemoryEventBus


class TestDemoPipeline:
    async def test_run_pipeline_emits_decision_and_signal_events(self) -> None:
        seen: list[object] = []

        async def capture(event: object) -> None:
            seen.append(event)

        bus = InMemoryEventBus()
        bus.subscribe("decision.made", capture, consumer_name="capture")
        bus.subscribe("signal.changed", capture, consumer_name="capture")

        # 65 steps guarantees a phase transition: max_green_s defaults to 60s, so the Signal
        # Controller must force a switch by then regardless of demand.
        await run_pipeline(bus=bus, steps=65, verbose=False)

        assert any(isinstance(event, DecisionMade) for event in seen)
        assert any(isinstance(event, SignalChanged) for event in seen)

    async def test_run_pipeline_updates_demo_view(self) -> None:
        view = DemoView()
        bus = InMemoryEventBus()

        await run_pipeline(bus=bus, steps=5, verbose=False, view=view)

        assert view.step_count > 0
        assert view.latest_state is not None
        assert view.latest_decision is not None
        assert view.latest_signal is not None

    def test_demo_view_payload_is_json_serializable(self) -> None:
        view = DemoView()

        payload = view.to_payload()

        assert payload["step_count"] == 0
        assert payload["latest_state"] is None
        assert payload["latest_decision"] is None
        assert payload["latest_signal"] is None

    def test_demo_http_server_serves_view_payload(self) -> None:
        view = DemoView()
        server, thread = create_demo_http_server(port=0, view=view)
        try:
            port = server.server_address[1]
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/demo/view") as response:
                payload = json.loads(response.read().decode("utf-8"))
            assert payload["step_count"] == 0
            assert payload["summary"].startswith("demo-view")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_demo_http_server_serves_frontend_entrypoint(self) -> None:
        view = DemoView()
        server, thread = create_demo_http_server(port=0, view=view)
        try:
            port = server.server_address[1]
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/") as response:
                body = response.read().decode("utf-8")
            assert "<!doctype html>" in body.lower()
            assert "Sentinel" in body
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
