"""Tests for the perception worker: publishing state and failure recovery."""

from __future__ import annotations

from typing import Any

from tests.conftest import make_box, make_detection

from sentinel.contracts.enums import Direction, SignalPhase
from sentinel.contracts.events import StateUpdated
from sentinel.messaging import InMemoryEventBus
from sentinel.observability.metrics import SENTINEL_REGISTRY
from sentinel.perception.detector import ObjectDetector
from sentinel.perception.geometry import default_calibration
from sentinel.perception.pipeline import PerceptionPipeline
from sentinel.perception.video import ScriptedFrameSource
from sentinel.perception.worker import PerceptionWorker


class _ListDetector(ObjectDetector):
    """Test detector: the 'frame' *is* the detection list, so bundles carry detections directly."""

    def detect(self, frame: Any) -> list:
        return list(frame) if frame else []


class _ExplodingDetector(ObjectDetector):
    def detect(self, frame: Any) -> list:
        raise RuntimeError("inference exploded")


def _bundle() -> dict[Direction, list]:
    return {Direction.NORTH: [make_detection(box=make_box(300, 200, 340, 260))]}


class TestPerceptionWorker:
    async def test_publishes_state_updates(self) -> None:
        received: list[StateUpdated] = []

        async def handler(event: StateUpdated) -> None:  # type: ignore[override]
            received.append(event)

        bus = InMemoryEventBus()
        bus.subscribe("state.updated", handler, consumer_name="test")
        worker = PerceptionWorker(
            pipeline=PerceptionPipeline(default_calibration()),
            detector=_ListDetector(),
            source=ScriptedFrameSource([_bundle(), _bundle()]),
            event_bus=bus,
            intersection_id="intersection-1",
            signal_provider=lambda: (SignalPhase.EW_GREEN, 5.0),
        )
        async with bus:
            processed = await worker.run()
            await bus.join()

        assert processed == 2
        assert len(received) == 2
        # By the second frame the stationary vehicle is confirmed and reported.
        assert received[-1].state.lanes[Direction.NORTH].vehicle_count == 1
        assert received[-1].state.current_phase is SignalPhase.EW_GREEN
        assert worker.last_state is not None

    async def test_detector_failure_is_recovered(self) -> None:
        published: list[StateUpdated] = []

        async def handler(event: StateUpdated) -> None:  # type: ignore[override]
            published.append(event)

        bus = InMemoryEventBus()
        bus.subscribe("state.updated", handler, consumer_name="test")
        labels = {"intersection": "intersection-1"}
        before = SENTINEL_REGISTRY.get_sample_value(
            "sentinel_perception_dropped_frames_total", labels
        ) or 0.0

        worker = PerceptionWorker(
            pipeline=PerceptionPipeline(default_calibration()),
            detector=_ExplodingDetector(),
            source=ScriptedFrameSource([_bundle(), _bundle()]),
            event_bus=bus,
            intersection_id="intersection-1",
        )
        async with bus:
            processed = await worker.run()  # must not raise
            await bus.join()

        after = SENTINEL_REGISTRY.get_sample_value(
            "sentinel_perception_dropped_frames_total", labels
        )
        assert processed == 0
        assert published == []  # no last-good state to fall back to
        assert after == before + 2  # both frames dropped and metered

    async def test_max_frames_limits_processing(self) -> None:
        bus = InMemoryEventBus()
        worker = PerceptionWorker(
            pipeline=PerceptionPipeline(default_calibration()),
            detector=_ListDetector(),
            source=ScriptedFrameSource([_bundle()] * 10),
            event_bus=bus,
            intersection_id="intersection-1",
        )
        async with bus:
            processed = await worker.run(max_frames=3)
        assert processed == 3
