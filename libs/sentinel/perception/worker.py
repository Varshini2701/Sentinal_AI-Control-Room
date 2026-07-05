"""The perception worker: the runnable heart of the fast Perception plane.

Ties a :class:`FrameSource` to the detector and :class:`PerceptionPipeline`, and publishes the
resulting :class:`IntersectionState` as a ``state.updated`` event on the bus (Redis Streams in
production). It owns the failure-recovery policy: a detector or pipeline error on one frame drops
that frame (metered) and re-publishes the last good state rather than crashing the plane.

The current signal phase is injected via ``signal_provider`` -- in the full system the Orchestrator
wires this to the latest ``signal.changed`` event; here it defaults to a safe constant.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from sentinel.contracts.enums import SignalPhase
from sentinel.contracts.events import StateUpdated
from sentinel.contracts.value_objects import IntersectionState
from sentinel.messaging.bus import EventBus
from sentinel.observability.logging import get_logger
from sentinel.observability.metrics import PERCEPTION_DROPPED_FRAMES, PERCEPTION_FPS
from sentinel.perception.detector import ObjectDetector
from sentinel.perception.pipeline import PerceptionPipeline
from sentinel.perception.video import FrameBundle, FrameSource

_log = get_logger("sentinel.perception.worker")

SignalProvider = Callable[[], tuple[SignalPhase, float]]


def _default_signal() -> tuple[SignalPhase, float]:
    return (SignalPhase.NS_GREEN, 0.0)


class PerceptionWorker:
    """Runs the perception loop and publishes intersection state to the event bus."""

    def __init__(
        self,
        *,
        pipeline: PerceptionPipeline,
        detector: ObjectDetector,
        source: FrameSource,
        event_bus: EventBus,
        intersection_id: str,
        source_name: str = "perception",
        signal_provider: SignalProvider | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._pipeline = pipeline
        self._detector = detector
        self._source = source
        self._bus = event_bus
        self._intersection_id = intersection_id
        self._source_name = source_name
        self._signal_provider = signal_provider or _default_signal
        self._clock = clock
        self._last_state: IntersectionState | None = None

    @property
    def last_state(self) -> IntersectionState | None:
        return self._last_state

    async def run(self, *, max_frames: int | None = None) -> int:
        """Process frames until the source is exhausted (or ``max_frames`` reached).

        Returns the number of frames successfully processed. Never raises on per-frame errors.
        """
        processed = 0
        for index, bundle in enumerate(self._source.frames()):
            if max_frames is not None and index >= max_frames:
                break
            start = self._clock()
            state = self._process_bundle(bundle)
            if state is None:
                continue
            await self._publish(state)
            processed += 1
            self._record_fps(start)
        return processed

    # -- internals ---------------------------------------------------------
    def _process_bundle(self, bundle: FrameBundle) -> IntersectionState | None:
        phase, elapsed = self._signal_provider()
        try:
            detections = {d: self._detector.detect(frame) for d, frame in bundle.items()}
            state = self._pipeline.process(
                detections, signal_phase=phase, phase_elapsed_s=elapsed
            )
        except Exception as exc:  # a bad frame must not take down the perception plane
            PERCEPTION_DROPPED_FRAMES.labels(intersection=self._intersection_id).inc()
            _log.warning("frame_dropped", intersection=self._intersection_id, error=str(exc))
            return self._last_state  # re-publish last good state if we have one
        self._last_state = state
        return state

    async def _publish(self, state: IntersectionState) -> None:
        await self._bus.publish(
            StateUpdated(
                source=self._source_name,
                intersection_id=self._intersection_id,
                state=state,
            )
        )

    def _record_fps(self, start: float) -> None:
        elapsed = self._clock() - start
        if elapsed > 0:
            PERCEPTION_FPS.labels(intersection=self._intersection_id).set(1.0 / elapsed)


__all__ = ["PerceptionWorker", "SignalProvider"]
