"""The perception pipeline: detections -> tracks -> movement -> density -> IntersectionState.

This is the pure state-estimation core of the fast Perception plane. It takes the raw detections
for each approach (produced by the detector, one camera per approach), and runs per-approach
tracking, movement analysis and density estimation to assemble the canonical
:class:`IntersectionState` that the Cognition plane consumes.

It is deliberately free of frames, ML and I/O -- those live in the detector and the worker -- so the
whole thing is deterministic and unit-testable with plain :class:`Detection` lists. The current
signal phase is *not* observed from vision; it is supplied by the caller (the worker fills it from
the latest ``signal.changed`` event / state store).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from sentinel.config.settings import PerceptionSettings
from sentinel.contracts.enums import Direction, SignalPhase, SystemMode
from sentinel.contracts.value_objects import Detection, IntersectionState, LaneState
from sentinel.observability.metrics import ACTIVE_TRACKS
from sentinel.perception.density import DensityEstimator
from sentinel.perception.geometry import IntersectionCalibration, LaneCalibration
from sentinel.perception.movement import MovementAnalyzer
from sentinel.perception.tracking import IouTracker, MultiObjectTracker

TrackerFactory = Callable[[], MultiObjectTracker]
AnalyzerFactory = Callable[[], MovementAnalyzer]


class PerceptionPipeline:
    """Assembles per-approach detections into an :class:`IntersectionState`."""

    def __init__(
        self,
        calibration: IntersectionCalibration,
        *,
        settings: PerceptionSettings | None = None,
        tracker_factory: TrackerFactory | None = None,
        analyzer_factory: AnalyzerFactory | None = None,
    ) -> None:
        self._calibration = calibration
        self._settings = settings or PerceptionSettings()
        self._default_dt = 1.0 / self._settings.target_fps
        threshold = self._settings.stopped_speed_threshold

        make_tracker = tracker_factory or (
            lambda: IouTracker(moving_speed_threshold=threshold)
        )
        make_analyzer = analyzer_factory or (
            lambda: MovementAnalyzer(stop_speed_threshold=threshold)
        )
        self._trackers: dict[Direction, MultiObjectTracker] = {
            d: make_tracker() for d in calibration.lanes
        }
        self._analyzers: dict[Direction, MovementAnalyzer] = {
            d: make_analyzer() for d in calibration.lanes
        }
        self._density = DensityEstimator()

    def reset(self) -> None:
        for tracker in self._trackers.values():
            tracker.reset()
        for analyzer in self._analyzers.values():
            analyzer.reset()

    def process(
        self,
        detections_by_direction: Mapping[Direction, list[Detection]],
        *,
        signal_phase: SignalPhase = SignalPhase.NS_GREEN,
        phase_elapsed_s: float = 0.0,
        mode: SystemMode = SystemMode.AI,
        dt: float | None = None,
    ) -> IntersectionState:
        step_dt = dt if dt is not None else self._default_dt
        lanes: dict[Direction, LaneState] = {}
        confidences: list[float] = []
        active_total = 0

        for direction, calib in self._calibration.lanes.items():
            raw = detections_by_direction.get(direction, [])
            kept = self._filter_roi(raw, calib)
            confidences.extend(d.confidence for d in kept)

            tracks = self._trackers[direction].update(kept, step_dt, lane=direction)
            movement = self._analyzers[direction].update(tracks, step_dt)
            lanes[direction] = self._density.estimate(direction, tracks, movement, calib)
            active_total += len(tracks)

        ACTIVE_TRACKS.labels(intersection=self._calibration.intersection_id).set(active_total)
        confidence = sum(confidences) / len(confidences) if confidences else 1.0

        return IntersectionState(
            intersection_id=self._calibration.intersection_id,
            lanes=lanes,
            current_phase=signal_phase,
            phase_elapsed_s=phase_elapsed_s,
            mode=mode,
            perception_confidence=confidence,
        )

    @staticmethod
    def _filter_roi(detections: list[Detection], calib: LaneCalibration) -> list[Detection]:
        return [d for d in detections if calib.contains(d.box.centroid)]


__all__ = ["AnalyzerFactory", "PerceptionPipeline", "TrackerFactory"]
