"""Tests for the perception pipeline: detections -> IntersectionState."""

from __future__ import annotations

from tests.conftest import make_box, make_detection

from sentinel.contracts.enums import Direction, SignalPhase
from sentinel.perception.geometry import (
    IntersectionCalibration,
    LaneCalibration,
    default_calibration,
)
from sentinel.perception.pipeline import PerceptionPipeline


class TestPipeline:
    def test_confirmed_vehicle_appears_in_state(self) -> None:
        pipeline = PerceptionPipeline(default_calibration())
        det = make_detection(box=make_box(300, 200, 340, 260), confidence=0.9)
        detections = {Direction.NORTH: [det]}
        # First tick: track tentative; second tick: confirmed (min_hits=2) and stationary.
        pipeline.process(detections, signal_phase=SignalPhase.NS_GREEN)
        state = pipeline.process(detections, signal_phase=SignalPhase.NS_GREEN)

        north = state.lanes[Direction.NORTH]
        assert north.vehicle_count == 1
        assert north.stopped_count == 1  # not moving between identical frames
        assert north.moving_count == 0
        assert state.perception_confidence == 0.9
        assert state.current_phase is SignalPhase.NS_GREEN

    def test_empty_input_gives_full_confidence_and_zero_counts(self) -> None:
        pipeline = PerceptionPipeline(default_calibration())
        state = pipeline.process({})
        assert state.total_vehicles == 0
        assert state.perception_confidence == 1.0
        assert set(state.lanes) == set(Direction)

    def test_roi_filters_out_of_region_detections(self) -> None:
        calib = IntersectionCalibration(
            intersection_id="i-roi",
            lanes={
                Direction.NORTH: LaneCalibration(
                    direction=Direction.NORTH,
                    roi_polygon=((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)),
                    stop_line=(5.0, 0.0),
                    metres_per_pixel=1.0,
                )
            },
        )
        pipeline = PerceptionPipeline(calib, tracker_factory=None)
        inside = make_detection(box=make_box(2, 2, 6, 6))  # centroid (4, 4) inside ROI
        outside = make_detection(box=make_box(50, 50, 60, 60))  # centroid (55, 55) outside
        detections = {Direction.NORTH: [inside, outside]}
        pipeline.process(detections)
        state = pipeline.process(detections)
        assert state.lanes[Direction.NORTH].vehicle_count == 1  # only the in-ROI vehicle

    def test_reset_clears_tracks(self) -> None:
        pipeline = PerceptionPipeline(default_calibration())
        det = make_detection(box=make_box(300, 200, 340, 260))
        pipeline.process({Direction.NORTH: [det]})
        pipeline.process({Direction.NORTH: [det]})
        pipeline.reset()
        state = pipeline.process({Direction.NORTH: [det]})
        # After reset the track is tentative again -> not yet confirmed.
        assert state.lanes[Direction.NORTH].vehicle_count == 0
