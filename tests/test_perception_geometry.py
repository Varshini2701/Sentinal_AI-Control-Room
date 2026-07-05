"""Tests for perception camera geometry: ROI membership and queue geometry."""

from __future__ import annotations

from sentinel.contracts.enums import Direction
from sentinel.perception.geometry import (
    LaneCalibration,
    default_calibration,
    distance,
    point_in_polygon,
)

_SQUARE = ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0))


class TestPointInPolygon:
    def test_inside(self) -> None:
        assert point_in_polygon((5.0, 5.0), _SQUARE) is True

    def test_outside(self) -> None:
        assert point_in_polygon((15.0, 5.0), _SQUARE) is False
        assert point_in_polygon((5.0, -1.0), _SQUARE) is False

    def test_triangle(self) -> None:
        triangle = ((0.0, 0.0), (10.0, 0.0), (5.0, 10.0))
        assert point_in_polygon((5.0, 2.0), triangle) is True
        assert point_in_polygon((1.0, 8.0), triangle) is False


class TestLaneCalibration:
    def _calib(self) -> LaneCalibration:
        return LaneCalibration(
            direction=Direction.NORTH,
            roi_polygon=_SQUARE,
            stop_line=(5.0, 0.0),
            metres_per_pixel=0.5,
            approach_length_m=100.0,
        )

    def test_contains(self) -> None:
        calib = self._calib()
        assert calib.contains((5.0, 5.0)) is True
        assert calib.contains((20.0, 20.0)) is False

    def test_queue_length_uses_furthest_stopped(self) -> None:
        calib = self._calib()
        # Two stopped vehicles at 4 px and 8 px from the stop line -> furthest = 8 px * 0.5 = 4 m.
        centroids = [(5.0, 4.0), (5.0, 8.0)]
        assert calib.queue_length_m(centroids) == 4.0

    def test_queue_length_empty(self) -> None:
        assert self._calib().queue_length_m([]) == 0.0

    def test_occupancy_clamped(self) -> None:
        calib = self._calib()
        assert calib.occupancy_pct(50.0) == 50.0
        assert calib.occupancy_pct(500.0) == 100.0


class TestDefaults:
    def test_default_calibration_has_four_lanes(self) -> None:
        calib = default_calibration()
        assert set(calib.lanes) == set(Direction)

    def test_distance(self) -> None:
        assert distance((0.0, 0.0), (3.0, 4.0)) == 5.0
