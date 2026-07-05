"""Tests for the density estimator: LaneState assembly from tracks + movement."""

from __future__ import annotations

from tests.conftest import make_box, make_track

from sentinel.contracts.enums import DensityLevel, Direction, VehicleClass
from sentinel.perception.density import DensityEstimator
from sentinel.perception.geometry import LaneCalibration
from sentinel.perception.movement import MovementInfo

_CALIB = LaneCalibration(
    direction=Direction.NORTH,
    roi_polygon=((0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)),
    stop_line=(0.0, 0.0),
    metres_per_pixel=1.0,
    approach_length_m=100.0,
)


def _info(track_id: int, *, is_moving: bool, stopped_s: float) -> MovementInfo:
    return MovementInfo(track_id=track_id, is_moving=is_moving, stopped_s=stopped_s, speed=0.0)


class TestDensityEstimator:
    def test_counts_moving_and_stopped(self) -> None:
        tracks = [
            make_track(1, box=make_box(2, 4, 10, 12)),  # centroid (6, 8), dist 10
            make_track(2, box=make_box(0, 2, 6, 6)),  # centroid (3, 4), dist 5
            make_track(3, box=make_box(0, 0, 6, 8)),  # moving
        ]
        movement = {
            1: _info(1, is_moving=False, stopped_s=3.0),
            2: _info(2, is_moving=False, stopped_s=5.0),
            3: _info(3, is_moving=True, stopped_s=0.0),
        }
        lane = DensityEstimator().estimate(Direction.NORTH, tracks, movement, _CALIB)
        assert lane.vehicle_count == 3
        assert lane.moving_count == 1
        assert lane.stopped_count == 2
        assert lane.queue_length_m == 10.0  # furthest stopped vehicle
        assert lane.avg_wait_s == 4.0  # mean of 3.0 and 5.0
        assert lane.occupancy_pct == 10.0
        assert lane.density is DensityLevel.FREE

    def test_emergency_flag(self) -> None:
        tracks = [make_track(1, vehicle_class=VehicleClass.AMBULANCE)]
        movement = {1: _info(1, is_moving=True, stopped_s=0.0)}
        lane = DensityEstimator().estimate(Direction.NORTH, tracks, movement, _CALIB)
        assert lane.has_emergency_vehicle is True

    def test_pedestrian_excluded_from_vehicle_count(self) -> None:
        tracks = [
            make_track(1, vehicle_class=VehicleClass.CAR),
            make_track(2, vehicle_class=VehicleClass.PEDESTRIAN),
        ]
        movement = {
            1: _info(1, is_moving=True, stopped_s=0.0),
            2: _info(2, is_moving=True, stopped_s=0.0),
        }
        lane = DensityEstimator().estimate(Direction.NORTH, tracks, movement, _CALIB)
        assert lane.vehicle_count == 1  # pedestrian not a vehicle
        assert lane.pedestrian_waiting is True

    def test_empty_lane(self) -> None:
        lane = DensityEstimator().estimate(Direction.NORTH, [], {}, _CALIB)
        assert lane.vehicle_count == 0
        assert lane.queue_length_m == 0.0
        assert lane.avg_wait_s == 0.0
        assert lane.density is DensityLevel.FREE
