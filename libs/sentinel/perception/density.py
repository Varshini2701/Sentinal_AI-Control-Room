"""Density / occupancy estimation: turn tracks into a :class:`LaneState`.

Given the confirmed tracks on one approach plus their movement classification, this produces the
decision-relevant lane summary: vehicle counts, queue length (metres), occupancy, average waiting
time, density level, and the emergency-vehicle / pedestrian flags. Pedestrians are counted for the
``pedestrian_waiting`` flag but excluded from vehicle counts and queue geometry.
"""

from __future__ import annotations

from sentinel.contracts.enums import DensityLevel, Direction, VehicleClass
from sentinel.contracts.value_objects import LaneState, Track
from sentinel.perception.geometry import LaneCalibration
from sentinel.perception.movement import MovementInfo


class DensityEstimator:
    """Aggregates tracks + movement into a :class:`LaneState` for one approach (stateless)."""

    def estimate(
        self,
        direction: Direction,
        tracks: list[Track],
        movement: dict[int, MovementInfo],
        calibration: LaneCalibration,
    ) -> LaneState:
        vehicles = [t for t in tracks if t.vehicle_class.is_vehicle]
        pedestrian_waiting = any(t.vehicle_class is VehicleClass.PEDESTRIAN for t in tracks)
        has_emergency = any(t.vehicle_class.is_emergency for t in vehicles)

        stopped_tracks = [t for t in vehicles if not self._is_moving(movement, t.track_id)]
        moving_count = len(vehicles) - len(stopped_tracks)

        queue_length_m = calibration.queue_length_m([t.box.centroid for t in stopped_tracks])
        occupancy = calibration.occupancy_pct(queue_length_m)
        avg_wait_s = self._avg_wait(movement, stopped_tracks)

        return LaneState(
            direction=direction,
            vehicle_count=len(vehicles),
            moving_count=moving_count,
            stopped_count=len(stopped_tracks),
            queue_length_m=queue_length_m,
            avg_wait_s=avg_wait_s,
            occupancy_pct=occupancy,
            density=DensityLevel.from_occupancy(occupancy),
            has_emergency_vehicle=has_emergency,
            pedestrian_waiting=pedestrian_waiting,
        )

    @staticmethod
    def _is_moving(movement: dict[int, MovementInfo], track_id: int) -> bool:
        info = movement.get(track_id)
        return info.is_moving if info is not None else True

    @staticmethod
    def _avg_wait(movement: dict[int, MovementInfo], stopped_tracks: list[Track]) -> float:
        waits = [
            movement[t.track_id].stopped_s
            for t in stopped_tracks
            if t.track_id in movement
        ]
        return sum(waits) / len(waits) if waits else 0.0


__all__ = ["DensityEstimator"]
