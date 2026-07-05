"""Camera calibration and image geometry for the perception plane.

Each approach camera is calibrated with:

* a **region of interest** polygon that isolates that approach's inbound lanes from cross-traffic
  and the opposite carriageway in the same frame,
* a **stop line** reference point at the intersection end of the approach, and
* a **metres-per-pixel** scale along the approach, used to turn the pixel distance from the stop
  line to the furthest queued vehicle into a queue length in metres.

All maths here is pure Python (ray-casting point-in-polygon, Euclidean distance) so lane assignment
and queue geometry are fully unit-testable without any image or ML dependency.
"""

from __future__ import annotations

from pydantic import Field, model_validator

from sentinel.contracts.base import FrozenModel
from sentinel.contracts.enums import Direction

Point = tuple[float, float]


def point_in_polygon(point: Point, polygon: tuple[Point, ...]) -> bool:
    """Return whether ``point`` lies inside ``polygon`` using the ray-casting algorithm.

    Points exactly on an edge are treated as inside for the horizontal-ray crossing test; this is
    adequate for ROI membership where sub-pixel edge cases do not matter.
    """
    x, y = point
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        # Does a horizontal ray from (x, y) going +x cross this edge?
        if (y1 > y) != (y2 > y):
            x_cross = x1 + (y - y1) / (y2 - y1) * (x2 - x1)
            if x <= x_cross:
                inside = not inside
    return inside


def distance(a: Point, b: Point) -> float:
    """Euclidean distance between two image points."""
    return float(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5)


class LaneCalibration(FrozenModel):
    """Calibration for a single approach camera."""

    direction: Direction
    roi_polygon: tuple[Point, ...] = Field(min_length=3)
    stop_line: Point
    metres_per_pixel: float = Field(gt=0.0)
    approach_length_m: float = Field(default=200.0, gt=0.0)

    def contains(self, point: Point) -> bool:
        """Whether an image point falls inside this approach's region of interest."""
        return point_in_polygon(point, self.roi_polygon)

    def queue_length_m(self, stopped_centroids: list[Point]) -> float:
        """Queue length: distance from the stop line to the furthest stopped vehicle, in metres."""
        if not stopped_centroids:
            return 0.0
        furthest_px = max(distance(self.stop_line, c) for c in stopped_centroids)
        return furthest_px * self.metres_per_pixel

    def occupancy_pct(self, queue_length_m: float) -> float:
        """Fraction of the approach occupied by queue, clamped to [0, 100]."""
        return min(100.0, queue_length_m / self.approach_length_m * 100.0)


class IntersectionCalibration(FrozenModel):
    """Calibration for a full four-approach intersection."""

    intersection_id: str = Field(min_length=1)
    lanes: dict[Direction, LaneCalibration]

    @model_validator(mode="after")
    def _check_directions(self) -> IntersectionCalibration:
        for direction, lane in self.lanes.items():
            if lane.direction != direction:
                raise ValueError(f"lane keyed under {direction} declares {lane.direction}")
        return self

    def get(self, direction: Direction) -> LaneCalibration | None:
        return self.lanes.get(direction)


def default_calibration(
    intersection_id: str = "intersection-1",
    *,
    frame_width: int = 640,
    frame_height: int = 480,
    metres_per_pixel: float = 0.25,
) -> IntersectionCalibration:
    """Build a simple, symmetric calibration whose ROI is the whole frame per approach.

    Useful for tests, the demo and single-camera setups. Real deployments replace this with
    surveyed polygons and stop lines per camera.
    """
    w, h = float(frame_width), float(frame_height)
    full_frame: tuple[Point, ...] = ((0.0, 0.0), (w, 0.0), (w, h), (0.0, h))
    # Stop line at the bottom-centre of each approach frame (nearest the intersection).
    stop_line: Point = (w / 2.0, h)
    return IntersectionCalibration(
        intersection_id=intersection_id,
        lanes={
            d: LaneCalibration(
                direction=d,
                roi_polygon=full_frame,
                stop_line=stop_line,
                metres_per_pixel=metres_per_pixel,
            )
            for d in Direction
        },
    )


__all__ = [
    "IntersectionCalibration",
    "LaneCalibration",
    "Point",
    "default_calibration",
    "distance",
    "point_in_polygon",
]
