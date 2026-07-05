"""Terminal demo of the Perception plane (Module 3).

Synthesises detections for two approaches -- a North queue that builds up and stops at the stop
line, and an East vehicle driving through -- and feeds them through the real
:class:`PerceptionPipeline`. It prints the :class:`IntersectionState` the pipeline produces each
few frames, showing detection -> tracking -> movement -> density -> state end to end, with no ML or
video needed. This is the exact contract the control loop (Module 2) consumes.

    python scripts/demo_perception.py
"""

from __future__ import annotations

from sentinel.contracts.enums import Direction, VehicleClass
from sentinel.contracts.value_objects import BoundingBox, Detection
from sentinel.perception.geometry import default_calibration
from sentinel.perception.pipeline import PerceptionPipeline

_STOP_LINE_Y = 480.0
_LANE_X = 320.0
_CAR_SPACING = 28.0


def _car(cx: float, cy: float, vehicle_class: VehicleClass = VehicleClass.CAR) -> Detection:
    return Detection(
        vehicle_class=vehicle_class,
        confidence=0.85,
        box=BoundingBox(x1=cx - 12, y1=cy - 16, x2=cx + 12, y2=cy + 16),
    )


def _north_queue(num_cars: int) -> list[Detection]:
    """A column of stationary cars backed up just inside the North stop line."""
    return [_car(_LANE_X, _STOP_LINE_Y - 8.0 - k * _CAR_SPACING) for k in range(num_cars)]


def _east_mover(frame: int) -> list[Detection]:
    """A single car crossing the East approach (advances ~8 px/frame -> stays tracked & moving)."""
    return [_car(60.0 + frame * 8.0, 240.0)]


def _render(frame: int, state) -> str:  # type: ignore[no-untyped-def]
    lines = [f"\n[ frame {frame:>2} ]  perception_confidence = {state.perception_confidence:.2f}"]
    for direction in (Direction.NORTH, Direction.EAST):
        lane = state.lanes[direction]
        lines.append(
            f"  {direction.value:<5}: {lane.vehicle_count:>2} veh  "
            f"(moving {lane.moving_count}, stopped {lane.stopped_count})  "
            f"queue {lane.queue_length_m:>5.1f} m  wait {lane.avg_wait_s:>4.1f}s  "
            f"[{lane.density.value}]"
        )
    return "\n".join(lines)


def main() -> None:
    pipeline = PerceptionPipeline(default_calibration())
    # Frame plan: North queue grows 1 -> 6 cars over the run; East car drives through.
    north_counts = [1, 2, 3, 4, 5, 6, 6, 6]

    print("=" * 64)
    print("  SENTINEL AI - Perception plane demo")
    print("  Synthetic detections -> tracking -> movement -> IntersectionState")
    print("=" * 64)

    for frame, north_n in enumerate(north_counts):
        detections = {
            Direction.NORTH: _north_queue(north_n),
            Direction.EAST: _east_mover(frame),
        }
        state = pipeline.process(detections)
        if frame >= 1:  # frame 0 tracks are still tentative (min_hits=2)
            print(_render(frame, state))

    print("\n" + "=" * 64)
    print("  The North queue length grows as vehicles stack up and stop;")
    print("  the East vehicle stays 'moving'. This IntersectionState is exactly")
    print("  what the Decision Agent consumes to control the signals.")
    print("=" * 64 + "\n")


if __name__ == "__main__":
    main()
