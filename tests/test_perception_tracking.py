"""Tests for the IoU tracker: association, identity stability, velocity, lifecycle."""

from __future__ import annotations

from tests.conftest import make_box, make_detection

from sentinel.contracts.enums import Direction, VehicleClass
from sentinel.perception.tracking import IouTracker, iou


class TestIou:
    def test_identical_boxes(self) -> None:
        box = make_box(0, 0, 10, 10)
        assert iou(box, box) == 1.0

    def test_disjoint_boxes(self) -> None:
        assert iou(make_box(0, 0, 10, 10), make_box(100, 100, 110, 110)) == 0.0

    def test_partial_overlap(self) -> None:
        # Two 10x10 boxes overlapping in a 5x10 region: inter=50, union=150 -> 1/3.
        score = iou(make_box(0, 0, 10, 10), make_box(5, 0, 15, 10))
        assert abs(score - (50 / 150)) < 1e-9


class TestLifecycle:
    def test_confirms_after_min_hits(self) -> None:
        tracker = IouTracker(min_hits=2)
        det = make_detection(box=make_box(0, 0, 10, 10))
        assert tracker.update([det], 1.0, lane=Direction.NORTH) == []  # tentative
        tracks = tracker.update([det], 1.0, lane=Direction.NORTH)
        assert len(tracks) == 1
        assert tracks[0].track_id == 1
        assert tracks[0].lane is Direction.NORTH

    def test_identity_stable_across_motion(self) -> None:
        tracker = IouTracker(min_hits=2)
        # A box drifting right by 4 px/frame keeps enough overlap to stay one track.
        boxes = [make_box(x, 0, x + 10, 10) for x in (0, 4, 8, 12)]
        ids = set()
        for box in boxes:
            tracks = tracker.update([make_detection(box=box)], 1.0, lane=Direction.EAST)
            ids.update(t.track_id for t in tracks)
        assert ids == {1}

    def test_velocity_estimated(self) -> None:
        tracker = IouTracker(min_hits=2)
        tracker.update([make_detection(box=make_box(0, 0, 10, 10))], 1.0, lane=Direction.NORTH)
        # Move 4 px/frame: boxes still overlap (IoU >= 0.3) so it stays the same track.
        tracks = tracker.update(
            [make_detection(box=make_box(4, 0, 14, 10))], 1.0, lane=Direction.NORTH
        )
        # centroid moved from x=5 to x=9 over dt=1 -> vx = 4; speed >= threshold -> moving.
        assert tracks[0].velocity.vx == 4.0
        assert tracks[0].is_moving is True

    def test_distinct_objects_get_distinct_ids(self) -> None:
        tracker = IouTracker(min_hits=1)
        tracks = tracker.update(
            [
                make_detection(box=make_box(0, 0, 10, 10)),
                make_detection(box=make_box(100, 100, 110, 110)),
            ],
            1.0,
            lane=Direction.NORTH,
        )
        assert {t.track_id for t in tracks} == {1, 2}

    def test_lost_track_is_evicted(self) -> None:
        tracker = IouTracker(min_hits=1, max_misses=2)
        det = make_detection(box=make_box(0, 0, 10, 10))
        tracker.update([det], 1.0, lane=Direction.NORTH)
        assert tracker.active_count == 1
        for _ in range(4):  # no detections -> misses accumulate past max_misses
            tracker.update([], 1.0, lane=Direction.NORTH)
        assert tracker.active_count == 0

    def test_reset_clears_ids(self) -> None:
        tracker = IouTracker(min_hits=1)
        tracker.update([make_detection()], 1.0, lane=Direction.NORTH)
        tracker.reset()
        assert tracker.active_count == 0
        tracks = tracker.update([make_detection()], 1.0, lane=Direction.NORTH)
        assert tracks[0].track_id == 1  # id counter reset

    def test_vehicle_class_carried(self) -> None:
        tracker = IouTracker(min_hits=1)
        tracks = tracker.update(
            [make_detection(vehicle_class=VehicleClass.BUS)], 1.0, lane=Direction.NORTH
        )
        assert tracks[0].vehicle_class is VehicleClass.BUS
