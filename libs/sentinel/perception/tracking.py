"""Multi-object tracking: stable identities across frames.

A pure-Python IoU/greedy-association tracker with a ByteTrack-style lifecycle (tentative ->
confirmed -> lost). It needs no ML dependency, is deterministic, and produces the velocity estimates
the movement analyzer relies on. Real deployments may swap in a heavier tracker behind the same
:class:`MultiObjectTracker` port; the association and lifecycle logic here is what the tests pin.

One tracker instance is used per approach camera, so every track it emits belongs to that approach.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

from sentinel.contracts.enums import Direction, VehicleClass
from sentinel.contracts.value_objects import BoundingBox, Detection, Track, Velocity


def iou(a: BoundingBox, b: BoundingBox) -> float:
    """Intersection-over-union of two boxes; 0.0 when disjoint."""
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    intersection = iw * ih
    if intersection <= 0.0:
        return 0.0
    union = a.area + b.area - intersection
    return intersection / union if union > 0 else 0.0


class MultiObjectTracker(abc.ABC):
    """Associates detections across frames into tracks with stable ids."""

    @abc.abstractmethod
    def update(self, detections: list[Detection], dt: float, *, lane: Direction) -> list[Track]:
        """Ingest one frame's detections and return the currently confirmed tracks."""

    @abc.abstractmethod
    def reset(self) -> None:
        """Drop all tracks and identity counters."""


@dataclass(slots=True)
class _TrackRecord:
    track_id: int
    box: BoundingBox
    vehicle_class: VehicleClass
    centroid: tuple[float, float]
    velocity: Velocity
    hits: int = 1
    misses: int = 0
    age: int = 0


class IouTracker(MultiObjectTracker):
    """Greedy IoU tracker with a confirmation/eviction lifecycle.

    Args:
        iou_threshold: Minimum IoU for a detection to match an existing track.
        max_misses: Consecutive frames a track may go unmatched before eviction.
        min_hits: Matches required before a track is emitted as confirmed.
        moving_speed_threshold: Pixel speed (px/s) above which a track is provisionally moving.
    """

    def __init__(
        self,
        *,
        iou_threshold: float = 0.3,
        max_misses: int = 5,
        min_hits: int = 2,
        moving_speed_threshold: float = 1.5,
    ) -> None:
        self._iou_threshold = iou_threshold
        self._max_misses = max_misses
        self._min_hits = min_hits
        self._moving_threshold = moving_speed_threshold
        self._tracks: dict[int, _TrackRecord] = {}
        self._next_id = 1

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1

    @property
    def active_count(self) -> int:
        return len(self._tracks)

    def update(self, detections: list[Detection], dt: float, *, lane: Direction) -> list[Track]:
        matches, unmatched_dets, unmatched_tracks = self._associate(detections)

        for track_id, det in matches:
            self._update_track(self._tracks[track_id], det, dt)
        for det in unmatched_dets:
            self._create_track(det)
        for track_id in unmatched_tracks:
            record = self._tracks[track_id]
            record.misses += 1
            record.age += 1
        self._evict()

        return [
            self._to_track(record, lane)
            for record in self._tracks.values()
            if record.hits >= self._min_hits and record.misses == 0
        ]

    # -- association -------------------------------------------------------
    def _associate(
        self, detections: list[Detection]
    ) -> tuple[list[tuple[int, Detection]], list[Detection], list[int]]:
        pairs: list[tuple[float, int, int]] = []  # (iou, track_id, det_index)
        for track_id, record in self._tracks.items():
            for di, det in enumerate(detections):
                score = iou(record.box, det.box)
                if score >= self._iou_threshold:
                    pairs.append((score, track_id, di))
        pairs.sort(reverse=True)

        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()
        matches: list[tuple[int, Detection]] = []
        for _score, track_id, di in pairs:
            if track_id in matched_tracks or di in matched_dets:
                continue
            matched_tracks.add(track_id)
            matched_dets.add(di)
            matches.append((track_id, detections[di]))

        unmatched_dets = [d for i, d in enumerate(detections) if i not in matched_dets]
        unmatched_tracks = [t for t in self._tracks if t not in matched_tracks]
        return matches, unmatched_dets, unmatched_tracks

    # -- lifecycle ---------------------------------------------------------
    def _create_track(self, det: Detection) -> None:
        self._tracks[self._next_id] = _TrackRecord(
            track_id=self._next_id,
            box=det.box,
            vehicle_class=det.vehicle_class,
            centroid=det.box.centroid,
            velocity=Velocity(vx=0.0, vy=0.0),
        )
        self._next_id += 1

    def _update_track(self, record: _TrackRecord, det: Detection, dt: float) -> None:
        new_centroid = det.box.centroid
        if dt > 0:
            record.velocity = Velocity(
                vx=(new_centroid[0] - record.centroid[0]) / dt,
                vy=(new_centroid[1] - record.centroid[1]) / dt,
            )
        record.centroid = new_centroid
        record.box = det.box
        record.vehicle_class = det.vehicle_class
        record.hits += 1
        record.misses = 0
        record.age += 1

    def _evict(self) -> None:
        dead = [tid for tid, rec in self._tracks.items() if rec.misses > self._max_misses]
        for tid in dead:
            del self._tracks[tid]

    def _to_track(self, record: _TrackRecord, lane: Direction) -> Track:
        return Track(
            track_id=record.track_id,
            vehicle_class=record.vehicle_class,
            box=record.box,
            velocity=record.velocity,
            lane=lane,
            is_moving=record.velocity.speed >= self._moving_threshold,
            age_frames=record.age,
        )


__all__ = ["IouTracker", "MultiObjectTracker", "iou"]
