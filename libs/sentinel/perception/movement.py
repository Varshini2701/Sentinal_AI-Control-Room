"""Movement analysis: classify each track as moving or stopped, and accrue waiting time.

Instantaneous speed is noisy, so a raw ``speed < threshold`` test flickers a vehicle between
moving and stopped every frame. This analyzer applies **hysteresis** (separate enter-stopped and
resume-moving thresholds) and tracks how long each vehicle has been continuously stopped -- the raw
material for per-lane waiting time and for the "no movement for N seconds" switch rule.

One analyzer instance is used per approach, mirroring the tracker.
"""

from __future__ import annotations

from dataclasses import dataclass

from sentinel.contracts.value_objects import Track


@dataclass(frozen=True, slots=True)
class MovementInfo:
    """Per-track movement classification for one tick."""

    track_id: int
    is_moving: bool
    stopped_s: float
    speed: float


class MovementAnalyzer:
    """Hysteretic moving/stopped classifier with per-track stopped-time accrual.

    Args:
        stop_speed_threshold: A moving track becomes *stopped* when its speed drops below this.
        resume_speed_threshold: A stopped track becomes *moving* only above this (higher) speed.
            Defaults to twice the stop threshold; the gap is the hysteresis band.
    """

    def __init__(
        self,
        *,
        stop_speed_threshold: float = 1.5,
        resume_speed_threshold: float | None = None,
    ) -> None:
        if resume_speed_threshold is not None and resume_speed_threshold < stop_speed_threshold:
            raise ValueError("resume_speed_threshold must be >= stop_speed_threshold")
        self._stop_threshold = stop_speed_threshold
        self._resume_threshold = resume_speed_threshold or stop_speed_threshold * 2.0
        self._is_moving: dict[int, bool] = {}
        self._stopped_s: dict[int, float] = {}

    def reset(self) -> None:
        self._is_moving.clear()
        self._stopped_s.clear()

    def update(self, tracks: list[Track], dt: float) -> dict[int, MovementInfo]:
        """Classify the current tracks, evicting state for tracks no longer present."""
        present = {t.track_id for t in tracks}
        self._evict(present)

        infos: dict[int, MovementInfo] = {}
        for track in tracks:
            speed = track.velocity.speed
            moving = self._classify(track.track_id, speed)
            stopped_s = self._accrue(track.track_id, moving, dt)
            infos[track.track_id] = MovementInfo(
                track_id=track.track_id, is_moving=moving, stopped_s=stopped_s, speed=speed
            )
        return infos

    # -- internals ---------------------------------------------------------
    def _classify(self, track_id: int, speed: float) -> bool:
        was_moving = self._is_moving.get(track_id, True)  # new tracks assumed moving
        threshold = self._stop_threshold if was_moving else self._resume_threshold
        moving = speed >= threshold
        self._is_moving[track_id] = moving
        return moving

    def _accrue(self, track_id: int, moving: bool, dt: float) -> float:
        stopped_s = 0.0 if moving else self._stopped_s.get(track_id, 0.0) + dt
        self._stopped_s[track_id] = stopped_s
        return stopped_s

    def _evict(self, present: set[int]) -> None:
        for track_id in list(self._is_moving):
            if track_id not in present:
                self._is_moving.pop(track_id, None)
                self._stopped_s.pop(track_id, None)


__all__ = ["MovementAnalyzer", "MovementInfo"]
