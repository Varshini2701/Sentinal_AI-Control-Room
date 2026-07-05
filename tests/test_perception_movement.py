"""Tests for movement analysis: hysteresis and stopped-time accrual."""

from __future__ import annotations

import pytest
from tests.conftest import make_track

from sentinel.perception.movement import MovementAnalyzer


class TestClassification:
    def test_stationary_track_is_stopped_and_accrues_wait(self) -> None:
        analyzer = MovementAnalyzer(stop_speed_threshold=1.5)
        track = make_track(1, vx=0.0, vy=0.0)
        info1 = analyzer.update([track], dt=1.0)[1]
        info2 = analyzer.update([track], dt=1.0)[1]
        assert info1.is_moving is False
        assert info1.stopped_s == 1.0
        assert info2.stopped_s == 2.0  # continuous stop accrues

    def test_fast_track_is_moving_with_zero_wait(self) -> None:
        analyzer = MovementAnalyzer(stop_speed_threshold=1.5)
        info = analyzer.update([make_track(1, vx=10.0)], dt=1.0)[1]
        assert info.is_moving is True
        assert info.stopped_s == 0.0

    def test_hysteresis_prevents_flicker(self) -> None:
        analyzer = MovementAnalyzer(stop_speed_threshold=1.5, resume_speed_threshold=3.0)
        # Stop the track, then feed a speed in the hysteresis band (1.5..3.0): stays stopped.
        analyzer.update([make_track(1, vx=0.0)], dt=1.0)
        info = analyzer.update([make_track(1, vx=2.0)], dt=1.0)[1]
        assert info.is_moving is False
        # Above the resume threshold it becomes moving again.
        info = analyzer.update([make_track(1, vx=5.0)], dt=1.0)[1]
        assert info.is_moving is True

    def test_stopped_time_resets_when_moving(self) -> None:
        analyzer = MovementAnalyzer(stop_speed_threshold=1.5)
        analyzer.update([make_track(1, vx=0.0)], dt=1.0)
        analyzer.update([make_track(1, vx=0.0)], dt=1.0)
        info = analyzer.update([make_track(1, vx=10.0)], dt=1.0)[1]
        assert info.stopped_s == 0.0

    def test_absent_track_state_evicted(self) -> None:
        analyzer = MovementAnalyzer()
        analyzer.update([make_track(1, vx=0.0)], dt=1.0)
        analyzer.update([make_track(2, vx=0.0)], dt=1.0)  # track 1 gone
        # Track 1 reappearing is treated as new (assumed moving until re-evaluated).
        infos = analyzer.update([make_track(1, vx=0.0)], dt=1.0)
        assert infos[1].stopped_s == 1.0  # fresh accrual, not continued from before

    def test_invalid_resume_threshold_rejected(self) -> None:
        with pytest.raises(ValueError, match="resume_speed_threshold"):
            MovementAnalyzer(stop_speed_threshold=3.0, resume_speed_threshold=1.0)
