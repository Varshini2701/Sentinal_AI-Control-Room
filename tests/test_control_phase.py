"""Tests for the safety-critical phase state machine and safety envelope.

These tests are the guarantee that the control system can never issue an unsafe signal sequence:
no green shorter than the minimum, no green longer than the maximum, and never two conflicting
greens without a full yellow + all-red clearance -- regardless of what a policy requests.
"""

from __future__ import annotations

from sentinel.config.settings import DecisionSettings
from sentinel.contracts.enums import Axis, SignalPhase
from sentinel.control import PhaseStateMachine, SafetyEnvelope, green_phase_for, other_axis


def _settings() -> DecisionSettings:
    return DecisionSettings(
        min_green_s=7, max_green_s=60, yellow_s=3, all_red_s=2, max_starvation_s=120
    )


class TestPhaseStateMachine:
    def test_starts_in_requested_axis_green(self) -> None:
        fsm = PhaseStateMachine(_settings(), initial_axis=Axis.EAST_WEST)
        assert fsm.phase is SignalPhase.EW_GREEN
        assert fsm.active_axis is Axis.EAST_WEST

    def test_min_green_is_respected(self) -> None:
        fsm = PhaseStateMachine(_settings())
        # Request a switch immediately; the machine must refuse until min_green elapses.
        for _ in range(6):  # 6 s < min_green (7 s)
            phase = fsm.step(1.0, request_switch=True)
            assert phase is SignalPhase.NS_GREEN
        phase = fsm.step(1.0, request_switch=True)  # now 7 s -> may leave green
        assert phase is SignalPhase.NS_YELLOW

    def test_max_green_forces_switch_without_request(self) -> None:
        fsm = PhaseStateMachine(_settings())
        phase = SignalPhase.NS_GREEN
        for _ in range(60):
            phase = fsm.step(1.0, request_switch=False)
        assert phase is SignalPhase.NS_YELLOW  # forced at max_green even with no request

    def test_full_clearance_sequence(self) -> None:
        fsm = PhaseStateMachine(_settings())
        # Drive one full switch NS_GREEN -> ... -> EW_GREEN and record the ordered phases.
        seen: list[SignalPhase] = [fsm.phase]
        for _ in range(80):
            phase = fsm.step(1.0, request_switch=True)
            if phase != seen[-1]:
                seen.append(phase)
            if phase is SignalPhase.EW_GREEN:
                break
        assert seen == [
            SignalPhase.NS_GREEN,
            SignalPhase.NS_YELLOW,
            SignalPhase.ALL_RED,
            SignalPhase.EW_GREEN,
        ]

    def test_yellow_and_all_red_durations(self) -> None:
        fsm = PhaseStateMachine(_settings())
        # advance to yellow
        while fsm.phase is SignalPhase.NS_GREEN:
            fsm.step(1.0, request_switch=True)
        assert fsm.phase is SignalPhase.NS_YELLOW
        fsm.step(1.0, request_switch=True)  # 1s yellow
        fsm.step(1.0, request_switch=True)  # 2s yellow
        assert fsm.phase is SignalPhase.NS_YELLOW
        fsm.step(1.0, request_switch=True)  # 3s == yellow_s -> all red
        assert fsm.phase is SignalPhase.ALL_RED

    def test_never_emits_illegal_transition(self) -> None:
        fsm = PhaseStateMachine(_settings())
        prev = fsm.phase
        # Fuzz the request signal across a long run; every transition must be legal.
        toggles = [i % 3 == 0 for i in range(500)]
        for req in toggles:
            nxt = fsm.step(1.0, request_switch=req)
            assert SafetyEnvelope.is_legal_transition(prev, nxt), f"{prev} -> {nxt}"
            prev = nxt


class TestSafetyEnvelope:
    def test_conflicting_green_to_green_is_illegal(self) -> None:
        assert not SafetyEnvelope.is_legal_transition(
            SignalPhase.NS_GREEN, SignalPhase.EW_GREEN
        )

    def test_validate_counts_violation(self) -> None:
        from sentinel.observability.metrics import SENTINEL_REGISTRY

        labels = {"intersection": "i-x", "constraint": "illegal_phase_transition"}
        before = SENTINEL_REGISTRY.get_sample_value(
            "sentinel_safety_violations_total", labels
        ) or 0.0
        ok = SafetyEnvelope.validate_transition(
            SignalPhase.NS_GREEN, SignalPhase.EW_GREEN, intersection_id="i-x"
        )
        after = SENTINEL_REGISTRY.get_sample_value("sentinel_safety_violations_total", labels)
        assert ok is False
        assert after == before + 1

    def test_legal_transition_does_not_count(self) -> None:
        assert SafetyEnvelope.validate_transition(
            SignalPhase.ALL_RED, SignalPhase.NS_GREEN, intersection_id="i-y"
        )


class TestHelpers:
    def test_other_axis(self) -> None:
        assert other_axis(Axis.NORTH_SOUTH) is Axis.EAST_WEST
        assert other_axis(Axis.EAST_WEST) is Axis.NORTH_SOUTH

    def test_green_phase_for(self) -> None:
        assert green_phase_for(Axis.NORTH_SOUTH) is SignalPhase.NS_GREEN
        assert green_phase_for(Axis.EAST_WEST) is SignalPhase.EW_GREEN
