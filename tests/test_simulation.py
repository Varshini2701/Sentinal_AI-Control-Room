"""Tests for the traffic simulation environment, controllers and benchmark harness.

The headline test -- :meth:`TestHarness.test_adaptive_beats_fixed_timer` -- is the project thesis
in executable form: on identical traffic, the adaptive controller must deliver lower average wait
than a fixed timer.
"""

from __future__ import annotations

from sentinel.config.settings import DecisionSettings
from sentinel.contracts.enums import DecisionAction, Direction, SignalPhase
from sentinel.contracts.value_objects import SignalCommand
from sentinel.simulation import (
    AdaptiveController,
    AnalyticalTrafficEnvironment,
    FixedTimerController,
    SimConfig,
    asymmetric_demand,
    run_comparison,
    run_controller,
    symmetric_demand,
)
from sentinel.simulation.config import EmergencyEvent, LaneDemand


def _hold(phase: SignalPhase, intersection_id: str = "intersection-1") -> SignalCommand:
    return SignalCommand(
        intersection_id=intersection_id,
        action=DecisionAction.KEEP_GREEN,
        target_phase=phase,
        duration_s=1.0,
        reason_code="test_hold",
    )


def _run(controller, config):  # type: ignore[no-untyped-def]
    """Run a controller against a fresh analytical environment for the scenario."""
    return run_controller(lambda: AnalyticalTrafficEnvironment(config), controller, config)


class TestAnalyticalEnvironment:
    def test_timestamp_advances_by_simulated_dt(self) -> None:
        # Guards against regressing to wall-clock timestamps: consumers (Decision Agent fairness,
        # Prediction Agent trend fitting) rely on IntersectionState.timestamp reflecting simulated
        # time, not real time -- a fast, sleep-free loop would otherwise report ~0s elapsed.
        config = SimConfig(horizon_s=10, dt_s=1.0, demand=symmetric_demand(0.1))
        env = AnalyticalTrafficEnvironment(config)
        first = env.reset()
        second = env.step(_hold(SignalPhase.NS_GREEN))
        third = env.step(_hold(SignalPhase.NS_GREEN))
        assert (second.timestamp - first.timestamp).total_seconds() == 1.0
        assert (third.timestamp - second.timestamp).total_seconds() == 1.0

    def test_deterministic_for_same_seed(self) -> None:
        config = SimConfig(horizon_s=300, seed=7, demand=symmetric_demand(0.15))
        a = _run(FixedTimerController(DecisionSettings()), config)
        b = _run(FixedTimerController(DecisionSettings()), config)
        assert a.total_arrived == b.total_arrived
        assert a.total_delay_veh_s == b.total_delay_veh_s
        assert a.total_served == b.total_served

    def test_served_never_exceeds_arrived(self) -> None:
        config = SimConfig(horizon_s=600, demand=symmetric_demand(0.2))
        summary = _run(AdaptiveController(DecisionSettings()), config)
        assert summary.total_served <= summary.total_arrived
        for lane in summary.lanes.values():
            assert lane.served <= lane.arrived
            assert lane.max_queue_veh >= 0

    def test_no_discharge_without_green(self) -> None:
        # Demand only on East (EW axis); hold NS green forever -> East never discharges.
        config = SimConfig(
            horizon_s=120,
            demand={
                Direction.EAST: LaneDemand(direction=Direction.EAST, arrival_rate_vps=0.3),
                Direction.NORTH: LaneDemand(direction=Direction.NORTH, arrival_rate_vps=0.0),
            },
        )
        env = AnalyticalTrafficEnvironment(config)
        env.reset()
        for _ in range(config.total_steps):
            env.step(_hold(SignalPhase.NS_GREEN))
        summary = env.metrics()
        assert summary.lanes[Direction.EAST].served == 0
        assert summary.lanes[Direction.EAST].arrived > 0
        assert summary.lanes[Direction.EAST].total_delay_veh_s > 0

    def test_emergency_flag_surfaces_in_state(self) -> None:
        config = SimConfig(
            horizon_s=30,
            demand=symmetric_demand(0.1),
            emergencies=(EmergencyEvent(direction=Direction.WEST, start_s=5, duration_s=10),),
        )
        env = AnalyticalTrafficEnvironment(config)
        env.reset()
        emergency_seen = False
        for _ in range(config.total_steps):
            state = env.step(_hold(SignalPhase.NS_GREEN))
            if state.lanes[Direction.WEST].has_emergency_vehicle:
                emergency_seen = True
        assert emergency_seen

    def test_green_axis_clears_queue(self) -> None:
        config = SimConfig(horizon_s=200, demand=symmetric_demand(0.1))
        env = AnalyticalTrafficEnvironment(config)
        env.reset()
        for _ in range(config.total_steps):
            state = env.step(_hold(SignalPhase.NS_GREEN))
        # North/South served continuously; their queues stay bounded and small.
        assert state.lanes[Direction.NORTH].vehicle_count <= 5


class TestControllers:
    def test_fixed_timer_serves_both_axes(self) -> None:
        config = SimConfig(horizon_s=600, demand=symmetric_demand(0.15))
        summary = _run(FixedTimerController(DecisionSettings(), green_s=25), config)
        assert summary.lanes[Direction.NORTH].served > 0
        assert summary.lanes[Direction.EAST].served > 0

    def test_adaptive_prioritises_busy_axis(self) -> None:
        config = SimConfig(horizon_s=900, demand=asymmetric_demand(0.22, 0.04))
        summary = _run(AdaptiveController(DecisionSettings()), config)
        ns = summary.lanes
        ns_served = ns[Direction.NORTH].served + ns[Direction.SOUTH].served
        ew_served = ns[Direction.EAST].served + ns[Direction.WEST].served
        assert ns_served > ew_served  # more capacity given to the busier axis

    def test_adaptive_preempts_for_emergency(self) -> None:
        # Heavy NS, an emergency parked on East. The controller must move green to the EW axis.
        settings = DecisionSettings(min_green_s=5, yellow_s=3, all_red_s=2)
        config = SimConfig(
            horizon_s=60,
            demand=asymmetric_demand(0.25, 0.02),
            emergencies=(EmergencyEvent(direction=Direction.EAST, start_s=8, duration_s=40),),
        )
        env = AnalyticalTrafficEnvironment(config)
        controller = AdaptiveController(settings)
        state = env.reset()
        reached_ew_green = False
        for _ in range(config.total_steps):
            command = controller.decide(state, config.dt_s)
            state = env.step(command)
            if state.current_phase is SignalPhase.EW_GREEN:
                reached_ew_green = True
        assert reached_ew_green


class TestHarness:
    def test_adaptive_beats_fixed_timer(self) -> None:
        settings = DecisionSettings()
        config = SimConfig(horizon_s=1500, seed=42, demand=asymmetric_demand(0.22, 0.05))
        result = run_comparison(
            lambda: AnalyticalTrafficEnvironment(config),
            [FixedTimerController(settings, green_s=30), AdaptiveController(settings)],
            config,
            baseline="fixed_timer",
        )
        fixed = result.summaries["fixed_timer"]
        adaptive = result.summaries["adaptive"]

        assert adaptive.avg_delay_s < fixed.avg_delay_s
        assert adaptive.clearance_rate >= fixed.clearance_rate
        assert result.wait_reduction_pct("adaptive") > 0.0

    def test_comparison_requires_baseline_present(self) -> None:
        import pytest

        config = SimConfig(horizon_s=60, demand=symmetric_demand(0.1))
        with pytest.raises(ValueError, match="baseline"):
            run_comparison(
                lambda: AnalyticalTrafficEnvironment(config),
                [AdaptiveController(DecisionSettings())],
                config,
                baseline="fixed_timer",
            )

    def test_wait_reduction_sign(self) -> None:
        from sentinel.simulation.kpi import ComparisonResult, KpiSummary

        def summary(name: str, delay_per_veh: float) -> KpiSummary:
            return KpiSummary(
                controller=name,
                sim_duration_s=100.0,
                total_arrived=100,
                total_served=100,
                total_delay_veh_s=delay_per_veh * 100,
                max_queue_veh=5,
                lanes={},
            )

        result = ComparisonResult(
            baseline="fixed_timer",
            summaries={
                "fixed_timer": summary("fixed_timer", 20.0),
                "adaptive": summary("adaptive", 15.0),
            },
        )
        assert result.wait_reduction_pct("adaptive") == 25.0  # 20 -> 15 is a 25% cut
