"""Tests for the fleet composition root: every agent wired to one bus, closing the loop."""

from __future__ import annotations

from sentinel.config.settings import Settings
from sentinel.fleet import build_fleet, run_simulated_loop
from sentinel.simulation import SimConfig, symmetric_demand


class TestBuildFleet:
    def test_builds_all_agents_unstarted(self) -> None:
        fleet = build_fleet("i-test")
        assert fleet.intersection_id == "i-test"
        assert len(fleet.agents) == 8

    def test_uses_provided_settings_and_config(self) -> None:
        settings = Settings()
        config = SimConfig(intersection_id="i-custom", horizon_s=60, demand=symmetric_demand(0.1))
        fleet = build_fleet("i-custom", settings=settings, sim_config=config)
        assert fleet.config is config


class TestRunSimulatedLoop:
    async def test_loop_closes_the_control_loop_safely(self) -> None:
        config = SimConfig(
            intersection_id="i-loop", horizon_s=20, dt_s=1.0, demand=symmetric_demand(0.15)
        )
        fleet = build_fleet("i-loop", sim_config=config)

        async with fleet.bus:
            for agent in fleet.agents:
                await agent.start()
            try:
                # Run for a bounded number of ticks rather than forever.
                import asyncio
                import contextlib

                task = asyncio.create_task(run_simulated_loop(fleet, config, tick_sleep_s=0.0))
                await asyncio.sleep(0.2)
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            finally:
                for agent in fleet.agents:
                    await agent.stop()

        snapshot = fleet.dashboard.snapshot()
        assert snapshot.state is not None
        assert snapshot.signal is not None
        assert snapshot.decision is not None
