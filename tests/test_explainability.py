"""Tests for the explanation generators and the Explainability Agent's fallback behaviour."""

from __future__ import annotations

from sentinel.config.settings import ExplainabilitySettings
from sentinel.contracts.enums import Direction, SignalPhase
from sentinel.contracts.events import DecisionMade, ExplanationGenerated, PredictionUpdated
from sentinel.contracts.value_objects import (
    Forecast,
    LaneForecast,
    SignalCommand,
)
from sentinel.decision.policy import DecisionAction
from sentinel.explainability import ExplainabilityAgent, TemplateExplainer
from sentinel.explainability.generator import ExplanationGenerator, counterfactual_for_opposing_axis
from sentinel.messaging import InMemoryEventBus


def _command(reason_code: str, phase: SignalPhase = SignalPhase.NS_GREEN) -> SignalCommand:
    return SignalCommand(
        intersection_id="i-1",
        action=DecisionAction.EXTEND_GREEN,
        target_phase=phase,
        duration_s=10.0,
        reason_code=reason_code,
        feature_snapshot={
            "queue_ns": 18.0,
            "wait_ns": 22.0,
            "since_green_ns": 5.0,
            "score_ns": 30.0,
        },
    )


def _forecast() -> Forecast:
    return Forecast(
        intersection_id="i-1",
        horizon_s=60.0,
        lanes={
            Direction.EAST: LaneForecast(
                direction=Direction.EAST, horizon_s=60.0, predicted_queue_length_m=40.0,
                predicted_wait_s=20.0, confidence=0.8, lower_bound_m=30.0, upper_bound_m=50.0,
            ),
            Direction.WEST: LaneForecast(
                direction=Direction.WEST, horizon_s=60.0, predicted_queue_length_m=10.0,
                predicted_wait_s=5.0, confidence=0.9, lower_bound_m=5.0, upper_bound_m=15.0,
            ),
        },
        model_version="linear-trend-v1",
    )


class TestTemplateExplainer:
    def test_known_reason_produces_grounded_text(self) -> None:
        text = TemplateExplainer().generate(_command("current_axis_congested"), None)
        assert "18" in text
        assert "North-South" in text

    def test_unknown_reason_falls_back_generically(self) -> None:
        text = TemplateExplainer().generate(_command("some_new_reason"), None)
        assert "some_new_reason" in text

    def test_counterfactual_appended(self) -> None:
        text = TemplateExplainer().generate(_command("current_axis_congested"), "extra context")
        assert "extra context" in text

    def test_name(self) -> None:
        assert TemplateExplainer().name == "template"
        assert TemplateExplainer().model_id is None


class TestCounterfactualHelper:
    def test_picks_worse_opposing_direction(self) -> None:
        text = counterfactual_for_opposing_axis(_command("current_axis_congested"), _forecast())
        assert text is not None
        assert "east" in text.lower()
        assert "40" in text

    def test_none_without_forecast(self) -> None:
        assert counterfactual_for_opposing_axis(_command("current_axis_congested"), None) is None


class _FlakyGenerator(ExplanationGenerator):
    """Always raises -- simulates an LLM failure to exercise the agent's fallback path."""

    @property
    def name(self) -> str:
        return "flaky"

    def generate(self, command: SignalCommand, counterfactual: str | None) -> str:
        raise RuntimeError("simulated LLM failure")


class _StubGenerator(ExplanationGenerator):
    """Returns a fixed string -- simulates a working LLM without a network dependency."""

    @property
    def name(self) -> str:
        return "stub-llm"

    @property
    def model_id(self) -> str | None:
        return "stub-model"

    def generate(self, command: SignalCommand, counterfactual: str | None) -> str:
        return "Stubbed narrative explanation."


class TestExplainabilityAgent:
    async def test_default_agent_uses_template(self) -> None:
        explanations: list[ExplanationGenerated] = []

        async def capture(event: ExplanationGenerated) -> None:  # type: ignore[override]
            explanations.append(event)

        bus = InMemoryEventBus()
        bus.subscribe("explanation.generated", capture, consumer_name="cap")
        ExplainabilityAgent(event_bus=bus, intersection_id="i-1", heartbeat_interval_s=0.0)

        async with bus:
            await bus.publish(
                DecisionMade(
                    source="decision", intersection_id="i-1",
                    command=_command("current_axis_congested"),
                )
            )
            await bus.join()

        assert len(explanations) == 1
        assert explanations[0].explanation.generator == "template"

    async def test_working_generator_is_used(self) -> None:
        explanations: list[ExplanationGenerated] = []

        async def capture(event: ExplanationGenerated) -> None:  # type: ignore[override]
            explanations.append(event)

        bus = InMemoryEventBus()
        bus.subscribe("explanation.generated", capture, consumer_name="cap")
        ExplainabilityAgent(
            event_bus=bus, intersection_id="i-1", generator=_StubGenerator(),
            heartbeat_interval_s=0.0,
        )

        async with bus:
            await bus.publish(
                DecisionMade(
                    source="decision", intersection_id="i-1",
                    command=_command("current_axis_congested"),
                )
            )
            await bus.join()

        assert explanations[0].explanation.text == "Stubbed narrative explanation."
        assert explanations[0].explanation.generator == "stub-llm"
        assert explanations[0].explanation.model_id == "stub-model"

    async def test_failing_generator_falls_back_to_template(self) -> None:
        explanations: list[ExplanationGenerated] = []

        async def capture(event: ExplanationGenerated) -> None:  # type: ignore[override]
            explanations.append(event)

        bus = InMemoryEventBus()
        bus.subscribe("explanation.generated", capture, consumer_name="cap")
        ExplainabilityAgent(
            event_bus=bus, intersection_id="i-1", generator=_FlakyGenerator(),
            heartbeat_interval_s=0.0,
        )

        async with bus:
            await bus.publish(
                DecisionMade(
                    source="decision", intersection_id="i-1",
                    command=_command("current_axis_congested"),
                )
            )
            await bus.join()

        assert len(explanations) == 1  # an explanation is still produced
        assert explanations[0].explanation.generator == "template"  # fell back

    async def test_uses_cached_forecast_for_counterfactual(self) -> None:
        explanations: list[ExplanationGenerated] = []

        async def capture(event: ExplanationGenerated) -> None:  # type: ignore[override]
            explanations.append(event)

        bus = InMemoryEventBus()
        bus.subscribe("explanation.generated", capture, consumer_name="cap")
        ExplainabilityAgent(
            event_bus=bus, intersection_id="i-1",
            settings=ExplainabilitySettings(timeout_s=2.0), heartbeat_interval_s=0.0,
        )

        async with bus:
            await bus.publish(
                PredictionUpdated(source="prediction", intersection_id="i-1", forecast=_forecast())
            )
            await bus.join()
            await bus.publish(
                DecisionMade(
                    source="decision", intersection_id="i-1",
                    command=_command("current_axis_congested"),
                )
            )
            await bus.join()

        assert explanations[0].explanation.counterfactual is not None
        assert "east" in explanations[0].explanation.counterfactual.lower()
