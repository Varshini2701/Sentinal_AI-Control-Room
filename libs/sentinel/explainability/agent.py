"""The Explainability Agent: narrates every decision, entirely outside the control loop.

Consumes ``decision.made`` (and caches the latest ``prediction.updated`` forecast for
counterfactuals), calls the configured :class:`ExplanationGenerator` with a bounded timeout, and
emits ``explanation.generated``. Any generator failure or timeout -- network error, malformed LLM
response, anything -- falls back to the deterministic :class:`TemplateExplainer`, so an
explanation is always produced and a stalled narrator can never stall a signal change (the command
that gets explained has *already* been actuated by the time this agent even sees it).
"""

from __future__ import annotations

import asyncio

from sentinel.agents.base import BaseAgent
from sentinel.config.settings import ExplainabilitySettings
from sentinel.contracts.events import (
    DecisionMade,
    DomainEvent,
    ExplanationGenerated,
    PredictionUpdated,
)
from sentinel.contracts.value_objects import Explanation, Forecast, SignalCommand
from sentinel.explainability.generator import (
    ExplanationGenerator,
    TemplateExplainer,
    counterfactual_for_opposing_axis,
)
from sentinel.messaging.bus import EventBus


class ExplainabilityAgent(BaseAgent):
    """Event-driven agent that produces a natural-language explanation for each decision."""

    def __init__(
        self,
        *,
        event_bus: EventBus,
        intersection_id: str,
        settings: ExplainabilitySettings | None = None,
        generator: ExplanationGenerator | None = None,
        fallback: ExplanationGenerator | None = None,
        heartbeat_interval_s: float = 5.0,
    ) -> None:
        self._settings = settings or ExplainabilitySettings()
        self._fallback = fallback or TemplateExplainer()
        self._generator = generator or self._fallback
        self._forecast: Forecast | None = None
        super().__init__(
            name="explainability-agent",
            event_bus=event_bus,
            intersection_id=intersection_id,
            heartbeat_interval_s=heartbeat_interval_s,
        )

    def _register(self) -> None:
        self._subscribe("decision.made", self._on_decision)
        self._subscribe("prediction.updated", self._on_prediction)

    async def _on_prediction(self, event: DomainEvent) -> None:
        if not isinstance(event, PredictionUpdated):
            return
        self._forecast = event.forecast

    async def _on_decision(self, event: DomainEvent) -> None:
        if not isinstance(event, DecisionMade):
            return
        command = event.command
        counterfactual = counterfactual_for_opposing_axis(command, self._forecast)

        text, generator_name, model_id = await self._generate(command, counterfactual)
        explanation = Explanation(
            intersection_id=self._intersection_id,
            decision_reason_code=command.reason_code,
            text=text,
            counterfactual=counterfactual,
            generator=generator_name,
            model_id=model_id,
        )
        await self._publish(
            ExplanationGenerated(
                source=self.name,
                intersection_id=self._intersection_id,
                explanation=explanation,
                correlation_id=event.correlation_id or event.event_id,
                causation_id=event.event_id,
            )
        )

    async def _generate(
        self, command: SignalCommand, counterfactual: str | None
    ) -> tuple[str, str, str | None]:
        if self._generator is self._fallback:
            return (
                self._fallback.generate(command, counterfactual),
                self._fallback.name,
                self._fallback.model_id,
            )
        try:
            text = await asyncio.wait_for(
                asyncio.to_thread(self._generator.generate, command, counterfactual),
                timeout=self._settings.timeout_s,
            )
            return text, self._generator.name, self._generator.model_id
        except Exception as exc:  # generator failure/timeout must never block the pipeline
            self._log.warning("explanation_generator_failed", error=str(exc), falling_back=True)
            return (
                self._fallback.generate(command, counterfactual),
                self._fallback.name,
                self._fallback.model_id,
            )


__all__ = ["ExplainabilityAgent"]
