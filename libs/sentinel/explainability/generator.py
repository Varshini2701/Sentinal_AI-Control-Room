"""Explanation generation: narrating decisions in natural language, strictly grounded in facts.

Two implementations satisfy :class:`ExplanationGenerator`. :class:`TemplateExplainer` is
deterministic, dependency-free, and **always available** -- it is both the default and the
guaranteed fallback. :class:`LlmExplainer` calls Claude to produce more natural prose, but the
prompt constrains it to the exact numbers already computed by the Decision Agent: the LLM narrates
facts, it does not compute or invent them, and it is never in the control loop -- a slow or failed
call only delays an explanation, never a signal change.
"""

from __future__ import annotations

import abc

from sentinel.contracts.enums import Axis
from sentinel.contracts.value_objects import Forecast, SignalCommand

_AXIS_LABEL = {Axis.NORTH_SOUTH: "North-South", Axis.EAST_WEST: "East-West"}

_TEMPLATES: dict[str, str] = {
    "current_axis_serving": (
        "{axis} remains green because it currently has the greater demand "
        "(queue {queue:.0f} vehicles, wait {wait:.0f}s)."
    ),
    "current_axis_congested": (
        "{axis} green was extended because its queue ({queue:.0f} vehicles) "
        "exceeds the congestion threshold."
    ),
    "current_axis_clearing": (
        "{axis} green will be released soon since its queue has cleared."
    ),
    "opposing_demand_higher": (
        "Switching to {axis} because its demand score ({score:.1f}) now exceeds "
        "the previously active axis."
    ),
    "fairness_anti_starvation": (
        "Switching to {axis} to guarantee it is served after waiting "
        "{since_green:.0f}s without green (fairness guarantee)."
    ),
    "emergency_preemption": (
        "EMERGENCY OVERRIDE: preempting the signal to clear the path for an "
        "emergency vehicle on {axis}."
    ),
    "clearance_interval": (
        "Signal is mid-transition (yellow/all-red clearance) before serving {axis}."
    ),
}


def _axis_features(command: SignalCommand) -> dict[str, float]:
    axis = command.target_phase.axis
    suffix = "ns" if axis is Axis.NORTH_SOUTH else "ew"
    snap = command.feature_snapshot
    return {
        "queue": snap.get(f"queue_{suffix}", 0.0),
        "wait": snap.get(f"wait_{suffix}", 0.0),
        "since_green": snap.get(f"since_green_{suffix}", 0.0),
        "score": snap.get(f"score_{suffix}", 0.0),
    }


def counterfactual_for_opposing_axis(
    command: SignalCommand, forecast: Forecast | None
) -> str | None:
    """Describe the projected cost of *not* serving the axis opposing this decision.

    Picks the worse (max) of the two directions on the opposing axis so the statement is a
    meaningful worst case, e.g. "if West stays red, its queue is projected to reach ~40m in 60s."
    """
    if forecast is None:
        return None
    axis = command.target_phase.axis
    if axis is None:
        return None
    opposing = Axis.EAST_WEST if axis is Axis.NORTH_SOUTH else Axis.NORTH_SOUTH

    candidates = [
        (direction, lane_forecast)
        for direction, lane_forecast in forecast.lanes.items()
        if direction.axis is opposing
    ]
    if not candidates:
        return None
    direction, lane_forecast = max(candidates, key=lambda pair: pair[1].predicted_queue_length_m)
    return (
        f"If {direction.value} stays unserved, its queue is projected to reach "
        f"~{lane_forecast.predicted_queue_length_m:.0f}m in {forecast.horizon_s:.0f}s "
        f"(confidence {lane_forecast.confidence:.0%})."
    )


class ExplanationGenerator(abc.ABC):
    """Produces a natural-language explanation for one :class:`SignalCommand`."""

    @abc.abstractmethod
    def generate(self, command: SignalCommand, counterfactual: str | None) -> str:
        """Return explanatory text for ``command``. May raise; callers must handle failure."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Identifier recorded in :class:`Explanation.generator`."""

    @property
    def model_id(self) -> str | None:
        """The underlying model identifier, if any (``None`` for non-LLM generators)."""
        return None


class TemplateExplainer(ExplanationGenerator):
    """Deterministic, fact-grounded template renderer. Always available; the safe fallback."""

    @property
    def name(self) -> str:
        return "template"

    def generate(self, command: SignalCommand, counterfactual: str | None) -> str:
        axis = command.target_phase.axis
        axis_label = _AXIS_LABEL.get(axis, "the intersection") if axis else "the intersection"
        template = _TEMPLATES.get(
            command.reason_code, "{axis}: {reason}"
        )
        text = template.format(
            axis=axis_label, reason=command.reason_code, **_axis_features(command)
        )
        if counterfactual:
            text = f"{text} {counterfactual}"
        return text


class LlmExplainer(ExplanationGenerator):
    """Claude-narrated explanation, strictly grounded in the decision's feature snapshot.

    Requires the optional ``anthropic`` package (lazy import) and an API key resolved by the
    Anthropic SDK's normal precedence (``ANTHROPIC_API_KEY`` env var by default). Never called
    from the control loop -- only from the Explainability Agent, out-of-band from actuation.
    """

    def __init__(self, *, model_id: str = "claude-sonnet-5", max_tokens: int = 200) -> None:
        self._model_id = model_id
        self._max_tokens = max_tokens

    @property
    def name(self) -> str:
        return f"llm:{self._model_id}"

    @property
    def model_id(self) -> str | None:
        return self._model_id

    def generate(self, command: SignalCommand, counterfactual: str | None) -> str:
        import anthropic  # lazy: only required when the LLM narrator is enabled

        client = anthropic.Anthropic()
        facts = _axis_features(command)
        prompt = (
            "You are narrating an autonomous traffic-signal decision for a city operator. "
            "Use ONLY the facts given below; do not invent or alter any number.\n\n"
            f"Action: {command.action.value}\n"
            f"Reason code: {command.reason_code}\n"
            f"Target phase: {command.target_phase.value}\n"
            f"Facts: queue={facts['queue']:.0f} veh, wait={facts['wait']:.0f}s, "
            f"since_green={facts['since_green']:.0f}s, score={facts['score']:.1f}\n"
            + (f"Additional context: {counterfactual}\n" if counterfactual else "")
            + "\nWrite exactly one or two sentences explaining the decision to a non-technical "
            "city operator."
        )
        response = client.messages.create(
            model=self._model_id,
            max_tokens=self._max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        ).strip()


__all__ = [
    "ExplanationGenerator",
    "LlmExplainer",
    "TemplateExplainer",
    "counterfactual_for_opposing_axis",
]
