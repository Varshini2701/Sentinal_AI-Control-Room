"""The Explainability Agent: narrates decisions in natural language, outside the control loop."""

from __future__ import annotations

from sentinel.explainability.agent import ExplainabilityAgent
from sentinel.explainability.generator import (
    ExplanationGenerator,
    LlmExplainer,
    TemplateExplainer,
    counterfactual_for_opposing_axis,
)

__all__ = [
    "ExplainabilityAgent",
    "ExplanationGenerator",
    "LlmExplainer",
    "TemplateExplainer",
    "counterfactual_for_opposing_axis",
]
