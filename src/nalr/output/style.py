from __future__ import annotations

import random
from typing import Any

from nalr.schemas.models import ExpressionProfile, RenderPlan, StochasticState


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _style_name(state: dict[str, Any], scenario: str) -> str:
    style_name = "task_focused"
    if state.get("body_energy", 1.0) < 0.45:
        style_name = "tired"
    elif state.get("focus") == "wander":
        style_name = "distracted"
    elif scenario == "companion" and state.get("mood", 0.5) < 0.45:
        style_name = "guarded"
    return style_name


def compute_style_profile(state: dict, scenario: str, output_styles: dict) -> dict:
    style_name = _style_name(state, scenario)
    profile = dict(output_styles.get(style_name, {}))
    profile["style_name"] = style_name
    return profile


def build_expression_profile(
    *,
    sampled_action: str,
    state: dict[str, Any],
    scenario: str,
    scenario_config: dict[str, Any],
    relation_state: dict[str, Any],
    stochastic: StochasticState,
) -> ExpressionProfile:
    rng = random.Random(stochastic.round_seed)
    intimacy = relation_state.get("closeness", 0.5)
    relationship_risk = relation_state.get("relationship_risk", 0.2)
    boundary_level = relation_state.get("boundary_level", 0.3)
    privacy_level = relation_state.get("privacy_level", 0.5)
    body_energy = state.get("body_energy", 0.7)
    mood = state.get("mood", 0.5)
    warmth_variance = scenario_config.get("output_warmth_variance", 0.1)
    delay_tolerance = scenario_config.get("delay_tolerance", 0.1)
    intensity = _clip(stochastic.r_intensity, 0.0, 1.0)

    d_min = 0.15 + delay_tolerance * 0.30
    d_max = 0.55 + delay_tolerance * 1.40 + max(0.0, 0.45 - body_energy) * 0.50
    s_min = 0.08 + intimacy * 0.05
    s_max = 0.30 + privacy_level * 0.40 + intimacy * 0.20
    t_min = 0.10 + max(0.0, -mood) * 0.02
    t_max = 0.35 + max(0.0, 0.6 - intimacy) * 0.10 + max(0.0, 0.5 - relationship_risk) * 0.08
    r_min = 0.18 + relationship_risk * 0.18
    r_max = 0.70 + boundary_level * 0.10

    reply_delay = d_min + intensity * (d_max - d_min)
    self_disclosure = s_min + intensity * (s_max - s_min)
    tone_sharpness = t_min + intensity * (t_max - t_min)
    repair_tendency = r_min + (1 - intensity) * (r_max - r_min)

    timing_jitter = _clip(rng.uniform(-0.08, 0.12) + stochastic.xi_emo * 0.02, -0.08, 0.12)
    fragmentation_jitter = _clip(rng.uniform(-0.05, 0.10) + stochastic.xi_mood * 0.03, -0.05, 0.10)

    warmth_level = _clip(0.35 + intimacy * 0.25 + warmth_variance * intensity - relationship_risk * 0.20 - boundary_level * 0.12, 0.0, 1.0)
    directness_level = _clip(0.55 + (0.15 if scenario == "task" else 0.0) + intensity * 0.08 - boundary_level * 0.12, 0.0, 1.0)
    hedging_level = _clip(0.20 + relationship_risk * 0.22 + (0.08 if sampled_action == "clarify" else 0.0) - directness_level * 0.08, 0.0, 1.0)
    sentence_fragmentation = _clip(0.08 + (1 - body_energy) * 0.18 + fragmentation_jitter + (0.06 if sampled_action == "wander" else 0.0), 0.0, 1.0)
    latency_style = _clip(0.35 + reply_delay * 0.45 + timing_jitter + max(0.0, 0.45 - body_energy) * 0.12, 0.0, 1.0)

    return ExpressionProfile(
        reply_delay=round(reply_delay, 4),
        latency_style=round(latency_style, 4),
        sentence_fragmentation=round(sentence_fragmentation, 4),
        hedging_level=round(hedging_level, 4),
        warmth_level=round(warmth_level, 4),
        directness_level=round(directness_level, 4),
        self_disclosure=round(_clip(self_disclosure, 0.0, 1.0), 4),
        tone_sharpness=round(_clip(tone_sharpness, 0.0, 1.0), 4),
        repair_tendency=round(_clip(repair_tendency, 0.0, 1.0), 4),
        timing_jitter=round(timing_jitter, 4),
        fragmentation_jitter=round(fragmentation_jitter, 4),
    )


def build_render_plan(
    *,
    sampled_action: str,
    expression: ExpressionProfile,
    safety_constraints: dict[str, Any],
    scenario: str,
    event_summary: str = "",
    target: str | None = None,
    relation_state: dict[str, Any] | None = None,
    perspective: dict[str, Any] | None = None,
) -> RenderPlan:
    message_plan = {
        "intent": sampled_action,
        "opening_style": "direct" if expression.directness_level >= 0.65 else "buffered",
        "warmth": expression.warmth_level,
        "repair": expression.repair_tendency,
        "scenario": scenario,
    }
    return RenderPlan(
        action=sampled_action,
        expression=expression,
        safety_constraints=safety_constraints,
        message_plan=message_plan,
        event_summary=event_summary,
        scenario=scenario,
        target=target,
        relation_state=relation_state or {},
        perspective=perspective or {},
    )
