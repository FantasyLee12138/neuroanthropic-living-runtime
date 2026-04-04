from __future__ import annotations

import random
from collections import Counter
from typing import Any


class ValueAgent:
    name = "ValueAgent"

    def apply(self, scores: dict[str, float], scenario_name: str, state: dict[str, Any], context: dict[str, Any]) -> tuple[dict[str, float], str]:
        adjusted = dict(scores)
        if scenario_name == "task":
            adjusted["plan"] = adjusted.get("plan", 0.0) + 0.12
        if scenario_name == "companion":
            adjusted["connect"] = adjusted.get("connect", 0.0) + (0.10 * context.get("closeness", 0.5))
        if state["body_energy"] < 0.45:
            adjusted["rest"] = adjusted.get("rest", 0.0) + 0.08
        return adjusted, "value-weighted priorities"


class ConflictMonitorAgent:
    name = "ConflictMonitorAgent"

    def score(self, scores: dict[str, float], proposals: list[dict[str, Any]], context: dict[str, Any]) -> tuple[float, int]:
        ordered = sorted(scores.values(), reverse=True)
        if not ordered:
            return 0.0, 0
        second = ordered[1] if len(ordered) > 1 else 0.0
        proposal_divergence = max(0.0, 1.0 - abs(ordered[0] - second))
        veto_tension = 1.0 if any(item.get("veto") for item in proposals) else 0.0
        value_gap = min(1.0, abs(ordered[0] - second))
        relation_risk_gap = 1.0 - context.get("closeness", 0.5)
        body_state_gap = 1.0 - context.get("body_energy", 0.5)
        conflict_score = (
            0.35 * proposal_divergence
            + 0.20 * veto_tension
            + 0.20 * value_gap
            + 0.15 * relation_risk_gap
            + 0.10 * body_state_gap
        )
        if conflict_score < 0.65:
            return round(conflict_score, 4), 0
        if conflict_score < 0.82:
            return round(conflict_score, 4), 1
        return round(conflict_score, 4), 2


class BehaviorPlausibilityGuard:
    name = "BehaviorPlausibilityGuard"

    def score(self, action_name: str, scenario_name: str, state: dict[str, Any], context: dict[str, Any]) -> float:
        disclosure_risk = 1.0 if action_name == "connect" and context.get("closeness", 0.5) < 0.55 else 0.0
        relation_mismatch = 1.0 if action_name == "connect" and context.get("closeness", 0.5) < 0.45 else 0.0
        body_mismatch = 1.0 if action_name == "plan" and state["body_energy"] < 0.35 else 0.0
        task_value_mismatch = 1.0 if scenario_name == "task" and action_name == "wander" else 0.0
        style_mismatch = 1.0 if scenario_name == "task" and context.get("style_name") == "distracted" else 0.0
        return round(
            0.30 * disclosure_risk
            + 0.25 * relation_mismatch
            + 0.20 * body_mismatch
            + 0.15 * task_value_mismatch
            + 0.10 * style_mismatch,
            4,
        )

    def apply(self, scores: dict[str, float], action_name: str, fail_score: float) -> tuple[dict[str, float], str]:
        adjusted = dict(scores)
        if 0.50 <= fail_score < 0.70:
            adjusted[action_name] = adjusted.get(action_name, 0.0) * 0.85
            return adjusted, "soft guard"
        if 0.70 <= fail_score < 0.85:
            adjusted[action_name] = max(0.0, adjusted.get(action_name, 0.0) * 0.50)
            adjusted["respond"] = adjusted.get("respond", 0.0) + 0.10
            return adjusted, "resample guard"
        if fail_score >= 0.85:
            adjusted[action_name] = 0.0
            adjusted["respond"] = adjusted.get("respond", 0.0) + 0.20
            adjusted["rest"] = adjusted.get("rest", 0.0) + 0.10
            return adjusted, "veto guard"
        return adjusted, "pass"


class ForcedModeSwitch:
    name = "ForcedModeSwitch"

    def apply(self, scores: dict[str, float], recent_actions: list[str], max_lock_rounds: int) -> tuple[dict[str, float], bool]:
        if len(recent_actions) < max_lock_rounds:
            return dict(scores), False
        tail = recent_actions[-max_lock_rounds:]
        if len(set(tail)) != 1:
            return dict(scores), False
        adjusted = dict(scores)
        current = tail[-1]
        if current == "respond":
            adjusted["plan"] = adjusted.get("plan", 0.0) + 0.16
        else:
            adjusted["respond"] = adjusted.get("respond", 0.0) + 0.16
        return adjusted, True


def sample_action(scores: dict[str, float], seed: int | None = None) -> tuple[str, dict[str, float]]:
    positive = {name: max(score, 0.0) for name, score in scores.items()}
    total = sum(positive.values()) or 1.0
    normalized = {name: round(score / total, 6) for name, score in positive.items()}
    if seed is None:
        return max(normalized, key=normalized.get), normalized
    rng = random.Random(seed)
    names = list(normalized)
    weights = [normalized[name] for name in names]
    return rng.choices(names, weights=weights, k=1)[0], normalized


def summarize_counts(items: list[str]) -> dict[str, int]:
    return dict(Counter(items))
