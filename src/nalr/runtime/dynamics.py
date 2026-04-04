from __future__ import annotations

import math


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def smooth_decay_rate(*, base_decay: float, recalled_within_window: bool, affect_intensity: float) -> float:
    recall_modifier = 0.72 if recalled_within_window else 1.0
    affect_support = sigmoid((float(affect_intensity) - 0.40) * 6.0)
    affect_modifier = 1.20 - 0.65 * affect_support
    return round(_clip(float(base_decay) * recall_modifier * affect_modifier, 1e-6, 0.05), 6)


def smooth_interference_penalty(*, similarity: float, old_strength: float) -> float:
    sim_gate = sigmoid((float(similarity) - 0.82) * 10.0)
    weakness = 1.0 - _clip(float(old_strength), 0.0, 1.0)
    return round(_clip(0.18 * sim_gate * (weakness ** 1.1), 0.0, 1.0), 4)


def smooth_resource_pressure(scarcity_index: float) -> float:
    return round(_clip(sigmoid((float(scarcity_index) - 0.45) * 5.5), 0.0, 1.0), 4)


def smooth_resource_biases(scarcity_index: float) -> dict[str, float]:
    scarcity = _clip(float(scarcity_index), 0.0, 1.0)
    pressure = smooth_resource_pressure(scarcity)
    return {
        "scarcity_pressure": pressure,
        "body_hunger_bias": round(_clip(0.10 + 0.55 * pressure, 0.0, 1.0), 4),
        "effort_avoidance_bias": round(_clip(0.05 + 0.45 * pressure, 0.0, 1.0), 4),
        "deliberation_compress": round(_clip(1.00 - 0.50 * pressure, 0.0, 1.0), 4),
        "rumination_bias": round(_clip(0.05 + 0.35 * pressure, 0.0, 1.0), 4),
        "action_shrink_scale": round(_clip(1.00 - 0.40 * pressure, 0.0, 1.0), 4),
    }


def bounded_drift_delta(*, previous_drift: float, push: float, recover_rate: float, max_abs: float = 0.25) -> float:
    recover = _clip(float(recover_rate), 0.0, 1.0)
    raw = float(previous_drift) * (1.0 - recover) + float(push)
    return round(_clip(raw, -abs(float(max_abs)), abs(float(max_abs))), 4)


def smooth_habit_recovery(*, context_recurrence: float, valence: float) -> float:
    recurrence = _clip(float(context_recurrence), 0.0, 1.0)
    positive = _clip(max(0.0, float(valence)), 0.0, 1.0)
    support = sigmoid((recurrence - 0.35) * 4.0)
    return round(_clip(0.02 + 0.03 * support + positive * 0.03, 0.0, 0.10), 4)
