from __future__ import annotations

import math
from typing import Any


AXES = ("E", "F", "S", "M")

TLH_ACTION_VECTORS: dict[str, dict[str, float]] = {
    "respond": {"E": 0.76, "F": 0.34, "S": 0.46, "M": 0.68},
    "plan": {"E": 0.62, "F": 0.30, "S": 0.42, "M": 0.84},
    "recall": {"E": 0.42, "F": 0.46, "S": 0.66, "M": 0.70},
    "rest": {"E": 0.20, "F": 0.84, "S": 0.22, "M": 0.38},
    "connect": {"E": 0.84, "F": 0.30, "S": 0.48, "M": 0.60},
    "clarify": {"E": 0.52, "F": 0.26, "S": 0.34, "M": 0.78},
    "wander": {"E": 0.36, "F": 0.42, "S": 0.84, "M": 0.46},
    "absorb": {"E": 0.30, "F": 0.58, "S": 0.74, "M": 0.72},
    "nothing": {"E": 0.18, "F": 0.66, "S": 0.34, "M": 0.40},
    "die": {"E": 0.08, "F": 0.88, "S": 0.22, "M": 0.12},
    "short_reply": {"E": 0.54, "F": 0.36, "S": 0.32, "M": 0.54},
}

REGION_ACTION_HINTS: dict[str, tuple[str, ...]] = {
    "express": ("respond", "connect", "clarify", "short_reply"),
    "withdraw": ("nothing",),
    "hibernate": ("rest",),
    "dissolve": ("die",),
    "absorb": ("absorb", "wander", "recall"),
}


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def normalize_axis_value(value: float) -> float:
    return round(_clip(0.5 + math.tanh(float(value)) * 0.5), 6)


def normalize_axis_map(values: dict[str, Any] | None, *, default: float = 0.5) -> dict[str, float]:
    payload = dict(values or {})
    return {
        axis: round(_clip(float(payload.get(axis, default) or default)), 6)
        for axis in AXES
    }


def signed_map(values: dict[str, Any] | None, *, default: float = 0.5) -> dict[str, float]:
    normalized = normalize_axis_map(values, default=default)
    return {axis: round(float(normalized[axis]) * 2.0 - 1.0, 6) for axis in AXES}


def unsigned_from_signed(values: dict[str, Any] | None) -> dict[str, float]:
    payload = dict(values or {})
    return {
        axis: round(_clip((float(payload.get(axis, 0.0) or 0.0) + 1.0) / 2.0), 6)
        for axis in AXES
    }


def cosine_similarity(left: dict[str, Any] | None, right: dict[str, Any] | None, *, default: float = 0.5) -> float:
    lhs = signed_map(left, default=default)
    rhs = signed_map(right, default=default)
    dot = sum(float(lhs[axis]) * float(rhs[axis]) for axis in AXES)
    lhs_norm = math.sqrt(sum(float(lhs[axis]) ** 2 for axis in AXES))
    rhs_norm = math.sqrt(sum(float(rhs[axis]) ** 2 for axis in AXES))
    if lhs_norm <= 1e-9 or rhs_norm <= 1e-9:
        return 0.0
    return round(_clip(dot / (lhs_norm * rhs_norm), -1.0, 1.0), 6)


def remap_similarity(value: float) -> float:
    return round(_clip((float(value) + 1.0) / 2.0), 6)


def mean_action_vector(actions: list[str] | tuple[str, ...], action_vectors: dict[str, dict[str, float]] | None = None) -> dict[str, float]:
    vectors = action_vectors or TLH_ACTION_VECTORS
    rows = [normalize_axis_map(vectors.get(action)) for action in actions if action in vectors]
    if not rows:
        return {axis: 0.5 for axis in AXES}
    return {
        axis: round(sum(float(row[axis]) for row in rows) / len(rows), 6)
        for axis in AXES
    }


def build_tlh_state_vectors(
    *,
    closeness: float,
    boundary: float,
    affect_residue: float,
    memory_activation: float,
    scarcity: float,
    interference: float,
    spontaneous: float,
    reject_all: float,
    meaning_strength: float,
    continuity: float,
    fatigue: float,
    fragments: float,
    body_energy: float,
    meaning_made_count: int,
    sketch_growth: float,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    raw_axes = {
        "E": round((meaning_strength - reject_all) + closeness * 0.25 - scarcity * 0.15, 6),
        "F": round(fatigue + affect_residue * 0.7 + boundary * 0.4 - body_energy * 0.3, 6),
        "S": round(spontaneous + fragments * 0.45 + memory_activation * 0.25 + interference * 0.2, 6),
        "M": round(meaning_strength + meaning_made_count * 0.08 - (1.0 - continuity) * 0.25, 6),
    }
    v_main = {axis: normalize_axis_value(raw_axes[axis]) for axis in AXES}
    v_mod = {
        "memory_fragments": round(_clip(float(fragments)), 6),
        "spontaneous": round(_clip(float(spontaneous)), 6),
        "reject_all": round(_clip(float(reject_all)), 6),
        "emergent_growth": round(_clip(float(sketch_growth)), 6),
    }
    return v_main, v_mod, raw_axes


def build_tlh_modulation_directions(
    *,
    action_vectors: dict[str, dict[str, float]] | None = None,
    sketch_vectors: list[dict[str, float]] | None = None,
) -> dict[str, dict[str, float]]:
    vectors = action_vectors or TLH_ACTION_VECTORS
    sketches = [normalize_axis_map(vector) for vector in list(sketch_vectors or []) if vector]
    if sketches:
        sketch_center = {
            axis: round(sum(float(vector[axis]) for vector in sketches) / len(sketches), 6)
            for axis in AXES
        }
    else:
        sketch_center = {axis: 0.5 for axis in AXES}
    return {
        "memory_fragments": mean_action_vector(["absorb", "recall", "nothing"], vectors),
        "spontaneous": mean_action_vector(["wander", "absorb", "recall"], vectors),
        "reject_all": mean_action_vector(["rest", "nothing", "die"], vectors),
        "emergent_growth": sketch_center,
    }


def infer_sketch_vector(sketch: Any, action_vectors: dict[str, dict[str, float]] | None = None) -> dict[str, float] | None:
    vectors = action_vectors or TLH_ACTION_VECTORS
    if sketch is None:
        return None
    payload = dict(sketch) if isinstance(sketch, dict) else {
        "signal_sources": list(getattr(sketch, "signal_sources", []) or []),
        "support_actions": dict(getattr(sketch, "support_actions", {}) or {}),
        "target_action_map": dict(getattr(sketch, "target_action_map", {}) or {}),
    }
    weights: dict[str, float] = {}
    for source_map_name in ("target_action_map", "support_actions"):
        for action, value in dict(payload.get(source_map_name, {}) or {}).items():
            action_name = str(action).strip()
            score = max(0.0, float(value or 0.0))
            if action_name in vectors and score > 0.0:
                weights[action_name] = round(weights.get(action_name, 0.0) + score, 6)
    for raw_source in list(payload.get("signal_sources", []) or []):
        source = str(raw_source or "").strip()
        if not source:
            continue
        if source.startswith("action:"):
            action_name = source.split(":", 1)[1].strip()
            if action_name in vectors:
                weights[action_name] = round(weights.get(action_name, 0.0) + 0.36, 6)
        elif source.startswith("sampled:"):
            action_name = source.split(":", 1)[1].strip()
            if action_name in vectors:
                weights[action_name] = round(weights.get(action_name, 0.0) + 0.22, 6)
        elif source.startswith("winner_region:"):
            region_name = source.split(":", 1)[1].strip()
            region_actions = REGION_ACTION_HINTS.get(region_name, ())
            for action_name in region_actions:
                weights[action_name] = round(weights.get(action_name, 0.0) + 0.12, 6)
    if not weights:
        return None
    total = sum(float(value) for value in weights.values()) or 1.0
    return {
        axis: round(
            sum(float(vectors[action][axis]) * float(weight) for action, weight in weights.items()) / total,
            6,
        )
        for axis in AXES
    }


def project_vector_to_action_support(
    vector: dict[str, float] | None,
    action_vectors: dict[str, dict[str, float]] | None = None,
    *,
    floor: float = 0.0,
    limit: int | None = None,
) -> dict[str, float]:
    if vector is None:
        return {}
    vectors = action_vectors or TLH_ACTION_VECTORS
    scored = {
        action: remap_similarity(cosine_similarity(vector, action_vector))
        for action, action_vector in vectors.items()
    }
    ranked = [
        (action, score)
        for action, score in sorted(scored.items(), key=lambda item: (item[1], item[0]), reverse=True)
        if float(score) > float(floor)
    ]
    if limit is not None:
        ranked = ranked[:limit]
    return {action: round(float(score), 6) for action, score in ranked}


def region_scores_from_match_scores(match_scores: dict[str, float]) -> dict[str, float]:
    mapped = {action: remap_similarity(float(value)) for action, value in dict(match_scores or {}).items()}
    return {
        "express": round(max(mapped.get("respond", 0.0), mapped.get("connect", 0.0), mapped.get("clarify", 0.0), mapped.get("short_reply", 0.0)), 6),
        "withdraw": round(mapped.get("nothing", 0.0), 6),
        "hibernate": round(mapped.get("rest", 0.0), 6),
        "dissolve": round(mapped.get("die", 0.0), 6),
        "absorb": round(max(mapped.get("absorb", 0.0), mapped.get("wander", 0.0), mapped.get("recall", 0.0)), 6),
    }
