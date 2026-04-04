from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from typing import Any


PROBABILITY_FIELD_SCHEMA_VERSION = "v0.57/probability-field"
PRIMARY_LAYERS = ("context", "memory", "action", "token")
ALL_LAYERS = (*PRIMARY_LAYERS, "global")


CONTEXT_STAGE_NAMES = {
    "state_update",
    "salience",
    "body",
    "emotion",
    "relationship",
    "resource",
    "perspective",
}
MEMORY_STAGE_NAMES = {
    "hippocampus",
    "dream",
}
ACTION_STAGE_NAMES = {
    "pfc",
    "habit",
    "desire",
    "dmn",
    "value",
    "unconscious",
    "cerebellar",
}
TOKEN_STAGE_NAMES = {
    "conflict",
    "thalamus",
    "plausibility_guard",
    "forced_mode_switch",
    "output_gate",
    "late_perspective",
    "renderer",
}
GLOBAL_STAGE_NAMES = {
    "identity",
    "authenticity",
    "vitality",
}


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _best_action(delta_map: dict[str, Any]) -> tuple[str, float]:
    if not delta_map:
        return "unknown", 0.0
    best_action = max(delta_map, key=lambda key: abs(_float(delta_map[key])))
    return best_action, _float(delta_map[best_action])


def _layer_for_stage(stage: str | None, module_name: str | None = None) -> str:
    normalized = str(stage or module_name or "").strip().lower()
    if normalized in CONTEXT_STAGE_NAMES:
        return "context"
    if normalized in MEMORY_STAGE_NAMES:
        return "memory"
    if normalized in ACTION_STAGE_NAMES:
        return "action"
    if normalized in TOKEN_STAGE_NAMES:
        return "token"
    if normalized in GLOBAL_STAGE_NAMES:
        return "global"
    if normalized in {"thalamusattentionagent", "thalamusattention", "thalamus"}:
        return "token"
    if normalized in {"hippocampusagent", "hippocampus"}:
        return "memory"
    if normalized in {"identityruntime", "identity", "authenticitypolicy", "authenticity", "vitalityengine", "vitality", "dreamorchestrator", "dream"}:
        return "global"
    return "action"


def _top_modules(contributions: list[dict[str, Any]], layer: str) -> list[dict[str, Any]]:
    rows = [item for item in contributions if item.get("layer") == layer]
    rows.sort(key=lambda item: (_float(item.get("confidence"), 0.0), abs(_float(item.get("delta_score"), 0.0))), reverse=True)
    return [
        {
            "module_name": item.get("module_name"),
            "target_name": item.get("target_name"),
            "confidence": round(_float(item.get("confidence"), 0.0), 4),
            "trace_reason": item.get("trace_reason", ""),
        }
        for item in rows[:3]
    ]


def _layer_summary(layer: str, contributions: list[dict[str, Any]], winner_posterior: dict[str, float]) -> dict[str, Any]:
    layer_rows = [item for item in contributions if item.get("layer") == layer]
    delta_logits: dict[str, float] = defaultdict(float)
    attention_bias: dict[str, float] = defaultdict(float)
    posterior: dict[str, float] = defaultdict(float)
    for item in layer_rows:
        for key, value in _mapping(item.get("delta_logits")).items():
            delta_logits[str(key)] += _float(value)
        for key, value in _mapping(item.get("attention_bias")).items():
            attention_bias[str(key)] += _float(value)
        for key, value in _mapping(item.get("posterior")).items():
            posterior[str(key)] = max(posterior[str(key)], _float(value))
    return {
        "layer": layer,
        "contribution_count": len(layer_rows),
        "top_modules": _top_modules(contributions, layer),
        "delta_logits": {key: round(value, 6) for key, value in sorted(delta_logits.items())},
        "attention_bias": {key: round(value, 6) for key, value in sorted(attention_bias.items())},
        "posterior": {
            key: round(value, 6)
            for key, value in sorted((posterior or winner_posterior).items())
        },
        "winner_action": max((posterior or winner_posterior), key=(posterior or winner_posterior).get) if (posterior or winner_posterior) else None,
    }


def _normalize_delta_map(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    return {str(key): round(_float(raw), 6) for key, raw in value.items() if isinstance(key, str)}


def _probability_baseline(payload: dict[str, Any]) -> dict[str, float]:
    candidate_distribution = _mapping(payload.get("candidate_distribution"))
    if candidate_distribution:
        return {str(key): round(_float(value), 6) for key, value in candidate_distribution.items()}
    distribution_state = _mapping(payload.get("distribution_state"))
    p_raw = _mapping(distribution_state.get("p_raw"))
    if p_raw:
        return {str(key): round(_float(value), 6) for key, value in p_raw.items()}
    p_final = _mapping(distribution_state.get("p_final"))
    if p_final:
        return {str(key): round(_float(value), 6) for key, value in p_final.items()}
    totals: dict[str, float] = defaultdict(float)
    for summary in _list(payload.get("proposal_summaries")):
        delta_map = _normalize_delta_map(summary.get("delta_p"))
        for key, value in delta_map.items():
            totals[key] += value
    if totals:
        total = sum(abs(value) for value in totals.values()) or 1.0
        return {key: round(_clip(abs(value) / total), 6) for key, value in sorted(totals.items())}
    sampled_action = str(payload.get("sampled_action") or "respond")
    return {sampled_action: 1.0}


def _derive_contribution(entry: dict[str, Any], payload: dict[str, Any], index: int) -> dict[str, Any]:
    delta_logits = _normalize_delta_map(entry.get("delta_logits") or entry.get("delta_p") or entry.get("action_preferences"))
    if not delta_logits and entry.get("top_action"):
        delta_logits = {str(entry.get("top_action")): _float(entry.get("confidence"), 0.0)}
    top_action, delta_score = _best_action(delta_logits)
    layer = str(entry.get("layer") or _layer_for_stage(entry.get("stage"), entry.get("module_name") or entry.get("agent_name"))).lower()
    module_name = str(entry.get("module_name") or entry.get("agent_name") or entry.get("owner_module") or "unknown")
    posterior = _normalize_delta_map(entry.get("posterior"))
    attention_bias = _normalize_delta_map(entry.get("attention_bias"))
    soft_mask = _mapping(entry.get("soft_mask"))
    hard_mask = _mapping(entry.get("hard_mask"))
    contribution = {
        "contribution_index": index,
        "layer": layer,
        "module_name": module_name,
        "target_name": str(entry.get("target_name") or entry.get("top_action") or top_action),
        "top_action": str(entry.get("top_action") or top_action),
        "selected": bool(entry.get("selected", False)),
        "confidence": round(_clip(_float(entry.get("confidence"), 0.0)), 4),
        "sigma_scale": round(_clip(_float(entry.get("sigma_scale"), 1.0), 0.0, 2.5), 4),
        "weight_applied": round(_float(entry.get("weight_applied"), 1.0), 4),
        "delta_score": round(delta_score, 6),
        "delta_logits": delta_logits,
        "delta_energy": _normalize_delta_map(entry.get("delta_energy")),
        "attention_bias": attention_bias,
        "soft_mask": soft_mask,
        "hard_mask": hard_mask,
        "posterior": posterior,
        "trace_reason": str(entry.get("trace_reason") or entry.get("reason") or ""),
        "peak_id": str(entry.get("peak_id") or f"{module_name}:{layer}:{index}"),
        "rank": int(entry.get("rank", index + 1)),
        "suppression_cause": str(entry.get("suppression_cause") or ""),
        "tags": _list(entry.get("tags")),
        "raw": deepcopy(entry),
    }
    if not contribution["suppression_cause"] and contribution["hard_mask"]:
        contribution["suppression_cause"] = "hard_mask"
    if not contribution["suppression_cause"] and any(value <= 0.0 for value in delta_logits.values()):
        contribution["suppression_cause"] = "negative_bias"
    return contribution


def _synthetic_global_contribution(payload: dict[str, Any], module_name: str, layer: str, *, confidence: float, trace_reason: str, delta_seed: dict[str, Any] | None = None, target_name: str | None = None) -> dict[str, Any]:
    delta_logits = _normalize_delta_map(delta_seed or {})
    if not delta_logits:
        delta_logits = {module_name.lower(): round(confidence, 6)}
    top_action, delta_score = _best_action(delta_logits)
    return {
        "contribution_index": 0,
        "layer": layer,
        "module_name": module_name,
        "target_name": target_name or top_action,
        "top_action": top_action,
        "selected": False,
        "confidence": round(_clip(confidence), 4),
        "sigma_scale": 1.0,
        "weight_applied": 1.0,
        "delta_score": round(delta_score, 6),
        "delta_logits": delta_logits,
        "delta_energy": {},
        "attention_bias": {},
        "soft_mask": {},
        "hard_mask": {},
        "posterior": {},
        "trace_reason": trace_reason,
        "peak_id": f"{module_name}:{layer}:synthetic",
        "rank": 0,
        "suppression_cause": "",
        "tags": [layer, module_name.lower()],
        "raw": {},
    }


def _base_contributions(payload: dict[str, Any]) -> list[dict[str, Any]]:
    contributions: list[dict[str, Any]] = []
    source_rows = _list(payload.get("probabilistic_contributions")) or _list(payload.get("proposal_summaries"))
    for index, summary in enumerate(source_rows):
        contributions.append(_derive_contribution(summary, payload, index))

    identity = _mapping(payload.get("identity_evolution"))
    if identity or payload.get("identity_evidence_score"):
        contributions.append(
            _synthetic_global_contribution(
                payload,
                "IdentityRuntime",
                "global",
                confidence=_float(payload.get("identity_evidence_score"), _float(identity.get("identity_score"), 0.0)),
                trace_reason=str(identity.get("rename_reason") or identity.get("evolution_reason") or "identity prior"),
                delta_seed={"identity": payload.get("identity_evidence_score", 0.0)},
                target_name=str(identity.get("current_display_name") or payload.get("sampled_action") or "identity"),
            )
        )

    authenticity = _mapping(payload.get("authenticity"))
    if authenticity:
        contributions.append(
            _synthetic_global_contribution(
                payload,
                "AuthenticityPolicy",
                "global",
                confidence=_float(authenticity.get("self_grounding_score"), 0.0),
                trace_reason=str(authenticity.get("guard_action") or "authenticity penalty"),
                delta_seed={"authenticity": authenticity.get("self_grounding_score", 0.0)},
                target_name=str(payload.get("sampled_action") or "authenticity"),
            )
        )

    vitality = _mapping(payload.get("vitality_snapshot"))
    if vitality:
        contributions.append(
            _synthetic_global_contribution(
                payload,
                "VitalityEngine",
                "global",
                confidence=_float(vitality.get("vitality_score"), _float(vitality.get("body_energy"), 0.0)),
                trace_reason=str(vitality.get("source") or "vitality modulation"),
                delta_seed={"vitality": vitality.get("body_energy", 0.0)},
                target_name=str(vitality.get("cue") or payload.get("sampled_action") or "vitality"),
            )
        )

    dream_effect = _mapping(payload.get("dream_effect_summary"))
    dream_guard = _mapping(payload.get("dream_guard_summary"))
    if dream_effect or dream_guard:
        contributions.append(
            _synthetic_global_contribution(
                payload,
                "DreamOrchestrator",
                "memory",
                confidence=_float(dream_effect.get("confidence"), _float(dream_guard.get("confidence"), 0.0)),
                trace_reason=str(dream_guard.get("reason") or dream_effect.get("trigger") or "dream recalibration"),
                delta_seed={"dream": dream_effect.get("confidence", dream_guard.get("confidence", 0.0))},
                target_name=str(payload.get("dream_run_id") or payload.get("sampled_action") or "dream"),
            )
        )

    return contributions


def _winner_posterior(payload: dict[str, Any], contributions: list[dict[str, Any]]) -> dict[str, float]:
    baseline = _probability_baseline(payload)
    raw = dict(baseline)
    for contribution in contributions:
        weight = _float(contribution.get("confidence"), 0.0) * max(_float(contribution.get("weight_applied"), 1.0), 0.0)
        for action, delta in contribution.get("delta_logits", {}).items():
            raw[action] = raw.get(action, 0.0) + delta * weight
    gate = _mapping(_mapping(payload.get("distribution_state")).get("gate"))
    if gate:
        for action, gate_value in gate.items():
            raw[action] = max(0.0, raw.get(action, 0.0) * _clip(_float(gate_value), 0.0, 1.0))
    total = sum(max(value, 0.0) for value in raw.values()) or 1.0
    return {action: round(max(value, 0.0) / total, 6) for action, value in sorted(raw.items())}


def _counterfactual_top_peaks(contributions: list[dict[str, Any]], winner_posterior: dict[str, float]) -> list[dict[str, Any]]:
    peaks: list[dict[str, Any]] = []
    for contribution in contributions:
        score = abs(contribution.get("delta_score", 0.0)) * max(contribution.get("confidence", 0.0), 0.0)
        if not score:
            score = max((abs(value) for value in contribution.get("delta_logits", {}).values()), default=0.0)
        peaks.append(
            {
                "module_name": contribution.get("module_name"),
                "layer": contribution.get("layer"),
                "top_action": contribution.get("top_action"),
                "score": round(_float(score), 6),
                "trace_reason": contribution.get("trace_reason", ""),
                "selected": contribution.get("selected", False),
            }
        )
    peaks.sort(key=lambda item: item["score"], reverse=True)
    for peak in peaks[:3]:
        peak["winner_posterior"] = winner_posterior
    return peaks[:5]


def _guard_summary(payload: dict[str, Any], winner_posterior: dict[str, float], contributions: list[dict[str, Any]]) -> dict[str, Any]:
    distribution_state = _mapping(payload.get("distribution_state"))
    plausibility_fail_score = _float(distribution_state.get("plausibility_fail_score"), 0.0)
    blocked_actions = [action for action, value in winner_posterior.items() if value <= 0.0]
    blocked_modules = [
        contribution.get("module_name")
        for contribution in contributions
        if contribution.get("suppression_cause") or contribution.get("selected") is False and contribution.get("confidence", 0.0) < 0.15
    ]
    return {
        "soft_penalty": round(_float(distribution_state.get("guard_penalty"), 0.0), 6),
        "hard_block": bool(_mapping(distribution_state.get("hard_block"))),
        "plausibility_fail_score": round(plausibility_fail_score, 6),
        "blockers": sorted({*blocked_actions, *[str(item) for item in blocked_modules if item]}),
    }


def _arbitration_summary(payload: dict[str, Any], winner_posterior: dict[str, float], contributions: list[dict[str, Any]]) -> dict[str, Any]:
    distribution_state = _mapping(payload.get("distribution_state"))
    conflict = _mapping(distribution_state.get("conflict"))
    layers = Counter(item.get("layer") for item in contributions if item.get("layer"))
    peak_count = sum(1 for value in winner_posterior.values() if value > 0.2)
    if not peak_count:
        peak_count = sum(1 for value in winner_posterior.values() if value > 0.0)
    return {
        "total_conflict_score": round(
            _float(conflict.get("total_score"), _float(conflict.get("score"), 0.0)),
            6,
        ),
        "critical_conflict": bool(conflict.get("critical_conflict", False)),
        "winning_priority": conflict.get("winning_priority"),
        "compromise_template": conflict.get("compromise", {}).get("template"),
        "repair_mode": conflict.get("repair_mode"),
        "repair_stage": conflict.get("repair_state_snapshot", {}).get("stage", "idle"),
        "components": conflict.get("components", {}),
        "peak_count": peak_count,
        "layer_counts": dict(sorted(layers.items())),
    }


def build_probability_field(payload: dict[str, Any]) -> dict[str, Any]:
    probability_field = _mapping(payload.get("probability_field"))
    if probability_field:
        field = deepcopy(probability_field)
        field.setdefault("schema_version", PROBABILITY_FIELD_SCHEMA_VERSION)
        contributions = _list(field.get("contributions"))
        if not contributions:
            contributions = _base_contributions(payload)
            field["contributions"] = contributions
        winner_posterior = _mapping(field.get("winner_posterior")) or _winner_posterior(payload, contributions)
        field["winner_posterior"] = winner_posterior
        field["counterfactual_top_peaks"] = _list(field.get("counterfactual_top_peaks")) or _counterfactual_top_peaks(contributions, winner_posterior)
        field["module_heatmap"] = _mapping(field.get("module_heatmap")) or build_module_heatmap(contributions)
        guard_summary = _guard_summary(payload, winner_posterior, contributions)
        field["guard"] = {**guard_summary, **_mapping(field.get("guard"))}
        arbitration_summary = _arbitration_summary(payload, winner_posterior, contributions)
        field["arbitration"] = {**arbitration_summary, **_mapping(field.get("arbitration"))}
        field["layers"] = _mapping(field.get("layers")) or build_probability_layers(contributions, winner_posterior)
        field.setdefault("context_attn_final", _field_projection(contributions, layer="context", key="attention_bias"))
        field.setdefault("memory_prior_final", _field_projection(contributions, layer="memory", key="posterior"))
        field.setdefault("action_logits_final", _field_projection(contributions, layer="action", key="delta_logits"))
        field.setdefault("token_logits_final", _field_projection(contributions, layer="token", key="delta_logits"))
        return field

    contributions = _base_contributions(payload)
    winner_posterior = _winner_posterior(payload, contributions)
    return {
        "schema_version": PROBABILITY_FIELD_SCHEMA_VERSION,
        "layers": build_probability_layers(contributions, winner_posterior),
        "contributions": contributions,
        "context_attn_final": _field_projection(contributions, layer="context", key="attention_bias"),
        "memory_prior_final": _field_projection(contributions, layer="memory", key="posterior"),
        "action_logits_final": _field_projection(contributions, layer="action", key="delta_logits"),
        "token_logits_final": _field_projection(contributions, layer="token", key="delta_logits"),
        "winner_posterior": winner_posterior,
        "counterfactual_top_peaks": _counterfactual_top_peaks(contributions, winner_posterior),
        "module_heatmap": build_module_heatmap(contributions),
        "guard": _guard_summary(payload, winner_posterior, contributions),
        "arbitration": _arbitration_summary(payload, winner_posterior, contributions),
    }


def build_probability_layers(contributions: list[dict[str, Any]], winner_posterior: dict[str, float]) -> dict[str, Any]:
    layers: dict[str, dict[str, Any]] = {}
    for layer in ALL_LAYERS:
        layers[layer] = _layer_summary(layer, contributions, winner_posterior)
    return layers


def _field_projection(contributions: list[dict[str, Any]], *, layer: str, key: str) -> dict[str, float]:
    merged: dict[str, float] = defaultdict(float)
    for contribution in contributions:
        if contribution.get("layer") != layer:
            continue
        for target, value in _mapping(contribution.get(key)).items():
            merged[str(target)] += _float(value)
    return {name: round(score, 6) for name, score in sorted(merged.items())}


def build_module_heatmap(contributions: list[dict[str, Any]]) -> dict[str, Any]:
    layers: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    modules: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for contribution in contributions:
        layer = str(contribution.get("layer") or "action")
        module = str(contribution.get("module_name") or "unknown")
        score = abs(_float(contribution.get("delta_score"), 0.0)) * max(_float(contribution.get("confidence"), 0.0), 0.0)
        layers[layer]["count"] += 1
        layers[layer]["score"] += score
        modules[module]["count"] += 1
        modules[module]["score"] += score
    return {
        "layers": {layer: {"count": int(metrics["count"]), "score": round(metrics["score"], 6)} for layer, metrics in sorted(layers.items())},
        "modules": {module: {"count": int(metrics["count"]), "score": round(metrics["score"], 6)} for module, metrics in sorted(modules.items())},
    }


def build_legacy_distribution_state(payload: dict[str, Any], field: dict[str, Any]) -> dict[str, Any]:
    distribution_state = deepcopy(_mapping(payload.get("distribution_state")))
    winner_posterior = _mapping(field.get("winner_posterior"))
    distribution_state.setdefault("p_raw", _probability_baseline(payload))
    distribution_state["p_final"] = winner_posterior
    distribution_state.setdefault("gate", {})
    existing_conflict = _mapping(distribution_state.get("conflict"))
    distribution_state["conflict"] = {**existing_conflict, **_mapping(field.get("arbitration"))}
    distribution_state["plausibility_fail_score"] = field.get("guard", {}).get("plausibility_fail_score", distribution_state.get("plausibility_fail_score", 0.0))
    distribution_state["guard_penalty"] = field.get("guard", {}).get("soft_penalty", distribution_state.get("guard_penalty", 0.0))
    return distribution_state


def build_legacy_proposal_summaries(payload: dict[str, Any], field: dict[str, Any]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for contribution in field.get("contributions", []):
        summaries.append(
            {
                "stage": contribution.get("layer"),
                "layer": contribution.get("layer"),
                "agent_name": contribution.get("module_name"),
                "action_name": contribution.get("target_name"),
                "top_action": contribution.get("top_action"),
                "selected": contribution.get("selected", False),
                "confidence": contribution.get("confidence", 0.0),
                "sigma_scale": contribution.get("sigma_scale", 1.0),
                "weight_applied": contribution.get("weight_applied", 1.0),
                "delta_p": contribution.get("delta_logits", {}),
                "delta_logits": contribution.get("delta_logits", {}),
                "attention_bias": contribution.get("attention_bias", {}),
                "soft_mask": contribution.get("soft_mask", {}),
                "hard_mask": contribution.get("hard_mask", {}),
                "posterior": contribution.get("posterior", {}),
                "resample_idx": contribution.get("rank", 0),
                "conflict_score": field.get("arbitration", {}).get("total_conflict_score", 0.0),
                "plausibility_fail_score": field.get("guard", {}).get("plausibility_fail_score", 0.0),
                "tags": contribution.get("tags", []),
                "trace_reason": contribution.get("trace_reason", ""),
                "peak_id": contribution.get("peak_id", ""),
                "suppression_cause": contribution.get("suppression_cause", ""),
                "rank": contribution.get("rank", 0),
            }
        )
    return summaries


def normalize_probability_trace(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(payload)
    field = build_probability_field(normalized)
    normalized["probability_field"] = field
    normalized["probabilistic_contributions"] = field["contributions"]
    normalized["module_heatmap"] = field["module_heatmap"]
    normalized["counterfactual_top_peaks"] = field["counterfactual_top_peaks"]
    normalized["candidate_distribution"] = field["winner_posterior"]
    normalized["distribution_state"] = build_legacy_distribution_state(normalized, field)
    normalized["proposal_summaries"] = build_legacy_proposal_summaries(normalized, field)
    normalized["top_drivers"] = [
        {
            "agent_name": item.get("module_name"),
            "action_name": item.get("top_action"),
            "score": round(abs(_float(item.get("delta_score"), 0.0)) * max(_float(item.get("confidence"), 0.0), 0.0), 6),
            "reason": item.get("trace_reason", ""),
        }
        for item in field.get("contributions", [])[:3]
    ]
    return normalized


def flatten_probability_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    normalized = normalize_probability_trace(payload)
    field = normalized["probability_field"]
    rows: list[dict[str, Any]] = []
    for contribution in field.get("contributions", []):
        rows.append(
            {
                "session_id": normalized["session_id"],
                "recorded_at": normalized["recorded_at"],
                "recorded_date": normalized["recorded_date"],
                "round_id": normalized.get("round_id"),
                "scenario": normalized.get("scenario"),
                "mode": normalized.get("mode"),
                "sampled_action": normalized.get("sampled_action"),
                "contribution_index": contribution.get("contribution_index", 0),
                "stage": contribution.get("layer"),
                "layer": contribution.get("layer"),
                "agent_name": contribution.get("module_name"),
                "module_name": contribution.get("module_name"),
                "action": contribution.get("target_name"),
                "target_name": contribution.get("target_name"),
                "top_action": contribution.get("top_action"),
                "selected": contribution.get("selected", False),
                "confidence": contribution.get("confidence", 0.0),
                "sigma_scale": contribution.get("sigma_scale", 1.0),
                "weight_applied": contribution.get("weight_applied", 1.0),
                "delta_p": contribution.get("delta_score", 0.0),
                "delta_logits_json": _json_blob(contribution.get("delta_logits", {})),
                "delta_energy_json": _json_blob(contribution.get("delta_energy", {})),
                "attention_bias_json": _json_blob(contribution.get("attention_bias", {})),
                "soft_mask_json": _json_blob(contribution.get("soft_mask", {})),
                "hard_mask_json": _json_blob(contribution.get("hard_mask", {})),
                "posterior_json": _json_blob(contribution.get("posterior", {})),
                "counterfactual_json": _json_blob(
                    {
                        "winner_posterior": field.get("winner_posterior", {}),
                        "top_peaks": field.get("counterfactual_top_peaks", []),
                    }
                ),
                "probability_field_json": _json_blob(field),
                "context_attn_json": _json_blob(field.get("context_attn_final", {})),
                "memory_prior_json": _json_blob(field.get("memory_prior_final", {})),
                "action_logits_json": _json_blob(field.get("action_logits_final", {})),
                "token_logits_json": _json_blob(field.get("token_logits_final", {})),
                "winner_posterior_json": _json_blob(field.get("winner_posterior", {})),
                "counterfactual_top_peaks_json": _json_blob(field.get("counterfactual_top_peaks", [])),
                "distribution_state_json": _json_blob(normalized.get("distribution_state", {})),
                "state_snapshot_json": _json_blob(normalized.get("state_snapshot", {})),
                "render_plan_json": _json_blob(normalized.get("render_plan", {})),
                "gate_decisions_json": _json_blob(normalized.get("gate_decisions", [])),
                "rendered_expression_json": _json_blob(normalized.get("rendered_expression", {})),
                "tags_json": _json_blob(contribution.get("tags", [])),
                "resample_count": normalized.get("resample_count", 0),
                "resample_idx": contribution.get("rank", 0),
                "conflict_score": field.get("arbitration", {}).get("total_conflict_score", 0.0),
                "plausibility_fail_score": field.get("guard", {}).get("plausibility_fail_score", 0.0),
                "trace_reason": contribution.get("trace_reason", ""),
                "peak_id": contribution.get("peak_id", ""),
                "suppression_cause": contribution.get("suppression_cause", ""),
            }
        )
    return rows


def build_trace_view(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_probability_trace(payload)
    field = normalized["probability_field"]
    return {
        "round_id": normalized.get("round_id"),
        "session_id": normalized.get("session_id"),
        "subject_id": normalized.get("subject_id"),
        "continuity_nonce": normalized.get("continuity_nonce"),
        "cause_type": normalized.get("cause_type"),
        "boundary_action": normalized.get("boundary_action"),
        "violation_code": normalized.get("violation_code"),
        "deprecation_warning": normalized.get("deprecation_warning"),
        "sampled_action": normalized.get("sampled_action"),
        "scenario": normalized.get("scenario"),
        "mode": normalized.get("mode"),
        "probability_field": field,
        "probabilistic_contributions": field.get("contributions", []),
        "candidate_distribution": normalized.get("candidate_distribution", {}),
        "distribution_state": normalized.get("distribution_state", {}),
        "top_drivers": normalized.get("top_drivers", []),
        "winner_posterior": field.get("winner_posterior", {}),
        "counterfactual_top_peaks": normalized.get("counterfactual_top_peaks", []),
        "module_heatmap": normalized.get("module_heatmap", {}),
        "state_snapshot": normalized.get("state_snapshot", {}),
        "render_plan": normalized.get("render_plan", {}),
        "rendered_expression": normalized.get("rendered_expression", {}),
        "authenticity": normalized.get("authenticity", {}),
        "identity_evolution": normalized.get("identity_evolution", {}),
        "vitality_snapshot": normalized.get("vitality_snapshot", {}),
        "vitality_events": normalized.get("vitality_events", []),
        "long_run_projection": normalized.get("long_run_projection", {}),
        "appraisal_snapshot": normalized.get("appraisal_snapshot", {}),
        "state_delta_before_clip": normalized.get("state_delta_before_clip", {}),
        "state_delta_after_clip": normalized.get("state_delta_after_clip", {}),
        "delta_suppression_reason": normalized.get("delta_suppression_reason", []),
        "run_context": normalized.get("run_context", {}),
        "run_contamination_detected": normalized.get("run_contamination_detected", False),
        "identity_evidence_score": normalized.get("identity_evidence_score", 0.0),
        "identity_trigger_blockers": normalized.get("identity_trigger_blockers", []),
        "temperament_window_summary": normalized.get("temperament_window_summary", {}),
        "storage": {},
    }


def build_contributions_view(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_probability_trace(payload)
    field = normalized["probability_field"]
    conflict = _mapping(normalized.get("distribution_state", {}).get("conflict"))
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for contribution in field.get("contributions", []):
        grouped[str(contribution.get("layer") or "action")].append(
            {
                "module_name": contribution.get("module_name"),
                "target_name": contribution.get("target_name"),
                "top_action": contribution.get("top_action"),
                "confidence": contribution.get("confidence", 0.0),
                "delta_logits": contribution.get("delta_logits", {}),
                "attention_bias": contribution.get("attention_bias", {}),
                "soft_mask": contribution.get("soft_mask", {}),
                "hard_mask": contribution.get("hard_mask", {}),
                "posterior": contribution.get("posterior", {}),
                "trace_reason": contribution.get("trace_reason", ""),
                "selected": contribution.get("selected", False),
            }
        )
    return {
        "round_id": normalized.get("round_id"),
        "sampled_action": normalized.get("sampled_action"),
        "layers": dict(sorted(grouped.items())),
        "repair": {
            "mode": conflict.get("repair_mode"),
            "stage": _mapping(conflict.get("repair_state_snapshot")).get("stage"),
            "post_error_adjustment": _mapping(conflict.get("post_error_adjustment")),
        },
        "module_heatmap": normalized.get("module_heatmap", {}),
        "winner_posterior": field.get("winner_posterior", {}),
        "counterfactual_top_peaks": field.get("counterfactual_top_peaks", []),
        "storage": {},
    }


def build_why_view(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_probability_trace(payload)
    base = build_trace_view(normalized)
    return {
        **base,
        "why": {
            "selected_action": normalized.get("sampled_action"),
            "winner_posterior": normalized.get("candidate_distribution", {}),
            "top_modules": normalized.get("top_drivers", []),
            "guard": normalized.get("probability_field", {}).get("guard", {}),
            "arbitration": normalized.get("probability_field", {}).get("arbitration", {}),
        },
    }


def build_why_not_view(payload: dict[str, Any], action: str) -> dict[str, Any]:
    normalized = normalize_probability_trace(payload)
    field = normalized["probability_field"]
    candidate_distribution = normalized.get("candidate_distribution", {})
    contributing = [
        item
        for item in field.get("contributions", [])
        if action in item.get("delta_logits", {}) or action == item.get("top_action")
    ]
    suppressing = [
        item
        for item in field.get("contributions", [])
        if item.get("suppression_cause")
        or (action in item.get("soft_mask", {}) and _float(item.get("soft_mask", {}).get(action), 1.0) <= 0.0)
        or (action in item.get("hard_mask", {}) and bool(item.get("hard_mask", {}).get(action)))
    ]
    blocked_by = list(field.get("guard", {}).get("blockers", []))
    if action not in candidate_distribution:
        blocked_by.insert(0, "not_proposed")
    if not contributing:
        blocked_by.append("no_positive_support")
    if suppressing:
        blocked_by.append("suppressed_by_module")
    return {
        "round_id": normalized.get("round_id"),
        "action": action,
        "selected_action": normalized.get("sampled_action"),
        "candidate_score": round(_float(candidate_distribution.get(action), 0.0), 6),
        "blocked_by": sorted(set(blocked_by)),
        "supporting_contributions": [
            {
                "module_name": item.get("module_name"),
                "layer": item.get("layer"),
                "confidence": item.get("confidence", 0.0),
                "trace_reason": item.get("trace_reason", ""),
                "delta_logits": item.get("delta_logits", {}),
            }
            for item in contributing[:5]
        ],
        "suppressing_contributions": [
            {
                "module_name": item.get("module_name"),
                "layer": item.get("layer"),
                "confidence": item.get("confidence", 0.0),
                "trace_reason": item.get("trace_reason", ""),
                "suppression_cause": item.get("suppression_cause", ""),
            }
            for item in suppressing[:5]
        ],
        "counterfactual_top_peaks": field.get("counterfactual_top_peaks", []),
        "winner_posterior": candidate_distribution,
        "storage": {},
    }


def build_replay_view(payload: dict[str, Any], *, seed: int = 0) -> dict[str, Any]:
    normalized = normalize_probability_trace(payload)
    field = normalized["probability_field"]
    contributions = list(field.get("contributions", []))
    ablations = []
    for contribution in contributions[:3]:
        action = contribution.get("top_action") or contribution.get("target_name")
        ablations.append(
            {
                "module_name": contribution.get("module_name"),
                "layer": contribution.get("layer"),
                "action": action,
                "delta": round(abs(_float(contribution.get("delta_score"), 0.0)) * max(_float(contribution.get("confidence"), 0.0), 0.0), 6),
                "trace_reason": contribution.get("trace_reason", ""),
            }
        )
    return {
        "round_id": normalized.get("round_id"),
        "original_action": normalized.get("sampled_action"),
        "replayed_action": normalized.get("sampled_action"),
        "counterfactual_distribution": field.get("winner_posterior", {}),
        "counterfactual_top_peaks": field.get("counterfactual_top_peaks", []),
        "ablations": ablations,
        "seed": seed,
        "storage": {},
    }


def build_conflict_timeline(rounds: list[dict[str, Any]]) -> dict[str, Any]:
    points = []
    for payload in rounds:
        normalized = normalize_probability_trace(payload)
        state_snapshot = _mapping(normalized.get("state_snapshot"))
        conflict_learning_state = _mapping(state_snapshot.get("conflict_learning_state"))
        arbitration = normalized.get("probability_field", {}).get("arbitration", {})
        distribution_conflict = _mapping(normalized.get("distribution_state")).get("conflict", {})
        ledger_tail = _list(distribution_conflict.get("repair_ledger_tail"))
        points.append(
            {
                "round_id": normalized.get("round_id"),
                "conflict_score": arbitration.get("total_conflict_score", 0.0),
                "components": arbitration.get("components", {}),
                "winning_priority": arbitration.get("winning_priority"),
                "template": arbitration.get("compromise_template"),
                "critical_conflict": arbitration.get("critical_conflict", False),
                "critical_conflict_streak": distribution_conflict.get("critical_conflict_streak", 0),
                "conflict_safe_mode_owned": distribution_conflict.get("conflict_safe_mode_owned", False),
                "repair_mode": arbitration.get("repair_mode"),
                "repair_stage": arbitration.get("repair_stage", "idle"),
                "last_post_error_adjustment": distribution_conflict.get("post_error_adjustment", {}),
                "repair_ledger_summary": {
                    "entries": len(ledger_tail),
                    "latest_reason": ledger_tail[-1].get("reason", "") if ledger_tail else "",
                },
                "repair_learning": {
                    "adjustment_reasons": conflict_learning_state.get("adjustment_reasons", {}),
                    "last_learning_signal": conflict_learning_state.get("last_learning_signal", {}),
                },
                "peak_count": arbitration.get("peak_count", 0),
                "layer_counts": arbitration.get("layer_counts", {}),
            }
        )
    return {"points": points, "storage": {}}


def build_heatmap_view(rounds: list[dict[str, Any]]) -> dict[str, Any]:
    action_counts: dict[str, int] = defaultdict(int)
    module_counts: dict[str, int] = defaultdict(int)
    layer_counts: dict[str, int] = defaultdict(int)
    layer_scores: dict[str, float] = defaultdict(float)
    for payload in rounds:
        normalized = normalize_probability_trace(payload)
        action_counts[str(normalized.get("sampled_action", "unknown"))] += 1
        field = normalized.get("probability_field", {})
        for contribution in field.get("contributions", []):
            module = str(contribution.get("module_name") or "unknown")
            layer = str(contribution.get("layer") or "action")
            score = abs(_float(contribution.get("delta_score"), 0.0)) * max(_float(contribution.get("confidence"), 0.0), 0.0)
            module_counts[module] += 1
            layer_counts[layer] += 1
            layer_scores[layer] += score
    return {
        "actions": dict(sorted(action_counts.items())),
        "modules": dict(sorted(module_counts.items())),
        "layers": {layer: {"count": count, "score": round(layer_scores[layer], 6)} for layer, count in sorted(layer_counts.items())},
        "storage": {},
    }


def _json_blob(payload: object) -> str:
    import json

    return json.dumps(payload, ensure_ascii=False, sort_keys=True)
