from __future__ import annotations

from typing import Any


def _value(source: Any, name: str, default: Any) -> Any:
    return getattr(source, name, default)


def adapt_proposal(proposal: Any, weight: float = 1.0, resample_idx: int = 0) -> dict[str, Any]:
    action_preferences = dict(_value(proposal, "action_preferences", {}) or {})
    top_action = max(action_preferences, key=action_preferences.get) if action_preferences else "respond"
    top_score = round(action_preferences.get(top_action, 0.0) * weight, 4)
    return {
        "agent_name": _value(proposal, "agent_name", "UnknownAgent"),
        "action_preferences": action_preferences,
        "confidence": float(_value(proposal, "confidence", 0.0)),
        "sigma_scale": float(_value(proposal, "sigma_scale", 1.0)),
        "veto": bool(_value(proposal, "veto", False)),
        "trace_tags": list(_value(proposal, "trace_tags", []) or []),
        "reason": str(_value(proposal, "reason", "")),
        "provider": str(_value(proposal, "provider", "upstream_contract")),
        "model": str(_value(proposal, "model", "pending-merge")),
        "latency_ms": int(_value(proposal, "latency_ms", 0)),
        "top_action": top_action,
        "top_score": top_score,
        "weight_applied": weight,
        "resample_idx": resample_idx,
    }


def adapt_skill_spec(spec: Any) -> dict[str, Any]:
    return {
        "name": _value(spec, "name", ""),
        "owner_module": _value(spec, "owner_module", ""),
        "input_schema": dict(_value(spec, "input_schema", {}) or {}),
        "output_schema": dict(_value(spec, "output_schema", {}) or {}),
        "timeout_ms": int(_value(spec, "timeout_ms", 0)),
        "cost_class": str(_value(spec, "cost_class", "L")),
        "failure_policy": str(_value(spec, "failure_policy", "return_neutral")),
        "trace_tags": list(_value(spec, "trace_tags", []) or []),
    }
