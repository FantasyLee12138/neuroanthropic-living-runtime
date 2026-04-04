from __future__ import annotations

from typing import Any

from nalr.schemas.models import ProbabilisticContribution


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _clip_delta(value: float) -> float:
    return max(-0.35, min(0.35, value))


def _top_reason(action_preferences: dict[str, float], reason: str) -> str:
    if not action_preferences:
        return reason
    action = max(action_preferences, key=action_preferences.get)
    return f"{reason}; top_action={action}" if reason else f"top_action={action}"


def _positive_distribution(values: dict[str, float]) -> dict[str, float]:
    positive = {key: max(float(value), 0.0) for key, value in values.items()}
    total = sum(positive.values()) or 1.0
    return {key: round(value / total, 6) for key, value in positive.items()}


def build_probabilistic_contribution(
    *,
    owner: str,
    prefs: dict[str, float],
    confidence: float,
    sigma_scale: float = 1.0,
    veto: bool = False,
    reason: str = "",
    trace_tags: list[str] | None = None,
    utility_shift: dict[str, float] | None = None,
    state_patch: dict | None = None,
    priority_bucket: str | None = None,
    control_domain: str | None = None,
    gated_actions: list[str] | None = None,
    risk_hints: dict[str, float] | None = None,
    module_name: str | None = None,
    layer: str = "action",
    delta_logits: dict[str, float] | None = None,
    delta_energy: dict[str, float] | None = None,
    attention_bias: dict[str, float] | None = None,
    soft_mask: dict[str, float] | None = None,
    hard_mask: list[str] | None = None,
    posterior: dict[str, float] | None = None,
    confidence_trace: dict[str, float] | None = None,
    trace_reason: str = "",
    trace_scope: str = "action",
    mode_switch: str | None = None,
    memory_ops: list[dict[str, Any]] | None = None,
) -> ProbabilisticContribution:
    clipped = {action: round(_clip_delta(score), 4) for action, score in prefs.items()}
    logits = {action: round(_clip_delta(score), 4) for action, score in (delta_logits or clipped).items()}
    energy = dict(delta_energy or {action: round(score * confidence, 4) for action, score in logits.items()})
    effective_gates = list(gated_actions or [])
    effective_hard_mask = list(hard_mask or ([] if not veto else effective_gates))
    effective_soft_mask = dict(soft_mask or {})
    if effective_gates:
        for action in effective_gates:
            effective_soft_mask.setdefault(action, 0.0 if action in effective_hard_mask else 0.15)
    contribution = ProbabilisticContribution(
        owner=owner,
        module_name=module_name or owner,
        layer=layer,
        level=layer,
        confidence=_clip(confidence, 0.0, 1.0),
        action_preferences=dict(clipped),
        delta_p=dict(clipped),
        delta_logits=logits,
        delta_energy=energy,
        attention_bias=dict(attention_bias or {}),
        soft_mask=effective_soft_mask,
        hard_mask=effective_hard_mask,
        posterior=dict(posterior or _positive_distribution(logits)),
        veto=veto,
        mode_switch=mode_switch,
        utility_shift=dict(utility_shift or {}),
        state_patch=dict(state_patch or {}),
        memory_ops=list(memory_ops or []),
        trace_tags=list(trace_tags or []),
        reason=_top_reason(clipped, reason),
        priority_bucket=priority_bucket or "task_goal",
        control_domain=control_domain or "task",
        gated_actions=effective_gates,
        risk_hints=dict(risk_hints or {}),
        confidence_trace=dict(confidence_trace or {"confidence": round(confidence, 4)}),
        trace_reason=trace_reason or reason,
        trace_scope=trace_scope or layer,
        sigma_scale=_clip(sigma_scale, 0.60, 1.60),
        weight_applied=1.0,
        target_name=max(logits, key=logits.get) if logits else "",
        top_action=max(logits, key=logits.get) if logits else "",
        tags=list(trace_tags or []),
        selected=False,
        suppression_cause="hard_mask" if effective_hard_mask else "",
    )
    return contribution
