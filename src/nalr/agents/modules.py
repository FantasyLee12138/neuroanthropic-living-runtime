from __future__ import annotations

from dataclasses import dataclass

from nalr.agents.probability_field import ProbabilisticContribution, build_probabilistic_contribution
from nalr.runtime.dynamics import bounded_drift_delta, smooth_resource_biases
from nalr.schemas.models import (
    ActionCandidate,
    ActionDistributionState,
    ConflictAssessment,
    ConflictComponents,
    ConflictPostErrorAdjustment,
    ConflictRepairLedgerEntry,
    ConflictRepairState,
    ConflictResolution,
    RoundEvent,
    RuntimeState,
    to_dict,
)


PRIORITY_ORDER = (
    "body_safety",
    "budget_overload",
    "relation_boundary",
    "task_goal",
    "immediate_desire",
    "roaming",
)

OWNER_PRIORITY_BUCKET = {
    "BodyStateAgent": "body_safety",
    "EmotionAgent": "body_safety",
    "ResourceAgent": "budget_overload",
    "RelationshipAgent": "relation_boundary",
    "PerspectiveModel": "relation_boundary",
    "PFCAgent": "task_goal",
    "SalienceAgent": "task_goal",
    "ValueAgent": "task_goal",
    "HippocampusAgent": "task_goal",
    "HabitAgent": "task_goal",
    "UnconsciousAgent": "task_goal",
    "CerebellarPredictor": "task_goal",
    "DesireAgent": "immediate_desire",
    "DMNAgent": "roaming",
    "ConflictMonitorAgent": "task_goal",
    "ThalamusAttentionAgent": "task_goal",
    "BehaviorPlausibilityGuard": "body_safety",
    "ForcedModeSwitch": "task_goal",
    "OutputGate": "task_goal",
}

OWNER_CONTROL_DOMAIN = {
    "BodyStateAgent": "body",
    "EmotionAgent": "body",
    "ResourceAgent": "resource",
    "RelationshipAgent": "relation",
    "PerspectiveModel": "relation",
    "PFCAgent": "task",
    "SalienceAgent": "task",
    "ValueAgent": "task",
    "HippocampusAgent": "task",
    "HabitAgent": "task",
    "UnconsciousAgent": "task",
    "CerebellarPredictor": "task",
    "DesireAgent": "desire",
    "DMNAgent": "dmn",
    "ConflictMonitorAgent": "task",
    "ThalamusAttentionAgent": "context",
    "BehaviorPlausibilityGuard": "guard",
    "ForcedModeSwitch": "guard",
    "OutputGate": "render",
}

TEMPLATE_BY_PRIORITY = {
    "body_safety": "body_first",
    "budget_overload": "budget_first",
    "relation_boundary": "relation_first",
    "task_goal": "task_first",
    "immediate_desire": "body_first",
    "roaming": "budget_first",
}

DEFAULT_BLOCKS = {
    "body_safety": {"connect", "wander"},
    "budget_overload": {"plan", "connect", "wander"},
    "relation_boundary": {"connect"},
    "task_goal": {"wander"},
    "immediate_desire": {"plan", "connect"},
    "roaming": {"plan", "connect"},
}

TEMPLATE_ACTION_SCALES = {
    "body_first": {"rest": 1.30, "respond": 1.05, "plan": 0.35, "connect": 0.0, "wander": 0.0},
    "relation_first": {"clarify": 1.20, "respond": 1.10, "connect": 0.20, "plan": 0.55, "wander": 0.35},
    "task_first": {"plan": 1.25, "respond": 1.10, "recall": 1.05, "rest": 0.55, "connect": 0.40, "wander": 0.0},
    "budget_first": {"respond": 1.15, "recall": 1.05, "rest": 1.10, "plan": 0.35, "connect": 0.0, "wander": 0.0},
}

SALIENCE_TASK_TOKENS = (
    "help",
    "remember",
    "urgent",
    "总结",
    "检查",
    "规划",
    "仓库",
    "代码",
    "文件",
    "测试",
    "修复",
    "分析",
    "summarize",
    "code",
    "file",
    "test",
    "fix",
    "analyze",
)


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _clip_delta(value: float) -> float:
    return _clip(value, -0.35, 0.35)


def _safe_mode_delta(before: bool, after: bool) -> str:
    if before == after:
        return "unchanged"
    return "entered" if after else "exited"


def _top_reason(action_preferences: dict[str, float], reason: str) -> str:
    if not action_preferences:
        return reason
    action = max(action_preferences, key=action_preferences.get)
    return f"{reason}; top_action={action}"


def compute_salience_signal(event: RoundEvent, scenario: dict | None = None) -> float:
    scenario = scenario or {}
    content = event.content.lower()
    task_signal = 0.0
    if any(token in content for token in SALIENCE_TASK_TOKENS):
        task_signal = 0.70 + min(float(scenario.get("pfc_base_share", 0.2)), 1.0) * 0.18
    affect_signal = abs(event.valence) * 0.18
    return round(_clip(0.10 + task_signal + affect_signal), 4)


def _bundle(
    owner: str,
    prefs: dict[str, float],
    *,
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
    layer: str = "action",
    delta_logits: dict[str, float] | None = None,
    delta_energy: dict[str, float] | None = None,
    attention_bias: dict[str, float] | None = None,
    soft_mask: dict[str, float] | None = None,
    hard_mask: list[str] | None = None,
    posterior: dict[str, float] | None = None,
    confidence_trace: dict[str, float] | None = None,
    trace_reason: str = "",
    trace_scope: str | None = None,
) -> ProbabilisticContribution:
    clipped = {action: _clip_delta(score) for action, score in prefs.items()}
    return build_probabilistic_contribution(
        owner=owner,
        prefs=clipped,
        confidence=_clip(confidence, 0.0, 1.0),
        sigma_scale=sigma_scale,
        veto=veto,
        reason=reason,
        trace_tags=trace_tags,
        utility_shift=utility_shift,
        state_patch=state_patch,
        priority_bucket=priority_bucket or OWNER_PRIORITY_BUCKET.get(owner, "task_goal"),
        control_domain=control_domain or OWNER_CONTROL_DOMAIN.get(owner, "task"),
        gated_actions=gated_actions,
        risk_hints=risk_hints,
        module_name=owner,
        layer=layer,
        delta_logits=delta_logits or clipped,
        delta_energy=delta_energy or {action: round(score * confidence, 4) for action, score in clipped.items()},
        attention_bias=attention_bias,
        soft_mask=soft_mask,
        hard_mask=hard_mask,
        posterior=posterior,
        confidence_trace=confidence_trace,
        trace_reason=trace_reason or reason,
        trace_scope=trace_scope or layer,
    )


@dataclass
class BaseAgent:
    name: str

    def run_skill(self, skill_name: str, *args, **kwargs):
        return getattr(self, skill_name)(*args, **kwargs)

    def contribute(self, *args, **kwargs) -> ProbabilisticContribution:
        if args and isinstance(args[0], RoundEvent):
            event, state, scenario, context = args[:4]
            return self.propose(event, state, scenario, context)
        if {"event", "state", "scenario", "context"} <= set(kwargs):
            return self.propose(kwargs["event"], kwargs["state"], kwargs["scenario"], kwargs["context"])
        raise TypeError(f"{self.name}.contribute() expects an event/state/scenario/context call")

    def propose(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> ProbabilisticContribution:
        raise NotImplementedError


class BodyStateAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="BodyStateAgent")

    def update_body_state(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> dict:
        appraisal = dict(context.get("appraisal", {}))
        fatigue_push = float(appraisal.get("fatigue_push", 0.0))
        cognitive_load = float(appraisal.get("cognitive_load", 0.0))
        repair_drag = 0.04 if state.repair_mode else 0.0
        next_energy = _clip(state.body_energy - fatigue_push * 0.03 - cognitive_load * 0.02 - repair_drag)
        return {"state_patch": {"body_energy": next_energy}}

    def compute_body_bias(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> ProbabilisticContribution:
        energy = _clip(state.body_energy)
        prefs = {"respond": 0.10}
        sigma_scale = 1.0
        if energy < 0.45:
            prefs["rest"] = 0.18 + (0.45 - energy) * 0.40
            sigma_scale = 1.15
        return _bundle(
            self.name,
            prefs,
            confidence=0.64,
            sigma_scale=sigma_scale,
            reason=f"energy={energy:.2f}",
            trace_tags=["body"],
            layer="global",
        )

    def apply_body_veto(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> dict:
        energy = _clip(state.body_energy + event.energy_delta)
        gated_actions = ["connect"] if energy < 0.15 else []
        return {"veto": False, "gated_actions": gated_actions}

    def propose(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> ProbabilisticContribution:
        return self.compute_body_bias(event, state, scenario, context)


class RelationshipAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="RelationshipAgent")

    def score_closeness(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> dict:
        return {"score": context["closeness"]}

    def compute_boundary_gate(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> dict:
        closeness = context["closeness"]
        boundary_level = _clip(0.8 - closeness + max(0.0, -event.valence) * 0.2)
        gate = {"connect": _clip(1.0 - boundary_level * 0.45, 0.0, 1.0)}
        return {"gate": gate, "boundary_level": boundary_level}

    def update_relation_trace(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> dict:
        return {"state_delta": {"relation_target": event.target}}

    def propose(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> ProbabilisticContribution:
        closeness = context["closeness"]
        prefs = {"respond": 0.04 + scenario.get("relationship_weight", 0.1) * closeness * 0.3}
        if closeness > 0.55 and event.target:
            prefs["connect"] = 0.08 + closeness * 0.18
        return _bundle(
            self.name,
            prefs,
            confidence=0.58,
            reason=f"closeness={closeness:.2f}",
            trace_tags=["relationship"],
            layer="context",
            attention_bias={"closeness": round(closeness, 4)},
        )


class DesireAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="DesireAgent")

    def score_immediate_reward(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> dict:
        content = event.content.lower()
        score = 0.0
        if any(token in content for token in {"quick", "easy", "coffee", "break", "rest"}):
            score = 0.45
        return {"score": score}

    def compute_effort_avoidance(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> ProbabilisticContribution:
        content = event.content.lower()
        prefs = {}
        if any(token in content for token in {"quick", "easy", "coffee", "break", "rest"}):
            prefs["rest"] = 0.12 + (1.0 - state.body_energy) * 0.18
            prefs["respond"] = 0.06
        return _bundle(self.name, prefs, confidence=0.54, reason="comfort seeking", trace_tags=["desire"], layer="action")

    def suggest_low_cost_action(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> dict:
        return {"action_hint": "rest" if state.body_energy < 0.4 else "respond"}

    def propose(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> ProbabilisticContribution:
        return self.compute_effort_avoidance(event, state, scenario, context)


class EmotionAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="EmotionAgent")

    def update_affect_state(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> dict:
        appraisal = dict(context.get("appraisal", {}))
        inertia = (state.mood - 0.55) * 0.08
        residue_pull = state.affect_residue * 0.06
        next_mood = _clip(state.mood + event.valence * 0.08 + inertia * 0.2 - residue_pull * 0.15)
        session_metadata = dict(state.session_metadata)
        session_metadata["last_appraisal_band"] = appraisal.get("appraisal_band", "muted")
        return {
            "state_patch": {
                "mood": next_mood,
                "session_metadata": session_metadata,
            }
        }

    def compute_affect_bias(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> ProbabilisticContribution:
        prefs = {}
        sigma_scale = 1.0
        if event.valence <= -0.25:
            prefs["clarify"] = 0.10 + abs(event.valence) * 0.14
            prefs["rest"] = 0.08 + abs(event.valence) * 0.12
            sigma_scale = 1.12
        elif event.valence >= 0.25:
            prefs["connect"] = 0.10 + event.valence * 0.12
        return _bundle(
            self.name,
            prefs,
            confidence=0.60,
            sigma_scale=sigma_scale,
            reason=f"valence={event.valence:.2f}",
            trace_tags=["emotion"],
            layer="global",
            attention_bias={"valence": round(event.valence, 4)},
        )

    def trigger_affect_veto(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> dict:
        return {"veto": event.valence < -0.95}

    def propose(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> ProbabilisticContribution:
        return self.compute_affect_bias(event, state, scenario, context)


class DMNAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="DMNAgent")

    def sample_dmn_intrusion(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> ProbabilisticContribution:
        prefs = {}
        if state.mode in {"idle", "interactive"}:
            prefs["wander"] = scenario.get("dmn_weight", 0.05) * (4.8 if state.mode == "idle" else 1.2)
        confidence = 0.72 if state.mode == "idle" else 0.46
        return _bundle(self.name, prefs, confidence=confidence, reason="background drift", trace_tags=["dmn"], layer="global")

    def score_rumination_pull(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> dict:
        return {"score": 0.18 if state.mode == "idle" else 0.05}

    def select_spontaneous_topic(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> dict:
        return {"topic_hint": context.get("cue") or "idle-thought"}

    def propose(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> ProbabilisticContribution:
        return self.sample_dmn_intrusion(event, state, scenario, context)


class PFCAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="PFCAgent")

    def fallback_generate_candidates(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> ProbabilisticContribution:
        content = event.content.lower()
        task_terms = (
            "plan",
            "help",
            "summary",
            "summarize",
            "check",
            "inspect",
            "repo",
            "code",
            "file",
            "test",
            "fix",
            "read",
            "search",
            "analyze",
            "analyse",
            "总结",
            "检查",
            "规划",
            "仓库",
            "代码",
            "文件",
            "目录",
            "测试",
            "修复",
            "读取",
            "搜索",
            "分析",
            "定位",
            "结构",
            "src",
            "tests",
        )
        memory_terms = ("remember", "recall", "记住", "记下", "提醒")
        social_terms = ("你好", "您好", "hello", "hi", "hey", "在吗", "你是谁", "why did you answer", "为什么这样回答")
        prefs = {"respond": 0.08}
        scarcity = float(state.resource_state.get("scarcity_index", 0.0))
        if any(token in content for token in social_terms):
            prefs["respond"] = 0.14
            prefs["plan"] = max(prefs.get("plan", 0.0), 0.0)
        if any(token in content for token in task_terms):
            prefs["plan"] = 0.18 + scenario.get("pfc_base_share", 0.2)
        if any(token in content for token in memory_terms) or context.get("cue"):
            prefs["recall"] = max(prefs.get("recall", 0.0), 0.18 + context.get("recall_strength", 0.0) * 0.15)
        if state.resource_state.get("resource_mode") == "starvation" or scarcity >= 0.75:
            prefs["plan"] = min(prefs.get("plan", 0.0), 0.06)
            prefs["short_reply"] = max(prefs.get("short_reply", 0.0), 0.08 + scarcity * 0.10)
        return _bundle(self.name, prefs, confidence=0.84, sigma_scale=0.92, reason="deliberate planner", trace_tags=["pfc"], layer="action")

    def generate_candidates(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> ProbabilisticContribution:
        return self.fallback_generate_candidates(event, state, scenario, context)

    def estimate_plan_depth(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> dict:
        content = event.content.lower()
        if state.resource_state.get("resource_mode") == "starvation":
            return {"plan_depth": 1}
        depth = 2 if "plan" in content or "help" in content else 1
        return {"plan_depth": depth}

    def bind_working_memory(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> dict:
        return {"wm_update": {"cue": context.get("cue"), "focus": state.focus}}

    def propose(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> ProbabilisticContribution:
        return self.generate_candidates(event, state, scenario, context)


class UnconsciousAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="UnconsciousAgent")

    def load_temperament(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> dict:
        baseline = dict(state.temperament_state.get("baseline", {}))
        return {"baseline": baseline}

    def compute_trait_bias(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> ProbabilisticContribution:
        current = dict(state.temperament_state.get("current", {}))
        attachment_need = _clip(float(current.get("attachment_need", 0.5)))
        boundary_softness = _clip(float(current.get("boundary_softness", 0.5)))
        sensitivity = _clip(float(current.get("sensitivity", 0.5)))
        patience = _clip(float(current.get("patience", 0.5)))
        extraversion = _clip(float(current.get("extraversion", 0.5)))
        cognitive_bandwidth = _clip(float(current.get("cognitive_bandwidth", 0.5)))
        desire_priority = _clip(float(current.get("desire_priority", attachment_need)))
        approach_tension = _clip(attachment_need * (1 - boundary_softness), 0.0, 1.0)
        reactive_fragility = _clip(sensitivity * (1 - patience), 0.0, 1.0)
        social_output_capacity = _clip(extraversion * cognitive_bandwidth, 0.0, 1.0)
        prefs = {"respond": 0.02 + social_output_capacity * 0.10}
        if approach_tension > 0.12:
            prefs["connect"] = 0.02 + approach_tension * 0.08
        if reactive_fragility > 0.10:
            prefs["clarify"] = 0.02 + reactive_fragility * 0.08
        if desire_priority > 0.55:
            prefs["plan"] = 0.01 + desire_priority * 0.05
        return _bundle(self.name, prefs, confidence=0.42, reason="temperament baseline", trace_tags=["trait"], layer="global")

    def apply_chronic_shift(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> dict:
        baseline = dict(state.temperament_state.get("baseline", {}))
        drift = {
            key: float(value)
            for key, value in dict(state.temperament_state.get("drift", {})).items()
        }
        current = dict(state.temperament_state.get("current", {}))
        correction_window = {
            key: [int(item) for item in values]
            for key, values in dict(state.temperament_state.get("correction_window", {})).items()
        }
        freeze_until_round = {
            key: int(value)
            for key, value in dict(state.temperament_state.get("freeze_until_round", {})).items()
        }
        recover_rates = {
            "patience": 0.03,
            "sensitivity": 0.02,
            "boundary_softness": 0.025,
            "desire_priority": 0.02,
        }
        chronic = dict(context.get("chronic_signal", {}))
        stress_load = _clip(
            state.affect_residue * 0.45
            + max(0.0, 0.45 - state.body_energy) * 0.50
            + float(state.resource_state.get("scarcity_index", 0.0)) * 0.35,
            0.0,
            1.0,
        )
        repeated_rejection = int(chronic.get("repeated_rejection_count", 0) or 0)
        repeated_confirmation = int(chronic.get("repeated_confirmation_count", 0) or 0)
        resource_stress_span = int(chronic.get("resource_stress_span", 0) or 0)
        identity_repeat = int(chronic.get("identity_cue_repeat_count", 0) or 0)
        relation_trend = float(chronic.get("relation_trend", 0.0) or 0.0)
        avg_negative_valence = float(chronic.get("avg_negative_valence", 0.0) or 0.0)
        avg_positive_valence = float(chronic.get("avg_positive_valence", 0.0) or 0.0)
        updated_drift: dict[str, float] = {}
        updated_current: dict[str, float] = {}
        correction_events: list[dict[str, float | str | int]] = []
        drift_diagnostics: dict[str, dict[str, float | str | bool]] = {}
        for key, baseline_value in baseline.items():
            old_drift = float(drift.get(key, 0.0))
            if state.round_count < freeze_until_round.get(key, 0):
                new_drift = old_drift
                drift_diagnostics[key] = {
                    "window_support": 0.0,
                    "recover_rate": 0.0,
                    "effective_push": 0.0,
                    "delta_reason": "frozen_after_clip",
                    "suppressed_by_clip": False,
                    "freeze_reason": "freeze_window_active",
                }
            else:
                reasons: list[str] = []
                window_support = 0.0
                effective_push = event.valence * 0.012
                if key == "sensitivity":
                    effective_push += repeated_rejection * 0.006 + avg_negative_valence * 0.04 + max(0.0, -relation_trend) * 0.05
                    window_support = repeated_rejection * 0.04 + avg_negative_valence * 0.4
                    reasons.append("repeated_rejection")
                elif key == "boundary_softness":
                    effective_push -= repeated_rejection * 0.006 + avg_negative_valence * 0.03
                    window_support = repeated_rejection * 0.04 + avg_negative_valence * 0.3
                    reasons.append("repeated_rejection")
                elif key == "attachment_need":
                    effective_push += repeated_confirmation * 0.005 + avg_positive_valence * 0.035 + identity_repeat * 0.002
                    window_support = repeated_confirmation * 0.035 + avg_positive_valence * 0.35
                    reasons.append("warm_confirmation")
                elif key == "patience":
                    effective_push -= repeated_rejection * 0.004 + stress_load * 0.02
                    window_support = repeated_rejection * 0.025 + stress_load * 0.2
                    reasons.append("sustained_friction")
                elif key == "cognitive_bandwidth":
                    effective_push -= resource_stress_span * 0.007 + stress_load * 0.025
                    window_support = resource_stress_span * 0.045 + stress_load * 0.2
                    reasons.append("resource_pressure")
                elif key == "extraversion":
                    effective_push -= resource_stress_span * 0.004 + max(0.0, -relation_trend) * 0.01
                    window_support = resource_stress_span * 0.03 + max(0.0, -relation_trend) * 0.2
                    reasons.append("withdrawal_pressure")
                elif key == "desire_priority":
                    effective_push += identity_repeat * 0.002 + avg_positive_valence * 0.01 - resource_stress_span * 0.002
                    window_support = identity_repeat * 0.02 + avg_positive_valence * 0.1
                    reasons.append("identity_repetition")
                recover_rate = recover_rates.get(key, 0.02)
                new_drift = bounded_drift_delta(
                    previous_drift=old_drift,
                    push=effective_push,
                    recover_rate=recover_rate,
                    max_abs=0.25,
                )
                drift_diagnostics[key] = {
                    "window_support": round(window_support, 4),
                    "recover_rate": round(recover_rate, 4),
                    "effective_push": round(effective_push, 4),
                    "delta_reason": "+".join(reasons) if reasons else "baseline_recovery",
                    "suppressed_by_clip": False,
                    "freeze_reason": "",
                }
            raw_value = float(baseline_value) + new_drift
            clipped_value = _clip(raw_value)
            updated_drift[key] = round(new_drift, 4)
            updated_current[key] = round(clipped_value, 4)
            window = [value for value in correction_window.get(key, []) if state.round_count - int(value) <= 20]
            if clipped_value != raw_value:
                window.append(state.round_count)
                correction_events.append({"parameter": key, "round": state.round_count, "raw": round(raw_value, 4)})
                drift_diagnostics[key]["suppressed_by_clip"] = True
                drift_diagnostics[key]["freeze_reason"] = "clip_guard"
                if len(window) > 3:
                    freeze_until_round[key] = state.round_count + 10
                    drift_diagnostics[key]["freeze_reason"] = "repeated_clip"
            elif not drift_diagnostics[key].get("freeze_reason"):
                drift_diagnostics[key]["freeze_reason"] = "none"
            correction_window[key] = window
        return {
            "state_patch": {
                "temperament_state": {
                    "baseline": baseline,
                    "drift": updated_drift,
                    "current": updated_current,
                    "correction_window": correction_window,
                    "freeze_until_round": freeze_until_round,
                    "last_correction_events": correction_events,
                    "drift_diagnostics": drift_diagnostics,
                }
            }
        }

    def propose(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> ProbabilisticContribution:
        return self.compute_trait_bias(event, state, scenario, context)


class HippocampusAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="HippocampusAgent")

    def encode_episode(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> dict:
        return {"writeback": {"cue": context.get("cue"), "round": state.round_count}}

    def retrieve_by_cue(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> dict:
        recall_strength = context.get("recall_strength", 0.0)
        recall_set = []
        if context.get("cue"):
            recall_set.append({"cue": context["cue"], "strength": recall_strength, "detail": recall_strength >= context.get("detail_threshold", 0.5)})
        return {"recall_set": recall_set}

    def apply_cue_weighted_decay(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> dict:
        return {"scalar": max(0.0, 1.0 - context.get("round_gap", 0) * 0.03)}

    def compute_memory_interference(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> dict:
        return {"scalar": _clip(context.get("interference", 0.0), 0.0, 1.0)}

    def fallback_to_gist_when_trace_weak(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> dict:
        return {"gist": {"cue": context.get("cue"), "strength": context.get("recall_strength", 0.0)}}

    def propose(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> ProbabilisticContribution:
        recall_strength = context.get("recall_strength", 0.0)
        prefs = {}
        if recall_strength > 0:
            prefs["recall"] = 0.10 + recall_strength * 0.22
        return _bundle(
            self.name,
            prefs,
            confidence=0.66,
            reason=f"recall={recall_strength:.2f}",
            trace_tags=["memory"],
            layer="memory",
            attention_bias={"recall_strength": round(float(recall_strength), 4)},
        )


class SalienceAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="SalienceAgent")

    def score_salience(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> ProbabilisticContribution:
        salience_signal = compute_salience_signal(event, scenario)
        prefs = {}
        if salience_signal >= 0.55:
            prefs["plan"] = 0.10 + scenario.get("pfc_base_share", 0.2) * 0.22
        if abs(event.valence) >= 0.3:
            prefs["clarify"] = max(prefs.get("clarify", 0.0), 0.08 + abs(event.valence) * 0.10)
        if not prefs:
            prefs["respond"] = 0.08
        return _bundle(
            self.name,
            prefs,
            confidence=0.64,
            reason=f"salience={salience_signal:.2f}",
            trace_tags=["salience"],
            layer="context",
            attention_bias={"salience": round(salience_signal, 4)},
        )

    def switch_mode(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> dict:
        if abs(event.valence) >= 0.6:
            return {"mode_flag": "interactive"}
        return {"mode_flag": state.mode}

    def interrupt_current_focus(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> dict:
        return {"interrupt": abs(event.valence) >= 0.5}

    def promote_event_to_workspace(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> dict:
        return {"workspace_delta": {"focus_hint": "plan" if "help" in event.content.lower() else state.focus}}

    def propose(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> ProbabilisticContribution:
        return self.score_salience(event, state, scenario, context)


class HabitAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="HabitAgent")

    def compute_feedback_decay(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> dict:
        return {"scalar": max(0.0, 1.0 - context.get("habit_strength", 0.0) * 0.1)}

    def update_habit_strength(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> dict:
        return {"state_update": {"habit_strength": context.get("habit_strength", 0.0)}}

    def suggest_default_action(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> ProbabilisticContribution:
        habit_strength = context.get("habit_strength", 0.0)
        prefs = {}
        if habit_strength > 0:
            prefs["respond"] = 0.06 + habit_strength * scenario.get("habit_weight", 0.1)
            prefs["recall"] = 0.04 + habit_strength * 0.12
        return _bundle(self.name, prefs, confidence=0.60, reason=f"habit={habit_strength:.2f}", trace_tags=["habit"], layer="action")

    def estimate_override_cost(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> dict:
        return {"cost": _clip(context.get("habit_strength", 0.0) * 0.4, 0.0, 1.0)}

    def propose(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> ProbabilisticContribution:
        return self.suggest_default_action(event, state, scenario, context)


class ValueAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="ValueAgent")

    def estimate_subjective_value(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> dict:
        content = event.content.lower()
        scores = {"respond": 0.08}
        if any(
            token in content
            for token in ("plan", "summary", "summarize", "check", "inspect", "repo", "code", "file", "总结", "检查", "规划", "仓库", "代码", "文件", "结构")
        ):
            scores["plan"] = 0.12 + scenario.get("pfc_base_share", 0.2) * 0.22
        if event.target and context.get("closeness", 0.5) > 0.55:
            scores["connect"] = 0.08 + context["closeness"] * 0.10
        return {"scores": scores}

    def discount_delayed_reward(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> dict:
        return {"scalar": 0.12 if scenario.get("delay_tolerance", 0.1) < 0.15 else 0.06}

    def price_social_cost(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> dict:
        return {"scalar": _clip((1.0 - context.get("closeness", 0.5)) * 0.25, 0.0, 1.0)}

    def score_uncertainty_penalty(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> dict:
        return {"penalty": _clip(state.action_ci.get("respond", 0.25), 0.0, 1.0)}

    def propose(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> ProbabilisticContribution:
        scores = self.estimate_subjective_value(event, state, scenario, context)["scores"]
        return _bundle(
            self.name,
            scores,
            confidence=0.59,
            reason="subjective value re-rank",
            trace_tags=["value"],
            layer="action",
        )


class ResourceAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="ResourceAgent")

    def compute_scarcity_index(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> dict:
        burn_rate_ratio = _clip(float(context.get("burn_rate_ratio", context.get("recent_burn_rate", 0.0))), 0.0, 1.0)
        low_balance_ratio = _clip(float(context.get("low_balance_ratio", 1.0 - state.budget_remaining)), 0.0, 1.0)
        queue_pressure = _clip(float(context.get("queue_pressure", 0.0)), 0.0, 1.0)
        latency_pressure = _clip(float(context.get("latency_pressure", 0.0)), 0.0, 1.0)
        scarcity = _clip(
            0.40 * burn_rate_ratio
            + 0.30 * low_balance_ratio
            + 0.20 * queue_pressure
            + 0.10 * latency_pressure,
            0.0,
            1.0,
        )
        return {"scalar": scarcity}

    def map_budget_to_bias(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> ProbabilisticContribution:
        scarcity = self.compute_scarcity_index(event, state, scenario, context)["scalar"]
        bias = smooth_resource_biases(scarcity)
        body_hunger_bias = bias["body_hunger_bias"]
        effort_avoidance_bias = bias["effort_avoidance_bias"]
        deliberation_compress = bias["deliberation_compress"]
        rumination_bias = bias["rumination_bias"]
        action_shrink_scale = bias["action_shrink_scale"]
        prefs = {
            "respond": 0.05 + 0.08 * effort_avoidance_bias,
            "recall": 0.04 * effort_avoidance_bias,
            "plan": -0.25 * effort_avoidance_bias,
            "wander": -0.20 * action_shrink_scale + 0.05 * rumination_bias,
        }
        if scarcity > 0.45:
            prefs["rest"] = 0.06 + 0.18 * body_hunger_bias
        if scarcity > 0.50:
            prefs["short_reply"] = 0.06 + scarcity * 0.22
        return _bundle(
            self.name,
            prefs,
            confidence=0.58,
            reason=f"budget={state.budget_remaining:.2f}",
            trace_tags=["resource"],
            risk_hints={
                "scarcity_pressure": bias["scarcity_pressure"],
                "body_hunger_bias": round(body_hunger_bias, 4),
                "effort_avoidance_bias": round(effort_avoidance_bias, 4),
                "deliberation_compress": round(deliberation_compress, 4),
                "rumination_bias": round(rumination_bias, 4),
                "action_shrink_scale": round(action_shrink_scale, 4),
            },
            layer="global",
        )

    def suggest_resource_mode(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> dict:
        scarcity = self.compute_scarcity_index(event, state, scenario, context)["scalar"]
        if scarcity >= 0.75:
            return {"mode_flag": "safe", "resource_mode": "starvation"}
        if scarcity >= 0.50:
            return {"mode_flag": state.mode, "resource_mode": "scarce"}
        if scarcity >= 0.25:
            return {"mode_flag": state.mode, "resource_mode": "stable"}
        return {"mode_flag": state.mode, "resource_mode": "abundant"}

    def propose(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> ProbabilisticContribution:
        return self.map_budget_to_bias(event, state, scenario, context)


class CerebellarPredictor(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="CerebellarPredictor")

    def predict_next_state(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> dict:
        return {"state_prediction": {"focus": state.last_action}}

    def compute_prediction_error(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> dict:
        return {"error": 0.0 if state.last_action == state.focus else 0.1}

    def smooth_response_timing(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> dict:
        return {"timing_delta": {"reply_delay_bias": -0.02 if state.last_action == "plan" else 0.0}}

    def micro_adjust_action(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> ProbabilisticContribution:
        prefs = {}
        if state.last_action == "clarify":
            prefs["clarify"] = 0.04
        elif state.last_action == "plan":
            prefs["plan"] = 0.04
        return _bundle(self.name, prefs, confidence=0.35, reason="micro smoothing", trace_tags=["timing"], layer="token")

    def propose(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> ProbabilisticContribution:
        return self.micro_adjust_action(event, state, scenario, context)


class PerspectiveModel(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="PerspectiveModel")

    def fallback_infer_other_state(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> dict:
        return {"state_hypothesis": {"confused": "clarify" in event.content.lower(), "closeness": context.get("closeness", 0.5)}}

    def infer_other_state(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> dict:
        return self.fallback_infer_other_state(event, state, scenario, context)

    def fallback_simulate_other_reaction(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> dict:
        closeness = context.get("closeness", 0.5)
        return {"reaction_hypothesis": {"positive": closeness > 0.6, "risk": 1 - closeness}}

    def simulate_other_reaction(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> dict:
        return self.fallback_simulate_other_reaction(event, state, scenario, context)

    def estimate_misunderstanding_risk(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> dict:
        risk = _clip((1.0 - context.get("closeness", 0.5)) * 0.5 + (0.15 if "clarify" in event.content.lower() else 0.0), 0.0, 1.0)
        return {"risk": risk}

    def adjust_social_interpretation(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> ProbabilisticContribution:
        prefs = {}
        if scenario.get("relationship_weight", 0.0) >= 0.22 and event.target:
            prefs["clarify"] = 0.05 + context["closeness"] * 0.08
        return _bundle(
            self.name,
            prefs,
            confidence=0.49,
            reason="social interpretation",
            trace_tags=["perspective"],
            layer="context",
            attention_bias={"closeness": round(float(context.get("closeness", 0.5)), 4)},
        )

    def propose(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> ProbabilisticContribution:
        return self.adjust_social_interpretation(event, state, scenario, context)


class ConflictMonitorAgent(BaseAgent):
    def __init__(
        self,
        *,
        conflict_high: float = 0.75,
        conflict_critical: float = 0.82,
        max_resample_rounds: int = 1,
    ) -> None:
        super().__init__(name="ConflictMonitorAgent")
        self.conflict_high = _clip(conflict_high)
        self.conflict_critical = _clip(conflict_critical)
        self.max_resample_rounds = max(0, int(max_resample_rounds))

    def contribute(self, *args, **kwargs) -> ProbabilisticContribution:
        if args and isinstance(args[0], ActionDistributionState):
            distribution_state = args[0]
            proposals = list(kwargs.get("proposals", []))
        else:
            proposals = list(kwargs.get("proposals", args[0] if args else []))
            distribution_state = kwargs.get("distribution_state", args[1] if len(args) > 1 else None)
        if not isinstance(distribution_state, ActionDistributionState):
            raise TypeError("ConflictMonitorAgent.contribute() expects distribution_state")
        attempts = int(kwargs.get("attempts", 0))
        hot_active = bool(kwargs.get("hot_active", False))
        assessment = self.score_conflict(proposals, distribution_state)
        resolution = self.trigger_control_escalation(assessment, distribution_state, attempts=attempts, hot_active=hot_active)
        blocked_actions = list(resolution.get("blocked_actions", []))
        score = float(assessment.get("score", 0.0))
        prefs = {action: -0.18 for action in blocked_actions}
        if not prefs:
            prefs = {"respond": 0.02}
        return _bundle(
            self.name,
            prefs,
            confidence=_clip(1.0 - score * 0.35, 0.0, 1.0),
            reason=f"conflict={score:.2f}",
            trace_tags=["conflict"],
            gated_actions=blocked_actions,
            risk_hints={"conflict_score": score, "critical_conflict": float(assessment.get("critical_conflict", False))},
            layer="global",
            soft_mask={action: 0.12 for action in blocked_actions},
            hard_mask=list(blocked_actions if bool(assessment.get("critical_conflict")) else []),
            posterior={"conflict": round(score, 4), "stability": round(1.0 - score, 4)},
            confidence_trace={"conflict_score": round(score, 4)},
            trace_reason=resolution.get("reason", "conflict arbitration"),
            trace_scope="global",
        )

    def _coerce_runtime_state(self, state: RuntimeState | dict) -> RuntimeState:
        return state if isinstance(state, RuntimeState) else RuntimeState(**state)

    def _priority_signals(self, proposals: list[ProposalBundle], distribution_state: ActionDistributionState) -> dict[str, float]:
        signals = {name: 0.0 for name in PRIORITY_ORDER}
        for bundle in proposals:
            bucket = getattr(bundle, "priority_bucket", OWNER_PRIORITY_BUCKET.get(bundle.owner, "task_goal"))
            top_delta = max((abs(score) for score in bundle.delta_p.values()), default=0.0)
            risk_strength = max(
                (
                    float(value)
                    for value in getattr(bundle, "risk_hints", {}).values()
                    if isinstance(value, (int, float))
                ),
                default=0.0,
            )
            gated_mass = sum(distribution_state.p_raw.get(action, 0.0) for action in getattr(bundle, "gated_actions", []))
            signal = _clip(0.20 + top_delta * bundle.confidence * 1.25 + risk_strength * 0.45 + gated_mass * 0.45, 0.0, 1.0)
            signals[bucket] = max(signals.get(bucket, 0.0), signal)
        return signals

    def _vector_conflict(self, left: dict[str, float], right: dict[str, float]) -> float:
        if not left or not right:
            return 0.0
        keys = set(left) | set(right)
        distance = sum(abs(left.get(key, 0.0) - right.get(key, 0.0)) for key in keys)
        normalizer = sum(abs(left.get(key, 0.0)) + abs(right.get(key, 0.0)) for key in keys) or 1.0
        left_top = max(left, key=left.get)
        right_top = max(right, key=right.get)
        mismatch_bonus = 0.18 if left_top != right_top else 0.0
        return _clip(0.22 + distance / normalizer + mismatch_bonus, 0.0, 1.0)

    def score_conflict(self, proposals: list[ProposalBundle], distribution_state: ActionDistributionState) -> dict:
        top_scores = sorted(distribution_state.p_raw.values(), reverse=True)
        if len(top_scores) < 2:
            empty = ConflictAssessment(
                score=0.0,
                components=ConflictComponents(),
                priority_signals={name: 0.0 for name in PRIORITY_ORDER},
                dominant_conflicts=[],
                critical_conflict=False,
            )
            payload = to_dict(empty)
            payload["total_score"] = payload["score"]
            return payload

        priority_signals = self._priority_signals(proposals, distribution_state)
        top_actions = [max(bundle.delta_p, key=bundle.delta_p.get) for bundle in proposals if bundle.delta_p]
        unique_ratio = 0.0
        if len(top_actions) > 1:
            unique_ratio = (len(set(top_actions)) - 1) / max(len(top_actions) - 1, 1)
        top_gap = top_scores[0] - top_scores[1]
        closeness = _clip(1.0 - top_gap / 0.30, 0.0, 1.0)
        proposal_divergence = _clip(0.32 + 0.40 * unique_ratio + 0.35 * closeness, 0.0, 1.0)

        gated_ratio = sum(1 for bundle in proposals if getattr(bundle, "gated_actions", [])) / max(len(proposals), 1)
        max_gated_mass = max(
            (
                sum(distribution_state.p_raw.get(action, 0.0) for action in getattr(bundle, "gated_actions", []))
                for bundle in proposals
            ),
            default=0.0,
        )
        veto_ratio = sum(1 for bundle in proposals if bundle.veto) / max(len(proposals), 1)
        veto_tension = _clip(0.12 + max_gated_mass * 1.20 + gated_ratio * 0.35 + veto_ratio * 0.45, 0.0, 1.0)

        task_vector: dict[str, float] = {}
        desire_vector: dict[str, float] = {}
        roam_vector: dict[str, float] = {}
        for bundle in proposals:
            bucket = getattr(bundle, "priority_bucket", OWNER_PRIORITY_BUCKET.get(bundle.owner, "task_goal"))
            target = None
            if bucket == "task_goal":
                target = task_vector
            elif bucket == "immediate_desire":
                target = desire_vector
            elif bucket == "roaming":
                target = roam_vector
            if target is None:
                continue
            for action, score in bundle.delta_p.items():
                target[action] = target.get(action, 0.0) + abs(score) * bundle.confidence

        value_gap = _clip(
            max(
                self._vector_conflict(task_vector, desire_vector),
                self._vector_conflict(task_vector, roam_vector),
                0.18 + priority_signals["task_goal"] * 0.55 + max(priority_signals["immediate_desire"], priority_signals["roaming"]) * 0.45,
            ),
            0.0,
            1.0,
        )

        relation_mass = distribution_state.p_raw.get("connect", 0.0) + distribution_state.p_raw.get("clarify", 0.0) * 0.35
        relation_risk_gap = _clip(
            0.08 + priority_signals["relation_boundary"] * 0.78 + relation_mass * 0.55,
            0.0,
            1.0,
        )

        risky_body_mass = distribution_state.p_raw.get("plan", 0.0) + distribution_state.p_raw.get("connect", 0.0) + distribution_state.p_raw.get("wander", 0.0)
        body_gap = _clip(
            0.12 + max(priority_signals["body_safety"], priority_signals["budget_overload"] * 0.92) * 0.80 + min(1.0, risky_body_mass) * 0.25,
            0.0,
            1.0,
        )

        components = ConflictComponents(
            proposal_divergence=round(proposal_divergence, 4),
            veto_tension=round(veto_tension, 4),
            value_gap=round(value_gap, 4),
            relation_risk_gap=round(relation_risk_gap, 4),
            body_gap=round(body_gap, 4),
        )
        score = _clip(
            0.35 * components.proposal_divergence
            + 0.20 * components.veto_tension
            + 0.20 * components.value_gap
            + 0.15 * components.relation_risk_gap
            + 0.10 * components.body_gap,
            0.0,
            1.0,
        )
        dominant_conflicts = [
            name
            for name, value in sorted(to_dict(components).items(), key=lambda item: item[1], reverse=True)
            if value >= 0.45
        ]
        assessment = ConflictAssessment(
            score=round(score, 4),
            components=components,
            priority_signals={key: round(value, 4) for key, value in priority_signals.items()},
            dominant_conflicts=dominant_conflicts,
            critical_conflict=score >= self.conflict_critical,
        )
        payload = to_dict(assessment)
        payload["total_score"] = payload["score"]
        return payload

    def trigger_control_escalation(
        self,
        assessment: dict | float,
        distribution_state: ActionDistributionState | None = None,
        attempts: int = 0,
        hot_active: bool = False,
    ) -> dict:
        if not isinstance(assessment, dict):
            assessment = {
                "score": float(assessment),
                "critical_conflict": float(assessment) >= self.conflict_critical,
                "priority_signals": {name: 0.0 for name in PRIORITY_ORDER},
            }
        score = float(assessment.get("score", 0.0))
        if score < self.conflict_high:
            return to_dict(ConflictResolution(flag=False, reason="below_conflict_threshold"))

        signals = {name: float(assessment.get("priority_signals", {}).get(name, 0.0)) for name in PRIORITY_ORDER}
        winning_priority = None
        for bucket in PRIORITY_ORDER:
            threshold = 0.48 if bucket in {"body_safety", "budget_overload", "relation_boundary"} else 0.42
            if signals.get(bucket, 0.0) >= threshold:
                winning_priority = bucket
                break
        if winning_priority is None:
            winning_priority = max(PRIORITY_ORDER, key=lambda name: signals.get(name, 0.0))

        blocked_actions = set(DEFAULT_BLOCKS.get(winning_priority, set()))
        if distribution_state is not None:
            if winning_priority == "body_safety":
                blocked_actions.update(action for action in {"plan", "connect", "wander"} if distribution_state.p_raw.get(action, 0.0) >= 0.06)
            elif winning_priority == "budget_overload":
                blocked_actions.update(action for action in {"plan", "connect", "wander"} if distribution_state.p_raw.get(action, 0.0) >= 0.04)
            elif winning_priority == "task_goal":
                blocked_actions.update(action for action in {"wander"} if distribution_state.p_raw.get(action, 0.0) >= 0.05)
            elif winning_priority == "relation_boundary":
                blocked_actions.update(action for action in {"connect"} if distribution_state.p_raw.get(action, 0.0) >= 0.03)
        if hot_active:
            blocked_actions.add("connect")

        action_scales = {action: 0.20 for action in blocked_actions}
        for action, scale in TEMPLATE_ACTION_SCALES.get(TEMPLATE_BY_PRIORITY[winning_priority], {}).items():
            action_scales[action] = max(action_scales.get(action, 1.0), scale) if scale > 1.0 else min(action_scales.get(action, 1.0), scale)

        resample_policy = self.request_resample(assessment, attempts)
        applied_template = None
        if bool(resample_policy.get("force_compromise")) and attempts >= int(resample_policy.get("allowed_resamples", 0)):
            applied_template = TEMPLATE_BY_PRIORITY[winning_priority]
            for action, scale in TEMPLATE_ACTION_SCALES.get(applied_template, {}).items():
                action_scales[action] = max(action_scales.get(action, 1.0), scale) if scale > 1.0 else min(action_scales.get(action, 1.0), scale)

        resolution = ConflictResolution(
            flag=True,
            winning_priority=winning_priority,
            blocked_actions=sorted(blocked_actions),
            action_scales=action_scales,
            applied_template=applied_template,
            reason=f"{winning_priority}:{score:.2f}",
        )
        payload = to_dict(resolution)
        payload["flag"] = True
        return payload

    def request_resample(self, assessment: dict | float, attempts: int) -> dict:
        if isinstance(assessment, dict):
            score = float(assessment.get("score", 0.0))
            critical = bool(assessment.get("critical_conflict", score >= self.conflict_critical))
        else:
            score = float(assessment)
            critical = score >= self.conflict_critical
        allowed_resamples = 0
        if score >= self.conflict_high:
            allowed_resamples = self.max_resample_rounds
        if critical:
            allowed_resamples = 2
        return {
            "flag": attempts < allowed_resamples,
            "allowed_resamples": allowed_resamples,
            "force_compromise": critical,
            "critical_conflict": critical,
        }

    def mark_post_error_adjustment(
        self,
        state: RuntimeState | dict,
        round_id: int,
        resolution: dict,
        compromise: dict,
        deadlock_fuse_triggered: bool,
        top_action_before: str | None,
        top_action_after: str | None,
        blocked_actions: list[str],
        safe_mode_before: bool,
    ) -> dict:
        runtime_state = self._coerce_runtime_state(state)
        previous_stage = runtime_state.repair_state.stage
        winning_priority = resolution.get("winning_priority")
        template = compromise.get("template") or resolution.get("applied_template")
        adjustment_reason = ""
        if deadlock_fuse_triggered:
            adjustment_reason = "deadlock_fuse"
        elif compromise.get("triggered"):
            adjustment_reason = "forced_compromise"
        elif top_action_before and top_action_after and top_action_before != top_action_after:
            adjustment_reason = "top_action_shift"

        if deadlock_fuse_triggered:
            next_stage = "repairing"
            transition_reason = "deadlock_fuse"
        elif adjustment_reason and len(runtime_state.repair_ledger) >= 2:
            next_stage = "repairing"
            transition_reason = "sustained_adjustment"
        elif adjustment_reason:
            next_stage = "adjusting"
            transition_reason = adjustment_reason
        elif previous_stage in {"adjusting", "repairing", "cooling"} and (
            runtime_state.conflict_hot_rounds > 0 or runtime_state.conflict_recovery_rounds > 0
        ):
            next_stage = "cooling"
            transition_reason = "cooldown"
        elif previous_stage in {"adjusting", "repairing", "cooling"}:
            next_stage = "recovered"
            transition_reason = "recovered"
        elif previous_stage == "recovered":
            next_stage = "recovered"
            transition_reason = "steady"
        else:
            next_stage = "idle"
            transition_reason = "idle"

        safe_mode_after = runtime_state.safe_mode
        conflict_safe_mode_owner = runtime_state.conflict_safe_mode_owner
        repair_cooldown_rounds_patch = runtime_state.conflict_recovery_rounds
        repair_mode = runtime_state.repair_mode

        if deadlock_fuse_triggered or (adjustment_reason and len(runtime_state.repair_ledger) >= 2):
            safe_mode_after = True
            conflict_safe_mode_owner = "conflict"
            repair_cooldown_rounds_patch = max(runtime_state.conflict_recovery_rounds, 2)
            repair_mode = "deadlock_fuse" if deadlock_fuse_triggered else "conflict_repair"
        elif next_stage == "adjusting":
            repair_mode = "post_error_adjustment"
        elif next_stage == "cooling":
            repair_mode = "conflict_cooling"
        elif next_stage == "recovered":
            if conflict_safe_mode_owner == "conflict":
                safe_mode_after = False
                conflict_safe_mode_owner = None
            repair_mode = None
        elif next_stage == "idle":
            repair_mode = None

        safe_mode_delta = _safe_mode_delta(safe_mode_before, safe_mode_after)
        adjustment = ConflictPostErrorAdjustment(
            triggered=bool(adjustment_reason),
            reason=adjustment_reason,
            winning_priority=winning_priority,
            template=template,
            top_action_before=top_action_before,
            top_action_after=top_action_after,
            blocked_actions=sorted(set(blocked_actions)),
            safe_mode_delta=safe_mode_delta,
        )
        repair_ledger_append: ConflictRepairLedgerEntry | dict[str, object] = {}
        if adjustment.triggered:
            repair_ledger_append = ConflictRepairLedgerEntry(
                round_id=round_id,
                reason=adjustment.reason,
                winning_priority=adjustment.winning_priority,
                template=adjustment.template,
                blocked_actions=list(adjustment.blocked_actions),
                top_action_before=adjustment.top_action_before,
                top_action_after=adjustment.top_action_after,
                safe_mode_delta=adjustment.safe_mode_delta,
                repair_stage_after=next_stage,
            )

        repair_state_snapshot = ConflictRepairState(
            stage=next_stage,
            active=next_stage in {"adjusting", "repairing", "cooling"},
            last_reason=adjustment.reason if adjustment.triggered else transition_reason,
            last_transition_round=round_id,
            cooldown_rounds_remaining=repair_cooldown_rounds_patch if next_stage in {"repairing", "cooling"} else 0,
            last_template=template or runtime_state.repair_state.last_template,
            last_winning_priority=winning_priority or runtime_state.repair_state.last_winning_priority,
        )
        repair_transition = {
            "from_stage": previous_stage,
            "to_stage": next_stage,
            "reason": transition_reason,
            "changed": previous_stage != next_stage,
            "safe_mode_delta": safe_mode_delta,
        }
        return {
            "repair_mode": repair_mode,
            "post_error_adjustment": to_dict(adjustment),
            "repair_transition": repair_transition,
            "repair_state_snapshot": to_dict(repair_state_snapshot),
            "repair_ledger_append": to_dict(repair_ledger_append),
            "conflict_safe_mode_owner": conflict_safe_mode_owner,
            "safe_mode_patch": safe_mode_after,
            "repair_cooldown_rounds_patch": repair_cooldown_rounds_patch,
        }

    def check_behavior_plausibility(self, action: str, scenario: str) -> dict:
        fail_score = 0.85 if scenario == "task" and action == "wander" else 0.05
        return {"pass": fail_score < 0.70, "plausibility_fail_score": fail_score}


class ThalamusAttentionAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="ThalamusAttentionAgent")

    def contribute(self, *args, **kwargs) -> ProbabilisticContribution:
        distribution_state = kwargs.get("distribution_state", args[0] if args else None)
        if not isinstance(distribution_state, ActionDistributionState):
            raise TypeError("ThalamusAttentionAgent.contribute() expects distribution_state")
        distribution = dict(distribution_state.p_raw or {})
        normalized = self.normalize_distribution(distribution)["distribution"]
        prefs = {action: score for action, score in distribution.items()}
        return _bundle(
            self.name,
            prefs,
            confidence=0.74,
            reason="context routing",
            trace_tags=["thalamus"],
            priority_bucket=OWNER_PRIORITY_BUCKET.get(self.name, "task_goal"),
            control_domain=OWNER_CONTROL_DOMAIN.get(self.name, "context"),
            layer="context",
            attention_bias=normalized,
            posterior=normalized,
            confidence_trace={"route_mass": round(sum(max(v, 0.0) for v in distribution.values()), 4)},
            trace_reason="thalamus context routing",
            trace_scope="context",
        )

    def aggregate_proposals(self, distribution_state: ActionDistributionState) -> dict:
        return {"distribution": distribution_state.p_raw}

    def normalize_distribution(self, distribution: dict[str, float]) -> dict:
        total = sum(max(value, 0.0) for value in distribution.values()) or 1.0
        normalized = {action: round(max(value, 0.0) / total, 6) for action, value in distribution.items()}
        return {"distribution": normalized}

    def sample_action(self, distribution: dict[str, float], sample_value: float = 0.5) -> dict:
        threshold = _clip(sample_value, 0.0, 1.0)
        cumulative = 0.0
        sampled_name = max(distribution, key=distribution.get)
        for action_name, probability in sorted(distribution.items()):
            cumulative += max(probability, 0.0)
            if threshold <= cumulative:
                sampled_name = action_name
                break
        return {"action": ActionCandidate(name=sampled_name, probability=distribution[sampled_name], rationale="thalamus sample")}


class BehaviorPlausibilityGuard(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="BehaviorPlausibilityGuard")

    def check_behavior_plausibility(self, action: str, scenario: str, relation_state: dict) -> dict:
        fail_score = 0.0
        if scenario == "task" and action == "wander":
            fail_score = 0.85
        if relation_state.get("boundary_level", 0.0) > 0.75 and action == "connect":
            fail_score = max(fail_score, 0.74)
        return {"pass": fail_score < 0.70, "plausibility_fail_score": fail_score}

    def request_second_sampling(self, fail_score: float, attempts: int) -> dict:
        return {"flag": fail_score >= 0.70 and attempts < 2}

    def escalate_value_reestimate(self, fail_score: float) -> dict:
        return {"flag": fail_score >= 0.82}


class ForcedModeSwitch(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="ForcedModeSwitch")

    def detect_mode_lock(self, history: list[str], focus_lock_count: int) -> dict:
        repeated = 1.0 if len(set(history[-6:])) == 1 and history else 0.0
        lock_score = _clip(repeated * 0.6 + focus_lock_count / 10, 0.0, 1.0)
        return {"lock_score": lock_score}

    def trigger_forced_focus_switch(self, lock_score: float) -> dict:
        return {"switch_flag": lock_score >= 0.75}


class OutputGate(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="OutputGate")

    def apply_output_gate(self, action: str, state: RuntimeState, scenario: str, relation_state: dict) -> dict:
        gate = 1.0
        if state.safe_mode and action not in {"respond", "rest", "short_reply"}:
            gate = 0.0
        if scenario == "task" and action == "connect":
            gate = 0.0
        if relation_state.get("boundary_level", 0.0) > 0.8 and action == "connect":
            gate = 0.0
        return {"gate": gate}

    def render_tone_profile(self, expression_profile) -> dict:
        return {
            "tone_params": {
                "warmth_level": expression_profile.warmth_level,
                "directness_level": expression_profile.directness_level,
                "hedging_level": expression_profile.hedging_level,
                "repair_tendency": expression_profile.repair_tendency,
            }
        }

    def compute_delay_profile(self, expression_profile) -> dict:
        return {"delay_params": {"reply_delay": expression_profile.reply_delay, "latency_style": expression_profile.latency_style}}


def build_agents(
    *,
    conflict_high: float = 0.75,
    conflict_critical: float = 0.82,
    max_resample_rounds: int = 1,
) -> list[BaseAgent]:
    return [
        SalienceAgent(),
        BodyStateAgent(),
        EmotionAgent(),
        RelationshipAgent(),
        ResourceAgent(),
        PFCAgent(),
        HabitAgent(),
        DesireAgent(),
        DMNAgent(),
        HippocampusAgent(),
        PerspectiveModel(),
        ValueAgent(),
        ConflictMonitorAgent(
            conflict_high=conflict_high,
            conflict_critical=conflict_critical,
            max_resample_rounds=max_resample_rounds,
        ),
        ThalamusAttentionAgent(),
        UnconsciousAgent(),
        CerebellarPredictor(),
        BehaviorPlausibilityGuard(),
        ForcedModeSwitch(),
        OutputGate(),
    ]
