from __future__ import annotations

from dataclasses import dataclass

from nalr.schemas.models import ActionCandidate, ActionDistributionState, ProposalBundle, RoundEvent, RuntimeState


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _clip_delta(value: float) -> float:
    return _clip(value, -0.35, 0.35)


def _top_reason(action_preferences: dict[str, float], reason: str) -> str:
    if not action_preferences:
        return reason
    action = max(action_preferences, key=action_preferences.get)
    return f"{reason}; top_action={action}"


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
) -> ProposalBundle:
    clipped = {action: _clip_delta(score) for action, score in prefs.items()}
    return ProposalBundle(
        owner=owner,
        confidence=_clip(confidence, 0.0, 1.0),
        action_preferences=clipped,
        delta_p=clipped,
        sigma_scale=_clip(sigma_scale, 0.60, 1.60),
        veto=veto,
        utility_shift=utility_shift or {},
        state_patch=state_patch or {},
        trace_tags=trace_tags or [],
        reason=_top_reason(clipped, reason),
    )


@dataclass
class BaseAgent:
    name: str

    def run_skill(self, skill_name: str, *args, **kwargs):
        return getattr(self, skill_name)(*args, **kwargs)

    def propose(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> ProposalBundle:
        raise NotImplementedError


class BodyStateAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="BodyStateAgent")

    def update_body_state(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> dict:
        next_energy = _clip(state.body_energy + event.energy_delta)
        return {"state_patch": {"body_energy": next_energy}}

    def compute_body_bias(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> ProposalBundle:
        energy = _clip(state.body_energy + event.energy_delta)
        prefs = {"respond": 0.10}
        sigma_scale = 1.0
        if energy < 0.45:
            prefs["rest"] = 0.18 + (0.45 - energy) * 0.40
            sigma_scale = 1.15
        return _bundle(self.name, prefs, confidence=0.64, sigma_scale=sigma_scale, reason=f"energy={energy:.2f}", trace_tags=["body"])

    def apply_body_veto(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> dict:
        energy = _clip(state.body_energy + event.energy_delta)
        gated_actions = ["connect"] if energy < 0.15 else []
        return {"veto": False, "gated_actions": gated_actions}

    def propose(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> ProposalBundle:
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

    def propose(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> ProposalBundle:
        closeness = context["closeness"]
        prefs = {"respond": 0.04 + scenario.get("relationship_weight", 0.1) * closeness * 0.3}
        if closeness > 0.55 and event.target:
            prefs["connect"] = 0.08 + closeness * 0.18
        return _bundle(self.name, prefs, confidence=0.58, reason=f"closeness={closeness:.2f}", trace_tags=["relationship"])


class DesireAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="DesireAgent")

    def score_immediate_reward(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> dict:
        content = event.content.lower()
        score = 0.0
        if any(token in content for token in {"quick", "easy", "coffee", "break", "rest"}):
            score = 0.45
        return {"score": score}

    def compute_effort_avoidance(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> ProposalBundle:
        content = event.content.lower()
        prefs = {}
        if any(token in content for token in {"quick", "easy", "coffee", "break", "rest"}):
            prefs["rest"] = 0.12 + (1.0 - state.body_energy) * 0.18
            prefs["respond"] = 0.06
        return _bundle(self.name, prefs, confidence=0.54, reason="comfort seeking", trace_tags=["desire"])

    def suggest_low_cost_action(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> dict:
        return {"action_hint": "rest" if state.body_energy < 0.4 else "respond"}

    def propose(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> ProposalBundle:
        return self.compute_effort_avoidance(event, state, scenario, context)


class EmotionAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="EmotionAgent")

    def update_affect_state(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> dict:
        next_mood = _clip(state.mood + event.valence * 0.08)
        return {"state_patch": {"mood": next_mood}}

    def compute_affect_bias(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> ProposalBundle:
        prefs = {}
        sigma_scale = 1.0
        if event.valence <= -0.25:
            prefs["clarify"] = 0.10 + abs(event.valence) * 0.14
            prefs["rest"] = 0.08 + abs(event.valence) * 0.12
            sigma_scale = 1.12
        elif event.valence >= 0.25:
            prefs["connect"] = 0.10 + event.valence * 0.12
        return _bundle(self.name, prefs, confidence=0.60, sigma_scale=sigma_scale, reason=f"valence={event.valence:.2f}", trace_tags=["emotion"])

    def trigger_affect_veto(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> dict:
        return {"veto": event.valence < -0.95}

    def propose(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> ProposalBundle:
        return self.compute_affect_bias(event, state, scenario, context)


class DMNAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="DMNAgent")

    def sample_dmn_intrusion(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> ProposalBundle:
        prefs = {}
        if state.mode in {"idle", "interactive"}:
            prefs["wander"] = scenario.get("dmn_weight", 0.05) * (4.8 if state.mode == "idle" else 1.2)
        confidence = 0.72 if state.mode == "idle" else 0.46
        return _bundle(self.name, prefs, confidence=confidence, reason="background drift", trace_tags=["dmn"])

    def score_rumination_pull(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> dict:
        return {"score": 0.18 if state.mode == "idle" else 0.05}

    def select_spontaneous_topic(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> dict:
        return {"topic_hint": context.get("cue") or "idle-thought"}

    def propose(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> ProposalBundle:
        return self.sample_dmn_intrusion(event, state, scenario, context)


class PFCAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="PFCAgent")

    def generate_candidates(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> ProposalBundle:
        content = event.content.lower()
        prefs = {"respond": 0.08}
        if scenario.get("pfc_base_share", 0.2) >= 0.30 or "plan" in content or "help" in content:
            prefs["plan"] = 0.18 + scenario.get("pfc_base_share", 0.2)
        if "remember" in content or context.get("cue"):
            prefs["recall"] = max(prefs.get("recall", 0.0), 0.18 + context.get("recall_strength", 0.0) * 0.15)
        return _bundle(self.name, prefs, confidence=0.84, sigma_scale=0.92, reason="deliberate planner", trace_tags=["pfc"])

    def estimate_plan_depth(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> dict:
        content = event.content.lower()
        depth = 2 if "plan" in content or "help" in content else 1
        return {"plan_depth": depth}

    def bind_working_memory(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> dict:
        return {"wm_update": {"cue": context.get("cue"), "focus": state.focus}}

    def propose(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> ProposalBundle:
        return self.generate_candidates(event, state, scenario, context)


class UnconsciousAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="UnconsciousAgent")

    def load_temperament(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> dict:
        return {"baseline": {"warmth_bias": 0.5, "directness_bias": 0.55}}

    def compute_trait_bias(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> ProposalBundle:
        prefs = {"respond": 0.04 + state.mood * 0.04}
        return _bundle(self.name, prefs, confidence=0.42, reason="temperament baseline", trace_tags=["trait"])

    def apply_chronic_shift(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> dict:
        return {"state_patch": {"mood": _clip(state.mood)}}

    def propose(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> ProposalBundle:
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

    def propose(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> ProposalBundle:
        recall_strength = context.get("recall_strength", 0.0)
        prefs = {}
        if recall_strength > 0:
            prefs["recall"] = 0.10 + recall_strength * 0.22
        return _bundle(self.name, prefs, confidence=0.66, reason=f"recall={recall_strength:.2f}", trace_tags=["memory"])


class SalienceAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="SalienceAgent")

    def score_salience(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> ProposalBundle:
        content = event.content.lower()
        prefs = {}
        if "help" in content or "remember" in content or "urgent" in content:
            prefs["plan"] = 0.10 + scenario.get("pfc_base_share", 0.2) * 0.22
        if abs(event.valence) >= 0.3:
            prefs["clarify"] = max(prefs.get("clarify", 0.0), 0.08 + abs(event.valence) * 0.10)
        if not prefs:
            prefs["respond"] = 0.08
        return _bundle(self.name, prefs, confidence=0.64, reason="salience promotion", trace_tags=["salience"])

    def switch_mode(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> dict:
        if abs(event.valence) >= 0.6:
            return {"mode_flag": "interactive"}
        return {"mode_flag": state.mode}

    def interrupt_current_focus(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> dict:
        return {"interrupt": abs(event.valence) >= 0.5}

    def promote_event_to_workspace(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> dict:
        return {"workspace_delta": {"focus_hint": "plan" if "help" in event.content.lower() else state.focus}}

    def propose(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> ProposalBundle:
        return self.score_salience(event, state, scenario, context)


class HabitAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="HabitAgent")

    def compute_feedback_decay(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> dict:
        return {"scalar": max(0.0, 1.0 - context.get("habit_strength", 0.0) * 0.1)}

    def update_habit_strength(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> dict:
        return {"state_update": {"habit_strength": context.get("habit_strength", 0.0)}}

    def suggest_default_action(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> ProposalBundle:
        habit_strength = context.get("habit_strength", 0.0)
        prefs = {}
        if habit_strength > 0:
            prefs["respond"] = 0.06 + habit_strength * scenario.get("habit_weight", 0.1)
            prefs["recall"] = 0.04 + habit_strength * 0.12
        return _bundle(self.name, prefs, confidence=0.60, reason=f"habit={habit_strength:.2f}", trace_tags=["habit"])

    def estimate_override_cost(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> dict:
        return {"cost": _clip(context.get("habit_strength", 0.0) * 0.4, 0.0, 1.0)}

    def propose(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> ProposalBundle:
        return self.suggest_default_action(event, state, scenario, context)


class ValueAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="ValueAgent")

    def estimate_subjective_value(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> dict:
        content = event.content.lower()
        scores = {"respond": 0.08}
        if scenario.get("pfc_base_share", 0.2) >= 0.30 or "plan" in content:
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

    def propose(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> ProposalBundle:
        scores = self.estimate_subjective_value(event, state, scenario, context)["scores"]
        return _bundle(self.name, scores, confidence=0.59, reason="subjective value re-rank", trace_tags=["value"])


class ResourceAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="ResourceAgent")

    def compute_scarcity_index(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> dict:
        burn_rate = context.get("recent_burn_rate", 0.0)
        scarcity = _clip((1.0 - state.budget_remaining) * 0.7 + burn_rate * 0.3, 0.0, 1.0)
        return {"scalar": scarcity}

    def map_budget_to_bias(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> ProposalBundle:
        prefs = {"respond": 0.05}
        scarcity = self.compute_scarcity_index(event, state, scenario, context)["scalar"]
        if scarcity > 0.45:
            prefs["rest"] = 0.08 + scarcity * 0.18
        return _bundle(self.name, prefs, confidence=0.58, reason=f"budget={state.budget_remaining:.2f}", trace_tags=["resource"])

    def suggest_resource_mode(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> dict:
        return {"mode_flag": "safe" if state.budget_remaining < 0.08 else state.mode}

    def propose(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> ProposalBundle:
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

    def micro_adjust_action(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> ProposalBundle:
        prefs = {}
        if state.last_action == "clarify":
            prefs["clarify"] = 0.04
        elif state.last_action == "plan":
            prefs["plan"] = 0.04
        return _bundle(self.name, prefs, confidence=0.35, reason="micro smoothing", trace_tags=["timing"])

    def propose(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> ProposalBundle:
        return self.micro_adjust_action(event, state, scenario, context)


class PerspectiveModel(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="PerspectiveModel")

    def infer_other_state(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> dict:
        return {"state_hypothesis": {"confused": "clarify" in event.content.lower(), "closeness": context.get("closeness", 0.5)}}

    def simulate_other_reaction(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> dict:
        closeness = context.get("closeness", 0.5)
        return {"reaction_hypothesis": {"positive": closeness > 0.6, "risk": 1 - closeness}}

    def estimate_misunderstanding_risk(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> dict:
        risk = _clip((1.0 - context.get("closeness", 0.5)) * 0.5 + (0.15 if "clarify" in event.content.lower() else 0.0), 0.0, 1.0)
        return {"risk": risk}

    def adjust_social_interpretation(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> ProposalBundle:
        prefs = {}
        if scenario.get("relationship_weight", 0.0) >= 0.22 and event.target:
            prefs["clarify"] = 0.05 + context["closeness"] * 0.08
        return _bundle(self.name, prefs, confidence=0.49, reason="social interpretation", trace_tags=["perspective"])

    def propose(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> ProposalBundle:
        return self.adjust_social_interpretation(event, state, scenario, context)


class ConflictMonitorAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="ConflictMonitorAgent")

    def score_conflict(self, proposals: list[ProposalBundle], distribution_state: ActionDistributionState) -> dict:
        top_scores = sorted(distribution_state.p_raw.values(), reverse=True)
        if len(top_scores) < 2:
            return {"score": 0.0}
        score = _clip(0.5 - (top_scores[0] - top_scores[1]) + len({max(p.delta_p, key=p.delta_p.get) for p in proposals if p.delta_p}) * 0.08, 0.0, 1.0)
        return {"score": score}

    def trigger_control_escalation(self, score: float) -> dict:
        return {"flag": score >= 0.65}

    def request_resample(self, score: float, attempts: int) -> dict:
        return {"flag": score >= 0.82 and attempts < 2}

    def check_behavior_plausibility(self, action: str, scenario: str) -> dict:
        fail_score = 0.85 if scenario == "task" and action == "wander" else 0.05
        return {"pass": fail_score < 0.70, "plausibility_fail_score": fail_score}


class ThalamusAttentionAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="ThalamusAttentionAgent")

    def aggregate_proposals(self, distribution_state: ActionDistributionState) -> dict:
        return {"distribution": distribution_state.p_raw}

    def normalize_distribution(self, distribution: dict[str, float]) -> dict:
        total = sum(max(value, 0.0) for value in distribution.values()) or 1.0
        normalized = {action: round(max(value, 0.0) / total, 6) for action, value in distribution.items()}
        return {"distribution": normalized}

    def sample_action(self, distribution: dict[str, float]) -> dict:
        sampled_name = max(distribution, key=distribution.get)
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
        if state.safe_mode and action not in {"respond", "rest"}:
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


def build_agents() -> list[BaseAgent]:
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
        ConflictMonitorAgent(),
        ThalamusAttentionAgent(),
        UnconsciousAgent(),
        CerebellarPredictor(),
        BehaviorPlausibilityGuard(),
        ForcedModeSwitch(),
        OutputGate(),
    ]
