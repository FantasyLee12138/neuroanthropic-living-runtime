from __future__ import annotations

import hashlib
import json
import math
import random
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from nalr.agents.modules import build_agents
from nalr.memory.store import MemoryStore
from nalr.output.style import build_expression_profile, build_render_plan, compute_style_profile
from nalr.runtime.model_gateway import ModelGateway
from nalr.schemas.models import (
    ActionCandidate,
    ActionDistributionState,
    AgentContribution,
    CheckpointRef,
    CommandResult,
    ExpressionProfile,
    HealthEvent,
    ProposalBundle,
    RenderPlan,
    RoundEvent,
    RoundResult,
    RoundTrace,
    RuntimeState,
    StochasticState,
    to_dict,
)
from nalr.skills.executor import SkillExecutor
from nalr.skills.registry import build_skill_registry
from nalr.trace.store import TraceStore


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _default_operator_level(command: str) -> str:
    parts = command.split()
    if not parts:
        return "read_only"
    domain = parts[0]
    if domain in {"safe", "checkpoint", "budget"}:
        return "ops_admin"
    if domain in {"body", "mood", "nudge", "mode"}:
        return "soft_intervene"
    if domain in {"agent", "debug", "suppress"}:
        return "debug_control"
    return "read_only"


PIPELINE_ORDER: list[tuple[str, tuple[str, ...]]] = [
    ("state_update", ()),
    ("salience", ("SalienceAgent",)),
    ("body", ("BodyStateAgent",)),
    ("emotion", ("EmotionAgent",)),
    ("relationship", ("RelationshipAgent",)),
    ("resource", ("ResourceAgent",)),
    ("pfc", ("PFCAgent",)),
    ("habit", ("HabitAgent",)),
    ("desire", ("DesireAgent",)),
    ("dmn", ("DMNAgent",)),
    ("hippocampus", ("HippocampusAgent",)),
    ("perspective", ("PerspectiveModel",)),
    ("value", ("ValueAgent",)),
    ("conflict", ("ConflictMonitorAgent",)),
    ("thalamus", ("ThalamusAttentionAgent",)),
    ("plausibility_guard", ("BehaviorPlausibilityGuard",)),
    ("forced_mode_switch", ("ForcedModeSwitch",)),
    ("output_gate", ("OutputGate",)),
    ("writeback", ()),
]


CORE_ACTIONS = ("respond", "plan", "recall", "rest", "connect", "clarify", "wander")


class RuntimeController:
    def __init__(self, project_root: Path, config_root: Path | None = None, home_path: Path | None = None) -> None:
        self.project_root = Path(project_root)
        self.config_root = Path(config_root) if config_root else self.project_root / "config"
        self.home_path = Path(home_path) if home_path else self.project_root / ".alive"
        self.runtime_dir = self.home_path / "runtime"
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.runtime_dir / "persona_state.json"
        self.checkpoint_dir = self.runtime_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.config = self._load_config()
        self.trace_store = TraceStore(self.home_path)
        self.memory_store = MemoryStore(self.home_path)
        self.agents = build_agents()
        self.agent_map = {agent.name: agent for agent in self.agents}
        self.skills = build_skill_registry()
        self.skill_executor = SkillExecutor(self.skills)
        self.model_gateway = ModelGateway.from_config(self.config["models"]) if self.config.get("models") else None

        if not self.state_path.exists():
            initial_state = RuntimeState(
                agents_enabled={
                    name: agent_cfg.get("enabled", True)
                    for name, agent_cfg in self.config["agents"]["agents"].items()
                }
            )
            self._save_state(initial_state)

    def _load_config(self) -> dict[str, Any]:
        def read_yaml(name: str) -> dict[str, Any]:
            path = self.config_root / name
            return yaml.safe_load(path.read_text(encoding="utf-8"))

        return {
            "agents": read_yaml("agents.yaml"),
            "modes": read_yaml("modes.yaml"),
            "scenarios": read_yaml("scenarios.yaml"),
            "thresholds": read_yaml("thresholds.yaml"),
            "temperament": read_yaml("temperament.yaml"),
            "resource_rules": read_yaml("resource_rules.yaml"),
            "output_style": read_yaml("output_style.yaml"),
            "models": read_yaml("models.yaml")["models"] if (self.config_root / "models.yaml").exists() else None,
        }

    def _save_state(self, state: RuntimeState) -> None:
        self.state_path.write_text(json.dumps(to_dict(state), ensure_ascii=False, indent=2), encoding="utf-8")

    def load_runtime_state(self) -> RuntimeState:
        payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        return RuntimeState(**payload)

    def _state_hash(self, state: RuntimeState) -> str:
        return hashlib.sha1(json.dumps(to_dict(state), sort_keys=True).encode("utf-8")).hexdigest()

    def _round_seed(self, state: RuntimeState, event: RoundEvent) -> int:
        payload = f"{state.round_count}:{event.source}:{event.content}:{event.target or ''}:{event.cue or ''}"
        return int(hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8], 16)

    def _apply_state_patch(self, state: RuntimeState, patch: dict[str, Any]) -> None:
        for key, value in patch.items():
            if hasattr(state, key):
                setattr(state, key, value)

    def _agent_weight(self, agent_name: str, state: RuntimeState) -> float:
        base = self.config["agents"]["agents"].get(agent_name, {}).get("weight", 1.0)
        return state.agent_weight_overrides.get(agent_name, base)

    def _record_skill_trace(self, bucket: list[dict[str, Any]], round_id: int, result) -> None:
        bucket.append(
            {
                "round_id": round_id,
                "skill_name": result.skill_name,
                "owner_module": result.owner_module,
                "latency_ms": result.latency_ms,
                "cost_class": result.cost_class,
                "input_hash": result.input_hash,
                "output_hash": result.output_hash,
                "failure_policy_applied": result.failure_policy_applied,
                "degraded": result.degraded,
                "seed_ref": result.seed_ref,
            }
        )

    def _execute_skill(
        self,
        *,
        round_id: int,
        skill_name: str,
        inputs: dict[str, Any],
        provider,
        skill_traces: list[dict[str, Any]],
        fallback_value: Any | None = None,
        seed_ref: int | None = None,
    ):
        output, result = self.skill_executor.run(
            round_id=round_id,
            skill_name=skill_name,
            inputs=inputs,
            provider=provider,
            fallback_value=fallback_value,
            seed_ref=seed_ref,
        )
        self._record_skill_trace(skill_traces, round_id, result)
        return output

    def _relation_state(self, event: RoundEvent, context: dict[str, Any]) -> dict[str, float]:
        closeness = context.get("closeness", 0.5)
        boundary_level = _clip(0.8 - closeness + max(0.0, -context.get("valence", 0.0)) * 0.2, 0.0, 1.0)
        relationship_risk = _clip((1.0 - closeness) * 0.5 + boundary_level * 0.25, 0.0, 1.0)
        privacy_level = 0.8 if event.target else 0.35
        return {
            "closeness": closeness,
            "boundary_level": boundary_level,
            "relationship_risk": relationship_risk,
            "privacy_level": privacy_level,
        }

    def _build_base_distribution(self, state: RuntimeState, scenario_cfg: dict[str, Any], mode_cfg: dict[str, Any], relation_state: dict[str, float]) -> dict[str, float]:
        mode_plan_scale = {
            "interactive": 0.7,
            "auto": 0.55,
            "idle": 0.18,
            "sleep": 0.05,
            "safe": 0.30,
        }.get(state.mode, 0.7)
        base = {
            "respond": mode_cfg.get("response_floor", 0.3),
            "plan": 0.04 + scenario_cfg.get("pfc_base_share", 0.2) * mode_plan_scale,
            "recall": 0.04 + scenario_cfg.get("memory_gist_bias", 0.5) * 0.12,
            "rest": 0.03 + max(0.0, 0.45 - state.body_energy) * 0.20,
            "connect": 0.02 + scenario_cfg.get("relationship_weight", 0.1) * relation_state["closeness"] * 0.40,
            "clarify": 0.08,
            "wander": scenario_cfg.get("dmn_weight", 0.05) * (1.0 if mode_cfg.get("allow_dmn", True) else 0.0),
        }
        return {action: _clip(value, 0.01, 0.85) for action, value in base.items()}

    def _compute_context_delta(self, state: RuntimeState, scenario_cfg: dict[str, Any]) -> dict[str, float]:
        delta = {action: 0.0 for action in CORE_ACTIONS}
        if state.focus_nudge:
            delta["respond"] += state.focus_nudge * 0.6
            delta["plan"] += state.focus_nudge * 0.6
            delta["clarify"] += state.focus_nudge * 0.4
        if state.safe_mode:
            delta["respond"] += 0.10
            delta["wander"] -= 0.12
        if scenario_cfg.get("delay_tolerance", 0.1) < 0.15:
            delta["respond"] += 0.03
            delta["clarify"] += 0.02
        return delta

    def _update_ci(self, state: RuntimeState, actions: list[str], context: dict[str, Any], scenario_cfg: dict[str, Any], thresholds: dict[str, Any]) -> dict[str, float]:
        ci_state = dict(state.action_ci)
        IF = _clip(0.25 + context.get("closeness", 0.5) * 0.35 + context.get("habit_strength", 0.0) * 0.30, 0.0, 1.0)
        NEG = _clip(max(0.0, -context.get("valence", 0.0)), 0.0, 1.0)
        HAB = _clip(context.get("habit_strength", 0.0), 0.0, 1.0)
        IL = _clip((1.0 - scenario_cfg.get("pfc_base_share", 0.2)) * (1.0 if state.mode in {"idle", "interactive"} else 0.5), 0.0, 1.0)
        UNC = _clip(0.3 + NEG * 0.2, 0.0, 1.0)
        OV = _clip(max(0.0, 1.0 - state.budget_remaining), 0.0, 1.0)
        for action in actions:
            current = ci_state.get(action, 0.25)
            narrow_factor = _clip(1 - 0.18 * IF - 0.22 * NEG - 0.15 * HAB, 0.55, 1.0)
            widen_factor = _clip(1 + 0.25 * IL + 0.10 * UNC, 1.0, 1.45)
            overload_guard = _clip(1 + 0.08 * OV, 1.0, 1.15)
            ci_state[action] = _clip(current * narrow_factor * widen_factor * overload_guard, thresholds["ci_min"], thresholds["ci_max"])
        state.action_ci = ci_state
        return ci_state

    def _build_distribution_state(
        self,
        bundles: list[ProposalBundle],
        state: RuntimeState,
        scenario_cfg: dict[str, Any],
        mode_cfg: dict[str, Any],
        relation_state: dict[str, float],
        context: dict[str, Any],
        resample_idx: int = 0,
    ) -> ActionDistributionState:
        actions = sorted({*CORE_ACTIONS, *(action for bundle in bundles for action in bundle.delta_p.keys())})
        thresholds = self.config["thresholds"]["thresholds"]
        p_base = self._build_base_distribution(state, scenario_cfg, mode_cfg, relation_state)
        context_delta = self._compute_context_delta(state, scenario_cfg)
        p_raw: dict[str, float] = {}
        risk_suppressor: dict[str, float] = {}
        gate = {action: 1.0 for action in actions}
        ci = self._update_ci(state, actions, context, scenario_cfg, thresholds)

        for action in actions:
            raw = p_base.get(action, 0.02) + context_delta.get(action, 0.0)
            for bundle in bundles:
                raw += self._agent_weight(bundle.owner, state) * bundle.delta_p.get(action, 0.0) * bundle.confidence
            p_raw[action] = _clip(raw, thresholds["p_floor"], thresholds["p_cap"])
            suppressor = 1.0
            if action == "wander" and not mode_cfg.get("allow_dmn", True):
                suppressor = 0.4
            if action == "connect" and relation_state["boundary_level"] > 0.7:
                suppressor = 0.65
            risk_suppressor[action] = suppressor

        return ActionDistributionState(
            p_base=p_base,
            p_raw=p_raw,
            p_mix={},
            p_final={},
            ci=ci,
            gate=gate,
            risk_suppressor=risk_suppressor,
            resample_idx=resample_idx,
        )

    def _normalize(self, distribution: dict[str, float]) -> dict[str, float]:
        total = sum(max(value, 0.0) for value in distribution.values()) or 1.0
        return {action: max(value, 0.0) / total for action, value in distribution.items()}

    def _kl_divergence(self, q_dist: dict[str, float], p_dist: dict[str, float]) -> float:
        kl = 0.0
        for action, q in q_dist.items():
            p = max(p_dist.get(action, 1e-9), 1e-9)
            q = max(q, 1e-9)
            kl += q * math.log(q / p)
        return kl

    def _stochastic_channel(self, distribution: dict[str, float]) -> str:
        top_action = max(distribution, key=distribution.get)
        if top_action in {"connect"}:
            return "warm"
        if top_action in {"plan", "clarify"}:
            return "controlled"
        if top_action in {"rest", "wander"}:
            return "withdrawn"
        if top_action in {"recall"}:
            return "reflective"
        return "neutral"

    def _trunc_normal(self, rng: random.Random, sigma: float, low: float, high: float) -> float:
        for _ in range(8):
            value = rng.gauss(0.0, sigma)
            if low <= value <= high:
                return value
        return _clip(rng.gauss(0.0, sigma), low, high)

    def _apply_stochastic_layer(
        self,
        deterministic: dict[str, float],
        distribution_state: ActionDistributionState,
        state: RuntimeState,
        event: RoundEvent,
        scenario_cfg: dict[str, Any],
        relation_state: dict[str, float],
        conflict_score: float,
        round_seed: int,
    ) -> tuple[dict[str, float], StochasticState]:
        rng = random.Random(round_seed)
        emo_channel = self._stochastic_channel(deterministic)
        sigma_emo = 0.08 + scenario_cfg.get("output_warmth_variance", 0.1) * 0.12
        sigma_mood = 0.06 + distribution_state.ci.get("respond", 0.25) * 0.12
        xi_emo = self._trunc_normal(rng, sigma_emo, -2 * sigma_emo, 2 * sigma_emo)
        xi_mood = self._trunc_normal(rng, sigma_mood, -sigma_mood, sigma_mood)
        lambda_noise = _clip(0.10 + scenario_cfg.get("output_warmth_variance", 0.1) * 0.40 - relation_state["relationship_risk"] * 0.12, 0.0, 0.60)

        modifiers: dict[str, float] = {}
        for action in deterministic:
            emo_weight = 1.0 if action in {"connect", "respond"} else -0.5 if action in {"rest", "wander"} else 0.4
            mood_weight = 0.6 if action in {"plan", "clarify", "recall"} else 0.2
            log_m = _clip(emo_weight * xi_emo + mood_weight * xi_mood, -0.35, 0.35)
            modifiers[action] = math.exp(log_m)

        q_noise = self._normalize({action: deterministic[action] * modifiers[action] for action in deterministic})
        kl = self._kl_divergence(q_noise, deterministic)
        noise_guard_triggered = False
        if kl > 0.15:
            noise_guard_triggered = True
            lambda_noise = max(0.0, lambda_noise * 0.5)
            q_noise = deterministic
        mixed = {action: (1 - lambda_noise) * deterministic[action] + lambda_noise * q_noise[action] for action in deterministic}

        affect_load = abs(event.valence)
        arousal = _clip(1.0 - state.body_energy + conflict_score * 0.3, 0.0, 1.0)
        privacy_level = relation_state["privacy_level"]
        self_control = _clip(0.55 + (0.18 if state.mode == "task" else 0.0), 0.0, 1.0)
        relationship_risk = relation_state["relationship_risk"]
        mu_int = _clip(
            1
            / (
                1
                + math.exp(
                    -(
                        -0.20
                        + 1.10 * affect_load
                        + 0.55 * arousal
                        + 0.35 * privacy_level
                        - 0.80 * self_control
                        - 0.60 * relationship_risk
                    )
                )
            ),
            0.05,
            0.95,
        )
        V_t = _clip(sum(distribution_state.ci.values()) / max(len(distribution_state.ci), 1), 0.0, 1.0)
        kappa = _clip(16 - 10 * V_t, 4, 16)
        r_intensity = rng.betavariate(mu_int * kappa, (1 - mu_int) * kappa)

        stochastic = StochasticState(
            emo_channel=emo_channel,
            xi_emo=round(xi_emo, 6),
            xi_mood=round(xi_mood, 6),
            lambda_noise=round(lambda_noise, 6),
            r_intensity=round(r_intensity, 6),
            noise_guard_triggered=noise_guard_triggered,
            round_seed=round_seed,
            kl_divergence=round(kl, 6),
        )
        return self._normalize(mixed), stochastic

    def _build_contributions(self, bundles: list[ProposalBundle], state: RuntimeState, selected_action: str, conflict_score: float, fail_score: float, resample_idx: int) -> tuple[list[AgentContribution], list[dict[str, Any]]]:
        contributions: list[AgentContribution] = []
        proposal_records: list[dict[str, Any]] = []
        for bundle in bundles:
            if bundle.veto:
                continue
            top_action = max(bundle.delta_p, key=bundle.delta_p.get) if bundle.delta_p else None
            weight_applied = self._agent_weight(bundle.owner, state)
            selected = selected_action == top_action
            if top_action is not None:
                contributions.append(
                    AgentContribution(
                        agent_name=bundle.owner,
                        action_name=top_action,
                        score=round(bundle.delta_p[top_action] * bundle.confidence * weight_applied, 4),
                        reason=bundle.reason,
                        delta_p=round(bundle.delta_p[top_action], 4),
                        sigma_scale=round(bundle.sigma_scale, 4),
                        confidence=round(bundle.confidence, 4),
                        weight_applied=round(weight_applied, 4),
                        resample_idx=resample_idx,
                        selected=selected,
                        latency_ms=0,
                        provider="rule",
                        model="fallback",
                        tags=list(bundle.trace_tags),
                    )
                )
            proposal_records.append(
                {
                    "stage": next((stage for stage, owners in PIPELINE_ORDER if bundle.owner in owners), "unknown"),
                    "agent_name": bundle.owner,
                    "top_action": top_action,
                    "confidence": round(bundle.confidence, 4),
                    "veto": bundle.veto,
                    "delta_p": bundle.delta_p,
                    "sigma_scale": round(bundle.sigma_scale, 4),
                    "weight_applied": round(weight_applied, 4),
                    "selected": selected,
                    "conflict_score": round(conflict_score, 4),
                    "plausibility_fail_score": round(fail_score, 4),
                    "resample_idx": resample_idx,
                    "tags": list(bundle.trace_tags),
                }
            )
        contributions = sorted(contributions, key=lambda item: item.score, reverse=True)
        return contributions, proposal_records

    def tick(self, event: RoundEvent, scenario: str, mode: str) -> RoundResult:
        state = self.load_runtime_state()
        pre_state = to_dict(state)
        requested_mode = "safe" if state.safe_mode else mode
        mode_cfg = self.config["modes"]["modes"].get(requested_mode, self.config["modes"]["modes"]["interactive"])
        scenario_cfg = self.config["scenarios"]["scenarios"][scenario]
        thresholds = self.config["thresholds"]["thresholds"]

        state.mode = requested_mode
        state.mode_history = (state.mode_history + [requested_mode])[-20:]
        state.round_count += 1
        state.body_energy = _clip(state.body_energy + event.energy_delta)
        state.budget_remaining = _clip(state.budget_remaining - 0.001 + (0.01 if requested_mode in {"idle", "sleep"} else 0.0))

        cue = self.memory_store.ingest_event(event)
        recall_payload = self.memory_store.recall(cue)
        plan_result = self.model_gateway.plan(f"scenario={scenario}; input={event.content}") if self.model_gateway else {
            "text": "",
            "provider": "rule_fallback",
            "model": "fallback",
        }
        context = {
            "cue": cue,
            "recall_strength": self.memory_store.recall_strength(cue),
            "recall": recall_payload,
            "habit_strength": self.memory_store.habit_strength(cue),
            "closeness": self.memory_store.closeness(event.target),
            "valence": event.valence,
            "detail_threshold": thresholds.get("detail_threshold", 0.5),
            "round_gap": 0,
            "interference": 0.0,
            "recent_burn_rate": 1.0 - state.budget_remaining,
            "pfc_model_text": plan_result.get("text", ""),
            "pfc_model_provider": plan_result.get("provider", "rule_fallback"),
            "pfc_model_name": plan_result.get("model", "fallback"),
        }
        relation_state = self._relation_state(event, context)

        pipeline_stages: list[str] = []
        gate_decisions: list[dict[str, Any]] = []
        skill_traces: list[dict[str, Any]] = []
        bundles: list[ProposalBundle] = []
        previous_focus = state.focus
        round_seed = self._round_seed(state, event)

        for stage_name, owners in PIPELINE_ORDER:
            pipeline_stages.append(stage_name)
            if stage_name in {"state_update", "conflict", "thalamus", "plausibility_guard", "forced_mode_switch", "output_gate", "writeback"}:
                continue
            owner = owners[0]
            agent = self.agent_map[owner]
            if not state.agents_enabled.get(owner, True):
                continue
            if owner == "BodyStateAgent":
                patch = self._execute_skill(round_id=state.round_count, skill_name="update_body_state", inputs={"event": to_dict(event), "state": to_dict(state)}, provider=lambda: agent.run_skill("update_body_state", event, state, scenario_cfg, context), skill_traces=skill_traces, seed_ref=round_seed)
                self._apply_state_patch(state, patch.get("state_patch", {}))
                bundle = self._execute_skill(round_id=state.round_count, skill_name="compute_body_bias", inputs={"event": to_dict(event), "state": to_dict(state)}, provider=lambda: agent.run_skill("compute_body_bias", event, state, scenario_cfg, context), skill_traces=skill_traces, fallback_value=ProposalBundle(owner=owner, confidence=0.1, delta_p={"respond": 0.01}, action_preferences={"respond": 0.01}), seed_ref=round_seed)
                veto_info = self._execute_skill(round_id=state.round_count, skill_name="apply_body_veto", inputs={"event": to_dict(event), "state": to_dict(state)}, provider=lambda: agent.run_skill("apply_body_veto", event, state, scenario_cfg, context), skill_traces=skill_traces, seed_ref=round_seed)
                bundle.veto = veto_info.get("veto", False)
                bundles.append(bundle)
                continue
            if owner == "EmotionAgent":
                patch = self._execute_skill(round_id=state.round_count, skill_name="update_affect_state", inputs={"event": to_dict(event), "state": to_dict(state)}, provider=lambda: agent.run_skill("update_affect_state", event, state, scenario_cfg, context), skill_traces=skill_traces, seed_ref=round_seed)
                self._apply_state_patch(state, patch.get("state_patch", {}))
                bundle = self._execute_skill(round_id=state.round_count, skill_name="compute_affect_bias", inputs={"event": to_dict(event), "state": to_dict(state)}, provider=lambda: agent.run_skill("compute_affect_bias", event, state, scenario_cfg, context), skill_traces=skill_traces, fallback_value=ProposalBundle(owner=owner, confidence=0.1, delta_p={"respond": 0.01}, action_preferences={"respond": 0.01}), seed_ref=round_seed)
                veto_info = self._execute_skill(round_id=state.round_count, skill_name="trigger_affect_veto", inputs={"event": to_dict(event), "state": to_dict(state)}, provider=lambda: agent.run_skill("trigger_affect_veto", event, state, scenario_cfg, context), skill_traces=skill_traces, seed_ref=round_seed)
                bundle.veto = veto_info.get("veto", False)
                bundles.append(bundle)
                continue
            if owner == "RelationshipAgent":
                closeness = self._execute_skill(round_id=state.round_count, skill_name="score_closeness", inputs={"target": event.target, "context": context}, provider=lambda: agent.run_skill("score_closeness", event, state, scenario_cfg, context), skill_traces=skill_traces, seed_ref=round_seed)
                context["closeness"] = closeness.get("score", context["closeness"])
                gate_info = self._execute_skill(round_id=state.round_count, skill_name="compute_boundary_gate", inputs={"target": event.target, "context": context}, provider=lambda: agent.run_skill("compute_boundary_gate", event, state, scenario_cfg, context), skill_traces=skill_traces, seed_ref=round_seed)
                relation_state["boundary_level"] = gate_info.get("boundary_level", relation_state["boundary_level"])
                self._execute_skill(round_id=state.round_count, skill_name="update_relation_trace", inputs={"target": event.target, "event": to_dict(event)}, provider=lambda: agent.run_skill("update_relation_trace", event, state, scenario_cfg, context), skill_traces=skill_traces, seed_ref=round_seed)
                bundles.append(agent.propose(event, state, scenario_cfg, context))
                continue
            if owner == "ResourceAgent":
                self._execute_skill(round_id=state.round_count, skill_name="compute_scarcity_index", inputs={"state": to_dict(state)}, provider=lambda: agent.run_skill("compute_scarcity_index", event, state, scenario_cfg, context), skill_traces=skill_traces, seed_ref=round_seed)
                bundle = self._execute_skill(round_id=state.round_count, skill_name="map_budget_to_bias", inputs={"state": to_dict(state)}, provider=lambda: agent.run_skill("map_budget_to_bias", event, state, scenario_cfg, context), skill_traces=skill_traces, fallback_value=ProposalBundle(owner=owner, confidence=0.1, delta_p={"respond": 0.01}, action_preferences={"respond": 0.01}), seed_ref=round_seed)
                mode_hint = self._execute_skill(round_id=state.round_count, skill_name="suggest_resource_mode", inputs={"state": to_dict(state)}, provider=lambda: agent.run_skill("suggest_resource_mode", event, state, scenario_cfg, context), skill_traces=skill_traces, seed_ref=round_seed)
                if mode_hint.get("mode_flag") == "safe":
                    state.safe_mode = True
                bundles.append(bundle)
                continue
            if owner == "PFCAgent":
                bundle = self._execute_skill(round_id=state.round_count, skill_name="generate_candidates", inputs={"event": to_dict(event), "state": to_dict(state)}, provider=lambda: agent.run_skill("generate_candidates", event, state, scenario_cfg, context), skill_traces=skill_traces, fallback_value=ProposalBundle(owner=owner, confidence=0.1, delta_p={"respond": 0.02}, action_preferences={"respond": 0.02}), seed_ref=round_seed)
                self._execute_skill(round_id=state.round_count, skill_name="estimate_plan_depth", inputs={"event": to_dict(event)}, provider=lambda: agent.run_skill("estimate_plan_depth", event, state, scenario_cfg, context), skill_traces=skill_traces, seed_ref=round_seed)
                self._execute_skill(round_id=state.round_count, skill_name="bind_working_memory", inputs={"event": to_dict(event), "state": to_dict(state)}, provider=lambda: agent.run_skill("bind_working_memory", event, state, scenario_cfg, context), skill_traces=skill_traces, seed_ref=round_seed)
                bundles.append(bundle)
                continue
            if owner == "HabitAgent":
                self._execute_skill(round_id=state.round_count, skill_name="compute_feedback_decay", inputs={"context": context}, provider=lambda: agent.run_skill("compute_feedback_decay", event, state, scenario_cfg, context), skill_traces=skill_traces, seed_ref=round_seed)
                self._execute_skill(round_id=state.round_count, skill_name="update_habit_strength", inputs={"context": context}, provider=lambda: agent.run_skill("update_habit_strength", event, state, scenario_cfg, context), skill_traces=skill_traces, seed_ref=round_seed)
                bundle = self._execute_skill(round_id=state.round_count, skill_name="suggest_default_action", inputs={"context": context}, provider=lambda: agent.run_skill("suggest_default_action", event, state, scenario_cfg, context), skill_traces=skill_traces, fallback_value=ProposalBundle(owner=owner, confidence=0.1, delta_p={}, action_preferences={}), seed_ref=round_seed)
                self._execute_skill(round_id=state.round_count, skill_name="estimate_override_cost", inputs={"context": context}, provider=lambda: agent.run_skill("estimate_override_cost", event, state, scenario_cfg, context), skill_traces=skill_traces, seed_ref=round_seed)
                bundles.append(bundle)
                continue
            if owner == "DesireAgent":
                self._execute_skill(round_id=state.round_count, skill_name="score_immediate_reward", inputs={"event": to_dict(event)}, provider=lambda: agent.run_skill("score_immediate_reward", event, state, scenario_cfg, context), skill_traces=skill_traces, seed_ref=round_seed)
                bundle = self._execute_skill(round_id=state.round_count, skill_name="compute_effort_avoidance", inputs={"event": to_dict(event), "state": to_dict(state)}, provider=lambda: agent.run_skill("compute_effort_avoidance", event, state, scenario_cfg, context), skill_traces=skill_traces, fallback_value=ProposalBundle(owner=owner, confidence=0.1, delta_p={}, action_preferences={}), seed_ref=round_seed)
                self._execute_skill(round_id=state.round_count, skill_name="suggest_low_cost_action", inputs={"event": to_dict(event)}, provider=lambda: agent.run_skill("suggest_low_cost_action", event, state, scenario_cfg, context), skill_traces=skill_traces, seed_ref=round_seed)
                bundles.append(bundle)
                continue
            if owner == "DMNAgent":
                bundle = self._execute_skill(round_id=state.round_count, skill_name="sample_dmn_intrusion", inputs={"state": to_dict(state)}, provider=lambda: agent.run_skill("sample_dmn_intrusion", event, state, scenario_cfg, context), skill_traces=skill_traces, fallback_value=ProposalBundle(owner=owner, confidence=0.1, delta_p={}, action_preferences={}), seed_ref=round_seed)
                self._execute_skill(round_id=state.round_count, skill_name="score_rumination_pull", inputs={"state": to_dict(state)}, provider=lambda: agent.run_skill("score_rumination_pull", event, state, scenario_cfg, context), skill_traces=skill_traces, seed_ref=round_seed)
                self._execute_skill(round_id=state.round_count, skill_name="select_spontaneous_topic", inputs={"context": context}, provider=lambda: agent.run_skill("select_spontaneous_topic", event, state, scenario_cfg, context), skill_traces=skill_traces, seed_ref=round_seed)
                bundles.append(bundle)
                continue
            if owner == "HippocampusAgent":
                self._execute_skill(round_id=state.round_count, skill_name="encode_episode", inputs={"event": to_dict(event), "state": to_dict(state)}, provider=lambda: agent.run_skill("encode_episode", event, state, scenario_cfg, context), skill_traces=skill_traces, seed_ref=round_seed)
                recall_set = self._execute_skill(round_id=state.round_count, skill_name="retrieve_by_cue", inputs={"context": context}, provider=lambda: agent.run_skill("retrieve_by_cue", event, state, scenario_cfg, context), skill_traces=skill_traces, seed_ref=round_seed)
                self._execute_skill(round_id=state.round_count, skill_name="apply_cue_weighted_decay", inputs={"context": context}, provider=lambda: agent.run_skill("apply_cue_weighted_decay", event, state, scenario_cfg, context), skill_traces=skill_traces, seed_ref=round_seed)
                self._execute_skill(round_id=state.round_count, skill_name="compute_memory_interference", inputs={"context": context}, provider=lambda: agent.run_skill("compute_memory_interference", event, state, scenario_cfg, context), skill_traces=skill_traces, seed_ref=round_seed)
                if not recall_set.get("recall_set"):
                    self._execute_skill(round_id=state.round_count, skill_name="fallback_to_gist_when_trace_weak", inputs={"context": context}, provider=lambda: agent.run_skill("fallback_to_gist_when_trace_weak", event, state, scenario_cfg, context), skill_traces=skill_traces, seed_ref=round_seed)
                bundles.append(agent.propose(event, state, scenario_cfg, context))
                continue
            if owner == "PerspectiveModel":
                self._execute_skill(round_id=state.round_count, skill_name="infer_other_state", inputs={"event": to_dict(event), "context": context}, provider=lambda: agent.run_skill("infer_other_state", event, state, scenario_cfg, context), skill_traces=skill_traces, seed_ref=round_seed)
                self._execute_skill(round_id=state.round_count, skill_name="simulate_other_reaction", inputs={"event": to_dict(event), "context": context}, provider=lambda: agent.run_skill("simulate_other_reaction", event, state, scenario_cfg, context), skill_traces=skill_traces, seed_ref=round_seed)
                self._execute_skill(round_id=state.round_count, skill_name="estimate_misunderstanding_risk", inputs={"event": to_dict(event), "context": context}, provider=lambda: agent.run_skill("estimate_misunderstanding_risk", event, state, scenario_cfg, context), skill_traces=skill_traces, seed_ref=round_seed)
                bundle = self._execute_skill(round_id=state.round_count, skill_name="adjust_social_interpretation", inputs={"event": to_dict(event), "context": context}, provider=lambda: agent.run_skill("adjust_social_interpretation", event, state, scenario_cfg, context), skill_traces=skill_traces, fallback_value=ProposalBundle(owner=owner, confidence=0.1, delta_p={}, action_preferences={}), seed_ref=round_seed)
                bundles.append(bundle)
                continue
            if owner == "ValueAgent":
                value_scores = self._execute_skill(round_id=state.round_count, skill_name="estimate_subjective_value", inputs={"event": to_dict(event), "context": context}, provider=lambda: agent.run_skill("estimate_subjective_value", event, state, scenario_cfg, context), skill_traces=skill_traces, seed_ref=round_seed)
                self._execute_skill(round_id=state.round_count, skill_name="discount_delayed_reward", inputs={"event": to_dict(event)}, provider=lambda: agent.run_skill("discount_delayed_reward", event, state, scenario_cfg, context), skill_traces=skill_traces, seed_ref=round_seed)
                self._execute_skill(round_id=state.round_count, skill_name="price_social_cost", inputs={"event": to_dict(event), "context": context}, provider=lambda: agent.run_skill("price_social_cost", event, state, scenario_cfg, context), skill_traces=skill_traces, seed_ref=round_seed)
                self._execute_skill(round_id=state.round_count, skill_name="score_uncertainty_penalty", inputs={"state": to_dict(state)}, provider=lambda: agent.run_skill("score_uncertainty_penalty", event, state, scenario_cfg, context), skill_traces=skill_traces, seed_ref=round_seed)
                bundles.append(
                    ProposalBundle(
                        owner=owner,
                        confidence=0.59,
                        action_preferences=value_scores.get("scores", {}),
                        delta_p=value_scores.get("scores", {}),
                        sigma_scale=0.98,
                        utility_shift=value_scores.get("scores", {}),
                        trace_tags=["value"],
                        reason="subjective value re-rank",
                    )
                )
                continue
            if owner == "SalienceAgent":
                bundle = self._execute_skill(round_id=state.round_count, skill_name="score_salience", inputs={"event": to_dict(event), "state": to_dict(state)}, provider=lambda: agent.run_skill("score_salience", event, state, scenario_cfg, context), skill_traces=skill_traces, fallback_value=ProposalBundle(owner=owner, confidence=0.1, delta_p={"respond": 0.02}, action_preferences={"respond": 0.02}), seed_ref=round_seed)
                self._execute_skill(round_id=state.round_count, skill_name="switch_mode", inputs={"event": to_dict(event), "state": to_dict(state)}, provider=lambda: agent.run_skill("switch_mode", event, state, scenario_cfg, context), skill_traces=skill_traces, seed_ref=round_seed)
                self._execute_skill(round_id=state.round_count, skill_name="interrupt_current_focus", inputs={"event": to_dict(event), "state": to_dict(state)}, provider=lambda: agent.run_skill("interrupt_current_focus", event, state, scenario_cfg, context), skill_traces=skill_traces, seed_ref=round_seed)
                self._execute_skill(round_id=state.round_count, skill_name="promote_event_to_workspace", inputs={"event": to_dict(event), "state": to_dict(state)}, provider=lambda: agent.run_skill("promote_event_to_workspace", event, state, scenario_cfg, context), skill_traces=skill_traces, seed_ref=round_seed)
                bundles.append(bundle)
                continue
            if owner == "UnconsciousAgent":
                self._execute_skill(round_id=state.round_count, skill_name="load_temperament", inputs={"state": to_dict(state)}, provider=lambda: agent.run_skill("load_temperament", event, state, scenario_cfg, context), skill_traces=skill_traces, seed_ref=round_seed)
                bundle = self._execute_skill(round_id=state.round_count, skill_name="compute_trait_bias", inputs={"state": to_dict(state)}, provider=lambda: agent.run_skill("compute_trait_bias", event, state, scenario_cfg, context), skill_traces=skill_traces, fallback_value=ProposalBundle(owner=owner, confidence=0.1, delta_p={"respond": 0.01}, action_preferences={"respond": 0.01}), seed_ref=round_seed)
                patch = self._execute_skill(round_id=state.round_count, skill_name="apply_chronic_shift", inputs={"state": to_dict(state)}, provider=lambda: agent.run_skill("apply_chronic_shift", event, state, scenario_cfg, context), skill_traces=skill_traces, seed_ref=round_seed)
                self._apply_state_patch(state, patch.get("state_patch", {}))
                bundles.append(bundle)
                continue
            if owner == "CerebellarPredictor":
                self._execute_skill(round_id=state.round_count, skill_name="predict_next_state", inputs={"state": to_dict(state)}, provider=lambda: agent.run_skill("predict_next_state", event, state, scenario_cfg, context), skill_traces=skill_traces, seed_ref=round_seed)
                self._execute_skill(round_id=state.round_count, skill_name="compute_prediction_error", inputs={"state": to_dict(state)}, provider=lambda: agent.run_skill("compute_prediction_error", event, state, scenario_cfg, context), skill_traces=skill_traces, seed_ref=round_seed)
                self._execute_skill(round_id=state.round_count, skill_name="smooth_response_timing", inputs={"state": to_dict(state)}, provider=lambda: agent.run_skill("smooth_response_timing", event, state, scenario_cfg, context), skill_traces=skill_traces, seed_ref=round_seed)
                bundle = self._execute_skill(round_id=state.round_count, skill_name="micro_adjust_action", inputs={"state": to_dict(state)}, provider=lambda: agent.run_skill("micro_adjust_action", event, state, scenario_cfg, context), skill_traces=skill_traces, fallback_value=ProposalBundle(owner=owner, confidence=0.1, delta_p={}, action_preferences={}), seed_ref=round_seed)
                bundles.append(bundle)

        distribution_state = self._build_distribution_state(bundles, state, scenario_cfg, mode_cfg, relation_state, context)

        conflict_agent = self.agent_map["ConflictMonitorAgent"]
        conflict_score = self._execute_skill(
            round_id=state.round_count,
            skill_name="score_conflict",
            inputs={"proposals": [to_dict(bundle) for bundle in bundles], "distribution_state": to_dict(distribution_state)},
            provider=lambda: conflict_agent.run_skill("score_conflict", bundles, distribution_state),
            skill_traces=skill_traces,
            seed_ref=round_seed,
        )["score"]
        escalate = self._execute_skill(
            round_id=state.round_count,
            skill_name="trigger_control_escalation",
            inputs={"score": conflict_score},
            provider=lambda: conflict_agent.run_skill("trigger_control_escalation", conflict_score),
            skill_traces=skill_traces,
            seed_ref=round_seed,
        )["flag"]
        resample_requested = self._execute_skill(
            round_id=state.round_count,
            skill_name="request_resample",
            inputs={"score": conflict_score, "attempts": distribution_state.resample_idx},
            provider=lambda: conflict_agent.run_skill("request_resample", conflict_score, distribution_state.resample_idx),
            skill_traces=skill_traces,
            seed_ref=round_seed,
        )["flag"]
        if escalate:
            gate_decisions.append({"stage": "conflict", "owner": "ConflictMonitorAgent", "allowed": False, "requires_resample": resample_requested, "reason": f"conflict={conflict_score:.2f}"})
        if resample_requested:
            distribution_state.resample_idx += 1
            top_action = max(distribution_state.p_raw, key=distribution_state.p_raw.get)
            distribution_state.p_raw[top_action] = _clip(distribution_state.p_raw[top_action] * 0.85, thresholds["p_floor"], thresholds["p_cap"])

        thalamus = self.agent_map["ThalamusAttentionAgent"]
        aggregated = self._execute_skill(
            round_id=state.round_count,
            skill_name="aggregate_proposals",
            inputs={"distribution_state": to_dict(distribution_state)},
            provider=lambda: thalamus.run_skill("aggregate_proposals", distribution_state),
            skill_traces=skill_traces,
            seed_ref=round_seed,
        )["distribution"]
        deterministic = self._execute_skill(
            round_id=state.round_count,
            skill_name="normalize_distribution",
            inputs={"distribution": aggregated},
            provider=lambda: thalamus.run_skill("normalize_distribution", aggregated),
            skill_traces=skill_traces,
            seed_ref=round_seed,
        )["distribution"]

        stochastic_distribution, stochastic_state = self._apply_stochastic_layer(
            deterministic,
            distribution_state,
            state,
            event,
            scenario_cfg,
            relation_state,
            conflict_score,
            round_seed,
        )
        distribution_state.p_mix = stochastic_distribution
        distribution_state.p_final = dict(stochastic_distribution)

        sampled_action = self._execute_skill(
            round_id=state.round_count,
            skill_name="sample_action",
            inputs={"distribution": distribution_state.p_final},
            provider=lambda: thalamus.run_skill("sample_action", distribution_state.p_final),
            skill_traces=skill_traces,
            seed_ref=round_seed,
        )["action"]

        plausibility_guard = self.agent_map["BehaviorPlausibilityGuard"]
        plausibility_by_action: dict[str, dict[str, Any]] = {}
        for action_name, probability in distribution_state.p_final.items():
            if probability <= thresholds["p_floor"]:
                continue
            plausibility_by_action[action_name] = self._execute_skill(
                round_id=state.round_count,
                skill_name="check_behavior_plausibility",
                inputs={"action": action_name, "scenario": scenario, "relation_state": relation_state, "probability": probability},
                provider=lambda action_name=action_name: plausibility_guard.run_skill("check_behavior_plausibility", action_name, scenario, relation_state),
                skill_traces=skill_traces,
                seed_ref=round_seed,
            )

        blocked_actions = {
            action_name: plausibility
            for action_name, plausibility in plausibility_by_action.items()
            if not plausibility.get("pass", True)
        }
        dominant_blocked_action = None
        dominant_blocked_probability = 0.0
        if blocked_actions:
            dominant_blocked_action = max(blocked_actions, key=lambda action_name: distribution_state.p_final.get(action_name, 0.0))
            dominant_blocked_probability = distribution_state.p_final.get(dominant_blocked_action, 0.0)
        selected_plausibility = plausibility_by_action.get(
            sampled_action.name,
            {"pass": True, "plausibility_fail_score": 0.0},
        )
        fail_score = max(
            selected_plausibility.get("plausibility_fail_score", 0.0),
            blocked_actions.get(dominant_blocked_action, {}).get("plausibility_fail_score", 0.0) if dominant_blocked_action else 0.0,
        )
        top_probability = max(distribution_state.p_final.values()) if distribution_state.p_final else 0.0
        preemptive_guard = dominant_blocked_action is not None and dominant_blocked_probability >= max(0.15, top_probability * 0.85)
        second_sampling = self._execute_skill(
            round_id=state.round_count,
            skill_name="request_second_sampling",
            inputs={"fail_score": fail_score, "attempts": distribution_state.resample_idx},
            provider=lambda: plausibility_guard.run_skill("request_second_sampling", fail_score, distribution_state.resample_idx),
            skill_traces=skill_traces,
            seed_ref=round_seed,
        )["flag"]
        self._execute_skill(
            round_id=state.round_count,
            skill_name="escalate_value_reestimate",
            inputs={"fail_score": fail_score},
            provider=lambda: plausibility_guard.run_skill("escalate_value_reestimate", fail_score),
            skill_traces=skill_traces,
            seed_ref=round_seed,
        )
        if second_sampling and (not selected_plausibility.get("pass", True) or preemptive_guard):
            gated_actions = [
                action_name
                for action_name, plausibility in blocked_actions.items()
                if distribution_state.p_final.get(action_name, 0.0) >= max(0.10, dominant_blocked_probability * 0.7)
            ]
            if not gated_actions and dominant_blocked_action is not None:
                gated_actions = [dominant_blocked_action]
            for action_name in gated_actions:
                distribution_state.gate[action_name] = 0.0
            distribution_state.resample_idx += 1
            filtered = {
                action: distribution_state.p_final[action] * distribution_state.gate.get(action, 1.0)
                for action in distribution_state.p_final
            }
            distribution_state.p_final = self._normalize(filtered)
            sampled_action = thalamus.run_skill("sample_action", distribution_state.p_final)["action"]
        gate_decisions.append(
            {
                "stage": "plausibility_guard",
                "owner": "BehaviorPlausibilityGuard",
                "allowed": not (second_sampling and (not selected_plausibility.get("pass", True) or preemptive_guard)),
                "requires_resample": second_sampling,
                "reason": (
                    f"fail_score={fail_score:.2f}; blocked={dominant_blocked_action or sampled_action.name}; "
                    f"blocked_p={dominant_blocked_probability:.3f}"
                ),
            }
        )

        if sampled_action.name == previous_focus:
            state.focus_lock_count += 1
        else:
            state.focus_lock_count = 1

        forced = self.agent_map["ForcedModeSwitch"]
        lock_score = self._execute_skill(
            round_id=state.round_count,
            skill_name="detect_mode_lock",
            inputs={"history": state.mode_history, "focus_lock_count": state.focus_lock_count},
            provider=lambda: forced.run_skill("detect_mode_lock", state.mode_history, state.focus_lock_count),
            skill_traces=skill_traces,
            seed_ref=round_seed,
        )["lock_score"]
        force_switch = self._execute_skill(
            round_id=state.round_count,
            skill_name="trigger_forced_focus_switch",
            inputs={"lock_score": lock_score},
            provider=lambda: forced.run_skill("trigger_forced_focus_switch", lock_score),
            skill_traces=skill_traces,
            seed_ref=round_seed,
        )["switch_flag"]
        if force_switch:
            sampled_action = ActionCandidate(name="respond", probability=sampled_action.probability, rationale="forced mode switch")
            gate_decisions.append({"stage": "forced_mode_switch", "owner": "ForcedModeSwitch", "allowed": False, "requires_resample": True, "reason": f"lock_score={lock_score:.2f}"})

        output_gate = self.agent_map["OutputGate"]
        gate = self._execute_skill(
            round_id=state.round_count,
            skill_name="apply_output_gate",
            inputs={"action": sampled_action.name, "state": to_dict(state), "scenario": scenario, "relation_state": relation_state},
            provider=lambda: output_gate.run_skill("apply_output_gate", sampled_action.name, state, scenario, relation_state),
            skill_traces=skill_traces,
            seed_ref=round_seed,
        )["gate"]
        distribution_state.gate[sampled_action.name] = min(distribution_state.gate.get(sampled_action.name, 1.0), gate)
        if gate == 0.0:
            filtered = {
                action: distribution_state.p_final[action] * distribution_state.gate.get(action, 1.0)
                for action in distribution_state.p_final
            }
            distribution_state.p_final = self._normalize(filtered)
            sampled_action = thalamus.run_skill("sample_action", distribution_state.p_final)["action"]
        gate_decisions.append({"stage": "output_gate", "owner": "OutputGate", "allowed": gate > 0, "requires_resample": gate == 0.0, "reason": f"gate={gate:.2f}"})

        expression = build_expression_profile(
            sampled_action=sampled_action.name,
            state=to_dict(state),
            scenario=scenario,
            scenario_config=scenario_cfg,
            relation_state=relation_state,
            stochastic=stochastic_state,
        )
        tone_params = self._execute_skill(
            round_id=state.round_count,
            skill_name="render_tone_profile",
            inputs={"expression": to_dict(expression)},
            provider=lambda: output_gate.run_skill("render_tone_profile", expression),
            skill_traces=skill_traces,
            seed_ref=round_seed,
        )
        delay_params = self._execute_skill(
            round_id=state.round_count,
            skill_name="compute_delay_profile",
            inputs={"expression": to_dict(expression)},
            provider=lambda: output_gate.run_skill("compute_delay_profile", expression),
            skill_traces=skill_traces,
            seed_ref=round_seed,
        )

        render_plan = build_render_plan(
            sampled_action=sampled_action.name,
            expression=ExpressionProfile(**{**to_dict(expression)}),
            safety_constraints={"gate": distribution_state.gate.get(sampled_action.name, 1.0), "tone_params": tone_params["tone_params"], "delay_params": delay_params["delay_params"]},
            scenario=scenario,
        )

        state.focus = sampled_action.name
        state.last_action = sampled_action.name
        if sampled_action.name in {"respond", "plan", "clarify"}:
            state.budget_remaining = _clip(state.budget_remaining + 0.0016)
        starvation = self.config["resource_rules"]["resource_defaults"]["starvation_threshold"]
        if state.budget_remaining <= starvation / 10:
            state.safe_mode = True
            state.mode = "safe"

        style_profile = compute_style_profile(to_dict(state), scenario, self.config["output_style"]["styles"])
        sampled_action.metadata["style_profile"] = style_profile
        sampled_action.metadata["expression_profile"] = to_dict(expression)
        sampled_action.metadata["render_plan"] = to_dict(render_plan)

        contributions, proposal_records = self._build_contributions(
            bundles,
            state,
            sampled_action.name,
            conflict_score,
            fail_score,
            distribution_state.resample_idx,
        )

        render_result = (
            self.model_gateway.render(sampled_action.name, style_profile, event.content)
            if self.model_gateway
            else {"text": "", "provider": "rule_fallback", "model": "fallback"}
        )
        sampled_action.metadata["rendered_output"] = render_result["text"]
        sampled_action.metadata["provider"] = render_result["provider"]
        sampled_action.metadata["model"] = render_result["model"]
        state.last_render_provider = render_result["provider"]
        state.last_render_model = render_result["model"]
        state.last_conflict_score = round(conflict_score, 4)
        state.last_plausibility_fail_score = round(fail_score, 4)

        trace = RoundTrace(
            round_id=state.round_count,
            scenario=scenario,
            mode=state.mode,
            sampled_action=sampled_action.name,
            contributions=contributions,
            top_drivers=contributions[:3],
            style_profile=style_profile,
            state_snapshot=to_dict(state),
            pipeline_stages=[stage for stage, _ in PIPELINE_ORDER],
            proposal_summaries=proposal_records,
            gate_decisions=gate_decisions,
            skill_traces=skill_traces,
            distribution_state=to_dict(distribution_state),
            stochastic_state=to_dict(stochastic_state),
            render_plan=to_dict(render_plan),
            resample_count=distribution_state.resample_idx,
            pre_state_snapshot=pre_state,
            event_payload=to_dict(event),
            decision_context=context,
            candidate_distribution=dict(distribution_state.p_final),
            conflict_score=round(conflict_score, 4),
            plausibility_fail_score=round(fail_score, 4),
            rendered_output=render_result["text"],
            provider=render_result["provider"],
            model=render_result["model"],
        )

        trace.state_snapshot["conflict_score"] = round(conflict_score, 4)
        trace.state_snapshot["plausibility_fail_score"] = round(fail_score, 4)
        trace.state_snapshot["last_render_provider"] = render_result["provider"]
        trace.state_snapshot["last_render_model"] = render_result["model"]

        health = HealthEvent(event="tick", status="ok", detail=f"budget={state.budget_remaining:.2f}; provider={render_result['provider']}")
        self._save_state(state)
        self.trace_store.write_round(trace)

        return RoundResult(
            round_id=state.round_count,
            sampled_action=sampled_action,
            trace=trace,
            state=state,
            health=health,
        )

    def apply_command(self, command: str, envelope=None) -> CommandResult:
        state = self.load_runtime_state()
        before_hash = self._state_hash(state)
        parts = command.split()
        operator_level = envelope.operator_level if envelope is not None else _default_operator_level(command)
        result = CommandResult(applied=False, scope="runtime", delta={}, operator_level=operator_level)

        if parts[:2] == ["safe", "on"]:
            state.safe_mode = True
            state.mode = "safe"
            result = CommandResult(applied=True, scope="runtime", delta={"safe_mode": True, "mode": "safe"}, risk_note="reduces spontaneity", rollback_hint="alive safe off", operator_level=operator_level, rollback_available=True)
        elif parts[:2] == ["safe", "off"]:
            state.safe_mode = False
            state.mode = "interactive"
            result = CommandResult(applied=True, scope="runtime", delta={"safe_mode": False, "mode": "interactive"}, risk_note="restores full runtime variability", rollback_hint="alive safe on", operator_level=operator_level, rollback_available=True)
        elif parts[:2] == ["mode", "set"] and len(parts) == 3:
            state.mode = parts[2]
            if parts[2] != "safe":
                state.safe_mode = False
            result = CommandResult(applied=True, scope="runtime", delta={"mode": parts[2]}, rollback_hint="alive mode set interactive", operator_level=operator_level, rollback_available=True)
        elif parts[:2] == ["agent", "disable"] and len(parts) == 3:
            state.agents_enabled[parts[2]] = False
            result = CommandResult(applied=True, scope=parts[2], delta={"enabled": False}, ttl="until re-enabled", rollback_hint=f"alive agent enable {parts[2]}", operator_level=operator_level, rollback_available=True)
        elif parts[:2] == ["agent", "enable"] and len(parts) == 3:
            state.agents_enabled[parts[2]] = True
            result = CommandResult(applied=True, scope=parts[2], delta={"enabled": True}, ttl="until changed", rollback_hint=f"alive agent disable {parts[2]}", operator_level=operator_level, rollback_available=True)
        elif parts[:2] == ["body", "rest"]:
            state.body_energy = _clip(state.body_energy + 0.15)
            result = CommandResult(applied=True, scope="body", delta={"body_energy": state.body_energy}, ttl="one round", rollback_hint="energy will decay naturally on next rounds", operator_level=operator_level, rollback_available=False)
        elif parts[:2] == ["mood", "calm"]:
            state.mood = _clip((state.mood + 0.65) / 2)
            result = CommandResult(applied=True, scope="mood", delta={"mood": state.mood}, ttl="one round", rollback_hint="mood will drift from future events", operator_level=operator_level, rollback_available=False)
        elif parts[:2] == ["debug", "weight"] and len(parts) == 4:
            weight = float(parts[3])
            state.agent_weight_overrides[parts[2]] = weight
            result = CommandResult(applied=True, scope=parts[2], delta={"weight": weight}, ttl="until changed", rollback_hint=f"alive debug weight {parts[2]} 1.0", operator_level=operator_level, rollback_available=True)
        elif parts[:2] == ["nudge", "focus"] and len(parts) == 3:
            delta = float(parts[2])
            state.focus_nudge = _clip(state.focus_nudge + delta, -0.5, 0.5)
            result = CommandResult(applied=True, scope="focus", delta={"focus_nudge": state.focus_nudge}, ttl="until changed", rollback_hint="alive nudge focus 0", operator_level=operator_level, rollback_available=True)
        elif parts[:2] == ["suppress", "dmn"]:
            state.agents_enabled["DMNAgent"] = False
            ttl = parts[2] if len(parts) >= 3 else "temporary"
            result = CommandResult(applied=True, scope="DMNAgent", delta={"enabled": False}, ttl=ttl, rollback_hint="alive agent enable DMNAgent", operator_level=operator_level, rollback_available=True)

        self._save_state(state)
        after_hash = self._state_hash(state)
        self.trace_store.append_command(command, result, before_hash, after_hash)
        return result

    def checkpoint(self) -> CheckpointRef:
        state = self.load_runtime_state()
        checkpoint_id = f"ckpt-{state.round_count:04d}"
        path = self.checkpoint_dir / f"{checkpoint_id}.json"
        shutil.copyfile(self.state_path, path)
        state.last_checkpoint_id = checkpoint_id
        self._save_state(state)
        return CheckpointRef(checkpoint_id=checkpoint_id, path=path)

    def rewind(self, checkpoint_id: str) -> CommandResult:
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.json"
        if not checkpoint_path.exists():
            return CommandResult(applied=False, scope="checkpoint", delta={"checkpoint_id": checkpoint_id, "restored": False}, risk_note="checkpoint not found", operator_level="ops_admin", rollback_available=False)
        before_state = self.load_runtime_state()
        before_hash = self._state_hash(before_state)
        shutil.copyfile(checkpoint_path, self.state_path)
        restored_state = self.load_runtime_state()
        after_hash = self._state_hash(restored_state)
        result = CommandResult(applied=True, scope="checkpoint", delta={"checkpoint_id": checkpoint_id, "restored": True, "safe_mode": restored_state.safe_mode, "mode": restored_state.mode}, rollback_hint="create a fresh checkpoint before further changes", operator_level="ops_admin", rollback_available=True)
        self.trace_store.append_command(f"checkpoint rewind {checkpoint_id}", result, before_hash, after_hash)
        return result

    def memory_top(self, limit: int = 5) -> list[dict]:
        return self.memory_store.memory_top(limit=limit)

    def habit_top(self, limit: int = 5) -> list[dict]:
        return self.memory_store.habit_top(limit=limit)

    def state_payload(self) -> dict[str, Any]:
        return to_dict(self.load_runtime_state())

    def trace_round(self, round_id: int) -> dict[str, Any]:
        return self.trace_store.read_round(round_id)

    def why_this(self, round_id: int) -> dict[str, Any]:
        trace = self.trace_round(round_id)
        return {
            "round_id": trace["round_id"],
            "sampled_action": trace["sampled_action"],
            "top_drivers": trace["top_drivers"],
            "style_profile": trace["style_profile"],
            "rendered_output": trace.get("rendered_output", ""),
            "distribution_state": trace.get("distribution_state", {}),
            "stochastic_state": trace.get("stochastic_state", {}),
            "render_plan": trace.get("render_plan", {}),
            "state_snapshot": {
                "mode": trace["state_snapshot"]["mode"],
                "safe_mode": trace["state_snapshot"]["safe_mode"],
                "focus": trace["state_snapshot"]["focus"],
                "budget_remaining": trace["state_snapshot"]["budget_remaining"],
            },
        }

    def contribution_breakdown(self, round_id: int) -> dict[str, Any]:
        trace = self.trace_round(round_id)
        return {"round_id": trace["round_id"], "sampled_action": trace["sampled_action"], "contributions": trace["contributions"]}

    def metrics_summary(self) -> dict[str, Any]:
        rounds = self.trace_store.list_rounds()
        sampled_actions: dict[str, int] = {}
        modes: dict[str, int] = {}
        driver_counts: dict[str, int] = {}
        budget_values: list[float] = []
        safe_mode_rounds = 0
        for trace in rounds:
            sampled_actions[trace["sampled_action"]] = sampled_actions.get(trace["sampled_action"], 0) + 1
            modes[trace["mode"]] = modes.get(trace["mode"], 0) + 1
            budget_values.append(trace["state_snapshot"]["budget_remaining"])
            if trace["state_snapshot"]["safe_mode"]:
                safe_mode_rounds += 1
            for driver in trace["top_drivers"]:
                driver_counts[driver["agent_name"]] = driver_counts.get(driver["agent_name"], 0) + 1
        top_agents = [{"agent_name": name, "count": count} for name, count in sorted(driver_counts.items(), key=lambda item: item[1], reverse=True)[:5]]
        avg_budget = round(sum(budget_values) / len(budget_values), 4) if budget_values else 0.0
        return {
            "total_rounds": len(rounds),
            "sampled_actions": sampled_actions,
            "mode_counts": modes,
            "safe_mode_rounds": safe_mode_rounds,
            "average_budget_remaining": avg_budget,
            "top_agents": top_agents,
        }

    def agent_list(self) -> list[dict[str, Any]]:
        state = self.load_runtime_state()
        known_names = sorted({*self.config["agents"]["agents"].keys(), *self.agent_map.keys()})
        return [
            {
                "name": name,
                "enabled": state.agents_enabled.get(name, self.config["agents"]["agents"].get(name, {}).get("enabled", True)),
                "weight": state.agent_weight_overrides.get(name, self.config["agents"]["agents"].get(name, {}).get("weight", 1.0)),
            }
            for name in known_names
        ]

    def skill_list(self) -> list[dict[str, Any]]:
        return [asdict(spec) for spec in self.skills.values()]

    def skill_stats(self) -> dict[str, Any]:
        base = self.trace_store.skill_stats()
        skill_rows = []
        for name, stats in sorted(base["skills"].items()):
            spec = self.skills.get(name)
            skill_rows.append(
                {
                    "skill_name": name,
                    "name": name,
                    "owner_module": spec.owner_module if spec else "",
                    "timeout_ms": spec.timeout_ms if spec else 0,
                    "cost_class": spec.cost_class if spec else "",
                    "failure_policy": spec.failure_policy if spec else "",
                    "trace_tags": spec.trace_tags if spec else [],
                    "observed_rounds": stats["count"],
                    "contribution_hits": stats["count"],
                    "selected_hits": 0,
                    "average_latency_ms": stats["average_latency_ms"],
                    "degraded_count": stats["degraded_count"],
                }
            )
        return {"total_calls": base["total_calls"], "skills": base["skills"], "skill_rows": skill_rows}

    def skill_profile(self, skill_name: str) -> dict[str, Any]:
        if skill_name not in self.skills:
            raise FileNotFoundError(f"skill {skill_name} not found")
        spec = self.skills[skill_name]
        stats = self.trace_store.skill_stats(skill_name=skill_name)
        return {
            "name": skill_name,
            "skill_name": skill_name,
            "owner_module": spec.owner_module,
            "timeout_ms": spec.timeout_ms,
            "cost_class": spec.cost_class,
            "failure_policy": spec.failure_policy,
            "trace_tags": spec.trace_tags,
            **stats,
        }

    def relation_show(self, target: str) -> dict[str, Any]:
        return self.memory_store.relation_state(target)

    def replay_round(self, round_id: int, seed: int | None = None) -> dict[str, Any]:
        return self.replay(round_id, seed=seed or 0)

    def replay(self, round_id: int, seed: int = 0) -> dict[str, Any]:
        trace = self.trace_round(round_id)
        proposal_rows = trace.get("proposal_summaries", [])
        ablations = []
        for item in proposal_rows[:3]:
            ablations.append(
                {
                    "agent": item.get("agent_name", "unknown"),
                    "action": item.get("top_action", trace["sampled_action"]),
                    "delta": round(item.get("delta_p", {}).get(item.get("top_action"), 0.0), 4) if isinstance(item.get("delta_p"), dict) else 0.0,
                }
            )
        return {
            "round_id": round_id,
            "original_action": trace["sampled_action"],
            "replayed_action": trace["sampled_action"],
            "candidate_distribution": trace.get("candidate_distribution", trace.get("distribution_state", {}).get("p_final", {})),
            "ablations": ablations,
            "seed": seed,
        }

    def why_not(self, round_id: int, action: str) -> dict[str, Any]:
        trace = self.trace_round(round_id)
        candidate_distribution = trace.get("candidate_distribution", trace.get("distribution_state", {}).get("p_final", {}))
        blocked_by = []
        if action not in candidate_distribution:
            blocked_by.append("not_proposed")
        if trace.get("plausibility_fail_score", 0.0) >= self.config["thresholds"]["thresholds"]["plausibility_fail"]:
            blocked_by.append("plausibility_guard")
        blocked_by.extend(item["agent_name"] for item in trace.get("top_drivers", [])[:2])
        return {
            "round_id": round_id,
            "action": action,
            "selected_action": trace["sampled_action"],
            "candidate_score": candidate_distribution.get(action, 0.0),
            "blocked_by": blocked_by,
        }

    def what_changed(self, window: int = 5) -> dict[str, Any]:
        rounds = self.trace_store.list_rounds()[-window:]
        if not rounds:
            return {"window": window, "action_counts": {}, "mode_counts": {}, "budget_delta": 0.0}
        budget_delta = rounds[-1]["state_snapshot"]["budget_remaining"] - rounds[0]["state_snapshot"]["budget_remaining"]
        return {
            "window": window,
            "action_counts": {item["sampled_action"]: sum(1 for row in rounds if row["sampled_action"] == item["sampled_action"]) for item in rounds},
            "mode_counts": {item["mode"]: sum(1 for row in rounds if row["mode"] == item["mode"]) for item in rounds},
            "budget_delta": round(budget_delta, 4),
        }

    def conflict_timeline(self) -> dict[str, Any]:
        points = []
        for trace in self.trace_store.list_rounds():
            points.append(
                {
                    "round_id": trace["round_id"],
                    "conflict_score": max((item.get("conflict_score", 0.0) for item in trace.get("proposal_summaries", [])), default=0.0),
                }
            )
        return {"points": points}

    def metrics_timeline(self) -> dict[str, Any]:
        return {
            "rounds": [
                {
                    "round_id": trace["round_id"],
                    "sampled_action": trace["sampled_action"],
                    "mode": trace["mode"],
                    "conflict_score": trace.get("conflict_score", 0.0),
                    "plausibility_fail_score": trace.get("plausibility_fail_score", 0.0),
                    "budget_remaining": trace["state_snapshot"].get("budget_remaining", 0.0),
                }
                for trace in self.trace_store.list_rounds()
            ]
        }

    def metrics_heatmap(self) -> dict[str, Any]:
        rounds = self.trace_store.list_rounds()
        actions = sorted({item["sampled_action"] for item in rounds} or {"respond"})
        matrix: dict[str, dict[str, int]] = {}
        for item in rounds:
            for contribution in item.get("contributions", []):
                matrix.setdefault(contribution["agent_name"], {action: 0 for action in actions})
                matrix[contribution["agent_name"]].setdefault(contribution["action_name"], 0)
                matrix[contribution["agent_name"]][contribution["action_name"]] += 1
        return {"actions": actions, "matrix": matrix}

    def mode_switch_timeline(self) -> dict[str, Any]:
        points = []
        for trace in self.trace_store.list_rounds():
            points.append(
                {
                    "round_id": trace["round_id"],
                    "mode": trace["mode"],
                    "forced_switch": any(item.get("stage") == "forced_mode_switch" for item in trace.get("gate_decisions", [])),
                }
            )
        return {"points": points}

    def ablation_summary(self) -> dict[str, Any]:
        implicit_modules = ["ResourceAgent", "HippocampusAgent", "PerspectiveModel", "UnconsciousAgent", "CerebellarPredictor"]
        rounds = self.trace_store.list_rounds()
        total_rounds = len(rounds) or 1
        modules = []
        for module_name in implicit_modules:
            appearances = 0
            approx_deltas: list[float] = []
            for trace in rounds:
                proposal_rows = [item for item in trace.get("proposal_summaries", []) if item.get("agent_name") == module_name]
                if not proposal_rows:
                    continue
                appearances += 1
                sampled_action = trace.get("sampled_action")
                distribution_state = trace.get("distribution_state", {})
                p_with = distribution_state.get("p_final", {})
                p_without = dict(distribution_state.get("p_raw", {}))
                for row in proposal_rows:
                    for action_name, delta in row.get("delta_p", {}).items():
                        p_without[action_name] = max(
                            0.0,
                            p_without.get(action_name, 0.0) - row.get("weight_applied", 1.0) * row.get("confidence", 0.0) * delta,
                        )
                filtered = {
                    action_name: p_without.get(action_name, 0.0) * distribution_state.get("gate", {}).get(action_name, 1.0)
                    for action_name in p_without
                }
                normalized_without = self._normalize(filtered)
                approx_deltas.append(
                    round(
                        p_with.get(sampled_action, 0.0) - normalized_without.get(sampled_action, 0.0),
                        6,
                    )
                )
            modules.append(
                {
                    "module": module_name,
                    "observed_rounds": appearances,
                    "coverage": round(appearances / total_rounds, 4),
                    "approx_gain": round(sum(approx_deltas) / len(approx_deltas), 6) if approx_deltas else 0.0,
                    "max_gain": round(max(approx_deltas), 6) if approx_deltas else 0.0,
                }
            )
        return {"modules": modules}

    def compact_traces(self) -> dict[str, Any]:
        return self.trace_store.compact_rounds()

    def eval_longrun(self, rounds: int = 1000) -> dict[str, Any]:
        start_round = self.load_runtime_state().round_count
        for idx in range(rounds):
            self.tick(
                RoundEvent(
                    source="simulation",
                    content=f"synthetic round {idx} task update",
                    target="sim-user",
                    cue=f"topic-{idx % 7}",
                    valence=0.05 if idx % 2 == 0 else -0.02,
                ),
                scenario="task" if idx % 3 == 0 else "chat",
                mode="interactive",
            )
        generated_rounds = self.trace_store.list_rounds()[start_round:]
        summary = self.metrics_summary()
        total = len(generated_rounds) or 1
        critical_conflicts = [item for item in generated_rounds if item.get("conflict_score", 0.0) >= self.config["thresholds"]["thresholds"]["conflict_critical"]]
        safe_mode_rounds = sum(1 for item in generated_rounds if item["state_snapshot"].get("safe_mode"))
        scarce_threshold = self.config["resource_rules"]["resource_defaults"]["scarce_threshold"]
        scarce_round = next((idx for idx, item in enumerate(generated_rounds) if item["state_snapshot"].get("budget_remaining", 1.0) <= scarce_threshold), None)
        scarcity_burn_drop = 0.0
        if scarce_round is not None and scarce_round >= 3 and scarce_round + 3 < len(generated_rounds):
            before = generated_rounds[scarce_round - 3 : scarce_round]
            after = generated_rounds[scarce_round : scarce_round + 3]
            before_burn = before[0]["state_snapshot"]["budget_remaining"] - before[-1]["state_snapshot"]["budget_remaining"]
            after_burn = after[0]["state_snapshot"]["budget_remaining"] - after[-1]["state_snapshot"]["budget_remaining"]
            if before_burn > 0:
                scarcity_burn_drop = round(max(0.0, (before_burn - after_burn) / before_burn), 4)
        recall_gist = sum(1 for item in generated_rounds if item.get("decision_context", {}).get("recall", {}).get("mode") == "gist")
        recall_detail = sum(1 for item in generated_rounds if item.get("decision_context", {}).get("recall", {}).get("mode") == "detail")
        task_rounds = [item for item in generated_rounds if item["scenario"] == "task"]
        task_successes = sum(1 for item in task_rounds if item["sampled_action"] in {"respond", "plan", "recall", "clarify"})
        relation_checks = []
        for item in generated_rounds:
            target = item.get("event_payload", {}).get("target")
            if not target:
                continue
            closeness = item.get("decision_context", {}).get("closeness", 0.5)
            action = item["sampled_action"]
            relation_checks.append(action in {"connect", "clarify", "respond", "recall"} if closeness >= 0.55 else action != "connect")
        habit_strengths = [item["strength"] for item in self.habit_top(limit=10)]
        summary["generated_rounds"] = total
        summary["crash_rate"] = 0.0
        summary["safe_mode_rate"] = round(safe_mode_rounds / total, 4)
        summary["conflict_deadloop_rate"] = round(len(critical_conflicts) / total, 4)
        summary["scarcity_burn_drop"] = scarcity_burn_drop
        summary["habit_gradient"] = round((sum(habit_strengths) / max(len(habit_strengths), 1)) / total, 4)
        summary["gist_detail_ratio"] = round(recall_gist / max(recall_detail, 1), 4)
        summary["relation_consistency"] = round(sum(1 for item in relation_checks if item) / max(len(relation_checks), 1), 4)
        summary["task_success_rate"] = round(task_successes / max(len(task_rounds), 1), 4)
        summary["top_driver_coverage"] = round(sum(1 for item in generated_rounds if len(item.get("top_drivers", [])) >= 3) / total, 4)
        return summary
