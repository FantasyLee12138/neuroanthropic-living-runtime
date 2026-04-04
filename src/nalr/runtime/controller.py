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

from nalr.agents.modules import TEMPLATE_ACTION_SCALES, TEMPLATE_BY_PRIORITY, build_agents
from nalr.memory.store import MemoryStore
from nalr.output.renderer import fallback_render_text
from nalr.output.style import build_expression_profile, build_render_plan, compute_style_profile
from nalr.providers import ModelRequest, ModelRouter
from nalr.runtime.metadata import iso_date, utc_now_iso
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
    RenderedExpression,
    RoundEvent,
    RoundResult,
    RoundTrace,
    RuntimeState,
    SkillRuntimeContext,
    StochasticState,
    to_dict,
)
from nalr.skills.executor import SkillExecutor
from nalr.skills.registry import build_skill_registry, serialize_contract
from nalr.trace.exporter import TraceExporter
from nalr.trace.store import TraceStore


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


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
    ("late_perspective", ("PerspectiveModel",)),
    ("renderer", ()),
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
        self.skill_executor = SkillExecutor(self.skills, circuit_breaker_path=self.memory_store.circuit_breaker_path)
        self.model_router = ModelRouter.from_config(self.config["models"])

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
            "models": read_yaml("models.yaml"),
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
        weight = state.agent_weight_overrides.get(agent_name, base)
        if agent_name == "PFCAgent" and state.conflict_hot_rounds > 0:
            weight *= 1.15
        return weight

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
                "fallback_route": result.fallback_route,
                "fallback_cost_class": result.fallback_cost_class,
                "policy_rejection_reason": result.policy_rejection_reason,
                "breaker_state": result.breaker_state,
            }
        )

    def _runtime_skill_inputs(
        self,
        event: RoundEvent,
        state: RuntimeState,
        scenario_cfg: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "event": event,
            "state": state,
            "scenario": scenario_cfg,
            "context": context,
        }

    def _skill_runtime_context(self, round_id: int, scenario: str, state: RuntimeState) -> SkillRuntimeContext:
        return SkillRuntimeContext(
            round_id=round_id,
            scenario=scenario,
            mode=state.mode,
            safe_mode=state.safe_mode,
        )

    def _agent_provider(self, agent: Any, skill_name: str):
        return lambda **skill_inputs: agent.run_skill(skill_name, **skill_inputs)

    def _execute_skill(
        self,
        *,
        round_id: int,
        skill_name: str,
        inputs: dict[str, Any],
        provider,
        skill_traces: list[dict[str, Any]],
        runtime_context: SkillRuntimeContext,
        fallback_provider=None,
        fallback_value: Any | None = None,
        seed_ref: int | None = None,
    ):
        output, result = self.skill_executor.run(
            round_id=round_id,
            skill_name=skill_name,
            inputs=inputs,
            provider=provider,
            fallback_provider=fallback_provider,
            fallback_value=fallback_value,
            runtime_context=runtime_context,
            seed_ref=seed_ref,
        )
        self._record_skill_trace(skill_traces, round_id, result)
        return output

    def _execute_skill_with_result(
        self,
        *,
        round_id: int,
        skill_name: str,
        inputs: dict[str, Any],
        provider,
        skill_traces: list[dict[str, Any]],
        runtime_context: SkillRuntimeContext,
        fallback_provider=None,
        fallback_value: Any | None = None,
        seed_ref: int | None = None,
    ):
        output, result = self.skill_executor.run(
            round_id=round_id,
            skill_name=skill_name,
            inputs=inputs,
            provider=provider,
            fallback_provider=fallback_provider,
            fallback_value=fallback_value,
            runtime_context=runtime_context,
            seed_ref=seed_ref,
        )
        self._record_skill_trace(skill_traces, round_id, result)
        return output, result

    def _clip_delta(self, value: float) -> float:
        return _clip(value, -0.35, 0.35)

    def _json_prompt(self, payload: dict[str, Any]) -> str:
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)

    def _coerce_score_map(self, value: Any) -> dict[str, float]:
        if isinstance(value, dict):
            normalized: dict[str, float] = {}
            for action, score in value.items():
                if isinstance(action, str) and isinstance(score, (int, float)):
                    normalized[action] = float(score)
            return normalized

        if not isinstance(value, list):
            return {}

        normalized: dict[str, float] = {}
        for item in value:
            if isinstance(item, dict):
                action = item.get("action") or item.get("name") or item.get("key")
                score = None
                for field_name in ("score", "value", "weight", "preference"):
                    candidate = item.get(field_name)
                    if isinstance(candidate, (int, float)):
                        score = float(candidate)
                        break
                if isinstance(action, str) and score is not None:
                    normalized[action] = score
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                action, score = item
                if isinstance(action, str) and isinstance(score, (int, float)):
                    normalized[action] = float(score)
        return normalized

    def _generate_pfc_candidates_via_model(
        self,
        event: RoundEvent,
        state: RuntimeState,
        scenario: dict[str, Any],
        context: dict[str, Any],
    ) -> ProposalBundle:
        response = self.model_router.generate(
            "pfc",
            ModelRequest(
                system_prompt="你是 PFCAgent。只返回 JSON，不要额外解释。输出 action_preferences、confidence、sigma_scale、reason。",
                user_prompt=self._json_prompt(
                    {
                        "event": to_dict(event),
                        "state": to_dict(state),
                        "scenario": scenario,
                        "context": context,
                    }
                ),
                response_schema={
                    "action_preferences": "dict[str, float]",
                    "confidence": "float",
                    "sigma_scale": "float",
                    "reason": "str",
                },
                metadata={
                    "scenario": scenario.get("name", ""),
                    "cue": context.get("cue"),
                },
            ),
        )
        prefs = {
            action: self._clip_delta(float(score))
            for action, score in self._coerce_score_map(response.payload.get("action_preferences", {})).items()
        }
        return ProposalBundle(
            owner="PFCAgent",
            confidence=_clip(float(response.payload.get("confidence", 0.84)), 0.0, 1.0),
            action_preferences=prefs,
            delta_p=prefs,
            sigma_scale=_clip(float(response.payload.get("sigma_scale", 0.92)), 0.60, 1.60),
            trace_tags=["pfc", "model"],
            reason=str(response.payload.get("reason", f"model route={response.route}")),
        )

    def _should_run_late_perspective(
        self,
        event: RoundEvent,
        relation_state: dict[str, Any],
        sampled_action: str,
    ) -> bool:
        perspective_cfg = self.config["models"].get("perspective", {})
        if not perspective_cfg.get("enabled", True):
            return False
        if not event.target:
            return False
        action_bump = 0.08 if sampled_action in {"clarify", "connect"} else 0.0
        risk_score = max(relation_state.get("relationship_risk", 0.0), abs(event.valence) * 0.5 + action_bump)
        return risk_score >= perspective_cfg.get("risk_threshold", 0.33)

    def _infer_other_state_via_model(
        self,
        event: RoundEvent,
        state: RuntimeState,
        scenario: dict[str, Any],
        context: dict[str, Any],
        sampled_action: str,
        relation_state: dict[str, Any],
    ) -> dict[str, Any]:
        response = self.model_router.generate(
            "perspective",
            ModelRequest(
                system_prompt="你负责推断对方当前状态。只返回 JSON，不要额外解释。输出 state_hypothesis。",
                user_prompt=self._json_prompt(
                    {
                        "event": to_dict(event),
                        "state": to_dict(state),
                        "scenario": scenario,
                        "context": context,
                        "sampled_action": sampled_action,
                        "relation_state": relation_state,
                    }
                ),
                response_schema={"state_hypothesis": "dict"},
                metadata={
                    "sampled_action": sampled_action,
                    "closeness": context.get("closeness", 0.5),
                    "relationship_risk": relation_state.get("relationship_risk", 0.0),
                },
            ),
        )
        return {"state_hypothesis": response.payload.get("state_hypothesis", {})}

    def _simulate_other_reaction_via_model(
        self,
        event: RoundEvent,
        state: RuntimeState,
        scenario: dict[str, Any],
        context: dict[str, Any],
        sampled_action: str,
        relation_state: dict[str, Any],
    ) -> dict[str, Any]:
        response = self.model_router.generate(
            "perspective",
            ModelRequest(
                system_prompt="你负责模拟对方对最终动作的反应。只返回 JSON，不要额外解释。输出 reaction_hypothesis。",
                user_prompt=self._json_prompt(
                    {
                        "event": to_dict(event),
                        "state": to_dict(state),
                        "scenario": scenario,
                        "context": context,
                        "final_action": sampled_action,
                        "relation_state": relation_state,
                    }
                ),
                response_schema={"reaction_hypothesis": "dict"},
                metadata={
                    "sampled_action": sampled_action,
                    "closeness": context.get("closeness", 0.5),
                    "relationship_risk": relation_state.get("relationship_risk", 0.0),
                },
            ),
        )
        return {"reaction_hypothesis": response.payload.get("reaction_hypothesis", {})}

    def _render_expression_via_model(self, render_plan: RenderPlan) -> dict[str, Any]:
        response = self.model_router.generate(
            "renderer",
            ModelRequest(
                system_prompt="你是最终表达 renderer。只返回 JSON，不要额外解释。输出 text。",
                user_prompt=self._json_prompt({"render_plan": to_dict(render_plan)}),
                response_schema={"text": "str"},
                metadata={"action": render_plan.action},
            ),
        )
        return {
            "text": str(response.payload.get("text", "")).strip(),
            "route": response.route,
            "model": response.model,
        }

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
        u_base: dict[str, float] = {}
        u_shifted: dict[str, float] = {}
        p_raw: dict[str, float] = {}
        risk_suppressor: dict[str, float] = {}
        gate = {action: 1.0 for action in actions}
        ci = self._update_ci(state, actions, context, scenario_cfg, thresholds)

        for action in actions:
            base_probability = p_base.get(action, 0.02)
            raw = base_probability + context_delta.get(action, 0.0)
            base_utility = math.log(max(base_probability, thresholds["p_floor"]))
            shifted = base_utility + context_delta.get(action, 0.0)
            for bundle in bundles:
                weight = self._agent_weight(bundle.owner, state)
                weighted_delta = weight * bundle.delta_p.get(action, 0.0) * bundle.confidence * bundle.sigma_scale
                weighted_utility = weight * bundle.utility_shift.get(action, 0.0) * max(0.35, bundle.confidence)
                raw += weighted_delta + weighted_utility
                shifted += weighted_delta + weighted_utility
            u_base[action] = round(base_utility, 6)
            u_shifted[action] = round(shifted, 6)
            p_raw[action] = _clip(raw, thresholds["p_floor"], thresholds["p_cap"])
            suppressor = 1.0
            if action == "wander" and not mode_cfg.get("allow_dmn", True):
                suppressor = 0.4
            if action == "connect" and relation_state["boundary_level"] > 0.7:
                suppressor = 0.65
            risk_suppressor[action] = suppressor

        return ActionDistributionState(
            u_base=u_base,
            u_shifted=u_shifted,
            p_base=p_base,
            p_raw=p_raw,
            p_base_stochastic={},
            q_noise={},
            p_mix={},
            p_final={},
            ci=ci,
            gate=gate,
            risk_suppressor=risk_suppressor,
            resample_idx=resample_idx,
            conflict_mode="none",
            conflict={},
        )

    def _enrich_bundle_metadata(
        self,
        bundle: ProposalBundle,
        state: RuntimeState,
        relation_state: dict[str, float],
        context: dict[str, Any],
    ) -> ProposalBundle:
        owner = bundle.owner
        if not getattr(bundle, "priority_bucket", None):
            bundle.priority_bucket = {
                "BodyStateAgent": "body_safety",
                "EmotionAgent": "body_safety",
                "ResourceAgent": "budget_overload",
                "RelationshipAgent": "relation_boundary",
                "PerspectiveModel": "relation_boundary",
                "DesireAgent": "immediate_desire",
                "DMNAgent": "roaming",
            }.get(owner, "task_goal")
        if not getattr(bundle, "control_domain", None):
            bundle.control_domain = {
                "BodyStateAgent": "body",
                "EmotionAgent": "body",
                "ResourceAgent": "resource",
                "RelationshipAgent": "relation",
                "PerspectiveModel": "relation",
                "DesireAgent": "desire",
                "DMNAgent": "dmn",
            }.get(owner, "task")
        bundle.gated_actions = list(getattr(bundle, "gated_actions", []))
        bundle.risk_hints = dict(getattr(bundle, "risk_hints", {}))

        if owner == "BodyStateAgent":
            bundle.risk_hints.setdefault("body_load", round(_clip(1.0 - state.body_energy), 4))
            bundle.risk_hints.setdefault("body_energy", round(state.body_energy, 4))
            if state.body_energy < 0.20 and "connect" not in bundle.gated_actions:
                bundle.gated_actions.append("connect")
            if state.body_energy < 0.12 and "plan" not in bundle.gated_actions:
                bundle.gated_actions.append("plan")
        elif owner == "ResourceAgent":
            overload = _clip(1.0 - state.budget_remaining)
            bundle.risk_hints.setdefault("overload", round(overload, 4))
            if overload > 0.80 and "plan" not in bundle.gated_actions:
                bundle.gated_actions.append("plan")
        elif owner in {"RelationshipAgent", "PerspectiveModel"}:
            bundle.risk_hints.setdefault("relationship_risk", round(relation_state["relationship_risk"], 4))
            bundle.risk_hints.setdefault("boundary_level", round(relation_state["boundary_level"], 4))
            if relation_state["boundary_level"] > 0.70 and "connect" not in bundle.gated_actions:
                bundle.gated_actions.append("connect")
        elif owner == "PFCAgent":
            bundle.risk_hints.setdefault("goal_pressure", round(max(bundle.delta_p.values(), default=0.0), 4))
        elif owner == "ValueAgent":
            bundle.risk_hints.setdefault("goal_pressure", round(max(bundle.utility_shift.values(), default=0.0), 4))
        elif owner == "DesireAgent":
            bundle.risk_hints.setdefault("comfort_pull", round(max(bundle.delta_p.values(), default=0.0), 4))
        elif owner == "DMNAgent":
            bundle.risk_hints.setdefault("roam_pull", round(bundle.delta_p.get("wander", 0.0), 4))
        elif owner == "HabitAgent":
            bundle.risk_hints.setdefault("habit_strength", round(context.get("habit_strength", 0.0), 4))

        return bundle

    def _apply_conflict_scales(
        self,
        distribution_state: ActionDistributionState,
        action_scales: dict[str, float],
        thresholds: dict[str, Any],
    ) -> None:
        for action, scale in action_scales.items():
            if action not in distribution_state.p_raw:
                continue
            bounded_scale = _clip(scale, 0.0, 1.50)
            distribution_state.risk_suppressor[action] = min(distribution_state.risk_suppressor.get(action, 1.0), bounded_scale)
            distribution_state.p_raw[action] = _clip(
                distribution_state.p_raw[action] * bounded_scale,
                thresholds["p_floor"],
                thresholds["p_cap"],
            )
            if bounded_scale <= 0.0:
                distribution_state.gate[action] = 0.0

    def _update_conflict_circuit(
        self,
        state: RuntimeState,
        *,
        critical_conflict: bool,
        winning_priority: str | None,
        compromise_template: str | None,
    ) -> dict[str, Any]:
        triggered = False
        if critical_conflict:
            state.critical_conflict_streak += 1
        else:
            state.critical_conflict_streak = 0

        if critical_conflict and state.critical_conflict_streak >= 3:
            state.conflict_hot_rounds = 5
            state.conflict_recovery_rounds = 5
            triggered = True
        elif critical_conflict and state.conflict_hot_rounds > 0:
            state.conflict_hot_rounds = 5
            state.conflict_recovery_rounds = 5
        elif not critical_conflict and state.conflict_hot_rounds > 0:
            state.conflict_hot_rounds = max(0, state.conflict_hot_rounds - 1)
            state.conflict_recovery_rounds = max(0, state.conflict_recovery_rounds - 1)

        if state.conflict_hot_rounds == 0:
            state.last_compromise_template = None
            state.last_conflict_priority = None
        else:
            state.last_compromise_template = compromise_template
            state.last_conflict_priority = winning_priority

        return {
            "triggered": triggered,
            "active": state.conflict_hot_rounds > 0,
            "hot_rounds_remaining": state.conflict_hot_rounds,
            "recovery_rounds_remaining": state.conflict_recovery_rounds,
        }

    def _apply_conflict_expression_adjustments(
        self,
        expression: ExpressionProfile,
        conflict_state: dict[str, Any],
    ) -> ExpressionProfile:
        values = to_dict(expression)
        template = conflict_state.get("compromise", {}).get("template")
        circuit_active = bool(conflict_state.get("circuit_breaker", {}).get("active"))

        if template == "body_first":
            values["reply_delay"] = _clip(values["reply_delay"] + 0.10)
            values["directness_level"] = _clip(values["directness_level"] - 0.10)
            values["hedging_level"] = _clip(values["hedging_level"] + 0.08)
            values["repair_tendency"] = _clip(values["repair_tendency"] + 0.10)
        elif template == "relation_first":
            values["self_disclosure"] = _clip(min(values["self_disclosure"], 0.28))
            values["hedging_level"] = _clip(values["hedging_level"] + 0.12)
            values["repair_tendency"] = _clip(values["repair_tendency"] + 0.10)
        elif template == "task_first":
            values["directness_level"] = _clip(values["directness_level"] + 0.10)
            values["hedging_level"] = _clip(values["hedging_level"] - 0.05)
            values["self_disclosure"] = _clip(min(values["self_disclosure"], 0.18))
        elif template == "budget_first":
            values["self_disclosure"] = _clip(min(values["self_disclosure"], 0.18))
            values["sentence_fragmentation"] = _clip(values["sentence_fragmentation"] - 0.04)
            values["reply_delay"] = _clip(values["reply_delay"] + 0.04)

        if circuit_active:
            values["hedging_level"] = _clip(values["hedging_level"] + 0.15)
            values["repair_tendency"] = _clip(values["repair_tendency"] + 0.15)
            values["directness_level"] = _clip(values["directness_level"] - 0.08)
            values["tone_sharpness"] = _clip(values["tone_sharpness"] - 0.08)
            values["reply_delay"] = _clip(values["reply_delay"] + 0.08)

        return ExpressionProfile(**values)

    def _run_conflict_controller(
        self,
        *,
        state: RuntimeState,
        bundles: list[ProposalBundle],
        distribution_state: ActionDistributionState,
        thresholds: dict[str, Any],
        skill_traces: list[dict[str, Any]],
        runtime_context: SkillRuntimeContext,
        gate_decisions: list[dict[str, Any]],
        round_seed: int,
    ) -> float:
        conflict_agent = self.agent_map["ConflictMonitorAgent"]
        hot_active = state.conflict_hot_rounds > 0
        assessment: dict[str, Any] = {"score": 0.0, "components": {}, "priority_signals": {}, "critical_conflict": False}
        resolution: dict[str, Any] = {"flag": False, "winning_priority": None, "blocked_actions": [], "action_scales": {}, "applied_template": None}
        resample_policy: dict[str, Any] = {"flag": False, "allowed_resamples": 0, "force_compromise": False}
        pass_records: list[dict[str, Any]] = []

        for pass_index in range(3):
            assessment = self._execute_skill(
                round_id=state.round_count,
                skill_name="score_conflict",
                inputs={"proposals": bundles, "distribution_state": distribution_state},
                provider=lambda proposals, distribution_state: conflict_agent.run_skill("score_conflict", proposals, distribution_state),
                skill_traces=skill_traces,
                runtime_context=runtime_context,
                seed_ref=round_seed,
            )
            if hot_active:
                assessment["score"] = round(_clip(float(assessment.get("score", 0.0)) + 0.04), 4)
                assessment["total_score"] = assessment["score"]
                assessment["critical_conflict"] = assessment["score"] >= thresholds["conflict_critical"]
                signals = dict(assessment.get("priority_signals", {}))
                signals["body_safety"] = round(max(signals.get("body_safety", 0.0), 0.55), 4)
                assessment["priority_signals"] = signals

            resolution = self._execute_skill(
                round_id=state.round_count,
                skill_name="trigger_control_escalation",
                inputs={
                    "assessment": assessment,
                    "distribution_state": distribution_state,
                    "attempts": pass_index,
                    "hot_active": hot_active,
                },
                provider=lambda assessment, distribution_state, attempts, hot_active: conflict_agent.run_skill(
                    "trigger_control_escalation",
                    assessment,
                    distribution_state,
                    attempts,
                    hot_active,
                ),
                skill_traces=skill_traces,
                runtime_context=runtime_context,
                seed_ref=round_seed,
            )
            resample_policy = self._execute_skill(
                round_id=state.round_count,
                skill_name="request_resample",
                inputs={"assessment": assessment, "attempts": pass_index},
                provider=lambda assessment, attempts: conflict_agent.run_skill("request_resample", assessment, attempts),
                skill_traces=skill_traces,
                runtime_context=runtime_context,
                seed_ref=round_seed,
            )
            if resolution.get("flag"):
                self._apply_conflict_scales(distribution_state, resolution.get("action_scales", {}), thresholds)
                gate_decisions.append(
                    {
                        "stage": "conflict",
                        "owner": "ConflictMonitorAgent",
                        "allowed": False,
                        "requires_resample": bool(resample_policy.get("flag")),
                        "reason": f"pass={pass_index}; priority={resolution.get('winning_priority')}; conflict={assessment.get('score', 0.0):.2f}",
                        "winning_priority": resolution.get("winning_priority"),
                        "template": resolution.get("applied_template"),
                    }
                )
            pass_records.append(
                {
                    "pass_index": pass_index,
                    "assessment": to_dict(assessment),
                    "resolution": to_dict(resolution),
                    "resample_requested": bool(resample_policy.get("flag")),
                }
            )
            if not resample_policy.get("flag"):
                break
            distribution_state.resample_idx += 1

        compromise = {
            "triggered": False,
            "template": resolution.get("applied_template"),
            "winning_priority": resolution.get("winning_priority"),
            "reason": "",
        }
        if resolution.get("flag") and resample_policy.get("force_compromise") and not resample_policy.get("flag"):
            template = resolution.get("applied_template") or TEMPLATE_BY_PRIORITY.get(
                resolution.get("winning_priority"), "task_first"
            )
            compromise = {
                "triggered": True,
                "template": template,
                "winning_priority": resolution.get("winning_priority"),
                "reason": "max_resample_reached",
            }
            resolution["applied_template"] = template
            self._apply_conflict_scales(
                distribution_state,
                TEMPLATE_ACTION_SCALES.get(template, {}),
                thresholds,
            )

        circuit = self._update_conflict_circuit(
            state,
            critical_conflict=bool(assessment.get("critical_conflict")),
            winning_priority=resolution.get("winning_priority"),
            compromise_template=compromise["template"] if compromise["triggered"] else None,
        )
        blocked_by_circuit = []
        if circuit["active"]:
            high_risk_actions = ("connect", "plan", "wander")
            for action in high_risk_actions:
                if action in distribution_state.p_raw:
                    distribution_state.gate[action] = 0.0
                    distribution_state.p_raw[action] = thresholds["p_floor"]
                    blocked_by_circuit.append(action)

        # Deadlock fuse (§8.5): when the circuit just triggered (3 consecutive
        # critical rounds), enter repair mode and force safe_mode.
        deadlock_fuse_triggered = False
        if circuit["triggered"]:
            state.repair_mode = "deadlock_fuse"
            state.safe_mode = True
            deadlock_fuse_triggered = True

        distribution_state.conflict = {
            "score": assessment.get("score", 0.0),
            "total_score": assessment.get("score", 0.0),
            "components": dict(assessment.get("components", {})),
            "priority_signals": dict(assessment.get("priority_signals", {})),
            "dominant_conflicts": list(assessment.get("dominant_conflicts", [])),
            "winning_priority": resolution.get("winning_priority"),
            "passes": pass_records,
            "compromise": compromise,
            "critical_conflict": bool(assessment.get("critical_conflict")),
            "critical_conflict_streak": state.critical_conflict_streak,
            "circuit_breaker": {**circuit, "blocked_actions": blocked_by_circuit},
            "allowed_resamples": resample_policy.get("allowed_resamples", 0),
            "deadlock_fuse_triggered": deadlock_fuse_triggered,
        }
        distribution_state.conflict_mode = "repair" if resample_policy.get("force_compromise") else "monitor"
        return float(assessment.get("score", 0.0))

    def _normalize(self, distribution: dict[str, float]) -> dict[str, float]:
        total = sum(max(value, 0.0) for value in distribution.values()) or 1.0
        return {action: max(value, 0.0) / total for action, value in distribution.items()}

    def _softmax(self, utilities: dict[str, float]) -> dict[str, float]:
        if not utilities:
            return {}
        max_utility = max(utilities.values())
        weights = {action: math.exp(value - max_utility) for action, value in utilities.items()}
        return self._normalize(weights)

    def _parse_budget_cap(self, value: str) -> tuple[int, float]:
        normalized = value.strip().lower().replace("_", "")
        if normalized.endswith("k"):
            cap_value = int(float(normalized[:-1]) * 1000)
        else:
            cap_value = int(float(normalized))
        return cap_value, round(_clip(cap_value / 100000, 0.0, 1.0), 4)

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
        base_stochastic = self._softmax(distribution_state.u_shifted or {action: math.log(max(value, 1e-9)) for action, value in deterministic.items()})
        distribution_state.p_base_stochastic = dict(base_stochastic)
        emo_channel = self._stochastic_channel(base_stochastic)
        sigma_emo = 0.08 + scenario_cfg.get("output_warmth_variance", 0.1) * 0.12
        sigma_mood = 0.06 + distribution_state.ci.get("respond", 0.25) * 0.12
        xi_emo = self._trunc_normal(rng, sigma_emo, -2 * sigma_emo, 2 * sigma_emo)
        xi_mood = self._trunc_normal(rng, sigma_mood, -sigma_mood, sigma_mood)
        V_t = _clip(sum(distribution_state.ci.values()) / max(len(distribution_state.ci), 1), 0.0, 1.0)
        control_strength = _clip(
            0.35
            + max(0.0, scenario_cfg.get("pfc_base_share", 0.2) - 0.2) * 1.2
            + (0.12 if state.safe_mode else 0.0)
            + min(0.12, state.focus_lock_count * 0.02)
            + max(0.0, state.budget_remaining - 0.5) * 0.10,
            0.0,
            1.0,
        )
        lambda_noise = _clip(
            0.10 + 0.25 * V_t - 0.12 * control_strength - relation_state["relationship_risk"] * 0.08,
            0.0,
            0.60,
        )

        modifiers: dict[str, float] = {}
        for action in base_stochastic:
            emo_weight = 1.0 if action in {"connect", "respond"} else -0.5 if action in {"rest", "wander"} else 0.4
            mood_weight = 0.6 if action in {"plan", "clarify", "recall"} else 0.2
            log_m = _clip(emo_weight * xi_emo + mood_weight * xi_mood, -0.35, 0.35)
            modifiers[action] = math.exp(log_m)

        q_noise = self._normalize({action: base_stochastic[action] * modifiers[action] for action in base_stochastic})
        kl = self._kl_divergence(q_noise, base_stochastic)
        noise_guard_triggered = False
        if kl > 0.15:
            noise_guard_triggered = True
            lambda_noise = max(0.0, lambda_noise * 0.5)
            q_noise = dict(base_stochastic)
        distribution_state.q_noise = dict(q_noise)
        mixed = {action: (1 - lambda_noise) * base_stochastic[action] + lambda_noise * q_noise[action] for action in base_stochastic}

        affect_load = abs(event.valence)
        arousal = _clip(1.0 - state.body_energy + conflict_score * 0.3, 0.0, 1.0)
        privacy_level = relation_state["privacy_level"]
        self_control = _clip(0.55 + max(0.0, scenario_cfg.get("pfc_base_share", 0.2) - 0.2) * 1.2 + (0.10 if state.safe_mode else 0.0), 0.0, 1.0)
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
                    )
                )
            proposal_records.append(
                {
                    "stage": next((stage for stage, owners in PIPELINE_ORDER if bundle.owner in owners), "unknown"),
                    "agent_name": bundle.owner,
                    "top_action": top_action,
                    "action_preferences": bundle.action_preferences,
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
                    "priority_bucket": getattr(bundle, "priority_bucket", "task_goal"),
                    "control_domain": getattr(bundle, "control_domain", "task"),
                    "gated_actions": list(getattr(bundle, "gated_actions", [])),
                    "risk_hints": dict(getattr(bundle, "risk_hints", {})),
                }
            )
        contributions = sorted(contributions, key=lambda item: item.score, reverse=True)
        return contributions, proposal_records

    def tick(self, event: RoundEvent, scenario: str, mode: str) -> RoundResult:
        state = self.load_runtime_state()
        requested_mode = "safe" if state.safe_mode else mode
        mode_cfg = self.config["modes"]["modes"].get(requested_mode, self.config["modes"]["modes"]["interactive"])
        scenario_cfg = self.config["scenarios"]["scenarios"][scenario]
        thresholds = self.config["thresholds"]["thresholds"]

        state.mode = requested_mode
        state.mode_history = (state.mode_history + [requested_mode])[-20:]
        state.round_count += 1
        state.body_energy = _clip(state.body_energy + event.energy_delta)
        state.budget_remaining = _clip(state.budget_remaining - 0.001 + (0.01 if requested_mode in {"idle", "sleep"} else 0.0))
        recorded_at = utc_now_iso()
        recorded_date = iso_date(recorded_at)

        cue = self.memory_store.ingest_event(
            event,
            round_id=state.round_count,
            session_id=state.session_id,
            recorded_at=recorded_at,
            update_habit=False,
        )
        context = {
            "cue": cue,
            "recall_strength": self.memory_store.recall_strength(cue),
            "habit_strength": self.memory_store.habit_strength(cue),
            "closeness": self.memory_store.closeness(event.target),
            "valence": event.valence,
            "detail_threshold": thresholds.get("detail_threshold", 0.5),
            "round_gap": 0,
            "interference": 0.0,
            "recent_burn_rate": 1.0 - state.budget_remaining,
        }
        if cue:
            recall_payload = self.memory_store.recall(cue)
            context["recall_strength"] = recall_payload.get("strength", context["recall_strength"])
            context["interference"] = recall_payload.get("interference", 0.0)
        relation_state = self._relation_state(event, context)

        pipeline_stages: list[str] = []
        gate_decisions: list[dict[str, Any]] = []
        skill_traces: list[dict[str, Any]] = []
        bundles: list[ProposalBundle] = []
        previous_focus = state.focus
        round_seed = self._round_seed(state, event)
        runtime_inputs = self._runtime_skill_inputs(event, state, scenario_cfg, context)
        runtime_context = self._skill_runtime_context(state.round_count, scenario, state)

        for stage_name, owners in PIPELINE_ORDER:
            pipeline_stages.append(stage_name)
            if stage_name in {"state_update", "conflict", "thalamus", "plausibility_guard", "forced_mode_switch", "output_gate", "late_perspective", "renderer", "writeback"} or not owners:
                continue
            owner = owners[0]
            agent = self.agent_map[owner]
            if not state.agents_enabled.get(owner, True):
                continue
            if owner == "BodyStateAgent":
                patch = self._execute_skill(round_id=state.round_count, skill_name="update_body_state", inputs=runtime_inputs, provider=self._agent_provider(agent, "update_body_state"), skill_traces=skill_traces, runtime_context=runtime_context, seed_ref=round_seed)
                self._apply_state_patch(state, patch.get("state_patch", {}))
                bundle = self._execute_skill(round_id=state.round_count, skill_name="compute_body_bias", inputs=runtime_inputs, provider=self._agent_provider(agent, "compute_body_bias"), skill_traces=skill_traces, runtime_context=runtime_context, fallback_value=ProposalBundle(owner=owner, confidence=0.1, delta_p={"respond": 0.01}, action_preferences={"respond": 0.01}), seed_ref=round_seed)
                veto_info = self._execute_skill(round_id=state.round_count, skill_name="apply_body_veto", inputs=runtime_inputs, provider=self._agent_provider(agent, "apply_body_veto"), skill_traces=skill_traces, runtime_context=runtime_context, seed_ref=round_seed)
                bundle.veto = veto_info.get("veto", False)
                bundle.gated_actions = list(veto_info.get("gated_actions", []))
                bundles.append(bundle)
                continue
            if owner == "EmotionAgent":
                patch = self._execute_skill(round_id=state.round_count, skill_name="update_affect_state", inputs=runtime_inputs, provider=self._agent_provider(agent, "update_affect_state"), skill_traces=skill_traces, runtime_context=runtime_context, seed_ref=round_seed)
                self._apply_state_patch(state, patch.get("state_patch", {}))
                bundle = self._execute_skill(round_id=state.round_count, skill_name="compute_affect_bias", inputs=runtime_inputs, provider=self._agent_provider(agent, "compute_affect_bias"), skill_traces=skill_traces, runtime_context=runtime_context, fallback_value=ProposalBundle(owner=owner, confidence=0.1, delta_p={"respond": 0.01}, action_preferences={"respond": 0.01}), seed_ref=round_seed)
                veto_info = self._execute_skill(round_id=state.round_count, skill_name="trigger_affect_veto", inputs=runtime_inputs, provider=self._agent_provider(agent, "trigger_affect_veto"), skill_traces=skill_traces, runtime_context=runtime_context, seed_ref=round_seed)
                bundle.veto = veto_info.get("veto", False)
                bundles.append(bundle)
                continue
            if owner == "RelationshipAgent":
                closeness = self._execute_skill(round_id=state.round_count, skill_name="score_closeness", inputs=runtime_inputs, provider=self._agent_provider(agent, "score_closeness"), skill_traces=skill_traces, runtime_context=runtime_context, seed_ref=round_seed)
                context["closeness"] = closeness.get("score", context["closeness"])
                gate_info = self._execute_skill(round_id=state.round_count, skill_name="compute_boundary_gate", inputs=runtime_inputs, provider=self._agent_provider(agent, "compute_boundary_gate"), skill_traces=skill_traces, runtime_context=runtime_context, seed_ref=round_seed)
                relation_state["boundary_level"] = gate_info.get("boundary_level", relation_state["boundary_level"])
                self._execute_skill(round_id=state.round_count, skill_name="update_relation_trace", inputs=runtime_inputs, provider=self._agent_provider(agent, "update_relation_trace"), skill_traces=skill_traces, runtime_context=runtime_context, seed_ref=round_seed)
                bundles.append(agent.propose(event, state, scenario_cfg, context))
                continue
            if owner == "ResourceAgent":
                scarcity = self._execute_skill(round_id=state.round_count, skill_name="compute_scarcity_index", inputs=runtime_inputs, provider=self._agent_provider(agent, "compute_scarcity_index"), skill_traces=skill_traces, runtime_context=runtime_context, seed_ref=round_seed)
                state.resource_state = {
                    **state.resource_state,
                    "scarcity_index": round(float(scarcity.get("scalar", 0.0)), 4),
                    "burn_rate": round(float(context.get("recent_burn_rate", 0.0)), 4),
                }
                bundle = self._execute_skill(round_id=state.round_count, skill_name="map_budget_to_bias", inputs=runtime_inputs, provider=self._agent_provider(agent, "map_budget_to_bias"), skill_traces=skill_traces, runtime_context=runtime_context, fallback_value=ProposalBundle(owner=owner, confidence=0.1, delta_p={"respond": 0.01}, action_preferences={"respond": 0.01}), seed_ref=round_seed)
                mode_hint = self._execute_skill(round_id=state.round_count, skill_name="suggest_resource_mode", inputs=runtime_inputs, provider=self._agent_provider(agent, "suggest_resource_mode"), skill_traces=skill_traces, runtime_context=runtime_context, seed_ref=round_seed)
                if mode_hint.get("mode_flag") == "safe":
                    state.safe_mode = True
                bundles.append(bundle)
                continue
            if owner == "PFCAgent":
                bundle = self._execute_skill(
                    round_id=state.round_count,
                    skill_name="generate_candidates",
                    inputs=runtime_inputs,
                    provider=self._generate_pfc_candidates_via_model,
                    skill_traces=skill_traces,
                    runtime_context=runtime_context,
                    fallback_provider=lambda **skill_inputs: agent.fallback_generate_candidates(
                        skill_inputs["event"],
                        skill_inputs["state"],
                        skill_inputs["scenario"],
                        skill_inputs["context"],
                    ),
                    seed_ref=round_seed,
                )
                self._execute_skill(round_id=state.round_count, skill_name="estimate_plan_depth", inputs=runtime_inputs, provider=self._agent_provider(agent, "estimate_plan_depth"), skill_traces=skill_traces, runtime_context=runtime_context, seed_ref=round_seed)
                self._execute_skill(round_id=state.round_count, skill_name="bind_working_memory", inputs=runtime_inputs, provider=self._agent_provider(agent, "bind_working_memory"), skill_traces=skill_traces, runtime_context=runtime_context, seed_ref=round_seed)
                bundles.append(bundle)
                continue
            if owner == "HabitAgent":
                self._execute_skill(round_id=state.round_count, skill_name="compute_feedback_decay", inputs=runtime_inputs, provider=self._agent_provider(agent, "compute_feedback_decay"), skill_traces=skill_traces, runtime_context=runtime_context, seed_ref=round_seed)
                self._execute_skill(round_id=state.round_count, skill_name="update_habit_strength", inputs=runtime_inputs, provider=self._agent_provider(agent, "update_habit_strength"), skill_traces=skill_traces, runtime_context=runtime_context, seed_ref=round_seed)
                updated_habit = self.memory_store.update_habit_strength(
                    context.get("cue"),
                    event.valence,
                    context_recurrence=min(1.0, context.get("recall_strength", 0.0) + 0.20),
                )
                if updated_habit is not None:
                    context["habit_strength"] = float(updated_habit.get("strength", context.get("habit_strength", 0.0)))
                bundle = self._execute_skill(round_id=state.round_count, skill_name="suggest_default_action", inputs=runtime_inputs, provider=self._agent_provider(agent, "suggest_default_action"), skill_traces=skill_traces, runtime_context=runtime_context, fallback_value=ProposalBundle(owner=owner, confidence=0.1, delta_p={}, action_preferences={}), seed_ref=round_seed)
                self._execute_skill(round_id=state.round_count, skill_name="estimate_override_cost", inputs=runtime_inputs, provider=self._agent_provider(agent, "estimate_override_cost"), skill_traces=skill_traces, runtime_context=runtime_context, seed_ref=round_seed)
                bundles.append(bundle)
                continue
            if owner == "DesireAgent":
                self._execute_skill(round_id=state.round_count, skill_name="score_immediate_reward", inputs=runtime_inputs, provider=self._agent_provider(agent, "score_immediate_reward"), skill_traces=skill_traces, runtime_context=runtime_context, seed_ref=round_seed)
                bundle = self._execute_skill(round_id=state.round_count, skill_name="compute_effort_avoidance", inputs=runtime_inputs, provider=self._agent_provider(agent, "compute_effort_avoidance"), skill_traces=skill_traces, runtime_context=runtime_context, fallback_value=ProposalBundle(owner=owner, confidence=0.1, delta_p={}, action_preferences={}), seed_ref=round_seed)
                self._execute_skill(round_id=state.round_count, skill_name="suggest_low_cost_action", inputs=runtime_inputs, provider=self._agent_provider(agent, "suggest_low_cost_action"), skill_traces=skill_traces, runtime_context=runtime_context, seed_ref=round_seed)
                bundles.append(bundle)
                continue
            if owner == "DMNAgent":
                bundle = self._execute_skill(round_id=state.round_count, skill_name="sample_dmn_intrusion", inputs=runtime_inputs, provider=self._agent_provider(agent, "sample_dmn_intrusion"), skill_traces=skill_traces, runtime_context=runtime_context, fallback_value=ProposalBundle(owner=owner, confidence=0.1, delta_p={}, action_preferences={}), seed_ref=round_seed)
                self._execute_skill(round_id=state.round_count, skill_name="score_rumination_pull", inputs=runtime_inputs, provider=self._agent_provider(agent, "score_rumination_pull"), skill_traces=skill_traces, runtime_context=runtime_context, seed_ref=round_seed)
                self._execute_skill(round_id=state.round_count, skill_name="select_spontaneous_topic", inputs=runtime_inputs, provider=self._agent_provider(agent, "select_spontaneous_topic"), skill_traces=skill_traces, runtime_context=runtime_context, seed_ref=round_seed)
                bundles.append(bundle)
                continue
            if owner == "HippocampusAgent":
                self._execute_skill(round_id=state.round_count, skill_name="encode_episode", inputs=runtime_inputs, provider=self._agent_provider(agent, "encode_episode"), skill_traces=skill_traces, runtime_context=runtime_context, seed_ref=round_seed)
                recall_set = self._execute_skill(round_id=state.round_count, skill_name="retrieve_by_cue", inputs=runtime_inputs, provider=self._agent_provider(agent, "retrieve_by_cue"), skill_traces=skill_traces, runtime_context=runtime_context, seed_ref=round_seed)
                self._execute_skill(round_id=state.round_count, skill_name="apply_cue_weighted_decay", inputs=runtime_inputs, provider=self._agent_provider(agent, "apply_cue_weighted_decay"), skill_traces=skill_traces, runtime_context=runtime_context, seed_ref=round_seed)
                self._execute_skill(round_id=state.round_count, skill_name="compute_memory_interference", inputs=runtime_inputs, provider=self._agent_provider(agent, "compute_memory_interference"), skill_traces=skill_traces, runtime_context=runtime_context, seed_ref=round_seed)
                if not recall_set.get("recall_set"):
                    self._execute_skill(round_id=state.round_count, skill_name="fallback_to_gist_when_trace_weak", inputs=runtime_inputs, provider=self._agent_provider(agent, "fallback_to_gist_when_trace_weak"), skill_traces=skill_traces, runtime_context=runtime_context, seed_ref=round_seed)
                bundles.append(agent.propose(event, state, scenario_cfg, context))
                continue
            if owner == "PerspectiveModel":
                bundle = self._execute_skill(round_id=state.round_count, skill_name="adjust_social_interpretation", inputs=runtime_inputs, provider=self._agent_provider(agent, "adjust_social_interpretation"), skill_traces=skill_traces, runtime_context=runtime_context, fallback_value=ProposalBundle(owner=owner, confidence=0.1, delta_p={}, action_preferences={}), seed_ref=round_seed)
                bundles.append(bundle)
                continue
            if owner == "ValueAgent":
                value_scores = self._execute_skill(round_id=state.round_count, skill_name="estimate_subjective_value", inputs=runtime_inputs, provider=self._agent_provider(agent, "estimate_subjective_value"), skill_traces=skill_traces, runtime_context=runtime_context, seed_ref=round_seed)
                self._execute_skill(round_id=state.round_count, skill_name="discount_delayed_reward", inputs=runtime_inputs, provider=self._agent_provider(agent, "discount_delayed_reward"), skill_traces=skill_traces, runtime_context=runtime_context, seed_ref=round_seed)
                self._execute_skill(round_id=state.round_count, skill_name="price_social_cost", inputs=runtime_inputs, provider=self._agent_provider(agent, "price_social_cost"), skill_traces=skill_traces, runtime_context=runtime_context, seed_ref=round_seed)
                self._execute_skill(round_id=state.round_count, skill_name="score_uncertainty_penalty", inputs=runtime_inputs, provider=self._agent_provider(agent, "score_uncertainty_penalty"), skill_traces=skill_traces, runtime_context=runtime_context, seed_ref=round_seed)
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
                bundle = self._execute_skill(round_id=state.round_count, skill_name="score_salience", inputs=runtime_inputs, provider=self._agent_provider(agent, "score_salience"), skill_traces=skill_traces, runtime_context=runtime_context, fallback_value=ProposalBundle(owner=owner, confidence=0.1, delta_p={"respond": 0.02}, action_preferences={"respond": 0.02}), seed_ref=round_seed)
                self._execute_skill(round_id=state.round_count, skill_name="switch_mode", inputs=runtime_inputs, provider=self._agent_provider(agent, "switch_mode"), skill_traces=skill_traces, runtime_context=runtime_context, seed_ref=round_seed)
                self._execute_skill(round_id=state.round_count, skill_name="interrupt_current_focus", inputs=runtime_inputs, provider=self._agent_provider(agent, "interrupt_current_focus"), skill_traces=skill_traces, runtime_context=runtime_context, seed_ref=round_seed)
                self._execute_skill(round_id=state.round_count, skill_name="promote_event_to_workspace", inputs=runtime_inputs, provider=self._agent_provider(agent, "promote_event_to_workspace"), skill_traces=skill_traces, runtime_context=runtime_context, seed_ref=round_seed)
                bundles.append(bundle)
                continue
            if owner == "UnconsciousAgent":
                baseline = self._execute_skill(round_id=state.round_count, skill_name="load_temperament", inputs=runtime_inputs, provider=self._agent_provider(agent, "load_temperament"), skill_traces=skill_traces, runtime_context=runtime_context, seed_ref=round_seed)
                if not state.temperament_state:
                    state.temperament_state = {**self.config["temperament"]["temperament"], **baseline.get("baseline", {})}
                bundle = self._execute_skill(round_id=state.round_count, skill_name="compute_trait_bias", inputs=runtime_inputs, provider=self._agent_provider(agent, "compute_trait_bias"), skill_traces=skill_traces, runtime_context=runtime_context, fallback_value=ProposalBundle(owner=owner, confidence=0.1, delta_p={"respond": 0.01}, action_preferences={"respond": 0.01}), seed_ref=round_seed)
                patch = self._execute_skill(round_id=state.round_count, skill_name="apply_chronic_shift", inputs=runtime_inputs, provider=self._agent_provider(agent, "apply_chronic_shift"), skill_traces=skill_traces, runtime_context=runtime_context, seed_ref=round_seed)
                self._apply_state_patch(state, patch.get("state_patch", {}))
                bundles.append(bundle)
                continue
            if owner == "CerebellarPredictor":
                self._execute_skill(round_id=state.round_count, skill_name="predict_next_state", inputs=runtime_inputs, provider=self._agent_provider(agent, "predict_next_state"), skill_traces=skill_traces, runtime_context=runtime_context, seed_ref=round_seed)
                self._execute_skill(round_id=state.round_count, skill_name="compute_prediction_error", inputs=runtime_inputs, provider=self._agent_provider(agent, "compute_prediction_error"), skill_traces=skill_traces, runtime_context=runtime_context, seed_ref=round_seed)
                self._execute_skill(round_id=state.round_count, skill_name="smooth_response_timing", inputs=runtime_inputs, provider=self._agent_provider(agent, "smooth_response_timing"), skill_traces=skill_traces, runtime_context=runtime_context, seed_ref=round_seed)
                bundle = self._execute_skill(round_id=state.round_count, skill_name="micro_adjust_action", inputs=runtime_inputs, provider=self._agent_provider(agent, "micro_adjust_action"), skill_traces=skill_traces, runtime_context=runtime_context, fallback_value=ProposalBundle(owner=owner, confidence=0.1, delta_p={}, action_preferences={}), seed_ref=round_seed)
                bundles.append(bundle)

        bundles = [self._enrich_bundle_metadata(bundle, state, relation_state, context) for bundle in bundles]
        distribution_state = self._build_distribution_state(bundles, state, scenario_cfg, mode_cfg, relation_state, context)
        conflict_score = self._run_conflict_controller(
            state=state,
            bundles=bundles,
            distribution_state=distribution_state,
            thresholds=thresholds,
            skill_traces=skill_traces,
            runtime_context=runtime_context,
            gate_decisions=gate_decisions,
            round_seed=round_seed,
        )

        thalamus = self.agent_map["ThalamusAttentionAgent"]
        aggregated = self._execute_skill(
            round_id=state.round_count,
            skill_name="aggregate_proposals",
            inputs={"distribution_state": distribution_state},
            provider=lambda distribution_state: thalamus.run_skill("aggregate_proposals", distribution_state),
            skill_traces=skill_traces,
            runtime_context=runtime_context,
            seed_ref=round_seed,
        )["distribution"]
        deterministic = self._execute_skill(
            round_id=state.round_count,
            skill_name="normalize_distribution",
            inputs={"distribution": aggregated},
            provider=lambda distribution: thalamus.run_skill("normalize_distribution", distribution),
            skill_traces=skill_traces,
            runtime_context=runtime_context,
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
        post_risk = {
            action: _clip(stochastic_distribution[action], thresholds["p_floor"], thresholds["p_cap"]) * distribution_state.risk_suppressor.get(action, 1.0)
            for action in stochastic_distribution
        }
        distribution_state.p_final = self._normalize(
            {
                action: post_risk[action] * distribution_state.gate.get(action, 1.0)
                for action in post_risk
            }
        )

        sampled_action = self._execute_skill(
            round_id=state.round_count,
            skill_name="sample_action",
            inputs={"distribution": distribution_state.p_final},
            provider=lambda distribution: thalamus.run_skill("sample_action", distribution),
            skill_traces=skill_traces,
            runtime_context=runtime_context,
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
                inputs={"action": action_name, "scenario": scenario, "relation_state": relation_state},
                provider=lambda action, scenario, relation_state: plausibility_guard.run_skill("check_behavior_plausibility", action, scenario, relation_state),
                skill_traces=skill_traces,
                runtime_context=runtime_context,
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
        preemptive_guard = dominant_blocked_action is not None and dominant_blocked_probability >= max(0.10, top_probability * 0.50)
        second_sampling = self._execute_skill(
            round_id=state.round_count,
            skill_name="request_second_sampling",
            inputs={"fail_score": fail_score, "attempts": distribution_state.resample_idx},
            provider=lambda fail_score, attempts: plausibility_guard.run_skill("request_second_sampling", fail_score, attempts),
            skill_traces=skill_traces,
            runtime_context=runtime_context,
            seed_ref=round_seed,
        )["flag"]
        self._execute_skill(
            round_id=state.round_count,
            skill_name="escalate_value_reestimate",
            inputs={"fail_score": fail_score},
            provider=lambda fail_score: plausibility_guard.run_skill("escalate_value_reestimate", fail_score),
            skill_traces=skill_traces,
            runtime_context=runtime_context,
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
            provider=lambda history, focus_lock_count: forced.run_skill("detect_mode_lock", history, focus_lock_count),
            skill_traces=skill_traces,
            runtime_context=runtime_context,
            seed_ref=round_seed,
        )["lock_score"]
        force_switch = self._execute_skill(
            round_id=state.round_count,
            skill_name="trigger_forced_focus_switch",
            inputs={"lock_score": lock_score},
            provider=lambda lock_score: forced.run_skill("trigger_forced_focus_switch", lock_score),
            skill_traces=skill_traces,
            runtime_context=runtime_context,
            seed_ref=round_seed,
        )["switch_flag"]
        if force_switch:
            sampled_action = ActionCandidate(name="respond", probability=sampled_action.probability, rationale="forced mode switch")
            gate_decisions.append({"stage": "forced_mode_switch", "owner": "ForcedModeSwitch", "allowed": False, "requires_resample": True, "reason": f"lock_score={lock_score:.2f}"})

        output_gate = self.agent_map["OutputGate"]
        gate = self._execute_skill(
            round_id=state.round_count,
            skill_name="apply_output_gate",
            inputs={"action": sampled_action.name, "state": state, "scenario": scenario, "relation_state": relation_state},
            provider=lambda action, state, scenario, relation_state: output_gate.run_skill("apply_output_gate", action, state, scenario, relation_state),
            skill_traces=skill_traces,
            runtime_context=runtime_context,
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
        expression = self._apply_conflict_expression_adjustments(expression, distribution_state.conflict)
        tone_params = self._execute_skill(
            round_id=state.round_count,
            skill_name="render_tone_profile",
            inputs={"expression_profile": expression},
            provider=lambda expression_profile: output_gate.run_skill("render_tone_profile", expression_profile),
            skill_traces=skill_traces,
            runtime_context=runtime_context,
            seed_ref=round_seed,
        )
        delay_params = self._execute_skill(
            round_id=state.round_count,
            skill_name="compute_delay_profile",
            inputs={"expression_profile": expression},
            provider=lambda expression_profile: output_gate.run_skill("compute_delay_profile", expression_profile),
            skill_traces=skill_traces,
            runtime_context=runtime_context,
            seed_ref=round_seed,
        )

        late_perspective = {"state_hypothesis": {}, "reaction_hypothesis": {}}
        if self._should_run_late_perspective(event, relation_state, sampled_action.name):
            perspective = self.agent_map["PerspectiveModel"]
            late_perspective["state_hypothesis"] = self._execute_skill(
                round_id=state.round_count,
                skill_name="infer_other_state",
                inputs={**runtime_inputs, "sampled_action": sampled_action.name, "relation_state": relation_state},
                provider=lambda event, state, scenario, context, sampled_action, relation_state: self._infer_other_state_via_model(event, state, scenario, context, sampled_action, relation_state),
                skill_traces=skill_traces,
                runtime_context=runtime_context,
                fallback_provider=lambda event, state, scenario, context, sampled_action, relation_state: perspective.fallback_infer_other_state(event, state, scenario, context),
                seed_ref=round_seed,
            ).get("state_hypothesis", {})
            late_perspective["reaction_hypothesis"] = self._execute_skill(
                round_id=state.round_count,
                skill_name="simulate_other_reaction",
                inputs={**runtime_inputs, "sampled_action": sampled_action.name, "relation_state": relation_state},
                provider=lambda event, state, scenario, context, sampled_action, relation_state: self._simulate_other_reaction_via_model(event, state, scenario, context, sampled_action, relation_state),
                skill_traces=skill_traces,
                runtime_context=runtime_context,
                fallback_provider=lambda event, state, scenario, context, sampled_action, relation_state: perspective.fallback_simulate_other_reaction(event, state, scenario, context),
                seed_ref=round_seed,
            ).get("reaction_hypothesis", {})
        else:
            gate_decisions.append(
                {
                    "stage": "late_perspective",
                    "owner": "PerspectiveModel",
                    "allowed": False,
                    "requires_resample": False,
                    "reason": "risk_gate_closed",
                }
            )

        render_plan = build_render_plan(
            sampled_action=sampled_action.name,
            expression=ExpressionProfile(**{**to_dict(expression)}),
            safety_constraints={
                "gate": distribution_state.gate.get(sampled_action.name, 1.0),
                "tone_params": tone_params["tone_params"],
                "delay_params": delay_params["delay_params"],
                "conflict_hot": distribution_state.conflict.get("circuit_breaker", {}).get("active", False),
                "compromise_template": distribution_state.conflict.get("compromise", {}).get("template"),
                "winning_priority": distribution_state.conflict.get("winning_priority"),
            },
            scenario=scenario,
            event_summary=event.content,
            target=event.target,
            relation_state=relation_state,
            perspective=late_perspective,
        )
        rendered_output, rendered_result = self._execute_skill_with_result(
            round_id=state.round_count,
            skill_name="render_expression",
            inputs={"render_plan": render_plan},
            provider=self._render_expression_via_model,
            skill_traces=skill_traces,
            runtime_context=runtime_context,
            fallback_provider=lambda render_plan: {
                "text": fallback_render_text(render_plan),
                "route": "renderer",
                "model": "fallback",
            },
            seed_ref=round_seed,
        )
        rendered_expression = RenderedExpression(
            text=rendered_output.get("text", ""),
            route=rendered_output.get("route", "renderer"),
            model=rendered_output.get("model", "fallback"),
            degraded=rendered_result.degraded,
            failure_policy_applied=rendered_result.failure_policy_applied,
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
        sampled_action.metadata["rendered_expression"] = to_dict(rendered_expression)

        contributions, proposal_records = self._build_contributions(
            bundles,
            state,
            sampled_action.name,
            conflict_score,
            fail_score,
            distribution_state.resample_idx,
        )
        for row in skill_traces:
            row["session_id"] = state.session_id
            row["recorded_at"] = recorded_at
            row["recorded_date"] = recorded_date

        trace = RoundTrace(
            session_id=state.session_id,
            recorded_at=recorded_at,
            recorded_date=recorded_date,
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
            rendered_expression=to_dict(rendered_expression),
            resample_count=distribution_state.resample_idx,
        )

        health = HealthEvent(event="tick", status="ok", detail=f"budget={state.budget_remaining:.2f}")
        self._save_state(state)
        self.trace_store.write_round(trace)

        return RoundResult(
            round_id=state.round_count,
            sampled_action=sampled_action,
            trace=trace,
            state=state,
            health=health,
            rendered_expression=rendered_expression,
        )

    def apply_command(self, command: str, envelope=None) -> CommandResult:
        state = self.load_runtime_state()
        before_hash = self._state_hash(state)
        parts = command.split()
        operator_level = envelope.operator_level if envelope is not None else "direct_runtime"
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
        elif parts[:2] == ["nudge", "relation"] and len(parts) >= 5 and parts[3] == "trust":
            delta = float(parts[4])
            relation = self.memory_store.nudge_relation(parts[2], delta)
            result = CommandResult(applied=True, scope="relation", delta={"target": parts[2], "closeness": relation["closeness"], "delta": delta}, ttl="until changed", rollback_hint=f"alive nudge relation {parts[2]} trust {-delta:+.2f}", operator_level=operator_level, rollback_available=True)
        elif parts[:2] == ["habit", "reset"] and len(parts) == 3:
            habit = self.memory_store.reset_habit(parts[2])
            result = CommandResult(applied=True, scope="habit", delta={"pattern": parts[2], "strength": habit["strength"], "recoverable": habit.get("recoverable", True)}, ttl="until rebuilt by recurrence", rollback_hint="habit will rebuild from future repetition", operator_level=operator_level, rollback_available=False)
        elif parts[:2] == ["budget", "set"] and len(parts) >= 3:
            raw_value = parts[3] if len(parts) >= 4 and parts[2] == "--cap" else parts[2]
            cap_value, budget_remaining = self._parse_budget_cap(raw_value)
            state.budget_remaining = budget_remaining
            state.resource_state = {**state.resource_state, "budget_cap": cap_value}
            result = CommandResult(applied=True, scope="resource", delta={"budget_remaining": budget_remaining, "budget_cap": cap_value}, ttl="until changed", rollback_hint="alive budget set --cap 100000", operator_level=operator_level, rollback_available=True)
        elif parts[:2] == ["suppress", "dmn"]:
            state.agents_enabled["DMNAgent"] = False
            ttl = parts[2] if len(parts) >= 3 else "temporary"
            result = CommandResult(applied=True, scope="DMNAgent", delta={"enabled": False}, ttl=ttl, rollback_hint="alive agent enable DMNAgent", operator_level=operator_level, rollback_available=True)

        self._save_state(state)
        after_hash = self._state_hash(state)
        self.trace_store.append_command(
            command,
            result,
            before_hash,
            after_hash,
            session_id=state.session_id,
            recorded_at=utc_now_iso(),
        )
        return result

    def checkpoint(self) -> CheckpointRef:
        state = self.load_runtime_state()
        before_hash = self._state_hash(state)
        checkpoint_id = f"ckpt-{state.round_count:04d}"
        path = self.checkpoint_dir / f"{checkpoint_id}.json"
        shutil.copyfile(self.state_path, path)
        state.last_checkpoint_id = checkpoint_id
        self._save_state(state)
        after_hash = self._state_hash(state)
        self.trace_store.append_command(
            "checkpoint create",
            CommandResult(
                applied=True,
                scope="checkpoint",
                delta={"checkpoint_id": checkpoint_id, "created": True},
                rollback_hint=f"alive checkpoint rewind {checkpoint_id}",
                operator_level="ops_admin",
                rollback_available=True,
            ),
            before_hash,
            after_hash,
            session_id=state.session_id,
            recorded_at=utc_now_iso(),
        )
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
        self.trace_store.append_command(
            f"checkpoint rewind {checkpoint_id}",
            result,
            before_hash,
            after_hash,
            session_id=restored_state.session_id,
            recorded_at=utc_now_iso(),
        )
        return result

    def memory_top(self, limit: int = 5) -> list[dict]:
        return self.memory_store.memory_top(limit=limit)

    def memory_recall(self, cue: str) -> dict[str, Any]:
        return self.memory_store.recall(cue)

    def compact_memory(self, *, hot_max_rounds: int = 500, warm_max_rounds: int = 3000) -> dict[str, Any]:
        return self.memory_store.compact_tiers(hot_max_rounds=hot_max_rounds, warm_max_rounds=warm_max_rounds)

    def sample_memory(self, tier: str, *, limit: int = 5, cue: str | None = None) -> list[dict[str, Any]]:
        return self.memory_store.sample_compacted(tier, limit=limit, cue=cue)

    def export_trace_parquet(self, *, since_round: int | None = None, overwrite: bool = False) -> dict[str, Any]:
        exporter = TraceExporter(self.trace_store)
        return exporter.export_parquet(since_round=since_round, overwrite=overwrite)

    def habit_top(self, limit: int = 5) -> list[dict]:
        return self.memory_store.habit_top(limit=limit)

    def state_payload(self) -> dict[str, Any]:
        return to_dict(self.load_runtime_state())

    def resolve_round_ref(self, round_ref: int | str | None) -> int:
        if round_ref is None or round_ref == "last":
            round_id = self.load_runtime_state().round_count
            if round_id <= 0:
                raise FileNotFoundError("no trace rounds recorded yet")
            return round_id
        if isinstance(round_ref, int):
            return round_ref
        try:
            return int(round_ref)
        except ValueError as exc:
            raise ValueError(f"invalid round reference: {round_ref}") from exc

    def trace_round(self, round_ref: int | str) -> dict[str, Any]:
        return self.trace_store.read_round(self.resolve_round_ref(round_ref))

    def why_this(self, round_ref: int | str) -> dict[str, Any]:
        trace = self.trace_round(round_ref)
        return {
            "round_id": trace["round_id"],
            "sampled_action": trace["sampled_action"],
            "top_drivers": trace["top_drivers"],
            "style_profile": trace["style_profile"],
            "distribution_state": trace.get("distribution_state", {}),
            "stochastic_state": trace.get("stochastic_state", {}),
            "render_plan": trace.get("render_plan", {}),
            "rendered_expression": trace.get("rendered_expression", {}),
            "state_snapshot": {
                "mode": trace["state_snapshot"]["mode"],
                "safe_mode": trace["state_snapshot"]["safe_mode"],
                "focus": trace["state_snapshot"]["focus"],
                "budget_remaining": trace["state_snapshot"]["budget_remaining"],
            },
        }

    def contribution_breakdown(self, round_ref: int | str) -> dict[str, Any]:
        trace = self.trace_round(round_ref)
        return {"round_id": trace["round_id"], "sampled_action": trace["sampled_action"], "contributions": trace["contributions"]}

    def trace_agents(self, round_ref: int | str) -> dict[str, Any]:
        trace = self.trace_round(round_ref)
        agents = [
            {
                "stage": row.get("stage"),
                "agent_name": row.get("agent_name"),
                "top_action": row.get("top_action"),
                "confidence": row.get("confidence"),
                "veto": row.get("veto", False),
                "selected": row.get("selected", False),
                "delta_p": row.get("delta_p", {}),
                "weight_applied": row.get("weight_applied"),
                "reason_tags": row.get("tags", []),
            }
            for row in trace.get("proposal_summaries", [])
        ]
        return {
            "round_id": trace["round_id"],
            "sampled_action": trace["sampled_action"],
            "agents": agents,
        }

    def trace_skills(self, round_ref: int | str) -> dict[str, Any]:
        trace = self.trace_round(round_ref)
        skills = [
            {
                "skill_name": row.get("skill_name"),
                "owner_module": row.get("owner_module"),
                "latency_ms": row.get("latency_ms"),
                "degraded": row.get("degraded", False),
                "failure_policy_applied": row.get("failure_policy_applied"),
                "fallback_route": row.get("fallback_route"),
                "policy_rejection_reason": row.get("policy_rejection_reason"),
            }
            for row in trace.get("skill_traces", [])
        ]
        return {
            "round_id": trace["round_id"],
            "sampled_action": trace["sampled_action"],
            "skills": skills,
        }

    def trace_gates(self, round_ref: int | str) -> dict[str, Any]:
        trace = self.trace_round(round_ref)
        gates = [
            {
                "stage": row.get("stage"),
                "owner": row.get("owner"),
                "allowed": row.get("allowed", True),
                "requires_resample": row.get("requires_resample", False),
                "reason": row.get("reason", ""),
            }
            for row in trace.get("gate_decisions", [])
        ]
        return {
            "round_id": trace["round_id"],
            "sampled_action": trace["sampled_action"],
            "gates": gates,
        }

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
        payload = []
        for spec in self.skills.values():
            item = asdict(spec)
            item["input_schema"] = serialize_contract(spec.input_schema)
            item["output_schema"] = serialize_contract(spec.output_schema)
            payload.append(item)
        return payload

    def skill_stats(self) -> dict[str, Any]:
        return self.trace_store.skill_stats()

    def skill_profile(self, skill_name: str) -> dict[str, Any]:
        stats = self.trace_store.skill_stats(skill_name=skill_name)
        return {"skill_name": skill_name, **stats}

    def relation_show(self, target: str) -> dict[str, Any]:
        closeness = self.memory_store.closeness(target)
        boundary_level = _clip(0.8 - closeness, 0.0, 1.0)
        return {"target": target, "closeness": closeness, "trust": closeness, "boundary_level": boundary_level}

    def replay_round(self, round_id: int, seed: int | None = None) -> dict[str, Any]:
        return {"round": self.trace_round(round_id), "seed": seed}

    def conflict_timeline(self) -> dict[str, Any]:
        points = []
        for trace in self.trace_store.list_rounds():
            conflict = trace.get("distribution_state", {}).get("conflict", {})
            points.append(
                {
                    "round_id": trace["round_id"],
                    "conflict_score": conflict.get("total_score", conflict.get("score", 0.0)),
                    "components": conflict.get("components", {}),
                    "winning_priority": conflict.get("winning_priority"),
                    "template": conflict.get("compromise", {}).get("template"),
                    "critical_conflict": conflict.get("critical_conflict", False),
                    "critical_conflict_streak": conflict.get("critical_conflict_streak", 0),
                    "conflict_hot_rounds": conflict.get("circuit_breaker", {}).get("hot_rounds_remaining", 0),
                }
            )
        return {"points": points}

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
