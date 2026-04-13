from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

import yaml

from nalr.agents.modules import (
    OWNER_CONTROL_DOMAIN,
    OWNER_PRIORITY_BUCKET,
    TEMPLATE_ACTION_SCALES,
    TEMPLATE_BY_PRIORITY,
    build_agents,
    compute_salience_signal,
)
from nalr.dream.orchestrator import DreamOrchestrator
from nalr.memory.store import MemoryStore, _derive_cue
from nalr.output.renderer import fallback_render_text
from nalr.output.style import build_expression_profile, build_render_plan, compute_style_profile
from nalr.providers import MissingModelCredentialError, ModelRequest, ModelRouteConfig, ModelRouter
from nalr.run import SupervisorLoop
from nalr.runtime.model_gateway import ModelGateway
from nalr.runtime.async_io import AsyncIOWorker
from nalr.runtime.dynamics import smooth_resource_biases
from nalr.runtime.entropy import MacOSSystemEntropyProvider, QuantumEntropyPool, QuantumEntropyUnavailableError
from nalr.runtime.authenticity import AuthenticityPolicy
from nalr.runtime.contracts import ArbitrationResult, CollectedContributions, EndogenousTickPayload, RoundContext
from nalr.runtime.endogenous_scheduler import EndogenousTickScheduler
from nalr.runtime.identity import IdentityRuntime
from nalr.runtime.initiative import InitiativeRuntime
from nalr.runtime.longrun import LongRunAnalyzer
from nalr.runtime.math_kernel import kl_divergence, normalize_distribution, sample_action_name, softmax_distribution
from nalr.runtime.metadata import iso_date, utc_now_iso
from nalr.runtime.monologue import MonologueStreamRuntime
from nalr.runtime.motivation_feedback import MotivationFeedbackUpdater
from nalr.runtime.motivation_pool import EndogenousMotivationPool
from nalr.runtime.probability_field import ProbabilityFieldIntegrator, compute_tlh_vector_collapse
from nalr.runtime.diagnostics_runtime import DiagnosticsRuntimeService
from nalr.runtime.state_runtime import StateRuntimeService
from nalr.runtime.tlh_vectors import (
    AXES,
    TLH_ACTION_VECTORS,
    build_tlh_modulation_directions,
    build_tlh_state_vectors,
    cosine_similarity,
    infer_sketch_vector,
    project_vector_to_action_support,
    region_scores_from_match_scores,
)
from nalr.runtime.vitality import VitalityEngine
from nalr.schemas.models import (
    ActionCandidate,
    ActionBookkeepingState,
    ActionEvidenceSignal,
    AutonomyLoopState,
    AutonomyPolicyState,
    AgentContribution,
    AuthenticityRecord,
    BodyState,
    CheckpointRef,
    CommandEnvelope,
    ExecutionBudget,
    CommandResult,
    ConflictPostErrorAdjustment,
    ConflictRepairLedgerEntry,
    ConflictRepairState,
    DisclosureIntentState,
    EndogenousMicroIntent,
    EndogenousMotivationSignal,
    EndogenousReplayChain,
    EndogenousRuntimeState,
    EndogenousSchedulerState,
    EndogenousSuppressionDecision,
    EndogenousTriggerContext,
    EndogenousTickTrigger,
    EmergentActionSketch,
    CrossLayerCouplingSpec,
    EnergyProjectionSpec,
    EmotionState,
    ExpressionProfile,
    HealthEvent,
    IdentityContext,
    IdentityState,
    InstinctFieldState,
    MotivationLearningState,
    MotivationPoolState,
    OrganicModeState,
    ProbabilityFieldSnapshot,
    ProbabilityLayerState,
    PersonalityAnchorState,
    ProbabilisticContribution,
    RenderPlan,
    RenderedExpression,
    RunPolicy,
    RunRequest,
    RunState,
    RoundEvent,
    RoundResult,
    RoundTrace,
    RuntimeState,
    QueryIntentState,
    SkillResult,
    SkillRuntimeContext,
    StopReason,
    StochasticState,
    SubjectCore,
    SubjectiveState,
    TaskNode,
    TokenFieldState,
    TurnExecution,
    TurnPlan,
    ToolResult,
    DesireState,
    normalize_temperament_state,
    to_dict,
)
from nalr.skills.executor import SkillExecutor
from nalr.skills.registry import build_skill_registry, serialize_contract
from nalr.storage.parquet_io import read_snapshot_rows, rewrite_snapshot
from nalr.trace.exporter import TraceExporter
from nalr.trace.store import TraceStore, canonical_probability_field_payload


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


PIPELINE_TELEMETRY_STAGES: list[str] = [
    "state_update",
    "salience",
    "body",
    "emotion",
    "relationship",
    "resource",
    "pfc",
    "habit",
    "desire",
    "dmn",
    "hippocampus",
    "perspective",
    "value",
    "unconscious",
    "cerebellar",
    "conflict",
    "thalamus",
    "plausibility_guard",
    "forced_mode_switch",
    "output_gate",
    "late_perspective",
    "renderer",
    "writeback",
]

ACTION_HEAD_STAGE_ORDER: list[tuple[str, str]] = [
    ("salience", "SalienceAgent"),
    ("body", "BodyStateAgent"),
    ("emotion", "EmotionAgent"),
    ("relationship", "RelationshipAgent"),
    ("resource", "ResourceAgent"),
    ("pfc", "PFCAgent"),
    ("habit", "HabitAgent"),
    ("desire", "DesireAgent"),
    ("dmn", "DMNAgent"),
    ("hippocampus", "HippocampusAgent"),
    ("perspective", "PerspectiveModel"),
    ("value", "ValueAgent"),
    ("unconscious", "UnconsciousAgent"),
    ("cerebellar", "CerebellarPredictor"),
]

ACTION_STAGE_BY_OWNER: dict[str, str] = {
    owner: stage
    for stage, owner in ACTION_HEAD_STAGE_ORDER
}
ACTION_STAGE_BY_OWNER["EndogenousMotivationPool"] = "motivation"
ACTION_STAGE_BY_OWNER["InstinctField"] = "instinct"
ACTION_STAGE_BY_OWNER["EmergentActionSketch"] = "emergent"


INNATE_ACTIONS = ("respond", "rest", "absorb", "wander", "monologue", "nothing", "die")
DERIVED_ACTIONS = ("plan", "recall", "connect", "clarify")
CORE_ACTIONS = ("respond", "plan", "recall", "rest", "connect", "clarify", "wander", "absorb", "monologue", "nothing", "die")
INTERNAL_RUNTIME_ACTIONS = CORE_ACTIONS + ("short_reply",)
MODEL_ROUTE_SKILL_BINDINGS: dict[str, tuple[str, ...]] = {
    "planner": (),
    "pfc": ("generate_candidates",),
    "perspective": ("infer_other_state", "simulate_other_reaction"),
    "renderer": ("render_expression",),
    "renderer_fallback_fast": ("render_expression_fallback",),
    "renderer_fallback_small": ("render_expression_fallback",),
    "monologue_stream": ("monologue_stream",),
}
MODEL_ROUTE_BINDING_KEYS: dict[str, str] = {
    "planner": "planner",
    "pfc": "PFCAgent",
    "perspective": "PerspectiveModel",
    "renderer": "Renderer",
    "renderer_fallback_fast": "Renderer",
    "renderer_fallback_small": "Renderer",
    "monologue_stream": "MonologueStream",
}

ARCHITECTURE_CORE_GOAL = "维持生命性、真实性与连续性"
RUNTIME_SCHEMA_VERSION = 2

if TYPE_CHECKING:
    from nalr.runtime.controller import RuntimeController


class _ControllerBackedRuntime:
    def __init__(self, controller: "RuntimeController") -> None:
        self.controller = controller

    def __getattr__(self, name: str):
        return getattr(self.controller, name)


class RoundPipelineRuntime(_ControllerBackedRuntime):
    def _tick_view(self, event: RoundEvent, scenario: str, mode: str) -> RoundResult:
        patched = getattr(self.controller, "__dict__", {}).get("tick")
        if callable(patched):
            return patched(event, scenario=scenario, mode=mode)
        return self.tick(event, scenario, mode)

    def _renderer_system_prompt(
        self,
        render_plan: RenderPlan,
        *,
        prompt_mode: str = "base",
        violation_types: list[str] | None = None,
    ) -> str:
        return self._renderer_system_prompt_impl(
            render_plan,
            prompt_mode=prompt_mode,
            violation_types=violation_types,
        )

    def _renderer_system_prompt_impl(
        self,
        render_plan: RenderPlan,
        *,
        prompt_mode: str = "base",
        violation_types: list[str] | None = None,
    ) -> str:
        identity = render_plan.identity_context
        slow_variables = render_plan.message_plan.get("slow_variables", {})
        repair_expression = render_plan.message_plan.get("repair_expression", {})
        lines = [
            "你是最终表达 renderer。仅回 JSON:{text}。",
            "必须从当前运行体视角说话；provider 不是本体。",
            f"本体={identity.display_label or self._unnamed_label()}/{identity.class_label}。",
            f"q={identity.query_kind};d={identity.disclosure_detail};qi={identity.query_intent};di={identity.disclosure_intent}.",
        ]
        lines.append(f"delivery_mode={render_plan.delivery_mode}; action={render_plan.action}.")
        if render_plan.delivery_mode == "monologue":
            lines.append("当 delivery_mode=monologue 时，这不是直接对用户说话，而是可见的内部独白。文本必须以【独白】开头。")
        lines.append(f"repair_expression={json.dumps(repair_expression, ensure_ascii=False, sort_keys=True)}。")
        lines.append(
            "按 repair 组织表达，"
            f"stage={repair_expression.get('stage')}; "
            f"opening_mode={repair_expression.get('opening_mode')}; "
            f"advance_mode={repair_expression.get('advance_mode')}; "
            f"visibility={repair_expression.get('visibility')}; "
            f"safety_invite={repair_expression.get('safety_invite')}。"
        )
        if repair_expression.get("visibility") == "implicit":
            lines.append("当 visibility=implicit 时，不得把 repair stage 直接翻译成元叙述，不要直说“我在修复/我在恢复”。")
        if repair_expression.get("stage") == "recovered":
            lines.append("当 stage=recovered 时，清除 repair tone，不要残留“收一下”“稳一点”“你可以打断我”这类修复腔调。")
        if identity.query_kind == "self_identity":
            lines.append("当用户在问你是谁时，只能说明自己是当前运行体实例，不得说“我是豆包/ChatGPT/provider”。")
        elif identity.query_kind == "provider_identity":
            lines.append("当用户追问底层能力时，先说明本体，再把 provider 作为底层能力来源说明。")
        elif identity.query_kind == "answer_explanation":
            lines.append("当用户追问为什么这样回答时，只能引用当前状态、记忆、关系、focus 和本轮表达计划。")
            lines.append(f"可引用的慢变量来源: {json.dumps(slow_variables, ensure_ascii=False, sort_keys=True)}。")
        elif any(
            token in render_plan.event_summary.lower()
            for token in ("你可以做什么", "你能做什么", "你会什么", "能帮我做什么", "what can you do")
        ):
            lines.append("当用户这样试探你会怎么说话时，要按自然对话去回应，像一个受记忆、关系、状态影响的活人。")
            lines.append("不能输出产品介绍、功能清单、命令帮助，也不要用帮助器或客服式口吻。")
            lines.append("可以自然提到你会回应、会记住、会受状态和关系影响，但必须说得像真人正在开口。")
        if prompt_mode == "violation" and violation_types:
            lines.append(f"上一版命中了这些违例：{', '.join(violation_types)}。重写时彻底避免这些违例。")
        return " ".join(lines)

    def _evaluate_authenticity(self, text: str, render_plan: RenderPlan) -> dict[str, Any]:
        patched = getattr(self.controller, "__dict__", {}).get("_evaluate_authenticity")
        if callable(patched):
            return patched(text, render_plan)
        return self._evaluate_authenticity_impl(text, render_plan)

    def _evaluate_authenticity_impl(self, text: str, render_plan: RenderPlan) -> dict[str, Any]:
        return self.authenticity_policy.evaluate_text(text, render_plan)

    def _coerce_score_map(self, value: Any) -> dict[str, float]:
        return self._coerce_score_map_impl(value)

    def _coerce_score_map_impl(self, value: Any) -> dict[str, float]:
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
        *,
        model_call_traces: list[dict[str, Any]] | None = None,
    ) -> ProbabilisticContribution:
        patched = getattr(self.controller, "__dict__", {}).get("_generate_pfc_candidates_via_model")
        if callable(patched):
            parameters = inspect.signature(patched).parameters
            if "model_call_traces" in parameters:
                return patched(
                    event,
                    state,
                    scenario,
                    context,
                    model_call_traces=model_call_traces,
                )
            return patched(event, state, scenario, context)
        return self._generate_pfc_candidates_via_model_impl(
            event,
            state,
            scenario,
            context,
            model_call_traces=model_call_traces,
        )

    def _generate_pfc_candidates_via_model_impl(
        self,
        event: RoundEvent,
        state: RuntimeState,
        scenario: dict[str, Any],
        context: dict[str, Any],
        *,
        model_call_traces: list[dict[str, Any]] | None = None,
    ) -> ProbabilisticContribution:
        request = ModelRequest(
            system_prompt="你是 PFCAgent。仅回 JSON:{action_preferences,confidence,sigma_scale,reason}。",
            user_prompt=self._json_prompt(self._build_pfc_model_payload(event, state, scenario, context)),
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
        )
        response = self._call_bound_model_route(
            "PFCAgent",
            route_name="pfc",
            request=request,
            binding_metadata={
                "relation_risk": float(context.get("relation_risk", 0.0) or 0.0),
                "disclosure_sensitivity": float(context.get("disclosure_sensitivity", 0.0) or 0.0),
                "authenticity_risk": float(context.get("authenticity_risk", 0.0) or 0.0),
                "conflict_score": float(context.get("conflict_score", 0.0) or 0.0),
                "resample_count": int(context.get("resample_count", 0) or 0),
            },
            model_call_traces=model_call_traces,
            skill_name="generate_candidates",
        )
        prefs = {
            action: self._clip_delta(float(score))
            for action, score in self._coerce_score_map(response.payload.get("action_preferences", {})).items()
        }
        confidence = _clip(float(response.payload.get("confidence", 0.84)), 0.0, 1.0)
        sigma_scale = _clip(float(response.payload.get("sigma_scale", 0.92)), 0.60, 1.60)
        top_action = max(prefs, key=prefs.get) if prefs else ""
        dependency_trace = [
            "priority:task_goal",
            "control:task",
            f"sigma_scale:{round(sigma_scale, 4)}",
        ]
        if top_action:
            dependency_trace.append(f"top_action:{top_action}")
        return ProbabilisticContribution(
            module_name="PFCAgent",
            module_type="executive",
            level="action",
            target_space="action",
            raw_signal=dict(prefs),
            modulated_delta=dict(prefs),
            confidence=confidence,
            confidence_calibrated=round(_clip(confidence * sigma_scale, 0.0, 1.0), 4),
            trace_reason=str(response.payload.get("reason", f"model route={response.route}")),
            projection_reason="executive prior projected from PFC model head",
            applied_at_stage="executive_prior",
            native_operator="executive_prior",
            dependency_trace=dependency_trace,
            projection=EnergyProjectionSpec(
                module_type="executive",
                target_space="action",
                module_temperature=sigma_scale,
            ),
        )

    def _invoke_pfc_model_generator(
        self,
        event: RoundEvent,
        state: RuntimeState,
        scenario: dict[str, Any],
        context: dict[str, Any],
        *,
        model_call_traces: list[dict[str, Any]] | None = None,
    ) -> ProbabilisticContribution:
        return self._invoke_pfc_model_generator_impl(
            event,
            state,
            scenario,
            context,
            model_call_traces=model_call_traces,
        )

    def _invoke_pfc_model_generator_impl(
        self,
        event: RoundEvent,
        state: RuntimeState,
        scenario: dict[str, Any],
        context: dict[str, Any],
        *,
        model_call_traces: list[dict[str, Any]] | None = None,
    ) -> ProbabilisticContribution:
        generator = self._generate_pfc_candidates_via_model
        parameters = inspect.signature(generator).parameters
        if "model_call_traces" in parameters:
            return generator(
                event,
                state,
                scenario,
                context,
                model_call_traces=model_call_traces,
            )
        return generator(event, state, scenario, context)

    def _should_run_late_perspective(
        self,
        event: RoundEvent,
        state: RuntimeState,
        relation_state: dict[str, Any],
        sampled_action: str,
        *,
        route_type: str = "",
    ) -> tuple[bool, str]:
        return self._should_run_late_perspective_impl(
            event,
            state,
            relation_state,
            sampled_action,
            route_type=route_type,
        )

    def _should_run_late_perspective_impl(
        self,
        event: RoundEvent,
        state: RuntimeState,
        relation_state: dict[str, Any],
        sampled_action: str,
        *,
        route_type: str = "",
    ) -> tuple[bool, str]:
        perspective_cfg = self.config["models"].get("perspective", {})
        if not perspective_cfg.get("enabled", True):
            return False, "perspective_disabled"
        if not event.target:
            return False, "no_target"
        if state.resource_state.get("resource_mode") == "starvation":
            return False, "resource_starvation"
        action_bump = 0.08 if sampled_action in {"clarify", "connect"} else 0.0
        risk_score = max(relation_state.get("relationship_risk", 0.0), abs(event.valence) * 0.5 + action_bump)
        risk_threshold = float(perspective_cfg.get("risk_threshold", 0.33) or 0.33)
        if risk_score < risk_threshold:
            return False, "risk_gate_closed"
        if route_type == "chat_standard":
            route_ready = (
                risk_score >= max(0.38, risk_threshold + 0.08)
                or abs(float(event.valence or 0.0)) >= 0.3
                or sampled_action in {"clarify", "connect"}
            )
            if not route_ready:
                return False, "route_gate_closed"
        elif route_type in {"chat_fast", "endogenous_light", "dream_sleep"}:
            return False, "route_gate_closed"
        return True, "risk_gate_open"

    def _infer_other_state_via_model(
        self,
        event: RoundEvent,
        state: RuntimeState,
        scenario: dict[str, Any],
        context: dict[str, Any],
        sampled_action: str,
        relation_state: dict[str, Any],
        *,
        model_call_traces: list[dict[str, Any]] | None = None,
        parallel_group: str | None = None,
    ) -> dict[str, Any]:
        return self._infer_other_state_via_model_impl(
            event,
            state,
            scenario,
            context,
            sampled_action,
            relation_state,
            model_call_traces=model_call_traces,
            parallel_group=parallel_group,
        )

    def _infer_other_state_via_model_impl(
        self,
        event: RoundEvent,
        state: RuntimeState,
        scenario: dict[str, Any],
        context: dict[str, Any],
        sampled_action: str,
        relation_state: dict[str, Any],
        *,
        model_call_traces: list[dict[str, Any]] | None = None,
        parallel_group: str | None = None,
    ) -> dict[str, Any]:
        request = ModelRequest(
            system_prompt="推断对方状态。仅回 JSON:{state_hypothesis}。",
            user_prompt=self._json_prompt(
                self._build_perspective_model_payload(
                    event,
                    state,
                    scenario,
                    context,
                    relation_state,
                    action_name=sampled_action,
                    output_key="sampled_action",
                )
            ),
            response_schema={"state_hypothesis": "dict"},
            metadata={
                "sampled_action": sampled_action,
                "closeness": context.get("closeness", 0.5),
                "relationship_risk": relation_state.get("relationship_risk", 0.0),
            },
        )
        response = self._call_bound_model_route(
            "PerspectiveModel",
            route_name="perspective",
            request=request,
            binding_metadata={
                "relation_risk": float(relation_state.get("relationship_risk", 0.0) or 0.0),
                "disclosure_sensitivity": float(context.get("disclosure_sensitivity", 0.0) or 0.0),
            },
            model_call_traces=model_call_traces,
            skill_name="infer_other_state",
            parallel_group=parallel_group,
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
        *,
        model_call_traces: list[dict[str, Any]] | None = None,
        parallel_group: str | None = None,
    ) -> dict[str, Any]:
        return self._simulate_other_reaction_via_model_impl(
            event,
            state,
            scenario,
            context,
            sampled_action,
            relation_state,
            model_call_traces=model_call_traces,
            parallel_group=parallel_group,
        )

    def _simulate_other_reaction_via_model_impl(
        self,
        event: RoundEvent,
        state: RuntimeState,
        scenario: dict[str, Any],
        context: dict[str, Any],
        sampled_action: str,
        relation_state: dict[str, Any],
        *,
        model_call_traces: list[dict[str, Any]] | None = None,
        parallel_group: str | None = None,
    ) -> dict[str, Any]:
        request = ModelRequest(
            system_prompt="模拟对方反应。仅回 JSON:{reaction_hypothesis}。",
            user_prompt=self._json_prompt(
                self._build_perspective_model_payload(
                    event,
                    state,
                    scenario,
                    context,
                    relation_state,
                    action_name=sampled_action,
                    output_key="final_action",
                )
            ),
            response_schema={"reaction_hypothesis": "dict"},
            metadata={
                "sampled_action": sampled_action,
                "closeness": context.get("closeness", 0.5),
                "relationship_risk": relation_state.get("relationship_risk", 0.0),
            },
        )
        response = self._call_bound_model_route(
            "PerspectiveModel",
            route_name="perspective",
            request=request,
            binding_metadata={
                "relation_risk": float(relation_state.get("relationship_risk", 0.0) or 0.0),
                "disclosure_sensitivity": float(context.get("disclosure_sensitivity", 0.0) or 0.0),
            },
            model_call_traces=model_call_traces,
            skill_name="simulate_other_reaction",
            parallel_group=parallel_group,
        )
        return {"reaction_hypothesis": response.payload.get("reaction_hypothesis", {})}

    def _render_expression_via_model(
        self,
        render_plan: RenderPlan,
        *,
        route_type: str = "",
        prompt_mode: str = "base",
        violation_types: list[str] | None = None,
        model_call_traces: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        return self._render_expression_via_model_impl(
            render_plan,
            route_type=route_type,
            prompt_mode=prompt_mode,
            violation_types=violation_types,
            model_call_traces=model_call_traces,
        )

    def _render_expression_via_model_impl(
        self,
        render_plan: RenderPlan,
        *,
        route_type: str = "",
        prompt_mode: str = "base",
        violation_types: list[str] | None = None,
        model_call_traces: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        request = ModelRequest(
            system_prompt=self._renderer_system_prompt(
                render_plan,
                prompt_mode=prompt_mode,
                violation_types=violation_types,
            ),
            user_prompt=self._json_prompt(self._build_render_model_payload(render_plan)),
            response_schema={"text": "str"},
            metadata={
                "action": render_plan.action,
                "query_kind": render_plan.identity_context.query_kind,
                "disclosure_detail": render_plan.identity_context.disclosure_detail,
            },
        )
        response = self._call_bound_model_route(
            "Renderer",
            route_name="renderer",
            request=request,
            binding_metadata={
                "route_type": route_type,
                "query_kind": render_plan.identity_context.query_kind,
                "relation_risk": float(render_plan.relation_state.get("relationship_risk", 0.0) or 0.0),
                "conflict_hot": bool(render_plan.safety_constraints.get("conflict_hot")),
            },
            model_call_traces=model_call_traces,
            skill_name="render_expression",
        )
        text = str(response.payload.get("text", "")).strip()
        if render_plan.delivery_mode == "monologue" and text and not text.startswith("【独白】"):
            text = f"【独白】{text}"
        return {
            "text": text,
            "route": response.route,
            "model": response.model,
        }

    def _should_allow_renderer_resample(
        self,
        *,
        route_type: str,
        render_plan: RenderPlan,
        violation_types: list[str],
        context: dict[str, Any],
    ) -> bool:
        return self._should_allow_renderer_resample_impl(
            route_type=route_type,
            render_plan=render_plan,
            violation_types=violation_types,
            context=context,
        )

    def _should_allow_renderer_resample_impl(
        self,
        *,
        route_type: str,
        render_plan: RenderPlan,
        violation_types: list[str],
        context: dict[str, Any],
    ) -> bool:
        if not violation_types:
            return False
        if render_plan.identity_context.query_kind in {"self_identity", "provider_identity", "answer_explanation"}:
            return True
        if route_type in {"chat_deep", "task_run", "endogenous_deep"}:
            return True
        if route_type in {"chat_fast", "endogenous_light", "dream_sleep"}:
            return False
        if any(item in {"provider_leak", "false_self_claim"} for item in violation_types):
            return True
        return (
            float(render_plan.relation_state.get("relationship_risk", 0.0) or 0.0) >= 0.62
            or float(context.get("disclosure_sensitivity", 0.0) or 0.0) >= 0.55
            or float(context.get("authenticity_risk", 0.0) or 0.0) >= 0.32
        )

    def _score_salience_via_model(
        self,
        event: RoundEvent,
        state: RuntimeState,
        scenario: dict[str, Any],
        context: dict[str, Any],
        *,
        model_call_traces: list[dict[str, Any]] | None = None,
        parallel_group: str | None = None,
    ) -> ProbabilisticContribution:
        return self._score_salience_via_model_impl(
            event,
            state,
            scenario,
            context,
            model_call_traces=model_call_traces,
            parallel_group=parallel_group,
        )

    def _score_salience_via_model_impl(
        self,
        event: RoundEvent,
        state: RuntimeState,
        scenario: dict[str, Any],
        context: dict[str, Any],
        *,
        model_call_traces: list[dict[str, Any]] | None = None,
        parallel_group: str | None = None,
    ) -> ProbabilisticContribution:
        route_config = self._route_config_for_binding("SalienceAgent", route_name="salience_small_model")
        if route_config is None:
            return self.agent_map["SalienceAgent"].build_direct_action_contribution(event, state, scenario, context)
        request = ModelRequest(
            system_prompt="评估 salience。仅回 JSON:{action_preferences,confidence,sigma_scale,reason}。",
            user_prompt=self._json_prompt(self._build_pfc_model_payload(event, state, scenario, context)),
            response_schema={
                "action_preferences": "dict[str, float]",
                "confidence": "float",
                "sigma_scale": "float",
                "reason": "str",
            },
            metadata={"cue": context.get("cue"), "stage": "salience"},
        )
        try:
            response = self.model_router.generate_config(route_config, request)
            if model_call_traces is not None:
                self._record_model_call(
                    model_call_traces,
                    skill_name="score_salience",
                    binding_key="SalienceAgent",
                    route_config=route_config,
                    response=response,
                    prompt_chars=len(request.system_prompt) + len(request.user_prompt),
                    parallel_group=parallel_group,
                )
            prefs = {
                action: self._clip_delta(float(score))
                for action, score in self._coerce_score_map(response.payload.get("action_preferences", {})).items()
            }
            confidence = _clip(float(response.payload.get("confidence", 0.64)), 0.0, 1.0)
            sigma_scale = _clip(float(response.payload.get("sigma_scale", 0.92)), 0.60, 1.60)
            top_action = max(prefs, key=prefs.get) if prefs else ""
            dependency_trace = [
                "priority:task_goal",
                "control:task",
                f"sigma_scale:{round(sigma_scale, 4)}",
            ]
            if top_action:
                dependency_trace.append(f"top_action:{top_action}")
            return ProbabilisticContribution(
                module_name="SalienceAgent",
                module_type="salience",
                level="action",
                target_space="action",
                raw_signal=dict(prefs),
                modulated_delta=dict(prefs),
                confidence=confidence,
                confidence_calibrated=round(_clip(confidence * sigma_scale, 0.0, 1.0), 4),
                trace_reason=str(response.payload.get("reason", "small-model salience")),
                projection_reason="salience bias projected from salience model head",
                applied_at_stage="salience_attention",
                native_operator="salience_bias",
                dependency_trace=dependency_trace,
                projection=EnergyProjectionSpec(
                    module_type="salience",
                    target_space="action",
                    module_temperature=sigma_scale,
                ),
            )
        except Exception:
            return self.agent_map["SalienceAgent"].build_direct_action_contribution(event, state, scenario, context)

    def _estimate_subjective_value_via_model(
        self,
        event: RoundEvent,
        state: RuntimeState,
        scenario: dict[str, Any],
        context: dict[str, Any],
        *,
        model_call_traces: list[dict[str, Any]] | None = None,
        parallel_group: str | None = None,
    ) -> dict[str, Any]:
        patched = getattr(self.controller, "__dict__", {}).get("_estimate_subjective_value_via_model")
        if callable(patched):
            parameters = inspect.signature(patched).parameters
            kwargs: dict[str, Any] = {}
            if "model_call_traces" in parameters:
                kwargs["model_call_traces"] = model_call_traces
            if "parallel_group" in parameters:
                kwargs["parallel_group"] = parallel_group
            return patched(event, state, scenario, context, **kwargs)
        return self._estimate_subjective_value_via_model_impl(
            event,
            state,
            scenario,
            context,
            model_call_traces=model_call_traces,
            parallel_group=parallel_group,
        )

    def _estimate_subjective_value_via_model_impl(
        self,
        event: RoundEvent,
        state: RuntimeState,
        scenario: dict[str, Any],
        context: dict[str, Any],
        *,
        model_call_traces: list[dict[str, Any]] | None = None,
        parallel_group: str | None = None,
    ) -> dict[str, Any]:
        route_config = self._route_config_for_binding("ValueAgent", route_name="value_small_model")
        if route_config is None:
            return self.agent_map["ValueAgent"].estimate_subjective_value(event, state, scenario, context)
        request = ModelRequest(
            system_prompt="评估主观行动价值。仅回 JSON:{scores}。",
            user_prompt=self._json_prompt(self._build_pfc_model_payload(event, state, scenario, context)),
            response_schema={"scores": "dict[str, float]"},
            metadata={"cue": context.get("cue"), "stage": "value"},
        )
        try:
            response = self.model_router.generate_config(route_config, request)
            if model_call_traces is not None:
                self._record_model_call(
                    model_call_traces,
                    skill_name="estimate_subjective_value",
                    binding_key="ValueAgent",
                    route_config=route_config,
                    response=response,
                    prompt_chars=len(request.system_prompt) + len(request.user_prompt),
                    parallel_group=parallel_group,
                )
            return {"scores": self._coerce_score_map(response.payload.get("scores", {}))}
        except Exception:
            return self.agent_map["ValueAgent"].estimate_subjective_value(event, state, scenario, context)

    def _build_base_distribution(
        self,
        state: RuntimeState,
        scenario_cfg: dict[str, Any],
        mode_cfg: dict[str, Any],
        relation_state: dict[str, float],
    ) -> dict[str, float]:
        return self._build_base_distribution_impl(
            state,
            scenario_cfg,
            mode_cfg,
            relation_state,
        )

    def _build_base_distribution_impl(
        self,
        state: RuntimeState,
        scenario_cfg: dict[str, Any],
        mode_cfg: dict[str, Any],
        relation_state: dict[str, float],
    ) -> dict[str, float]:
        scarcity_index = float(state.resource_state.get("scarcity_index", _clip(1.0 - state.budget_remaining)))
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
            "absorb": 0.025 + float(state.memory_fragments or 0.0) * 0.14 + float(state.subjective_state.spontaneous or 0.0) * 0.06,
            "nothing": 0.015 + float(state.subjective_state.reject_all or 0.0) * 0.06,
            "die": 0.01 + (1.0 - float(state.self_continuity or 0.0)) * 0.02 + max(0.0, 0.25 - float(state.meaning_strength or 0.0)) * 0.03,
        }
        if state.mode in {"idle", "sleep"} or str(state.mode).startswith("endogenous"):
            base["monologue"] = 0.018 + 0.035
        if state.organic_mode.enabled:
            subjective_pressure = self._subjective_pressure(state)
            innate_gain = 1.0 + subjective_pressure * 0.58 + float(state.organic_mode.guard_relaxation or 0.0) * 0.16
            derived_scale = max(
                0.32,
                1.0 - subjective_pressure * 0.46 - max(0.0, float(state.organic_mode.subjective_weight or 1.0) - 1.0) * 0.1,
            )
            for action in INNATE_ACTIONS:
                if action in base:
                    base[action] *= innate_gain
            for action in DERIVED_ACTIONS:
                if action in base:
                    base[action] *= derived_scale
        if scarcity_index >= 0.60 or state.budget_remaining <= 0.10:
            base["short_reply"] = 0.02 + scarcity_index * 0.12 + max(0.0, 0.35 - state.body_energy) * 0.20
        return {action: _clip(value, 0.01, 0.85) for action, value in base.items()}

    def _compute_context_delta(
        self,
        state: RuntimeState,
        scenario_cfg: dict[str, Any],
    ) -> dict[str, float]:
        return self._compute_context_delta_impl(state, scenario_cfg)

    def _compute_context_delta_impl(
        self,
        state: RuntimeState,
        scenario_cfg: dict[str, Any],
    ) -> dict[str, float]:
        delta = {action: 0.0 for action in INTERNAL_RUNTIME_ACTIONS}
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
        scarcity_index = float(state.resource_state.get("scarcity_index", _clip(1.0 - state.budget_remaining)))
        if scarcity_index >= 0.60 or state.budget_remaining <= 0.10:
            delta["short_reply"] += 0.05 + scarcity_index * 0.10
            delta["plan"] -= 0.04 * scarcity_index
        return delta

    def _update_ci(
        self,
        state: RuntimeState,
        actions: list[str],
        context: dict[str, Any],
        scenario_cfg: dict[str, Any],
        thresholds: dict[str, Any],
    ) -> dict[str, float]:
        return self._update_ci_impl(state, actions, context, scenario_cfg, thresholds)

    def _update_ci_impl(
        self,
        state: RuntimeState,
        actions: list[str],
        context: dict[str, Any],
        scenario_cfg: dict[str, Any],
        thresholds: dict[str, Any],
    ) -> dict[str, float]:
        ci_state = dict(state.action_ci)
        if_value = _clip(0.25 + context.get("closeness", 0.5) * 0.35 + context.get("habit_strength", 0.0) * 0.30, 0.0, 1.0)
        neg = _clip(max(0.0, -context.get("valence", 0.0)), 0.0, 1.0)
        hab = _clip(context.get("habit_strength", 0.0), 0.0, 1.0)
        il = _clip((1.0 - scenario_cfg.get("pfc_base_share", 0.2)) * (1.0 if state.mode in {"idle", "interactive"} else 0.5), 0.0, 1.0)
        unc = _clip(0.3 + neg * 0.2, 0.0, 1.0)
        ov = _clip(max(0.0, 1.0 - state.budget_remaining), 0.0, 1.0)
        for action in actions:
            current = ci_state.get(action, 0.25)
            narrow_factor = _clip(1 - 0.18 * if_value - 0.22 * neg - 0.15 * hab, 0.55, 1.0)
            widen_factor = _clip(1 + 0.25 * il + 0.10 * unc, 1.0, 1.45)
            overload_guard = _clip(1 + 0.08 * ov, 1.0, 1.15)
            ci_state[action] = _clip(
                current * narrow_factor * widen_factor * overload_guard,
                thresholds["ci_min"],
                thresholds["ci_max"],
            )
        state.action_ci = ci_state
        return ci_state

    def _build_action_bookkeeping(
        self,
        rows: list[ActionEvidenceSignal | ProbabilisticContribution],
        state: RuntimeState,
        scenario_cfg: dict[str, Any],
        mode_cfg: dict[str, Any],
        relation_state: dict[str, float],
        context: dict[str, Any],
        query_state: QueryIntentState | None = None,
        disclosure_state: DisclosureIntentState | None = None,
        resample_idx: int = 0,
    ) -> ActionBookkeepingState:
        patched = getattr(self.controller, "__dict__", {}).get("_build_action_bookkeeping")
        if callable(patched):
            return patched(
                rows,
                state,
                scenario_cfg,
                mode_cfg,
                relation_state,
                context,
                query_state=query_state,
                disclosure_state=disclosure_state,
                resample_idx=resample_idx,
            )
        return self._build_action_bookkeeping_impl(
            rows,
            state,
            scenario_cfg,
            mode_cfg,
            relation_state,
            context,
            query_state=query_state,
            disclosure_state=disclosure_state,
            resample_idx=resample_idx,
        )

    def _build_action_bookkeeping_impl(
        self,
        rows: list[ActionEvidenceSignal | ProbabilisticContribution],
        state: RuntimeState,
        scenario_cfg: dict[str, Any],
        mode_cfg: dict[str, Any],
        relation_state: dict[str, float],
        context: dict[str, Any],
        query_state: QueryIntentState | None = None,
        disclosure_state: DisclosureIntentState | None = None,
        resample_idx: int = 0,
    ) -> ActionBookkeepingState:
        del resample_idx
        thresholds = self.config["thresholds"]["thresholds"]
        p_base = self._build_base_distribution(state, scenario_cfg, mode_cfg, relation_state)
        signals = self._coerce_action_signals(rows)
        actions = sorted({*p_base.keys(), *(action for signal in signals for action in signal.action_delta.keys())})
        u_base: dict[str, float] = {}
        risk_suppressor: dict[str, float] = {}
        gate = {action: 1.0 for action in actions}
        ci = self._update_ci(state, actions, context, scenario_cfg, thresholds)

        for action in actions:
            base_probability = p_base.get(action, 0.02)
            base_utility = math.log(max(base_probability, thresholds["p_floor"]))
            u_base[action] = round(base_utility, 6)
            suppressor = 1.0
            if action == "wander" and not mode_cfg.get("allow_dmn", True):
                suppressor = 0.4
            if action == "connect" and relation_state["boundary_level"] > 0.7:
                suppressor = 0.65
            risk_suppressor[action] = suppressor

        return ActionBookkeepingState(
            u_base=u_base,
            p_base=p_base,
            ci=ci,
            gate=gate,
            risk_suppressor=risk_suppressor,
            query_intent={
                "top_intent": query_state.posterior.top_intent if query_state is not None else "",
                "posterior": dict(query_state.posterior.posterior) if query_state is not None else {},
                "legacy_query_kind": query_state.legacy_query_kind if query_state is not None else "general",
            },
            disclosure_intent={
                "top_intent": disclosure_state.posterior.top_intent if disclosure_state is not None else "",
                "posterior": dict(disclosure_state.posterior.posterior) if disclosure_state is not None else {},
                "legacy_disclosure_detail": disclosure_state.legacy_disclosure_detail if disclosure_state is not None else "none",
                "clipped": list(disclosure_state.posterior.clipped) if disclosure_state is not None else [],
            },
        )

    def _top_action_name(self, distribution: dict[str, float]) -> str | None:
        return self._top_action_name_impl(distribution)

    def _top_action_name_impl(self, distribution: dict[str, float]) -> str | None:
        if not distribution:
            return None
        return max(distribution.items(), key=lambda item: (item[1], item[0]))[0]

    def _build_repair_expression_policy(self, conflict_state: dict[str, Any]) -> dict[str, Any]:
        return self._build_repair_expression_policy_impl(conflict_state)

    def _build_repair_expression_policy_impl(self, conflict_state: dict[str, Any]) -> dict[str, Any]:
        repair_transition = conflict_state.get("repair_transition", {}) or {}
        repair_snapshot = conflict_state.get("repair_state_snapshot", {}) or {}
        stage = str(repair_snapshot.get("stage") or repair_transition.get("to_stage") or "idle")
        template = conflict_state.get("compromise", {}).get("template")
        transition_reason = repair_transition.get("reason")
        source = "conflict" if stage in {"adjusting", "repairing", "cooling", "recovered"} else None

        policy = {
            "source": source,
            "stage": stage,
            "visibility": "implicit",
            "opening_mode": "none",
            "advance_mode": "normal",
            "safety_invite": False,
            "template": template,
            "transition_reason": transition_reason,
        }

        if stage == "adjusting":
            policy["opening_mode"] = "buffered"
            policy["advance_mode"] = "limited"
        elif stage == "repairing":
            policy["opening_mode"] = "buffered"
            policy["advance_mode"] = "limited"
            policy["safety_invite"] = True
        elif stage == "cooling":
            policy["opening_mode"] = "soft_resume"
            policy["advance_mode"] = "resume"

        return policy

    def _safe_mode_delta(self, before: bool, after: bool) -> str:
        return self._safe_mode_delta_impl(before, after)

    def _safe_mode_delta_impl(self, before: bool, after: bool) -> str:
        if before == after:
            return "unchanged"
        return "entered" if after else "exited"

    def _update_conflict_circuit(
        self,
        state: RuntimeState,
        *,
        critical_conflict: bool,
        winning_priority: str | None,
        compromise_template: str | None,
    ) -> dict[str, Any]:
        return self._update_conflict_circuit_impl(
            state,
            critical_conflict=critical_conflict,
            winning_priority=winning_priority,
            compromise_template=compromise_template,
        )

    def _update_conflict_circuit_impl(
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
        return self._apply_conflict_expression_adjustments_impl(expression, conflict_state)

    def _apply_conflict_expression_adjustments_impl(
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

    def _fallback_render_route_names(self) -> list[str]:
        preferred_groups = [
            ("renderer_fallback_fast", "chat_fast"),
            ("renderer_fallback_small", "monologue_stream"),
        ]
        selected: list[str] = []
        for group in preferred_groups:
            for route_name in group:
                route_cfg = self.model_router.route_configs.get(route_name)
                if route_cfg is None or not route_cfg.enabled or route_name in selected:
                    continue
                selected.append(route_name)
                break
        return selected

    def _build_deterministic_render_fallback_output(
        self,
        render_plan: RenderPlan,
        *,
        model_label: str,
    ) -> dict[str, Any]:
        return {
            "text": fallback_render_text(render_plan),
            "route": "renderer",
            "model": model_label,
        }

    def _fallback_renderer_system_prompt(
        self,
        render_plan: RenderPlan,
        *,
        fallback_reason: str,
        violation_types: list[str] | None = None,
    ) -> str:
        base_prompt = self._renderer_system_prompt(
            render_plan,
            prompt_mode="violation" if violation_types else "base",
            violation_types=violation_types,
        )
        extra_lines = [
            "当前处于降级表达链，但还没有到最终模板兜底。",
            "你会收到 deterministic_bottom_line，它只是最后失败时的表达底线，不是你应该直接照抄的输出。",
            "保持同一语义和边界，把它改写成更自然、更像此刻真人开口的表达。",
            "不要客服腔，不要功能介绍腔，不要元叙述，不要暴露 provider 视角。",
            f"fallback_reason={fallback_reason}。",
        ]
        return " ".join([base_prompt, *extra_lines])

    def _build_fallback_render_model_payload(
        self,
        render_plan: RenderPlan,
        *,
        fallback_reason: str,
        violation_types: list[str] | None = None,
    ) -> dict[str, Any]:
        payload = self._build_render_model_payload(render_plan)
        payload["fallback_reason"] = fallback_reason
        payload["violation_types"] = list(violation_types or [])
        payload["deterministic_bottom_line"] = fallback_render_text(render_plan)
        payload["rewrite_target"] = "humanized_first_person_runtime_reply"
        return payload

    def _render_expression_via_humanized_fallback_chain(
        self,
        render_plan: RenderPlan,
        *,
        fallback_reason: str,
        deterministic_model: str,
        violation_types: list[str] | None = None,
        model_call_traces: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        request = ModelRequest(
            system_prompt=self._fallback_renderer_system_prompt(
                render_plan,
                fallback_reason=fallback_reason,
                violation_types=violation_types,
            ),
            user_prompt=self._json_prompt(
                self._build_fallback_render_model_payload(
                    render_plan,
                    fallback_reason=fallback_reason,
                    violation_types=violation_types,
                )
            ),
            response_schema={"text": "str"},
            metadata={
                "action": render_plan.action,
                "query_kind": render_plan.identity_context.query_kind,
                "disclosure_detail": render_plan.identity_context.disclosure_detail,
                "fallback_reason": fallback_reason,
            },
        )
        for route_name in self._fallback_render_route_names():
            route_cfg = self.model_router.route_configs.get(route_name)
            if route_cfg is None or not route_cfg.enabled:
                continue
            try:
                response = self.model_router.generate(route_name, request)
            except Exception:
                continue
            text = str(response.payload.get("text", "")).strip()
            if not text:
                continue
            if render_plan.delivery_mode == "monologue" and not text.startswith("【独白】"):
                text = f"【独白】{text}"
            if model_call_traces is not None:
                self._record_model_call(
                    model_call_traces,
                    skill_name="render_expression_fallback",
                    binding_key="Renderer",
                    route_config=route_cfg,
                    response=response,
                    prompt_chars=len(request.system_prompt) + len(request.user_prompt),
                )
            return {
                "text": text,
                "route": "renderer",
                "model": str(response.model or route_cfg.model),
            }
        return self._build_deterministic_render_fallback_output(
            render_plan,
            model_label=deterministic_model,
        )

    def _execute_parallel_skills(
        self,
        *,
        round_id: int,
        tasks: list[dict[str, Any]],
        skill_traces: list[dict[str, Any]],
        runtime_context: SkillRuntimeContext,
        parallel_traces: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        results: dict[str, Any] = {}
        if not tasks:
            return results
        executor = ThreadPoolExecutor(max_workers=len(tasks))
        futures: dict[Any, tuple[dict[str, Any], float]] = {}
        try:
            for task in tasks:
                task_type = str(task.get("task_type", "skill" if task.get("skill_name") else "callable"))
                started = time.perf_counter()
                if task_type == "skill":
                    future = executor.submit(
                        self.skill_executor.run,
                        round_id=round_id,
                        skill_name=task["skill_name"],
                        inputs=task["inputs"],
                        provider=task["provider"],
                        fallback_provider=task.get("fallback_provider"),
                        fallback_value=task.get("fallback_value"),
                        runtime_context=runtime_context,
                        seed_ref=task.get("seed_ref"),
                    )
                else:
                    future = executor.submit(
                        self.skill_executor._invoke_callable,
                        task["provider"],
                        dict(task.get("inputs", {})),
                    )
                futures[future] = (task, started)

            for future, (task, started) in futures.items():
                task_type = str(task.get("task_type", "skill" if task.get("skill_name") else "callable"))
                task_name = str(task["name"])
                priority = str(task.get("priority", "required"))
                parallel_group = task.get("parallel_group")
                binding_tier = task.get("binding_tier")
                timeout_ms = int(task.get("timeout_ms", 0) or 0)
                task_outcome = "completed"
                try:
                    if timeout_ms > 0:
                        raw_result = future.result(timeout=max(timeout_ms / 1000.0, 0.001))
                    else:
                        raw_result = future.result()
                except FutureTimeoutError:
                    task_outcome = "timeout"
                    output = self._parallel_fallback_output(task)
                    latency_ms = max(1, timeout_ms or int((time.perf_counter() - started) * 1000))
                    if task_type == "skill":
                        result = self._parallel_timeout_skill_result(
                            round_id=round_id,
                            task=task,
                            latency_ms=latency_ms,
                        )
                        self._record_skill_trace(
                            skill_traces,
                            round_id,
                            result,
                            extra={
                                "parallel_group": parallel_group,
                                "binding_tier": binding_tier,
                                "task_priority": priority,
                                "task_outcome": task_outcome,
                                "task_type": task_type,
                                "timeout_ms": timeout_ms,
                            },
                        )
                    results[task_name] = output
                except Exception:
                    task_outcome = "fallback"
                    output = self._parallel_fallback_output(task)
                    latency_ms = max(1, int((time.perf_counter() - started) * 1000))
                    if task_type == "skill":
                        result = self._parallel_timeout_skill_result(
                            round_id=round_id,
                            task=task,
                            latency_ms=latency_ms,
                            failure_policy="parallel_exception_fallback",
                        )
                        self._record_skill_trace(
                            skill_traces,
                            round_id,
                            result,
                            extra={
                                "parallel_group": parallel_group,
                                "binding_tier": binding_tier,
                                "task_priority": priority,
                                "task_outcome": task_outcome,
                                "task_type": task_type,
                                "timeout_ms": timeout_ms,
                            },
                        )
                    results[task_name] = output
                else:
                    latency_ms = max(1, int((time.perf_counter() - started) * 1000))
                    if task_type == "skill":
                        output, result = raw_result
                        result.parallel_group = parallel_group
                        result.binding_tier = binding_tier
                        setattr(result, "task_priority", priority)
                        setattr(result, "task_outcome", "fallback" if result.degraded else "completed")
                        setattr(result, "task_type", task_type)
                        setattr(result, "timeout_ms", timeout_ms)
                        self._record_skill_trace(
                            skill_traces,
                            round_id,
                            result,
                            extra={
                                "parallel_group": parallel_group,
                                "binding_tier": binding_tier,
                                "task_priority": priority,
                                "task_outcome": getattr(result, "task_outcome", "completed"),
                                "task_type": task_type,
                                "timeout_ms": timeout_ms,
                            },
                        )
                        task_outcome = getattr(result, "task_outcome", "completed")
                        results[task_name] = output
                    else:
                        results[task_name] = raw_result
                if parallel_traces is not None:
                    parallel_traces.append(
                        {
                            "task_name": task_name,
                            "task_type": task_type,
                            "parallel_group": parallel_group,
                            "task_priority": priority,
                            "task_outcome": task_outcome,
                            "timeout_ms": timeout_ms,
                            "latency_ms": latency_ms,
                            "binding_tier": binding_tier,
                            "started_at_ms": round(started * 1000, 3),
                            "finished_at_ms": round(time.perf_counter() * 1000, 3),
                        }
                    )
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
        return results

    def _parallel_fallback_output(self, task: dict[str, Any]) -> Any:
        fallback_provider = task.get("fallback_provider")
        if callable(fallback_provider):
            try:
                return self.skill_executor._invoke_callable(
                    fallback_provider,
                    dict(task.get("inputs", {})),
                )
            except Exception:
                pass
        if "fallback_value" in task:
            return task.get("fallback_value")
        return {}

    def _parallel_timeout_skill_result(
        self,
        *,
        round_id: int,
        task: dict[str, Any],
        latency_ms: int,
        failure_policy: str = "parallel_timeout",
    ) -> SkillResult:
        spec = self.skill_executor.registry[str(task["skill_name"])]
        fallback_output = self._parallel_fallback_output(task)
        normalized_output = to_dict(fallback_output)
        return SkillResult(
            skill_name=str(task["skill_name"]),
            owner_module=spec.owner_module,
            output=normalized_output if isinstance(normalized_output, dict) else {"value": normalized_output},
            latency_ms=max(1, latency_ms),
            cost_class=spec.cost_class,
            degraded=True,
            failure_policy_applied=failure_policy,
            seed_ref=task.get("seed_ref"),
            fallback_route=spec.fallback_route.target if spec.fallback_route else None,
            fallback_cost_class=spec.fallback_route.cost_class if spec.fallback_route else None,
            breaker_state={},
            parallel_group=task.get("parallel_group"),
            binding_tier=task.get("binding_tier"),
            task_priority=str(task.get("priority", "required")),
            task_outcome="timeout" if failure_policy == "parallel_timeout" else "fallback",
            task_type=str(task.get("task_type", "skill")),
            timeout_ms=int(task.get("timeout_ms", 0) or 0),
        )

    def _action_evidence_from_contribution(
        self,
        contribution: ProbabilisticContribution,
        *,
        priority_bucket: str | None = None,
        control_domain: str | None = None,
        gated_actions: list[str] | None = None,
        risk_hints: dict[str, Any] | None = None,
        veto: bool = False,
        trace_tags: list[str] | None = None,
    ) -> ActionEvidenceSignal:
        projected_delta = self._projected_action_delta_from_contribution(contribution)
        projection = contribution.projection or EnergyProjectionSpec(
            module_type=contribution.module_type,
            target_space=contribution.target_space,
        )
        hard_masked = {
            action
            for action, blocked in dict(contribution.hard_mask or {}).items()
            if blocked
        }
        hard_masked.update(
            str(action)
            for action in list(gated_actions or [])
            if isinstance(action, str) and action
        )
        return ActionEvidenceSignal(
            module_name=contribution.module_name,
            module_type=contribution.module_type,
            confidence=float(contribution.confidence),
            action_delta=projected_delta,
            utility_shift=dict(projected_delta),
            sigma_scale=max(0.6, min(1.6, float(projection.module_temperature or 1.0))),
            priority_bucket=priority_bucket or OWNER_PRIORITY_BUCKET.get(contribution.module_name, "task_goal"),
            control_domain=control_domain or OWNER_CONTROL_DOMAIN.get(contribution.module_name, "task"),
            gated_actions=sorted(hard_masked),
            risk_hints=dict(risk_hints or {}),
            veto=bool(veto),
            trace_reason=contribution.trace_reason,
            trace_tags=list(trace_tags or [str(contribution.module_type)]),
        )

    def _projected_action_delta_from_contribution(
        self,
        contribution: ProbabilisticContribution,
    ) -> dict[str, float]:
        projected: dict[str, float] = {
            str(action): float(value)
            for action, value in dict(contribution.modulated_delta or contribution.raw_signal or {}).items()
            if isinstance(action, str) and action
        }
        for action, value in dict(contribution.inhibitory_drive or {}).items():
            action_name = str(action).strip()
            if not action_name:
                continue
            projected[action_name] = round(
                projected.get(action_name, 0.0) - abs(float(value)),
                6,
            )
        return {
            action: round(float(value), 6)
            for action, value in projected.items()
            if abs(float(value)) >= 1e-9
        }

    def _action_signal_metadata(
        self,
        *,
        owner: str,
        state: RuntimeState,
        relation_state: dict[str, float],
        context: dict[str, Any],
        module_type: str | None = None,
        priority_bucket: str | None = None,
        control_domain: str | None = None,
        projected_delta: dict[str, float] | None = None,
        utility_shift: dict[str, float] | None = None,
        gated_actions: list[str] | None = None,
        risk_hints: dict[str, Any] | None = None,
        veto: bool = False,
        trace_tags: list[str] | None = None,
    ) -> dict[str, Any]:
        metadata = {
            "priority_bucket": priority_bucket
            or {
                "BodyStateAgent": "body_safety",
                "EmotionAgent": "body_safety",
                "ResourceAgent": "budget_overload",
                "RelationshipAgent": "relation_boundary",
                "PerspectiveModel": "relation_boundary",
                "DesireAgent": "immediate_desire",
                "DMNAgent": "roaming",
            }.get(owner, "task_goal"),
            "control_domain": control_domain
            or {
                "BodyStateAgent": "body",
                "EmotionAgent": "body",
                "ResourceAgent": "resource",
                "RelationshipAgent": "relation",
                "PerspectiveModel": "relation",
                "DesireAgent": "desire",
                "DMNAgent": "dmn",
            }.get(owner, "task"),
            "gated_actions": [
                str(action)
                for action in list(gated_actions or [])
                if isinstance(action, str) and action
            ],
            "risk_hints": dict(risk_hints or {}),
            "veto": bool(veto),
            "trace_tags": list(trace_tags or ([str(module_type)] if module_type else [])),
        }

        gated = set(metadata["gated_actions"])
        hints = dict(metadata["risk_hints"])
        projected = {
            str(action): float(value)
            for action, value in dict(projected_delta or {}).items()
            if isinstance(action, str) and action
        }
        utility = {
            str(action): float(value)
            for action, value in dict(utility_shift or {}).items()
            if isinstance(action, str) and action
        }

        if owner == "BodyStateAgent":
            hints.setdefault("body_load", round(_clip(1.0 - state.body_energy), 4))
            hints.setdefault("body_energy", round(state.body_energy, 4))
            if state.body_energy < 0.20:
                gated.add("connect")
            if state.body_energy < 0.12:
                gated.add("plan")
        elif owner == "ResourceAgent":
            overload = round(_clip(1.0 - state.budget_remaining), 4)
            hints.setdefault("overload", overload)
            for key in (
                "scarcity_pressure",
                "body_hunger_bias",
                "effort_avoidance_bias",
                "deliberation_compress",
                "rumination_bias",
                "action_shrink_scale",
            ):
                value = state.resource_state.get(key)
                if isinstance(value, (int, float)):
                    hints.setdefault(key, round(float(value), 4))
            if overload > 0.80:
                gated.add("plan")
        elif owner in {"RelationshipAgent", "PerspectiveModel"}:
            hints.setdefault("relationship_risk", round(relation_state["relationship_risk"], 4))
            hints.setdefault("boundary_level", round(relation_state["boundary_level"], 4))
            if relation_state["boundary_level"] > 0.70:
                gated.add("connect")
        elif owner == "PFCAgent":
            hints.setdefault("goal_pressure", round(max(projected.values(), default=0.0), 4))
        elif owner == "ValueAgent":
            hints.setdefault("goal_pressure", round(max(utility.values(), default=0.0), 4))
        elif owner == "DesireAgent":
            hints.setdefault("comfort_pull", round(max(projected.values(), default=0.0), 4))
        elif owner == "DMNAgent":
            hints.setdefault("roam_pull", round(projected.get("wander", 0.0), 4))
        elif owner == "HabitAgent":
            hints.setdefault("habit_strength", round(float(context.get("habit_strength", 0.0)), 4))

        metadata["gated_actions"] = sorted(gated)
        metadata["risk_hints"] = hints
        return metadata

    def _coerce_action_signals(
        self,
        rows: list[ActionEvidenceSignal | ProbabilisticContribution],
    ) -> list[ActionEvidenceSignal]:
        signals: list[ActionEvidenceSignal] = []
        for row in rows:
            if isinstance(row, ActionEvidenceSignal):
                signals.append(row)
            else:
                signals.append(self._action_evidence_from_contribution(row))
        return signals

    def _build_value_action_contribution(
        self,
        value_scores: dict[str, Any],
    ) -> ProbabilisticContribution:
        scores = {
            str(action): float(value)
            for action, value in dict(value_scores.get("scores", {}) or {}).items()
        }
        sigma_scale = 0.98
        top_action = max(scores, key=scores.get) if scores else ""
        dependency_trace = [
            "priority:task_goal",
            "control:task",
            f"sigma_scale:{round(sigma_scale, 4)}",
        ]
        if top_action:
            dependency_trace.append(f"top_action:{top_action}")
        return ProbabilisticContribution(
            module_name="ValueAgent",
            module_type="value",
            level="action",
            target_space="action",
            raw_signal=dict(scores),
            modulated_delta=dict(scores),
            confidence=0.59,
            confidence_calibrated=round(_clip(0.59 * sigma_scale, 0.0, 1.0), 4),
            trace_reason="subjective value re-rank",
            projection_reason="value bias projected from value head",
            applied_at_stage="subjective_value",
            native_operator="value_bias",
            dependency_trace=dependency_trace,
            projection=EnergyProjectionSpec(
                module_type="value",
                target_space="action",
                module_temperature=sigma_scale,
            ),
        )

    def _candidate_distribution_from_trace(self, trace: dict[str, Any]) -> dict[str, float]:
        action_layer = self._action_layer_from_trace(trace)
        winner_posterior = action_layer.get("winner_posterior", {}) if isinstance(action_layer, dict) else {}
        if isinstance(winner_posterior, dict) and winner_posterior:
            return {
                str(action): round(float(value), 6)
                for action, value in winner_posterior.items()
            }
        candidate_distribution = trace.get("candidate_distribution", {})
        if isinstance(candidate_distribution, dict) and candidate_distribution:
            return {
                str(action): round(float(value), 6)
                for action, value in candidate_distribution.items()
            }
        return {}

    def _probability_field_couplings(self) -> list[CrossLayerCouplingSpec]:
        return [
            CrossLayerCouplingSpec(
                source_layer="context",
                target_layer="memory",
                carrier_signal="context_route",
                projection_rule="field_native",
                allowed_phase="tick",
            ),
            CrossLayerCouplingSpec(
                source_layer="memory",
                target_layer="action",
                carrier_signal="memory_prior",
                projection_rule="field_native",
                allowed_phase="tick",
            ),
            CrossLayerCouplingSpec(
                source_layer="memory",
                target_layer="action",
                carrier_signal="organic_memory",
                projection_rule="field_native",
                allowed_phase="tick",
            ),
            CrossLayerCouplingSpec(
                source_layer="action",
                target_layer="token",
                carrier_signal="render_plan",
                projection_rule="field_native",
                allowed_phase="render",
            ),
        ]

    def _collect_probability_field_contributions(
        self,
        *,
        direct_action_contributions: dict[str, ProbabilisticContribution] | None,
        render_plan: RenderPlan | None,
        event: RoundEvent,
        state: RuntimeState,
        scenario_cfg: dict[str, Any],
        context: dict[str, Any],
        identity_context: IdentityContext | None,
        authenticity: AuthenticityRecord | None,
        vitality_snapshot: dict[str, Any] | None,
        long_run_contribution: ProbabilisticContribution | None,
        include_conflict: bool,
        include_token: bool,
        control_ledger: dict[str, Any] | None = None,
        action_truth: dict[str, Any] | None = None,
    ) -> tuple[list[ProbabilisticContribution], TokenFieldState]:
        thalamus_agent = self.agent_map["ThalamusAttentionAgent"]
        hippocampus_agent = self.agent_map["HippocampusAgent"]
        conflict_agent = self.agent_map["ConflictMonitorAgent"]

        contribution_rows: list[ProbabilisticContribution] = []
        contribution_rows.append(
            thalamus_agent.build_context_routing_contribution(event, state, scenario_cfg, context)
        )
        contribution_rows.append(
            thalamus_agent.build_memory_routing_contribution(event, state, scenario_cfg, context)
        )
        contribution_rows.append(
            hippocampus_agent.build_memory_prior_contribution(event, state, scenario_cfg, context)
        )
        tlh_memory_contribution = self._build_tlh_memory_contribution(state)
        if tlh_memory_contribution is not None:
            contribution_rows.append(tlh_memory_contribution)
        if include_token and render_plan is not None:
            contribution_rows.append(self._build_renderer_token_contribution(render_plan, context))
        tool_contribution = self._build_tool_affordance_contribution(state, context)
        if tool_contribution is not None:
            contribution_rows.append(tool_contribution)
        if identity_context is not None:
            contribution_rows.append(
                self.identity_runtime.build_identity_prior_contribution(
                    identity_context=identity_context,
                    state=state,
                )
            )
        if authenticity is not None:
            contribution_rows.append(
                self.authenticity_policy.build_action_penalty_contribution(authenticity)
            )
        if vitality_snapshot is not None:
            contribution_rows.append(
                self.vitality_engine.build_vitality_modulation_contribution(vitality_snapshot)
            )
        if long_run_contribution is not None:
            contribution_rows.append(long_run_contribution)

        for direct_contribution in dict(direct_action_contributions or {}).values():
            contribution_rows.append(direct_contribution)

        if include_conflict:
            conflict_state = dict((control_ledger or {}).get("conflict", {}) or {})
            conflict_view = dict(action_truth or {})
            if conflict_state and conflict_view:
                contribution_rows.append(
                    conflict_agent.build_arbitration_contribution(conflict_state, conflict_view)
                )

        token_state = TokenFieldState(
            step_index=state.round_count,
            prefix_tokens=event.content.split()[:12],
            active_module_sources=sorted({row.module_name for row in contribution_rows} | {"Renderer"}),
            generated_delta_sources=[],
        )
        return contribution_rows, token_state

    def _integrate_probability_field_snapshot(
        self,
        *,
        direct_action_contributions: dict[str, ProbabilisticContribution] | None,
        action_base: dict[str, float],
        event: RoundEvent,
        state: RuntimeState,
        scenario_cfg: dict[str, Any],
        context: dict[str, Any],
        identity_context: IdentityContext | None,
        authenticity: AuthenticityRecord | None = None,
        vitality_snapshot: dict[str, Any] | None = None,
        long_run_contribution: ProbabilisticContribution | None = None,
        include_conflict: bool = False,
        include_token: bool = False,
        render_plan: RenderPlan | None = None,
        source_chain: list[str] | None = None,
        control_ledger: dict[str, Any] | None = None,
        action_truth: dict[str, Any] | None = None,
    ) -> tuple[ProbabilityFieldSnapshot, list[ProbabilisticContribution], TokenFieldState]:
        patched = getattr(self.controller, "__dict__", {}).get("_integrate_probability_field_snapshot")
        if callable(patched):
            return patched(
                direct_action_contributions=direct_action_contributions,
                action_base=action_base,
                event=event,
                state=state,
                scenario_cfg=scenario_cfg,
                context=context,
                identity_context=identity_context,
                authenticity=authenticity,
                vitality_snapshot=vitality_snapshot,
                long_run_contribution=long_run_contribution,
                include_conflict=include_conflict,
                include_token=include_token,
                render_plan=render_plan,
                source_chain=source_chain,
                control_ledger=control_ledger,
                action_truth=action_truth,
            )
        return self._integrate_probability_field_snapshot_impl(
            direct_action_contributions=direct_action_contributions,
            action_base=action_base,
            event=event,
            state=state,
            scenario_cfg=scenario_cfg,
            context=context,
            identity_context=identity_context,
            authenticity=authenticity,
            vitality_snapshot=vitality_snapshot,
            long_run_contribution=long_run_contribution,
            include_conflict=include_conflict,
            include_token=include_token,
            render_plan=render_plan,
            source_chain=source_chain,
            control_ledger=control_ledger,
            action_truth=action_truth,
        )

    def _integrate_probability_field_snapshot_impl(
        self,
        *,
        direct_action_contributions: dict[str, ProbabilisticContribution] | None,
        action_base: dict[str, float],
        event: RoundEvent,
        state: RuntimeState,
        scenario_cfg: dict[str, Any],
        context: dict[str, Any],
        identity_context: IdentityContext | None,
        authenticity: AuthenticityRecord | None = None,
        vitality_snapshot: dict[str, Any] | None = None,
        long_run_contribution: ProbabilisticContribution | None = None,
        include_conflict: bool = False,
        include_token: bool = False,
        render_plan: RenderPlan | None = None,
        source_chain: list[str] | None = None,
        control_ledger: dict[str, Any] | None = None,
        action_truth: dict[str, Any] | None = None,
    ) -> tuple[ProbabilityFieldSnapshot, list[ProbabilisticContribution], TokenFieldState]:
        contribution_rows, token_state = self._collect_probability_field_contributions(
            direct_action_contributions=direct_action_contributions,
            control_ledger=control_ledger,
            action_truth=action_truth,
            render_plan=render_plan,
            event=event,
            state=state,
            scenario_cfg=scenario_cfg,
            context=context,
            identity_context=identity_context,
            authenticity=authenticity,
            vitality_snapshot=vitality_snapshot,
            long_run_contribution=long_run_contribution,
            include_conflict=include_conflict,
            include_token=include_token,
        )
        snapshot = self.probability_integrator.integrate(
            context_base={
                "task_relevance": round(float(scenario_cfg.get("pfc_base_share", 0.0) or 0.0), 6),
                "relation_context": round(float(context.get("closeness", 0.0) or 0.0), 6),
            },
            memory_base={
                str(context.get("cue") or "memory:empty"): round(
                    float(context.get("recall_strength", 0.0) or 0.0),
                    6,
                ),
            },
            action_base={
                str(action): float(value)
                for action, value in dict(action_base or {}).items()
            },
            token_state=token_state,
            contributions=contribution_rows,
            couplings=self._probability_field_couplings(),
            source_chain=source_chain or ["probability_field_native", "context_memory_action"],
        )
        return snapshot, contribution_rows, token_state

    def _finalize_action_bookkeeping_from_action_layer(
        self,
        action_bookkeeping: ActionBookkeepingState,
        action_layer: dict[str, Any] | ProbabilityLayerState,
        *,
        finalize_stage: str,
    ) -> None:
        patched = getattr(self.controller, "__dict__", {}).get(
            "_finalize_action_bookkeeping_from_action_layer"
        )
        if callable(patched):
            return patched(
                action_bookkeeping,
                action_layer,
                finalize_stage=finalize_stage,
            )
        return self._finalize_action_bookkeeping_from_action_layer_impl(
            action_bookkeeping,
            action_layer,
            finalize_stage=finalize_stage,
        )

    def _finalize_action_bookkeeping_from_action_layer_impl(
        self,
        action_bookkeeping: ActionBookkeepingState,
        action_layer: dict[str, Any] | ProbabilityLayerState,
        *,
        finalize_stage: str,
    ) -> None:
        del finalize_stage
        if isinstance(action_layer, ProbabilityLayerState):
            action_keys = {
                *action_bookkeeping.p_base.keys(),
                *dict(action_layer.final_energy or {}).keys(),
                *dict(action_layer.winner_posterior or {}).keys(),
            }
        else:
            action_keys = {
                *action_bookkeeping.p_base.keys(),
                *dict(action_layer.get("final_energy", {}) or {}).keys(),
                *dict(action_layer.get("winner_posterior", {}) or {}).keys(),
            }
        all_actions = sorted(
            str(action) for action in action_keys if isinstance(action, str) and action
        )
        action_bookkeeping.gate = {
            action: float(action_bookkeeping.gate.get(action, 1.0) or 1.0)
            for action in all_actions
        }
        action_bookkeeping.risk_suppressor = {
            action: float(action_bookkeeping.risk_suppressor.get(action, 1.0) or 1.0)
            for action in all_actions
        }

    def _reintegrate_probability_snapshot(
        self,
        *,
        snapshot: ProbabilityFieldSnapshot,
        contributions: list[ProbabilisticContribution],
        token_state: TokenFieldState,
        source_chain: list[str] | None = None,
    ) -> ProbabilityFieldSnapshot:
        return self.probability_integrator.reintegrate_action_token_layers(
            snapshot=snapshot,
            contributions=contributions,
            token_state=token_state,
            source_chain=source_chain,
        )

    def _action_truth_from_field(
        self,
        action_layer: ProbabilityLayerState | dict[str, Any],
        *,
        gate: dict[str, float] | None = None,
        conflict_mode: str = "field_native",
    ) -> dict[str, Any]:
        if isinstance(action_layer, ProbabilityLayerState):
            winner_posterior = {
                str(action): float(value)
                for action, value in dict(action_layer.winner_posterior or {}).items()
            }
            final_energy = {
                str(action): float(value)
                for action, value in dict(action_layer.final_energy or {}).items()
            }
            hard_masked_targets = [
                str(action)
                for action in list(action_layer.hard_masked_targets or [])
            ]
        else:
            winner_posterior = {
                str(action): float(value)
                for action, value in dict(action_layer.get("winner_posterior", {}) or {}).items()
            }
            final_energy = {
                str(action): float(value)
                for action, value in dict(action_layer.get("final_energy", {}) or {}).items()
            }
            hard_masked_targets = [
                str(action)
                for action in list(action_layer.get("hard_masked_targets", []) or [])
            ]

        gate_map = {
            str(action): float(value)
            for action, value in dict(gate or {}).items()
        }
        for action in winner_posterior:
            gate_map.setdefault(action, 0.0 if action in hard_masked_targets else 1.0)
        return {
            "winner_posterior": winner_posterior,
            "final_energy": final_energy,
            "gate": gate_map,
            "hard_masked_targets": sorted(
                {
                    *hard_masked_targets,
                    *[action for action, value in gate_map.items() if value <= 0.0],
                }
            ),
            "conflict_mode": conflict_mode,
        }

    def _refresh_action_truth(
        self,
        action_layer: ProbabilityLayerState | dict[str, Any],
        current_action_truth: dict[str, Any] | None = None,
        *,
        conflict_mode: str = "field_native",
    ) -> dict[str, Any]:
        gate = (
            {
                str(action): float(value)
                for action, value in dict(current_action_truth.get("gate", {}) or {}).items()
            }
            if isinstance(current_action_truth, dict)
            else None
        )
        effective_mode = (
            str(current_action_truth.get("conflict_mode") or conflict_mode)
            if isinstance(current_action_truth, dict)
            else conflict_mode
        )
        return self._action_truth_from_field(
            action_layer,
            gate=gate,
            conflict_mode=effective_mode,
        )

    def _control_ledger_from_action_bookkeeping(
        self,
        action_bookkeeping: ActionBookkeepingState,
    ) -> dict[str, Any]:
        return {
            "ci": {
                str(action): float(value)
                for action, value in dict(action_bookkeeping.ci or {}).items()
            },
            "gate": {
                str(action): float(value)
                for action, value in dict(action_bookkeeping.gate or {}).items()
            },
            "risk_suppressor": {
                str(action): float(value)
                for action, value in dict(action_bookkeeping.risk_suppressor or {}).items()
            },
            "resample_idx": 0,
            "conflict_mode": "none",
            "conflict": {},
        }

    def _merge_control_ledger_into_action_bookkeeping(
        self,
        action_bookkeeping: ActionBookkeepingState,
        control_ledger: dict[str, Any],
    ) -> None:
        action_bookkeeping.ci = {
            str(action): float(value)
            for action, value in dict(control_ledger.get("ci", {}) or {}).items()
            if isinstance(action, str) and action
        }
        action_bookkeeping.risk_suppressor = {
            str(action): float(value)
            for action, value in dict(control_ledger.get("risk_suppressor", {}) or {}).items()
            if isinstance(action, str) and action
        }

    def _action_bookkeeping_payload(
        self,
        *,
        action_bookkeeping: ActionBookkeepingState,
        probability_field_snapshot: ProbabilityFieldSnapshot,
        control_ledger: dict[str, Any],
        action_truth: dict[str, Any],
        stochastic_state: StochasticState,
    ) -> dict[str, Any]:
        del probability_field_snapshot
        del stochastic_state
        payload = to_dict(action_bookkeeping)
        payload["gate"] = {
            str(action): float(value)
            for action, value in dict(action_truth.get("gate", {}) or {}).items()
        }
        payload["risk_suppressor"] = {
            str(action): float(value)
            for action, value in dict(control_ledger.get("risk_suppressor", {}) or {}).items()
        }
        payload["ci"] = {
            str(action): float(value)
            for action, value in dict(control_ledger.get("ci", {}) or {}).items()
        }
        payload["resample_idx"] = int(control_ledger.get("resample_idx", 0) or 0)
        return payload

    def _build_distribution_delta_contribution(
        self,
        *,
        module_name: str,
        module_type: str,
        from_distribution: dict[str, float],
        to_distribution: dict[str, float],
        trace_reason: str,
        projection_reason: str,
        applied_at_stage: str,
        native_operator: str,
        dependency_trace: list[str] | None = None,
        hard_mask: dict[str, bool] | None = None,
        confidence: float = 1.0,
        module_temperature: float = 1.0,
    ) -> ProbabilisticContribution | None:
        modulated_delta: dict[str, float] = {}
        inhibitory_drive: dict[str, float] = {}
        all_actions = sorted(
            {*from_distribution.keys(), *to_distribution.keys(), *(hard_mask or {}).keys()}
        )
        for action in all_actions:
            if hard_mask and hard_mask.get(action):
                inhibitory_drive[str(action)] = 1.0
                continue
            base_probability = max(float(from_distribution.get(action, 0.0) or 0.0), 1e-9)
            target_probability = max(float(to_distribution.get(action, 0.0) or 0.0), 1e-9)
            shift = round(math.log(target_probability) - math.log(base_probability), 6)
            if abs(shift) >= 1e-9:
                modulated_delta[action] = shift
        effective_hard_mask = {
            str(action): bool(flag)
            for action, flag in dict(hard_mask or {}).items()
            if flag
        }
        if not modulated_delta and not inhibitory_drive and not effective_hard_mask:
            return None
        return ProbabilisticContribution(
            module_name=module_name,
            module_type=module_type,
            level="action",
            target_space="action",
            raw_signal=dict(modulated_delta),
            modulated_delta=modulated_delta,
            inhibitory_drive=inhibitory_drive,
            hard_mask=effective_hard_mask,
            confidence=confidence,
            confidence_calibrated=round(_clip(confidence), 4),
            trace_reason=trace_reason,
            projection_reason=projection_reason,
            applied_at_stage=applied_at_stage,
            native_operator=native_operator,
            dependency_trace=list(dependency_trace or []),
            projection=EnergyProjectionSpec(
                module_type=module_type,
                target_space="action",
                module_temperature=module_temperature,
            ),
        )

    def _expressive_action_biases(
        self,
        payload: dict[str, Any],
        *,
        scenario: str,
        endogenous_turn: bool,
    ) -> dict[str, float]:
        del endogenous_turn
        delta = {action: 0.0 for action in INTERNAL_RUNTIME_ACTIONS}
        metrics = dict(payload.get("metrics", {}) or {})
        top_intent = str(payload.get("top_intent") or "stay_silent")
        expression_mode = str(payload.get("expression_mode") or "silent")
        relation_strength = float(metrics.get("relation_strength", 0.0) or 0.0)
        grounding_score = float(payload.get("grounding_score", 0.0) or 0.0)
        speech_cost = float(payload.get("speech_cost", 0.0) or 0.0)
        intrinsic_value = float(payload.get("intrinsic_value", 0.0) or 0.0)
        external_drive = max(0.0, intrinsic_value - speech_cost)

        if top_intent == "share_memory":
            delta["recall"] += 0.06 + external_drive * 0.14
            delta["connect"] += 0.02 + relation_strength * 0.04
        elif top_intent == "check_relation":
            delta["connect"] += 0.04 + relation_strength * 0.08
            delta["clarify"] += 0.02 + external_drive * 0.08
        elif top_intent == "express_state":
            delta["respond"] += 0.04 + external_drive * 0.12
            delta["connect"] += 0.02 + relation_strength * 0.05
        elif top_intent == "follow_up_task":
            delta["plan"] += 0.05 + external_drive * 0.14
            delta["clarify"] += 0.02 + external_drive * 0.06
        else:
            delta["nothing"] += 0.04 + speech_cost * 0.06

        if expression_mode == "external":
            delta["respond"] += 0.04 + external_drive * 0.10
        else:
            delta["nothing"] += 0.03 + speech_cost * 0.08
            delta["respond"] -= min(0.04, speech_cost * 0.04)
            delta["connect"] -= min(0.03, speech_cost * 0.03)

        if scenario == "task":
            delta["plan"] += 0.01
        if grounding_score < 0.25:
            delta["respond"] -= 0.01
            delta["connect"] -= 0.01

        return {
            action: round(value, 6)
            for action, value in delta.items()
            if abs(float(value)) > 0.0
        }

    def _build_expressive_action_contribution(
        self,
        *,
        current_distribution: dict[str, float],
        payload: dict[str, Any],
        scenario: str,
        endogenous_turn: bool,
    ) -> ProbabilisticContribution | None:
        modulated_delta = self._expressive_action_biases(
            payload,
            scenario=scenario,
            endogenous_turn=endogenous_turn,
        )
        if not modulated_delta:
            return None
        adjusted_distribution = dict(current_distribution or {})
        for action_name, delta in modulated_delta.items():
            adjusted_distribution[action_name] = max(
                0.0,
                float(adjusted_distribution.get(action_name, 0.0) or 0.0) + float(delta),
            )
        return self._build_distribution_delta_contribution(
            module_name="ExpressiveImpulse",
            module_type="expression",
            from_distribution=dict(current_distribution or {}),
            to_distribution=self._normalize(adjusted_distribution),
            trace_reason=(
                f"expression_mode={payload.get('expression_mode', 'silent')} "
                f"intent={payload.get('top_intent', 'stay_silent')} "
                f"speech_cost={round(float(payload.get('speech_cost', 0.0) or 0.0), 4)}"
            ),
            projection_reason="expressive impulse projected into unified action field",
            applied_at_stage="expressive_field",
            native_operator="expressive_reweight",
            dependency_trace=[
                f"proposal_type:{payload.get('proposal_type', 'speak')}",
                f"expression_mode:{payload.get('expression_mode', 'silent')}",
                f"grounding_score:{round(float(payload.get('grounding_score', 0.0) or 0.0), 4)}",
                f"endogenous:{str(bool(endogenous_turn)).lower()}",
            ],
            confidence=max(0.35, min(1.0, float(payload.get("intrinsic_value", 0.0) or 0.0) + 0.2)),
        )

    def build_round_context(self, event: RoundEvent, scenario: str, mode: str) -> RoundContext:
        state = self.load_runtime_state()
        self._normalize_temperament_runtime_state(state)
        prior_state = RuntimeState(**to_dict(state))
        latest_round_views = self.trace_store.recent_round_signal_views(limit=1)
        latest_round_recorded_at = latest_round_views[-1].get("recorded_at") if latest_round_views else None
        requested_mode = "safe" if state.safe_mode else mode
        endogenous_turn = event.source == "endogenous" or requested_mode.startswith("endogenous")
        mode_cfg = self.config["modes"]["modes"].get(requested_mode, self.config["modes"]["modes"]["interactive"])
        scenario_cfg = self.config["scenarios"]["scenarios"][scenario]
        thresholds = self.config["thresholds"]["thresholds"]
        prior_closeness = self.memory_store.closeness(event.target)
        appraisal = self._infer_appraisal(
            event,
            state,
            scenario=scenario,
            requested_mode=requested_mode,
            closeness=prior_closeness,
        )
        event = self._event_with_appraisal(event, appraisal)

        state.mode = requested_mode
        state.mode_history = (state.mode_history + [requested_mode])[-20:]
        state.round_count += 1
        state.body_energy = _clip(state.body_energy + event.energy_delta)
        self._update_affect_residue(state, event, requested_mode)
        state.budget_remaining = _clip(state.budget_remaining - 0.001 + (0.01 if requested_mode in {"idle", "sleep"} else 0.0))
        recorded_at = utc_now_iso()
        recorded_date = iso_date(recorded_at)
        resource_telemetry = self._compute_resource_telemetry(state)
        state.resource_state = {**state.resource_state, **resource_telemetry}

        cue = self.memory_store.ingest_event(
            event,
            round_id=state.round_count,
            session_id=state.session_id,
            recorded_at=recorded_at,
            update_habit=False,
            cue_quality=event.cue_quality,
            resource_pressure=float(resource_telemetry.get("scarcity_pressure", 0.0) or 0.0),
        )
        memory_write_gate = self.memory_store.last_ingest_diagnostics()
        memory_retrieval_budget = self._memory_retrieval_budget(event, scenario_cfg)
        context = {
            "cue": cue,
            "memory_retrieval_budget": memory_retrieval_budget,
            "recall_strength": self.memory_store.recall_strength(cue, tier_budget=memory_retrieval_budget),
            "habit_strength": self.memory_store.habit_strength(cue),
            "closeness": self.memory_store.closeness(event.target),
            "valence": event.valence,
            "appraisal": appraisal,
            "detail_threshold": thresholds.get("detail_threshold", 0.5),
            "round_gap": 0,
            "interference": 0.0,
            "recent_burn_rate": 1.0 - state.budget_remaining,
            "burn_rate_ratio": resource_telemetry["burn_rate_ratio"],
            "low_balance_ratio": resource_telemetry["low_balance_ratio"],
            "queue_pressure": resource_telemetry["queue_pressure"],
            "latency_pressure": resource_telemetry["latency_pressure"],
            "memory_write_gate": memory_write_gate,
        }
        if cue:
            recall_payload = self.memory_store.recall(cue, tier_budget=memory_retrieval_budget)
            context["interference"] = recall_payload.get("interference", 0.0)
            context["episode_id"] = recall_payload.get("episode_id", "")
            context["separation_id"] = recall_payload.get("separation_id", "")
            context["memory_prior_vector"] = dict(recall_payload.get("prior_vector", {}) or {})
            context["recall_payload"] = dict(recall_payload)
        relation_state = self._relation_state(event, context)
        context["chronic_signal"] = self._build_chronic_signal(
            event,
            state=state,
            context=context,
            relation_state=relation_state,
            scenario=scenario,
        )
        shaping_events, dream_payload = self._apply_noninteractive_shaping(state, requested_mode, cue, relation_state)
        rename_event = self._maybe_update_identity_from_evidence(state)
        identity_evidence = self._augment_identity_evidence(state, self.memory_store.identity_evidence())
        self._update_personality_anchor(state, identity_evidence)

        gate_decisions: list[dict[str, Any]] = [
            {
                "stage": "memory_write_gate",
                "owner": "MemoryWriteGate",
                "allowed": not bool(memory_write_gate.get("suppressed", False)),
                "reason": memory_write_gate.get("reason", "unknown"),
                "cue": memory_write_gate.get("cue"),
            }
        ]
        skill_traces: list[dict[str, Any]] = []
        model_call_traces: list[dict[str, Any]] = []
        parallel_traces: list[dict[str, Any]] = []
        previous_focus = state.focus
        round_seed = self._round_seed(state, event)
        reasoning_state, reasoning_meta = self._state_for_reasoning(state, event, scenario, requested_mode)
        context = {**context, **reasoning_meta}
        runtime_context = self._skill_runtime_context(state.round_count, scenario, state)
        slow_variables = self._build_slow_variable_payload(
            state=state,
            context=context,
            relation_state=relation_state,
            prior_closeness=prior_closeness,
        )
        return cast(RoundContext, {
            "event": event,
            "state": state,
            "prior_state": prior_state,
            "latest_round_recorded_at": latest_round_recorded_at,
            "requested_mode": requested_mode,
            "endogenous_turn": endogenous_turn,
            "mode_cfg": mode_cfg,
            "scenario_cfg": scenario_cfg,
            "thresholds": thresholds,
            "prior_closeness": prior_closeness,
            "appraisal": appraisal,
            "recorded_at": recorded_at,
            "recorded_date": recorded_date,
            "resource_telemetry": resource_telemetry,
            "cue": cue,
            "memory_write_gate": memory_write_gate,
            "context": context,
            "relation_state": relation_state,
            "shaping_events": shaping_events,
            "dream_payload": dream_payload,
            "rename_event": rename_event,
            "identity_evidence": identity_evidence,
            "gate_decisions": gate_decisions,
            "skill_traces": skill_traces,
            "model_call_traces": model_call_traces,
            "parallel_traces": parallel_traces,
            "previous_focus": previous_focus,
            "round_seed": round_seed,
            "reasoning_state": reasoning_state,
            "runtime_context": runtime_context,
            "slow_variables": slow_variables,
        })

    def collect_parallel_contributions(
        self,
        *,
        round_context: RoundContext,
        scenario: str,
    ) -> CollectedContributions:
        event = round_context["event"]
        state = round_context["state"]
        requested_mode = round_context["requested_mode"]
        mode_cfg = round_context["mode_cfg"]
        scenario_cfg = round_context["scenario_cfg"]
        context = round_context["context"]
        relation_state = round_context["relation_state"]
        shaping_events = round_context["shaping_events"]
        rename_event = round_context["rename_event"]
        skill_traces = round_context["skill_traces"]
        model_call_traces = round_context["model_call_traces"]
        parallel_traces = round_context["parallel_traces"]
        round_seed = round_context["round_seed"]
        reasoning_state = round_context["reasoning_state"]
        runtime_context = round_context["runtime_context"]
        slow_variables = round_context["slow_variables"]
        pfc_fallback_only = self._endogenous_route_type(requested_mode) == "endogenous_light"

        intent_prefetch = self._execute_parallel_skills(
            round_id=state.round_count,
            tasks=[
                {
                    "name": "query_state",
                    "inputs": {},
                    "provider": lambda: self._infer_query_intent(
                        event=event,
                        scenario=scenario,
                        state=state,
                        relation_state=relation_state,
                        slow_variables=slow_variables,
                    ),
                    "parallel_group": "intent_prefetch",
                    "priority": "required",
                    "task_type": "callable",
                    "binding_tier": "state_machine",
                },
                {
                    "name": "grounding_capsule",
                    "inputs": {},
                    "provider": lambda: self._build_grounding_capsule(
                        state=state,
                        context=context,
                        relation_state=relation_state,
                        slow_variables=slow_variables,
                        shaping_events=shaping_events,
                        rename_event=rename_event,
                    ),
                    "parallel_group": "intent_prefetch",
                    "priority": "speculative",
                    "task_type": "callable",
                    "binding_tier": "state_machine",
                },
            ],
            skill_traces=skill_traces,
            runtime_context=runtime_context,
            parallel_traces=parallel_traces,
        )
        query_state = intent_prefetch.get("query_state")
        if not isinstance(query_state, QueryIntentState):
            query_state = self._infer_query_intent(
                event=event,
                scenario=scenario,
                state=state,
                relation_state=relation_state,
                slow_variables=slow_variables,
            )
        disclosure_state = self._infer_disclosure_intent(
            query_state=query_state,
            scenario=scenario,
            state=state,
            relation_state=relation_state,
            slow_variables=slow_variables,
        )
        context["relation_risk"] = float(relation_state.get("relationship_risk", 0.0) or 0.0)
        context["disclosure_sensitivity"] = float(
            disclosure_state.posterior.posterior.get("withhold", 0.0)
            if hasattr(disclosure_state.posterior, "posterior")
            else 0.0
        )
        context["authenticity_risk"] = round(
            max(
                float(state.affect_residue or 0.0),
                float(relation_state.get("relationship_risk", 0.0) or 0.0) * 0.5,
            ),
            6,
        )
        grounding_capsule = intent_prefetch.get("grounding_capsule")
        if not isinstance(grounding_capsule, dict):
            grounding_capsule = self._build_grounding_capsule(
                state=state,
                context=context,
                relation_state=relation_state,
                slow_variables=slow_variables,
                shaping_events=shaping_events,
                rename_event=rename_event,
            )
        context["grounding_capsule"] = grounding_capsule
        runtime_inputs = self._runtime_skill_inputs(event, reasoning_state, scenario_cfg, context)
        action_head_state = RuntimeState(**to_dict(reasoning_state))
        action_head_context = dict(context)
        action_head_inputs = self._runtime_skill_inputs(event, action_head_state, scenario_cfg, action_head_context)
        prefetch_route_type = (
            "task_run"
            if scenario == "task"
            else self._route_type_for_turn_plan(
                route="direct_chat",
                text=event.content,
                scenario=scenario,
                mode=requested_mode,
                probe={"task_mass": 0.0, "top_action": str(query_state.posterior.top_intent or "")},
            )
        )
        value_prefetch_timeout_ms = 1200 if prefetch_route_type == "chat_standard" else 0
        prefetched_outputs = self._execute_parallel_skills(
            round_id=state.round_count,
            tasks=[
                {
                    "name": "salience_contribution",
                    "skill_name": "score_salience",
                    "inputs": runtime_inputs,
                    "provider": lambda event, state, scenario, context: self.agent_map["SalienceAgent"].build_direct_action_contribution(
                        event,
                        state,
                        scenario,
                        context,
                    ),
                    "fallback_value": ProbabilisticContribution(
                        module_name="SalienceAgent",
                        module_type="salience",
                        level="action",
                        target_space="action",
                        raw_signal={"respond": 0.02},
                        modulated_delta={"respond": 0.02},
                        confidence=0.1,
                        trace_reason="typed fallback",
                        projection_reason="typed fallback",
                    ),
                    "parallel_group": "salience_value_prefetch",
                    "binding_tier": self._pipeline_tier("SalienceAgent"),
                },
                {
                    "name": "value_scores",
                    "skill_name": "estimate_subjective_value",
                    "inputs": runtime_inputs,
                    "provider": lambda event, state, scenario, context: self._estimate_subjective_value_via_model(
                        event,
                        state,
                        scenario,
                        context,
                        model_call_traces=model_call_traces,
                        parallel_group="salience_value_prefetch",
                    ),
                    "fallback_value": {"scores": {}},
                    "parallel_group": "salience_value_prefetch",
                    "binding_tier": self._pipeline_tier("ValueAgent"),
                    "priority": "optional" if value_prefetch_timeout_ms > 0 else "required",
                    "timeout_ms": value_prefetch_timeout_ms,
                },
            ],
            skill_traces=skill_traces,
            runtime_context=runtime_context,
            parallel_traces=parallel_traces,
        )
        direct_action_contributions: dict[str, ProbabilisticContribution] = {}
        direct_action_signal_metadata: dict[str, dict[str, Any]] = {}

        for stage_name, owner in ACTION_HEAD_STAGE_ORDER:
            agent = self.agent_map[owner]
            if not state.agents_enabled.get(owner, True):
                continue
            if owner == "BodyStateAgent":
                patch = self._execute_skill(round_id=state.round_count, skill_name="update_body_state", inputs=runtime_inputs, provider=self._agent_provider(agent, "update_body_state"), skill_traces=skill_traces, runtime_context=runtime_context, seed_ref=round_seed)
                self._apply_state_patch(state, patch.get("state_patch", {}))
                contribution = self._execute_skill(
                    round_id=state.round_count,
                    skill_name="compute_body_bias",
                    inputs=action_head_inputs,
                    provider=lambda event, state, scenario, context: agent.build_direct_action_contribution(event, state, scenario, context),
                    skill_traces=skill_traces,
                    runtime_context=runtime_context,
                    fallback_value=ProbabilisticContribution(
                        module_name=owner,
                        module_type="body",
                        level="action",
                        target_space="action",
                        raw_signal={"respond": 0.01},
                        modulated_delta={"respond": 0.01},
                        confidence=0.1,
                        trace_reason="typed fallback",
                        projection_reason="typed fallback",
                    ),
                    seed_ref=round_seed,
                )
                veto_info = self._execute_skill(round_id=state.round_count, skill_name="apply_body_veto", inputs=runtime_inputs, provider=self._agent_provider(agent, "apply_body_veto"), skill_traces=skill_traces, runtime_context=runtime_context, seed_ref=round_seed)
                contribution.hard_mask = {
                    **dict(contribution.hard_mask or {}),
                    **{action: True for action in list(veto_info.get("gated_actions", []))},
                }
                direct_action_contributions[owner] = contribution
                direct_action_signal_metadata[owner] = self._action_signal_metadata(
                    owner=owner,
                    state=state,
                    relation_state=relation_state,
                    context=context,
                    module_type=contribution.module_type,
                    projected_delta=self._projected_action_delta_from_contribution(contribution),
                    utility_shift=self._projected_action_delta_from_contribution(contribution),
                    gated_actions=list(veto_info.get("gated_actions", [])),
                    veto=bool(veto_info.get("veto", False)),
                )
                continue
            if owner == "EmotionAgent":
                patch = self._execute_skill(round_id=state.round_count, skill_name="update_affect_state", inputs=runtime_inputs, provider=self._agent_provider(agent, "update_affect_state"), skill_traces=skill_traces, runtime_context=runtime_context, seed_ref=round_seed)
                self._apply_state_patch(state, patch.get("state_patch", {}))
                contribution = self._execute_skill(
                    round_id=state.round_count,
                    skill_name="compute_affect_bias",
                    inputs=action_head_inputs,
                    provider=lambda event, state, scenario, context: agent.build_direct_action_contribution(event, state, scenario, context),
                    skill_traces=skill_traces,
                    runtime_context=runtime_context,
                    fallback_value=ProbabilisticContribution(
                        module_name=owner,
                        module_type="emotion",
                        level="action",
                        target_space="action",
                        raw_signal={"respond": 0.01},
                        modulated_delta={"respond": 0.01},
                        confidence=0.1,
                        trace_reason="typed fallback",
                        projection_reason="typed fallback",
                    ),
                    seed_ref=round_seed,
                )
                veto_info = self._execute_skill(round_id=state.round_count, skill_name="trigger_affect_veto", inputs=runtime_inputs, provider=self._agent_provider(agent, "trigger_affect_veto"), skill_traces=skill_traces, runtime_context=runtime_context, seed_ref=round_seed)
                direct_action_contributions[owner] = contribution
                direct_action_signal_metadata[owner] = self._action_signal_metadata(
                    owner=owner,
                    state=state,
                    relation_state=relation_state,
                    context=context,
                    module_type=contribution.module_type,
                    projected_delta=self._projected_action_delta_from_contribution(contribution),
                    utility_shift=self._projected_action_delta_from_contribution(contribution),
                    veto=bool(veto_info.get("veto", False)),
                )
                continue
            if owner == "RelationshipAgent":
                closeness = self._execute_skill(round_id=state.round_count, skill_name="score_closeness", inputs=runtime_inputs, provider=self._agent_provider(agent, "score_closeness"), skill_traces=skill_traces, runtime_context=runtime_context, seed_ref=round_seed)
                context["closeness"] = closeness.get("score", context["closeness"])
                gate_info = self._execute_skill(round_id=state.round_count, skill_name="compute_boundary_gate", inputs=runtime_inputs, provider=self._agent_provider(agent, "compute_boundary_gate"), skill_traces=skill_traces, runtime_context=runtime_context, seed_ref=round_seed)
                relation_state["boundary_level"] = gate_info.get("boundary_level", relation_state["boundary_level"])
                self._execute_skill(round_id=state.round_count, skill_name="update_relation_trace", inputs=runtime_inputs, provider=self._agent_provider(agent, "update_relation_trace"), skill_traces=skill_traces, runtime_context=runtime_context, seed_ref=round_seed)
                contribution = agent.build_direct_action_contribution(event, action_head_state, scenario_cfg, action_head_context)
                direct_action_contributions[owner] = contribution
                direct_action_signal_metadata[owner] = self._action_signal_metadata(
                    owner=owner,
                    state=state,
                    relation_state=relation_state,
                    context=context,
                    module_type=contribution.module_type,
                    projected_delta=self._projected_action_delta_from_contribution(contribution),
                    utility_shift=self._projected_action_delta_from_contribution(contribution),
                )
                continue
            if owner == "ResourceAgent":
                scarcity = self._execute_skill(round_id=state.round_count, skill_name="compute_scarcity_index", inputs=runtime_inputs, provider=self._agent_provider(agent, "compute_scarcity_index"), skill_traces=skill_traces, runtime_context=runtime_context, seed_ref=round_seed)
                state.resource_state = {**state.resource_state, "scarcity_index": round(float(scarcity.get("scalar", 0.0)), 4)}
                state.resource_state = {**state.resource_state, **self._resource_bias_snapshot(state, context)}
                contribution = self._execute_skill(
                    round_id=state.round_count,
                    skill_name="map_budget_to_bias",
                    inputs=action_head_inputs,
                    provider=lambda event, state, scenario, context: agent.build_direct_action_contribution(event, state, scenario, context),
                    skill_traces=skill_traces,
                    runtime_context=runtime_context,
                    fallback_value=ProbabilisticContribution(
                        module_name=owner,
                        module_type="resource",
                        level="action",
                        target_space="action",
                        raw_signal={"respond": 0.01},
                        modulated_delta={"respond": 0.01},
                        confidence=0.1,
                        trace_reason="typed fallback",
                        projection_reason="typed fallback",
                    ),
                    seed_ref=round_seed,
                )
                direct_action_contributions[owner] = contribution
                mode_hint = self._execute_skill(round_id=state.round_count, skill_name="suggest_resource_mode", inputs=runtime_inputs, provider=self._agent_provider(agent, "suggest_resource_mode"), skill_traces=skill_traces, runtime_context=runtime_context, seed_ref=round_seed)
                if mode_hint.get("resource_mode"):
                    state.resource_state = {**state.resource_state, "resource_mode": mode_hint["resource_mode"]}
                if mode_hint.get("mode_flag") == "safe":
                    state.safe_mode = True
                direct_action_signal_metadata[owner] = self._action_signal_metadata(
                    owner=owner,
                    state=state,
                    relation_state=relation_state,
                    context=context,
                    module_type=contribution.module_type,
                    projected_delta=self._projected_action_delta_from_contribution(contribution),
                    utility_shift=self._projected_action_delta_from_contribution(contribution),
                )
                continue
            if owner == "PFCAgent":
                if pfc_fallback_only:
                    contribution = agent.build_direct_action_contribution(
                        action_head_inputs["event"],
                        action_head_inputs["state"],
                        action_head_inputs["scenario"],
                        action_head_inputs["context"],
                    )
                else:
                    contribution = self._execute_skill(
                        round_id=state.round_count,
                        skill_name="generate_candidates",
                        inputs=action_head_inputs,
                        provider=lambda event, state, scenario, context: self._invoke_pfc_model_generator(
                            event,
                            state,
                            scenario,
                            context,
                            model_call_traces=model_call_traces,
                        ),
                        skill_traces=skill_traces,
                        runtime_context=runtime_context,
                        fallback_provider=lambda **skill_inputs: agent.build_direct_action_contribution(
                            skill_inputs["event"],
                            skill_inputs["state"],
                            skill_inputs["scenario"],
                            skill_inputs["context"],
                        ),
                        seed_ref=round_seed,
                    )
                self._execute_skill(round_id=state.round_count, skill_name="estimate_plan_depth", inputs=runtime_inputs, provider=self._agent_provider(agent, "estimate_plan_depth"), skill_traces=skill_traces, runtime_context=runtime_context, seed_ref=round_seed)
                self._execute_skill(round_id=state.round_count, skill_name="bind_working_memory", inputs=runtime_inputs, provider=self._agent_provider(agent, "bind_working_memory"), skill_traces=skill_traces, runtime_context=runtime_context, seed_ref=round_seed)
                direct_action_contributions[owner] = contribution
                direct_action_signal_metadata[owner] = self._action_signal_metadata(
                    owner=owner,
                    state=state,
                    relation_state=relation_state,
                    context=context,
                    module_type=contribution.module_type,
                    projected_delta=self._projected_action_delta_from_contribution(contribution),
                    utility_shift=self._projected_action_delta_from_contribution(contribution),
                )
                continue
            if owner == "HabitAgent":
                self._execute_skill(round_id=state.round_count, skill_name="compute_feedback_decay", inputs=runtime_inputs, provider=self._agent_provider(agent, "compute_feedback_decay"), skill_traces=skill_traces, runtime_context=runtime_context, seed_ref=round_seed)
                self._execute_skill(round_id=state.round_count, skill_name="update_habit_strength", inputs=runtime_inputs, provider=self._agent_provider(agent, "update_habit_strength"), skill_traces=skill_traces, runtime_context=runtime_context, seed_ref=round_seed)
                updated_habit = self.memory_store.update_habit_strength(
                    context.get("cue"),
                    event.valence,
                    context_recurrence=min(1.0, context.get("recall_strength", 0.0) + 0.20),
                    context_slot=f"{event.source.lower()}::{(event.target or 'none').lower()}",
                    round_id=state.round_count,
                )
                if updated_habit is not None:
                    context["habit_strength"] = float(updated_habit.get("strength", context.get("habit_strength", 0.0)))
                habit_contribution = agent.build_direct_action_contribution(event, action_head_state, scenario_cfg, action_head_context)
                direct_action_contributions[owner] = habit_contribution
                self._execute_skill(round_id=state.round_count, skill_name="estimate_override_cost", inputs=runtime_inputs, provider=self._agent_provider(agent, "estimate_override_cost"), skill_traces=skill_traces, runtime_context=runtime_context, seed_ref=round_seed)
                direct_action_signal_metadata[owner] = self._action_signal_metadata(
                    owner=owner,
                    state=state,
                    relation_state=relation_state,
                    context=context,
                    module_type=habit_contribution.module_type,
                    projected_delta=self._projected_action_delta_from_contribution(habit_contribution),
                    utility_shift=self._projected_action_delta_from_contribution(habit_contribution),
                )
                continue
            if owner == "DesireAgent":
                self._execute_skill(round_id=state.round_count, skill_name="score_immediate_reward", inputs=runtime_inputs, provider=self._agent_provider(agent, "score_immediate_reward"), skill_traces=skill_traces, runtime_context=runtime_context, seed_ref=round_seed)
                desire_contribution = agent.build_direct_action_contribution(
                    event,
                    action_head_state,
                    scenario_cfg,
                    action_head_context,
                )
                direct_action_contributions[owner] = desire_contribution
                self._execute_skill(round_id=state.round_count, skill_name="suggest_low_cost_action", inputs=runtime_inputs, provider=self._agent_provider(agent, "suggest_low_cost_action"), skill_traces=skill_traces, runtime_context=runtime_context, seed_ref=round_seed)
                direct_action_signal_metadata[owner] = self._action_signal_metadata(
                    owner=owner,
                    state=state,
                    relation_state=relation_state,
                    context=context,
                    module_type=desire_contribution.module_type,
                    projected_delta=self._projected_action_delta_from_contribution(desire_contribution),
                    utility_shift=self._projected_action_delta_from_contribution(desire_contribution),
                )
                continue
            if owner == "DMNAgent":
                dmn_contribution = agent.build_direct_action_contribution(
                    event,
                    action_head_state,
                    scenario_cfg,
                    action_head_context,
                )
                direct_action_contributions[owner] = dmn_contribution
                self._execute_skill(round_id=state.round_count, skill_name="score_rumination_pull", inputs=runtime_inputs, provider=self._agent_provider(agent, "score_rumination_pull"), skill_traces=skill_traces, runtime_context=runtime_context, seed_ref=round_seed)
                self._execute_skill(round_id=state.round_count, skill_name="select_spontaneous_topic", inputs=runtime_inputs, provider=self._agent_provider(agent, "select_spontaneous_topic"), skill_traces=skill_traces, runtime_context=runtime_context, seed_ref=round_seed)
                direct_action_signal_metadata[owner] = self._action_signal_metadata(
                    owner=owner,
                    state=state,
                    relation_state=relation_state,
                    context=context,
                    module_type=dmn_contribution.module_type,
                    projected_delta=self._projected_action_delta_from_contribution(dmn_contribution),
                    utility_shift=self._projected_action_delta_from_contribution(dmn_contribution),
                )
                continue
            if owner == "HippocampusAgent":
                self._execute_skill(round_id=state.round_count, skill_name="encode_episode", inputs=runtime_inputs, provider=self._agent_provider(agent, "encode_episode"), skill_traces=skill_traces, runtime_context=runtime_context, seed_ref=round_seed)
                recall_set = self._execute_skill(round_id=state.round_count, skill_name="retrieve_by_cue", inputs=runtime_inputs, provider=self._agent_provider(agent, "retrieve_by_cue"), skill_traces=skill_traces, runtime_context=runtime_context, seed_ref=round_seed)
                self._execute_skill(round_id=state.round_count, skill_name="apply_cue_weighted_decay", inputs=runtime_inputs, provider=self._agent_provider(agent, "apply_cue_weighted_decay"), skill_traces=skill_traces, runtime_context=runtime_context, seed_ref=round_seed)
                self._execute_skill(round_id=state.round_count, skill_name="compute_memory_interference", inputs=runtime_inputs, provider=self._agent_provider(agent, "compute_memory_interference"), skill_traces=skill_traces, runtime_context=runtime_context, seed_ref=round_seed)
                if not recall_set.get("recall_set"):
                    self._execute_skill(round_id=state.round_count, skill_name="fallback_to_gist_when_trace_weak", inputs=runtime_inputs, provider=self._agent_provider(agent, "fallback_to_gist_when_trace_weak"), skill_traces=skill_traces, runtime_context=runtime_context, seed_ref=round_seed)
                contribution = agent.build_direct_action_contribution(event, action_head_state, scenario_cfg, action_head_context)
                direct_action_contributions[owner] = contribution
                direct_action_signal_metadata[owner] = self._action_signal_metadata(
                    owner=owner,
                    state=state,
                    relation_state=relation_state,
                    context=context,
                    module_type=contribution.module_type,
                    projected_delta=self._projected_action_delta_from_contribution(contribution),
                    utility_shift=self._projected_action_delta_from_contribution(contribution),
                )
                continue
            if owner == "PerspectiveModel":
                if state.resource_state.get("resource_mode") == "starvation":
                    continue
                perspective_contribution = agent.build_direct_action_contribution(
                    event,
                    action_head_state,
                    scenario_cfg,
                    action_head_context,
                )
                direct_action_contributions[owner] = perspective_contribution
                direct_action_signal_metadata[owner] = self._action_signal_metadata(
                    owner=owner,
                    state=state,
                    relation_state=relation_state,
                    context=context,
                    module_type=perspective_contribution.module_type,
                    projected_delta=self._projected_action_delta_from_contribution(perspective_contribution),
                    utility_shift=self._projected_action_delta_from_contribution(perspective_contribution),
                )
                continue
            if owner == "ValueAgent":
                if "value_scores" in prefetched_outputs:
                    value_scores = prefetched_outputs["value_scores"]
                else:
                    value_scores = self._execute_skill(
                        round_id=state.round_count,
                        skill_name="estimate_subjective_value",
                        inputs=runtime_inputs,
                        provider=lambda event, state, scenario, context: self._estimate_subjective_value_via_model(
                            event,
                            state,
                            scenario,
                            context,
                            model_call_traces=model_call_traces,
                        ),
                        skill_traces=skill_traces,
                        runtime_context=runtime_context,
                        seed_ref=round_seed,
                    )
                self._execute_skill(round_id=state.round_count, skill_name="discount_delayed_reward", inputs=runtime_inputs, provider=self._agent_provider(agent, "discount_delayed_reward"), skill_traces=skill_traces, runtime_context=runtime_context, seed_ref=round_seed)
                self._execute_skill(round_id=state.round_count, skill_name="price_social_cost", inputs=runtime_inputs, provider=self._agent_provider(agent, "price_social_cost"), skill_traces=skill_traces, runtime_context=runtime_context, seed_ref=round_seed)
                self._execute_skill(round_id=state.round_count, skill_name="score_uncertainty_penalty", inputs=runtime_inputs, provider=self._agent_provider(agent, "score_uncertainty_penalty"), skill_traces=skill_traces, runtime_context=runtime_context, seed_ref=round_seed)
                value_contribution = self._build_value_action_contribution(value_scores)
                direct_action_contributions[owner] = value_contribution
                direct_action_signal_metadata[owner] = self._action_signal_metadata(
                    owner=owner,
                    state=state,
                    relation_state=relation_state,
                    context=context,
                    module_type=value_contribution.module_type,
                    projected_delta=self._projected_action_delta_from_contribution(value_contribution),
                    utility_shift=self._projected_action_delta_from_contribution(value_contribution),
                )
                continue
            if owner == "SalienceAgent":
                if "salience_contribution" in prefetched_outputs:
                    contribution = prefetched_outputs["salience_contribution"]
                else:
                    contribution = self._execute_skill(
                        round_id=state.round_count,
                        skill_name="score_salience",
                        inputs=action_head_inputs,
                        provider=lambda event, state, scenario, context: agent.build_direct_action_contribution(event, state, scenario, context),
                        skill_traces=skill_traces,
                        runtime_context=runtime_context,
                        fallback_value=ProbabilisticContribution(
                            module_name=owner,
                            module_type="salience",
                            level="action",
                            target_space="action",
                            raw_signal={"respond": 0.02},
                            modulated_delta={"respond": 0.02},
                            confidence=0.1,
                            trace_reason="typed fallback",
                            projection_reason="typed fallback",
                        ),
                        seed_ref=round_seed,
                    )
                direct_action_contributions[owner] = contribution
                self._execute_skill(round_id=state.round_count, skill_name="switch_mode", inputs=runtime_inputs, provider=self._agent_provider(agent, "switch_mode"), skill_traces=skill_traces, runtime_context=runtime_context, seed_ref=round_seed)
                self._execute_skill(round_id=state.round_count, skill_name="interrupt_current_focus", inputs=runtime_inputs, provider=self._agent_provider(agent, "interrupt_current_focus"), skill_traces=skill_traces, runtime_context=runtime_context, seed_ref=round_seed)
                self._execute_skill(round_id=state.round_count, skill_name="promote_event_to_workspace", inputs=runtime_inputs, provider=self._agent_provider(agent, "promote_event_to_workspace"), skill_traces=skill_traces, runtime_context=runtime_context, seed_ref=round_seed)
                direct_action_signal_metadata[owner] = self._action_signal_metadata(
                    owner=owner,
                    state=state,
                    relation_state=relation_state,
                    context=context,
                    module_type=contribution.module_type,
                    projected_delta=self._projected_action_delta_from_contribution(contribution),
                    utility_shift=self._projected_action_delta_from_contribution(contribution),
                )
                continue
            if owner == "UnconsciousAgent":
                baseline = self._execute_skill(round_id=state.round_count, skill_name="load_temperament", inputs=runtime_inputs, provider=self._agent_provider(agent, "load_temperament"), skill_traces=skill_traces, runtime_context=runtime_context, seed_ref=round_seed)
                self._normalize_temperament_runtime_state(state, baseline.get("baseline", {}))
                contribution = self._execute_skill(
                    round_id=state.round_count,
                    skill_name="compute_trait_bias",
                    inputs=action_head_inputs,
                    provider=lambda event, state, scenario, context: agent.build_direct_action_contribution(event, state, scenario, context),
                    skill_traces=skill_traces,
                    runtime_context=runtime_context,
                    fallback_value=ProbabilisticContribution(
                        module_name=owner,
                        module_type="trait",
                        level="action",
                        target_space="action",
                        raw_signal={"respond": 0.01},
                        modulated_delta={"respond": 0.01},
                        confidence=0.1,
                        trace_reason="typed fallback",
                        projection_reason="typed fallback",
                    ),
                    seed_ref=round_seed,
                )
                patch = self._execute_skill(round_id=state.round_count, skill_name="apply_chronic_shift", inputs=runtime_inputs, provider=self._agent_provider(agent, "apply_chronic_shift"), skill_traces=skill_traces, runtime_context=runtime_context, seed_ref=round_seed)
                self._apply_state_patch(state, patch.get("state_patch", {}))
                self._normalize_temperament_runtime_state(state)
                direct_action_contributions[owner] = contribution
                direct_action_signal_metadata[owner] = self._action_signal_metadata(
                    owner=owner,
                    state=state,
                    relation_state=relation_state,
                    context=context,
                    module_type=contribution.module_type,
                    projected_delta=self._projected_action_delta_from_contribution(contribution),
                    utility_shift=self._projected_action_delta_from_contribution(contribution),
                )
                continue
            if owner == "CerebellarPredictor":
                self._execute_skill(round_id=state.round_count, skill_name="predict_next_state", inputs=runtime_inputs, provider=self._agent_provider(agent, "predict_next_state"), skill_traces=skill_traces, runtime_context=runtime_context, seed_ref=round_seed)
                self._execute_skill(round_id=state.round_count, skill_name="compute_prediction_error", inputs=runtime_inputs, provider=self._agent_provider(agent, "compute_prediction_error"), skill_traces=skill_traces, runtime_context=runtime_context, seed_ref=round_seed)
                self._execute_skill(round_id=state.round_count, skill_name="smooth_response_timing", inputs=runtime_inputs, provider=self._agent_provider(agent, "smooth_response_timing"), skill_traces=skill_traces, runtime_context=runtime_context, seed_ref=round_seed)
                contribution = self._execute_skill(
                    round_id=state.round_count,
                    skill_name="micro_adjust_action",
                    inputs=action_head_inputs,
                    provider=lambda event, state, scenario, context: agent.build_direct_action_contribution(event, state, scenario, context),
                    skill_traces=skill_traces,
                    runtime_context=runtime_context,
                    fallback_value=ProbabilisticContribution(
                        module_name=owner,
                        module_type="predictive",
                        level="action",
                        target_space="action",
                        raw_signal={},
                        modulated_delta={},
                        confidence=0.1,
                        trace_reason="typed fallback",
                        projection_reason="typed fallback",
                    ),
                    seed_ref=round_seed,
                )
                direct_action_contributions[owner] = contribution
                direct_action_signal_metadata[owner] = self._action_signal_metadata(
                    owner=owner,
                    state=state,
                    relation_state=relation_state,
                    context=context,
                    module_type=contribution.module_type,
                    projected_delta=self._projected_action_delta_from_contribution(contribution),
                    utility_shift=self._projected_action_delta_from_contribution(contribution),
                )

        online_long_run_projection = self.long_run_analyzer.build_online_projection(
            round_id=state.round_count,
            slow_variables=slow_variables,
            shaping_events=shaping_events,
        )
        motivation_pool_state = self.endogenous_motivation_pool.evaluate(
            state=state,
            context=context,
            relation_state=relation_state,
            slow_variables=slow_variables,
            long_run_projection=online_long_run_projection,
        )
        state.motivation_pool_state = motivation_pool_state
        motivation_contribution = self.endogenous_motivation_pool.build_action_contribution(
            state=state,
            pool_state=motivation_pool_state,
        )
        if motivation_contribution is not None:
            direct_action_contributions["EndogenousMotivationPool"] = motivation_contribution
            direct_action_signal_metadata["EndogenousMotivationPool"] = self._action_signal_metadata(
                owner="EndogenousMotivationPool",
                state=state,
                relation_state=relation_state,
                context=context,
                module_type=motivation_contribution.module_type,
                projected_delta=self._projected_action_delta_from_contribution(motivation_contribution),
                utility_shift=self._projected_action_delta_from_contribution(motivation_contribution),
                trace_tags=["motivation", *[item.motivation_type for item in motivation_pool_state.active_motivations]],
            )
        autonomy_self_run_contribution = self._build_autonomy_self_run_action_contribution(
            state=state,
            scenario=scenario,
            endogenous_turn=bool(round_context.get("endogenous_turn", False)),
        )
        if autonomy_self_run_contribution is not None:
            direct_action_contributions["AutonomySelfRun"] = autonomy_self_run_contribution
            direct_action_signal_metadata["AutonomySelfRun"] = self._action_signal_metadata(
                owner="AutonomySelfRun",
                state=state,
                relation_state=relation_state,
                context=context,
                module_type=autonomy_self_run_contribution.module_type,
                projected_delta=self._projected_action_delta_from_contribution(autonomy_self_run_contribution),
                utility_shift=self._projected_action_delta_from_contribution(autonomy_self_run_contribution),
                trace_tags=["autonomy", "self_run"],
            )
        monologue_stream_contribution, monologue_stream_trace = self._build_monologue_stream_action_contribution(
            state=state,
            scenario=scenario,
            endogenous_turn=bool(round_context.get("endogenous_turn", False)),
        )
        context["monologue_stream_trace"] = monologue_stream_trace
        if monologue_stream_contribution is not None:
            direct_action_contributions["MonologueStream"] = monologue_stream_contribution
            direct_action_signal_metadata["MonologueStream"] = self._action_signal_metadata(
                owner="MonologueStream",
                state=state,
                relation_state=relation_state,
                context=context,
                module_type=monologue_stream_contribution.module_type,
                projected_delta=self._projected_action_delta_from_contribution(monologue_stream_contribution),
                utility_shift=self._projected_action_delta_from_contribution(monologue_stream_contribution),
                trace_tags=["monologue", "hidden_stream"],
            )
        instinct_contribution = self._build_instinct_field_contribution(
            state=state,
            context=context,
            relation_state=relation_state,
            slow_variables=slow_variables,
        )
        if instinct_contribution is not None:
            direct_action_contributions["InstinctField"] = instinct_contribution
            direct_action_signal_metadata["InstinctField"] = self._action_signal_metadata(
                owner="InstinctField",
                state=state,
                relation_state=relation_state,
                context=context,
                module_type=instinct_contribution.module_type,
                projected_delta=self._projected_action_delta_from_contribution(instinct_contribution),
                utility_shift=self._projected_action_delta_from_contribution(instinct_contribution),
                trace_tags=["instinct", state.instinct_field.winner_region],
            )
        emergent_contribution = self._build_emergent_action_contribution(state)
        if emergent_contribution is not None:
            direct_action_contributions["EmergentActionSketch"] = emergent_contribution
            direct_action_signal_metadata["EmergentActionSketch"] = self._action_signal_metadata(
                owner="EmergentActionSketch",
                state=state,
                relation_state=relation_state,
                context=context,
                module_type=emergent_contribution.module_type,
                projected_delta=self._projected_action_delta_from_contribution(emergent_contribution),
                utility_shift=self._projected_action_delta_from_contribution(emergent_contribution),
                trace_tags=["emergent", *[sketch.name for sketch in state.emergent_action_sketches[:3]]],
            )

        action_signals: list[ActionEvidenceSignal] = []
        for owner, contribution in direct_action_contributions.items():
            action_signals.append(
                self._action_evidence_from_contribution(
                    contribution,
                    **direct_action_signal_metadata.get(owner, {}),
                )
            )
        action_bookkeeping = self._build_action_bookkeeping(
            action_signals,
            state,
            scenario_cfg,
            mode_cfg,
            relation_state,
            context,
            query_state=query_state,
            disclosure_state=disclosure_state,
        )
        identity_context = self._build_identity_context(
            query_state=query_state,
            disclosure_state=disclosure_state,
            scenario=scenario,
            state=state,
            shaping_events=shaping_events,
            slow_variables=slow_variables,
            state_sources=list(grounding_capsule.get("state_sources", [])),
            rename_event=rename_event,
        )
        return cast(CollectedContributions, {
            "query_state": query_state,
            "disclosure_state": disclosure_state,
            "runtime_inputs": runtime_inputs,
            "direct_action_contributions": direct_action_contributions,
            "action_signals": action_signals,
            "action_bookkeeping": action_bookkeeping,
            "identity_context": identity_context,
            "online_long_run_projection": online_long_run_projection,
            "motivation_pool_state": motivation_pool_state,
            "monologue_stream_trace": dict(context.get("monologue_stream_trace", {}) or {}),
        })

    def integrate_and_arbitrate(
        self,
        *,
        round_context: RoundContext,
        collected: CollectedContributions,
        scenario: str,
        turn_started: float,
    ) -> ArbitrationResult:
        event = round_context["event"]
        state = round_context["state"]
        prior_state = round_context["prior_state"]
        latest_round_recorded_at = round_context["latest_round_recorded_at"]
        requested_mode = round_context["requested_mode"]
        endogenous_turn = round_context["endogenous_turn"]
        scenario_cfg = round_context["scenario_cfg"]
        thresholds = round_context["thresholds"]
        prior_closeness = round_context["prior_closeness"]
        appraisal = round_context["appraisal"]
        recorded_at = round_context["recorded_at"]
        recorded_date = round_context["recorded_date"]
        memory_write_gate = round_context["memory_write_gate"]
        context = round_context["context"]
        relation_state = round_context["relation_state"]
        shaping_events = round_context["shaping_events"]
        dream_payload = round_context["dream_payload"]
        rename_event = round_context["rename_event"]
        identity_evidence = round_context["identity_evidence"]
        gate_decisions = round_context["gate_decisions"]
        skill_traces = round_context["skill_traces"]
        model_call_traces = round_context["model_call_traces"]
        parallel_traces = round_context["parallel_traces"]
        previous_focus = round_context["previous_focus"]
        round_seed = round_context["round_seed"]
        runtime_context = round_context["runtime_context"]

        query_state = collected["query_state"]
        disclosure_state = collected["disclosure_state"]
        runtime_inputs = collected["runtime_inputs"]
        direct_action_contributions = collected["direct_action_contributions"]
        action_signals = collected["action_signals"]
        action_bookkeeping = collected["action_bookkeeping"]
        identity_context = collected["identity_context"]
        online_long_run_projection = collected["online_long_run_projection"]
        motivation_pool_state = collected["motivation_pool_state"]
        monologue_stream_trace = dict(collected.get("monologue_stream_trace", {}) or {})

        online_long_run_contribution = self.long_run_analyzer.build_long_run_prior_contribution(
            online_long_run_projection
        )
        action_snapshot, action_contributions, token_state = self._integrate_probability_field_snapshot(
            direct_action_contributions=direct_action_contributions,
            action_base=dict(action_bookkeeping.u_base),
            event=event,
            state=state,
            scenario_cfg=scenario_cfg,
            context=context,
            identity_context=identity_context,
            long_run_contribution=online_long_run_contribution,
        )
        control_ledger = self._control_ledger_from_action_bookkeeping(action_bookkeeping)
        action_truth = self._refresh_action_truth(
            action_snapshot.action,
            {
                "gate": dict(control_ledger.get("gate", {}) or {}),
                "conflict_mode": str(control_ledger.get("conflict_mode", "none") or "none"),
            },
            conflict_mode=str(control_ledger.get("conflict_mode", "none") or "none"),
        )

        conflict_score, conflict_action_truth, control_ledger = self._run_conflict_controller(
            state=state,
            signals=action_signals,
            control_ledger=control_ledger,
            action_truth=action_truth,
            thresholds=thresholds,
            skill_traces=skill_traces,
            runtime_context=runtime_context,
            gate_decisions=gate_decisions,
            round_seed=round_seed,
        )
        action_truth = dict(conflict_action_truth)
        conflict_contribution = self.agent_map["ConflictMonitorAgent"].build_arbitration_contribution(
            dict(control_ledger.get("conflict", {}) or {}),
            conflict_action_truth,
        )
        if conflict_contribution.modulated_delta or conflict_contribution.inhibitory_drive or conflict_contribution.hard_mask:
            action_contributions.append(conflict_contribution)
            action_snapshot = self._reintegrate_probability_snapshot(
                snapshot=action_snapshot,
                contributions=action_contributions,
                token_state=token_state,
                source_chain=["action_field_after_conflict"],
            )
            action_truth = self._refresh_action_truth(action_snapshot.action, action_truth)

        deterministic = dict(action_snapshot.action.winner_posterior or {})

        stochastic_distribution, stochastic_state = self._apply_stochastic_layer(
            deterministic,
            action_bookkeeping,
            control_ledger,
            state,
            event,
            scenario_cfg,
            relation_state,
            conflict_score,
            round_seed,
            action_energy=dict(action_truth.get("final_energy", {}) or dict(action_snapshot.action.final_energy or {})),
        )
        stochastic_contribution = self._build_distribution_delta_contribution(
            module_name="StochasticPolicy",
            module_type="stochastic",
            from_distribution=deterministic,
            to_distribution=stochastic_distribution,
            trace_reason=(
                f"stochastic mixing lambda={stochastic_state.lambda_noise:.4f} "
                f"channel={stochastic_state.emo_channel}"
            ),
            projection_reason="stochastic action mixing projected from entropy-controlled noise",
            applied_at_stage="stochastic_mixing",
            native_operator="posterior_reweight",
            dependency_trace=[
                f"lambda_noise:{stochastic_state.lambda_noise:.4f}",
                f"emo_channel:{stochastic_state.emo_channel}",
                f"kl:{stochastic_state.kl_divergence:.4f}",
                f"v_t:{stochastic_state.v_t:.4f}",
            ],
            confidence=max(0.25, min(1.0, stochastic_state.lambda_noise + 0.2)),
            module_temperature=0.9,
        )
        if stochastic_contribution is not None:
            action_contributions.append(stochastic_contribution)
            action_snapshot = self._reintegrate_probability_snapshot(
                snapshot=action_snapshot,
                contributions=action_contributions,
                token_state=token_state,
                source_chain=["action_field_after_stochastic"],
            )
            action_truth = self._refresh_action_truth(action_snapshot.action, action_truth)

        current_final_distribution = dict(action_snapshot.action.winner_posterior or {})
        current_final_distribution, candidate_penalties, sampling_penalty_applied = self.authenticity_policy.apply_sampling_penalties(
            current_final_distribution,
            query_kind=query_state.legacy_query_kind,
            disclosure_intent=disclosure_state.posterior.top_intent,
            slow_variables=round_context["slow_variables"],
            memory_cue=context.get("cue"),
            shaping_events=shaping_events,
        )
        if candidate_penalties or sampling_penalty_applied > 0.0:
            authenticity_contribution = self.authenticity_policy.build_action_penalty_contribution(
                AuthenticityRecord(
                    self_grounding_score=float(
                        context.get("grounding_capsule", {}).get("state_summary", {}).get("body_energy", state.body_energy)
                        or 0.0
                    ),
                    guard_action="sampling_penalty",
                    disclosure_detail=disclosure_state.posterior.top_intent or "none",
                    state_sources=list(context.get("grounding_capsule", {}).get("state_sources", [])),
                    candidate_penalties=dict(candidate_penalties),
                    sampling_penalty_applied=float(sampling_penalty_applied),
                )
            )
            action_contributions.append(authenticity_contribution)
            action_snapshot = self._reintegrate_probability_snapshot(
                snapshot=action_snapshot,
                contributions=action_contributions,
                token_state=token_state,
                source_chain=["action_field_after_authenticity"],
            )
        action_truth = self._refresh_action_truth(action_snapshot.action, action_truth)
        if candidate_penalties:
            gate_decisions.append(
                {
                    "stage": "authenticity_sampling",
                    "owner": "AuthenticityPolicy",
                    "allowed": True,
                    "requires_resample": False,
                    "reason": (
                        f"query_kind={query_state.legacy_query_kind}; "
                        f"disclosure_intent={disclosure_state.posterior.top_intent}; "
                        f"penalized={','.join(sorted(candidate_penalties))}"
                    ),
                }
            )

        plausibility_guard = self.agent_map["BehaviorPlausibilityGuard"]
        plausibility_by_action: dict[str, dict[str, Any]] = {}
        for action_name, probability in dict(action_snapshot.action.winner_posterior or {}).items():
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
        field_distribution = dict(action_snapshot.action.winner_posterior or {})
        if blocked_actions:
            dominant_blocked_action = max(blocked_actions, key=lambda action_name: field_distribution.get(action_name, 0.0))
            dominant_blocked_probability = field_distribution.get(dominant_blocked_action, 0.0)
        provisional_action_name = str(action_snapshot.action.winner_target or self._top_action_name(field_distribution))
        selected_plausibility = plausibility_by_action.get(
            provisional_action_name,
            {"pass": True, "plausibility_fail_score": 0.0},
        )
        fail_score = max(
            selected_plausibility.get("plausibility_fail_score", 0.0),
            blocked_actions.get(dominant_blocked_action, {}).get("plausibility_fail_score", 0.0) if dominant_blocked_action else 0.0,
        )
        top_probability = max(field_distribution.values()) if field_distribution else 0.0
        preemptive_guard = dominant_blocked_action is not None and dominant_blocked_probability >= max(0.10, top_probability * 0.50)
        second_sampling = self._execute_skill(
            round_id=state.round_count,
            skill_name="request_second_sampling",
            inputs={"fail_score": fail_score, "attempts": 0},
            provider=lambda fail_score, attempts: plausibility_guard.run_skill("request_second_sampling", fail_score, attempts),
            skill_traces=skill_traces,
            runtime_context=runtime_context,
            seed_ref=round_seed,
        )["flag"]
        if state.resource_state.get("resource_mode") == "starvation":
            second_sampling = False
        self._execute_skill(
            round_id=state.round_count,
            skill_name="escalate_value_reestimate",
            inputs={"fail_score": fail_score},
            provider=lambda fail_score: plausibility_guard.run_skill("escalate_value_reestimate", fail_score),
            skill_traces=skill_traces,
            runtime_context=runtime_context,
            seed_ref=round_seed,
        )
        plausibility_reason_actions = [dominant_blocked_action] if dominant_blocked_action is not None else [provisional_action_name]
        if second_sampling and (not selected_plausibility.get("pass", True) or preemptive_guard):
            gated_actions = [
                action_name
                for action_name, plausibility in blocked_actions.items()
                if field_distribution.get(action_name, 0.0) >= max(0.10, dominant_blocked_probability * 0.7)
            ]
            if not gated_actions and dominant_blocked_action is not None:
                gated_actions = [dominant_blocked_action]
            plausibility_reason_actions = list(gated_actions or plausibility_reason_actions)
            plausibility_contribution = self._build_distribution_delta_contribution(
                module_name="BehaviorPlausibilityGuard",
                module_type="guard",
                from_distribution=dict(action_snapshot.action.winner_posterior or {}),
                to_distribution={
                    action: probability
                    for action, probability in dict(action_snapshot.action.winner_posterior or {}).items()
                    if action not in set(gated_actions)
                },
                hard_mask={action_name: True for action_name in gated_actions},
                trace_reason=f"plausibility resample fail_score={fail_score:.4f}",
                projection_reason="plausibility guard projected from blocked action set",
                applied_at_stage="plausibility_guard",
                native_operator="hard_mask",
                dependency_trace=[
                    f"fail_score:{fail_score:.4f}",
                    f"blocked:{','.join(sorted(gated_actions))}",
                ],
                confidence=max(0.4, min(1.0, fail_score + 0.2)),
            )
            if plausibility_contribution is not None:
                action_contributions.append(plausibility_contribution)
                action_snapshot = self._reintegrate_probability_snapshot(
                    snapshot=action_snapshot,
                    contributions=action_contributions,
                    token_state=token_state,
                    source_chain=["action_field_after_plausibility"],
                )
                action_truth = self._refresh_action_truth(action_snapshot.action, action_truth)
        gate_decisions.append(
            {
                "stage": "plausibility_guard",
                "owner": "BehaviorPlausibilityGuard",
                "allowed": not (second_sampling and (not selected_plausibility.get("pass", True) or preemptive_guard)),
                "requires_resample": False,
                "reason": (
                    f"fail_score={fail_score:.2f}; blocked={','.join(sorted(action for action in plausibility_reason_actions if action))}; "
                    f"blocked_p={dominant_blocked_probability:.3f}"
                ),
            }
        )

        provisional_action_name = str(action_snapshot.action.winner_target or self._top_action_name(dict(action_snapshot.action.winner_posterior or {})))
        provisional_focus_lock_count = state.focus_lock_count + 1 if provisional_action_name == previous_focus else 1

        forced = self.agent_map["ForcedModeSwitch"]
        lock_score = self._execute_skill(
            round_id=state.round_count,
            skill_name="detect_mode_lock",
            inputs={"history": state.mode_history, "focus_lock_count": provisional_focus_lock_count},
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
            forced_distribution = dict(action_snapshot.action.winner_posterior or {})
            if provisional_action_name in forced_distribution and provisional_action_name != "respond":
                shifted_mass = forced_distribution.pop(provisional_action_name, 0.0)
                forced_distribution["respond"] = forced_distribution.get("respond", 0.0) + max(shifted_mass, 0.25)
            forced_mode_contribution = self._build_distribution_delta_contribution(
                module_name="ForcedModeSwitch",
                module_type="guard",
                from_distribution=dict(action_snapshot.action.winner_posterior or {}),
                to_distribution=self._normalize(forced_distribution),
                trace_reason=f"forced focus switch lock_score={lock_score:.4f}",
                projection_reason="forced mode switch projected from focus-lock override",
                applied_at_stage="forced_mode_switch",
                native_operator="focus_override",
                dependency_trace=[
                    f"lock_score:{lock_score:.4f}",
                    f"blocked_action:{provisional_action_name}",
                ],
                confidence=max(0.45, min(1.0, lock_score)),
            )
            if forced_mode_contribution is not None:
                action_contributions.append(forced_mode_contribution)
                action_snapshot = self._reintegrate_probability_snapshot(
                    snapshot=action_snapshot,
                    contributions=action_contributions,
                    token_state=token_state,
                    source_chain=["action_field_after_forced_switch"],
                )
                action_truth = self._refresh_action_truth(action_snapshot.action, action_truth)
            gate_decisions.append({"stage": "forced_mode_switch", "owner": "ForcedModeSwitch", "allowed": False, "requires_resample": False, "reason": f"lock_score={lock_score:.2f}"})

        output_gate = self.agent_map["OutputGate"]
        action_truth.setdefault("gate", {})
        control_ledger.setdefault("gate", {})
        gate_inputs = {
            action_name: probability
            for action_name, probability in dict(action_snapshot.action.winner_posterior or {}).items()
            if probability > thresholds["p_floor"]
        }
        if not gate_inputs and provisional_action_name:
            gate_inputs[provisional_action_name] = float(field_distribution.get(provisional_action_name, 0.0) or 0.0)
        gate_values: dict[str, float] = {}
        for action_name in gate_inputs:
            gate = float(
                self._execute_skill(
                    round_id=state.round_count,
                    skill_name="apply_output_gate",
                    inputs={"action": action_name, "state": state, "scenario": scenario, "relation_state": relation_state},
                    provider=lambda action, state, scenario, relation_state: output_gate.run_skill("apply_output_gate", action, state, scenario, relation_state),
                    skill_traces=skill_traces,
                    runtime_context=runtime_context,
                    seed_ref=round_seed,
                ).get("gate", 1.0)
                or 1.0
            )
            gate_values[action_name] = gate
            action_truth["gate"][action_name] = min(float(action_truth["gate"].get(action_name, 1.0) or 1.0), gate)
            control_ledger["gate"][action_name] = min(float(control_ledger["gate"].get(action_name, 1.0) or 1.0), gate)
        if any(gate < 1.0 for gate in gate_values.values()):
            gated_distribution = {
                action_name: probability * gate_values.get(action_name, 1.0)
                for action_name, probability in dict(action_snapshot.action.winner_posterior or {}).items()
            }
            output_gate_contribution = self._build_distribution_delta_contribution(
                module_name="OutputGate",
                module_type="guard",
                from_distribution=dict(action_snapshot.action.winner_posterior or {}),
                to_distribution=self._normalize(gated_distribution or dict(action_snapshot.action.winner_posterior or {})),
                hard_mask={action_name: gate == 0.0 for action_name, gate in gate_values.items()},
                trace_reason=(
                    "output gate applied gates="
                    + ",".join(f"{action}:{round(gate, 4)}" for action, gate in sorted(gate_values.items()))
                ),
                projection_reason="output gate projected from final action suppression",
                applied_at_stage="output_gate",
                native_operator="final_gate",
                dependency_trace=[
                    f"peak:{provisional_action_name}",
                    *[f"gate:{action}:{round(gate, 4)}" for action, gate in sorted(gate_values.items())],
                ],
                confidence=max(0.5, min(1.0, max((1.0 - gate for gate in gate_values.values()), default=0.0) + 0.2)),
            )
            if output_gate_contribution is not None:
                action_contributions.append(output_gate_contribution)
                action_snapshot = self._reintegrate_probability_snapshot(
                    snapshot=action_snapshot,
                    contributions=action_contributions,
                    token_state=token_state,
                    source_chain=["action_field_after_output_gate"],
                )
                action_truth = self._refresh_action_truth(action_snapshot.action, action_truth)
        peak_gate = float(action_truth.get("gate", {}).get(provisional_action_name, 1.0) or 1.0)
        gate_decisions.append(
            {
                "stage": "output_gate",
                "owner": "OutputGate",
                "allowed": peak_gate > 0,
                "requires_resample": False,
                "reason": f"peak={provisional_action_name}; gate={peak_gate:.2f}",
            }
        )

        context_memory_backing = {}
        if context.get("cue") or float(context.get("recall_strength", 0.0) or 0.0) > 0.0:
            context_memory_backing = {
                "cue": context.get("cue"),
                "strength": round(float(context.get("recall_strength", 0.0) or 0.0), 4),
                "summary": context.get("cue") or event.content[:80],
            }
        expressive_overlay = self._initiative_distribution_payload(
            state,
            relation_state=relation_state,
            vitality_snapshot={},
            cue=context.get("cue"),
            memory_backing=context_memory_backing,
            latest_recorded_at=latest_round_recorded_at,
            current_goal=context.get("run_context", {}).get("goal") if isinstance(context.get("run_context"), dict) else state.current_goal,
        )
        expressive_contribution = self._build_expressive_action_contribution(
            current_distribution=dict(action_snapshot.action.winner_posterior or {}),
            payload=expressive_overlay,
            scenario=scenario,
            endogenous_turn=endogenous_turn,
        )
        if expressive_contribution is not None:
            action_contributions.append(expressive_contribution)
            action_snapshot = self._reintegrate_probability_snapshot(
                snapshot=action_snapshot,
                contributions=action_contributions,
                token_state=token_state,
                source_chain=["action_field_after_expressive"],
            )
            action_truth = self._refresh_action_truth(action_snapshot.action, action_truth)
        gate_decisions.append(
            {
                "stage": "expressive_field",
                "owner": "InitiativeRuntime",
                "allowed": expressive_overlay.get("expression_mode") != "silent",
                "requires_resample": False,
                "reason": (
                    f"intent={expressive_overlay.get('top_intent', 'stay_silent')}; "
                    f"mode={expressive_overlay.get('expression_mode', 'silent')}; "
                    f"speech_cost={float(expressive_overlay.get('speech_cost', 0.0) or 0.0):.2f}"
                ),
            }
        )

        provisional_action_name = str(action_snapshot.action.winner_target or self._top_action_name(dict(action_snapshot.action.winner_posterior or {})))
        provisional_action = ActionCandidate(
            name=provisional_action_name,
            probability=float((action_snapshot.action.winner_posterior or {}).get(provisional_action_name, 0.0) or 0.0),
            rationale="field peak before terminal sampling",
        )
        contributions, proposal_records = self._build_contributions(
            action_signals,
            state,
            provisional_action_name,
            conflict_score,
            fail_score,
            int(control_ledger.get("resample_idx", 0) or 0),
        )
        vitality_snapshot = self._build_vitality_snapshot(
            state=state,
            context=context,
            relation_state=relation_state,
            prior_closeness=prior_closeness,
            scenario=scenario,
            sampled_action=provisional_action,
            contributions=contributions,
            gate_decisions=gate_decisions,
            shaping_events=shaping_events,
        )
        vitality_contribution = self.vitality_engine.build_vitality_modulation_contribution(vitality_snapshot)
        action_contributions.append(vitality_contribution)
        action_snapshot = self._reintegrate_probability_snapshot(
            snapshot=action_snapshot,
            contributions=action_contributions,
            token_state=token_state,
            source_chain=["action_field_after_vitality"],
        )
        action_truth = self._refresh_action_truth(action_snapshot.action, action_truth)
        sample_value, action_entropy_ref = self.entropy_pool.uniform(
            purpose=f"round-{round_seed}-action-sample",
            node_name="action_sample",
        )
        sampled_action = self._sample_action_from_distribution(
            dict(action_snapshot.action.winner_posterior or {}),
            sample_value,
        )
        sampled_action = self._collapse_internal_sampled_action(sampled_action)
        route_type = self._runtime_route_type_for_round(
            event=event,
            scenario=scenario,
            mode=requested_mode,
            top_action=sampled_action.name,
            model_call_count=0,
        )
        stochastic_state.entropy_ref = action_entropy_ref if not stochastic_state.entropy_ref.source else stochastic_state.entropy_ref
        stochastic_state.entropy_refs_by_node["action_sample"] = to_dict(action_entropy_ref)
        state.focus_lock_count = state.focus_lock_count + 1 if sampled_action.name == previous_focus else 1
        locked_action_before_render = sampled_action.name
        locked_probability_before_render = round(float((action_snapshot.action.winner_posterior or {}).get(sampled_action.name, 0.0) or 0.0), 6)
        gate_at_render = round(float(action_truth.get("gate", {}).get(sampled_action.name, 1.0) or 0.0), 6)

        expression = build_expression_profile(
            sampled_action=sampled_action.name,
            state=to_dict(state),
            scenario=scenario,
            scenario_config=scenario_cfg,
            relation_state=relation_state,
            stochastic=stochastic_state,
            intent_context={
                "query_intent": query_state.posterior.top_intent,
                "disclosure_intent": disclosure_state.posterior.top_intent,
            },
        )
        expression = self._apply_conflict_expression_adjustments(
            expression,
            dict(control_ledger.get("conflict", {}) or {}),
        )
        output_profiles = self._execute_parallel_skills(
            round_id=state.round_count,
            tasks=[
                {
                    "name": "tone_params",
                    "skill_name": "render_tone_profile",
                    "inputs": {"expression_profile": expression},
                    "provider": lambda expression_profile: output_gate.run_skill("render_tone_profile", expression_profile),
                    "parallel_group": "output_profiles",
                    "binding_tier": self._pipeline_tier("OutputGate"),
                },
                {
                    "name": "delay_params",
                    "skill_name": "compute_delay_profile",
                    "inputs": {"expression_profile": expression},
                    "provider": lambda expression_profile: output_gate.run_skill("compute_delay_profile", expression_profile),
                    "parallel_group": "output_profiles",
                    "binding_tier": self._pipeline_tier("OutputGate"),
                },
            ],
            skill_traces=skill_traces,
            runtime_context=runtime_context,
            parallel_traces=parallel_traces,
        )
        tone_params = output_profiles["tone_params"]
        delay_params = output_profiles["delay_params"]

        late_perspective = {"state_hypothesis": {}, "reaction_hypothesis": {}}
        allow_late_perspective, late_perspective_reason = self._should_run_late_perspective(
            event,
            state,
            relation_state,
            sampled_action.name,
            route_type=route_type,
        )
        if allow_late_perspective:
            perspective = self.agent_map["PerspectiveModel"]
            perspective_results = self._execute_parallel_skills(
                round_id=state.round_count,
                tasks=[
                    {
                        "name": "state_hypothesis",
                        "skill_name": "infer_other_state",
                        "inputs": {**runtime_inputs, "sampled_action": sampled_action.name, "relation_state": relation_state},
                        "provider": lambda event, state, scenario, context, sampled_action, relation_state: self._infer_other_state_via_model(
                            event,
                            state,
                            scenario,
                            context,
                            sampled_action,
                            relation_state,
                            model_call_traces=model_call_traces,
                            parallel_group="late_perspective",
                        ),
                        "fallback_provider": lambda event, state, scenario, context, sampled_action, relation_state: perspective.fallback_infer_other_state(event, state, scenario, context),
                        "parallel_group": "late_perspective",
                        "binding_tier": self._pipeline_tier("PerspectiveModel"),
                    },
                    {
                        "name": "reaction_hypothesis",
                        "skill_name": "simulate_other_reaction",
                        "inputs": {**runtime_inputs, "sampled_action": sampled_action.name, "relation_state": relation_state},
                        "provider": lambda event, state, scenario, context, sampled_action, relation_state: self._simulate_other_reaction_via_model(
                            event,
                            state,
                            scenario,
                            context,
                            sampled_action,
                            relation_state,
                            model_call_traces=model_call_traces,
                            parallel_group="late_perspective",
                        ),
                        "fallback_provider": lambda event, state, scenario, context, sampled_action, relation_state: perspective.fallback_simulate_other_reaction(event, state, scenario, context),
                        "parallel_group": "late_perspective",
                        "binding_tier": self._pipeline_tier("PerspectiveModel"),
                    },
                ],
                skill_traces=skill_traces,
                runtime_context=runtime_context,
                parallel_traces=parallel_traces,
            )
            late_perspective["state_hypothesis"] = perspective_results["state_hypothesis"].get("state_hypothesis", {})
            late_perspective["reaction_hypothesis"] = perspective_results["reaction_hypothesis"].get("reaction_hypothesis", {})
        else:
            gate_decisions.append(
                {
                    "stage": "late_perspective",
                    "owner": "PerspectiveModel",
                    "allowed": False,
                    "requires_resample": False,
                    "reason": late_perspective_reason,
                }
            )

        slow_variables = self._build_slow_variable_payload(
            state=state,
            context=context,
            relation_state=relation_state,
            prior_closeness=prior_closeness,
        )
        render_plan = build_render_plan(
            sampled_action=sampled_action.name,
            expression=ExpressionProfile(**{**to_dict(expression)}),
            safety_constraints={
                "gate": action_truth.get("gate", {}).get(sampled_action.name, 1.0),
                "tone_params": tone_params["tone_params"],
                "delay_params": delay_params["delay_params"],
                "conflict_hot": dict(control_ledger.get("conflict", {}) or {}).get("circuit_breaker", {}).get("active", False),
                "compromise_template": dict(control_ledger.get("conflict", {}) or {}).get("compromise", {}).get("template"),
                "winning_priority": dict(control_ledger.get("conflict", {}) or {}).get("winning_priority"),
                "repair_stage": dict(control_ledger.get("conflict", {}) or {}).get("repair_state_snapshot", {}).get("stage"),
            },
            scenario=scenario,
            event_summary=event.content,
            target=event.target,
            relation_state=relation_state,
            perspective=late_perspective,
            identity_context=identity_context,
            state_focus=sampled_action.name,
            memory_cue=context.get("cue"),
            recall_strength=float(context.get("recall_strength", 0.0)),
            slow_variables=slow_variables,
            shaping_events=shaping_events,
            repair_expression=self._build_repair_expression_policy(dict(control_ledger.get("conflict", {}) or {})),
        )
        if sampled_action.name == "self_run":
            rendered_output = {
                "text": "",
                "route": "runtime",
                "model": "self_run_realizer",
            }
            rendered_result = SimpleNamespace(degraded=False, failure_policy_applied="")
        elif route_type == "endogenous_light":
            rendered_output = self._render_expression_via_humanized_fallback_chain(
                render_plan,
                fallback_reason="endogenous_light_local_only",
                deterministic_model="fallback",
                model_call_traces=model_call_traces,
            )
            rendered_result = SimpleNamespace(degraded=True, failure_policy_applied="endogenous_light_local_only")
        else:
            rendered_output, rendered_result = self._execute_skill_with_result(
                round_id=state.round_count,
                skill_name="render_expression",
                inputs={"render_plan": render_plan},
                provider=lambda render_plan: self._render_expression_via_model(
                    render_plan,
                    route_type=route_type,
                    model_call_traces=model_call_traces,
                ),
                skill_traces=skill_traces,
                runtime_context=runtime_context,
                fallback_provider=lambda render_plan: self._render_expression_via_humanized_fallback_chain(
                    render_plan,
                    fallback_reason="renderer_failure",
                    deterministic_model="fallback",
                    model_call_traces=model_call_traces,
                ),
                seed_ref=round_seed,
            )
        initial_auth = self._evaluate_authenticity(rendered_output.get("text", ""), render_plan)
        guard_action = "pass"
        violation_types = list(initial_auth.get("violation_types", []))
        final_auth = initial_auth
        final_rendered_output = rendered_output

        if violation_types:
            if self._should_allow_renderer_resample(
                route_type=route_type,
                render_plan=render_plan,
                violation_types=violation_types,
                context=context,
            ):
                guard_action = "resample"
                try:
                    resampled_output = self._render_expression_via_model(
                        render_plan,
                        route_type=route_type,
                        prompt_mode="violation",
                        violation_types=violation_types,
                        model_call_traces=model_call_traces,
                    )
                except Exception:
                    resampled_output = None
                if resampled_output is not None:
                    resampled_auth = self._evaluate_authenticity(resampled_output.get("text", ""), render_plan)
                    if not resampled_auth.get("violation_types"):
                        final_rendered_output = resampled_output
                        final_auth = resampled_auth
                    else:
                        guard_action = "fallback"
                else:
                    guard_action = "fallback"
            else:
                guard_action = "fallback"

            if guard_action == "fallback":
                final_rendered_output = self._render_expression_via_humanized_fallback_chain(
                    render_plan,
                    fallback_reason="authenticity_guard_fallback",
                    deterministic_model="authenticity_fallback",
                    violation_types=violation_types,
                    model_call_traces=model_call_traces,
                )
                final_auth = self._evaluate_authenticity(final_rendered_output["text"], render_plan)

        final_auth_payload = self.authenticity_policy.build_record(
            evaluation=final_auth,
            guard_action=guard_action,
            disclosure_detail=render_plan.identity_context.disclosure_detail,
            rename_event=rename_event,
            candidate_penalties=candidate_penalties,
            sampling_penalty_applied=sampling_penalty_applied,
        )
        render_plan.identity_context.self_description_sources = sorted(
            set(render_plan.identity_context.self_description_sources + list(final_auth_payload.state_sources))
        )
        renderer_decision_integrity = {
            "locked_action": locked_action_before_render,
            "render_plan_action": render_plan.action,
            "post_render_action": sampled_action.name,
            "locked_probability": locked_probability_before_render,
            "gate_at_render": gate_at_render,
            "auth_guard_action": guard_action,
            "decision_mutated": False,
            "renderer_consumes_final_field": render_plan.action == locked_action_before_render == sampled_action.name,
            "mutation_reasons": [],
        }
        gate_decisions.append(
            {
                "stage": "authenticity_guard",
                "owner": "AuthenticityGuard",
                "allowed": guard_action == "pass",
                "requires_resample": guard_action == "resample",
                "reason": ",".join(violation_types) or "grounded",
            }
        )
        rendered_expression = RenderedExpression(
            text=final_rendered_output.get("text", ""),
            route=final_rendered_output.get("route", "renderer"),
            model=final_rendered_output.get("model", "fallback"),
            delivery_mode=render_plan.delivery_mode,
            degraded=rendered_result.degraded or guard_action == "fallback",
            failure_policy_applied=rendered_result.failure_policy_applied,
            authenticity=final_auth_payload,
        )

        state.focus = sampled_action.name
        state.last_action = sampled_action.name
        if sampled_action.name in {"respond", "plan", "clarify"}:
            state.budget_remaining = _clip(state.budget_remaining + 0.0016)
        anchor_alignment = self._current_axis_alignment(
            anchor=state.personality_anchor,
            normalized_axes={
                axis: float(
                    dict(
                        state.instinct_field.collapse_trace.get("subject_vector", {})
                        or state.instinct_field.collapse_trace.get("coupled_axes", {})
                        or state.instinct_field.collapse_trace.get("normalized_axes", {})
                        or {}
                    ).get(axis, 0.5)
                    or 0.5
                )
                for axis in AXES
            },
            action=sampled_action.name,
            match_scores=dict(state.instinct_field.collapse_trace.get("match_scores", {}) or {}),
        )
        self._apply_subject_dynamics_after_round(
            state=state,
            appraisal=appraisal,
            slow_variables=slow_variables,
            relation_state=relation_state,
            sampled_action=sampled_action.name,
            anchor_alignment=anchor_alignment,
            requested_mode=requested_mode,
        )
        starvation = self.config["resource_rules"]["resource_defaults"]["starvation_threshold"]
        recovered_this_round = dict(control_ledger.get("conflict", {}) or {}).get("repair_transition", {}).get("to_stage") == "recovered"
        if state.budget_remaining <= starvation / 10 and not recovered_this_round:
            state.safe_mode = True
            state.mode = "safe"

        style_profile = compute_style_profile(to_dict(state), scenario, self.config["output_style"]["styles"])
        sampled_action.metadata["style_profile"] = style_profile
        sampled_action.metadata["expression_profile"] = to_dict(expression)
        sampled_action.metadata["render_plan"] = to_dict(render_plan)
        sampled_action.metadata["rendered_expression"] = to_dict(rendered_expression)
        identity_evolution = self.identity_runtime.build_identity_evolution_payload(
            state=state,
            rename_event=rename_event,
            shaping_events=shaping_events,
            slow_variables=slow_variables,
            identity_context=render_plan.identity_context,
        )
        long_run_projection = self.long_run_analyzer.build_round_projection(
            round_id=state.round_count,
            vitality_snapshot=vitality_snapshot,
            authenticity=to_dict(final_auth_payload),
            identity_evolution=identity_evolution,
            shaping_events=shaping_events,
            personality_anchor=to_dict(state.personality_anchor),
            top_drivers=to_dict(contributions[:3]),
        )
        long_run_projection["online_prior"] = dict(online_long_run_projection)
        long_run_projection["anchor_alignment"] = round(float(state.personality_anchor.alignment or 0.0), 4)
        long_run_projection["anchor_drift"] = round(float(state.personality_anchor.drift or 0.0), 4)
        initiative_payload = self.evaluate_initiative_overlay(
            state=state,
            relation_state=relation_state,
            vitality_snapshot=vitality_snapshot,
            context=context,
            latest_round_recorded_at=latest_round_recorded_at,
            endogenous_turn=endogenous_turn,
            rendered_preview=rendered_expression.text,
        )
        expressive_trace_payload = {
            **initiative_payload,
            "monologue_stream": monologue_stream_trace,
        }
        for row in skill_traces:
            row["session_id"] = state.session_id
            row["recorded_at"] = recorded_at
            row["recorded_date"] = recorded_date
        parallel_groups = sorted(
            {
                row.get("parallel_group")
                for row in [*model_call_traces, *parallel_traces]
                if row.get("parallel_group")
            }
        )
        runtime_metrics = {
            "route_type": route_type,
            "model_call_count": len(model_call_traces),
            "parallel_task_count": len(parallel_traces),
            "parallel_groups": parallel_groups,
            "optional_timeout_count": sum(
                1
                for row in parallel_traces
                if row.get("task_priority") == "optional" and row.get("task_outcome") == "timeout"
            ),
            "speculative_drop_count": sum(
                1
                for row in parallel_traces
                if row.get("task_priority") == "speculative" and row.get("task_outcome") != "completed"
            ),
            "total_model_wait_ms": int(sum(int(row.get("latency_ms", 0) or 0) for row in model_call_traces)),
            "total_turn_ms": max(1, int((time.perf_counter() - turn_started) * 1000)),
        }
        renderer_contribution = self._build_renderer_token_contribution(render_plan, context)
        probability_field_snapshot = self._reintegrate_probability_snapshot(
            snapshot=action_snapshot,
            contributions=[*action_contributions, renderer_contribution],
            token_state=token_state,
            source_chain=[
                "probability_field_native",
                "context_memory_action_token",
                "field_first_tick",
            ],
        )
        self._finalize_action_bookkeeping_from_action_layer(
            action_bookkeeping,
            probability_field_snapshot.action,
            finalize_stage="final",
        )
        self._merge_control_ledger_into_action_bookkeeping(action_bookkeeping, control_ledger)
        probability_field = to_dict(probability_field_snapshot)
        trace_action_bookkeeping = self._action_bookkeeping_payload(
            action_bookkeeping=action_bookkeeping,
            probability_field_snapshot=probability_field_snapshot,
            control_ledger=control_ledger,
            action_truth=action_truth,
            stochastic_state=stochastic_state,
        )
        self._update_emergent_action_sketches(
            state=state,
            action_posterior=dict(probability_field_snapshot.action.winner_posterior or {}),
            sampled_action=sampled_action.name,
        )
        recall_payload = dict(context.get("recall_payload", {}) or {})
        prior_contract_state = RuntimeState(**to_dict(prior_state))
        self._sync_plan16_state(prior_contract_state)
        prior_history_burden = self._build_history_burden(prior_contract_state)
        state.history_burden = self._build_history_burden(state)
        event_log_ref = {
            "round_trace_ref": f"round://{state.round_count}",
            "round_trace_jsonl": str(self.trace_store.rounds_jsonl_path),
            "command_trace_jsonl": str(self.trace_store.commands_jsonl_path),
            "memory_event_ref": f"memory://raw/{state.session_id}/{state.round_count}",
            "memory_events_jsonl": str(self.memory_store.raw_events_path),
        }
        memory_evidence = {
            "event_log_ref": event_log_ref,
            "cue": str(context.get("cue") or ""),
            "hit": bool(recall_payload.get("found", False)),
            "miss": bool(recall_payload.get("recall_failure", False)),
            "mode": str(recall_payload.get("mode") or "none"),
            "detail": bool(recall_payload.get("detail", False)),
            "cue_rescue": bool(recall_payload.get("cue_rescue", False)),
            "interference": round(float(recall_payload.get("interference", 0.0) or 0.0), 4),
            "contamination": dict(recall_payload.get("contamination", {}) or {}),
            "latency_cost": {
                "latency_ms": int(recall_payload.get("latency_ms", 0) or 0),
                "cost_tier": str(recall_payload.get("cost_tier") or ""),
                "search_trace": dict(recall_payload.get("search_trace", {}) or {}),
            },
            "behavioral_consequence": {
                "selected_action": sampled_action.name,
                "recall_strength": round(float(context.get("recall_strength", 0.0) or 0.0), 4),
                "initiative_top_intent": str(initiative_payload.get("top_intent") or ""),
                "initiative_memory_backing": dict(initiative_payload.get("memory_backing", {}) or {}),
                "probability_peak": dict(probability_field_snapshot.action.winner_posterior or {}),
            },
        }
        arbitration_record = {
            "competing_proposals": [
                {
                    "agent_name": str(item.get("agent_name") or ""),
                    "stage": str(item.get("stage") or ""),
                    "top_action": str(item.get("top_action") or ""),
                    "selected": bool(item.get("selected", False)),
                    "gated_actions": list(item.get("gated_actions", []) or []),
                }
                for item in proposal_records
            ],
            "evidence_sources": sorted(
                {
                    str(item.get("agent_name") or "")
                    for item in proposal_records
                    if str(item.get("agent_name") or "").strip()
                }
            ),
            "priority_scores": {
                str(item.get("agent_name") or f"proposal:{index}"): round(
                    float(item.get("confidence", 0.0) or 0.0) * float(item.get("weight_applied", 0.0) or 0.0),
                    4,
                )
                for index, item in enumerate(proposal_records, start=1)
            },
            "budget_consumption": {
                "budget_remaining": round(float(state.budget_remaining or 0.0), 4),
                "resample_count": int(control_ledger.get("resample_idx", 0) or 0),
                "window_tool_actions": int(state.autonomy_loop.window_tool_actions or 0),
                "window_endogenous_rounds": int(state.autonomy_loop.window_endogenous_rounds or 0),
            },
            "selected_winner": sampled_action.name,
            "rejected_options": [
                {
                    "agent_name": str(item.get("agent_name") or ""),
                    "top_action": str(item.get("top_action") or ""),
                    "reason": "not_selected" if not item.get("selected", False) else "selected",
                }
                for item in proposal_records
                if str(item.get("top_action") or "") and not bool(item.get("selected", False))
            ],
            "hard_masks": list(conflict_action_truth.get("hard_masked_targets", []) or []),
            "consequence_patch": {
                "repair_state": to_dict(state.repair_state),
                "suppressed_actions": to_dict(state.agency_loop.suppressed_actions),
                "budget_residue": round(float(state.budget_remaining or 0.0), 4),
                "scheduled_task_residue": int(state.agency_loop.budget.get("scheduled_due", 0) or 0),
                "history_inertia_score": round(float(state.history_burden.get("history_inertia_score", 0.0) or 0.0), 4),
                "safe_mode_owner": str(state.conflict_safe_mode_owner or ""),
            },
        }
        state.arbitration_state = self._build_arbitration_state(state, round_arbitration=arbitration_record)
        history_burden_delta = self._history_burden_delta_payload(prior_history_burden, state.history_burden)
        snapshot_continuity = {
            "continuity_nonce": state.subject_core.continuity_nonce,
            "runtime_revision": int(state.runtime_revision or 0) + 1,
            "last_checkpoint_id": state.last_checkpoint_id,
            "rewind_supported": True,
            "snapshot_dir": str(self.snapshot_dir),
            "self_continuity": round(float(state.self_continuity or 0.0), 4),
            "repair_stage": str(state.repair_state.stage or ""),
            "history_burden_delta": history_burden_delta,
        }

        trace = RoundTrace(
            session_id=state.session_id,
            recorded_at=recorded_at,
            recorded_date=recorded_date,
            round_id=state.round_count,
            subject_id=state.subject_core.subject_id,
            continuity_nonce=state.subject_core.continuity_nonce,
            scenario=scenario,
            mode=state.mode,
            sampled_action=sampled_action.name,
            contributions=contributions,
            top_drivers=contributions[:3],
            style_profile=style_profile,
            state_snapshot=to_dict(state),
            cause_type="endogenous" if endogenous_turn else "external_stimulus",
            boundary_action="allow_internal",
            pipeline_stages=list(PIPELINE_TELEMETRY_STAGES),
            proposal_summaries=proposal_records,
            gate_decisions=gate_decisions,
            skill_traces=skill_traces,
            parallel_traces=parallel_traces,
            action_bookkeeping=trace_action_bookkeeping,
            candidate_distribution=dict(probability_field_snapshot.action.winner_posterior or {}),
            probability_field=probability_field,
            stochastic_state=to_dict(stochastic_state),
            conflict_arbitration={
                **dict(control_ledger.get("conflict", {}) or {}),
                "hard_masked_targets": list(conflict_action_truth.get("hard_masked_targets", []) or []),
                "winner_peak_posterior": dict(conflict_action_truth.get("winner_posterior", {}) or {}),
                "conflict_mode": str(conflict_action_truth.get("conflict_mode") or control_ledger.get("conflict_mode", "monitor")),
            },
            event_log_ref=event_log_ref,
            memory_evidence=memory_evidence,
            arbitration_record=arbitration_record,
            history_burden_delta=history_burden_delta,
            snapshot_continuity=snapshot_continuity,
            render_plan=to_dict(render_plan),
            rendered_expression=to_dict(rendered_expression),
            renderer_decision_integrity=renderer_decision_integrity,
            memory_write_gate=memory_write_gate,
            authenticity=to_dict(final_auth_payload),
            identity_evolution=identity_evolution,
            vitality_snapshot=vitality_snapshot,
            vitality_events=shaping_events,
            long_run_projection=long_run_projection,
            initiative=initiative_payload,
            expressive_trace=expressive_trace_payload,
            appraisal_snapshot=appraisal,
            state_delta_before_clip={
                "mood": round(float(event.valence) * 0.08, 4),
                "body_energy": round(float(event.energy_delta), 4),
                "affect_residue": round(float(state.affect_residue) - float(prior_state.affect_residue), 4),
            },
            state_delta_after_clip={
                "mood": round(float(state.mood) - float(prior_state.mood), 4),
                "body_energy": round(float(state.body_energy) - float(prior_state.body_energy), 4),
                "affect_residue": round(float(state.affect_residue) - float(prior_state.affect_residue), 4),
            },
            delta_suppression_reason=[
                reason
                for reason in (
                    "delta_clipped"
                    if abs(float(event.valence)) > 0.01 and round(float(state.mood) - float(prior_state.mood), 4) == 0.0
                    else None,
                    "expression_threshold_not_met"
                    if abs(float(event.valence)) > 0.01 and round(float(state.affect_residue) - float(prior_state.affect_residue), 4) == 0.0
                    else None,
                )
                if reason is not None
            ],
            run_context=context.get("run_context", {}),
            run_contamination_detected=bool(context.get("run_contamination_detected", False)),
            identity_evidence_score=round(float(identity_evidence.get("identity_score", 0.0)), 4),
            identity_trigger_blockers=list(identity_evidence.get("rejection_reasons", [])),
            temperament_window_summary=dict(state.temperament_state.get("drift_diagnostics", {})),
            dream_run_id=dream_payload.get("run_id"),
            dream_trigger=dream_payload.get("trigger"),
            dream_guard_summary=dream_payload.get("guard_summary", {}),
            dream_trace_ref=dream_payload.get("trace_ref"),
            dream_effect_summary=dream_payload.get("effect_summary", {}),
            model_call_traces=model_call_traces,
            runtime_metrics=runtime_metrics,
            resample_count=int(control_ledger.get("resample_idx", 0) or 0),
        )

        updated_learning_state, motivation_feedback_payload = self.motivation_feedback_updater.update(
            learning_state=state.motivation_learning_state,
            pool_state=motivation_pool_state,
            trace_payload=to_dict(trace),
        )
        state.motivation_learning_state = updated_learning_state
        state.motivation_pool_state.last_feedback_update_at = recorded_at
        latest_trigger = None
        trigger_context = None
        micro_intent = None
        replay_chain = None
        if endogenous_turn and self._pending_endogenous_trigger is not None:
            latest_trigger = EndogenousTickTrigger(**to_dict(self._pending_endogenous_trigger))
            trigger_context = (
                EndogenousTriggerContext(**to_dict(self._pending_endogenous_trigger_context))
                if self._pending_endogenous_trigger_context is not None
                else EndogenousTriggerContext(scenario=scenario)
            )
            state.endogenous_scheduler_state = self.endogenous_scheduler.update_state(
                scheduler_state=state.endogenous_scheduler_state,
                trigger=latest_trigger,
                recorded_at=recorded_at,
            )
            micro_intent = self._build_endogenous_micro_intent(
                state=state,
                trigger=latest_trigger,
                pool_state=motivation_pool_state,
            )
            state.endogenous_state = EndogenousRuntimeState(
                current_intent=micro_intent,
                stability=micro_intent.stability,
                history=(*state.endogenous_state.history, micro_intent)[-20:],
                last_trigger=latest_trigger.trigger_type,
                last_suppression=state.endogenous_state.last_suppression,
            )
            replay_chain = EndogenousReplayChain(
                trigger=latest_trigger,
                trigger_context=trigger_context,
                motivation_pool=self.endogenous_motivation_pool.trace_payload(motivation_pool_state),
                motivation_feedback=motivation_feedback_payload,
                micro_intent=micro_intent,
                policy_shift=updated_learning_state.endogenous_policy_shift,
            )
        trace.motivation_pool = self.endogenous_motivation_pool.trace_payload(motivation_pool_state)
        trace.motivation_feedback = motivation_feedback_payload
        if latest_trigger is None:
            latest_trigger = (
                state.endogenous_scheduler_state.recent_triggers[-1]
                if state.endogenous_scheduler_state.recent_triggers
                else None
            )
        trace.endogenous_tick_reason = self.endogenous_scheduler.trace_payload(
            state.endogenous_scheduler_state,
            latest_trigger,
        )
        trace.endogenous_policy_shift = dict(updated_learning_state.endogenous_policy_shift)
        trace.endogenous_trigger_context = trigger_context
        trace.endogenous_suppression = state.endogenous_state.last_suppression
        trace.micro_intent = micro_intent
        trace.endogenous_replay_chain = replay_chain
        trace.state_snapshot = self._trace_state_snapshot_payload(
            state,
            endogenous_turn=endogenous_turn,
        )
        return cast(ArbitrationResult, {
            "trace": trace,
            "sampled_action": sampled_action,
            "rendered_expression": rendered_expression,
            "control_ledger": control_ledger,
        })

    def render_and_commit(
        self,
        *,
        state: RuntimeState,
        trace: RoundTrace,
        sampled_action: ActionSelection,
        rendered_expression: RenderedExpression,
        scenario: str,
        endogenous_turn: bool,
        control_ledger: dict[str, Any],
        recorded_at: str,
    ) -> RoundResult:
        health = HealthEvent(event="tick", status="ok", detail=f"budget={state.budget_remaining:.2f}")
        state.entropy_health_state = self.entropy_pool.health_snapshot()
        state.last_entropy_failure = {}
        sync_tick_writeback = self._should_sync_tick_writeback(
            round_id=state.round_count,
            scenario=scenario,
            mode=state.mode,
            endogenous_turn=endogenous_turn,
        )
        self._save_state(state, sync=sync_tick_writeback)
        self.trace_store.write_round(trace, sync=sync_tick_writeback)
        if dict(control_ledger.get("conflict", {}) or {}).get("post_error_adjustment", {}).get("triggered") and state.repair_ledger:
            latest_repair_entry = state.repair_ledger[-1]
            if latest_repair_entry.round_id == state.round_count:
                self.trace_store.append_repair_entry(
                    to_dict(latest_repair_entry),
                    session_id=state.session_id,
                    recorded_at=recorded_at,
                    sync=sync_tick_writeback,
                )
        self._maybe_flush_cold_path(
            scenario=scenario,
            mode=state.mode,
            endogenous_turn=endogenous_turn,
            raise_on_error=True,
        )
        if sampled_action.name == "self_run":
            pending_progress = self._autonomy_pending_readonly_repo_scan(state, state.autonomy_policy)
            if pending_progress is not None:
                self.resolve_run_tool_approval(
                    str(pending_progress.get("run_id") or ""),
                    str(pending_progress.get("call_id") or ""),
                    approved=True,
                )
            elif not (state.active_run_id and state.run_status in {"running", "paused"}):
                self.start_run(
                    self._autonomy_self_run_goal(state),
                    allow_commit=False,
                    operator_level="read_only",
                    include_details=False,
                    sync_hot_path=True,
                    defer_bootstrap_tool=True,
                )
            state = self.load_runtime_state()
        return RoundResult(
            round_id=state.round_count,
            sampled_action=sampled_action,
            trace=trace,
            state=state,
            health=health,
            rendered_expression=rendered_expression,
        )

    def tick(self, event: RoundEvent, scenario: str, mode: str) -> RoundResult:
        try:
            result = self._tick_impl(event, scenario, mode)
        except QuantumEntropyUnavailableError as exc:
            state = self.load_runtime_state()
            self._record_entropy_failure(state, exc)
            raise
        if event.source != "endogenous":
            self._maybe_schedule_endogenous_followup(result=result, event=event, scenario=scenario)
        return result

    def run_endogenous_tick(
        self,
        *,
        trigger: str = "idle",
        mode: str | None = None,
        scenario: str | None = None,
        trigger_context: EndogenousTriggerContext | None = None,
        trigger_payload: EndogenousTickTrigger | None = None,
        respect_suppression: bool = False,
    ) -> EndogenousTickPayload:
        state = self.load_runtime_state()
        self._ensure_subject_core(state)
        recent_rounds = self.trace_store.recent_rounds(limit=1)
        latest_round = recent_rounds[-1] if recent_rounds else {}
        effective_scenario = scenario or str(latest_round.get("scenario") or "companion")
        effective_context = trigger_context or self._build_endogenous_trigger_context(
            state=state,
            scenario=effective_scenario,
            trace_payload=latest_round,
        )
        scheduler_trigger = trigger_payload or self.endogenous_scheduler.build_trigger(
            state=state,
            context=effective_context.context,
            relation_state=effective_context.relation_state,
            slow_variables=effective_context.slow_variables,
            pool_state=state.motivation_pool_state,
        )
        if scheduler_trigger is None:
            scheduler_trigger = EndogenousTickTrigger(
                trigger_type=trigger,
                trigger_score=round(float(state.motivation_pool_state.endogenous_activation_score or 0.0), 6),
                source_metrics={key: round(float(value), 6) for key, value in effective_context.slow_variables.items()},
                selected_mode=mode or "endogenous_light",
                audit_reason=f"manual endogenous trigger fallback: {trigger}",
            )
        if trigger and trigger != scheduler_trigger.trigger_type:
            scheduler_trigger.trigger_type = trigger
            scheduler_trigger.audit_reason = f"manual trigger override: {trigger}"
        if mode is not None:
            scheduler_trigger.selected_mode = mode
        if respect_suppression:
            suppression = self._endogenous_suppression_decision(
                state=state,
                scenario=effective_scenario,
                trigger=scheduler_trigger,
            )
            if suppression.suppressed:
                state.endogenous_scheduler_state = self.endogenous_scheduler.update_state(
                    scheduler_state=state.endogenous_scheduler_state,
                    trigger=None,
                    suppression_reason=suppression.reason or None,
                )
                state.endogenous_state.last_suppression = suppression
                self._save_state(state, sync=True)
                return cast(EndogenousTickPayload, {
                    "round_id": None,
                    "micro_intent": {},
                    "cause_type": "suppressed",
                    "boundary_action": "allow_internal",
                    "trigger": to_dict(scheduler_trigger),
                    "selected_mode": scheduler_trigger.selected_mode,
                    "suppressed": True,
                    "suppression_reason": suppression.reason,
                    "trace_ref": None,
                    "initiative": {},
                })
        self._pending_endogenous_trigger = scheduler_trigger
        self._pending_endogenous_trigger_context = effective_context
        try:
            result = self._tick_view(
                RoundEvent(
                    source="endogenous",
                    content=f"endogenous trigger {scheduler_trigger.trigger_type}",
                    target="self",
                    cue=f"endogenous:{scheduler_trigger.trigger_type}",
                    cue_quality=0.45,
                ),
                scenario=effective_scenario,
                mode=scheduler_trigger.selected_mode or mode or "endogenous_light",
            )
        finally:
            self._pending_endogenous_trigger = None
            self._pending_endogenous_trigger_context = None
        micro_intent = to_dict(result.trace.micro_intent) if result.trace.micro_intent else {}
        initiative_payload = dict(getattr(result.trace, "initiative", {}) or {})
        auto_sent = False
        auto_session_id = None
        if initiative_payload:
            should_send = bool(initiative_payload.get("should_send"))
            if should_send:
                auto_sent, auto_session_id = self._dispatch_initiative_to_session(initiative_payload, result.rendered_expression.text)
            recorded_initiative = self._record_initiative_outcome(
                round_id=result.round_id,
                recorded_at=result.trace.recorded_at,
                proposal=initiative_payload,
                message=result.rendered_expression.text.strip(),
                auto_sent=auto_sent,
                session_id=auto_session_id,
            )
            initiative_payload = {
                **recorded_initiative,
                "auto_sent": auto_sent,
                "target_session_id": auto_session_id or initiative_payload.get("target_session_id"),
            }
        return cast(EndogenousTickPayload, {
            "round_id": result.round_id,
            "micro_intent": micro_intent,
            "cause_type": result.trace.cause_type,
            "boundary_action": result.trace.boundary_action,
            "trigger": to_dict(scheduler_trigger),
            "selected_mode": scheduler_trigger.selected_mode,
            "suppressed": False,
            "suppression_reason": "",
            "trace_ref": f"round://{result.round_id}",
            "initiative": initiative_payload,
        })
