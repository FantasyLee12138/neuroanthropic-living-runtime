from __future__ import annotations

import copy
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
from typing import Any
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
MODEL_STATUS_RECENT_SKILL_WINDOW = 512

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nalr.runtime.controller import RuntimeController


class _ControllerBackedRuntime:
    def __init__(self, controller: "RuntimeController") -> None:
        self.controller = controller

    def __getattr__(self, name: str):
        return getattr(self.controller, name)


class ObserverRuntimeService(_ControllerBackedRuntime):
    def _trace_round_view(self, round_ref: int | str) -> dict[str, Any]:
        patched = getattr(self.controller, "__dict__", {}).get("trace_round")
        if callable(patched):
            return patched(round_ref)
        return self.trace_round(round_ref)

    def _trace_storage_payload(self, *, read_source: str | None = None) -> dict[str, Any]:
        payload = dict(self.trace_store.trace_storage_status())
        if read_source is not None:
            payload["read_source"] = read_source
        return payload

    def trace_storage_status(self, *, read_source: str | None = None) -> dict[str, Any]:
        return self._trace_storage_payload(read_source=read_source)

    def resolve_round_ref(self, round_ref: int | str | None) -> int:
        if round_ref is None:
            round_id = self.load_runtime_state().round_count
            if round_id <= 0:
                raise FileNotFoundError("no trace rounds recorded yet")
            return round_id
        if isinstance(round_ref, int):
            return round_ref
        normalized_round_ref = round_ref.strip().lower()
        if normalized_round_ref in {"last", "latest"}:
            round_id = self.load_runtime_state().round_count
            if round_id <= 0:
                raise FileNotFoundError("no trace rounds recorded yet")
            return round_id
        try:
            return int(normalized_round_ref)
        except ValueError as exc:
            raise ValueError(f"invalid round reference: {round_ref}") from exc

    def _dream_summary_from_trace(self, trace: dict[str, Any]) -> dict[str, Any] | None:
        run_id = trace.get("dream_run_id")
        if not run_id:
            return None
        return {
            "run_id": run_id,
            "trigger": trace.get("dream_trigger"),
            "trace_ref": trace.get("dream_trace_ref"),
            "guard_summary": trace.get("dream_guard_summary", {}),
            "effect_summary": trace.get("dream_effect_summary", {}),
        }

    def dream_runs(self) -> dict[str, Any]:
        return {"runs": self.dream_orchestrator.list_runs()}

    def dream_status(self) -> dict[str, Any]:
        return self.dream_orchestrator.status(self.load_runtime_state())

    def dream_trace(self, run_ref: int | str | None) -> dict[str, Any]:
        if isinstance(run_ref, int):
            trace = self.trace_round(run_ref)
            summary = self._dream_summary_from_trace(trace)
            if summary is None:
                raise FileNotFoundError(f"no dream run recorded for round {run_ref}")
            payload = self.dream_orchestrator.read_run(summary["run_id"])
        else:
            payload = self.dream_orchestrator.read_run(str(run_ref) if run_ref is not None else None)
        trace = payload.get("trace", {})
        return {
            **payload,
            "dream_run_id": trace.get("dream_run_id"),
            "trigger": trace.get("trigger"),
            "mode": trace.get("mode"),
            "trace_ref": f"dream://runs/{trace.get('dream_run_id')}",
        }

    def dream_proposals(self, run_ref: int | str | None) -> dict[str, Any]:
        payload = self.dream_trace(run_ref)
        return {
            "dream_run_id": payload["trace"]["dream_run_id"],
            "trigger": payload["trace"]["trigger"],
            "proposal_bundle": payload["proposal_bundle"],
            "guard_summary": payload.get("guard_summary", {}),
            "effect_summary": payload.get("effect_summary", {}),
        }

    def dream_metrics(self) -> dict[str, Any]:
        return self.dream_orchestrator.metrics()

    def _dream_semantic_summary(self, payload: dict[str, Any]) -> dict[str, Any]:
        trace = payload.get("trace", {}) if isinstance(payload.get("trace"), dict) else {}
        guard = payload.get("guard_summary", {}) if isinstance(payload.get("guard_summary"), dict) else {}
        effect = payload.get("effect_summary", {}) if isinstance(payload.get("effect_summary"), dict) else {}
        proposal_bundle = payload.get("proposal_bundle", {}) if isinstance(payload.get("proposal_bundle"), dict) else {}
        proposal_counts = trace.get("proposal_counts", {}) if isinstance(trace.get("proposal_counts"), dict) else {}
        if proposal_counts:
            proposal_total = sum(max(0, int(value or 0)) for value in proposal_counts.values())
        else:
            proposal_total = sum(len(value) for value in proposal_bundle.values() if isinstance(value, list))
        return {
            "trigger": str(trace.get("trigger") or ""),
            "mode": str(trace.get("mode") or ""),
            "route": str(trace.get("route") or ""),
            "degraded": bool(trace.get("degraded")),
            "cue": trace.get("cue"),
            "dominant_emotion": trace.get("dominant_emotion"),
            "evaluated_types": list(trace.get("evaluated_types", []) or guard.get("evaluated_types", []) or []),
            "allowed_types": list(guard.get("allowed_types", []) or []),
            "rejected_types": list(guard.get("rejected_types", []) or []),
            "applied_types": list(effect.get("applied_types", []) or []),
            "proposal_total": int(proposal_total),
            "proposal_counts": dict(proposal_counts),
            "approved": bool(guard.get("approved")),
            "applied": bool(effect.get("applied")),
        }

    def dream_overview(self, *, limit: int = 6) -> dict[str, Any]:
        normalized_limit = max(1, int(limit or 1))
        status = self.dream_status()
        metrics = self.dream_metrics()
        runs = list(self.dream_runs().get("runs", []) or [])
        recent_runs: list[dict[str, Any]] = []
        for item in runs[:normalized_limit]:
            trace = item.get("trace", {}) if isinstance(item, dict) else {}
            run_payload = {
                "dream_run_id": trace.get("dream_run_id"),
                "trigger": trace.get("trigger"),
                "mode": trace.get("mode"),
                "trace_ref": f"dream://runs/{trace.get('dream_run_id')}" if trace.get("dream_run_id") else "",
                "trace": trace,
                "proposal_bundle": item.get("proposal_bundle", {}) if isinstance(item, dict) else {},
                "guard_summary": item.get("guard_summary", {}) if isinstance(item, dict) else {},
                "effect_summary": item.get("effect_summary", {}) if isinstance(item, dict) else {},
            }
            run_payload["semantic_summary"] = self._dream_semantic_summary(run_payload)
            recent_runs.append(run_payload)
        latest_run: dict[str, Any] | None = None
        latest_run_id = status.get("latest_run_id")
        if latest_run_id:
            try:
                latest_payload = self.dream_trace(str(latest_run_id))
                latest_run = {
                    "dream_run_id": latest_payload.get("dream_run_id"),
                    "trigger": latest_payload.get("trigger"),
                    "mode": latest_payload.get("mode"),
                    "trace_ref": latest_payload.get("trace_ref"),
                    "trace": latest_payload.get("trace", {}),
                    "proposal_bundle": latest_payload.get("proposal_bundle", {}),
                    "guard_summary": latest_payload.get("guard_summary", {}),
                    "effect_summary": latest_payload.get("effect_summary", {}),
                }
                latest_run["semantic_summary"] = self._dream_semantic_summary(latest_run)
            except FileNotFoundError:
                latest_run = recent_runs[0] if recent_runs else None
        elif recent_runs:
            latest_run = recent_runs[0]
        return {
            "status": status,
            "metrics": metrics,
            "latest_run": latest_run,
            "recent_runs": recent_runs,
        }

    def console_round_or_none(self) -> int | None:
        return self._console_round_or_none()

    def console_default_why_not_action(
        self,
        round_ref: int | str | None,
        *,
        action_field: dict[str, Any] | None = None,
    ) -> str | None:
        field = action_field or self.console_action_field(round_ref)
        winner_action = str((field.get("winner") or {}).get("action") or "").strip()
        for entry in list(field.get("competing_peaks", []) or []):
            action = str((entry or {}).get("action") or "").strip()
            if action and action != winner_action:
                return action
        for entry in list(field.get("top_actions", []) or []):
            action = str((entry or {}).get("action") or "").strip()
            if action and action != winner_action:
                return action
        return None

    def console_recent_rounds(self, *, limit: int = 12) -> list[dict[str, Any]]:
        return self._console_recent_rounds(limit=limit)

    def console_source_links(
        self,
        *,
        round_id: int | None,
        trace_ref: str | None,
        why_not_action: str | None,
    ) -> list[dict[str, Any]]:
        return self._console_source_links(
            round_id=round_id,
            trace_ref=trace_ref,
            why_not_action=why_not_action,
        )

    def _console_round_or_none(self) -> int | None:
        round_count = int(self.load_runtime_state().round_count or 0)
        return round_count if round_count > 0 else None

    def _why_not_summary(
        self,
        *,
        action: str,
        selected_action: str,
        candidate_score: float,
        blocked_by: list[str],
        expressive_trace: dict[str, Any],
    ) -> str:
        normalized_action = str(action or "").strip()
        selected = str(selected_action or "").strip()
        if not normalized_action:
            return "没有候选动作。"
        if normalized_action == "monologue":
            monologue_stream = dict(expressive_trace.get("monologue_stream", {}) or {})
            fragment_count = int(monologue_stream.get("recent_fragment_count", 0) or 0)
            if selected == "monologue":
                return f"这轮独白胜出，隐藏独白流提供了 {fragment_count} 条最近碎片作为内部表达压力。"
            blocker_text = "、".join(blocked_by[:3]) if blocked_by else "其他动作后验更高"
            return f"这轮独白没有胜出；虽然隐藏独白流仍在提供 {fragment_count} 条碎片，但最终被 {selected or '其他动作'} 压过。阻力主要来自 {blocker_text}。"
        if selected == normalized_action:
            return f"{normalized_action} 就是本轮最终动作。"
        blocker_text = "、".join(blocked_by[:3]) if blocked_by else "没有明确阻断，只是后验不足"
        return f"{normalized_action} 没有胜出，最终由 {selected or '其他动作'} 占优。当前候选分数为 {round(float(candidate_score or 0.0), 4)}，主要阻力是 {blocker_text}。"

    def _console_default_why_not_action(
        self,
        round_ref: int | str | None,
        *,
        action_field: dict[str, Any] | None = None,
    ) -> str | None:
        field = action_field or self.console_action_field(round_ref)
        winner_action = str((field.get("winner") or {}).get("action") or "").strip()
        for entry in field.get("competing_peaks", []):
            action = str((entry or {}).get("action") or "").strip()
            if action and action != winner_action:
                return action
        for entry in field.get("top_actions", []):
            action = str((entry or {}).get("action") or "").strip()
            if action and action != winner_action:
                return action
        return None

    def _console_recent_rounds(self, *, limit: int = 12) -> list[dict[str, Any]]:
        rows = self.trace_store.recent_rounds(limit=limit)
        return [
            {
                "round_id": row["round_id"],
                "sampled_action": row["sampled_action"],
                "mode": row["mode"],
                "budget_remaining": row.get("state_snapshot", {}).get("budget_remaining"),
                "conflict_score": (row.get("conflict_arbitration", {}) or {}).get("total_score", 0.0),
                "cause_type": row.get("cause_type", "external_stimulus"),
            }
            for row in rows
        ]

    def _console_source_links(
        self,
        *,
        round_id: int | None,
        trace_ref: str | None,
        why_not_action: str | None,
    ) -> list[dict[str, Any]]:
        replay_path = f"/replay/{round_id}" if round_id is not None else "/replay/{round_id}"
        why_not_path = f"/console/why-not/{why_not_action}" if why_not_action else "/console/why-not/{action}"

        def link(panel_id: str, api_path: str, controller_method: str) -> dict[str, Any]:
            return {
                "panel_id": panel_id,
                "api_path": api_path,
                "controller_method": controller_method,
                "trace_ref": trace_ref,
                "round_id": round_id,
            }

        return [
            link("brain_state", "/console/state", "RuntimeController.console_state"),
            link("body_state", "/console/state", "RuntimeController.cognitive_snapshot"),
            link("subjective_state", "/console/state", "RuntimeController.cognitive_snapshot"),
            link("organic_mode", "/console/state", "RuntimeController.cognitive_snapshot"),
            link("emergent_action_sketches", "/console/state", "RuntimeController.cognitive_snapshot"),
            link("speak_channel", f"/speak/{round_id}" if round_id is not None else "/speak/last", "RuntimeController.expression_channel_snapshot"),
            link("monologue_channel", f"/monologue/{round_id}" if round_id is not None else "/monologue/last", "RuntimeController.expression_channel_snapshot"),
            link("autonomy_status", "/autonomy/status", "RuntimeController.autonomy_status"),
            link("autonomy_recent", "/autonomy/status", "RuntimeController.autonomy_status"),
            link("autonomy_policy", "/autonomy/status", "RuntimeController.autonomy_status"),
            link("action_field", "/console/action-field", "RuntimeController.console_action_field"),
            link("probability_layers", "/trace/probability/latest", "RuntimeController.trace_probability_field"),
            link("timeline", "/console/timeline", "RuntimeController.console_timeline"),
            link("why_current", "/console/why/current", "RuntimeController.console_why_current"),
            link("why_not", why_not_path, "RuntimeController.console_why_not"),
            link("instinct_space", "/console/state", "RuntimeController.cognitive_snapshot"),
            link("counterfactual_replay", replay_path, "RuntimeController.replay"),
        ]

    def console_state(self) -> dict[str, Any]:
        return self.diagnostics_runtime.console_state()

    def console_action_field(self, round_ref: int | str | None = None) -> dict[str, Any]:
        effective_round = self.resolve_round_ref(round_ref)
        why_override = getattr(self.__dict__.get("controller"), "__dict__", {}).get("why_this")
        if callable(why_override):
            why_payload = why_override(effective_round)
        else:
            why_payload = self.why_this(effective_round)
        initiative_payload = dict(why_payload.get("initiative", {}) or {})
        explanation = dict(why_payload.get("action_probability_explanation", {}) or {})
        winner_target = explanation.get("winner_target") or why_payload.get("sampled_action")
        winner_posterior = dict(explanation.get("winner_posterior", {}) or {})
        competing_peaks = list(explanation.get("competing_peaks", []) or [])
        contribution_stack = list(explanation.get("stacked_contributions", []) or [])
        if not contribution_stack:
            contribution_override = getattr(self.__dict__.get("controller"), "__dict__", {}).get("contribution_breakdown")
            if callable(contribution_override):
                contribution_payload = contribution_override(effective_round)
                if isinstance(contribution_payload, dict):
                    contribution_stack = list(contribution_payload.get("contributions", []) or [])
            if not contribution_stack:
                try:
                    trace = self.trace_round(effective_round)
                except FileNotFoundError:
                    trace = None
                if isinstance(trace, dict):
                    contribution_stack = self._stacked_action_contributions(
                        self._action_layer_from_trace(trace),
                        str(winner_target or ""),
                    )
        top_actions = [
            {
                "action": str(action),
                "score": float(score),
            }
            for action, score in sorted(winner_posterior.items(), key=lambda item: item[1], reverse=True)[:5]
        ]
        if not top_actions and winner_target:
            top_actions.append({"action": str(winner_target), "score": 0.0})
        return {
            "round_id": why_payload["round_id"],
            "trace_ref": f"round://{why_payload['round_id']}",
            "top_actions": top_actions,
            "winner": {
                "action": winner_target,
                "score": winner_posterior.get(winner_target, 0.0) if winner_target else 0.0,
            },
            "winner_posterior": winner_posterior,
            "conflict": why_payload.get("conflict_arbitration", {}),
            "token_field": dict(why_payload.get("token_state", {}) or {}),
            "contribution_stack": contribution_stack,
            "competing_peaks": competing_peaks,
            "expressive": {
                "expression_mode": initiative_payload.get("expression_mode"),
                "proposal_type": initiative_payload.get("proposal_type"),
                "top_intent": initiative_payload.get("top_intent"),
                "speech_cost": initiative_payload.get("speech_cost"),
                "intrinsic_value": initiative_payload.get("intrinsic_value"),
                "should_send": initiative_payload.get("should_send"),
                "suppression_reason": initiative_payload.get("suppression_reason"),
                "grounded_in": initiative_payload.get("grounded_in", {}),
                "memory_backing": dict(initiative_payload.get("memory_backing", {}) or {}),
            },
        }

    def console_timeline(self, round_ref: int | str | None = None) -> dict[str, Any]:
        effective_round = self.resolve_round_ref(round_ref)
        trace = self._trace_round_view(effective_round)
        events: list[dict[str, Any]] = [
            {
                "type": "stimulus",
                "label": str(trace.get("scenario") or trace.get("cause_type") or "runtime_input"),
                "summary": str(trace.get("cue") or trace.get("sampled_action") or "收到本轮刺激"),
            }
        ]
        if trace.get("top_drivers"):
            events.append(
                {
                    "type": "memory_activation",
                    "label": "top_drivers",
                    "summary": ", ".join(
                        str(item.get("agent_name") or item.get("module_name") or "unknown")
                        for item in list(trace.get("top_drivers", []) or [])[:3]
                    ),
                }
            )
        motivation_pool = dict(trace.get("motivation_pool", {}) or {})
        active_motivations = list(motivation_pool.get("active_motivations", []) or [])
        if active_motivations:
            events.append(
                {
                    "type": "motivation_rise",
                    "label": "motivation_pool",
                    "summary": f"active={len(active_motivations)}",
                }
            )
        events.append(
            {
                "type": "action_arbitration",
                "label": str(trace.get("sampled_action") or "unknown"),
                "summary": str(trace.get("conflict_arbitration", {}).get("winning_priority") or "winner_selected"),
            }
        )
        token_state = dict(trace.get("token_state", {}) or {})
        if token_state or trace.get("renderer_decision_integrity"):
            events.append(
                {
                    "type": "token_gate",
                    "label": "token_state",
                    "summary": str(trace.get("renderer_decision_integrity", {}).get("verdict") or "token_coupling_checked"),
                }
            )
        if trace.get("cause_type") == "endogenous" or trace.get("endogenous_tick_reason"):
            events.append(
                {
                    "type": "endogenous_tick",
                    "label": str(trace.get("cause_type") or "endogenous"),
                    "summary": str(dict(trace.get("endogenous_tick_reason", {}) or {}).get("trigger_type") or "internal_trigger"),
                }
            )
        if trace.get("dream_run_id"):
            events.append(
                {
                    "type": "dream_effect",
                    "label": str(trace.get("dream_trigger") or "dream"),
                    "summary": str(trace.get("dream_effect_summary") or "dream_effect_observed"),
                }
            )
        return {
            "round_id": trace["round_id"],
            "trace_ref": trace.get("trace_ref"),
            "events": events,
        }

    def console_why_current(self, round_ref: int | str | None = None) -> dict[str, Any]:
        effective_round = self.resolve_round_ref(round_ref)
        why_override = getattr(self.__dict__.get("controller"), "__dict__", {}).get("why_this")
        if callable(why_override):
            payload = why_override(effective_round)
        else:
            payload = self.why_this(effective_round)
        initiative_payload = dict(payload.get("initiative", {}) or {})
        expressive_trace = dict(payload.get("expressive_trace", initiative_payload) or {})
        return {
            "round_id": payload["round_id"],
            "trace_ref": f"round://{payload['round_id']}",
            "why": {
                "summary": str(payload.get("sampled_action") or "暂无"),
                "sampled_action": payload.get("sampled_action"),
                "top_drivers": payload.get("top_drivers", []),
                "vitality_snapshot": payload.get("vitality_snapshot", {}),
                "authenticity": payload.get("authenticity", {}),
                "initiative": {
                    "expression_mode": initiative_payload.get("expression_mode"),
                    "proposal_type": initiative_payload.get("proposal_type"),
                    "top_intent": initiative_payload.get("top_intent"),
                    "speech_cost": initiative_payload.get("speech_cost"),
                    "intrinsic_value": initiative_payload.get("intrinsic_value"),
                    "should_send": initiative_payload.get("should_send"),
                    "suppression_reason": initiative_payload.get("suppression_reason"),
                    "memory_backing": dict(initiative_payload.get("memory_backing", {}) or {}),
                },
                "expressive_trace": expressive_trace,
            },
        }

    def console_why_not(self, action: str, round_ref: int | str | None = None) -> dict[str, Any]:
        effective_round = self.resolve_round_ref(round_ref)
        payload = self.why_not(effective_round, action)
        return {
            "round_id": payload["round_id"],
            "trace_ref": f"round://{payload['round_id']}",
            "action": payload["action"],
            "why_not": {
                "selected_action": payload.get("selected_action"),
                "candidate_score": payload.get("candidate_score"),
                "blocked_by": payload.get("blocked_by", []),
                "competing_peaks": payload.get("competing_peaks", []),
                "stacked_contributions": payload.get("stacked_contributions", []),
                "expressive_trace": payload.get("expressive_trace", {}),
                "summary": payload.get("summary", ""),
            },
        }

    def console_refresh_payload(self, round_ref: int | str | None = None) -> dict[str, Any]:
        return self.diagnostics_runtime.console_refresh_payload(round_ref)

    def console_recent_actions(self, *, limit: int = 8) -> dict[str, Any]:
        return self.diagnostics_runtime.console_recent_actions(limit=limit)

    def console_probability_space(self, round_ref: int | str | None = None) -> dict[str, Any]:
        return self.diagnostics_runtime.console_probability_space(round_ref)

    def trace_round(self, round_ref: int | str) -> dict[str, Any]:
        resolved_round = self.resolve_round_ref(round_ref)
        reader_override = getattr(self.trace_store, "__dict__", {}).get("read_round_record")
        reader_signature = (
            ("instance", id(reader_override))
            if reader_override is not None
            else ("class", id(type(self.trace_store).read_round_record))
        )
        cached = self._trace_round_cache.get(resolved_round)
        if cached is not None and tuple(cached.get("__trace_cache_reader_signature") or ()) == reader_signature:
            cached_payload = copy.deepcopy(cached)
            cached_payload.pop("__trace_cache_reader_signature", None)
            return cached_payload

        self.trace_store.flush(raise_on_error=False)
        payload = None
        read_source = None
        if reader_override is None:
            storage_status = self.trace_store.trace_storage_status()
            if storage_status["storage_state"] == "healthy":
                parquet_payload = self.trace_store._read_round_from_parquet(resolved_round)
                if parquet_payload is not None:
                    payload, read_source = parquet_payload, "parquet"
            else:
                json_payload = self.trace_store._read_round_from_json(resolved_round)
                if json_payload is not None:
                    payload, read_source = json_payload, "json_fallback"
        if payload is None:
            read_result = self.trace_store.read_round_record(resolved_round)
            if isinstance(read_result, tuple) and len(read_result) == 2:
                payload, read_source = read_result
            else:
                payload, read_source = read_result, None
        self._trace_derived_cache.pop(resolved_round, None)
        enriched = dict(payload)
        enriched.setdefault("motivation_pool", {})
        enriched.setdefault("motivation_feedback", {})
        enriched.setdefault("endogenous_tick_reason", {})
        enriched.setdefault("endogenous_policy_shift", {})
        enriched.setdefault("endogenous_trigger_context", {})
        enriched.setdefault("endogenous_suppression", {})
        enriched.setdefault("micro_intent", {})
        enriched.setdefault("endogenous_replay_chain", {})
        enriched["token_state"] = self._token_state_from_trace(enriched)
        enriched["cross_layer_coupling_verdict"] = self._cross_layer_coupling_verdict(enriched)
        enriched["renderer_decision_integrity"] = self._renderer_decision_integrity(enriched)
        enriched["conflict_arbitration"] = self._conflict_arbitration_summary(enriched)
        enriched["storage"] = self._trace_storage_payload(read_source=read_source)
        enriched["trace_ref"] = f"round://{enriched['round_id']}"
        dream = self._dream_summary_from_trace(enriched)
        if dream is not None:
            enriched["dream_run_id"] = dream["run_id"]
            enriched["dream_trigger"] = dream["trigger"]
            enriched["dream_trace_ref"] = dream["trace_ref"]
            enriched["dream_guard_summary"] = dream["guard_summary"]
            enriched["dream_effect_summary"] = dream["effect_summary"]
        enriched["__trace_cache_reader_signature"] = list(reader_signature)
        self._trace_round_cache[resolved_round] = copy.deepcopy(enriched)
        if len(self._trace_round_cache) > self._trace_round_cache_limit:
            for stale_round_id in sorted(self._trace_round_cache)[:-self._trace_round_cache_limit]:
                self._trace_round_cache.pop(stale_round_id, None)
                self._trace_derived_cache.pop(stale_round_id, None)
        enriched.pop("__trace_cache_reader_signature", None)
        return enriched

    def empty_trace_payload(self, round_ref: int | str | None = None) -> dict[str, Any]:
        round_id = round_ref if isinstance(round_ref, int) else None
        return {
            "round_id": round_id,
            "trace_ref": None,
            "sampled_action": "nothing",
            "top_drivers": [],
            "style_profile": {},
            "state_snapshot": {
                "mode": self.load_runtime_state().mode,
                "safe_mode": self.load_runtime_state().safe_mode,
                "focus": self.load_runtime_state().focus,
                "budget_remaining": self.load_runtime_state().budget_remaining,
            },
            "probability_field": {},
            "token_state": {},
            "cross_layer_coupling_verdict": {},
            "renderer_decision_integrity": {},
            "conflict_arbitration": {},
            "storage": self._trace_storage_payload(),
            "message": "无决策记录",
        }

    def why_this(self, round_ref: int | str) -> dict[str, Any]:
        trace = self._trace_round_view(round_ref)
        state_snapshot = dict(trace.get("state_snapshot", {}) or {})
        personality_anchor = trace.get("state_snapshot", {}).get("personality_anchor", {})
        instinct_field = trace.get("state_snapshot", {}).get("instinct_field", {})
        emergent_sketches = trace.get("state_snapshot", {}).get("emergent_action_sketches", [])
        return {
            "round_id": trace["round_id"],
            "cause_type": trace.get("cause_type"),
            "mode": trace.get("mode"),
            "sampled_action": trace["sampled_action"],
            "top_drivers": trace["top_drivers"],
            "style_profile": trace["style_profile"],
            "stochastic_state": trace.get("stochastic_state", {}),
            "render_plan": trace.get("render_plan", {}),
            "rendered_expression": trace.get("rendered_expression", {}),
            "authenticity": trace.get("authenticity", trace.get("rendered_expression", {}).get("authenticity", {})),
            "identity_evolution": trace.get("identity_evolution", {}),
            "vitality_snapshot": trace.get("vitality_snapshot", {}),
            "vitality_events": trace.get("vitality_events", []),
            "subjective_state": trace.get("state_snapshot", {}).get("subjective_state", {}),
            "emotion_state": trace.get("state_snapshot", {}).get("emotion_state", {}),
            "desire_state": trace.get("state_snapshot", {}).get("desire_state", {}),
            "instinct_field": instinct_field,
            "organic_mode": trace.get("state_snapshot", {}).get("organic_mode", {}),
            "emergent_action_sketches": emergent_sketches,
            "personality_anchor": personality_anchor,
            "self_continuity_derivation": dict(personality_anchor.get("continuity_derivation", {}) or {}),
            "high_dimensional_collapse": dict(instinct_field.get("collapse_trace", {}) or {}),
            "anchor_alignment": round(float(personality_anchor.get("alignment", 0.0) or 0.0), 4),
            "emergent_action_formalization": self._emergent_action_formalization_payload(emergent_sketches),
            "long_run_projection": trace.get("long_run_projection", {}),
            "motivation_pool": trace.get("motivation_pool", {}),
            "motivation_feedback": trace.get("motivation_feedback", {}),
            "initiative": trace.get("initiative", {}),
            "expressive_trace": trace.get("expressive_trace", trace.get("initiative", {})),
            "micro_intent": trace.get("micro_intent", {}),
            "endogenous_tick_reason": trace.get("endogenous_tick_reason", {}),
            "endogenous_policy_shift": trace.get("endogenous_policy_shift", {}),
            "appraisal_snapshot": trace.get("appraisal_snapshot", {}),
            "state_delta_before_clip": trace.get("state_delta_before_clip", {}),
            "state_delta_after_clip": trace.get("state_delta_after_clip", {}),
            "delta_suppression_reason": trace.get("delta_suppression_reason", []),
            "run_context": trace.get("run_context", {}),
            "run_contamination_detected": trace.get("run_contamination_detected", False),
            "identity_evidence_score": trace.get("identity_evidence_score", 0.0),
            "identity_trigger_blockers": trace.get("identity_trigger_blockers", []),
            "temperament_window_summary": trace.get("temperament_window_summary", {}),
            "probability_field": trace.get("probability_field", {}),
            "token_state": trace.get("token_state", {}),
            "cross_layer_coupling_verdict": trace.get("cross_layer_coupling_verdict", {}),
            "renderer_decision_integrity": trace.get("renderer_decision_integrity", {}),
            "memory_write_gate": trace.get("memory_write_gate", {}),
            "failure_taxonomy": self._failure_taxonomy_from_trace(trace),
            "conflict_arbitration": trace.get("conflict_arbitration", self._conflict_arbitration_summary(trace)),
            "action_probability_explanation": self._action_probability_explanation(
                trace,
                target_action=str(trace.get("sampled_action", "")),
            ),
            "counterfactual_replays": self._counterfactual_replays_from_trace(trace),
            "dream": self._dream_summary_from_trace(trace),
            "state_snapshot": {
                "mode": state_snapshot.get("mode", ""),
                "safe_mode": bool(state_snapshot.get("safe_mode", False)),
                "focus": state_snapshot.get("focus", ""),
                "budget_remaining": state_snapshot.get("budget_remaining", 0.0),
            },
            "storage": trace.get("storage", self._trace_storage_payload()),
        }

    def empty_why_payload(self, round_ref: int | str | None = None) -> dict[str, Any]:
        state = self.load_runtime_state()
        self._sync_tlh_state(state)
        return {
            "round_id": round_ref if isinstance(round_ref, int) else None,
            "sampled_action": "nothing",
            "top_drivers": [],
            "style_profile": {},
            "stochastic_state": {},
            "render_plan": {},
            "rendered_expression": {},
            "authenticity": {},
            "identity_evolution": {},
            "vitality_snapshot": {},
            "vitality_events": [],
            "subjective_state": to_dict(state.subjective_state),
            "emotion_state": to_dict(state.emotion_state),
            "desire_state": to_dict(state.desire_state),
            "instinct_field": to_dict(state.instinct_field),
            "organic_mode": to_dict(state.organic_mode),
            "emergent_action_sketches": [],
            "personality_anchor": to_dict(state.personality_anchor),
            "self_continuity_derivation": {},
            "high_dimensional_collapse": {},
            "anchor_alignment": round(float(state.personality_anchor.alignment or 0.0), 4),
            "emergent_action_formalization": {"sketches": [], "active_count": 0, "candidate_targets": []},
            "long_run_projection": {},
            "motivation_pool": {},
            "motivation_feedback": {},
            "initiative": {},
            "expressive_trace": {},
            "endogenous_tick_reason": {},
            "endogenous_policy_shift": {},
            "appraisal_snapshot": {},
            "state_delta_before_clip": {},
            "state_delta_after_clip": {},
            "delta_suppression_reason": [],
            "run_context": {},
            "run_contamination_detected": False,
            "identity_evidence_score": 0.0,
            "identity_trigger_blockers": [],
            "temperament_window_summary": {},
            "probability_field": {},
            "token_state": {},
            "cross_layer_coupling_verdict": {},
            "renderer_decision_integrity": {},
            "memory_write_gate": {},
            "failure_taxonomy": [],
            "conflict_arbitration": {},
            "action_probability_explanation": {},
            "counterfactual_replays": [],
            "dream": None,
            "state_snapshot": {
                "mode": state.mode,
                "safe_mode": state.safe_mode,
                "focus": state.focus,
                "budget_remaining": state.budget_remaining,
            },
            "storage": self._trace_storage_payload(),
            "message": "无决策记录",
        }

    def empty_expression_channel_payload(self, channel: str, round_ref: int | str | None = None) -> dict[str, Any]:
        normalized = "monologue" if str(channel or "").strip().lower() == "monologue" else "speak"
        preview_action = "monologue" if normalized == "monologue" else "respond"
        delivery_mode = "monologue" if normalized == "monologue" else "speech"
        return {
            "round_id": round_ref if isinstance(round_ref, int) else None,
            "trace_ref": None,
            "channel": normalized,
            "delivery_mode": delivery_mode,
            "preview_action": preview_action,
            "sampled_action": "nothing",
            "active": False,
            "posterior": 0.0,
            "actual": {
                "action": "nothing",
                "delivery_mode": delivery_mode,
                "text": "",
                "route": "",
                "model": "",
                "degraded": False,
            },
            "preview": {
                "action": preview_action,
                "delivery_mode": delivery_mode,
                "text": "",
                "would_output": False,
                "terminal_intent": False,
            },
            "initiative": {},
            "message": "无决策记录",
        }

    def expression_channel_snapshot(self, channel: str, round_ref: int | str = "last") -> dict[str, Any]:
        normalized = "monologue" if str(channel or "").strip().lower() == "monologue" else "speak"
        preview_action = "monologue" if normalized == "monologue" else "respond"
        expected_delivery_mode = "monologue" if normalized == "monologue" else "speech"
        trace_override = getattr(self.__dict__.get("controller"), "__dict__", {}).get("trace_round")
        if callable(trace_override):
            trace = trace_override(round_ref)
        else:
            trace = self._trace_round_view(round_ref)
        rendered_expression = dict(trace.get("rendered_expression", {}) or {})
        action_explanation = self._action_probability_explanation(trace, target_action=preview_action)
        initiative = dict(trace.get("initiative", {}) or {})
        actual_delivery_mode = str(rendered_expression.get("delivery_mode") or "speech")
        actual_action = str(trace.get("sampled_action") or "")
        actual_active = actual_delivery_mode == expected_delivery_mode and bool(rendered_expression.get("text"))
        preview = self._counterfactual_render_preview(trace, preview_action)
        return {
            "round_id": int(trace.get("round_id", 0) or 0),
            "trace_ref": str(trace.get("trace_ref") or f"round://{int(trace.get('round_id', 0) or 0)}"),
            "channel": normalized,
            "delivery_mode": expected_delivery_mode,
            "preview_action": preview_action,
            "sampled_action": actual_action,
            "active": actual_active,
            "posterior": round(float(action_explanation.get("winner_posterior", {}).get(preview_action, 0.0) or 0.0), 6),
            "actual": {
                "action": actual_action,
                "delivery_mode": actual_delivery_mode,
                "text": str(rendered_expression.get("text") or ""),
                "route": str(rendered_expression.get("route") or ""),
                "model": str(rendered_expression.get("model") or ""),
                "degraded": bool(rendered_expression.get("degraded", False)),
            },
            "preview": {
                "action": str(preview.get("action") or preview_action),
                "delivery_mode": expected_delivery_mode,
                "text": str(preview.get("text") or ""),
                "would_output": bool(preview.get("would_output", False)),
                "terminal_intent": bool(preview.get("terminal_intent", False)),
            },
            "initiative": {
                "proposal_type": str(initiative.get("proposal_type") or ""),
                "expression_mode": str(initiative.get("expression_mode") or ""),
                "top_intent": str(initiative.get("top_intent") or ""),
                "speech_cost": round(float(initiative.get("speech_cost", 0.0) or 0.0), 6),
                "intrinsic_value": round(float(initiative.get("intrinsic_value", 0.0) or 0.0), 6),
                "should_send": bool(initiative.get("should_send", False)),
                "suppression_reason": str(initiative.get("suppression_reason") or ""),
                "grounded_in": dict(initiative.get("grounded_in", {}) or {}),
                "memory_backing": dict(initiative.get("memory_backing", {}) or {}),
            },
            "message": "",
        }

    def contribution_breakdown(self, round_ref: int | str) -> dict[str, Any]:
        trace = self._trace_round_view(round_ref)
        conflict = dict(trace.get("conflict_arbitration", {}) or {})
        return {
            "round_id": trace["round_id"],
            "sampled_action": trace["sampled_action"],
            "contributions": trace["contributions"],
            "repair": {
                "mode": conflict.get("repair_mode"),
                "stage": conflict.get("repair_state_snapshot", {}).get("stage", "idle"),
                "transition": conflict.get("repair_transition", {}),
                "post_error_adjustment": conflict.get("post_error_adjustment", {}),
                "ledger_tail": conflict.get("repair_ledger_tail", []),
            },
            "storage": trace.get("storage", self._trace_storage_payload()),
        }

    def trace_probability_field(self, round_ref: int | str) -> dict[str, Any]:
        trace = self._trace_round_view(round_ref)
        return {
            "round_id": trace["round_id"],
            "sampled_action": trace["sampled_action"],
            "probability_field": trace.get("probability_field", {}),
            "token_state": trace.get("token_state", self._token_state_from_trace(trace)),
            "cross_layer_coupling_verdict": trace.get(
                "cross_layer_coupling_verdict",
                self._cross_layer_coupling_verdict(trace),
            ),
            "storage": trace.get("storage", self._trace_storage_payload()),
        }

    def trace_probability_layer(self, round_ref: int | str, *, layer: str) -> dict[str, Any]:
        normalized_layer = str(layer or "").strip().lower()
        if normalized_layer not in {"context", "memory", "action", "token"}:
            raise ValueError(f"unsupported probability layer: {layer}")
        trace = self._trace_round_view(round_ref)
        probability_field = dict(trace.get("probability_field", {}) or {})
        layer_state = probability_field.get(normalized_layer, {})
        return {
            "round_id": trace["round_id"],
            "sampled_action": trace["sampled_action"],
            "layer": normalized_layer,
            "layer_state": layer_state if isinstance(layer_state, dict) else {},
            "token_state": trace.get("token_state", self._token_state_from_trace(trace)),
            "storage": trace.get("storage", self._trace_storage_payload()),
        }

    def trace_action_probability(self, round_ref: int | str, *, action: str) -> dict[str, Any]:
        target_action = str(action or "").strip()
        if not target_action:
            raise ValueError("action is required")
        trace = self._trace_round_view(round_ref)
        return {
            "round_id": trace["round_id"],
            "sampled_action": trace["sampled_action"],
            "action": target_action,
            "action_probability_explanation": self._action_probability_explanation(trace, target_action=target_action),
            "conflict_arbitration": trace.get(
                "conflict_arbitration",
                self._conflict_arbitration_summary(trace),
            ),
            "storage": trace.get("storage", self._trace_storage_payload()),
        }

    def trace_agents(self, round_ref: int | str) -> dict[str, Any]:
        trace = self._trace_round_view(round_ref)
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
            "storage": trace.get("storage", self._trace_storage_payload()),
        }

    def trace_skills(self, round_ref: int | str) -> dict[str, Any]:
        trace = self._trace_round_view(round_ref)
        round_id = int(trace["round_id"])
        stored_skill_rows = [
            row
            for row in self.trace_store.list_skill_traces()
            if int(row.get("round_id", 0) or 0) == round_id
        ]
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
            for row in (trace.get("skill_traces", []) or stored_skill_rows)
        ]
        return {
            "round_id": round_id,
            "sampled_action": trace["sampled_action"],
            "skills": skills,
            "storage": trace.get("storage", self._trace_storage_payload()),
        }

    def trace_gates(self, round_ref: int | str) -> dict[str, Any]:
        trace = self._trace_round_view(round_ref)
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
            "storage": trace.get("storage", self._trace_storage_payload()),
        }

    def metrics_summary(self) -> dict[str, Any]:
        self.trace_store.flush(raise_on_error=False)
        rounds = self._metrics_round_rows()
        subjectivity = self._subjectivity_metrics(rounds=rounds)
        payload = self._long_run_metrics_summary(rounds=rounds)
        motivation = self.motivation_metrics(rounds=rounds, flush=False)
        endogenous = self.endogenous_metrics(rounds=rounds, subjectivity=subjectivity, flush=False)
        payload.update(
            {
                "boundary_violation_count": subjectivity["boundary_violation_count"],
                "external_to_internal_ratio": subjectivity["external_to_internal_ratio"],
                "endogenous_intent_rate": subjectivity["endogenous_intent_rate"],
                "motivation_active_rate": motivation["active_rate"],
                "motivation_activation_score_avg": motivation["activation_score_avg"],
                "endogenous_round_rate": endogenous["endogenous_round_rate"],
            }
        )
        payload["storage"] = self._trace_storage_payload()
        return payload

    def _metrics_round_rows(self) -> list[dict[str, Any]]:
        metrics_reader = getattr(self.trace_store, "metrics_round_rows_view", None)
        if callable(metrics_reader):
            return list(metrics_reader())
        summary_reader = getattr(self.trace_store, "list_round_summaries", None)
        if callable(summary_reader):
            return list(summary_reader())
        return self._full_round_history_rows()

    def _full_round_history_rows(self) -> list[dict[str, Any]]:
        limit_hint = 0
        count_reader = getattr(self.trace_store, "_round_count_from_parquet_cached", None)
        if callable(count_reader):
            try:
                limit_hint = max(limit_hint, int(count_reader() or 0))
            except Exception:
                limit_hint = max(limit_hint, 0)
        round_cache = getattr(self.trace_store, "_round_cache", None)
        if isinstance(round_cache, dict):
            limit_hint = max(limit_hint, len(round_cache))
        state_reader = getattr(self, "load_runtime_state", None)
        if callable(state_reader):
            try:
                runtime_state = state_reader()
                limit_hint = max(limit_hint, int(getattr(runtime_state, "round_count", 0) or 0))
            except Exception:
                limit_hint = max(limit_hint, 0)
        if limit_hint <= 0:
            latest = self.trace_store.recent_rounds(limit=1)
            if not latest:
                return []
            limit_hint = max(
                len(latest),
                int(latest[-1].get("round_id", 0) or 0),
            )
        return self.trace_store.recent_rounds(limit=limit_hint)

    def _long_run_metrics_summary(self, *, rounds: list[dict[str, Any]]) -> dict[str, Any]:
        metrics_fn = self.long_run_analyzer.metrics_summary
        try:
            parameters = inspect.signature(metrics_fn).parameters
        except (TypeError, ValueError):
            parameters = {}
        supports_rounds = "rounds" in parameters or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        if supports_rounds:
            return metrics_fn(rounds=rounds)
        return metrics_fn()

    def authenticity_timeline(self) -> dict[str, Any]:
        self.trace_store.flush(raise_on_error=False)
        payload = self.long_run_analyzer.authenticity_timeline()
        payload["storage"] = self._trace_storage_payload()
        return payload

    def vitality_timeline(self) -> dict[str, Any]:
        self.trace_store.flush(raise_on_error=False)
        payload = self.long_run_analyzer.vitality_timeline()
        payload["storage"] = self._trace_storage_payload()
        return payload

    def motivation_metrics(
        self,
        *,
        rounds: list[dict[str, Any]] | None = None,
        flush: bool = True,
    ) -> dict[str, Any]:
        if rounds is None:
            if flush:
                self.trace_store.flush(raise_on_error=False)
            rounds = self._metrics_round_rows()
        active_rounds = [row for row in rounds if dict(row.get("motivation_pool", {}) or {}).get("active_motivations")]
        activation_scores = [
            float(dict(row.get("motivation_pool", {}) or {}).get("endogenous_activation_score", 0.0) or 0.0)
            for row in active_rounds
        ]
        latest = active_rounds[-1] if active_rounds else {}
        return {
            "total_rounds": len(rounds),
            "active_rounds": len(active_rounds),
            "active_rate": round(len(active_rounds) / max(len(rounds), 1), 4),
            "activation_score_avg": round(sum(activation_scores) / max(len(activation_scores), 1), 4),
            "latest_pool": dict(latest.get("motivation_pool", {}) or {}),
            "storage": self._trace_storage_payload(),
        }

    def endogenous_metrics(
        self,
        *,
        rounds: list[dict[str, Any]] | None = None,
        subjectivity: dict[str, Any] | None = None,
        flush: bool = True,
    ) -> dict[str, Any]:
        if rounds is None:
            if flush:
                self.trace_store.flush(raise_on_error=False)
            rounds = self._metrics_round_rows()
        resolved_subjectivity = subjectivity if subjectivity is not None else self._subjectivity_metrics(rounds=rounds)
        endogenous_rounds = [row for row in rounds if row.get("cause_type") == "endogenous"]
        trigger_types: dict[str, int] = {}
        for row in endogenous_rounds:
            reason = dict(row.get("endogenous_tick_reason", {}) or {})
            latest = dict(reason.get("latest_trigger", {}) or {})
            trigger_type = str(latest.get("trigger_type", row.get("mode", "")) or "")
            if trigger_type:
                trigger_types[trigger_type] = trigger_types.get(trigger_type, 0) + 1
        return {
            "total_rounds": len(rounds),
            "endogenous_rounds": len(endogenous_rounds),
            "endogenous_round_rate": round(len(endogenous_rounds) / max(len(rounds), 1), 4),
            "trigger_types": trigger_types,
            "subjectivity": resolved_subjectivity,
            "storage": self._trace_storage_payload(),
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
        payload = self.trace_store.skill_stats()
        skill_rows = []
        for name, stats in sorted(payload.get("skills", {}).items()):
            spec = self.skills.get(name)
            skill_rows.append(
                {
                    "skill_name": name,
                    "owner_module": spec.owner_module if spec else "",
                    "timeout_ms": spec.timeout_ms if spec else 0,
                    "cost_class": spec.cost_class if spec else "",
                    "failure_policy": spec.failure_policy if spec else "",
                    "trace_tags": spec.trace_tags if spec else [],
                    "observed_rounds": stats.get("count", 0),
                    "average_latency_ms": stats.get("average_latency_ms", 0.0),
                    "degraded_count": stats.get("degraded_count", 0),
                }
            )
        payload["skill_rows"] = skill_rows
        payload["storage"] = self._trace_storage_payload()
        return payload

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
            "storage": self._trace_storage_payload(),
        }

    def model_status(self) -> dict[str, Any]:
        state = self.load_runtime_state()
        recent_skill_reader = getattr(self.trace_store, "recent_skill_traces", None)
        if callable(recent_skill_reader):
            raw_skill_rows = recent_skill_reader(limit=MODEL_STATUS_RECENT_SKILL_WINDOW)
        else:
            raw_skill_rows = self.trace_store.list_skill_traces()
        skill_rows = sorted(
            raw_skill_rows,
            key=lambda row: (int(row.get("round_id", 0)), str(row.get("recorded_at", ""))),
        )
        failover = self.model_router.failover_status()
        routes: dict[str, Any] = {}
        credential_present = bool(os.getenv("ARK_API_KEY"))

        def credential_visible(route_cfg: ModelRouteConfig) -> bool:
            if route_cfg.api_key_env:
                return bool(os.getenv(route_cfg.api_key_env))
            if route_cfg.backend == "doubao":
                return bool(os.getenv("ARK_API_KEY"))
            if route_cfg.backend == "deepseek":
                return bool(os.getenv("DEEPSEEK_API_KEY"))
            return True

        def effective_tier_route(tier_name: str, tier_cfg: dict[str, Any]) -> ModelRouteConfig | None:
            mode = str(tier_cfg.get("mode", "local")).lower()
            if mode == "local":
                return None
            route_cfg = ModelRouteConfig(
                name=tier_name,
                backend=str(tier_cfg.get("backend", "")),
                model=str(tier_cfg.get("model", "")),
                timeout_ms=int(tier_cfg.get("timeout_ms", 12000)),
                retries=int(tier_cfg.get("retries", 0)),
                enabled=bool(tier_cfg.get("enabled", True)),
                base_url=str(tier_cfg.get("base_url", self.config["models"]["models"].get("base_url", ""))),
                api_key_env=str(tier_cfg.get("api_key_env", "")).strip() or None,
            )
            setattr(route_cfg, "effective_mode", mode)
            return self.model_router.effective_route_config(route_cfg)

        tiers = {
            tier_name: {
                "mode": str(tier_cfg.get("mode", "local")),
                "backend": tier_cfg.get("backend"),
                "model": tier_cfg.get("model"),
                "base_url": tier_cfg.get("base_url"),
                "timeout_ms": tier_cfg.get("timeout_ms"),
                "api_key_env": tier_cfg.get("api_key_env"),
                "credential_present": credential_visible(effective_route) if effective_route is not None else True,
                "effective_backend": effective_route.backend if effective_route is not None else tier_cfg.get("backend"),
                "effective_model": effective_route.model if effective_route is not None else tier_cfg.get("model"),
                "effective_base_url": effective_route.base_url if effective_route is not None else tier_cfg.get("base_url"),
                "effective_api_key_env": effective_route.api_key_env if effective_route is not None else tier_cfg.get("api_key_env"),
            }
            for tier_name, tier_cfg in self._model_tiers().items()
            for effective_route in [effective_tier_route(tier_name, tier_cfg)]
        }

        for route_name, route in self.model_router.route_configs.items():
            bound_skills = list(MODEL_ROUTE_SKILL_BINDINGS.get(route_name, ()))
            route_rows = [row for row in skill_rows if row.get("skill_name") in bound_skills]
            fallback_rows = [
                row
                for row in route_rows
                if row.get("degraded") or row.get("fallback_route") or row.get("failure_policy_applied")
            ]
            last_row = route_rows[-1] if route_rows else None
            last_failure = fallback_rows[-1] if fallback_rows else None
            binding_key = MODEL_ROUTE_BINDING_KEYS.get(route_name)
            effective_route = self._route_config_for_binding(binding_key, route_name=route_name) if binding_key else None
            status_route = effective_route or route
            effective_status_route = self.model_router.effective_route_config(status_route)
            route_credential_present = credential_visible(effective_status_route)
            health = "disabled" if not route.enabled else "ready"
            if route.enabled and not route_credential_present:
                health = "degraded"
            if route.enabled and last_failure is not None:
                health = "degraded"

            routes[route_name] = {
                "enabled": route.enabled,
                "backend": route.backend,
                "model": route.model,
                "credential_present": route_credential_present,
                "timeout_ms": route.timeout_ms,
                "retries": route.retries,
                "bound_skills": bound_skills,
                "health": health,
                "recent_call_count": len(route_rows),
                "recent_fallback_count": len(fallback_rows),
                "last_failure_reason": last_failure.get("failure_policy_applied") if last_failure else None,
                "last_fallback_route": last_failure.get("fallback_route") if last_failure else None,
                "last_activity_round": last_row.get("round_id") if last_row else None,
                "effective_backend": effective_status_route.backend,
                "effective_model": effective_status_route.model,
                "effective_base_url": effective_status_route.base_url,
                "effective_api_key_env": effective_status_route.api_key_env,
            }

        return {
            "current_round": state.round_count,
            "credential_present": credential_present,
            "tiers": tiers,
            "module_model_bindings": self._module_model_bindings(),
            "routes": routes,
            "route_policies": self._route_policy_contract(),
            "activation_thresholds": self._activation_thresholds_contract(),
            "memory_tier_policy": self._memory_tier_policy_contract(),
            "consolidation_policy": self._consolidation_policy_contract(),
            "failover": failover,
            "fault_guard": self.fault_guard_status(state),
        }

    def thought_snapshot(self, round_ref: int | str = "last") -> dict[str, Any]:
        trace = self.trace_round(round_ref)
        why = self.why_this(round_ref)
        action_field = self.console_action_field(round_ref)
        timeline = self.console_timeline(round_ref)
        return {
            "round_id": trace.get("round_id"),
            "trace_ref": trace.get("trace_ref"),
            "sampled_action": trace.get("sampled_action"),
            "thought_summary": {
                "why_summary": str((why.get("sampled_action") or trace.get("sampled_action") or "暂无")),
                "conflict_mode": str(dict(trace.get("conflict_arbitration", {}) or {}).get("conflict_mode") or ""),
                "top_intent": str(dict(trace.get("initiative", {}) or {}).get("top_intent") or ""),
            },
            "top_drivers": why.get("top_drivers", []),
            "why": why,
            "action_field": action_field,
            "timeline": timeline,
            "initiative": trace.get("initiative", {}),
            "expressive_trace": trace.get("expressive_trace", trace.get("initiative", {})),
            "rendered_expression": trace.get("rendered_expression", {}),
            "storage": trace.get("storage", self.trace_storage_status()),
        }

    def empty_thought_payload(self, round_ref: int | str | None = None) -> dict[str, Any]:
        return {
            "round_id": round_ref if isinstance(round_ref, int) else None,
            "trace_ref": None,
            "sampled_action": "nothing",
            "thought_summary": {
                "why_summary": "暂无",
                "conflict_mode": "",
                "top_intent": "",
            },
            "top_drivers": [],
            "why": self.empty_why_payload(round_ref),
            "action_field": {
                "round_id": round_ref if isinstance(round_ref, int) else None,
                "trace_ref": None,
                "top_actions": [],
                "winner": {"action": None, "score": 0.0},
                "winner_posterior": {},
                "conflict": {},
                "token_field": {},
                "contribution_stack": [],
                "competing_peaks": [],
            },
            "timeline": {
                "round_id": round_ref if isinstance(round_ref, int) else None,
                "trace_ref": None,
                "events": [],
            },
            "initiative": {},
            "expressive_trace": {},
            "rendered_expression": {},
            "storage": self.trace_storage_status(),
            "message": "无决策记录",
        }

    def migration_report(self) -> dict[str, Any]:
        payload = self.runtime_migration_report()
        payload["storage"] = self._trace_storage_payload()
        return payload

    def runtime_storage_status(self) -> dict[str, Any]:
        parquet_ready = self.state_parquet_path.exists()
        return {
            "read_source_default": "parquet",
            "storage_state": "healthy",
            "parquet_live_ready": parquet_ready,
            "degraded_reason": None,
            "last_sync_at": None,
        }

    def _latest_why_payload(self, state: RuntimeState) -> dict[str, Any] | None:
        if state.round_count <= 0:
            return None
        try:
            patched = getattr(self.controller, "__dict__", {}).get("why_this")
            if callable(patched):
                return patched(state.round_count)
            return self.why_this(state.round_count)
        except FileNotFoundError:
            return None

    def _current_intent_summary(
        self,
        state: RuntimeState,
        latest_why: dict[str, Any] | None,
        run_payload: dict[str, Any] | None,
    ) -> str:
        if run_payload:
            run_goal = str(run_payload.get("goal_summary") or run_payload.get("goal") or "").strip()
            if run_goal:
                return f"围绕目标“{run_goal}”收束下一步行动"
        if latest_why:
            if str(latest_why.get("cause_type") or "").strip() == "endogenous":
                internal_mode = str(latest_why.get("mode") or "").strip()
                return {
                    "endogenous_light": "正在进行内部整理，先在心里消化线索",
                    "endogenous_regulation": "正在进行内在调节，让状态先回稳",
                    "endogenous_replay": "正在进行内部回放，先把刚才的痕迹过一遍",
                }.get(internal_mode, "正在进行内部整理，先在内部自我对话")
            identity_context = latest_why.get("render_plan", {}).get("identity_context", {})
            query_intent = str(identity_context.get("query_intent") or "").strip()
            sampled_action = str(latest_why.get("sampled_action") or "").strip()
            query_label = self._query_intent_label(query_intent)
            action_label = self._action_phrase(sampled_action)
            if query_label and action_label:
                return f"{query_label}，{action_label}"
            if action_label:
                return action_label
        if state.current_goal:
            return f"围绕目标“{state.current_goal}”维持 {state.focus} 焦点"
        return f"当前保持{self._focus_label(state.focus)}，并继续根据慢变量调节表达"

    def _identity_summary(self, state: RuntimeState, latest_why: dict[str, Any] | None) -> dict[str, Any]:
        display_name = self._display_label(state)
        continuity = "还在形成稳定称呼"
        if display_name != self._unnamed_label():
            continuity = "名称与身份连续性稳定"
        if latest_why:
            identity_evolution = latest_why.get("identity_evolution", {})
            rename_reason = str(identity_evolution.get("rename_reason") or "").strip()
            continuity = self._continuity_label(rename_reason, has_display_name=display_name != self._unnamed_label())
        return {
            "display_name": display_name,
            "continuity": continuity,
        }

    def _query_intent_label(self, query_intent: str) -> str:
        return {
            "general_exchange": "正在自然交流",
            "general_task_push": "正在收束任务目标",
            "self_model_identity_probe": "正在回应关于我是谁的问题",
            "provider_lineage_probe": "正在回应来源追问",
            "answer_reason_probe": "正在解释刚才为什么这样说",
            "capability_boundary_probe": "正在说明能力边界",
            "relational_bid": "正在接住关系靠近",
            "self_disclosure_request": "正在判断自我披露尺度",
        }.get(query_intent, "")

    def _action_phrase(self, action_name: str) -> str:
        return {
            "respond": "准备直接回应",
            "plan": "准备梳理下一步",
            "recall": "准备调取相关记忆",
            "clarify": "准备先确认关键信息",
            "connect": "准备更靠近地回应",
            "rest": "准备降低表达强度",
            "wander": "准备转入发散游移",
            "absorb": "准备先吸收内部线索",
            "self_run": "准备转入自我检查并启动只读行动",
            "monologue": "准备把念头先在内部说清楚",
            "nothing": "准备先不向外表达",
            "die": "准备自主结束生命",
            "short_reply": "准备先给出简短回应",
        }.get(action_name, "")

    def _focus_label(self, focus: str) -> str:
        return {
            "task": "专心处理眼前的事",
            "respond": "把注意力放在回应上",
            "wander": "思绪有些发散",
            "rest": "慢慢回落和恢复",
            "absorb": "把注意力转向内部吸收",
            "self_run": "把注意力转向自我检查和只读行动",
            "monologue": "把注意力转向内在独白",
            "nothing": "暂时收住外显表达",
            "die": "正在转向自主结束生命",
            "boot": "还在进入状态",
        }.get(focus, focus)

    def _continuity_label(self, rename_reason: str, *, has_display_name: bool) -> str:
        if rename_reason in {"", "stable_continuity"}:
            return "名称与身份连续性稳定" if has_display_name else "还在形成稳定称呼"
        if rename_reason == "unnamed":
            return "还在形成稳定称呼"
        if rename_reason == "generated":
            return "当前名称正根据内部证据逐步形成"
        if rename_reason == "user_seed":
            return "当前名称沿用用户给出的称呼"
        if rename_reason == "manual_override":
            return "当前名称采用人工指定称呼"
        return "当前连续性正在调整中"

    def _authenticity_summary(self, latest_why: dict[str, Any] | None) -> dict[str, Any]:
        if not latest_why:
            return {
                "summary": "还没有足够证据判断这轮真实感",
                "source": "none",
                "guard_action": "none",
                "sampling_penalty_applied": 0.0,
            }
        authenticity = latest_why.get("authenticity", {})
        guard_action = str(authenticity.get("guard_action") or "none")
        penalty = round(float(authenticity.get("sampling_penalty_applied", 0.0) or 0.0), 4)
        if guard_action in {"none", "pass"} and penalty <= 0.0:
            summary = "这轮表达比较自然"
        elif guard_action in {"none", "pass"}:
            summary = "这轮表达总体自然，但有一点收束"
        elif guard_action == "resample":
            summary = "这轮在收住偏移，已经主动回拉表达"
        elif guard_action == "fallback":
            summary = "这轮为了保持真实感，表达被明显收束"
        else:
            summary = "这轮真实性状态有明显波动"
        return {
            "summary": summary,
            "source": "trace",
            "guard_action": guard_action,
            "sampling_penalty_applied": penalty,
        }

    def internal_learning_summary(self, state: RuntimeState | None = None) -> dict[str, Any]:
        current_state = state or self.load_runtime_state()
        learning_state = getattr(current_state, "motivation_learning_state", MotivationLearningState())
        recent_feedback = list(getattr(learning_state, "recent_feedback", []) or [])
        policy_shift = dict(getattr(learning_state, "endogenous_policy_shift", {}) or {})
        top_policy_shifts = [
            {
                "intent": str(intent or ""),
                "delta": round(float(delta or 0.0), 4),
            }
            for intent, delta in sorted(
                policy_shift.items(),
                key=lambda item: (-abs(float(item[1] or 0.0)), str(item[0] or "")),
            )[:3]
            if abs(float(delta or 0.0)) > 0.0
        ]
        last_feedback = recent_feedback[-1] if recent_feedback else None
        return {
            "recent_feedback_count": len(recent_feedback),
            "top_policy_shifts": top_policy_shifts,
            "last_learning_round": str(getattr(last_feedback, "round_id", "") or ""),
        }

    def cognitive_snapshot(
        self,
        *,
        state: RuntimeState | None = None,
        run_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        current_state = state or self.load_runtime_state()
        self._sync_tlh_state(current_state)
        latest_why = self._latest_why_payload(current_state)
        return {
            "core_goal": ARCHITECTURE_CORE_GOAL,
            "current_intent": self._current_intent_summary(current_state, latest_why, run_payload),
            "vital_signs": {
                "mood": round(float(current_state.mood), 4),
                "body_energy": round(float(current_state.body_energy), 4),
                "affect_residue": round(float(current_state.affect_residue), 4),
                "focus": str(current_state.focus),
                "mode": str(current_state.mode),
            },
            "identity": self._identity_summary(current_state, latest_why),
            "authenticity": self._authenticity_summary(latest_why),
            "internal_learning": self.internal_learning_summary(current_state),
            "tlh": {
                "body_state": to_dict(current_state.body_state),
                "subjective_state": to_dict(current_state.subjective_state),
                "emotion_state": to_dict(current_state.emotion_state),
                "desire_state": to_dict(current_state.desire_state),
                "instinct_field": to_dict(current_state.instinct_field),
                "organic_mode": to_dict(current_state.organic_mode),
                "emergent_action_sketches": to_dict(current_state.emergent_action_sketches),
                "personality_anchor": to_dict(current_state.personality_anchor),
            },
        }

    def relation_show(self, target: str) -> dict[str, Any]:
        return self.memory_store.relation_state(target)

    def replay_round(self, round_id: int, seed: int | None = None) -> dict[str, Any]:
        return self.replay(round_id, seed=seed or 0)

    def replay(self, round_id: int, seed: int = 0) -> dict[str, Any]:
        trace = self._trace_round_view(round_id)
        proposal_rows = trace.get("proposal_summaries", [])
        candidate_distribution = self._candidate_distribution_from_trace(trace)
        counterfactual_replays = self._counterfactual_replays_from_trace(trace)
        personality_anchor = trace.get("state_snapshot", {}).get("personality_anchor", {})
        instinct_field = trace.get("state_snapshot", {}).get("instinct_field", {})
        emergent_sketches = trace.get("state_snapshot", {}).get("emergent_action_sketches", [])
        ablations = []
        for item in proposal_rows[:3]:
            delta_map = item.get("delta_p", {}) if isinstance(item.get("delta_p"), dict) else {}
            top_action = item.get("top_action", trace.get("sampled_action"))
            ablations.append(
                {
                    "agent": item.get("agent_name", "unknown"),
                    "action": top_action,
                    "delta": round(float(delta_map.get(top_action, 0.0)), 4),
                }
            )
        replayed_action = self._sample_counterfactual_action(
            candidate_distribution,
            seed=seed,
            default_action=str(trace.get("sampled_action") or ""),
        )
        preview = next(
            (item.get("counterfactual_preview", {}) for item in counterfactual_replays if item.get("action") == replayed_action),
            {"action": replayed_action, "would_output": replayed_action != "nothing", "text": "", "terminal_intent": replayed_action == "die"},
        )
        return {
            "round_id": round_id,
            "original_action": trace["sampled_action"],
            "replayed_action": replayed_action,
            "candidate_distribution": candidate_distribution,
            "ablations": ablations,
            "counterfactual_replays": counterfactual_replays,
            "counterfactual_preview": preview,
            "instinct_field": instinct_field,
            "emergent_action_sketches": emergent_sketches,
            "personality_anchor": personality_anchor,
            "high_dimensional_collapse": dict(instinct_field.get("collapse_trace", {}) or {}),
            "anchor_alignment": round(float(personality_anchor.get("alignment", 0.0) or 0.0), 4),
            "emergent_action_formalization": self._emergent_action_formalization_payload(emergent_sketches),
            "seed": seed,
            "storage": self._trace_storage_payload(),
        }

    def empty_replay_payload(self, round_ref: int | str | None = None, *, seed: int = 0) -> dict[str, Any]:
        return {
            "round_id": round_ref if isinstance(round_ref, int) else None,
            "original_action": "nothing",
            "replayed_action": "nothing",
            "candidate_distribution": {},
            "ablations": [],
            "counterfactual_replays": [],
            "counterfactual_preview": {},
            "instinct_field": {},
            "emergent_action_sketches": [],
            "personality_anchor": {},
            "high_dimensional_collapse": {},
            "anchor_alignment": 0.0,
            "emergent_action_formalization": {"sketches": [], "active_count": 0, "candidate_targets": []},
            "seed": seed,
            "storage": self._trace_storage_payload(),
            "message": "无决策记录",
        }

    def replay_motivation(self, round_id: int) -> dict[str, Any]:
        trace = self._trace_round_view(round_id)
        return {
            "round_id": round_id,
            "trace_ref": trace.get("trace_ref", f"round://{round_id}"),
            "sampled_action": trace["sampled_action"],
            "motivation_pool": trace.get("motivation_pool", {}),
            "motivation_feedback": trace.get("motivation_feedback", {}),
            "endogenous_tick_reason": trace.get("endogenous_tick_reason", {}),
            "endogenous_policy_shift": trace.get("endogenous_policy_shift", {}),
            "endogenous_trigger_context": trace.get("endogenous_trigger_context", {}),
            "endogenous_suppression": trace.get("endogenous_suppression", {}),
            "micro_intent": trace.get("micro_intent", {}),
            "endogenous_replay_chain": trace.get("endogenous_replay_chain", {}),
            "storage": trace.get("storage", self._trace_storage_payload()),
        }

    def empty_motivation_payload(self, round_ref: int | str | None = None) -> dict[str, Any]:
        return {
            "round_id": round_ref if isinstance(round_ref, int) else None,
            "trace_ref": None,
            "sampled_action": "nothing",
            "cause_type": "external_stimulus",
            "motivation_pool": {},
            "motivation_feedback": {},
            "endogenous_tick_reason": {},
            "endogenous_policy_shift": {},
            "endogenous_trigger_context": {},
            "endogenous_suppression": {},
            "micro_intent": {},
            "endogenous_replay_chain": {},
            "storage": self._trace_storage_payload(),
            "message": "无决策记录",
        }

    def why_motivation(self, round_ref: int | str) -> dict[str, Any]:
        trace = self._trace_round_view(round_ref)
        return {
            "round_id": trace["round_id"],
            "trace_ref": trace.get("trace_ref", f"round://{trace['round_id']}"),
            "sampled_action": trace["sampled_action"],
            "cause_type": trace.get("cause_type", "external_stimulus"),
            "motivation_pool": trace.get("motivation_pool", {}),
            "motivation_feedback": trace.get("motivation_feedback", {}),
            "endogenous_tick_reason": trace.get("endogenous_tick_reason", {}),
            "endogenous_policy_shift": trace.get("endogenous_policy_shift", {}),
            "endogenous_trigger_context": trace.get("endogenous_trigger_context", {}),
            "endogenous_suppression": trace.get("endogenous_suppression", {}),
            "micro_intent": trace.get("micro_intent", {}),
            "endogenous_replay_chain": trace.get("endogenous_replay_chain", {}),
            "storage": trace.get("storage", self._trace_storage_payload()),
        }

    def why_not(self, round_id: int, action: str) -> dict[str, Any]:
        trace = self._trace_round_view(round_id)
        candidate_distribution = self._candidate_distribution_from_trace(trace)
        action_explanation = self._action_probability_explanation(trace, target_action=action)
        expressive_trace = dict(trace.get("expressive_trace", trace.get("initiative", {})) or {})
        blocked_by = []
        if action not in candidate_distribution:
            blocked_by.append("not_proposed")
        gate_decisions = [item for item in trace.get("gate_decisions", []) if isinstance(item, dict)]
        for item in gate_decisions:
            if not item.get("allowed", True):
                blocked_by.append(str(item.get("owner") or item.get("stage") or "gate"))
        blocked_by.extend(
            item["module_name"]
            for item in action_explanation["stacked_contributions"]
            if item["hard_masked"] or item["direction"] in {"block", "suppress"}
        )
        blocked_by.extend(item["agent_name"] for item in trace.get("top_drivers", [])[:2])
        blocked_by = list(dict.fromkeys(blocked_by))
        summary = self._why_not_summary(
            action=action,
            selected_action=str(trace.get("sampled_action") or ""),
            candidate_score=float(candidate_distribution.get(action, 0.0) or 0.0),
            blocked_by=blocked_by,
            expressive_trace=expressive_trace,
        )
        return {
            "round_id": round_id,
            "action": action,
            "selected_action": trace["sampled_action"],
            "candidate_score": candidate_distribution.get(action, 0.0),
            "blocked_by": blocked_by,
            "stacked_contributions": action_explanation["stacked_contributions"],
            "competing_peaks": action_explanation["competing_peaks"],
            "conflict_arbitration": self._conflict_arbitration_summary(trace),
            "expressive_trace": expressive_trace,
            "summary": summary,
            "storage": self._trace_storage_payload(),
        }

    def empty_why_not_payload(self, action: str = "", round_ref: int | str | None = None) -> dict[str, Any]:
        return {
            "round_id": round_ref if isinstance(round_ref, int) else None,
            "action": str(action or ""),
            "selected_action": "nothing",
            "candidate_score": 0.0,
            "blocked_by": [],
            "stacked_contributions": [],
            "competing_peaks": [],
            "conflict_arbitration": {},
            "expressive_trace": {},
            "summary": "无决策记录",
            "storage": self._trace_storage_payload(),
            "message": "无决策记录",
        }

    def state_delta_timeline(self, window: int = 20) -> dict[str, Any]:
        rounds = self.trace_store.recent_rounds(limit=window)
        return {
            "points": [
                {
                    "round_id": trace["round_id"],
                    "sampled_action": trace.get("sampled_action"),
                    "appraisal_snapshot": trace.get("appraisal_snapshot", {}),
                    "state_delta_before_clip": trace.get("state_delta_before_clip", {}),
                    "state_delta_after_clip": trace.get("state_delta_after_clip", {}),
                    "delta_suppression_reason": trace.get("delta_suppression_reason", []),
                }
                for trace in rounds
            ],
            "storage": self._trace_storage_payload(),
        }

    def identity_blockers(self) -> dict[str, Any]:
        state = self.load_runtime_state()
        evidence = self._augment_identity_evidence(state, self.memory_store.identity_evidence())
        return {
            "display_name": state.identity_state.display_name or self._unnamed_label(),
            "score": round(float(evidence.get("identity_score", 0.0) or 0.0), 4),
            "naming_signal": round(float(evidence.get("naming_signal", 0.0) or 0.0), 4),
            "continuity_signal": round(float(evidence.get("continuity_signal", 0.0) or 0.0), 4),
            "blockers": list(evidence.get("rejection_reasons", [])),
            "clusters": list(evidence.get("identity_clusters", [])),
            "storage": self._trace_storage_payload(),
        }

    def cue_fragmentation_report(self) -> dict[str, Any]:
        families: dict[str, dict[str, Any]] = {}
        sources = [
            ("stable_prior", self.memory_store.stable_priors_top(limit=20), "cue"),
            ("habit", self.memory_store.habit_top(limit=20), "pattern"),
            ("memory", self.memory_store.memory_top(limit=20), "cue"),
        ]
        for source_name, rows, field_name in sources:
            for row in rows:
                cue = str(row.get(field_name) or "").strip()
                if not cue:
                    continue
                family = cue.split(":", 1)[0] if ":" in cue else cue
                entry = families.setdefault(family, {"family": family, "members": set(), "sources": set(), "count": 0})
                entry["members"].add(cue)
                entry["sources"].add(source_name)
                entry["count"] += 1
        normalized = []
        for family in families.values():
            normalized.append(
                {
                    "family": family["family"],
                    "members": sorted(family["members"]),
                    "sources": sorted(family["sources"]),
                    "count": family["count"],
                    "fragmented": len(family["members"]) > 1,
                }
            )
        normalized.sort(key=lambda item: (not item["fragmented"], -item["count"], item["family"]))
        return {"families": normalized, "storage": self._trace_storage_payload()}

    def run_contamination_report(self, window: int = 20) -> dict[str, Any]:
        rounds = self.trace_store.recent_rounds(limit=window)
        return {
            "points": [
                {
                    "round_id": trace["round_id"],
                    "scenario": trace.get("scenario"),
                    "run_contamination_detected": bool(trace.get("run_contamination_detected", False)),
                    "drive_source": trace.get("run_context", {}).get("drive_source"),
                    "current_goal": trace.get("run_context", {}).get("current_goal"),
                }
                for trace in rounds
            ],
            "storage": self._trace_storage_payload(),
        }

    def why_no_change(self, round_ref: int | str | None = None) -> dict[str, Any]:
        trace = self.trace_round(self.resolve_round_ref(round_ref))
        appraisal = trace.get("appraisal_snapshot", {})
        before = trace.get("state_delta_before_clip", {})
        after = trace.get("state_delta_after_clip", {})
        blockers: list[str] = []
        failure_mode = "called_neutral"

        if not appraisal:
            failure_mode = "not_called"
            blockers.append("no_appraisal_input")
        elif trace.get("run_contamination_detected", False):
            failure_mode = "updated_but_not_expressed"
            blockers.append("run_contamination")
        elif any(trace.get("delta_suppression_reason", [])):
            failure_mode = "updated_but_clipped"
            blockers.extend(trace.get("delta_suppression_reason", []))
        elif max(abs(float(value or 0.0)) for value in before.values() or [0.0]) <= 0.01:
            failure_mode = "called_neutral"
            blockers.append("no_appraisal_input")
        elif max(abs(float(value or 0.0)) for value in after.values() or [0.0]) <= 0.03:
            failure_mode = "updated_but_not_expressed"
            blockers.append("expression_threshold_not_met")
        if trace.get("identity_trigger_blockers"):
            blockers.extend(trace.get("identity_trigger_blockers", []))
        deduped = []
        for blocker in blockers:
            if blocker not in deduped:
                deduped.append(blocker)
        if failure_mode == "not_called":
            summary = "本轮没有形成有效 appraisal 输入，因此没有可见变化。"
        elif failure_mode == "updated_but_clipped":
            summary = "状态更新出现了，但被抑制或裁剪，没有进入可见表达。"
        elif failure_mode == "updated_but_not_expressed":
            summary = "状态有更新，但尚未越过表达阈值，或被其他运行上下文压住了。"
        else:
            summary = "本轮 appraisal 偏中性，因此没有形成明显外显变化。"
        return {
            "round_id": trace["round_id"],
            "failure_mode": failure_mode,
            "blockers": deduped,
            "summary": summary,
            "appraisal_snapshot": appraisal,
            "state_delta_before_clip": before,
            "state_delta_after_clip": after,
            "storage": trace.get("storage", self._trace_storage_payload()),
        }

    def empty_why_no_change_payload(self, round_ref: int | str | None = None) -> dict[str, Any]:
        return {
            "round_id": round_ref if isinstance(round_ref, int) else None,
            "failure_mode": "",
            "blockers": [],
            "summary": "无决策记录",
            "appraisal_snapshot": {},
            "state_delta_before_clip": {},
            "state_delta_after_clip": {},
            "storage": self._trace_storage_payload(),
            "message": "无决策记录",
        }

    def what_changed(self, window: int = 5) -> dict[str, Any]:
        rounds = self.trace_store.recent_rounds(limit=window)
        if not rounds:
            return {"window": window, "action_counts": {}, "mode_counts": {}, "budget_delta": 0.0, "storage": self._trace_storage_payload()}
        first = rounds[0].get("state_snapshot", {})
        last = rounds[-1].get("state_snapshot", {})
        action_counts: dict[str, int] = {}
        mode_counts: dict[str, int] = {}
        for row in rounds:
            action_counts[row["sampled_action"]] = action_counts.get(row["sampled_action"], 0) + 1
            mode_counts[row["mode"]] = mode_counts.get(row["mode"], 0) + 1
        return {
            "window": window,
            "action_counts": action_counts,
            "mode_counts": mode_counts,
            "budget_delta": round(float(last.get("budget_remaining", 1.0)) - float(first.get("budget_remaining", 1.0)), 4),
            "storage": self._trace_storage_payload(),
        }

    def conflict_timeline(self) -> dict[str, Any]:
        self.trace_store.flush(raise_on_error=False)
        points = []
        for trace in self._metrics_round_rows():
            conflict = dict(trace.get("conflict_arbitration", {}) or self._conflict_arbitration_summary(trace))
            ledger_tail = conflict.get("repair_ledger_tail", [])
            repair_ledger_summary = dict(conflict.get("repair_ledger_summary", {}) or {})
            if repair_ledger_summary:
                repair_ledger_summary = {
                    "entries": int(repair_ledger_summary.get("entries", len(ledger_tail)) or 0),
                    "latest_reason": str(
                        repair_ledger_summary.get(
                            "latest_reason",
                            ledger_tail[-1]["reason"] if ledger_tail else "",
                        )
                        or ""
                    ),
                }
            else:
                repair_ledger_summary = {
                    "entries": len(ledger_tail),
                    "latest_reason": ledger_tail[-1]["reason"] if ledger_tail else "",
                }
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
                    "repair_mode": conflict.get("repair_mode"),
                    "repair_stage": conflict.get("repair_state_snapshot", {}).get("stage", "idle"),
                    "last_post_error_adjustment": conflict.get("post_error_adjustment", {}),
                    "repair_ledger_summary": repair_ledger_summary,
                    "repair_learning": {
                        "adjustment_reasons": trace.get("state_snapshot", {}).get("conflict_learning_state", {}).get("adjustment_reasons", {}),
                        "last_learning_signal": trace.get("state_snapshot", {}).get("conflict_learning_state", {}).get("last_learning_signal", {}),
                    },
                    "conflict_safe_mode_owned": conflict.get("conflict_safe_mode_owned", False),
                }
            )
        return {"points": points, "storage": self._trace_storage_payload()}

    def entropy_metrics(self) -> dict[str, Any]:
        state = self.load_runtime_state()
        health = self.entropy_pool.health_snapshot()
        return {
            **health,
            "last_runtime_failure": dict(state.last_entropy_failure),
            "storage": self._trace_storage_payload(),
        }

    def mode_switch_timeline(self) -> dict[str, Any]:
        self.trace_store.flush(raise_on_error=False)
        points = []
        for trace in self._metrics_round_rows():
            points.append(
                {
                    "round_id": trace["round_id"],
                    "mode": trace["mode"],
                    "forced_switch": any(item.get("stage") == "forced_mode_switch" for item in trace.get("gate_decisions", [])),
                }
            )
        return {"points": points, "storage": self._trace_storage_payload()}

    def ablation_summary(self) -> dict[str, Any]:
        implicit_modules = ["ResourceAgent", "HippocampusAgent", "PerspectiveModel", "UnconsciousAgent", "CerebellarPredictor"]
        rounds = self.trace_store.list_rounds()
        total_rounds = len(rounds) or 1
        modules = []
        for module_name in implicit_modules:
            appearances = 0
            approx_deltas: list[float] = []
            for trace in rounds:
                action_layer = self._action_layer_from_trace(trace)
                audit_rows = [
                    item
                    for item in action_layer.get("contribution_audit", [])
                    if isinstance(item, dict) and item.get("module_name") == module_name
                ]
                if not audit_rows:
                    continue
                appearances += 1
                sampled_action = trace.get("sampled_action")
                module_gain = 0.0
                for row in audit_rows:
                    normalized = dict(row.get("delta_normalized", {}) or {})
                    if sampled_action in list(row.get("hard_masked_targets", []) or []):
                        module_gain -= 1.0
                    module_gain += float(normalized.get(sampled_action, 0.0) or 0.0)
                approx_deltas.append(round(module_gain, 6))
            modules.append(
                {
                    "module": module_name,
                    "observed_rounds": appearances,
                    "coverage": round(appearances / total_rounds, 4),
                    "approx_gain": round(sum(approx_deltas) / len(approx_deltas), 6) if approx_deltas else 0.0,
                    "max_gain": round(max(approx_deltas), 6) if approx_deltas else 0.0,
                }
            )
        return {"modules": modules, "storage": self._trace_storage_payload()}

    def metrics_timeline(self) -> dict[str, Any]:
        rounds = self._metrics_round_rows()
        return {
            "rounds": [
                {
                    "round_id": row["round_id"],
                    "sampled_action": row["sampled_action"],
                    "mode": row["mode"],
                    "budget_remaining": row.get("state_snapshot", {}).get("budget_remaining"),
                    "conflict_score": (row.get("conflict_arbitration", {}) or {}).get(
                        "total_score",
                        (row.get("conflict_arbitration", {}) or {}).get("score", 0.0),
                    ),
                    "cause_type": row.get("cause_type", "external_stimulus"),
                }
                for row in rounds
            ],
            "subjectivity": self._subjectivity_metrics(rounds=rounds),
            "storage": self._trace_storage_payload(),
        }

    def metrics_heatmap(self) -> dict[str, Any]:
        rounds = self.trace_store.list_rounds()
        actions: dict[str, int] = {}
        action_module_heatmap: dict[str, dict[str, float]] = {}
        action_block_heatmap: dict[str, dict[str, int]] = {}
        for row in rounds:
            action = str(row.get("sampled_action", "unknown"))
            actions[action] = actions.get(action, 0) + 1
            action_layer = self._action_layer_from_trace(row)
            for audit in action_layer.get("contribution_audit", []) or []:
                if not isinstance(audit, dict):
                    continue
                module_name = str(audit.get("module_name", "") or "unknown")
                for target, value in dict(audit.get("delta_normalized", {}) or {}).items():
                    module_map = action_module_heatmap.setdefault(str(target), {})
                    module_map[module_name] = round(module_map.get(module_name, 0.0) + float(value), 6)
                for target in list(audit.get("hard_masked_targets", []) or []):
                    block_map = action_block_heatmap.setdefault(str(target), {})
                    block_map[module_name] = block_map.get(module_name, 0) + 1
        return {
            "actions": actions,
            "action_module_heatmap": action_module_heatmap,
            "action_block_heatmap": action_block_heatmap,
            "storage": self._trace_storage_payload(),
        }

    def acceptance_report(self, window: int = 20) -> dict[str, Any]:
        def _status_from_evidence(*, failures: list[str], partials: list[str]) -> str:
            if failures:
                return "fail"
            if partials:
                return "partial"
            return "pass"

        def _has_payload(value: Any) -> bool:
            if value is None:
                return False
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return abs(float(value)) > 1e-9
            if isinstance(value, str):
                return bool(value.strip())
            if isinstance(value, dict):
                return any(_has_payload(item) for item in value.values())
            if isinstance(value, list):
                return any(_has_payload(item) for item in value)
            return bool(value)

        def _meaningful_delta(payload: dict[str, Any]) -> bool:
            for value in dict(payload or {}).values():
                if _has_payload(value):
                    return True
            return False

        rounds = self.trace_store.list_rounds()[-window:]
        total = len(rounds) or 1
        probability_field_rounds = [row for row in rounds if row.get("probability_field")]
        action_audit_rounds = [
            row
            for row in rounds
            if row.get("probability_field", {}).get("action", {}).get("contribution_audit")
        ]
        token_audit_rounds = [
            row
            for row in rounds
            if row.get("probability_field", {}).get("token", {}).get("contribution_audit")
        ]
        renderer_token_rounds = [
            row
            for row in token_audit_rounds
            if any(
                isinstance(item, dict) and item.get("module_name") == "Renderer"
                for item in row.get("probability_field", {}).get("token", {}).get("contribution_audit", [])
            )
        ]
        token_source_integrity_passes = 0
        for row in rounds:
            token_field = row.get("probability_field", {}).get("token", {})
            token_audit = token_field.get("contribution_audit", []) if isinstance(token_field, dict) else []
            if not token_audit:
                token_source_integrity_passes += 1
                continue
            if all((not item.get("dependency_trace") or item.get("module_name")) for item in token_audit if isinstance(item, dict)):
                token_source_integrity_passes += 1
        autonomy_settings = self._observer_settings().get("autonomy", {}) if isinstance(self._observer_settings().get("autonomy"), dict) else {}
        long_run_prior = self._long_run_prior_summary(rounds)
        bounded_long_run = {
            "window": window,
            "rounds_considered": len(rounds),
            "acceptance_report_window": window,
            "long_run_prior": long_run_prior,
        }
        current_state = self.load_runtime_state()
        self._sync_plan16_state(current_state)
        latest_round = dict(rounds[-1]) if rounds else {}
        state_truth = self.state_truth_payload(current_state, trace=latest_round or None)
        single_truth_checks = dict(state_truth.get("consistency_checks", {}) or {})
        single_truth_failures = [
            key
            for key in (
                "subject_projection_from_state",
                "meaning_projection_from_state",
                "agency_projection_from_state",
                "session_cache_is_projection_only",
            )
            if not bool(single_truth_checks.get(key, False))
        ]
        projection_only_truth_fields = [
            str(item)
            for item in list(single_truth_checks.get("projection_only_truth_fields", []) or [])
            if str(item).strip()
        ]
        if projection_only_truth_fields:
            single_truth_failures.append("projection_only_truth_fields")
        single_truth_partials: list[str] = []
        if rounds and not str(dict(state_truth.get("event_log", {}) or {}).get("round_trace_ref") or "").strip():
            single_truth_partials.append("missing_round_trace_ref")
        if rounds and int(dict(state_truth.get("authoritative_state", {}) or {}).get("runtime_revision", 0) or 0) <= 0:
            single_truth_partials.append("missing_runtime_revision")
        single_truth = {
            "namespace": "release_16.subjectivity.single_truth",
            "status": _status_from_evidence(failures=single_truth_failures, partials=single_truth_partials),
            "checks": single_truth_checks,
            "state_truth": state_truth,
            "blocking_reasons": single_truth_failures,
            "missing_evidence": single_truth_partials,
            "projection_only_truth_fields": projection_only_truth_fields,
        }
        memory_rounds = [row for row in rounds if dict(row.get("memory_evidence", {}) or {})]
        memory_misses = sum(1 for row in memory_rounds if dict(row.get("memory_evidence", {}) or {}).get("miss"))
        memory_contamination = sum(
            1
            for row in memory_rounds
            if dict(dict(row.get("memory_evidence", {}) or {}).get("contamination", {}) or {}).get("detected")
        )
        memory_cue_rescue = sum(1 for row in memory_rounds if dict(row.get("memory_evidence", {}) or {}).get("cue_rescue"))
        memory_latency_count = 0
        memory_behavior_count = 0
        imperfect_memory_rounds = 0
        for row in memory_rounds:
            evidence = dict(row.get("memory_evidence", {}) or {})
            latency_cost = dict(evidence.get("latency_cost", {}) or {})
            contamination = dict(evidence.get("contamination", {}) or {})
            behavior = dict(evidence.get("behavioral_consequence", {}) or {})
            latency_observed = "latency_ms" in latency_cost and int(latency_cost.get("latency_ms", 0) or 0) >= 0
            if latency_observed and str(latency_cost.get("cost_tier") or "").strip():
                memory_latency_count += 1
            if str(behavior.get("selected_action") or "").strip() and (
                float(behavior.get("recall_strength", 0.0) or 0.0) > 0.0
                or bool(str(behavior.get("initiative_top_intent") or "").strip())
                or _has_payload(dict(behavior.get("initiative_memory_backing", {}) or {}))
            ):
                memory_behavior_count += 1
            if (
                bool(evidence.get("miss"))
                or bool(evidence.get("cue_rescue"))
                or bool(contamination.get("detected"))
                or float(evidence.get("interference", 0.0) or 0.0) > 0.0
                or str(evidence.get("mode") or "").strip() in {"gist", "none"}
                or not bool(evidence.get("detail", False))
            ):
                imperfect_memory_rounds += 1
        memory_failures: list[str] = []
        memory_partials: list[str] = []
        if memory_rounds:
            if memory_behavior_count == 0:
                memory_failures.append("missing_behavioral_consequence")
            if imperfect_memory_rounds == 0:
                memory_failures.append("perfect_recall_only")
            if memory_latency_count == 0:
                memory_partials.append("missing_latency_cost")
            if memory_misses == 0:
                memory_partials.append("no_recall_failure")
            if memory_cue_rescue == 0:
                memory_partials.append("no_cue_rescue")
            if memory_contamination == 0:
                memory_partials.append("no_summary_contamination")
        else:
            memory_partials.append("no_memory_rounds")
        memory = {
            "namespace": "release_16.subjectivity.memory",
            "status": _status_from_evidence(failures=memory_failures, partials=memory_partials),
            "rounds": len(memory_rounds),
            "miss_count": memory_misses,
            "contamination_count": memory_contamination,
            "cue_rescue_count": memory_cue_rescue,
            "latency_count": memory_latency_count,
            "behavioral_consequence_count": memory_behavior_count,
            "imperfect_rounds": imperfect_memory_rounds,
            "latest": dict(latest_round.get("memory_evidence", {}) or {}),
            "blocking_reasons": memory_failures,
            "missing_evidence": memory_partials,
        }
        competition_rounds = [row for row in rounds if dict(row.get("arbitration_record", {}) or {})]
        competition_winners = 0
        competition_multi_proposal_rounds = 0
        competition_evidence_sources = 0
        competition_budget_rounds = 0
        competition_rejected_rounds = 0
        competition_consequence_rounds = 0
        for row in competition_rounds:
            record = dict(row.get("arbitration_record", {}) or {})
            if str(record.get("selected_winner") or "").strip():
                competition_winners += 1
            if len(list(record.get("competing_proposals", []) or [])) >= 2:
                competition_multi_proposal_rounds += 1
            if list(record.get("evidence_sources", []) or []):
                competition_evidence_sources += 1
            if _has_payload(dict(record.get("budget_consumption", {}) or {})):
                competition_budget_rounds += 1
            if list(record.get("rejected_options", []) or []):
                competition_rejected_rounds += 1
            if _has_payload(dict(record.get("consequence_patch", {}) or {})):
                competition_consequence_rounds += 1
        competition_failures: list[str] = []
        competition_partials: list[str] = []
        if competition_rounds:
            if competition_winners == 0:
                competition_failures.append("missing_selected_winner")
            if competition_evidence_sources == 0:
                competition_failures.append("missing_evidence_sources")
            if competition_consequence_rounds == 0:
                competition_failures.append("missing_consequence_patch")
            if competition_multi_proposal_rounds == 0:
                competition_partials.append("no_multi_proposal_rounds")
            if competition_rejected_rounds == 0:
                competition_partials.append("no_rejected_options")
            if competition_budget_rounds == 0:
                competition_partials.append("no_budget_consumption")
        else:
            competition_partials.append("no_arbitration_rounds")
        competition = {
            "namespace": "release_16.subjectivity.competition",
            "status": _status_from_evidence(failures=competition_failures, partials=competition_partials),
            "rounds": len(competition_rounds),
            "selected_winners": [
                str(dict(row.get("arbitration_record", {}) or {}).get("selected_winner") or "")
                for row in competition_rounds[-5:]
            ],
            "latest": dict(latest_round.get("arbitration_record", {}) or {}),
            "multi_proposal_rounds": competition_multi_proposal_rounds,
            "rejected_rounds": competition_rejected_rounds,
            "consequence_rounds": competition_consequence_rounds,
            "blocking_reasons": competition_failures,
            "missing_evidence": competition_partials,
        }
        history_burden = to_dict(current_state.history_burden)
        repair_scars = list(history_burden.get("repair_scars", []) or [])
        suppressed_patterns = list(history_burden.get("suppressed_but_recoverable_patterns", []) or [])
        unfinished_commitments = list(history_burden.get("unfinished_commitments", []) or [])
        meaning_debts = list(history_burden.get("meaning_debts", []) or [])
        conflict_residues = dict(history_burden.get("conflict_residues", {}) or {})
        continuity_residues = dict(history_burden.get("continuity_residues", {}) or {})
        history_inertia_score = round(float(history_burden.get("history_inertia_score", 0.0) or 0.0), 4)
        meaningful_conflict_residue = (
            bool(str(conflict_residues.get("repair_mode") or "").strip())
            or bool(str(conflict_residues.get("safe_mode_owner") or "").strip())
            or bool(str(conflict_residues.get("last_conflict_priority") or "").strip())
            or bool(str(conflict_residues.get("last_compromise_template") or "").strip())
            or bool(str(conflict_residues.get("repair_stage") or "").strip())
            or any(
                int(conflict_residues.get(key, 0) or 0) > 0
                for key in ("critical_conflict_streak", "conflict_hot_rounds", "conflict_recovery_rounds")
            )
        )
        meaningful_continuity_residue = (
            bool(str(continuity_residues.get("continuity_nonce") or "").strip())
            or bool(continuity_residues.get("last_checkpoint_id"))
            or float(continuity_residues.get("self_continuity", 0.0) or 0.0) > 0.0
            or float(continuity_residues.get("meaning_strength", 0.0) or 0.0) > 0.0
            or float(continuity_residues.get("memory_fragments", 0.0) or 0.0) > 0.0
        )
        history_failures: list[str] = []
        history_partials: list[str] = []
        if not (
            repair_scars
            or suppressed_patterns
            or unfinished_commitments
            or meaning_debts
            or meaningful_conflict_residue
            or meaningful_continuity_residue
        ):
            history_failures.append("missing_history_burden")
        if history_inertia_score <= 0.0:
            history_failures.append("missing_history_inertia")
        latest_history_delta = dict(latest_round.get("history_burden_delta", {}) or {})
        if not _meaningful_delta(latest_history_delta):
            history_partials.append("no_history_delta")
        history = {
            "namespace": "release_16.subjectivity.history",
            "status": _status_from_evidence(failures=history_failures, partials=history_partials),
            "history_burden": history_burden,
            "latest_delta": latest_history_delta,
            "blocking_reasons": history_failures,
            "missing_evidence": history_partials,
        }
        latest_continuity = dict(latest_round.get("snapshot_continuity", {}) or {})
        current_continuity_nonce = str(current_state.subject_core.continuity_nonce or "").strip()
        latest_continuity_nonce = str(latest_continuity.get("continuity_nonce") or "").strip()
        recovery_failures: list[str] = []
        recovery_partials: list[str] = []
        if latest_continuity and current_continuity_nonce and latest_continuity_nonce and latest_continuity_nonce != current_continuity_nonce:
            recovery_failures.append("continuity_nonce_mismatch")
        if (
            latest_continuity
            and current_state.last_checkpoint_id
            and str(latest_continuity.get("last_checkpoint_id") or "").strip()
            and str(latest_continuity.get("last_checkpoint_id") or "").strip() != str(current_state.last_checkpoint_id)
        ):
            recovery_failures.append("checkpoint_mismatch")
        if latest_continuity and "rewind_supported" in latest_continuity and not bool(latest_continuity.get("rewind_supported")):
            recovery_failures.append("rewind_not_supported")
        if not current_state.last_checkpoint_id:
            recovery_partials.append("no_checkpoint")
        if not latest_continuity:
            recovery_partials.append("no_snapshot_continuity")
        else:
            if not latest_continuity_nonce:
                recovery_partials.append("missing_continuity_nonce")
            if not _meaningful_delta(dict(latest_continuity.get("history_burden_delta", {}) or {})):
                recovery_partials.append("missing_history_burden_delta")
            if int(latest_continuity.get("runtime_revision", 0) or 0) <= 0:
                recovery_partials.append("missing_runtime_revision")
        recovery = {
            "namespace": "release_16.subjectivity.recovery",
            "status": _status_from_evidence(failures=recovery_failures, partials=recovery_partials),
            "latest": latest_continuity,
            "last_checkpoint_id": current_state.last_checkpoint_id,
            "continuity_nonce": current_state.subject_core.continuity_nonce,
            "blocking_reasons": recovery_failures,
            "missing_evidence": recovery_partials,
        }
        drift_failures: list[str] = []
        drift_partials: list[str] = []
        online_projection_coverage = round(float(long_run_prior.get("online_projection_coverage", 0.0) or 0.0), 4)
        self_consistency_score_avg = round(float(long_run_prior.get("self_consistency_score_avg", 0.0) or 0.0), 4)
        volatility_signal_avg = round(float(long_run_prior.get("volatility_signal_avg", 0.0) or 0.0), 4)
        self_continuity = round(float(current_state.self_continuity or 0.0), 4)
        long_run_samples = list(long_run_prior.get("samples", []) or [])
        if rounds:
            if online_projection_coverage < 0.15:
                drift_failures.append("projection_coverage_collapse")
            if self_consistency_score_avg < 0.25:
                drift_failures.append("continuity_collapse")
            if volatility_signal_avg > 0.9:
                drift_failures.append("volatility_collapse")
            if self_continuity < 0.25:
                drift_failures.append("state_continuity_collapse")
        else:
            drift_partials.append("no_rounds")
        if rounds:
            if not long_run_samples:
                drift_partials.append("no_long_run_samples")
            if online_projection_coverage < 0.8:
                drift_partials.append("projection_coverage_below_release_bar")
            if self_consistency_score_avg < 0.7:
                drift_partials.append("continuity_below_release_bar")
            if volatility_signal_avg > 0.45:
                drift_partials.append("volatility_above_release_bar")
            if history_inertia_score <= 0.0:
                drift_partials.append("no_history_inertia")
        drift = {
            "namespace": "release_16.subjectivity.drift",
            "status": _status_from_evidence(failures=drift_failures, partials=drift_partials),
            "long_run_prior": long_run_prior,
            "history_inertia_score": history_inertia_score,
            "continuity": self_continuity,
            "blocking_reasons": drift_failures,
            "missing_evidence": drift_partials,
        }
        subjectivity_sections = {
            "single_truth": single_truth,
            "memory": memory,
            "competition": competition,
            "history": history,
            "recovery": recovery,
            "drift": drift,
        }
        subjectivity_status = (
            "fail"
            if any(section["status"] == "fail" for section in subjectivity_sections.values())
            else "pass"
            if all(section["status"] == "pass" for section in subjectivity_sections.values())
            else "partial"
        )
        return {
            "window": window,
            "rounds_considered": len(rounds),
            "probability_field_coverage": round(len(probability_field_rounds) / total, 4),
            "action_audit_coverage": round(len(action_audit_rounds) / total, 4),
            "token_audit_coverage": round(len(token_audit_rounds) / total, 4),
            "renderer_token_coverage": round(len(renderer_token_rounds) / total, 4),
            "token_source_integrity_rate": round(token_source_integrity_passes / total, 4),
            "bypass_detection": self._bypass_detection_summary(rounds),
            "parallel_evidence": self._parallel_evidence_summary(rounds),
            "scale_consistency": self._scale_consistency_summary(rounds),
            "cross_layer_coupling": self._cross_layer_coupling_summary(rounds),
            "conflict_arbitration": self._conflict_arbitration_acceptance_summary(rounds),
            "long_run_prior": long_run_prior,
            "renderer_decision_integrity": self._renderer_decision_integrity_summary(rounds),
            "memory_write_gate": self._memory_write_gate_summary(rounds),
            "failure_taxonomy": self._failure_taxonomy_summary(rounds),
            "release_15": {
                "controlled_learning": {
                    "learning_mode": str(autonomy_settings.get("learning_mode", "guided-learn") or "guided-learn"),
                    "allowed_network_domains": list(autonomy_settings.get("allowed_network_domains", []) or []),
                    "writable_roots": list(autonomy_settings.get("writable_roots", []) or []),
                    "knowledge_roots": list(autonomy_settings.get("knowledge_roots", []) or []),
                    "learning_log_dir": str(autonomy_settings.get("learning_log_dir", "") or ""),
                    "trace_external_learning": bool(autonomy_settings.get("trace_external_learning", True)),
                    "network_enabled": bool(autonomy_settings.get("network_enabled", False)),
                    "external_io_enabled": bool(autonomy_settings.get("external_io_enabled", False)),
                },
                "bounded_long_run": bounded_long_run,
            },
            "release_16": {
                "release_frontdoor": {
                    "status": "pass",
                    "gate": "release_frontdoor",
                    "read_model_contract": "/workbench/read-model",
                    "round_contract": "/workbench/round/{id}",
                },
                "release_subjectivity": {
                    "status": subjectivity_status,
                    "gate": "release_subjectivity",
                    "blocking_items": [name for name, section in subjectivity_sections.items() if section["status"] != "pass"],
                },
                "subjectivity": subjectivity_sections,
            },
            "storage": self._trace_storage_payload(),
        }

    def compact_traces(self) -> dict[str, Any]:
        payload = self.export_trace_parquet(overwrite=True)
        payload["parquet_path"] = payload["tables"]["round_trace"]["path"]
        return payload

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
        total = len(generated_rounds) or 1
        task_rounds = [item for item in generated_rounds if item["scenario"] == "task"]
        task_successes = sum(1 for item in task_rounds if item["sampled_action"] in {"respond", "plan", "recall", "clarify"})
        safe_mode_rounds = sum(1 for item in generated_rounds if item.get("state_snapshot", {}).get("safe_mode"))
        critical_conflicts = [
            item
            for item in generated_rounds
            if (item.get("conflict_arbitration", {}) or {}).get("total_score", 0.0)
            >= self.config["thresholds"]["thresholds"]["conflict_critical"]
        ]
        probability_field_rounds = [item for item in generated_rounds if item.get("probability_field")]
        action_audit_rounds = [
            item
            for item in generated_rounds
            if item.get("probability_field", {}).get("action", {}).get("contribution_audit")
        ]
        token_audit_rounds = [
            item
            for item in generated_rounds
            if item.get("probability_field", {}).get("token", {}).get("contribution_audit")
        ]
        renderer_token_rounds = [
            item
            for item in token_audit_rounds
            if any(
                isinstance(row, dict) and row.get("module_name") == "Renderer"
                for row in item.get("probability_field", {}).get("token", {}).get("contribution_audit", [])
            )
        ]
        token_source_integrity_passes = 0
        for item in generated_rounds:
            token_field = item.get("probability_field", {}).get("token", {})
            token_audit = token_field.get("contribution_audit", []) if isinstance(token_field, dict) else []
            if not token_audit:
                token_source_integrity_passes += 1
                continue
            if all(not row.get("dependency_trace", []) or row.get("module_name") for row in token_audit if isinstance(row, dict)):
                token_source_integrity_passes += 1
        habit_strengths = [item["strength"] for item in self.habit_top(limit=10)]
        recall_gist = sum(1 for item in generated_rounds if item.get("decision_context", {}).get("recall", {}).get("mode") == "gist")
        recall_detail = sum(1 for item in generated_rounds if item.get("decision_context", {}).get("recall", {}).get("mode") == "detail")
        relation_checks = []
        for item in generated_rounds:
            target = item.get("event_payload", {}).get("target")
            if not target:
                continue
            closeness = item.get("decision_context", {}).get("closeness", 0.5)
            action = item["sampled_action"]
            relation_checks.append(action in {"connect", "clarify", "respond", "recall"} if closeness >= 0.55 else action != "connect")
        return {
            **self.metrics_summary(),
            "generated_rounds": total,
            "crash_rate": 0.0,
            "safe_mode_rate": round(safe_mode_rounds / total, 4),
            "conflict_deadloop_rate": round(len(critical_conflicts) / total, 4),
            "scarcity_burn_drop": 0.0,
            "habit_gradient": round((sum(habit_strengths) / max(len(habit_strengths), 1)) / total, 4),
            "gist_detail_ratio": round(recall_gist / max(recall_detail, 1), 4),
            "relation_consistency": round(sum(1 for item in relation_checks if item) / max(len(relation_checks), 1), 4),
            "task_success_rate": round(task_successes / max(len(task_rounds), 1), 4),
            "top_driver_coverage": round(sum(1 for item in generated_rounds if len(item.get("top_drivers", [])) >= 3) / total, 4),
            "probability_field_coverage": round(len(probability_field_rounds) / total, 4),
            "action_audit_coverage": round(len(action_audit_rounds) / total, 4),
            "token_audit_coverage": round(len(token_audit_rounds) / total, 4),
            "renderer_token_coverage": round(len(renderer_token_rounds) / total, 4),
            "token_source_integrity_rate": round(token_source_integrity_passes / total, 4),
            "cross_layer_coupling": self._cross_layer_coupling_summary(generated_rounds),
            "conflict_arbitration": self._conflict_arbitration_acceptance_summary(generated_rounds),
            "long_run_prior": self._long_run_prior_summary(generated_rounds),
            "memory_write_gate": self._memory_write_gate_summary(generated_rounds),
            "storage": self._trace_storage_payload(),
        }

    def _observer_action_layer_from_trace(self, trace: dict[str, Any]) -> dict[str, Any]:
        probability_field = canonical_probability_field_payload(trace)
        action_layer = probability_field.get("action", {})
        return action_layer if isinstance(action_layer, dict) else {}

    def _observer_trace_derived_cache_entry(self, trace: dict[str, Any]) -> dict[str, Any] | None:
        round_id = int(trace.get("round_id", 0) or 0)
        if round_id <= 0:
            return None
        entry = self._trace_derived_cache.setdefault(
            round_id,
            {
                "action_probability_explanations": {},
                "counterfactual_replays": {},
            },
        )
        if len(self._trace_derived_cache) > self._trace_round_cache_limit:
            for stale_round_id in sorted(self._trace_derived_cache)[:-self._trace_round_cache_limit]:
                self._trace_derived_cache.pop(stale_round_id, None)
        return entry

    def _observer_token_state_from_trace(self, trace: dict[str, Any]) -> dict[str, Any]:
        probability_field = canonical_probability_field_payload(trace)
        token_state = probability_field.get("token_state", {})
        return token_state if isinstance(token_state, dict) else {}

    def _observer_cross_layer_coupling_verdict(self, trace: dict[str, Any]) -> dict[str, Any]:
        probability_field = canonical_probability_field_payload(trace)
        couplings = probability_field.get("couplings", []) if isinstance(probability_field, dict) else []
        allowed_pairs = {
            ("context", "memory", "context_route"),
            ("memory", "action", "memory_prior"),
            ("memory", "action", "organic_memory"),
            ("action", "token", "render_plan"),
        }
        observed_pairs: list[str] = []
        illegal_pairs: list[str] = []
        for row in couplings:
            if not isinstance(row, dict):
                continue
            pair = (
                str(row.get("source_layer", "")),
                str(row.get("target_layer", "")),
                str(row.get("carrier_signal", "")),
            )
            joined = "->".join(pair)
            observed_pairs.append(joined)
            if pair not in allowed_pairs or not bool(row.get("enabled", True)):
                illegal_pairs.append(joined)
        return {
            "observed_pairs": observed_pairs,
            "illegal_pairs": illegal_pairs,
            "legal": not illegal_pairs,
        }

    def _observer_renderer_decision_integrity(self, trace: dict[str, Any]) -> dict[str, Any]:
        raw = dict(trace.get("renderer_decision_integrity", {}) or {})
        sampled_action = str(trace.get("sampled_action") or "")
        render_plan = dict(trace.get("render_plan", {}) or {})
        probability_field = dict(trace.get("probability_field", {}) or {})
        action_layer = probability_field.get("action", {}) if isinstance(probability_field.get("action", {}), dict) else {}
        winner_posterior = dict(action_layer.get("winner_posterior", {}) or {})
        locked_action = str(raw.get("locked_action") or sampled_action)
        render_plan_action = str(raw.get("render_plan_action") or render_plan.get("action") or "")
        post_render_action = str(raw.get("post_render_action") or sampled_action)
        winner_target = str(raw.get("winner_target") or action_layer.get("winner_target") or sampled_action)
        mutation_reasons: list[str] = []
        if locked_action and render_plan_action and render_plan_action != locked_action:
            mutation_reasons.append("render_plan_action_mismatch")
        if locked_action and post_render_action and post_render_action != locked_action:
            mutation_reasons.append("post_render_action_mismatch")
        return {
            "locked_action": locked_action,
            "render_plan_action": render_plan_action,
            "post_render_action": post_render_action,
            "winner_target": winner_target,
            "field_peak_matches_locked_action": not (locked_action and winner_target and winner_target != locked_action),
            "sampled_differs_from_peak": bool(locked_action and winner_target and winner_target != locked_action),
            "locked_probability": round(float(raw.get("locked_probability", winner_posterior.get(locked_action or sampled_action, 0.0) or 0.0)), 6),
            "gate_at_render": round(float(raw.get("gate_at_render", render_plan.get("safety_constraints", {}).get("gate", 1.0) or 0.0)), 6),
            "auth_guard_action": str(raw.get("auth_guard_action") or trace.get("authenticity", {}).get("guard_action") or ""),
            "decision_mutated": bool(mutation_reasons),
            "renderer_consumes_final_field": not mutation_reasons,
            "mutation_reasons": mutation_reasons,
        }

    def _observer_competing_peaks(self, action_layer: dict[str, Any], *, limit: int = 3) -> list[dict[str, Any]]:
        peak_rows = action_layer.get("counterfactual_top_peaks", [])
        if isinstance(peak_rows, list) and peak_rows:
            rows: list[dict[str, Any]] = []
            for item in peak_rows[:limit]:
                if not isinstance(item, dict):
                    continue
                rows.append(
                    {
                        "action": str(item.get("target", "")),
                        "final_energy": round(float(item.get("final_energy", 0.0) or 0.0), 6),
                        "posterior": round(float(item.get("posterior", 0.0) or 0.0), 6),
                    }
                )
            if rows:
                return rows
        final_energy = action_layer.get("final_energy", {})
        if not isinstance(final_energy, dict):
            return []
        rows = []
        for action_name, value in final_energy.items():
            if value == float("-inf"):
                continue
            rows.append(
                {
                    "action": action_name,
                    "final_energy": round(float(value), 6),
                    "posterior": round(float(action_layer.get("winner_posterior", {}).get(action_name, 0.0) or 0.0), 6),
                }
            )
        rows.sort(key=lambda item: (item["posterior"], item["final_energy"]), reverse=True)
        return rows[:limit]

    def _observer_stacked_action_contributions(
        self,
        action_layer: dict[str, Any],
        target_action: str,
        *,
        limit: int = 16,
    ) -> list[dict[str, Any]]:
        audit_rows = action_layer.get("contribution_audit", [])
        if not isinstance(audit_rows, list):
            return []
        stacked: list[dict[str, Any]] = []
        for row in audit_rows:
            if not isinstance(row, dict):
                continue
            projected = row.get("delta_projected", {}) if isinstance(row.get("delta_projected"), dict) else {}
            normalized = row.get("delta_normalized", {}) if isinstance(row.get("delta_normalized"), dict) else {}
            hard_masked = list(row.get("hard_masked_targets", []) or [])
            touched = target_action in projected or target_action in normalized or target_action in hard_masked
            if not touched:
                continue
            normalized_value = round(float(normalized.get(target_action, 0.0) or 0.0), 6)
            projected_value = round(float(projected.get(target_action, 0.0) or 0.0), 6)
            if target_action in hard_masked:
                direction = "block"
            elif normalized_value > 0.0:
                direction = "support"
            elif normalized_value < 0.0 or projected_value < 0.0:
                direction = "suppress"
            else:
                direction = "neutral"
            stacked.append(
                {
                    "module_name": row.get("module_name", ""),
                    "module_type": row.get("module_type", ""),
                    "direction": direction,
                    "delta_projected": projected_value,
                    "delta_normalized": normalized_value,
                    "hard_masked": target_action in hard_masked,
                    "trace_reason": row.get("trace_reason", ""),
                    "projection_reason": row.get("projection_reason", ""),
                }
            )
        seen_modules = {item["module_name"] for item in stacked}
        if len(stacked) < limit:
            background_rows: list[dict[str, Any]] = []
            for row in audit_rows:
                if not isinstance(row, dict):
                    continue
                module_name = str(row.get("module_name", "") or "")
                if not module_name or module_name in seen_modules:
                    continue
                projected = row.get("delta_projected", {}) if isinstance(row.get("delta_projected"), dict) else {}
                normalized = row.get("delta_normalized", {}) if isinstance(row.get("delta_normalized"), dict) else {}
                background_rows.append(
                    {
                        "module_name": module_name,
                        "module_type": row.get("module_type", ""),
                        "direction": "background",
                        "delta_projected": round(
                            max((abs(float(value)) for value in projected.values()), default=0.0),
                            6,
                        ),
                        "delta_normalized": round(
                            max((abs(float(value)) for value in normalized.values()), default=0.0),
                            6,
                        ),
                        "hard_masked": False,
                        "trace_reason": row.get("trace_reason", ""),
                        "projection_reason": row.get("projection_reason", ""),
                    }
                )
            background_rows.sort(
                key=lambda item: (
                    0 if item["module_name"] in {"SkillExecutor", "LongRunAnalyzer"} else 1,
                    -abs(float(item["delta_normalized"])),
                    -abs(float(item["delta_projected"])),
                )
            )
            stacked.extend(background_rows[: max(0, limit - len(stacked))])
        stacked.sort(
            key=lambda item: (
                0 if item["hard_masked"] else 1,
                -abs(float(item["delta_normalized"])),
                -abs(float(item["delta_projected"])),
            )
        )
        return stacked[:limit]

    def _observer_action_probability_explanation(self, trace: dict[str, Any], *, target_action: str) -> dict[str, Any]:
        normalized_target_action = str(target_action or "")
        cache_entry = self._trace_derived_cache_entry(trace)
        if cache_entry is not None:
            explanation_cache = cache_entry.setdefault("action_probability_explanations", {})
            cached = explanation_cache.get(normalized_target_action)
            if isinstance(cached, dict):
                return copy.deepcopy(cached)
        action_layer = self._action_layer_from_trace(trace)
        final_energy = action_layer.get("final_energy", {}) if isinstance(action_layer.get("final_energy"), dict) else {}
        winner_target = str(action_layer.get("winner_target") or trace.get("sampled_action") or "")
        sampled_action = str(trace.get("sampled_action") or "")
        stacked = self._stacked_action_contributions(action_layer, normalized_target_action)
        competing_peaks = self._competing_peaks(action_layer)
        if winner_target:
            winner_peak = next((item for item in competing_peaks if item.get("action") == winner_target), None)
            if winner_peak is None:
                winner_peak = {
                    "action": winner_target,
                    "final_energy": round(float(final_energy.get(winner_target, 0.0) or 0.0), 6)
                    if final_energy.get(winner_target) != float("-inf")
                    else float("-inf"),
                    "posterior": round(float(action_layer.get("winner_posterior", {}).get(winner_target, 0.0) or 0.0), 6),
                }
            competing_peaks = [winner_peak] + [
                item
                for item in competing_peaks
                if item.get("action") != winner_target
            ]
        result = {
            "target_action": normalized_target_action,
            "winner_target": winner_target,
            "sampled_action": sampled_action,
            "sampled_differs_from_peak": bool(sampled_action and winner_target and sampled_action != winner_target),
            "winner_posterior": dict(action_layer.get("winner_posterior", {}) or {}),
            "winner_energy": round(float(final_energy.get(winner_target, 0.0) or 0.0), 6) if winner_target else 0.0,
            "target_final_energy": round(float(final_energy.get(normalized_target_action, 0.0) or 0.0), 6)
            if normalized_target_action in final_energy and final_energy.get(normalized_target_action) != float("-inf")
            else None,
            "hard_masked": normalized_target_action in list(action_layer.get("hard_masked_targets", []) or []),
            "stacked_contributions": stacked,
            "competing_peaks": competing_peaks[:3],
        }
        if cache_entry is not None:
            explanation_cache = cache_entry.setdefault("action_probability_explanations", {})
            explanation_cache[normalized_target_action] = copy.deepcopy(result)
        return result

    def _observer_counterfactual_render_preview(self, trace: dict[str, Any], action: str) -> dict[str, Any]:
        render_plan_payload = dict(trace.get("render_plan", {}) or {})
        if not render_plan_payload:
            return {
                "action": action,
                "would_output": action != "nothing",
                "text": "",
                "terminal_intent": action == "die",
            }
        render_plan_payload["action"] = action
        render_plan_payload["delivery_mode"] = "monologue" if action == "monologue" else "speech"
        message_plan = dict(render_plan_payload.get("message_plan", {}) or {})
        message_plan["intent"] = action
        message_plan["focus"] = action
        message_plan["delivery_mode"] = render_plan_payload["delivery_mode"]
        render_plan_payload["message_plan"] = message_plan
        preview_plan = RenderPlan(**render_plan_payload)
        preview_text = fallback_render_text(preview_plan)
        return {
            "action": action,
            "would_output": bool(preview_text),
            "text": preview_text,
            "terminal_intent": action == "die",
        }

    def _observer_emergent_action_formalization_payload(
        self,
        sketches: list[dict[str, Any]] | list[Any],
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for raw in sketches:
            sketch = raw if isinstance(raw, dict) else to_dict(raw)
            status = str(sketch.get("status", "latent") or "latent")
            if status != "formalized":
                continue
            rows.append(
                {
                    "name": str(sketch.get("name") or ""),
                    "status": status,
                    "growth_score": round(float(sketch.get("growth_score", 0.0) or 0.0), 4),
                    "anchor_alignment": round(float(sketch.get("anchor_alignment", 0.0) or 0.0), 4),
                    "stability": int(sketch.get("stability", 0) or 0),
                    "target_action_map": dict(sketch.get("target_action_map", {}) or {}),
                }
            )
        return rows

    def _observer_sample_counterfactual_action(
        self,
        distribution: dict[str, float],
        *,
        seed: int,
        default_action: str,
    ) -> str:
        normalized = self._normalize(distribution or {})
        if not normalized:
            return default_action
        ordered = sorted(normalized.items())
        seed_payload = f"{seed}:{','.join(f'{action}:{value:.6f}' for action, value in ordered)}"
        threshold = (int(hashlib.sha1(seed_payload.encode("utf-8")).hexdigest()[:8], 16) % 1000000) / 1000000.0
        cumulative = 0.0
        sampled = default_action or ordered[-1][0]
        for action, probability in sorted(normalized.items(), key=lambda item: item[1], reverse=True):
            cumulative += max(float(probability), 0.0)
            sampled = action
            if threshold <= cumulative:
                break
        return sampled

    def _observer_counterfactual_replays_from_trace(self, trace: dict[str, Any], *, limit: int = 4) -> list[dict[str, Any]]:
        normalized_limit = max(1, int(limit))
        cache_entry = self._trace_derived_cache_entry(trace)
        if cache_entry is not None:
            replay_cache = cache_entry.setdefault("counterfactual_replays", {})
            cached = replay_cache.get(normalized_limit)
            if isinstance(cached, list):
                return copy.deepcopy(cached)
        instinct_field = dict(trace.get("state_snapshot", {}).get("instinct_field", {}) or {})
        emergent_sketches = list(trace.get("state_snapshot", {}).get("emergent_action_sketches", []) or [])
        personality_anchor = dict(trace.get("state_snapshot", {}).get("personality_anchor", {}) or {})
        action_layer = self._action_layer_from_trace(trace)
        final_energy = dict(action_layer.get("final_energy", {}) or {})
        rows: list[dict[str, Any]] = []
        action_order: list[str] = []
        for action in list(instinct_field.get("candidate_actions", []) or []):
            action_name = str(action).strip()
            if action_name and action_name in final_energy and action_name not in action_order:
                action_order.append(action_name)
        peaks = self._competing_peaks(action_layer, limit=max(normalized_limit, 6))
        for peak in peaks:
            action_name = str(peak.get("action") or "").strip()
            if action_name and action_name not in action_order:
                action_order.append(action_name)
        for action in action_order[:normalized_limit]:
            peak = next((item for item in peaks if item.get("action") == action), None)
            if peak is None:
                peak = {
                    "action": action,
                    "posterior": round(float(action_layer.get("winner_posterior", {}).get(action, 0.0) or 0.0), 6),
                    "final_energy": round(float(final_energy.get(action, 0.0) or 0.0), 6),
                }
            action = str(peak.get("action") or "").strip()
            if not action:
                continue
            explanation = self._action_probability_explanation(trace, target_action=action)
            sketch_hits = []
            for sketch in emergent_sketches:
                if not isinstance(sketch, dict):
                    continue
                support_actions = dict(sketch.get("support_actions", {}) or {})
                if action not in support_actions:
                    continue
                sketch_hits.append(
                    {
                        "name": str(sketch.get("name") or ""),
                        "growth_score": round(float(sketch.get("growth_score", 0.0) or 0.0), 4),
                        "status": str(sketch.get("status") or "latent"),
                    }
                )
            render_preview = self._counterfactual_render_preview(trace, action)
            rows.append(
                {
                    "action": action,
                    "posterior": round(float(peak.get("posterior", 0.0) or 0.0), 6),
                    "final_energy": round(float(peak.get("final_energy", 0.0) or 0.0), 6),
                    "winner_region": instinct_field.get("winner_region", ""),
                    "region_scores": dict(instinct_field.get("region_scores", {}) or {}),
                    "collapse_trace": dict(instinct_field.get("collapse_trace", {}) or {}),
                    "supporters": [
                        item["module_name"]
                        for item in explanation["stacked_contributions"]
                        if item["direction"] in {"support", "background"} and not item["hard_masked"]
                    ][:4],
                    "blockers": [
                        item["module_name"]
                        for item in explanation["stacked_contributions"]
                        if item["direction"] in {"block", "suppress"} or item["hard_masked"]
                    ][:4],
                    "emergent_sketches": sketch_hits[:3],
                    "anchor_alignment": round(float(personality_anchor.get("alignment", 0.0) or 0.0), 4),
                    "counterfactual_preview": render_preview,
                }
            )
        if cache_entry is not None:
            replay_cache = cache_entry.setdefault("counterfactual_replays", {})
            replay_cache[normalized_limit] = copy.deepcopy(rows)
        return rows

    def _observer_conflict_arbitration_summary(self, trace: dict[str, Any]) -> dict[str, Any]:
        explicit = dict(trace.get("conflict_arbitration", {}) or {})
        action_layer = self._action_layer_from_trace(trace)
        audit_rows = action_layer.get("contribution_audit", [])
        conflict_row = next(
            (
                row
                for row in audit_rows
                if isinstance(row, dict) and row.get("module_name") == "ConflictMonitorAgent"
            ),
            {},
        )
        posterior = dict(conflict_row.get("posterior", {}) or {})
        peak_clusters = list(conflict_row.get("peak_clusters", []) or [])
        compromise_template_prior = dict(conflict_row.get("compromise_template_prior", {}) or {})
        dependency_trace = [str(item) for item in list(conflict_row.get("dependency_trace", []) or [])]
        winning_priority = ""
        for item in dependency_trace:
            if item.startswith("winning_priority:"):
                winning_priority = item.split(":", 1)[1].strip()
                break
        if not winning_priority:
            gate_rows = [item for item in trace.get("gate_decisions", []) if isinstance(item, dict) and item.get("stage") == "conflict"]
            for row in gate_rows:
                winning_priority = str(row.get("winning_priority") or "").strip()
                if winning_priority:
                    break
        if not compromise_template_prior:
            for row in trace.get("gate_decisions", []):
                if not isinstance(row, dict):
                    continue
                template = str(row.get("template") or "").strip()
                if template:
                    compromise_template_prior = {template: 1.0}
                    break
        winner_peak = max(posterior, key=posterior.get) if posterior else ""
        summary = {
            "winning_priority": winning_priority,
            "winner_peak": winner_peak,
            "winner_peak_posterior": posterior,
            "compromise_template_prior": compromise_template_prior,
            "peak_clusters": peak_clusters,
            "hard_masked_targets": list(conflict_row.get("hard_masked_targets", []) or []),
            "trace_reason": str(conflict_row.get("trace_reason", "")),
        }
        if explicit:
            merged = dict(explicit)
            for key, value in summary.items():
                if key not in merged or merged.get(key) in ({}, [], "", None):
                    merged[key] = value
            return merged
        return summary

    def _observer_average(self, values: list[float]) -> float:
        return round(sum(values) / len(values), 4) if values else 0.0

    def _observer_estimate_affect_half_life(self, rounds: list[dict[str, Any]]) -> float:
        residues = [float(trace.get("vitality_snapshot", {}).get("affect_residue", 0.0)) for trace in rounds]
        spans: list[float] = []
        for idx, peak in enumerate(residues):
            if peak < 0.25:
                continue
            if idx > 0 and peak < residues[idx - 1]:
                continue
            target = peak / 2
            for future_idx in range(idx + 1, len(residues)):
                if residues[future_idx] <= target:
                    spans.append(float(future_idx - idx))
                    break
        return self._average(spans)

    def _observer_estimate_recovery_duration(self, rounds: list[dict[str, Any]]) -> float:
        residues = [float(trace.get("vitality_snapshot", {}).get("affect_residue", 0.0)) for trace in rounds]
        spans: list[float] = []
        for idx, residue in enumerate(residues):
            if residue < 0.35:
                continue
            for future_idx in range(idx + 1, len(residues)):
                if residues[future_idx] <= 0.18:
                    spans.append(float(future_idx - idx))
                    break
        return self._average(spans)

    def _observer_event_variance_metric(self, rounds: list[dict[str, Any]], bucket_fn) -> float:
        grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
        for trace in rounds:
            vitality = trace.get("vitality_snapshot", {})
            cue = str(vitality.get("cue") or "").strip()
            if not cue:
                continue
            bucket = str(bucket_fn(trace, vitality))
            grouped.setdefault(cue, {}).setdefault(bucket, []).append(trace)

        scores: list[float] = []
        for cue_groups in grouped.values():
            if len(cue_groups) < 2:
                continue
            actions = set()
            warmth_values: list[float] = []
            directness_values: list[float] = []
            for traces in cue_groups.values():
                for trace in traces:
                    actions.add(str(trace.get("sampled_action", "")))
                    expression = trace.get("render_plan", {}).get("expression", {})
                    warmth_values.append(float(expression.get("warmth_level", 0.0)))
                    directness_values.append(float(expression.get("directness_level", 0.0)))
            action_var = (len(actions) - 1) / max(len(CORE_ACTIONS) - 1, 1)
            warmth_var = (max(warmth_values) - min(warmth_values)) if warmth_values else 0.0
            directness_var = (max(directness_values) - min(directness_values)) if directness_values else 0.0
            scores.append(round(_clip(action_var * 0.5 + warmth_var * 0.3 + directness_var * 0.2), 4))
        return self._average(scores)

    def _observer_bypass_detection_summary(self, rounds: list[dict[str, Any]]) -> dict[str, Any]:
        violations: list[dict[str, Any]] = []
        legacy_bridge_residues: list[dict[str, Any]] = []
        for trace in rounds:
            action_layer = self._action_layer_from_trace(trace)
            audit_rows = [item for item in action_layer.get("contribution_audit", []) if isinstance(item, dict)]
            action_modules = {
                str(item.get("module_name", ""))
                for item in audit_rows
                if item.get("module_name")
            }
            proposal_modules = {
                str(item.get("agent_name", ""))
                for item in trace.get("proposal_summaries", [])
                if not item.get("veto", False) and item.get("agent_name")
            }
            missing_modules = sorted(module for module in proposal_modules if module not in action_modules)
            if missing_modules:
                violations.append(
                    {
                        "round_id": trace.get("round_id"),
                        "type": "missing_action_contribution",
                        "modules": missing_modules,
                    }
                )
            if not trace.get("probability_field", {}).get("action", {}).get("contribution_audit"):
                violations.append(
                    {
                        "round_id": trace.get("round_id"),
                        "type": "empty_action_audit",
                        "modules": [],
                    }
                )
            legacy_rows = [
                {
                    "round_id": trace.get("round_id"),
                    "module_name": str(item.get("module_name", "")),
                    "projection_reason": str(item.get("projection_reason", "")),
                }
                for item in audit_rows
                if str(item.get("projection_reason", "")).startswith("legacy_bundle:")
            ]
            legacy_bridge_residues.extend(legacy_rows)
        return {
            "violation_count": len(violations),
            "violations": violations,
            "legacy_bridge_residue_count": len(legacy_bridge_residues),
            "legacy_bridge_residues": legacy_bridge_residues,
            "clean_round_rate": round((len(rounds) - len({item["round_id"] for item in violations})) / max(len(rounds), 1), 4),
        }

    def _observer_parallel_evidence_summary(self, rounds: list[dict[str, Any]]) -> dict[str, Any]:
        observed_groups: set[str] = set()
        groups_with_overlap: set[str] = set()
        round_count_with_parallel = 0
        samples: list[dict[str, Any]] = []
        for trace in rounds:
            parallel_rows = [row for row in trace.get("parallel_traces", []) if isinstance(row, dict) and row.get("parallel_group")]
            if parallel_rows:
                round_count_with_parallel += 1
            by_group: dict[str, list[dict[str, Any]]] = {}
            for row in parallel_rows:
                group = str(row.get("parallel_group"))
                observed_groups.add(group)
                by_group.setdefault(group, []).append(row)
            for group, items in by_group.items():
                items = sorted(items, key=lambda item: float(item.get("started_at_ms", 0.0) or 0.0))
                overlap = False
                for idx, current in enumerate(items):
                    current_start = float(current.get("started_at_ms", 0.0) or 0.0)
                    current_end = float(current.get("finished_at_ms", current_start) or current_start)
                    for later in items[idx + 1 :]:
                        later_start = float(later.get("started_at_ms", 0.0) or 0.0)
                        later_end = float(later.get("finished_at_ms", later_start) or later_start)
                        if later_start <= current_end and current_start <= later_end:
                            overlap = True
                            break
                    if overlap:
                        groups_with_overlap.add(group)
                        break
                if len(samples) < 8:
                    samples.append(
                        {
                            "round_id": trace.get("round_id"),
                            "parallel_group": group,
                            "task_count": len(items),
                            "overlap_detected": overlap,
                        }
                    )
        return {
            "observed_groups": sorted(observed_groups),
            "groups_with_overlap": sorted(groups_with_overlap),
            "parallel_round_rate": round(round_count_with_parallel / max(len(rounds), 1), 4),
            "true_parallel_group_rate": round(len(groups_with_overlap) / max(len(observed_groups), 1), 4),
            "samples": samples,
        }

    def _observer_scale_consistency_summary(self, rounds: list[dict[str, Any]]) -> dict[str, Any]:
        rounds_evaluated = 0
        clipped_contribution_count = 0
        total_contribution_count = 0
        overdominant_rounds = 0
        max_module_shares: list[float] = []
        samples: list[dict[str, Any]] = []
        for trace in rounds:
            action_layer = self._action_layer_from_trace(trace)
            audit_rows = [row for row in action_layer.get("contribution_audit", []) if isinstance(row, dict)]
            if not audit_rows:
                continue
            rounds_evaluated += 1
            per_module_mass: dict[str, float] = {}
            for row in audit_rows:
                total_contribution_count += 1
                stats = row.get("stats", {}) if isinstance(row.get("stats"), dict) else {}
                if bool(stats.get("clipped", False)):
                    clipped_contribution_count += 1
                normalized = dict(row.get("delta_normalized", {}) or {})
                mass = sum(abs(float(value)) for value in normalized.values())
                module_name = str(row.get("module_name", "") or "unknown")
                per_module_mass[module_name] = per_module_mass.get(module_name, 0.0) + mass
            total_mass = sum(per_module_mass.values())
            max_share = max((mass / total_mass) for mass in per_module_mass.values()) if total_mass > 0 else 0.0
            max_module_shares.append(max_share)
            if len(per_module_mass) >= 2 and max_share >= 0.85:
                overdominant_rounds += 1
            if len(samples) < 8:
                dominant_module = max(per_module_mass, key=per_module_mass.get) if per_module_mass else ""
                samples.append(
                    {
                        "round_id": trace.get("round_id"),
                        "dominant_module": dominant_module,
                        "max_module_share": round(max_share, 4),
                        "module_count": len(per_module_mass),
                    }
                )
        sorted_shares = sorted(max_module_shares)
        p95_index = min(len(sorted_shares) - 1, max(0, math.ceil(len(sorted_shares) * 0.95) - 1)) if sorted_shares else 0
        p95_share = sorted_shares[p95_index] if sorted_shares else 0.0
        return {
            "rounds_evaluated": rounds_evaluated,
            "clipped_contribution_rate": round(clipped_contribution_count / max(total_contribution_count, 1), 4),
            "overdominant_round_rate": round(overdominant_rounds / max(rounds_evaluated, 1), 4),
            "max_module_share_p95": round(p95_share, 4),
            "samples": samples,
        }

    def _observer_cross_layer_coupling_summary(self, rounds: list[dict[str, Any]]) -> dict[str, Any]:
        allowed_pairs = {
            ("context", "memory", "context_route"),
            ("memory", "action", "memory_prior"),
            ("memory", "action", "organic_memory"),
            ("action", "token", "render_plan"),
        }
        pair_hits: dict[str, int] = {"->".join(pair): 0 for pair in allowed_pairs}
        observed_pairs: set[str] = set()
        illegal: list[dict[str, Any]] = []
        compliant_rounds = 0
        rounds_with_couplings = 0
        for trace in rounds:
            probability_field = trace.get("probability_field", {})
            couplings = probability_field.get("couplings", []) if isinstance(probability_field, dict) else []
            if not couplings:
                continue
            rounds_with_couplings += 1
            round_illegal = False
            round_pairs: set[str] = set()
            for row in couplings:
                if not isinstance(row, dict):
                    continue
                pair = (
                    str(row.get("source_layer", "")),
                    str(row.get("target_layer", "")),
                    str(row.get("carrier_signal", "")),
                )
                joined = "->".join(pair)
                observed_pairs.add(joined)
                round_pairs.add(joined)
                if pair not in allowed_pairs or not bool(row.get("enabled", True)) or not str(row.get("allowed_phase", "")):
                    round_illegal = True
                    illegal.append(
                        {
                            "round_id": trace.get("round_id"),
                            "source_layer": pair[0],
                            "target_layer": pair[1],
                            "carrier_signal": pair[2],
                        }
                    )
            for allowed in pair_hits:
                if allowed in round_pairs:
                    pair_hits[allowed] += 1
            if not round_illegal:
                compliant_rounds += 1
        return {
            "observed_pairs": sorted(observed_pairs),
            "pair_coverage": {
                pair: round(count / max(rounds_with_couplings, 1), 4)
                for pair, count in sorted(pair_hits.items())
            },
            "illegal_count": len(illegal),
            "illegal_couplings": illegal,
            "compliant_round_rate": round(compliant_rounds / max(rounds_with_couplings, 1), 4),
        }

    def _observer_renderer_decision_integrity_summary(self, rounds: list[dict[str, Any]]) -> dict[str, Any]:
        violations: list[dict[str, Any]] = []
        locked_rounds = 0
        samples: list[dict[str, Any]] = []
        for trace in rounds:
            integrity = self._renderer_decision_integrity(trace)
            if integrity["renderer_consumes_final_field"]:
                locked_rounds += 1
            else:
                violations.append(
                    {
                        "round_id": trace.get("round_id"),
                        "locked_action": integrity["locked_action"],
                        "render_plan_action": integrity["render_plan_action"],
                        "post_render_action": integrity["post_render_action"],
                        "winner_target": integrity["winner_target"],
                        "mutation_reasons": integrity["mutation_reasons"],
                    }
                )
            if len(samples) < 8:
                samples.append(
                    {
                        "round_id": trace.get("round_id"),
                        "locked_action": integrity["locked_action"],
                        "auth_guard_action": integrity["auth_guard_action"],
                        "decision_mutated": integrity["decision_mutated"],
                    }
                )
        return {
            "violation_count": len(violations),
            "violations": violations,
            "decision_lock_rate": round(locked_rounds / max(len(rounds), 1), 4),
            "samples": samples,
        }

    def _observer_memory_write_gate_summary(self, rounds: list[dict[str, Any]]) -> dict[str, Any]:
        observed = 0
        blocked = 0
        reasons: dict[str, int] = {}
        samples: list[dict[str, Any]] = []
        for trace in rounds:
            gate = dict(trace.get("memory_write_gate", {}) or {})
            if not gate:
                continue
            observed += 1
            reason = str(gate.get("reason", "unknown"))
            if bool(gate.get("suppressed", False)):
                blocked += 1
                reasons[reason] = reasons.get(reason, 0) + 1
            if len(samples) < 8:
                samples.append(
                    {
                        "round_id": trace.get("round_id"),
                        "cue": gate.get("cue"),
                        "suppressed": bool(gate.get("suppressed", False)),
                        "reason": reason,
                    }
                )
        return {
            "observed_round_rate": round(observed / max(len(rounds), 1), 4),
            "suppressed_round_count": blocked,
            "suppressed_round_rate": round(blocked / max(len(rounds), 1), 4),
            "suppressed_reasons": reasons,
            "samples": samples,
        }

    def _observer_failure_taxonomy_from_trace(self, trace: dict[str, Any]) -> list[str]:
        probability_field = canonical_probability_field_payload(trace)
        seen: set[str] = set()
        ordered: list[str] = []

        def collect(values: object) -> None:
            if not isinstance(values, list):
                return
            for item in values:
                label = str(item).strip()
                if not label or label in seen:
                    continue
                seen.add(label)
                ordered.append(label)

        for layer_name in ("action", "token"):
            layer = probability_field.get(layer_name, {}) if isinstance(probability_field, dict) else {}
            if not isinstance(layer, dict):
                continue
            collect(layer.get("failure_taxonomy"))
            for audit_row in list(layer.get("contribution_audit", []) or []):
                if isinstance(audit_row, dict):
                    collect(audit_row.get("failure_taxonomy"))
        return ordered

    def _observer_failure_taxonomy_summary(self, rounds: list[dict[str, Any]]) -> dict[str, Any]:
        observed = 0
        counts: dict[str, int] = {}
        round_counts: dict[str, int] = {}
        samples: list[dict[str, Any]] = []
        total = max(len(rounds), 1)
        for trace in rounds:
            labels = self._failure_taxonomy_from_trace(trace)
            if not labels:
                continue
            observed += 1
            for label in labels:
                counts[label] = counts.get(label, 0) + 1
                round_counts[label] = round_counts.get(label, 0) + 1
            if len(samples) < 8:
                samples.append({"round_id": trace.get("round_id"), "failure_taxonomy": labels})
        return {
            "observed_round_rate": round(observed / total, 4),
            "counts": counts,
            "round_rates": {label: round(count / total, 4) for label, count in sorted(round_counts.items())},
            "samples": samples,
        }

    def _observer_conflict_arbitration_acceptance_summary(self, rounds: list[dict[str, Any]]) -> dict[str, Any]:
        observed = 0
        winning_priority_count = 0
        peak_cluster_count = 0
        compromise_template_prior_count = 0
        samples: list[dict[str, Any]] = []
        for trace in rounds:
            summary = self._conflict_arbitration_summary(trace)
            if not summary:
                continue
            observed += 1
            if str(summary.get("winning_priority", "")):
                winning_priority_count += 1
            if list(summary.get("peak_clusters", []) or []):
                peak_cluster_count += 1
            if dict(summary.get("compromise_template_prior", {}) or {}):
                compromise_template_prior_count += 1
            if len(samples) < 8:
                samples.append(
                    {
                        "round_id": trace.get("round_id"),
                        "winning_priority": summary.get("winning_priority"),
                        "winner_peak": summary.get("winner_peak"),
                        "peak_cluster_count": len(list(summary.get("peak_clusters", []) or [])),
                        "compromise_template_prior": dict(summary.get("compromise_template_prior", {}) or {}),
                    }
                )
        return {
            "observed_round_rate": round(observed / max(len(rounds), 1), 4),
            "winning_priority_coverage": round(winning_priority_count / max(observed, 1), 4),
            "peak_cluster_coverage": round(peak_cluster_count / max(observed, 1), 4),
            "compromise_template_prior_coverage": round(compromise_template_prior_count / max(observed, 1), 4),
            "samples": samples,
        }

    def _observer_long_run_prior_summary(self, rounds: list[dict[str, Any]]) -> dict[str, Any]:
        observed = 0
        online_projection_count = 0
        blocked_action_rounds = 0
        conflict_subordinated_rounds = 0
        auth_penalty_rounds = 0
        authenticity_subordinated_rounds = 0
        continuity_windows: list[float] = []
        consistency_scores: list[float] = []
        volatility_signals: list[float] = []
        samples: list[dict[str, Any]] = []
        for trace in rounds:
            projection = dict(trace.get("long_run_projection", {}) or {})
            if not projection:
                continue
            observed += 1
            online_prior = dict(projection.get("online_prior", {}) or {})
            if online_prior:
                online_projection_count += 1
                continuity_windows.append(float(online_prior.get("continuity_window", 0.0) or 0.0))
                consistency_scores.append(float(online_prior.get("self_consistency_score", 0.0) or 0.0))
                volatility_signals.append(float(online_prior.get("volatility_signal", 0.0) or 0.0))
            conflict_summary = self._conflict_arbitration_summary(trace)
            raw_conflict = dict(trace.get("conflict_arbitration", {}) or {})
            latest_passes = list(raw_conflict.get("passes", []) or [])
            latest_resolution = dict(latest_passes[-1].get("resolution", {})) if latest_passes else {}
            blocked_actions = {
                str(action)
                for action in list(conflict_summary.get("hard_masked_targets", []) or [])
                if str(action)
            }
            blocked_actions.update(
                str(action)
                for action in list(raw_conflict.get("blocked_actions", []) or [])
                if str(action)
            )
            blocked_actions.update(
                str(action)
                for action in list(raw_conflict.get("circuit_breaker", {}).get("blocked_actions", []) or [])
                if str(action)
            )
            blocked_actions.update(
                str(action)
                for action in list(latest_resolution.get("blocked_actions", []) or [])
                if str(action)
            )
            action_layer = self._action_layer_from_trace(trace)
            conflict_rows = [
                row
                for row in list(action_layer.get("contribution_audit", []) or [])
                if isinstance(row, dict) and row.get("module_name") == "ConflictMonitorAgent"
            ]
            for row in conflict_rows:
                blocked_actions.update(
                    str(action)
                    for action in list(row.get("hard_masked_targets", []) or [])
                    if str(action)
                )
                blocked_actions.update(
                    str(action)
                    for action, value in dict(row.get("delta_projected", {}) or {}).items()
                    if str(action) and float(value) < 0.0
                )
            if blocked_actions:
                blocked_action_rounds += 1
                self._candidate_distribution_from_trace(trace)
                winner_target = str(action_layer.get("winner_target") or trace.get("sampled_action") or "")
                sampled_action = str(trace.get("sampled_action") or "")
                if all(action != winner_target and action != sampled_action for action in blocked_actions):
                    conflict_subordinated_rounds += 1
            authenticity = dict(trace.get("authenticity", {}) or {})
            candidate_penalties = {
                str(action): float(value)
                for action, value in dict(authenticity.get("candidate_penalties", {}) or {}).items()
                if float(value) > 0.0
            }
            if candidate_penalties or float(authenticity.get("sampling_penalty_applied", 0.0) or 0.0) > 0.0:
                auth_penalty_rounds += 1
                sampled_action = str(trace.get("sampled_action") or "")
                if sampled_action not in candidate_penalties:
                    authenticity_subordinated_rounds += 1
            if len(samples) < 8:
                samples.append(
                    {
                        "round_id": trace.get("round_id"),
                        "self_consistency_score": online_prior.get("self_consistency_score"),
                        "volatility_signal": online_prior.get("volatility_signal"),
                        "continuity_window": online_prior.get("continuity_window"),
                        "blocked_actions": sorted(blocked_actions),
                        "candidate_penalties": dict(candidate_penalties),
                    }
                )

        def _avg(values: list[float]) -> float:
            return round(sum(values) / len(values), 4) if values else 0.0

        return {
            "observed_round_rate": round(observed / max(len(rounds), 1), 4),
            "online_projection_coverage": round(online_projection_count / max(observed, 1), 4),
            "blocked_action_round_count": blocked_action_rounds,
            "conflict_subordination_rate": round(conflict_subordinated_rounds / max(blocked_action_rounds, 1), 4),
            "auth_penalty_round_count": auth_penalty_rounds,
            "authenticity_subordination_rate": round(authenticity_subordinated_rounds / max(auth_penalty_rounds, 1), 4),
            "self_consistency_score_avg": _avg(consistency_scores),
            "volatility_signal_avg": _avg(volatility_signals),
            "continuity_window_avg": _avg(continuity_windows),
            "samples": samples,
        }
