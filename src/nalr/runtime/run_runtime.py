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
from nalr.output.renderer import deterministic_short_chat_reply, fallback_render_text
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

FAULT_GUARD_SKILLS: tuple[str, ...] = (
    "heartbeat_check",
    "replace_failed_agent_with_baseline",
    "rollback_invalid_sigma",
    "switch_to_light_cache_mode",
)

ARCHITECTURE_CORE_GOAL = "维持生命性、真实性与连续性"
RUNTIME_SCHEMA_VERSION = 2
HEARTBEAT_NOOP_SIDE_CHANNEL_THROTTLE_SECONDS = 2.0
STALE_DIRTY_WORKTREE_RUN_SECONDS = 300.0
COMMAND_SNAPSHOT_RETENTION_LIMIT = 32

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nalr.runtime.controller import RuntimeController


class _ControllerBackedRuntime:
    def __init__(self, controller: "RuntimeController") -> None:
        self.controller = controller

    def __getattr__(self, name: str):
        return getattr(self.controller, name)

    def _controller_callable_override(self, name: str):
        patched = getattr(self.controller, "__dict__", {}).get(name)
        return patched if callable(patched) else None

    def _controller_utc_now_iso(self) -> str:
        controller_module = inspect.getmodule(self.controller.__class__)
        now_fn = getattr(controller_module, "utc_now_iso", utc_now_iso) if controller_module is not None else utc_now_iso
        return str(now_fn())


class RunRuntimeService(_ControllerBackedRuntime):
    def _run_policy(self, *, allow_commit: bool, operator_level: str) -> RunPolicy:
        return RunPolicy(
            allow_commit=allow_commit,
            operator_level=operator_level,
            dirty_worktree_policy="pause",
            pause_on_commit_boundary=True,
            continue_on_recoverable_failure=True,
            max_retries_per_step=2,
        )

    def _run_budget(self) -> ExecutionBudget:
        return ExecutionBudget(max_steps=12, max_retries_per_step=2)

    def _dirty_worktree_snapshot(self) -> dict[str, Any]:
        patched = getattr(self.controller, "__dict__", {}).get("_dirty_worktree_snapshot")
        if callable(patched):
            return patched()
        if not (self.project_root / ".git").exists():
            return {"detected": False, "entries": [], "reason": "not_git_repo"}
        try:
            result = subprocess.run(
                ["git", "status", "--short"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError:
            return {"detected": False, "entries": [], "reason": "git_unavailable"}
        if result.returncode != 0:
            return {"detected": False, "entries": [], "reason": "git_status_failed"}
        entries = [line for line in result.stdout.splitlines() if line.strip()]
        return {"detected": bool(entries), "entries": entries[:20], "reason": "ok"}

    def _infer_operator_level(self, domain: str) -> str:
        if domain in {"body", "mood", "nudge", "mode", "run", "identity"}:
            return "soft_intervene"
        if domain in {"agent", "debug", "suppress"}:
            return "debug_control"
        if domain in {"safe", "checkpoint", "budget", "snapshot"}:
            return "ops_admin"
        return "read_only"

    def _default_mutation_scope(self, domain: str, target: str | None = None) -> str:
        if domain in {"safe", "mode"}:
            return "runtime"
        if domain == "budget":
            return "resource"
        if domain == "nudge":
            return "relation" if target == "relation" else "focus"
        if domain == "snapshot":
            return "snapshot"
        if target:
            return target
        return domain

    def _legacy_command_envelope(self, command: str) -> CommandEnvelope:
        parts = command.split()
        domain = parts[0] if parts else "runtime"
        verb = parts[1] if len(parts) > 1 else "show"
        target = parts[2] if len(parts) > 2 else None
        mutation_target = verb if domain == "nudge" else target
        return CommandEnvelope(
            command_id=uuid4().hex,
            domain=domain,
            verb=verb,
            target=target,
            canonical=command,
            mutation_scope=self._default_mutation_scope(domain, mutation_target),
            operator_level=self._infer_operator_level(domain),
            rollback_available=True,
        )

    def _command_snapshot_path(self, snapshot_id: str) -> Path:
        return self.snapshot_dir / snapshot_id

    def _create_command_snapshot(self, state: RuntimeState, envelope: CommandEnvelope) -> str:
        self.flush_pending_io(raise_on_error=True)
        snapshot_id = f"snap-{uuid4().hex[:12]}"
        snapshot_path = self._command_snapshot_path(snapshot_id)
        snapshot_path.mkdir(parents=True, exist_ok=True)
        rewrite_snapshot(
            snapshot_path / "state.parquet",
            [{"payload_json": json.dumps(to_dict(state), ensure_ascii=False, sort_keys=True)}],
            schema={"payload_json": "VARCHAR"},
        )
        memory_snapshot = snapshot_path / "memory"
        if self.memory_store.memory_dir.exists():
            shutil.copytree(self.memory_store.memory_dir, memory_snapshot, dirs_exist_ok=True)
        rewrite_snapshot(
            snapshot_path / "snapshot_meta.parquet",
            [{
                "payload_json": json.dumps(
                    {
                        "snapshot_id": snapshot_id,
                        "command_id": envelope.command_id,
                        "canonical": envelope.canonical,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            }],
            schema={"payload_json": "VARCHAR"},
        )
        self._trim_command_snapshots(
            keep_latest=COMMAND_SNAPSHOT_RETENTION_LIMIT,
            preserve_snapshot_id=snapshot_id,
        )
        return snapshot_id

    def _trim_command_snapshots(self, *, keep_latest: int, preserve_snapshot_id: str) -> None:
        try:
            snapshot_dirs: list[tuple[int, str, Path]] = []
            for entry in self.snapshot_dir.iterdir():
                if not entry.is_dir():
                    continue
                if not entry.name.startswith("snap-"):
                    continue
                try:
                    mtime_ns = entry.stat().st_mtime_ns
                except OSError:
                    continue
                snapshot_dirs.append((mtime_ns, entry.name, entry))
            snapshot_dirs.sort(key=lambda item: (item[0], item[1]), reverse=True)

            to_delete: list[Path] = []
            for _, name, path in snapshot_dirs[keep_latest:]:
                if name == preserve_snapshot_id:
                    continue
                to_delete.append(path)

            for path in to_delete:
                try:
                    shutil.rmtree(path)
                except OSError:
                    continue
        except OSError:
            return

    def _apply_snapshot_restore(self, snapshot_id: str, envelope: CommandEnvelope) -> CommandResult:
        snapshot_path = self._command_snapshot_path(snapshot_id)
        operator_level = envelope.operator_level if envelope is not None else "ops_admin"
        if not snapshot_path.exists():
            return self._mark_boundary_result(
                CommandResult(
                    applied=False,
                    scope="snapshot",
                    delta={"snapshot_id": snapshot_id, "restored": False},
                    risk_note="snapshot not found",
                    operator_level=operator_level,
                    mutation_scope="snapshot",
                    command_id=envelope.command_id,
                    canonical=envelope.canonical,
                    parsed_args=envelope.parsed_args,
                    flags=envelope.flags,
                    rollback_available=False,
                ),
                boundary_action="allow_internal",
            )

        before_state = self.load_runtime_state()
        before_hash = self._state_hash(before_state)
        rollback_snapshot_id = self._create_command_snapshot(before_state, envelope)
        self.flush_pending_io(raise_on_error=True)
        shutil.copyfile(snapshot_path / "state.parquet", self.state_parquet_path)
        memory_snapshot = snapshot_path / "memory"
        if self.memory_store.memory_dir.exists():
            shutil.rmtree(self.memory_store.memory_dir)
        if memory_snapshot.exists():
            shutil.copytree(memory_snapshot, self.memory_store.memory_dir, dirs_exist_ok=True)
        self.memory_store = MemoryStore(self.home_path)
        self.controller.memory_store = self.memory_store
        self.dream_orchestrator.memory_store = self.memory_store
        rows = read_snapshot_rows(self.state_parquet_path, "select payload_json from read_parquet(?)")
        restored_state = RuntimeState(**json.loads(rows[0]["payload_json"]))
        restored_state, violation_code = self._preserve_subject_core_on_restore(
            before_state,
            restored_state,
            violation_code="subject_core_violation",
        )
        self._save_state(restored_state, sync=True)
        restored_state = self.load_runtime_state()
        after_hash = self._state_hash(restored_state)
        rollback = {
            "strategy": "snapshot_restore",
            "command": f"snapshot restore {rollback_snapshot_id}",
            "snapshot_id": rollback_snapshot_id,
            "restores_scope": "runtime+memory",
            "human_hint": f"alive snapshot restore {rollback_snapshot_id}",
        }
        result = CommandResult(
            applied=True,
            scope="snapshot",
            delta={"snapshot_id": snapshot_id, "restored": True},
            rollback_hint=rollback["human_hint"],
            operator_level=operator_level,
            rollback_available=True,
            mutation_scope="snapshot",
            snapshot_id=rollback_snapshot_id,
            command_id=envelope.command_id,
            canonical=envelope.canonical,
            parsed_args=envelope.parsed_args,
            flags=envelope.flags,
            rollback=rollback,
        )
        result = self._mark_boundary_result(
            result,
            boundary_action="allow_internal",
            violation_code=violation_code,
        )
        self._enrich_command_result(result, envelope)
        self.trace_store.append_command(
            envelope.canonical,
            result,
            before_hash,
            after_hash,
            session_id=restored_state.session_id,
            recorded_at=utc_now_iso(),
            sync=True,
        )
        return result

    def _enrich_command_result(self, result: CommandResult, envelope: CommandEnvelope) -> CommandResult:
        result.command_id = envelope.command_id
        result.canonical = envelope.canonical
        result.parsed_args = dict(envelope.parsed_args)
        result.flags = dict(envelope.flags)
        if result.mutation_scope is None:
            result.mutation_scope = envelope.mutation_scope
        state = self.load_runtime_state()
        core = self._ensure_subject_core(state)
        result.subject_id = core.subject_id
        result.continuity_nonce = core.continuity_nonce
        return result

    def _build_rollback(
        self,
        envelope: CommandEnvelope,
        before_state: RuntimeState,
        result: CommandResult,
        snapshot_id: str,
    ) -> dict[str, Any]:
        key = (envelope.domain, envelope.verb)
        command: str | None = None
        strategy = "domain_inverse"
        hint: str | None = None

        if key == ("safe", "on"):
            command = "safe off"
        elif key == ("safe", "off"):
            command = "safe on"
        elif key == ("mode", "set"):
            command = "safe on" if before_state.mode == "safe" or before_state.safe_mode else f"mode set {before_state.mode}"
        elif key == ("agent", "disable"):
            command = f"agent enable {envelope.target}"
        elif key == ("agent", "enable"):
            command = f"agent disable {envelope.target}"
        elif key == ("debug", "weight"):
            agent_name = str(envelope.parsed_args.get("agent_name", envelope.target))
            previous = float(before_state.agent_weight_overrides.get(agent_name, 1.0))
            command = f"debug weight {agent_name} {previous:g}"
        elif key == ("nudge", "focus"):
            delta = float(envelope.parsed_args.get("delta", 0.0))
            command = f"nudge focus {-delta:+.2f}"
        elif key == ("nudge", "relation"):
            delta = float(envelope.parsed_args.get("delta", 0.0))
            target = str(envelope.parsed_args.get("relation_target", envelope.target))
            command = f"nudge relation {target} trust {-delta:+.2f}"
        elif key == ("budget", "set"):
            previous_cap = before_state.resource_state.get("budget_cap")
            if previous_cap is None:
                previous_cap = int(round(float(before_state.budget_remaining) * 100000))
            command = f"budget set --cap {int(previous_cap)}"
        elif key == ("suppress", "dmn"):
            command = "agent enable DMNAgent"
        elif key == ("identity", "set-name"):
            previous_name = before_state.identity_state.display_name or before_state.identity_state.internal_handle
            command = f"identity set-name {previous_name}"
        else:
            strategy = "snapshot_restore"
            command = f"snapshot restore {snapshot_id}"
            hint = f"alive snapshot restore {snapshot_id}"

        return {
            "strategy": strategy,
            "command": command,
            "snapshot_id": snapshot_id,
            "restores_scope": result.mutation_scope or result.scope,
            "human_hint": hint or f"alive {command}",
        }

    def _mark_boundary_result(
        self,
        result: CommandResult,
        *,
        boundary_action: str,
        cause_type: str = "external_stimulus",
        violation_code: str = "",
        deprecation_warning: str = "",
    ) -> CommandResult:
        result.boundary_action = boundary_action
        result.cause_type = cause_type
        result.violation_code = violation_code
        result.deprecation_warning = deprecation_warning
        return result

    def _boundary_deprecation(self, command: str) -> str:
        return f"{command} is now boundary-mediated; the legacy direct-write path has been deprecated."

    def _reject_boundary_command(
        self,
        state: RuntimeState,
        *,
        scope: str,
        operator_level: str,
        violation_code: str,
        risk_note: str,
    ) -> CommandResult:
        state.safe_mode = True
        state.mode = "safe"
        return self._mark_boundary_result(
            CommandResult(
                applied=False,
                scope=scope,
                delta={},
                risk_note=risk_note,
                operator_level=operator_level,
                rollback_available=False,
            ),
            boundary_action="reject",
            violation_code=violation_code,
        )

    def _sync_run_state_to_runtime(self, state: RuntimeState, run_state: RunState) -> None:
        state.active_run_id = run_state.run_id
        state.run_status = run_state.status
        state.run_mode = "autonomous"
        state.current_goal = run_state.goal
        state.current_step_id = run_state.current_step_id
        state.pending_steps = to_dict(run_state.pending_steps)
        state.completed_steps = to_dict(run_state.completed_steps)
        state.last_tool_result = to_dict(run_state.last_tool_result) if run_state.last_tool_result else {}
        state.stop_reason = to_dict(run_state.stop_reason) if run_state.stop_reason else {}
        state.dirty_worktree_detected = run_state.dirty_worktree_detected
        state.commit_permission_required = run_state.commit_permission_required

    def _current_task_node(self, run_state: RunState | dict[str, Any]) -> TaskNode | None:
        normalized = run_state if isinstance(run_state, RunState) else RunState(**run_state)
        for step in normalized.pending_steps:
            if step.node_id == normalized.current_step_id:
                return step
        return normalized.pending_steps[0] if normalized.pending_steps else None

    def _task_turn_message(self, run_payload: dict[str, Any]) -> str:
        if run_payload.get("status") == "paused" and run_payload.get("dirty_worktree_detected"):
            return "任务已建立，但当前处于暂停状态。可用 /status /why 查看原因。"
        if not run_payload.get("last_tool_result"):
            return "任务已建立，等待工具审批。批准后会继续执行。"
        return "已进入只读任务处理。可用 /status /why /steps /tools 查看进度。"

    def _run_trace_ref(self, run_id: str) -> str:
        return f"run://{run_id}"

    def _run_step_trace_ref(self, run_id: str, step_id: str) -> str:
        return f"{self._run_trace_ref(run_id)}/steps/{step_id}"

    def _run_tool_trace_ref(self, run_id: str, call_id: str) -> str:
        return f"{self._run_trace_ref(run_id)}/tools/{call_id}"

    def _run_status_payload(self, run_state: RunState) -> dict[str, Any]:
        state = self.load_runtime_state()
        occupancy = self._active_run_occupancy(state)
        current_step = self._current_task_node(run_state)
        return {
            "runtime_revision": int(state.runtime_revision or 0),
            "last_mutation_at": str(state.last_mutation_at or ""),
            "run_id": run_state.run_id,
            "status": run_state.status,
            "goal": run_state.goal,
            "goal_summary": run_state.goal_summary,
            "current_step_id": run_state.current_step_id,
            "current_step": to_dict(current_step) if current_step else None,
            "dirty_worktree_detected": run_state.dirty_worktree_detected,
            "commit_permission_required": run_state.commit_permission_required,
            "stop_reason": to_dict(run_state.stop_reason) if run_state.stop_reason else {},
            "pending_steps": len(run_state.pending_steps),
            "completed_steps": len(run_state.completed_steps),
            "last_tool_result": to_dict(run_state.last_tool_result) if run_state.last_tool_result else {},
            "created_at": run_state.created_at,
            "updated_at": run_state.updated_at,
            "run_blocking": bool(occupancy.get("run_blocking")) and str(occupancy.get("run_id") or "") == str(run_state.run_id or ""),
            "run_block_reason": occupancy.get("run_block_reason") if str(occupancy.get("run_id") or "") == str(run_state.run_id or "") else "",
            "round_id": None,
            "trace_ref": self._run_trace_ref(run_state.run_id),
        }

    def _run_explain_payload(self, run_state: RunState) -> dict[str, Any]:
        current_step = self._current_task_node(run_state)
        return {
            "run_id": run_state.run_id,
            "status": run_state.status,
            "goal": run_state.goal,
            "goal_summary": run_state.goal_summary,
            "current_step": to_dict(current_step) if current_step else None,
            "last_tool_result": to_dict(run_state.last_tool_result) if run_state.last_tool_result else {},
            "policy": to_dict(run_state.policy),
            "budget": to_dict(run_state.budget),
            "dirty_worktree_detected": run_state.dirty_worktree_detected,
            "stop_reason": to_dict(run_state.stop_reason) if run_state.stop_reason else {},
            "round_id": None,
            "trace_ref": self._run_trace_ref(run_state.run_id),
        }

    def _run_stop_reason_code(self, run_state: RunState | dict[str, Any] | None) -> str:
        if run_state is None:
            return ""
        if isinstance(run_state, RunState):
            stop_reason = run_state.stop_reason
            return str(getattr(stop_reason, "code", "") or "")
        return str(dict(run_state.get("stop_reason", {}) or {}).get("code") or "")

    def _run_has_pending_approval(self, run_id: str) -> bool:
        if not str(run_id or "").strip():
            return False
        try:
            tool_rows = list(self.trace_store.list_run_tools(run_id) or [])
        except Exception:
            return False
        return any(str(row.get("status") or "") == "awaiting_approval" for row in tool_rows)

    def _active_run_occupancy(self, state: RuntimeState) -> dict[str, Any]:
        active_run_id = str(state.active_run_id or "").strip()
        default_payload = {
            "run_visible": False,
            "run_id": "",
            "run_status": "idle",
            "run_blocking": False,
            "run_block_reason": "",
            "run_block_run_id": "",
            "run_pending_approval": False,
            "run_dirty_worktree": False,
            "run_stop_reason": "",
            "run_stale_dirty_worktree": False,
        }
        if not active_run_id or state.run_status not in {"running", "paused"}:
            return default_payload
        try:
            run_state = self._load_run_state(active_run_id)
        except FileNotFoundError:
            return default_payload
        stale_info = self._stale_dirty_worktree_run_info(run_state)
        pending_approval = self._run_has_pending_approval(active_run_id)
        run_status = str(run_state.status or state.run_status or "idle")
        stop_reason = self._run_stop_reason_code(run_state)
        dirty_worktree = bool(getattr(run_state, "dirty_worktree_detected", False))
        block_reason = ""
        blocking = False
        if pending_approval:
            blocking = True
            block_reason = "awaiting_approval"
        elif run_status == "running":
            blocking = True
            block_reason = "running"
        elif stale_info is not None:
            block_reason = "stale_dirty_worktree"
        elif stop_reason:
            block_reason = stop_reason
        elif run_status == "paused":
            block_reason = "paused"
        return {
            "run_visible": True,
            "run_id": active_run_id,
            "run_status": run_status,
            "run_blocking": blocking,
            "run_block_reason": block_reason,
            "run_block_run_id": active_run_id if blocking else "",
            "run_pending_approval": pending_approval,
            "run_dirty_worktree": dirty_worktree,
            "run_stop_reason": stop_reason,
            "run_stale_dirty_worktree": stale_info is not None,
        }

    def runtime_status_truth_payload(self, state: RuntimeState | None = None) -> dict[str, Any]:
        current = state or self.load_runtime_state()
        occupancy = self._active_run_occupancy(current)
        return {
            "runtime_revision": int(current.runtime_revision or 0),
            "last_mutation_at": str(current.last_mutation_at or ""),
            **occupancy,
            "fault_guard": self.fault_guard_status(current),
        }

    def fault_guard_status(self, state: RuntimeState | None = None) -> dict[str, Any]:
        current = state or self.load_runtime_state()
        registered = {name: name in self.skills for name in FAULT_GUARD_SKILLS}
        return {
            "contract_status": "contract_only",
            "heartbeat_check": {
                "registered": registered["heartbeat_check"],
                "status": "implemented" if registered["heartbeat_check"] else "missing",
                "reads_runtime_state": True,
                "writes_state": False,
                "process_replace": False,
            },
            "replace_failed_agent_with_baseline": {
                "registered": registered["replace_failed_agent_with_baseline"],
                "status": "contract_only" if registered["replace_failed_agent_with_baseline"] else "missing",
                "process_replace": False,
                "fallback_contract": "baseline_or_neutral_delta",
                "baseline_source": "agent baseline bias / neutral delta",
                "checkpoint_backed": False,
            },
            "rollback_invalid_sigma": {
                "registered": registered["rollback_invalid_sigma"],
                "status": "checkpoint_backed_contract" if registered["rollback_invalid_sigma"] else "missing",
                "checkpoint_backed": True,
                "uses_checkpoint_create": "checkpoint_create" in self.skills,
                "uses_checkpoint_rewind": "rewind_state" in self.skills,
                "process_replace": False,
            },
            "switch_to_light_cache_mode": {
                "registered": registered["switch_to_light_cache_mode"],
                "status": "contract_only" if registered["switch_to_light_cache_mode"] else "missing",
                "process_replace": False,
                "cache_mode": "light_cache",
            },
            "checkpoint_contract": {
                "create": "checkpoint_create" in self.skills,
                "rewind": "rewind_state" in self.skills,
                "rollback_binding": "checkpoint-backed",
            },
            "state": {
                "mode": current.mode,
                "safe_mode": current.safe_mode,
                "last_checkpoint_id": current.last_checkpoint_id,
            },
        }

    def _stale_dirty_worktree_run_info(self, run_state: RunState | dict[str, Any] | None) -> dict[str, Any] | None:
        if run_state is None:
            return None
        normalized = run_state if isinstance(run_state, RunState) else RunState(**to_dict(run_state))
        if str(normalized.status or "") != "paused":
            return None
        stop_reason = getattr(normalized.stop_reason, "code", "")
        if not stop_reason and isinstance(normalized.stop_reason, dict):
            stop_reason = str(normalized.stop_reason.get("code") or "")
        if stop_reason != "dirty_worktree":
            return None
        updated_at = str(normalized.updated_at or normalized.created_at or "").strip()
        if not updated_at:
            return None
        anchor = self._autonomy_window_anchor(updated_at)
        if anchor is None:
            return None
        age_seconds = max(0.0, (datetime.now(timezone.utc) - anchor).total_seconds())
        if age_seconds < STALE_DIRTY_WORKTREE_RUN_SECONDS:
            return None
        return {"run_id": normalized.run_id, "age_seconds": round(age_seconds, 3)}

    def _active_run_blocks_background_progress(self, state: RuntimeState) -> bool:
        return bool(self._active_run_occupancy(state).get("run_blocking"))

    def _clear_stale_dirty_worktree_run_blocker(self, state: RuntimeState) -> bool:
        active_run_id = str(state.active_run_id or "").strip()
        if not active_run_id or state.run_status not in {"running", "paused"}:
            return False
        try:
            run_state = self._load_run_state(active_run_id)
        except FileNotFoundError:
            return False
        stale_info = self._stale_dirty_worktree_run_info(run_state)
        if stale_info is None:
            return False
        self.interrupt_run(
            active_run_id,
            reason="stale_dirty_worktree_blocker",
            message="stale dirty-worktree blocker cleared for autonomy recovery",
        )
        refreshed = self.load_runtime_state()
        if str(refreshed.active_run_id or "").strip() == active_run_id:
            refreshed.active_run_id = None
            refreshed.run_status = "idle"
            refreshed.run_mode = None
            refreshed.current_goal = None
            refreshed.current_step_id = None
            refreshed.pending_steps = []
            refreshed.completed_steps = []
            refreshed.last_tool_result = {}
            refreshed.stop_reason = {}
            refreshed.dirty_worktree_detected = False
            self._append_autonomy_recent_action(
                refreshed,
                action_type="run_cleanup",
                summary=f"cleared stale paused run blocker after {int(stale_info['age_seconds'])}s",
            )
            self._save_state(refreshed, sync=True)
        return True

    def _resolve_run_id(self, run_id: str | None = None) -> str:
        state = self.load_runtime_state()
        effective = run_id or state.active_run_id
        if not effective:
            raise FileNotFoundError("no active run recorded")
        return effective

    def _load_run_state(self, run_id: str | None = None) -> RunState:
        return RunState(**self.trace_store.read_run(self._resolve_run_id(run_id)))

    def _persist_run_state(self, run_state: RunState) -> None:
        state = self.load_runtime_state()
        self._sync_run_state_to_runtime(state, run_state)
        self._save_state(state, sync=True)
        self.trace_store.write_run(
            to_dict(run_state),
            session_id=state.session_id,
            recorded_at=self._controller_utc_now_iso(),
            sync=True,
        )

    def _build_task_bootstrap(
        self,
        goal: str,
        *,
        allow_commit: bool,
        operator_level: str,
        defer_bootstrap_tool: bool = False,
    ) -> tuple[RunState, dict[str, Any], dict[str, Any]]:
        state = self.load_runtime_state()
        request = RunRequest(
            goal=goal.strip(),
            scenario="task",
            mode=state.mode or "interactive",
            allow_commit=allow_commit,
            operator_level=operator_level,
        )
        supervisor = SupervisorLoop(project_root=self.project_root, planner=self._plan_run_via_model)
        if defer_bootstrap_tool:
            return supervisor.prepare_bootstrap(
                request,
                session_id=state.session_id,
                recorded_at=self._controller_utc_now_iso(),
            )
        return supervisor.bootstrap(
            request,
            session_id=state.session_id,
            recorded_at=self._controller_utc_now_iso(),
        )

    def prepare_task_bootstrap(
        self,
        goal: str,
        *,
        allow_commit: bool,
        operator_level: str,
        defer_bootstrap_tool: bool = False,
    ) -> tuple[RunState, dict[str, Any], dict[str, Any]]:
        return self._build_task_bootstrap(
            goal,
            allow_commit=allow_commit,
            operator_level=operator_level,
            defer_bootstrap_tool=defer_bootstrap_tool,
        )

    def _plan_run_via_model(self, goal: str, repo_metadata: dict[str, Any]) -> dict[str, Any]:
        planner_cfg = self.config["models"]["model_routes"].get("planner", {})
        route_name = "planner" if planner_cfg.get("enabled", False) else "pfc"
        response = self._call_bound_model_route(
            "planner",
            route_name=route_name,
            request=ModelRequest(
                system_prompt="你是 planner。仅回 JSON:{goal_summary,next_step,detail,expected_observation,success_criteria,tool_choice,confidence}。",
                user_prompt=self._json_prompt(
                    {
                        "goal": goal,
                        "repo": {
                            "cwd": repo_metadata.get("cwd"),
                            "dirty_worktree": bool(repo_metadata.get("dirty_worktree_detected", False)),
                            "tracked_files": int(repo_metadata.get("tracked_file_count", 0) or 0),
                        },
                    }
                ),
                response_schema={
                    "goal_summary": "str",
                    "next_step": "str",
                    "detail": "str",
                    "expected_observation": "str",
                    "success_criteria": "str",
                    "tool_choice": "str",
                    "confidence": "float",
                },
                metadata={"goal": goal, "stage": "run_planner"},
            ),
        )
        payload = response.payload
        required = {"goal_summary", "next_step", "expected_observation", "success_criteria", "tool_choice"}
        if not isinstance(payload, dict) or not required.issubset(payload):
            raise ValueError("planner response missing required fields")
        return payload

    def set_dream_enabled(self, enabled: bool) -> dict[str, Any]:
        state = self.load_runtime_state()
        state.session_metadata["dream_enabled_override"] = bool(enabled)
        self._save_state(state)
        return self.dream_status()

    def run_dream(self, *, mode: str = "sleep", cue: str | None = None) -> dict[str, Any]:
        state = self.load_runtime_state()
        relation_state = {"closeness": self.memory_store.closeness("user"), "boundary_level": 0.0}
        outcome = self.dream_orchestrator.run(
            state=state,
            mode=mode,
            cue=cue,
            relation_state=relation_state,
        )
        self._save_state(state)
        if not outcome.get("run_id"):
            raise RuntimeError("dream run did not produce a sidecar result")
        return self.dream_trace(outcome["run_id"])

    def execute_command(self, envelope: CommandEnvelope) -> CommandResult:
        if envelope.domain == "snapshot" and envelope.verb == "restore" and envelope.target:
            return self._apply_snapshot_restore(envelope.target, envelope)

        state = self.load_runtime_state()
        self._ensure_subject_core(state)
        before_state = RuntimeState(**to_dict(state))
        before_hash = self._state_hash(before_state)
        parts = envelope.canonical.split()
        snapshot_id = self._create_command_snapshot(before_state, envelope)
        operator_level = envelope.operator_level
        result = CommandResult(
            applied=False,
            scope=envelope.mutation_scope or envelope.domain,
            delta={},
            operator_level=operator_level,
            rollback_available=True,
        )

        if envelope.domain == "safe" and envelope.verb == "on":
            state.safe_mode = True
            state.mode = "safe"
            result = self._mark_boundary_result(
                CommandResult(applied=True, scope="runtime", delta={"safe_mode": True, "mode": "safe"}, risk_note="reduces spontaneity", operator_level=operator_level, rollback_available=True),
                boundary_action="allow_internal",
            )
        elif envelope.domain == "safe" and envelope.verb == "off":
            state.safe_mode = False
            state.mode = "interactive"
            result = self._mark_boundary_result(
                CommandResult(applied=True, scope="runtime", delta={"safe_mode": False, "mode": "interactive"}, risk_note="restores full runtime variability", operator_level=operator_level, rollback_available=True),
                boundary_action="allow_internal",
            )
        elif envelope.domain == "mode" and envelope.verb == "set":
            mode_name = str(envelope.parsed_args.get("mode", envelope.target or (parts[2] if len(parts) > 2 else "interactive")))
            state.mode = mode_name
            if mode_name != "safe":
                state.safe_mode = False
            result = self._mark_boundary_result(
                CommandResult(applied=True, scope="runtime", delta={"mode": mode_name}, operator_level=operator_level, rollback_available=True),
                boundary_action="allow_internal",
            )
        elif envelope.domain == "agent" and envelope.verb == "disable":
            agent_name = str(envelope.parsed_args.get("agent_name", envelope.target))
            state.agents_enabled[agent_name] = False
            result = self._mark_boundary_result(
                CommandResult(applied=True, scope=agent_name, delta={"enabled": False}, ttl="until re-enabled", operator_level=operator_level, rollback_available=True),
                boundary_action="allow_internal",
            )
        elif envelope.domain == "agent" and envelope.verb == "enable":
            agent_name = str(envelope.parsed_args.get("agent_name", envelope.target))
            state.agents_enabled[agent_name] = True
            result = self._mark_boundary_result(
                CommandResult(applied=True, scope=agent_name, delta={"enabled": True}, ttl="until changed", operator_level=operator_level, rollback_available=True),
                boundary_action="allow_internal",
            )
        elif envelope.domain == "body" and envelope.verb == "rest":
            state.body_energy = _clip(state.body_energy + 0.15)
            embodied_fatigue = _clip((1.0 - state.body_energy) * 0.82 + float(state.affect_residue or 0.0) * 0.14)
            state.fatigue = _clip(float(state.fatigue or 0.0) * 0.82 + embodied_fatigue * 0.18 - 0.05)
            result = self._mark_boundary_result(
                CommandResult(
                    applied=True,
                    scope="body",
                    delta={"body_energy": state.body_energy, "fatigue": round(float(state.fatigue), 4)},
                    ttl="one round",
                    operator_level=operator_level,
                    rollback_available=True,
                ),
                boundary_action="downgrade_to_stimulus",
                deprecation_warning=self._boundary_deprecation(envelope.canonical),
            )
        elif envelope.domain == "mood" and envelope.verb == "calm":
            state.mood = _clip((state.mood + 0.65) / 2)
            result = self._mark_boundary_result(
                CommandResult(applied=True, scope="mood", delta={"mood": state.mood}, ttl="one round", operator_level=operator_level, rollback_available=True),
                boundary_action="downgrade_to_stimulus",
                deprecation_warning=self._boundary_deprecation(envelope.canonical),
            )
        elif envelope.domain == "identity" and envelope.verb == "set-name":
            result = self._reject_boundary_command(
                state,
                scope="identity",
                operator_level=operator_level,
                violation_code="identity_seed_locked",
                risk_note="identity name can only be set during bootstrap seed or controlled migration",
            )
        elif envelope.domain == "debug" and envelope.verb == "weight":
            agent_name = str(envelope.parsed_args.get("agent_name", parts[2]))
            weight = float(envelope.parsed_args.get("weight", parts[3]))
            state.agent_weight_overrides[agent_name] = weight
            result = self._mark_boundary_result(
                CommandResult(applied=True, scope=agent_name, delta={"weight": weight}, ttl="until changed", operator_level=operator_level, rollback_available=True),
                boundary_action="allow_internal",
            )
        elif envelope.domain == "nudge" and envelope.verb == "focus":
            delta = float(envelope.parsed_args.get("delta", envelope.target or parts[2]))
            state.focus_nudge = _clip(state.focus_nudge + delta, -0.5, 0.5)
            result = self._mark_boundary_result(
                CommandResult(applied=True, scope="focus", delta={"focus_nudge": state.focus_nudge}, ttl="until changed", operator_level=operator_level, rollback_available=True),
                boundary_action="downgrade_to_stimulus",
                deprecation_warning=self._boundary_deprecation(envelope.canonical),
            )
        elif envelope.domain == "nudge" and envelope.verb == "relation":
            target = str(envelope.parsed_args.get("relation_target", parts[2]))
            delta = float(envelope.parsed_args.get("delta", parts[4]))
            relation = self.memory_store.nudge_relation(target, delta)
            result = self._mark_boundary_result(
                CommandResult(applied=True, scope="relation", delta={"target": target, "closeness": relation["closeness"], "delta": delta}, ttl="until changed", operator_level=operator_level, rollback_available=True),
                boundary_action="proposal_route",
                deprecation_warning=self._boundary_deprecation(envelope.canonical),
            )
        elif envelope.domain == "habit" and envelope.verb == "reset":
            pattern = str(envelope.parsed_args.get("pattern", envelope.target or parts[2]))
            habit = self.memory_store.reset_habit(pattern)
            result = self._mark_boundary_result(
                CommandResult(applied=True, scope="habit", delta={"pattern": pattern, "strength": habit["strength"], "recoverable": habit.get("recoverable", True)}, ttl="until rebuilt by recurrence", operator_level=operator_level, rollback_available=True),
                boundary_action="proposal_route",
                deprecation_warning=self._boundary_deprecation(envelope.canonical),
            )
        elif envelope.domain == "budget" and envelope.verb == "set":
            raw_value = envelope.parsed_args.get("value")
            if raw_value is None:
                raw_value = parts[3] if len(parts) >= 4 and parts[2] == "--cap" else parts[2]
            cap_value, budget_remaining = self._parse_budget_cap(str(raw_value))
            state.budget_remaining = budget_remaining
            state.resource_state = {**state.resource_state, "budget_cap": cap_value}
            result = self._mark_boundary_result(
                CommandResult(applied=True, scope="resource", delta={"budget_remaining": budget_remaining, "budget_cap": cap_value}, ttl="until changed", operator_level=operator_level, rollback_available=True),
                boundary_action="downgrade_to_stimulus",
                deprecation_warning=self._boundary_deprecation(envelope.canonical),
            )
        elif envelope.domain == "suppress" and envelope.verb == "dmn":
            state.agents_enabled["DMNAgent"] = False
            ttl = str(envelope.parsed_args.get("ttl", parts[2] if len(parts) >= 3 else "temporary"))
            result = self._mark_boundary_result(
                CommandResult(applied=True, scope="DMNAgent", delta={"enabled": False}, ttl=ttl, operator_level=operator_level, rollback_available=True),
                boundary_action="allow_internal",
            )
        elif envelope.domain == "checkpoint" and envelope.verb == "create":
            self.flush_pending_io(raise_on_error=True)
            checkpoint_id = f"ckpt-{state.round_count:04d}"
            path = self.checkpoint_dir / f"{checkpoint_id}.parquet"
            shutil.copyfile(self.state_parquet_path, path)
            state.last_checkpoint_id = checkpoint_id
            result = self._mark_boundary_result(
                CommandResult(applied=True, scope="checkpoint", delta={"checkpoint_id": checkpoint_id, "created": True}, operator_level=operator_level, rollback_available=True),
                boundary_action="allow_internal",
            )
        elif envelope.domain == "checkpoint" and envelope.verb == "rewind":
            checkpoint_id = str(envelope.parsed_args.get("checkpoint_id", envelope.target or parts[2]))
            checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.parquet"
            if not checkpoint_path.exists():
                result = self._mark_boundary_result(
                    CommandResult(applied=False, scope="checkpoint", delta={"checkpoint_id": checkpoint_id, "restored": False}, risk_note="checkpoint not found", operator_level=operator_level, rollback_available=False),
                    boundary_action="allow_internal",
                )
            else:
                self.flush_pending_io(raise_on_error=True)
                shutil.copyfile(checkpoint_path, self.state_parquet_path)
                rows = read_snapshot_rows(self.state_parquet_path, "select payload_json from read_parquet(?)")
                restored_state = RuntimeState(**json.loads(rows[0]["payload_json"]))
                restored_state, violation_code = self._preserve_subject_core_on_restore(
                    before_state,
                    restored_state,
                    violation_code="subject_core_violation",
                )
                self._save_state(restored_state, sync=True)
                state = self.load_runtime_state()
                result = self._mark_boundary_result(
                    CommandResult(applied=True, scope="checkpoint", delta={"checkpoint_id": checkpoint_id, "restored": True, "safe_mode": state.safe_mode, "mode": state.mode}, operator_level=operator_level, rollback_available=True),
                    boundary_action="allow_internal",
                    violation_code=violation_code,
                )
        else:
            raise ValueError(f"unsupported mutable command: {envelope.canonical}")

        self._enrich_command_result(result, envelope)
        result.snapshot_id = snapshot_id
        result.rollback = self._build_rollback(envelope, before_state, result, snapshot_id)
        result.rollback_hint = result.rollback["human_hint"]
        sync_disk = envelope.domain == "checkpoint"
        self._save_state(state, sync=sync_disk)
        after_hash = self._state_hash(state)
        self.trace_store.append_command(
            envelope.canonical,
            result,
            before_hash,
            after_hash,
            session_id=state.session_id,
            recorded_at=self._controller_utc_now_iso(),
            sync=sync_disk,
        )
        return result
    def prepare_monologue_stream_advance(
        self,
        *,
        state: RuntimeState | None = None,
        now_iso: str | None = None,
    ) -> dict[str, Any]:
        current = state or self.load_runtime_state()
        bucket = self._monologue_state_bucket(current)
        prepared = self.monologue_runtime.prepare_catch_up(bucket, now_iso=now_iso or self._controller_utc_now_iso())
        return {
            "runtime_revision": int(current.runtime_revision or 0),
            "prepared": prepared,
            "due": bool(list(prepared.get("due_pulses", []) or [])),
        }
    def execute_monologue_stream_advance(self, ticket: dict[str, Any]) -> list[dict[str, Any]]:
        prepared = dict(ticket.get("prepared", {}) or {})
        fragment_builder = getattr(self.controller, "_generate_monologue_fragments_via_model")
        return self.monologue_runtime.execute_catch_up(
            prepared,
            fragment_builder=fragment_builder,
        )

    def _monologue_state_bucket(self, state: RuntimeState) -> dict[str, Any]:
        bucket = state.session_metadata.setdefault("monologue_stream", {})
        if not isinstance(bucket, dict):
            bucket = {}
            state.session_metadata["monologue_stream"] = bucket
        normalized = self.monologue_runtime.ensure_bucket(bucket)
        state.session_metadata["monologue_stream"] = normalized
        return normalized

    def commit_monologue_stream_advance(self, ticket: dict[str, Any], generated: list[dict[str, Any]]) -> dict[str, Any]:
        state = self.load_runtime_state()
        expected_revision = int(ticket.get("runtime_revision", -1) or -1)
        if expected_revision >= 0 and int(state.runtime_revision or 0) != expected_revision:
            return {
                "committed": False,
                "reason": "runtime_revision_changed",
                "runtime_revision": int(state.runtime_revision or 0),
                "bucket": self._monologue_state_bucket(state),
                "generated": [],
            }
        prepared = dict(ticket.get("prepared", {}) or {})
        committed_bucket, committed_rows = self.monologue_runtime.commit_catch_up(prepared, list(generated or []))
        state.session_metadata["monologue_stream"] = committed_bucket
        self._save_state(state, sync=True)
        return {
            "committed": True,
            "reason": "",
            "runtime_revision": int(state.runtime_revision or 0),
            "bucket": committed_bucket,
            "generated": committed_rows,
        }

    def advance_monologue_stream_background(self, *, now_iso: str | None = None) -> dict[str, Any]:
        ticket = self.prepare_monologue_stream_advance(now_iso=now_iso)
        if not bool(ticket.get("due")):
            state = self.load_runtime_state()
            return {
                "committed": False,
                "reason": "not_due",
                "runtime_revision": int(state.runtime_revision or 0),
                "bucket": self._monologue_state_bucket(state),
                "generated": [],
            }
        generated = self.execute_monologue_stream_advance(ticket)
        return self.commit_monologue_stream_advance(ticket, generated)

    def _sync_monologue_stream(
        self,
        state: RuntimeState,
        *,
        mark_viewed: bool = False,
        generate_if_due: bool = True,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        bucket = self._monologue_state_bucket(state)
        original_bucket = dict(bucket)
        now_iso = self._controller_utc_now_iso()
        updated_bucket = self.monologue_runtime.ensure_bucket(bucket, now_iso=now_iso)
        generated: list[dict[str, Any]] = []
        if generate_if_due:
            updated_bucket, generated = self.monologue_runtime.catch_up(
                updated_bucket,
                now_iso=now_iso,
                fragment_builder=getattr(self.controller, "_generate_monologue_fragments_via_model"),
            )
        if mark_viewed:
            viewed_at = now_iso
            if str(updated_bucket.get("last_viewed_at") or "") != viewed_at:
                updated_bucket["last_viewed_at"] = viewed_at
        state.session_metadata["monologue_stream"] = updated_bucket
        if updated_bucket != original_bucket:
            self._save_state(state, sync=True)
        return updated_bucket, generated

    def _refresh_heartbeat_side_channels(self, state: RuntimeState) -> dict[str, Any]:
        bucket, _generated = self._sync_monologue_stream(state, generate_if_due=False)
        original_initiative_bucket = dict(state.session_metadata.get("initiative", {}) or {})
        latest_view = self.trace_store.recent_round_signal_views(limit=1)
        latest_recorded_at = latest_view[-1].get("recorded_at") if latest_view else None
        initiative_bucket = self._initiative_state_bucket(state)
        state.session_metadata["initiative"] = initiative_bucket
        if initiative_bucket != original_initiative_bucket:
            self._save_state(state, sync=True)
        evaluation = dict(initiative_bucket.get("last_evaluation", {}) or {})
        return {
            "monologue_generated_total": int(bucket.get("generated_total", 0) or 0),
            "initiative_ready": bool(evaluation.get("should_send")),
            "initiative_last_evaluated_at": str(initiative_bucket.get("last_evaluated_at") or ""),
            "initiative_latest_recorded_at": latest_recorded_at,
        }

    def _heartbeat_side_channel_refresh_due(self, state: RuntimeState, *, noop_heartbeat: bool) -> bool:
        if not noop_heartbeat:
            return True
        self._sync_autonomy_state(state)
        if str(state.autonomy_loop.last_action_type or "") != "nothing":
            return True
        initiative_bucket = self._initiative_state_bucket(state)
        if str(initiative_bucket.get("last_evaluation_source") or "") != "heartbeat":
            return True
        last_evaluated_at = self._autonomy_window_anchor(str(initiative_bucket.get("last_evaluated_at") or ""))
        now_at = self._autonomy_window_anchor(self._controller_utc_now_iso())
        if last_evaluated_at is None or now_at is None:
            return True
        return (now_at - last_evaluated_at).total_seconds() >= HEARTBEAT_NOOP_SIDE_CHANNEL_THROTTLE_SECONDS

    def _collect_monologue_stream_fragments(
        self,
        state: RuntimeState,
        *,
        allow_model_generation: bool = True,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
        del allow_model_generation
        bucket = self._monologue_state_bucket(state)
        recent_fragments = list(self.monologue_runtime.read_fragments(limit=3))
        return bucket, recent_fragments, []

    def _monologue_stream_trace_payload(
        self,
        *,
        bucket: dict[str, Any],
        recent_fragments: list[dict[str, Any]],
        generated: list[dict[str, Any]],
    ) -> dict[str, Any]:
        pressure = self._monologue_stream_pressure_payload(
            recent_fragments=recent_fragments,
            generated=generated,
        )
        return {
            "active": bool(recent_fragments),
            "hidden_by_default": bool(dict(bucket.get("settings", {}) or {}).get("hidden_by_default", True)),
            "fresh_generated": bool(generated),
            "generated_total": int(bucket.get("generated_total", 0) or 0),
            "last_generated_at": str(bucket.get("last_generated_at") or ""),
            "recent_fragment_count": len(recent_fragments),
            "pressure_score": pressure["score"],
            "blank_ratio": pressure["blank_ratio"],
            "sample_fragments": [
                {
                    "fragment_id": str(row.get("fragment_id") or ""),
                    "recorded_at": str(row.get("recorded_at") or ""),
                    "category": str(row.get("category") or ""),
                    "content": str(row.get("content") or ""),
                    "source": str(row.get("source") or ""),
                }
                for row in list(recent_fragments or [])[:2]
            ],
        }

    def _monologue_stream_pressure_payload(
        self,
        *,
        recent_fragments: list[dict[str, Any]],
        generated: list[dict[str, Any]],
    ) -> dict[str, float]:
        fragment_count = len(list(recent_fragments or []))
        if fragment_count <= 0:
            return {
                "score": 0.0,
                "fragment_count": 0.0,
                "blank_ratio": 0.0,
                "fresh_generated": 0.0,
            }
        blank_count = sum(
            1
            for row in list(recent_fragments or [])
            if str(row.get("category") or "") == "blank_fragment"
            or not str(row.get("content") or "").strip().strip(".。!！?？… ")
        )
        blank_ratio = blank_count / max(fragment_count, 1)
        freshness_bonus = 0.08 if generated else 0.03
        score = _clip(0.18 + fragment_count * 0.14 + blank_ratio * 0.08 + freshness_bonus, 0.0, 1.0)
        return {
            "score": round(float(score), 4),
            "fragment_count": round(float(fragment_count), 4),
            "blank_ratio": round(float(blank_ratio), 6),
            "fresh_generated": round(float(1.0 if generated else 0.0), 6),
        }

    def _monologue_stream_pressure_score(self, state: RuntimeState) -> float:
        _bucket, recent_fragments, generated = self._collect_monologue_stream_fragments(
            state,
            allow_model_generation=False,
        )
        pressure = self._monologue_stream_pressure_payload(
            recent_fragments=recent_fragments,
            generated=generated,
        )
        return float(pressure["score"] or 0.0)

    def _build_monologue_stream_action_contribution(
        self,
        *,
        state: RuntimeState,
        scenario: str,
        endogenous_turn: bool,
    ) -> tuple[ProbabilisticContribution | None, dict[str, Any]]:
        if not endogenous_turn or scenario == "task":
            return None, {}
        bucket, recent_fragments, generated = self._collect_monologue_stream_fragments(
            state,
            allow_model_generation=False,
        )
        trace_payload = self._monologue_stream_trace_payload(
            bucket=bucket,
            recent_fragments=recent_fragments,
            generated=generated,
        )
        if not recent_fragments:
            return None, trace_payload
        pressure = self._monologue_stream_pressure_payload(
            recent_fragments=recent_fragments,
            generated=generated,
        )
        fragment_count = int(pressure["fragment_count"] or 0.0)
        blank_ratio = float(pressure["blank_ratio"] or 0.0)
        score = float(pressure["score"] or 0.0)
        confidence = round(max(0.34, min(0.88, 0.4 + score * 0.3)), 4)
        return ProbabilisticContribution(
            module_name="MonologueStream",
            module_type="monologue",
            level="action",
            target_space="action",
            raw_signal={
                "fragment_count": round(float(fragment_count), 6),
                "blank_ratio": round(float(blank_ratio), 6),
                "fresh_generated": round(float(1.0 if generated else 0.0), 6),
            },
            modulated_delta={
                "monologue": round(0.14 + score * 0.32, 6),
                "absorb": round(0.03 + score * 0.06, 6),
                "respond": round(-0.02 * score, 6),
                "nothing": round(-0.01 * score, 6),
            },
            confidence=confidence,
            confidence_calibrated=confidence,
            trace_reason="hidden monologue fragments re-enter the endogenous action field as low-weight internal speech pressure",
            projection_reason="continuous hidden monologue stream projects a monologue candidate back into the unified action field",
            applied_at_stage="monologue_stream",
            native_operator="hidden_monologue_feedback",
            dependency_trace=[
                f"fragments:{fragment_count}",
                f"blank_ratio:{round(blank_ratio, 4)}",
                f"fresh:{1 if generated else 0}",
            ],
            projection=EnergyProjectionSpec(module_type="monologue", target_space="action", module_temperature=0.78),
        ), trace_payload

    def _generate_monologue_fragments_via_model(
        self,
        *,
        seed: str,
        pulse_index: int,
        pulse_dt: str,
        fragment_count: int,
    ) -> list[dict[str, Any]]:
        request = ModelRequest(
            system_prompt=(
                "你是一个隐藏意识流生成器。只返回 JSON:{fragments:[{content,category,source}] }。"
                "生成彼此松散、碎片化、无明确任务指向的第一人称内在念头。"
                "不要对用户说话，不要解释，不要总结，不要输出第二人称开头。"
            ),
            user_prompt=self._json_prompt(
                {
                    "seed": seed,
                    "pulse_index": pulse_index,
                    "pulse_dt": pulse_dt,
                    "fragment_count": fragment_count,
                    "categories": [
                        "environment_notice",
                        "free_association",
                        "memory_fragment",
                        "daydream",
                        "blank_fragment",
                    ],
                }
            ),
            response_schema={"fragments": "list"},
            metadata={
                "seed": seed,
                "pulse_index": pulse_index,
                "fragment_count": fragment_count,
            },
        )
        try:
            response = self._call_bound_model_route(
                "MonologueStream",
                route_name="monologue_stream",
                request=request,
                skill_name="monologue_stream",
            )
        except Exception:
            return []
        payload = dict(getattr(response, "payload", {}) or {})
        rows: list[dict[str, Any]] = []
        for item in list(payload.get("fragments", []) or []):
            if isinstance(item, str):
                content = item.strip()
                if not content:
                    continue
                rows.append({"content": content, "category": "model_fragment", "source": "model"})
                continue
            if not isinstance(item, dict):
                continue
            content = str(item.get("content") or "").strip()
            if not content:
                continue
            rows.append(
                {
                    "content": content,
                    "category": str(item.get("category") or "model_fragment"),
                    "source": str(item.get("source") or "model"),
                }
            )
        return rows[: max(1, int(fragment_count or 1))]

    def _initiative_memory_backing(
        self,
        cue: str | None = None,
        *,
        current_goal: str | None = None,
        active_session: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if cue:
            recall = self.memory_recall(cue)
            if recall.get("found"):
                return {
                    "cue": recall.get("cue") or cue,
                    "strength": round(float(recall.get("strength", 0.0) or 0.0), 4),
                    "summary": recall.get("summary") or recall.get("cue") or cue,
                }
        for topic_cue, topic_source in self._initiative_topic_cues(current_goal=current_goal, active_session=active_session):
            recall = self.memory_recall(topic_cue)
            if recall.get("found"):
                return {
                    "cue": recall.get("cue") or topic_cue,
                    "strength": round(float(recall.get("strength", 0.0) or 0.0), 4),
                    "summary": recall.get("summary") or recall.get("cue") or topic_cue,
                    "topic_relevance": 1.0,
                    "topic_source": topic_source,
                }
        top_memories = self.memory_top(limit=24)
        if not top_memories:
            return {}
        unique_memories: list[dict[str, Any]] = []
        seen_cues: set[str] = set()
        for item in top_memories:
            memory = dict(item or {})
            cue_value = str(memory.get("cue") or "").strip()
            if not cue_value or cue_value in seen_cues:
                continue
            seen_cues.add(cue_value)
            unique_memories.append(memory)
        if not unique_memories:
            return {}

        def _memory_rank(item: dict[str, Any]) -> tuple[float, float, float, float, float, float]:
            cue_value = str(item.get("cue") or "").strip()
            context_slot = str(item.get("context_slot") or "").strip()
            strength = max(float(item.get("detail_strength", 0.0) or 0.0), float(item.get("gist_strength", 0.0) or 0.0))
            last_recalled_round = float(int(item.get("last_recalled_round", 0) or 0))
            recorded_rank = 0.0
            recorded_at = str(item.get("recorded_at") or "").strip()
            if recorded_at:
                try:
                    recorded_rank = datetime.fromisoformat(recorded_at.replace("Z", "+00:00")).timestamp()
                except ValueError:
                    recorded_rank = 0.0
            return (
                0.0 if cue_value.startswith("endogenous:") else 1.0,
                1.0 if context_slot.startswith("user::") else 0.0,
                last_recalled_round,
                recorded_rank,
                strength,
                float(item.get("count", 0.0) or 0.0),
            )

        preferred = max(unique_memories, key=_memory_rank)
        top = dict(preferred or {})
        return {
            "cue": top.get("cue"),
            "strength": round(
                max(float(top.get("detail_strength", 0.0) or 0.0), float(top.get("gist_strength", 0.0) or 0.0)),
                4,
            ),
            "summary": top.get("summary") or top.get("cue") or "",
            "topic_relevance": 0.0,
        }

    def _initiative_session_selection(self, *, now_iso: str | None = None) -> dict[str, Any]:
        resolved_now_iso = now_iso or self._controller_utc_now_iso()
        session, metadata = self.terminal_sessions.select_fresh_active_session(
            max_age_seconds=900,
            now_iso=resolved_now_iso,
        )
        selected_session = to_dict(session) if session is not None else None
        selected_session_id = metadata.get("selected_session_id")
        return {
            "active_session": selected_session,
            "selected_session_id": selected_session_id,
            "selected_session_age_seconds": metadata.get("selected_session_age_seconds"),
            "selected_session_reason": metadata.get("selected_session_reason") or "",
            "session_fresh": bool(metadata.get("session_fresh", False)),
            "active_session_available": bool(metadata.get("active_session_available", False)),
            "fresh_session_available": bool(metadata.get("session_fresh", False) and selected_session_id),
        }

    def _initiative_active_session(self) -> dict[str, Any] | None:
        return self._initiative_session_selection().get("active_session")

    def _initiative_has_pending_approval(self, active_session: dict[str, Any] | None) -> bool:
        if not isinstance(active_session, dict):
            return False
        approvals = [item for item in list(active_session.get("approvals_pending", []) or []) if isinstance(item, dict)]
        return any(str(item.get("status") or "pending") == "pending" for item in approvals)

    def _initiative_recent_history_stats(
        self,
        history: list[dict[str, Any]],
        *,
        now_iso: str,
        settings: dict[str, Any],
    ) -> dict[str, Any]:
        now_dt = datetime.fromisoformat(now_iso.replace("Z", "+00:00"))
        recent_hour = 0
        last_auto_sent_at = None
        for item in reversed(history):
            if not item.get("auto_sent"):
                continue
            recorded_at = str(item.get("recorded_at") or "")
            try:
                recorded_dt = datetime.fromisoformat(recorded_at.replace("Z", "+00:00"))
            except ValueError:
                continue
            if last_auto_sent_at is None:
                last_auto_sent_at = recorded_at
            if (now_dt - recorded_dt).total_seconds() <= 3600:
                recent_hour += 1
        cooldown_remaining = 0
        if last_auto_sent_at:
            last_auto_dt = datetime.fromisoformat(last_auto_sent_at.replace("Z", "+00:00"))
            cooldown_remaining = max(0, int(settings["cooldown_seconds"]) - int((now_dt - last_auto_dt).total_seconds()))
        return {
            "sent_last_hour": recent_hour,
            "hourly_limit": int(settings["hourly_limit"]),
            "cooldown_remaining_seconds": cooldown_remaining,
            "last_auto_sent_at": last_auto_sent_at,
        }

    def _initiative_topic_cues(
        self,
        *,
        current_goal: str | None = None,
        active_session: dict[str, Any] | None = None,
    ) -> list[tuple[str, str]]:
        cues: list[tuple[str, str]] = []
        seen: set[str] = set()

        def _append_topic_cue(text: Any, *, source: str) -> None:
            normalized = str(text or "").strip()
            if not normalized:
                return
            derived = str(_derive_cue(RoundEvent(source="user", content=normalized, target="user")) or "").strip().lower()
            for candidate in (derived, normalized.lower()):
                if candidate and candidate not in seen:
                    seen.add(candidate)
                    cues.append((candidate, source))

        _append_topic_cue(current_goal, source="current_goal")
        if isinstance(active_session, dict):
            transcript_lines = list(active_session.get("transcript_lines", []) or [])
            for row in reversed(transcript_lines):
                if not isinstance(row, dict):
                    continue
                if str(row.get("kind") or "").strip() != "user":
                    continue
                _append_topic_cue(row.get("text"), source="recent_user_turn")
                break
        return cues

    def _initiative_active_run_payload(self, state: RuntimeState) -> dict[str, Any] | None:
        occupancy = self._active_run_occupancy(state)
        if not bool(occupancy.get("run_visible")):
            return None
        return {
            "run_id": occupancy.get("run_id"),
            "status": occupancy.get("run_status"),
            "dirty_worktree_detected": bool(occupancy.get("run_dirty_worktree")),
            "awaiting_approval": bool(occupancy.get("run_pending_approval")),
            "approval_state": "awaiting_approval" if occupancy.get("run_pending_approval") else "",
            "stop_reason": {"code": occupancy.get("run_stop_reason")},
        }

    def _initiative_distribution_payload(
        self,
        state: RuntimeState,
        *,
        relation_state: dict[str, Any] | None = None,
        vitality_snapshot: dict[str, Any] | None = None,
        cue: str | None = None,
        memory_backing: dict[str, Any] | None = None,
        latest_recorded_at: str | None = None,
        current_goal: str | None = None,
    ) -> dict[str, Any]:
        ticket = self.prepare_initiative_background(state=state, latest_recorded_at=latest_recorded_at)
        prepared = dict(ticket.get("prepared", {}) or {})
        if vitality_snapshot:
            prepared["vitality_snapshot"] = dict(vitality_snapshot)
        if relation_state:
            prepared["relation_state"] = dict(relation_state)
        if memory_backing:
            prepared["memory_backing"] = dict(memory_backing)
            ticket["memory_backing"] = dict(memory_backing)
        elif cue and not dict(prepared.get("memory_backing", {}) or {}):
            prepared["memory_backing"] = self._initiative_memory_backing(
                cue,
                current_goal=current_goal if current_goal is not None else prepared.get("current_goal"),
                active_session=ticket.get("active_session"),
            )
            ticket["memory_backing"] = dict(prepared["memory_backing"])
        if current_goal is not None:
            prepared["current_goal"] = current_goal
            if not memory_backing and not cue:
                prepared["memory_backing"] = self._initiative_memory_backing(
                    None,
                    current_goal=current_goal,
                    active_session=ticket.get("active_session"),
                )
                ticket["memory_backing"] = dict(prepared["memory_backing"])
        ticket["prepared"] = prepared
        payload = self.evaluate_initiative_background(ticket)
        settings = dict(payload.get("settings", {}) or {})
        history_stats = dict(payload.get("history_stats", {}) or {})
        hourly_limit = int(settings["hourly_limit"])
        if hourly_limit > 0 and history_stats["sent_last_hour"] >= hourly_limit:
            payload["should_send"] = False
            payload["expression_mode"] = "silent"
            payload["suppression_reason"] = "hourly_limit_reached"
        elif history_stats["cooldown_remaining_seconds"] > 0:
            payload["should_send"] = False
            payload["expression_mode"] = "silent"
            payload["suppression_reason"] = "cooldown_active"
        payload["settings"] = settings
        payload["history_stats"] = history_stats
        payload["auto_send_enabled"] = bool(settings["auto_send_enabled"])
        payload["memory_backing"] = dict(ticket.get("memory_backing", {}) or payload.get("memory_backing", {}) or {})
        payload["run_occupancy"] = self._active_run_occupancy(state)
        payload["selected_session_id"] = ticket.get("selected_session_id")
        payload["selected_session_age_seconds"] = ticket.get("selected_session_age_seconds")
        payload["selected_session_reason"] = ticket.get("selected_session_reason")
        payload["session_fresh"] = bool(ticket.get("session_fresh", False))
        payload["active_session_available"] = bool(ticket.get("active_session_available", False))
        payload["fresh_session_available"] = bool(ticket.get("fresh_session_available", False))
        payload["monologue_score"] = round(float(ticket.get("monologue_score", payload.get("monologue_score", 0.0)) or 0.0), 4)
        if bool(payload.get("active_session_available")) and not bool(payload.get("session_fresh")) and not payload.get("active_session"):
            payload["should_send"] = False
            payload["expression_mode"] = "silent"
            payload["suppression_reason"] = "no_fresh_session"
            payload["target_session_id"] = None
        if not payload.get("should_send"):
            payload["expression_mode"] = "silent"
        return payload

    def _initiative_delivery_text(self, *, proposal: dict[str, Any], message: str) -> str:
        base = str(message or "").strip()
        memory_backing = dict(proposal.get("memory_backing", {}) or {})
        cue = str(memory_backing.get("cue") or "").strip()
        topic_relevance = float(memory_backing.get("topic_relevance", 0.0) or 0.0)
        if not cue or cue.startswith("endogenous:") or topic_relevance < 0.8:
            return base

        def _compact(value: str) -> str:
            return (
                value.strip()
                .replace(" ", "")
                .replace("\n", "")
                .replace("。", "")
                .replace("，", "")
                .replace("？", "")
                .replace("！", "")
                .lower()
            )

        if cue in base or _compact(cue) in _compact(base):
            return base

        anchor = f"我还挂着“{cue}”这条线。"
        return f"{anchor} {base}".strip() if base else anchor

    def _initiative_context_memory_backing(
        self,
        *,
        cue: Any,
        recall_strength: Any,
        fallback_summary: str,
    ) -> dict[str, Any]:
        cue_value = str(cue or "").strip()
        strength_value = round(float(recall_strength or 0.0), 4)
        if not cue_value and strength_value <= 0.0:
            return {}
        if cue_value.startswith("endogenous:"):
            return {}
        return {
            "cue": cue_value,
            "strength": strength_value,
            "summary": cue_value or fallback_summary[:80],
            "topic_relevance": 1.0,
            "topic_source": "context_cue",
        }

    def _record_initiative_feedback_delta(self, text: str) -> float:
        positive_tokens = ("好", "可以", "继续", "想", "愿意", "谢谢", "ok", "yes", "sure", "love")
        negative_tokens = ("不要", "别", "停", "烦", "no", "stop", "later")
        lowered = text.lower()
        if any(token in text for token in positive_tokens) or any(token in lowered for token in positive_tokens):
            return 0.05
        if any(token in text for token in negative_tokens) or any(token in lowered for token in negative_tokens):
            return -0.05
        return 0.01

    def record_initiative_feedback(self, text: str, *, session_id: str | None = None, target: str = "user") -> dict[str, Any]:
        state = self.load_runtime_state()
        bucket = self._initiative_state_bucket(state)
        history = list(bucket.get("history", []) or [])
        pending = next(
            (
                item
                for item in reversed(history)
                if item.get("auto_sent") and not item.get("feedback_recorded") and (session_id is None or item.get("session_id") == session_id)
            ),
            None,
        )
        if pending is None:
            return {"recorded": False}
        delta = self._record_initiative_feedback_delta(text)
        self.memory_store.nudge_relation(target, delta)
        cue = str(dict(pending.get("memory_backing", {}) or {}).get("cue") or "")
        if cue:
            self.memory_store.update_habit_strength(cue, delta, round_id=state.round_count)
        intent = str(pending.get("top_intent") or "initiative")
        state.desire_state.latent_drives[intent] = round(float(state.desire_state.latent_drives.get(intent, 0.0) or 0.0) + delta, 6)
        state.motivation_learning_state.endogenous_policy_shift[intent] = round(
            float(state.motivation_learning_state.endogenous_policy_shift.get(intent, 0.0) or 0.0) + delta,
            6,
        )
        pending["feedback_recorded"] = True
        pending["response_preview"] = text[:120]
        response = {
            "initiative_response_to": pending.get("proposal_id"),
            "recorded_at": self._controller_utc_now_iso(),
            "session_id": session_id,
            "relation_delta": round(delta, 4),
            "habit_cue": cue or None,
            "response_preview": text[:120],
        }
        feedback = dict(bucket.get("feedback", {}) or {})
        feedback["recent"] = [*list(feedback.get("recent", []) or []), response][-20:]
        bucket["feedback"] = feedback
        bucket["history"] = history
        self._save_state(state, sync=True)
        return {"recorded": True, **response}

    def _dispatch_initiative_to_session(self, proposal: dict[str, Any], text: str) -> tuple[bool, str | None]:
        session_id = str(proposal.get("target_session_id") or "").strip()
        if not session_id or not text.strip():
            return False, None
        try:
            session = self.terminal_sessions.read(session_id)
        except FileNotFoundError:
            return False, None
        memory_backing = dict(proposal.get("memory_backing", {}) or {})
        session.transcript_lines.append(
            {
                "kind": "assistant",
                "text": text.strip(),
                "recorded_at": self._controller_utc_now_iso(),
                "delivery_mode": "speech",
                "initiative_proposal_id": proposal.get("proposal_id"),
                "initiative_memory_cue": memory_backing.get("cue"),
                "initiative_topic_source": memory_backing.get("topic_source"),
                "initiative_topic_relevance": memory_backing.get("topic_relevance"),
            }
        )
        self.terminal_sessions.write(session)
        return True, session_id

    def _record_initiative_outcome(
        self,
        *,
        round_id: int,
        recorded_at: str,
        proposal: dict[str, Any],
        message: str,
        auto_sent: bool,
        session_id: str | None,
    ) -> dict[str, Any]:
        state = self.load_runtime_state()
        bucket = self._initiative_state_bucket(state)
        history = list(bucket.get("history", []) or [])
        entry = {
            **dict(proposal or {}),
            "round_id": round_id,
            "recorded_at": recorded_at,
            "message": message,
            "auto_sent": bool(auto_sent),
            "session_id": session_id,
            "feedback_recorded": False,
        }
        history.append(entry)
        bucket["history"] = history[-50:]
        bucket["last_proposal"] = entry
        if auto_sent:
            bucket["last_auto_send"] = entry
        else:
            bucket["last_suppressed"] = entry
        self._save_state(state, sync=True)
        return entry

    def _autonomy_policy_for_profile(self, profile: str = "tool_level") -> AutonomyPolicyState:
        normalized = str(profile or "tool_level").strip() or "tool_level"
        policy = AutonomyPolicyState(profile=normalized)
        if normalized == "observer_only":
            policy.allowed_operator_levels = ["read_only"]
            policy.allowed_commands = [
                "endogenous status",
                "dream status",
                "replay",
                "memory recall",
                "memory top",
                "trace why",
                "why not",
            ]
        elif normalized == "full_runtime":
            policy.allowed_operator_levels = ["read_only", "soft_intervene", "debug_control"]
        policy = self._apply_observer_autonomy_preferences(policy)
        return policy

    def _apply_observer_autonomy_preferences(self, policy: AutonomyPolicyState) -> AutonomyPolicyState:
        settings = self._observer_settings()
        autonomy = settings.get("autonomy", {}) if isinstance(settings.get("autonomy"), dict) else {}
        learning_mode = str(autonomy.get("learning_mode", policy.learning_mode) or policy.learning_mode).strip().lower()
        if learning_mode not in {"observe", "guided-learn", "active-learn"}:
            learning_mode = policy.learning_mode
        policy.learning_mode = learning_mode
        policy.network_enabled = bool(autonomy.get("network_enabled", policy.network_enabled))
        policy.external_io_enabled = bool(autonomy.get("external_io_enabled", policy.external_io_enabled))
        policy.allow_commit = bool(autonomy.get("allow_commit", policy.allow_commit))
        policy.allowed_network_domains = self._normalize_observer_string_list(
            list(autonomy.get("allowed_network_domains", policy.allowed_network_domains))
        )
        policy.writable_roots = self._normalize_observer_path_list(
            list(autonomy.get("writable_roots", policy.writable_roots))
        )
        policy.knowledge_roots = self._normalize_observer_path_list(
            list(autonomy.get("knowledge_roots", policy.knowledge_roots))
        )
        policy.learning_log_dir = self._normalize_observer_path(str(autonomy.get("learning_log_dir", policy.learning_log_dir) or ""))
        policy.trace_external_learning = bool(autonomy.get("trace_external_learning", policy.trace_external_learning))
        policy.allowed_operator_levels = self._normalize_observer_string_list(
            list(autonomy.get("allowed_operator_levels", policy.allowed_operator_levels))
        )
        allowed = self._normalize_observer_string_list(list(autonomy.get("allowed_commands", policy.allowed_commands)))
        blocked = self._normalize_observer_string_list(list(autonomy.get("blocked_commands", policy.blocked_commands)))
        policy.allowed_commands = allowed
        policy.blocked_commands = blocked
        try:
            policy.max_rounds_per_hour = max(0, int(autonomy.get("max_rounds_per_hour", policy.max_rounds_per_hour) or 0))
        except (TypeError, ValueError):
            policy.max_rounds_per_hour = max(0, int(policy.max_rounds_per_hour or 0))
        try:
            policy.max_tool_actions_per_hour = max(0, int(autonomy.get("max_tool_actions_per_hour", policy.max_tool_actions_per_hour) or 0))
        except (TypeError, ValueError):
            policy.max_tool_actions_per_hour = max(0, int(policy.max_tool_actions_per_hour or 0))
        try:
            policy.failure_trip_threshold = max(1, int(autonomy.get("failure_trip_threshold", policy.failure_trip_threshold) or 1))
        except (TypeError, ValueError):
            policy.failure_trip_threshold = max(1, int(policy.failure_trip_threshold or 1))
        policy.auto_safe_mode = bool(autonomy.get("auto_safe_mode", policy.auto_safe_mode))
        quiet_hours: list[int] = []
        for raw in list(autonomy.get("quiet_hours", policy.quiet_hours) or []):
            try:
                hour = int(raw)
            except (TypeError, ValueError):
                continue
            if 0 <= hour <= 23 and hour not in quiet_hours:
                quiet_hours.append(hour)
        policy.quiet_hours = quiet_hours
        return policy

    def _generate_autonomy_self_run_goal_via_model(self, state: RuntimeState) -> str | None:
        request = ModelRequest(
            system_prompt=(
                "你是自治运行体的只读自查意图生成器。"
                "只返回 JSON:{goal,reason}。"
                "goal 必须是一句简短英文任务，限定为 inspect/read/identify/summarize 这类只读动作。"
                "不要包含写入、修改、删除、提交、push。"
            ),
            user_prompt=self._json_prompt(
                {
                    "state": {
                        "round_count": int(state.round_count or 0),
                        "body_energy": self._compact_float(state.body_energy),
                        "fatigue": self._compact_float(state.fatigue),
                        "memory_fragments": self._compact_float(state.memory_fragments),
                        "self_continuity": self._compact_float(state.self_continuity),
                        "meaning_strength": self._compact_float(state.meaning_strength),
                        "mode": state.mode,
                    },
                    "constraints": {
                        "operator_level": "read_only",
                        "allow_commit": False,
                        "goal_style": "single_sentence",
                    },
                }
            ),
            response_schema={"goal": "string", "reason": "string"},
            metadata={
                "round_count": int(state.round_count or 0),
                "memory_fragments": float(state.memory_fragments or 0.0),
                "self_continuity": float(state.self_continuity or 0.0),
            },
        )
        try:
            response = self._call_bound_model_route(
                "AutonomySelfRun",
                route_name="autonomy_self_run",
                request=request,
                skill_name="autonomy_self_run",
            )
        except Exception:
            return None
        payload = dict(getattr(response, "payload", {}) or {})
        goal = str(payload.get("goal") or "").strip()
        if not goal:
            return None
        lowered = goal.lower()
        blocked = ("write", "modify", "delete", "remove", "commit", "push", "overwrite")
        if any(token in lowered for token in blocked):
            return None
        return goal

    def _sync_autonomy_state(self, state: RuntimeState) -> None:
        if isinstance(state.autonomy_policy, dict):
            state.autonomy_policy = AutonomyPolicyState(**state.autonomy_policy)
        if isinstance(state.autonomy_loop, dict):
            state.autonomy_loop = AutonomyLoopState(**state.autonomy_loop)
        if not state.autonomy_policy.profile:
            state.autonomy_policy.profile = "tool_level"
        if not state.autonomy_loop.profile:
            state.autonomy_loop.profile = state.autonomy_policy.profile

    def _ensure_default_autonomy_runtime(self, state: RuntimeState, *, force: bool = False) -> bool:
        self._sync_autonomy_state(state)
        if not force and bool(state.session_metadata.get("autonomy_user_disabled", False)):
            return False

        target_profile = str(state.autonomy_loop.profile or state.autonomy_policy.profile or "tool_level").strip() or "tool_level"
        default_policy = self._autonomy_policy_for_profile(target_profile)
        changed = False

        if state.autonomy_policy.profile != default_policy.profile:
            state.autonomy_policy.profile = default_policy.profile
            changed = True
        for command in default_policy.allowed_commands:
            if command not in state.autonomy_policy.allowed_commands:
                state.autonomy_policy.allowed_commands.append(command)
                changed = True
        for level in default_policy.allowed_operator_levels:
            if level not in state.autonomy_policy.allowed_operator_levels:
                state.autonomy_policy.allowed_operator_levels.append(level)
                changed = True
        if not state.autonomy_policy.enabled:
            state.autonomy_policy.enabled = True
            changed = True
        if not state.autonomy_loop.running:
            state.autonomy_loop.running = True
            changed = True
        if state.autonomy_loop.profile != default_policy.profile:
            state.autonomy_loop.profile = default_policy.profile
            changed = True
        if not state.autonomy_loop.window_started_at:
            state.autonomy_loop.window_started_at = datetime.now(timezone.utc).isoformat()
            changed = True
        if "autonomy_default_enabled_at" not in state.session_metadata:
            state.session_metadata["autonomy_default_enabled_at"] = utc_now_iso()
            changed = True
        return changed

    def _autonomy_matches_prefix(self, command: str, prefix: str) -> bool:
        normalized_command = str(command or "").strip().lower()
        normalized_prefix = str(prefix or "").strip().lower()
        return bool(normalized_prefix) and (
            normalized_command == normalized_prefix or normalized_command.startswith(f"{normalized_prefix} ")
        )

    def _autonomy_command_allowed(
        self,
        command: str,
        policy: AutonomyPolicyState | None = None,
    ) -> tuple[bool, str]:
        effective_policy = policy or self.load_runtime_state().autonomy_policy
        for blocked in effective_policy.blocked_commands:
            if self._autonomy_matches_prefix(command, blocked):
                return False, "blocked_by_policy"
        if effective_policy.allowed_commands and not any(
            self._autonomy_matches_prefix(command, allowed) for allowed in effective_policy.allowed_commands
        ):
            return False, "not_in_allowlist"
        envelope = self._legacy_command_envelope(command)
        if effective_policy.allow_commit is False and command.startswith("git commit"):
            return False, "commit_disallowed"
        if effective_policy.allowed_operator_levels and envelope.operator_level not in effective_policy.allowed_operator_levels:
            return False, "operator_level_disallowed"
        return True, ""

    def _autonomy_operator_allowed(
        self,
        operator_level: str,
        policy: AutonomyPolicyState | None = None,
    ) -> tuple[bool, str]:
        effective_policy = policy or self.load_runtime_state().autonomy_policy
        if effective_policy.allowed_operator_levels and operator_level not in effective_policy.allowed_operator_levels:
            return False, "operator_level_disallowed"
        return True, ""

    def _autonomy_budget_usage(self, state: RuntimeState) -> dict[str, Any]:
        self._sync_autonomy_state(state)
        return {
            "remaining": round(float(state.budget_remaining or 0.0), 4),
            "heartbeat_count": int(state.autonomy_loop.heartbeat_count or 0),
            "tool_actions": int(state.autonomy_loop.window_tool_actions or 0),
            "endogenous_rounds": int(state.autonomy_loop.window_endogenous_rounds or 0),
            "total_tool_actions": int(state.autonomy_loop.total_tool_actions or 0),
            "total_endogenous_rounds": int(state.autonomy_loop.total_endogenous_rounds or 0),
            "max_rounds_per_hour": int(state.autonomy_policy.max_rounds_per_hour or 0),
            "max_tool_actions_per_hour": int(state.autonomy_policy.max_tool_actions_per_hour or 0),
            "window_started_at": state.autonomy_loop.window_started_at,
        }

    def _append_autonomy_recent_action(
        self,
        state: RuntimeState,
        *,
        action_type: str,
        summary: str,
        round_id: int | None = None,
        trace_ref: str | None = None,
    ) -> None:
        self._sync_autonomy_state(state)
        recorded_at = utc_now_iso()
        if action_type != "failure":
            state.autonomy_loop.failure_count = 0
            state.autonomy_loop.last_error = ""
        state.autonomy_loop.last_step_at = recorded_at
        state.autonomy_loop.last_action_type = action_type
        state.autonomy_loop.last_action_summary = summary
        state.autonomy_loop.last_round_id = round_id
        state.autonomy_loop.last_trace_ref = trace_ref
        state.autonomy_loop.recent_actions.append(
            {
                "recorded_at": recorded_at,
                "action_type": action_type,
                "summary": summary,
                "round_id": round_id,
                "trace_ref": trace_ref,
            }
        )
        state.autonomy_loop.recent_actions = state.autonomy_loop.recent_actions[-12:]

    def _autonomy_window_anchor(self, value: str | None) -> datetime | None:
        if not value:
            return None
        normalized = str(value).replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def _ensure_autonomy_window(self, state: RuntimeState) -> None:
        self._sync_autonomy_state(state)
        loop = state.autonomy_loop
        now = datetime.now(timezone.utc)
        anchor = self._autonomy_window_anchor(loop.window_started_at)
        if anchor is None or (now - anchor).total_seconds() >= 3600:
            loop.window_started_at = now.isoformat()
            loop.window_tool_actions = 0
            loop.window_endogenous_rounds = 0

    def _autonomy_step_context(self, state: RuntimeState) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        context = {"cue": "", "recall_strength": 0.0}
        relation_state = {"relationship_risk": 0.0, "closeness": 0.5, "boundary_level": float(state.subjective_state.boundary or 0.5)}
        slow_variables = {
            "affect_residue": round(float(state.affect_residue or 0.0), 6),
            "memory_activation": round(float(state.memory_fragments or 0.0) * 0.45, 6),
            "resource_scarcity": round(1.0 - float(state.budget_remaining or 0.0), 6),
        }
        return context, relation_state, slow_variables

    def _autonomy_self_run_score(
        self,
        state: RuntimeState,
        policy: AutonomyPolicyState,
    ) -> float:
        if "self_run" in {str(item or "").strip() for item in list(policy.blocked_commands or [])}:
            return 0.0
        pending_progress = self._autonomy_pending_readonly_repo_scan(state, policy)
        if pending_progress is not None:
            return 0.94
        if self._active_run_blocks_background_progress(state):
            return 0.0
        allowed, _reason = self._autonomy_operator_allowed("read_only", policy)
        if not allowed:
            return 0.0
        if int(state.round_count or 0) <= 0:
            return 0.0
        if float(state.budget_remaining or 0.0) < 0.18:
            return 0.0
        if float(state.body_energy or 0.0) < 0.38 or float(state.fatigue or 0.0) > 0.62:
            return 0.0
        recent_actions = list(state.autonomy_loop.recent_actions or [])[-4:]
        if any(str(item.get("action_type") or "") == "self_run" for item in recent_actions):
            return 0.0
        continuity_gap = max(0.0, 0.58 - float(state.self_continuity or 0.0))
        meaning_gap = max(0.0, 0.44 - float(state.meaning_strength or 0.0))
        memory_pull = max(float(state.memory_fragments or 0.0), float(state.subjective_state.spontaneous or 0.0) * 0.3)
        if continuity_gap + meaning_gap + memory_pull < 0.28:
            return 0.0
        vitality_room = max(0.0, float(state.body_energy or 0.0) - 0.34)
        fatigue_headroom = max(0.0, 0.64 - float(state.fatigue or 0.0))
        return min(
            1.0,
            continuity_gap * 0.42
            + meaning_gap * 0.28
            + memory_pull * 0.22
            + vitality_room * 0.18
            + fatigue_headroom * 0.12
            + (0.14 if int(state.round_count or 0) > 0 else 0.0),
        )

    def _autonomy_record_failure(self, state: RuntimeState, exc: Exception) -> None:
        self._sync_autonomy_state(state)
        state.autonomy_loop.failure_count += 1
        state.autonomy_loop.last_error = str(exc)
        threshold = int(state.autonomy_policy.failure_trip_threshold or 1)
        if state.autonomy_loop.failure_count >= threshold:
            state.autonomy_loop.running = False
            state.autonomy_policy.enabled = False
            state.autonomy_loop.stop_reason = "failure_trip_threshold"
            if state.autonomy_policy.auto_safe_mode:
                state.safe_mode = True
                state.mode = "safe"

    def _autonomy_execute_command(self, command: str) -> dict[str, Any]:
        state = self.load_runtime_state()
        self._sync_autonomy_state(state)
        self._ensure_autonomy_window(state)
        allowed, reason = self._autonomy_command_allowed(command, state.autonomy_policy)
        if not allowed:
            self._append_autonomy_recent_action(
                state,
                action_type="blocked_command",
                summary=f"{command} blocked: {reason}",
            )
            self._save_state(state, sync=True)
            return {
                "allowed": False,
                "command": command,
                "reason": reason,
                "blocked_commands": list(state.autonomy_policy.blocked_commands),
            }

        apply_command = self._controller_callable_override("apply_command")
        result = apply_command(command) if apply_command is not None else self.apply_command(command)
        state = self.load_runtime_state()
        self._sync_autonomy_state(state)
        self._ensure_autonomy_window(state)
        state.autonomy_loop.total_tool_actions += 1
        state.autonomy_loop.window_tool_actions += 1
        self._append_autonomy_recent_action(
            state,
            action_type="command",
            summary=f"{command}: {'applied' if result.applied else 'rejected'}",
        )
        self._autonomy_trace_append(
            state,
            action_type="command",
            summary=command,
            delta={"command": command, "applied": result.applied, "scope": result.scope},
        )
        self._save_state(state, sync=True)
        return {
            "allowed": True,
            "command": command,
            "reason": "",
            "applied": result.applied,
            "scope": result.scope,
            "boundary_action": result.boundary_action,
        }

    def _autonomy_status_payload(self, *, include_diagnostics: bool) -> dict[str, Any]:
        state = self.load_runtime_state()
        self._sync_autonomy_state(state)
        self._ensure_autonomy_window(state)
        policy = state.autonomy_policy
        loop = state.autonomy_loop
        payload = {
            **self.runtime_status_truth_payload(state),
            "enabled": bool(policy.enabled),
            "running": bool(loop.running and policy.enabled),
            "profile": loop.profile or policy.profile,
            "last_step_at": loop.last_step_at,
            "last_action_type": loop.last_action_type,
            "last_action_summary": loop.last_action_summary,
            "failure_count": int(loop.failure_count or 0),
            "stop_reason": loop.stop_reason,
            "last_error": loop.last_error,
            "budget_usage": self._autonomy_budget_usage(state),
            "kill_switch_available": True,
            "allowed_commands": list(policy.allowed_commands),
            "blocked_commands": list(policy.blocked_commands),
            "recent_actions": list(loop.recent_actions),
        }
        if not include_diagnostics:
            return payload

        field_probe = self._autonomy_action_field_probe()
        candidate_scores = {
            name: round(float(score or 0.0), 6)
            for name, score in self._autonomy_projected_candidate_scores(state, policy, field_probe).items()
        }
        candidate_peak = max(candidate_scores, key=candidate_scores.get)
        return {
            **payload,
            "decision_surface": "autonomy_action_field_probe" if field_probe else "autonomy_candidate_competition",
            "field_probe_top_action": str(field_probe.get("top_action") or "") if isinstance(field_probe, dict) else "",
            "candidate_scores": candidate_scores,
            "candidate_peak": candidate_peak,
        }

    def autonomy_status(self) -> dict[str, Any]:
        return self._autonomy_status_payload(include_diagnostics=True)

    def autonomy_runtime_status(self) -> dict[str, Any]:
        return self._autonomy_status_payload(include_diagnostics=False)

    def start_autonomy(self, profile: str = "tool_level", *, clear_safe_mode: bool = False) -> dict[str, Any]:
        state = self.load_runtime_state()
        clear_requested = clear_safe_mode or bool(self._observer_settings().get("autonomy", {}).get("clear_safe_mode_on_start", True))
        if clear_requested:
            self._apply_startup_unlock_preferences(state)
        state.autonomy_policy = self._autonomy_policy_for_profile(profile)
        state.autonomy_policy.enabled = True
        state.autonomy_loop.running = True
        state.autonomy_loop.profile = state.autonomy_policy.profile
        state.autonomy_loop.failure_count = 0
        state.autonomy_loop.stop_reason = ""
        state.autonomy_loop.last_error = ""
        state.autonomy_loop.window_started_at = datetime.now(timezone.utc).isoformat()
        state.autonomy_loop.window_tool_actions = 0
        state.autonomy_loop.window_endogenous_rounds = 0
        state.session_metadata["autonomy_user_disabled"] = False
        self._append_autonomy_recent_action(
            state,
            action_type="start",
            summary=f"autonomy started in {state.autonomy_policy.profile}{' after leaving safe mode' if clear_requested else ''}",
        )
        self._save_state(state, sync=True)
        return self.autonomy_status()

    def stop_autonomy(self, reason: str = "manual_stop") -> dict[str, Any]:
        state = self.load_runtime_state()
        self._sync_autonomy_state(state)
        state.autonomy_policy.enabled = False
        state.autonomy_loop.running = False
        state.autonomy_loop.stop_reason = str(reason or "manual_stop")
        if state.autonomy_loop.stop_reason in {"manual_stop", "user_stop", "operator_stop"}:
            state.session_metadata["autonomy_user_disabled"] = True
        self._append_autonomy_recent_action(
            state,
            action_type="stop",
            summary=f"autonomy stopped: {state.autonomy_loop.stop_reason}",
        )
        self._save_state(state, sync=True)
        return self.autonomy_status()

    def prepare_initiative_background(
        self,
        *,
        state: RuntimeState | None = None,
        latest_recorded_at: str | None = None,
    ) -> dict[str, Any]:
        current = state or self.load_runtime_state()
        bucket = self._initiative_state_bucket(current)
        settings = self.initiative_runtime.merge_settings(bucket.get("settings"))
        now_iso = self._controller_utc_now_iso()
        self._apply_subjective_baseline_floor(current)
        history = list(bucket.get("history", []) or [])
        history_stats = self._initiative_recent_history_stats(history, now_iso=now_iso, settings=settings)
        session_selection = self._initiative_session_selection(now_iso=now_iso)
        active_session = session_selection.get("active_session")
        pending_approval = self._initiative_has_pending_approval(active_session)
        idle_seconds = self.initiative_runtime.idle_seconds(latest_recorded_at, now_iso=now_iso)
        vitality = {
            "body_energy": round(float(current.body_energy or 0.0), 4),
            "affect_residue": round(float(current.affect_residue or 0.0), 4),
            "mood": round(float(current.mood or 0.0), 4),
        }
        relation = dict(self.memory_store.relation_state("user"))
        habit_strength = max((float(item.get("strength", 0.0) or 0.0) for item in self.habit_top(limit=1)), default=0.0)
        memory_backing_builder = self._controller_callable_override("_initiative_memory_backing")
        if memory_backing_builder is not None:
            resolved_memory_backing = memory_backing_builder(
                None,
                current_goal=current.current_goal,
                active_session=active_session,
            )
        else:
            resolved_memory_backing = self._initiative_memory_backing(
                None,
                current_goal=current.current_goal,
                active_session=active_session,
            )
        prepared = self.initiative_runtime.prepare(
            settings=settings,
            idle_seconds=idle_seconds,
            vitality_snapshot=vitality,
            relation_state=relation,
            subjective_state=to_dict(current.subjective_state) if getattr(current, "subjective_state", None) else {},
            temperament_state=dict(getattr(current, "temperament_state", {}) or {}),
            habit_strength=habit_strength,
            memory_backing=resolved_memory_backing,
            active_session=active_session,
            active_run=self._initiative_active_run_payload(current),
            pending_approval=pending_approval,
            safe_mode=bool(current.safe_mode),
            recent_history=history,
            current_goal=current.current_goal,
        )
        pressure_scorer = self._controller_callable_override("_monologue_stream_pressure_score")
        monologue_score_value = pressure_scorer(current) if pressure_scorer is not None else self._monologue_stream_pressure_score(current)
        monologue_score = round(float(monologue_score_value or 0.0), 4)
        return {
            "runtime_revision": int(current.runtime_revision or 0),
            "prepared": prepared,
            "history_stats": history_stats,
            "settings": settings,
            "memory_backing": resolved_memory_backing,
            "active_session": active_session,
            "monologue_score": monologue_score,
            **session_selection,
        }
    def evaluate_initiative_background(self, ticket: dict[str, Any]) -> dict[str, Any]:
        prepared = dict(ticket.get("prepared", {}) or {})
        evaluation = self.initiative_runtime.evaluate(prepared)
        evaluation["settings"] = dict(ticket.get("settings", {}) or evaluation.get("settings", {}) or {})
        evaluation["history_stats"] = dict(ticket.get("history_stats", {}) or {})
        evaluation["auto_send_enabled"] = bool(dict(ticket.get("settings", {}) or {}).get("auto_send_enabled", True))
        evaluation["memory_backing"] = dict(ticket.get("memory_backing", {}) or {})
        evaluation["active_session"] = ticket.get("active_session")
        evaluation["selected_session_id"] = ticket.get("selected_session_id")
        evaluation["selected_session_age_seconds"] = ticket.get("selected_session_age_seconds")
        evaluation["selected_session_reason"] = ticket.get("selected_session_reason")
        evaluation["session_fresh"] = bool(ticket.get("session_fresh", False))
        evaluation["active_session_available"] = bool(ticket.get("active_session_available", False))
        evaluation["fresh_session_available"] = bool(ticket.get("fresh_session_available", False))
        evaluation["monologue_score"] = round(float(ticket.get("monologue_score", 0.0) or 0.0), 4)
        evaluation["run_occupancy"] = dict(prepared.get("active_run") or {})
        return evaluation

    def execute_initiative_background(self, ticket: dict[str, Any]) -> dict[str, Any]:
        return self.evaluate_initiative_background(ticket)

    def _initiative_state_bucket(self, state: RuntimeState) -> dict[str, Any]:
        bucket = state.session_metadata.setdefault("initiative", {})
        if not isinstance(bucket, dict):
            bucket = {}
            state.session_metadata["initiative"] = bucket
        bucket["settings"] = self.initiative_runtime.merge_settings(bucket.get("settings"))
        bucket["history"] = [
            item
            for item in list(bucket.get("history", []) or [])
            if isinstance(item, dict)
        ][-50:]
        feedback = dict(bucket.get("feedback", {}) or {})
        feedback["recent"] = [
            item
            for item in list(feedback.get("recent", []) or [])
            if isinstance(item, dict)
        ][-20:]
        bucket["feedback"] = feedback
        if not isinstance(bucket.get("last_evaluation"), dict):
            bucket["last_evaluation"] = {}
        bucket["last_evaluated_at"] = str(bucket.get("last_evaluated_at") or "")
        bucket["last_evaluation_source"] = str(bucket.get("last_evaluation_source") or "")
        return bucket

    def commit_initiative_background(self, ticket: dict[str, Any], evaluation: dict[str, Any]) -> dict[str, Any]:
        state = self.load_runtime_state()
        expected_revision = int(ticket.get("runtime_revision", -1) or -1)
        if expected_revision >= 0 and int(state.runtime_revision or 0) != expected_revision:
            bucket = self._initiative_state_bucket(state)
            return {
                "committed": False,
                "reason": "runtime_revision_changed",
                "evaluation": dict(bucket.get("last_evaluation", {}) or {}),
            }
        committed = self.initiative_runtime.commit(dict(ticket.get("prepared", {}) or {}), dict(evaluation or {}))
        bucket = self._initiative_state_bucket(state)
        bucket["last_evaluation"] = committed
        bucket["last_evaluated_at"] = self._controller_utc_now_iso()
        bucket["last_evaluation_source"] = "background"
        state.session_metadata["initiative"] = bucket
        self._save_state(state, sync=True)
        return {
            "committed": True,
            "reason": "",
            "evaluation": committed,
        }

    def advance_initiative_background(self, *, latest_recorded_at: str | None = None) -> dict[str, Any]:
        ticket = self.prepare_initiative_background(latest_recorded_at=latest_recorded_at)
        evaluation = self.evaluate_initiative_background(ticket)
        return self.commit_initiative_background(ticket, evaluation)

    def _initiative_status_payload(self, state: RuntimeState) -> dict[str, Any]:
        bucket = self._initiative_state_bucket(state)
        distribution = dict(bucket.get("last_evaluation", {}) or {})
        if not distribution:
            latest_view = self.trace_store.recent_round_signal_views(limit=1)
            distribution_builder = self._controller_callable_override("_initiative_distribution_payload")
            if distribution_builder is not None:
                distribution = distribution_builder(
                    state,
                    latest_recorded_at=latest_view[-1].get("recorded_at") if latest_view else None,
                )
            else:
                distribution = self._initiative_distribution_payload(
                    state,
                    latest_recorded_at=latest_view[-1].get("recorded_at") if latest_view else None,
                )
        feedback = dict(bucket.get("feedback", {}) or {})
        recent_feedback = list(feedback.get("recent", []) or [])
        proposal = {
            "proposal_id": distribution.get("proposal_id"),
            "proposal_type": distribution.get("proposal_type"),
            "expression_mode": distribution.get("expression_mode"),
            "top_intent": distribution.get("top_intent"),
            "posterior": distribution.get("posterior", {}),
            "readiness": distribution.get("readiness", 0.0),
            "intrinsic_value": distribution.get("intrinsic_value", 0.0),
            "speech_cost": distribution.get("speech_cost", 0.0),
            "suppression_reason": distribution.get("suppression_reason", ""),
            "should_send": distribution.get("should_send", False),
            "target_session_id": distribution.get("target_session_id"),
        }
        return {
            **self.runtime_status_truth_payload(state),
            "settings": distribution["settings"],
            "ready": bool(distribution.get("should_send")),
            "last_evaluated_at": bucket.get("last_evaluated_at"),
            "last_evaluation_source": bucket.get("last_evaluation_source"),
            "last_evaluation": dict(bucket.get("last_evaluation", {}) or {}),
            "proposal": proposal,
            "top_intent": proposal["top_intent"],
            "suppression_reason": proposal["suppression_reason"],
            "should_send": proposal["should_send"],
            "speech_cost": proposal["speech_cost"],
            "monologue_score": distribution.get("monologue_score", 0.0),
            "selected_session_id": distribution.get("selected_session_id"),
            "selected_session_age_seconds": distribution.get("selected_session_age_seconds"),
            "selected_session_reason": distribution.get("selected_session_reason"),
            "session_fresh": bool(distribution.get("session_fresh", False)),
            "memory_backing": dict(distribution.get("memory_backing", {}) or {}),
            "run_occupancy": dict(distribution.get("run_occupancy", {}) or {}),
            "history": list(bucket.get("history", []) or [])[-10:],
            "history_stats": distribution.get("history_stats", {}),
            "feedback": {
                "recent_count": len(recent_feedback),
                "last_response": recent_feedback[-1] if recent_feedback else None,
            },
        }

    def initiative_status(self) -> dict[str, Any]:
        state = self.load_runtime_state()
        return self._initiative_status_payload(state)

    def initiative_distribution(self) -> dict[str, Any]:
        state = self.load_runtime_state()
        latest_view = self.trace_store.recent_round_signal_views(limit=1)
        distribution_builder = self._controller_callable_override("_initiative_distribution_payload")
        if distribution_builder is not None:
            return distribution_builder(
                state,
                latest_recorded_at=latest_view[-1].get("recorded_at") if latest_view else None,
            )
        return self._initiative_distribution_payload(
            state,
            latest_recorded_at=latest_view[-1].get("recorded_at") if latest_view else None,
        )

    def monologue_status(self) -> dict[str, Any]:
        state = self.load_runtime_state()
        sync_monologue = self._controller_callable_override("_sync_monologue_stream")
        if sync_monologue is not None:
            bucket, _generated = sync_monologue(state)
        else:
            bucket, _generated = self._sync_monologue_stream(state)
        return self.monologue_runtime.status_payload(bucket, now_iso=self._controller_utc_now_iso())

    def monologue_status_lightweight(self) -> dict[str, Any]:
        state = self.load_runtime_state()
        sync_monologue = self._controller_callable_override("_sync_monologue_stream")
        if sync_monologue is not None:
            bucket, _generated = sync_monologue(state, generate_if_due=False)
        else:
            bucket, _generated = self._sync_monologue_stream(state, generate_if_due=False)
        return self.monologue_runtime.status_payload(bucket, now_iso=self._controller_utc_now_iso())

    def monologue_show(self, limit: int | None = None) -> dict[str, Any]:
        state = self.load_runtime_state()
        sync_monologue = self._controller_callable_override("_sync_monologue_stream")
        if sync_monologue is not None:
            bucket, _generated = sync_monologue(state, mark_viewed=True)
        else:
            bucket, _generated = self._sync_monologue_stream(state, mark_viewed=True)
        return self.monologue_runtime.show_payload(bucket, limit=limit)

    def monologue_show_lightweight(self, limit: int | None = None) -> dict[str, Any]:
        state = self.load_runtime_state()
        sync_monologue = self._controller_callable_override("_sync_monologue_stream")
        if sync_monologue is not None:
            bucket, _generated = sync_monologue(state, mark_viewed=False, generate_if_due=False)
        else:
            bucket, _generated = self._sync_monologue_stream(state, mark_viewed=False, generate_if_due=False)
        return self.monologue_runtime.show_payload(bucket, limit=limit)

    def _initiative_force_settings(self, settings: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.initiative_runtime.merge_settings(
            settings,
            {
                "idle_seconds_threshold": 0,
                "vitality_threshold": 0.0,
                "relation_strength_threshold": 0.0,
                "habit_strength_threshold": 0.0,
                "proposal_posterior_threshold": 0.0,
                "hourly_limit": 0,
                "cooldown_seconds": 0,
                "require_memory_backing": False,
            },
        )

    def initiative_update_settings(self, **settings: Any) -> dict[str, Any]:
        state = self.load_runtime_state()
        bucket = self._initiative_state_bucket(state)
        bucket["settings"] = self.initiative_runtime.merge_settings(bucket.get("settings"), settings)
        self._save_state(state, sync=True)
        return self._initiative_status_payload(state)

    def _initiative_why_summary(self, initiative: dict[str, Any] | None) -> str:
        payload = dict(initiative or {})
        suppression_reason = str(payload.get("suppression_reason") or "").strip()
        if suppression_reason:
            return suppression_reason
        top_intent = str(payload.get("top_intent") or "").strip()
        memory_backing = dict(payload.get("memory_backing", {}) or {})
        cue = str(memory_backing.get("cue") or memory_backing.get("summary") or "").strip()
        if top_intent and cue:
            return f"{top_intent} via {cue}"
        if top_intent:
            return top_intent
        expression_mode = str(payload.get("expression_mode") or "").strip()
        if expression_mode:
            return expression_mode
        proposal_type = str(payload.get("proposal_type") or "").strip()
        if proposal_type:
            return proposal_type
        return "initiative why"

    def initiative_why(self, round_ref: int | str = "last") -> dict[str, Any]:
        trace = self.trace_round(round_ref)
        initiative_payload = dict(trace.get("initiative", {}) or {})
        patched = getattr(self.controller, "__dict__", {}).get("_initiative_why_summary")
        summary = patched(initiative_payload) if callable(patched) else self._initiative_why_summary(initiative_payload)
        return {
            "round_id": trace.get("round_id"),
            "trace_ref": trace.get("trace_ref"),
            "initiative": initiative_payload,
            "summary": summary,
        }

    def initiative_trigger_now(
        self,
        *,
        trigger: str = "idle",
        mode: str | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        original_settings: dict[str, Any] | None = None
        had_settings = False
        if force:
            forced_state = self.load_runtime_state()
            bucket = self._initiative_state_bucket(forced_state)
            had_settings = isinstance(bucket.get("settings"), dict)
            original_settings = dict(bucket.get("settings", {}) or {})
            bucket["settings"] = self._initiative_force_settings(bucket.get("settings"))
            self._save_state(forced_state, sync=True)
        try:
            preflight = self.initiative_distribution()
            tick_payload = self.run_endogenous_tick(trigger=trigger, mode=mode or "endogenous_light")
        finally:
            if force:
                restore_state = self.load_runtime_state()
                bucket = self._initiative_state_bucket(restore_state)
                if had_settings:
                    bucket["settings"] = dict(original_settings or {})
                else:
                    bucket.pop("settings", None)
                self._save_state(restore_state, sync=True)
        proposal = dict(tick_payload.get("initiative") or preflight)
        return {
            "trigger": trigger,
            "distribution": preflight,
            "tick": tick_payload,
            "proposal": proposal,
            "auto_sent": bool(proposal.get("auto_sent", False)),
            "forced": bool(force),
        }

    def _prepare_autonomy_step_state(self, state: RuntimeState | None = None) -> RuntimeState:
        current = state or self.load_runtime_state()
        self._sync_tlh_state(current)
        self._sync_autonomy_state(current)
        self._ensure_autonomy_window(current)
        if self._clear_stale_dirty_worktree_run_blocker(current):
            current = self.load_runtime_state()
            self._sync_tlh_state(current)
            self._sync_autonomy_state(current)
            self._ensure_autonomy_window(current)
        return current

    def _autonomy_default_self_run_goal(self, state: RuntimeState) -> str:
        continuity_gap = max(0.0, 0.58 - float(state.self_continuity or 0.0))
        meaning_gap = max(0.0, 0.44 - float(state.meaning_strength or 0.0))
        memory_pull = float(state.memory_fragments or 0.0)
        if continuity_gap >= meaning_gap and continuity_gap >= memory_pull * 0.8:
            return "Inspect recent runtime traces, continuity drift, and memory pressure, then identify the next read-only stabilizing step."
        if memory_pull >= 0.34:
            return "Inspect recent runtime traces and surfaced memory fragments, then identify the next read-only organizing step."
        return "Inspect the repository and current runtime state, then identify the next read-only self-directed step."

    def _autonomy_after_command_step(self, result: dict[str, Any]) -> dict[str, Any]:
        state = self.load_runtime_state()
        self._sync_autonomy_state(state)
        state.autonomy_loop.heartbeat_count += 1
        self._save_state(state, sync=True)
        return self.autonomy_runtime_status()

    def _autonomy_run_dream_pass(self, state: RuntimeState) -> dict[str, Any]:
        dream_runner = self._controller_callable_override("run_dream")
        dream_payload = (
            dream_runner(mode=state.mode)
            if dream_runner is not None
            else self.run_dream(mode=state.mode)
        )
        state = self.load_runtime_state()
        self._sync_autonomy_state(state)
        self._ensure_autonomy_window(state)
        state.autonomy_loop.heartbeat_count += 1
        state.autonomy_loop.total_tool_actions += 1
        state.autonomy_loop.window_tool_actions += 1
        self._append_autonomy_recent_action(
            state,
            action_type="dream_pass",
            summary=f"dream pass in {state.mode}",
            trace_ref=str(dream_payload.get("trace_ref") or ""),
        )
        self._save_state(state, sync=True)
        return self.autonomy_runtime_status()

    def _autonomy_self_run_goal(self, state: RuntimeState) -> str:
        goal_builder = self._controller_callable_override("_generate_autonomy_self_run_goal_via_model")
        model_goal = goal_builder(state) if goal_builder is not None else self._generate_autonomy_self_run_goal_via_model(state)
        if model_goal:
            return model_goal
        return self._autonomy_default_self_run_goal(state)

    def _autonomy_continue_readonly_run_directive(
        self,
        state: RuntimeState,
        policy: AutonomyPolicyState,
    ) -> dict[str, str] | None:
        pending = self._autonomy_pending_readonly_repo_scan(state, policy)
        if pending is None:
            return None
        return {"kind": "repo_scan_approval", **pending}

    def _autonomy_start_readonly_run(self, goal: str) -> dict[str, Any]:
        start_run = self._controller_callable_override("start_run")
        if start_run is not None:
            run_payload = start_run(
                goal,
                allow_commit=False,
                operator_level="read_only",
                include_details=False,
                sync_hot_path=True,
                defer_bootstrap_tool=True,
            )
        else:
            run_payload = self.start_run(
                goal,
                allow_commit=False,
                operator_level="read_only",
                include_details=False,
                sync_hot_path=True,
                defer_bootstrap_tool=True,
            )
        state = self.load_runtime_state()
        self._sync_autonomy_state(state)
        self._ensure_autonomy_window(state)
        state.autonomy_loop.heartbeat_count += 1
        state.autonomy_loop.total_tool_actions += 1
        state.autonomy_loop.window_tool_actions += 1
        summary = str(run_payload.get("goal_summary") or run_payload.get("goal") or goal).strip()
        trace_ref = str(run_payload.get("trace_ref") or "")
        self._append_autonomy_recent_action(
            state,
            action_type="self_run",
            summary=f"autonomy self-study: {summary}",
            trace_ref=trace_ref or None,
        )
        self._autonomy_trace_append(
            state,
            action_type="self_run",
            summary=summary,
            delta={
                "run_id": run_payload.get("run_id"),
                "goal": goal,
                "status": run_payload.get("status"),
                "trace_ref": trace_ref,
            },
        )
        self._save_state(state, sync=True)
        return self.autonomy_runtime_status()

    def _autonomy_continue_readonly_run(self, run_id: str, call_id: str) -> dict[str, Any]:
        resolver = self._controller_callable_override("resolve_run_tool_approval")
        payload = (
            resolver(run_id, call_id, approved=True)
            if resolver is not None
            else self.resolve_run_tool_approval(run_id, call_id, approved=True)
        )
        state = self.load_runtime_state()
        self._sync_autonomy_state(state)
        self._ensure_autonomy_window(state)
        state.autonomy_loop.heartbeat_count += 1
        state.autonomy_loop.total_tool_actions += 1
        state.autonomy_loop.window_tool_actions += 1
        tool = dict(payload.get("tool", {}) or {})
        summary = f"auto-approved {str(tool.get('tool_name') or 'tool')} for read-only run"
        self._append_autonomy_recent_action(
            state,
            action_type="self_run_tool",
            summary=summary,
            trace_ref=str((payload.get("run") or {}).get("trace_ref") or ""),
        )
        self._autonomy_trace_append(
            state,
            action_type="self_run_tool",
            summary=summary,
            delta={
                "run_id": run_id,
                "call_id": call_id,
                "tool_name": tool.get("tool_name"),
                "tool_status": tool.get("status"),
            },
        )
        self._save_state(state, sync=True)
        return self.autonomy_runtime_status()

    def _autonomy_rest_realization(
        self,
        state: RuntimeState,
        policy: AutonomyPolicyState,
    ) -> dict[str, str] | None:
        energy = float(state.body_energy if state.body_energy is not None else state.body_state.energy)
        fatigue = float(state.fatigue if state.fatigue is not None else state.body_state.fatigue)
        severe = fatigue >= 0.94 or energy <= 0.08
        high = fatigue >= 0.82 or energy <= 0.18
        elevated = fatigue >= 0.68 or energy <= 0.28

        if severe:
            if state.mode == "sleep" and self._autonomy_command_allowed("dream run", policy)[0]:
                dream_status = self.dream_status()
                if dream_status.get("enabled"):
                    return {"kind": "dream"}
            if state.mode != "sleep" and self._autonomy_command_allowed("mode set sleep", policy)[0]:
                return {"kind": "command", "command": "mode set sleep"}
        if high and state.mode != "idle" and self._autonomy_command_allowed("mode set idle", policy)[0]:
            return {"kind": "command", "command": "mode set idle"}
        if elevated and self._autonomy_command_allowed("body rest", policy)[0]:
            return {"kind": "command", "command": "body rest"}
        return None

    def _autonomy_pending_readonly_repo_scan(
        self,
        state: RuntimeState,
        policy: AutonomyPolicyState,
    ) -> dict[str, str] | None:
        active_run_id = str(state.active_run_id or "").strip()
        if not active_run_id or state.run_status not in {"running", "paused"}:
            return None
        allowed, _reason = self._autonomy_operator_allowed("read_only", policy)
        if not allowed:
            return None
        try:
            run_state = self._load_run_state(active_run_id)
        except FileNotFoundError:
            return None
        if self._stale_dirty_worktree_run_info(run_state) is not None:
            return None
        if bool(getattr(run_state.policy, "allow_commit", False)):
            return None
        if str(getattr(run_state.policy, "operator_level", "") or "") != "read_only":
            return None
        tool_rows = list(self.trace_store.list_run_tools(active_run_id) or [])
        pending_tool = next(
            (
                row
                for row in reversed(tool_rows)
                if str(row.get("status") or "") == "awaiting_approval"
            ),
            None,
        )
        if pending_tool is None:
            return None
        if str(pending_tool.get("tool_name") or "") != "repo_scan":
            return None
        return {
            "run_id": active_run_id,
            "call_id": str(pending_tool.get("call_id") or ""),
        }

    def _autonomy_due_scheduled_task(
        self,
        state: RuntimeState,
        policy: AutonomyPolicyState,
    ) -> dict[str, str] | None:
        due_tasks = self.scheduled_task_store.due_tasks(reference_at=self._controller_utc_now_iso())
        for task in due_tasks:
            toolset_policy = dict(task.toolset_policy or {})
            operator_level = str(toolset_policy.get("operator_level") or "read_only").strip() or "read_only"
            allowed, _reason = self._autonomy_operator_allowed(operator_level, policy)
            if not allowed:
                continue
            if bool(toolset_policy.get("allow_commit", False)) and not bool(policy.allow_commit):
                continue
            summary = str(task.prompt or task.task_payload.get("prompt") or task.task_payload.get("goal") or task.skill_name).strip()
            return {
                "task_id": task.task_id,
                "summary": summary or f"scheduled task {task.task_id}",
            }
        return None

    def _autonomy_candidate_scores(
        self,
        state: RuntimeState,
        policy: AutonomyPolicyState,
    ) -> dict[str, float]:
        self._sync_tlh_state(state)
        rest_score = max(1.0 - float(state.body_energy or 0.0), float(state.fatigue or 0.0))
        absorb_score = max(float(state.memory_fragments or 0.0), float(state.subjective_state.reject_all or 0.0) * 0.85)
        wander_score = min(
            1.0,
            float(state.subjective_state.spontaneous or 0.0) * 0.75
            + (1.0 - float(state.subjective_state.boundary or 0.0)) * 0.25,
        )
        nothing_score = min(
            1.0,
            float(state.subjective_state.reject_all or 0.0) * 0.7
            + float(state.fatigue or 0.0) * 0.3,
        )
        die_score = min(
            1.0,
            max(0.0, 0.12 - float(state.self_continuity or 0.0))
            + max(0.0, 0.12 - float(state.meaning_strength or 0.0)),
        )
        return {
            "rest": rest_score,
            "absorb": absorb_score,
            "wander": wander_score,
            "nothing": nothing_score,
            "die": die_score,
            "self_run": self._autonomy_self_run_score(state, policy),
        }

    def _autonomy_candidate_action(
        self,
        state: RuntimeState,
        policy: AutonomyPolicyState,
    ) -> str:
        field_probe = self._autonomy_action_field_probe()
        scores = self._autonomy_projected_candidate_scores(state, policy, field_probe)
        return max(scores, key=scores.get)

    def _autonomy_projected_candidate_scores(
        self,
        state: RuntimeState,
        policy: AutonomyPolicyState,
        field_probe: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        prior_scores = {
            str(name): max(0.0, float(score or 0.0))
            for name, score in self._autonomy_candidate_scores(state, policy).items()
        }
        if not isinstance(field_probe, dict):
            return prior_scores
        distribution = dict(field_probe.get("action_distribution", {}) or {})
        if not distribution:
            return prior_scores

        projected_actions = list(prior_scores.keys())
        raw_scores = {
            action: max(0.0, float(distribution.get(action, 0.0) or 0.0))
            for action in projected_actions
        }
        positive_prior_actions = [action for action, score in prior_scores.items() if score > 0.0]
        raw_total = sum(raw_scores.get(action, 0.0) for action in positive_prior_actions)
        prior_total = sum(prior_scores.get(action, 0.0) for action in positive_prior_actions)
        if raw_total <= 0.0 or prior_total <= 0.0:
            return prior_scores

        projected_scores: dict[str, float] = {}
        for action in projected_actions:
            prior_score = prior_scores.get(action, 0.0)
            if prior_score <= 0.0:
                projected_scores[action] = 0.0
                continue
            raw_component = raw_scores.get(action, 0.0) / raw_total
            prior_component = prior_score / prior_total
            projected_scores[action] = round(raw_component * 0.2 + prior_component * 0.8, 6)
        return projected_scores

    def _autonomy_trace_append(
        self,
        state: RuntimeState,
        *,
        action_type: str,
        summary: str,
        delta: dict[str, Any] | None = None,
    ) -> None:
        before_hash = self._state_hash(state)
        result = CommandResult(
            applied=True,
            scope="autonomy",
            delta=delta or {"action_type": action_type},
            operator_level="read_only",
            rollback_available=False,
            canonical=f"autonomy {action_type}",
            command_id=f"autonomy-{uuid4().hex[:12]}",
            cause_type="endogenous",
            boundary_action="allow_internal",
        )
        self.trace_store.append_command(
            f"autonomy {action_type}",
            result,
            before_hash,
            before_hash,
            session_id=state.session_id,
            recorded_at=utc_now_iso(),
            sync=False,
        )

    def prepare_autonomy_background(self, *, state: RuntimeState | None = None) -> dict[str, Any]:
        current = self._prepare_autonomy_step_state(state)
        policy = current.autonomy_policy
        loop = current.autonomy_loop
        ticket: dict[str, Any] = {
            "runtime_revision": int(current.runtime_revision or 0),
            "action": "inactive",
        }

        if not policy.enabled or not loop.running:
            return ticket
        if current.safe_mode:
            return {**ticket, "action": "stop_safe_mode"}
        if float(current.budget_remaining or 0.0) <= 0.0:
            return {**ticket, "action": "stop_budget_exhausted"}
        if policy.max_rounds_per_hour > 0 and loop.window_endogenous_rounds >= policy.max_rounds_per_hour:
            return {**ticket, "action": "stop_round_budget"}
        if policy.max_tool_actions_per_hour > 0 and loop.window_tool_actions >= policy.max_tool_actions_per_hour:
            return {**ticket, "action": "stop_tool_budget"}
        current_hour = time.localtime().tm_hour
        if current_hour in policy.quiet_hours:
            return {**ticket, "action": "quiet"}

        action = self._autonomy_candidate_action(current, policy)
        if action == "rest":
            recovery = self._autonomy_rest_realization(current, policy)
            if recovery is not None:
                if recovery.get("kind") == "dream":
                    return {**ticket, "action": "dream_pass"}
                return {
                    **ticket,
                    "action": "command",
                    "command": str(recovery.get("command") or ""),
                }
            return {**ticket, "action": "command", "command": "body rest"}

        scheduled_task = self._autonomy_due_scheduled_task(current, policy)
        if scheduled_task is not None:
            return {
                **ticket,
                "action": "scheduled_task",
                "task_id": str(scheduled_task.get("task_id") or ""),
                "summary": str(scheduled_task.get("summary") or "scheduled task"),
            }

        context, relation_state, slow_variables = self._autonomy_step_context(current)
        trigger = self.endogenous_scheduler.build_trigger(
            state=current,
            context=context,
            relation_state=relation_state,
            slow_variables=slow_variables,
            pool_state=current.motivation_pool_state,
        )
        session_selection = self._initiative_session_selection()
        if trigger is not None and self._autonomy_command_allowed("endogenous tick", policy)[0]:
            if not bool(session_selection.get("fresh_session_available")):
                return {
                    **ticket,
                    "action": "nothing",
                    "refresh_due": True,
                    "suppression_reason": "no_fresh_session",
                }
            return {
                **ticket,
                "action": "endogenous_tick",
                "trigger_type": str(trigger.trigger_type or "idle"),
                "selected_mode": str(trigger.selected_mode or ""),
                "summary": str(trigger.audit_reason or trigger.trigger_type or "endogenous tick"),
            }

        if action == "die":
            return {**ticket, "action": "terminal_intent"}
        if action == "absorb":
            return {**ticket, "action": "absorb"}
        if action == "wander" and current.round_count > 0:
            return {
                **ticket,
                "action": "wander",
                "round_id": int(current.round_count or 0),
            }
        if action == "self_run":
            pending_progress = self._autonomy_pending_readonly_repo_scan(current, policy)
            if pending_progress is not None:
                return {
                    **ticket,
                    "action": "self_run_continue",
                    "run_id": str(pending_progress.get("run_id") or ""),
                    "call_id": str(pending_progress.get("call_id") or ""),
                }
            return {
                **ticket,
                "action": "self_run_start",
                "goal_fallback": self._autonomy_default_self_run_goal(current),
                "goal_state": to_dict(current),
            }
        return {
            **ticket,
            "action": "nothing",
            "refresh_due": bool(self._heartbeat_side_channel_refresh_due(current, noop_heartbeat=True)),
        }
    def execute_autonomy_background(self, ticket: dict[str, Any]) -> dict[str, Any]:
        action = str(ticket.get("action") or "")
        if action != "self_run_start":
            return {}
        fallback_goal = str(ticket.get("goal_fallback") or "").strip()
        raw_goal_state = ticket.get("goal_state")
        if not isinstance(raw_goal_state, dict):
            return {
                "goal": fallback_goal,
                "used_model": False,
            }
        goal_builder = self._controller_callable_override("_generate_autonomy_self_run_goal_via_model")
        state = RuntimeState(**raw_goal_state)
        model_goal = goal_builder(state) if goal_builder is not None else self._generate_autonomy_self_run_goal_via_model(state)
        return {
            "goal": str(model_goal or fallback_goal).strip(),
            "used_model": bool(model_goal),
        }
    def commit_autonomy_background(
        self,
        ticket: dict[str, Any],
        execution: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        state = self.load_runtime_state()
        expected_revision = int(ticket.get("runtime_revision", -1) or -1)
        if expected_revision >= 0 and int(state.runtime_revision or 0) != expected_revision:
            return {
                **self.autonomy_runtime_status(),
                "committed": False,
                "reason": "runtime_revision_changed",
            }

        action = str(ticket.get("action") or "inactive")
        execution_payload = dict(execution or {})
        try:
            if action == "inactive":
                return {
                    **self.autonomy_runtime_status(),
                    "committed": False,
                    "reason": "autonomy_not_running",
                }
            if action == "stop_safe_mode":
                self._sync_autonomy_state(state)
                state.autonomy_policy.enabled = False
                state.autonomy_loop.running = False
                state.autonomy_loop.stop_reason = "safe_mode_active"
                self._append_autonomy_recent_action(state, action_type="stop", summary="autonomy stopped by safe mode")
                self._save_state(state, sync=True)
                return {
                    **self.autonomy_runtime_status(),
                    "committed": True,
                    "reason": "",
                }
            if action == "stop_budget_exhausted":
                self._sync_autonomy_state(state)
                state.autonomy_policy.enabled = False
                state.autonomy_loop.running = False
                state.autonomy_loop.stop_reason = "budget_exhausted"
                self._append_autonomy_recent_action(state, action_type="stop", summary="autonomy stopped by exhausted budget")
                self._save_state(state, sync=True)
                return {
                    **self.autonomy_runtime_status(),
                    "committed": True,
                    "reason": "",
                }
            if action == "stop_round_budget":
                self._sync_autonomy_state(state)
                state.autonomy_policy.enabled = False
                state.autonomy_loop.running = False
                state.autonomy_loop.stop_reason = "round_budget_reached"
                self._append_autonomy_recent_action(state, action_type="stop", summary="autonomy stopped by round budget")
                self._save_state(state, sync=True)
                return {
                    **self.autonomy_runtime_status(),
                    "committed": True,
                    "reason": "",
                }
            if action == "stop_tool_budget":
                self._sync_autonomy_state(state)
                state.autonomy_policy.enabled = False
                state.autonomy_loop.running = False
                state.autonomy_loop.stop_reason = "tool_budget_reached"
                self._append_autonomy_recent_action(state, action_type="stop", summary="autonomy stopped by tool budget")
                self._save_state(state, sync=True)
                return {
                    **self.autonomy_runtime_status(),
                    "committed": True,
                    "reason": "",
                }
            if action == "quiet":
                self._sync_autonomy_state(state)
                self._refresh_heartbeat_side_channels(state)
                state.autonomy_loop.heartbeat_count += 1
                self._append_autonomy_recent_action(state, action_type="quiet", summary="quiet-hours heartbeat")
                self._save_state(state, sync=True)
                return {
                    **self.autonomy_runtime_status(),
                    "committed": True,
                    "reason": "",
                }
            if action == "command":
                result = self._autonomy_after_command_step(
                    self._autonomy_execute_command(str(ticket.get("command") or ""))
                )
                return {
                    **result,
                    "committed": True,
                    "reason": "",
                }
            if action == "dream_pass":
                result = self._autonomy_run_dream_pass(state)
                return {
                    **result,
                    "committed": True,
                    "reason": "",
                }
            if action == "scheduled_task":
                trigger_payload = self.trigger_scheduled_task(str(ticket.get("task_id") or ""))
                state = self.load_runtime_state()
                self._sync_autonomy_state(state)
                self._ensure_autonomy_window(state)
                state.autonomy_loop.heartbeat_count += 1
                state.autonomy_loop.total_tool_actions += 1
                state.autonomy_loop.window_tool_actions += 1
                run_payload = dict(trigger_payload.get("run") or {})
                summary = str(run_payload.get("goal_summary") or ticket.get("summary") or ticket.get("task_id") or "scheduled task").strip()
                trace_ref = str(run_payload.get("trace_ref") or "")
                self._append_autonomy_recent_action(
                    state,
                    action_type="scheduled_task",
                    summary=summary,
                    trace_ref=trace_ref or None,
                )
                self._autonomy_trace_append(
                    state,
                    action_type="scheduled_task",
                    summary=summary,
                    delta={
                        "task_id": ticket.get("task_id"),
                        "run_id": run_payload.get("run_id"),
                        "trace_ref": trace_ref,
                    },
                )
                self._save_state(state, sync=True)
                return {
                    **self.autonomy_runtime_status(),
                    "committed": True,
                    "reason": "",
                }
            if action == "endogenous_tick":
                self._refresh_heartbeat_side_channels(state)
                tick_payload = self.run_endogenous_tick(
                    trigger=str(ticket.get("trigger_type") or "idle"),
                    mode=str(ticket.get("selected_mode") or "") or None,
                )
                state = self.load_runtime_state()
                self._sync_autonomy_state(state)
                self._ensure_autonomy_window(state)
                state.autonomy_loop.heartbeat_count += 1
                state.autonomy_loop.total_endogenous_rounds += 1
                state.autonomy_loop.window_endogenous_rounds += 1
                round_id = tick_payload.get("round_id")
                trace_ref = f"round://{round_id}" if round_id is not None else None
                self._append_autonomy_recent_action(
                    state,
                    action_type="endogenous_tick",
                    summary=str(ticket.get("summary") or ticket.get("trigger_type") or "endogenous tick"),
                    round_id=round_id,
                    trace_ref=trace_ref,
                )
                self._save_state(state, sync=True)
                return {
                    **self.autonomy_runtime_status(),
                    "committed": True,
                    "reason": "",
                }
            if action == "terminal_intent":
                self._sync_autonomy_state(state)
                self._refresh_heartbeat_side_channels(state)
                state.autonomy_loop.heartbeat_count += 1
                if state.autonomy_policy.auto_safe_mode:
                    state.safe_mode = True
                    state.mode = "safe"
                state.autonomy_policy.enabled = False
                state.autonomy_loop.running = False
                state.autonomy_loop.stop_reason = "terminal_intent_boundary"
                self._append_autonomy_recent_action(
                    state,
                    action_type="terminal_intent",
                    summary="die mapped to bounded terminal intent",
                )
                self._autonomy_trace_append(
                    state,
                    action_type="terminal_intent",
                    summary="die mapped to bounded terminal intent",
                    delta={"terminal_intent": True},
                )
                self._save_state(state, sync=True)
                return {
                    **self.autonomy_runtime_status(),
                    "committed": True,
                    "reason": "",
                }
            if action == "absorb":
                self._sync_autonomy_state(state)
                self._refresh_heartbeat_side_channels(state)
                payload = self.memory_top(limit=3)
                state.autonomy_loop.heartbeat_count += 1
                state.autonomy_loop.total_tool_actions += 1
                state.autonomy_loop.window_tool_actions += 1
                self._append_autonomy_recent_action(
                    state,
                    action_type="absorb",
                    summary=f"absorbed {len(payload)} memory cues",
                )
                self._autonomy_trace_append(
                    state,
                    action_type="absorb",
                    summary="memory top",
                    delta={"memory_count": len(payload)},
                )
                self._save_state(state, sync=True)
                return {
                    **self.autonomy_runtime_status(),
                    "committed": True,
                    "reason": "",
                }
            if action == "wander":
                self._sync_autonomy_state(state)
                self._refresh_heartbeat_side_channels(state)
                round_id = int(ticket.get("round_id", 0) or 0)
                replay_payload = self.replay(round_id, seed=max(1, round_id))
                state.autonomy_loop.heartbeat_count += 1
                state.autonomy_loop.total_tool_actions += 1
                state.autonomy_loop.window_tool_actions += 1
                self._append_autonomy_recent_action(
                    state,
                    action_type="wander",
                    summary=f"replayed round {round_id}",
                    round_id=round_id,
                    trace_ref=f"round://{round_id}",
                )
                self._autonomy_trace_append(
                    state,
                    action_type="wander",
                    summary="replay latest round",
                    delta={"replayed_action": replay_payload.get("replayed_action")},
                )
                self._save_state(state, sync=True)
                return {
                    **self.autonomy_runtime_status(),
                    "committed": True,
                    "reason": "",
                }
            if action == "self_run_continue":
                result = self._autonomy_continue_readonly_run(
                    str(ticket.get("run_id") or ""),
                    str(ticket.get("call_id") or ""),
                )
                return {
                    **result,
                    "committed": True,
                    "reason": "",
                }
            if action == "self_run_start":
                goal = str(execution_payload.get("goal") or ticket.get("goal_fallback") or "").strip()
                if not goal:
                    goal = self._autonomy_default_self_run_goal(state)
                result = self._autonomy_start_readonly_run(goal)
                return {
                    **result,
                    "committed": True,
                    "reason": "",
                }

            self._sync_autonomy_state(state)
            state.autonomy_loop.heartbeat_count += 1
            if bool(ticket.get("refresh_due")):
                self._refresh_heartbeat_side_channels(state)
            self._append_autonomy_recent_action(state, action_type="nothing", summary="no outward autonomy action this heartbeat")
            self._autonomy_trace_append(
                state,
                action_type="nothing",
                summary="no outward autonomy action",
                delta={"would_output": False},
            )
            self._save_state(state, sync=True)
            return {
                **self.autonomy_runtime_status(),
                "committed": True,
                "reason": "",
            }
        except Exception as exc:
            state = self.load_runtime_state()
            self._autonomy_record_failure(state, exc)
            self._append_autonomy_recent_action(state, action_type="failure", summary=str(exc))
            self._save_state(state, sync=True)
            return {
                **self.autonomy_runtime_status(),
                "committed": True,
                "reason": "exception",
            }
    def autonomy_step(self) -> dict[str, Any]:
        ticket = self.prepare_autonomy_background()
        execution = self.execute_autonomy_background(ticket)
        return self.commit_autonomy_background(ticket, execution)
    def start_run(
        self,
        goal: str,
        *,
        allow_commit: bool = False,
        operator_level: str = "read_only",
        replace_active: bool = False,
        interrupt_reason: str = "interrupted_by_user",
        bootstrap: tuple[RunState, dict[str, Any], dict[str, Any]] | None = None,
        include_details: bool = False,
        sync_hot_path: bool = False,
        defer_bootstrap_tool: bool = False,
    ) -> dict[str, Any]:
        state = self.load_runtime_state()
        if state.active_run_id and state.run_status in {"running", "paused"}:
            if replace_active:
                self.interrupt_run(
                    state.active_run_id,
                    reason=interrupt_reason,
                    message="run interrupted by a new terminal input",
                )
                state = self.load_runtime_state()
            else:
                return self.run_status(state.active_run_id)

        recorded_at = self._controller_utc_now_iso()
        before_hash = self._state_hash(state)
        dirty = self._dirty_worktree_snapshot()
        request = RunRequest(
            goal=goal.strip(),
            scenario="task",
            mode=state.mode or "interactive",
            allow_commit=allow_commit,
            operator_level=operator_level,
        )
        if bootstrap is None:
            supervisor = SupervisorLoop(project_root=self.project_root, planner=self._plan_run_via_model)
            if defer_bootstrap_tool:
                run_state, step_trace, tool_trace = supervisor.prepare_bootstrap(
                    request,
                    session_id=state.session_id,
                    recorded_at=recorded_at,
                )
            else:
                run_state, step_trace, tool_trace = supervisor.bootstrap(
                    request,
                    session_id=state.session_id,
                    recorded_at=recorded_at,
                )
        else:
            run_state, step_trace, tool_trace = bootstrap
            run_state = RunState(**to_dict(run_state))
            step_trace = dict(step_trace)
            tool_trace = dict(tool_trace)
            run_state.goal = request.goal
            run_state.policy = self._run_policy(allow_commit=allow_commit, operator_level=operator_level)
            run_state.budget = self._run_budget()
            run_state.session_id = state.session_id
            run_state.created_at = run_state.created_at or recorded_at
            run_state.updated_at = recorded_at
        call_id = str(tool_trace.get("call_id") or f"{run_state.run_id}:tool:0")
        step_trace.setdefault("run_id", run_state.run_id)
        step_trace.setdefault("trace_ref", self._run_step_trace_ref(run_state.run_id, str(step_trace.get("step_id") or run_state.current_step_id or "step")))
        step_trace.setdefault("round_id", None)
        tool_trace.setdefault("run_id", run_state.run_id)
        tool_trace.setdefault("call_id", call_id)
        tool_trace.setdefault("trace_ref", self._run_tool_trace_ref(run_state.run_id, call_id))
        tool_trace.setdefault("round_id", None)
        tool_trace.setdefault("input", {"goal": request.goal})
        if dirty["detected"]:
            run_state.status = "paused"
            run_state.dirty_worktree_detected = True
            run_state.stop_reason = StopReason(
                code="dirty_worktree",
                message="detected existing uncommitted changes before mutation phase",
                retryable=True,
                needs_operator=True,
            )
            step_trace["status"] = "paused"
            step_trace["pause_reason"] = run_state.stop_reason.code
            tool_trace["dirty_entries"] = dirty["entries"]
        self._sync_run_state_to_runtime(state, run_state)
        self._save_state(state, sync=sync_hot_path)
        after_hash = self._state_hash(state)
        tool_payload = {
            **tool_trace,
            "before_state_hash": before_hash,
            "after_state_hash": after_hash,
        }
        self.trace_store.write_run(to_dict(run_state), session_id=state.session_id, recorded_at=recorded_at, sync=sync_hot_path)
        self.trace_store.append_step_trace(step_trace, session_id=state.session_id, recorded_at=recorded_at, sync=sync_hot_path)
        self.trace_store.append_tool_trace(
            tool_payload,
            session_id=state.session_id,
            recorded_at=recorded_at,
            sync=sync_hot_path,
        )
        run_payload = self._run_status_payload(run_state)
        if not include_details:
            return run_payload
        return {
            "run": run_payload,
            "explain": self._run_explain_payload(run_state),
            "steps": [step_trace],
            "tools": [tool_payload],
        }
    def resolve_run_tool_approval(self, run_id: str, call_id: str, *, approved: bool) -> dict[str, Any]:
        run_state = self._load_run_state(run_id)
        tool_rows = self.trace_store.list_run_tools(run_id)
        pending_tool = next(
            (
                row
                for row in reversed(tool_rows)
                if str(row.get("call_id") or "") == call_id
            ),
            None,
        )
        if pending_tool is None:
            raise FileNotFoundError(f"run tool {call_id} not found")
        recorded_at = self._controller_utc_now_iso()
        current_step = self._current_task_node(run_state)
        step_trace = None
        if current_step is not None:
            step_trace = {
                "run_id": run_state.run_id,
                "step_id": current_step.node_id,
                "title": current_step.title,
                "detail": current_step.detail,
                "status": "paused" if not approved else current_step.status,
                "tool_choice": current_step.tool_choice,
                "expected_observation": current_step.expected_observation,
                "success_criteria": current_step.success_criteria,
                "confidence": current_step.confidence,
                "matched_files": list(current_step.metadata.get("matched_files", []) or []),
                "trace_ref": self._run_step_trace_ref(run_state.run_id, current_step.node_id),
                "round_id": None,
            }

        if approved:
            supervisor = SupervisorLoop(project_root=self.project_root, planner=self._plan_run_via_model)
            result = supervisor.execute_prepared_tool(
                str(pending_tool.get("tool_name") or "repo_scan"),
                goal=str((pending_tool.get("input") or {}).get("goal") or run_state.goal),
            )
            run_state.last_tool_result = result
            run_state.updated_at = recorded_at
            run_state.stop_reason = None
            if current_step is not None:
                current_step.metadata["matched_files"] = list(result.metadata.get("matched_files", []) or [])
            tool_payload = {
                **dict(pending_tool),
                "status": result.status,
                "summary": result.summary,
                "output_excerpt": result.output_excerpt,
                "before_state_hash": result.before_state_hash,
                "after_state_hash": result.after_state_hash,
                "matched_files": list(result.metadata.get("matched_files", []) or []),
                "file_count": int(result.metadata.get("file_count", 0) or 0),
                "recorded_at": recorded_at,
                "recorded_date": iso_date(recorded_at),
            }
        else:
            run_state.status = "paused"
            run_state.stop_reason = StopReason(
                code="operator_rejected",
                message="tool execution rejected by operator",
                retryable=True,
                needs_operator=True,
            )
            run_state.updated_at = recorded_at
            if current_step is not None:
                current_step.status = "paused"
            tool_payload = {
                **dict(pending_tool),
                "status": "rejected",
                "summary": str(pending_tool.get("summary") or "tool execution rejected"),
                "output_excerpt": "",
                "recorded_at": recorded_at,
                "recorded_date": iso_date(recorded_at),
            }

        self._persist_run_state(run_state)
        if step_trace is not None:
            self.trace_store.append_step_trace(
                step_trace,
                session_id=run_state.session_id,
                recorded_at=recorded_at,
                sync=True,
            )
        self.trace_store.append_tool_trace(
            tool_payload,
            session_id=run_state.session_id,
            recorded_at=recorded_at,
            sync=True,
        )
        return {
            "run": self._run_status_payload(run_state),
            "explain": self._run_explain_payload(run_state),
            "tool": tool_payload,
        }
    def interrupt_run(
        self,
        run_id: str | None = None,
        *,
        reason: str = "interrupted_by_user",
        message: str = "run interrupted",
    ) -> dict[str, Any]:
        run_state = self._load_run_state(run_id)
        if run_state.status not in {"completed", "aborted", "interrupted"}:
            run_state.status = "interrupted"
            run_state.stop_reason = StopReason(
                code=reason,
                message=message,
                retryable=True,
                needs_operator=False,
            )
            run_state.updated_at = self._controller_utc_now_iso()
            self._persist_run_state(run_state)
        return self.run_status(run_state.run_id)
    def run_status(self, run_id: str | None = None) -> dict[str, Any]:
        run_state = self._load_run_state(run_id)
        return self._run_status_payload(run_state)
    def pause_run(self, run_id: str | None = None) -> dict[str, Any]:
        run_state = self._load_run_state(run_id)
        if run_state.status not in {"completed", "aborted"}:
            run_state.status = "paused"
            run_state.stop_reason = StopReason(
                code="operator_paused",
                message="paused by operator",
                retryable=True,
                needs_operator=True,
            )
            run_state.updated_at = self._controller_utc_now_iso()
            self._persist_run_state(run_state)
        return self.run_status(run_state.run_id)
    def resume_run(self, run_id: str | None = None) -> dict[str, Any]:
        run_state = self._load_run_state(run_id)
        dirty = self._dirty_worktree_snapshot()
        if dirty["detected"]:
            run_state.status = "paused"
            run_state.dirty_worktree_detected = True
            run_state.stop_reason = StopReason(
                code="dirty_worktree",
                message="cannot resume while worktree is dirty",
                retryable=True,
                needs_operator=True,
            )
        elif run_state.status not in {"completed", "aborted"}:
            run_state.status = "running"
            run_state.dirty_worktree_detected = False
            run_state.stop_reason = None
        run_state.updated_at = self._controller_utc_now_iso()
        self._persist_run_state(run_state)
        return self.run_status(run_state.run_id)
    def abort_run(self, run_id: str | None = None, *, reason: str = "operator_requested") -> dict[str, Any]:
        run_state = self._load_run_state(run_id)
        run_state.status = "aborted"
        run_state.stop_reason = StopReason(
            code=reason,
            message="run aborted",
            retryable=False,
            needs_operator=False,
        )
        run_state.updated_at = self._controller_utc_now_iso()
        self._persist_run_state(run_state)
        return self.run_status(run_state.run_id)
    def explain_run(self, run_id: str | None = None) -> dict[str, Any]:
        run_state = self._load_run_state(run_id)
        return self._run_explain_payload(run_state)
    def run_steps(self, run_id: str | None = None) -> dict[str, Any]:
        effective_run_id = self._resolve_run_id(run_id)
        return {"run_id": effective_run_id, "steps": self.trace_store.list_run_steps(effective_run_id)}
    def run_tools(self, run_id: str | None = None) -> dict[str, Any]:
        effective_run_id = self._resolve_run_id(run_id)
        return {"run_id": effective_run_id, "tools": self.trace_store.list_run_tools(effective_run_id)}
