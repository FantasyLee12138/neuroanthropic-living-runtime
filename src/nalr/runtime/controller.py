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
from nalr.runtime.chat_kernel_v2 import ChatKernelV2
from nalr.runtime.diagnostics_runtime import DiagnosticsRuntimeService
from nalr.runtime.model_routing_runtime import ModelRoutingRuntime
from nalr.runtime.model_payload_runtime import ModelPayloadRuntime
from nalr.runtime.observer_settings_runtime import ObserverSettingsRuntime
from nalr.runtime.state_runtime import StateRuntimeService
from nalr.runtime.observer_runtime import ObserverRuntimeService
from nalr.runtime.round_pipeline import RoundPipelineRuntime
from nalr.runtime.run_runtime import RunRuntimeService
from nalr.runtime.scheduled_tasks import ScheduledTaskRunRecord, ScheduledTaskSchedule, ScheduledTaskSpec, ScheduledTaskStore
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
    AgencyLoopState,
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
    ProactiveActionState,
    ProbabilisticContribution,
    PurposeMemoryEntry,
    RenderPlan,
    RenderedExpression,
    RelationshipCommitmentState,
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
    SubjectKernelState,
    SubjectiveState,
    MeaningConflict,
    MeaningRepair,
    MeaningSource,
    MeaningSystemState,
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

RUNTIME_SCHEMA_VERSION = 2
HEARTBEAT_NOOP_SIDE_CHANNEL_THROTTLE_SECONDS = 2.0
STALE_DIRTY_WORKTREE_RUN_SECONDS = 300.0


class RuntimeController:
    def __init__(self, project_root: Path, config_root: Path | None = None, home_path: Path | None = None) -> None:
        self.project_root = Path(project_root)
        self.config_root = Path(config_root) if config_root else self.project_root / "config"
        self.home_path = Path(home_path) if home_path else self.project_root / ".alive"
        self.runtime_dir = self.home_path / "runtime"
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.runtime_dir / "persona_state.json"
        self.runtime_migration_status_path = self.runtime_dir / "runtime_migration_status.json"
        self.observer_settings_path = self.runtime_dir / "observer_settings.json"
        self.runtime_parquet_dir = self.runtime_dir / "parquet"
        self.state_parquet_path = self.runtime_parquet_dir / "persona_state.parquet"
        self.checkpoint_dir = self.runtime_dir / "checkpoints"
        self.snapshot_dir = self.runtime_dir / "snapshots"
        self.runtime_parquet_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.observer_settings_runtime = ObserverSettingsRuntime(
            project_root=self.project_root,
            config_root=self.config_root,
            home_path=self.home_path,
            observer_settings_path=self.observer_settings_path,
        )
        self._state_io = AsyncIOWorker("nalr-state-io")
        self._state_cache: RuntimeState | None = None
        self._trace_round_cache: dict[int, dict[str, Any]] = {}
        self._trace_derived_cache: dict[int, dict[str, Any]] = {}
        self._trace_round_cache_limit = 16
        self._cold_flush_round_interval = 8
        self._cold_flush_interval_seconds = 2.0
        self._rounds_since_cold_flush = 0
        self._last_cold_flush_at = time.monotonic()

        self.config = self._load_config()
        thresholds = self.config["thresholds"]["thresholds"]
        self.trace_store = TraceStore(self.home_path)
        self.memory_store = MemoryStore(self.home_path)
        from nalr.terminal_bridge.session import TerminalSessionStore

        self.terminal_sessions = TerminalSessionStore(self.runtime_dir)
        self.scheduled_task_store = ScheduledTaskStore(self.runtime_dir / "scheduled_tasks")
        self.agents = build_agents(
            conflict_high=thresholds["conflict_high"],
            conflict_critical=thresholds["conflict_critical"],
            max_resample_rounds=thresholds["max_resample_rounds"],
        )
        self.agent_map = {agent.name: agent for agent in self.agents}
        self.skills = build_skill_registry()
        self.skill_executor = SkillExecutor(
            self.skills,
            circuit_breaker_path=self.memory_store.circuit_breaker_path,
            environment_fingerprint=self._breaker_environment_fingerprint(),
        )
        self.model_router = ModelRouter.from_config(self.config["models"])
        self.model_routing_runtime = ModelRoutingRuntime(self)
        self.model_payload_runtime = ModelPayloadRuntime(self)
        model_gateway_cfg = self.config["models"].get("models")
        self.model_gateway = ModelGateway.from_config(model_gateway_cfg) if isinstance(model_gateway_cfg, dict) else None
        self.identity_runtime = IdentityRuntime(self._identity_cfg(), self._provider_descriptor)
        entropy_cfg = dict(self.config.get("entropy", {}).get("entropy", {}))
        provider_name = str(entropy_cfg.get("provider", "macos_os_urandom"))
        if provider_name != "macos_os_urandom":
            raise ValueError(f"unsupported entropy provider '{provider_name}'; expected 'macos_os_urandom'")
        entropy_provider = MacOSSystemEntropyProvider()
        self.entropy_pool = QuantumEntropyPool(
            provider=entropy_provider,
            prefetch_bytes=int(entropy_cfg.get("prefetch_bytes", 256)),
            hard_block_on_unavailable=bool(entropy_cfg.get("hard_block_on_unavailable", True)),
        )
        if os.getenv("PYTEST_CURRENT_TEST") and not os.getenv("NALR_DISABLE_TEST_QRNG_SEED"):
            self.entropy_pool.ingest_bytes(bytes([128]) * (256 * 64), source="pytest_qrng_fixture", reason="test harness")
        self.authenticity_policy = AuthenticityPolicy(self._identity_cfg())
        self.vitality_engine = VitalityEngine()
        self.probability_integrator = ProbabilityFieldIntegrator()
        self.endogenous_motivation_pool = EndogenousMotivationPool()
        self.motivation_feedback_updater = MotivationFeedbackUpdater()
        self.endogenous_scheduler = EndogenousTickScheduler()
        self._pending_endogenous_trigger: EndogenousTickTrigger | None = None
        self._pending_endogenous_trigger_context: EndogenousTriggerContext | None = None
        self.dream_orchestrator = DreamOrchestrator(
            project_root=self.project_root,
            home_path=self.home_path,
            config=self.config["dream"]["dream"],
            memory_store=self.memory_store,
            vitality_engine=self.vitality_engine,
        )
        self.long_run_analyzer = LongRunAnalyzer(self.trace_store, self.identity_payload, CORE_ACTIONS)
        self.initiative_runtime = InitiativeRuntime(self.config["models"].get("initiative_trigger", {}))
        self.monologue_runtime = MonologueStreamRuntime(
            self.runtime_dir / "monologue_fragments.jsonl",
            self.config["models"].get("monologue_stream", {}),
        )
        self.state_runtime = StateRuntimeService(self)
        self.diagnostics_runtime = DiagnosticsRuntimeService(self)
        self.chat_kernel_v2 = ChatKernelV2(self)
        self.round_pipeline = RoundPipelineRuntime(self)
        self.run_runtime = RunRuntimeService(self)
        self.observer_runtime = ObserverRuntimeService(self)

        if not self.state_parquet_path.exists() and self.state_path.exists():
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            rewrite_snapshot(
                self.state_parquet_path,
                [{"payload_json": json.dumps(payload, ensure_ascii=False, sort_keys=True)}],
                schema={"payload_json": "VARCHAR"},
            )

        if not self.state_parquet_path.exists() and not self.state_path.exists():
            initial_state = RuntimeState(
                agents_enabled={
                    name: agent_cfg.get("enabled", True)
                    for name, agent_cfg in self.config["agents"]["agents"].items()
                }
            )
            self._ensure_default_autonomy_runtime(initial_state, force=True)
            self._save_state(initial_state, sync=True)
        else:
            self._state_cache = self.load_runtime_state()
        self._migrate_runtime_schema_if_needed()

    def _load_config(self) -> dict[str, Any]:
        def read_yaml(name: str) -> dict[str, Any]:
            path = self.config_root / name
            return yaml.safe_load(path.read_text(encoding="utf-8"))

        temperament_cfg = self._normalize_temperament_config(read_yaml("temperament.yaml"))
        resource_rules = read_yaml("resource_rules.yaml")
        resource_rules.setdefault("resource_defaults", {})
        resource_rules["resource_defaults"].setdefault("latency_sla_ms", 250)
        resource_rules["resource_defaults"].setdefault("target_burn_ratio", 0.20)
        observer_settings = self._load_observer_settings_file()
        models_cfg = self._apply_observer_settings_to_models(read_yaml("models.yaml"), observer_settings)
        return {
            "agents": read_yaml("agents.yaml"),
            "modes": read_yaml("modes.yaml"),
            "scenarios": read_yaml("scenarios.yaml"),
            "thresholds": read_yaml("thresholds.yaml"),
            "entropy": read_yaml("entropy.yaml"),
            "temperament": temperament_cfg,
            "resource_rules": resource_rules,
            "output_style": read_yaml("output_style.yaml"),
            "models": models_cfg,
            "identity": read_yaml("identity.yaml"),
            "dream": read_yaml("dream.yaml"),
            "observer_settings": observer_settings,
        }

    def _default_observer_settings(self) -> dict[str, Any]:
        return self.observer_settings_runtime.default_settings()

    def _load_observer_settings_file(self) -> dict[str, Any]:
        return self.observer_settings_runtime.load_settings_file()

    def _normalize_observer_settings(self, payload: dict[str, Any], *, base: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.observer_settings_runtime.normalize_settings(
            payload,
            base=base,
            current_config=getattr(self, "config", None),
        )

    def _normalize_model_config(self, payload: dict[str, Any], *, tier_mode: bool) -> dict[str, Any]:
        return self.observer_settings_runtime.normalize_model_config(payload, tier_mode=tier_mode)

    def _known_model_tier_names(self, *, observer_model_tiers: dict[str, Any] | None = None) -> set[str]:
        return self.observer_settings_runtime.known_model_tier_names(
            observer_model_tiers=observer_model_tiers,
            current_config=getattr(self, "config", None),
        )

    def _normalize_observer_string_list(self, values: list[Any]) -> list[str]:
        return self.observer_settings_runtime.normalize_string_list(values)

    def _normalize_observer_path(self, value: str) -> str:
        return self.observer_settings_runtime.normalize_path(value)

    def _normalize_observer_path_list(self, values: list[Any]) -> list[str]:
        return self.observer_settings_runtime.normalize_path_list(values)

    def _apply_observer_settings_to_models(self, models_cfg: dict[str, Any], observer_settings: dict[str, Any]) -> dict[str, Any]:
        return self.observer_settings_runtime.apply_settings_to_models(models_cfg, observer_settings)

    def _observer_settings(self) -> dict[str, Any]:
        return self.observer_settings_runtime.current_settings(getattr(self, "config", None))

    def observer_settings_current(self) -> dict[str, Any]:
        return self._observer_settings()

    def _newborn_subjective_baseline(self) -> dict[str, float]:
        newborn = dict(self._observer_settings().get("newborn", {}) or {})
        subjective = dict(newborn.get("subjective_state", {}) or {})
        return {
            "spontaneous": round(_clip(float(subjective.get("spontaneous", 0.0) or 0.0)), 4),
        }

    def _apply_subjective_baseline_floor(self, state: RuntimeState) -> None:
        baseline = self._newborn_subjective_baseline()
        state.subjective_state.spontaneous = round(
            max(float(state.subjective_state.spontaneous or 0.0), float(baseline.get("spontaneous", 0.0) or 0.0)),
            4,
        )

    def normalize_observer_settings_payload(
        self,
        payload: dict[str, Any],
        *,
        base: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.observer_settings_runtime.normalize_settings(
            payload,
            base=base,
            current_config=getattr(self, "config", None),
        )

    def refresh_runtime_components_from_config(self) -> None:
        previous_failover = self.model_router.failover_status() if hasattr(self, "model_router") else {}
        self.config = self._load_config()
        self.model_router = ModelRouter.from_config(self.config["models"])
        if previous_failover.get("active"):
            self.model_router.failover.update(
                {
                    "active": True,
                    "reason": str(previous_failover.get("reason") or ""),
                    "activated_at": str(previous_failover.get("activated_at") or ""),
                }
            )
        model_gateway_cfg = self.config["models"].get("models")
        self.model_gateway = ModelGateway.from_config(model_gateway_cfg) if isinstance(model_gateway_cfg, dict) else None
        self.identity_runtime = IdentityRuntime(self._identity_cfg(), self._provider_descriptor)
        self.authenticity_policy = AuthenticityPolicy(self._identity_cfg())
        self.initiative_runtime = self.initiative_runtime.__class__(self.config["models"].get("initiative_trigger", {}))
        self.monologue_runtime = self.monologue_runtime.__class__(
            self.runtime_dir / "monologue_fragments.jsonl",
            self.config["models"].get("monologue_stream", {}),
        )

    def _normalize_temperament_config(self, payload: dict[str, Any]) -> dict[str, Any]:
        temperament = dict(payload.get("temperament", {}))
        if "attachment_need" not in temperament and "attachment" in temperament:
            temperament["attachment_need"] = temperament["attachment"]
        if "boundary_softness" not in temperament:
            if "boundary" in temperament:
                temperament["boundary_softness"] = round(1.0 - float(temperament["boundary"]), 4)
            else:
                temperament["boundary_softness"] = 0.5
        if "desire_priority" not in temperament:
            temperament["desire_priority"] = temperament.get("attachment_need", 0.5)
        normalized = {}
        for key in ("attachment_need", "boundary_softness", "sensitivity", "patience", "extraversion", "cognitive_bandwidth", "desire_priority"):
            normalized[key] = round(_clip(float(temperament.get(key, 0.5))), 4)
        return {**payload, "temperament": normalized}

    def _write_state_snapshot_payload(self, payload: dict[str, Any]) -> None:
        rewrite_snapshot(
            self.state_parquet_path,
            [{"payload_json": json.dumps(payload, ensure_ascii=False, sort_keys=True)}],
            schema={"payload_json": "VARCHAR"},
        )
        self.state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _save_state(self, state: RuntimeState, *, sync: bool = False) -> None:
        self._apply_subjective_baseline_floor(state)
        self._sync_plan16_state(state)
        state.runtime_revision = max(0, int(state.runtime_revision or 0)) + 1
        state.last_mutation_at = utc_now_iso()
        payload = to_dict(state)
        snapshot = RuntimeState(**payload)
        self._ensure_subject_core(snapshot)
        self._state_cache = snapshot
        self._state_io.submit(lambda payload=payload: self._write_state_snapshot_payload(payload))
        if sync:
            self._state_io.flush(raise_on_error=True)

    def _breaker_environment_fingerprint(self) -> str:
        payload = {
            "guard_version": "2026-04-03-model-recovery-v1",
            "models": self.config.get("models", {}),
            "has_ark_api_key": bool(os.getenv("ARK_API_KEY")),
        }
        return hashlib.sha1(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()

    def load_runtime_state(self) -> RuntimeState:
        if self._state_cache is None:
            rows = read_snapshot_rows(self.state_parquet_path, "select payload_json from read_parquet(?)")
            if rows:
                payload = json.loads(rows[0]["payload_json"])
            else:
                payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            self._state_cache = RuntimeState(**payload)
            self._ensure_subject_core(self._state_cache)
            self._ensure_default_autonomy_runtime(self._state_cache)
            self._apply_subjective_baseline_floor(self._state_cache)
        return RuntimeState(**to_dict(self._state_cache))

    def _run_stop_reason_code(self, run_state: RunState | dict[str, Any] | None) -> str:
        return self.run_runtime._run_stop_reason_code(run_state)

    def _run_has_pending_approval(self, run_id: str) -> bool:
        return self.run_runtime._run_has_pending_approval(run_id)

    def _active_run_occupancy(self, state: RuntimeState) -> dict[str, Any]:
        return self.run_runtime._active_run_occupancy(state)

    def runtime_status_truth_payload(self, state: RuntimeState | None = None) -> dict[str, Any]:
        return self.run_runtime.runtime_status_truth_payload(state)

    def fault_guard_status(self, state: RuntimeState | None = None) -> dict[str, Any]:
        return self.run_runtime.fault_guard_status(state)

    def _runtime_migration_status(self) -> dict[str, Any]:
        if not self.runtime_migration_status_path.exists():
            return {}
        try:
            payload = json.loads(self.runtime_migration_status_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}

    def _estimate_subject_birth_ts(self) -> str:
        candidates: list[str] = []
        if self.state_path.exists():
            candidates.append(utc_now_iso())
        try:
            rounds = self.trace_store.list_rounds()
            candidates.extend(str(item.get("recorded_at") or "").strip() for item in rounds if item.get("recorded_at"))
        except Exception:
            pass
        candidates = [item for item in candidates if item]
        return min(candidates) if candidates else utc_now_iso()

    def _default_vitality_anchor(self, state: RuntimeState) -> dict[str, Any]:
        return {
            "mood": round(float(state.mood), 4),
            "body_energy": round(float(state.body_energy), 4),
            "affect_residue": round(float(state.affect_residue), 4),
            "budget_remaining": round(float(state.budget_remaining), 4),
        }

    def _ensure_subject_core(self, state: RuntimeState) -> SubjectCore:
        core = state.subject_core if isinstance(state.subject_core, SubjectCore) else SubjectCore(**to_dict(state.subject_core))
        if not core.subject_id:
            core.subject_id = f"subject-{uuid4().hex}"
        if not core.birth_ts:
            core.birth_ts = self._estimate_subject_birth_ts()
        if not core.continuity_nonce:
            core.continuity_nonce = f"continuity-{uuid4().hex}"
        if not core.original_vitality_anchor:
            core.original_vitality_anchor = self._default_vitality_anchor(state)
        core.core_boundary_version = max(int(core.core_boundary_version or 0), 1)
        state.subject_core = core
        return core

    def _continuity_summary(self, continuity_score: float) -> str:
        if continuity_score >= 0.75:
            return "主体连续性稳定，当前自我叙述保持连贯。"
        if continuity_score >= 0.5:
            return "主体连续性可维持，但仍需要持续锚定当前身份。"
        return "主体连续性偏弱，需要重新收束当前身份与意义线索。"

    def _purpose_memory_focus_label(self, summary: str) -> str:
        text = str(summary or "").strip()
        for prefix in ("当前目标：", "主导驱力：", "意义债务：", "关系承诺：", "计划任务："):
            if text.startswith(prefix):
                return text[len(prefix) :].strip()
        return text

    def _purpose_memory_focus_entry(self, state: RuntimeState) -> PurposeMemoryEntry | None:
        entries = [
            item if isinstance(item, PurposeMemoryEntry) else PurposeMemoryEntry(**item)
            for item in list(state.purpose_memory or [])
        ]
        status_rank = {"active": 0, "repair": 1, "background": 2, "archived": 3}
        ordered = sorted(
            (entry for entry in entries if entry.summary),
            key=lambda entry: (
                status_rank.get(str(entry.status or "").strip(), 4),
                -float(entry.strength or 0.0),
                str(entry.recorded_at or ""),
            ),
        )
        return ordered[0] if ordered else None

    def _current_subject_narrative(self, state: RuntimeState, display_name: str) -> str:
        goal = str(state.current_goal or "").strip()
        focus = str(state.focus or "").strip() or "整理当前状态"
        if goal:
            return f"{display_name} 正在围绕“{goal}”维持自我延续，并把注意力收束在 {focus}。"
        purpose_entry = self._purpose_memory_focus_entry(state)
        purpose_label = self._purpose_memory_focus_label(purpose_entry.summary) if purpose_entry is not None else ""
        if purpose_label:
            return f"{display_name} 正在围绕“{purpose_label}”维持自我延续，并把注意力收束在 {focus}。"
        if state.meaning_strength >= 0.55:
            return f"{display_name} 正在维持一条可解释的存在叙事，当前更偏向 {focus}。"
        return f"{display_name} 仍在通过 {focus} 重新确认自己为何继续存在。"

    def _build_subject_kernel(self, state: RuntimeState) -> SubjectKernelState:
        identity = state.identity_state
        display_name = str(identity.display_name or identity.internal_handle or self._unnamed_label()).strip()
        boundary = float(state.subjective_state.boundary or 0.0)
        continuity_score = round(float(state.self_continuity or 0.0), 4)
        purpose_entry = self._purpose_memory_focus_entry(state)
        purpose_label = self._purpose_memory_focus_label(purpose_entry.summary) if purpose_entry is not None else ""
        boundary_principles = [
            "优先维持主体连续性，不让短期成功直接改写核心身份。",
            "外部反馈只能进入二级评估，不能绕过主体内核。",
            "保留拒绝与延后响应的边界权。",
        ]
        if boundary >= 0.65:
            boundary_principles.append("当前边界偏紧，优先压低高社交风险与高暴露动作。")
        else:
            boundary_principles.append("当前边界允许有限接近，但仍保留主体自我保护。")
        core_commitments = [
            "持续维护可解释的自我叙述。",
            "把意义感不足和连续性下降视为需要修复的运行事实。",
        ]
        if state.current_goal:
            core_commitments.insert(0, f"继续围绕“{state.current_goal}”维持当前轨迹。")
        elif purpose_label:
            core_commitments.insert(0, f"继续围绕“{purpose_label}”维持当前轨迹。")
        return SubjectKernelState(
            display_name=display_name,
            current_narrative=self._current_subject_narrative(state, display_name),
            continuity_score=continuity_score,
            continuity_summary=self._continuity_summary(continuity_score),
            boundary_principles=boundary_principles,
            core_commitments=core_commitments,
            integrity_source="subject_core",
        )

    def _build_meaning_system(
        self,
        state: RuntimeState,
        *,
        initiative_status: dict[str, Any],
    ) -> MeaningSystemState:
        sources: list[MeaningSource] = []
        seen_sources: set[tuple[str, str]] = set()

        def _append_source(source: MeaningSource) -> None:
            key = (str(source.kind or "").strip(), str(source.label or "").strip())
            if not key[1] or key in seen_sources:
                return
            seen_sources.add(key)
            sources.append(source)

        if state.current_goal:
            _append_source(
                MeaningSource(
                    label=state.current_goal,
                    kind="goal",
                    strength=max(0.55, float(state.meaning_strength or 0.0)),
                    source="current_goal",
                    evidence="当前运行目标仍在前台。",
                )
            )
        for item in list(state.subjective_state.meaning_made or [])[:3]:
            _append_source(
                MeaningSource(
                    label=item,
                    kind="subjective_trace",
                    strength=min(1.0, 0.42 + float(state.meaning_strength or 0.0) * 0.35),
                    source="subjective_state.meaning_made",
                    evidence="主观意义片段仍在持续影响当前行为。",
                )
            )
        for entry in list(state.purpose_memory or [])[:3]:
            resolved_entry = entry if isinstance(entry, PurposeMemoryEntry) else PurposeMemoryEntry(**entry)
            label = self._purpose_memory_focus_label(resolved_entry.summary)
            if not label:
                continue
            _append_source(
                MeaningSource(
                    label=label,
                    kind="purpose_memory",
                    strength=max(0.35, float(resolved_entry.strength or 0.0)),
                    source="purpose_memory",
                    evidence="持续保留的目的记忆仍在塑形当前判断。",
                )
            )
        memory_backing = dict(initiative_status.get("memory_backing", {}) or {})
        if memory_backing.get("summary"):
            _append_source(
                MeaningSource(
                    label=str(memory_backing.get("summary") or ""),
                    kind="memory_pull",
                    strength=float(memory_backing.get("strength", 0.0) or 0.0),
                    source="initiative.memory_backing",
                    evidence="最近的主动性评估仍被这条线索牵引。",
                )
            )

        conflicts: list[MeaningConflict] = []
        if float(state.subjective_state.spontaneous or 0.0) >= 0.45 and float(state.subjective_state.reject_all or 0.0) >= 0.35:
            conflicts.append(
                MeaningConflict(
                    label="想靠近又想收回",
                    tension=max(state.subjective_state.spontaneous, state.subjective_state.reject_all),
                    summary="主动表达的冲动与收缩倾向同时存在，主体需要决定如何行动。",
                )
            )
        if state.current_goal and float(state.fatigue or 0.0) >= 0.55:
            conflicts.append(
                MeaningConflict(
                    label="目标推进与疲劳恢复冲突",
                    tension=max(float(state.fatigue or 0.0), float(state.meaning_strength or 0.0)),
                    summary="继续推进目标与先恢复状态之间存在直接拉扯。",
                )
            )

        debts: list[str] = []
        repairs: list[MeaningRepair] = []
        if float(state.meaning_strength or 0.0) < 0.45:
            debts.append("意义感偏弱，需要重新生成当前为何继续的叙事。")
            repairs.append(MeaningRepair(label="意义修复", status="pending", summary="优先从当前目标、记忆牵引与关系线索中重建存在理由。"))
        if float(state.self_continuity or 0.0) < 0.55:
            debts.append("自我连续性下降，需要重新锚定主体身份。")
            repairs.append(MeaningRepair(label="连续性锚定", status="pending", summary="用主体核与最近行动重新收束当前自我叙述。"))
        if float(state.memory_fragments or 0.0) > 0.55:
            debts.append("记忆碎片偏高，需要整理最近经历。")
            repairs.append(MeaningRepair(label="经历整理", status="planned", summary="通过内部独白与回放把分散线索压回可解释结构。"))

        if float(state.meaning_strength or 0.0) >= 0.6:
            survival_narrative = "当前存在叙事相对稳定，主体仍能解释自己为什么继续。"
        elif state.current_goal:
            survival_narrative = f"主体目前主要依靠“{state.current_goal}”维持继续存在的方向感。"
        else:
            survival_narrative = "主体正在从最近的记忆牵引、边界感和主动性冲动里重新寻找继续存在的理由。"

        return MeaningSystemState(
            survival_narrative=survival_narrative,
            sources=sources,
            conflicts=conflicts,
            debts=debts,
            repairs=repairs,
        )

    def _build_purpose_memory(self, state: RuntimeState, meaning_system: MeaningSystemState) -> list[PurposeMemoryEntry]:
        entries: list[PurposeMemoryEntry] = []
        now_iso = utc_now_iso()
        existing_entries = [
            item if isinstance(item, PurposeMemoryEntry) else PurposeMemoryEntry(**item)
            for item in list(state.purpose_memory or [])
        ]
        existing_by_key = {
            (str(entry.source or "").strip(), str(entry.summary or "").strip()): entry
            for entry in existing_entries
            if entry.summary
        }
        if state.current_goal:
            entries.append(
                PurposeMemoryEntry(
                    summary=f"当前目标：{state.current_goal}",
                    source="runtime.current_goal",
                    strength=max(0.45, float(state.meaning_strength or 0.0)),
                    recorded_at=now_iso,
                    status="active",
                )
            )
        if state.desire_state.dominant_drive:
            entries.append(
                PurposeMemoryEntry(
                    summary=f"主导驱力：{state.desire_state.dominant_drive}",
                    source="desire_state",
                    strength=float(state.desire_state.drive_tension or 0.0),
                    recorded_at=now_iso,
                    status="background",
                )
            )
        if meaning_system.debts:
            entries.append(
                PurposeMemoryEntry(
                    summary=meaning_system.debts[0],
                    source="meaning_system",
                    strength=max(0.35, 1.0 - float(state.meaning_strength or 0.0)),
                    recorded_at=now_iso,
                    status="repair",
                )
            )
        merged: list[PurposeMemoryEntry] = []
        seen_keys: set[tuple[str, str]] = set()
        for entry in entries:
            key = (str(entry.source or "").strip(), str(entry.summary or "").strip())
            previous = existing_by_key.get(key)
            merged.append(
                PurposeMemoryEntry(
                    summary=entry.summary,
                    source=entry.source,
                    strength=max(float(entry.strength or 0.0), float(previous.strength or 0.0) if previous else 0.0),
                    recorded_at=str(previous.recorded_at or entry.recorded_at or now_iso) if previous else str(entry.recorded_at or now_iso),
                    status=entry.status,
                )
            )
            seen_keys.add(key)

        for entry in existing_entries:
            key = (str(entry.source or "").strip(), str(entry.summary or "").strip())
            if not entry.summary or key in seen_keys:
                continue
            carry_status = str(entry.status or "background").strip() or "background"
            if carry_status == "active" and str(entry.source or "").strip() != "runtime.current_goal":
                carry_status = "background"
            merged.append(
                PurposeMemoryEntry(
                    summary=entry.summary,
                    source=entry.source,
                    strength=max(0.2, float(entry.strength or 0.0)),
                    recorded_at=str(entry.recorded_at or now_iso),
                    status=carry_status,
                )
            )

        status_rank = {"active": 0, "repair": 1, "background": 2, "archived": 3}
        merged.sort(
            key=lambda entry: (
                status_rank.get(str(entry.status or "").strip(), 4),
                -float(entry.strength or 0.0),
                str(entry.recorded_at or ""),
            )
        )
        return merged[:6]

    def _build_relationship_commitments(
        self,
        state: RuntimeState,
        *,
        initiative_status: dict[str, Any],
    ) -> list[RelationshipCommitmentState]:
        relation = dict(self.memory_store.relation_state("user"))
        closeness = float(relation.get("closeness", 0.0) or 0.0)
        top_intent = str(initiative_status.get("top_intent") or "")
        if closeness < 0.45 and top_intent not in {"check_relation", "share_memory"}:
            return []
        if float(state.subjective_state.boundary or 0.0) >= 0.7:
            commitment = "维持联系，但不让关系压力绕过当前边界。"
        else:
            commitment = "维持关系脉冲，避免在重要关系上完全沉默。"
        return [
            RelationshipCommitmentState(
                target="user",
                commitment=commitment,
                strength=max(closeness, float(state.subjective_state.spontaneous or 0.0)),
                status="active",
                evidence=f"closeness={closeness:.2f}, top_intent={top_intent or 'none'}",
            )
        ]

    def _build_agency_state(
        self,
        state: RuntimeState,
        *,
        initiative_status: dict[str, Any],
        relationship_commitments: list[RelationshipCommitmentState],
    ) -> tuple[AgencyLoopState, list[ProactiveActionState]]:
        now_iso = utc_now_iso()
        monologue_bucket = self._monologue_state_bucket(state)
        monologue_status = self.monologue_runtime.status_payload(monologue_bucket, now_iso=now_iso)
        scheduled_tasks = self.scheduled_task_store.list_tasks()
        due_tasks = self.scheduled_task_store.due_tasks(reference_at=now_iso)
        running_tasks = [task for task in scheduled_tasks if task.status == "running"]
        backlog: list[ProactiveActionState] = []
        suppressed_actions: list[ProactiveActionState] = []
        for task in due_tasks[:3]:
            summary = str(task.prompt or task.task_payload.get("prompt") or task.task_payload.get("goal") or task.skill_name).strip()
            backlog.append(
                ProactiveActionState(
                    kind="scheduled_task",
                    summary=summary or f"执行计划任务：{task.task_id}",
                    priority=0.88,
                    channel="tooling",
                    status="due",
                )
            )
        for task in running_tasks[:2]:
            summary = str(task.prompt or task.task_payload.get("prompt") or task.task_payload.get("goal") or task.skill_name).strip()
            backlog.append(
                ProactiveActionState(
                    kind="scheduled_task",
                    summary=f"计划任务执行中：{summary or task.task_id}",
                    priority=0.62,
                    channel="tooling",
                    status="running",
                )
            )
        if monologue_status.get("stale") or monologue_status.get("next_pulse_due"):
            backlog.append(
                ProactiveActionState(
                    kind="self_dialogue",
                    summary="继续内部独白并整理刚才残留的线索。",
                    priority=max(0.45, float(state.memory_fragments or 0.0)),
                    channel="internal",
                    status="pending",
                )
            )
        if state.current_goal:
            backlog.append(
                ProactiveActionState(
                    kind="plan_update",
                    summary=f"围绕“{state.current_goal}”更新当前计划与推进顺序。",
                    priority=max(0.5, float(state.meaning_strength or 0.0)),
                    channel="planning",
                    status="pending",
                )
            )
        if relationship_commitments:
            backlog.append(
                ProactiveActionState(
                    kind="relationship_maintenance",
                    summary=relationship_commitments[0].commitment,
                    priority=float(relationship_commitments[0].strength or 0.0),
                    channel="outward",
                    status="queued",
                )
            )
        if float(state.meaning_strength or 0.0) < 0.45:
            backlog.append(
                ProactiveActionState(
                    kind="meaning_repair",
                    summary="先修复意义感，再决定是否继续外显行动。",
                    priority=max(0.5, 1.0 - float(state.meaning_strength or 0.0)),
                    channel="internal",
                    status="pending",
                )
            )

        suppression_reason = str(initiative_status.get("suppression_reason") or "")
        if suppression_reason:
            suppressed_actions.append(
                ProactiveActionState(
                    kind=str(initiative_status.get("top_intent") or initiative_status.get("proposal", {}).get("proposal_type") or "initiative"),
                    summary="当前存在对外主动动作，但被运行时边界抑制。",
                    priority=max(0.35, float(initiative_status.get("speech_cost", 0.0) or 0.0)),
                    channel="outward",
                    status="suppressed",
                    suppression_reason=suppression_reason,
                )
            )
        if due_tasks and (not state.autonomy_policy.enabled or not state.autonomy_loop.running):
            suppressed_actions.append(
                ProactiveActionState(
                    kind="scheduled_task",
                    summary="至少一条已到期的计划任务仍在等待自治恢复。",
                    priority=0.78,
                    channel="tooling",
                    status="suppressed",
                    suppression_reason="autonomy_paused",
                )
            )

        history_stats = dict(initiative_status.get("history_stats", {}) or {})
        hourly_limit = int(history_stats.get("hourly_limit", 0) or 0)
        sent_last_hour = int(history_stats.get("sent_last_hour", 0) or 0)
        outward_remaining = max(hourly_limit - sent_last_hour, 0) if hourly_limit > 0 else 0
        budget = {
            "outward_dispatch_remaining": outward_remaining,
            "cooldown_remaining_seconds": int(history_stats.get("cooldown_remaining_seconds", 0) or 0),
            "autonomy_tool_actions": int(state.autonomy_loop.window_tool_actions or 0),
            "autonomy_endogenous_rounds": int(state.autonomy_loop.window_endogenous_rounds or 0),
            "scheduled_due": len(due_tasks),
            "scheduled_running": len(running_tasks),
            "scheduled_total": len(scheduled_tasks),
            "backlog_size": len(backlog),
        }
        summary = "主动性由对外发言、内部独白、目标整理和关系维护共同组成。"
        if due_tasks:
            summary = "当前主动性已经形成闭环：到期计划任务会进入 backlog，并等待自治执行。"
        elif suppressed_actions:
            summary = "当前主动性仍在运转，但至少一条对外动作被运行时边界抑制。"
        elif backlog:
            summary = "当前主动性主要体现在 backlog 中，尚未全部转成外显动作。"
        return (
            AgencyLoopState(
                summary=summary,
                outward_channel={
                    "top_intent": initiative_status.get("top_intent"),
                    "should_send": bool(initiative_status.get("should_send")),
                    "suppression_reason": suppression_reason,
                },
                internal_channel={
                    "monologue_due": bool(monologue_status.get("next_pulse_due")),
                    "monologue_stale": bool(monologue_status.get("stale")),
                    "monologue_generated_total": int(monologue_status.get("generated_total", 0) or 0),
                },
                budget=budget,
                suppressed_actions=suppressed_actions,
            ),
            backlog,
        )

    def _latest_subjectivity_trace(self) -> dict[str, Any] | None:
        try:
            rounds = self.trace_store.recent_rounds(limit=1)
        except Exception:
            return None
        if not rounds:
            return None
        return dict(rounds[-1])

    def _build_history_burden(self, state: RuntimeState) -> dict[str, Any]:
        repair_scars = [
            {
                "round_id": int(item.round_id or 0),
                "reason": str(item.reason or ""),
                "winning_priority": str(item.winning_priority or ""),
                "template": str(item.template or ""),
                "repair_stage_after": str(item.repair_stage_after or ""),
                "blocked_actions": list(item.blocked_actions or []),
                "conflict_score": round(float(item.conflict_score or 0.0), 4),
            }
            for item in list(state.repair_ledger or [])[-6:]
        ]
        suppressed_patterns = [
            {
                "pattern": str(item.get("pattern") or ""),
                "suppressed_by": str(item.get("suppressed_by") or ""),
                "status": str(item.get("status") or ""),
                "suppressed_at_round": item.get("suppressed_at_round"),
                "strength": round(float(item.get("strength", 0.0) or 0.0), 4),
            }
            for item in self.memory_store.habit_top(limit=12)
            if str(item.get("status") or "") == "suppressed_recoverable"
        ]
        unfinished_commitments = [
            {
                "kind": str(item.get("kind") or ""),
                "summary": str(item.get("summary") or ""),
                "status": str(item.get("status") or ""),
                "priority": round(float(item.get("priority", 0.0) or 0.0), 4),
            }
            for item in to_dict(state.proactive_backlog)
            if str(item.get("status") or "") not in {"done", "archived"}
        ]
        unfinished_commitments.extend(
            {
                "kind": "relationship_commitment",
                "summary": str(item.get("commitment") or ""),
                "status": str(item.get("status") or ""),
                "priority": round(float(item.get("strength", 0.0) or 0.0), 4),
            }
            for item in to_dict(state.relationship_commitments)
            if str(item.get("status") or "") == "active"
        )
        meaning_debts = [
            {
                "summary": str(item),
                "status": "open",
            }
            for item in list(state.meaning_system.debts or [])
            if str(item).strip()
        ]
        continuity_residues = {
            "continuity_nonce": state.subject_core.continuity_nonce,
            "self_continuity": round(float(state.self_continuity or 0.0), 4),
            "meaning_strength": round(float(state.meaning_strength or 0.0), 4),
            "memory_fragments": round(float(state.memory_fragments or 0.0), 4),
            "last_checkpoint_id": state.last_checkpoint_id,
            "active_run_id": state.active_run_id,
            "run_status": str(state.run_status or "idle"),
        }
        conflict_residues = {
            "repair_mode": str(state.repair_mode or ""),
            "safe_mode_owner": str(state.conflict_safe_mode_owner or ""),
            "critical_conflict_streak": int(state.critical_conflict_streak or 0),
            "conflict_hot_rounds": int(state.conflict_hot_rounds or 0),
            "conflict_recovery_rounds": int(state.conflict_recovery_rounds or 0),
            "last_conflict_priority": str(state.last_conflict_priority or ""),
            "last_compromise_template": str(state.last_compromise_template or ""),
            "repair_stage": str(state.repair_state.stage or ""),
        }
        inertia_score = round(
            min(
                1.0,
                len(repair_scars) * 0.16
                + len(suppressed_patterns) * 0.10
                + len(unfinished_commitments) * 0.06
                + len(meaning_debts) * 0.08
                + max(0.0, 1.0 - float(state.self_continuity or 0.0)) * 0.35,
            ),
            4,
        )
        return {
            "repair_scars": repair_scars,
            "suppressed_but_recoverable_patterns": suppressed_patterns,
            "unfinished_commitments": unfinished_commitments[:8],
            "meaning_debts": meaning_debts[:6],
            "conflict_residues": conflict_residues,
            "continuity_residues": continuity_residues,
            "history_inertia_score": inertia_score,
        }

    def _history_burden_delta_payload(
        self,
        before: dict[str, Any] | None,
        after: dict[str, Any] | None,
    ) -> dict[str, Any]:
        previous = dict(before or {})
        current = dict(after or {})
        previous_continuity = dict(previous.get("continuity_residues", {}) or {})
        current_continuity = dict(current.get("continuity_residues", {}) or {})
        return {
            "repair_scars_added": max(
                0,
                len(list(current.get("repair_scars", []) or [])) - len(list(previous.get("repair_scars", []) or [])),
            ),
            "suppressed_patterns_delta": len(list(current.get("suppressed_but_recoverable_patterns", []) or []))
            - len(list(previous.get("suppressed_but_recoverable_patterns", []) or [])),
            "unfinished_commitments_delta": len(list(current.get("unfinished_commitments", []) or []))
            - len(list(previous.get("unfinished_commitments", []) or [])),
            "meaning_debts_delta": len(list(current.get("meaning_debts", []) or []))
            - len(list(previous.get("meaning_debts", []) or [])),
            "continuity_shift": round(
                float(current_continuity.get("self_continuity", 0.0) or 0.0)
                - float(previous_continuity.get("self_continuity", 0.0) or 0.0),
                4,
            ),
            "history_inertia_score": round(float(current.get("history_inertia_score", 0.0) or 0.0), 4),
        }

    def _build_autonomy_arbitration_state(self, state: RuntimeState) -> dict[str, Any]:
        now_iso = utc_now_iso()
        due_tasks = self.scheduled_task_store.due_tasks(reference_at=now_iso)
        backlog = [dict(item) for item in to_dict(state.proactive_backlog)]
        suppressed = [dict(item) for item in to_dict(state.agency_loop.suppressed_actions)]
        competing_proposals = [
            {
                "proposal_id": f"{str(item.get('kind') or 'proposal')}:{index}",
                "kind": str(item.get("kind") or ""),
                "summary": str(item.get("summary") or ""),
                "channel": str(item.get("channel") or ""),
                "status": str(item.get("status") or ""),
            }
            for index, item in enumerate(backlog[:6], start=1)
        ]
        if not competing_proposals and state.current_goal:
            competing_proposals.append(
                {
                    "proposal_id": "goal:1",
                    "kind": "goal",
                    "summary": str(state.current_goal),
                    "channel": "planning",
                    "status": "active",
                }
            )
        priority_scores = {
            str(item.get("proposal_id") or f"proposal:{index}"): round(float(backlog_item.get("priority", 0.0) or 0.0), 4)
            for index, (item, backlog_item) in enumerate(zip(competing_proposals, backlog[: len(competing_proposals)]), start=1)
        }
        hard_masks: list[str] = []
        if state.safe_mode:
            hard_masks.append("safe_mode_active")
        if float(state.budget_remaining or 0.0) <= 0.0:
            hard_masks.append("budget_exhausted")
        if state.active_run_id and str(state.run_status or "") in {"running", "paused"}:
            hard_masks.append("run_occupancy")
        selected_winner = (
            str(state.autonomy_loop.last_action_type or "").strip()
            or (competing_proposals[0]["proposal_id"] if competing_proposals else "idle")
        )
        return {
            "competing_proposals": competing_proposals,
            "evidence_sources": ["proactive_backlog", "scheduled_tasks", "agency_loop", "initiative"],
            "priority_scores": priority_scores,
            "budget_consumption": {
                "budget_remaining": round(float(state.budget_remaining or 0.0), 4),
                "window_tool_actions": int(state.autonomy_loop.window_tool_actions or 0),
                "window_endogenous_rounds": int(state.autonomy_loop.window_endogenous_rounds or 0),
                "scheduled_due": len(due_tasks),
            },
            "selected_winner": selected_winner,
            "rejected_options": [
                {
                    "kind": str(item.get("kind") or ""),
                    "summary": str(item.get("summary") or ""),
                    "reason": str(item.get("suppression_reason") or item.get("status") or ""),
                }
                for item in suppressed[:6]
            ],
            "hard_masks": hard_masks,
            "consequence_patch": {
                "safe_mode_owner": str(state.conflict_safe_mode_owner or ""),
                "budget_remaining": round(float(state.budget_remaining or 0.0), 4),
                "scheduled_task_residue": len(due_tasks),
                "run_status": str(state.run_status or "idle"),
                "last_action_type": str(state.autonomy_loop.last_action_type or ""),
                "last_round_id": int(state.autonomy_loop.last_round_id or 0),
            },
        }

    def _build_arbitration_state(
        self,
        state: RuntimeState,
        *,
        round_arbitration: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        resolved_round = dict(round_arbitration or {})
        if not resolved_round:
            latest_trace = self._latest_subjectivity_trace() or {}
            resolved_round = dict(latest_trace.get("arbitration_record", {}) or latest_trace.get("conflict_arbitration", {}) or {})
        return {
            "round": resolved_round,
            "autonomy": self._build_autonomy_arbitration_state(state),
        }

    def state_truth_payload(
        self,
        state: RuntimeState | None = None,
        *,
        trace: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        current = state or self.load_runtime_state()
        resolved_trace = dict(trace or self._latest_subjectivity_trace() or {})
        try:
            session_projection_status = self.terminal_sessions.projection_only_status()
        except Exception as exc:
            session_projection_status = {
                "checked_sessions": 0,
                "projection_only": False,
                "violations": [f"terminal_sessions:{type(exc).__name__}"],
                "error": str(exc),
            }
        event_log_ref = dict(resolved_trace.get("event_log_ref", {}) or {})
        round_trace_ref = str(
            event_log_ref.get("round_trace_ref")
            or (f"round://{int(resolved_trace.get('round_id', 0) or 0)}" if int(resolved_trace.get("round_id", 0) or 0) > 0 else "")
        )
        return {
            "event_log": {
                "round_trace_ref": round_trace_ref,
                "round_trace_jsonl": str(self.trace_store.rounds_jsonl_path),
                "command_trace_jsonl": str(self.trace_store.commands_jsonl_path),
                "memory_events_jsonl": str(self.memory_store.raw_events_path),
                "append_only": True,
            },
            "authoritative_state": {
                "runtime_revision": int(current.runtime_revision or 0),
                "session_id": current.session_id,
                "last_mutation_at": str(current.last_mutation_at or ""),
                "runtime_state_path": str(self.state_parquet_path),
                "memory_state_root": str(self.memory_store.current_dir),
                "scheduled_task_state_path": str(self.scheduled_task_store.tasks_path),
            },
            "snapshot": {
                "last_checkpoint_id": current.last_checkpoint_id,
                "checkpoint_dir": str(self.checkpoint_dir),
                "snapshot_dir": str(self.snapshot_dir),
                "rewind_supported": True,
            },
            "projection": {
                "terminal_view": "derived",
                "observer_status": "derived",
                "workbench_read_model": "derived",
                "round_detail": "derived",
                "terminal_session_cache": "projection_only",
            },
            "consistency_checks": {
                "subject_projection_from_state": bool(current.subject_kernel.display_name),
                "meaning_projection_from_state": bool(current.meaning_system.survival_narrative),
                "agency_projection_from_state": bool(current.agency_loop.summary),
                "projection_only_truth_fields": list(session_projection_status.get("violations", []) or []),
                "session_cache_is_projection_only": bool(session_projection_status.get("projection_only", False)),
                "checked_session_caches": int(session_projection_status.get("checked_sessions", 0) or 0),
            },
        }

    def memory_evidence_payload(self, trace: dict[str, Any] | None = None) -> dict[str, Any] | None:
        resolved_trace = dict(trace or self._latest_subjectivity_trace() or {})
        if not resolved_trace:
            return None
        payload = dict(resolved_trace.get("memory_evidence", {}) or {})
        if not payload:
            return None
        payload.setdefault("event_log_ref", dict(resolved_trace.get("event_log_ref", {}) or {}))
        return payload

    def competition_evidence_payload(self, trace: dict[str, Any] | None = None) -> dict[str, Any] | None:
        resolved_trace = dict(trace or self._latest_subjectivity_trace() or {})
        if resolved_trace:
            payload = dict(resolved_trace.get("arbitration_record", {}) or {})
            if payload:
                return payload
        state = self.load_runtime_state()
        self._sync_plan16_state(state)
        return dict(state.arbitration_state.get("round", {}) or {})

    def continuity_evidence_payload(
        self,
        trace: dict[str, Any] | None = None,
        *,
        state: RuntimeState | None = None,
    ) -> dict[str, Any] | None:
        current = state or self.load_runtime_state()
        resolved_trace = dict(trace or self._latest_subjectivity_trace() or {})
        payload = dict(resolved_trace.get("snapshot_continuity", {}) or {})
        if not payload:
            payload = {
                "continuity_nonce": current.subject_core.continuity_nonce,
                "runtime_revision": int(current.runtime_revision or 0),
                "last_checkpoint_id": current.last_checkpoint_id,
            }
        payload.setdefault("continuity_nonce", current.subject_core.continuity_nonce)
        if resolved_trace.get("history_burden_delta") and "history_burden_delta" not in payload:
            payload["history_burden_delta"] = dict(resolved_trace.get("history_burden_delta", {}) or {})
        return payload

    def _sync_plan16_state(self, state: RuntimeState) -> None:
        existing_round_arbitration = dict(dict(state.arbitration_state or {}).get("round", {}) or {})
        self._ensure_subject_core(state)
        self._sync_tlh_state(state)
        self._sync_autonomy_state(state)
        initiative_status = self._initiative_status_payload(state)
        state.subject_kernel = self._build_subject_kernel(state)
        state.meaning_system = self._build_meaning_system(state, initiative_status=initiative_status)
        state.purpose_memory = self._build_purpose_memory(state, state.meaning_system)
        state.relationship_commitments = self._build_relationship_commitments(state, initiative_status=initiative_status)
        agency_loop, backlog = self._build_agency_state(
            state,
            initiative_status=initiative_status,
            relationship_commitments=state.relationship_commitments,
        )
        state.agency_loop = agency_loop
        state.proactive_backlog = backlog[:6]
        state.history_burden = self._build_history_burden(state)
        state.arbitration_state = self._build_arbitration_state(state, round_arbitration=existing_round_arbitration)

    def sync_plan16_state(self, state: RuntimeState) -> None:
        self._sync_plan16_state(state)

    def subject_status(self) -> dict[str, Any]:
        state = self.load_runtime_state()
        self._sync_plan16_state(state)
        return {
            "subject_kernel": to_dict(state.subject_kernel),
            "subject_core_integrity": True,
        }

    def meaning_status(self) -> dict[str, Any]:
        state = self.load_runtime_state()
        self._sync_plan16_state(state)
        return {
            "meaning_system": to_dict(state.meaning_system),
            "purpose_memory": to_dict(state.purpose_memory),
            "relationship_commitments": to_dict(state.relationship_commitments),
        }

    def agency_status(self) -> dict[str, Any]:
        state = self.load_runtime_state()
        self._sync_plan16_state(state)
        return {
            "agency_loop": to_dict(state.agency_loop),
            "proactive_backlog": to_dict(state.proactive_backlog),
            "initiative": self._initiative_status_payload(state),
            "monologue": self.monologue_runtime.status_payload(self._monologue_state_bucket(state), now_iso=utc_now_iso()),
            "scheduled_tasks": to_dict(self.scheduled_task_store.list_tasks()),
        }

    def list_scheduled_tasks(self) -> dict[str, Any]:
        return {"tasks": to_dict(self.scheduled_task_store.list_tasks())}

    def list_scheduled_task_runs(self, task_id: str | None = None) -> dict[str, Any]:
        return {"runs": to_dict(self.scheduled_task_store.list_runs(task_id=task_id))}

    def upsert_scheduled_task(self, payload: dict[str, Any]) -> dict[str, Any]:
        recorded_at = utc_now_iso()
        schedule_payload = dict(payload.get("schedule", {}) or {})
        if "schedule_type" not in schedule_payload:
            schedule_payload["schedule_type"] = "disabled"
        spec = ScheduledTaskSpec(
            task_id=str(payload.get("task_id") or f"task-{uuid4().hex[:12]}").strip(),
            skill_name=str(payload.get("skill_name") or "scheduled_self_run").strip() or "scheduled_self_run",
            prompt=str(payload.get("prompt") or "").strip(),
            task_payload=dict(payload.get("task_payload", {}) or {}),
            toolset_policy=dict(payload.get("toolset_policy", {}) or {}),
            fresh_session=bool(payload.get("fresh_session", True)),
            schedule=ScheduledTaskSchedule(**schedule_payload),
        )
        saved = self.scheduled_task_store.upsert_task(spec, recorded_at=recorded_at)
        return {"task": to_dict(saved)}

    def trigger_scheduled_task(self, task_id: str) -> dict[str, Any]:
        recorded_at = utc_now_iso()
        task = self.scheduled_task_store.read_task(task_id)
        run_id = f"scheduled-{uuid4().hex[:12]}"
        scheduled_session_id = None
        if bool(task.fresh_session):
            from nalr.terminal_bridge.session import TerminalSessionState

            scheduled_session_id = f"scheduled-session-{uuid4().hex[:12]}"
            self.terminal_sessions.write(
                TerminalSessionState(
                    session_id=scheduled_session_id,
                    cwd=str(self.project_root),
                    status="scheduled",
                    mode="plan",
                    permission_mode="plan",
                ),
                mark_current=False,
            )
        running_task = self.scheduled_task_store.mark_task_running(task_id, run_id=run_id, recorded_at=recorded_at)
        goal = (
            task.prompt
            or str(task.task_payload.get("prompt") or "").strip()
            or str(task.task_payload.get("goal") or "").strip()
            or f"Inspect the repository in a read-only way for scheduled task {task.skill_name}"
        )
        toolset_policy = dict(task.toolset_policy or {})
        run_payload = self.start_run(
            goal,
            allow_commit=bool(toolset_policy.get("allow_commit", False)),
            operator_level=str(toolset_policy.get("operator_level") or "read_only"),
            include_details=True,
            defer_bootstrap_tool=True,
        )
        run_record = self.scheduled_task_store.append_run(
            ScheduledTaskRunRecord(
                run_id=run_id,
                task_id=task_id,
                status="running",
                scheduled_for=running_task.next_run_at,
            ),
            recorded_at=recorded_at,
        )
        return {
            "task": to_dict(self.scheduled_task_store.read_task(task_id)),
            "run": run_payload,
            "run_record": to_dict(run_record),
            "scheduled_session_id": scheduled_session_id,
        }

    def workbench_round_payload(self, round_ref: int | str, *, action: str = "respond") -> dict[str, Any]:
        resolved_action = str(action or "respond").strip() or "respond"
        trace = self.trace_round(round_ref)
        state = self.load_runtime_state()
        self._sync_plan16_state(state)
        return {
            "trace": trace,
            "initiativeWhy": self.initiative_why(round_ref),
            "thought": self.thought_snapshot(round_ref),
            "why": self.why_this(round_ref),
            "contributions": self.contribution_breakdown(round_ref),
            "probability": self.console_probability_space(round_ref),
            "whyNot": self.console_why_not(resolved_action, round_ref),
            "replay": self.replay(int(self.resolve_round_ref(round_ref))),
            "state_truth": self.state_truth_payload(state, trace=trace),
            "memory_evidence": self.memory_evidence_payload(trace) or {},
            "competition_evidence": self.competition_evidence_payload(trace) or {},
            "continuity_evidence": self.continuity_evidence_payload(trace, state=state) or {},
        }

    def _preserve_subject_core_on_restore(
        self,
        current_state: RuntimeState,
        restored_state: RuntimeState,
        *,
        violation_code: str,
    ) -> tuple[RuntimeState, str]:
        current_core = SubjectCore(**to_dict(self._ensure_subject_core(current_state)))
        restored_core = SubjectCore(**to_dict(self._ensure_subject_core(restored_state)))
        if to_dict(current_core) != to_dict(restored_core):
            restored_state.subject_core = current_core
            restored_state.safe_mode = True
            restored_state.mode = "safe"
            return restored_state, violation_code
        restored_state.subject_core = current_core
        return restored_state, ""

    def _migrate_runtime_schema_if_needed(self) -> None:
        status = self._runtime_migration_status()
        if int(status.get("schema_version", 0) or 0) >= RUNTIME_SCHEMA_VERSION:
            return

        payload: dict[str, Any] = {}
        if self.state_parquet_path.exists():
            rows = read_snapshot_rows(self.state_parquet_path, "select payload_json from read_parquet(?)")
            if rows:
                payload = json.loads(rows[0]["payload_json"])
        elif self.state_path.exists():
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))

        raw_identity = dict(payload.get("identity_state", {})) if isinstance(payload.get("identity_state", {}), dict) else {}
        raw_aliases = list(raw_identity.get("aliases", [])) if isinstance(raw_identity.get("aliases", []), list) else []
        migrated_state = RuntimeState(**payload) if payload else self.load_runtime_state()
        self._ensure_subject_core(migrated_state)
        removed_aliases_count = max(0, len(raw_aliases) - len(migrated_state.identity_state.aliases))
        if payload:
            self._save_state(migrated_state, sync=True)
        else:
            self._state_cache = RuntimeState(**to_dict(migrated_state))
        report = {
            **status,
            "schema_version": RUNTIME_SCHEMA_VERSION,
            "migrated_at": utc_now_iso(),
            "removed_aliases_count": removed_aliases_count,
            "preserved_memory_count": self.memory_store.migration_report().get("preserved_memory_count", 0),
            "identity_evidence_rebuilt": True,
        }
        self.runtime_migration_status_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    def runtime_migration_report(self) -> dict[str, Any]:
        runtime_status = self._runtime_migration_status()
        memory_status = self.memory_store.migration_report()
        trace_sync_status = self.trace_store.trace_storage_status()
        trace_migration_status: dict[str, Any] = {}
        if self.trace_store.migration_status_path.exists():
            try:
                trace_migration_status = json.loads(self.trace_store.migration_status_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                trace_migration_status = {}
        return {
            "runtime": {
                "schema_version": int(runtime_status.get("schema_version", 0) or 0),
                "removed_aliases_count": int(runtime_status.get("removed_aliases_count", 0) or 0),
                "preserved_memory_count": int(runtime_status.get("preserved_memory_count", 0) or 0),
                "identity_evidence_rebuilt": bool(runtime_status.get("identity_evidence_rebuilt", False)),
                "migrated_at": runtime_status.get("migrated_at"),
            },
            "memory": memory_status,
            "trace": {
                "status": str(trace_migration_status.get("status") or "not_needed"),
                "migrated_at": trace_migration_status.get("migrated_at"),
                "parquet_live_ready": bool(trace_sync_status.get("parquet_live_ready", False)),
                "trace_sync_state": str(trace_sync_status.get("trace_sync_state", trace_sync_status.get("storage_state", "healthy"))),
                "storage_state": str(trace_sync_status.get("storage_state", "healthy")),
                "degraded_reason": trace_sync_status.get("degraded_reason"),
                "last_sync_at": trace_sync_status.get("last_sync_at"),
            },
        }

    def flush_pending_io(self, *, raise_on_error: bool = False) -> None:
        self._state_io.flush(raise_on_error=raise_on_error)
        self.memory_store.flush(raise_on_error=raise_on_error)
        self.trace_store.flush(raise_on_error=raise_on_error)
        self._rounds_since_cold_flush = 0
        self._last_cold_flush_at = time.monotonic()

    def newborn_runtime_state(self) -> RuntimeState:
        return self.state_runtime.newborn_runtime_state()

    def _rebind_runtime_storage(self) -> None:
        self.state_runtime.rebind_runtime_storage()

    def reset_persona(self) -> RuntimeState:
        return self.state_runtime.reset_persona()

    def _maybe_flush_cold_path(
        self,
        *,
        scenario: str = "",
        mode: str = "",
        endogenous_turn: bool = False,
        raise_on_error: bool = False,
    ) -> bool:
        self._rounds_since_cold_flush += 1
        now = time.monotonic()
        round_interval = self._cold_flush_round_interval
        time_interval = self._cold_flush_interval_seconds
        round_due = round_interval > 0 and self._rounds_since_cold_flush >= round_interval
        time_due = time_interval > 0 and (now - self._last_cold_flush_at) >= time_interval
        if not round_due and not time_due:
            return False
        self.flush_pending_io(raise_on_error=raise_on_error)
        return True

    def _should_sync_tick_writeback(self, *, round_id: int, scenario: str, mode: str, endogenous_turn: bool) -> bool:
        if endogenous_turn:
            return True
        if mode in {"idle", "sleep", "safe"}:
            return True
        return False

    def _normalize_temperament_state(self, state: RuntimeState) -> dict[str, Any]:
        baseline = dict(state.temperament_state.get("baseline", {}))
        if not baseline:
            baseline = dict(self.config["temperament"]["temperament"])
        drift = {
            key: round(float(value), 4)
            for key, value in dict(state.temperament_state.get("drift", {})).items()
        }
        current = {
            key: round(_clip(float(value)), 4)
            for key, value in dict(state.temperament_state.get("current", {})).items()
        }
        for key, baseline_value in baseline.items():
            drift.setdefault(key, 0.0)
            current.setdefault(key, round(_clip(float(baseline_value) + float(drift[key])), 4))
        normalized = {
            "baseline": {key: round(_clip(float(value)), 4) for key, value in baseline.items()},
            "drift": drift,
            "current": current,
            "correction_window": dict(state.temperament_state.get("correction_window", {})),
            "freeze_until_round": dict(state.temperament_state.get("freeze_until_round", {})),
            "last_correction_events": list(state.temperament_state.get("last_correction_events", [])),
            "drift_diagnostics": dict(state.temperament_state.get("drift_diagnostics", {})),
        }
        state.temperament_state = normalized
        return normalized

    def _compute_resource_telemetry(self, state: RuntimeState) -> dict[str, float | str]:
        defaults = self.config["resource_rules"]["resource_defaults"]
        target_burn_ratio = max(float(defaults.get("target_burn_ratio", 0.20)), 0.01)
        burn_rate_ratio = _clip((1.0 - state.budget_remaining) / target_burn_ratio, 0.0, 1.0)
        low_balance_ratio = _clip(1.0 - state.budget_remaining, 0.0, 1.0)
        queue_cap = max(int(defaults.get("queue_cap", 64)), 1)
        queue_depth = max(len(state.pending_steps), int(state.resource_state.get("queue_depth", 0) or 0))
        queue_pressure = _clip(queue_depth / queue_cap, 0.0, 1.0)
        recent_skills = self.trace_store.recent_skill_traces(limit=20)
        avg_latency_ms = (
            sum(float(row.get("latency_ms", 0) or 0.0) for row in recent_skills) / len(recent_skills)
            if recent_skills
            else 0.0
        )
        latency_sla_ms = max(float(defaults.get("latency_sla_ms", 250)), 1.0)
        latency_pressure = _clip(avg_latency_ms / latency_sla_ms, 0.0, 1.0) if recent_skills else 0.0
        scarcity_index = _clip(
            0.40 * burn_rate_ratio
            + 0.30 * low_balance_ratio
            + 0.20 * queue_pressure
            + 0.10 * latency_pressure,
            0.0,
            1.0,
        )
        smooth_bias = smooth_resource_biases(scarcity_index)
        if scarcity_index >= 0.75:
            resource_mode = "starvation"
        elif scarcity_index >= 0.50:
            resource_mode = "scarce"
        elif scarcity_index >= 0.25:
            resource_mode = "stable"
        else:
            resource_mode = "abundant"
        return {
            "burn_rate_ratio": round(burn_rate_ratio, 4),
            "low_balance_ratio": round(low_balance_ratio, 4),
            "queue_pressure": round(queue_pressure, 4),
            "latency_pressure": round(latency_pressure, 4),
            "scarcity_index": round(scarcity_index, 4),
            "resource_mode": resource_mode,
            "scarcity_pressure": float(smooth_bias["scarcity_pressure"]),
            "body_hunger_bias": float(smooth_bias["body_hunger_bias"]),
            "effort_avoidance_bias": float(smooth_bias["effort_avoidance_bias"]),
            "deliberation_compress": float(smooth_bias["deliberation_compress"]),
            "rumination_bias": float(smooth_bias["rumination_bias"]),
            "action_shrink_scale": float(smooth_bias["action_shrink_scale"]),
            "queue_depth": queue_depth,
            "average_latency_ms": round(avg_latency_ms, 2),
        }

    def _state_hash(self, state: RuntimeState) -> str:
        return hashlib.sha1(json.dumps(to_dict(state), sort_keys=True).encode("utf-8")).hexdigest()

    def _trace_state_snapshot_payload(self, state: RuntimeState, *, endogenous_turn: bool) -> dict[str, Any]:
        snapshot = {
            "mode": state.mode,
            "safe_mode": state.safe_mode,
            "focus": state.focus,
            "budget_remaining": state.budget_remaining,
            "focus_lock_count": state.focus_lock_count,
            "mood": state.mood,
            "conflict_learning_state": to_dict(state.conflict_learning_state),
            "body_state": to_dict(state.body_state),
            "subjective_state": to_dict(state.subjective_state),
            "emotion_state": to_dict(state.emotion_state),
            "desire_state": to_dict(state.desire_state),
            "instinct_field": to_dict(state.instinct_field),
            "organic_mode": to_dict(state.organic_mode),
            "emergent_action_sketches": to_dict(state.emergent_action_sketches),
            "personality_anchor": to_dict(state.personality_anchor),
        }
        if endogenous_turn:
            snapshot["motivation_learning_state"] = to_dict(state.motivation_learning_state)
            snapshot["motivation_pool_state"] = to_dict(state.motivation_pool_state)
            snapshot["endogenous_state"] = to_dict(state.endogenous_state)
        return snapshot

    def _infer_operator_level(self, domain: str) -> str:
        return self.run_runtime._infer_operator_level(domain)

    def _default_mutation_scope(self, domain: str, target: str | None = None) -> str:
        return self.run_runtime._default_mutation_scope(domain, target)

    def _legacy_command_envelope(self, command: str) -> CommandEnvelope:
        return self.run_runtime._legacy_command_envelope(command)

    def _command_snapshot_path(self, snapshot_id: str) -> Path:
        return self.run_runtime._command_snapshot_path(snapshot_id)

    def _create_command_snapshot(self, state: RuntimeState, envelope: CommandEnvelope) -> str:
        return self.run_runtime._create_command_snapshot(state, envelope)

    def _apply_snapshot_restore(self, snapshot_id: str, envelope: CommandEnvelope) -> CommandResult:
        return self.run_runtime._apply_snapshot_restore(snapshot_id, envelope)

    def _enrich_command_result(self, result: CommandResult, envelope: CommandEnvelope) -> CommandResult:
        return self.run_runtime._enrich_command_result(result, envelope)

    def _build_rollback(self, envelope: CommandEnvelope, before_state: RuntimeState, result: CommandResult, snapshot_id: str) -> dict[str, Any]:
        return self.run_runtime._build_rollback(envelope, before_state, result, snapshot_id)

    def _mark_boundary_result(
        self,
        result: CommandResult,
        *,
        boundary_action: str,
        cause_type: str = "external_stimulus",
        violation_code: str = "",
        deprecation_warning: str = "",
    ) -> CommandResult:
        return self.run_runtime._mark_boundary_result(
            result,
            boundary_action=boundary_action,
            cause_type=cause_type,
            violation_code=violation_code,
            deprecation_warning=deprecation_warning,
        )

    def _boundary_deprecation(self, command: str) -> str:
        return self.run_runtime._boundary_deprecation(command)

    def _reject_boundary_command(
        self,
        state: RuntimeState,
        *,
        scope: str,
        operator_level: str,
        violation_code: str,
        risk_note: str,
    ) -> CommandResult:
        return self.run_runtime._reject_boundary_command(
            state,
            scope=scope,
            operator_level=operator_level,
            violation_code=violation_code,
            risk_note=risk_note,
        )

    def _subjectivity_metrics(
        self,
        *,
        rounds: list[dict[str, Any]] | None = None,
        commands: list[dict[str, Any]] | None = None,
        command_totals: dict[str, Any] | None = None,
        round_totals: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        state = self.load_runtime_state()
        core = self._ensure_subject_core(state)
        if commands is not None:
            commands = list(commands)
        elif command_totals is None:
            totals_reader = getattr(self.trace_store, "command_subjectivity_totals", None)
            if callable(totals_reader):
                command_totals = dict(totals_reader())
            else:
                commands = self.trace_store.list_commands()
        if rounds is not None:
            rounds = list(rounds)
        elif round_totals is None:
            round_totals_reader = getattr(self.trace_store, "round_subjectivity_totals", None)
            if callable(round_totals_reader):
                round_totals = dict(round_totals_reader())
            else:
                summary_reader = getattr(self.trace_store, "list_round_summaries", None)
                if callable(summary_reader):
                    rounds = list(summary_reader())
                else:
                    rounds = self.trace_store.list_rounds()
        if command_totals is not None:
            boundary_violation_count = int(command_totals.get("boundary_violation_count", 0) or 0)
            external_command_count = int(command_totals.get("external_count", 0) or 0)
            internal_command_count = int(command_totals.get("internal_count", 0) or 0)
        else:
            boundary_violation_count = sum(1 for item in commands or [] if item.get("violation_code"))
            external_command_count = sum(1 for item in commands or [] if item.get("cause_type") == "external_stimulus")
            internal_command_count = sum(1 for item in commands or [] if item.get("cause_type") == "endogenous")
        if round_totals is not None:
            external_round_count = int(round_totals.get("external_round_count", 0) or 0)
            endogenous_round_count = int(round_totals.get("endogenous_round_count", 0) or 0)
            total_rounds = int(round_totals.get("total_rounds", 0) or 0)
        else:
            external_round_count = sum(
                1 for item in rounds or [] if item.get("cause_type", "external_stimulus") == "external_stimulus"
            )
            endogenous_round_count = sum(1 for item in rounds or [] if item.get("cause_type") == "endogenous")
            total_rounds = len(rounds or [])
        external_count = external_command_count + external_round_count
        internal_count = internal_command_count + endogenous_round_count
        ratio = round(external_count / max(internal_count, 1), 4)
        return {
            "subject_id": core.subject_id,
            "continuity_nonce": core.continuity_nonce,
            "subject_core_integrity": bool(core.subject_id and core.continuity_nonce and core.birth_ts),
            "boundary_violation_count": boundary_violation_count,
            "external_to_internal_ratio": ratio,
            "endogenous_intent_rate": round(endogenous_round_count / max(total_rounds, 1), 4),
        }

    def subjectivity_metrics(self) -> dict[str, Any]:
        return self._subjectivity_metrics()

    def _round_seed(self, state: RuntimeState, event: RoundEvent) -> int:
        payload = f"{state.round_count}:{event.source}:{event.content}:{event.target or ''}:{event.cue or ''}"
        return int(hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8], 16)

    def _apply_state_patch(self, state: RuntimeState, patch: dict[str, Any]) -> None:
        for key, value in patch.items():
            if hasattr(state, key):
                setattr(state, key, value)
        self._sync_tlh_state(state)

    def _sync_tlh_state(self, state: RuntimeState) -> None:
        state.body_state.energy = round(_clip(float(state.body_energy), 0.0, 1.0), 4)
        state.body_state.fatigue = state.fatigue = round(_clip(float(state.fatigue), 0.0, 1.0), 4)
        state.body_state.memory_fragments = state.memory_fragments = round(_clip(float(state.memory_fragments), 0.0, 1.0), 4)
        state.body_state.self_continuity = state.self_continuity = round(_clip(float(state.self_continuity), 0.0, 1.0), 4)
        state.body_state.meaning_strength = state.meaning_strength = round(_clip(float(state.meaning_strength), 0.0, 1.0), 4)
        state.body_state.metabolism = state.base_metabolism = round(max(0.0, float(state.base_metabolism)), 4)
        state.subjective_state.boundary = round(_clip(float(state.subjective_state.boundary), 0.0, 1.0), 4)
        state.subjective_state.spontaneous = round(_clip(float(state.subjective_state.spontaneous), 0.0, 1.0), 4)
        state.subjective_state.reject_all = round(_clip(float(state.subjective_state.reject_all), 0.0, 1.0), 4)
        emotion_valence = _clip((float(state.mood) - 0.55) / 0.45, -1.0, 1.0)
        emotion_arousal = _clip(
            abs(emotion_valence) * 0.45
            + float(state.affect_residue or 0.0) * 0.4
            + float(state.subjective_state.spontaneous or 0.0) * 0.15,
            0.0,
            1.0,
        )
        appraisal_band = str(state.session_metadata.get("last_appraisal_band", state.emotion_state.appraisal_band or "steady") or "steady")
        state.emotion_state.valence = round(emotion_valence, 4)
        state.emotion_state.arousal = round(emotion_arousal, 4)
        state.emotion_state.residue = round(_clip(float(state.affect_residue or 0.0), 0.0, 1.0), 4)
        state.emotion_state.appraisal_band = appraisal_band

        latent_drives = {
            "comfort": round(_clip((1.0 - float(state.body_energy or 0.0)) * 0.5 + float(state.fatigue or 0.0) * 0.35 + float(state.affect_residue or 0.0) * 0.15), 6),
            "meaning": round(_clip(float(state.meaning_strength or 0.0) * 0.65 + min(1.0, len(state.subjective_state.meaning_made) * 0.16) * 0.35), 6),
            "relation": round(_clip((1.0 - float(state.subjective_state.boundary or 0.0)) * 0.35 + max(0.0, state.emotion_state.valence) * 0.2 + (1.0 - float(state.subjective_state.reject_all or 0.0)) * 0.25 + float(state.subjective_state.spontaneous or 0.0) * 0.2), 6),
            "exploration": round(_clip(float(state.subjective_state.spontaneous or 0.0) * 0.45 + float(state.memory_fragments or 0.0) * 0.35 + (1.0 - float(state.self_continuity or 0.0)) * 0.2), 6),
            "completion": round(_clip(float(state.meaning_strength or 0.0) * 0.25 + (1.0 - float(state.body_state.energy or 0.0)) * -0.08 + (1.0 - float(state.resource_state.get("scarcity_index", 0.0) or 0.0)) * 0.28 + (1.0 - float(state.subjective_state.reject_all or 0.0)) * 0.2 + (1.0 - float(state.memory_fragments or 0.0)) * 0.15), 6),
        }
        dominant_drive = max(latent_drives, key=latent_drives.get) if latent_drives else ""
        state.desire_state.latent_drives = latent_drives
        state.desire_state.dominant_drive = dominant_drive
        state.desire_state.drive_tension = round(max(latent_drives.values(), default=0.0), 4)

    def sync_tlh_state(self, state: RuntimeState) -> None:
        self._sync_tlh_state(state)

    def _subjective_pressure(self, state: RuntimeState) -> float:
        self._sync_tlh_state(state)
        meaning_density = min(1.0, len(state.subjective_state.meaning_made) * 0.2)
        continuity_drop = 1.0 - float(state.self_continuity or 0.0)
        pressure = (
            float(state.subjective_state.reject_all or 0.0) * 0.24
            + float(state.subjective_state.spontaneous or 0.0) * 0.22
            + float(state.fatigue or 0.0) * 0.15
            + float(state.memory_fragments or 0.0) * 0.13
            + continuity_drop * 0.1
            + (1.0 - float(state.meaning_strength or 0.0)) * 0.06
            + meaning_density * 0.1
        )
        if state.organic_mode.enabled:
            pressure *= 1.0 + max(
                0.0,
                (float(state.organic_mode.body_weight) + float(state.organic_mode.subjective_weight)) / 2.0 - 1.0,
            ) * 0.18
        return round(_clip(pressure, 0.0, 1.0), 6)

    def _active_sketch_growth_scalar(self, state: RuntimeState) -> float:
        return round(
            max((float(sketch.growth_score or 0.0) for sketch in state.emergent_action_sketches), default=0.0),
            6,
        )

    def _recent_action_bias(self, *, limit: int = 12) -> dict[str, float]:
        rounds = self.trace_store.recent_rounds(limit=limit)
        counts: dict[str, float] = {}
        total = 0.0
        for row in rounds:
            action = str(row.get("sampled_action") or "").strip()
            if not action:
                action = ""
            if action:
                counts[action] = counts.get(action, 0.0) + 1.0
                total += 1.0
            for driver in list(row.get("top_drivers", []) or [])[:3]:
                if not isinstance(driver, dict):
                    continue
                driver_action = str(driver.get("action_name") or "").strip()
                driver_score = abs(float(driver.get("score", 0.0) or 0.0))
                if not driver_action or driver_score <= 0.0:
                    continue
                weight = min(0.85, 0.22 + driver_score * 2.5)
                counts[driver_action] = counts.get(driver_action, 0.0) + weight
                total += weight
        if total <= 0.0:
            return {}
        return {action: round(value / total, 6) for action, value in counts.items()}

    def _update_personality_anchor(
        self,
        state: RuntimeState,
        identity_evidence: dict[str, Any],
    ) -> PersonalityAnchorState:
        self._sync_tlh_state(state)
        anchor = state.personality_anchor
        prior_axes = dict(anchor.axis_baseline or {})
        prior_stability = float(anchor.stability or 0.5)
        collapse_trace = dict(state.instinct_field.collapse_trace or {})
        subject_vector = dict(collapse_trace.get("subject_vector", {}) or collapse_trace.get("coupled_axes", {}) or collapse_trace.get("normalized_axes", {}) or {})
        if not subject_vector:
            subject_vector = {axis: float(prior_axes.get(axis, 0.5) or 0.5) for axis in AXES}
        alpha = round(_clip(1.0 / min(max(int(state.round_count or 1), 1), 12), 0.08, 0.25), 6)
        blended_axes = {
            axis: round(
                _clip(float(prior_axes.get(axis, 0.5) or 0.5) * (1.0 - alpha) + float(subject_vector.get(axis, 0.5) or 0.5) * alpha),
                6,
            )
            for axis in AXES
        }
        drift = round(
            sum(abs(blended_axes[axis] - float(prior_axes.get(axis, 0.5) or 0.5)) for axis in AXES) / len(AXES),
            6,
        )
        recent_bias = self._recent_action_bias()
        latest_rounds = self.trace_store.recent_rounds(limit=1)
        latest_trace = latest_rounds[-1] if latest_rounds else {}
        latest_action_posterior = (
            dict(latest_trace.get("probability_field", {}).get("action", {}).get("winner_posterior", {}) or {})
            if isinstance(latest_trace, dict)
            else {}
        )
        if not latest_action_posterior:
            raw_match_scores = {
                str(action): max(0.0, float(score or 0.0))
                for action, score in dict(collapse_trace.get("match_scores", {}) or {}).items()
            }
            match_total = sum(raw_match_scores.values()) or 1.0
            latest_action_posterior = {
                action: round(score / match_total, 6)
                for action, score in raw_match_scores.items()
                if score > 0.0
            }
        action_bias = {
            action: round(
                float(anchor.action_bias.get(action, 0.0) or 0.0) * (1.0 - alpha)
                + float(latest_action_posterior.get(action, recent_bias.get(action, 0.0)) or 0.0) * alpha,
                6,
            )
            for action in INTERNAL_RUNTIME_ACTIONS
            if (
                float(anchor.action_bias.get(action, 0.0) or 0.0) > 0.0
                or float(latest_action_posterior.get(action, 0.0) or 0.0) > 0.0
                or float(recent_bias.get(action, 0.0) or 0.0) > 0.0
            )
        }
        stability = round(
            _clip(prior_stability * (1.0 - alpha) + (1.0 - min(drift * 1.5, 1.0)) * alpha),
            4,
        )
        alignment = round(_clip((cosine_similarity(subject_vector, blended_axes) + 1.0) / 2.0, 0.0, 1.0), 6)
        state.personality_anchor = PersonalityAnchorState(
            axis_baseline=blended_axes,
            action_bias=action_bias,
            evidence_anchors=list(identity_evidence.get("anchors", []) or []),
            anchor_signature=str(identity_evidence.get("signature", "") or ""),
            stability=stability,
            drift=round(_clip(drift), 4),
            alignment=alignment,
            updated_round=state.round_count,
            continuity_derivation=dict(anchor.continuity_derivation or {}),
        )
        return state.personality_anchor

    def _normalize_axis_value(self, value: float) -> float:
        return round(_clip(0.5 + math.tanh(float(value)) * 0.5), 6)

    def _current_axis_alignment(
        self,
        *,
        anchor: PersonalityAnchorState,
        normalized_axes: dict[str, float],
        action: str | None = None,
        match_scores: dict[str, float] | None = None,
    ) -> float:
        axis_alignment = _clip((cosine_similarity(normalized_axes, anchor.axis_baseline) + 1.0) / 2.0, 0.0, 1.0)
        action_alignment = float(anchor.action_bias.get(str(action or ""), 0.0) or 0.0)
        collapse_alignment = _clip((float((match_scores or {}).get(str(action or ""), 0.0) or 0.0) + 1.0) / 2.0, 0.0, 1.0)
        return round(_clip(axis_alignment * 0.72 + action_alignment * 0.18 + collapse_alignment * 0.10), 6)

    def _high_dimensional_collapse(
        self,
        *,
        state: RuntimeState,
        v_main: dict[str, float],
        v_mod: dict[str, float],
        weight: float,
        round_seed: int,
    ) -> tuple[dict[str, float], dict[str, Any], str]:
        sketch_vectors = [
            vector
            for vector in (
                infer_sketch_vector(sketch, TLH_ACTION_VECTORS)
                for sketch in state.emergent_action_sketches[:4]
            )
            if vector is not None
        ]
        collapse_payload = compute_tlh_vector_collapse(
            v_main=v_main,
            v_mod=v_mod,
            v_anchor=dict(state.personality_anchor.axis_baseline or {}),
            action_vectors=TLH_ACTION_VECTORS,
            modulation_directions=build_tlh_modulation_directions(
                action_vectors=TLH_ACTION_VECTORS,
                sketch_vectors=sketch_vectors,
            ),
            action_bias=dict(state.personality_anchor.action_bias or {}),
            weight=weight,
            tie_break_seed=round_seed,
        )
        modulated_delta = dict(collapse_payload.get("collapse_delta", {}) or {})
        selected_action = str(collapse_payload.get("selected_action") or "")
        return modulated_delta, collapse_payload, selected_action

    def _apply_subject_dynamics_after_round(
        self,
        *,
        state: RuntimeState,
        appraisal: dict[str, Any],
        slow_variables: dict[str, Any],
        relation_state: dict[str, float],
        sampled_action: str,
        anchor_alignment: float,
        requested_mode: str,
    ) -> None:
        action_relief = {
            "respond": 0.0,
            "plan": -0.02,
            "recall": 0.04,
            "rest": 0.18,
            "connect": -0.01,
            "clarify": -0.01,
            "wander": 0.03,
            "absorb": 0.12,
            "nothing": 0.06,
            "die": -0.04,
            "short_reply": 0.02,
        }
        mode_recovery = {"interactive": 0.0, "idle": 0.10, "sleep": 0.22, "safe": 0.04}
        cognitive_load = float(appraisal.get("cognitive_load", 0.0) or 0.0)
        interference = float(slow_variables.get("memory_interference", 0.0) or 0.0)
        scarcity = float(slow_variables.get("resource_scarcity", 0.0) or 0.0)
        affect_residue = float(slow_variables.get("affect_residue", 0.0) or 0.0)
        relief = float(action_relief.get(sampled_action, 0.0))
        fatigue_delta = cognitive_load * 0.12 + max(0.0, 0.45 - float(state.body_energy or 0.0)) * 0.08 + scarcity * 0.04 - relief - float(mode_recovery.get(requested_mode, 0.0))
        fragment_delta = interference * 0.14 + float(state.subjective_state.spontaneous or 0.0) * 0.03 - (0.10 if sampled_action in {"absorb", "recall"} else 0.0) - float(mode_recovery.get(requested_mode, 0.0)) * 0.28
        meaning_delta = anchor_alignment * 0.08 + min(1.0, len(state.subjective_state.meaning_made) * 0.08) * 0.04 - float(state.subjective_state.reject_all or 0.0) * 0.03
        if sampled_action == "die":
            meaning_delta -= 0.14
        state.fatigue = round(_clip(float(state.fatigue or 0.0) + fatigue_delta), 4)
        state.memory_fragments = round(_clip(float(state.memory_fragments or 0.0) + fragment_delta), 4)
        state.meaning_strength = round(_clip(float(state.meaning_strength or 0.0) + meaning_delta), 4)
        state.subjective_state.spontaneous = round(
            _clip(float(state.subjective_state.spontaneous or 0.0) * 0.84 + float(state.memory_fragments or 0.0) * 0.10 + affect_residue * 0.06),
            4,
        )
        self._apply_subjective_baseline_floor(state)
        state.subjective_state.reject_all = round(
            _clip(float(state.subjective_state.reject_all or 0.0) * 0.78 + float(state.fatigue or 0.0) * 0.10 + (1.0 - anchor_alignment) * 0.12),
            4,
        )
        state.subjective_state.boundary = round(
            _clip(float(state.subjective_state.boundary or 0.0) * 0.76 + float(relation_state.get("boundary_level", state.subjective_state.boundary) or state.subjective_state.boundary) * 0.12 + scarcity * 0.06 + float(state.subjective_state.reject_all or 0.0) * 0.06),
            4,
        )
        continuity_target = _clip(
            anchor_alignment * 0.44
            + (1.0 - float(state.memory_fragments or 0.0)) * 0.24
            + float(state.meaning_strength or 0.0) * 0.18
            + float(state.personality_anchor.stability or 0.0) * 0.14,
            0.0,
            1.0,
        )
        state.self_continuity = round(
            _clip(float(state.self_continuity or 0.0) * 0.56 + continuity_target * 0.44),
            4,
        )
        state.personality_anchor.alignment = round(anchor_alignment, 4)
        state.personality_anchor.continuity_derivation = {
            "anchor_alignment": round(anchor_alignment, 6),
            "memory_integration": round(1.0 - float(state.memory_fragments or 0.0), 6),
            "meaning_strength": round(float(state.meaning_strength or 0.0), 6),
            "anchor_stability": round(float(state.personality_anchor.stability or 0.0), 6),
            "continuity_target": round(continuity_target, 6),
            "sampled_action": sampled_action,
        }

    def _autonomy_policy_for_profile(self, profile: str = "tool_level") -> AutonomyPolicyState:
        return self.run_runtime._autonomy_policy_for_profile(profile)

    def _apply_observer_autonomy_preferences(self, policy: AutonomyPolicyState) -> AutonomyPolicyState:
        return self.run_runtime._apply_observer_autonomy_preferences(policy)

    def _sync_autonomy_state(self, state: RuntimeState) -> None:
        self.run_runtime._sync_autonomy_state(state)

    def sync_autonomy_state(self, state: RuntimeState) -> None:
        self.run_runtime._sync_autonomy_state(state)

    def _ensure_default_autonomy_runtime(self, state: RuntimeState, *, force: bool = False) -> bool:
        return self.run_runtime._ensure_default_autonomy_runtime(state, force=force)

    def _autonomy_matches_prefix(self, command: str, prefix: str) -> bool:
        return self.run_runtime._autonomy_matches_prefix(command, prefix)

    def _autonomy_command_allowed(
        self,
        command: str,
        policy: AutonomyPolicyState | None = None,
    ) -> tuple[bool, str]:
        return self.run_runtime._autonomy_command_allowed(command, policy)

    def _autonomy_operator_allowed(
        self,
        operator_level: str,
        policy: AutonomyPolicyState | None = None,
    ) -> tuple[bool, str]:
        return self.run_runtime._autonomy_operator_allowed(operator_level, policy)

    def _generate_autonomy_self_run_goal_via_model(self, state: RuntimeState) -> str | None:
        return self.run_runtime._generate_autonomy_self_run_goal_via_model(state)

    def _autonomy_budget_usage(self, state: RuntimeState) -> dict[str, Any]:
        return self.run_runtime._autonomy_budget_usage(state)

    def _append_autonomy_recent_action(
        self,
        state: RuntimeState,
        *,
        action_type: str,
        summary: str,
        round_id: int | None = None,
        trace_ref: str | None = None,
    ) -> None:
        self.run_runtime._append_autonomy_recent_action(
            state,
            action_type=action_type,
            summary=summary,
            round_id=round_id,
            trace_ref=trace_ref,
        )

    def _autonomy_window_anchor(self, value: str | None) -> datetime | None:
        return self.run_runtime._autonomy_window_anchor(value)

    def _ensure_autonomy_window(self, state: RuntimeState) -> None:
        self.run_runtime._ensure_autonomy_window(state)

    def _autonomy_step_context(self, state: RuntimeState) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        return self.run_runtime._autonomy_step_context(state)

    def _autonomy_self_run_score(
        self,
        state: RuntimeState,
        policy: AutonomyPolicyState,
    ) -> float:
        return self.run_runtime._autonomy_self_run_score(state, policy)

    def _autonomy_pending_readonly_repo_scan(
        self,
        state: RuntimeState,
        policy: AutonomyPolicyState,
    ) -> dict[str, str] | None:
        return self.run_runtime._autonomy_pending_readonly_repo_scan(state, policy)

    def _stale_dirty_worktree_run_info(self, run_state: RunState | dict[str, Any] | None) -> dict[str, Any] | None:
        return self.run_runtime._stale_dirty_worktree_run_info(run_state)

    def _active_run_blocks_background_progress(self, state: RuntimeState) -> bool:
        return self.run_runtime._active_run_blocks_background_progress(state)

    def _clear_stale_dirty_worktree_run_blocker(self, state: RuntimeState) -> bool:
        return self.run_runtime._clear_stale_dirty_worktree_run_blocker(state)

    def _autonomy_candidate_scores(
        self,
        state: RuntimeState,
        policy: AutonomyPolicyState,
    ) -> dict[str, float]:
        return self.run_runtime._autonomy_candidate_scores(state, policy)

    def _autonomy_candidate_action(
        self,
        state: RuntimeState,
        policy: AutonomyPolicyState,
    ) -> str:
        return self.run_runtime._autonomy_candidate_action(state, policy)

    def _autonomy_projected_candidate_scores(
        self,
        state: RuntimeState,
        policy: AutonomyPolicyState,
        field_probe: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        return self.run_runtime._autonomy_projected_candidate_scores(state, policy, field_probe)

    def _autonomy_trace_append(
        self,
        state: RuntimeState,
        *,
        action_type: str,
        summary: str,
        delta: dict[str, Any] | None = None,
    ) -> None:
        self.run_runtime._autonomy_trace_append(
            state,
            action_type=action_type,
            summary=summary,
            delta=delta,
        )

    def _autonomy_record_failure(self, state: RuntimeState, exc: Exception) -> None:
        self.run_runtime._autonomy_record_failure(state, exc)

    def _autonomy_execute_command(self, command: str) -> dict[str, Any]:
        return self.run_runtime._autonomy_execute_command(command)

    def _build_tlh_memory_contribution(self, state: RuntimeState) -> ProbabilisticContribution | None:
        self._sync_tlh_state(state)
        continuity_drop = round(1.0 - float(state.self_continuity or 0.0), 6)
        memory_signal: dict[str, float] = {
            "state:fatigue": round(float(state.fatigue or 0.0), 6),
            "state:fragments": round(float(state.memory_fragments or 0.0), 6),
            "state:continuity_drop": continuity_drop,
            "state:meaning_strength": round(float(state.meaning_strength or 0.0), 6),
            "state:reject_all": round(float(state.subjective_state.reject_all or 0.0), 6),
            "state:spontaneous": round(float(state.subjective_state.spontaneous or 0.0), 6),
        }
        for item in state.subjective_state.felt[:3]:
            memory_signal[f"felt:{item}"] = round(memory_signal.get(f"felt:{item}", 0.0) + 0.12, 6)
        for item in state.subjective_state.meaning_made[:3]:
            memory_signal[f"meaning:{item}"] = round(memory_signal.get(f"meaning:{item}", 0.0) + 0.16, 6)
        if max(memory_signal.values(), default=0.0) <= 0.0:
            return None
        return ProbabilisticContribution(
            module_name="TLHMemoryBridge",
            module_type="subjective_memory",
            level="memory",
            target_space="memory",
            raw_signal=memory_signal,
            modulated_delta=memory_signal,
            confidence=round(_clip(0.4 + self._subjective_pressure(state) * 0.4, 0.0, 1.0), 4),
            trace_reason="body state and subjectivity write internal state markers into the memory field",
            projection_reason="TLH internal state projected into memory energy for downstream action coupling",
            applied_at_stage="tlh_memory_bridge",
            native_operator="tlh_memory_bridge",
            dependency_trace=[
                f"felt_count:{len(state.subjective_state.felt)}",
                f"meaning_count:{len(state.subjective_state.meaning_made)}",
                f"subjective_pressure:{self._subjective_pressure(state)}",
            ],
            projection=EnergyProjectionSpec(module_type="subjective_memory", target_space="memory", module_temperature=0.9),
        )

    def _build_emergent_action_contribution(self, state: RuntimeState) -> ProbabilisticContribution | None:
        active_sketches = [
            sketch
            for sketch in state.emergent_action_sketches
            if float((sketch.get("growth_score", 0.0) if isinstance(sketch, dict) else getattr(sketch, "growth_score", 0.0)) or 0.0) >= 0.28
            or str(sketch.get("status", "latent") if isinstance(sketch, dict) else getattr(sketch, "status", "latent")) in {"growing", "ready", "formalized"}
        ]
        if not active_sketches:
            return None
        modulated_delta: dict[str, float] = {}
        dependency_trace: list[str] = []
        for sketch in active_sketches[:4]:
            sketch_status = str(sketch.get("status", "latent") if isinstance(sketch, dict) else getattr(sketch, "status", "latent"))
            sketch_growth = float((sketch.get("growth_score", 0.0) if isinstance(sketch, dict) else getattr(sketch, "growth_score", 0.0)) or 0.0)
            sketch_name = str(sketch.get("name", "") if isinstance(sketch, dict) else getattr(sketch, "name", ""))
            base_scale = 0.18 if sketch_status == "formalized" else 0.12 if sketch_status == "ready" else 0.07
            scale = round(base_scale + sketch_growth * 0.18, 6)
            dependency_trace.append(
                f"{sketch_name}:{sketch_status}:{round(sketch_growth, 4)}"
            )
            sketch_vector = infer_sketch_vector(sketch, TLH_ACTION_VECTORS)
            if sketch_vector is not None:
                support_map = project_vector_to_action_support(
                    sketch_vector,
                    TLH_ACTION_VECTORS,
                    floor=0.45,
                    limit=4,
                )
            else:
                raw_target_action_map = sketch.get("target_action_map", {}) if isinstance(sketch, dict) else getattr(sketch, "target_action_map", {})
                raw_support_actions = sketch.get("support_actions", {}) if isinstance(sketch, dict) else getattr(sketch, "support_actions", {})
                support_map = dict(raw_target_action_map or raw_support_actions or {})
            for action, value in support_map.items():
                score = round(float(value or 0.0) * scale, 6)
                if score == 0.0:
                    continue
                modulated_delta[str(action)] = round(modulated_delta.get(str(action), 0.0) + score, 6)
        if not modulated_delta:
            return None
        return ProbabilisticContribution(
            module_name="EmergentActionSketch",
            module_type="emergent",
            level="action",
            target_space="action",
            raw_signal=dict(modulated_delta),
            modulated_delta=modulated_delta,
            confidence=round(
                _clip(
                    0.32
                    + max(
                        float((sketch.get("growth_score", 0.0) if isinstance(sketch, dict) else getattr(sketch, "growth_score", 0.0)) or 0.0)
                        for sketch in active_sketches[:4]
                    )
                    * 0.45,
                    0.0,
                    1.0,
                ),
                4,
            ),
            trace_reason="repeated internal pressure grows learned action sketches that re-enter action competition",
            projection_reason="emergent action sketches projected back into the action field",
            applied_at_stage="emergent_sketch",
            native_operator="emergent_action_sketch",
            dependency_trace=dependency_trace,
            projection=EnergyProjectionSpec(module_type="emergent", target_space="action", module_temperature=0.95),
        )

    def _update_emergent_action_sketches(
        self,
        *,
        state: RuntimeState,
        action_posterior: dict[str, float],
        sampled_action: str,
    ) -> None:
        self._sync_tlh_state(state)
        instinct = state.instinct_field
        candidates = [str(action).strip() for action in instinct.candidate_actions if str(action).strip()]
        if not candidates or not instinct.winner_region:
            return
        subjective_pressure = self._subjective_pressure(state)
        dominant_region = max((float(value) for value in instinct.region_scores.values()), default=0.0)
        growth_boost = _clip(
            dominant_region * 0.36
            + subjective_pressure * 0.32
            + float(action_posterior.get(sampled_action, 0.0) or 0.0) * 0.18
            + min(1.0, len(state.subjective_state.felt) * 0.06)
            + min(1.0, len(state.subjective_state.meaning_made) * 0.08),
            0.0,
            1.0,
        )
        if growth_boost < 0.28:
            return

        sketch_name = f"{instinct.winner_region}:{'+'.join(candidates[:2])}"
        support_actions = {
            action: round(float(action_posterior.get(action, 0.0) or 0.0), 6)
            for action in candidates[:4]
            if float(action_posterior.get(action, 0.0) or 0.0) > 0.0
        }
        sources: list[str] = [f"winner_region:{instinct.winner_region}"]
        sources.extend(f"felt:{item}" for item in state.subjective_state.felt[:2])
        sources.extend(f"meaning:{item}" for item in state.subjective_state.meaning_made[:2])
        sources.extend(f"action:{item}" for item in candidates[:3])
        if sampled_action:
            sources.append(f"sampled:{sampled_action}")
        anchor_alignment = self._current_axis_alignment(
            anchor=state.personality_anchor,
            normalized_axes={
                axis: float(
                    dict(
                        instinct.collapse_trace.get("subject_vector", {})
                        or instinct.collapse_trace.get("coupled_axes", {})
                        or instinct.collapse_trace.get("normalized_axes", {})
                        or {}
                    ).get(axis, 0.5)
                    or 0.5
                )
                for axis in AXES
            },
            action=sampled_action,
            match_scores=dict(instinct.collapse_trace.get("match_scores", {}) or {}),
        )

        upgraded: list[Any] = []
        matched = False
        for raw_sketch in state.emergent_action_sketches:
            sketch = raw_sketch if hasattr(raw_sketch, "name") else raw_sketch
            if str(sketch.name) != sketch_name:
                upgraded.append(sketch)
                continue
            matched = True
            sketch.signal_sources = sorted(set(list(sketch.signal_sources) + sources))
            sketch.support_actions = {
                key: round(max(float(sketch.support_actions.get(key, 0.0) or 0.0), value), 6)
                for key, value in {**sketch.support_actions, **support_actions}.items()
            }
            sketch.growth_score = round(_clip(max(float(sketch.growth_score or 0.0), float(sketch.growth_score or 0.0) + growth_boost * 0.35), 0.0, 1.0), 4)
            sketch.stability = max(int(sketch.stability or 0) + 1, 1)
            sketch.anchor_alignment = round(max(float(sketch.anchor_alignment or 0.0), anchor_alignment), 4)
            sketch.target_action_map = {
                action: round(float(value), 6)
                for action, value in sorted(sketch.support_actions.items(), key=lambda item: item[1], reverse=True)[:3]
            }
            if sketch.growth_score >= sketch.upgrade_threshold and sketch.stability >= 2 and sketch.anchor_alignment >= 0.58:
                sketch.status = "formalized"
            elif sketch.growth_score >= sketch.upgrade_threshold:
                sketch.status = "ready"
            elif sketch.growth_score >= max(0.35, sketch.upgrade_threshold * 0.5):
                sketch.status = "growing"
            else:
                sketch.status = "latent"
            upgraded.append(sketch)
        if not matched:
            threshold = 0.66 if sketch_name.split(":", 1)[0] != "dissolve" else 0.72
            score = round(_clip(growth_boost * 0.7, 0.0, 1.0), 4)
            status = "ready" if score >= threshold else "growing" if score >= max(0.35, threshold * 0.5) else "latent"
            upgraded.append(
                EmergentActionSketch(
                    name=sketch_name,
                    signal_sources=sources,
                    support_actions=support_actions,
                    growth_score=score,
                    upgrade_threshold=threshold,
                    status="formalized" if score >= threshold and anchor_alignment >= 0.7 else status,
                    target_action_map={
                        action: round(float(value), 6)
                        for action, value in sorted(support_actions.items(), key=lambda item: item[1], reverse=True)[:3]
                    },
                    anchor_alignment=anchor_alignment,
                    stability=1,
                )
            )
        upgraded.sort(key=lambda item: (float(getattr(item, "growth_score", 0.0)), str(getattr(item, "name", ""))), reverse=True)
        state.emergent_action_sketches = upgraded[:8]

    def _build_instinct_field_contribution(
        self,
        *,
        state: RuntimeState,
        context: dict[str, Any],
        relation_state: dict[str, float],
        slow_variables: dict[str, Any],
    ) -> ProbabilisticContribution | None:
        self._sync_tlh_state(state)
        if not state.organic_mode.enabled:
            return None

        closeness = float(relation_state.get("closeness", context.get("closeness", 0.5)) or 0.5)
        boundary = max(float(relation_state.get("boundary_level", state.subjective_state.boundary) or state.subjective_state.boundary), state.subjective_state.boundary)
        affect_residue = float(slow_variables.get("affect_residue", state.affect_residue) or 0.0)
        memory_activation = float(slow_variables.get("memory_activation", 0.0) or 0.0)
        scarcity = float(slow_variables.get("resource_scarcity", state.resource_state.get("scarcity_index", 0.0)) or 0.0)
        interference = float(context.get("interference", 0.0) or 0.0)
        spontaneous = float(state.subjective_state.spontaneous or 0.0)
        reject_all = float(state.subjective_state.reject_all or 0.0)
        meaning_strength = float(state.meaning_strength or 0.0)
        continuity = float(state.self_continuity or 0.0)
        fatigue = float(state.fatigue or 0.0)
        fragments = float(state.memory_fragments or 0.0)
        body_energy = float(state.body_energy or 0.0)
        anchor = state.personality_anchor
        growth_scalar = self._active_sketch_growth_scalar(state)
        v_main, v_mod, raw_axis_values = build_tlh_state_vectors(
            closeness=closeness,
            boundary=boundary,
            affect_residue=affect_residue,
            memory_activation=memory_activation,
            scarcity=scarcity,
            interference=interference,
            spontaneous=spontaneous,
            reject_all=reject_all,
            meaning_strength=meaning_strength,
            continuity=continuity,
            fatigue=fatigue,
            fragments=fragments,
            body_energy=body_energy,
            meaning_made_count=len(state.subjective_state.meaning_made),
            sketch_growth=growth_scalar,
        )
        anchor_axes = dict(anchor.axis_baseline or {})
        weight = max(1.0, float(state.organic_mode.body_weight) * 0.5 + float(state.organic_mode.subjective_weight) * 0.5)
        modulated_delta, collapse_payload, selected_action = self._high_dimensional_collapse(
            state=state,
            v_main=v_main,
            v_mod=v_mod,
            weight=weight,
            round_seed=max(1, int(state.round_count or 1)),
        )
        region_scores = region_scores_from_match_scores(dict(collapse_payload.get("match_scores", {}) or {}))
        winner_region = max(region_scores, key=region_scores.get)
        candidate_actions = [
            name
            for name, value in sorted(modulated_delta.items(), key=lambda item: item[1], reverse=True)
            if value > 0.0
        ][:6]
        state.instinct_field.axis_values = v_main
        state.instinct_field.region_scores = region_scores
        state.instinct_field.winner_region = winner_region
        state.instinct_field.candidate_actions = candidate_actions
        state.instinct_field.collapse_trace = {
            **collapse_payload,
            "raw_axes": dict(raw_axis_values),
            "body_energy": round(body_energy, 6),
            "fatigue": round(fatigue, 6),
            "reject_all": round(reject_all, 6),
            "meaning_strength": round(meaning_strength, 6),
            "boundary": round(boundary, 6),
            "anchor_axes": dict(anchor_axes),
            "selected_action": selected_action,
        }
        return ProbabilisticContribution(
            module_name="InstinctField",
            module_type="instinct",
            level="action",
            target_space="action",
            raw_signal=dict(collapse_payload.get("match_scores", {}) or {}),
            modulated_delta=modulated_delta,
            confidence=round(_clip(0.45 + region_scores[winner_region] * 0.35, 0.0, 1.0), 4),
            trace_reason="body state, subjective state, personality anchor, and 4D collapse co-resolve into action pressure",
            projection_reason="4D subject-space collapse projected from TLH runtime state",
            applied_at_stage="instinct_field",
            native_operator="high_dimensional_collapse",
            dependency_trace=[
                f"winner_region:{winner_region}",
                f"selected_action:{selected_action}",
                f"felt:{'|'.join(state.subjective_state.felt[:3])}" if state.subjective_state.felt else "felt:none",
            ],
        )

    def _agent_weight(self, agent_name: str, state: RuntimeState) -> float:
        base = self.config["agents"]["agents"].get(agent_name, {}).get("weight", 1.0)
        weight = state.agent_weight_overrides.get(agent_name, base)
        if agent_name == "PFCAgent" and state.conflict_hot_rounds > 0:
            weight *= 1.15
        return weight

    def _record_skill_trace(
        self,
        bucket: list[dict[str, Any]],
        round_id: int,
        result,
        *,
        extra: dict[str, Any] | None = None,
    ) -> None:
        row = {
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
            "parallel_group": getattr(result, "parallel_group", None),
            "agent_tier": getattr(result, "agent_tier", None),
            "task_priority": getattr(result, "task_priority", None),
            "task_outcome": getattr(result, "task_outcome", None),
            "task_type": getattr(result, "task_type", None),
            "timeout_ms": getattr(result, "timeout_ms", None),
        }
        if extra:
            row.update(extra)
        bucket.append(row)

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

    def _looks_like_task_followup(self, text: str) -> bool:
        lowered = text.lower()
        task_tokens = (
            "repo",
            "code",
            "file",
            "files",
            "test",
            "tests",
            "fix",
            "debug",
            "implement",
            "continue",
            "next step",
            "run",
            "branch",
            "commit",
            "仓库",
            "代码",
            "文件",
            "测试",
            "修复",
            "实现",
            "继续做",
            "下一步",
        )
        return any(token in lowered for token in task_tokens)

    def _sanitize_execution_state(self, state: RuntimeState) -> RuntimeState:
        sanitized = RuntimeState(**to_dict(state))
        sanitized.active_run_id = None
        sanitized.run_status = "idle"
        sanitized.run_mode = None
        sanitized.current_goal = None
        sanitized.current_step_id = None
        sanitized.pending_steps = []
        sanitized.completed_steps = []
        sanitized.last_tool_result = {}
        sanitized.stop_reason = {}
        sanitized.dirty_worktree_detected = False
        return sanitized

    def _state_for_reasoning(
        self,
        state: RuntimeState,
        event: RoundEvent,
        scenario: str,
        requested_mode: str,
    ) -> tuple[RuntimeState, dict[str, Any]]:
        has_active_run = bool(state.active_run_id and state.run_status in {"running", "paused"})
        if requested_mode in {"idle", "sleep"}:
            drive_source = "noninteractive_shaping"
        elif scenario == "task":
            drive_source = "active_task" if has_active_run else "user_input"
        elif has_active_run and self._looks_like_task_followup(event.content):
            drive_source = "mixed"
        else:
            drive_source = "user_input"
        contamination_detected = bool(has_active_run and scenario != "task" and drive_source == "user_input")
        reasoning_state = self._sanitize_execution_state(state) if contamination_detected else RuntimeState(**to_dict(state))
        run_context = {
            "active_run_id": state.active_run_id,
            "run_status": state.run_status,
            "current_goal": state.current_goal,
            "drive_source": drive_source,
            "execution_state_visible": not contamination_detected,
        }
        return reasoning_state, {"run_context": run_context, "run_contamination_detected": contamination_detected}

    def _infer_appraisal(
        self,
        event: RoundEvent,
        state: RuntimeState,
        *,
        scenario: str,
        requested_mode: str,
        closeness: float,
    ) -> dict[str, Any]:
        lowered = event.content.lower()
        negative_tokens = ("难过", "伤心", "失望", "烦", "糟", "痛苦", "sad", "upset", "hurt", "tired of")
        positive_tokens = ("开心", "高兴", "轻松", "喜欢", "安心", "happy", "glad", "relieved")
        fatigue_tokens = ("累", "困", "疲惫", "想睡", "睡了", "休息", "tired", "sleepy", "exhausted")
        load_tokens = ("忙", "好多", "压力", "撑不住", "难", "复杂", "overwhelmed", "busy", "hard")
        identity_tokens = ("你是谁", "你叫什么", "名字", "name", "who are you", "what are you")
        relation_tokens = ("记得我", "不记得我", "在乎", "陪我", "remember me", "forget me")

        semantic_valence = 0.0
        if any(token in lowered for token in negative_tokens):
            semantic_valence -= 0.42
        if any(token in lowered for token in positive_tokens):
            semantic_valence += 0.34
        semantic_valence += (closeness - 0.5) * 0.08

        fatigue_push = 0.62 if any(token in lowered for token in fatigue_tokens) else 0.0
        cognitive_load = 0.55 if any(token in lowered for token in load_tokens) else 0.0
        semantic_arousal = _clip(abs(semantic_valence) * 0.7 + cognitive_load * 0.35 + fatigue_push * 0.2, 0.0, 1.0)
        social_approach_pull = _clip(0.35 + closeness * 0.4 + max(semantic_valence, 0.0) * 0.2, 0.0, 1.0)
        inferred_energy_delta = -0.16 * fatigue_push - 0.08 * cognitive_load + max(semantic_valence, 0.0) * 0.04
        identity_salience = 0.78 if any(token in lowered for token in identity_tokens) else 0.0
        relation_charge = _clip((closeness - 0.5) * 0.6 + semantic_valence * 0.45, -1.0, 1.0)
        if any(token in lowered for token in relation_tokens):
            relation_charge = _clip(relation_charge - 0.12 if semantic_valence < 0 else relation_charge + 0.12, -1.0, 1.0)

        mood_band = "low" if state.mood + semantic_valence * 0.08 < 0.42 else "high" if state.mood + semantic_valence * 0.08 > 0.68 else "steady"
        energy_band = "low" if state.body_energy + inferred_energy_delta < 0.38 else "high" if state.body_energy + inferred_energy_delta > 0.72 else "steady"
        appraisal_band = "charged" if semantic_arousal > 0.58 else "muted" if abs(semantic_valence) < 0.08 and fatigue_push == 0.0 else "engaged"
        return {
            "semantic_valence": round(semantic_valence, 4),
            "semantic_arousal": round(semantic_arousal, 4),
            "social_approach_pull": round(social_approach_pull, 4),
            "cognitive_load": round(cognitive_load, 4),
            "fatigue_push": round(fatigue_push, 4),
            "inferred_energy_delta": round(inferred_energy_delta, 4),
            "identity_salience": round(identity_salience, 4),
            "relation_charge": round(relation_charge, 4),
            "mood_band": mood_band,
            "energy_band": energy_band,
            "appraisal_band": appraisal_band,
            "scenario": scenario,
            "mode": requested_mode,
        }

    def _build_chronic_signal(
        self,
        event: RoundEvent,
        *,
        state: RuntimeState,
        context: dict[str, Any],
        relation_state: dict[str, Any],
        scenario: str,
        window: int = 20,
    ) -> dict[str, Any]:
        recent = self.trace_store.recent_round_signal_views(limit=window)
        appraisal = dict(context.get("appraisal", {}))
        valences = [
            float(trace.get("appraisal_snapshot", {}).get("semantic_valence", 0.0) or 0.0)
            for trace in recent
        ]
        valences.append(float(appraisal.get("semantic_valence", 0.0) or 0.0))
        relation_charges = [
            float(trace.get("appraisal_snapshot", {}).get("relation_charge", 0.0) or 0.0)
            for trace in recent
        ]
        relation_charges.append(float(appraisal.get("relation_charge", 0.0) or 0.0))
        closeness_series = [
            float(trace.get("vitality_snapshot", {}).get("relationship_closeness", 0.5) or 0.5)
            for trace in recent
        ]
        closeness_series.append(float(relation_state.get("closeness", 0.5) or 0.5))
        resource_series = [
            float(trace.get("vitality_snapshot", {}).get("resource_scarcity", 0.0) or 0.0)
            for trace in recent
        ]
        resource_series.append(float(state.resource_state.get("scarcity_index", 0.0) or 0.0))
        identity_salience = [
            float(trace.get("appraisal_snapshot", {}).get("identity_salience", 0.0) or 0.0)
            for trace in recent
        ]
        identity_salience.append(float(appraisal.get("identity_salience", 0.0) or 0.0))
        recent_noninteractive = sum(
            1
            for trace in recent[-6:]
            if any(item.get("non_interactive") for item in trace.get("vitality_events", []))
        )
        resource_stress_span = 0
        for value in reversed(resource_series):
            if value >= 0.45:
                resource_stress_span += 1
                continue
            break
        rejection_count = sum(1 for charge in relation_charges if charge <= -0.12)
        confirmation_count = sum(1 for charge in relation_charges if charge >= 0.12)
        relation_trend = (
            round(closeness_series[-1] - closeness_series[0], 4)
            if len(closeness_series) >= 2
            else 0.0
        )
        return {
            "window_size": len(recent) + 1,
            "avg_positive_valence": round(sum(max(value, 0.0) for value in valences) / max(len(valences), 1), 4),
            "avg_negative_valence": round(sum(abs(min(value, 0.0)) for value in valences) / max(len(valences), 1), 4),
            "relation_trend": relation_trend,
            "repeated_rejection_count": rejection_count,
            "repeated_confirmation_count": confirmation_count,
            "resource_stress_span": resource_stress_span,
            "identity_cue_repeat_count": sum(1 for value in identity_salience if value >= 0.35),
            "noninteractive_residue": recent_noninteractive,
            "current_relation_charge": round(float(appraisal.get("relation_charge", 0.0) or 0.0), 4),
            "scenario": scenario,
        }

    def _event_with_appraisal(self, event: RoundEvent, appraisal: dict[str, Any]) -> RoundEvent:
        inferred_valence = float(appraisal.get("semantic_valence", 0.0))
        inferred_energy_delta = float(appraisal.get("inferred_energy_delta", 0.0))
        return RoundEvent(
            source=event.source,
            content=event.content,
            target=event.target,
            cue=event.cue,
            valence=event.valence if abs(event.valence) > 1e-9 else inferred_valence,
            energy_delta=event.energy_delta if abs(event.energy_delta) > 1e-9 else inferred_energy_delta,
            cue_quality=event.cue_quality,
        )

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

    def _run_policy(self, *, allow_commit: bool, operator_level: str) -> RunPolicy:
        return self.run_runtime._run_policy(allow_commit=allow_commit, operator_level=operator_level)

    def _run_budget(self) -> ExecutionBudget:
        return self.run_runtime._run_budget()

    def _dirty_worktree_snapshot(self) -> dict[str, Any]:
        return self.run_runtime._dirty_worktree_snapshot()

    def _plan_run_via_model(self, goal: str, repo_metadata: dict[str, Any]) -> dict[str, Any]:
        return self.run_runtime._plan_run_via_model(goal, repo_metadata)

    def _sync_run_state_to_runtime(self, state: RuntimeState, run_state: RunState) -> None:
        self.run_runtime._sync_run_state_to_runtime(state, run_state)

    def _current_task_node(self, run_state: RunState | dict[str, Any]) -> TaskNode | None:
        return self.run_runtime._current_task_node(run_state)

    def _identity_cfg(self) -> dict[str, Any]:
        return self.config["identity"]["identity"]

    def identity_config(self) -> dict[str, Any]:
        return self._identity_cfg()

    def _provider_descriptor_for_route(self, route_name: str = "renderer") -> tuple[str, str]:
        return self.model_routing_runtime._provider_descriptor_for_route(route_name)

    def _provider_descriptor(self) -> tuple[str, str]:
        return self.model_routing_runtime._provider_descriptor()

    def provider_descriptor(self) -> tuple[str, str]:
        return self._provider_descriptor()

    def _model_tiers(self) -> dict[str, Any]:
        return self.model_routing_runtime._model_tiers()

    def _agent_model_bindings(self) -> dict[str, str]:
        return self.model_routing_runtime._agent_model_bindings()

    def _module_model_bindings(self) -> dict[str, str]:
        return self.model_routing_runtime._module_model_bindings()

    def _agent_tier(self, binding_key: str) -> str:
        return self.model_routing_runtime._agent_tier(binding_key)

    def _effective_agent_tier(self, binding_key: str, *, metadata: dict[str, Any] | None = None) -> str:
        return self.model_routing_runtime._effective_agent_tier(binding_key, metadata=metadata)

    def _tier_config(self, tier_name: str) -> dict[str, Any]:
        return self.model_routing_runtime._tier_config(tier_name)

    def _infer_tier_name_for_route(self, route_cfg: ModelRouteConfig | None) -> str | None:
        return self.model_routing_runtime._infer_tier_name_for_route(route_cfg)

    def _route_policy_contract(self) -> dict[str, Any]:
        return self.model_routing_runtime._route_policy_contract()

    def _activation_thresholds_contract(self) -> dict[str, Any]:
        return self.model_routing_runtime._activation_thresholds_contract()

    def _memory_tier_policy_contract(self) -> dict[str, Any]:
        return self.model_routing_runtime._memory_tier_policy_contract()

    def _consolidation_policy_contract(self) -> dict[str, Any]:
        return self.model_routing_runtime._consolidation_policy_contract()

    def _route_config_for_binding(
        self,
        binding_key: str,
        *,
        route_name: str,
        metadata: dict[str, Any] | None = None,
    ) -> ModelRouteConfig | None:
        return self.model_routing_runtime._route_config_for_binding(
            binding_key,
            route_name=route_name,
            metadata=metadata,
        )

    def _record_model_call(
        self,
        bucket: list[dict[str, Any]],
        *,
        skill_name: str,
        binding_key: str,
        route_config: ModelRouteConfig,
        response,
        prompt_chars: int,
        parallel_group: str | None = None,
    ) -> None:
        self.model_routing_runtime._record_model_call(
            bucket,
            skill_name=skill_name,
            binding_key=binding_key,
            route_config=route_config,
            response=response,
            prompt_chars=prompt_chars,
            parallel_group=parallel_group,
        )

    def _call_bound_model_route(
        self,
        binding_key: str,
        *,
        route_name: str,
        request: ModelRequest,
        binding_metadata: dict[str, Any] | None = None,
        model_call_traces: list[dict[str, Any]] | None = None,
        skill_name: str | None = None,
        parallel_group: str | None = None,
    ):
        return self.model_routing_runtime._call_bound_model_route(
            binding_key,
            route_name=route_name,
            request=request,
            binding_metadata=binding_metadata,
            model_call_traces=model_call_traces,
            skill_name=skill_name,
            parallel_group=parallel_group,
        )

    def _compact_float(self, value: Any) -> float:
        return self.model_payload_runtime._compact_float(value)

    def _build_state_summary(self, state: RuntimeState) -> dict[str, Any]:
        return self.model_payload_runtime._build_state_summary(state)

    def _build_context_summary(self, context: dict[str, Any]) -> dict[str, Any]:
        return self.model_payload_runtime._build_context_summary(context)

    def _build_relation_summary(self, relation_state: dict[str, Any]) -> dict[str, Any]:
        return self.model_payload_runtime._build_relation_summary(relation_state)

    def _build_event_summary(self, event: RoundEvent) -> dict[str, Any]:
        return self.model_payload_runtime._build_event_summary(event)

    def _build_pfc_model_payload(
        self,
        event: RoundEvent,
        state: RuntimeState,
        scenario: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        return self.model_payload_runtime._build_pfc_model_payload(event, state, scenario, context)

    def _build_perspective_model_payload(
        self,
        event: RoundEvent,
        state: RuntimeState,
        scenario: dict[str, Any],
        context: dict[str, Any],
        relation_state: dict[str, Any],
        *,
        action_name: str,
        output_key: str,
    ) -> dict[str, Any]:
        return self.model_payload_runtime._build_perspective_model_payload(
            event,
            state,
            scenario,
            context,
            relation_state,
            action_name=action_name,
            output_key=output_key,
        )

    def _build_render_model_payload(self, render_plan: RenderPlan) -> dict[str, Any]:
        return self.model_payload_runtime._build_render_model_payload(render_plan)

    def _fallback_render_route_names(self) -> list[str]:
        return self.round_pipeline._fallback_render_route_names()

    def _build_deterministic_render_fallback_output(self, render_plan: RenderPlan, *, model_label: str) -> dict[str, Any]:
        return self.round_pipeline._build_deterministic_render_fallback_output(
            render_plan,
            model_label=model_label,
        )

    def _fallback_renderer_system_prompt(
        self,
        render_plan: RenderPlan,
        *,
        fallback_reason: str,
        violation_types: list[str] | None = None,
    ) -> str:
        return self.round_pipeline._fallback_renderer_system_prompt(
            render_plan,
            fallback_reason=fallback_reason,
            violation_types=violation_types,
        )

    def _build_fallback_render_model_payload(
        self,
        render_plan: RenderPlan,
        *,
        fallback_reason: str,
        violation_types: list[str] | None = None,
    ) -> dict[str, Any]:
        return self.round_pipeline._build_fallback_render_model_payload(
            render_plan,
            fallback_reason=fallback_reason,
            violation_types=violation_types,
        )

    def _render_expression_via_humanized_fallback_chain(
        self,
        render_plan: RenderPlan,
        *,
        fallback_reason: str,
        deterministic_model: str,
        violation_types: list[str] | None = None,
        model_call_traces: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        return self.round_pipeline._render_expression_via_humanized_fallback_chain(
            render_plan,
            fallback_reason=fallback_reason,
            deterministic_model=deterministic_model,
            violation_types=violation_types,
            model_call_traces=model_call_traces,
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
        return self.round_pipeline._execute_parallel_skills(
            round_id=round_id,
            tasks=tasks,
            skill_traces=skill_traces,
            runtime_context=runtime_context,
            parallel_traces=parallel_traces,
        )

    def _parallel_fallback_output(self, task: dict[str, Any]) -> Any:
        return self.round_pipeline._parallel_fallback_output(task)

    def _parallel_timeout_skill_result(
        self,
        *,
        round_id: int,
        task: dict[str, Any],
        latency_ms: int,
        failure_policy: str = "parallel_timeout",
    ) -> SkillResult:
        return self.round_pipeline._parallel_timeout_skill_result(
            round_id=round_id,
            task=task,
            latency_ms=latency_ms,
            failure_policy=failure_policy,
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
        return self.round_pipeline._action_evidence_from_contribution(
            contribution,
            priority_bucket=priority_bucket,
            control_domain=control_domain,
            gated_actions=gated_actions,
            risk_hints=risk_hints,
            veto=veto,
            trace_tags=trace_tags,
        )

    def _projected_action_delta_from_contribution(
        self,
        contribution: ProbabilisticContribution,
    ) -> dict[str, float]:
        return self.round_pipeline._projected_action_delta_from_contribution(contribution)

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
        return self.round_pipeline._action_signal_metadata(
            owner=owner,
            state=state,
            relation_state=relation_state,
            context=context,
            module_type=module_type,
            priority_bucket=priority_bucket,
            control_domain=control_domain,
            projected_delta=projected_delta,
            utility_shift=utility_shift,
            gated_actions=gated_actions,
            risk_hints=risk_hints,
            veto=veto,
            trace_tags=trace_tags,
        )

    def _coerce_action_signals(
        self,
        rows: list[ActionEvidenceSignal | ProbabilisticContribution],
    ) -> list[ActionEvidenceSignal]:
        return self.round_pipeline._coerce_action_signals(rows)

    def _build_value_action_contribution(
        self,
        value_scores: dict[str, Any],
    ) -> ProbabilisticContribution:
        return self.round_pipeline._build_value_action_contribution(value_scores)

    def _candidate_distribution_from_trace(self, trace: dict[str, Any]) -> dict[str, float]:
        return self.round_pipeline._candidate_distribution_from_trace(trace)

    def _probability_field_couplings(self) -> list[CrossLayerCouplingSpec]:
        return self.round_pipeline._probability_field_couplings()

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
        return self.round_pipeline._collect_probability_field_contributions(
            direct_action_contributions=direct_action_contributions,
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
            control_ledger=control_ledger,
            action_truth=action_truth,
        )

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
        impl = getattr(
            self.round_pipeline,
            "_integrate_probability_field_snapshot_impl",
            self.round_pipeline._integrate_probability_field_snapshot,
        )
        return impl(
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

    def _finalize_action_bookkeeping_from_action_layer(
        self,
        action_bookkeeping: ActionBookkeepingState,
        action_layer: dict[str, Any] | ProbabilityLayerState,
        *,
        finalize_stage: str,
    ) -> None:
        impl = getattr(
            self.round_pipeline,
            "_finalize_action_bookkeeping_from_action_layer_impl",
            self.round_pipeline._finalize_action_bookkeeping_from_action_layer,
        )
        return impl(
            action_bookkeeping,
            action_layer,
            finalize_stage=finalize_stage,
        )

    def _reintegrate_probability_snapshot(
        self,
        *,
        snapshot: ProbabilityFieldSnapshot,
        contributions: list[ProbabilisticContribution],
        token_state: TokenFieldState,
        source_chain: list[str] | None = None,
    ) -> ProbabilityFieldSnapshot:
        return self.round_pipeline._reintegrate_probability_snapshot(
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
        return self.round_pipeline._action_truth_from_field(
            action_layer,
            gate=gate,
            conflict_mode=conflict_mode,
        )

    def _refresh_action_truth(
        self,
        action_layer: ProbabilityLayerState | dict[str, Any],
        current_action_truth: dict[str, Any] | None = None,
        *,
        conflict_mode: str = "field_native",
    ) -> dict[str, Any]:
        return self.round_pipeline._refresh_action_truth(
            action_layer,
            current_action_truth=current_action_truth,
            conflict_mode=conflict_mode,
        )

    def _control_ledger_from_action_bookkeeping(
        self,
        action_bookkeeping: ActionBookkeepingState,
    ) -> dict[str, Any]:
        return self.round_pipeline._control_ledger_from_action_bookkeeping(action_bookkeeping)

    def _merge_control_ledger_into_action_bookkeeping(
        self,
        action_bookkeeping: ActionBookkeepingState,
        control_ledger: dict[str, Any],
    ) -> None:
        return self.round_pipeline._merge_control_ledger_into_action_bookkeeping(
            action_bookkeeping,
            control_ledger,
        )

    def _action_bookkeeping_payload(
        self,
        *,
        action_bookkeeping: ActionBookkeepingState,
        probability_field_snapshot: ProbabilityFieldSnapshot,
        control_ledger: dict[str, Any],
        action_truth: dict[str, Any],
        stochastic_state: StochasticState,
    ) -> dict[str, Any]:
        return self.round_pipeline._action_bookkeeping_payload(
            action_bookkeeping=action_bookkeeping,
            probability_field_snapshot=probability_field_snapshot,
            control_ledger=control_ledger,
            action_truth=action_truth,
            stochastic_state=stochastic_state,
        )

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
        return self.round_pipeline._build_distribution_delta_contribution(
            module_name=module_name,
            module_type=module_type,
            from_distribution=from_distribution,
            to_distribution=to_distribution,
            trace_reason=trace_reason,
            projection_reason=projection_reason,
            applied_at_stage=applied_at_stage,
            native_operator=native_operator,
            dependency_trace=dependency_trace,
            hard_mask=hard_mask,
            confidence=confidence,
            module_temperature=module_temperature,
        )

    def _unnamed_label(self) -> str:
        return self.identity_runtime.unnamed_label()

    def _normalize_identity_name(self, name: str | None) -> str:
        return self.identity_runtime.normalize_name(name)

    def _display_label(self, state: RuntimeState) -> str:
        return self.identity_runtime.display_label(state)

    def _set_identity_name(
        self,
        state: RuntimeState,
        name: str,
        *,
        source_hint: str,
        round_id: int | None = None,
        reset_baseline: bool = True,
    ) -> dict[str, Any] | None:
        normalized = self._normalize_identity_name(name)
        if not normalized:
            return None

        previous = state.identity_state.display_name
        if previous == normalized:
            if state.identity_state.name_source is None:
                state.identity_state.name_source = "user_seed" if previous is None and source_hint == "user_seed" else source_hint
            return None

        effective_source = source_hint
        if source_hint in {"user_seed", "manual_override"}:
            effective_source = "user_seed" if previous is None else "manual_override"
        if previous and previous not in state.identity_state.aliases:
            state.identity_state.aliases.append(previous)
        state.identity_state.display_name = normalized
        state.identity_state.name_source = effective_source
        state.identity_state.provider_disclosure_mode = str(self._identity_cfg().get("provider_disclosure_mode", "adaptive"))
        if reset_baseline:
            state.identity_state.evidence_signature = ""
            state.identity_state.evidence_anchors = []
            state.identity_state.drift_credit = 0.0
        if round_id is not None:
            state.identity_state.last_name_change_round = round_id
        return {"from": previous, "to": normalized, "source": effective_source}

    def seed_identity_name(self, name: str, *, source_hint: str = "user_seed") -> dict[str, Any]:
        state = self.load_runtime_state()
        rename_event = self._set_identity_name(state, name, source_hint=source_hint, round_id=state.round_count)
        if rename_event:
            self._save_state(state, sync=True)
        return self.identity_payload()

    def _initial_identity_threshold_met(self, evidence: dict[str, Any]) -> bool:
        return self.identity_runtime.initial_threshold_met(evidence)

    def _generate_identity_name(self, state: RuntimeState, evidence_signature: str) -> str:
        return self.identity_runtime.generate_identity_name(state, evidence_signature)

    def _anchor_drift_score(self, previous_anchors: list[str], current_anchors: list[str]) -> float:
        return self.identity_runtime.anchor_drift_score(previous_anchors, current_anchors)

    def _augment_identity_evidence(self, state: RuntimeState, evidence: dict[str, Any]) -> dict[str, Any]:
        return self.identity_runtime.augment_identity_evidence(state, evidence)

    def _maybe_update_identity_from_evidence(self, state: RuntimeState) -> dict[str, Any] | None:
        return self.identity_runtime.maybe_update_from_evidence(
            state,
            self.memory_store,
            round_id=state.round_count,
            set_identity_name=self._set_identity_name,
        )

    def _update_affect_residue(self, state: RuntimeState, event: RoundEvent, requested_mode: str) -> None:
        shock = abs(event.valence) * (1.12 if event.valence < 0 else 0.92)
        decay = 0.84
        input_scale = 0.42
        if requested_mode == "idle":
            decay = 0.90
            input_scale = 0.18
        elif requested_mode == "sleep":
            decay = 0.72
            input_scale = 0.08
        state.affect_residue = round(_clip(state.affect_residue * decay + shock * input_scale), 4)

    def _apply_noninteractive_shaping(
        self,
        state: RuntimeState,
        requested_mode: str,
        cue: str | None,
        relation_state: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        outcome = self.dream_orchestrator.run(
            state=state,
            mode=requested_mode,
            cue=cue,
            relation_state=relation_state,
        )
        return outcome["shaping_events"], outcome

    def _build_slow_variable_payload(
        self,
        *,
        state: RuntimeState,
        context: dict[str, Any],
        relation_state: dict[str, float],
        prior_closeness: float,
    ) -> dict[str, Any]:
        return self.vitality_engine.build_slow_variable_payload(
            state=state,
            context=context,
            relation_state=relation_state,
            prior_closeness=prior_closeness,
        )

    def _build_endogenous_trigger_context(
        self,
        *,
        state: RuntimeState,
        scenario: str,
        event: RoundEvent | None = None,
        trace_payload: dict[str, Any] | None = None,
    ) -> EndogenousTriggerContext:
        payload = dict(trace_payload or {})
        vitality = dict(payload.get("vitality_snapshot", {}) or {})
        source_round_id = payload.get("round_id")
        cue = payload.get("run_context", {}).get("cue") if isinstance(payload.get("run_context"), dict) else None
        if cue is None and event is not None:
            cue = event.cue
        context = {
            "cue": cue,
            "recall_strength": float(vitality.get("memory_activation", 0.0) or 0.0),
            "closeness": float(self.memory_store.closeness("self")),
            "interference": float(vitality.get("memory_interference", 0.0) or 0.0),
        }
        relation_state = {
            "relationship_risk": float(vitality.get("relationship_drift", 0.0) or 0.0),
            "closeness": float(context["closeness"]),
        }
        slow_variables = {
            "affect_residue": float(state.affect_residue),
            "memory_activation": float(vitality.get("memory_activation", 0.0) or 0.0),
            "relationship_drift": float(vitality.get("relationship_drift", 0.0) or 0.0),
            "resource_scarcity": float(vitality.get("resource_scarcity", 0.0) or 0.0),
        }
        return EndogenousTriggerContext(
            scenario=scenario,
            source_round_id=int(source_round_id) if isinstance(source_round_id, int) else None,
            context=context,
            relation_state=relation_state,
            slow_variables=slow_variables,
        )

    def _endogenous_suppression_decision(
        self,
        *,
        state: RuntimeState,
        scenario: str,
        trigger: EndogenousTickTrigger | None,
    ) -> EndogenousSuppressionDecision:
        if trigger is None:
            return EndogenousSuppressionDecision(suppressed=True, reason="no_trigger")
        if state.safe_mode or state.mode == "safe":
            return EndogenousSuppressionDecision(suppressed=True, reason="safe_mode")
        subjective_pressure = self._subjective_pressure(state)
        instinct_relax = (
            state.organic_mode.enabled
            and state.organic_mode.instinct_first
            and (
                (
                    float(state.organic_mode.endogenous_autonomy or 0.0) >= 0.7
                    and float(trigger.trigger_score or 0.0) >= max(0.55, 0.82 - float(state.organic_mode.endogenous_autonomy or 0.0) * 0.25)
                )
                or (
                    subjective_pressure >= 0.58
                    and float(trigger.trigger_score or 0.0) >= max(0.48, 0.76 - subjective_pressure * 0.18)
                )
            )
        )
        if scenario == "task" and not instinct_relax:
            return EndogenousSuppressionDecision(suppressed=True, reason="task_scenario")
        if scenario != "companion" and not instinct_relax:
            return EndogenousSuppressionDecision(suppressed=True, reason="non_companion_scenario")
        if state.active_run_id and state.run_status in {"running", "paused"}:
            return EndogenousSuppressionDecision(
                suppressed=True,
                reason="active_run",
                details={"run_id": state.active_run_id, "run_status": state.run_status},
            )
        threshold = 0.7 if not instinct_relax else max(
            0.46,
            0.7 - float(state.organic_mode.endogenous_autonomy or 0.0) * 0.2 - subjective_pressure * 0.12,
        )
        if float(trigger.trigger_score or 0.0) < threshold:
            return EndogenousSuppressionDecision(
                suppressed=True,
                reason="score_below_auto_threshold",
                details={
                    "trigger_score": float(trigger.trigger_score or 0.0),
                    "threshold": threshold,
                    "subjective_pressure": subjective_pressure,
                },
            )
        return EndogenousSuppressionDecision(suppressed=False, reason="")

    def _build_endogenous_micro_intent(
        self,
        *,
        state: RuntimeState,
        trigger: EndogenousTickTrigger,
        pool_state,
    ) -> EndogenousMicroIntent:
        first_motivation = pool_state.active_motivations[0].motivation_type if pool_state.active_motivations else "latent"
        previous_intent = state.endogenous_state.current_intent
        stability = state.endogenous_state.stability
        if previous_intent is not None and previous_intent.name == first_motivation:
            stability += 1
        else:
            stability = 1
        return EndogenousMicroIntent(
            name=first_motivation,
            trigger=trigger.trigger_type,
            bias=pool_state.active_motivations[0].target_actions if pool_state.active_motivations else {},
            evidence=trigger.source_metrics,
            stability=stability,
            source_round_id=state.round_count,
        )

    def _build_grounding_capsule(
        self,
        *,
        state: RuntimeState,
        context: dict[str, Any],
        relation_state: dict[str, float],
        slow_variables: dict[str, Any],
        shaping_events: list[dict[str, Any]],
        rename_event: dict[str, Any] | None,
    ) -> dict[str, Any]:
        compact_slow_variables = {
            key: self._compact_float(value)
            for key, value in slow_variables.items()
            if key in {"resource_scarcity", "memory_activation", "relationship_heat", "identity_salience", "relationship_drift"}
        }
        state_sources = []
        if context.get("cue"):
            state_sources.append("memory_cue")
        if state.focus:
            state_sources.append("focus")
        if rename_event is not None:
            state_sources.append("identity_rename")
        state_sources.extend(self.identity_runtime.shaping_sources(shaping_events, slow_variables))
        return {
            "state_sources": sorted(set(state_sources)),
            "state_summary": self._build_state_summary(state),
            "context_summary": self._build_context_summary(context),
            "relation_state": self._build_relation_summary(relation_state),
            "slow_variables": compact_slow_variables,
        }

    def _build_vitality_snapshot(
        self,
        *,
        state: RuntimeState,
        context: dict[str, Any],
        relation_state: dict[str, float],
        prior_closeness: float,
        scenario: str,
        sampled_action: ActionCandidate,
        contributions: list[AgentContribution],
        gate_decisions: list[dict[str, Any]],
        shaping_events: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return self.vitality_engine.build_vitality_snapshot(
            state=state,
            context=context,
            relation_state=relation_state,
            prior_closeness=prior_closeness,
            scenario=scenario,
            sampled_action=sampled_action,
            contributions=contributions,
            gate_decisions=gate_decisions,
            shaping_events=shaping_events,
        )

    def _classify_identity_query(self, text: str) -> str:
        return self.identity_runtime.classify_query(text)

    def _infer_query_intent(
        self,
        *,
        event: RoundEvent,
        scenario: str,
        state: RuntimeState,
        relation_state: dict[str, float],
        slow_variables: dict[str, Any],
    ) -> QueryIntentState:
        return self.identity_runtime.intent_runtime.infer_query_intent(
            text=event.content,
            scenario=scenario,
            state=state,
            relation_state=relation_state,
            slow_variables=slow_variables,
        )

    def _infer_disclosure_intent(
        self,
        *,
        query_state: QueryIntentState,
        scenario: str,
        state: RuntimeState,
        relation_state: dict[str, float],
        slow_variables: dict[str, Any],
    ) -> DisclosureIntentState:
        return self.identity_runtime.intent_runtime.infer_disclosure_intent(
            query_state=query_state,
            scenario=scenario,
            state=state,
            relation_state=relation_state,
            slow_variables=slow_variables,
        )

    def _build_identity_context(
        self,
        *,
        query_state: QueryIntentState,
        disclosure_state: DisclosureIntentState,
        scenario: str,
        state: RuntimeState,
        shaping_events: list[dict[str, Any]],
        slow_variables: dict[str, Any],
        state_sources: list[str],
        rename_event: dict[str, Any] | None,
        ) -> IdentityContext:
        return self.identity_runtime.build_identity_context(
            query_state=query_state,
            disclosure_state=disclosure_state,
            scenario=scenario,
            state=state,
            shaping_events=shaping_events,
            slow_variables=slow_variables,
            state_sources=state_sources,
            rename_event=rename_event,
        )

    def _build_repair_expression_policy(self, conflict_state: dict[str, Any]) -> dict[str, Any]:
        impl = getattr(
            self.round_pipeline,
            "_build_repair_expression_policy_impl",
            self.round_pipeline._build_repair_expression_policy,
        )
        return impl(conflict_state)

    def _renderer_system_prompt(
        self,
        render_plan: RenderPlan,
        *,
        prompt_mode: str = "base",
        violation_types: list[str] | None = None,
    ) -> str:
        impl = getattr(
            self.round_pipeline,
            "_renderer_system_prompt_impl",
            self.round_pipeline._renderer_system_prompt,
        )
        return impl(
            render_plan,
            prompt_mode=prompt_mode,
            violation_types=violation_types,
        )

    def _evaluate_authenticity(self, text: str, render_plan: RenderPlan) -> dict[str, Any]:
        impl = getattr(
            self.round_pipeline,
            "_evaluate_authenticity_impl",
            self.round_pipeline._evaluate_authenticity,
        )
        return impl(text, render_plan)

    def _coerce_score_map(self, value: Any) -> dict[str, float]:
        impl = getattr(
            self.round_pipeline,
            "_coerce_score_map_impl",
            self.round_pipeline._coerce_score_map,
        )
        return impl(value)

    def _generate_pfc_candidates_via_model(
        self,
        event: RoundEvent,
        state: RuntimeState,
        scenario: dict[str, Any],
        context: dict[str, Any],
        *,
        model_call_traces: list[dict[str, Any]] | None = None,
    ) -> ProbabilisticContribution:
        impl = getattr(
            self.round_pipeline,
            "_generate_pfc_candidates_via_model_impl",
            self.round_pipeline._generate_pfc_candidates_via_model,
        )
        return impl(
            event,
            state,
            scenario,
            context,
            model_call_traces=model_call_traces,
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
        impl = getattr(
            self.round_pipeline,
            "_invoke_pfc_model_generator_impl",
            self.round_pipeline._invoke_pfc_model_generator,
        )
        return impl(
            event,
            state,
            scenario,
            context,
            model_call_traces=model_call_traces,
        )

    def _should_run_late_perspective(
        self,
        event: RoundEvent,
        state: RuntimeState,
        relation_state: dict[str, Any],
        sampled_action: str,
        *,
        route_type: str = "",
    ) -> tuple[bool, str]:
        impl = getattr(
            self.round_pipeline,
            "_should_run_late_perspective_impl",
            self.round_pipeline._should_run_late_perspective,
        )
        return impl(
            event,
            state,
            relation_state,
            sampled_action,
            route_type=route_type,
        )

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
        impl = getattr(
            self.round_pipeline,
            "_infer_other_state_via_model_impl",
            self.round_pipeline._infer_other_state_via_model,
        )
        return impl(
            event,
            state,
            scenario,
            context,
            sampled_action,
            relation_state,
            model_call_traces=model_call_traces,
            parallel_group=parallel_group,
        )

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
        impl = getattr(
            self.round_pipeline,
            "_simulate_other_reaction_via_model_impl",
            self.round_pipeline._simulate_other_reaction_via_model,
        )
        return impl(
            event,
            state,
            scenario,
            context,
            sampled_action,
            relation_state,
            model_call_traces=model_call_traces,
            parallel_group=parallel_group,
        )

    def _render_expression_via_model(
        self,
        render_plan: RenderPlan,
        *,
        prompt_mode: str = "base",
        violation_types: list[str] | None = None,
        model_call_traces: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        impl = getattr(
            self.round_pipeline,
            "_render_expression_via_model_impl",
            self.round_pipeline._render_expression_via_model,
        )
        return impl(
            render_plan,
            prompt_mode=prompt_mode,
            violation_types=violation_types,
            model_call_traces=model_call_traces,
        )

    def _should_allow_renderer_resample(
        self,
        *,
        route_type: str,
        render_plan: RenderPlan,
        violation_types: list[str],
        context: dict[str, Any],
    ) -> bool:
        impl = getattr(
            self.round_pipeline,
            "_should_allow_renderer_resample_impl",
            self.round_pipeline._should_allow_renderer_resample,
        )
        return impl(
            route_type=route_type,
            render_plan=render_plan,
            violation_types=violation_types,
            context=context,
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
        impl = getattr(
            self.round_pipeline,
            "_score_salience_via_model_impl",
            self.round_pipeline._score_salience_via_model,
        )
        return impl(
            event,
            state,
            scenario,
            context,
            model_call_traces=model_call_traces,
            parallel_group=parallel_group,
        )

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
        impl = getattr(
            self.round_pipeline,
            "_estimate_subjective_value_via_model_impl",
            self.round_pipeline._estimate_subjective_value_via_model,
        )
        return impl(
            event,
            state,
            scenario,
            context,
            model_call_traces=model_call_traces,
            parallel_group=parallel_group,
        )

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

    def _resource_bias_snapshot(self, state: RuntimeState, context: dict[str, Any]) -> dict[str, float]:
        telemetry = self._compute_resource_telemetry(state)
        return {
            "scarcity_index": float(telemetry["scarcity_index"]),
            "burn_rate_ratio": float(telemetry["burn_rate_ratio"]),
            "low_balance_ratio": float(telemetry["low_balance_ratio"]),
            "queue_pressure": float(telemetry["queue_pressure"]),
            "latency_pressure": float(telemetry["latency_pressure"]),
            "resource_mode": str(telemetry["resource_mode"]),
            "scarcity_pressure": float(telemetry["scarcity_pressure"]),
            "body_hunger_bias": float(telemetry["body_hunger_bias"]),
            "effort_avoidance_bias": float(telemetry["effort_avoidance_bias"]),
            "deliberation_compress": float(telemetry["deliberation_compress"]),
            "rumination_bias": float(telemetry["rumination_bias"]),
            "action_shrink_scale": float(telemetry["action_shrink_scale"]),
            "queue_depth": int(telemetry["queue_depth"]),
            "average_latency_ms": float(telemetry["average_latency_ms"]),
        }

    def _normalize_temperament_runtime_state(self, state: RuntimeState, baseline_patch: dict[str, Any] | None = None) -> None:
        structured = self._normalize_temperament_state(state)
        baseline = dict(self.config["temperament"]["temperament"])
        baseline.update(structured.get("baseline", {}))
        if baseline_patch:
            baseline.update({key: value for key, value in baseline_patch.items() if isinstance(value, (int, float))})
        baseline = {key: round(_clip(float(value), 0.0, 1.0), 4) for key, value in baseline.items()}
        drift = {key: round(float(structured.get("drift", {}).get(key, 0.0)), 4) for key in baseline}
        current = {key: round(_clip(baseline[key] + drift.get(key, 0.0), 0.0, 1.0), 4) for key in baseline}
        state.temperament_state = {
            "baseline": baseline,
            "drift": drift,
            "current": current,
            "correction_window": dict(structured.get("correction_window", {})),
            "freeze_until_round": dict(structured.get("freeze_until_round", {})),
            "last_correction_events": list(structured.get("last_correction_events", [])),
            "drift_diagnostics": dict(structured.get("drift_diagnostics", {})),
        }

    def _record_entropy_failure(self, state: RuntimeState, exc: QuantumEntropyUnavailableError) -> None:
        state.entropy_health_state = self.entropy_pool.health_snapshot()
        state.last_entropy_failure = {
            "node_name": exc.node_name,
            "purpose": exc.purpose,
            "failure_class": exc.failure_class,
            "detail": exc.detail,
            "blocked_by_entropy": True,
            "recorded_at": utc_now_iso(),
        }
        self._save_state(state)

    def _build_base_distribution(self, state: RuntimeState, scenario_cfg: dict[str, Any], mode_cfg: dict[str, Any], relation_state: dict[str, float]) -> dict[str, float]:
        impl = getattr(
            self.round_pipeline,
            "_build_base_distribution_impl",
            self.round_pipeline._build_base_distribution,
        )
        return impl(state, scenario_cfg, mode_cfg, relation_state)

    def _compute_context_delta(self, state: RuntimeState, scenario_cfg: dict[str, Any]) -> dict[str, float]:
        impl = getattr(
            self.round_pipeline,
            "_compute_context_delta_impl",
            self.round_pipeline._compute_context_delta,
        )
        return impl(state, scenario_cfg)

    def _update_ci(self, state: RuntimeState, actions: list[str], context: dict[str, Any], scenario_cfg: dict[str, Any], thresholds: dict[str, Any]) -> dict[str, float]:
        impl = getattr(
            self.round_pipeline,
            "_update_ci_impl",
            self.round_pipeline._update_ci,
        )
        return impl(state, actions, context, scenario_cfg, thresholds)

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
        impl = getattr(
            self.round_pipeline,
            "_build_action_bookkeeping_impl",
            self.round_pipeline._build_action_bookkeeping,
        )
        return impl(
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

    def _apply_conflict_scales(
        self,
        action_truth: dict[str, Any],
        action_bookkeeping: ActionBookkeepingState | dict[str, Any],
        action_scales: dict[str, float],
        thresholds: dict[str, Any],
        hard_blocked_actions: list[str] | None = None,
    ) -> dict[str, Any]:
        bookkeeping = (
            {
                "risk_suppressor": action_bookkeeping.risk_suppressor,
                "gate": action_bookkeeping.gate,
            }
            if isinstance(action_bookkeeping, ActionBookkeepingState)
            else action_bookkeeping
        )
        distribution = {
            str(action): float(value)
            for action, value in dict(action_truth.get("winner_posterior", {}) or {}).items()
        }
        gate = {
            str(action): float(value)
            for action, value in dict(action_truth.get("gate", {}) or {}).items()
        }
        hard_masked = {
            str(action)
            for action in list(action_truth.get("hard_masked_targets", []) or [])
        }
        for action, scale in action_scales.items():
            if action not in distribution:
                continue
            bounded_scale = _clip(scale, 0.0, 1.50)
            bookkeeping.setdefault("risk_suppressor", {})
            bookkeeping["risk_suppressor"][action] = min(
                float(bookkeeping["risk_suppressor"].get(action, 1.0) or 1.0),
                bounded_scale,
            )
            distribution[action] = _clip(
                distribution[action] * bounded_scale,
                thresholds["p_floor"],
                thresholds["p_cap"],
            )
            if bounded_scale <= 0.0:
                bookkeeping.setdefault("gate", {})
                bookkeeping["gate"][action] = 0.0
                gate[action] = 0.0
                hard_masked.add(action)
        for action in hard_blocked_actions or []:
            if action not in distribution:
                continue
            bookkeeping.setdefault("risk_suppressor", {})
            bookkeeping.setdefault("gate", {})
            bookkeeping["risk_suppressor"][action] = 0.0
            bookkeeping["gate"][action] = 0.0
            distribution[action] = thresholds["p_floor"]
            gate[action] = 0.0
            hard_masked.add(action)
        action_truth["winner_posterior"] = distribution
        action_truth["gate"] = gate
        action_truth["hard_masked_targets"] = sorted(hard_masked)
        return action_truth

    def _top_action_name(self, distribution: dict[str, float]) -> str | None:
        impl = getattr(
            self.round_pipeline,
            "_top_action_name_impl",
            self.round_pipeline._top_action_name,
        )
        return impl(distribution)

    def _safe_mode_delta(self, before: bool, after: bool) -> str:
        impl = getattr(
            self.round_pipeline,
            "_safe_mode_delta_impl",
            self.round_pipeline._safe_mode_delta,
        )
        return impl(before, after)

    def _update_conflict_circuit(
        self,
        state: RuntimeState,
        *,
        critical_conflict: bool,
        winning_priority: str | None,
        compromise_template: str | None,
    ) -> dict[str, Any]:
        impl = getattr(
            self.round_pipeline,
            "_update_conflict_circuit_impl",
            self.round_pipeline._update_conflict_circuit,
        )
        return impl(
            state,
            critical_conflict=critical_conflict,
            winning_priority=winning_priority,
            compromise_template=compromise_template,
        )

    def _apply_conflict_repair_output(
        self,
        state: RuntimeState,
        *,
        repair_output: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
        state.repair_mode = repair_output.get("repair_mode")
        state.safe_mode = bool(repair_output.get("safe_mode_patch", state.safe_mode))
        state.conflict_safe_mode_owner = repair_output.get("conflict_safe_mode_owner")
        state.conflict_recovery_rounds = int(
            repair_output.get("repair_cooldown_rounds_patch", state.conflict_recovery_rounds)
        )

        adjustment = ConflictPostErrorAdjustment(**repair_output.get("post_error_adjustment", {"triggered": False}))
        if adjustment.triggered:
            state.last_post_error_adjustment = adjustment

        ledger_append_payload = dict(repair_output.get("repair_ledger_append", {}))
        conflict_context = dict(repair_output.get("conflict_learning_context", {}))
        if ledger_append_payload:
            learning_state = dict(state.conflict_learning_state or {})
            reason_counts = dict(learning_state.get("adjustment_reasons", {}))
            template_counts = dict(learning_state.get("template_counts", {}))
            blocked_action_counts = dict(learning_state.get("blocked_action_counts", {}))
            reason = str(ledger_append_payload.get("reason") or "idle")
            template = str(ledger_append_payload.get("template") or "none")
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
            template_counts[template] = template_counts.get(template, 0) + 1
            for action_name in ledger_append_payload.get("blocked_actions", []):
                blocked_action_counts[action_name] = blocked_action_counts.get(action_name, 0) + 1
            total_adjustments = sum(reason_counts.values()) or 1
            learning_signal = {
                "resample_budget_bias": round(min(0.20, reason_counts[reason] * 0.02), 4),
                "template_preference_bias": round(template_counts[template] / total_adjustments, 4),
                "blocked_action_persistence": round(min(0.50, max(blocked_action_counts.values(), default=0) * 0.05), 4),
                "cooldown_length": int(2 + min(3, reason_counts[reason] // 2)),
            }
            ledger_append_payload.update(
                {
                    "conflict_score": round(float(conflict_context.get("conflict_score", 0.0)), 4),
                    "dominant_conflicts": list(conflict_context.get("dominant_conflicts", [])),
                    "pass_count": int(conflict_context.get("pass_count", 0)),
                    "resample_count": int(conflict_context.get("resample_count", 0)),
                    "safe_mode_owned": bool(conflict_context.get("safe_mode_owned", False)),
                    "action_delta_summary": {
                        "top_action_before": ledger_append_payload.get("top_action_before"),
                        "top_action_after": ledger_append_payload.get("top_action_after"),
                        "blocked_actions_added": len(ledger_append_payload.get("blocked_actions", [])),
                    },
                    "learning_signal": learning_signal,
                    "stage_before": str(conflict_context.get("stage_before", state.repair_state.stage)),
                    "stage_after": str(ledger_append_payload.get("repair_stage_after", state.repair_state.stage)),
                }
            )
            state.conflict_learning_state = {
                "adjustment_reasons": reason_counts,
                "template_counts": template_counts,
                "blocked_action_counts": blocked_action_counts,
                "last_learning_signal": learning_signal,
                "last_stage": ledger_append_payload["stage_after"],
            }
            state.repair_ledger.append(ConflictRepairLedgerEntry(**ledger_append_payload))

        state.repair_state = ConflictRepairState(
            **repair_output.get("repair_state_snapshot", {"stage": state.repair_state.stage})
        )
        transition = dict(repair_output.get("repair_transition", {}))
        return (
            to_dict(adjustment),
            transition,
            to_dict(state.repair_state),
            [to_dict(item) for item in state.repair_ledger[-5:]],
        )

    def _apply_conflict_expression_adjustments(
        self,
        expression: ExpressionProfile,
        conflict_state: dict[str, Any],
    ) -> ExpressionProfile:
        impl = getattr(
            self.round_pipeline,
            "_apply_conflict_expression_adjustments_impl",
            self.round_pipeline._apply_conflict_expression_adjustments,
        )
        return impl(expression, conflict_state)

    def _run_conflict_controller(
        self,
        *,
        state: RuntimeState,
        signals: list[ActionEvidenceSignal],
        control_ledger: dict[str, Any],
        action_truth: dict[str, Any] | None,
        thresholds: dict[str, Any],
        skill_traces: list[dict[str, Any]],
        runtime_context: SkillRuntimeContext,
        gate_decisions: list[dict[str, Any]],
        round_seed: int,
    ) -> tuple[float, dict[str, Any], dict[str, Any]]:
        conflict_agent = self.agent_map["ConflictMonitorAgent"]
        hot_active = state.conflict_hot_rounds > 0
        safe_mode_before = state.safe_mode
        current_action_truth = dict(action_truth or {})
        if not current_action_truth:
            current_action_truth = self._action_truth_from_field(
                {
                    "winner_posterior": {},
                    "final_energy": {},
                },
                gate=dict(control_ledger.get("gate", {}) or {}),
                conflict_mode=str(control_ledger.get("conflict_mode", "none") or "none"),
            )
        top_action_before = self._top_action_name(
            dict(current_action_truth.get("winner_posterior", {}) or {})
        )
        assessment: dict[str, Any] = {"score": 0.0, "components": {}, "priority_signals": {}, "critical_conflict": False}
        resolution: dict[str, Any] = {
            "flag": False,
            "winning_priority": None,
            "blocked_actions": [],
            "action_scales": {},
            "applied_template": None,
            "repair_mode": state.repair_mode,
            "post_error_adjustment": {"triggered": False},
            "repair_transition": {},
            "repair_state_snapshot": {"stage": state.repair_state.stage},
            "repair_ledger_tail": [],
        }
        resample_policy: dict[str, Any] = {"flag": False, "allowed_resamples": 0, "force_compromise": False}
        peak_assessment: dict[str, Any] = dict(assessment)
        effective_resolution: dict[str, Any] = dict(resolution)
        critical_seen = False
        force_compromise_seen = False
        max_allowed_resamples = 0
        pass_records: list[dict[str, Any]] = []

        for pass_index in range(3):
            probability_field_view = {
                "action": dict(current_action_truth or {}),
                "source_chain": ["conflict_runtime"],
            }
            assessment = self._execute_skill(
                round_id=state.round_count,
                skill_name="score_conflict",
                inputs={"signals": signals, "probability_field": probability_field_view},
                provider=lambda signals, probability_field: conflict_agent.run_skill("score_conflict", signals, probability_field),
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
            if float(assessment.get("score", 0.0)) >= float(peak_assessment.get("score", -1.0)):
                peak_assessment = dict(assessment)
            critical_seen = critical_seen or bool(assessment.get("critical_conflict"))

            resolution = self._execute_skill(
                round_id=state.round_count,
                skill_name="trigger_control_escalation",
                inputs={
                    "assessment": assessment,
                    "probability_field": probability_field_view,
                    "conflict_arbitration": dict(control_ledger.get("conflict", {}) or {}),
                    "attempts": pass_index,
                    "hot_active": hot_active,
                },
                provider=lambda assessment, probability_field, conflict_arbitration, attempts, hot_active: conflict_agent.run_skill(
                    "trigger_control_escalation",
                    assessment,
                    probability_field,
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
            force_compromise_seen = force_compromise_seen or bool(resample_policy.get("force_compromise"))
            max_allowed_resamples = max(max_allowed_resamples, int(resample_policy.get("allowed_resamples", 0)))
            if resolution.get("flag"):
                effective_resolution = dict(resolution)
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
            control_ledger["resample_idx"] = int(control_ledger.get("resample_idx", 0) or 0) + 1

        effective_assessment = dict(peak_assessment)
        repair_components = {
            str(name): float(value)
            for name, value in dict(effective_assessment.get("components", {}) or {}).items()
        }
        sustained_repair_pressure = (
            float(repair_components.get("body_gap", 0.0)) >= 0.92
            or float(repair_components.get("veto_tension", 0.0)) >= 0.32
            or float(repair_components.get("relation_risk_gap", 0.0)) >= 0.42
        )
        sustained_critical = (
            (bool(state.repair_state.active) or bool(state.repair_ledger))
            and (
                bool(effective_resolution.get("flag"))
                or (
                    float(effective_assessment.get("score", 0.0))
                    >= max(0.64, float(conflict_agent.conflict_high) - 0.11)
                    and sustained_repair_pressure
                )
            )
        )
        effective_assessment["critical_conflict"] = (
            critical_seen
            or bool(effective_assessment.get("critical_conflict"))
            or sustained_critical
        )
        resolution = dict(effective_resolution if effective_resolution.get("flag") else resolution)
        effective_resample_policy = {
            "flag": bool(resample_policy.get("flag")),
            "allowed_resamples": max_allowed_resamples,
            "force_compromise": force_compromise_seen,
            "critical_conflict": critical_seen,
        }
        compromise = {
            "triggered": False,
            "template": resolution.get("applied_template"),
            "winning_priority": resolution.get("winning_priority"),
            "reason": "",
        }
        if resolution.get("flag") and effective_resample_policy.get("force_compromise") and not effective_resample_policy.get("flag"):
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

        circuit = self._update_conflict_circuit(
            state,
            critical_conflict=bool(effective_assessment.get("critical_conflict")),
            winning_priority=resolution.get("winning_priority"),
            compromise_template=compromise["template"] if compromise["triggered"] else None,
        )
        blocked_by_circuit = []
        if circuit["active"]:
            high_risk_actions = ("connect", "plan", "wander")
            for action in high_risk_actions:
                blocked_by_circuit.append(action)
        current_action_truth["conflict_mode"] = str(control_ledger.get("conflict_mode", "monitor") or "monitor")

        deadlock_fuse_triggered = bool(circuit["triggered"])
        top_action_after = self._top_action_name(dict(current_action_truth.get("winner_posterior", {}) or {}))
        repair_output = self._execute_skill(
            round_id=state.round_count,
            skill_name="mark_post_error_adjustment",
            inputs={
                "state": state,
                "round_id": state.round_count,
                "resolution": resolution,
                "compromise": compromise,
                "deadlock_fuse_triggered": deadlock_fuse_triggered,
                "top_action_before": top_action_before,
                "top_action_after": top_action_after,
                "blocked_actions": [*resolution.get("blocked_actions", []), *blocked_by_circuit],
                "safe_mode_before": safe_mode_before,
            },
            provider=lambda state, round_id, resolution, compromise, deadlock_fuse_triggered, top_action_before, top_action_after, blocked_actions, safe_mode_before: conflict_agent.run_skill(
                "mark_post_error_adjustment",
                state,
                round_id,
                resolution,
                compromise,
                deadlock_fuse_triggered,
                top_action_before,
                top_action_after,
                blocked_actions,
                safe_mode_before,
            ),
            skill_traces=skill_traces,
            runtime_context=runtime_context,
            seed_ref=round_seed,
        )
        post_error_adjustment, repair_transition, repair_state_snapshot, repair_ledger_tail = self._apply_conflict_repair_output(
            state,
            repair_output={
                **repair_output,
                "conflict_learning_context": {
                    "conflict_score": effective_assessment.get("score", 0.0),
                    "dominant_conflicts": list(effective_assessment.get("dominant_conflicts", [])),
                    "pass_count": len(pass_records),
                    "resample_count": int(control_ledger.get("resample_idx", 0) or 0),
                    "safe_mode_owned": state.conflict_safe_mode_owner == "conflict",
                    "stage_before": state.repair_state.stage,
                },
            },
        )
        resolution["repair_mode"] = repair_output.get("repair_mode")
        resolution["post_error_adjustment"] = post_error_adjustment
        resolution["repair_transition"] = repair_transition
        resolution["repair_state_snapshot"] = repair_state_snapshot
        resolution["repair_ledger_tail"] = repair_ledger_tail

        control_ledger["conflict"] = {
            "score": effective_assessment.get("score", 0.0),
            "total_score": effective_assessment.get("score", 0.0),
            "components": dict(effective_assessment.get("components", {})),
            "priority_signals": dict(effective_assessment.get("priority_signals", {})),
            "dominant_conflicts": list(effective_assessment.get("dominant_conflicts", [])),
            "winning_priority": resolution.get("winning_priority"),
            "passes": pass_records,
            "compromise": compromise,
            "critical_conflict": bool(effective_assessment.get("critical_conflict")),
            "critical_conflict_streak": state.critical_conflict_streak,
            "circuit_breaker": {**circuit, "blocked_actions": blocked_by_circuit},
            "allowed_resamples": effective_resample_policy.get("allowed_resamples", 0),
            "deadlock_fuse_triggered": deadlock_fuse_triggered,
            "repair_mode": state.repair_mode,
            "post_error_adjustment": post_error_adjustment,
            "repair_transition": repair_transition,
            "repair_state_snapshot": repair_state_snapshot,
            "repair_ledger_tail": repair_ledger_tail,
            "conflict_safe_mode_owned": state.conflict_safe_mode_owner == "conflict",
        }
        control_ledger["conflict_mode"] = (
            "repair"
            if state.repair_state.active
            else ("recovered" if state.repair_state.stage == "recovered" else "monitor")
        )
        return float(effective_assessment.get("score", 0.0)), current_action_truth, control_ledger

    def _normalize(self, distribution: dict[str, float]) -> dict[str, float]:
        return normalize_distribution(distribution)

    def _sample_action_from_distribution(self, distribution: dict[str, float], sample_value: float = 0.5) -> ActionCandidate:
        sampled_name = sample_action_name(distribution, sample_value)
        return ActionCandidate(
            name=sampled_name,
            probability=float(distribution.get(sampled_name, 0.0) or 0.0),
            rationale="controller sample",
        )

    def _softmax(self, utilities: dict[str, float]) -> dict[str, float]:
        return softmax_distribution(utilities)

    def _parse_budget_cap(self, value: str) -> tuple[int, float]:
        normalized = value.strip().lower().replace("_", "")
        if normalized.endswith("k"):
            cap_value = int(float(normalized[:-1]) * 1000)
        else:
            cap_value = int(float(normalized))
        return cap_value, round(_clip(cap_value / 100000, 0.0, 1.0), 4)

    def _kl_divergence(self, q_dist: dict[str, float], p_dist: dict[str, float]) -> float:
        return kl_divergence(q_dist, p_dist)

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

    def _apply_stochastic_layer(
        self,
        deterministic: dict[str, float],
        action_bookkeeping: ActionBookkeepingState,
        control_ledger: dict[str, Any] | None,
        state: RuntimeState,
        event: RoundEvent,
        scenario_cfg: dict[str, Any],
        relation_state: dict[str, float],
        conflict_score: float,
        round_seed: int,
        action_energy: dict[str, float],
    ) -> tuple[dict[str, float], StochasticState]:
        base_energy = (
            {
                str(action): float(value)
                for action, value in dict(action_energy or {}).items()
                if float(value) != float("-inf")
            }
            or {action: math.log(max(value, 1e-9)) for action, value in deterministic.items()}
        )
        base_stochastic = self._softmax(base_energy)
        base_top_action = self._top_action_name(base_stochastic) or ""
        ordered_base = sorted(base_stochastic.values(), reverse=True)
        top_gap = (ordered_base[0] - ordered_base[1]) if len(ordered_base) > 1 else 1.0
        emo_channel = self._stochastic_channel(base_stochastic)
        novelty = _clip(1.0 - max(base_stochastic.values(), default=0.0), 0.0, 1.0)
        emotion_volatility = _clip(abs(event.valence - (state.mood - 0.5)), 0.0, 1.0)
        fatigue = _clip(1.0 - state.body_energy, 0.0, 1.0)
        privacy_shift = _clip(
            abs(relation_state["privacy_level"] - float(state.session_metadata.get("last_privacy_level", relation_state["privacy_level"]))),
            0.0,
            1.0,
        )
        location_shift = _clip(1.0 if event.target and event.target != state.session_metadata.get("last_target") else 0.0, 0.0, 1.0)
        circadian_hour = time.localtime().tm_hour
        circadian_offset = _clip(abs(circadian_hour - 14) / 14.0, 0.0, 1.0)
        resource_scarcity = _clip(float(state.resource_state.get("scarcity_index", 1.0 - state.budget_remaining)), 0.0, 1.0)
        salience_lock = _clip(state.focus_lock_count / 6.0, 0.0, 1.0)
        ci_payload = (
            dict(control_ledger.get("ci", {}) or {})
            if isinstance(control_ledger, dict)
            else dict(action_bookkeeping.ci or {})
        )
        ci_state = {
            str(action): float(value)
            for action, value in ci_payload.items()
            if isinstance(action, str) and action
        }
        habit_strength = _clip(1.0 - (sum(ci_state.values()) / max(len(ci_state), 1)) / 0.55, 0.0, 1.0)
        v_t_components = {
            "novelty": round(novelty, 4),
            "emotion_volatility": round(emotion_volatility, 4),
            "fatigue": round(fatigue, 4),
            "privacy_shift": round(privacy_shift, 4),
            "location_shift": round(location_shift, 4),
            "circadian_offset": round(circadian_offset, 4),
            "resource_scarcity": round(resource_scarcity, 4),
            "salience_lock": round(salience_lock, 4),
            "habit_strength": round(habit_strength, 4),
        }
        V_t = _clip(
            0.25 * novelty
            + 0.20 * emotion_volatility
            + 0.15 * fatigue
            + 0.10 * privacy_shift
            + 0.10 * location_shift
            + 0.10 * circadian_offset
            + 0.10 * resource_scarcity
            - 0.10 * salience_lock
            - 0.10 * habit_strength,
            0.0,
            1.0,
        )
        sigma_emo = 0.03 + (0.18 - 0.03) * V_t
        sigma_mood = 0.015 + (0.09 - 0.015) * V_t
        entropy_refs_by_node: dict[str, Any] = {}
        xi_emo, entropy_ref = self.entropy_pool.truncated_normal(
            sigma=sigma_emo,
            low=-2 * sigma_emo,
            high=2 * sigma_emo,
            purpose=f"round-{round_seed}-xi-emo",
            node_name="xi_emo",
        )
        entropy_refs_by_node["xi_emo"] = to_dict(entropy_ref)
        xi_mood, mood_entropy_ref = self.entropy_pool.truncated_normal(
            sigma=sigma_mood,
            low=-sigma_mood,
            high=sigma_mood,
            purpose=f"round-{round_seed}-xi-mood",
            node_name="xi_mood",
        )
        entropy_refs_by_node["xi_mood"] = to_dict(mood_entropy_ref)
        control_strength = _clip(
            0.35
            + max(0.0, scenario_cfg.get("pfc_base_share", 0.2) - 0.2) * 1.2
            + (0.12 if state.safe_mode else 0.0)
            + min(0.12, state.focus_lock_count * 0.02)
            + max(0.0, state.budget_remaining - 0.5) * 0.10,
            0.0,
            1.0,
        )
        lambda_noise_pre_guard = _clip(
            0.12
            + 0.42 * V_t
            + max(0.0, 0.08 - top_gap) * 2.0
            - 0.10 * control_strength
            - relation_state["relationship_risk"] * 0.06,
            0.0,
            0.75,
        )
        lambda_noise = lambda_noise_pre_guard

        modifiers: dict[str, float] = {}
        log_m_guard_triggered = False
        for action in base_stochastic:
            emo_weight = 1.0 if action in {"connect", "respond"} else -0.5 if action in {"rest", "wander"} else 0.4
            mood_weight = 0.6 if action in {"plan", "clarify", "recall"} else 0.2
            raw_log_m = emo_weight * xi_emo + mood_weight * xi_mood
            if abs(raw_log_m) > 0.35:
                log_m_guard_triggered = True
            log_m = _clip(raw_log_m, -0.35, 0.35)
            modifiers[action] = math.exp(log_m)

        q_noise = self._normalize({action: base_stochastic[action] * modifiers[action] for action in base_stochastic})
        q_noise_pre_guard = dict(q_noise)
        kl = self._kl_divergence(q_noise, base_stochastic)
        noise_guard_triggered = False
        guard_reason = ""
        if log_m_guard_triggered:
            lambda_noise = max(0.0, lambda_noise * 0.92)
            guard_reason = "log_m_guard"
        if kl > 0.15:
            noise_guard_triggered = True
            lambda_noise = max(0.0, lambda_noise * 0.7)
            guard_reason = f"{guard_reason}+kl_guard" if guard_reason else "kl_guard"
            q_noise = self._normalize(
                {
                    action: (base_stochastic[action] * 0.45) + (q_noise[action] * 0.55)
                    for action in base_stochastic
                }
            )
        mixed = {action: (1 - lambda_noise) * base_stochastic[action] + lambda_noise * q_noise[action] for action in base_stochastic}
        mixed = self._normalize(mixed)
        mixed_top_action = self._top_action_name(mixed) or ""

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
        r_intensity, intensity_ref = self.entropy_pool.beta_like(
            mu=mu_int,
            kappa=kappa,
            purpose=f"round-{round_seed}-intensity",
            node_name="r_intensity",
        )
        entropy_refs_by_node["r_intensity"] = to_dict(intensity_ref)
        timing_jitter, timing_ref = self.entropy_pool.uniform_range(
            -0.08,
            0.12,
            purpose=f"round-{round_seed}-timing-jitter",
            node_name="timing_jitter",
        )
        fragmentation_jitter, fragmentation_ref = self.entropy_pool.uniform_range(
            -0.05,
            0.10,
            purpose=f"round-{round_seed}-fragment-jitter",
            node_name="fragmentation_jitter",
        )
        entropy_refs_by_node["timing_jitter"] = to_dict(timing_ref)
        entropy_refs_by_node["fragmentation_jitter"] = to_dict(fragmentation_ref)

        stochastic = StochasticState(
            emo_channel=emo_channel,
            xi_emo=round(xi_emo, 6),
            xi_mood=round(xi_mood, 6),
            lambda_noise=round(lambda_noise, 6),
            r_intensity=round(r_intensity, 6),
            noise_guard_triggered=noise_guard_triggered,
            round_seed=round_seed,
            kl_divergence=round(kl, 6),
            timing_jitter=round(timing_jitter, 6),
            fragmentation_jitter=round(fragmentation_jitter, 6),
            v_t=round(V_t, 6),
            v_t_components=v_t_components,
            sigma_emo=round(sigma_emo, 6),
            sigma_mood=round(sigma_mood, 6),
            lambda_noise_pre_guard=round(lambda_noise_pre_guard, 6),
            log_m_guard_triggered=log_m_guard_triggered,
            guard_reason=guard_reason,
            base_stochastic_distribution={key: round(value, 6) for key, value in base_stochastic.items()},
            q_noise_distribution={key: round(value, 6) for key, value in q_noise.items()},
            q_noise_pre_guard_summary={key: round(value, 6) for key, value in q_noise_pre_guard.items()},
            winner_flip_detected=bool(base_top_action and mixed_top_action and base_top_action != mixed_top_action),
            winner_flip_from=base_top_action,
            winner_flip_to=mixed_top_action,
            entropy_refs_by_node=entropy_refs_by_node,
            entropy_ref=entropy_ref,
        )
        return mixed, stochastic

    def _build_contributions(
        self,
        signals: list[ActionEvidenceSignal],
        state: RuntimeState,
        selected_action: str,
        conflict_score: float,
        fail_score: float,
        resample_idx: int,
    ) -> tuple[list[AgentContribution], list[dict[str, Any]]]:
        contributions: list[AgentContribution] = []
        proposal_records: list[dict[str, Any]] = []
        for signal in signals:
            if signal.veto:
                continue
            top_action = max(signal.action_delta, key=signal.action_delta.get) if signal.action_delta else None
            weight_applied = self._agent_weight(signal.module_name, state)
            selected = selected_action == top_action
            if top_action is not None:
                contributions.append(
                    AgentContribution(
                        agent_name=signal.module_name,
                        action_name=top_action,
                        score=round(signal.action_delta[top_action] * signal.confidence * weight_applied, 4),
                        reason=signal.trace_reason,
                    )
                )
            proposal_records.append(
                {
                    "stage": ACTION_STAGE_BY_OWNER.get(signal.module_name, "unknown"),
                    "agent_name": signal.module_name,
                    "top_action": top_action,
                    "action_preferences": signal.action_delta,
                    "confidence": round(signal.confidence, 4),
                    "veto": signal.veto,
                    "delta_p": signal.action_delta,
                    "sigma_scale": round(signal.sigma_scale, 4),
                    "weight_applied": round(weight_applied, 4),
                    "selected": selected,
                    "conflict_score": round(conflict_score, 4),
                    "plausibility_fail_score": round(fail_score, 4),
                    "resample_idx": resample_idx,
                    "tags": list(signal.trace_tags),
                    "priority_bucket": signal.priority_bucket,
                    "control_domain": signal.control_domain,
                    "gated_actions": list(signal.gated_actions),
                    "risk_hints": dict(signal.risk_hints),
                }
            )
        contributions = sorted(contributions, key=lambda item: item.score, reverse=True)
        return contributions, proposal_records

    def _build_renderer_token_contribution(
        self,
        render_plan: RenderPlan,
        context: dict[str, Any],
    ) -> ProbabilisticContribution:
        cue = str(context.get("cue") or "")
        action = str(render_plan.action or "respond")
        modulated_delta = {
            f"act:{action}": 0.32,
            f"tone:direct:{round(float(render_plan.expression.directness_level), 2)}": round(float(render_plan.expression.directness_level) * 0.18, 6),
            f"tone:warm:{round(float(render_plan.expression.warmth_level), 2)}": round(float(render_plan.expression.warmth_level) * 0.16, 6),
        }
        if bool(render_plan.safety_constraints.get("conflict_hot", False)):
            modulated_delta["safety:guarded"] = 0.22
        if cue:
            modulated_delta[f"cue:{cue}"] = round(0.14 + float(context.get("recall_strength", 0.0) or 0.0) * 0.18, 6)
        dependency_trace = [
            f"action:{action}",
            f"gate:{round(float(render_plan.safety_constraints.get('gate', 1.0) or 0.0), 4)}",
            f"query_kind:{render_plan.identity_context.query_kind or 'none'}",
        ]
        if cue:
            dependency_trace.append(f"cue:{cue}")
        return ProbabilisticContribution(
            module_name="Renderer",
            module_type="token",
            level="token",
            target_space="token",
            raw_signal=dict(modulated_delta),
            modulated_delta=modulated_delta,
            confidence=0.68,
            confidence_calibrated=0.68,
            trace_reason="render plan unfolds action and expression into token field",
            projection_reason="render plan projected into token field",
            applied_at_stage="render_token_unfold",
            native_operator="token_render_plan",
            dependency_trace=dependency_trace,
            projection=EnergyProjectionSpec(module_type="token", target_space="token", module_temperature=0.92),
        )

    def _build_tool_affordance_contribution(
        self,
        state: RuntimeState,
        context: dict[str, Any],
    ) -> ProbabilisticContribution | None:
        if not state.active_run_id or state.run_status not in {"running", "paused"}:
            return None
        current_step = self._current_task_node(
            {
                "current_step_id": state.current_step_id,
                "pending_steps": state.pending_steps,
                "completed_steps": state.completed_steps,
            }
        )
        if current_step is None:
            return None
        tool_choice = str(current_step.tool_choice or "repo_scan")
        matched_files = list(current_step.metadata.get("matched_files", []) or state.last_tool_result.get("matched_files", []) or [])
        matched_count = len(matched_files)
        scarcity = float(state.resource_state.get("scarcity_index", 0.0) or 0.0)
        dirty = bool(state.dirty_worktree_detected)
        drive_source = str(context.get("run_context", {}).get("drive_source") or "")

        tool_affordance_prior = min(1.0, 0.28 + matched_count * 0.08 + (0.12 if drive_source == "active_task" else 0.0))
        tool_expected_value = min(1.0, tool_affordance_prior * (0.75 + min(matched_count, 4) * 0.08))
        tool_cost_penalty = min(1.0, scarcity * 0.45 + (0.18 if dirty else 0.0))

        modulated_delta = {
            "plan": round(0.18 * tool_expected_value, 6),
            "recall": round(0.10 * tool_affordance_prior, 6),
            "clarify": round(0.06 * max(tool_expected_value - tool_cost_penalty * 0.4, 0.0), 6),
            "wander": round(-0.12 * max(tool_affordance_prior, tool_expected_value), 6),
        }
        if tool_cost_penalty > 0.0:
            modulated_delta["short_reply"] = round(0.08 * tool_cost_penalty, 6)
            modulated_delta["rest"] = round(0.04 * tool_cost_penalty, 6)
        if tool_choice != "repo_scan":
            modulated_delta["respond"] = round(0.05 * tool_affordance_prior, 6)

        dependency_trace = [
            f"tool_choice:{tool_choice}",
            f"tool_affordance_prior:{round(tool_affordance_prior, 4)}",
            f"tool_expected_value:{round(tool_expected_value, 4)}",
            f"tool_cost_penalty:{round(tool_cost_penalty, 4)}",
            f"matched_files:{matched_count}",
            f"dirty_worktree:{str(dirty).lower()}",
        ]
        if drive_source:
            dependency_trace.append(f"drive_source:{drive_source}")
        temperature = max(0.55, min(1.2, 1.0 - tool_expected_value * 0.18 + tool_cost_penalty * 0.22))
        confidence = max(0.35, min(0.9, 0.42 + tool_expected_value * 0.35 - tool_cost_penalty * 0.12))
        return ProbabilisticContribution(
            module_name="SkillExecutor",
            module_type="tooling",
            level="action",
            target_space="action",
            raw_signal=dict(modulated_delta),
            modulated_delta=modulated_delta,
            confidence=round(confidence, 4),
            confidence_calibrated=round(confidence, 4),
            trace_reason="active run tool context biases action field toward executable next steps",
            projection_reason="tool affordance prior projected from active run context",
            applied_at_stage="tool_affordance",
            native_operator="tool_affordance_prior",
            dependency_trace=dependency_trace,
            projection=EnergyProjectionSpec(module_type="tooling", target_space="action", module_temperature=temperature),
        )

    def _build_autonomy_self_run_action_contribution(
        self,
        *,
        state: RuntimeState,
        scenario: str,
        endogenous_turn: bool,
    ) -> ProbabilisticContribution | None:
        if not endogenous_turn or scenario == "task":
            return None
        score = self._autonomy_self_run_score(state, state.autonomy_policy)
        if score <= 0.0:
            return None
        confidence = round(max(0.35, min(0.92, 0.4 + score * 0.35)), 4)
        return ProbabilisticContribution(
            module_name="AutonomySelfRun",
            module_type="autonomy",
            level="action",
            target_space="action",
            raw_signal={"self_run_drive": round(score, 6)},
            modulated_delta={
                "self_run": round(0.18 + score * 0.42, 6),
                "nothing": round(-0.04 * score, 6),
                "wander": round(-0.03 * score, 6),
            },
            confidence=confidence,
            confidence_calibrated=confidence,
            trace_reason="endogenous self-directed run intention enters the main action field",
            projection_reason="autonomy self-run pressure projected into endogenous action competition",
            applied_at_stage="autonomy_self_run",
            native_operator="autonomy_endogenous_bias",
            dependency_trace=[
                f"score:{round(score, 4)}",
                f"mode:{state.mode}",
                f"round:{int(state.round_count or 0)}",
            ],
            projection=EnergyProjectionSpec(module_type="autonomy", target_space="action", module_temperature=0.82),
        )

    def _collapse_internal_sampled_action(self, sampled_action: ActionCandidate) -> ActionCandidate:
        if sampled_action.name != "short_reply":
            return sampled_action
        return ActionCandidate(
            name="respond",
            probability=sampled_action.probability,
            rationale="short_reply_folded_to_respond",
        )

    def _probe_context(
        self,
        event: RoundEvent,
        scenario: str,
        mode: str,
    ) -> tuple[RuntimeState, str, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, float], int]:
        state = RuntimeState(**to_dict(self.load_runtime_state()))
        self._normalize_temperament_runtime_state(state)
        requested_mode = "safe" if state.safe_mode else mode
        mode_cfg = self.config["modes"]["modes"].get(requested_mode, self.config["modes"]["modes"]["interactive"])
        scenario_cfg = self.config["scenarios"]["scenarios"][scenario]
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

        cue = _derive_cue(event)
        memory_retrieval_budget = self._memory_retrieval_budget(event, scenario_cfg)
        resource_telemetry = self._compute_resource_telemetry(state)
        state.resource_state = {**state.resource_state, **resource_telemetry}
        context = {
            "cue": cue,
            "memory_retrieval_budget": memory_retrieval_budget,
            "recall_strength": self.memory_store.recall_strength(cue, tier_budget=memory_retrieval_budget),
            "habit_strength": self.memory_store.habit_strength(cue),
            "closeness": self.memory_store.closeness(event.target),
            "valence": event.valence,
            "appraisal": appraisal,
            "detail_threshold": self.config["thresholds"]["thresholds"].get("detail_threshold", 0.5),
            "round_gap": 0,
            "interference": 0.0,
            "recent_burn_rate": 1.0 - state.budget_remaining,
            "burn_rate_ratio": resource_telemetry["burn_rate_ratio"],
            "low_balance_ratio": resource_telemetry["low_balance_ratio"],
            "queue_pressure": resource_telemetry["queue_pressure"],
            "latency_pressure": resource_telemetry["latency_pressure"],
        }
        if cue:
            recall_payload = self.memory_store.recall(cue, tier_budget=memory_retrieval_budget)
            context["interference"] = recall_payload.get("interference", 0.0)
            context["episode_id"] = recall_payload.get("episode_id", "")
            context["separation_id"] = recall_payload.get("separation_id", "")
            context["memory_prior_vector"] = dict(recall_payload.get("prior_vector", {}) or {})
        relation_state = self._relation_state(event, context)
        context["chronic_signal"] = self._build_chronic_signal(
            event,
            state=state,
            context=context,
            relation_state=relation_state,
            scenario=scenario,
        )
        round_seed = self._round_seed(state, event)
        return state, requested_mode, mode_cfg, scenario_cfg, context, relation_state, round_seed

    def _collect_probe_signals(
        self,
        *,
        event: RoundEvent,
        state: RuntimeState,
        scenario: str,
        scenario_cfg: dict[str, Any],
        context: dict[str, Any],
        relation_state: dict[str, float],
        allow_model_reasoning: bool = True,
    ) -> tuple[list[ActionEvidenceSignal], dict[str, ProbabilisticContribution], dict[str, Any], dict[str, float]]:
        direct_action_contributions: dict[str, ProbabilisticContribution] = {}
        direct_action_signal_metadata: dict[str, dict[str, Any]] = {}

        for stage_name, owner in ACTION_HEAD_STAGE_ORDER:
            if not state.agents_enabled.get(owner, True):
                continue
            agent = self.agent_map[owner]

            if owner == "BodyStateAgent":
                patch = agent.update_body_state(event, state, scenario_cfg, context)
                self._apply_state_patch(state, patch.get("state_patch", {}))
                contribution = agent.build_direct_action_contribution(event, state, scenario_cfg, context)
                veto_info = agent.apply_body_veto(event, state, scenario_cfg, context)
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
                patch = agent.update_affect_state(event, state, scenario_cfg, context)
                self._apply_state_patch(state, patch.get("state_patch", {}))
                contribution = agent.build_direct_action_contribution(event, state, scenario_cfg, context)
                veto_info = agent.trigger_affect_veto(event, state, scenario_cfg, context)
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
                closeness = agent.score_closeness(event, state, scenario_cfg, context)
                context["closeness"] = closeness.get("score", context["closeness"])
                relation_state = self._relation_state(event, context)
                gate_info = agent.compute_boundary_gate(event, state, scenario_cfg, context)
                relation_state["boundary_level"] = gate_info.get("boundary_level", relation_state["boundary_level"])
                relation_state["relationship_risk"] = _clip(
                    (1.0 - context["closeness"]) * 0.5 + relation_state["boundary_level"] * 0.25,
                    0.0,
                    1.0,
                )
                contribution = agent.build_direct_action_contribution(event, state, scenario_cfg, context)
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
                scarcity = agent.compute_scarcity_index(event, state, scenario_cfg, context)
                state.resource_state = {**state.resource_state, "scarcity_index": round(float(scarcity.get("scalar", 0.0)), 4)}
                state.resource_state = {**state.resource_state, **self._resource_bias_snapshot(state, context)}
                contribution = agent.build_direct_action_contribution(event, state, scenario_cfg, context)
                direct_action_contributions[owner] = contribution
                mode_hint = agent.suggest_resource_mode(event, state, scenario_cfg, context)
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
                reasoning_state, reasoning_meta = self._state_for_reasoning(state, event, scenario, state.mode)
                context = {**context, **reasoning_meta}
                if allow_model_reasoning:
                    try:
                        contribution = self._generate_pfc_candidates_via_model(event, reasoning_state, scenario_cfg, context)
                    except Exception:
                        contribution = agent.build_direct_action_contribution(event, reasoning_state, scenario_cfg, context)
                else:
                    contribution = agent.build_direct_action_contribution(event, reasoning_state, scenario_cfg, context)
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

            if owner == "HippocampusAgent":
                agent.encode_episode(event, state, scenario_cfg, context)
                recall_set = agent.retrieve_by_cue(event, state, scenario_cfg, context)
                agent.apply_cue_weighted_decay(event, state, scenario_cfg, context)
                agent.compute_memory_interference(event, state, scenario_cfg, context)
                if not recall_set.get("recall_set"):
                    agent.fallback_to_gist_when_trace_weak(event, state, scenario_cfg, context)
                contribution = agent.build_direct_action_contribution(event, state, scenario_cfg, context)
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

            if hasattr(agent, "build_direct_action_contribution"):
                contribution = agent.build_direct_action_contribution(event, state, scenario_cfg, context)
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

        autonomy_self_run_contribution = self._build_autonomy_self_run_action_contribution(
            state=state,
            scenario=scenario,
            endogenous_turn=event.source == "endogenous",
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

        action_signals = [
            self._action_evidence_from_contribution(
                contribution,
                **direct_action_signal_metadata.get(owner, {}),
            )
            for owner, contribution in direct_action_contributions.items()
        ]
        return (
            action_signals,
            direct_action_contributions,
            context,
            relation_state,
        )

    def _probe_distribution_for_scenario(
        self,
        event: RoundEvent,
        *,
        scenario: str,
        mode: str,
        allow_model_reasoning: bool = True,
    ) -> dict[str, Any]:
        state, requested_mode, mode_cfg, scenario_cfg, context, relation_state, round_seed = self._probe_context(event, scenario, mode)
        probe_signals, probe_direct_contributions, context, relation_state = self._collect_probe_signals(
            event=event,
            state=state,
            scenario=scenario,
            scenario_cfg=scenario_cfg,
            context=context,
            relation_state=relation_state,
            allow_model_reasoning=allow_model_reasoning,
        )
        slow_variables = self._build_slow_variable_payload(
            state=state,
            context=context,
            relation_state=relation_state,
            prior_closeness=context.get("closeness", 0.5),
        )
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
        identity_context = self._build_identity_context(
            query_state=query_state,
            disclosure_state=disclosure_state,
            scenario=scenario,
            state=state,
            shaping_events=[],
            slow_variables=slow_variables,
            state_sources=[],
            rename_event=None,
        )
        probe_action_bookkeeping = self._build_action_bookkeeping(
            probe_signals,
            state,
            scenario_cfg,
            mode_cfg,
            relation_state,
            context,
            query_state=query_state,
            disclosure_state=disclosure_state,
        )
        online_long_run_projection = self.long_run_analyzer.build_online_projection(
            round_id=state.round_count,
            slow_variables=slow_variables,
            shaping_events=[],
        )
        online_long_run_contribution = self.long_run_analyzer.build_long_run_prior_contribution(
            online_long_run_projection
        )
        probe_snapshot, probe_contributions, token_state = self._integrate_probability_field_snapshot(
            direct_action_contributions=probe_direct_contributions,
            action_base=dict(probe_action_bookkeeping.u_base),
            event=event,
            state=state,
            scenario_cfg=scenario_cfg,
            context=context,
            identity_context=identity_context,
            long_run_contribution=online_long_run_contribution,
        )
        current_distribution = dict(probe_snapshot.action.winner_posterior or {})
        penalized_distribution, candidate_penalties, sampling_penalty_applied = self.authenticity_policy.apply_sampling_penalties(
            current_distribution,
            query_kind=query_state.legacy_query_kind,
            disclosure_intent=disclosure_state.posterior.top_intent,
            slow_variables=slow_variables,
            memory_cue=context.get("cue"),
            shaping_events=[],
        )
        authenticity_contribution = None
        if candidate_penalties or sampling_penalty_applied > 0.0:
            authenticity_contribution = self.authenticity_policy.build_action_penalty_contribution(
                AuthenticityRecord(
                    self_grounding_score=0.0,
                    guard_action="probe_sampling_penalty",
                    disclosure_detail=disclosure_state.posterior.top_intent or "none",
                    state_sources=[],
                    candidate_penalties=dict(candidate_penalties),
                    sampling_penalty_applied=float(sampling_penalty_applied),
                )
            )
        if authenticity_contribution is not None:
            probe_contributions.append(authenticity_contribution)
            probe_snapshot = self._reintegrate_probability_snapshot(
                snapshot=probe_snapshot,
                contributions=probe_contributions,
                token_state=token_state,
                source_chain=["probe_after_authenticity"],
            )

        plausibility_guard = self.agent_map["BehaviorPlausibilityGuard"]
        output_gate = self.agent_map["OutputGate"]
        gated_distribution: dict[str, float] = {}
        hard_mask: dict[str, bool] = {}
        for action_name, probability in dict(probe_snapshot.action.winner_posterior or {}).items():
            if probability <= 0.0:
                continue
            plausibility = plausibility_guard.check_behavior_plausibility(action_name, scenario, relation_state)
            gate = 1.0 if plausibility.get("pass", True) else 0.0
            gate *= float(output_gate.apply_output_gate(action_name, state, scenario, relation_state).get("gate", 1.0))
            gated_distribution[action_name] = probability * gate
            if gate <= 0.0:
                hard_mask[action_name] = True
        gate_contribution = self._build_distribution_delta_contribution(
            module_name="ProbeGuard",
            module_type="guard",
            from_distribution=dict(probe_snapshot.action.winner_posterior or {}),
            to_distribution=self._normalize(gated_distribution or dict(probe_snapshot.action.winner_posterior or {})),
            hard_mask=hard_mask,
            trace_reason="probe guard projected from plausibility and output gates",
            projection_reason="probe guard projected from field-native gating",
            applied_at_stage="probe_guard",
            native_operator="probe_gate",
            dependency_trace=[
                f"scenario:{scenario}",
                f"blocked:{','.join(sorted(action for action, flag in hard_mask.items() if flag))}",
            ],
            confidence=1.0,
        )
        if gate_contribution is not None:
            probe_contributions.append(gate_contribution)
            probe_snapshot = self._reintegrate_probability_snapshot(
                snapshot=probe_snapshot,
                contributions=probe_contributions,
                token_state=token_state,
                source_chain=["probe_after_guard"],
            )
        top_action = probe_snapshot.action.winner_target or max(probe_snapshot.action.winner_posterior, key=probe_snapshot.action.winner_posterior.get)
        sampled_action = self._collapse_internal_sampled_action(
            ActionCandidate(name=top_action, probability=probe_snapshot.action.winner_posterior[top_action], rationale="probe argmax")
        )
        final_distribution = dict(probe_snapshot.action.winner_posterior or {})
        task_mass = round(sum(final_distribution.get(action, 0.0) for action in ("plan", "recall", "clarify")), 6)
        chat_mass = round(sum(final_distribution.get(action, 0.0) for action in ("respond", "connect", "rest")), 6)
        return {
            "scenario": scenario,
            "mode": requested_mode,
            "top_action": sampled_action.name,
            "action_distribution": final_distribution,
            "task_mass": task_mass,
            "chat_mass": chat_mass,
        }

    def _autonomy_action_field_probe(self) -> dict[str, Any] | None:
        try:
            return self._probe_distribution_for_scenario(
                RoundEvent(
                    source="endogenous",
                    content="autonomy heartbeat",
                    target="self",
                    cue="endogenous:autonomy",
                ),
                scenario="companion",
                mode="endogenous_light",
                allow_model_reasoning=False,
            )
        except Exception:
            return None

    def _terminal_route_scenario_hint(self, text: str) -> str:
        normalized = text.strip().lower()
        if not normalized:
            return "chat"
        task_tokens = (
            ".py",
            ".ts",
            ".tsx",
            ".js",
            ".jsx",
            "repo",
            "trace",
            "bug",
            "debug",
            "fix",
            "test",
            "tests",
            "file",
            "files",
            "code",
            "worker",
            "planner",
            "app.py",
            "仓库",
            "代码",
            "文件",
            "目录",
            "测试",
            "修复",
            "检查",
            "总结",
            "规划",
            "实现",
            "重构",
        )
        return "task" if any(token in normalized for token in task_tokens) else "chat"

    def _turn_reason(self, route: str, probe: dict[str, Any], scenario: str) -> str:
        task_mass = float(probe.get("task_mass", 0.0))
        chat_mass = float(probe.get("chat_mass", 0.0))
        if route == "task_run":
            return f"{scenario}_probe task_mass={task_mass:.3f} chat_mass={chat_mass:.3f}"
        return f"{scenario}_probe chat_mass={chat_mass:.3f} task_mass={task_mass:.3f}"

    def _endogenous_route_type(self, mode: str | None) -> str:
        normalized = str(mode or "").strip().lower()
        if normalized in {"endogenous_replay", "endogenous_regulation"}:
            return "endogenous_deep"
        if normalized.startswith("endogenous"):
            return "endogenous_light"
        if normalized == "sleep":
            return "dream_sleep"
        return ""

    def _direct_chat_route_type(
        self,
        text: str,
        *,
        probe: dict[str, Any] | None = None,
        top_action: str = "",
        model_call_count: int = 0,
        target: str | None = "user",
    ) -> str:
        normalized = str(text or "").strip()
        lowered = normalized.lower()
        normalized_target = str(target or "user").strip().lower()
        top = str(top_action or str((probe or {}).get("top_action", ""))).strip()
        task_mass = float((probe or {}).get("task_mass", 0.0) or 0.0)
        deep_tokens = ("详细", "深入", "系统", "比较", "分析", "展开", "long-form", "step by step", "deeply")
        standard_tokens = (
            "记得",
            "还记得",
            "上次",
            "之前",
            "冲突",
            "矛盾",
            "工具",
            "查一下",
            "搜索",
            "不确定",
            "拿不准",
            "remember",
            "conflict",
            "uncertain",
        )
        has_deep_token = any(token in lowered for token in deep_tokens)
        has_standard_token = any(token in lowered for token in standard_tokens)
        if has_standard_token and not has_deep_token and len(normalized) <= 140:
            return "chat_standard"
        if (
            len(normalized) > 140
            or model_call_count >= 4
            or task_mass >= 0.35
            or (top in {"plan", "recall", "clarify", "connect"} and len(normalized) > 48)
            or has_deep_token
        ):
            return "chat_deep"
        if normalized_target not in {"", "user", "self"}:
            return "chat_standard"
        if has_standard_token:
            return "chat_standard"
        return "chat_fast"

    def _route_type_for_turn_plan(self, *, route: str, text: str, scenario: str, mode: str, probe: dict[str, Any] | None = None, target: str | None = "user") -> str:
        if route == "fast_chat":
            return "chat_micro"
        if route == "task_run":
            return "task_run"
        endogenous_route = self._endogenous_route_type(mode)
        if endogenous_route:
            return endogenous_route
        if scenario == "task":
            return "task_run"
        return self._direct_chat_route_type(text, probe=probe, target=target)

    def _runtime_route_type_for_round(
        self,
        *,
        event: RoundEvent,
        scenario: str,
        mode: str,
        top_action: str,
        model_call_count: int,
        probe: dict[str, Any] | None = None,
    ) -> str:
        endogenous_route = self._endogenous_route_type(mode)
        if endogenous_route:
            return endogenous_route
        if scenario == "task":
            return "task_run"
        return self._direct_chat_route_type(
            event.content,
            probe=probe,
            top_action=top_action,
            model_call_count=model_call_count,
            target=event.target,
        )

    def _is_fast_chat_candidate(self, text: str, *, target: str = "user", mode: str = "interactive") -> bool:
        normalized = text.strip()
        if not normalized or len(normalized) > 80:
            return False
        if self._is_known_user_probe(normalized):
            return True
        if deterministic_short_chat_reply(normalized):
            return True
        lowered = normalized.lower()
        greeting_tokens = ("你好", "您好", "hello", "hi", "hey", "在吗")
        direct_identity_tokens = (
            "你是谁",
            "你叫什么",
            "有名字吗",
            "who are you",
            "what's your name",
            "what is your name",
        )
        provider_tokens = (
            "你是chatgpt",
            "你是豆包",
            "你底层是什么",
            "what model are you",
            "which model are you",
        )
        explanation_tokens = (
            "为什么这样回答",
            "为什么这么回答",
            "why did you answer",
            "why did you say",
        )
        capability_tokens = (
            "你能做什么",
            "你可以做什么",
            "你会什么",
            "能帮我做什么",
            "what can you do",
        )
        if len(normalized) <= 24 and any(token in lowered for token in greeting_tokens):
            return True
        if len(normalized) <= 32 and any(
            token in lowered
            for token in (
                *direct_identity_tokens,
                *provider_tokens,
                *explanation_tokens,
                *capability_tokens,
            )
        ):
            return True
        event = RoundEvent(source="user", content=normalized, target=target)
        state, _, _, _, context, relation_state, _ = self._probe_context(event, "chat", mode)
        slow_variables = self._build_slow_variable_payload(
            state=state,
            context=context,
            relation_state=relation_state,
            prior_closeness=context.get("closeness", 0.5),
        )
        query_state = self._infer_query_intent(
            event=event,
            scenario="chat",
            state=state,
            relation_state=relation_state,
            slow_variables=slow_variables,
        )
        return query_state.posterior.top_intent in {
            "self_model_identity_probe",
            "provider_lineage_probe",
            "answer_reason_probe",
            "capability_boundary_probe",
        }

    def _compact_fast_chat_text(self, text: str) -> str:
        return "".join(str(text or "").strip().lower().split())

    def _is_known_user_probe(self, text: str) -> bool:
        compact = self._compact_fast_chat_text(text)
        return any(
            token in compact
            for token in (
                "你知道我是谁吗",
                "你知道我是谁",
                "你记得我是谁吗",
                "你还记得我是谁吗",
                "doyouknowwhoiam",
                "doyourememberwhoiam",
            )
        )

    def _cached_fast_chat_reply(self, state: RuntimeState, text: str) -> str:
        capsule = dict(state.session_metadata.get("fast_chat_capsule", {}) or {})
        cached_text = str(capsule.get("assistant_text") or "").strip()
        cached_user_text = str(capsule.get("user_text") or "").strip()
        if not cached_text or not cached_user_text:
            return ""
        if self._compact_fast_chat_text(cached_user_text) != self._compact_fast_chat_text(text):
            return ""
        return cached_text

    def _local_fast_chat_reply(self, text: str, capsule: dict[str, Any]) -> str:
        identity = dict(capsule.get("identity_context") or {})
        display_label = str(identity.get("display_label") or self._unnamed_label()).strip() or self._unnamed_label()
        provider_label = str(identity.get("provider_label") or "当前模型路由").strip() or "当前模型路由"
        query_kind = str(identity.get("query_kind") or "").strip()
        query_intent = str(identity.get("query_intent") or "").strip()
        if self._is_known_user_probe(text):
            relation_state = dict(capsule.get("relation_state") or {})
            slow_variables = dict(capsule.get("slow_variables") or {})
            closeness = float(relation_state.get("closeness", 0.5) or 0.5)
            memory_activation = float(slow_variables.get("memory_activation", 0.0) or 0.0)
            if closeness >= 0.62 or memory_activation >= 0.18:
                return "我知道现在正在和我说话的是你。我对你已经有一些连续的关系和记忆线索，但只靠当前这些线索，还不能稳定确认一个更具体的身份。"
            return "我知道现在正在和我说话的是你。但就这段对话里已经留下来的线索，我还不能稳定确认一个更具体的身份。"
        if query_kind == "self_identity":
            return f"我是{display_label}，当前这个运行体。你可以直接这样叫我。"
        if query_kind == "provider_identity":
            return f"如果你在问底层路由，我现在走的是{provider_label}；但和你说话的这个运行体是{display_label}。"
        if query_kind == "answer_explanation":
            return "我刚才是先按你这句话里最需要接住的部分来回应，再把语气收得更贴近当前状态。"
        if query_intent == "capability_boundary_probe":
            return "我可以直接和你对话，也能帮你梳理思路、拆任务、看代码，或者解释我现在的运行状态。"
        return ""

    def _build_fast_chat_request(self, plan: TurnPlan) -> tuple[RuntimeState, dict[str, Any], ModelRequest]:
        current_state = self.load_runtime_state()
        event = RoundEvent(source="user", content=plan.text, target=plan.target)
        probe_state, _, _, _, context, relation_state, _ = self._probe_context(event, "chat", plan.mode)
        reasoning_state, _ = self._state_for_reasoning(current_state, event, "chat", plan.mode)
        slow_variables = self._build_slow_variable_payload(
            state=probe_state,
            context=context,
            relation_state=relation_state,
            prior_closeness=context.get("closeness", 0.5),
        )
        query_state = self._infer_query_intent(
            event=event,
            scenario="chat",
            state=probe_state,
            relation_state=relation_state,
            slow_variables=slow_variables,
        )
        disclosure_state = self._infer_disclosure_intent(
            query_state=query_state,
            scenario="chat",
            state=probe_state,
            relation_state=relation_state,
            slow_variables=slow_variables,
        )
        identity_context = self._build_identity_context(
            query_state=query_state,
            disclosure_state=disclosure_state,
            scenario="chat",
            state=current_state,
            shaping_events=[],
            slow_variables=slow_variables,
            state_sources=["focus"],
            rename_event=None,
        )
        provider_label, model_label = self._provider_descriptor_for_route("chat_fast")
        identity_context.provider_label = provider_label
        identity_context.model_label = model_label
        capsule = {
            "user_text": plan.text,
            "identity_context": to_dict(identity_context),
            "state_summary": {
                "mode": reasoning_state.mode,
                "focus": reasoning_state.focus,
                "mood": round(float(reasoning_state.mood), 4),
                "body_energy": round(float(reasoning_state.body_energy), 4),
                "affect_residue": round(float(reasoning_state.affect_residue), 4),
                "run_status": reasoning_state.run_status,
                "current_goal": reasoning_state.current_goal,
            },
            "relation_state": relation_state,
            "slow_variables": slow_variables,
            "cognitive_snapshot": self.cognitive_snapshot(state=reasoning_state),
        }
        request = ModelRequest(
            system_prompt=self._fast_chat_system_prompt(identity_context),
            user_prompt=self._json_prompt({"capsule": capsule}),
            response_schema={"text": "str"},
            metadata={"query_kind": identity_context.query_kind, "route": "chat_fast"},
        )
        return current_state, capsule, request

    def _persist_fast_chat_capsule(self, state: RuntimeState, capsule: dict[str, Any], message: str) -> None:
        state.session_metadata["fast_chat_capsule"] = {
            **capsule,
            "assistant_text": message,
            "recorded_at": utc_now_iso(),
        }
        self._save_state(state)

    def _fast_chat_system_prompt(self, identity: IdentityContext) -> str:
        lines = [
            "你是 NALR 的 fast-chat 表达层。",
            "你只输出最终给用户看的自然语言正文，不要输出 JSON，不要解释规则。",
            "你必须从当前运行体视角表达，不能把底层 provider 当成本体身份。",
            f"当前本体标签: {identity.display_label or self._unnamed_label()} / {identity.class_label}.",
            f"query_kind={identity.query_kind}; disclosure_detail={identity.disclosure_detail}.",
            f"query_intent={identity.query_intent}; disclosure_intent={identity.disclosure_intent}.",
            "回答应自然、简洁、像同一个运行体正在开口，不要写成客服、产品介绍或命令帮助。",
        ]
        if identity.query_kind == "self_identity":
            lines.append("当用户在问你是谁时，只能说明自己是当前运行体实例，不得说“我是豆包/ChatGPT/provider”。")
        elif identity.query_kind == "provider_identity":
            lines.append("当用户追问底层能力时，先说明本体，再把 provider 作为底层能力来源说明。")
        elif identity.query_kind == "answer_explanation":
            lines.append("当用户追问为什么这样回答时，只能引用当前状态、记忆、关系、focus 和本轮表达收束。")
        elif identity.query_intent == "capability_boundary_probe":
            lines.append("当用户问你能做什么时，自然说明你会如何响应、记住、规划或协助，不要列清单。")
        return " ".join(lines)

    def _execute_fast_chat_turn(self, plan: TurnPlan) -> TurnExecution:
        state, capsule, request = self._build_fast_chat_request(plan)
        cached_reply = self._cached_fast_chat_reply(state, plan.text)
        if cached_reply:
            final_message = cached_reply
            self._persist_fast_chat_capsule(state, capsule, final_message)
            return TurnExecution(
                route="fast_chat",
                route_type="chat_micro",
                assistant_final=final_message,
                payload={"capsule": capsule, "cache_hit": True},
            )
        local_reply = self._local_fast_chat_reply(plan.text, capsule)
        if local_reply:
            final_message = local_reply
            self._persist_fast_chat_capsule(state, capsule, final_message)
            return TurnExecution(
                route="fast_chat",
                route_type="chat_micro",
                assistant_final=final_message,
                payload={"capsule": capsule},
            )
        deterministic_reply = deterministic_short_chat_reply(plan.text)
        if deterministic_reply:
            final_message = deterministic_reply
            self._persist_fast_chat_capsule(state, capsule, final_message)
            return TurnExecution(
                route="fast_chat",
                route_type="chat_micro",
                assistant_final=final_message,
                payload={"capsule": capsule},
            )
        try:
            response = self.model_router.generate("chat_fast", request)
            final_message = str(response.payload.get("text", "")).strip()
        except Exception:
            result = self.tick(
                RoundEvent(source="user", content=plan.text, target=plan.target),
                scenario="chat",
                mode=plan.mode,
            )
            final_message = result.rendered_expression.text.strip()
        final_message = final_message or "你好，我在。你想让我帮你做什么？"
        self._persist_fast_chat_capsule(state, capsule, final_message)
        return TurnExecution(
            route="fast_chat",
            route_type="chat_micro",
            assistant_final=final_message,
            payload={"capsule": capsule},
        )

    def stream_fast_chat_turn(self, plan: TurnPlan) -> tuple[list[str], TurnExecution]:
        state, capsule, request = self._build_fast_chat_request(plan)
        cached_reply = self._cached_fast_chat_reply(state, plan.text)
        if cached_reply:
            self._persist_fast_chat_capsule(state, capsule, cached_reply)
            return [cached_reply], TurnExecution(
                route="fast_chat",
                route_type="chat_micro",
                assistant_final=cached_reply,
                payload={"capsule": capsule, "stream_deltas": [cached_reply], "cache_hit": True},
            )
        local_reply = self._local_fast_chat_reply(plan.text, capsule)
        if local_reply:
            self._persist_fast_chat_capsule(state, capsule, local_reply)
            return [local_reply], TurnExecution(
                route="fast_chat",
                route_type="chat_micro",
                assistant_final=local_reply,
                payload={"capsule": capsule, "stream_deltas": [local_reply]},
            )
        deterministic_reply = deterministic_short_chat_reply(plan.text)
        if deterministic_reply:
            self._persist_fast_chat_capsule(state, capsule, deterministic_reply)
            return [deterministic_reply], TurnExecution(
                route="fast_chat",
                route_type="chat_micro",
                assistant_final=deterministic_reply,
                payload={"capsule": capsule, "stream_deltas": [deterministic_reply]},
            )
        try:
            deltas = [chunk for chunk in self.model_router.stream_generate("chat_fast", request) if isinstance(chunk, str) and chunk]
            final_message = "".join(deltas).strip()
            if not final_message:
                response = self.model_router.generate("chat_fast", request)
                final_message = str(response.payload.get("text", "")).strip()
                deltas = [final_message] if final_message else []
        except Exception:
            result = self.tick(
                RoundEvent(source="user", content=plan.text, target=plan.target),
                scenario="chat",
                mode=plan.mode,
            )
            final_message = result.rendered_expression.text.strip()
            deltas = [final_message] if final_message else []
        if not final_message:
            final_message = "你好，我在。你想让我帮你做什么？"
            deltas = [final_message]
        self._persist_fast_chat_capsule(state, capsule, final_message)
        return deltas, TurnExecution(
            route="fast_chat",
            route_type="chat_micro",
            assistant_final=final_message,
            payload={"capsule": capsule, "stream_deltas": deltas},
        )

    def _build_task_bootstrap(
        self,
        goal: str,
        *,
        allow_commit: bool,
        operator_level: str,
        defer_bootstrap_tool: bool = False,
    ) -> tuple[RunState, dict[str, Any], dict[str, Any]]:
        return self.run_runtime._build_task_bootstrap(
            goal,
            allow_commit=allow_commit,
            operator_level=operator_level,
            defer_bootstrap_tool=defer_bootstrap_tool,
        )

    def prepare_task_bootstrap(
        self,
        goal: str,
        *,
        allow_commit: bool,
        operator_level: str,
        defer_bootstrap_tool: bool = False,
    ) -> tuple[RunState, dict[str, Any], dict[str, Any]]:
        return self.run_runtime.prepare_task_bootstrap(
            goal,
            allow_commit=allow_commit,
            operator_level=operator_level,
            defer_bootstrap_tool=defer_bootstrap_tool,
        )

    def plan_turn(
        self,
        text: str,
        *,
        target: str = "user",
        mode: str = "interactive",
        allow_commit: bool = False,
        operator_level: str = "read_only",
        defer_bootstrap_tool: bool = False,
    ) -> TurnPlan:
        normalized = text.strip()
        if not normalized:
            return TurnPlan(
                text=normalized,
                route="direct_chat",
                scenario="chat",
                route_type="chat_fast",
                mode=mode,
                target=target,
                reason="empty_input",
            )

        scenario = self._terminal_route_scenario_hint(normalized)
        if scenario == "chat" and self._is_fast_chat_candidate(normalized, target=target, mode=mode):
            return TurnPlan(
                text=normalized,
                route="fast_chat",
                scenario=scenario,
                route_type="chat_micro",
                mode=mode,
                target=target,
                reason="chat_fast_heuristic",
                top_action="respond",
                precomputed_distribution={
                    "action_distribution": {"respond": 1.0},
                    "task_mass": 0.0,
                    "chat_mass": 1.0,
                },
                task_bootstrap=None,
            )
        event = RoundEvent(source="user", content=normalized, target=target)
        try:
            probe = self._probe_distribution_for_scenario(
                event,
                scenario=scenario,
                mode=mode,
                allow_model_reasoning=scenario != "chat",
            )
        except TypeError as exc:
            if "allow_model_reasoning" not in str(exc):
                raise
            probe = self._probe_distribution_for_scenario(
                event,
                scenario=scenario,
                mode=mode,
            )
        task_selected = (
            probe["task_mass"] >= 0.55
            and probe["task_mass"] - probe["chat_mass"] >= 0.10
        )
        route = "task_run" if task_selected else "direct_chat"
        task_bootstrap = None
        if route == "task_run":
            task_bootstrap = self._build_task_bootstrap(
                normalized,
                allow_commit=allow_commit,
                operator_level=operator_level,
                defer_bootstrap_tool=defer_bootstrap_tool,
            )
        return TurnPlan(
            text=normalized,
            route=route,
            scenario=scenario,
            route_type=self._route_type_for_turn_plan(route=route, text=normalized, scenario=scenario, mode=mode, probe=probe, target=target),
            mode=mode,
            target=target,
            reason=self._turn_reason(route, probe, scenario),
            top_action=str(probe.get("top_action", "")),
            precomputed_distribution={
                "action_distribution": dict(probe.get("action_distribution", {})),
                "task_mass": float(probe.get("task_mass", 0.0)),
                "chat_mass": float(probe.get("chat_mass", 0.0)),
            },
            task_bootstrap=task_bootstrap,
        )

    def execute_turn(
        self,
        plan: TurnPlan,
        *,
        allow_commit: bool = False,
        operator_level: str = "read_only",
        replace_active: bool = False,
        interrupt_reason: str = "interrupted_by_user",
    ) -> TurnExecution:
        if plan.route == "fast_chat":
            return self._execute_fast_chat_turn(plan)
        if plan.route == "direct_chat":
            execution = self.chat_kernel_v2.execute(plan)
            execution.payload = {
                "reason": plan.reason,
                "top_action": plan.top_action,
                **dict(execution.payload or {}),
            }
            return execution

        run_details = self.start_run(
            plan.text,
            allow_commit=allow_commit,
            operator_level=operator_level,
            replace_active=replace_active,
            interrupt_reason=interrupt_reason,
            bootstrap=plan.task_bootstrap,
            include_details=True,
            sync_hot_path=False,
        )
        return TurnExecution(
            route="task_run",
            route_type=plan.route_type or "task_run",
            assistant_preamble=self._task_turn_message(run_details["run"]),
            assistant_final=self._task_turn_message(run_details["run"]),
            run=run_details["run"],
            explain=run_details["explain"],
            steps=run_details["steps"],
            tools=run_details["tools"],
            payload={"reason": plan.reason, "top_action": plan.top_action, "route_type": plan.route_type or "task_run"},
        )

    def probe_terminal_route(
        self,
        text: str,
        *,
        target: str = "user",
        mode: str = "interactive",
    ) -> dict[str, Any]:
        plan = self.plan_turn(text, target=target, mode=mode)
        selected = plan.precomputed_distribution
        return {
            "route": plan.route,
            "route_type": plan.route_type,
            "top_action": plan.top_action,
            "action_distribution": selected["action_distribution"],
            "task_mass": selected["task_mass"],
            "chat_mass": selected["chat_mass"],
            "reason": plan.reason,
            "scenario": plan.scenario,
        }

    def _memory_retrieval_budget(self, event: RoundEvent, scenario_cfg: dict[str, Any]) -> tuple[str, ...]:
        salience_signal = compute_salience_signal(event, scenario_cfg)
        salience_high = float(self.config["thresholds"]["thresholds"].get("salience_high", 0.78))
        if salience_signal < salience_high:
            return ("hot",)
        return ("hot", "warm", "cold")

    def build_round_context(self, event: RoundEvent, scenario: str, mode: str) -> dict[str, Any]:
        return self.round_pipeline.build_round_context(event, scenario, mode)

    def collect_parallel_contributions(
        self,
        *,
        round_context: dict[str, Any],
        scenario: str,
    ) -> dict[str, Any]:
        return self.round_pipeline.collect_parallel_contributions(round_context=round_context, scenario=scenario)

    def integrate_and_arbitrate(
        self,
        *,
        round_context: dict[str, Any],
        collected: dict[str, Any],
        scenario: str,
        turn_started: float,
    ) -> dict[str, Any]:
        return self.round_pipeline.integrate_and_arbitrate(round_context=round_context, collected=collected, scenario=scenario, turn_started=turn_started)

    def evaluate_initiative_overlay(
        self,
        *,
        state: RuntimeState,
        relation_state: dict[str, Any],
        vitality_snapshot: dict[str, Any],
        context: dict[str, Any],
        latest_round_recorded_at: str | None,
        endogenous_turn: bool,
        rendered_preview: str,
    ) -> dict[str, Any]:
        context_memory_backing = self._initiative_context_memory_backing(
            cue=context.get("cue"),
            recall_strength=context.get("recall_strength", 0.0),
            fallback_summary=rendered_preview,
        )
        payload = self._initiative_distribution_payload(
            state,
            relation_state=relation_state,
            vitality_snapshot=vitality_snapshot,
            cue=context.get("cue"),
            memory_backing=context_memory_backing,
            latest_recorded_at=latest_round_recorded_at,
            current_goal=context.get("run_context", {}).get("goal") if isinstance(context.get("run_context"), dict) else state.current_goal,
        )
        if not endogenous_turn and payload.get("should_send"):
            payload["should_send"] = False
            payload["expression_mode"] = "silent"
            payload["suppression_reason"] = str(payload.get("suppression_reason") or "reactive_round")
        payload["rendered_preview"] = rendered_preview[:200].strip()
        payload["source_round_id"] = state.round_count
        if payload.get("should_send") and not bool(payload.get("auto_send_enabled")):
            payload["should_send"] = False
            payload["expression_mode"] = "silent"
            payload["suppression_reason"] = "auto_send_disabled"
        return payload

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
        return self.round_pipeline.render_and_commit(state=state, trace=trace, sampled_action=sampled_action, rendered_expression=rendered_expression, scenario=scenario, endogenous_turn=endogenous_turn, control_ledger=control_ledger, recorded_at=recorded_at)

    def _tick_impl(self, event: RoundEvent, scenario: str, mode: str) -> RoundResult:
        turn_started = time.perf_counter()
        round_context = self.build_round_context(event, scenario, mode)
        state = round_context["state"]
        endogenous_turn = round_context["endogenous_turn"]
        recorded_at = round_context["recorded_at"]
        collected = self.collect_parallel_contributions(
            round_context=round_context,
            scenario=scenario,
        )
        arbitration = self.integrate_and_arbitrate(
            round_context=round_context,
            collected=collected,
            scenario=scenario,
            turn_started=turn_started,
        )
        trace = arbitration["trace"]
        sampled_action = arbitration["sampled_action"]
        rendered_expression = arbitration["rendered_expression"]
        control_ledger = arbitration["control_ledger"]

        return self.render_and_commit(
            state=state,
            trace=trace,
            sampled_action=sampled_action,
            rendered_expression=rendered_expression,
            scenario=scenario,
            endogenous_turn=endogenous_turn,
            control_ledger=control_ledger,
            recorded_at=recorded_at,
        )

    def tick(self, event: RoundEvent, scenario: str, mode: str) -> RoundResult:
        return self.round_pipeline.tick(event, scenario, mode)

    def _maybe_schedule_endogenous_followup(self, *, result: RoundResult, event: RoundEvent, scenario: str) -> None:
        current_suppression = to_dict(result.state.endogenous_state.last_suppression) if result.state.endogenous_state.last_suppression else {}
        current_scheduler_reason = str(result.state.endogenous_scheduler_state.suppression_reason or "")

        def _persist_suppression_if_changed(suppression: EndogenousSuppressionDecision) -> None:
            next_scheduler_state = self.endogenous_scheduler.update_state(
                scheduler_state=result.state.endogenous_scheduler_state,
                trigger=None,
                suppression_reason=suppression.reason,
            )
            result.state.endogenous_scheduler_state = next_scheduler_state
            result.state.endogenous_state.last_suppression = suppression
            if (
                current_suppression == to_dict(suppression)
                and current_scheduler_reason == str(suppression.reason or "")
            ):
                return
            self._save_state(result.state, sync=False)

        if event.source != "user" or str(result.state.mode or "") != "interactive":
            suppression = EndogenousSuppressionDecision(
                suppressed=True,
                reason="non_user_or_non_interactive_turn",
                details={"event_source": event.source, "mode": str(result.state.mode or "")},
            )
            _persist_suppression_if_changed(suppression)
            return
        if scenario == "task":
            _persist_suppression_if_changed(
                EndogenousSuppressionDecision(suppressed=True, reason="task_scenario")
            )
            return
        if scenario != "companion":
            _persist_suppression_if_changed(
                EndogenousSuppressionDecision(suppressed=True, reason="non_companion_scenario")
            )
            return
        trigger_context = self._build_endogenous_trigger_context(
            state=result.state,
            scenario=scenario,
            event=event,
            trace_payload=to_dict(result.trace),
        )
        trigger = self.endogenous_scheduler.build_trigger(
            state=result.state,
            context=trigger_context.context,
            relation_state=trigger_context.relation_state,
            slow_variables=trigger_context.slow_variables,
            pool_state=result.state.motivation_pool_state,
        )
        suppression = self._endogenous_suppression_decision(
            state=result.state,
            scenario=scenario,
            trigger=trigger,
        )
        if suppression.suppressed:
            _persist_suppression_if_changed(suppression)
        else:
            result.state.endogenous_scheduler_state = self.endogenous_scheduler.update_state(
                scheduler_state=result.state.endogenous_scheduler_state,
                trigger=None,
                suppression_reason=suppression.reason or None,
            )
            result.state.endogenous_state.last_suppression = None
            self._save_state(result.state, sync=False)
        if suppression.suppressed or trigger is None:
            return
        self.run_endogenous_tick(
            trigger=trigger.trigger_type,
            mode=trigger.selected_mode or None,
            scenario=scenario,
            trigger_context=trigger_context,
            trigger_payload=trigger,
        )

    def run_endogenous_tick(
        self,
        *,
        trigger: str = "idle",
        mode: str | None = None,
        scenario: str | None = None,
        trigger_context: EndogenousTriggerContext | None = None,
        trigger_payload: EndogenousTickTrigger | None = None,
        respect_suppression: bool = False,
    ) -> dict[str, Any]:
        return self.round_pipeline.run_endogenous_tick(trigger=trigger, mode=mode, scenario=scenario, trigger_context=trigger_context, trigger_payload=trigger_payload, respect_suppression=respect_suppression)

    def execute_command(self, envelope: CommandEnvelope) -> CommandResult:
        return self.run_runtime.execute_command(envelope)

    def apply_command(self, command: str, envelope=None) -> CommandResult:
        effective_envelope = envelope or self._legacy_command_envelope(command)
        if not effective_envelope.canonical:
            effective_envelope.canonical = command
        return self.execute_command(effective_envelope)

    def checkpoint(self) -> CheckpointRef:
        return self.state_runtime.checkpoint()

    def rewind(self, checkpoint_id: str) -> CommandResult:
        return self.state_runtime.rewind(checkpoint_id)

    def memory_top(self, limit: int = 5) -> list[dict]:
        return self.state_runtime.memory_top(limit=limit)

    def memory_recall(self, cue: str) -> dict[str, Any]:
        return self.state_runtime.memory_recall(cue)

    def compact_memory(self, *, hot_max_rounds: int = 500, warm_max_rounds: int = 3000) -> dict[str, Any]:
        return self.state_runtime.compact_memory(hot_max_rounds=hot_max_rounds, warm_max_rounds=warm_max_rounds)

    def sample_memory(self, tier: str, *, limit: int = 5, cue: str | None = None) -> list[dict[str, Any]]:
        return self.state_runtime.sample_memory(tier, limit=limit, cue=cue)

    def export_trace_parquet(self, *, since_round: int | None = None, overwrite: bool = False) -> dict[str, Any]:
        return self.state_runtime.export_trace_parquet(since_round=since_round, overwrite=overwrite)

    def habit_top(self, limit: int = 5) -> list[dict]:
        return self.state_runtime.habit_top(limit=limit)

    def identity_payload(self) -> dict[str, Any]:
        return self.state_runtime.identity_payload()

    def observer_settings_payload(self) -> dict[str, Any]:
        return self.state_runtime.observer_settings_payload()

    def _apply_startup_unlock_preferences(self, state: RuntimeState) -> None:
        self.state_runtime._apply_startup_unlock_preferences(state)

    def apply_startup_unlock_preferences(self, state: RuntimeState) -> None:
        self.state_runtime.apply_startup_unlock_preferences(state)

    def initiative_status_from_state(self, state: RuntimeState) -> dict[str, Any]:
        return self.state_runtime.initiative_status_from_state(state)

    def save_runtime_state(self, state: RuntimeState, *, sync: bool = False) -> None:
        self.state_runtime.save_runtime_state(state, sync=sync)

    def console_round_or_none(self) -> int | None:
        return self._console_round_or_none()

    def console_default_why_not_action(
        self,
        round_ref: int | str | None,
        *,
        action_field: dict[str, Any] | None = None,
    ) -> str | None:
        return self._console_default_why_not_action(round_ref, action_field=action_field)

    def console_recent_rounds(self, *, limit: int = 12) -> list[dict[str, Any]]:
        return self._console_recent_rounds(limit=limit)

    def console_source_links(
        self,
        *,
        round_id: int | None,
        trace_ref: str | None,
        why_not_action: str | None,
    ) -> list[dict[str, Any]]:
        return self._console_source_links(round_id=round_id, trace_ref=trace_ref, why_not_action=why_not_action)

    def update_observer_settings(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.state_runtime.update_observer_settings(payload)

    def state_payload(self) -> dict[str, Any]:
        return self.state_runtime.state_payload()

    def state_hot_payload(self) -> dict[str, Any]:
        return self.state_runtime.state_hot_payload()

    def latest_runtime_metrics(self) -> dict[str, Any]:
        return self.diagnostics_runtime.latest_runtime_metrics()

    def runtime_performance_payload(self) -> dict[str, Any]:
        return self.diagnostics_runtime.performance_payload()

    def _initiative_state_bucket(self, state: RuntimeState) -> dict[str, Any]:
        return self.run_runtime._initiative_state_bucket(state)

    def _monologue_state_bucket(self, state: RuntimeState) -> dict[str, Any]:
        return self.run_runtime._monologue_state_bucket(state)

    def prepare_monologue_stream_advance(
        self,
        *,
        state: RuntimeState | None = None,
        now_iso: str | None = None,
    ) -> dict[str, Any]:
        return self.run_runtime.prepare_monologue_stream_advance(state=state, now_iso=now_iso)

    def execute_monologue_stream_advance(self, ticket: dict[str, Any]) -> list[dict[str, Any]]:
        return self.run_runtime.execute_monologue_stream_advance(ticket)

    def commit_monologue_stream_advance(self, ticket: dict[str, Any], generated: list[dict[str, Any]]) -> dict[str, Any]:
        return self.run_runtime.commit_monologue_stream_advance(ticket, generated)

    def advance_monologue_stream_background(self, *, now_iso: str | None = None) -> dict[str, Any]:
        return self.run_runtime.advance_monologue_stream_background(now_iso=now_iso)

    def _sync_monologue_stream(
        self,
        state: RuntimeState,
        *,
        mark_viewed: bool = False,
        generate_if_due: bool = True,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        return self.run_runtime._sync_monologue_stream(
            state,
            mark_viewed=mark_viewed,
            generate_if_due=generate_if_due,
        )

    def _refresh_heartbeat_side_channels(self, state: RuntimeState) -> dict[str, Any]:
        return self.run_runtime._refresh_heartbeat_side_channels(state)

    def _heartbeat_side_channel_refresh_due(self, state: RuntimeState, *, noop_heartbeat: bool) -> bool:
        return self.run_runtime._heartbeat_side_channel_refresh_due(state, noop_heartbeat=noop_heartbeat)

    def _collect_monologue_stream_fragments(
        self,
        state: RuntimeState,
        *,
        allow_model_generation: bool = True,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
        return self.run_runtime._collect_monologue_stream_fragments(
            state,
            allow_model_generation=allow_model_generation,
        )

    def _monologue_stream_trace_payload(
        self,
        *,
        bucket: dict[str, Any],
        recent_fragments: list[dict[str, Any]],
        generated: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return self.run_runtime._monologue_stream_trace_payload(
            bucket=bucket,
            recent_fragments=recent_fragments,
            generated=generated,
        )

    def _monologue_stream_pressure_payload(
        self,
        *,
        recent_fragments: list[dict[str, Any]],
        generated: list[dict[str, Any]],
    ) -> dict[str, float]:
        return self.run_runtime._monologue_stream_pressure_payload(
            recent_fragments=recent_fragments,
            generated=generated,
        )

    def _monologue_stream_pressure_score(self, state: RuntimeState) -> float:
        return self.run_runtime._monologue_stream_pressure_score(state)

    def _build_monologue_stream_action_contribution(
        self,
        *,
        state: RuntimeState,
        scenario: str,
        endogenous_turn: bool,
    ) -> tuple[ProbabilisticContribution | None, dict[str, Any]]:
        return self.run_runtime._build_monologue_stream_action_contribution(
            state=state,
            scenario=scenario,
            endogenous_turn=endogenous_turn,
        )

    def _generate_monologue_fragments_via_model(
        self,
        *,
        seed: str,
        pulse_index: int,
        pulse_dt: str,
        fragment_count: int,
    ) -> list[dict[str, Any]]:
        return self.run_runtime._generate_monologue_fragments_via_model(
            seed=seed,
            pulse_index=pulse_index,
            pulse_dt=pulse_dt,
            fragment_count=fragment_count,
        )

    def _initiative_session_selection(self, *, now_iso: str | None = None) -> dict[str, Any]:
        return self.run_runtime._initiative_session_selection(now_iso=now_iso)

    def _initiative_active_session(self) -> dict[str, Any] | None:
        return self.run_runtime._initiative_active_session()

    def _initiative_has_pending_approval(self, active_session: dict[str, Any] | None) -> bool:
        return self.run_runtime._initiative_has_pending_approval(active_session)

    def _initiative_recent_history_stats(
        self,
        history: list[dict[str, Any]],
        *,
        now_iso: str,
        settings: dict[str, Any],
    ) -> dict[str, Any]:
        return self.run_runtime._initiative_recent_history_stats(
            history,
            now_iso=now_iso,
            settings=settings,
        )

    def _initiative_topic_cues(
        self,
        *,
        current_goal: str | None = None,
        active_session: dict[str, Any] | None = None,
    ) -> list[tuple[str, str]]:
        return self.run_runtime._initiative_topic_cues(
            current_goal=current_goal,
            active_session=active_session,
        )

    def _initiative_memory_backing(
        self,
        cue: str | None = None,
        *,
        current_goal: str | None = None,
        active_session: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.run_runtime._initiative_memory_backing(
            cue,
            current_goal=current_goal,
            active_session=active_session,
        )

    def _initiative_active_run_payload(self, state: RuntimeState) -> dict[str, Any] | None:
        return self.run_runtime._initiative_active_run_payload(state)

    def prepare_initiative_background(
        self,
        *,
        state: RuntimeState | None = None,
        latest_recorded_at: str | None = None,
    ) -> dict[str, Any]:
        return self.run_runtime.prepare_initiative_background(state=state, latest_recorded_at=latest_recorded_at)

    def evaluate_initiative_background(self, ticket: dict[str, Any]) -> dict[str, Any]:
        return self.run_runtime.evaluate_initiative_background(ticket)

    def commit_initiative_background(self, ticket: dict[str, Any], evaluation: dict[str, Any]) -> dict[str, Any]:
        return self.run_runtime.commit_initiative_background(ticket, evaluation)

    def advance_initiative_background(self, *, latest_recorded_at: str | None = None) -> dict[str, Any]:
        return self.run_runtime.advance_initiative_background(latest_recorded_at=latest_recorded_at)

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
        return self.run_runtime._initiative_distribution_payload(
            state,
            relation_state=relation_state,
            vitality_snapshot=vitality_snapshot,
            cue=cue,
            memory_backing=memory_backing,
            latest_recorded_at=latest_recorded_at,
            current_goal=current_goal,
        )

    def _expressive_action_biases(
        self,
        payload: dict[str, Any],
        *,
        scenario: str,
        endogenous_turn: bool,
    ) -> dict[str, float]:
        return self.round_pipeline._expressive_action_biases(
            payload,
            scenario=scenario,
            endogenous_turn=endogenous_turn,
        )

    def _build_expressive_action_contribution(
        self,
        *,
        current_distribution: dict[str, float],
        payload: dict[str, Any],
        scenario: str,
        endogenous_turn: bool,
    ) -> ProbabilisticContribution | None:
        return self.round_pipeline._build_expressive_action_contribution(
            current_distribution=current_distribution,
            payload=payload,
            scenario=scenario,
            endogenous_turn=endogenous_turn,
        )

    def _initiative_status_payload(self, state: RuntimeState) -> dict[str, Any]:
        return self.run_runtime._initiative_status_payload(state)

    def initiative_status(self) -> dict[str, Any]:
        return self.run_runtime.initiative_status()

    def initiative_distribution(self) -> dict[str, Any]:
        return self.run_runtime.initiative_distribution()

    def monologue_status(self) -> dict[str, Any]:
        return self.run_runtime.monologue_status()

    def monologue_status_lightweight(self) -> dict[str, Any]:
        return self.run_runtime.monologue_status_lightweight()

    def monologue_show(self, limit: int | None = None) -> dict[str, Any]:
        return self.run_runtime.monologue_show(limit=limit)

    def monologue_show_lightweight(self, limit: int | None = None) -> dict[str, Any]:
        return self.run_runtime.monologue_show_lightweight(limit=limit)

    def _initiative_force_settings(self, settings: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.run_runtime._initiative_force_settings(settings)

    def initiative_update_settings(self, **settings: Any) -> dict[str, Any]:
        return self.run_runtime.initiative_update_settings(**settings)

    def _initiative_why_summary(self, initiative: dict[str, Any] | None) -> str:
        return self.run_runtime._initiative_why_summary(initiative)

    def initiative_why(self, round_ref: int | str = "last") -> dict[str, Any]:
        return self.run_runtime.initiative_why(round_ref)

    def initiative_trigger_now(self, *, trigger: str = "idle", mode: str | None = None, force: bool = False) -> dict[str, Any]:
        return self.run_runtime.initiative_trigger_now(trigger=trigger, mode=mode, force=force)

    def _initiative_delivery_text(self, *, proposal: dict[str, Any], message: str) -> str:
        return self.run_runtime._initiative_delivery_text(proposal=proposal, message=message)

    def _initiative_context_memory_backing(
        self,
        *,
        cue: Any,
        recall_strength: Any,
        fallback_summary: str,
    ) -> dict[str, Any]:
        return self.run_runtime._initiative_context_memory_backing(
            cue=cue,
            recall_strength=recall_strength,
            fallback_summary=fallback_summary,
        )

    def thought_snapshot(self, round_ref: int | str = "last") -> dict[str, Any]:
        return self.observer_runtime.thought_snapshot(round_ref)

    def empty_thought_payload(self, round_ref: int | str | None = None) -> dict[str, Any]:
        return self.observer_runtime.empty_thought_payload(round_ref)

    def _record_initiative_feedback_delta(self, text: str) -> float:
        return self.run_runtime._record_initiative_feedback_delta(text)

    def record_initiative_feedback(self, text: str, *, session_id: str | None = None, target: str = "user") -> dict[str, Any]:
        return self.run_runtime.record_initiative_feedback(text, session_id=session_id, target=target)

    def _dispatch_initiative_to_session(self, proposal: dict[str, Any], text: str) -> tuple[bool, str | None]:
        return self.run_runtime._dispatch_initiative_to_session(proposal, text)

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
        return self.run_runtime._record_initiative_outcome(
            round_id=round_id,
            recorded_at=recorded_at,
            proposal=proposal,
            message=message,
            auto_sent=auto_sent,
            session_id=session_id,
        )

    def autonomy_status(self) -> dict[str, Any]:
        return self._autonomy_status_payload(include_diagnostics=True)

    def autonomy_runtime_status(self) -> dict[str, Any]:
        return self._autonomy_status_payload(include_diagnostics=False)

    def _autonomy_status_payload(self, *, include_diagnostics: bool) -> dict[str, Any]:
        return self.run_runtime._autonomy_status_payload(include_diagnostics=include_diagnostics)

    def start_autonomy(self, profile: str = "tool_level", *, clear_safe_mode: bool = False) -> dict[str, Any]:
        return self.run_runtime.start_autonomy(profile=profile, clear_safe_mode=clear_safe_mode)

    def stop_autonomy(self, reason: str = "manual_stop") -> dict[str, Any]:
        return self.run_runtime.stop_autonomy(reason=reason)

    def _prepare_autonomy_step_state(self, state: RuntimeState | None = None) -> RuntimeState:
        return self.run_runtime._prepare_autonomy_step_state(state)

    def _autonomy_default_self_run_goal(self, state: RuntimeState) -> str:
        return self.run_runtime._autonomy_default_self_run_goal(state)

    def prepare_autonomy_background(self, *, state: RuntimeState | None = None) -> dict[str, Any]:
        return self.run_runtime.prepare_autonomy_background(state=state)

    def execute_autonomy_background(self, ticket: dict[str, Any]) -> dict[str, Any]:
        return self.run_runtime.execute_autonomy_background(ticket)

    def commit_autonomy_background(
        self,
        ticket: dict[str, Any],
        execution: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.run_runtime.commit_autonomy_background(ticket, execution)

    def autonomy_step(self) -> dict[str, Any]:
        return self.run_runtime.autonomy_step()

    def _autonomy_after_command_step(self, result: dict[str, Any]) -> dict[str, Any]:
        return self.run_runtime._autonomy_after_command_step(result)

    def _autonomy_run_dream_pass(self, state: RuntimeState) -> dict[str, Any]:
        return self.run_runtime._autonomy_run_dream_pass(state)

    def _autonomy_self_run_goal(self, state: RuntimeState) -> str:
        return self.run_runtime._autonomy_self_run_goal(state)

    def _autonomy_continue_readonly_run_directive(
        self,
        state: RuntimeState,
        policy: AutonomyPolicyState,
    ) -> dict[str, str] | None:
        return self.run_runtime._autonomy_continue_readonly_run_directive(state, policy)

    def _autonomy_start_readonly_run(self, goal: str) -> dict[str, Any]:
        return self.run_runtime._autonomy_start_readonly_run(goal)

    def _autonomy_continue_readonly_run(self, run_id: str, call_id: str) -> dict[str, Any]:
        return self.run_runtime._autonomy_continue_readonly_run(run_id, call_id)

    def _autonomy_rest_realization(
        self,
        state: RuntimeState,
        policy: AutonomyPolicyState,
    ) -> dict[str, str] | None:
        return self.run_runtime._autonomy_rest_realization(state, policy)

    def _console_round_or_none(self) -> int | None:
        return self.observer_runtime._console_round_or_none()

    def console_state(self) -> dict[str, Any]:
        return self.observer_runtime.console_state()

    def console_action_field(self, round_ref: int | str | None = None) -> dict[str, Any]:
        return self.observer_runtime.console_action_field(round_ref)

    def console_timeline(self, round_ref: int | str | None = None) -> dict[str, Any]:
        return self.observer_runtime.console_timeline(round_ref)

    def console_why_current(self, round_ref: int | str | None = None) -> dict[str, Any]:
        return self.observer_runtime.console_why_current(round_ref)

    def console_why_not(self, action: str, round_ref: int | str | None = None) -> dict[str, Any]:
        return self.observer_runtime.console_why_not(action, round_ref)

    def _why_not_summary(
        self,
        *,
        action: str,
        selected_action: str,
        candidate_score: float,
        blocked_by: list[str],
        expressive_trace: dict[str, Any],
    ) -> str:
        return self.observer_runtime._why_not_summary(
            action=action,
            selected_action=selected_action,
            candidate_score=candidate_score,
            blocked_by=blocked_by,
            expressive_trace=expressive_trace,
        )

    def _console_default_why_not_action(
        self,
        round_ref: int | str | None,
        *,
        action_field: dict[str, Any] | None = None,
    ) -> str | None:
        return self.observer_runtime._console_default_why_not_action(round_ref, action_field=action_field)

    def _console_recent_rounds(self, *, limit: int = 12) -> list[dict[str, Any]]:
        return self.observer_runtime._console_recent_rounds(limit=limit)

    def _console_source_links(
        self,
        *,
        round_id: int | None,
        trace_ref: str | None,
        why_not_action: str | None,
    ) -> list[dict[str, Any]]:
        return self.observer_runtime._console_source_links(
            round_id=round_id,
            trace_ref=trace_ref,
            why_not_action=why_not_action,
        )

    def console_refresh_payload(self, round_ref: int | str | None = None) -> dict[str, Any]:
        return self.observer_runtime.console_refresh_payload(round_ref)

    def console_recent_actions(self, *, limit: int = 8) -> dict[str, Any]:
        return self.observer_runtime.console_recent_actions(limit=limit)

    def console_probability_space(self, round_ref: int | str | None = None) -> dict[str, Any]:
        return self.observer_runtime.console_probability_space(round_ref)

    def state_delta_timeline(self, window: int = 20) -> dict[str, Any]:
        return self.observer_runtime.state_delta_timeline(window)

    def identity_blockers(self) -> dict[str, Any]:
        return self.observer_runtime.identity_blockers()

    def cue_fragmentation_report(self) -> dict[str, Any]:
        return self.observer_runtime.cue_fragmentation_report()

    def run_contamination_report(self, window: int = 20) -> dict[str, Any]:
        return self.observer_runtime.run_contamination_report(window)

    def why_no_change(self, round_ref: int | str | None = None) -> dict[str, Any]:
        return self.observer_runtime.why_no_change(round_ref)

    def empty_why_no_change_payload(self, round_ref: int | str | None = None) -> dict[str, Any]:
        return self.observer_runtime.empty_why_no_change_payload(round_ref)

    def migration_report(self) -> dict[str, Any]:
        return self.observer_runtime.migration_report()

    def runtime_storage_status(self) -> dict[str, Any]:
        return self.observer_runtime.runtime_storage_status()

    def _latest_why_payload(self, state: RuntimeState) -> dict[str, Any] | None:
        return self.observer_runtime._latest_why_payload(state)

    def _current_intent_summary(
        self,
        state: RuntimeState,
        latest_why: dict[str, Any] | None,
        run_payload: dict[str, Any] | None,
    ) -> str:
        return self.observer_runtime._current_intent_summary(state, latest_why, run_payload)

    def _identity_summary(self, state: RuntimeState, latest_why: dict[str, Any] | None) -> dict[str, Any]:
        return self.observer_runtime._identity_summary(state, latest_why)

    def _query_intent_label(self, query_intent: str) -> str:
        return self.observer_runtime._query_intent_label(query_intent)

    def _action_phrase(self, action_name: str) -> str:
        return self.observer_runtime._action_phrase(action_name)

    def _focus_label(self, focus: str) -> str:
        return self.observer_runtime._focus_label(focus)

    def _continuity_label(self, rename_reason: str, *, has_display_name: bool) -> str:
        return self.observer_runtime._continuity_label(rename_reason, has_display_name=has_display_name)

    def _authenticity_summary(self, latest_why: dict[str, Any] | None) -> dict[str, Any]:
        return self.observer_runtime._authenticity_summary(latest_why)

    def cognitive_snapshot(
        self,
        *,
        state: RuntimeState | None = None,
        run_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.observer_runtime.cognitive_snapshot(state=state, run_payload=run_payload)

    def internal_learning_summary(self, state: RuntimeState | None = None) -> dict[str, Any]:
        return self.observer_runtime.internal_learning_summary(state)

    def _trace_storage_payload(self, *, read_source: str | None = None) -> dict[str, Any]:
        return self.observer_runtime._trace_storage_payload(read_source=read_source)

    def trace_storage_status(self, *, read_source: str | None = None) -> dict[str, Any]:
        return self.observer_runtime.trace_storage_status(read_source=read_source)

    def _dream_summary_from_trace(self, trace: dict[str, Any]) -> dict[str, Any] | None:
        return self.observer_runtime._dream_summary_from_trace(trace)

    def dream_runs(self) -> dict[str, Any]:
        return self.observer_runtime.dream_runs()

    def dream_status(self) -> dict[str, Any]:
        return self.observer_runtime.dream_status()

    def dream_trace(self, run_ref: int | str | None) -> dict[str, Any]:
        return self.observer_runtime.dream_trace(run_ref)

    def dream_proposals(self, run_ref: int | str | None) -> dict[str, Any]:
        return self.observer_runtime.dream_proposals(run_ref)

    def dream_metrics(self) -> dict[str, Any]:
        return self.observer_runtime.dream_metrics()

    def _dream_semantic_summary(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.observer_runtime._dream_semantic_summary(payload)

    def dream_overview(self, *, limit: int = 6) -> dict[str, Any]:
        return self.observer_runtime.dream_overview(limit=limit)

    def set_dream_enabled(self, enabled: bool) -> dict[str, Any]:
        return self.run_runtime.set_dream_enabled(enabled)

    def run_dream(self, *, mode: str = "sleep", cue: str | None = None) -> dict[str, Any]:
        return self.run_runtime.run_dream(mode=mode, cue=cue)

    def _task_turn_message(self, run_payload: dict[str, Any]) -> str:
        return self.run_runtime._task_turn_message(run_payload)

    def _run_trace_ref(self, run_id: str) -> str:
        return self.run_runtime._run_trace_ref(run_id)

    def _run_step_trace_ref(self, run_id: str, step_id: str) -> str:
        return self.run_runtime._run_step_trace_ref(run_id, step_id)

    def _run_tool_trace_ref(self, run_id: str, call_id: str) -> str:
        return self.run_runtime._run_tool_trace_ref(run_id, call_id)

    def _run_status_payload(self, run_state: RunState) -> dict[str, Any]:
        return self.run_runtime._run_status_payload(run_state)

    def _run_explain_payload(self, run_state: RunState) -> dict[str, Any]:
        return self.run_runtime._run_explain_payload(run_state)

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
        return self.run_runtime.start_run(goal, allow_commit=allow_commit, operator_level=operator_level, replace_active=replace_active, interrupt_reason=interrupt_reason, bootstrap=bootstrap, include_details=include_details, sync_hot_path=sync_hot_path, defer_bootstrap_tool=defer_bootstrap_tool)

    def resolve_run_tool_approval(self, run_id: str, call_id: str, *, approved: bool) -> dict[str, Any]:
        return self.run_runtime.resolve_run_tool_approval(run_id, call_id, approved=approved)

    def interrupt_run(
        self,
        run_id: str | None = None,
        *,
        reason: str = "interrupted_by_user",
        message: str = "run interrupted",
    ) -> dict[str, Any]:
        return self.run_runtime.interrupt_run(run_id, reason=reason, message=message)

    def _resolve_run_id(self, run_id: str | None = None) -> str:
        return self.run_runtime._resolve_run_id(run_id)

    def _load_run_state(self, run_id: str | None = None) -> RunState:
        return self.run_runtime._load_run_state(run_id)

    def _persist_run_state(self, run_state: RunState) -> None:
        self.run_runtime._persist_run_state(run_state)

    def run_status(self, run_id: str | None = None) -> dict[str, Any]:
        return self.run_runtime.run_status(run_id)

    def pause_run(self, run_id: str | None = None) -> dict[str, Any]:
        return self.run_runtime.pause_run(run_id)

    def resume_run(self, run_id: str | None = None) -> dict[str, Any]:
        return self.run_runtime.resume_run(run_id)

    def abort_run(self, run_id: str | None = None, *, reason: str = "operator_requested") -> dict[str, Any]:
        return self.run_runtime.abort_run(run_id, reason=reason)

    def explain_run(self, run_id: str | None = None) -> dict[str, Any]:
        return self.run_runtime.explain_run(run_id)

    def run_steps(self, run_id: str | None = None) -> dict[str, Any]:
        return self.run_runtime.run_steps(run_id)

    def run_tools(self, run_id: str | None = None) -> dict[str, Any]:
        return self.run_runtime.run_tools(run_id)

    def resolve_round_ref(self, round_ref: int | str | None) -> int:
        return self.observer_runtime.resolve_round_ref(round_ref)

    def trace_round(self, round_ref: int | str) -> dict[str, Any]:
        return self.observer_runtime.trace_round(round_ref)

    def empty_trace_payload(self, round_ref: int | str | None = None) -> dict[str, Any]:
        return self.observer_runtime.empty_trace_payload(round_ref)

    def why_this(self, round_ref: int | str) -> dict[str, Any]:
        return self.observer_runtime.why_this(round_ref)

    def empty_why_payload(self, round_ref: int | str | None = None) -> dict[str, Any]:
        return self.observer_runtime.empty_why_payload(round_ref)

    def empty_expression_channel_payload(self, channel: str, round_ref: int | str | None = None) -> dict[str, Any]:
        return self.observer_runtime.empty_expression_channel_payload(channel, round_ref)

    def expression_channel_snapshot(self, channel: str, round_ref: int | str = "last") -> dict[str, Any]:
        return self.observer_runtime.expression_channel_snapshot(channel, round_ref)

    def _action_layer_from_trace(self, trace: dict[str, Any]) -> dict[str, Any]:
        return self.observer_runtime._observer_action_layer_from_trace(trace)

    def _trace_derived_cache_entry(self, trace: dict[str, Any]) -> dict[str, Any] | None:
        return self.observer_runtime._observer_trace_derived_cache_entry(trace)

    def _token_state_from_trace(self, trace: dict[str, Any]) -> dict[str, Any]:
        return self.observer_runtime._observer_token_state_from_trace(trace)

    def _cross_layer_coupling_verdict(self, trace: dict[str, Any]) -> dict[str, Any]:
        return self.observer_runtime._observer_cross_layer_coupling_verdict(trace)

    def _renderer_decision_integrity(self, trace: dict[str, Any]) -> dict[str, Any]:
        return self.observer_runtime._observer_renderer_decision_integrity(trace)

    def _competing_peaks(self, action_layer: dict[str, Any], *, limit: int = 3) -> list[dict[str, Any]]:
        return self.observer_runtime._observer_competing_peaks(action_layer, limit=limit)

    def _stacked_action_contributions(
        self,
        action_layer: dict[str, Any],
        target_action: str,
        *,
        limit: int = 16,
    ) -> list[dict[str, Any]]:
        return self.observer_runtime._observer_stacked_action_contributions(
            action_layer,
            target_action,
            limit=limit,
        )

    def _action_probability_explanation(self, trace: dict[str, Any], *, target_action: str) -> dict[str, Any]:
        return self.observer_runtime._observer_action_probability_explanation(
            trace,
            target_action=target_action,
        )

    def _counterfactual_render_preview(self, trace: dict[str, Any], action: str) -> dict[str, Any]:
        return self.observer_runtime._observer_counterfactual_render_preview(trace, action)

    def _emergent_action_formalization_payload(self, sketches: list[dict[str, Any]] | list[Any]) -> list[dict[str, Any]]:
        return self.observer_runtime._observer_emergent_action_formalization_payload(sketches)

    def _sample_counterfactual_action(
        self,
        distribution: dict[str, float],
        *,
        seed: int,
        default_action: str,
    ) -> str:
        return self.observer_runtime._observer_sample_counterfactual_action(
            distribution,
            seed=seed,
            default_action=default_action,
        )

    def _counterfactual_replays_from_trace(self, trace: dict[str, Any], *, limit: int = 4) -> list[dict[str, Any]]:
        return self.observer_runtime._observer_counterfactual_replays_from_trace(trace, limit=limit)

    def _conflict_arbitration_summary(self, trace: dict[str, Any]) -> dict[str, Any]:
        return self.observer_runtime._observer_conflict_arbitration_summary(trace)

    def contribution_breakdown(self, round_ref: int | str) -> dict[str, Any]:
        return self.observer_runtime.contribution_breakdown(round_ref)

    def trace_probability_field(self, round_ref: int | str) -> dict[str, Any]:
        return self.observer_runtime.trace_probability_field(round_ref)

    def trace_probability_layer(self, round_ref: int | str, *, layer: str) -> dict[str, Any]:
        return self.observer_runtime.trace_probability_layer(round_ref, layer=layer)

    def trace_action_probability(self, round_ref: int | str, *, action: str) -> dict[str, Any]:
        return self.observer_runtime.trace_action_probability(round_ref, action=action)

    def trace_agents(self, round_ref: int | str) -> dict[str, Any]:
        return self.observer_runtime.trace_agents(round_ref)

    def trace_skills(self, round_ref: int | str) -> dict[str, Any]:
        return self.observer_runtime.trace_skills(round_ref)

    def trace_gates(self, round_ref: int | str) -> dict[str, Any]:
        return self.observer_runtime.trace_gates(round_ref)

    def _average(self, values: list[float]) -> float:
        return self.observer_runtime._observer_average(values)

    def _estimate_affect_half_life(self, rounds: list[dict[str, Any]]) -> float:
        return self.observer_runtime._observer_estimate_affect_half_life(rounds)

    def _estimate_recovery_duration(self, rounds: list[dict[str, Any]]) -> float:
        return self.observer_runtime._observer_estimate_recovery_duration(rounds)

    def _event_variance_metric(self, rounds: list[dict[str, Any]], bucket_fn) -> float:
        return self.observer_runtime._observer_event_variance_metric(rounds, bucket_fn)

    def metrics_summary(self) -> dict[str, Any]:
        return self.observer_runtime.metrics_summary()

    def authenticity_timeline(self) -> dict[str, Any]:
        return self.observer_runtime.authenticity_timeline()

    def vitality_timeline(self) -> dict[str, Any]:
        return self.observer_runtime.vitality_timeline()

    def motivation_metrics(self) -> dict[str, Any]:
        return self.observer_runtime.motivation_metrics()

    def endogenous_metrics(self) -> dict[str, Any]:
        return self.observer_runtime.endogenous_metrics()

    def agent_list(self) -> list[dict[str, Any]]:
        return self.observer_runtime.agent_list()

    def skill_list(self) -> list[dict[str, Any]]:
        return self.observer_runtime.skill_list()

    def skill_stats(self) -> dict[str, Any]:
        return self.observer_runtime.skill_stats()

    def skill_profile(self, skill_name: str) -> dict[str, Any]:
        return self.observer_runtime.skill_profile(skill_name)

    def model_status(self) -> dict[str, Any]:
        return self.observer_runtime.model_status()

    def relation_show(self, target: str) -> dict[str, Any]:
        return self.observer_runtime.relation_show(target)

    def replay_round(self, round_id: int, seed: int | None = None) -> dict[str, Any]:
        return self.observer_runtime.replay_round(round_id, seed)

    def replay(self, round_id: int, seed: int = 0) -> dict[str, Any]:
        return self.observer_runtime.replay(round_id, seed)

    def empty_replay_payload(self, round_ref: int | str | None = None, *, seed: int = 0) -> dict[str, Any]:
        return self.observer_runtime.empty_replay_payload(round_ref, seed=seed)

    def replay_motivation(self, round_id: int) -> dict[str, Any]:
        return self.observer_runtime.replay_motivation(round_id)

    def empty_motivation_payload(self, round_ref: int | str | None = None) -> dict[str, Any]:
        return self.observer_runtime.empty_motivation_payload(round_ref)

    def why_motivation(self, round_ref: int | str) -> dict[str, Any]:
        return self.observer_runtime.why_motivation(round_ref)

    def why_not(self, round_id: int, action: str) -> dict[str, Any]:
        return self.observer_runtime.why_not(round_id, action)

    def empty_why_not_payload(self, action: str = "", round_ref: int | str | None = None) -> dict[str, Any]:
        return self.observer_runtime.empty_why_not_payload(action, round_ref)

    def what_changed(self, window: int = 5) -> dict[str, Any]:
        return self.observer_runtime.what_changed(window)

    def conflict_timeline(self) -> dict[str, Any]:
        return self.observer_runtime.conflict_timeline()

    def entropy_metrics(self) -> dict[str, Any]:
        return self.observer_runtime.entropy_metrics()

    def mode_switch_timeline(self) -> dict[str, Any]:
        return self.observer_runtime.mode_switch_timeline()

    def ablation_summary(self) -> dict[str, Any]:
        return self.observer_runtime.ablation_summary()

    def metrics_timeline(self) -> dict[str, Any]:
        return self.observer_runtime.metrics_timeline()

    def metrics_heatmap(self) -> dict[str, Any]:
        return self.observer_runtime.metrics_heatmap()

    def _bypass_detection_summary(self, rounds: list[dict[str, Any]]) -> dict[str, Any]:
        return self.observer_runtime._observer_bypass_detection_summary(rounds)

    def _parallel_evidence_summary(self, rounds: list[dict[str, Any]]) -> dict[str, Any]:
        return self.observer_runtime._observer_parallel_evidence_summary(rounds)

    def _scale_consistency_summary(self, rounds: list[dict[str, Any]]) -> dict[str, Any]:
        return self.observer_runtime._observer_scale_consistency_summary(rounds)

    def _cross_layer_coupling_summary(self, rounds: list[dict[str, Any]]) -> dict[str, Any]:
        return self.observer_runtime._observer_cross_layer_coupling_summary(rounds)

    def _renderer_decision_integrity_summary(self, rounds: list[dict[str, Any]]) -> dict[str, Any]:
        return self.observer_runtime._observer_renderer_decision_integrity_summary(rounds)

    def _memory_write_gate_summary(self, rounds: list[dict[str, Any]]) -> dict[str, Any]:
        return self.observer_runtime._observer_memory_write_gate_summary(rounds)

    def _failure_taxonomy_from_trace(self, trace: dict[str, Any]) -> list[str]:
        return self.observer_runtime._observer_failure_taxonomy_from_trace(trace)

    def _failure_taxonomy_summary(self, rounds: list[dict[str, Any]]) -> dict[str, Any]:
        return self.observer_runtime._observer_failure_taxonomy_summary(rounds)

    def _conflict_arbitration_acceptance_summary(self, rounds: list[dict[str, Any]]) -> dict[str, Any]:
        return self.observer_runtime._observer_conflict_arbitration_acceptance_summary(rounds)

    def _long_run_prior_summary(self, rounds: list[dict[str, Any]]) -> dict[str, Any]:
        return self.observer_runtime._observer_long_run_prior_summary(rounds)

    def acceptance_report(self, window: int = 20) -> dict[str, Any]:
        return self.observer_runtime.acceptance_report(window)

    def compact_traces(self) -> dict[str, Any]:
        return self.observer_runtime.compact_traces()

    def eval_longrun(self, rounds: int = 1000) -> dict[str, Any]:
        return self.observer_runtime.eval_longrun(rounds)
