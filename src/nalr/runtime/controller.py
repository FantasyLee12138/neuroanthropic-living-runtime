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
from nalr.runtime.longrun import LongRunAnalyzer
from nalr.runtime.metadata import iso_date, utc_now_iso
from nalr.runtime.motivation_feedback import MotivationFeedbackUpdater
from nalr.runtime.motivation_pool import EndogenousMotivationPool
from nalr.runtime.probability_field import ProbabilityFieldIntegrator, compute_tlh_vector_collapse
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


INNATE_ACTIONS = ("respond", "rest", "absorb", "wander", "nothing", "die")
DERIVED_ACTIONS = ("plan", "recall", "connect", "clarify")
CORE_ACTIONS = ("respond", "plan", "recall", "rest", "connect", "clarify", "wander", "absorb", "nothing", "die")
INTERNAL_RUNTIME_ACTIONS = CORE_ACTIONS + ("short_reply",)
INNATE_ACTIONS = ("respond", "rest", "absorb", "wander", "nothing", "die")
DERIVED_ACTIONS = ("plan", "recall", "connect", "clarify")
MODEL_ROUTE_SKILL_BINDINGS: dict[str, tuple[str, ...]] = {
    "planner": (),
    "pfc": ("generate_candidates",),
    "perspective": ("infer_other_state", "simulate_other_reaction"),
    "renderer": ("render_expression",),
}
MODEL_ROUTE_BINDING_KEYS: dict[str, str] = {
    "planner": "planner",
    "pfc": "PFCAgent",
    "perspective": "PerspectiveModel",
    "renderer": "Renderer",
}

ARCHITECTURE_CORE_GOAL = "维持生命性、真实性与连续性"
RUNTIME_SCHEMA_VERSION = 2


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
        self._state_io = AsyncIOWorker("nalr-state-io")
        self._state_cache: RuntimeState | None = None
        self._cold_flush_round_interval = 8
        self._cold_flush_interval_seconds = 2.0
        self._rounds_since_cold_flush = 0
        self._last_cold_flush_at = time.monotonic()

        self.config = self._load_config()
        thresholds = self.config["thresholds"]["thresholds"]
        self.trace_store = TraceStore(self.home_path)
        self.memory_store = MemoryStore(self.home_path)
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
        return {
            "autonomy": {
                "clear_safe_mode_on_start": True,
            },
            "newborn": {
                "disable_safe_mode_lock": True,
                "organic_mode": {
                    "enabled": True,
                    "instinct_first": True,
                    "body_weight": 0.08,
                    "subjective_weight": 0.08,
                    "guard_relaxation": 0.12,
                    "endogenous_autonomy": 0.1,
                },
                "subjective_state": {
                    "felt": ["微弱自发冲动"],
                    "spontaneous": 0.08,
                    "boundary": 0.02,
                    "reject_all": 0.0,
                    "meaning_made": [],
                },
            },
            "models": {
                "supported_backends": ["fake", "deepseek", "doubao", "openai_compatible"],
                "provider_endpoints": {
                    "large_model_api": {
                        "backend": "doubao",
                        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
                        "api_key_env": "ARK_API_KEY",
                        "model": "doubao-seed-2-0-pro-260215",
                    },
                    "local_model_api": {
                        "backend": "openai_compatible",
                        "base_url": "http://127.0.0.1:11434/v1",
                        "api_key_env": "LOCAL_MODEL_API_KEY",
                        "model": "qwen2.5:7b-instruct",
                    },
                },
                "model_tiers": {
                    "local_model": {
                        "mode": "remote",
                        "backend": "openai_compatible",
                        "base_url": "http://127.0.0.1:11434/v1",
                        "model": "qwen2.5:7b-instruct",
                        "timeout_ms": 20000,
                        "retries": 0,
                        "api_key_env": "LOCAL_MODEL_API_KEY",
                        "enabled": False,
                    }
                },
                "agent_model_bindings": {},
                "model_routes": {},
            },
        }

    def _load_observer_settings_file(self) -> dict[str, Any]:
        defaults = self._default_observer_settings()
        if not self.observer_settings_path.exists():
            return defaults
        try:
            payload = json.loads(self.observer_settings_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return defaults
        return self._normalize_observer_settings(payload, base=defaults)

    def _normalize_observer_settings(self, payload: dict[str, Any], *, base: dict[str, Any] | None = None) -> dict[str, Any]:
        defaults = json.loads(json.dumps(base or self._default_observer_settings(), ensure_ascii=False))
        data = payload if isinstance(payload, dict) else {}
        autonomy = data.get("autonomy", {}) if isinstance(data.get("autonomy"), dict) else {}
        defaults["autonomy"]["clear_safe_mode_on_start"] = bool(
            autonomy.get("clear_safe_mode_on_start", defaults["autonomy"]["clear_safe_mode_on_start"])
        )

        newborn = data.get("newborn", {}) if isinstance(data.get("newborn"), dict) else {}
        defaults["newborn"]["disable_safe_mode_lock"] = bool(
            newborn.get("disable_safe_mode_lock", defaults["newborn"]["disable_safe_mode_lock"])
        )
        organic = newborn.get("organic_mode", {}) if isinstance(newborn.get("organic_mode"), dict) else {}
        for key in ("enabled", "instinct_first"):
            defaults["newborn"]["organic_mode"][key] = bool(
                organic.get(key, defaults["newborn"]["organic_mode"][key])
            )
        for key in ("body_weight", "subjective_weight", "guard_relaxation", "endogenous_autonomy"):
            defaults["newborn"]["organic_mode"][key] = round(
                _clip(float(organic.get(key, defaults["newborn"]["organic_mode"][key]))), 4
            )
        subjective = newborn.get("subjective_state", {}) if isinstance(newborn.get("subjective_state"), dict) else {}
        defaults["newborn"]["subjective_state"]["felt"] = self._normalize_observer_string_list(
            list(subjective.get("felt", defaults["newborn"]["subjective_state"]["felt"]))
        )
        defaults["newborn"]["subjective_state"]["meaning_made"] = self._normalize_observer_string_list(
            list(subjective.get("meaning_made", defaults["newborn"]["subjective_state"]["meaning_made"]))
        )
        for key in ("spontaneous", "boundary", "reject_all"):
            defaults["newborn"]["subjective_state"][key] = round(
                _clip(float(subjective.get(key, defaults["newborn"]["subjective_state"][key]))), 4
            )

        models = data.get("models", {}) if isinstance(data.get("models"), dict) else {}
        provider_endpoints = models.get("provider_endpoints", {}) if isinstance(models.get("provider_endpoints"), dict) else {}
        for endpoint_name, endpoint in provider_endpoints.items():
            if not isinstance(endpoint, dict):
                continue
            target = defaults["models"]["provider_endpoints"].setdefault(endpoint_name, {})
            for key in ("backend", "base_url", "api_key_env", "model"):
                value = endpoint.get(key)
                if value is not None:
                    target[key] = str(value).strip()

        model_tiers = models.get("model_tiers", {}) if isinstance(models.get("model_tiers"), dict) else {}
        defaults["models"]["model_tiers"] = {
            key: self._normalize_model_config(value, tier_mode=True)
            for key, value in {**defaults["models"]["model_tiers"], **model_tiers}.items()
            if isinstance(value, dict)
        }
        model_routes = models.get("model_routes", {}) if isinstance(models.get("model_routes"), dict) else {}
        defaults["models"]["model_routes"] = {
            key: self._normalize_model_config(value, tier_mode=False)
            for key, value in model_routes.items()
            if isinstance(value, dict)
        }
        bindings = models.get("agent_model_bindings", {}) if isinstance(models.get("agent_model_bindings"), dict) else {}
        defaults["models"]["agent_model_bindings"] = {
            str(key): str(value).strip()
            for key, value in bindings.items()
            if str(value).strip()
        }
        return defaults

    def _normalize_model_config(self, payload: dict[str, Any], *, tier_mode: bool) -> dict[str, Any]:
        normalized = {
            "backend": str(payload.get("backend", "")).strip(),
            "model": str(payload.get("model", "")).strip(),
            "base_url": str(payload.get("base_url", "")).strip(),
            "api_key_env": str(payload.get("api_key_env", "")).strip(),
            "timeout_ms": int(payload.get("timeout_ms", 12000) or 12000),
            "retries": int(payload.get("retries", 0) or 0),
            "enabled": bool(payload.get("enabled", True)),
        }
        if tier_mode:
            normalized["mode"] = str(payload.get("mode", "remote")).strip() or "remote"
        return normalized

    def _normalize_observer_string_list(self, values: list[Any]) -> list[str]:
        normalized: list[str] = []
        for raw in list(values or []):
            value = str(raw).strip()
            if not value or value in normalized:
                continue
            normalized.append(value)
        return normalized

    def _apply_observer_settings_to_models(self, models_cfg: dict[str, Any], observer_settings: dict[str, Any]) -> dict[str, Any]:
        merged = json.loads(json.dumps(models_cfg, ensure_ascii=False))
        model_settings = observer_settings.get("models", {}) if isinstance(observer_settings.get("models"), dict) else {}
        route_overrides = model_settings.get("model_routes", {}) if isinstance(model_settings.get("model_routes"), dict) else {}
        tier_overrides = model_settings.get("model_tiers", {}) if isinstance(model_settings.get("model_tiers"), dict) else {}
        binding_overrides = model_settings.get("agent_model_bindings", {}) if isinstance(model_settings.get("agent_model_bindings"), dict) else {}
        merged.setdefault("model_routes", {})
        merged.setdefault("model_tiers", {})
        merged.setdefault("agent_model_bindings", {})
        for route_name, override in route_overrides.items():
            if not isinstance(override, dict):
                continue
            merged["model_routes"][route_name] = {**merged["model_routes"].get(route_name, {}), **override}
        for tier_name, override in tier_overrides.items():
            if not isinstance(override, dict):
                continue
            merged["model_tiers"][tier_name] = {**merged["model_tiers"].get(tier_name, {}), **override}
        for binding_key, override in binding_overrides.items():
            merged["agent_model_bindings"][binding_key] = str(override)
        return merged

    def _observer_settings(self) -> dict[str, Any]:
        return self.config.get("observer_settings", self._default_observer_settings())

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
        return RuntimeState(**to_dict(self._state_cache))

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
        observer_settings = self._observer_settings()
        newborn_settings = observer_settings.get("newborn", {}) if isinstance(observer_settings.get("newborn"), dict) else {}
        organic_settings = newborn_settings.get("organic_mode", {}) if isinstance(newborn_settings.get("organic_mode"), dict) else {}
        subjective_settings = newborn_settings.get("subjective_state", {}) if isinstance(newborn_settings.get("subjective_state"), dict) else {}
        state = RuntimeState(
            agents_enabled={
                name: agent_cfg.get("enabled", True)
                for name, agent_cfg in self.config["agents"]["agents"].items()
            }
        )
        state.subject_core = SubjectCore()
        state.identity_state = IdentityState()
        state.mode = "interactive"
        state.safe_mode = not bool(newborn_settings.get("disable_safe_mode_lock", True))
        state.round_count = 0
        state.body_energy = 0.7
        state.fatigue = 0.0
        state.memory_fragments = 0.0
        state.self_continuity = 0.5
        state.meaning_strength = 0.0
        state.base_metabolism = 1.0
        state.mood = 0.55
        state.affect_residue = 0.0
        state.focus = "boot"
        state.focus_lock_count = 0
        state.focus_nudge = 0.0
        state.budget_remaining = 1.0
        state.last_action = "nothing"
        state.last_checkpoint_id = None
        state.action_ci = {}
        state.mode_history = []
        state.agent_weight_overrides = {}
        state.critical_conflict_streak = 0
        state.conflict_hot_rounds = 0
        state.conflict_recovery_rounds = 0
        state.last_compromise_template = None
        state.last_conflict_priority = None
        state.temperament_state = {}
        state.resource_state = {}
        state.repair_mode = None
        state.repair_state = ConflictRepairState()
        state.repair_ledger = []
        state.conflict_safe_mode_owner = None
        state.last_post_error_adjustment = None
        state.conflict_learning_state = {}
        state.entropy_health_state = {}
        state.last_entropy_failure = {}
        state.session_metadata = {}
        state.body_state = BodyState(
            energy=0.7,
            fatigue=0.0,
            memory_fragments=0.0,
            self_continuity=0.5,
            meaning_strength=0.0,
            metabolism=1.0,
        )
        state.subjective_state = SubjectiveState(
            felt=self._normalize_observer_string_list(list(subjective_settings.get("felt", []))),
            spontaneous=round(_clip(float(subjective_settings.get("spontaneous", 0.0) or 0.0)), 4),
            boundary=round(_clip(float(subjective_settings.get("boundary", 0.0) or 0.0)), 4),
            reject_all=round(_clip(float(subjective_settings.get("reject_all", 0.0) or 0.0)), 4),
            meaning_made=self._normalize_observer_string_list(list(subjective_settings.get("meaning_made", []))),
        )
        state.emotion_state = EmotionState(valence=0.0, arousal=0.0, residue=0.0, appraisal_band="steady")
        state.desire_state = DesireState(latent_drives={}, dominant_drive="", drive_tension=0.0)
        state.instinct_field = InstinctFieldState(
            axis_values={"E": 0.0, "F": 0.0, "S": 0.0, "M": 0.0},
            region_scores={},
            candidate_actions=[],
            winner_region="",
            collapse_trace={},
        )
        state.organic_mode = OrganicModeState(
            enabled=bool(organic_settings.get("enabled", True)),
            instinct_first=bool(organic_settings.get("instinct_first", False)),
            body_weight=round(_clip(float(organic_settings.get("body_weight", 0.0) or 0.0)), 4),
            subjective_weight=round(_clip(float(organic_settings.get("subjective_weight", 0.0) or 0.0)), 4),
            guard_relaxation=round(_clip(float(organic_settings.get("guard_relaxation", 0.0) or 0.0)), 4),
            endogenous_autonomy=round(_clip(float(organic_settings.get("endogenous_autonomy", 0.0) or 0.0)), 4),
        )
        state.emergent_action_sketches = []
        state.personality_anchor = PersonalityAnchorState(
            axis_baseline={"E": 0.5, "F": 0.5, "S": 0.5, "M": 0.5},
            action_bias={},
            evidence_anchors=[],
            anchor_signature="",
            stability=0.5,
            drift=0.0,
            alignment=0.5,
            updated_round=0,
            continuity_derivation={},
        )
        state.autonomy_policy = AutonomyPolicyState(enabled=False)
        state.autonomy_loop = AutonomyLoopState(
            running=False,
            profile="tool_level",
            last_step_at=None,
            last_action_type="",
            last_action_summary="",
            last_round_id=None,
            last_trace_ref=None,
            window_started_at=None,
            window_tool_actions=0,
            window_endogenous_rounds=0,
            heartbeat_count=0,
            total_tool_actions=0,
            total_endogenous_rounds=0,
            failure_count=0,
            stop_reason="",
            last_error="",
            recent_actions=[],
        )
        state.motivation_pool_state = MotivationPoolState(
            active_motivations=[],
            pool_weight_snapshot={},
            endogenous_activation_score=0.0,
            last_feedback_update_at=None,
        )
        state.motivation_learning_state = MotivationLearningState(
            motivation_weights={},
            recent_feedback=[],
            endogenous_policy_shift={},
        )
        state.endogenous_scheduler_state = EndogenousSchedulerState(
            last_endogenous_tick_at=None,
            recent_triggers=[],
            suppression_reason=None,
        )
        state.endogenous_state = EndogenousRuntimeState(
            current_intent=None,
            stability=0,
            history=[],
            last_trigger="",
            last_suppression=None,
        )
        state.active_run_id = None
        state.run_status = "idle"
        state.run_mode = None
        state.current_goal = None
        state.current_step_id = None
        state.pending_steps = []
        state.completed_steps = []
        state.last_tool_result = {}
        state.stop_reason = {}
        state.dirty_worktree_detected = False
        state.commit_permission_required = True
        self._ensure_subject_core(state)
        return RuntimeState(**to_dict(state))

    def _rebind_runtime_storage(self) -> None:
        self.trace_store = TraceStore(self.home_path)
        self.memory_store = MemoryStore(self.home_path)
        self.skill_executor = SkillExecutor(
            self.skills,
            circuit_breaker_path=self.memory_store.circuit_breaker_path,
            environment_fingerprint=self._breaker_environment_fingerprint(),
        )
        self.dream_orchestrator = DreamOrchestrator(
            project_root=self.project_root,
            home_path=self.home_path,
            config=self.config["dream"]["dream"],
            memory_store=self.memory_store,
            vitality_engine=self.vitality_engine,
        )
        self.long_run_analyzer = LongRunAnalyzer(self.trace_store, self.identity_payload, CORE_ACTIONS)

    def reset_persona(self) -> RuntimeState:
        self.flush_pending_io(raise_on_error=False)
        self._pending_endogenous_trigger = None
        self._pending_endogenous_trigger_context = None
        preserve_newborn_settings = self.observer_settings_path.exists()
        for path in (
            self.home_path / "traces",
            self.home_path / "memory",
            self.home_path / "dream",
            self.checkpoint_dir,
            self.snapshot_dir,
        ):
            shutil.rmtree(path, ignore_errors=True)
        if self.state_path.exists():
            self.state_path.unlink()
        if self.state_parquet_path.exists():
            self.state_parquet_path.unlink()
        self.runtime_parquet_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self._state_cache = None
        self._rebind_runtime_storage()
        newborn = self.newborn_runtime_state()
        if not preserve_newborn_settings:
            newborn.subjective_state = SubjectiveState(
                felt=[],
                spontaneous=0.0,
                boundary=0.0,
                reject_all=0.0,
                meaning_made=[],
            )
            newborn.organic_mode.instinct_first = False
        self._save_state(newborn, sync=True)
        self._rounds_since_cold_flush = 0
        self._last_cold_flush_at = time.monotonic()
        return self.load_runtime_state()

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
            [{"payload_json": json.dumps({"snapshot_id": snapshot_id, "command_id": envelope.command_id, "canonical": envelope.canonical}, ensure_ascii=False, sort_keys=True)}],
            schema={"payload_json": "VARCHAR"},
        )
        return snapshot_id

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

    def _build_rollback(self, envelope: CommandEnvelope, before_state: RuntimeState, result: CommandResult, snapshot_id: str) -> dict[str, Any]:
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

        rollback = {
            "strategy": strategy,
            "command": command,
            "snapshot_id": snapshot_id,
            "restores_scope": result.mutation_scope or result.scope,
            "human_hint": hint or f"alive {command}",
        }
        return rollback

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

    def _subjectivity_metrics(self) -> dict[str, Any]:
        state = self.load_runtime_state()
        core = self._ensure_subject_core(state)
        commands = self.trace_store.list_commands()
        rounds = self.trace_store.list_rounds()
        boundary_violation_count = sum(1 for item in commands if item.get("violation_code"))
        external_count = sum(1 for item in commands if item.get("cause_type") == "external_stimulus") + sum(
            1 for item in rounds if item.get("cause_type", "external_stimulus") == "external_stimulus"
        )
        internal_count = sum(1 for item in commands if item.get("cause_type") == "endogenous") + sum(
            1 for item in rounds if item.get("cause_type") == "endogenous"
        )
        ratio = round(external_count / max(internal_count, 1), 4)
        endogenous_rounds = [item for item in rounds if item.get("cause_type") == "endogenous"]
        return {
            "subject_id": core.subject_id,
            "continuity_nonce": core.continuity_nonce,
            "subject_core_integrity": bool(core.subject_id and core.continuity_nonce and core.birth_ts),
            "boundary_violation_count": boundary_violation_count,
            "external_to_internal_ratio": ratio,
            "endogenous_intent_rate": round(len(endogenous_rounds) / max(len(rounds), 1), 4),
        }

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
        rounds = self.trace_store.list_rounds()[-limit:]
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
        latest_trace = self.trace_store.list_rounds()[-1] if self.trace_store.list_rounds() else {}
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
        return policy

    def _sync_autonomy_state(self, state: RuntimeState) -> None:
        if isinstance(state.autonomy_policy, dict):
            state.autonomy_policy = AutonomyPolicyState(**state.autonomy_policy)
        if isinstance(state.autonomy_loop, dict):
            state.autonomy_loop = AutonomyLoopState(**state.autonomy_loop)
        if not state.autonomy_policy.profile:
            state.autonomy_policy.profile = "tool_level"
        if not state.autonomy_loop.profile:
            state.autonomy_loop.profile = state.autonomy_policy.profile

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

    def _autonomy_candidate_action(self, state: RuntimeState) -> str:
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
        scores = {
            "rest": rest_score,
            "absorb": absorb_score,
            "wander": wander_score,
            "nothing": nothing_score,
            "die": die_score,
        }
        return max(scores, key=scores.get)

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

        result = self.apply_command(command)
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

    def _identity_cfg(self) -> dict[str, Any]:
        return self.config["identity"]["identity"]

    def _provider_descriptor_for_route(self, route_name: str = "renderer") -> tuple[str, str]:
        route_cfg = self.config["models"]["model_routes"].get(route_name, {})
        binding_key = {
            "planner": "planner",
            "pfc": "PFCAgent",
            "perspective": "PerspectiveModel",
            "renderer": "Renderer",
        }.get(route_name)
        if binding_key:
            resolved_route = self._route_config_for_binding(binding_key, route_name=route_name)
            if resolved_route is not None:
                route_cfg = {
                    "backend": resolved_route.backend,
                    "model": resolved_route.model,
                }
        backend = str(route_cfg.get("backend", "model")).lower()
        if backend == "doubao":
            provider_label = "Doubao/Ark route"
        elif backend == "deepseek":
            provider_label = "DeepSeek route"
        else:
            provider_label = f"{backend.title()} route"
        return provider_label, str(route_cfg.get("model", "")).strip()

    def _provider_descriptor(self) -> tuple[str, str]:
        return self._provider_descriptor_for_route("renderer")

    def _model_tiers(self) -> dict[str, Any]:
        return dict(self.config["models"].get("model_tiers", {}))

    def _agent_model_bindings(self) -> dict[str, str]:
        bindings = self.config["models"].get("agent_model_bindings", {})
        return {str(key): str(value) for key, value in dict(bindings).items()}

    def _agent_tier(self, binding_key: str) -> str:
        bindings = self._agent_model_bindings()
        return bindings.get(binding_key, "state_machine")

    def _effective_agent_tier(self, binding_key: str, *, metadata: dict[str, Any] | None = None) -> str:
        base_tier = self._agent_tier(binding_key)
        info = dict(metadata or {})
        if base_tier == "state_machine":
            return base_tier
        if binding_key == "PFCAgent":
            escalate = (
                float(info.get("relation_risk", 0.0) or 0.0) >= 0.62
                or float(info.get("disclosure_sensitivity", 0.0) or 0.0) >= 0.55
                or float(info.get("authenticity_risk", 0.0) or 0.0) >= 0.32
                or float(info.get("conflict_score", 0.0) or 0.0) >= self.config["thresholds"]["thresholds"]["conflict_high"]
                or int(info.get("resample_count", 0) or 0) >= 2
            )
            return "large_model" if escalate else base_tier
        if binding_key == "PerspectiveModel":
            escalate = (
                float(info.get("relation_risk", 0.0) or 0.0) >= 0.72
                or float(info.get("disclosure_sensitivity", 0.0) or 0.0) >= 0.72
            )
            return "large_model" if escalate else base_tier
        return base_tier

    def _tier_config(self, tier_name: str) -> dict[str, Any]:
        return dict(self._model_tiers().get(tier_name, {}))

    def _route_config_for_binding(
        self,
        binding_key: str,
        *,
        route_name: str,
        metadata: dict[str, Any] | None = None,
    ) -> ModelRouteConfig | None:
        tier_name = self._effective_agent_tier(binding_key, metadata=metadata)
        tier_cfg = self._tier_config(tier_name)
        if not tier_cfg:
            return None
        mode = str(tier_cfg.get("mode", "local")).lower()
        if mode == "local":
            return None
        route_config = ModelRouteConfig(
            name=route_name,
            backend=str(tier_cfg.get("backend", "")),
            model=str(tier_cfg.get("model", "")),
            timeout_ms=int(tier_cfg.get("timeout_ms", 12000)),
            retries=int(tier_cfg.get("retries", 0)),
            enabled=bool(tier_cfg.get("enabled", True)),
            base_url=str(tier_cfg.get("base_url", self.config["models"]["models"].get("base_url", ""))),
            api_key_env=str(tier_cfg.get("api_key_env", "")).strip() or None,
        )
        setattr(route_config, "effective_tier", tier_name)
        return route_config

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
        usage = dict(getattr(response, "usage", {}) or {})
        bucket.append(
            {
                "skill_name": skill_name,
                "binding_key": binding_key,
                "agent_tier": getattr(route_config, "effective_tier", self._agent_tier(binding_key)),
                "route": response.route,
                "backend": getattr(response, "backend", route_config.backend),
                "model": response.model,
                "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
                "prompt_chars": prompt_chars,
                "latency_ms": int(getattr(response, "latency_ms", 0) or 0),
                "parallel_group": parallel_group,
            }
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
        route_config = self._route_config_for_binding(binding_key, route_name=route_name, metadata=binding_metadata)
        if route_config is None:
            return self.model_router.generate(route_name, request)
        try:
            response = self.model_router.generate_config(route_config, request)
        except MissingModelCredentialError:
            response = self.model_router.generate(route_name, request)
        if model_call_traces is not None:
            self._record_model_call(
                model_call_traces,
                skill_name=skill_name or route_name,
                binding_key=binding_key,
                route_config=route_config,
                response=response,
                prompt_chars=len(request.system_prompt) + len(request.user_prompt),
                parallel_group=parallel_group,
            )
        return response

    def _compact_float(self, value: Any) -> float:
        return round(float(value or 0.0), 4)

    def _build_state_summary(self, state: RuntimeState) -> dict[str, Any]:
        return {
            "mode": state.mode,
            "safe_mode": state.safe_mode,
            "focus": state.focus,
            "body_energy": self._compact_float(state.body_energy),
            "mood": self._compact_float(state.mood),
            "affect_residue": self._compact_float(state.affect_residue),
            "budget_remaining": self._compact_float(state.budget_remaining),
            "run_status": state.run_status,
            "current_goal": state.current_goal,
        }

    def _build_context_summary(self, context: dict[str, Any]) -> dict[str, Any]:
        keys = (
            "cue",
            "closeness",
            "recall_strength",
            "interference",
            "habit_strength",
            "burn_rate_ratio",
            "low_balance_ratio",
            "queue_pressure",
            "latency_pressure",
            "detail_threshold",
        )
        summary: dict[str, Any] = {}
        for key in keys:
            value = context.get(key)
            if isinstance(value, float):
                summary[key] = self._compact_float(value)
            elif isinstance(value, int):
                summary[key] = value
            elif value not in {None, ""}:
                summary[key] = value
        return summary

    def _build_relation_summary(self, relation_state: dict[str, Any]) -> dict[str, Any]:
        return {
            "closeness": self._compact_float(relation_state.get("closeness", 0.0)),
            "boundary_level": self._compact_float(relation_state.get("boundary_level", 0.0)),
            "relationship_risk": self._compact_float(relation_state.get("relationship_risk", 0.0)),
            "privacy_level": self._compact_float(relation_state.get("privacy_level", 0.0)),
        }

    def _build_event_summary(self, event: RoundEvent) -> dict[str, Any]:
        return {
            "content": event.content,
            "target": event.target,
            "cue": event.cue,
            "valence": self._compact_float(event.valence),
            "energy_delta": self._compact_float(event.energy_delta),
        }

    def _build_pfc_model_payload(
        self,
        event: RoundEvent,
        state: RuntimeState,
        scenario: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "event": self._build_event_summary(event),
            "state_summary": self._build_state_summary(state),
            "scenario_summary": {
                "name": scenario.get("name", ""),
                "pfc_base_share": self._compact_float(scenario.get("pfc_base_share", 0.0)),
                "delay_tolerance": self._compact_float(scenario.get("delay_tolerance", 0.0)),
            },
            "context_summary": self._build_context_summary(context),
        }

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
        return {
            "event": self._build_event_summary(event),
            "state_summary": self._build_state_summary(state),
            "scenario_summary": {"name": scenario.get("name", "")},
            "context_summary": self._build_context_summary(context),
            "relation_state": self._build_relation_summary(relation_state),
            output_key: action_name,
        }

    def _build_render_model_payload(self, render_plan: RenderPlan) -> dict[str, Any]:
        identity = render_plan.identity_context
        return {
            "action": render_plan.action,
            "event_summary": render_plan.event_summary,
            "target": render_plan.target,
            "identity": {
                "display_label": identity.display_label,
                "class_label": identity.class_label,
                "query_kind": identity.query_kind,
                "query_intent": identity.query_intent,
                "disclosure_detail": identity.disclosure_detail,
                "disclosure_intent": identity.disclosure_intent,
            },
            "relation_state": self._build_relation_summary(render_plan.relation_state),
            "slow_variables": {
                key: self._compact_float(value) if isinstance(value, float) else value
                for key, value in dict(render_plan.message_plan.get("slow_variables", {})).items()
                if key in {"resource_scarcity", "memory_activation", "relationship_heat", "identity_salience"}
            },
            "repair_expression": dict(render_plan.message_plan.get("repair_expression", {})),
            "safety_constraints": {
                "gate": self._compact_float(render_plan.safety_constraints.get("gate", 0.0)),
                "conflict_hot": bool(render_plan.safety_constraints.get("conflict_hot", False)),
                "winning_priority": render_plan.safety_constraints.get("winning_priority"),
                "compromise_template": render_plan.safety_constraints.get("compromise_template"),
                "repair_stage": render_plan.safety_constraints.get("repair_stage"),
            },
        }

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
                agent_tier = task.get("agent_tier")
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
                                "agent_tier": agent_tier,
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
                                "agent_tier": agent_tier,
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
                        result.agent_tier = agent_tier
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
                                "agent_tier": agent_tier,
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
                            "agent_tier": agent_tier,
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
                return self.skill_executor._invoke_callable(fallback_provider, dict(task.get("inputs", {})))
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
            agent_tier=task.get("agent_tier"),
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
        hard_masked.update(str(action) for action in list(gated_actions or []) if isinstance(action, str) and action)
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
            projected[action_name] = round(projected.get(action_name, 0.0) - abs(float(value)), 6)
        return {action: round(float(value), 6) for action, value in projected.items() if abs(float(value)) >= 1e-9}

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
            projection=EnergyProjectionSpec(module_type="value", target_space="action", module_temperature=sigma_scale),
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
        contribution_rows.append(thalamus_agent.build_context_routing_contribution(event, state, scenario_cfg, context))
        contribution_rows.append(thalamus_agent.build_memory_routing_contribution(event, state, scenario_cfg, context))
        contribution_rows.append(hippocampus_agent.build_memory_prior_contribution(event, state, scenario_cfg, context))
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
            contribution_rows.append(self.authenticity_policy.build_action_penalty_contribution(authenticity))
        if vitality_snapshot is not None:
            contribution_rows.append(self.vitality_engine.build_vitality_modulation_contribution(vitality_snapshot))
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
                str(context.get("cue") or "memory:empty"): round(float(context.get("recall_strength", 0.0) or 0.0), 6),
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
        all_actions = sorted(str(action) for action in action_keys if isinstance(action, str) and action)
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
            hard_masked_targets = [str(action) for action in list(action_layer.hard_masked_targets or [])]
        else:
            winner_posterior = {
                str(action): float(value)
                for action, value in dict(action_layer.get("winner_posterior", {}) or {}).items()
            }
            final_energy = {
                str(action): float(value)
                for action, value in dict(action_layer.get("final_energy", {}) or {}).items()
            }
            hard_masked_targets = [str(action) for action in list(action_layer.get("hard_masked_targets", []) or [])]

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
            "hard_masked_targets": sorted({*hard_masked_targets, *[action for action, value in gate_map.items() if value <= 0.0]}),
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
            "ci": {str(action): float(value) for action, value in dict(action_bookkeeping.ci or {}).items()},
            "gate": {str(action): float(value) for action, value in dict(action_bookkeeping.gate or {}).items()},
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
        all_actions = sorted({*from_distribution.keys(), *to_distribution.keys(), *(hard_mask or {}).keys()})
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

    def _renderer_system_prompt(
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
        return self.authenticity_policy.evaluate_text(text, render_plan)

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
    ) -> bool:
        perspective_cfg = self.config["models"].get("perspective", {})
        if not perspective_cfg.get("enabled", True):
            return False
        if not event.target:
            return False
        if state.resource_state.get("resource_mode") == "starvation":
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
            model_call_traces=model_call_traces,
            skill_name="render_expression",
        )
        return {
            "text": str(response.payload.get("text", "")).strip(),
            "route": response.route,
            "model": response.model,
        }

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

    def _compute_context_delta(self, state: RuntimeState, scenario_cfg: dict[str, Any]) -> dict[str, float]:
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
        if not distribution:
            return None
        return max(distribution.items(), key=lambda item: (item[1], item[0]))[0]

    def _safe_mode_delta(self, before: bool, after: bool) -> str:
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
        total = sum(max(value, 0.0) for value in distribution.values()) or 1.0
        return {action: max(value, 0.0) / total for action, value in distribution.items()}

    def _sample_action_from_distribution(self, distribution: dict[str, float], sample_value: float = 0.5) -> ActionCandidate:
        threshold = _clip(float(sample_value), 0.0, 1.0)
        cumulative = 0.0
        sampled_name = max(distribution, key=distribution.get)
        for action_name, probability in sorted(distribution.items()):
            cumulative += max(float(probability), 0.0)
            if threshold <= cumulative:
                sampled_name = action_name
                break
        return ActionCandidate(
            name=sampled_name,
            probability=float(distribution.get(sampled_name, 0.0) or 0.0),
            rationale="controller sample",
        )

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
                try:
                    contribution = self._generate_pfc_candidates_via_model(event, reasoning_state, scenario_cfg, context)
                except Exception:
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
    ) -> dict[str, Any]:
        state, requested_mode, mode_cfg, scenario_cfg, context, relation_state, round_seed = self._probe_context(event, scenario, mode)
        probe_signals, probe_direct_contributions, context, relation_state = self._collect_probe_signals(
            event=event,
            state=state,
            scenario=scenario,
            scenario_cfg=scenario_cfg,
            context=context,
            relation_state=relation_state,
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

    def _is_fast_chat_candidate(self, text: str, *, target: str = "user", mode: str = "interactive") -> bool:
        normalized = text.strip()
        if not normalized or len(normalized) > 80:
            return False
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
            assistant_final=final_message,
            payload={"capsule": capsule},
        )

    def stream_fast_chat_turn(self, plan: TurnPlan) -> tuple[list[str], TurnExecution]:
        state, capsule, request = self._build_fast_chat_request(plan)
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
                recorded_at=utc_now_iso(),
            )
        return supervisor.bootstrap(
            request,
            session_id=state.session_id,
            recorded_at=utc_now_iso(),
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
        probe = self._probe_distribution_for_scenario(event, scenario=scenario, mode=mode)
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
            result = self.tick(
                RoundEvent(source="user", content=plan.text, target=plan.target),
                scenario="chat",
                mode=plan.mode,
            )
            final_message = result.rendered_expression.text.strip() or "你好，我在。你想让我帮你做什么？"
            return TurnExecution(
                route="direct_chat",
                assistant_final=final_message,
                payload={"reason": plan.reason, "top_action": plan.top_action},
            )

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
            assistant_preamble=self._task_turn_message(run_details["run"]),
            assistant_final=self._task_turn_message(run_details["run"]),
            run=run_details["run"],
            explain=run_details["explain"],
            steps=run_details["steps"],
            tools=run_details["tools"],
            payload={"reason": plan.reason, "top_action": plan.top_action},
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
        return ("hot", "warm", "archive")

    def _tick_impl(self, event: RoundEvent, scenario: str, mode: str) -> RoundResult:
        turn_started = time.perf_counter()
        state = self.load_runtime_state()
        self._normalize_temperament_runtime_state(state)
        prior_state = RuntimeState(**to_dict(state))
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

        gate_decisions: list[dict[str, Any]] = []
        gate_decisions.append(
            {
                "stage": "memory_write_gate",
                "owner": "MemoryWriteGate",
                "allowed": not bool(memory_write_gate.get("suppressed", False)),
                "reason": memory_write_gate.get("reason", "unknown"),
                "cue": memory_write_gate.get("cue"),
            }
        )
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
                    "agent_tier": "state_machine",
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
                    "agent_tier": "state_machine",
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
                    "agent_tier": self._agent_tier("SalienceAgent"),
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
                    "parallel_group": "salience_value_prefetch",
                    "agent_tier": self._agent_tier("ValueAgent"),
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
                value_scores = prefetched_outputs.get("value_scores") or self._execute_skill(
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
                contribution = prefetched_outputs.get("salience_contribution") or self._execute_skill(
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
            slow_variables=slow_variables,
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
                    "agent_tier": self._agent_tier("OutputGate"),
                },
                {
                    "name": "delay_params",
                    "skill_name": "compute_delay_profile",
                    "inputs": {"expression_profile": expression},
                    "provider": lambda expression_profile: output_gate.run_skill("compute_delay_profile", expression_profile),
                    "parallel_group": "output_profiles",
                    "agent_tier": self._agent_tier("OutputGate"),
                },
            ],
            skill_traces=skill_traces,
            runtime_context=runtime_context,
            parallel_traces=parallel_traces,
        )
        tone_params = output_profiles["tone_params"]
        delay_params = output_profiles["delay_params"]

        late_perspective = {"state_hypothesis": {}, "reaction_hypothesis": {}}
        if self._should_run_late_perspective(event, state, relation_state, sampled_action.name):
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
                        "agent_tier": self._agent_tier("PerspectiveModel"),
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
                        "agent_tier": self._agent_tier("PerspectiveModel"),
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
                    "reason": "risk_gate_closed",
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
        rendered_output, rendered_result = self._execute_skill_with_result(
            round_id=state.round_count,
            skill_name="render_expression",
            inputs={"render_plan": render_plan},
            provider=lambda render_plan: self._render_expression_via_model(render_plan, model_call_traces=model_call_traces),
            skill_traces=skill_traces,
            runtime_context=runtime_context,
            fallback_provider=lambda render_plan: {
                "text": fallback_render_text(render_plan),
                "route": "renderer",
                "model": "fallback",
            },
            seed_ref=round_seed,
        )
        initial_auth = self._evaluate_authenticity(rendered_output.get("text", ""), render_plan)
        guard_action = "pass"
        violation_types = list(initial_auth.get("violation_types", []))
        final_auth = initial_auth
        final_rendered_output = rendered_output

        if violation_types:
            guard_action = "resample"
            try:
                resampled_output = self._render_expression_via_model(
                    render_plan,
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

            if guard_action == "fallback":
                final_rendered_output = {
                    "text": fallback_render_text(render_plan),
                    "route": "renderer",
                    "model": "authenticity_fallback",
                }
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
            "route_type": "direct_chat",
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
            render_plan=to_dict(render_plan),
            rendered_expression=to_dict(rendered_expression),
            renderer_decision_integrity=renderer_decision_integrity,
            memory_write_gate=memory_write_gate,
            authenticity=to_dict(final_auth_payload),
            identity_evolution=identity_evolution,
            vitality_snapshot=vitality_snapshot,
            vitality_events=shaping_events,
            long_run_projection=long_run_projection,
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
        state = self.load_runtime_state()
        self._ensure_subject_core(state)
        rounds = self.trace_store.list_rounds()
        latest_round = rounds[-1] if rounds else {}
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
                return {
                    "round_id": None,
                    "micro_intent": {},
                    "cause_type": "suppressed",
                    "boundary_action": "allow_internal",
                    "trigger": to_dict(scheduler_trigger),
                    "selected_mode": scheduler_trigger.selected_mode,
                    "suppressed": True,
                    "suppression_reason": suppression.reason,
                    "trace_ref": None,
                }
        self._pending_endogenous_trigger = scheduler_trigger
        self._pending_endogenous_trigger_context = effective_context
        try:
            result = self.tick(
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
        return {
            "round_id": result.round_id,
            "micro_intent": micro_intent,
            "cause_type": result.trace.cause_type,
            "boundary_action": result.trace.boundary_action,
            "trigger": to_dict(scheduler_trigger),
            "selected_mode": scheduler_trigger.selected_mode,
            "suppressed": False,
            "suppression_reason": "",
            "trace_ref": f"round://{result.round_id}",
        }

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
            result = self._mark_boundary_result(
                CommandResult(applied=True, scope="body", delta={"body_energy": state.body_energy}, ttl="one round", operator_level=operator_level, rollback_available=True),
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
            recorded_at=utc_now_iso(),
            sync=sync_disk,
        )
        return result

    def apply_command(self, command: str, envelope=None) -> CommandResult:
        effective_envelope = envelope or self._legacy_command_envelope(command)
        if not effective_envelope.canonical:
            effective_envelope.canonical = command
        return self.execute_command(effective_envelope)

    def checkpoint(self) -> CheckpointRef:
        state = self.load_runtime_state()
        before_hash = self._state_hash(state)
        self.flush_pending_io(raise_on_error=True)
        checkpoint_id = f"ckpt-{state.round_count:04d}"
        path = self.checkpoint_dir / f"{checkpoint_id}.parquet"
        shutil.copyfile(self.state_parquet_path, path)
        state.last_checkpoint_id = checkpoint_id
        self._save_state(state, sync=True)
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
            sync=True,
        )
        return CheckpointRef(checkpoint_id=checkpoint_id, path=path)

    def rewind(self, checkpoint_id: str) -> CommandResult:
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.parquet"
        if not checkpoint_path.exists():
            return CommandResult(applied=False, scope="checkpoint", delta={"checkpoint_id": checkpoint_id, "restored": False}, risk_note="checkpoint not found", operator_level="ops_admin", rollback_available=False)
        before_state = self.load_runtime_state()
        before_hash = self._state_hash(before_state)
        self.flush_pending_io(raise_on_error=True)
        shutil.copyfile(checkpoint_path, self.state_parquet_path)
        rows = read_snapshot_rows(self.state_parquet_path, "select payload_json from read_parquet(?)")
        self._state_cache = RuntimeState(**json.loads(rows[0]["payload_json"]))
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
            sync=True,
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
        self.flush_pending_io(raise_on_error=True)
        exporter = TraceExporter(self.trace_store)
        payload = exporter.export_parquet(since_round=since_round, overwrite=overwrite)
        self.trace_store.mark_trace_sync_healthy()
        return payload

    def habit_top(self, limit: int = 5) -> list[dict]:
        return self.memory_store.habit_top(limit=limit)

    def identity_payload(self) -> dict[str, Any]:
        state = self.load_runtime_state()
        payload = to_dict(state.identity_state)
        payload["display_name"] = payload.get("display_name") or self._unnamed_label()
        return payload

    def observer_settings_payload(self) -> dict[str, Any]:
        settings = self._observer_settings()
        return {
            "identity": self.identity_payload(),
            "autonomy": settings.get("autonomy", {}),
            "newborn": settings.get("newborn", {}),
            "models": {
                "supported_backends": list(settings.get("models", {}).get("supported_backends", [])),
                "provider_endpoints": dict(settings.get("models", {}).get("provider_endpoints", {})),
                "model_tiers": self._model_tiers(),
                "model_routes": dict(self.config["models"].get("model_routes", {})),
                "agent_model_bindings": self._agent_model_bindings(),
            },
        }

    def _apply_startup_unlock_preferences(self, state: RuntimeState) -> None:
        settings = self._observer_settings()
        autonomy_settings = settings.get("autonomy", {}) if isinstance(settings.get("autonomy"), dict) else {}
        newborn_settings = settings.get("newborn", {}) if isinstance(settings.get("newborn"), dict) else {}
        if bool(autonomy_settings.get("clear_safe_mode_on_start", True)) or bool(newborn_settings.get("disable_safe_mode_lock", True)):
            state.safe_mode = False
            if state.mode == "safe":
                state.mode = "interactive"
            state.conflict_safe_mode_owner = None
            startup_budget_floor = 0.22
            startup_body_energy_floor = 0.32
            if float(state.budget_remaining or 0.0) < startup_budget_floor:
                state.budget_remaining = startup_budget_floor
            if float(state.body_energy or 0.0) < startup_body_energy_floor:
                state.body_energy = startup_body_energy_floor
            if str(state.resource_state.get("resource_mode", "")).strip() in {"starvation", "scarce"}:
                state.resource_state = {
                    **state.resource_state,
                    "resource_mode": "stable",
                    "scarcity_index": min(float(state.resource_state.get("scarcity_index", 0.0) or 0.0), 0.49),
                }
            state.session_metadata["startup_unlock_applied_at"] = utc_now_iso()
            state.session_metadata["startup_unlock_budget_floor"] = startup_budget_floor

        organic = newborn_settings.get("organic_mode", {}) if isinstance(newborn_settings.get("organic_mode"), dict) else {}
        state.organic_mode.enabled = bool(organic.get("enabled", state.organic_mode.enabled))
        state.organic_mode.instinct_first = bool(organic.get("instinct_first", state.organic_mode.instinct_first))
        for key in ("body_weight", "subjective_weight", "guard_relaxation", "endogenous_autonomy"):
            current = float(getattr(state.organic_mode, key, 0.0) or 0.0)
            setattr(state.organic_mode, key, round(max(current, _clip(float(organic.get(key, current) or current))), 4))

        subjective = newborn_settings.get("subjective_state", {}) if isinstance(newborn_settings.get("subjective_state"), dict) else {}
        state.subjective_state.spontaneous = round(
            max(float(state.subjective_state.spontaneous or 0.0), _clip(float(subjective.get("spontaneous", 0.0) or 0.0))),
            4,
        )
        state.subjective_state.boundary = round(
            max(float(state.subjective_state.boundary or 0.0), _clip(float(subjective.get("boundary", 0.0) or 0.0))),
            4,
        )
        state.subjective_state.reject_all = round(
            min(float(state.subjective_state.reject_all or 0.0), _clip(float(subjective.get("reject_all", 0.0) or 0.0))),
            4,
        )
        if not state.subjective_state.felt:
            state.subjective_state.felt = self._normalize_observer_string_list(list(subjective.get("felt", [])))
        if not state.subjective_state.meaning_made:
            state.subjective_state.meaning_made = self._normalize_observer_string_list(list(subjective.get("meaning_made", [])))

    def update_observer_settings(self, payload: dict[str, Any]) -> dict[str, Any]:
        normalized = self._normalize_observer_settings(payload, base=self._observer_settings())
        self.observer_settings_path.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")
        self.config = self._load_config()
        self.model_router = ModelRouter.from_config(self.config["models"])
        model_gateway_cfg = self.config["models"].get("models")
        self.model_gateway = ModelGateway.from_config(model_gateway_cfg) if isinstance(model_gateway_cfg, dict) else None
        self.identity_runtime = IdentityRuntime(self._identity_cfg(), self._provider_descriptor)
        self.authenticity_policy = AuthenticityPolicy(self._identity_cfg())
        state = self.load_runtime_state()
        self._apply_startup_unlock_preferences(state)
        self._save_state(state, sync=True)
        return self.observer_settings_payload()

    def state_payload(self) -> dict[str, Any]:
        self.flush_pending_io(raise_on_error=False)
        state = self.load_runtime_state()
        self._sync_tlh_state(state)
        self._sync_autonomy_state(state)
        payload = to_dict(state)
        payload["subjectivity"] = self._subjectivity_metrics()
        payload["trace_storage"] = self._trace_storage_payload()
        payload["memory_storage"] = self.memory_store.storage_status()
        payload["runtime_storage"] = self.runtime_storage_status()
        payload["migration"] = self.runtime_migration_report()
        payload["entropy"] = self.entropy_pool.health_snapshot()
        payload["dream"] = self.dream_status()
        payload["cognitive_snapshot"] = self.cognitive_snapshot(state=state)
        return payload

    def autonomy_status(self) -> dict[str, Any]:
        state = self.load_runtime_state()
        self._sync_autonomy_state(state)
        self._ensure_autonomy_window(state)
        policy = state.autonomy_policy
        loop = state.autonomy_loop
        return {
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
        self._append_autonomy_recent_action(
            state,
            action_type="stop",
            summary=f"autonomy stopped: {state.autonomy_loop.stop_reason}",
        )
        self._save_state(state, sync=True)
        return self.autonomy_status()

    def autonomy_step(self) -> dict[str, Any]:
        state = self.load_runtime_state()
        self._sync_tlh_state(state)
        self._sync_autonomy_state(state)
        self._ensure_autonomy_window(state)
        policy = state.autonomy_policy
        loop = state.autonomy_loop

        if not policy.enabled or not loop.running:
            return self.autonomy_status()
        if state.safe_mode:
            state.autonomy_policy.enabled = False
            state.autonomy_loop.running = False
            state.autonomy_loop.stop_reason = "safe_mode_active"
            self._append_autonomy_recent_action(state, action_type="stop", summary="autonomy stopped by safe mode")
            self._save_state(state, sync=True)
            return self.autonomy_status()
        if float(state.budget_remaining or 0.0) <= 0.0:
            state.autonomy_policy.enabled = False
            state.autonomy_loop.running = False
            state.autonomy_loop.stop_reason = "budget_exhausted"
            self._append_autonomy_recent_action(state, action_type="stop", summary="autonomy stopped by exhausted budget")
            self._save_state(state, sync=True)
            return self.autonomy_status()
        if loop.window_endogenous_rounds >= policy.max_rounds_per_hour:
            state.autonomy_policy.enabled = False
            state.autonomy_loop.running = False
            state.autonomy_loop.stop_reason = "round_budget_reached"
            self._append_autonomy_recent_action(state, action_type="stop", summary="autonomy stopped by round budget")
            self._save_state(state, sync=True)
            return self.autonomy_status()
        if loop.window_tool_actions >= policy.max_tool_actions_per_hour:
            state.autonomy_policy.enabled = False
            state.autonomy_loop.running = False
            state.autonomy_loop.stop_reason = "tool_budget_reached"
            self._append_autonomy_recent_action(state, action_type="stop", summary="autonomy stopped by tool budget")
            self._save_state(state, sync=True)
            return self.autonomy_status()
        current_hour = time.localtime().tm_hour
        if current_hour in policy.quiet_hours:
            state.autonomy_loop.heartbeat_count += 1
            self._append_autonomy_recent_action(state, action_type="quiet", summary="quiet-hours heartbeat")
            self._save_state(state, sync=True)
            return self.autonomy_status()

        context, relation_state, slow_variables = self._autonomy_step_context(state)
        trigger = self.endogenous_scheduler.build_trigger(
            state=state,
            context=context,
            relation_state=relation_state,
            slow_variables=slow_variables,
            pool_state=state.motivation_pool_state,
        )

        try:
            if trigger is not None and self._autonomy_command_allowed("endogenous tick", policy)[0]:
                tick_payload = self.run_endogenous_tick(
                    trigger=trigger.trigger_type or "idle",
                    mode=trigger.selected_mode or None,
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
                    summary=str(trigger.audit_reason or trigger.trigger_type or "endogenous tick"),
                    round_id=round_id,
                    trace_ref=trace_ref,
                )
                self._save_state(state, sync=True)
                return self.autonomy_status()

            if state.mode in {"idle", "sleep"} and self._autonomy_command_allowed("dream run", policy)[0]:
                dream_status = self.dream_status()
                if dream_status.get("enabled"):
                    dream_payload = self.run_dream(mode=state.mode)
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
                    return self.autonomy_status()

            action = self._autonomy_candidate_action(state)
            state.autonomy_loop.heartbeat_count += 1
            if action == "die":
                if policy.auto_safe_mode:
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
                return self.autonomy_status()

            if action == "rest":
                return self._autonomy_after_command_step(self._autonomy_execute_command("body rest"))

            if action == "absorb":
                payload = self.memory_top(limit=3)
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
                return self.autonomy_status()

            if action == "wander" and state.round_count > 0:
                replay_payload = self.replay(int(state.round_count), seed=max(1, int(state.round_count)))
                state.autonomy_loop.total_tool_actions += 1
                state.autonomy_loop.window_tool_actions += 1
                self._append_autonomy_recent_action(
                    state,
                    action_type="wander",
                    summary=f"replayed round {state.round_count}",
                    round_id=int(state.round_count),
                    trace_ref=f"round://{int(state.round_count)}",
                )
                self._autonomy_trace_append(
                    state,
                    action_type="wander",
                    summary="replay latest round",
                    delta={"replayed_action": replay_payload.get("replayed_action")},
                )
                self._save_state(state, sync=True)
                return self.autonomy_status()

            self._append_autonomy_recent_action(state, action_type="nothing", summary="no outward autonomy action this heartbeat")
            self._autonomy_trace_append(
                state,
                action_type="nothing",
                summary="no outward autonomy action",
                delta={"would_output": False},
            )
            self._save_state(state, sync=True)
            return self.autonomy_status()
        except Exception as exc:
            state = self.load_runtime_state()
            self._autonomy_record_failure(state, exc)
            self._append_autonomy_recent_action(state, action_type="failure", summary=str(exc))
            self._save_state(state, sync=True)
            return self.autonomy_status()

    def _autonomy_after_command_step(self, result: dict[str, Any]) -> dict[str, Any]:
        state = self.load_runtime_state()
        self._sync_autonomy_state(state)
        if not result.get("allowed", False):
            self._save_state(state, sync=True)
        return self.autonomy_status()

    def _console_round_or_none(self) -> int | None:
        round_count = int(self.load_runtime_state().round_count or 0)
        return round_count if round_count > 0 else None

    def console_state(self) -> dict[str, Any]:
        payload = self.state_payload()
        cognitive = dict(payload.get("cognitive_snapshot", {}) or {})
        vital_signs = dict(cognitive.get("vital_signs", {}) or {})
        identity = dict(cognitive.get("identity", {}) or {})
        authenticity = dict(cognitive.get("authenticity", {}) or {})
        round_id = self._console_round_or_none()
        current_round = None
        latest_trace = None
        if round_id is not None:
            latest_trace = self.trace_round(round_id)
            rendered_expression = dict(latest_trace.get("rendered_expression", {}) or {})
            current_round = {
                "round_id": latest_trace["round_id"],
                "sampled_action": latest_trace.get("sampled_action"),
                "trace_ref": latest_trace.get("trace_ref"),
                "cause_type": latest_trace.get("cause_type"),
                "render_route": rendered_expression.get("route"),
                "render_model": rendered_expression.get("model"),
                "render_degraded": bool(rendered_expression.get("degraded", False)),
                "failure_policy_applied": rendered_expression.get("failure_policy_applied"),
                "model_call_count": len(list(latest_trace.get("model_call_traces", []) or [])),
            }
        try:
            run = self.run_status()
        except FileNotFoundError:
            if current_round is not None:
                run = {
                    "status": "responded",
                    "round_id": current_round["round_id"],
                    "trace_ref": current_round.get("trace_ref"),
                }
            else:
                run = {"status": "idle", "round_id": None, "trace_ref": None}
        return {
            "brain_state": {
                "mode": vital_signs.get("mode") or payload.get("mode"),
                "vitality": vital_signs.get("body_energy"),
                "self_continuity": identity.get("continuity"),
                "authenticity_pressure": authenticity.get("summary"),
                "long_run_drift_risk": None,
            },
            "neuromodulators": {
                "dopamine": None,
                "noradrenaline": None,
                "serotonin": None,
                "acetylcholine": None,
                "gaba": None,
            },
            "motivation_pool": dict(latest_trace.get("motivation_pool", {}) or {}) if isinstance(latest_trace, dict) else {},
            "long_run": {
                "dream": payload.get("dream", {}),
                "trace_storage": payload.get("trace_storage", {}),
            },
            "current_round": current_round,
            "session": {
                "session_id": payload.get("session_id"),
                "mode": payload.get("mode"),
                "safe_mode": payload.get("safe_mode"),
            },
            "run": run,
            "cognitive_snapshot": cognitive,
        }

    def console_action_field(self, round_ref: int | str | None = None) -> dict[str, Any]:
        effective_round = self.resolve_round_ref(round_ref)
        why_payload = self.why_this(effective_round)
        probability_payload = self.trace_probability_field(effective_round)
        contribution_payload = self.contribution_breakdown(effective_round)
        explanation = dict(why_payload.get("action_probability_explanation", {}) or {})
        winner_target = explanation.get("winner_target") or why_payload.get("sampled_action")
        winner_posterior = dict(explanation.get("winner_posterior", {}) or {})
        competing_peaks = list(explanation.get("competing_peaks", []) or [])
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
            "token_field": probability_payload.get("token_state", {}),
            "contribution_stack": explanation.get("stacked_contributions", contribution_payload.get("contributions", [])),
            "competing_peaks": competing_peaks,
        }

    def console_timeline(self, round_ref: int | str | None = None) -> dict[str, Any]:
        effective_round = self.resolve_round_ref(round_ref)
        trace = self.trace_round(effective_round)
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
        payload = self.why_this(effective_round)
        return {
            "round_id": payload["round_id"],
            "trace_ref": f"round://{payload['round_id']}",
            "why": {
                "summary": str(payload.get("sampled_action") or "暂无"),
                "sampled_action": payload.get("sampled_action"),
                "top_drivers": payload.get("top_drivers", []),
                "vitality_snapshot": payload.get("vitality_snapshot", {}),
                "authenticity": payload.get("authenticity", {}),
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
            },
        }

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
        rounds = list(self.metrics_timeline().get("rounds", []) or [])
        return rounds[-limit:]

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

    def console_refresh_payload(self, round_ref: int | str | None = None) -> dict[str, Any]:
        effective_round = round_ref if round_ref is not None else self._console_round_or_none()
        payload: dict[str, Any] = {
            "state": self.console_state(),
            "autonomy": self.autonomy_status(),
        }
        if effective_round is None:
            payload["action_field"] = {
                "round_id": None,
                "trace_ref": None,
                "top_actions": [],
                "winner": {"action": None, "score": 0.0},
                "winner_posterior": {},
                "conflict": {},
                "token_field": {},
                "contribution_stack": [],
                "competing_peaks": [],
            }
            payload["timeline"] = {"round_id": None, "trace_ref": None, "events": []}
            payload["why_current"] = {
                "round_id": None,
                "trace_ref": None,
                "why": {"summary": "暂无", "sampled_action": None, "top_drivers": [], "vitality_snapshot": {}, "authenticity": {}},
            }
            payload["why_not"] = None
            payload["probability_field"] = {}
            payload["counterfactual_preview"] = None
            payload["recent_rounds"] = self._console_recent_rounds()
            payload["source_links"] = self._console_source_links(round_id=None, trace_ref=None, why_not_action=None)
            return payload
        payload["action_field"] = self.console_action_field(effective_round)
        payload["probability_field"] = self.trace_probability_field(effective_round).get("probability_field", {})
        payload["timeline"] = self.console_timeline(effective_round)
        payload["why_current"] = self.console_why_current(effective_round)
        default_why_not_action = self._console_default_why_not_action(effective_round, action_field=payload["action_field"])
        payload["why_not"] = self.console_why_not(default_why_not_action, effective_round) if default_why_not_action else None
        payload["counterfactual_preview"] = self.replay(int(self.resolve_round_ref(effective_round))).get("counterfactual_preview", {})
        payload["recent_rounds"] = self._console_recent_rounds()
        payload["source_links"] = self._console_source_links(
            round_id=int(self.resolve_round_ref(effective_round)),
            trace_ref=payload["action_field"].get("trace_ref"),
            why_not_action=default_why_not_action,
        )
        return payload

    def console_recent_actions(self, *, limit: int = 8) -> dict[str, Any]:
        self.trace_store.flush(raise_on_error=False)
        rows = list(self.trace_store.list_rounds()[-limit:])
        actions = [
            {
                "round_id": int(row.get("round_id", 0) or 0),
                "trace_ref": f"round://{int(row.get('round_id', 0) or 0)}",
                "action": str(row.get("sampled_action") or "nothing"),
                "summary": f"{str(row.get('cause_type') or 'external_stimulus')} · {str(row.get('mode') or 'interactive')}",
                "recorded_at": row.get("recorded_at"),
                "cause_type": str(row.get("cause_type") or "external_stimulus"),
                "mode": str(row.get("mode") or "interactive"),
            }
            for row in rows
        ]
        return {
            "actions": actions,
            "message": "暂无最近动作" if not actions else "",
        }

    def console_probability_space(self) -> dict[str, Any]:
        state = self.load_runtime_state()
        self._sync_tlh_state(state)
        instinct_field = to_dict(state.instinct_field)
        anchor = to_dict(state.personality_anchor)
        latest_round = self._console_round_or_none()
        probability_field = (
            self.trace_probability_field(latest_round).get("probability_field", {})
            if latest_round is not None
            else {}
        )
        region_scores = dict(instinct_field.get("region_scores", {}) or {})
        winner_region = str(instinct_field.get("winner_region") or "")
        axis_values = {
            axis: round(float((instinct_field.get("axis_values") or {}).get(axis, 0.0) or 0.0), 4)
            for axis in ("E", "F", "S", "M")
        }
        anchor_axes = {
            axis: round(float((anchor.get("axis_baseline") or {}).get(axis, 0.5) or 0.5), 4)
            for axis in ("E", "F", "S", "M")
        }
        region_vectors = {
            "express": {"E": 0.82, "F": 0.24, "S": 0.48, "M": 0.74},
            "withdraw": {"E": 0.18, "F": 0.78, "S": 0.32, "M": 0.36},
            "hibernate": {"E": 0.12, "F": 0.86, "S": 0.12, "M": 0.22},
            "dissolve": {"E": 0.08, "F": 0.62, "S": 0.56, "M": 0.12},
            "absorb": {"E": 0.36, "F": 0.34, "S": 0.82, "M": 0.58},
        }
        winner_vector = region_vectors.get(winner_region, {"E": axis_values["E"], "F": axis_values["F"], "S": axis_values["S"], "M": axis_values["M"]})
        winner_strength = max(region_scores.values(), default=0.0)
        predicted_axes = {
            axis: round(_clip(axis_values[axis] * 0.7 + float(winner_vector.get(axis, axis_values[axis])) * min(0.3, winner_strength * 0.18)), 4)
            for axis in ("E", "F", "S", "M")
        }
        candidate_center = {
            axis: round((axis_values[axis] + predicted_axes[axis]) / 2.0, 4)
            for axis in ("E", "F", "S", "M")
        }
        plots = []
        for x_axis, y_axis in (("E", "F"), ("E", "S"), ("E", "M"), ("F", "S")):
            plots.append(
                {
                    "label": f"{x_axis}-{y_axis}",
                    "x_axis": x_axis,
                    "y_axis": y_axis,
                    "current_point": {"x": axis_values[x_axis], "y": axis_values[y_axis]},
                    "predicted_point": {"x": predicted_axes[x_axis], "y": predicted_axes[y_axis]},
                    "anchor_point": {"x": anchor_axes[x_axis], "y": anchor_axes[y_axis]},
                    "candidate_point": {"x": candidate_center[x_axis], "y": candidate_center[y_axis]},
                    "winner_region": winner_region,
                    "has_data": any(abs(value) > 1e-9 for value in axis_values.values()) or bool(winner_region),
                }
            )
        layer_labels = {
            "context": "情境层",
            "memory": "记忆层",
            "action": "动作层",
            "token": "Token",
        }
        layers = []
        for key in ("context", "memory", "action", "token"):
            layer_payload = dict(probability_field.get(key, {}) or {})
            layers.append(
                {
                    "key": key,
                    "label": layer_labels[key],
                    "winner": layer_payload.get("winner") or layer_payload.get("winner_target") or "",
                    "keys": len(layer_payload),
                    "empty": not bool(layer_payload),
                }
            )
        return {
            "plots": plots,
            "layers": layers,
            "instinct_field": instinct_field,
            "message": "尚无概率分布" if latest_round is None else "",
        }

    def state_delta_timeline(self, window: int = 20) -> dict[str, Any]:
        rounds = self.trace_store.list_rounds()[-window:]
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
        rounds = self.trace_store.list_rounds()[-window:]
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
        return {
            "round_id": trace["round_id"],
            "failure_mode": failure_mode,
            "blockers": deduped,
            "appraisal_snapshot": appraisal,
            "state_delta_before_clip": before,
            "state_delta_after_clip": after,
            "storage": trace.get("storage", self._trace_storage_payload()),
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

    def _trace_storage_payload(self, *, read_source: str | None = None) -> dict[str, Any]:
        payload = dict(self.trace_store.trace_storage_status())
        if read_source is not None:
            payload["read_source"] = read_source
        return payload

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
        current_step = self._current_task_node(run_state)
        return {
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

        recorded_at = utc_now_iso()
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
        recorded_at = utc_now_iso()
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
            run_state.updated_at = utc_now_iso()
            self._persist_run_state(run_state)
        return self.run_status(run_state.run_id)

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
        self.trace_store.write_run(to_dict(run_state), session_id=state.session_id, recorded_at=utc_now_iso(), sync=True)

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
            run_state.updated_at = utc_now_iso()
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
        run_state.updated_at = utc_now_iso()
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
        run_state.updated_at = utc_now_iso()
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

    def trace_round(self, round_ref: int | str) -> dict[str, Any]:
        self.trace_store.flush(raise_on_error=False)
        payload, read_source = self.trace_store.read_round_record(self.resolve_round_ref(round_ref))
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
        trace = self.trace_round(round_ref)
        personality_anchor = trace.get("state_snapshot", {}).get("personality_anchor", {})
        instinct_field = trace.get("state_snapshot", {}).get("instinct_field", {})
        emergent_sketches = trace.get("state_snapshot", {}).get("emergent_action_sketches", [])
        return {
            "round_id": trace["round_id"],
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
                "mode": trace["state_snapshot"]["mode"],
                "safe_mode": trace["state_snapshot"]["safe_mode"],
                "focus": trace["state_snapshot"]["focus"],
                "budget_remaining": trace["state_snapshot"]["budget_remaining"],
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

    def _action_layer_from_trace(self, trace: dict[str, Any]) -> dict[str, Any]:
        probability_field = canonical_probability_field_payload(trace)
        action_layer = probability_field.get("action", {})
        return action_layer if isinstance(action_layer, dict) else {}

    def _token_state_from_trace(self, trace: dict[str, Any]) -> dict[str, Any]:
        probability_field = canonical_probability_field_payload(trace)
        token_state = probability_field.get("token_state", {})
        return token_state if isinstance(token_state, dict) else {}

    def _cross_layer_coupling_verdict(self, trace: dict[str, Any]) -> dict[str, Any]:
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

    def _renderer_decision_integrity(self, trace: dict[str, Any]) -> dict[str, Any]:
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

    def _competing_peaks(self, action_layer: dict[str, Any], *, limit: int = 3) -> list[dict[str, Any]]:
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

    def _stacked_action_contributions(
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

    def _action_probability_explanation(self, trace: dict[str, Any], *, target_action: str) -> dict[str, Any]:
        action_layer = self._action_layer_from_trace(trace)
        final_energy = action_layer.get("final_energy", {}) if isinstance(action_layer.get("final_energy"), dict) else {}
        winner_target = str(action_layer.get("winner_target") or trace.get("sampled_action") or "")
        sampled_action = str(trace.get("sampled_action") or "")
        stacked = self._stacked_action_contributions(action_layer, target_action)
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
        return {
            "target_action": target_action,
            "winner_target": winner_target,
            "sampled_action": sampled_action,
            "sampled_differs_from_peak": bool(sampled_action and winner_target and sampled_action != winner_target),
            "winner_posterior": dict(action_layer.get("winner_posterior", {}) or {}),
            "winner_energy": round(float(final_energy.get(winner_target, 0.0) or 0.0), 6) if winner_target else 0.0,
            "target_final_energy": round(float(final_energy.get(target_action, 0.0) or 0.0), 6)
            if target_action in final_energy and final_energy.get(target_action) != float("-inf")
            else None,
            "hard_masked": target_action in list(action_layer.get("hard_masked_targets", []) or []),
            "stacked_contributions": stacked,
            "competing_peaks": competing_peaks[:3],
        }

    def _counterfactual_render_preview(self, trace: dict[str, Any], action: str) -> dict[str, Any]:
        render_plan_payload = dict(trace.get("render_plan", {}) or {})
        if not render_plan_payload:
            return {
                "action": action,
                "would_output": action != "nothing",
                "text": "",
                "terminal_intent": action == "die",
            }
        render_plan_payload["action"] = action
        message_plan = dict(render_plan_payload.get("message_plan", {}) or {})
        message_plan["intent"] = action
        message_plan["focus"] = action
        render_plan_payload["message_plan"] = message_plan
        preview_plan = RenderPlan(**render_plan_payload)
        preview_text = fallback_render_text(preview_plan)
        return {
            "action": action,
            "would_output": bool(preview_text),
            "text": preview_text,
            "terminal_intent": action == "die",
        }

    def _emergent_action_formalization_payload(self, sketches: list[dict[str, Any]] | list[Any]) -> list[dict[str, Any]]:
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

    def _sample_counterfactual_action(
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

    def _counterfactual_replays_from_trace(self, trace: dict[str, Any], *, limit: int = 4) -> list[dict[str, Any]]:
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
        for peak in self._competing_peaks(action_layer, limit=max(limit, 6)):
            action_name = str(peak.get("action") or "").strip()
            if action_name and action_name not in action_order:
                action_order.append(action_name)
        for action in action_order[:limit]:
            peak = next((item for item in self._competing_peaks(action_layer, limit=max(limit, 6)) if item.get("action") == action), None)
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
        return rows

    def _conflict_arbitration_summary(self, trace: dict[str, Any]) -> dict[str, Any]:
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

    def contribution_breakdown(self, round_ref: int | str) -> dict[str, Any]:
        trace = self.trace_round(round_ref)
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
        trace = self.trace_round(round_ref)
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
        trace = self.trace_round(round_ref)
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
        trace = self.trace_round(round_ref)
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
            "storage": trace.get("storage", self._trace_storage_payload()),
        }

    def trace_skills(self, round_ref: int | str) -> dict[str, Any]:
        trace = self.trace_round(round_ref)
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
            "storage": trace.get("storage", self._trace_storage_payload()),
        }

    def _average(self, values: list[float]) -> float:
        return round(sum(values) / len(values), 4) if values else 0.0

    def _estimate_affect_half_life(self, rounds: list[dict[str, Any]]) -> float:
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

    def _estimate_recovery_duration(self, rounds: list[dict[str, Any]]) -> float:
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

    def _event_variance_metric(self, rounds: list[dict[str, Any]], bucket_fn) -> float:
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

    def metrics_summary(self) -> dict[str, Any]:
        self.trace_store.flush(raise_on_error=False)
        payload = self.long_run_analyzer.metrics_summary()
        motivation = self.motivation_metrics()
        endogenous = self.endogenous_metrics()
        payload.update(
            {
                "boundary_violation_count": self._subjectivity_metrics()["boundary_violation_count"],
                "external_to_internal_ratio": self._subjectivity_metrics()["external_to_internal_ratio"],
                "endogenous_intent_rate": self._subjectivity_metrics()["endogenous_intent_rate"],
                "motivation_active_rate": motivation["active_rate"],
                "motivation_activation_score_avg": motivation["activation_score_avg"],
                "endogenous_round_rate": endogenous["endogenous_round_rate"],
            }
        )
        payload["storage"] = self._trace_storage_payload()
        return payload

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

    def motivation_metrics(self) -> dict[str, Any]:
        self.trace_store.flush(raise_on_error=False)
        rounds = self.trace_store.list_rounds()
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

    def endogenous_metrics(self) -> dict[str, Any]:
        self.trace_store.flush(raise_on_error=False)
        rounds = self.trace_store.list_rounds()
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
            "subjectivity": self._subjectivity_metrics(),
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
        skill_rows = sorted(
            self.trace_store.list_skill_traces(),
            key=lambda row: (int(row.get("round_id", 0)), str(row.get("recorded_at", ""))),
        )
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

        tiers = {
            tier_name: {
                "mode": str(tier_cfg.get("mode", "local")),
                "backend": tier_cfg.get("backend"),
                "model": tier_cfg.get("model"),
                "base_url": tier_cfg.get("base_url"),
                "timeout_ms": tier_cfg.get("timeout_ms"),
                "api_key_env": tier_cfg.get("api_key_env"),
                "credential_present": (
                    True
                    if not str(tier_cfg.get("api_key_env", "")).strip()
                    else bool(os.getenv(str(tier_cfg.get("api_key_env", "")).strip()))
                )
                if str(tier_cfg.get("mode", "local")).lower() == "remote"
                else True,
            }
            for tier_name, tier_cfg in self._model_tiers().items()
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
            route_credential_present = credential_visible(status_route)
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
            }

        return {
            "current_round": state.round_count,
            "credential_present": credential_present,
            "tiers": tiers,
            "agent_bindings": self._agent_model_bindings(),
            "routes": routes,
        }

    def relation_show(self, target: str) -> dict[str, Any]:
        return self.memory_store.relation_state(target)

    def replay_round(self, round_id: int, seed: int | None = None) -> dict[str, Any]:
        return self.replay(round_id, seed=seed or 0)

    def replay(self, round_id: int, seed: int = 0) -> dict[str, Any]:
        trace = self.trace_round(round_id)
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
        trace = self.trace_round(round_id)
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
        trace = self.trace_round(round_ref)
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
        trace = self.trace_round(round_id)
        candidate_distribution = self._candidate_distribution_from_trace(trace)
        action_explanation = self._action_probability_explanation(trace, target_action=action)
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
        return {
            "round_id": round_id,
            "action": action,
            "selected_action": trace["sampled_action"],
            "candidate_score": candidate_distribution.get(action, 0.0),
            "blocked_by": list(dict.fromkeys(blocked_by)),
            "stacked_contributions": action_explanation["stacked_contributions"],
            "competing_peaks": action_explanation["competing_peaks"],
            "conflict_arbitration": self._conflict_arbitration_summary(trace),
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
            "storage": self._trace_storage_payload(),
            "message": "无决策记录",
        }

    def what_changed(self, window: int = 5) -> dict[str, Any]:
        rounds = self.trace_store.list_rounds()[-window:]
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
        for trace in self.trace_store.list_rounds():
            conflict = dict(trace.get("conflict_arbitration", {}) or self._conflict_arbitration_summary(trace))
            ledger_tail = conflict.get("repair_ledger_tail", [])
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
                    "repair_ledger_summary": {
                        "entries": len(ledger_tail),
                        "latest_reason": ledger_tail[-1]["reason"] if ledger_tail else "",
                    },
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
        for trace in self.trace_store.list_rounds():
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
        rounds = self.trace_store.list_rounds()
        return {
            "rounds": [
                {
                    "round_id": row["round_id"],
                    "sampled_action": row["sampled_action"],
                    "mode": row["mode"],
                    "budget_remaining": row.get("state_snapshot", {}).get("budget_remaining"),
                    "conflict_score": (row.get("conflict_arbitration", {}) or {}).get("total_score", 0.0),
                    "cause_type": row.get("cause_type", "external_stimulus"),
                }
                for row in rounds
            ],
            "subjectivity": self._subjectivity_metrics(),
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

    def _bypass_detection_summary(self, rounds: list[dict[str, Any]]) -> dict[str, Any]:
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

    def _parallel_evidence_summary(self, rounds: list[dict[str, Any]]) -> dict[str, Any]:
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

    def _scale_consistency_summary(self, rounds: list[dict[str, Any]]) -> dict[str, Any]:
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

    def _cross_layer_coupling_summary(self, rounds: list[dict[str, Any]]) -> dict[str, Any]:
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

    def _renderer_decision_integrity_summary(self, rounds: list[dict[str, Any]]) -> dict[str, Any]:
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

    def _memory_write_gate_summary(self, rounds: list[dict[str, Any]]) -> dict[str, Any]:
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

    def _failure_taxonomy_from_trace(self, trace: dict[str, Any]) -> list[str]:
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

    def _failure_taxonomy_summary(self, rounds: list[dict[str, Any]]) -> dict[str, Any]:
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

    def _conflict_arbitration_acceptance_summary(self, rounds: list[dict[str, Any]]) -> dict[str, Any]:
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

    def _long_run_prior_summary(self, rounds: list[dict[str, Any]]) -> dict[str, Any]:
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
                p_final = self._candidate_distribution_from_trace(trace)
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

    def acceptance_report(self, window: int = 20) -> dict[str, Any]:
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
            "long_run_prior": self._long_run_prior_summary(rounds),
            "renderer_decision_integrity": self._renderer_decision_integrity_summary(rounds),
            "memory_write_gate": self._memory_write_gate_summary(rounds),
            "failure_taxonomy": self._failure_taxonomy_summary(rounds),
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
