from __future__ import annotations

import json
import shutil
import time
from typing import TYPE_CHECKING, Any

from nalr.dream.orchestrator import DreamOrchestrator
from nalr.memory.store import MemoryStore
from nalr.runtime.longrun import LongRunAnalyzer
from nalr.runtime.metadata import utc_now_iso
from nalr.schemas.models import (
    AutonomyLoopState,
    BodyState,
    CheckpointRef,
    CommandResult,
    ConflictRepairState,
    DesireState,
    EmotionState,
    EndogenousRuntimeState,
    EndogenousSchedulerState,
    IdentityState,
    InstinctFieldState,
    MotivationLearningState,
    MotivationPoolState,
    OrganicModeState,
    PersonalityAnchorState,
    RuntimeState,
    SubjectCore,
    SubjectiveState,
    to_dict,
)
from nalr.skills.executor import SkillExecutor
from nalr.storage.parquet_io import read_snapshot_rows
from nalr.trace.exporter import TraceExporter
from nalr.trace.store import TraceStore

if TYPE_CHECKING:
    from nalr.runtime.controller import RuntimeController


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


CORE_ACTIONS = ("respond", "plan", "recall", "rest", "connect", "clarify", "wander", "absorb", "monologue", "nothing", "die")


class StateRuntimeService:
    def __init__(self, controller: "RuntimeController") -> None:
        self.controller = controller

    def checkpoint(self) -> CheckpointRef:
        controller = self.controller
        state = controller.load_runtime_state()
        before_hash = controller._state_hash(state)
        controller.flush_pending_io(raise_on_error=True)
        checkpoint_id = f"ckpt-{state.round_count:04d}"
        path = controller.checkpoint_dir / f"{checkpoint_id}.parquet"
        shutil.copyfile(controller.state_parquet_path, path)
        state.last_checkpoint_id = checkpoint_id
        controller._save_state(state, sync=True)
        after_hash = controller._state_hash(state)
        controller.trace_store.append_command(
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
        controller = self.controller
        checkpoint_path = controller.checkpoint_dir / f"{checkpoint_id}.parquet"
        if not checkpoint_path.exists():
            return CommandResult(
                applied=False,
                scope="checkpoint",
                delta={"checkpoint_id": checkpoint_id, "restored": False},
                risk_note="checkpoint not found",
                operator_level="ops_admin",
                rollback_available=False,
            )
        before_state = controller.load_runtime_state()
        before_hash = controller._state_hash(before_state)
        controller.flush_pending_io(raise_on_error=True)
        shutil.copyfile(checkpoint_path, controller.state_parquet_path)
        rows = read_snapshot_rows(controller.state_parquet_path, "select payload_json from read_parquet(?)")
        controller._state_cache = RuntimeState(**json.loads(rows[0]["payload_json"]))
        restored_state = controller.load_runtime_state()
        after_hash = controller._state_hash(restored_state)
        result = CommandResult(
            applied=True,
            scope="checkpoint",
            delta={
                "checkpoint_id": checkpoint_id,
                "restored": True,
                "safe_mode": restored_state.safe_mode,
                "mode": restored_state.mode,
            },
            rollback_hint="create a fresh checkpoint before further changes",
            operator_level="ops_admin",
            rollback_available=True,
        )
        controller.trace_store.append_command(
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
        return self.controller.memory_store.memory_top(limit=limit)

    def memory_recall(self, cue: str) -> dict[str, Any]:
        return self.controller.memory_store.recall(cue)

    def compact_memory(self, *, hot_max_rounds: int = 500, warm_max_rounds: int = 3000) -> dict[str, Any]:
        return self.controller.memory_store.compact_tiers(
            hot_max_rounds=hot_max_rounds,
            warm_max_rounds=warm_max_rounds,
        )

    def sample_memory(self, tier: str, *, limit: int = 5, cue: str | None = None) -> list[dict[str, Any]]:
        return self.controller.memory_store.sample_compacted(tier, limit=limit, cue=cue)

    def export_trace_parquet(self, *, since_round: int | None = None, overwrite: bool = False) -> dict[str, Any]:
        controller = self.controller
        controller.flush_pending_io(raise_on_error=True)
        exporter = TraceExporter(controller.trace_store)
        payload = exporter.export_parquet(since_round=since_round, overwrite=overwrite)
        controller.trace_store.mark_trace_sync_healthy()
        return payload

    def habit_top(self, limit: int = 5) -> list[dict]:
        return self.controller.memory_store.habit_top(limit=limit)

    def identity_payload(self) -> dict[str, Any]:
        controller = self.controller
        state = controller.load_runtime_state()
        payload = to_dict(state.identity_state)
        payload["display_name"] = payload.get("display_name") or controller._unnamed_label()
        return payload

    def observer_settings_payload(self) -> dict[str, Any]:
        controller = self.controller
        settings = controller._observer_settings()
        return {
            "identity": self.identity_payload(),
            "autonomy": settings.get("autonomy", {}),
            "newborn": settings.get("newborn", {}),
            "models": {
                "supported_backends": list(settings.get("models", {}).get("supported_backends", [])),
                "provider_endpoints": dict(settings.get("models", {}).get("provider_endpoints", {})),
                "model_tiers": controller._model_tiers(),
                "model_routes": dict(controller.config["models"].get("model_routes", {})),
                "module_model_bindings": controller._module_model_bindings(),
            },
        }

    def newborn_runtime_state(self) -> RuntimeState:
        controller = self.controller
        observer_settings = controller._observer_settings()
        newborn_settings = observer_settings.get("newborn", {}) if isinstance(observer_settings.get("newborn"), dict) else {}
        organic_settings = newborn_settings.get("organic_mode", {}) if isinstance(newborn_settings.get("organic_mode"), dict) else {}
        subjective_settings = (
            newborn_settings.get("subjective_state", {})
            if isinstance(newborn_settings.get("subjective_state"), dict)
            else {}
        )
        state = RuntimeState(
            agents_enabled={
                name: agent_cfg.get("enabled", True)
                for name, agent_cfg in controller.config["agents"]["agents"].items()
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
            felt=controller._normalize_observer_string_list(list(subjective_settings.get("felt", []))),
            spontaneous=round(_clip(float(subjective_settings.get("spontaneous", 0.0) or 0.0)), 4),
            boundary=round(_clip(float(subjective_settings.get("boundary", 0.0) or 0.0)), 4),
            reject_all=round(_clip(float(subjective_settings.get("reject_all", 0.0) or 0.0)), 4),
            meaning_made=controller._normalize_observer_string_list(list(subjective_settings.get("meaning_made", []))),
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
        state.autonomy_policy = controller._autonomy_policy_for_profile("tool_level")
        state.autonomy_policy.enabled = True
        state.autonomy_loop = AutonomyLoopState(
            running=True,
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
        state.session_metadata["autonomy_default_enabled_at"] = utc_now_iso()
        controller._ensure_subject_core(state)
        return RuntimeState(**to_dict(state))

    def rebind_runtime_storage(self) -> None:
        controller = self.controller
        controller.trace_store = TraceStore(controller.home_path)
        controller.memory_store = MemoryStore(controller.home_path)
        controller.skill_executor = SkillExecutor(
            controller.skills,
            circuit_breaker_path=controller.memory_store.circuit_breaker_path,
            environment_fingerprint=controller._breaker_environment_fingerprint(),
        )
        controller.dream_orchestrator = DreamOrchestrator(
            project_root=controller.project_root,
            home_path=controller.home_path,
            config=controller.config["dream"]["dream"],
            memory_store=controller.memory_store,
            vitality_engine=controller.vitality_engine,
        )
        controller.long_run_analyzer = LongRunAnalyzer(
            controller.trace_store,
            controller.identity_payload,
            CORE_ACTIONS,
        )

    def reset_persona(self) -> RuntimeState:
        controller = self.controller
        controller.flush_pending_io(raise_on_error=False)
        controller._pending_endogenous_trigger = None
        controller._pending_endogenous_trigger_context = None
        controller._trace_round_cache.clear()
        controller._trace_derived_cache.clear()
        preserve_newborn_settings = controller.observer_settings_path.exists()
        for path in (
            controller.home_path / "traces",
            controller.home_path / "memory",
            controller.home_path / "dream",
            controller.checkpoint_dir,
            controller.snapshot_dir,
        ):
            shutil.rmtree(path, ignore_errors=True)
        if controller.state_path.exists():
            controller.state_path.unlink()
        if controller.state_parquet_path.exists():
            controller.state_parquet_path.unlink()
        controller.runtime_parquet_dir.mkdir(parents=True, exist_ok=True)
        controller.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        controller.snapshot_dir.mkdir(parents=True, exist_ok=True)
        controller._state_cache = None
        self.rebind_runtime_storage()
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
        newborn.session_metadata.pop("autonomy_user_disabled", None)
        controller._ensure_default_autonomy_runtime(newborn, force=True)
        controller._save_state(newborn, sync=True)
        controller._rounds_since_cold_flush = 0
        controller._last_cold_flush_at = time.monotonic()
        return controller.load_runtime_state()

    def _apply_startup_unlock_preferences(self, state: RuntimeState) -> None:
        controller = self.controller
        settings = controller._observer_settings()
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
            state.subjective_state.felt = controller._normalize_observer_string_list(list(subjective.get("felt", [])))
        if not state.subjective_state.meaning_made:
            state.subjective_state.meaning_made = controller._normalize_observer_string_list(list(subjective.get("meaning_made", [])))
        controller._ensure_default_autonomy_runtime(state)

    def apply_startup_unlock_preferences(self, state: RuntimeState) -> None:
        self._apply_startup_unlock_preferences(state)

    def initiative_status_from_state(self, state: RuntimeState) -> dict[str, Any]:
        return self.controller._initiative_status_payload(state)

    def save_runtime_state(self, state: RuntimeState, *, sync: bool = False) -> None:
        self.controller._save_state(state, sync=sync)

    def update_observer_settings(self, payload: dict[str, Any]) -> dict[str, Any]:
        controller = self.controller
        normalized = controller.normalize_observer_settings_payload(payload, base=controller.observer_settings_current())
        controller.observer_settings_path.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")
        controller.refresh_runtime_components_from_config()
        state = controller.load_runtime_state()
        controller.apply_startup_unlock_preferences(state)
        if getattr(state.autonomy_loop, "profile", ""):
            previous_enabled = bool(getattr(state.autonomy_policy, "enabled", False))
            state.autonomy_policy = controller._autonomy_policy_for_profile(state.autonomy_loop.profile or state.autonomy_policy.profile)
            state.autonomy_policy.enabled = previous_enabled
        controller.save_runtime_state(state, sync=True)
        return controller.observer_settings_payload()

    def state_payload(self) -> dict[str, Any]:
        controller = self.controller
        controller.flush_pending_io(raise_on_error=False)
        state = controller.load_runtime_state()
        controller.sync_tlh_state(state)
        controller.sync_autonomy_state(state)
        controller.sync_plan16_state(state)
        runtime_metrics = controller.latest_runtime_metrics()
        payload = to_dict(state)
        payload.update(controller.runtime_status_truth_payload(state))
        payload["initiative"] = controller.initiative_status_from_state(state)
        payload["subjectivity"] = controller.subjectivity_metrics()
        payload["state_truth"] = controller.state_truth_payload(state)
        payload["trace_storage"] = controller.trace_storage_status()
        payload["memory_storage"] = controller.memory_store.storage_status()
        payload["runtime_storage"] = controller.runtime_storage_status()
        payload["migration"] = controller.runtime_migration_report()
        payload["entropy"] = controller.entropy_pool.health_snapshot()
        payload["dream"] = controller.dream_status()
        payload["cognitive_snapshot"] = controller.cognitive_snapshot(state=state)
        payload["internal_learning"] = controller.internal_learning_summary(state)
        payload["runtime_metrics"] = runtime_metrics
        payload["performance"] = controller.diagnostics_runtime.performance_payload_from_runtime_metrics(runtime_metrics)
        return payload

    def state_hot_payload(self) -> dict[str, Any]:
        controller = self.controller
        state = controller.load_runtime_state()
        controller.sync_tlh_state(state)
        controller.sync_autonomy_state(state)
        controller.sync_plan16_state(state)
        runtime_metrics = controller.latest_runtime_metrics()
        payload = to_dict(state)
        payload.update(controller.runtime_status_truth_payload(state))
        payload["initiative"] = controller.initiative_status_from_state(state)
        payload["subjectivity"] = controller.subjectivity_metrics()
        payload["state_truth"] = controller.state_truth_payload(state)
        payload["cognitive_snapshot"] = controller.cognitive_snapshot(state=state)
        payload["runtime_metrics"] = runtime_metrics
        payload["performance"] = controller.diagnostics_runtime.performance_payload_from_runtime_metrics(runtime_metrics)
        return payload
