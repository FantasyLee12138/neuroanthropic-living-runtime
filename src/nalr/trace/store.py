from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import time
from uuid import uuid4

import duckdb

from nalr.runtime.async_io import AsyncIOWorker
from nalr.runtime.metadata import ensure_recorded_fields, iso_date, utc_now_iso
from nalr.schemas.models import CommandResult, RoundTrace, to_dict
from nalr.storage.parquet_io import append_dataset, read_dataset_rows


def _mtime_iso(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_json_cell(value: object, default):
    if value in (None, ""):
        return copy.deepcopy(default)
    if isinstance(value, (dict, list)):
        return copy.deepcopy(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if parsed is None:
                return copy.deepcopy(default)
            return parsed
        except json.JSONDecodeError:
            return copy.deepcopy(default)
    return copy.deepcopy(default)


def _looks_like_probability_field(payload: dict[str, object]) -> bool:
    return any(key in payload for key in ("context", "memory", "action", "token_state", "couplings"))


def _float_map(payload: object) -> dict[str, float]:
    if not isinstance(payload, dict):
        return {}
    result: dict[str, float] = {}
    for action, value in payload.items():
        if not isinstance(action, str) or not action:
            continue
        try:
            result[action] = float(value)
        except (TypeError, ValueError):
            continue
    return result


def canonical_probability_field_payload(payload: dict[str, object]) -> dict[str, object]:
    probability_field = payload.get("probability_field")
    if isinstance(probability_field, dict) and probability_field:
        return copy.deepcopy(probability_field)
    return {}


def legacy_probability_field_payload(payload: dict[str, object]) -> dict[str, object]:
    distribution_state = payload.get("distribution_state")
    if not isinstance(distribution_state, dict) or not distribution_state:
        return {}
    if _looks_like_probability_field(distribution_state):
        return copy.deepcopy(distribution_state)

    winner_posterior = (
        _float_map(distribution_state.get("p_final"))
        or _float_map(distribution_state.get("p_mix"))
        or _float_map(distribution_state.get("p_raw"))
        or _float_map(distribution_state.get("p_base"))
    )
    final_energy = _float_map(distribution_state.get("u_shifted")) or _float_map(distribution_state.get("u_base"))
    hard_masked_targets = [
        str(action)
        for action in list(dict(distribution_state.get("conflict", {}) or {}).get("hard_masked_targets", []) or [])
        if str(action)
    ]
    winner_target = str(payload.get("sampled_action") or "")
    if winner_posterior:
        winner_target = max(winner_posterior, key=winner_posterior.get)

    action_layer: dict[str, object] = {}
    if winner_posterior:
        action_layer["winner_posterior"] = winner_posterior
    if final_energy:
        action_layer["final_energy"] = final_energy
    if winner_target:
        action_layer["winner_target"] = winner_target
    if hard_masked_targets:
        action_layer["hard_masked_targets"] = hard_masked_targets
    if action_layer:
        action_layer.setdefault("contribution_audit", [])

    rebuilt: dict[str, object] = {}
    if action_layer:
        rebuilt["action"] = action_layer
    token_state = payload.get("token_state")
    if isinstance(token_state, dict) and token_state:
        rebuilt["token_state"] = copy.deepcopy(token_state)
    couplings = payload.get("couplings")
    if isinstance(couplings, list) and couplings:
        rebuilt["couplings"] = copy.deepcopy(couplings)
    return rebuilt


def normalize_round_payload(payload: dict[str, object]) -> dict[str, object]:
    normalized = copy.deepcopy(payload)
    probability_field = canonical_probability_field_payload(normalized)
    if probability_field:
        normalized["probability_field"] = probability_field
        candidate_distribution = normalized.get("candidate_distribution")
        action_layer = probability_field.get("action", {}) if isinstance(probability_field, dict) else {}
        winner_posterior = action_layer.get("winner_posterior", {}) if isinstance(action_layer, dict) else {}
        if (not isinstance(candidate_distribution, dict) or not candidate_distribution) and isinstance(winner_posterior, dict) and winner_posterior:
            normalized["candidate_distribution"] = copy.deepcopy(winner_posterior)
    normalized.pop("distribution_state", None)
    return normalized


def rewrite_round_payload(payload: dict[str, object]) -> dict[str, object]:
    normalized = copy.deepcopy(payload)
    probability_field = canonical_probability_field_payload(normalized) or legacy_probability_field_payload(normalized)
    if probability_field:
        normalized["probability_field"] = probability_field
        candidate_distribution = normalized.get("candidate_distribution")
        action_layer = probability_field.get("action", {}) if isinstance(probability_field, dict) else {}
        winner_posterior = action_layer.get("winner_posterior", {}) if isinstance(action_layer, dict) else {}
        if (not isinstance(candidate_distribution, dict) or not candidate_distribution) and isinstance(winner_posterior, dict) and winner_posterior:
            normalized["candidate_distribution"] = copy.deepcopy(winner_posterior)
    normalized.pop("distribution_state", None)
    return normalized


ROUND_CANONICAL_SCHEMA = {
    "session_id": "VARCHAR",
    "recorded_at": "VARCHAR",
    "recorded_date": "VARCHAR",
    "round_id": "BIGINT",
    "payload_json": "VARCHAR",
}

ROUND_TRACE_SCHEMA = {
    "session_id": "VARCHAR",
    "recorded_at": "VARCHAR",
    "recorded_date": "VARCHAR",
    "round_id": "BIGINT",
    "scenario": "VARCHAR",
    "mode": "VARCHAR",
    "sampled_action": "VARCHAR",
    "stage": "VARCHAR",
    "agent_name": "VARCHAR",
    "action": "VARCHAR",
    "top_action": "VARCHAR",
    "selected": "BOOLEAN",
    "confidence": "DOUBLE",
    "sigma_scale": "DOUBLE",
    "weight_applied": "DOUBLE",
    "delta_p": "DOUBLE",
    "resample_count": "BIGINT",
    "resample_idx": "BIGINT",
    "conflict_score": "DOUBLE",
    "plausibility_fail_score": "DOUBLE",
    "tags_json": "VARCHAR",
    "probability_field_json": "VARCHAR",
    "token_state_json": "VARCHAR",
    "cross_layer_coupling_verdict_json": "VARCHAR",
    "renderer_decision_integrity_json": "VARCHAR",
    "memory_write_gate_json": "VARCHAR",
    "conflict_arbitration_json": "VARCHAR",
    "state_snapshot_json": "VARCHAR",
    "render_plan_json": "VARCHAR",
    "gate_decisions_json": "VARCHAR",
    "rendered_expression_json": "VARCHAR",
    "long_run_projection_json": "VARCHAR",
    "long_run_projection_online_prior_json": "VARCHAR",
    "motivation_pool_json": "VARCHAR",
    "motivation_feedback_json": "VARCHAR",
    "endogenous_tick_reason_json": "VARCHAR",
    "endogenous_policy_shift_json": "VARCHAR",
}

RUN_CANONICAL_SCHEMA = {
    "session_id": "VARCHAR",
    "recorded_at": "VARCHAR",
    "recorded_date": "VARCHAR",
    "run_id": "VARCHAR",
    "payload_json": "VARCHAR",
}

STEP_TOOL_SCHEMA = {
    "session_id": "VARCHAR",
    "recorded_at": "VARCHAR",
    "recorded_date": "VARCHAR",
    "run_id": "VARCHAR",
    "payload_json": "VARCHAR",
}

SKILL_CANONICAL_SCHEMA = {
    "session_id": "VARCHAR",
    "recorded_at": "VARCHAR",
    "recorded_date": "VARCHAR",
    "round_id": "BIGINT",
    "skill_name": "VARCHAR",
    "payload_json": "VARCHAR",
}

SKILL_TRACE_SCHEMA = {
    "session_id": "VARCHAR",
    "recorded_at": "VARCHAR",
    "recorded_date": "VARCHAR",
    "round_id": "BIGINT",
    "skill_name": "VARCHAR",
    "owner_module": "VARCHAR",
    "latency_ms": "BIGINT",
    "cost_class": "VARCHAR",
    "input_hash": "VARCHAR",
    "output_hash": "VARCHAR",
    "failure_policy_applied": "VARCHAR",
    "degraded": "BOOLEAN",
    "seed_ref": "BIGINT",
}

COMMAND_CANONICAL_SCHEMA = {
    "session_id": "VARCHAR",
    "recorded_at": "VARCHAR",
    "recorded_date": "VARCHAR",
    "command": "VARCHAR",
    "payload_json": "VARCHAR",
}

COMMAND_TRACE_SCHEMA = {
    "session_id": "VARCHAR",
    "recorded_at": "VARCHAR",
    "recorded_date": "VARCHAR",
    "command": "VARCHAR",
    "command_id": "VARCHAR",
    "canonical": "VARCHAR",
    "applied": "BOOLEAN",
    "scope": "VARCHAR",
    "delta_json": "VARCHAR",
    "ttl": "VARCHAR",
    "risk_note": "VARCHAR",
    "rollback_hint": "VARCHAR",
    "operator_level": "VARCHAR",
    "rollback_available": "BOOLEAN",
    "mutation_scope": "VARCHAR",
    "parsed_args_json": "VARCHAR",
    "flags_json": "VARCHAR",
    "snapshot_id": "VARCHAR",
    "rollback_json": "VARCHAR",
    "before_state_hash": "VARCHAR",
    "after_state_hash": "VARCHAR",
    "subject_id": "VARCHAR",
    "continuity_nonce": "VARCHAR",
    "cause_type": "VARCHAR",
    "boundary_action": "VARCHAR",
    "violation_code": "VARCHAR",
    "deprecation_warning": "VARCHAR",
}

REPAIR_LEDGER_SCHEMA = {
    "session_id": "VARCHAR",
    "recorded_at": "VARCHAR",
    "recorded_date": "VARCHAR",
    "round_id": "BIGINT",
    "reason": "VARCHAR",
    "winning_priority": "VARCHAR",
    "template": "VARCHAR",
    "repair_stage_after": "VARCHAR",
    "conflict_score": "DOUBLE",
    "payload_json": "VARCHAR",
}

ROUND_SUMMARY_SCHEMA = {
    "session_id": "VARCHAR",
    "recorded_at": "VARCHAR",
    "recorded_date": "VARCHAR",
    "round_id": "BIGINT",
    "payload_json": "VARCHAR",
}

ROUND_SUMMARY_VERSION = 2
COMMAND_SUBJECTIVITY_TOTALS_VERSION = 1
ROUND_SUBJECTIVITY_TOTALS_VERSION = 1


class TraceStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.rounds_dir = self.root / "traces" / "rounds"
        self.rounds_jsonl_path = self.root / "traces" / "round_traces.jsonl"
        self.round_subjectivity_totals_path = self.root / "traces" / "round_subjectivity_totals.json"
        self.runs_dir = self.root / "traces" / "runs"
        self.runs_jsonl_path = self.root / "traces" / "run_traces.jsonl"
        self.step_jsonl_path = self.root / "traces" / "step_traces.jsonl"
        self.tool_jsonl_path = self.root / "traces" / "tool_traces.jsonl"
        self.skills_dir = self.root / "traces" / "skills"
        self.skill_jsonl_path = self.skills_dir / "skill_traces.jsonl"
        self.commands_path = self.root / "traces" / "command_traces.json"
        self.commands_jsonl_path = self.root / "traces" / "command_traces.jsonl"
        self.command_subjectivity_totals_path = self.root / "traces" / "command_subjectivity_totals.json"
        self.repair_jsonl_path = self.root / "traces" / "repair_ledger.jsonl"
        self.parquet_dir = self.root / "traces" / "parquet"
        self.round_canonical_dir = self.parquet_dir / "round_canonical"
        self.round_trace_dir = self.parquet_dir / "round_trace"
        self.round_summary_dir = self.parquet_dir / "round_summary"
        self.run_canonical_dir = self.parquet_dir / "run_canonical"
        self.step_trace_dir = self.parquet_dir / "step_trace"
        self.tool_trace_dir = self.parquet_dir / "tool_trace"
        self.skill_canonical_dir = self.parquet_dir / "skill_canonical"
        self.skill_trace_dir = self.parquet_dir / "skill_trace"
        self.command_canonical_dir = self.parquet_dir / "command_canonical"
        self.command_trace_dir = self.parquet_dir / "command_trace"
        self.repair_ledger_dir = self.parquet_dir / "repair_ledger"
        self.migration_status_path = self.root / "traces" / "trace_migration_status.json"
        self.trace_sync_status_path = self.root / "traces" / "trace_sync_status.json"
        self.rounds_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        self.commands_path.parent.mkdir(parents=True, exist_ok=True)
        self.parquet_dir.mkdir(parents=True, exist_ok=True)
        if not self.commands_path.exists():
            self.commands_path.write_text("[]", encoding="utf-8")
        for path in (
            self.rounds_jsonl_path,
            self.runs_jsonl_path,
            self.step_jsonl_path,
            self.tool_jsonl_path,
            self.skill_jsonl_path,
            self.commands_jsonl_path,
            self.repair_jsonl_path,
        ):
            if not path.exists():
                path.write_text("", encoding="utf-8")
        if not self.trace_sync_status_path.exists():
            self._write_trace_sync_status(self._default_trace_sync_status())
        self._migrate_legacy_if_needed()
        self._io_worker = AsyncIOWorker("nalr-trace-io")
        self._legacy_round_parquet_rewrite_pending = False
        self._round_cache: dict[int, dict] = {}
        self._round_cache_complete = False
        self._round_summary_cache: list[dict] = []
        self._round_summary_loaded = False
        self._round_metrics_cache: list[dict] = []
        self._round_metrics_loaded = False
        self._round_subjectivity_totals_cache: dict[str, object] | None = None
        self._run_cache: dict[str, dict] = {}
        self._run_cache_loaded = False
        self._step_cache: list[dict] = []
        self._step_cache_loaded = False
        self._tool_cache: list[dict] = []
        self._tool_cache_loaded = False
        self._skill_cache: list[dict] = []
        self._skill_cache_loaded = False
        self._command_cache: list[dict] = []
        self._command_cache_loaded = False
        self._command_subjectivity_totals_cache: dict[str, object] | None = None
        self._repair_cache: list[dict] = []
        self._repair_cache_loaded = False
        self._trace_sync_status_cache = json.loads(self.trace_sync_status_path.read_text(encoding="utf-8"))
        self._trace_sync_status_cache.setdefault("parquet_live_ready", self._detect_parquet_live_ready())
        self._signal_view_cache_limit = 100
        self._signal_view_cache: dict[int, dict] = {}
        self._round_count_cache_value: int | None = None
        self._round_count_cache_checked_at = 0.0
        self._round_count_cache_ttl = 1.0

    def _round_signal_view(self, payload: dict) -> dict:
        action_field = dict((dict(payload.get("probability_field", {}) or {}).get("action", {}) or {}))
        appraisal = dict(payload.get("appraisal_snapshot", {}) or {})
        vitality = dict(payload.get("vitality_snapshot", {}) or {})
        conflict = dict(payload.get("conflict_arbitration", {}) or {})
        initiative = dict(payload.get("initiative", {}) or {})
        return {
            "round_id": int(payload.get("round_id", 0) or 0),
            "recorded_at": payload.get("recorded_at"),
            "appraisal_snapshot": {
                "semantic_valence": float(appraisal.get("semantic_valence", 0.0) or 0.0),
                "relation_charge": float(appraisal.get("relation_charge", 0.0) or 0.0),
            },
            "vitality_snapshot": {
                "relationship_closeness": float(vitality.get("relationship_closeness", 0.5) or 0.5),
                "body_energy": float(vitality.get("body_energy", payload.get("body_energy", 0.0)) or 0.0),
            },
            "u_base": dict(dict(payload.get("action_bookkeeping", {}) or {}).get("u_base", {}) or {}),
            "u_shifted": dict(dict(payload.get("action_bookkeeping", {}) or {}).get("u_shifted", {}) or {}),
            "p_final": dict(action_field.get("winner_posterior", {}) or dict(payload.get("candidate_distribution", {}) or {})),
            "conflict_mode": str(conflict.get("conflict_mode") or ""),
            "winner_posterior": dict(action_field.get("winner_posterior", {}) or {}),
            "initiative": {
                "top_intent": str(initiative.get("top_intent") or ""),
                "suppression_reason": str(initiative.get("suppression_reason") or ""),
                "should_send": bool(initiative.get("should_send", False)),
            },
        }

    def prewarm_recent_views(self, *, limit: int = 20) -> dict[str, int]:
        effective_limit = max(0, min(int(limit), self._signal_view_cache_limit))
        if effective_limit == 0:
            self._signal_view_cache = {}
            return {"round_count": 0}
        rounds = self.recent_rounds(limit=effective_limit)
        self._signal_view_cache = {
            int(payload.get("round_id", 0) or 0): self._round_signal_view(payload)
            for payload in rounds
            if int(payload.get("round_id", 0) or 0) > 0
        }
        return {"round_count": len(self._signal_view_cache)}

    def _load_round_records(self) -> list[dict]:
        parquet_rows = self._raw_round_payloads_from_parquet()
        if parquet_rows:
            self._legacy_round_parquet_rewrite_pending = any(
                self._round_payload_requires_rewrite(payload) for payload in parquet_rows
            ) or "distribution_state_json" in self._dataset_columns(self.round_trace_dir)
            return [rewrite_round_payload(payload) for payload in parquet_rows]
        self._legacy_round_parquet_rewrite_pending = False
        return [
            rewrite_round_payload(ensure_recorded_fields(json.loads(path.read_text(encoding="utf-8")), recorded_at=_mtime_iso(path)))
            for path in sorted(self.rounds_dir.glob("round_*.json"))
        ]

    def _load_run_records(self) -> list[dict]:
        parquet_rows = self._load_payload_dataset(self.run_canonical_dir)
        if parquet_rows:
            return parquet_rows
        return [
            ensure_recorded_fields(json.loads(path.read_text(encoding="utf-8")), recorded_at=_mtime_iso(path))
            for path in sorted(self.runs_dir.glob("run_*.json"))
        ]

    def _payload_json(self, payload: dict) -> str:
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)

    def _row_signature(self, row: dict) -> str:
        return json.dumps(row, ensure_ascii=False, sort_keys=True)

    def _merge_cache_rows(self, persisted_rows: list[dict], pending_rows: list[dict]) -> list[dict]:
        rows = [copy.deepcopy(row) for row in persisted_rows]
        seen = {self._row_signature(row) for row in rows}
        for row in pending_rows:
            signature = self._row_signature(row)
            if signature in seen:
                continue
            rows.append(copy.deepcopy(row))
            seen.add(signature)
        return rows

    def _ensure_run_cache_loaded(self) -> None:
        if self._run_cache_loaded:
            return
        loaded = {str(payload["run_id"]): payload for payload in self._load_run_records()}
        loaded.update({run_id: copy.deepcopy(payload) for run_id, payload in self._run_cache.items()})
        self._run_cache = loaded
        self._run_cache_loaded = True

    def _ensure_step_cache_loaded(self) -> None:
        if self._step_cache_loaded:
            return
        persisted = self._load_payload_dataset(self.step_trace_dir) or self._read_jsonl(self.step_jsonl_path)
        self._step_cache = self._merge_cache_rows(persisted, self._step_cache)
        self._step_cache_loaded = True

    def _ensure_tool_cache_loaded(self) -> None:
        if self._tool_cache_loaded:
            return
        persisted = self._load_payload_dataset(self.tool_trace_dir) or self._read_jsonl(self.tool_jsonl_path)
        self._tool_cache = self._merge_cache_rows(persisted, self._tool_cache)
        self._tool_cache_loaded = True

    def _ensure_skill_cache_loaded(self) -> None:
        if self._skill_cache_loaded:
            return
        persisted = self._load_payload_dataset(self.skill_canonical_dir) or self._read_jsonl(self.skill_jsonl_path)
        self._skill_cache = self._merge_cache_rows(persisted, self._skill_cache)
        self._skill_cache_loaded = True

    def _ensure_command_cache_loaded(self) -> None:
        if self._command_cache_loaded:
            return
        persisted = self._load_payload_dataset(self.command_canonical_dir) or self._read_jsonl(self.commands_jsonl_path)
        self._command_cache = self._merge_cache_rows(persisted, self._command_cache)
        self._command_cache_loaded = True

    def _default_command_subjectivity_totals(self) -> dict[str, object]:
        return {
            "summary_version": COMMAND_SUBJECTIVITY_TOTALS_VERSION,
            "total_commands": 0,
            "boundary_violation_count": 0,
            "external_count": 0,
            "internal_count": 0,
            "last_recorded_at": "",
        }

    def _normalize_command_subjectivity_totals(self, payload: dict[str, object] | None) -> dict[str, object]:
        normalized = dict(self._default_command_subjectivity_totals())
        if isinstance(payload, dict):
            normalized["summary_version"] = int(payload.get("summary_version", COMMAND_SUBJECTIVITY_TOTALS_VERSION) or 0)
            normalized["total_commands"] = int(payload.get("total_commands", 0) or 0)
            normalized["boundary_violation_count"] = int(payload.get("boundary_violation_count", 0) or 0)
            normalized["external_count"] = int(payload.get("external_count", 0) or 0)
            normalized["internal_count"] = int(payload.get("internal_count", 0) or 0)
            normalized["last_recorded_at"] = str(payload.get("last_recorded_at", "") or "")
        return normalized

    def _read_command_subjectivity_totals_file(self) -> dict[str, object] | None:
        if not self.command_subjectivity_totals_path.exists():
            return None
        try:
            return self._normalize_command_subjectivity_totals(
                json.loads(self.command_subjectivity_totals_path.read_text(encoding="utf-8"))
            )
        except (OSError, json.JSONDecodeError, ValueError, TypeError):
            return None

    def _write_command_subjectivity_totals(self, payload: dict[str, object]) -> None:
        normalized = self._normalize_command_subjectivity_totals(payload)
        self.command_subjectivity_totals_path.write_text(
            json.dumps(normalized, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._command_subjectivity_totals_cache = normalized

    def _command_subjectivity_totals_stale(self) -> bool:
        if not self.command_subjectivity_totals_path.exists():
            return True
        try:
            summary_mtime = self.command_subjectivity_totals_path.stat().st_mtime
            if self.commands_jsonl_path.exists() and self.commands_jsonl_path.stat().st_mtime > summary_mtime:
                return True
        except OSError:
            return True
        return False

    def _compute_command_subjectivity_totals_from_rows(self, rows: list[dict]) -> dict[str, object]:
        totals = self._default_command_subjectivity_totals()
        totals["total_commands"] = len(rows)
        totals["boundary_violation_count"] = sum(1 for row in rows if str(row.get("violation_code") or "").strip())
        totals["external_count"] = sum(1 for row in rows if row.get("cause_type") == "external_stimulus")
        totals["internal_count"] = sum(1 for row in rows if row.get("cause_type") == "endogenous")
        totals["last_recorded_at"] = str(rows[-1].get("recorded_at", "") or "") if rows else ""
        return totals

    def _backfill_command_subjectivity_totals(self) -> dict[str, object]:
        aggregate_rows = read_dataset_rows(
            self.command_trace_dir,
            """
            select
              count(*) as total_commands,
              sum(case when coalesce(violation_code, '') <> '' then 1 else 0 end) as boundary_violation_count,
              sum(case when cause_type = 'external_stimulus' then 1 else 0 end) as external_count,
              sum(case when cause_type = 'endogenous' then 1 else 0 end) as internal_count,
              max(recorded_at) as last_recorded_at
            from read_parquet(?)
            """,
        )
        if aggregate_rows:
            totals = self._normalize_command_subjectivity_totals(aggregate_rows[0])
        else:
            self._ensure_command_cache_loaded()
            totals = self._compute_command_subjectivity_totals_from_rows(self._command_cache)
        self._write_command_subjectivity_totals(totals)
        return dict(totals)

    def command_subjectivity_totals(self) -> dict[str, object]:
        cached = self._command_subjectivity_totals_cache
        if cached is not None and not self._command_subjectivity_totals_stale():
            return dict(cached)
        if not self._command_subjectivity_totals_stale():
            payload = self._read_command_subjectivity_totals_file()
            if payload is not None:
                self._command_subjectivity_totals_cache = payload
                return dict(payload)
        return self._backfill_command_subjectivity_totals()

    def _ensure_repair_cache_loaded(self) -> None:
        if self._repair_cache_loaded:
            return
        persisted = self._load_payload_dataset(self.repair_ledger_dir) or self._read_jsonl(self.repair_jsonl_path)
        self._repair_cache = self._merge_cache_rows(persisted, self._repair_cache)
        self._repair_cache_loaded = True

    def _ensure_round_summary_loaded(self) -> None:
        if self._round_summary_loaded:
            return
        persisted = self._load_round_summary_records()
        if persisted and self._round_summary_refresh_required(persisted):
            persisted = self._backfill_round_summary_records(force=True)
            self._round_metrics_loaded = False
            self._round_metrics_cache = []
        if not persisted:
            persisted = self._backfill_round_summary_records()
        self._round_summary_cache = self._merge_cache_rows(persisted, self._round_summary_cache)
        self._round_summary_loaded = True

    def _round_summary_refresh_required(self, rows: list[dict]) -> bool:
        return any(int(row.get("summary_version", 0) or 0) < ROUND_SUMMARY_VERSION for row in rows)

    def _default_round_subjectivity_totals(self) -> dict[str, object]:
        return {
            "summary_version": ROUND_SUBJECTIVITY_TOTALS_VERSION,
            "total_rounds": 0,
            "external_round_count": 0,
            "endogenous_round_count": 0,
            "last_recorded_at": "",
        }

    def _normalize_round_subjectivity_totals(self, payload: dict[str, object] | None) -> dict[str, object]:
        normalized = dict(self._default_round_subjectivity_totals())
        if isinstance(payload, dict):
            normalized["summary_version"] = int(payload.get("summary_version", ROUND_SUBJECTIVITY_TOTALS_VERSION) or 0)
            normalized["total_rounds"] = int(payload.get("total_rounds", 0) or 0)
            normalized["external_round_count"] = int(payload.get("external_round_count", 0) or 0)
            normalized["endogenous_round_count"] = int(payload.get("endogenous_round_count", 0) or 0)
            normalized["last_recorded_at"] = str(payload.get("last_recorded_at", "") or "")
        return normalized

    def _read_round_subjectivity_totals_file(self) -> dict[str, object] | None:
        if not self.round_subjectivity_totals_path.exists():
            return None
        try:
            return self._normalize_round_subjectivity_totals(
                json.loads(self.round_subjectivity_totals_path.read_text(encoding="utf-8"))
            )
        except (OSError, json.JSONDecodeError, ValueError, TypeError):
            return None

    def _write_round_subjectivity_totals(self, payload: dict[str, object]) -> None:
        normalized = self._normalize_round_subjectivity_totals(payload)
        self.round_subjectivity_totals_path.write_text(
            json.dumps(normalized, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._round_subjectivity_totals_cache = normalized

    def _round_subjectivity_totals_stale(self) -> bool:
        if not self.round_subjectivity_totals_path.exists():
            return True
        try:
            summary_mtime = self.round_subjectivity_totals_path.stat().st_mtime
            if self.rounds_jsonl_path.exists() and self.rounds_jsonl_path.stat().st_mtime > summary_mtime:
                return True
        except OSError:
            return True
        return False

    def _compute_round_subjectivity_totals_from_rows(self, rows: list[dict]) -> dict[str, object]:
        totals = self._default_round_subjectivity_totals()
        totals["total_rounds"] = len(rows)
        totals["external_round_count"] = sum(
            1 for row in rows if row.get("cause_type", "external_stimulus") == "external_stimulus"
        )
        totals["endogenous_round_count"] = sum(1 for row in rows if row.get("cause_type") == "endogenous")
        totals["last_recorded_at"] = str(rows[-1].get("recorded_at", "") or "") if rows else ""
        return totals

    def _backfill_round_subjectivity_totals(self) -> dict[str, object]:
        rows = self._load_round_summary_records()
        if not rows:
            self._ensure_round_summary_loaded()
            rows = self._round_summary_cache
        totals = self._compute_round_subjectivity_totals_from_rows(rows)
        self._write_round_subjectivity_totals(totals)
        return dict(totals)

    def round_subjectivity_totals(self) -> dict[str, object]:
        cached = self._round_subjectivity_totals_cache
        if cached is not None and not self._round_subjectivity_totals_stale():
            return dict(cached)
        if not self._round_subjectivity_totals_stale():
            payload = self._read_round_subjectivity_totals_file()
            if payload is not None:
                self._round_subjectivity_totals_cache = payload
                return dict(payload)
        return self._backfill_round_subjectivity_totals()

    def _probability_field_payload(self, payload: dict) -> dict:
        probability_field = canonical_probability_field_payload(payload)
        return probability_field if isinstance(probability_field, dict) else {}

    def _conflict_arbitration_payload(self, payload: dict) -> dict:
        direct = dict(payload.get("conflict_arbitration", {}) or {})
        probability_field = self._probability_field_payload(payload)
        action_layer = probability_field.get("action", {}) if isinstance(probability_field, dict) else {}
        audit_rows = action_layer.get("contribution_audit", []) if isinstance(action_layer, dict) else []
        conflict_row = next(
            (
                row
                for row in audit_rows
                if isinstance(row, dict) and row.get("module_name") == "ConflictMonitorAgent"
            ),
            {},
        )
        posterior = dict(direct.get("winner_peak_posterior", {}) or conflict_row.get("posterior", {}) or {})
        compromise_template_prior = dict(
            direct.get("compromise_template_prior", {}) or conflict_row.get("compromise_template_prior", {}) or {}
        )
        dependency_trace = [str(item) for item in list(conflict_row.get("dependency_trace", []) or [])]
        payload_summary = dict(direct)
        if not payload_summary.get("winning_priority"):
            for item in dependency_trace:
                if item.startswith("winning_priority:"):
                    payload_summary["winning_priority"] = item.split(":", 1)[1].strip()
                    break
        if not payload_summary.get("winning_priority"):
            for row in payload.get("gate_decisions", []):
                if not isinstance(row, dict) or row.get("stage") != "conflict":
                    continue
                winning_priority = str(row.get("winning_priority") or "").strip()
                if winning_priority:
                    payload_summary["winning_priority"] = winning_priority
                    break
        if posterior and not payload_summary.get("winner_peak_posterior"):
            payload_summary["winner_peak_posterior"] = posterior
        if posterior and not payload_summary.get("winner_peak"):
            payload_summary["winner_peak"] = max(posterior, key=posterior.get)
        if compromise_template_prior and not payload_summary.get("compromise_template_prior"):
            payload_summary["compromise_template_prior"] = compromise_template_prior
        if not payload_summary.get("compromise_template_prior"):
            for row in payload.get("gate_decisions", []):
                if not isinstance(row, dict):
                    continue
                template = str(row.get("template") or "").strip()
                if template:
                    payload_summary["compromise_template_prior"] = {template: 1.0}
                    break
        if not payload_summary.get("peak_clusters") and conflict_row.get("peak_clusters"):
            payload_summary["peak_clusters"] = list(conflict_row.get("peak_clusters", []) or [])
        if not payload_summary.get("hard_masked_targets") and conflict_row.get("hard_masked_targets"):
            payload_summary["hard_masked_targets"] = list(conflict_row.get("hard_masked_targets", []) or [])
        if not payload_summary.get("trace_reason") and conflict_row.get("trace_reason"):
            payload_summary["trace_reason"] = str(conflict_row.get("trace_reason", ""))
        return payload_summary

    def _load_payload_dataset(self, dataset_dir: Path) -> list[dict]:
        rows = read_dataset_rows(dataset_dir, "select payload_json from read_parquet(?)")
        return [ensure_recorded_fields(json.loads(row["payload_json"])) for row in rows]

    def _load_round_summary_records(self) -> list[dict]:
        rows = read_dataset_rows(
            self.round_summary_dir,
            "select payload_json from read_parquet(?) order by round_id, recorded_at",
        )
        return [ensure_recorded_fields(json.loads(row["payload_json"])) for row in rows]

    def _metrics_round_payload(self, summary: dict) -> dict:
        state_snapshot = dict(summary.get("state_snapshot", {}) or {})
        render_plan = dict(summary.get("render_plan", {}) or {})
        identity_context = dict(render_plan.get("identity_context", {}) or {})
        expression = dict(render_plan.get("expression", {}) or {})
        authenticity = dict(summary.get("authenticity", {}) or {})
        identity_evolution = dict(summary.get("identity_evolution", {}) or {})
        vitality_snapshot = dict(summary.get("vitality_snapshot", {}) or {})
        conflict_arbitration = dict(summary.get("conflict_arbitration", {}) or {})
        motivation_pool = dict(summary.get("motivation_pool", {}) or {})
        latest_trigger = dict(dict(summary.get("endogenous_tick_reason", {}) or {}).get("latest_trigger", {}) or {})
        return {
            "round_id": int(summary.get("round_id", 0) or 0),
            "recorded_at": str(summary.get("recorded_at", "") or ""),
            "scenario": str(summary.get("scenario", "") or ""),
            "mode": str(summary.get("mode", "") or ""),
            "sampled_action": str(summary.get("sampled_action", "") or ""),
            "cause_type": str(summary.get("cause_type", "external_stimulus") or "external_stimulus"),
            "top_drivers": copy.deepcopy(list(summary.get("top_drivers", []) or [])),
            "state_snapshot": {
                "budget_remaining": float(state_snapshot.get("budget_remaining", 0.0) or 0.0),
                "safe_mode": bool(state_snapshot.get("safe_mode", False)),
                "focus_lock_count": float(state_snapshot.get("focus_lock_count", 0.0) or 0.0),
                "mood": float(state_snapshot.get("mood", 0.0) or 0.0),
                "conflict_learning_state": copy.deepcopy(dict(state_snapshot.get("conflict_learning_state", {}) or {})),
            },
            "rendered_expression": {
                "text": str(dict(summary.get("rendered_expression", {}) or {}).get("text", "") or ""),
            },
            "render_plan": {
                "identity_context": {
                    "query_kind": str(identity_context.get("query_kind", "") or ""),
                    "query_intent": str(identity_context.get("query_intent", "") or ""),
                    "display_label": str(identity_context.get("display_label", "") or ""),
                    "disclosure_detail": str(identity_context.get("disclosure_detail", "") or ""),
                    "disclosure_intent": str(identity_context.get("disclosure_intent", "") or ""),
                },
                "expression": {
                    "repair_tendency": float(expression.get("repair_tendency", 0.0) or 0.0),
                    "warmth_level": float(expression.get("warmth_level", 0.0) or 0.0),
                    "directness_level": float(expression.get("directness_level", 0.0) or 0.0),
                },
            },
            "authenticity": {
                "violation_types": copy.deepcopy(list(authenticity.get("violation_types", []) or [])),
                "self_grounding_score": float(authenticity.get("self_grounding_score", 1.0) or 1.0),
                "provider_leak_detected": bool(authenticity.get("provider_leak_detected", False)),
                "false_self_claim_detected": bool(authenticity.get("false_self_claim_detected", False)),
                "provider_leak_penalty": float(authenticity.get("provider_leak_penalty", 0.0) or 0.0),
                "false_self_claim_penalty": float(authenticity.get("false_self_claim_penalty", 0.0) or 0.0),
                "guard_action": str(authenticity.get("guard_action", "pass") or "pass"),
                "state_sources": copy.deepcopy(list(authenticity.get("state_sources", []) or [])),
            },
            "identity_evolution": {
                "rename_event": bool(identity_evolution.get("rename_event", False)),
                "rename_reason": str(identity_evolution.get("rename_reason", "") or ""),
                "identity_shaping_sources": copy.deepcopy(list(identity_evolution.get("identity_shaping_sources", []) or [])),
            },
            "vitality_snapshot": {
                "cue": str(vitality_snapshot.get("cue", "") or ""),
                "cue_present": bool(vitality_snapshot.get("cue_present", False)),
                "memory_activation": float(vitality_snapshot.get("memory_activation", 0.0) or 0.0),
                "memory_interference": float(vitality_snapshot.get("memory_interference", 0.0) or 0.0),
                "detail_available": bool(vitality_snapshot.get("detail_available", False)),
                "affect_residue": float(vitality_snapshot.get("affect_residue", 0.0) or 0.0),
                "trigger_valence": float(vitality_snapshot.get("trigger_valence", 0.0) or 0.0),
                "habit_takeover": bool(vitality_snapshot.get("habit_takeover", False)),
                "habit_readiness": float(vitality_snapshot.get("habit_readiness", 0.0) or 0.0),
                "relationship_closeness": float(vitality_snapshot.get("relationship_closeness", 0.5) or 0.5),
                "relationship_drift": float(vitality_snapshot.get("relationship_drift", 0.0) or 0.0),
                "resource_scarcity": float(vitality_snapshot.get("resource_scarcity", 0.0) or 0.0),
            },
            "vitality_events": copy.deepcopy(list(summary.get("vitality_events", []) or [])),
            "gate_decisions": copy.deepcopy(list(summary.get("gate_decisions", []) or [])),
            "conflict_arbitration": {
                "total_score": float(conflict_arbitration.get("total_score", 0.0) or 0.0),
                "components": copy.deepcopy(dict(conflict_arbitration.get("components", {}) or {})),
                "critical_conflict": bool(conflict_arbitration.get("critical_conflict", False)),
                "critical_conflict_streak": int(conflict_arbitration.get("critical_conflict_streak", 0) or 0),
                "circuit_breaker": copy.deepcopy(dict(conflict_arbitration.get("circuit_breaker", {}) or {})),
                "winning_priority": str(conflict_arbitration.get("winning_priority", "") or ""),
                "compromise": copy.deepcopy(dict(conflict_arbitration.get("compromise", {}) or {})),
                "repair_mode": str(conflict_arbitration.get("repair_mode", "") or ""),
                "repair_state_snapshot": copy.deepcopy(dict(conflict_arbitration.get("repair_state_snapshot", {}) or {})),
                "post_error_adjustment": copy.deepcopy(dict(conflict_arbitration.get("post_error_adjustment", {}) or {})),
                "repair_ledger_summary": copy.deepcopy(dict(conflict_arbitration.get("repair_ledger_summary", {}) or {})),
                "conflict_safe_mode_owned": bool(conflict_arbitration.get("conflict_safe_mode_owned", False)),
            },
            "motivation_pool": {
                "active_motivations": copy.deepcopy(list(motivation_pool.get("active_motivations", []) or [])),
                "endogenous_activation_score": float(motivation_pool.get("endogenous_activation_score", 0.0) or 0.0),
            },
            "endogenous_tick_reason": {
                "latest_trigger": {
                    "trigger_type": str(latest_trigger.get("trigger_type", "") or ""),
                }
            },
        }

    def _load_round_metric_records(self) -> list[dict]:
        rows = read_dataset_rows(
            self.round_summary_dir,
            """
            select
              round_id,
              recorded_at,
              json_extract_string(payload_json, '$.scenario') as scenario,
              json_extract_string(payload_json, '$.mode') as mode,
              json_extract_string(payload_json, '$.sampled_action') as sampled_action,
              coalesce(json_extract_string(payload_json, '$.cause_type'), 'external_stimulus') as cause_type,
              json_extract(payload_json, '$.top_drivers') as top_drivers_json,
              try_cast(json_extract_string(payload_json, '$.state_snapshot.budget_remaining') as double) as state_budget_remaining,
              coalesce(try_cast(json_extract_string(payload_json, '$.state_snapshot.safe_mode') as boolean), false) as state_safe_mode,
              try_cast(json_extract_string(payload_json, '$.state_snapshot.focus_lock_count') as double) as state_focus_lock_count,
              try_cast(json_extract_string(payload_json, '$.state_snapshot.mood') as double) as state_mood,
              json_extract(payload_json, '$.state_snapshot.conflict_learning_state') as state_conflict_learning_json,
              json_extract_string(payload_json, '$.rendered_expression.text') as rendered_text,
              json_extract_string(payload_json, '$.render_plan.identity_context.query_kind') as query_kind,
              json_extract_string(payload_json, '$.render_plan.identity_context.query_intent') as query_intent,
              json_extract_string(payload_json, '$.render_plan.identity_context.display_label') as display_label,
              json_extract_string(payload_json, '$.render_plan.identity_context.disclosure_detail') as disclosure_detail,
              json_extract_string(payload_json, '$.render_plan.identity_context.disclosure_intent') as disclosure_intent,
              try_cast(json_extract_string(payload_json, '$.render_plan.expression.repair_tendency') as double) as expression_repair_tendency,
              try_cast(json_extract_string(payload_json, '$.render_plan.expression.warmth_level') as double) as expression_warmth_level,
              try_cast(json_extract_string(payload_json, '$.render_plan.expression.directness_level') as double) as expression_directness_level,
              json_extract(payload_json, '$.authenticity.violation_types') as authenticity_violation_types_json,
              try_cast(json_extract_string(payload_json, '$.authenticity.self_grounding_score') as double) as authenticity_self_grounding_score,
              coalesce(try_cast(json_extract_string(payload_json, '$.authenticity.provider_leak_detected') as boolean), false) as authenticity_provider_leak_detected,
              coalesce(try_cast(json_extract_string(payload_json, '$.authenticity.false_self_claim_detected') as boolean), false) as authenticity_false_self_claim_detected,
              try_cast(json_extract_string(payload_json, '$.authenticity.provider_leak_penalty') as double) as authenticity_provider_leak_penalty,
              try_cast(json_extract_string(payload_json, '$.authenticity.false_self_claim_penalty') as double) as authenticity_false_self_claim_penalty,
              json_extract_string(payload_json, '$.authenticity.guard_action') as authenticity_guard_action,
              json_extract(payload_json, '$.authenticity.state_sources') as authenticity_state_sources_json,
              coalesce(try_cast(json_extract_string(payload_json, '$.identity_evolution.rename_event') as boolean), false) as identity_rename_event,
              json_extract_string(payload_json, '$.identity_evolution.rename_reason') as identity_rename_reason,
              json_extract(payload_json, '$.identity_evolution.identity_shaping_sources') as identity_shaping_sources_json,
              json_extract_string(payload_json, '$.vitality_snapshot.cue') as vitality_cue,
              coalesce(try_cast(json_extract_string(payload_json, '$.vitality_snapshot.cue_present') as boolean), false) as vitality_cue_present,
              try_cast(json_extract_string(payload_json, '$.vitality_snapshot.memory_activation') as double) as vitality_memory_activation,
              try_cast(json_extract_string(payload_json, '$.vitality_snapshot.memory_interference') as double) as vitality_memory_interference,
              coalesce(try_cast(json_extract_string(payload_json, '$.vitality_snapshot.detail_available') as boolean), false) as vitality_detail_available,
              try_cast(json_extract_string(payload_json, '$.vitality_snapshot.affect_residue') as double) as vitality_affect_residue,
              try_cast(json_extract_string(payload_json, '$.vitality_snapshot.trigger_valence') as double) as vitality_trigger_valence,
              coalesce(try_cast(json_extract_string(payload_json, '$.vitality_snapshot.habit_takeover') as boolean), false) as vitality_habit_takeover,
              try_cast(json_extract_string(payload_json, '$.vitality_snapshot.habit_readiness') as double) as vitality_habit_readiness,
              try_cast(json_extract_string(payload_json, '$.vitality_snapshot.relationship_closeness') as double) as vitality_relationship_closeness,
              try_cast(json_extract_string(payload_json, '$.vitality_snapshot.relationship_drift') as double) as vitality_relationship_drift,
              try_cast(json_extract_string(payload_json, '$.vitality_snapshot.resource_scarcity') as double) as vitality_resource_scarcity,
              json_extract(payload_json, '$.vitality_events') as vitality_events_json,
              json_extract_string(payload_json, '$.gate_decisions[0].stage') as forced_stage,
              try_cast(json_extract_string(payload_json, '$.conflict_arbitration.total_score') as double) as conflict_total_score,
              json_extract(payload_json, '$.conflict_arbitration.components') as conflict_components_json,
              coalesce(try_cast(json_extract_string(payload_json, '$.conflict_arbitration.critical_conflict') as boolean), false) as conflict_critical_conflict,
              try_cast(json_extract_string(payload_json, '$.conflict_arbitration.critical_conflict_streak') as bigint) as conflict_critical_conflict_streak,
              json_extract(payload_json, '$.conflict_arbitration.circuit_breaker') as conflict_circuit_breaker_json,
              json_extract_string(payload_json, '$.conflict_arbitration.winning_priority') as conflict_winning_priority,
              json_extract_string(payload_json, '$.conflict_arbitration.compromise.template') as conflict_compromise_template,
              json_extract_string(payload_json, '$.conflict_arbitration.repair_mode') as conflict_repair_mode,
              json_extract_string(payload_json, '$.conflict_arbitration.repair_state_snapshot.stage') as conflict_repair_stage,
              json_extract(payload_json, '$.conflict_arbitration.post_error_adjustment') as conflict_post_error_adjustment_json,
              try_cast(json_extract_string(payload_json, '$.conflict_arbitration.repair_ledger_summary.entries') as bigint) as conflict_repair_entries,
              json_extract_string(payload_json, '$.conflict_arbitration.repair_ledger_summary.latest_reason') as conflict_repair_latest_reason,
              coalesce(try_cast(json_extract_string(payload_json, '$.conflict_arbitration.conflict_safe_mode_owned') as boolean), false) as conflict_safe_mode_owned,
              json_extract(payload_json, '$.motivation_pool.active_motivations') as motivation_active_motivations_json,
              try_cast(json_extract_string(payload_json, '$.motivation_pool.endogenous_activation_score') as double) as motivation_endogenous_activation_score,
              json_extract_string(payload_json, '$.endogenous_tick_reason.latest_trigger.trigger_type') as latest_trigger_type
            from read_parquet(?)
            order by round_id, recorded_at
            """,
        )
        if not rows:
            return []
        projected: list[dict] = []
        for row in rows:
            conflict_learning_state = _parse_json_cell(row.get("state_conflict_learning_json"), {})
            projected.append(
                {
                    "round_id": int(row.get("round_id", 0) or 0),
                    "recorded_at": str(row.get("recorded_at", "") or ""),
                    "scenario": str(row.get("scenario", "") or ""),
                    "mode": str(row.get("mode", "") or ""),
                    "sampled_action": str(row.get("sampled_action", "") or ""),
                    "cause_type": str(row.get("cause_type", "external_stimulus") or "external_stimulus"),
                    "top_drivers": _parse_json_cell(row.get("top_drivers_json"), []),
                    "state_snapshot": {
                        "budget_remaining": float(row.get("state_budget_remaining", 0.0) or 0.0),
                        "safe_mode": bool(row.get("state_safe_mode", False)),
                        "focus_lock_count": float(row.get("state_focus_lock_count", 0.0) or 0.0),
                        "mood": float(row.get("state_mood", 0.0) or 0.0),
                        "conflict_learning_state": {
                            "adjustment_reasons": dict(_parse_json_cell(dict(conflict_learning_state).get("adjustment_reasons"), {})),
                            "last_learning_signal": dict(_parse_json_cell(dict(conflict_learning_state).get("last_learning_signal"), {})),
                        },
                    },
                    "rendered_expression": {
                        "text": str(row.get("rendered_text", "") or ""),
                    },
                    "render_plan": {
                        "identity_context": {
                            "query_kind": str(row.get("query_kind", "") or ""),
                            "query_intent": str(row.get("query_intent", "") or ""),
                            "display_label": str(row.get("display_label", "") or ""),
                            "disclosure_detail": str(row.get("disclosure_detail", "") or ""),
                            "disclosure_intent": str(row.get("disclosure_intent", "") or ""),
                        },
                        "expression": {
                            "repair_tendency": float(row.get("expression_repair_tendency", 0.0) or 0.0),
                            "warmth_level": float(row.get("expression_warmth_level", 0.0) or 0.0),
                            "directness_level": float(row.get("expression_directness_level", 0.0) or 0.0),
                        },
                    },
                    "authenticity": {
                        "violation_types": _parse_json_cell(row.get("authenticity_violation_types_json"), []),
                        "self_grounding_score": float(row.get("authenticity_self_grounding_score", 1.0) or 1.0),
                        "provider_leak_detected": bool(row.get("authenticity_provider_leak_detected", False)),
                        "false_self_claim_detected": bool(row.get("authenticity_false_self_claim_detected", False)),
                        "provider_leak_penalty": float(row.get("authenticity_provider_leak_penalty", 0.0) or 0.0),
                        "false_self_claim_penalty": float(row.get("authenticity_false_self_claim_penalty", 0.0) or 0.0),
                        "guard_action": str(row.get("authenticity_guard_action", "pass") or "pass"),
                        "state_sources": _parse_json_cell(row.get("authenticity_state_sources_json"), []),
                    },
                    "identity_evolution": {
                        "rename_event": bool(row.get("identity_rename_event", False)),
                        "rename_reason": str(row.get("identity_rename_reason", "") or ""),
                        "identity_shaping_sources": _parse_json_cell(row.get("identity_shaping_sources_json"), []),
                    },
                    "vitality_snapshot": {
                        "cue": str(row.get("vitality_cue", "") or ""),
                        "cue_present": bool(row.get("vitality_cue_present", False)),
                        "memory_activation": float(row.get("vitality_memory_activation", 0.0) or 0.0),
                        "memory_interference": float(row.get("vitality_memory_interference", 0.0) or 0.0),
                        "detail_available": bool(row.get("vitality_detail_available", False)),
                        "affect_residue": float(row.get("vitality_affect_residue", 0.0) or 0.0),
                        "trigger_valence": float(row.get("vitality_trigger_valence", 0.0) or 0.0),
                        "habit_takeover": bool(row.get("vitality_habit_takeover", False)),
                        "habit_readiness": float(row.get("vitality_habit_readiness", 0.0) or 0.0),
                        "relationship_closeness": float(row.get("vitality_relationship_closeness", 0.5) or 0.5),
                        "relationship_drift": float(row.get("vitality_relationship_drift", 0.0) or 0.0),
                        "resource_scarcity": float(row.get("vitality_resource_scarcity", 0.0) or 0.0),
                    },
                    "vitality_events": _parse_json_cell(row.get("vitality_events_json"), []),
                    "gate_decisions": [{"stage": "forced_mode_switch"}] if str(row.get("forced_stage", "") or "") == "forced_mode_switch" else [],
                    "conflict_arbitration": {
                        "total_score": float(row.get("conflict_total_score", 0.0) or 0.0),
                        "components": _parse_json_cell(row.get("conflict_components_json"), {}),
                        "critical_conflict": bool(row.get("conflict_critical_conflict", False)),
                        "critical_conflict_streak": int(row.get("conflict_critical_conflict_streak", 0) or 0),
                        "circuit_breaker": _parse_json_cell(row.get("conflict_circuit_breaker_json"), {}),
                        "winning_priority": str(row.get("conflict_winning_priority", "") or ""),
                        "compromise": {"template": str(row.get("conflict_compromise_template", "") or "")},
                        "repair_mode": str(row.get("conflict_repair_mode", "") or ""),
                        "repair_state_snapshot": {"stage": str(row.get("conflict_repair_stage", "idle") or "idle")},
                        "post_error_adjustment": _parse_json_cell(row.get("conflict_post_error_adjustment_json"), {}),
                        "repair_ledger_summary": {
                            "entries": int(row.get("conflict_repair_entries", 0) or 0),
                            "latest_reason": str(row.get("conflict_repair_latest_reason", "") or ""),
                        },
                        "conflict_safe_mode_owned": bool(row.get("conflict_safe_mode_owned", False)),
                    },
                    "motivation_pool": {
                        "active_motivations": _parse_json_cell(row.get("motivation_active_motivations_json"), []),
                        "endogenous_activation_score": float(row.get("motivation_endogenous_activation_score", 0.0) or 0.0),
                    },
                    "endogenous_tick_reason": {
                        "latest_trigger": {
                            "trigger_type": str(row.get("latest_trigger_type", "") or ""),
                        }
                    },
                }
            )
        return projected

    def _ensure_round_metrics_loaded(self) -> None:
        if self._round_metrics_loaded:
            return
        projected = self._load_round_metric_records()
        if not projected:
            self._ensure_round_summary_loaded()
            projected = [self._metrics_round_payload(summary) for summary in self._round_summary_cache]
        self._round_metrics_cache = projected
        self._round_metrics_loaded = True

    def metrics_round_rows_view(self) -> list[dict]:
        self._ensure_round_metrics_loaded()
        return self._round_metrics_cache

    def _append_dataset_rows(
        self,
        dataset_dir: Path,
        rows: list[dict],
        *,
        schema: dict[str, str],
        partition_keys: tuple[str, ...] = ("recorded_date",),
    ) -> None:
        append_dataset(dataset_dir, rows, schema=schema, partition_keys=partition_keys)

    def _rewrite_dataset_rows(
        self,
        dataset_dir: Path,
        rows: list[dict],
        *,
        schema: dict[str, str],
        partition_keys: tuple[str, ...] = ("recorded_date",),
    ) -> None:
        temp_dir = dataset_dir.with_name(f"{dataset_dir.name}.rewrite-{uuid4().hex}")
        shutil.rmtree(temp_dir, ignore_errors=True)
        temp_dir.mkdir(parents=True, exist_ok=True)
        if rows:
            append_dataset(temp_dir, rows, schema=schema, partition_keys=partition_keys)
        shutil.rmtree(dataset_dir, ignore_errors=True)
        temp_dir.replace(dataset_dir)

    def _dataset_columns(self, dataset_dir: Path) -> list[str]:
        if not dataset_dir.exists():
            return []
        conn = duckdb.connect()
        try:
            try:
                cursor = conn.execute(
                    "select * from read_parquet(?) limit 0",
                    [str(dataset_dir / "**" / "*.parquet")],
                )
            except duckdb.IOException as exc:
                if "No files found" in str(exc):
                    return []
                raise
            except duckdb.Error:
                return []
            return [item[0] for item in cursor.description]
        finally:
            conn.close()

    def _raw_round_payloads_from_parquet(self) -> list[dict]:
        rows = read_dataset_rows(
            self.round_canonical_dir,
            "select payload_json from read_parquet(?) order by round_id, recorded_at",
        )
        return [ensure_recorded_fields(json.loads(row["payload_json"])) for row in rows]

    def _round_payload_requires_rewrite(self, payload: dict) -> bool:
        if "distribution_state" in payload:
            return True
        probability_field = payload.get("probability_field")
        return probability_field is None and bool(legacy_probability_field_payload(payload))

    def rewrite_legacy_round_parquet_history(self, *, force: bool = False) -> dict[str, object]:
        raw_payloads = self._raw_round_payloads_from_parquet()
        legacy_payload_rows = sum(1 for payload in raw_payloads if self._round_payload_requires_rewrite(payload))
        removed_trace_columns = sorted(
            column
            for column in self._dataset_columns(self.round_trace_dir)
            if column == "distribution_state_json"
        )
        rewrite_needed = force or legacy_payload_rows > 0 or bool(removed_trace_columns)
        summary: dict[str, object] = {
            "rewritten": False,
            "round_count": len(raw_payloads),
            "legacy_payload_rows": legacy_payload_rows,
            "removed_trace_columns": removed_trace_columns,
            "round_trace_row_count": 0,
        }
        if not rewrite_needed:
            self._legacy_round_parquet_rewrite_pending = False
            return summary

        canonical_payloads = [rewrite_round_payload(payload) for payload in raw_payloads]
        round_trace_rows: list[dict] = []
        for payload in canonical_payloads:
            round_trace_rows.extend(self._round_trace_rows(payload))

        if raw_payloads or self.round_canonical_dir.exists():
            canonical_rows: list[dict] = []
            summary_rows: list[dict] = []
            for payload in canonical_payloads:
                canonical_rows.extend(self._round_canonical_rows(payload))
                summary_rows.extend(self._round_summary_rows(payload))
            self._rewrite_dataset_rows(
                self.round_canonical_dir,
                canonical_rows,
                schema=ROUND_CANONICAL_SCHEMA,
                partition_keys=("recorded_date", "round_id"),
            )
            self._rewrite_dataset_rows(
                self.round_summary_dir,
                summary_rows,
                schema=ROUND_SUMMARY_SCHEMA,
                partition_keys=("recorded_date", "round_id"),
            )
        if round_trace_rows or self.round_trace_dir.exists():
            self._rewrite_dataset_rows(self.round_trace_dir, round_trace_rows, schema=ROUND_TRACE_SCHEMA)

        if hasattr(self, "_round_cache"):
            self._round_cache = {int(payload["round_id"]): copy.deepcopy(payload) for payload in canonical_payloads}
            self._round_cache_complete = True
            self._round_count_cache_value = len(self._round_cache)
            self._round_count_cache_checked_at = time.monotonic()
        if hasattr(self, "_round_summary_cache"):
            self._round_summary_cache = [self._round_summary_payload(payload) for payload in canonical_payloads]
            self._round_summary_loaded = True

        self._legacy_round_parquet_rewrite_pending = False
        summary["rewritten"] = True
        summary["round_trace_row_count"] = len(round_trace_rows)
        return summary

    def _legacy_round_parquet_pending_rewrite(self) -> bool:
        if self._legacy_round_parquet_rewrite_pending is None:
            self._legacy_round_parquet_rewrite_pending = any(
                self._round_payload_requires_rewrite(payload) for payload in self._raw_round_payloads_from_parquet()
            ) or "distribution_state_json" in self._dataset_columns(self.round_trace_dir)
        return self._legacy_round_parquet_rewrite_pending

    def _round_canonical_rows(self, payload: dict) -> list[dict]:
        return [
            {
                "session_id": payload["session_id"],
                "recorded_at": payload["recorded_at"],
                "recorded_date": payload["recorded_date"],
                "round_id": payload["round_id"],
                "payload_json": self._payload_json(payload),
            }
        ]

    def _round_trace_rows(self, payload: dict) -> list[dict]:
        rows: list[dict] = []
        probability_field = self._probability_field_payload(payload)
        long_run_projection = payload.get("long_run_projection", {})
        long_run_online_prior = long_run_projection.get("online_prior", {}) if isinstance(long_run_projection, dict) else {}
        conflict_arbitration = self._conflict_arbitration_payload(payload)
        cross_layer_coupling_verdict = {
            "observed_pairs": [
                "->".join(
                    (
                        str(item.get("source_layer", "")),
                        str(item.get("target_layer", "")),
                        str(item.get("carrier_signal", "")),
                    )
                )
                for item in probability_field.get("couplings", [])
                if isinstance(item, dict)
            ],
            "illegal_pairs": [],
            "legal": True,
        }
        for summary in payload.get("proposal_summaries", []):
            delta_map = summary.get("delta_p", {}) or {summary.get("top_action") or "unknown": None}
            for action_name, delta_value in delta_map.items():
                rows.append(
                    {
                        "session_id": payload["session_id"],
                        "recorded_at": payload["recorded_at"],
                        "recorded_date": payload["recorded_date"],
                        "round_id": payload.get("round_id"),
                        "scenario": payload.get("scenario"),
                        "mode": payload.get("mode"),
                        "sampled_action": payload.get("sampled_action"),
                        "stage": summary.get("stage"),
                        "agent_name": summary.get("agent_name"),
                        "action": action_name,
                        "top_action": summary.get("top_action"),
                        "selected": summary.get("selected"),
                        "confidence": summary.get("confidence"),
                        "sigma_scale": summary.get("sigma_scale"),
                        "weight_applied": summary.get("weight_applied"),
                        "delta_p": delta_value,
                        "resample_count": payload.get("resample_count", 0),
                        "resample_idx": summary.get("resample_idx", 0),
                        "conflict_score": summary.get("conflict_score"),
                        "plausibility_fail_score": summary.get("plausibility_fail_score"),
                        "tags_json": json.dumps(summary.get("tags", []), ensure_ascii=False, sort_keys=True),
                        "probability_field_json": json.dumps(probability_field, ensure_ascii=False, sort_keys=True),
                        "token_state_json": json.dumps(probability_field.get("token_state", {}), ensure_ascii=False, sort_keys=True),
                        "cross_layer_coupling_verdict_json": json.dumps(cross_layer_coupling_verdict, ensure_ascii=False, sort_keys=True),
                        "renderer_decision_integrity_json": json.dumps(payload.get("renderer_decision_integrity", {}), ensure_ascii=False, sort_keys=True),
                        "memory_write_gate_json": json.dumps(payload.get("memory_write_gate", {}), ensure_ascii=False, sort_keys=True),
                        "conflict_arbitration_json": json.dumps(conflict_arbitration, ensure_ascii=False, sort_keys=True),
                        "state_snapshot_json": json.dumps(payload.get("state_snapshot", {}), ensure_ascii=False, sort_keys=True),
                        "render_plan_json": json.dumps(payload.get("render_plan", {}), ensure_ascii=False, sort_keys=True),
                        "gate_decisions_json": json.dumps(payload.get("gate_decisions", []), ensure_ascii=False, sort_keys=True),
                        "rendered_expression_json": json.dumps(payload.get("rendered_expression", {}), ensure_ascii=False, sort_keys=True),
                        "long_run_projection_json": json.dumps(long_run_projection, ensure_ascii=False, sort_keys=True),
                        "long_run_projection_online_prior_json": json.dumps(long_run_online_prior, ensure_ascii=False, sort_keys=True),
                        "motivation_pool_json": json.dumps(payload.get("motivation_pool", {}), ensure_ascii=False, sort_keys=True),
                        "motivation_feedback_json": json.dumps(payload.get("motivation_feedback", {}), ensure_ascii=False, sort_keys=True),
                        "endogenous_tick_reason_json": json.dumps(payload.get("endogenous_tick_reason", {}), ensure_ascii=False, sort_keys=True),
                        "endogenous_policy_shift_json": json.dumps(payload.get("endogenous_policy_shift", {}), ensure_ascii=False, sort_keys=True),
                    }
                )
        return rows

    def _round_summary_payload(self, payload: dict) -> dict:
        render_plan = dict(payload.get("render_plan", {}) or {})
        expression = dict(render_plan.get("expression", {}) or {})
        identity_context = dict(render_plan.get("identity_context", {}) or {})
        rendered_expression = dict(payload.get("rendered_expression", {}) or {})
        authenticity = dict(payload.get("authenticity", rendered_expression.get("authenticity", {})) or {})
        vitality = dict(payload.get("vitality_snapshot", {}) or {})
        state_snapshot = dict(payload.get("state_snapshot", {}) or {})
        conflict_learning = dict(state_snapshot.get("conflict_learning_state", {}) or {})
        motivation_pool = dict(payload.get("motivation_pool", {}) or {})
        endogenous_tick_reason = dict(payload.get("endogenous_tick_reason", {}) or {})
        latest_trigger = dict(endogenous_tick_reason.get("latest_trigger", {}) or {})
        conflict_arbitration = self._conflict_arbitration_payload(payload)
        repair_ledger_tail = list(conflict_arbitration.get("repair_ledger_tail", []) or [])
        latest_repair_entry = dict(repair_ledger_tail[-1]) if repair_ledger_tail else {}
        forced_mode_switch = any(
            isinstance(item, dict) and item.get("stage") == "forced_mode_switch"
            for item in list(payload.get("gate_decisions", []) or [])
        )
        top_drivers = []
        for driver in list(payload.get("top_drivers", []) or [])[:5]:
            if not isinstance(driver, dict):
                continue
            top_drivers.append(
                {
                    "agent_name": str(driver.get("agent_name") or ""),
                    "action_name": str(driver.get("action_name") or ""),
                    "score": float(driver.get("score", 0.0) or 0.0),
                }
            )
        summary = {
            "summary_version": ROUND_SUMMARY_VERSION,
            "session_id": payload.get("session_id"),
            "recorded_at": payload.get("recorded_at"),
            "recorded_date": payload.get("recorded_date"),
            "round_id": payload.get("round_id"),
            "scenario": payload.get("scenario"),
            "mode": payload.get("mode"),
            "sampled_action": payload.get("sampled_action"),
            "cause_type": payload.get("cause_type", "external_stimulus"),
            "top_drivers": top_drivers,
            "state_snapshot": {
                "budget_remaining": float(state_snapshot.get("budget_remaining", 0.0) or 0.0),
                "safe_mode": bool(state_snapshot.get("safe_mode", False)),
                "focus_lock_count": float(state_snapshot.get("focus_lock_count", 0.0) or 0.0),
                "mood": float(state_snapshot.get("mood", 0.0) or 0.0),
                "conflict_learning_state": {
                    "adjustment_reasons": dict(conflict_learning.get("adjustment_reasons", {}) or {}),
                    "last_learning_signal": dict(conflict_learning.get("last_learning_signal", {}) or {}),
                },
            },
            "rendered_expression": {
                "text": str(rendered_expression.get("text", "") or ""),
            },
            "render_plan": {
                "identity_context": {
                    "query_kind": str(identity_context.get("query_kind", "") or ""),
                    "query_intent": str(identity_context.get("query_intent", "") or ""),
                    "display_label": str(identity_context.get("display_label", "") or ""),
                    "disclosure_detail": str(identity_context.get("disclosure_detail", "") or ""),
                    "disclosure_intent": str(identity_context.get("disclosure_intent", "") or ""),
                },
                "expression": {
                    "repair_tendency": float(expression.get("repair_tendency", 0.0) or 0.0),
                    "warmth_level": float(expression.get("warmth_level", 0.0) or 0.0),
                    "directness_level": float(expression.get("directness_level", 0.0) or 0.0),
                },
            },
            "authenticity": {
                "violation_types": list(authenticity.get("violation_types", []) or []),
                "self_grounding_score": float(authenticity.get("self_grounding_score", 1.0) or 1.0),
                "provider_leak_detected": bool(authenticity.get("provider_leak_detected", False)),
                "false_self_claim_detected": bool(authenticity.get("false_self_claim_detected", False)),
                "provider_leak_penalty": float(authenticity.get("provider_leak_penalty", 0.0) or 0.0),
                "false_self_claim_penalty": float(authenticity.get("false_self_claim_penalty", 0.0) or 0.0),
                "guard_action": str(authenticity.get("guard_action", "pass") or "pass"),
                "state_sources": list(authenticity.get("state_sources", []) or []),
            },
            "identity_evolution": {
                "rename_event": bool(dict(payload.get("identity_evolution", {}) or {}).get("rename_event")),
                "rename_reason": str(dict(payload.get("identity_evolution", {}) or {}).get("rename_reason", "") or ""),
                "identity_shaping_sources": list(dict(payload.get("identity_evolution", {}) or {}).get("identity_shaping_sources", []) or []),
            },
            "vitality_snapshot": {
                "cue": str(vitality.get("cue", "") or ""),
                "cue_present": bool(vitality.get("cue_present", False)),
                "memory_activation": float(vitality.get("memory_activation", 0.0) or 0.0),
                "memory_interference": float(vitality.get("memory_interference", 0.0) or 0.0),
                "detail_available": bool(vitality.get("detail_available", False)),
                "affect_residue": float(vitality.get("affect_residue", 0.0) or 0.0),
                "trigger_valence": float(vitality.get("trigger_valence", 0.0) or 0.0),
                "habit_takeover": bool(vitality.get("habit_takeover", False)),
                "habit_readiness": float(vitality.get("habit_readiness", 0.0) or 0.0),
                "relationship_closeness": float(vitality.get("relationship_closeness", 0.5) or 0.5),
                "relationship_drift": float(vitality.get("relationship_drift", 0.0) or 0.0),
                "resource_scarcity": float(vitality.get("resource_scarcity", 0.0) or 0.0),
            },
            "vitality_events": [
                {
                    "non_interactive": bool(item.get("non_interactive", False)),
                    "shaping_detail": str(item.get("shaping_detail", "") or ""),
                }
                for item in list(payload.get("vitality_events", []) or [])
                if isinstance(item, dict) and (item.get("non_interactive") or item.get("shaping_detail"))
            ],
            "gate_decisions": [{"stage": "forced_mode_switch"}] if forced_mode_switch else [],
            "conflict_arbitration": {
                "total_score": float(
                    conflict_arbitration.get("total_score", conflict_arbitration.get("score", 0.0)) or 0.0
                ),
                "components": dict(conflict_arbitration.get("components", {}) or {}),
                "critical_conflict": bool(conflict_arbitration.get("critical_conflict", False)),
                "critical_conflict_streak": int(conflict_arbitration.get("critical_conflict_streak", 0) or 0),
                "circuit_breaker": dict(conflict_arbitration.get("circuit_breaker", {}) or {}),
                "winning_priority": str(conflict_arbitration.get("winning_priority", "") or ""),
                "compromise": {
                    "template": str(dict(conflict_arbitration.get("compromise", {}) or {}).get("template", "") or ""),
                },
                "repair_mode": str(conflict_arbitration.get("repair_mode", "") or ""),
                "repair_state_snapshot": {
                    "stage": str(dict(conflict_arbitration.get("repair_state_snapshot", {}) or {}).get("stage", "idle") or "idle"),
                },
                "post_error_adjustment": dict(conflict_arbitration.get("post_error_adjustment", {}) or {}),
                "repair_ledger_summary": {
                    "entries": len(repair_ledger_tail),
                    "latest_reason": str(latest_repair_entry.get("reason", "") or ""),
                },
                "conflict_safe_mode_owned": bool(conflict_arbitration.get("conflict_safe_mode_owned", False)),
            },
            "motivation_pool": {
                "active_motivations": list(motivation_pool.get("active_motivations", []) or []),
                "endogenous_activation_score": float(motivation_pool.get("endogenous_activation_score", 0.0) or 0.0),
            },
            "endogenous_tick_reason": {
                "latest_trigger": {
                    "trigger_type": str(latest_trigger.get("trigger_type", "") or ""),
                }
            },
        }
        return ensure_recorded_fields(
            summary,
            session_id=str(summary.get("session_id") or "legacy"),
            recorded_at=str(summary.get("recorded_at") or utc_now_iso()),
        )

    def _round_summary_rows(self, payload: dict) -> list[dict]:
        summary = self._round_summary_payload(payload)
        return [
            {
                "session_id": summary["session_id"],
                "recorded_at": summary["recorded_at"],
                "recorded_date": summary["recorded_date"],
                "round_id": summary["round_id"],
                "payload_json": self._payload_json(summary),
            }
        ]

    def _backfill_round_summary_records(self, *, force: bool = False) -> list[dict]:
        if not force and self.round_summary_dir.exists() and any(self.round_summary_dir.rglob("*.parquet")):
            return self._load_round_summary_records()
        source_rounds = self.list_rounds()
        if not source_rounds:
            return []
        summary_rows: list[dict] = []
        summary_payloads: list[dict] = []
        for payload in source_rounds:
            summary_payload = self._round_summary_payload(payload)
            summary_payloads.append(summary_payload)
            summary_rows.append(
                {
                    "session_id": summary_payload["session_id"],
                    "recorded_at": summary_payload["recorded_at"],
                    "recorded_date": summary_payload["recorded_date"],
                    "round_id": summary_payload["round_id"],
                    "payload_json": self._payload_json(summary_payload),
                }
            )
        self._rewrite_dataset_rows(
            self.round_summary_dir,
            summary_rows,
            schema=ROUND_SUMMARY_SCHEMA,
            partition_keys=("recorded_date", "round_id"),
        )
        return summary_payloads

    def _skill_rows(self, rows: list[dict]) -> tuple[list[dict], list[dict]]:
        canonical = []
        flat = []
        for row in rows:
            canonical.append(
                {
                    "session_id": row.get("session_id", "legacy"),
                    "recorded_at": row.get("recorded_at", utc_now_iso()),
                    "recorded_date": row.get("recorded_date", row.get("recorded_at", utc_now_iso())[:10]),
                    "round_id": row.get("round_id"),
                    "skill_name": row.get("skill_name"),
                    "payload_json": self._payload_json(row),
                }
            )
            flat.append(
                {
                    "session_id": row.get("session_id", "legacy"),
                    "recorded_at": row.get("recorded_at", utc_now_iso()),
                    "recorded_date": row.get("recorded_date", row.get("recorded_at", utc_now_iso())[:10]),
                    "round_id": row.get("round_id"),
                    "skill_name": row.get("skill_name"),
                    "owner_module": row.get("owner_module"),
                    "latency_ms": row.get("latency_ms"),
                    "cost_class": row.get("cost_class"),
                    "input_hash": row.get("input_hash"),
                    "output_hash": row.get("output_hash"),
                    "failure_policy_applied": row.get("failure_policy_applied"),
                    "degraded": row.get("degraded", False),
                    "seed_ref": row.get("seed_ref"),
                }
            )
        return canonical, flat

    def _migrate_legacy_if_needed(self) -> None:
        if self.migration_status_path.exists():
            return
        has_legacy = any(self.rounds_dir.glob("round_*.json")) or self.commands_path.exists() or self.runs_dir.exists()
        has_dataset = any(self.parquet_dir.rglob("*.parquet"))
        if not has_legacy or has_dataset:
            return
        try:
            for payload in self._load_round_records():
                self._append_dataset_rows(
                    self.round_canonical_dir,
                    self._round_canonical_rows(payload),
                    schema=ROUND_CANONICAL_SCHEMA,
                    partition_keys=("recorded_date", "round_id"),
                )
                self._append_dataset_rows(
                    self.round_summary_dir,
                    self._round_summary_rows(payload),
                    schema=ROUND_SUMMARY_SCHEMA,
                    partition_keys=("recorded_date", "round_id"),
                )
                round_trace_rows = self._round_trace_rows(payload)
                if round_trace_rows:
                    self._append_dataset_rows(self.round_trace_dir, round_trace_rows, schema=ROUND_TRACE_SCHEMA)
                skill_rows = payload.get("skill_traces", [])
                canonical, flat = self._skill_rows(skill_rows)
                if canonical:
                    self._append_dataset_rows(self.skill_canonical_dir, canonical, schema=SKILL_CANONICAL_SCHEMA)
                    self._append_dataset_rows(self.skill_trace_dir, flat, schema=SKILL_TRACE_SCHEMA)
            round_rows = self._load_round_summary_records()
            if round_rows:
                self._write_round_subjectivity_totals(self._compute_round_subjectivity_totals_from_rows(round_rows))
            for payload in self._load_run_records():
                self._append_dataset_rows(
                    self.run_canonical_dir,
                    [
                        {
                            "session_id": payload["session_id"],
                            "recorded_at": payload["recorded_at"],
                            "recorded_date": payload["recorded_date"],
                            "run_id": payload["run_id"],
                            "payload_json": self._payload_json(payload),
                        }
                    ],
                    schema=RUN_CANONICAL_SCHEMA,
                )
            for row in self._read_jsonl(self.step_jsonl_path):
                self._append_dataset_rows(
                    self.step_trace_dir,
                    [{"session_id": row["session_id"], "recorded_at": row["recorded_at"], "recorded_date": row["recorded_date"], "run_id": row.get("run_id"), "payload_json": self._payload_json(row)}],
                    schema=STEP_TOOL_SCHEMA,
                )
            for row in self._read_jsonl(self.tool_jsonl_path):
                self._append_dataset_rows(
                    self.tool_trace_dir,
                    [{"session_id": row["session_id"], "recorded_at": row["recorded_at"], "recorded_date": row["recorded_date"], "run_id": row.get("run_id"), "payload_json": self._payload_json(row)}],
                    schema=STEP_TOOL_SCHEMA,
                )
            for row in self._read_jsonl(self.commands_jsonl_path):
                self._append_dataset_rows(
                    self.command_canonical_dir,
                    [{"session_id": row["session_id"], "recorded_at": row["recorded_at"], "recorded_date": row["recorded_date"], "command": row.get("command"), "payload_json": self._payload_json(row)}],
                    schema=COMMAND_CANONICAL_SCHEMA,
                )
                self._append_dataset_rows(
                    self.command_trace_dir,
                    [{
                        "session_id": row["session_id"],
                        "recorded_at": row["recorded_at"],
                        "recorded_date": row["recorded_date"],
                        "command": row.get("command"),
                        "command_id": row.get("command_id"),
                        "canonical": row.get("canonical"),
                        "applied": row.get("applied"),
                        "scope": row.get("scope"),
                        "delta_json": json.dumps(row.get("delta", {}), ensure_ascii=False, sort_keys=True),
                        "ttl": row.get("ttl"),
                        "risk_note": row.get("risk_note"),
                        "rollback_hint": row.get("rollback_hint"),
                        "operator_level": row.get("operator_level"),
                        "rollback_available": row.get("rollback_available"),
                        "mutation_scope": row.get("mutation_scope"),
                        "parsed_args_json": json.dumps(row.get("parsed_args", {}), ensure_ascii=False, sort_keys=True),
                        "flags_json": json.dumps(row.get("flags", {}), ensure_ascii=False, sort_keys=True),
                        "snapshot_id": row.get("snapshot_id"),
                        "rollback_json": json.dumps(row.get("rollback", {}), ensure_ascii=False, sort_keys=True),
                        "before_state_hash": row.get("before_state_hash"),
                        "after_state_hash": row.get("after_state_hash"),
                        "subject_id": row.get("subject_id"),
                        "continuity_nonce": row.get("continuity_nonce"),
                        "cause_type": row.get("cause_type"),
                        "boundary_action": row.get("boundary_action"),
                        "violation_code": row.get("violation_code"),
                        "deprecation_warning": row.get("deprecation_warning"),
                    }],
                    schema=COMMAND_TRACE_SCHEMA,
                )
            command_rows = self._read_jsonl(self.commands_jsonl_path)
            if command_rows:
                self._write_command_subjectivity_totals(self._compute_command_subjectivity_totals_from_rows(command_rows))
            for row in self._read_jsonl(self.repair_jsonl_path):
                self._append_dataset_rows(
                    self.repair_ledger_dir,
                    [{
                        "session_id": row["session_id"],
                        "recorded_at": row["recorded_at"],
                        "recorded_date": row["recorded_date"],
                        "round_id": row.get("round_id", 0),
                        "reason": row.get("reason", ""),
                        "winning_priority": row.get("winning_priority"),
                        "template": row.get("template"),
                        "repair_stage_after": row.get("repair_stage_after"),
                        "conflict_score": row.get("conflict_score", 0.0),
                        "payload_json": self._payload_json(row),
                    }],
                    schema=REPAIR_LEDGER_SCHEMA,
                )
            self.migration_status_path.write_text(json.dumps({"status": "completed", "migrated_at": utc_now_iso()}, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            shutil.rmtree(self.parquet_dir, ignore_errors=True)
            self.parquet_dir.mkdir(parents=True, exist_ok=True)
            raise

    def write_round(self, trace: RoundTrace, *, sync: bool = False) -> None:
        payload = rewrite_round_payload(
            ensure_recorded_fields(
            to_dict(trace),
            session_id=trace.session_id,
            recorded_at=trace.recorded_at,
            )
        )
        skill_rows = list(payload.get("skill_traces", []))
        round_payload = dict(payload)
        round_payload.pop("skill_traces", None)
        summary_payload = self._round_summary_payload(round_payload)
        self._round_cache[int(round_payload["round_id"])] = round_payload
        self._signal_view_cache[int(round_payload["round_id"])] = self._round_signal_view(round_payload)
        if len(self._signal_view_cache) > self._signal_view_cache_limit:
            for stale_round_id in sorted(self._signal_view_cache)[:-self._signal_view_cache_limit]:
                self._signal_view_cache.pop(stale_round_id, None)
        self._round_summary_cache.append(summary_payload)
        self._round_metrics_cache.append(self._metrics_round_payload(summary_payload))
        existing_round_totals = self._round_subjectivity_totals_cache or self._read_round_subjectivity_totals_file()
        if existing_round_totals is None:
            if self._round_summary_loaded:
                existing_round_totals = self._compute_round_subjectivity_totals_from_rows(self._round_summary_cache[:-1])
            else:
                existing_round_totals = self._default_round_subjectivity_totals()
        next_round_totals = self._normalize_round_subjectivity_totals(existing_round_totals)
        next_round_totals["summary_version"] = ROUND_SUBJECTIVITY_TOTALS_VERSION
        next_round_totals["total_rounds"] = int(next_round_totals.get("total_rounds", 0) or 0) + 1
        if round_payload.get("cause_type", "external_stimulus") == "external_stimulus":
            next_round_totals["external_round_count"] = int(next_round_totals.get("external_round_count", 0) or 0) + 1
        if round_payload.get("cause_type") == "endogenous":
            next_round_totals["endogenous_round_count"] = int(next_round_totals.get("endogenous_round_count", 0) or 0) + 1
        next_round_totals["last_recorded_at"] = str(round_payload.get("recorded_at", "") or "")
        self._round_subjectivity_totals_cache = next_round_totals
        self._skill_cache.extend(skill_rows)
        if skill_rows:
            with self.skill_jsonl_path.open("a", encoding="utf-8") as handle:
                for row in skill_rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        def write() -> None:
            path = self.rounds_dir / f"round_{trace.round_id}.json"
            path.write_text(json.dumps(round_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            with self.rounds_jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(round_payload, ensure_ascii=False) + "\n")
            self._append_dataset_rows(
                self.round_canonical_dir,
                self._round_canonical_rows(round_payload),
                schema=ROUND_CANONICAL_SCHEMA,
                partition_keys=("recorded_date", "round_id"),
            )
            self._append_dataset_rows(
                self.round_summary_dir,
                self._round_summary_rows(round_payload),
                schema=ROUND_SUMMARY_SCHEMA,
                partition_keys=("recorded_date", "round_id"),
            )
            self._write_round_subjectivity_totals(next_round_totals)
            round_trace_rows = self._round_trace_rows(round_payload)
            if round_trace_rows:
                self._append_dataset_rows(self.round_trace_dir, round_trace_rows, schema=ROUND_TRACE_SCHEMA)
            skill_canonical_rows, skill_trace_rows = self._skill_rows(skill_rows)
            if skill_canonical_rows:
                self._append_dataset_rows(self.skill_canonical_dir, skill_canonical_rows, schema=SKILL_CANONICAL_SCHEMA)
                self._append_dataset_rows(self.skill_trace_dir, skill_trace_rows, schema=SKILL_TRACE_SCHEMA)
            self._sync_parquet_mirror_safely(track_live_status=True)

        self._io_worker.submit(write, on_error=lambda exc: self.mark_trace_sync_degraded(str(exc)))
        if sync:
            self.flush(raise_on_error=True)

    def read_round(self, round_id: int) -> dict:
        payload, _ = self.read_round_record(round_id)
        return payload

    def read_round_record(self, round_id: int) -> tuple[dict, str]:
        cached = self._round_cache.get(round_id)
        if cached is not None:
            return copy.deepcopy(cached), "memory_cache"
        if self._legacy_round_parquet_pending_rewrite():
            self.rewrite_legacy_round_parquet_history(force=True)
        status = self.trace_storage_status()
        if status["storage_state"] == "healthy":
            parquet_payload = self._read_round_from_parquet(round_id)
            if parquet_payload is not None:
                self._round_cache[int(round_id)] = copy.deepcopy(parquet_payload)
                return parquet_payload, "parquet"
        else:
            json_payload = self._read_round_from_json(round_id)
            if json_payload is not None:
                self._round_cache[int(round_id)] = copy.deepcopy(json_payload)
                return json_payload, "json_fallback"
        parquet_payload = self._read_round_from_parquet(round_id)
        if parquet_payload is not None:
            self._round_cache[int(round_id)] = copy.deepcopy(parquet_payload)
            return parquet_payload, "parquet"
        json_payload = self._read_round_from_json(round_id)
        if json_payload is not None:
            self._round_cache[int(round_id)] = copy.deepcopy(json_payload)
            return json_payload, "json_fallback"
        raise FileNotFoundError(f"trace round {round_id} not found")

    def list_rounds(self) -> list[dict]:
        if self._legacy_round_parquet_pending_rewrite():
            self.rewrite_legacy_round_parquet_history(force=True)
        if self._round_cache_complete and self._round_cache:
            return [copy.deepcopy(self._round_cache[key]) for key in sorted(self._round_cache)]
        status = self.trace_storage_status()
        if status["storage_state"] == "healthy":
            if self._round_cache:
                round_count = self._round_count_from_parquet_cached()
                if round_count and round_count <= len(self._round_cache):
                    if round_count == len(self._round_cache):
                        round_count = self._round_count_from_parquet_cached(force_refresh=True)
                    if round_count and round_count <= len(self._round_cache):
                        if round_count == len(self._round_cache):
                            self._round_cache_complete = True
                        return [copy.deepcopy(self._round_cache[key]) for key in sorted(self._round_cache)]
            parquet_rows = self._list_rounds_from_parquet()
            if parquet_rows or self.round_canonical_dir.exists():
                return self._merge_round_rows(parquet_rows)
        traces = []
        for path in sorted(self.rounds_dir.glob("round_*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if "session_id" not in payload or "recorded_at" not in payload or "recorded_date" not in payload:
                payload = ensure_recorded_fields(payload, recorded_at=_mtime_iso(path))
            traces.append(payload)
        return self._merge_round_rows(traces)

    def recent_rounds(self, *, limit: int = 20) -> list[dict]:
        effective_limit = max(int(limit), 0)
        if effective_limit == 0:
            return []
        if self._round_cache_complete and self._round_cache:
            keys = sorted(self._round_cache)[-effective_limit:]
            return [copy.deepcopy(self._round_cache[key]) for key in keys]
        status = self.trace_storage_status()
        if status["storage_state"] == "healthy":
            if self._round_cache:
                round_count = self._round_count_from_parquet_cached()
                if round_count and round_count <= len(self._round_cache):
                    if round_count == len(self._round_cache):
                        round_count = self._round_count_from_parquet_cached(force_refresh=True)
                    if round_count and round_count <= len(self._round_cache):
                        if round_count == len(self._round_cache):
                            self._round_cache_complete = True
                        keys = sorted(self._round_cache)[-effective_limit:]
                        return [copy.deepcopy(self._round_cache[key]) for key in keys]
            json_rows = self._recent_rounds_from_jsonl(effective_limit)
            if json_rows:
                return self._merge_round_rows(json_rows, limit=effective_limit)
            parquet_rows = self._recent_rounds_from_parquet(effective_limit)
            if parquet_rows or self.round_canonical_dir.exists():
                return self._merge_round_rows(parquet_rows, limit=effective_limit)
        return self.list_rounds()[-effective_limit:]

    def recent_round_signal_views(self, *, limit: int = 20) -> list[dict]:
        effective_limit = max(int(limit), 0)
        if effective_limit == 0:
            return []
        if self._signal_view_cache and effective_limit <= len(self._signal_view_cache):
            keys = sorted(self._signal_view_cache)[-effective_limit:]
            return [copy.deepcopy(self._signal_view_cache[key]) for key in keys]
        if self._round_cache_complete and self._round_cache:
            keys = sorted(self._round_cache)[-effective_limit:]
            return [self._round_signal_view(self._round_cache[key]) for key in keys]
        return [self._round_signal_view(row) for row in self.recent_rounds(limit=effective_limit)]

    def list_commands(self) -> list[dict]:
        self._ensure_command_cache_loaded()
        return [copy.deepcopy(item) for item in self._command_cache]

    def list_round_summaries(self) -> list[dict]:
        self._ensure_round_summary_loaded()
        return copy.deepcopy(self._round_summary_cache)

    def write_run(
        self,
        run_payload: dict,
        *,
        session_id: str,
        recorded_at: str | None = None,
        sync: bool = False,
    ) -> None:
        effective_recorded_at = recorded_at or utc_now_iso()
        payload = ensure_recorded_fields(run_payload, session_id=session_id, recorded_at=effective_recorded_at)
        self._run_cache[str(payload["run_id"])] = copy.deepcopy(payload)

        def write() -> None:
            path = self.runs_dir / f"run_{payload['run_id']}.json"
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            self._append_dataset_rows(
                self.run_canonical_dir,
                [
                    {
                        "session_id": payload["session_id"],
                        "recorded_at": payload["recorded_at"],
                        "recorded_date": payload["recorded_date"],
                        "run_id": payload["run_id"],
                        "payload_json": self._payload_json(payload),
                    }
                ],
                schema=RUN_CANONICAL_SCHEMA,
            )
            self._sync_parquet_mirror_safely(track_live_status=False)

        self._io_worker.submit(write, on_error=lambda exc: self.mark_trace_sync_degraded(str(exc)))
        if sync:
            self.flush(raise_on_error=True)

    def read_run(self, run_id: str) -> dict:
        cached = self._run_cache.get(run_id)
        if cached is not None:
            return copy.deepcopy(cached)
        rows = read_dataset_rows(
            self.run_canonical_dir,
            "select payload_json from read_parquet(?) where run_id = ? order by recorded_at desc limit 1",
            [run_id],
        )
        if rows:
            return ensure_recorded_fields(json.loads(rows[0]["payload_json"]))
        path = self.runs_dir / f"run_{run_id}.json"
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if "session_id" in payload and "recorded_at" in payload and "recorded_date" in payload:
                return payload
            return ensure_recorded_fields(payload, recorded_at=_mtime_iso(path))
        raise FileNotFoundError(f"run {run_id} not found")

    def append_step_trace(
        self,
        payload: dict,
        *,
        session_id: str,
        recorded_at: str | None = None,
        sync: bool = False,
    ) -> None:
        entry = ensure_recorded_fields(payload, session_id=session_id, recorded_at=recorded_at or utc_now_iso())
        self._step_cache.append(copy.deepcopy(entry))

        def write() -> None:
            with self.step_jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
            self._append_dataset_rows(
                self.step_trace_dir,
                [
                    {
                        "session_id": entry["session_id"],
                        "recorded_at": entry["recorded_at"],
                        "recorded_date": entry["recorded_date"],
                        "run_id": entry.get("run_id"),
                        "payload_json": self._payload_json(entry),
                    }
                ],
                schema=STEP_TOOL_SCHEMA,
            )
            self._sync_parquet_mirror_safely(track_live_status=False)

        self._io_worker.submit(write, on_error=lambda exc: self.mark_trace_sync_degraded(str(exc)))
        if sync:
            self.flush(raise_on_error=True)

    def append_tool_trace(
        self,
        payload: dict,
        *,
        session_id: str,
        recorded_at: str | None = None,
        sync: bool = False,
    ) -> None:
        entry = ensure_recorded_fields(payload, session_id=session_id, recorded_at=recorded_at or utc_now_iso())
        self._tool_cache.append(copy.deepcopy(entry))

        def write() -> None:
            with self.tool_jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
            self._append_dataset_rows(
                self.tool_trace_dir,
                [
                    {
                        "session_id": entry["session_id"],
                        "recorded_at": entry["recorded_at"],
                        "recorded_date": entry["recorded_date"],
                        "run_id": entry.get("run_id"),
                        "payload_json": self._payload_json(entry),
                    }
                ],
                schema=STEP_TOOL_SCHEMA,
            )
            self._sync_parquet_mirror_safely(track_live_status=False)

        self._io_worker.submit(write, on_error=lambda exc: self.mark_trace_sync_degraded(str(exc)))
        if sync:
            self.flush(raise_on_error=True)

    def list_run_steps(self, run_id: str) -> list[dict]:
        self._ensure_step_cache_loaded()
        return [copy.deepcopy(row) for row in self._step_cache if row.get("run_id") == run_id]

    def list_run_tools(self, run_id: str) -> list[dict]:
        self._ensure_tool_cache_loaded()
        return [copy.deepcopy(row) for row in self._tool_cache if row.get("run_id") == run_id]

    def append_command(
        self,
        command: str,
        result: CommandResult,
        before_state_hash: str,
        after_state_hash: str,
        *,
        session_id: str,
        recorded_at: str | None = None,
        sync: bool = False,
    ) -> None:
        self._ensure_command_cache_loaded()
        effective_recorded_at = recorded_at or utc_now_iso()
        entry = {
            "session_id": session_id,
            "recorded_at": effective_recorded_at,
            "recorded_date": iso_date(effective_recorded_at),
            "command": command,
            "command_id": result.command_id,
            "canonical": result.canonical,
            "applied": result.applied,
            "scope": result.scope,
            "delta": result.delta,
            "ttl": result.ttl,
            "risk_note": result.risk_note,
            "rollback_hint": result.rollback_hint,
            "operator_level": result.operator_level,
            "rollback_available": result.rollback_available,
            "mutation_scope": result.mutation_scope,
            "parsed_args": result.parsed_args,
            "flags": result.flags,
            "snapshot_id": result.snapshot_id,
            "rollback": result.rollback,
            "before_state_hash": before_state_hash,
            "after_state_hash": after_state_hash,
            "subject_id": result.subject_id,
            "continuity_nonce": result.continuity_nonce,
            "cause_type": result.cause_type,
            "boundary_action": result.boundary_action,
            "violation_code": result.violation_code,
            "deprecation_warning": result.deprecation_warning,
        }
        self._command_cache.append(copy.deepcopy(entry))
        existing_totals = self._command_subjectivity_totals_cache or self._read_command_subjectivity_totals_file()
        if existing_totals is None:
            if self._command_cache_loaded:
                existing_totals = self._compute_command_subjectivity_totals_from_rows(self._command_cache[:-1])
            else:
                existing_totals = self._default_command_subjectivity_totals()
        next_totals = self._normalize_command_subjectivity_totals(existing_totals)
        next_totals["summary_version"] = COMMAND_SUBJECTIVITY_TOTALS_VERSION
        next_totals["total_commands"] = int(next_totals.get("total_commands", 0) or 0) + 1
        if str(entry.get("violation_code") or "").strip():
            next_totals["boundary_violation_count"] = int(next_totals.get("boundary_violation_count", 0) or 0) + 1
        if entry.get("cause_type") == "external_stimulus":
            next_totals["external_count"] = int(next_totals.get("external_count", 0) or 0) + 1
        if entry.get("cause_type") == "endogenous":
            next_totals["internal_count"] = int(next_totals.get("internal_count", 0) or 0) + 1
        next_totals["last_recorded_at"] = str(entry.get("recorded_at", "") or "")
        self._command_subjectivity_totals_cache = next_totals

        def write() -> None:
            self.commands_path.write_text(json.dumps(self._command_cache, ensure_ascii=False, indent=2), encoding="utf-8")
            with self.commands_jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
            self._append_dataset_rows(
                self.command_canonical_dir,
                [
                    {
                        "session_id": entry["session_id"],
                        "recorded_at": entry["recorded_at"],
                        "recorded_date": entry["recorded_date"],
                        "command": entry.get("command"),
                        "payload_json": self._payload_json(entry),
                    }
                ],
                schema=COMMAND_CANONICAL_SCHEMA,
            )
            self._append_dataset_rows(
                self.command_trace_dir,
                [
                    {
                        "session_id": entry["session_id"],
                        "recorded_at": entry["recorded_at"],
                        "recorded_date": entry["recorded_date"],
                        "command": entry.get("command"),
                        "command_id": entry.get("command_id"),
                        "canonical": entry.get("canonical"),
                        "applied": entry.get("applied"),
                        "scope": entry.get("scope"),
                        "delta_json": json.dumps(entry.get("delta", {}), ensure_ascii=False, sort_keys=True),
                        "ttl": entry.get("ttl"),
                        "risk_note": entry.get("risk_note"),
                        "rollback_hint": entry.get("rollback_hint"),
                        "operator_level": entry.get("operator_level"),
                        "rollback_available": entry.get("rollback_available"),
                        "mutation_scope": entry.get("mutation_scope"),
                        "parsed_args_json": json.dumps(entry.get("parsed_args", {}), ensure_ascii=False, sort_keys=True),
                        "flags_json": json.dumps(entry.get("flags", {}), ensure_ascii=False, sort_keys=True),
                        "snapshot_id": entry.get("snapshot_id"),
                        "rollback_json": json.dumps(entry.get("rollback", {}), ensure_ascii=False, sort_keys=True),
                        "before_state_hash": entry.get("before_state_hash"),
                        "after_state_hash": entry.get("after_state_hash"),
                        "subject_id": entry.get("subject_id"),
                        "continuity_nonce": entry.get("continuity_nonce"),
                        "cause_type": entry.get("cause_type"),
                        "boundary_action": entry.get("boundary_action"),
                        "violation_code": entry.get("violation_code"),
                        "deprecation_warning": entry.get("deprecation_warning"),
                    }
                ],
                schema=COMMAND_TRACE_SCHEMA,
            )
            self._write_command_subjectivity_totals(next_totals)
            self._sync_parquet_mirror_safely(track_live_status=True)

        self._io_worker.submit(write, on_error=lambda exc: self.mark_trace_sync_degraded(str(exc)))
        if sync:
            self.flush(raise_on_error=True)

    def append_repair_entry(
        self,
        payload: dict,
        *,
        session_id: str,
        recorded_at: str | None = None,
        sync: bool = False,
    ) -> None:
        entry = ensure_recorded_fields(payload, session_id=session_id, recorded_at=recorded_at or utc_now_iso())
        self._repair_cache.append(copy.deepcopy(entry))

        def write() -> None:
            with self.repair_jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
            self._append_dataset_rows(
                self.repair_ledger_dir,
                [
                    {
                        "session_id": entry["session_id"],
                        "recorded_at": entry["recorded_at"],
                        "recorded_date": entry["recorded_date"],
                        "round_id": entry.get("round_id", 0),
                        "reason": entry.get("reason", ""),
                        "winning_priority": entry.get("winning_priority"),
                        "template": entry.get("template"),
                        "repair_stage_after": entry.get("repair_stage_after"),
                        "conflict_score": entry.get("conflict_score", 0.0),
                        "payload_json": self._payload_json(entry),
                    }
                ],
                schema=REPAIR_LEDGER_SCHEMA,
            )
            self._sync_parquet_mirror_safely(track_live_status=True)

        self._io_worker.submit(write, on_error=lambda exc: self.mark_trace_sync_degraded(str(exc)))
        if sync:
            self.flush(raise_on_error=True)

    def _read_jsonl(self, path: Path) -> list[dict]:
        if not path.exists():
            return []
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rows.append(ensure_recorded_fields(json.loads(line)))
        return rows

    def _read_jsonl_tail(self, path: Path, *, limit: int) -> list[dict]:
        effective_limit = max(int(limit), 0)
        if effective_limit == 0 or not path.exists():
            return []
        chunks: list[bytes] = []
        newline_count = 0
        chunk_size = 64 * 1024
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            position = handle.tell()
            while position > 0 and newline_count <= effective_limit:
                read_size = min(chunk_size, position)
                position -= read_size
                handle.seek(position)
                chunk = handle.read(read_size)
                chunks.append(chunk)
                newline_count += chunk.count(b"\n")
        buffer = b"".join(reversed(chunks))
        lines = [line for line in buffer.decode("utf-8", errors="ignore").splitlines() if line.strip()]
        return [ensure_recorded_fields(json.loads(line)) for line in lines[-effective_limit:]]

    def list_skill_traces(self) -> list[dict]:
        self._ensure_skill_cache_loaded()
        return copy.deepcopy(self._skill_cache)

    def recent_skill_traces(self, *, limit: int = 20) -> list[dict]:
        effective_limit = max(int(limit), 0)
        if effective_limit == 0:
            return []
        if self._skill_cache_loaded:
            return copy.deepcopy(self._skill_cache[-effective_limit:])
        status = self.trace_storage_status()
        if status["storage_state"] == "healthy":
            json_rows = self._read_jsonl_tail(self.skill_jsonl_path, limit=effective_limit)
            if json_rows or self._skill_cache:
                merged = self._merge_cache_rows(json_rows, self._skill_cache)
                return copy.deepcopy(merged[-effective_limit:])
            parquet_rows = self._recent_skill_rows_from_parquet(effective_limit)
            if parquet_rows or self.skill_canonical_dir.exists():
                merged = self._merge_cache_rows(parquet_rows, self._skill_cache)
                return copy.deepcopy(merged[-effective_limit:])
        self._ensure_skill_cache_loaded()
        return copy.deepcopy(self._skill_cache[-effective_limit:])

    def latest_round_projection_seed(self) -> dict[str, object]:
        if self._round_cache_complete and self._round_cache:
            latest_round_id = max(self._round_cache)
            latest = self._round_cache[latest_round_id]
            return {
                "round_id": latest_round_id,
                "long_run_projection": copy.deepcopy(dict(latest.get("long_run_projection", {}) or {})),
                "authenticity": copy.deepcopy(dict(latest.get("authenticity", {}) or {})),
            }
        rounds = self.recent_rounds(limit=1)
        if not rounds:
            return {}
        latest = rounds[-1]
        return {
            "round_id": latest.get("round_id"),
            "long_run_projection": copy.deepcopy(dict(latest.get("long_run_projection", {}) or {})),
            "authenticity": copy.deepcopy(dict(latest.get("authenticity", {}) or {})),
        }

    def list_command_traces(self) -> list[dict]:
        self._ensure_command_cache_loaded()
        return copy.deepcopy(self._command_cache)

    def list_repair_entries(self) -> list[dict]:
        self._ensure_repair_cache_loaded()
        return copy.deepcopy(self._repair_cache)

    def skill_stats(self, skill_name: str | None = None) -> dict:
        rows = self.list_skill_traces()
        if skill_name:
            rows = [row for row in rows if row["skill_name"] == skill_name]
        skills: dict[str, dict] = {}
        for row in rows:
            bucket = skills.setdefault(
                row["skill_name"],
                {
                    "count": 0,
                    "degraded_count": 0,
                    "fallback_count": 0,
                    "breaker_trip_count": 0,
                    "average_latency_ms": 0.0,
                },
            )
            bucket["count"] += 1
            bucket["degraded_count"] += int(bool(row.get("degraded")))
            bucket["fallback_count"] += int(bool(row.get("fallback_route")))
            bucket["breaker_trip_count"] += int(row.get("failure_policy_applied") == "trip_circuit_breaker")
            bucket["average_latency_ms"] += row.get("latency_ms", 0)
        for bucket in skills.values():
            bucket["average_latency_ms"] = round(bucket["average_latency_ms"] / bucket["count"], 2)
        return {"total_calls": len(rows), "skills": skills}

    def _sync_parquet_mirror(self) -> None:
        return None

    def _sync_parquet_mirror_safely(self, *, track_live_status: bool) -> bool:
        try:
            self._sync_parquet_mirror()
        except Exception as exc:
            if track_live_status:
                self.mark_trace_sync_degraded(str(exc))
            return False
        if track_live_status:
            self.mark_trace_sync_healthy()
        return True

    def _detect_parquet_live_ready(self) -> bool:
        return self.round_canonical_dir.exists() and any(self.round_canonical_dir.rglob("*.parquet"))

    def _default_trace_sync_status(self) -> dict:
        parquet_live_ready = self._detect_parquet_live_ready()
        return {
            "read_source_default": "parquet",
            "storage_state": "healthy",
            "trace_sync_state": "healthy",
            "parquet_live_ready": parquet_live_ready,
            "degraded_reason": None,
            "last_sync_at": None,
        }

    def _write_trace_sync_status(self, payload: dict) -> None:
        self.trace_sync_status_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def trace_storage_status(self) -> dict:
        if not self.trace_sync_status_path.exists():
            payload = self._default_trace_sync_status()
            self._write_trace_sync_status(payload)
            self._trace_sync_status_cache = payload
            return payload
        payload = dict(self._trace_sync_status_cache)
        payload.setdefault("read_source_default", "parquet")
        payload.setdefault("storage_state", payload.get("trace_sync_state", "healthy"))
        payload.setdefault("trace_sync_state", payload["storage_state"])
        if "parquet_live_ready" not in payload:
            payload["parquet_live_ready"] = self._detect_parquet_live_ready()
        payload["parquet_live_ready"] = bool(payload["parquet_live_ready"]) if payload["storage_state"] == "healthy" else False
        payload.setdefault("degraded_reason", None)
        payload.setdefault("last_sync_at", None)
        payload["signal_view_cache_size"] = len(self._signal_view_cache)
        payload["signal_view_cache_limit"] = self._signal_view_cache_limit
        self._trace_sync_status_cache = payload
        return payload

    def mark_trace_sync_healthy(self) -> dict:
        payload = self.trace_storage_status()
        payload["storage_state"] = "healthy"
        payload["trace_sync_state"] = "healthy"
        payload["parquet_live_ready"] = True
        payload["degraded_reason"] = None
        payload["last_sync_at"] = utc_now_iso()
        self._trace_sync_status_cache = payload
        self._io_worker.submit(lambda: self._write_trace_sync_status(payload))
        return payload

    def mark_trace_sync_degraded(self, reason: str) -> dict:
        payload = self.trace_storage_status()
        payload["storage_state"] = "degraded"
        payload["trace_sync_state"] = "degraded"
        payload["parquet_live_ready"] = False
        payload["degraded_reason"] = reason
        self._trace_sync_status_cache = payload
        self._io_worker.submit(lambda: self._write_trace_sync_status(payload))
        return payload

    def _read_round_from_json(self, round_id: int) -> dict | None:
        path = self.rounds_dir / f"round_{round_id}.json"
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "session_id" in payload and "recorded_at" in payload and "recorded_date" in payload:
            return rewrite_round_payload(payload)
        recorded_at = _mtime_iso(path)
        return rewrite_round_payload(ensure_recorded_fields(payload, recorded_at=recorded_at))

    def _read_round_from_parquet(self, round_id: int) -> dict | None:
        rows = read_dataset_rows(
            self.round_canonical_dir,
            "select payload_json from read_parquet(?) where round_id = ? order by recorded_at desc limit 1",
            [round_id],
        )
        if not rows:
            return None
        payload = ensure_recorded_fields(json.loads(rows[0]["payload_json"]))
        if self._round_payload_requires_rewrite(payload):
            self.rewrite_legacy_round_parquet_history(force=True)
            rows = read_dataset_rows(
                self.round_canonical_dir,
                "select payload_json from read_parquet(?) where round_id = ? order by recorded_at desc limit 1",
                [round_id],
            )
            if not rows:
                return None
            payload = ensure_recorded_fields(json.loads(rows[0]["payload_json"]))
        return normalize_round_payload(payload)

    def _list_rounds_from_parquet(self) -> list[dict]:
        payloads = self._raw_round_payloads_from_parquet()
        if any(self._round_payload_requires_rewrite(payload) for payload in payloads):
            self.rewrite_legacy_round_parquet_history(force=True)
            payloads = self._raw_round_payloads_from_parquet()
        return [normalize_round_payload(payload) for payload in payloads]

    def _recent_rounds_from_parquet(self, limit: int) -> list[dict]:
        rows = self._query_payload_rows(
            self.round_canonical_dir,
            "select payload_json from read_parquet(?) order by round_id desc, recorded_at desc limit ?",
            [str(self.round_canonical_dir / "**" / "*.parquet"), int(limit)],
        )
        payloads = [normalize_round_payload(ensure_recorded_fields(json.loads(row["payload_json"]))) for row in rows]
        payloads.reverse()
        return payloads

    def _recent_rounds_from_jsonl(self, limit: int) -> list[dict]:
        return [rewrite_round_payload(row) for row in self._read_jsonl_tail(self.rounds_jsonl_path, limit=limit)]

    def _recent_skill_rows_from_parquet(self, limit: int) -> list[dict]:
        rows = self._query_payload_rows(
            self.skill_canonical_dir,
            "select payload_json from read_parquet(?) order by recorded_at desc, round_id desc, skill_name desc limit ?",
            [str(self.skill_canonical_dir / "**" / "*.parquet"), int(limit)],
        )
        payloads = [ensure_recorded_fields(json.loads(row["payload_json"])) for row in rows]
        payloads.reverse()
        return payloads

    def _merge_round_rows(self, persisted_rows: list[dict], *, limit: int | None = None) -> list[dict]:
        if not self._round_cache:
            rows = [copy.deepcopy(row) for row in persisted_rows]
        else:
            merged = {int(row["round_id"]): copy.deepcopy(row) for row in persisted_rows}
            merged.update({round_id: copy.deepcopy(payload) for round_id, payload in self._round_cache.items()})
            keys = sorted(merged)
            if limit is not None:
                keys = keys[-max(int(limit), 0):]
            rows = [merged[key] for key in keys]
        if limit is not None and len(rows) > limit:
            return rows[-limit:]
        return rows

    def _round_count_from_parquet(self) -> int:
        rows = self._query_payload_rows(
            self.round_canonical_dir,
            "select count(*) as row_count from read_parquet(?)",
            [str(self.round_canonical_dir / "**" / "*.parquet")],
        )
        if not rows:
            return 0
        return int(rows[0].get("row_count", 0) or 0)

    def _round_count_from_parquet_cached(self, *, force_refresh: bool = False) -> int:
        now = time.monotonic()
        if not force_refresh:
            cached = self._round_count_cache_value
            if cached is not None and now - self._round_count_cache_checked_at <= self._round_count_cache_ttl:
                return cached
        count = self._round_count_from_parquet()
        self._round_count_cache_value = count
        self._round_count_cache_checked_at = now
        return count

    def _list_payload_rows_from_parquet(self, filename: str) -> list[dict]:
        path = self.parquet_dir / filename
        rows = self._query_payload_rows(
            path,
            "select payload_json from read_parquet(?) order by recorded_at",
            [str(path)],
        )
        return [ensure_recorded_fields(json.loads(row["payload_json"])) for row in rows]

    def _query_payload_rows(self, path: Path, query: str, params: list[object]) -> list[dict]:
        if not path.exists():
            return []
        conn = duckdb.connect()
        try:
            try:
                cursor = conn.execute(query, params)
            except duckdb.IOException as exc:
                if "No files found" in str(exc):
                    return []
                raise
            except duckdb.InvalidInputException as exc:
                message = str(exc)
                if "too small to be a Parquet file" in message or "No magic bytes found at end of file" in message:
                    return []
                raise
            columns = [item[0] for item in cursor.description]
            return [dict(zip(columns, row, strict=False)) for row in cursor.fetchall()]
        finally:
            conn.close()

    def flush(self, *, raise_on_error: bool = False) -> None:
        self._io_worker.flush(raise_on_error=raise_on_error)
