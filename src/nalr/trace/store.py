from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from uuid import uuid4

import duckdb

from nalr.runtime.async_io import AsyncIOWorker
from nalr.runtime.metadata import ensure_recorded_fields, iso_date, utc_now_iso
from nalr.schemas.models import CommandResult, RoundTrace, to_dict
from nalr.storage.parquet_io import append_dataset, read_dataset_rows


def _mtime_iso(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


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


class TraceStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.rounds_dir = self.root / "traces" / "rounds"
        self.rounds_jsonl_path = self.root / "traces" / "round_traces.jsonl"
        self.runs_dir = self.root / "traces" / "runs"
        self.runs_jsonl_path = self.root / "traces" / "run_traces.jsonl"
        self.step_jsonl_path = self.root / "traces" / "step_traces.jsonl"
        self.tool_jsonl_path = self.root / "traces" / "tool_traces.jsonl"
        self.skills_dir = self.root / "traces" / "skills"
        self.skill_jsonl_path = self.skills_dir / "skill_traces.jsonl"
        self.commands_path = self.root / "traces" / "command_traces.json"
        self.commands_jsonl_path = self.root / "traces" / "command_traces.jsonl"
        self.repair_jsonl_path = self.root / "traces" / "repair_ledger.jsonl"
        self.parquet_dir = self.root / "traces" / "parquet"
        self.round_canonical_dir = self.parquet_dir / "round_canonical"
        self.round_trace_dir = self.parquet_dir / "round_trace"
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
        self._legacy_round_parquet_rewrite_pending: bool | None = None
        self._round_cache = {int(payload["round_id"]): payload for payload in self._load_round_records()}
        self._run_cache = {str(payload["run_id"]): payload for payload in self._load_run_records()}
        self._step_cache = self._load_payload_dataset(self.step_trace_dir) or self._read_jsonl(self.step_jsonl_path)
        self._tool_cache = self._load_payload_dataset(self.tool_trace_dir) or self._read_jsonl(self.tool_jsonl_path)
        self._skill_cache = self._load_payload_dataset(self.skill_canonical_dir) or self._read_jsonl(self.skill_jsonl_path)
        self._command_cache = self._load_payload_dataset(self.command_canonical_dir) or self._read_jsonl(self.commands_jsonl_path)
        self._repair_cache = self._load_payload_dataset(self.repair_ledger_dir) or self._read_jsonl(self.repair_jsonl_path)
        self._trace_sync_status_cache = json.loads(self.trace_sync_status_path.read_text(encoding="utf-8"))
        self._signal_view_cache_limit = 100
        self._signal_view_cache: dict[int, dict] = {}
        self.prewarm_recent_views(limit=min(self._signal_view_cache_limit, max(len(self._round_cache), 20)))

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
            for payload in canonical_payloads:
                canonical_rows.extend(self._round_canonical_rows(payload))
            self._rewrite_dataset_rows(
                self.round_canonical_dir,
                canonical_rows,
                schema=ROUND_CANONICAL_SCHEMA,
                partition_keys=("recorded_date", "round_id"),
            )
        if round_trace_rows or self.round_trace_dir.exists():
            self._rewrite_dataset_rows(self.round_trace_dir, round_trace_rows, schema=ROUND_TRACE_SCHEMA)

        if hasattr(self, "_round_cache"):
            self._round_cache = {int(payload["round_id"]): copy.deepcopy(payload) for payload in canonical_payloads}

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
                round_trace_rows = self._round_trace_rows(payload)
                if round_trace_rows:
                    self._append_dataset_rows(self.round_trace_dir, round_trace_rows, schema=ROUND_TRACE_SCHEMA)
                skill_rows = payload.get("skill_traces", [])
                canonical, flat = self._skill_rows(skill_rows)
                if canonical:
                    self._append_dataset_rows(self.skill_canonical_dir, canonical, schema=SKILL_CANONICAL_SCHEMA)
                    self._append_dataset_rows(self.skill_trace_dir, flat, schema=SKILL_TRACE_SCHEMA)
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
        self._round_cache[int(round_payload["round_id"])] = round_payload
        self._signal_view_cache[int(round_payload["round_id"])] = self._round_signal_view(round_payload)
        if len(self._signal_view_cache) > self._signal_view_cache_limit:
            for stale_round_id in sorted(self._signal_view_cache)[:-self._signal_view_cache_limit]:
                self._signal_view_cache.pop(stale_round_id, None)
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
        if self._legacy_round_parquet_pending_rewrite():
            self.rewrite_legacy_round_parquet_history(force=True)
        status = self.trace_storage_status()
        if status["storage_state"] == "healthy":
            parquet_payload = self._read_round_from_parquet(round_id)
            if parquet_payload is not None:
                return parquet_payload, "parquet"
            cached = self._round_cache.get(round_id)
            if cached is not None:
                return copy.deepcopy(cached), "memory_cache"
        else:
            json_payload = self._read_round_from_json(round_id)
            if json_payload is not None:
                return json_payload, "json_fallback"
            cached = self._round_cache.get(round_id)
            if cached is not None:
                return copy.deepcopy(cached), "memory_cache"
        parquet_payload = self._read_round_from_parquet(round_id)
        if parquet_payload is not None:
            return parquet_payload, "parquet"
        json_payload = self._read_round_from_json(round_id)
        if json_payload is not None:
            return json_payload, "json_fallback"
        cached = self._round_cache.get(round_id)
        if cached is not None:
            return copy.deepcopy(cached), "memory_cache"
        raise FileNotFoundError(f"trace round {round_id} not found")

    def list_rounds(self) -> list[dict]:
        if self._legacy_round_parquet_pending_rewrite():
            self.rewrite_legacy_round_parquet_history(force=True)
        if self._round_cache:
            return [copy.deepcopy(self._round_cache[key]) for key in sorted(self._round_cache)]
        status = self.trace_storage_status()
        if status["storage_state"] == "healthy":
            parquet_rows = self._list_rounds_from_parquet()
            if parquet_rows or self.round_canonical_dir.exists():
                return parquet_rows
        traces = []
        for path in sorted(self.rounds_dir.glob("round_*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if "session_id" not in payload or "recorded_at" not in payload or "recorded_date" not in payload:
                payload = ensure_recorded_fields(payload, recorded_at=_mtime_iso(path))
            traces.append(payload)
        return traces

    def recent_rounds(self, *, limit: int = 20) -> list[dict]:
        effective_limit = max(int(limit), 0)
        if effective_limit == 0:
            return []
        if self._round_cache:
            keys = sorted(self._round_cache)[-effective_limit:]
            return [copy.deepcopy(self._round_cache[key]) for key in keys]
        return self.list_rounds()[-effective_limit:]

    def recent_round_signal_views(self, *, limit: int = 20) -> list[dict]:
        effective_limit = max(int(limit), 0)
        if effective_limit == 0:
            return []
        if self._signal_view_cache and effective_limit <= len(self._signal_view_cache):
            keys = sorted(self._signal_view_cache)[-effective_limit:]
            return [copy.deepcopy(self._signal_view_cache[key]) for key in keys]
        if self._round_cache:
            keys = sorted(self._round_cache)[-effective_limit:]
            return [self._round_signal_view(self._round_cache[key]) for key in keys]
        return [self._round_signal_view(row) for row in self.list_rounds()[-effective_limit:]]

    def list_commands(self) -> list[dict]:
        return [copy.deepcopy(item) for item in self._command_cache]

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
        return [copy.deepcopy(row) for row in self._step_cache if row.get("run_id") == run_id]

    def list_run_tools(self, run_id: str) -> list[dict]:
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

    def list_skill_traces(self) -> list[dict]:
        if self._skill_cache:
            return copy.deepcopy(self._skill_cache)
        status = self.trace_storage_status()
        if status["storage_state"] == "healthy":
            parquet_rows = self._load_payload_dataset(self.skill_canonical_dir)
            if parquet_rows or self.skill_canonical_dir.exists():
                return parquet_rows
        return self._read_jsonl(self.skill_jsonl_path)

    def recent_skill_traces(self, *, limit: int = 20) -> list[dict]:
        effective_limit = max(int(limit), 0)
        if effective_limit == 0:
            return []
        if self._skill_cache:
            return copy.deepcopy(self._skill_cache[-effective_limit:])
        return self.list_skill_traces()[-effective_limit:]

    def latest_round_projection_seed(self) -> dict[str, object]:
        if self._round_cache:
            latest_round_id = max(self._round_cache)
            latest = self._round_cache[latest_round_id]
            return {
                "round_id": latest_round_id,
                "long_run_projection": copy.deepcopy(dict(latest.get("long_run_projection", {}) or {})),
                "authenticity": copy.deepcopy(dict(latest.get("authenticity", {}) or {})),
            }
        rounds = self.list_rounds()
        if not rounds:
            return {}
        latest = rounds[-1]
        return {
            "round_id": latest.get("round_id"),
            "long_run_projection": copy.deepcopy(dict(latest.get("long_run_projection", {}) or {})),
            "authenticity": copy.deepcopy(dict(latest.get("authenticity", {}) or {})),
        }

    def list_command_traces(self) -> list[dict]:
        if self._command_cache:
            return copy.deepcopy(self._command_cache)
        status = self.trace_storage_status()
        if status["storage_state"] == "healthy":
            parquet_rows = self._load_payload_dataset(self.command_canonical_dir)
            if parquet_rows or self.command_canonical_dir.exists():
                return parquet_rows
        return self._read_jsonl(self.commands_jsonl_path)

    def list_repair_entries(self) -> list[dict]:
        if self._repair_cache:
            return copy.deepcopy(self._repair_cache)
        status = self.trace_storage_status()
        if status["storage_state"] == "healthy":
            parquet_rows = self._load_payload_dataset(self.repair_ledger_dir)
            if parquet_rows or self.repair_ledger_dir.exists():
                return parquet_rows
        return self._read_jsonl(self.repair_jsonl_path)

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

    def _default_trace_sync_status(self) -> dict:
        parquet_live_ready = self.round_canonical_dir.exists() and any(self.round_canonical_dir.rglob("*.parquet"))
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
        parquet_live_ready = self.round_canonical_dir.exists() and any(self.round_canonical_dir.rglob("*.parquet"))
        payload.setdefault("read_source_default", "parquet")
        payload.setdefault("storage_state", payload.get("trace_sync_state", "healthy"))
        payload.setdefault("trace_sync_state", payload["storage_state"])
        payload["parquet_live_ready"] = parquet_live_ready if payload["storage_state"] == "healthy" else False
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
            columns = [item[0] for item in cursor.description]
            return [dict(zip(columns, row, strict=False)) for row in cursor.fetchall()]
        finally:
            conn.close()

    def flush(self, *, raise_on_error: bool = False) -> None:
        self._io_worker.flush(raise_on_error=raise_on_error)
