from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import duckdb

from nalr.runtime.metadata import ensure_recorded_fields, utc_now_iso
from nalr.trace.store import TraceStore, canonical_probability_field_payload


ROUND_SCHEMA = {
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

ROUND_CANONICAL_SCHEMA = {
    "session_id": "VARCHAR",
    "recorded_at": "VARCHAR",
    "recorded_date": "VARCHAR",
    "round_id": "BIGINT",
    "payload_json": "VARCHAR",
}

SKILL_SCHEMA = {
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

SKILL_CANONICAL_SCHEMA = {
    "session_id": "VARCHAR",
    "recorded_at": "VARCHAR",
    "recorded_date": "VARCHAR",
    "round_id": "BIGINT",
    "skill_name": "VARCHAR",
    "payload_json": "VARCHAR",
}

COMMAND_SCHEMA = {
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
}

COMMAND_CANONICAL_SCHEMA = {
    "session_id": "VARCHAR",
    "recorded_at": "VARCHAR",
    "recorded_date": "VARCHAR",
    "command": "VARCHAR",
    "payload_json": "VARCHAR",
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


def _json_blob(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _mtime_iso(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _conflict_arbitration_payload(payload: dict) -> dict:
    direct = dict(payload.get("conflict_arbitration", {}) or {})
    probability_field = canonical_probability_field_payload(payload)
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
    compromise_template_prior = dict(direct.get("compromise_template_prior", {}) or conflict_row.get("compromise_template_prior", {}) or {})
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


class TraceExporter:
    def __init__(self, store: TraceStore) -> None:
        self.store = store

    def export_parquet(self, *, since_round: int | None = None, overwrite: bool = False) -> dict:
        history_rewrite = self.store.rewrite_legacy_round_parquet_history()
        outputs = {
            "round": self.store.parquet_dir / "round_trace.parquet",
            "round_canonical": self.store.parquet_dir / "round_canonical.parquet",
            "skill": self.store.parquet_dir / "skill_trace.parquet",
            "skill_canonical": self.store.parquet_dir / "skill_canonical.parquet",
            "command": self.store.parquet_dir / "command_trace.parquet",
            "command_canonical": self.store.parquet_dir / "command_canonical.parquet",
            "repair_ledger": self.store.parquet_dir / "repair_ledger.parquet",
        }
        for path in outputs.values():
            if path.exists() and not overwrite:
                raise FileExistsError(f"{path.name} already exists; pass --overwrite to replace it")
            if path.exists():
                path.unlink()

        round_rows = self._flatten_round_rows(since_round=since_round)
        round_canonical_rows = self._canonical_round_rows(since_round=since_round)
        skill_rows = self._flatten_skill_rows(since_round=since_round)
        skill_canonical_rows = self._canonical_skill_rows(since_round=since_round)
        command_rows = self._flatten_command_rows()
        command_canonical_rows = self._canonical_command_rows()
        repair_rows = self._canonical_repair_rows()
        self._write_rows(round_rows, outputs["round"], ROUND_SCHEMA)
        self._write_rows(round_canonical_rows, outputs["round_canonical"], ROUND_CANONICAL_SCHEMA)
        self._write_rows(skill_rows, outputs["skill"], SKILL_SCHEMA)
        self._write_rows(skill_canonical_rows, outputs["skill_canonical"], SKILL_CANONICAL_SCHEMA)
        self._write_rows(command_rows, outputs["command"], COMMAND_SCHEMA)
        self._write_rows(command_canonical_rows, outputs["command_canonical"], COMMAND_CANONICAL_SCHEMA)
        self._write_rows(repair_rows, outputs["repair_ledger"], REPAIR_LEDGER_SCHEMA)
        return {
            "mode": "rebuild",
            "parquet_dir": str(self.store.parquet_dir),
            "history_rewrite": history_rewrite,
            "tables": {
                "round_trace": {"path": str(outputs["round"]), "row_count": len(round_rows)},
                "round_canonical": {"path": str(outputs["round_canonical"]), "row_count": len(round_canonical_rows)},
                "skill_trace": {"path": str(outputs["skill"]), "row_count": len(skill_rows)},
                "skill_canonical": {"path": str(outputs["skill_canonical"]), "row_count": len(skill_canonical_rows)},
                "command_trace": {"path": str(outputs["command"]), "row_count": len(command_rows)},
                "command_canonical": {"path": str(outputs["command_canonical"]), "row_count": len(command_canonical_rows)},
                "repair_ledger": {"path": str(outputs["repair_ledger"]), "row_count": len(repair_rows)},
            },
        }

    def _canonical_round_rows(self, *, since_round: int | None) -> list[dict]:
        rows: list[dict] = []
        for payload in self.store.list_rounds():
            if since_round is not None and payload.get("round_id", 0) < since_round:
                continue
            rows.append(
                {
                    "session_id": payload["session_id"],
                    "recorded_at": payload["recorded_at"],
                    "recorded_date": payload["recorded_date"],
                    "round_id": payload.get("round_id"),
                    "payload_json": _json_blob(payload),
                }
            )
        return rows

    def _flatten_round_rows(self, *, since_round: int | None) -> list[dict]:
        rows: list[dict] = []
        for payload in self.store.list_rounds():
            if since_round is not None and payload.get("round_id", 0) < since_round:
                continue
            probability_field = canonical_probability_field_payload(payload)
            token_state = probability_field.get("token_state", {}) if isinstance(probability_field, dict) else {}
            couplings = probability_field.get("couplings", []) if isinstance(probability_field, dict) else []
            summaries = payload.get("proposal_summaries", [])
            for summary in summaries:
                delta_map = summary.get("delta_p", {}) or {summary.get("top_action") or "unknown": None}
                long_run_projection = payload.get("long_run_projection", {})
                long_run_online_prior = long_run_projection.get("online_prior", {}) if isinstance(long_run_projection, dict) else {}
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
                            "tags_json": _json_blob(summary.get("tags", [])),
                            "probability_field_json": _json_blob(probability_field),
                            "token_state_json": _json_blob(token_state),
                            "cross_layer_coupling_verdict_json": _json_blob(
                                {
                                    "observed_pairs": [
                                        "->".join(
                                            (
                                                str(item.get("source_layer", "")),
                                                str(item.get("target_layer", "")),
                                                str(item.get("carrier_signal", "")),
                                            )
                                        )
                                        for item in couplings
                                        if isinstance(item, dict)
                                    ],
                                    "illegal_pairs": [],
                                    "legal": True,
                                }
                            ),
                            "renderer_decision_integrity_json": _json_blob(payload.get("renderer_decision_integrity", {})),
                            "memory_write_gate_json": _json_blob(payload.get("memory_write_gate", {})),
                            "conflict_arbitration_json": _json_blob(_conflict_arbitration_payload(payload)),
                            "state_snapshot_json": _json_blob(payload.get("state_snapshot", {})),
                            "render_plan_json": _json_blob(payload.get("render_plan", {})),
                            "gate_decisions_json": _json_blob(payload.get("gate_decisions", [])),
                            "rendered_expression_json": _json_blob(payload.get("rendered_expression", {})),
                            "long_run_projection_json": _json_blob(long_run_projection),
                            "long_run_projection_online_prior_json": _json_blob(long_run_online_prior),
                            "motivation_pool_json": _json_blob(payload.get("motivation_pool", {})),
                            "motivation_feedback_json": _json_blob(payload.get("motivation_feedback", {})),
                            "endogenous_tick_reason_json": _json_blob(payload.get("endogenous_tick_reason", {})),
                            "endogenous_policy_shift_json": _json_blob(payload.get("endogenous_policy_shift", {})),
                        }
                    )
        return rows

    def _flatten_skill_rows(self, *, since_round: int | None) -> list[dict]:
        rows = []
        for row in self.store.list_skill_traces():
            if since_round is not None and row.get("round_id", 0) < since_round:
                continue
            rows.append(
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
        return rows

    def _canonical_skill_rows(self, *, since_round: int | None) -> list[dict]:
        rows = []
        for row in self.store.list_skill_traces():
            if since_round is not None and row.get("round_id", 0) < since_round:
                continue
            rows.append(
                {
                    "session_id": row.get("session_id", "legacy"),
                    "recorded_at": row.get("recorded_at", utc_now_iso()),
                    "recorded_date": row.get("recorded_date", row.get("recorded_at", utc_now_iso())[:10]),
                    "round_id": row.get("round_id"),
                    "skill_name": row.get("skill_name"),
                    "payload_json": _json_blob(row),
                }
            )
        return rows

    def _flatten_command_rows(self) -> list[dict]:
        rows = []
        for row in self.store.list_command_traces():
            rows.append(
                {
                    "session_id": row.get("session_id", "legacy"),
                    "recorded_at": row.get("recorded_at", utc_now_iso()),
                    "recorded_date": row.get("recorded_date", row.get("recorded_at", utc_now_iso())[:10]),
                    "command": row.get("command"),
                    "command_id": row.get("command_id"),
                    "canonical": row.get("canonical", row.get("command")),
                    "applied": row.get("applied", False),
                    "scope": row.get("scope"),
                    "delta_json": _json_blob(row.get("delta", {})),
                    "ttl": row.get("ttl"),
                    "risk_note": row.get("risk_note"),
                    "rollback_hint": row.get("rollback_hint"),
                    "operator_level": row.get("operator_level"),
                    "rollback_available": row.get("rollback_available", False),
                    "mutation_scope": row.get("mutation_scope"),
                    "parsed_args_json": _json_blob(row.get("parsed_args", {})),
                    "flags_json": _json_blob(row.get("flags", {})),
                    "snapshot_id": row.get("snapshot_id"),
                    "rollback_json": _json_blob(row.get("rollback", {})),
                    "before_state_hash": row.get("before_state_hash"),
                    "after_state_hash": row.get("after_state_hash"),
                }
            )
        return rows

    def _canonical_command_rows(self) -> list[dict]:
        rows = []
        for row in self.store.list_command_traces():
            rows.append(
                {
                    "session_id": row.get("session_id", "legacy"),
                    "recorded_at": row.get("recorded_at", utc_now_iso()),
                    "recorded_date": row.get("recorded_date", row.get("recorded_at", utc_now_iso())[:10]),
                    "command": row.get("command"),
                    "payload_json": _json_blob(row),
                }
            )
        return rows

    def _canonical_repair_rows(self) -> list[dict]:
        rows = []
        for row in self.store.list_repair_entries():
            rows.append(
                {
                    "session_id": row.get("session_id", "legacy"),
                    "recorded_at": row.get("recorded_at", utc_now_iso()),
                    "recorded_date": row.get("recorded_date", row.get("recorded_at", utc_now_iso())[:10]),
                    "round_id": row.get("round_id", 0),
                    "reason": row.get("reason", ""),
                    "winning_priority": row.get("winning_priority"),
                    "template": row.get("template"),
                    "repair_stage_after": row.get("repair_stage_after"),
                    "conflict_score": row.get("conflict_score", 0.0),
                    "payload_json": _json_blob(row),
                }
            )
        return rows

    def _write_rows(self, rows: list[dict], output_path: Path, schema: dict[str, str]) -> None:
        conn = duckdb.connect()
        table_name = "export_rows"
        try:
            schema_sql = ", ".join(f"{name} {dtype}" for name, dtype in schema.items())
            conn.execute(f"create table {table_name} ({schema_sql})")
            if rows:
                with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".jsonl", delete=False) as handle:
                    temp_path = Path(handle.name)
                    for row in rows:
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                try:
                    columns = ", ".join(schema.keys())
                    conn.execute(
                        f"insert into {table_name} ({columns}) select {columns} from read_json_auto(?, format='newline_delimited')",
                        [str(temp_path)],
                    )
                finally:
                    temp_path.unlink(missing_ok=True)
            conn.execute(f"copy {table_name} to ? (format parquet)", [str(output_path)])
        finally:
            conn.close()
