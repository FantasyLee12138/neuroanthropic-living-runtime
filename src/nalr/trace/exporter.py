from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import duckdb

from nalr.runtime.metadata import ensure_recorded_fields, utc_now_iso
from nalr.trace.store import TraceStore


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
    "distribution_state_json": "VARCHAR",
    "state_snapshot_json": "VARCHAR",
    "render_plan_json": "VARCHAR",
    "gate_decisions_json": "VARCHAR",
    "rendered_expression_json": "VARCHAR",
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

COMMAND_SCHEMA = {
    "session_id": "VARCHAR",
    "recorded_at": "VARCHAR",
    "recorded_date": "VARCHAR",
    "command": "VARCHAR",
    "applied": "BOOLEAN",
    "scope": "VARCHAR",
    "delta_json": "VARCHAR",
    "ttl": "VARCHAR",
    "risk_note": "VARCHAR",
    "rollback_hint": "VARCHAR",
    "operator_level": "VARCHAR",
    "rollback_available": "BOOLEAN",
    "before_state_hash": "VARCHAR",
    "after_state_hash": "VARCHAR",
}


def _json_blob(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _mtime_iso(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


class TraceExporter:
    def __init__(self, store: TraceStore) -> None:
        self.store = store

    def export_parquet(self, *, since_round: int | None = None, overwrite: bool = False) -> dict:
        outputs = {
            "round": self.store.parquet_dir / "round_trace.parquet",
            "skill": self.store.parquet_dir / "skill_trace.parquet",
            "command": self.store.parquet_dir / "command_trace.parquet",
        }
        for path in outputs.values():
            if path.exists() and not overwrite:
                raise FileExistsError(f"{path.name} already exists; pass --overwrite to replace it")
            if path.exists():
                path.unlink()

        round_rows = self._flatten_round_rows(since_round=since_round)
        skill_rows = self._flatten_skill_rows(since_round=since_round)
        command_rows = self._flatten_command_rows()
        self._write_rows(round_rows, outputs["round"], ROUND_SCHEMA)
        self._write_rows(skill_rows, outputs["skill"], SKILL_SCHEMA)
        self._write_rows(command_rows, outputs["command"], COMMAND_SCHEMA)
        return {
            "parquet_dir": str(self.store.parquet_dir),
            "tables": {
                "round_trace": {"path": str(outputs["round"]), "row_count": len(round_rows)},
                "skill_trace": {"path": str(outputs["skill"]), "row_count": len(skill_rows)},
                "command_trace": {"path": str(outputs["command"]), "row_count": len(command_rows)},
            },
        }

    def _flatten_round_rows(self, *, since_round: int | None) -> list[dict]:
        rows: list[dict] = []
        for path in sorted(self.store.rounds_dir.glob("round_*.json")):
            payload = ensure_recorded_fields(json.loads(path.read_text(encoding="utf-8")), recorded_at=_mtime_iso(path))
            if since_round is not None and payload.get("round_id", 0) < since_round:
                continue
            summaries = payload.get("proposal_summaries", [])
            for summary in summaries:
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
                            "tags_json": _json_blob(summary.get("tags", [])),
                            "distribution_state_json": _json_blob(payload.get("distribution_state", {})),
                            "state_snapshot_json": _json_blob(payload.get("state_snapshot", {})),
                            "render_plan_json": _json_blob(payload.get("render_plan", {})),
                            "gate_decisions_json": _json_blob(payload.get("gate_decisions", [])),
                            "rendered_expression_json": _json_blob(payload.get("rendered_expression", {})),
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

    def _flatten_command_rows(self) -> list[dict]:
        rows = []
        for row in self.store.list_command_traces():
            rows.append(
                {
                    "session_id": row.get("session_id", "legacy"),
                    "recorded_at": row.get("recorded_at", utc_now_iso()),
                    "recorded_date": row.get("recorded_date", row.get("recorded_at", utc_now_iso())[:10]),
                    "command": row.get("command"),
                    "applied": row.get("applied", False),
                    "scope": row.get("scope"),
                    "delta_json": _json_blob(row.get("delta", {})),
                    "ttl": row.get("ttl"),
                    "risk_note": row.get("risk_note"),
                    "rollback_hint": row.get("rollback_hint"),
                    "operator_level": row.get("operator_level"),
                    "rollback_available": row.get("rollback_available", False),
                    "before_state_hash": row.get("before_state_hash"),
                    "after_state_hash": row.get("after_state_hash"),
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
