from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import duckdb

from nalr.runtime.async_io import AsyncIOWorker
from nalr.runtime.metadata import ensure_recorded_fields, iso_date, utc_now_iso
from nalr.schemas.models import CommandResult, RoundTrace, to_dict
from nalr.storage.parquet_io import append_dataset, read_dataset_rows


def _mtime_iso(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


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
    "distribution_state_json": "VARCHAR",
    "state_snapshot_json": "VARCHAR",
    "render_plan_json": "VARCHAR",
    "gate_decisions_json": "VARCHAR",
    "rendered_expression_json": "VARCHAR",
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
        self._round_cache = {int(payload["round_id"]): payload for payload in self._load_round_records()}
        self._run_cache = {str(payload["run_id"]): payload for payload in self._load_run_records()}
        self._step_cache = self._load_payload_dataset(self.step_trace_dir) or self._read_jsonl(self.step_jsonl_path)
        self._tool_cache = self._load_payload_dataset(self.tool_trace_dir) or self._read_jsonl(self.tool_jsonl_path)
        self._skill_cache = self._load_payload_dataset(self.skill_canonical_dir) or self._read_jsonl(self.skill_jsonl_path)
        self._command_cache = self._load_payload_dataset(self.command_canonical_dir) or self._read_jsonl(self.commands_jsonl_path)
        self._repair_cache = self._load_payload_dataset(self.repair_ledger_dir) or self._read_jsonl(self.repair_jsonl_path)
        self._trace_sync_status_cache = json.loads(self.trace_sync_status_path.read_text(encoding="utf-8"))

    def _load_round_records(self) -> list[dict]:
        parquet_rows = self._list_rounds_from_parquet()
        if parquet_rows:
            return parquet_rows
        return [
            ensure_recorded_fields(json.loads(path.read_text(encoding="utf-8")), recorded_at=_mtime_iso(path))
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
                        "distribution_state_json": json.dumps(payload.get("distribution_state", {}), ensure_ascii=False, sort_keys=True),
                        "state_snapshot_json": json.dumps(payload.get("state_snapshot", {}), ensure_ascii=False, sort_keys=True),
                        "render_plan_json": json.dumps(payload.get("render_plan", {}), ensure_ascii=False, sort_keys=True),
                        "gate_decisions_json": json.dumps(payload.get("gate_decisions", []), ensure_ascii=False, sort_keys=True),
                        "rendered_expression_json": json.dumps(payload.get("rendered_expression", {}), ensure_ascii=False, sort_keys=True),
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
        payload = ensure_recorded_fields(
            to_dict(trace),
            session_id=trace.session_id,
            recorded_at=trace.recorded_at,
        )
        self._round_cache[int(payload["round_id"])] = copy.deepcopy(payload)
        self._skill_cache.extend(copy.deepcopy(payload.get("skill_traces", [])))
        if payload.get("skill_traces", []):
            with self.skill_jsonl_path.open("a", encoding="utf-8") as handle:
                for row in payload["skill_traces"]:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        def write() -> None:
            path = self.rounds_dir / f"round_{trace.round_id}.json"
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            with self.rounds_jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self._append_dataset_rows(
                self.round_canonical_dir,
                self._round_canonical_rows(payload),
                schema=ROUND_CANONICAL_SCHEMA,
                partition_keys=("recorded_date", "round_id"),
            )
            round_trace_rows = self._round_trace_rows(payload)
            if round_trace_rows:
                self._append_dataset_rows(self.round_trace_dir, round_trace_rows, schema=ROUND_TRACE_SCHEMA)
            skill_canonical_rows, skill_trace_rows = self._skill_rows(payload.get("skill_traces", []))
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
            return payload
        recorded_at = _mtime_iso(path)
        return ensure_recorded_fields(payload, recorded_at=recorded_at)

    def _read_round_from_parquet(self, round_id: int) -> dict | None:
        rows = read_dataset_rows(
            self.round_canonical_dir,
            "select payload_json from read_parquet(?) where round_id = ? order by recorded_at desc limit 1",
            [round_id],
        )
        if not rows:
            return None
        return ensure_recorded_fields(json.loads(rows[0]["payload_json"]))

    def _list_rounds_from_parquet(self) -> list[dict]:
        rows = read_dataset_rows(
            self.round_canonical_dir,
            "select payload_json from read_parquet(?) order by round_id, recorded_at",
        )
        return [ensure_recorded_fields(json.loads(row["payload_json"])) for row in rows]

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
