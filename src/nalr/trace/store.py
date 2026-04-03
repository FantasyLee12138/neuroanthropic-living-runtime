from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import duckdb

from nalr.schemas.models import CommandResult, RoundTrace, to_dict
from nalr.runtime.metadata import ensure_recorded_fields, iso_date, utc_now_iso


def _mtime_iso(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


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

    def write_round(self, trace: RoundTrace) -> None:
        path = self.rounds_dir / f"round_{trace.round_id}.json"
        payload = ensure_recorded_fields(
            to_dict(trace),
            session_id=trace.session_id,
            recorded_at=trace.recorded_at,
        )
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        with self.rounds_jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        with self.skill_jsonl_path.open("a", encoding="utf-8") as handle:
            for skill_trace in payload.get("skill_traces", []):
                handle.write(json.dumps(skill_trace, ensure_ascii=False) + "\n")
        self._sync_parquet_mirror_safely(track_live_status=True)

    def read_round(self, round_id: int) -> dict:
        payload, _ = self.read_round_record(round_id)
        return payload

    def read_round_record(self, round_id: int) -> tuple[dict, str]:
        status = self.trace_storage_status()
        if status["trace_sync_state"] == "healthy":
            parquet_payload = self._read_round_from_parquet(round_id)
            if parquet_payload is not None:
                return parquet_payload, "parquet"
        else:
            json_payload = self._read_round_from_json(round_id)
            if json_payload is not None:
                return json_payload, "json_fallback"
        parquet_payload = self._read_round_from_parquet(round_id)
        if parquet_payload is not None:
            return parquet_payload, "parquet"
        json_payload = self._read_round_from_json(round_id)
        if json_payload is not None:
            return json_payload, "json_fallback"
        raise FileNotFoundError(f"trace round {round_id} not found")

    def list_rounds(self) -> list[dict]:
        status = self.trace_storage_status()
        round_canonical_path = self.parquet_dir / "round_canonical.parquet"
        if status["trace_sync_state"] == "healthy":
            parquet_rows = self._list_rounds_from_parquet()
            if parquet_rows or round_canonical_path.exists():
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
    ) -> None:
        effective_recorded_at = recorded_at or utc_now_iso()
        payload = ensure_recorded_fields(run_payload, session_id=session_id, recorded_at=effective_recorded_at)
        path = self.runs_dir / f"run_{payload['run_id']}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        with self.runs_jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._sync_parquet_mirror_safely(track_live_status=False)

    def read_run(self, run_id: str) -> dict:
        path = self.runs_dir / f"run_{run_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"run {run_id} not found")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "session_id" in payload and "recorded_at" in payload and "recorded_date" in payload:
            return payload
        return ensure_recorded_fields(payload, recorded_at=_mtime_iso(path))

    def append_step_trace(
        self,
        payload: dict,
        *,
        session_id: str,
        recorded_at: str | None = None,
    ) -> None:
        entry = ensure_recorded_fields(payload, session_id=session_id, recorded_at=recorded_at or utc_now_iso())
        with self.step_jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._sync_parquet_mirror_safely(track_live_status=False)

    def append_tool_trace(
        self,
        payload: dict,
        *,
        session_id: str,
        recorded_at: str | None = None,
    ) -> None:
        entry = ensure_recorded_fields(payload, session_id=session_id, recorded_at=recorded_at or utc_now_iso())
        with self.tool_jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._sync_parquet_mirror_safely(track_live_status=False)

    def list_run_steps(self, run_id: str) -> list[dict]:
        return [row for row in self._read_jsonl(self.step_jsonl_path) if row.get("run_id") == run_id]

    def list_run_tools(self, run_id: str) -> list[dict]:
        return [row for row in self._read_jsonl(self.tool_jsonl_path) if row.get("run_id") == run_id]

    def append_command(
        self,
        command: str,
        result: CommandResult,
        before_state_hash: str,
        after_state_hash: str,
        *,
        session_id: str,
        recorded_at: str | None = None,
    ) -> None:
        effective_recorded_at = recorded_at or utc_now_iso()
        payload = json.loads(self.commands_path.read_text(encoding="utf-8"))
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
        payload.append(entry)
        self.commands_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        with self.commands_jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._sync_parquet_mirror_safely(track_live_status=True)

    def append_repair_entry(
        self,
        payload: dict,
        *,
        session_id: str,
        recorded_at: str | None = None,
    ) -> None:
        entry = ensure_recorded_fields(payload, session_id=session_id, recorded_at=recorded_at or utc_now_iso())
        with self.repair_jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._sync_parquet_mirror_safely(track_live_status=True)

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
        status = self.trace_storage_status()
        path = self.parquet_dir / "skill_canonical.parquet"
        if status["trace_sync_state"] == "healthy":
            parquet_rows = self._list_payload_rows_from_parquet("skill_canonical.parquet")
            if parquet_rows or path.exists():
                return parquet_rows
        return self._read_jsonl(self.skill_jsonl_path)

    def list_command_traces(self) -> list[dict]:
        status = self.trace_storage_status()
        path = self.parquet_dir / "command_canonical.parquet"
        if status["trace_sync_state"] == "healthy":
            parquet_rows = self._list_payload_rows_from_parquet("command_canonical.parquet")
            if parquet_rows or path.exists():
                return parquet_rows
        return self._read_jsonl(self.commands_jsonl_path)

    def list_repair_entries(self) -> list[dict]:
        status = self.trace_storage_status()
        path = self.parquet_dir / "repair_ledger.parquet"
        if status["trace_sync_state"] == "healthy":
            parquet_rows = self._list_payload_rows_from_parquet("repair_ledger.parquet")
            if parquet_rows or path.exists():
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
        try:
            from nalr.trace.exporter import TraceExporter

            TraceExporter(self).export_parquet(overwrite=True)
        except Exception:
            raise

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
        parquet_live_ready = all(
            (self.parquet_dir / filename).exists()
            for filename in (
                "round_canonical.parquet",
                "skill_canonical.parquet",
                "command_canonical.parquet",
                "repair_ledger.parquet",
            )
        )
        return {
            "read_source_default": "parquet",
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
            return payload
        payload = json.loads(self.trace_sync_status_path.read_text(encoding="utf-8"))
        parquet_live_ready = all(
            (self.parquet_dir / filename).exists()
            for filename in (
                "round_canonical.parquet",
                "skill_canonical.parquet",
                "command_canonical.parquet",
                "repair_ledger.parquet",
            )
        )
        payload.setdefault("read_source_default", "parquet")
        payload.setdefault("trace_sync_state", "healthy")
        payload["parquet_live_ready"] = parquet_live_ready if payload["trace_sync_state"] == "healthy" else False
        payload.setdefault("degraded_reason", None)
        payload.setdefault("last_sync_at", None)
        return payload

    def mark_trace_sync_healthy(self) -> dict:
        payload = self.trace_storage_status()
        payload["trace_sync_state"] = "healthy"
        payload["parquet_live_ready"] = True
        payload["degraded_reason"] = None
        payload["last_sync_at"] = utc_now_iso()
        self._write_trace_sync_status(payload)
        return payload

    def mark_trace_sync_degraded(self, reason: str) -> dict:
        payload = self.trace_storage_status()
        payload["trace_sync_state"] = "degraded"
        payload["parquet_live_ready"] = False
        payload["degraded_reason"] = reason
        self._write_trace_sync_status(payload)
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
        rows = self._query_payload_rows(
            self.parquet_dir / "round_canonical.parquet",
            "select payload_json from read_parquet(?) where round_id = ? limit 1",
            [str(self.parquet_dir / "round_canonical.parquet"), round_id],
        )
        if not rows:
            return None
        return ensure_recorded_fields(json.loads(rows[0]["payload_json"]))

    def _list_rounds_from_parquet(self) -> list[dict]:
        rows = self._query_payload_rows(
            self.parquet_dir / "round_canonical.parquet",
            "select payload_json from read_parquet(?) order by round_id",
            [str(self.parquet_dir / "round_canonical.parquet")],
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
            self._sync_parquet_mirror_safely(track_live_status=True)
        if not path.exists():
            return []
        conn = duckdb.connect()
        try:
            cursor = conn.execute(query, params)
            columns = [item[0] for item in cursor.description]
            return [dict(zip(columns, row, strict=False)) for row in cursor.fetchall()]
        finally:
            conn.close()
