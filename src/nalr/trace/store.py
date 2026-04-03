from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from nalr.schemas.models import CommandResult, RoundTrace, to_dict
from nalr.runtime.metadata import ensure_recorded_fields, iso_date, utc_now_iso


def _mtime_iso(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


class TraceStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.rounds_dir = self.root / "traces" / "rounds"
        self.rounds_jsonl_path = self.root / "traces" / "round_traces.jsonl"
        self.skills_dir = self.root / "traces" / "skills"
        self.skill_jsonl_path = self.skills_dir / "skill_traces.jsonl"
        self.commands_path = self.root / "traces" / "command_traces.json"
        self.commands_jsonl_path = self.root / "traces" / "command_traces.jsonl"
        self.parquet_dir = self.root / "traces" / "parquet"
        self.rounds_dir.mkdir(parents=True, exist_ok=True)
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        self.commands_path.parent.mkdir(parents=True, exist_ok=True)
        self.parquet_dir.mkdir(parents=True, exist_ok=True)
        if not self.commands_path.exists():
            self.commands_path.write_text("[]", encoding="utf-8")
        for path in (self.rounds_jsonl_path, self.skill_jsonl_path, self.commands_jsonl_path):
            if not path.exists():
                path.write_text("", encoding="utf-8")

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

    def read_round(self, round_id: int) -> dict:
        path = self.rounds_dir / f"round_{round_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"trace round {round_id} not found")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "session_id" in payload and "recorded_at" in payload and "recorded_date" in payload:
            return payload
        recorded_at = _mtime_iso(path)
        return ensure_recorded_fields(payload, recorded_at=recorded_at)

    def list_rounds(self) -> list[dict]:
        traces = []
        for path in sorted(self.rounds_dir.glob("round_*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if "session_id" not in payload or "recorded_at" not in payload or "recorded_date" not in payload:
                payload = ensure_recorded_fields(payload, recorded_at=_mtime_iso(path))
            traces.append(payload)
        return traces

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
            "applied": result.applied,
            "scope": result.scope,
            "delta": result.delta,
            "ttl": result.ttl,
            "risk_note": result.risk_note,
            "rollback_hint": result.rollback_hint,
            "operator_level": result.operator_level,
            "rollback_available": result.rollback_available,
            "before_state_hash": before_state_hash,
            "after_state_hash": after_state_hash,
        }
        payload.append(entry)
        self.commands_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        with self.commands_jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

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
        return self._read_jsonl(self.skill_jsonl_path)

    def list_command_traces(self) -> list[dict]:
        return self._read_jsonl(self.commands_jsonl_path)

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
