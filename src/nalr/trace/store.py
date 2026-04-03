from __future__ import annotations

import json
from pathlib import Path

from nalr.schemas.models import CommandResult, RoundTrace, to_dict


class TraceStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.rounds_dir = self.root / "traces" / "rounds"
        self.rounds_jsonl_path = self.root / "traces" / "round_traces.jsonl"
        self.skills_dir = self.root / "traces" / "skills"
        self.skill_jsonl_path = self.skills_dir / "skill_traces.jsonl"
        self.commands_path = self.root / "traces" / "command_traces.json"
        self.commands_jsonl_path = self.root / "traces" / "command_traces.jsonl"
        self.rounds_dir.mkdir(parents=True, exist_ok=True)
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        self.commands_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.commands_path.exists():
            self.commands_path.write_text("[]", encoding="utf-8")
        for path in (self.rounds_jsonl_path, self.skill_jsonl_path, self.commands_jsonl_path):
            if not path.exists():
                path.write_text("", encoding="utf-8")

    def write_round(self, trace: RoundTrace) -> None:
        path = self.rounds_dir / f"round_{trace.round_id}.json"
        payload = to_dict(trace)
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
        return json.loads(path.read_text(encoding="utf-8"))

    def list_rounds(self) -> list[dict]:
        traces = []
        for path in sorted(self.rounds_dir.glob("round_*.json")):
            traces.append(json.loads(path.read_text(encoding="utf-8")))
        return traces

    def append_command(self, command: str, result: CommandResult, before_state_hash: str, after_state_hash: str) -> None:
        payload = json.loads(self.commands_path.read_text(encoding="utf-8"))
        entry = {
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
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def list_skill_traces(self) -> list[dict]:
        return self._read_jsonl(self.skill_jsonl_path)

    def skill_stats(self, skill_name: str | None = None) -> dict:
        rows = self.list_skill_traces()
        if skill_name:
            rows = [row for row in rows if row["skill_name"] == skill_name]
        skills: dict[str, dict] = {}
        for row in rows:
            bucket = skills.setdefault(
                row["skill_name"],
                {"count": 0, "degraded_count": 0, "average_latency_ms": 0.0},
            )
            bucket["count"] += 1
            bucket["degraded_count"] += int(bool(row.get("degraded")))
            bucket["average_latency_ms"] += row.get("latency_ms", 0)
        for bucket in skills.values():
            bucket["average_latency_ms"] = round(bucket["average_latency_ms"] / bucket["count"], 2)
        return {"total_calls": len(rows), "skills": skills}
