from __future__ import annotations

import json
from pathlib import Path

import duckdb

from nalr.schemas.models import CommandResult, RoundTrace, to_dict


class TraceStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.rounds_dir = self.root / "traces" / "rounds"
        self.jsonl_dir = self.root / "traces" / "jsonl"
        self.parquet_dir = self.root / "traces" / "parquet"
        self.jsonl_path = self.jsonl_dir / "rounds.jsonl"
        self.parquet_path = self.parquet_dir / "rounds.parquet"
        self.commands_path = self.root / "traces" / "command_traces.json"
        self.rounds_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_dir.mkdir(parents=True, exist_ok=True)
        self.parquet_dir.mkdir(parents=True, exist_ok=True)
        self.commands_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.commands_path.exists():
            self.commands_path.write_text("[]", encoding="utf-8")
        if not self.jsonl_path.exists():
            self.jsonl_path.write_text("", encoding="utf-8")

    def write_round(self, trace: RoundTrace) -> None:
        path = self.rounds_dir / f"round_{trace.round_id}.json"
        payload = to_dict(trace)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

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
        payload.append(
            {
                "command": command,
                "applied": result.applied,
                "scope": result.scope,
                "delta": result.delta,
                "ttl": result.ttl,
                "risk_note": result.risk_note,
                "rollback_hint": result.rollback_hint,
                "before_state_hash": before_state_hash,
                "after_state_hash": after_state_hash,
            }
        )
        self.commands_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def compact_rounds(self) -> dict:
        rounds = self.list_rounds()
        connection = duckdb.connect()
        try:
            connection.execute(
                f"""
                COPY (
                  SELECT
                    round_id,
                    scenario,
                    mode,
                    sampled_action,
                    conflict_score,
                    plausibility_fail_score,
                    provider,
                    model,
                    state_snapshot.budget_remaining AS budget_remaining
                  FROM read_json_auto('{self.jsonl_path}', format='newline_delimited')
                ) TO '{self.parquet_path}' (FORMAT PARQUET)
                """
            )
        finally:
            connection.close()
        return {
            "jsonl_path": str(self.jsonl_path),
            "parquet_path": str(self.parquet_path),
            "rows_written": len(rounds),
        }
