from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class DreamStore:
    def __init__(self, root: Path) -> None:
        self.root = root / "dream"
        self.runs_dir = self.root / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.runs_jsonl = self.root / "dream_runs.jsonl"
        if not self.runs_jsonl.exists():
            self.runs_jsonl.write_text("", encoding="utf-8")

    def write_run(self, payload: dict[str, Any]) -> str:
        run_id = str(payload["trace"]["dream_run_id"])
        path = self.runs_dir / f"{run_id}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        with self.runs_jsonl.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return f"dream://runs/{run_id}"

    def list_runs(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for path in sorted(self.runs_dir.glob("dream-*.json")):
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        return rows

    def read_run(self, run_ref: str | None = None) -> dict[str, Any]:
        if run_ref in {None, "last"}:
            rows = self.list_runs()
            if not rows:
                raise FileNotFoundError("no dream runs recorded yet")
            return rows[-1]
        path = self.runs_dir / f"{run_ref}.json"
        if not path.exists():
            raise FileNotFoundError(f"dream run {run_ref} not found")
        return json.loads(path.read_text(encoding="utf-8"))

    def metrics(self) -> dict[str, Any]:
        runs = self.list_runs()
        runs_by_trigger: dict[str, int] = {}
        runs_by_mode: dict[str, int] = {}
        identity_approvals = 0
        for row in runs:
            trace = row.get("trace", {})
            guard = row.get("guard_summary", {})
            effect = row.get("effect_summary", {})
            trigger = str(trace.get("trigger", "unknown"))
            mode = str(trace.get("mode", "unknown"))
            runs_by_trigger[trigger] = runs_by_trigger.get(trigger, 0) + 1
            runs_by_mode[mode] = runs_by_mode.get(mode, 0) + 1
            if "limited_identity_drift" in effect.get("applied_types", []) and guard.get("approved"):
                identity_approvals += 1
        latest = runs[-1]["trace"]["dream_run_id"] if runs else None
        return {
            "total_runs": len(runs),
            "runs_by_trigger": runs_by_trigger,
            "runs_by_mode": runs_by_mode,
            "identity_proposal_approvals": identity_approvals,
            "latest_run_id": latest,
        }
