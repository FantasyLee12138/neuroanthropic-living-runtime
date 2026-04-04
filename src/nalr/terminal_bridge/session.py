from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nalr.runtime.metadata import utc_now_iso
from nalr.schemas.models import to_dict


@dataclass
class TerminalSessionState:
    session_id: str
    cwd: str
    status: str = "active"
    mode: str = "plan"
    permission_mode: str = "plan"
    compact: bool = False
    active_run_id: str | None = None
    last_run_id: str | None = None
    created_at: str = ""
    updated_at: str = ""
    approvals_pending: list[dict[str, Any]] = field(default_factory=list)
    transcript_lines: list[dict[str, Any]] = field(default_factory=list)
    tool_timeline: list[dict[str, Any]] = field(default_factory=list)
    transcript_mode: str = "full"


class TerminalSessionStore:
    def __init__(self, runtime_dir: Path) -> None:
        self.runtime_dir = Path(runtime_dir)
        self.sessions_dir = self.runtime_dir / "terminal_sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.current_path = self.runtime_dir / "current_terminal_session.json"

    def _path_for(self, session_id: str) -> Path:
        return self.sessions_dir / f"{session_id}.json"

    def _hydrate_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        hydrated = dict(payload)
        hydrated.setdefault("approvals_pending", [])
        hydrated.setdefault("transcript_lines", [])
        hydrated.setdefault("tool_timeline", [])
        transcript_mode = str(hydrated.get("transcript_mode") or "")
        if transcript_mode not in {"full", "compact"}:
            transcript_mode = "compact" if bool(hydrated.get("compact")) else "full"
        hydrated["transcript_mode"] = transcript_mode
        hydrated["compact"] = transcript_mode == "compact"
        return hydrated

    def write(self, state: TerminalSessionState, *, mark_current: bool = True) -> TerminalSessionState:
        if not state.created_at:
            state.created_at = utc_now_iso()
        state.updated_at = utc_now_iso()
        state.transcript_mode = "compact" if state.transcript_mode == "compact" or state.compact else "full"
        state.compact = state.transcript_mode == "compact"
        payload = to_dict(state)
        path = self._path_for(state.session_id)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        if mark_current:
            self.current_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return state

    def read(self, session_id: str) -> TerminalSessionState:
        path = self._path_for(session_id)
        if not path.exists():
            raise FileNotFoundError(f"terminal session {session_id} not found")
        return TerminalSessionState(**self._hydrate_payload(json.loads(path.read_text(encoding="utf-8"))))

    def read_current(self) -> TerminalSessionState:
        if not self.current_path.exists():
            raise FileNotFoundError("no current terminal session recorded")
        return TerminalSessionState(**self._hydrate_payload(json.loads(self.current_path.read_text(encoding="utf-8"))))
