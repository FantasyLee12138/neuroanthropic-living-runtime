from __future__ import annotations

import json
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path
import shutil
from typing import Any
from uuid import uuid4

from nalr.runtime.metadata import utc_now_iso
from nalr.schemas.models import to_dict


def _parse_iso(value: str | None) -> datetime | None:
    normalized = str(value or "").strip()
    if not normalized:
        return None
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _default_truth_contract() -> dict[str, Any]:
    return {
        "layer": "projection",
        "projection_only": True,
        "authoritative": False,
        "source": "terminal_session_projection",
        "authoritative_source": "authoritative_runtime_state",
        "event_log_backed": False,
    }


def _normalize_truth_contract(payload: dict[str, Any] | None) -> dict[str, Any]:
    normalized = _default_truth_contract()
    if isinstance(payload, dict):
        normalized.update({key: value for key, value in payload.items() if key in normalized})
    normalized["layer"] = "projection"
    normalized["projection_only"] = bool(normalized.get("projection_only", True))
    normalized["authoritative"] = bool(normalized.get("authoritative", False))
    normalized["source"] = str(normalized.get("source") or "terminal_session_projection")
    normalized["authoritative_source"] = str(normalized.get("authoritative_source") or "authoritative_runtime_state")
    normalized["event_log_backed"] = bool(normalized.get("event_log_backed", False))
    return normalized


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
    truth_contract: dict[str, Any] = field(default_factory=_default_truth_contract)


class TerminalSessionStore:
    def __init__(self, runtime_dir: Path) -> None:
        self.runtime_dir = Path(runtime_dir)
        self.sessions_dir = self.runtime_dir / "terminal_sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.current_path = self.runtime_dir / "current_terminal_session.json"

    def _path_for(self, session_id: str) -> Path:
        return self.sessions_dir / f"{session_id}.json"

    def _write_json_atomic(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temp_path.replace(path)

    def _read_json(self, path: Path) -> dict[str, Any] | None:
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError:
            return None
        if not raw.strip():
            return None
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None
        return payload

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
        hydrated["truth_contract"] = _normalize_truth_contract(dict(hydrated.get("truth_contract", {}) or {}))
        return hydrated

    def _current_session_id(self) -> str | None:
        payload = self._read_json(self.current_path)
        if not isinstance(payload, dict):
            return None
        session_id = str(payload.get("session_id") or "").strip()
        return session_id or None

    def _session_updated_dt(self, session: TerminalSessionState) -> datetime | None:
        return _parse_iso(session.updated_at or session.created_at)

    def _session_age_seconds(self, session: TerminalSessionState, *, now_dt: datetime) -> int | None:
        updated_dt = self._session_updated_dt(session)
        if updated_dt is None:
            return None
        return max(0, int((now_dt - updated_dt).total_seconds()))

    def _session_has_recent_user_turn(self, session: TerminalSessionState) -> bool:
        for row in reversed(list(session.transcript_lines or [])):
            if not isinstance(row, dict):
                continue
            if str(row.get("kind") or "").strip() == "user" and str(row.get("text") or "").strip():
                return True
        return False

    def write(self, state: TerminalSessionState, *, mark_current: bool = True) -> TerminalSessionState:
        if not state.created_at:
            state.created_at = utc_now_iso()
        state.updated_at = utc_now_iso()
        state.transcript_mode = "compact" if state.transcript_mode == "compact" or state.compact else "full"
        state.compact = state.transcript_mode == "compact"
        state.truth_contract = _normalize_truth_contract(dict(state.truth_contract or {}))
        payload = to_dict(state)
        path = self._path_for(state.session_id)
        self._write_json_atomic(path, payload)
        if mark_current:
            self._write_json_atomic(
                self.current_path,
                {
                    "session_id": state.session_id,
                    "updated_at": state.updated_at,
                },
            )
        return state

    def read(self, session_id: str) -> TerminalSessionState:
        path = self._path_for(session_id)
        if not path.exists():
            raise FileNotFoundError(f"terminal session {session_id} not found")
        payload = self._read_json(path)
        if payload is None:
            raise FileNotFoundError(f"terminal session {session_id} not found")
        return TerminalSessionState(**self._hydrate_payload(payload))

    def read_current(self) -> TerminalSessionState:
        if not self.current_path.exists():
            raise FileNotFoundError("no current terminal session recorded")
        payload = self._read_json(self.current_path)
        if isinstance(payload, dict):
            if "cwd" in payload:
                return TerminalSessionState(**self._hydrate_payload(payload))
            session_id = str(payload.get("session_id") or "").strip()
            if session_id:
                try:
                    return self.read(session_id)
                except FileNotFoundError:
                    pass
        for session in self.list_sessions():
            if session.status == "active":
                return session
        raise FileNotFoundError("no current terminal session recorded")

    def select_fresh_active_session(
        self,
        *,
        max_age_seconds: int = 900,
        now_iso: str | None = None,
    ) -> tuple[TerminalSessionState | None, dict[str, Any]]:
        now_dt = _parse_iso(now_iso or utc_now_iso()) or datetime.now(timezone.utc)
        active_sessions = [session for session in self.list_sessions() if session.status == "active"]
        selected: TerminalSessionState | None = None
        reason = "no_fresh_active_session"
        current_session_id = self._current_session_id()
        max_age_seconds = max(0, int(max_age_seconds))

        def _metadata(session: TerminalSessionState | None, selection_reason: str) -> dict[str, Any]:
            age_seconds = self._session_age_seconds(session, now_dt=now_dt) if session is not None else None
            session_fresh = session is not None and age_seconds is not None and age_seconds < max_age_seconds
            return {
                "selected_session_id": session.session_id if session is not None else None,
                "selected_session_age_seconds": age_seconds,
                "selected_session_reason": selection_reason,
                "session_fresh": bool(session_fresh),
                "active_session_available": bool(active_sessions),
            }

        if current_session_id:
            try:
                current_session = self.read(current_session_id)
            except FileNotFoundError:
                current_session = None
            if current_session is not None and current_session.status == "active":
                current_age_seconds = self._session_age_seconds(current_session, now_dt=now_dt)
                if current_age_seconds is not None and current_age_seconds < max_age_seconds:
                    selected = current_session
                    reason = (
                        "current_pointer_recent_user_turn"
                        if self._session_has_recent_user_turn(current_session)
                        else "current_pointer"
                    )
                    return selected, _metadata(selected, reason)

        fresh_sessions: list[TerminalSessionState] = []
        for session in active_sessions:
            age_seconds = self._session_age_seconds(session, now_dt=now_dt)
            if age_seconds is None or age_seconds >= max_age_seconds:
                continue
            fresh_sessions.append(session)
        if not fresh_sessions:
            return None, _metadata(None, reason)

        fresh_sessions.sort(
            key=lambda session: (
                1 if self._session_has_recent_user_turn(session) else 0,
                self._session_updated_dt(session) or datetime.fromtimestamp(0, tz=timezone.utc),
            ),
            reverse=True,
        )
        selected = fresh_sessions[0]
        reason = "fallback_recent_user_turn" if self._session_has_recent_user_turn(selected) else "fallback_latest_active"
        return selected, _metadata(selected, reason)

    def list_sessions(self) -> list[TerminalSessionState]:
        sessions: list[TerminalSessionState] = []
        for path in sorted(self.sessions_dir.glob("*.json")):
            payload = self._read_json(path)
            if payload is None:
                continue
            sessions.append(TerminalSessionState(**self._hydrate_payload(payload)))
        sessions.sort(key=lambda item: item.updated_at or item.created_at or "", reverse=True)
        return sessions

    def projection_only_status(self) -> dict[str, Any]:
        violations: list[str] = []
        checked_sessions = 0
        for session in self.list_sessions():
            checked_sessions += 1
            contract = _normalize_truth_contract(dict(session.truth_contract or {}))
            if not bool(contract.get("projection_only", False)):
                violations.append(f"{session.session_id}:projection_only=false")
            if bool(contract.get("authoritative", False)):
                violations.append(f"{session.session_id}:authoritative=true")
        pointer_payload = self._read_json(self.current_path)
        if isinstance(pointer_payload, dict) and "cwd" in pointer_payload:
            checked_sessions += 1
            contract = _normalize_truth_contract(dict(pointer_payload.get("truth_contract", {}) or {}))
            if not bool(contract.get("projection_only", False)):
                violations.append("current_pointer:projection_only=false")
            if bool(contract.get("authoritative", False)):
                violations.append("current_pointer:authoritative=true")
        return {
            "checked_sessions": checked_sessions,
            "projection_only": not violations,
            "violations": violations,
            "projection_store_root": str(self.sessions_dir),
        }

    def clear(self) -> None:
        for path in self.sessions_dir.glob("*.json"):
            path.unlink(missing_ok=True)
        self.current_path.unlink(missing_ok=True)

    def clear_current(self, session_id: str | None = None) -> None:
        if session_id is not None and self._current_session_id() != session_id:
            return
        self.current_path.unlink(missing_ok=True)

    def delete(self, session_id: str) -> None:
        self._path_for(session_id).unlink(missing_ok=True)
        self.clear_current(session_id)

    def prune(self, *, statuses: set[str] | None = None) -> list[str]:
        allowed_statuses = {str(status).strip() for status in (statuses or set()) if str(status).strip()}
        removed: list[str] = []
        for session in self.list_sessions():
            if allowed_statuses and session.status not in allowed_statuses:
                continue
            self.delete(session.session_id)
            removed.append(session.session_id)
        return removed

    def clear_all(self) -> None:
        shutil.rmtree(self.sessions_dir, ignore_errors=True)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        if self.current_path.exists():
            self.current_path.unlink()
