import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from nalr.terminal_bridge.session import TerminalSessionState, TerminalSessionStore


def _state(session_id: str, *, status: str = "active") -> TerminalSessionState:
    return TerminalSessionState(session_id=session_id, cwd="/tmp/demo", status=status)


def _rewrite_session(
    store: TerminalSessionStore,
    session_id: str,
    *,
    updated_at: str,
    transcript_lines: list[dict] | None = None,
) -> None:
    payload = json.loads(store._path_for(session_id).read_text(encoding="utf-8"))
    payload["updated_at"] = updated_at
    payload["created_at"] = updated_at
    if transcript_lines is not None:
        payload["transcript_lines"] = transcript_lines
    store._path_for(session_id).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_write_marks_current_session_with_pointer_payload(tmp_path):
    store = TerminalSessionStore(tmp_path)

    written = store.write(_state("sess-pointer"))
    current_payload = json.loads(store.current_path.read_text(encoding="utf-8"))

    assert current_payload == {
        "session_id": "sess-pointer",
        "updated_at": written.updated_at,
    }


def test_read_current_accepts_legacy_full_session_payload(tmp_path):
    store = TerminalSessionStore(tmp_path)
    expected = store.write(_state("sess-legacy"))
    legacy_payload = {
        "session_id": expected.session_id,
        "cwd": expected.cwd,
        "status": expected.status,
        "mode": expected.mode,
        "permission_mode": expected.permission_mode,
        "compact": expected.compact,
        "active_run_id": expected.active_run_id,
        "last_run_id": expected.last_run_id,
        "created_at": expected.created_at,
        "updated_at": expected.updated_at,
        "approvals_pending": expected.approvals_pending,
        "transcript_lines": expected.transcript_lines,
        "tool_timeline": expected.tool_timeline,
        "transcript_mode": expected.transcript_mode,
    }
    store.current_path.write_text(json.dumps(legacy_payload, ensure_ascii=False), encoding="utf-8")

    current = store.read_current()

    assert current.session_id == "sess-legacy"
    assert current.cwd == "/tmp/demo"


def test_read_current_falls_back_to_active_session_when_pointer_file_is_corrupted(tmp_path):
    store = TerminalSessionStore(tmp_path)
    store.write(_state("sess-active"))
    store.write(_state("sess-ended", status="ended"), mark_current=False)
    store.current_path.write_text("{", encoding="utf-8")

    current = store.read_current()

    assert current.session_id == "sess-active"


def test_list_sessions_skips_corrupted_payloads(tmp_path):
    store = TerminalSessionStore(tmp_path)
    store.write(_state("sess-good"))
    (store.sessions_dir / "sess-bad.json").write_text("{", encoding="utf-8")

    sessions = store.list_sessions()

    assert [session.session_id for session in sessions] == ["sess-good"]


def test_read_current_does_not_surface_json_decode_errors_during_parallel_access(tmp_path):
    store = TerminalSessionStore(tmp_path)
    for index in range(3):
        store.write(_state(f"sess-{index}"), mark_current=index == 0)

    errors: list[Exception] = []

    def writer(index: int) -> None:
        try:
            store.write(_state(f"sess-{index % 3}"))
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    def reader() -> None:
        try:
            store.read_current()
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    with ThreadPoolExecutor(max_workers=8) as executor:
        for index in range(100):
            executor.submit(writer, index)
            executor.submit(reader)

    assert errors == []


def test_terminal_session_store_marks_sessions_as_projection_only(tmp_path):
    store = TerminalSessionStore(tmp_path)

    store.write(_state("sess-projection"))
    loaded = store.read("sess-projection")
    projection_status = store.projection_only_status()

    assert loaded.truth_contract["projection_only"] is True
    assert loaded.truth_contract["authoritative"] is False
    assert loaded.truth_contract["source"] == "terminal_session_projection"
    assert projection_status["projection_only"] is True
    assert projection_status["violations"] == []


def test_terminal_session_store_projection_status_detects_shadow_truth_payload(tmp_path):
    store = TerminalSessionStore(tmp_path)
    store.write(_state("sess-shadow"))

    payload = json.loads(store._path_for("sess-shadow").read_text(encoding="utf-8"))
    payload["truth_contract"] = {
        "projection_only": False,
        "authoritative": True,
        "source": "shadow_truth_cache",
    }
    store._path_for("sess-shadow").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    projection_status = store.projection_only_status()

    assert projection_status["projection_only"] is False
    assert any("sess-shadow" in item for item in projection_status["violations"])


def test_delete_session_removes_current_pointer_and_prune_clears_inactive(tmp_path):
    store = TerminalSessionStore(tmp_path)
    store.write(_state("sess-active"))
    store.write(_state("sess-ended", status="ended"), mark_current=False)
    store.write(_state("sess-detached", status="detached"), mark_current=False)

    removed = store.prune(statuses={"ended", "detached"})

    assert set(removed) == {"sess-ended", "sess-detached"}
    assert [session.session_id for session in store.list_sessions()] == ["sess-active"]

    store.delete("sess-active")

    assert store.current_path.exists() is False


def test_select_fresh_active_session_skips_stale_pointer_and_prefers_recent_user_turn(tmp_path):
    store = TerminalSessionStore(tmp_path)
    store.write(_state("sess-stale"))
    store.write(_state("sess-assistant-only"), mark_current=False)
    store.write(_state("sess-user-recent"), mark_current=False)

    _rewrite_session(
        store,
        "sess-stale",
        updated_at="2026-04-10T00:00:00Z",
        transcript_lines=[{"kind": "user", "text": "旧会话"}],
    )
    _rewrite_session(
        store,
        "sess-assistant-only",
        updated_at="2026-04-10T00:14:40Z",
        transcript_lines=[{"kind": "assistant", "text": "只有输出"}],
    )
    _rewrite_session(
        store,
        "sess-user-recent",
        updated_at="2026-04-10T00:14:20Z",
        transcript_lines=[{"kind": "user", "text": "我在这里"}],
    )
    store.current_path.write_text(
        json.dumps({"session_id": "sess-stale", "updated_at": "2026-04-10T00:00:00Z"}, ensure_ascii=False),
        encoding="utf-8",
    )

    current, metadata = store.select_fresh_active_session(
        max_age_seconds=900,
        now_iso="2026-04-10T00:15:00Z",
    )

    assert current is not None
    assert current.session_id == "sess-user-recent"
    assert metadata["selected_session_reason"] == "fallback_recent_user_turn"
    assert metadata["selected_session_age_seconds"] == 40
    assert metadata["session_fresh"] is True


def test_select_fresh_active_session_returns_none_when_all_active_sessions_are_stale(tmp_path):
    store = TerminalSessionStore(tmp_path)
    store.write(_state("sess-stale"))
    _rewrite_session(store, "sess-stale", updated_at="2026-04-10T00:00:00Z")

    current, metadata = store.select_fresh_active_session(
        max_age_seconds=900,
        now_iso="2026-04-10T00:30:00Z",
    )

    assert current is None
    assert metadata["selected_session_id"] is None
    assert metadata["selected_session_reason"] == "no_fresh_active_session"
    assert metadata["session_fresh"] is False
