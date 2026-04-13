from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


CONFIG_ROOT = Path(__file__).resolve().parents[2] / "config"


def _send(handle, payload: dict) -> None:
    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    handle.flush()


def _read_until(handle, event_type: str) -> list[dict]:
    rows: list[dict] = []
    while True:
        line = handle.readline()
        assert line, f"bridge exited before emitting {event_type}"
        payload = json.loads(line)
        rows.append(payload)
        if payload["type"] == event_type:
            return rows


def _read_exactly(handle, count: int) -> list[dict]:
    rows: list[dict] = []
    while len(rows) < count:
        line = handle.readline()
        assert line, f"bridge exited before emitting {count} events"
        rows.append(json.loads(line))
    return rows


def test_stdio_bridge_runs_terminal_session_and_emits_events(tmp_path):
    env = os.environ.copy()
    env["NALR_HOME"] = str(tmp_path / ".alive")
    env["NALR_CONFIG_DIR"] = str(CONFIG_ROOT)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "src")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "layout.py").write_text("VALUE = 1\n", encoding="utf-8")

    process = subprocess.Popen(
        [sys.executable, "-m", "nalr.terminal_bridge.bridge"],
        cwd=tmp_path,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        assert process.stdin is not None
        assert process.stdout is not None
        _send(process.stdin, {"type": "start_session", "session_id": "sess-stdio", "cwd": str(tmp_path)})
        started = _read_exactly(process.stdout, 2)
        _send(
            process.stdin,
            {"type": "control_command", "session_id": "sess-stdio", "command": "permissions", "value": "ask"},
        )
        permissions_events = _read_until(process.stdout, "assistant_final")
        _send(process.stdin, {"type": "user_turn", "session_id": "sess-stdio", "text": "总结这个仓库结构"})
        turn_events = _read_until(process.stdout, "approval_request")
        turn_snapshot = _read_until(process.stdout, "sidebar_snapshot")[-1]
        approval = next((item for item in turn_events if item["type"] == "approval_request"), None)
        assert approval is not None
        assert all(item["type"] != "tool_result" for item in turn_events)
        _send(
            process.stdin,
            {
                "type": "approve",
                "session_id": "sess-stdio",
                "call_id": approval["call_id"],
                "approved": True,
            },
        )
        approve_events = _read_until(process.stdout, "assistant_final")
        _send(process.stdin, {"type": "control_command", "session_id": "sess-stdio", "command": "status"})
        status_events = _read_until(process.stdout, "assistant_final")
        _send(process.stdin, {"type": "control_command", "session_id": "sess-stdio", "command": "tools"})
        tool_events = _read_until(process.stdout, "assistant_final")
        _send(process.stdin, {"type": "control_command", "session_id": "sess-stdio", "command": "state"})
        state_events = _read_until(process.stdout, "assistant_final")
        _send(process.stdin, {"type": "close_session", "session_id": "sess-stdio"})
        ended = _read_until(process.stdout, "session_ended")
    finally:
        process.terminate()
        process.wait(timeout=5)

    started_types = [item["type"] for item in started]
    permissions_types = [item["type"] for item in permissions_events]
    turn_types = [item["type"] for item in turn_events]
    approve_types = [item["type"] for item in approve_events]
    status_types = [item["type"] for item in status_events]
    tool_types = [item["type"] for item in tool_events]
    state_types = [item["type"] for item in state_events]
    ended_types = [item["type"] for item in ended]

    assert "session_started" in started_types
    started_snapshot = next(item for item in started if item["type"] == "sidebar_snapshot")
    assert started_snapshot["permission_mode"] == "plan"
    assert started_snapshot["statusline"]["cwd"] == str(tmp_path)
    assert started_snapshot["controlled_learning"]["learning_mode"] == "guided-learn"
    assert started_snapshot["controlled_learning"]["trace_external_learning"] is True
    assert permissions_types[-1] == "assistant_final"
    permission_snapshot = next(item for item in permissions_events if item["type"] == "sidebar_snapshot")
    assert permission_snapshot["permission_mode"] == "ask"
    assert "run_status" in turn_types
    assert "step_update" in turn_types
    assert "tool_call" in turn_types
    assert "approval_request" in turn_types
    assert turn_snapshot["pending_approval_count"] >= 1
    assert "tool_result" in [item["type"] for item in approve_events]
    assert turn_snapshot["goal_summary"]
    assert turn_snapshot["current_step"]
    assert turn_snapshot["ui_actions"]["primary"]
    assert turn_types[-1] == "approval_request"
    assert approval["choices"][0]["id"] == "approve"
    assert "sidebar_snapshot" in approve_types
    assert approve_types[-1] == "assistant_final"
    assert "run_status" in status_types
    assert status_types[-1] == "assistant_final"
    assert "tool_result" in tool_types
    assert tool_types[-1] == "assistant_final"
    assert "sidebar_snapshot" in state_types
    state_snapshot = next(item for item in state_events if item["type"] == "sidebar_snapshot")
    assert state_snapshot["controlled_learning"]["learning_mode"] == "guided-learn"
    assert state_types[-1] == "assistant_final"
    assert "session_ended" in ended_types
