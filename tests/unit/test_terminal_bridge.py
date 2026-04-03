import pytest
from pathlib import Path

from nalr.runtime.controller import RuntimeController
from nalr.terminal_bridge.handlers import TerminalEventHandler
from nalr.terminal_bridge.protocol import ProtocolError, validate_inbound_event
from nalr.terminal_bridge.session import TerminalSessionStore


CONFIG_ROOT = Path(__file__).resolve().parents[2] / "config"


def test_protocol_validates_known_terminal_events():
    payload = validate_inbound_event({"type": "start_session", "session_id": "sess-1", "cwd": "/tmp/demo"})

    assert payload["type"] == "start_session"
    assert payload["session_id"] == "sess-1"
    control = validate_inbound_event(
        {"type": "control_command", "session_id": "sess-1", "command": "permissions", "value": "ask"}
    )
    assert control["value"] == "ask"

    with pytest.raises(ProtocolError):
        validate_inbound_event({"type": "user_turn", "session_id": "sess-1"})

    with pytest.raises(ProtocolError):
        validate_inbound_event({"type": "unknown_event", "session_id": "sess-1"})

    with pytest.raises(ProtocolError):
        validate_inbound_event({"type": "control_command", "session_id": "sess-1", "command": "permissions", "value": 1})


def test_user_turn_emits_read_only_run_sequence_and_persists_session_mapping(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "planner.py").write_text("VALUE = 1\n", encoding="utf-8")
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    handler = TerminalEventHandler(controller)

    started = handler.handle({"type": "start_session", "session_id": "sess-1", "cwd": str(tmp_path)})
    events = handler.handle({"type": "user_turn", "session_id": "sess-1", "text": "检查 planner.py 并规划下一步"})

    event_types = [item["type"] for item in events]
    final_event = next(item for item in events if item["type"] == "assistant_final")
    session_state = TerminalSessionStore(controller.runtime_dir).read("sess-1")

    assert started[0]["type"] == "session_started"
    assert any(item["type"] == "sidebar_snapshot" for item in started)
    started_snapshot = next(item for item in started if item["type"] == "sidebar_snapshot")
    assert started_snapshot["permission_mode"] == "plan"
    assert started_snapshot["pending_approval_count"] == 0
    assert started_snapshot["statusline"]["session_id"] == "sess-1"
    assert "run_status" in event_types
    assert "step_update" in event_types
    assert "tool_call" in event_types
    assert "tool_result" in event_types
    assert "sidebar_snapshot" in event_types
    assert event_types[-1] == "assistant_final"
    assert final_event["message"]
    run_snapshot = next(item for item in events if item["type"] == "sidebar_snapshot")
    assert run_snapshot["goal_summary"]
    assert run_snapshot["current_step"]
    assert run_snapshot["run_status"] in {"running", "paused"}
    assert run_snapshot["cognitive_snapshot"]["core_goal"]
    assert run_snapshot["cognitive_snapshot"]["current_intent"]
    assert "vital_signs" in run_snapshot["cognitive_snapshot"]
    assert "identity" in run_snapshot["cognitive_snapshot"]
    assert "authenticity" in run_snapshot["cognitive_snapshot"]
    assert session_state.active_run_id is not None
    assert session_state.status == "active"


def test_greeting_user_turn_uses_direct_chat_without_starting_run(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    handler = TerminalEventHandler(controller)

    handler.handle({"type": "start_session", "session_id": "sess-greet", "cwd": str(tmp_path)})
    events = handler.handle({"type": "user_turn", "session_id": "sess-greet", "text": "你好"})
    session_state = TerminalSessionStore(controller.runtime_dir).read("sess-greet")

    assert [item["type"] for item in events] == ["assistant_final"]
    assert "你好" in events[0]["message"]
    assert session_state.active_run_id is None


def test_identity_compound_user_turn_uses_direct_chat_without_starting_run(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    handler = TerminalEventHandler(controller)

    handler.handle({"type": "start_session", "session_id": "sess-identity", "cwd": str(tmp_path)})
    events = handler.handle({"type": "user_turn", "session_id": "sess-identity", "text": "你好，你是谁？你有名字吗？"})
    session_state = TerminalSessionStore(controller.runtime_dir).read("sess-identity")

    assert [item["type"] for item in events] == ["assistant_final"]
    assert "runtime_instance" not in events[0]["message"]
    assert "我是" in events[0]["message"]
    assert session_state.active_run_id is None


def test_task_run_uses_system_confirmation_message(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "planner.py").write_text("VALUE = 1\n", encoding="utf-8")
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    handler = TerminalEventHandler(controller)

    handler.handle({"type": "start_session", "session_id": "sess-system", "cwd": str(tmp_path)})
    events = handler.handle({"type": "user_turn", "session_id": "sess-system", "text": "检查 planner.py 并规划下一步"})

    final_event = next(item for item in events if item["type"] == "assistant_final")

    assert final_event["message"] == "已进入只读任务处理。可用 /status /why /steps /tools 查看进度。"


def test_task_run_uses_paused_system_confirmation_when_dirty_worktree_detected(tmp_path, monkeypatch):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "planner.py").write_text("VALUE = 1\n", encoding="utf-8")
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    handler = TerminalEventHandler(controller)

    monkeypatch.setattr(controller, "_dirty_worktree_snapshot", lambda: {"detected": True, "entries": ["M planner.py"]})

    handler.handle({"type": "start_session", "session_id": "sess-dirty", "cwd": str(tmp_path)})
    events = handler.handle({"type": "user_turn", "session_id": "sess-dirty", "text": "检查 planner.py 并规划下一步"})

    final_event = next(item for item in events if item["type"] == "assistant_final")
    run_event = next(item for item in events if item["type"] == "run_status")

    assert run_event["run"]["status"] == "paused"
    assert final_event["message"] == "任务已建立，但当前处于暂停状态。可用 /status /why 查看原因。"


def test_second_user_turn_interrupts_previous_run_before_starting_new_one(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "worker.py").write_text("VALUE = 1\n", encoding="utf-8")
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    handler = TerminalEventHandler(controller)

    handler.handle({"type": "start_session", "session_id": "sess-2", "cwd": str(tmp_path)})
    first_events = handler.handle({"type": "user_turn", "session_id": "sess-2", "text": "检查 worker.py"})
    first_run_id = TerminalSessionStore(controller.runtime_dir).read("sess-2").active_run_id

    second_events = handler.handle({"type": "user_turn", "session_id": "sess-2", "text": "重新检查 src 目录结构"})
    second_run_id = TerminalSessionStore(controller.runtime_dir).read("sess-2").active_run_id
    session_state = TerminalSessionStore(controller.runtime_dir).read("sess-2")

    assert any(item["type"] == "run_status" for item in first_events)
    assert any(item["type"] == "assistant_final" for item in first_events)
    assert any(item["type"] == "run_status" for item in second_events)
    assert any(item["type"] == "assistant_final" for item in second_events)
    assert second_run_id != first_run_id
    assert controller.run_status(first_run_id)["status"] == "interrupted"
    assert controller.run_status(first_run_id)["stop_reason"]["code"] == "interrupted_by_user"
    assert session_state.active_run_id == second_run_id
    assert session_state.last_run_id == second_run_id


def test_control_commands_expose_run_views_without_cil(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    handler = TerminalEventHandler(controller)

    handler.handle({"type": "start_session", "session_id": "sess-3", "cwd": str(tmp_path)})
    handler.handle({"type": "user_turn", "session_id": "sess-3", "text": "检查 app.py"})

    status_events = handler.handle({"type": "control_command", "session_id": "sess-3", "command": "status"})
    why_events = handler.handle({"type": "control_command", "session_id": "sess-3", "command": "why"})
    step_events = handler.handle({"type": "control_command", "session_id": "sess-3", "command": "steps"})
    tool_events = handler.handle({"type": "control_command", "session_id": "sess-3", "command": "tools"})

    assert any(item["type"] == "run_status" for item in status_events)
    assert any(item["type"] == "assistant_final" for item in why_events)
    assert any(item["type"] == "step_update" for item in step_events)
    assert any(item["type"] == "tool_result" for item in tool_events)
    assert "状态：" in next(item for item in status_events if item["type"] == "assistant_final")["message"]
    assert "当前目标：" in next(item for item in why_events if item["type"] == "assistant_final")["message"]
    assert "最近工具：" in next(item for item in tool_events if item["type"] == "assistant_final")["message"]


def test_control_command_mode_permissions_state_and_compact(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    handler = TerminalEventHandler(controller)
    session_store = TerminalSessionStore(controller.runtime_dir)

    handler.handle({"type": "start_session", "session_id": "sess-cmd", "cwd": str(tmp_path)})

    mode_events = handler.handle(
        {"type": "control_command", "session_id": "sess-cmd", "command": "mode", "value": "plan"}
    )
    permission_events = handler.handle(
        {"type": "control_command", "session_id": "sess-cmd", "command": "permissions", "value": "ask"}
    )
    compact_events = handler.handle(
        {"type": "control_command", "session_id": "sess-cmd", "command": "compact", "value": "on"}
    )
    state_events = handler.handle({"type": "control_command", "session_id": "sess-cmd", "command": "state"})
    invalid_permission_events = handler.handle(
        {"type": "control_command", "session_id": "sess-cmd", "command": "permissions", "value": "nope"}
    )
    persisted = session_store.read("sess-cmd")

    assert any(item["type"] == "assistant_final" for item in mode_events)
    assert any(item["type"] == "assistant_final" for item in permission_events)
    assert any(item["type"] == "assistant_final" for item in compact_events)
    assert any(item["type"] == "sidebar_snapshot" for item in state_events)
    state_final = next(item for item in state_events if item["type"] == "assistant_final")
    assert state_final["payload"]["session"]["mode"] == "plan"
    assert state_final["payload"]["session"]["permission_mode"] == "ask"
    assert state_final["payload"]["session"]["compact"] is True
    assert state_final["payload"]["cognitive_snapshot"]["core_goal"]
    assert "vital_signs" in state_final["payload"]["cognitive_snapshot"]
    state_snapshot = next(item for item in state_events if item["type"] == "sidebar_snapshot")
    assert state_snapshot["permission_mode"] == "ask"
    assert state_snapshot["statusline"]["permission_mode"] == "ask"
    assert state_snapshot["cognitive_snapshot"]["authenticity"]["summary"]
    assert invalid_permission_events[0]["type"] == "error"
    assert persisted.permission_mode == "ask"
    assert persisted.compact is True


def test_ask_permissions_emit_approval_request_and_persist_pending_approvals(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "worker.py").write_text("VALUE = 1\n", encoding="utf-8")
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    handler = TerminalEventHandler(controller)
    session_store = TerminalSessionStore(controller.runtime_dir)

    handler.handle({"type": "start_session", "session_id": "sess-ask", "cwd": str(tmp_path)})
    handler.handle({"type": "control_command", "session_id": "sess-ask", "command": "permissions", "value": "ask"})
    turn_events = handler.handle({"type": "user_turn", "session_id": "sess-ask", "text": "检查 worker.py"})
    approval = next(item for item in turn_events if item["type"] == "approval_request")

    assert approval["mode"] == "ask"
    assert approval["status"] == "pending"
    assert approval["run_id"]
    assert approval["tool"] == "repo_scan"
    assert approval["actions"] == ["approve", "reject"]
    assert approval["risk_level"] == "medium"
    assert approval["summary"]
    assert approval["action_preview"]
    persisted = session_store.read("sess-ask")
    assert persisted.permission_mode == "ask"
    assert persisted.approvals_pending
    assert persisted.approvals_pending[0]["call_id"] == approval["call_id"]

    approved_events = handler.handle(
        {
            "type": "approve",
            "session_id": "sess-ask",
            "call_id": approval["call_id"],
            "approved": True,
        }
    )
    approved_final = next(item for item in approved_events if item["type"] == "assistant_final")
    assert approved_final["payload"]["approval"]["status"] == "approved"
    assert approved_final["payload"]["approval"]["approved"] is True
    assert any(item["type"] == "sidebar_snapshot" for item in approved_events)
    approved_persisted = session_store.read("sess-ask")
    match = next(item for item in approved_persisted.approvals_pending if item["call_id"] == approval["call_id"])
    assert match["status"] == "approved"


def test_start_session_keeps_permission_mode_and_pending_approvals(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "persist.py").write_text("VALUE = 1\n", encoding="utf-8")
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    handler = TerminalEventHandler(controller)
    session_store = TerminalSessionStore(controller.runtime_dir)

    handler.handle({"type": "start_session", "session_id": "sess-persist", "cwd": str(tmp_path)})
    handler.handle({"type": "control_command", "session_id": "sess-persist", "command": "permissions", "value": "ask"})
    turn_events = handler.handle({"type": "user_turn", "session_id": "sess-persist", "text": "检查 persist.py"})
    pending = next(item for item in turn_events if item["type"] == "approval_request")["call_id"]
    before_restart = session_store.read("sess-persist")

    restarted = handler.handle({"type": "start_session", "session_id": "sess-persist", "cwd": str(tmp_path)})
    after_restart = session_store.read("sess-persist")

    assert any(item["type"] == "session_started" for item in restarted)
    assert before_restart.permission_mode == "ask"
    assert any(item["call_id"] == pending for item in before_restart.approvals_pending)
    assert after_restart.permission_mode == "ask"
    assert any(item["call_id"] == pending for item in after_restart.approvals_pending)


def test_close_session_marks_session_ended(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    handler = TerminalEventHandler(controller)

    handler.handle({"type": "start_session", "session_id": "sess-4", "cwd": str(tmp_path)})
    events = handler.handle({"type": "close_session", "session_id": "sess-4"})
    session_state = TerminalSessionStore(controller.runtime_dir).read("sess-4")

    assert events[0]["type"] == "session_ended"
    assert session_state.status == "ended"


def test_why_summary_labels_bootstrap_planning_without_runtime_evidence(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "planner.py").write_text("VALUE = 1\n", encoding="utf-8")
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    handler = TerminalEventHandler(controller)

    handler.handle({"type": "start_session", "session_id": "sess-why", "cwd": str(tmp_path)})
    handler.handle({"type": "user_turn", "session_id": "sess-why", "text": "检查 planner.py 并规划下一步"})

    why_events = handler.handle({"type": "control_command", "session_id": "sess-why", "command": "why"})
    why_message = next(item for item in why_events if item["type"] == "assistant_final")["message"]

    assert "说明类型：bootstrap规划" in why_message
    assert "真实性证据：暂无轮次证据" in why_message


def test_control_command_dream_runs_manual_dream_without_active_run(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    handler = TerminalEventHandler(controller)

    handler.handle({"type": "start_session", "session_id": "sess-dream", "cwd": str(tmp_path)})
    events = handler.handle({"type": "control_command", "session_id": "sess-dream", "command": "dream", "value": "tea"})

    final_event = next(item for item in events if item["type"] == "assistant_final")
    payload = final_event["payload"]

    assert any(item["type"] == "sidebar_snapshot" for item in events)
    assert "已手动触发 Dream" in final_event["message"]
    assert payload["trace"]["mode"] == "sleep"
    assert payload["trace"]["cue"] == "tea"
    assert payload["dream_run_id"].startswith("dream-")
