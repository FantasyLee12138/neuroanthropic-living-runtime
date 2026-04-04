import pytest
from pathlib import Path

from nalr.runtime.controller import RuntimeController
from nalr.terminal_bridge.handlers import TerminalEventHandler
from nalr.terminal_bridge.protocol import ProtocolError, validate_inbound_event
from nalr.terminal_bridge.session import TerminalSessionStore


CONFIG_ROOT = Path(__file__).resolve().parents[2] / "config"


def _seed_entropy(controller: RuntimeController) -> None:
    controller.entropy_pool.ingest_bytes(bytes([index % 256 for index in range(4096)]), source="test_qrng")


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
    assert session_state.transcript_lines[0] == {"kind": "user", "text": "检查 planner.py 并规划下一步"}
    assert session_state.transcript_lines[1] == {"kind": "assistant", "text": "已进入只读任务处理。可用 /status /why /steps /tools 查看进度。"}
    assert any(item["kind"] == "system" and item["text"].startswith("Step: ") for item in session_state.transcript_lines)
    assert any(item["kind"] == "call" and item["tool"] == "repo_scan" for item in session_state.tool_timeline)
    assert any(item["kind"] == "result" and item["tool"] == "repo_scan" for item in session_state.tool_timeline)


def test_greeting_user_turn_uses_direct_chat_without_starting_run(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    _seed_entropy(controller)
    handler = TerminalEventHandler(controller)

    handler.handle({"type": "start_session", "session_id": "sess-greet", "cwd": str(tmp_path)})
    events = handler.handle({"type": "user_turn", "session_id": "sess-greet", "text": "你好"})
    session_state = TerminalSessionStore(controller.runtime_dir).read("sess-greet")

    assert [item["type"] for item in events] == ["assistant_token", "assistant_final"]
    assert "你好" in events[-1]["message"]
    assert session_state.active_run_id is None
    assert session_state.transcript_lines == [
        {"kind": "user", "text": "你好"},
        {"kind": "assistant", "text": events[0]["message"]},
    ]


def test_identity_compound_user_turn_uses_direct_chat_without_starting_run(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    _seed_entropy(controller)
    handler = TerminalEventHandler(controller)

    handler.handle({"type": "start_session", "session_id": "sess-identity", "cwd": str(tmp_path)})
    events = handler.handle({"type": "user_turn", "session_id": "sess-identity", "text": "你好，你是谁？你有名字吗？"})
    session_state = TerminalSessionStore(controller.runtime_dir).read("sess-identity")

    assert [item["type"] for item in events] == ["assistant_token", "assistant_final"]
    assert "runtime_instance" not in events[-1]["message"]
    assert "我是" in events[-1]["message"]
    assert session_state.active_run_id is None


def test_fast_chat_user_turn_streams_tokens_without_starting_run(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    handler = TerminalEventHandler(controller)

    handler.handle({"type": "start_session", "session_id": "sess-fast", "cwd": str(tmp_path)})

    class _Plan:
        route = "fast_chat"

    controller.plan_turn = lambda text, **kwargs: _Plan()
    monkeypatch.setattr(
        controller,
        "stream_fast_chat_turn",
        lambda plan: (
            ["你好", "，我是当前运行体实例。"],
            {
                "route": "fast_chat",
                "assistant_final": "你好，我是当前运行体实例。",
                "payload": {"stream_deltas": ["你好", "，我是当前运行体实例。"]},
            },
        ),
    )

    events = handler.handle({"type": "user_turn", "session_id": "sess-fast", "text": "你是谁？"})

    assert [item["type"] for item in events] == ["assistant_token", "assistant_token", "assistant_final"]
    assert "".join(item["delta"] for item in events if item["type"] == "assistant_token") == "你好，我是当前运行体实例。"


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


def test_user_turn_uses_plan_and_execute_paths_instead_of_legacy_route_and_trace_refetch(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    handler = TerminalEventHandler(controller)

    handler.handle({"type": "start_session", "session_id": "sess-plan", "cwd": str(tmp_path)})

    monkeypatch.setattr(controller, "probe_terminal_route", lambda *args, **kwargs: pytest.fail("legacy route probe should not be used"))
    monkeypatch.setattr(controller, "start_run", lambda *args, **kwargs: pytest.fail("legacy start_run path should not be used"))
    monkeypatch.setattr(controller, "tick", lambda *args, **kwargs: pytest.fail("legacy tick path should not be used"))
    monkeypatch.setattr(controller, "explain_run", lambda *args, **kwargs: pytest.fail("explain_run should not be called during streamed turn"))
    monkeypatch.setattr(controller, "run_steps", lambda *args, **kwargs: pytest.fail("run_steps should not be called during streamed turn"))
    monkeypatch.setattr(controller, "run_tools", lambda *args, **kwargs: pytest.fail("run_tools should not be called during streamed turn"))

    class _Plan:
        route = "task_run"

    controller.plan_turn = lambda text, **kwargs: _Plan()
    controller.execute_turn = lambda plan, **kwargs: {
        "route": "task_run",
        "run": {
            "run_id": "run-1",
            "status": "running",
            "goal": "检查 app.py",
            "goal_summary": "检查 app.py",
            "current_step_id": "step-1",
            "current_step": {"node_id": "step-1", "title": "阅读 app.py", "tool_choice": "repo_scan"},
            "dirty_worktree_detected": False,
            "commit_permission_required": True,
            "stop_reason": {},
            "pending_steps": 1,
            "completed_steps": 0,
            "last_tool_result": {"tool_name": "repo_scan", "summary": "scanned 1 files"},
            "created_at": "2026-04-04T00:00:00Z",
            "updated_at": "2026-04-04T00:00:00Z",
        },
        "assistant_preamble": "已进入只读任务处理。可用 /status /why /steps /tools 查看进度。",
        "assistant_final": "已进入只读任务处理。可用 /status /why /steps /tools 查看进度。",
        "explain": {
            "run_id": "run-1",
            "status": "running",
            "goal": "检查 app.py",
            "goal_summary": "检查 app.py",
            "current_step": {"node_id": "step-1", "title": "阅读 app.py", "tool_choice": "repo_scan"},
            "last_tool_result": {"tool_name": "repo_scan", "summary": "scanned 1 files"},
            "policy": {"allow_commit": False},
            "budget": {"max_steps": 12},
            "dirty_worktree_detected": False,
            "stop_reason": {},
        },
        "steps": [
            {
                "run_id": "run-1",
                "step_id": "step-1",
                "title": "阅读 app.py",
                "detail": "先检查相关文件",
                "status": "running",
                "tool_choice": "repo_scan",
                "expected_observation": "找到入口文件",
                "success_criteria": "定位到相关上下文",
                "confidence": 0.82,
            }
        ],
        "tools": [
            {
                "run_id": "run-1",
                "tool_name": "repo_scan",
                "status": "ok",
                "summary": "scanned 1 files",
                "output_excerpt": "src/app.py",
                "input": {},
            }
        ],
    }

    events = list(handler.handle_stream({"type": "user_turn", "session_id": "sess-plan", "text": "检查 app.py"}))

    assert [event["type"] for event in events] == [
        "run_status",
        "assistant_token",
        "step_update",
        "tool_call",
        "tool_result",
        "sidebar_snapshot",
        "assistant_final",
    ]


def test_task_user_turn_streams_visible_token_before_followup_events(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    handler = TerminalEventHandler(controller)

    handler.handle({"type": "start_session", "session_id": "sess-stream", "cwd": str(tmp_path)})

    class _Plan:
        route = "task_run"

    observed: list[str] = []
    controller.plan_turn = lambda text, **kwargs: _Plan()

    def fake_execute_turn(plan, **kwargs):
        observed.append("execute_turn")
        return {
            "route": "task_run",
            "run": {
                "run_id": "run-2",
                "status": "running",
                "goal": "检查 worker.py",
                "goal_summary": "检查 worker.py",
                "current_step_id": "step-2",
                "current_step": {"node_id": "step-2", "title": "阅读 worker.py", "tool_choice": "repo_scan"},
                "dirty_worktree_detected": False,
                "commit_permission_required": True,
                "stop_reason": {},
                "pending_steps": 1,
                "completed_steps": 0,
                "last_tool_result": {"tool_name": "repo_scan", "summary": "scanned 1 files"},
                "created_at": "2026-04-04T00:00:00Z",
                "updated_at": "2026-04-04T00:00:00Z",
            },
            "assistant_preamble": "已进入只读任务处理。可用 /status /why /steps /tools 查看进度。",
            "assistant_final": "已进入只读任务处理。可用 /status /why /steps /tools 查看进度。",
            "explain": {
                "run_id": "run-2",
                "status": "running",
                "goal": "检查 worker.py",
                "goal_summary": "检查 worker.py",
                "current_step": {"node_id": "step-2", "title": "阅读 worker.py", "tool_choice": "repo_scan"},
                "last_tool_result": {"tool_name": "repo_scan", "summary": "scanned 1 files"},
                "policy": {"allow_commit": False},
                "budget": {"max_steps": 12},
                "dirty_worktree_detected": False,
                "stop_reason": {},
            },
            "steps": [{"run_id": "run-2", "step_id": "step-2", "title": "阅读 worker.py", "status": "running", "tool_choice": "repo_scan"}],
            "tools": [{"run_id": "run-2", "tool_name": "repo_scan", "status": "ok", "summary": "scanned 1 files", "input": {}}],
        }

    controller.execute_turn = fake_execute_turn

    stream = handler.handle_stream({"type": "user_turn", "session_id": "sess-stream", "text": "检查 worker.py"})
    first = next(stream)
    second = next(stream)

    assert observed == ["execute_turn"]
    assert first["type"] == "run_status"
    assert second["type"] == "assistant_token"


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
    assert after_restart.transcript_lines == before_restart.transcript_lines
    assert after_restart.tool_timeline == before_restart.tool_timeline


def test_detach_session_preserves_transcript_and_restarts_same_session(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    _seed_entropy(controller)
    handler = TerminalEventHandler(controller)
    session_store = TerminalSessionStore(controller.runtime_dir)

    handler.handle({"type": "start_session", "session_id": "sess-detach", "cwd": str(tmp_path)})
    turn_events = handler.handle({"type": "user_turn", "session_id": "sess-detach", "text": "你好"})
    detached = handler.handle({"type": "close_session", "session_id": "sess-detach", "detach": True, "transcript_mode": "compact"})
    detached_state = session_store.read("sess-detach")

    assert detached[0]["type"] == "session_ended"
    assert detached_state.status == "detached"
    assert detached_state.transcript_mode == "compact"
    assert detached_state.compact is True
    assert detached_state.transcript_lines == [
        {"kind": "user", "text": "你好"},
        {"kind": "assistant", "text": next(item for item in turn_events if item["type"] == "assistant_final")["message"]},
    ]

    restarted = handler.handle({"type": "start_session", "session_id": "sess-detach", "cwd": str(tmp_path)})
    restarted_state = session_store.read("sess-detach")

    assert restarted_state.status == "active"
    assert restarted_state.transcript_lines == detached_state.transcript_lines
    assert restarted_state.transcript_mode == "compact"
    assert restarted[0]["session"]["transcript_lines"] == detached_state.transcript_lines


def test_non_persistent_start_session_does_not_replace_current_restorable_session(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    handler = TerminalEventHandler(controller)
    session_store = TerminalSessionStore(controller.runtime_dir)

    handler.handle({"type": "start_session", "session_id": "sess-interactive", "cwd": str(tmp_path)})
    current_before = session_store.read_current()

    handler.handle({"type": "start_session", "session_id": "sess-oneshot", "cwd": str(tmp_path), "persist_current": False})
    current_after = session_store.read_current()
    oneshot_state = session_store.read("sess-oneshot")

    assert current_before.session_id == "sess-interactive"
    assert current_after.session_id == "sess-interactive"
    assert oneshot_state.session_id == "sess-oneshot"


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
