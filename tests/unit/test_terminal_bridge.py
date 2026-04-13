import pytest
from pathlib import Path

from nalr.runtime.controller import RuntimeController
from nalr.schemas.models import RoundEvent
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
    assert "console" in run_snapshot
    assert run_snapshot["console"]["state"]
    assert run_snapshot["console"]["action_field"]
    assert run_snapshot["console"]["why_current"]
    assert run_snapshot["cognitive_snapshot"]["core_goal"]
    assert run_snapshot["cognitive_snapshot"]["current_intent"]
    assert "ui_actions" in run_snapshot
    assert run_snapshot["ui_actions"]["primary"]
    assert any(item["id"] == "status" for item in run_snapshot["ui_actions"]["primary"])
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


def test_start_session_stream_emits_started_before_sidebar_snapshot(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    handler = TerminalEventHandler(controller)

    stream = iter(handler.handle_stream({"type": "start_session", "session_id": "sess-stream-start", "cwd": str(tmp_path)}))
    first = next(stream)
    second = next(stream)

    assert first["type"] == "session_started"
    assert second["type"] == "sidebar_snapshot"


def test_sidebar_snapshot_surfaces_controlled_learning_policy(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    controller.update_observer_settings(
        {
            "autonomy": {
                "learning_mode": "active-learn",
                "network_enabled": True,
                "external_io_enabled": True,
                "allowed_network_domains": ["example.com", "docs.python.org"],
                "writable_roots": [str(workspace_root)],
                "knowledge_roots": [str(tmp_path / "docs")],
                "learning_log_dir": str(tmp_path / ".alive" / "learning-cache"),
                "trace_external_learning": False,
            }
        }
    )
    handler = TerminalEventHandler(controller)

    started = handler.handle({"type": "start_session", "session_id": "sess-learning", "cwd": str(tmp_path)})
    snapshot = next(item for item in started if item["type"] == "sidebar_snapshot")

    assert snapshot["controlled_learning"]["learning_mode"] == "active-learn"
    assert snapshot["controlled_learning"]["network_enabled"] is True
    assert snapshot["controlled_learning"]["external_io_enabled"] is True
    assert snapshot["controlled_learning"]["trace_external_learning"] is False
    assert snapshot["controlled_learning"]["allowed_network_domains"] == ["example.com", "docs.python.org"]
    assert snapshot["controlled_learning"]["writable_roots"] == [str(workspace_root)]
    assert snapshot["controlled_learning"]["learning_log_dir"] == str(tmp_path / ".alive" / "learning-cache")


def test_greeting_user_turn_uses_direct_chat_without_starting_run(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    _seed_entropy(controller)
    handler = TerminalEventHandler(controller)

    handler.handle({"type": "start_session", "session_id": "sess-greet", "cwd": str(tmp_path)})
    events = handler.handle({"type": "user_turn", "session_id": "sess-greet", "text": "你好"})
    session_state = TerminalSessionStore(controller.runtime_dir).read("sess-greet")

    assert [item["type"] for item in events] == ["assistant_token", "sidebar_snapshot", "assistant_final"]
    snapshot_event = next(item for item in events if item["type"] == "sidebar_snapshot")
    assert "你好" in events[-1]["message"]
    assert snapshot_event["console"]["state"]["brain_state"]["mode"] == "interactive"
    assert snapshot_event["console"]["state"]["current_round"]["round_id"] is not None
    assert snapshot_event["console"]["timeline"]["round_id"] is not None
    assert snapshot_event["console"]["why_current"]["round_id"] is not None
    assert snapshot_event["console"]["why_current"]["why"]["summary"] != "暂无"
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

    assert [item["type"] for item in events] == ["assistant_token", "sidebar_snapshot", "assistant_final"]
    snapshot_event = next(item for item in events if item["type"] == "sidebar_snapshot")
    assert "runtime_instance" not in events[-1]["message"]
    assert "我是" in events[-1]["message"]
    assert snapshot_event["console"]["state"]["brain_state"]["mode"] == "interactive"
    assert snapshot_event["console"]["state"]["current_round"]["round_id"] is not None
    assert snapshot_event["console"]["timeline"]["round_id"] is not None
    assert snapshot_event["console"]["why_current"]["round_id"] is not None
    assert snapshot_event["console"]["why_current"]["why"]["summary"] != "暂无"
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

    assert [item["type"] for item in events] == ["assistant_token", "assistant_token", "sidebar_snapshot", "assistant_final"]
    snapshot_event = next(item for item in events if item["type"] == "sidebar_snapshot")
    assert "".join(item["delta"] for item in events if item["type"] == "assistant_token") == "你好，我是当前运行体实例。"
    assert snapshot_event["console"]["state"]["brain_state"]["mode"] == "interactive"
    assert snapshot_event["console"]["state"]["current_round"]["round_id"] is not None
    assert snapshot_event["console"]["timeline"]["round_id"] is not None
    assert snapshot_event["console"]["why_current"]["round_id"] is not None
    assert snapshot_event["console"]["why_current"]["why"]["summary"] != "暂无"


def test_substantive_direct_chat_turn_refreshes_console_round(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    _seed_entropy(controller)
    handler = TerminalEventHandler(controller)

    handler.handle({"type": "start_session", "session_id": "sess-chat", "cwd": str(tmp_path)})
    events = handler.handle({"type": "user_turn", "session_id": "sess-chat", "text": "我现在有点乱，你觉得我应该先做哪一步？"})

    assert [item["type"] for item in events] == ["assistant_token", "sidebar_snapshot", "assistant_final"]
    snapshot_event = next(item for item in events if item["type"] == "sidebar_snapshot")

    assert snapshot_event["console"]["state"]["current_round"]["round_id"] is not None
    assert snapshot_event["console"]["action_field"]["winner"]["action"]
    assert snapshot_event["console"]["timeline"]["events"]
    assert snapshot_event["console"]["why_current"]["why"]["summary"]


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


def test_debug_control_commands_surface_trace_and_motivation_views(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    handler = TerminalEventHandler(controller)

    handler.handle({"type": "start_session", "session_id": "sess-debug", "cwd": str(tmp_path)})
    monkeypatch.setattr(controller, "resolve_round_ref", lambda round_ref: 7)
    monkeypatch.setattr(
        controller,
        "run_endogenous_tick",
        lambda **kwargs: {
            "round_id": 7,
            "boundary_action": "allow_internal",
            "selected_mode": kwargs.get("mode"),
            "trigger": {"trigger_type": kwargs.get("trigger"), "selected_mode": kwargs.get("mode")},
            "micro_intent": {"name": "curiosity"},
            "trace_ref": "trace://round/7",
        },
    )
    monkeypatch.setattr(
        controller,
        "why_motivation",
        lambda round_ref: {
            "round_id": 7,
            "sampled_action": "plan",
            "cause_type": "endogenous",
            "motivation_pool": {"active_motivations": [1]},
            "motivation_feedback": {"reward_signal": 0.5},
            "trace_ref": "trace://round/7",
        },
    )
    monkeypatch.setattr(
        controller,
        "replay_motivation",
        lambda round_id: {
            "round_id": round_id,
            "sampled_action": "plan",
            "motivation_pool": {"active_motivations": [1]},
            "motivation_feedback": {"reward_signal": 0.5},
            "trace_ref": "trace://round/7",
        },
    )
    monkeypatch.setattr(
        controller,
        "replay",
        lambda round_id, seed=0: {
            "round_id": round_id,
            "original_action": "plan",
            "replayed_action": "respond",
            "seed": seed,
            "trace_ref": "trace://round/7",
        },
    )
    monkeypatch.setattr(
        controller,
        "why_not",
        lambda round_id, action: {
            "round_id": round_id,
            "action": action,
            "selected_action": "respond",
            "blocked_by": ["ConflictMonitorAgent"],
            "trace_ref": "trace://round/7",
        },
    )
    monkeypatch.setattr(
        controller,
        "what_changed",
        lambda window=5: {
            "window": window,
            "action_counts": {"plan": 2},
            "mode_counts": {"interactive": 1},
            "budget_delta": 0.25,
            "trace_ref": "trace://round/7",
        },
    )
    monkeypatch.setattr(
        controller,
        "eval_longrun",
        lambda rounds=1000: {
            "generated_rounds": rounds,
            "task_success_rate": 0.75,
            "safe_mode_rate": 0.1,
            "trace_ref": "trace://round/7",
        },
    )

    cases = [
        ("endogenous", {"type": "control_command", "session_id": "sess-debug", "command": "endogenous", "value": "idle boot"}, "内源触发"),
        ("why-motivation", {"type": "control_command", "session_id": "sess-debug", "command": "why-motivation", "value": "last"}, "动机原因"),
        ("replay-motivation", {"type": "control_command", "session_id": "sess-debug", "command": "replay-motivation", "value": "7"}, "动机重放"),
        ("replay", {"type": "control_command", "session_id": "sess-debug", "command": "replay", "value": "7 11"}, "重放"),
        ("why-not", {"type": "control_command", "session_id": "sess-debug", "command": "why-not", "value": "7 plan"}, "为什么不是"),
        ("what-changed", {"type": "control_command", "session_id": "sess-debug", "command": "what-changed", "value": "8"}, "变化窗口"),
        ("eval", {"type": "control_command", "session_id": "sess-debug", "command": "eval", "value": "2"}, "长跑评估"),
    ]

    for _, payload, marker in cases:
        events = handler.handle(payload)
        final_event = next(item for item in events if item["type"] == "assistant_final")
        assert marker in final_event["message"]
        if "round_id" in final_event:
            assert final_event["round_id"] == 7
        if "trace_ref" in final_event:
            assert final_event["trace_ref"] == "trace://round/7"


def test_terminal_handler_uses_public_prepare_task_bootstrap_for_active_run(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    _seed_entropy(controller)
    handler = TerminalEventHandler(controller)
    session_store = TerminalSessionStore(controller.runtime_dir)

    handler.handle({"type": "start_session", "session_id": "sess-bootstrap", "cwd": str(tmp_path)})
    session = session_store.read("sess-bootstrap")
    session.active_run_id = "run-active"
    session.last_run_id = "run-active"
    session_store.write(session)

    class _Plan:
        route = "direct_chat"
        scenario = "chat"
        mode = "interactive"
        target = "user"
        task_bootstrap = None

    monkeypatch.setattr(controller, "plan_turn", lambda *args, **kwargs: _Plan())
    monkeypatch.setattr(controller, "run_status", lambda run_id=None: {"status": "running"})
    monkeypatch.setattr(controller, "explain_run", lambda run_id=None: {"goal_summary": "继续当前任务", "current_step": {}, "stop_reason": {}})
    monkeypatch.setattr(controller, "run_steps", lambda run_id=None: {"steps": []})
    monkeypatch.setattr(controller, "run_tools", lambda run_id=None: {"tools": []})
    recorded: dict[str, object] = {}
    monkeypatch.setattr(
        controller,
        "prepare_task_bootstrap",
        lambda *args, **kwargs: recorded.update({"goal": args[0]}) or ("run", {}, {}),
    )

    def _private_should_not_be_used(*args, **kwargs):
        raise AssertionError("terminal bridge should not call controller._build_task_bootstrap directly")

    monkeypatch.setattr(controller, "_build_task_bootstrap", _private_should_not_be_used)
    monkeypatch.setattr(
        controller,
        "execute_turn",
        lambda *args, **kwargs: {"route": "direct_chat", "assistant_final": "ok", "payload": {"stream_deltas": ["ok"]}},
    )

    events = handler.handle({"type": "user_turn", "session_id": "sess-bootstrap", "text": "继续当前任务"})

    assert recorded["goal"] == "继续当前任务"
    assert next(item for item in events if item["type"] == "assistant_final")["message"] == "ok"


def test_terminal_control_command_surfaces_initiative_views(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    handler = TerminalEventHandler(controller)

    handler.handle({"type": "start_session", "session_id": "sess-initiative-cmd", "cwd": str(tmp_path)})
    monkeypatch.setattr(controller, "initiative_status", lambda: {"summary": "initiative status", "ready": True})
    monkeypatch.setattr(
        controller,
        "initiative_distribution",
        lambda: {"summary": "initiative distribution", "top_intent": "share_memory"},
    )
    monkeypatch.setattr(
        controller,
        "initiative_trigger_now",
        lambda **kwargs: {"summary": "initiative trigger", "proposal": {"proposal_id": "prop-1"}, "auto_sent": False, "forced": kwargs.get("force", False)},
    )
    monkeypatch.setattr(
        controller,
        "initiative_why",
        lambda round_ref="last": {"summary": "initiative why", "round_id": 7},
    )

    cases = [
        ("status", "initiative status"),
        ("distribution", "initiative distribution"),
        ("trigger", "initiative trigger"),
        ("why", "initiative why"),
    ]

    for subcommand, marker in cases:
        events = handler.handle(
            {"type": "control_command", "session_id": "sess-initiative-cmd", "command": "initiative", "value": subcommand}
        )
        final_event = next(item for item in events if item["type"] == "assistant_final")
        assert marker in final_event["message"]

    forced_events = handler.handle(
        {"type": "control_command", "session_id": "sess-initiative-cmd", "command": "initiative", "value": "trigger idle force"}
    )
    forced_final = next(item for item in forced_events if item["type"] == "assistant_final")
    assert "initiative trigger" in forced_final["message"]


def test_terminal_control_command_surfaces_thought_snapshot(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    handler = TerminalEventHandler(controller)

    handler.handle({"type": "start_session", "session_id": "sess-thought-cmd", "cwd": str(tmp_path)})
    monkeypatch.setattr(
        controller,
        "thought_snapshot",
        lambda round_ref="last": {"round_id": 3, "thought_summary": {"why_summary": "plan"}, "trace_ref": "round://3"},
    )

    events = handler.handle(
        {"type": "control_command", "session_id": "sess-thought-cmd", "command": "thought", "value": "last"}
    )
    final_event = next(item for item in events if item["type"] == "assistant_final")

    assert "thought snapshot" in final_event["message"]
    assert final_event["payload"]["round_id"] == 3


def test_terminal_control_command_surfaces_monologue_views(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    handler = TerminalEventHandler(controller)

    handler.handle({"type": "start_session", "session_id": "sess-monologue-cmd", "cwd": str(tmp_path)})
    monkeypatch.setattr(controller, "monologue_status", lambda: {"summary": "monologue status", "hidden": True, "generated_total": 12})
    monkeypatch.setattr(
        controller,
        "monologue_show",
        lambda limit=12: {
            "summary": "monologue show",
            "hidden": True,
            "fragments": [{"content": "突然想到一个词", "category": "free_association"}],
            "returned": limit,
        },
    )

    status_events = handler.handle(
        {"type": "control_command", "session_id": "sess-monologue-cmd", "command": "monologue", "value": "status"}
    )
    show_events = handler.handle(
        {"type": "control_command", "session_id": "sess-monologue-cmd", "command": "monologue", "value": "show 7"}
    )

    status_final = next(item for item in status_events if item["type"] == "assistant_final")
    show_final = next(item for item in show_events if item["type"] == "assistant_final")

    assert "monologue status" in status_final["message"]
    assert "monologue show" in show_final["message"]
    assert show_final["payload"]["returned"] == 7


def test_permission_mode_task_run_emits_approval_choices_and_contextual_ui_actions(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    handler = TerminalEventHandler(controller)

    handler.handle({"type": "start_session", "session_id": "sess-choices", "cwd": str(tmp_path)})
    handler.handle({"type": "control_command", "session_id": "sess-choices", "command": "permissions", "value": "ask"})
    events = handler.handle({"type": "user_turn", "session_id": "sess-choices", "text": "检查 app.py 并规划下一步"})

    approval_event = next(item for item in events if item["type"] == "approval_request")
    snapshot_event = next(item for item in events if item["type"] == "sidebar_snapshot")

    assert approval_event["choices"] == [
        {"id": "approve", "label": "批准", "kind": "approval", "value": "approve"},
        {"id": "reject", "label": "拒绝", "kind": "approval", "value": "reject"},
        {"id": "details", "label": "详情", "kind": "drawer", "value": "approvals"},
        {"id": "next", "label": "下一个", "kind": "approval_nav", "value": "next"},
    ]
    assert any(item["id"] == "approvals" and item["disabled"] is False for item in snapshot_event["ui_actions"]["primary"])
    assert any(item["id"] == "abort" for item in snapshot_event["ui_actions"]["secondary"])


def test_control_command_probability_views_surface_round_observability_without_active_run(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    handler = TerminalEventHandler(controller)

    controller.tick(
        RoundEvent(
            source="user",
            content="Please help me plan carefully, but I also want to wander and rest. Remember tea too.",
            target="friend",
            cue="tea",
            valence=0.05,
        ),
        scenario="task",
        mode="interactive",
    )

    handler.handle({"type": "start_session", "session_id": "sess-prob", "cwd": str(tmp_path)})
    probability_events = handler.handle({"type": "control_command", "session_id": "sess-prob", "command": "probability"})
    layer_events = handler.handle(
        {"type": "control_command", "session_id": "sess-prob", "command": "probability", "value": "layer action"}
    )
    action_events = handler.handle(
        {"type": "control_command", "session_id": "sess-prob", "command": "probability", "value": "action wander"}
    )

    probability_message = next(item for item in probability_events if item["type"] == "assistant_final")["message"]
    layer_message = next(item for item in layer_events if item["type"] == "assistant_final")["message"]
    action_message = next(item for item in action_events if item["type"] == "assistant_final")["message"]

    assert "概率场" in probability_message
    assert "action" in probability_message
    assert "token_state" in probability_message
    assert "action 层" in layer_message
    assert "winner=" in layer_message
    assert "动作概率" in action_message
    assert "wander" in action_message


def test_model_control_command_reports_tiers_and_bindings(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    handler = TerminalEventHandler(controller)

    handler.handle({"type": "start_session", "session_id": "sess-model", "cwd": str(tmp_path)})
    events = handler.handle({"type": "control_command", "session_id": "sess-model", "command": "model"})

    final_event = next(item for item in events if item["type"] == "assistant_final")
    snapshot_event = next(item for item in events if item["type"] == "sidebar_snapshot")

    assert "small_model" in final_event["message"]
    assert "medium_model" in final_event["message"]
    assert "large_model" in final_event["message"]
    assert "SalienceAgent -> small_model" in final_event["message"]
    assert "PerspectiveModel -> medium_model" in final_event["message"]
    assert "Renderer -> large_model" in final_event["message"]
    assert "model_status" in snapshot_event
    assert snapshot_event["model_status"]["tiers"]["small_model"]["model"] == "ep-20260404191810-qfn7s"
    assert snapshot_event["model_status"]["tiers"]["medium_model"]["model"] == "deepseek-chat"
    assert snapshot_event["model_status"]["route_policies"]["chat_standard"]["hot_path"] == "full_tick"


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
    turn_types = [item["type"] for item in turn_events]
    run_event = next(item for item in turn_events if item["type"] == "run_status")

    assert approval["mode"] == "ask"
    assert approval["status"] == "pending"
    assert approval["run_id"]
    assert approval["tool"] == "repo_scan"
    assert approval["actions"] == ["approve", "reject"]
    assert approval["risk_level"] == "medium"
    assert approval["summary"]
    assert approval["action_preview"]
    assert "tool_result" not in turn_types
    assert "sidebar_snapshot" in turn_types
    assert run_event["run"]["last_tool_result"] == {}
    persisted = session_store.read("sess-ask")
    assert persisted.permission_mode == "ask"
    assert persisted.approvals_pending
    assert persisted.approvals_pending[0]["call_id"] == approval["call_id"]
    assert persisted.approvals_pending[0]["trace_ref"] == approval["trace_ref"]

    approved_events = handler.handle(
        {
            "type": "approve",
            "session_id": "sess-ask",
            "call_id": approval["call_id"],
            "approved": True,
        }
    )
    approved_types = [item["type"] for item in approved_events]
    assert "tool_result" in approved_types
    assert "sidebar_snapshot" in approved_types
    approved_persisted = session_store.read("sess-ask")
    assert approved_persisted.approvals_pending == []
    assert any(
        item["kind"] == "result" and item["callId"] == approval["call_id"] and item["traceRef"]
        for item in approved_persisted.tool_timeline
    )


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
    assert any(item.get("traceRef") for item in before_restart.tool_timeline if item["callId"] == pending)
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
