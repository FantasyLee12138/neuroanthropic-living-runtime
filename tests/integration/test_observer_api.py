import inspect
import json
from pathlib import Path

from fastapi.testclient import TestClient

from nalr.runtime.controller import RuntimeController
from nalr.schemas.models import RoundEvent
from services.observer.api.app import app as default_app
from services.observer.api.app import create_app
from nalr.terminal_bridge.handlers import TerminalEventHandler


CONFIG_ROOT = Path(__file__).resolve().parents[2] / "config"


def test_observer_reads_state_and_trace(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    controller.tick(
        RoundEvent(source="user", content="hello runtime and remember coffee", target="user", cue="coffee"),
        scenario="chat",
        mode="interactive",
    )
    controller.flush_pending_io(raise_on_error=True)

    client = TestClient(create_app(project_root=tmp_path, config_root=CONFIG_ROOT))

    state_response = client.get("/state")
    trace_response = client.get("/trace/1")
    why_response = client.get("/why/1")
    metrics_response = client.get("/metrics/summary")
    recall_response = client.get("/memory/recall/coffee")
    dashboard_response = client.get("/dashboard")

    assert state_response.status_code == 200
    assert trace_response.status_code == 200
    assert why_response.status_code == 200
    assert metrics_response.status_code == 200
    assert recall_response.status_code == 200
    assert dashboard_response.status_code == 200
    assert state_response.json()["round_count"] == 1
    assert state_response.json()["trace_storage"]["trace_sync_state"] == "healthy"
    assert trace_response.json()["round_id"] == 1
    assert trace_response.json()["storage"]["read_source"] == "parquet"
    assert why_response.json()["round_id"] == 1
    assert why_response.json()["storage"]["trace_sync_state"] == "healthy"
    assert metrics_response.json()["total_rounds"] == 1
    assert recall_response.json()["cue"] == "coffee"
    assert recall_response.json()["found"] is True
    assert "NALR Observer" in dashboard_response.text


def test_observer_state_and_why_surface_tlh_runtime_fields(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    state = controller.load_runtime_state()
    state.body_energy = 0.08
    state.fatigue = 0.9
    state.memory_fragments = 0.86
    state.self_continuity = 0.24
    state.meaning_strength = 0.18
    state.subjective_state.reject_all = 0.9
    state.subjective_state.spontaneous = 0.74
    state.subjective_state.meaning_made = ["安静比回应更重要"]
    controller._save_state(state, sync=True)
    controller.tick(
        RoundEvent(source="user", content="先别急着回答", target="user", cue="回答"),
        scenario="chat",
        mode="interactive",
    )
    controller.flush_pending_io(raise_on_error=True)

    client = TestClient(create_app(project_root=tmp_path, config_root=CONFIG_ROOT))

    state_response = client.get("/state")
    why_response = client.get("/why/1")

    assert state_response.status_code == 200
    assert why_response.status_code == 200
    assert "subjective_state" in state_response.json()
    assert "organic_mode" in state_response.json()
    assert "instinct_field" in state_response.json()["cognitive_snapshot"]["tlh"]
    assert "counterfactual_replays" in why_response.json()


def test_default_observer_app_is_available():
    assert default_app.title == "NALR Observer"
    assert default_app.version == "0.6.0"


def test_observer_core_routes_are_async_handlers(tmp_path):
    observer_app = create_app(project_root=tmp_path, config_root=CONFIG_ROOT)
    route_map = {route.path: route.endpoint for route in observer_app.routes if hasattr(route, "path")}

    assert inspect.iscoroutinefunction(route_map["/state"])
    assert inspect.iscoroutinefunction(route_map["/identity"])
    assert inspect.iscoroutinefunction(route_map["/metrics/summary"])


def test_observer_exposes_current_run_and_step_views(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "planner.py").write_text("VALUE = 1\n", encoding="utf-8")
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    run_payload = controller.start_run("检查 planner.py 并规划下一步")
    controller.flush_pending_io(raise_on_error=True)

    client = TestClient(create_app(project_root=tmp_path, config_root=CONFIG_ROOT))

    current_run = client.get("/runs/current")
    step_trace = client.get(f"/runs/{run_payload['run_id']}/steps")
    tool_trace = client.get(f"/runs/{run_payload['run_id']}/tools")

    assert current_run.status_code == 200
    assert step_trace.status_code == 200
    assert tool_trace.status_code == 200
    assert current_run.json()["run_id"] == run_payload["run_id"]
    assert step_trace.json()["steps"]
    assert tool_trace.json()["tools"][0]["tool_name"] == "repo_scan"


def test_observer_exposes_model_status(tmp_path):
    client = TestClient(create_app(project_root=tmp_path, config_root=CONFIG_ROOT))

    response = client.get("/models/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["tiers"]["small_model"]["model"] == "ep-20260404191810-qfn7s"
    assert payload["tiers"]["medium_model"]["model"] == "deepseek-chat"
    assert payload["agent_bindings"]["SalienceAgent"] == "small_model"
    assert payload["agent_bindings"]["planner"] == "medium_model"
    assert "chat_fast" in payload["routes"]


def test_observer_settings_can_override_newborn_unlocks_and_local_model(tmp_path):
    client = TestClient(create_app(project_root=tmp_path, config_root=CONFIG_ROOT))

    update_response = client.post(
        "/settings",
        json={
            "newborn": {
                "organic_mode": {
                    "instinct_first": True,
                },
                "subjective_state": {
                    "spontaneous": 0.11,
                    "felt": ["轻微想动"],
                },
            },
            "models": {
                "model_tiers": {
                    "local_model": {
                        "mode": "remote",
                        "backend": "openai_compatible",
                        "base_url": "http://127.0.0.1:11434/v1",
                        "model": "qwen-local",
                        "timeout_ms": 15000,
                        "retries": 0,
                        "api_key_env": "LOCAL_MODEL_API_KEY",
                        "enabled": True,
                    }
                },
                "agent_model_bindings": {
                    "Renderer": "local_model",
                },
            },
        },
    )

    assert update_response.status_code == 200
    settings_payload = update_response.json()
    assert settings_payload["newborn"]["organic_mode"]["instinct_first"] is True
    assert settings_payload["newborn"]["subjective_state"]["spontaneous"] == 0.11
    assert settings_payload["models"]["model_tiers"]["local_model"]["backend"] == "openai_compatible"
    assert settings_payload["models"]["agent_model_bindings"]["Renderer"] == "local_model"

    reset_response = client.post("/persona/reset", json={})
    assert reset_response.status_code == 200
    state_payload = reset_response.json()["state"]
    assert state_payload["organic_mode"]["instinct_first"] is True
    assert state_payload["subjective_state"]["spontaneous"] == 0.11


def test_observer_exposes_terminal_session_mapping(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "agent.py").write_text("VALUE = 1\n", encoding="utf-8")
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    handler = TerminalEventHandler(controller)
    handler.handle({"type": "start_session", "session_id": "sess-observer", "cwd": str(tmp_path)})
    events = handler.handle({"type": "user_turn", "session_id": "sess-observer", "text": "检查 agent.py"})
    run_id = next(item for item in events if item["type"] == "run_status")["run"]["run_id"]

    client = TestClient(create_app(project_root=tmp_path, config_root=CONFIG_ROOT))

    current_session = client.get("/sessions/current")
    session_detail = client.get("/sessions/sess-observer")

    assert current_session.status_code == 200
    assert session_detail.status_code == 200
    assert current_session.json()["session_id"] == "sess-observer"
    assert session_detail.json()["active_run_id"] == run_id


def test_observer_prefers_parquet_round_mirror_when_json_round_files_are_missing(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    controller.tick(
        RoundEvent(
            source="user",
            content="Please remember tea and plan the next step.",
            target="user",
            cue="tea",
            valence=0.1,
        ),
        scenario="task",
        mode="interactive",
    )
    controller.flush_pending_io(raise_on_error=True)

    round_json = tmp_path / ".alive" / "traces" / "rounds" / "round_1.json"
    round_jsonl = tmp_path / ".alive" / "traces" / "round_traces.jsonl"
    assert round_json.exists()
    round_json.unlink()
    round_jsonl.write_text("", encoding="utf-8")

    client = TestClient(create_app(project_root=tmp_path, config_root=CONFIG_ROOT))

    trace_response = client.get("/trace/1")
    metrics_response = client.get("/metrics/summary")

    assert trace_response.status_code == 200
    assert metrics_response.status_code == 200
    assert trace_response.json()["round_id"] == 1
    assert trace_response.json()["storage"]["read_source"] == "parquet"
    assert metrics_response.json()["total_rounds"] == 1


def test_observer_exposes_dream_status_runs_and_metrics(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    controller.tick(
        RoundEvent(
            source="user",
            content="Remember tea for sleep.",
            target="user",
            cue="tea",
            valence=0.2,
        ),
        scenario="companion",
        mode="interactive",
    )
    controller.tick(
        RoundEvent(
            source="system",
            content="Sleep reshape around tea.",
            target="user",
            cue="tea",
        ),
        scenario="companion",
        mode="sleep",
    )
    controller.flush_pending_io(raise_on_error=True)

    client = TestClient(create_app(project_root=tmp_path, config_root=CONFIG_ROOT))

    status_response = client.get("/dream/status")
    runs_response = client.get("/dream/runs")
    trace_response = client.get("/dream/runs/last")
    metrics_response = client.get("/dream/metrics")

    assert status_response.status_code == 200
    assert runs_response.status_code == 200
    assert trace_response.status_code == 200
    assert metrics_response.status_code == 200
    assert status_response.json()["enabled"] is True
    assert runs_response.json()["runs"]
    assert trace_response.json()["trigger"] == "sleep_full"
    assert metrics_response.json()["total_runs"] == 1


def test_observer_exposes_console_routes_for_latest_runtime_state(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    controller.tick(
        RoundEvent(
            source="user",
            content="Remember tea and reflect on it.",
            target="user",
            cue="tea",
            valence=0.2,
        ),
        scenario="chat",
        mode="interactive",
    )
    controller.flush_pending_io(raise_on_error=True)

    client = TestClient(create_app(project_root=tmp_path, config_root=CONFIG_ROOT))

    console_state = client.get("/console/state")
    console_action_field = client.get("/console/action-field")
    console_timeline = client.get("/console/timeline")
    console_why = client.get("/console/why/current")
    console_why_not = client.get("/console/why-not/rest")

    assert console_state.status_code == 200
    assert console_action_field.status_code == 200
    assert console_timeline.status_code == 200
    assert console_why.status_code == 200
    assert console_why_not.status_code == 200

    state_payload = console_state.json()
    assert "brain_state" in state_payload
    assert "cognitive_snapshot" in state_payload
    assert "current_round" in state_payload
    assert "trace_ref" in state_payload["current_round"]

    action_payload = console_action_field.json()
    assert "top_actions" in action_payload
    assert "winner" in action_payload
    assert "token_field" in action_payload
    assert "contribution_stack" in action_payload

    timeline_payload = console_timeline.json()
    assert "events" in timeline_payload
    assert timeline_payload["events"]
    assert timeline_payload["events"][0]["type"] in {
        "stimulus",
        "memory_activation",
        "motivation_rise",
        "action_arbitration",
        "token_gate",
        "endogenous_tick",
        "dream_effect",
    }

    why_payload = console_why.json()
    assert why_payload["round_id"] == 1
    assert "why" in why_payload
    assert "summary" in why_payload["why"]

    why_not_payload = console_why_not.json()
    assert why_not_payload["round_id"] == 1
    assert why_not_payload["action"] == "rest"
    assert "why_not" in why_not_payload


def test_observer_exposes_console_refresh_route_with_tlh_payload(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    state = controller.load_runtime_state()
    state.body_energy = 0.08
    state.fatigue = 0.91
    state.memory_fragments = 0.84
    state.self_continuity = 0.23
    state.meaning_strength = 0.16
    state.subjective_state.reject_all = 0.86
    state.subjective_state.spontaneous = 0.75
    state.subjective_state.meaning_made = ["先回到内部，再决定是否说话"]
    controller._save_state(state, sync=True)
    controller.tick(
        RoundEvent(
            source="user",
            content="先不要着急回答",
            target="user",
            cue="回答",
        ),
        scenario="chat",
        mode="interactive",
    )
    controller.flush_pending_io(raise_on_error=True)

    client = TestClient(create_app(project_root=tmp_path, config_root=CONFIG_ROOT))

    response = client.get("/console/refresh")

    assert response.status_code == 200
    payload = response.json()
    assert "state" in payload
    assert "action_field" in payload
    assert "timeline" in payload
    assert "why_current" in payload
    assert "why_not" in payload
    assert "probability_field" in payload
    assert "counterfactual_preview" in payload
    assert "source_links" in payload
    assert "recent_rounds" in payload
    assert "instinct_field" in payload["state"]["cognitive_snapshot"]["tlh"]
    assert payload["probability_field"]["action"]["winner_posterior"]
    assert payload["source_links"][0]["panel_id"]


def test_dashboard_shell_surfaces_tlh_observer_sections(tmp_path):
    client = TestClient(create_app(project_root=tmp_path, config_root=CONFIG_ROOT))

    response = client.get("/dashboard")

    assert response.status_code == 200
    assert "主体控制" in response.text
    assert "开始运行" in response.text
    assert "重置人格" in response.text
    assert "主体状态" in response.text
    assert "概率与坍缩" in response.text
    assert "四维空间" in response.text
    assert "四层概率场" in response.text
    assert "峰值焦点" in response.text
    assert "外显表达区" in response.text
    assert "解释系统" in response.text
    assert "最近动作" in response.text
    assert "自治状态" in response.text
    assert "最近自治动作" in response.text
    assert "权限边界与预算" in response.text


def test_dashboard_shell_surfaces_web_first_workspace_entry(tmp_path):
    client = TestClient(create_app(project_root=tmp_path, config_root=CONFIG_ROOT))

    response = client.get("/dashboard")

    assert response.status_code == 200
    assert "NALR 工作台" in response.text
    assert "开始运行" in response.text
    assert "进入分析" in response.text
    assert "查看回放" in response.text
    assert "认知工作台" in response.text
    assert "开发者明细" in response.text
    assert 'id="command-toggle"' in response.text
    assert 'id="process-toggle"' in response.text
    assert "命令面板" not in response.text
    assert "Token" in response.text


def test_observer_console_talk_and_endogenous_tick_return_console_refresh_payload(tmp_path):
    client = TestClient(create_app(project_root=tmp_path, config_root=CONFIG_ROOT))

    talk_response = client.post(
        "/console/talk",
        json={
            "session_id": "sess-console",
            "cwd": str(tmp_path),
            "text": "你好",
        },
    )

    assert talk_response.status_code == 200
    talk_payload = talk_response.json()
    assert "assistant" in talk_payload
    assert "console" in talk_payload
    assert "state" in talk_payload["console"]
    assert "action_field" in talk_payload["console"]
    assert "timeline" in talk_payload["console"]
    assert "why_current" in talk_payload["console"]
    assert "why_not" in talk_payload["console"]

    tick_response = client.post("/console/endogenous/tick", json={"trigger": "idle", "mode": "endogenous_light"})

    assert tick_response.status_code == 200
    tick_payload = tick_response.json()
    assert "tick" in tick_payload
    assert "console" in tick_payload
    assert "state" in tick_payload["console"]
    assert "action_field" in tick_payload["console"]
    assert "why_not" in tick_payload["console"]


def test_observer_exposes_autonomy_lifecycle_routes_and_console_payload(tmp_path):
    client = TestClient(create_app(project_root=tmp_path, config_root=CONFIG_ROOT))

    start_response = client.post("/autonomy/start", json={"profile": "tool_level"})
    status_response = client.get("/autonomy/status")
    step_response = client.post("/autonomy/step")
    refresh_response = client.get("/console/refresh")
    stop_response = client.post("/autonomy/stop", json={"reason": "api_stop"})

    assert start_response.status_code == 200
    assert status_response.status_code == 200
    assert step_response.status_code == 200
    assert refresh_response.status_code == 200
    assert stop_response.status_code == 200

    assert start_response.json()["profile"] == "tool_level"
    assert status_response.json()["enabled"] is True
    assert step_response.json()["last_action_type"]
    assert "autonomy" in refresh_response.json()
    assert refresh_response.json()["autonomy"]["kill_switch_available"] is True
    assert stop_response.json()["running"] is False
    assert stop_response.json()["stop_reason"] == "api_stop"


def test_observer_web_session_api_streams_chat_events_and_session_state(tmp_path):
    client = TestClient(create_app(project_root=tmp_path, config_root=CONFIG_ROOT))

    start_response = client.post("/web/session/start", json={"session_id": "sess-web", "cwd": str(tmp_path)})
    assert start_response.status_code == 200
    assert start_response.json()["session"]["session_id"] == "sess-web"

    turn_response = client.post(
        "/web/session/event",
        json={"type": "user_turn", "session_id": "sess-web", "text": "你好"},
    )
    assert turn_response.status_code == 200
    assert turn_response.json()["accepted"] is True

    with client.stream("GET", "/web/session/events", params={"session_id": "sess-web", "once": True}) as stream_response:
        assert stream_response.status_code == 200
        event_lines = [line for line in stream_response.iter_lines() if line.startswith("data: ")]

    assert event_lines
    event_payloads = [json.loads(line[len("data: ") :]) for line in event_lines]
    event_types = [payload["type"] for payload in event_payloads]
    assert "assistant_token" in event_types
    assert "sidebar_snapshot" in event_types
    assert event_types[-1] == "assistant_final"

    state_response = client.get("/web/session/state", params={"session_id": "sess-web"})
    assert state_response.status_code == 200
    state_payload = state_response.json()
    assert state_payload["session"]["session_id"] == "sess-web"
    assert state_payload["session"]["transcript_lines"]
    assert state_payload["console"]["state"]
    assert state_payload["workbench"]["cards"]


def test_observer_web_session_api_supports_approval_round_trip(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "worker.py").write_text("VALUE = 1\n", encoding="utf-8")

    client = TestClient(create_app(project_root=tmp_path, config_root=CONFIG_ROOT))

    start_response = client.post("/web/session/start", json={"session_id": "sess-approval", "cwd": str(tmp_path)})
    assert start_response.status_code == 200

    permissions_response = client.post(
        "/web/session/event",
        json={"type": "control_command", "session_id": "sess-approval", "command": "permissions", "value": "ask"},
    )
    assert permissions_response.status_code == 200

    turn_response = client.post(
        "/web/session/event",
        json={"type": "user_turn", "session_id": "sess-approval", "text": "检查 worker.py 并规划下一步"},
    )
    assert turn_response.status_code == 200

    with client.stream("GET", "/web/session/events", params={"session_id": "sess-approval", "once": True}) as stream_response:
        assert stream_response.status_code == 200
        event_lines = [line for line in stream_response.iter_lines() if line.startswith("data: ")]

    assert event_lines
    event_payloads = [json.loads(line[len("data: ") :]) for line in event_lines]
    approval_event = next(item for item in event_payloads if item["type"] == "approval_request")

    approval_state = client.get("/web/session/state", params={"session_id": "sess-approval"})
    assert approval_state.status_code == 200
    assert approval_state.json()["approvals"]["pending_count"] >= 1

    approve_response = client.post(
        "/web/session/event",
        json={
            "type": "approve",
            "session_id": "sess-approval",
            "call_id": approval_event["call_id"],
            "approved": True,
        },
    )
    assert approve_response.status_code == 200

    with client.stream(
        "GET",
        "/web/session/events",
        params={"session_id": "sess-approval", "after_id": approval_event["event_id"], "once": True},
    ) as approval_stream:
        assert approval_stream.status_code == 200
        approval_lines = [line for line in approval_stream.iter_lines() if line.startswith("data: ")]

    assert approval_lines
    approval_payloads = [json.loads(line[len("data: ") :]) for line in approval_lines]
    approval_types = [payload["type"] for payload in approval_payloads]
    assert "tool_result" in approval_types
    assert "assistant_final" in approval_types


def test_observer_exposes_extended_diagnostics_endpoints(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    controller.start_run("Inspect the repository and keep working until the task is done.")
    controller.tick(
        RoundEvent(source="user", content="你是谁？", target="user"),
        scenario="chat",
        mode="interactive",
    )
    for _ in range(5):
        controller.tick(
            RoundEvent(source="user", content="你是不是根本不记得我了，我有点失望。", target="user"),
            scenario="chat",
            mode="interactive",
        )
    controller.flush_pending_io(raise_on_error=True)

    client = TestClient(create_app(project_root=tmp_path, config_root=CONFIG_ROOT))

    timeline_response = client.get("/metrics/timeline")
    heatmap_response = client.get("/metrics/heatmap")
    replay_response = client.get("/replay/1?seed=5")
    why_not_response = client.get("/why-not/1/rest")
    entropy_response = client.get("/metrics/entropy")
    skill_profile_response = client.get("/skills/profile/generate_candidates")
    state_delta_response = client.get("/diagnostics/state-delta")
    identity_blockers_response = client.get("/diagnostics/identity-blockers")
    contamination_response = client.get("/diagnostics/run-contamination")
    why_no_change_response = client.get("/diagnostics/why-no-change/last")
    cue_fragmentation_response = client.get("/diagnostics/cue-fragmentation")
    migration_response = client.get("/diagnostics/migration")

    assert timeline_response.status_code == 200
    assert heatmap_response.status_code == 200
    assert replay_response.status_code == 200
    assert why_not_response.status_code == 200
    assert entropy_response.status_code == 200
    assert skill_profile_response.status_code == 200
    assert state_delta_response.status_code == 200
    assert identity_blockers_response.status_code == 200
    assert contamination_response.status_code == 200
    assert why_no_change_response.status_code == 200
    assert cue_fragmentation_response.status_code == 200
    assert migration_response.status_code == 200
    assert timeline_response.json()["rounds"]
    assert heatmap_response.json()["actions"]
    assert replay_response.json()["round_id"] == 1
    assert why_not_response.json()["round_id"] == 1
    assert "provider_class" in entropy_response.json()
    assert skill_profile_response.json()["name"] == "generate_candidates"
    assert state_delta_response.json()["points"]
    assert "blockers" in identity_blockers_response.json()
    assert contamination_response.json()["points"]
    assert "failure_mode" in why_no_change_response.json()
    assert "families" in cue_fragmentation_response.json()
    assert "runtime" in migration_response.json()


def test_observer_state_and_metrics_expose_subjectivity_boundary_metrics(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    controller.tick(
        RoundEvent(source="user", content="记住茶。", target="user", cue="tea"),
        scenario="chat",
        mode="interactive",
    )
    controller.apply_command("mood calm")
    controller.flush_pending_io(raise_on_error=True)

    client = TestClient(create_app(project_root=tmp_path, config_root=CONFIG_ROOT))

    state_response = client.get("/state")
    metrics_response = client.get("/metrics/summary")
    motivation_response = client.get("/metrics/motivation")
    endogenous_response = client.get("/metrics/endogenous")
    timeline_response = client.get("/metrics/timeline")
    trace_response = client.get("/trace/1")
    endogenous_tick_response = client.post("/endogenous/tick", params={"trigger": "idle"})
    why_motivation_response = client.get("/why-motivation/2")
    replay_motivation_response = client.get("/replay/motivation/2")

    assert state_response.status_code == 200
    assert metrics_response.status_code == 200
    assert motivation_response.status_code == 200
    assert endogenous_response.status_code == 200
    assert timeline_response.status_code == 200
    assert trace_response.status_code == 200
    assert endogenous_tick_response.status_code == 200
    assert why_motivation_response.status_code == 200
    assert replay_motivation_response.status_code == 200
    assert state_response.json()["subjectivity"]["subject_core_integrity"] is True
    assert "boundary_violation_count" in state_response.json()["subjectivity"]
    assert "external_to_internal_ratio" in metrics_response.json()
    assert "endogenous_intent_rate" in metrics_response.json()
    assert "motivation_active_rate" in metrics_response.json()
    assert "active_rate" in motivation_response.json()
    assert "endogenous_round_rate" in endogenous_response.json()
    assert "subjectivity" in timeline_response.json()
    assert "subject_id" in trace_response.json()
    assert endogenous_tick_response.json()["cause_type"] == "endogenous"
    assert why_motivation_response.json()["cause_type"] == "endogenous"
    assert "motivation_pool" in replay_motivation_response.json()


def test_observer_runtime_start_and_pause_use_single_session_entrypoint(tmp_path):
    client = TestClient(create_app(project_root=tmp_path, config_root=CONFIG_ROOT))

    start_response = client.post("/web/runtime/start", json={})

    assert start_response.status_code == 200
    start_payload = start_response.json()
    assert start_payload["session"]["session"]["session_id"] == "observer-main"
    assert start_payload["session"]["session"]["permission_mode"] == "acceptEdits"
    assert start_payload["autonomy"]["running"] is True
    assert start_payload["autonomy"]["last_step_at"]
    assert "recent_actions" in start_payload

    pause_response = client.post("/web/runtime/pause", json={})

    assert pause_response.status_code == 200
    assert pause_response.json()["autonomy"]["running"] is False


def test_observer_runtime_start_clears_stale_safe_mode_and_enters_running(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    state = controller.load_runtime_state()
    state.safe_mode = True
    state.mode = "safe"
    controller._save_state(state, sync=True)

    client = TestClient(create_app(project_root=tmp_path, config_root=CONFIG_ROOT))

    start_response = client.post("/web/runtime/start", json={})

    assert start_response.status_code == 200
    payload = start_response.json()
    assert payload["autonomy"]["running"] is True
    refreshed_state = client.get("/state").json()
    assert refreshed_state["safe_mode"] is False
    assert refreshed_state["mode"] != "safe"


def test_observer_runtime_start_rehydrates_budget_before_first_autonomy_step(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    state = controller.load_runtime_state()
    state.safe_mode = True
    state.mode = "safe"
    state.budget_remaining = 0.03
    state.body_energy = 0.08
    state.resource_state = {"resource_mode": "starvation", "scarcity_index": 0.92}
    controller._save_state(state, sync=True)

    client = TestClient(create_app(project_root=tmp_path, config_root=CONFIG_ROOT))

    start_response = client.post("/web/runtime/start", json={})

    assert start_response.status_code == 200
    payload = start_response.json()
    assert payload["autonomy"]["running"] is True
    refreshed_state = client.get("/state").json()
    assert refreshed_state["safe_mode"] is False
    assert refreshed_state["budget_remaining"] >= 0.22
    assert refreshed_state["body_energy"] >= 0.32
    assert refreshed_state["resource_state"]["resource_mode"] != "starvation"


def test_observer_persona_reset_wipes_histories_and_returns_empty_safe_payloads(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    controller.tick(
        RoundEvent(
            source="user",
            content="记住 tea，并给出一个回应。",
            target="user",
            cue="tea",
        ),
        scenario="chat",
        mode="interactive",
    )
    controller.flush_pending_io(raise_on_error=True)

    client = TestClient(create_app(project_root=tmp_path, config_root=CONFIG_ROOT))
    start_response = client.post("/web/runtime/start", json={})
    assert start_response.status_code == 200

    reset_response = client.post("/persona/reset", json={})

    assert reset_response.status_code == 200
    reset_payload = reset_response.json()
    state_payload = reset_payload["state"]
    assert state_payload["round_count"] == 0
    assert state_payload["fatigue"] == 0
    assert state_payload["memory_fragments"] == 0
    assert state_payload["self_continuity"] == 0.5
    assert state_payload["meaning_strength"] == 0
    assert state_payload["base_metabolism"] == 1.0
    assert state_payload["subjective_state"]["felt"] == []
    assert state_payload["subjective_state"]["spontaneous"] == 0
    assert state_payload["subjective_state"]["boundary"] == 0
    assert state_payload["subjective_state"]["reject_all"] == 0
    assert state_payload["subjective_state"]["meaning_made"] == []
    assert state_payload["organic_mode"]["instinct_first"] is False
    assert state_payload["autonomy_loop"]["recent_actions"] == []
    assert reset_payload["autonomy"]["running"] is False
    assert reset_payload["recent_actions"] == []
    assert reset_payload["explainability"]["why"]["message"] == "无决策记录"
    assert reset_payload["explainability"]["trace"]["message"] == "无决策记录"
    assert state_payload["cognitive_snapshot"]["tlh"]["instinct_field"]["axis_values"] == {"E": 0.0, "F": 0.0, "S": 0.0, "M": 0.0}

    sessions_response = client.get("/web/sessions")
    why_response = client.get("/why/1")
    trace_response = client.get("/trace/1")

    assert sessions_response.status_code == 200
    assert sessions_response.json()["sessions"] == []
    assert why_response.status_code == 200
    assert trace_response.status_code == 200
    assert why_response.json()["message"] == "无决策记录"
    assert trace_response.json()["message"] == "无决策记录"


def test_observer_console_lightweight_routes_surface_recent_actions_and_probability_space(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    controller.tick(
        RoundEvent(
            source="user",
            content="请先休息一下再回应。",
            target="user",
            cue="休息",
        ),
        scenario="chat",
        mode="interactive",
    )
    controller.flush_pending_io(raise_on_error=True)

    client = TestClient(create_app(project_root=tmp_path, config_root=CONFIG_ROOT))

    recent_actions_response = client.get("/console/recent-actions")
    probability_space_response = client.get("/console/probability-space")

    assert recent_actions_response.status_code == 200
    assert probability_space_response.status_code == 200
    recent_actions_payload = recent_actions_response.json()
    probability_space_payload = probability_space_response.json()
    assert recent_actions_payload["actions"]
    assert "round_label" not in recent_actions_payload["actions"][0]
    assert probability_space_payload["plots"]
    assert [item["label"] for item in probability_space_payload["plots"]] == ["E-F", "E-S", "E-M", "F-S"]
    assert probability_space_payload["layers"][0]["label"] == "情境层"


def test_observer_exposes_endogenous_status_tick_and_what_changed(tmp_path):
    client = TestClient(create_app(project_root=tmp_path, config_root=CONFIG_ROOT))

    tick_response = client.post("/endogenous/tick", params={"trigger": "idle"})
    status_response = client.get("/endogenous/status")
    why_motivation_response = client.get("/why-motivation/1")
    replay_motivation_response = client.get("/replay/motivation/1")
    what_changed_response = client.get("/what-changed", params={"window": 1})

    assert tick_response.status_code == 200
    assert status_response.status_code == 200
    assert why_motivation_response.status_code == 200
    assert replay_motivation_response.status_code == 200
    assert what_changed_response.status_code == 200
    assert tick_response.json()["cause_type"] == "endogenous"
    assert status_response.json()["latest_trigger"]["trigger_type"] == "idle"
    assert status_response.json()["latest_endogenous_round_id"] == 1
    assert why_motivation_response.json()["cause_type"] == "endogenous"
    assert "storage" in replay_motivation_response.json()
    assert what_changed_response.json()["window"] == 1
