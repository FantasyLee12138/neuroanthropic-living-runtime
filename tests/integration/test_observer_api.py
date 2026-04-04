import inspect
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


def test_default_observer_app_is_available():
    assert default_app.title == "NALR Observer"


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
    controller.run_endogenous_tick(trigger="idle")
    controller.flush_pending_io(raise_on_error=True)

    client = TestClient(create_app(project_root=tmp_path, config_root=CONFIG_ROOT))

    state_response = client.get("/state")
    metrics_response = client.get("/metrics/summary")
    timeline_response = client.get("/metrics/timeline")
    trace_response = client.get("/trace/1")

    assert state_response.status_code == 200
    assert metrics_response.status_code == 200
    assert timeline_response.status_code == 200
    assert trace_response.status_code == 200
    assert state_response.json()["subjectivity"]["subject_core_integrity"] is True
    assert "boundary_violation_count" in state_response.json()["subjectivity"]
    assert "external_to_internal_ratio" in metrics_response.json()
    assert "endogenous_intent_rate" in metrics_response.json()
    assert "subjectivity" in timeline_response.json()
    assert "subject_id" in trace_response.json()


def test_observer_probability_field_routes_expose_canonical_field_shape(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    controller.tick(
        RoundEvent(
            source="user",
            content="Plan dinner and remember rice.",
            target="user",
            cue="rice",
            valence=0.15,
        ),
        scenario="task",
        mode="interactive",
    )
    controller.flush_pending_io(raise_on_error=True)

    client = TestClient(create_app(project_root=tmp_path, config_root=CONFIG_ROOT))

    probability_field_response = client.get("/probability-field/1")
    trace_probability_field_response = client.get("/trace/1/probability-field")

    assert probability_field_response.status_code == 200
    assert trace_probability_field_response.status_code == 200
    probability_field = probability_field_response.json()
    trace_probability_field = trace_probability_field_response.json()

    assert probability_field["round_id"] == 1
    assert trace_probability_field["round_id"] == 1
    assert probability_field["schema_version"] == "v0.57/probability-field"
    assert trace_probability_field["schema_version"] == "v0.57/probability-field"
    assert "probability_field" not in probability_field
    assert "probability_field" not in trace_probability_field
    assert probability_field["guard"]["soft_penalty"] >= 0.0
    assert "hard_block" in probability_field["guard"]
    assert trace_probability_field["guard"]["soft_penalty"] >= 0.0
    assert "hard_block" in trace_probability_field["guard"]
    assert probability_field["storage"]["read_source"] == "parquet"
