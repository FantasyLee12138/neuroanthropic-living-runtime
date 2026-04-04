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
