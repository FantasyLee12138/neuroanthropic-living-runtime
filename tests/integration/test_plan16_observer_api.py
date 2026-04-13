from pathlib import Path
import sys
import time

from fastapi.testclient import TestClient

from nalr.schemas.models import RoundEvent
from nalr.terminal_bridge.handlers import TerminalEventHandler

sys.path.append(str(Path(__file__).resolve().parents[2]))

from services.observer.api.app import create_app


CONFIG_ROOT = Path(__file__).resolve().parents[2] / "config"


def test_plan16_status_endpoints_and_workbench_read_model(tmp_path):
    app = create_app(project_root=tmp_path, config_root=CONFIG_ROOT)
    controller = app.state.controller
    controller.tick(
        RoundEvent(
            source="user",
            content="Inspect the runtime so the workbench has subject, meaning, and agency context.",
            target="user",
            cue="plan16",
        ),
        scenario="chat",
        mode="interactive",
    )

    client = TestClient(app)

    subject_response = client.get("/subject/status")
    meaning_response = client.get("/meaning/status")
    agency_response = client.get("/agency/status")
    performance_response = client.get("/performance/hot-path")
    read_model_response = client.get(
        "/workbench/read-model",
        params={
            "analysis": True,
            "inner_space": True,
            "settings": True,
            "chat_session_id": "workbench-chat",
        },
    )

    assert subject_response.status_code == 200
    assert subject_response.json()["subject_kernel"]["display_name"]

    assert meaning_response.status_code == 200
    assert meaning_response.json()["meaning_system"]["survival_narrative"]

    assert agency_response.status_code == 200
    assert "agency_loop" in agency_response.json()
    assert "proactive_backlog" in agency_response.json()

    assert performance_response.status_code == 200
    assert "latency" in performance_response.json()

    assert read_model_response.status_code == 200
    read_model = read_model_response.json()
    assert "bootstrap" in read_model
    assert "summary" in read_model
    assert "subject" in read_model
    assert "meaning" in read_model
    assert "agency" in read_model
    assert "performance" in read_model
    assert "inner_space" in read_model
    assert "state_truth" in read_model
    assert "memory_evidence" in read_model
    assert "competition_evidence" in read_model
    assert "continuity_evidence" in read_model
    assert read_model["state_truth"]["projection"]["workbench_read_model"] == "derived"


def test_workbench_round_endpoint_returns_single_round_payload(tmp_path):
    app = create_app(project_root=tmp_path, config_root=CONFIG_ROOT)
    controller = app.state.controller
    controller.tick(
        RoundEvent(
            source="user",
            content="Produce one round so the workbench detail endpoint can hydrate it in one request.",
            target="user",
            cue="single-round",
        ),
        scenario="chat",
        mode="interactive",
    )

    client = TestClient(app)
    response = client.get("/workbench/round/1", params={"action": "respond"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["trace"]["round_id"] == 1
    assert payload["why"]["round_id"] == 1
    assert payload["contributions"]["round_id"] == 1
    assert payload["probability"]["source_round_id"] == 1
    assert payload["whyNot"]["round_id"] == 1
    assert payload["replay"]["round_id"] == 1
    assert payload["thought"]["round_id"] == 1
    assert payload["memory_evidence"]["event_log_ref"]["round_trace_ref"] == "round://1"
    assert payload["competition_evidence"]["selected_winner"] == payload["trace"]["sampled_action"]
    assert payload["continuity_evidence"]["continuity_nonce"] == payload["trace"]["continuity_nonce"]


def test_scheduled_task_endpoints_support_upsert_list_and_trigger(tmp_path):
    app = create_app(project_root=tmp_path, config_root=CONFIG_ROOT)
    client = TestClient(app)

    create_response = client.post(
        "/scheduled-tasks",
        json={
            "task_id": "weekly-repo-scan",
            "skill_name": "repo_scan",
            "prompt": "Inspect the repository and summarize notable changes in a read-only way.",
            "fresh_session": True,
            "toolset_policy": {"operator_level": "read_only", "allow_commit": False},
            "schedule": {
                "schedule_type": "hourly",
                "interval_hours": 6,
            },
        },
    )
    list_response = client.get("/scheduled-tasks")

    assert create_response.status_code == 200
    created_task = create_response.json()["task"]
    assert created_task["task_id"] == "weekly-repo-scan"
    assert created_task["next_run_at"]

    assert list_response.status_code == 200
    assert list_response.json()["tasks"][0]["task_id"] == "weekly-repo-scan"

    trigger_response = client.post("/scheduled-tasks/weekly-repo-scan/trigger")
    runs_response = client.get("/scheduled-tasks/runs", params={"task_id": "weekly-repo-scan"})

    assert trigger_response.status_code == 200
    trigger_payload = trigger_response.json()
    assert trigger_payload["task"]["status"] == "running"
    assert trigger_payload["run_record"]["status"] == "running"
    assert trigger_payload["scheduled_session_id"]

    assert runs_response.status_code == 200
    assert runs_response.json()["runs"][0]["task_id"] == "weekly-repo-scan"


def test_scheduled_task_runner_auto_triggers_due_tasks(tmp_path, monkeypatch):
    monkeypatch.setenv("NALR_SCHEDULED_TASK_HEARTBEAT_SECONDS", "0.05")
    app = create_app(project_root=tmp_path, config_root=CONFIG_ROOT)
    client = TestClient(app)

    started_runs: list[tuple[str, str]] = []

    def _fake_start_run(goal: str, **kwargs):
        started_runs.append((goal, str(kwargs.get("operator_level") or "")))
        return {"run_id": f"run-{len(started_runs)}", "status": "running", "goal_summary": goal}

    monkeypatch.setattr(app.state.controller, "start_run", _fake_start_run)

    with client:
        create_response = client.post(
            "/scheduled-tasks",
            json={
                "task_id": "auto-repo-scan",
                "skill_name": "repo_scan",
                "prompt": "Inspect the repository and summarize notable changes in a read-only way.",
                "fresh_session": True,
                "toolset_policy": {"operator_level": "read_only", "allow_commit": False},
                "schedule": {
                    "schedule_type": "hourly",
                    "interval_hours": 1,
                },
            },
        )
        assert create_response.status_code == 200

        due_task = app.state.controller.scheduled_task_store.read_task("auto-repo-scan")
        app.state.controller.scheduled_task_store.upsert_task(due_task, recorded_at="2000-01-01T00:00:00Z")

        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            runs = app.state.controller.scheduled_task_store.list_runs(task_id="auto-repo-scan")
            if runs:
                break
            time.sleep(0.05)
        else:
            raise AssertionError("scheduled task runner did not trigger the due task")

    task = app.state.controller.scheduled_task_store.read_task("auto-repo-scan")
    runs = app.state.controller.scheduled_task_store.list_runs(task_id="auto-repo-scan")
    assert started_runs
    assert task.status == "running"
    assert runs[0].status == "running"
    assert runs[0].task_id == "auto-repo-scan"


def test_workbench_read_model_infers_observer_session_state_from_bootstrap(tmp_path):
    app = create_app(project_root=tmp_path, config_root=CONFIG_ROOT)
    client = TestClient(app)

    session_response = client.post(
        "/web/session/start",
        json={
            "session_id": "observer-main",
            "persist_current": False,
        },
    )
    assert session_response.status_code == 200

    read_model_response = client.get("/workbench/read-model")

    assert read_model_response.status_code == 200
    payload = read_model_response.json()
    assert payload["bootstrap"]["session_attached"] is True
    assert payload["sessionState"]["session"]["session_id"] == "observer-main"


def test_workbench_read_model_uses_light_session_state_payload(tmp_path, monkeypatch):
    app = create_app(project_root=tmp_path, config_root=CONFIG_ROOT)
    client = TestClient(app)

    session_response = client.post(
        "/web/session/start",
        json={
            "session_id": "observer-main",
            "persist_current": False,
        },
    )
    assert session_response.status_code == 200

    def _unexpected_snapshot(self, session_id: str):
        raise AssertionError(f"snapshot_session should stay unused for /workbench/read-model: {session_id}")

    monkeypatch.setattr(TerminalEventHandler, "snapshot_session", _unexpected_snapshot)

    read_model_response = client.get("/workbench/read-model")

    assert read_model_response.status_code == 200
    payload = read_model_response.json()
    assert payload["sessionState"]["session"]["session_id"] == "observer-main"
    assert payload["sessionState"]["console"] is None
    assert payload["sessionState"]["workbench"] == {"cards": []}


def test_workbench_read_model_skips_performance_payload_outside_settings(tmp_path, monkeypatch):
    app = create_app(project_root=tmp_path, config_root=CONFIG_ROOT)
    client = TestClient(app)

    def _unexpected_performance_payload():
        raise AssertionError("runtime_performance_payload should stay unused for default /workbench/read-model")

    monkeypatch.setattr(app.state.controller, "runtime_performance_payload", _unexpected_performance_payload)

    response = client.get("/workbench/read-model")

    assert response.status_code == 200
    payload = response.json()
    assert payload["performance"] is None


def test_workbench_read_model_includes_performance_payload_for_settings(tmp_path):
    app = create_app(project_root=tmp_path, config_root=CONFIG_ROOT)
    client = TestClient(app)

    response = client.get("/workbench/read-model", params={"settings": True})

    assert response.status_code == 200
    payload = response.json()
    assert "latency" in payload["performance"]


def test_observer_lifespan_keeps_background_runners_lazy_until_needed(tmp_path, monkeypatch):
    monkeypatch.setenv("NALR_MONOLOGUE_HEARTBEAT_SECONDS", "0.05")
    monkeypatch.setenv("NALR_INITIATIVE_HEARTBEAT_SECONDS", "0.05")
    app = create_app(project_root=tmp_path, config_root=CONFIG_ROOT)

    monologue_calls = 0
    initiative_calls = 0

    original_prepare_monologue = app.state.controller.prepare_monologue_stream_advance
    original_prepare_initiative = app.state.controller.prepare_initiative_background

    def _count_prepare_monologue(*args, **kwargs):
        nonlocal monologue_calls
        monologue_calls += 1
        return original_prepare_monologue(*args, **kwargs)

    def _count_prepare_initiative(*args, **kwargs):
        nonlocal initiative_calls
        initiative_calls += 1
        return original_prepare_initiative(*args, **kwargs)

    monkeypatch.setattr(app.state.controller, "prepare_monologue_stream_advance", _count_prepare_monologue)
    monkeypatch.setattr(app.state.controller, "prepare_initiative_background", _count_prepare_initiative)

    with TestClient(app) as client:
        response = client.get("/service/status")
        assert response.status_code == 200
        time.sleep(0.2)

    assert monologue_calls == 0
    assert initiative_calls == 0


def test_service_and_bootstrap_skip_autonomy_candidate_projection(tmp_path, monkeypatch):
    app = create_app(project_root=tmp_path, config_root=CONFIG_ROOT)
    client = TestClient(app)

    def _unexpected_candidate_projection(*args, **kwargs):
        raise AssertionError("candidate projection should stay unused for lightweight service/bootstrap payloads")

    monkeypatch.setattr(app.state.controller, "_autonomy_projected_candidate_scores", _unexpected_candidate_projection)

    service_response = client.get("/service/status")
    bootstrap_response = client.get("/web/runtime/bootstrap")

    assert service_response.status_code == 200
    assert bootstrap_response.status_code == 200


def test_workbench_read_model_reuses_single_recent_actions_fetch(tmp_path, monkeypatch):
    app = create_app(project_root=tmp_path, config_root=CONFIG_ROOT)
    client = TestClient(app)

    original_console_recent_actions = app.state.controller.console_recent_actions
    limits: list[int] = []

    def _count_console_recent_actions(*, limit: int = 8):
        limits.append(limit)
        return original_console_recent_actions(limit=limit)

    monkeypatch.setattr(app.state.controller, "console_recent_actions", _count_console_recent_actions)

    response = client.get("/workbench/read-model")

    assert response.status_code == 200
    assert limits == [12]
