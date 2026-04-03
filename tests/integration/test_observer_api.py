from pathlib import Path

from fastapi.testclient import TestClient

from nalr.runtime.controller import RuntimeController
from nalr.schemas.models import RoundEvent
from services.observer.api.app import app as default_app
from services.observer.api.app import create_app


CONFIG_ROOT = Path(__file__).resolve().parents[2] / "config"


def test_observer_reads_state_and_trace(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    controller.tick(RoundEvent(source="user", content="hello runtime"), scenario="chat", mode="interactive")

    client = TestClient(create_app(project_root=tmp_path, config_root=CONFIG_ROOT))

    state_response = client.get("/state")
    trace_response = client.get("/trace/1")
    why_response = client.get("/why/1")
    metrics_response = client.get("/metrics/summary")
    timeline_response = client.get("/metrics/timeline")
    heatmap_response = client.get("/metrics/heatmap")
    replay_response = client.get("/replay/1?seed=5")
    why_not_response = client.get("/why-not/1/rest")
    skill_stats_response = client.get("/skills/stats")
    skill_profile_response = client.get("/skills/profile/generate_candidates")
    dashboard_response = client.get("/dashboard")

    assert state_response.status_code == 200
    assert trace_response.status_code == 200
    assert why_response.status_code == 200
    assert metrics_response.status_code == 200
    assert timeline_response.status_code == 200
    assert heatmap_response.status_code == 200
    assert replay_response.status_code == 200
    assert why_not_response.status_code == 200
    assert skill_stats_response.status_code == 200
    assert skill_profile_response.status_code == 200
    assert dashboard_response.status_code == 200
    assert state_response.json()["round_count"] == 1
    assert trace_response.json()["round_id"] == 1
    assert why_response.json()["round_id"] == 1
    assert metrics_response.json()["total_rounds"] == 1
    assert timeline_response.json()["rounds"]
    assert heatmap_response.json()["actions"]
    assert replay_response.json()["round_id"] == 1
    assert why_not_response.json()["round_id"] == 1
    assert skill_stats_response.json()["skills"]
    assert skill_profile_response.json()["name"] == "generate_candidates"
    assert "NALR Observer" in dashboard_response.text


def test_default_observer_app_is_available():
    assert default_app.title == "NALR Observer"
