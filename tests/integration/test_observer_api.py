from pathlib import Path

from fastapi.testclient import TestClient

from nalr.runtime.controller import RuntimeController
from nalr.schemas.models import RoundEvent
from services.observer.api.app import create_app


CONFIG_ROOT = Path(__file__).resolve().parents[2] / "config"


def test_observer_reads_state_and_trace(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    controller.tick(RoundEvent(source="user", content="hello runtime"), scenario="chat", mode="interactive")

    client = TestClient(create_app(project_root=tmp_path, config_root=CONFIG_ROOT))

    state_response = client.get("/state")
    trace_response = client.get("/trace/1")

    assert state_response.status_code == 200
    assert trace_response.status_code == 200
    assert state_response.json()["round_count"] == 1
    assert trace_response.json()["round_id"] == 1

