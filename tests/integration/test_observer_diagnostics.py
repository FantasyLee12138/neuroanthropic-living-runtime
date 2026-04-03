from pathlib import Path

from fastapi.testclient import TestClient

from nalr.runtime.controller import RuntimeController
from nalr.schemas.models import RoundEvent
from services.observer.api.app import create_app


CONFIG_ROOT = Path(__file__).resolve().parents[2] / "config"


def test_observer_exposes_skill_conflict_mode_and_ablation_views(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    for idx in range(25):
        controller.tick(
            RoundEvent(
                source="user",
                content=f"help me remember tea round {idx}",
                target="user",
                cue="tea",
                valence=0.1 if idx % 2 == 0 else -0.1,
            ),
            scenario="task",
            mode="interactive",
        )

    client = TestClient(create_app(project_root=tmp_path, config_root=CONFIG_ROOT))

    skill_response = client.get("/skills/stats")
    conflict_response = client.get("/metrics/conflicts")
    mode_response = client.get("/metrics/mode-switches")
    ablation_response = client.get("/analysis/ablation")

    assert skill_response.status_code == 200
    assert conflict_response.status_code == 200
    assert mode_response.status_code == 200
    assert ablation_response.status_code == 200
    assert skill_response.json()["total_calls"] > 0
    assert conflict_response.json()["points"]
    assert mode_response.json()["points"]
    assert ablation_response.json()["modules"]
    assert "approx_gain" in ablation_response.json()["modules"][0]
    assert "coverage" in ablation_response.json()["modules"][0]
