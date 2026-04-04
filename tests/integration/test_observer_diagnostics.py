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
    entropy_response = client.get("/metrics/entropy")

    assert skill_response.status_code == 200
    assert conflict_response.status_code == 200
    assert mode_response.status_code == 200
    assert ablation_response.status_code == 200
    assert entropy_response.status_code == 200
    assert skill_response.json()["total_calls"] > 0
    assert conflict_response.json()["points"]
    assert "components" in conflict_response.json()["points"][0]
    assert "critical_conflict_streak" in conflict_response.json()["points"][0]
    assert mode_response.json()["points"]
    assert ablation_response.json()["modules"]
    assert entropy_response.json()["provider_class"]
    assert "approx_gain" in ablation_response.json()["modules"][0]
    assert "coverage" in ablation_response.json()["modules"][0]


def test_observer_conflict_timeline_includes_repair_visibility(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    state = controller.load_runtime_state()
    state.body_energy = 0.06
    state.budget_remaining = 0.03
    controller._save_state(state)

    for _ in range(3):
        controller.tick(
            RoundEvent(
                source="user",
                content="I need a quick easy break, but help me plan carefully, reply to Alex, and let me drift.",
                target="alex",
                cue="break",
                valence=-0.45,
                energy_delta=-0.20,
            ),
            scenario="task",
            mode="interactive",
        )

    client = TestClient(create_app(project_root=tmp_path, config_root=CONFIG_ROOT))
    conflict_response = client.get("/metrics/conflicts")

    assert conflict_response.status_code == 200
    point = conflict_response.json()["points"][-1]
    assert point["repair_mode"] == "deadlock_fuse"
    assert point["repair_stage"] == "repairing"
    assert point["conflict_safe_mode_owned"] is True
    assert point["last_post_error_adjustment"]["triggered"] is True
    assert point["repair_ledger_summary"]["entries"] >= 1
    assert point["repair_learning"]["adjustment_reasons"]


def test_observer_exposes_identity_and_authenticity_metrics(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    controller.seed_identity_name("阿澜")
    controller.tick(
        RoundEvent(
            source="user",
            content="你是谁？",
            target="user",
            valence=0.1,
        ),
        scenario="chat",
        mode="interactive",
    )

    client = TestClient(create_app(project_root=tmp_path, config_root=CONFIG_ROOT))

    identity_response = client.get("/identity")
    authenticity_response = client.get("/metrics/authenticity")

    assert identity_response.status_code == 200
    assert authenticity_response.status_code == 200
    assert identity_response.json()["display_name"] == "阿澜"
    assert authenticity_response.json()["points"]
    assert authenticity_response.json()["points"][0]["query_kind"] == "self_identity"
    assert authenticity_response.json()["points"][0]["query_intent"] == "self_model_identity_probe"
    assert "disclosure_intent" in authenticity_response.json()["points"][0]
    assert "guard_action" in authenticity_response.json()["points"][0]


def test_observer_exposes_vitality_metrics_and_noninteractive_events(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    controller.tick(
        RoundEvent(source="user", content="记住 tea 这条线索", target="user", cue="tea", valence=-0.15),
        scenario="companion",
        mode="interactive",
    )
    controller.tick(
        RoundEvent(source="system", content="idle shaping tick", target="user", cue="tea"),
        scenario="companion",
        mode="idle",
    )

    client = TestClient(create_app(project_root=tmp_path, config_root=CONFIG_ROOT))

    vitality_response = client.get("/metrics/vitality")
    summary_response = client.get("/metrics/summary")

    assert vitality_response.status_code == 200
    assert summary_response.status_code == 200
    assert vitality_response.json()["points"]
    assert vitality_response.json()["points"][-1]["non_interactive_events"]
    assert "cue_recall_success_rate" in summary_response.json()
