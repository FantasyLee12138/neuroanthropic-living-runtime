import json
from pathlib import Path

from typer.testing import CliRunner

from nalr.cli.app import app
from nalr.runtime.controller import RuntimeController
from nalr.schemas.models import RoundEvent


CONFIG_ROOT = Path(__file__).resolve().parents[2] / "config"
RUNNER = CliRunner()
EXPECTED_CHAIN_LAYERS = ["perception", "memory", "cognition", "decision", "execution", "feedback"]


def test_acceptance_report_surfaces_15_controlled_learning_summary(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    payload = controller.acceptance_report(window=1)
    summary = payload["release_15"]["controlled_learning"]

    assert summary["learning_mode"] == "guided-learn"
    assert summary["trace_external_learning"] is True
    assert "docs.python.org" in summary["allowed_network_domains"]
    assert summary["learning_log_dir"] == str(tmp_path / ".alive" / "learning")
    assert str(tmp_path) in summary["writable_roots"]


def test_acceptance_report_uses_current_observer_learning_preferences(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    controller.update_observer_settings(
        {
            "autonomy": {
                "learning_mode": "active-learn",
                "allowed_network_domains": ["example.com"],
                "writable_roots": [str(tmp_path / "workspace")],
                "knowledge_roots": [str(tmp_path / "docs")],
                "learning_log_dir": str(tmp_path / ".alive" / "learning-cache"),
                "trace_external_learning": False,
            }
        }
    )

    payload = controller.acceptance_report(window=1)
    summary = payload["release_15"]["controlled_learning"]

    assert summary["learning_mode"] == "active-learn"
    assert summary["allowed_network_domains"] == ["example.com"]
    assert summary["writable_roots"] == [str(tmp_path / "workspace")]
    assert summary["knowledge_roots"] == [str(tmp_path / "docs")]
    assert summary["learning_log_dir"] == str(tmp_path / ".alive" / "learning-cache")
    assert summary["trace_external_learning"] is False


def test_release_acceptance_report_surfaces_six_layer_chain_and_layer_replay(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    controller.tick(
        RoundEvent(
            source="user",
            content="Remember tea and explain the full reasoning chain.",
            target="user",
            cue="tea",
            valence=0.2,
        ),
        scenario="chat",
        mode="interactive",
    )

    why_payload = controller.why_this(1)
    layer_chain = controller.layer_chain(1)
    report = controller.acceptance_report(window=1)

    assert [item["layer"] for item in why_payload["cognitive_chain"]] == EXPECTED_CHAIN_LAYERS
    assert [item["layer"] for item in layer_chain["cognitive_chain"]] == EXPECTED_CHAIN_LAYERS
    assert report["rounds_considered"] == 1
    assert report["release_15"]["controlled_learning"]["learning_mode"] == "guided-learn"
    for layer in EXPECTED_CHAIN_LAYERS:
        replay_payload = controller.replay_layer(1, layer)
        assert replay_payload["round_id"] == 1
        assert replay_payload["layer"] == layer
        assert replay_payload["snapshot"]["input_vector"]
        assert replay_payload["snapshot"]["transform_summary"]


def test_release_acceptance_report_surfaces_control_proposal_auditability_and_observer_parity(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    proposal = controller.create_control_proposal(
        {
            "title": "Tighten initiative cadence",
            "target": "initiative",
            "patch": {
                "behavior_policies": {
                    "initiative": {
                        "trigger_interval_minutes": 15,
                    }
                }
            },
        }
    )

    controller.apply_control_proposal(proposal["proposal_id"], approved=True)
    controller.promote_control_proposal(proposal["proposal_id"])

    controls = controller.controls_current()
    observer = controller.observer_settings_payload()
    report = controller.acceptance_report(window=1)
    proposal_entry = next(item for item in controls["proposals"] if item["proposal_id"] == proposal["proposal_id"])
    command_names = [item["command"] for item in controller.trace_store.list_commands()]

    assert proposal_entry["status"] == "promoted"
    assert proposal_entry["approved"] is True
    assert proposal_entry["created_at"]
    assert proposal_entry["approved_at"]
    assert proposal_entry["promoted_at"]
    assert controls["layer_controls"]["behavior_policies"]["initiative"]["trigger_interval_minutes"] == 15
    assert controls["layer_controls"] == observer["runtime"]["layer_controls"]
    assert report["release_15"]["controlled_learning"]["learning_mode"] == observer["autonomy"]["learning_mode"]
    assert report["release_15"]["controlled_learning"]["allowed_network_domains"] == observer["autonomy"]["allowed_network_domains"]
    assert f"controls proposal create {proposal['proposal_id']}" in command_names
    assert f"controls proposal apply {proposal['proposal_id']}" in command_names
    assert f"controls proposal promote {proposal['proposal_id']}" in command_names


def test_eval_acceptance_report_command_surfaces_release_15_controlled_learning_summary(tmp_path, monkeypatch):
    monkeypatch.setenv("NALR_HOME", str(tmp_path / ".alive"))
    monkeypatch.setenv("NALR_CONFIG_DIR", str(CONFIG_ROOT))

    result = RUNNER.invoke(app, ["eval", "acceptance-report", "--window", "1"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    summary = payload["release_15"]["controlled_learning"]

    assert summary["learning_mode"] == "guided-learn"
    assert summary["trace_external_learning"] is True
    assert "docs.python.org" in summary["allowed_network_domains"]
