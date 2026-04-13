import json
from pathlib import Path

from typer.testing import CliRunner

from nalr.cli.app import app
from nalr.runtime.controller import RuntimeController


CONFIG_ROOT = Path(__file__).resolve().parents[2] / "config"
RUNNER = CliRunner()


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
