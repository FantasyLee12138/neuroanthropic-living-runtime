import json
from pathlib import Path

from typer.testing import CliRunner

from nalr.cli.app import app


RUNNER = CliRunner()


def test_cli_state_show_and_safe_on(tmp_path, monkeypatch):
    monkeypatch.setenv("NALR_HOME", str(tmp_path / ".alive"))
    monkeypatch.setenv("NALR_CONFIG_DIR", str(Path(__file__).resolve().parents[2] / "config"))

    safe_result = RUNNER.invoke(app, ["safe", "on"])
    state_result = RUNNER.invoke(app, ["state", "show"])

    assert safe_result.exit_code == 0
    assert state_result.exit_code == 0

    payload = json.loads(state_result.stdout)
    assert payload["safe_mode"] is True
    assert payload["mode"] == "safe"


def test_cli_agent_list_and_trace_round(tmp_path, monkeypatch):
    monkeypatch.setenv("NALR_HOME", str(tmp_path / ".alive"))
    monkeypatch.setenv("NALR_CONFIG_DIR", str(Path(__file__).resolve().parents[2] / "config"))

    RUNNER.invoke(app, ["focus", "show"])
    list_result = RUNNER.invoke(app, ["agent", "list"])
    trace_result = RUNNER.invoke(app, ["trace", "round", "1"])

    assert list_result.exit_code == 0
    assert "PFCAgent" in list_result.stdout
    assert trace_result.exit_code == 0
    assert '"round_id": 1' in trace_result.stdout

