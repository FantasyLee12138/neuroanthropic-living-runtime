import json
from pathlib import Path

from typer.testing import CliRunner

from nalr.cli.app import app
from nalr.runtime.controller import RuntimeController
from nalr.schemas.models import RoundEvent


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


def test_cli_checkpoint_rewind_and_trace_why(tmp_path, monkeypatch):
    monkeypatch.setenv("NALR_HOME", str(tmp_path / ".alive"))
    monkeypatch.setenv("NALR_CONFIG_DIR", str(Path(__file__).resolve().parents[2] / "config"))

    RUNNER.invoke(app, ["safe", "on"])
    checkpoint_result = RUNNER.invoke(app, ["checkpoint", "create"])
    RUNNER.invoke(app, ["safe", "off"])
    rewind_result = RUNNER.invoke(app, ["checkpoint", "rewind", "ckpt-0000"])
    RUNNER.invoke(app, ["focus", "show"])
    why_result = RUNNER.invoke(app, ["trace", "why", "1"])

    assert checkpoint_result.exit_code == 0
    assert rewind_result.exit_code == 0
    assert '"safe_mode": true' in rewind_result.stdout
    assert why_result.exit_code == 0
    assert '"top_drivers"' in why_result.stdout


def test_cli_debug_weight_replay_why_not_and_trace_compact(tmp_path, monkeypatch):
    monkeypatch.setenv("NALR_HOME", str(tmp_path / ".alive"))
    monkeypatch.setenv("NALR_CONFIG_DIR", str(Path(__file__).resolve().parents[2] / "config"))

    RUNNER.invoke(app, ["focus", "show"])
    weight_result = RUNNER.invoke(app, ["debug", "weight", "PFCAgent", "0.3"])
    replay_result = RUNNER.invoke(app, ["replay", "round", "1", "--seed", "11"])
    why_not_result = RUNNER.invoke(app, ["why", "not", "rest", "1"])
    compact_result = RUNNER.invoke(app, ["trace", "compact"])

    assert weight_result.exit_code == 0
    assert '"weight": 0.3' in weight_result.stdout
    assert replay_result.exit_code == 0
    assert '"original_action"' in replay_result.stdout
    assert why_not_result.exit_code == 0
    assert '"blocked_by"' in why_not_result.stdout
    assert compact_result.exit_code == 0
    assert '"rows_written"' in compact_result.stdout


def test_cli_skill_relation_and_user_alias_commands(tmp_path, monkeypatch):
    monkeypatch.setenv("NALR_HOME", str(tmp_path / ".alive"))
    monkeypatch.setenv("NALR_CONFIG_DIR", str(Path(__file__).resolve().parents[2] / "config"))

    controller = RuntimeController(project_root=tmp_path, config_root=Path(__file__).resolve().parents[2] / "config")
    controller.tick(
        RoundEvent(source="user", content="hello there", target="user", valence=0.3),
        scenario="companion",
        mode="interactive",
    )

    skill_stats_result = RUNNER.invoke(app, ["skill", "stats"])
    skill_profile_result = RUNNER.invoke(app, ["skill", "profile", "generate_candidates"])
    relation_result = RUNNER.invoke(app, ["relation", "show", "user"])
    rest_result = RUNNER.invoke(app, ["rest"])
    calm_result = RUNNER.invoke(app, ["calm"])

    assert skill_stats_result.exit_code == 0
    assert '"skill_name": "generate_candidates"' in skill_stats_result.stdout
    assert skill_profile_result.exit_code == 0
    assert '"owner_module": "PFCAgent"' in skill_profile_result.stdout
    assert relation_result.exit_code == 0
    assert '"target": "user"' in relation_result.stdout
    assert rest_result.exit_code == 0
    assert '"body_energy"' in rest_result.stdout
    assert calm_result.exit_code == 0
    assert '"mood"' in calm_result.stdout
