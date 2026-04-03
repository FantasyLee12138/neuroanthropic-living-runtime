import json
from pathlib import Path

from typer.testing import CliRunner

from nalr.cli.app import app
from nalr.cil.runtime import CommandInterfaceLayer
from nalr.runtime.controller import RuntimeController
from nalr.schemas.models import RoundEvent


RUNNER = CliRunner()
CONFIG_ROOT = Path(__file__).resolve().parents[2] / "config"


def test_cil_exposes_skill_stats_and_relation_show(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    controller.tick(
        RoundEvent(
            source="user",
            content="Help me remember noodles and plan dinner.",
            target="user",
            cue="noodles",
            valence=0.2,
        ),
        scenario="task",
        mode="interactive",
    )
    cil = CommandInterfaceLayer(controller)

    stats = cil.execute("skill stats")
    relation = cil.execute("relation show user")

    assert stats["total_calls"] > 0
    assert "generate_candidates" in stats["skills"]
    assert relation["target"] == "user"
    assert relation["closeness"] >= 0.5


def test_cli_user_alias_rest_routes_through_cil(tmp_path, monkeypatch):
    monkeypatch.setenv("NALR_HOME", str(tmp_path / ".alive"))
    monkeypatch.setenv("NALR_CONFIG_DIR", str(CONFIG_ROOT))

    result = RUNNER.invoke(app, ["rest"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["applied"] is True
    assert payload["scope"] == "body"
    assert "body_energy" in payload["delta"]


def test_cil_command_trace_records_operator_level_and_rollback(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    cil = CommandInterfaceLayer(controller)

    payload = cil.execute("safe on")

    trace_path = tmp_path / ".alive" / "traces" / "command_traces.json"
    traces = json.loads(trace_path.read_text(encoding="utf-8"))

    assert payload.operator_level == "ops_admin"
    assert payload.rollback_available is True
    assert traces[-1]["operator_level"] == "ops_admin"
    assert traces[-1]["rollback_available"] is True
