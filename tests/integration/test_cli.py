import json
import subprocess
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


def test_cli_chat_prints_human_readable_reply_and_round_summary(tmp_path, monkeypatch):
    monkeypatch.setenv("NALR_HOME", str(tmp_path / ".alive"))
    monkeypatch.setenv("NALR_CONFIG_DIR", str(Path(__file__).resolve().parents[2] / "config"))

    result = RUNNER.invoke(app, ["chat", "帮我记住晚饭想吃面，并规划今晚。"])

    assert result.exit_code == 0
    assert "Round 1" in result.stdout
    assert "Action:" in result.stdout
    assert "NALR:" in result.stdout


def test_cli_chat_json_and_trace_views_support_last_round(tmp_path, monkeypatch):
    monkeypatch.setenv("NALR_HOME", str(tmp_path / ".alive"))
    monkeypatch.setenv("NALR_CONFIG_DIR", str(Path(__file__).resolve().parents[2] / "config"))

    chat_result = RUNNER.invoke(
        app,
        [
            "chat",
            "帮我记住晚饭想吃面，并规划今晚。",
            "--json",
        ],
    )
    agents_result = RUNNER.invoke(app, ["trace", "agents", "last"])
    skills_result = RUNNER.invoke(app, ["trace", "skills", "last"])
    gates_result = RUNNER.invoke(app, ["trace", "gates", "last"])
    why_result = RUNNER.invoke(app, ["trace", "why", "last"])

    assert chat_result.exit_code == 0
    payload = json.loads(chat_result.stdout)
    assert payload["round_id"] == 1
    assert "rendered_expression" in payload
    assert payload["top_drivers"]

    assert agents_result.exit_code == 0
    assert "Agent Proposals" in agents_result.stdout
    assert "PFCAgent" in agents_result.stdout

    assert skills_result.exit_code == 0
    assert "Skill Trace" in skills_result.stdout
    assert "generate_candidates" in skills_result.stdout

    assert gates_result.exit_code == 0
    assert "Gate Decisions" in gates_result.stdout
    assert "OutputGate" in gates_result.stdout

    assert why_result.exit_code == 0
    assert '"round_id": 1' in why_result.stdout


def test_cli_supports_memory_recall_habit_reset_relation_nudge_and_budget_set(tmp_path, monkeypatch):
    monkeypatch.setenv("NALR_HOME", str(tmp_path / ".alive"))
    monkeypatch.setenv("NALR_CONFIG_DIR", str(Path(__file__).resolve().parents[2] / "config"))

    RUNNER.invoke(
        app,
        [
            "chat",
            "Remember that coffee helps me focus every morning.",
            "--target",
            "user",
            "--cue",
            "coffee",
            "--scenario",
            "companion",
        ],
    )

    recall_result = RUNNER.invoke(app, ["memory", "recall", "coffee"])
    reset_result = RUNNER.invoke(app, ["habit", "reset", "coffee"])
    nudge_result = RUNNER.invoke(app, ["nudge", "relation", "user", "trust", "+0.05"])
    budget_result = RUNNER.invoke(app, ["budget", "set", "--cap", "50000"])

    assert recall_result.exit_code == 0
    assert '"cue": "coffee"' in recall_result.stdout
    assert reset_result.exit_code == 0
    assert '"applied": true' in reset_result.stdout
    assert nudge_result.exit_code == 0
    assert '"scope": "relation"' in nudge_result.stdout
    assert budget_result.exit_code == 0
    assert '"budget_remaining": 0.5' in budget_result.stdout


def test_cli_repl_supports_chat_trace_shortcuts_and_runtime_controls(tmp_path, monkeypatch):
    monkeypatch.setenv("NALR_HOME", str(tmp_path / ".alive"))
    monkeypatch.setenv("NALR_CONFIG_DIR", str(Path(__file__).resolve().parents[2] / "config"))

    result = RUNNER.invoke(
        app,
        ["repl"],
        input="帮我记住晚饭想吃面\n/agents\n/skills\n/mode task\n/safe on\n/exit\n",
    )

    assert result.exit_code == 0
    assert "NALR:" in result.stdout
    assert "Agent Proposals" in result.stdout
    assert "Skill Trace" in result.stdout
    assert "mode" in result.stdout
    assert "safe_mode" in result.stdout


def test_cli_trace_agents_last_is_friendly_when_no_rounds_exist(tmp_path, monkeypatch):
    monkeypatch.setenv("NALR_HOME", str(tmp_path / ".alive"))
    monkeypatch.setenv("NALR_CONFIG_DIR", str(Path(__file__).resolve().parents[2] / "config"))

    result = RUNNER.invoke(app, ["trace", "agents", "last"])

    assert result.exit_code == 1
    assert "No rounds yet. Send a message first." in result.stdout


def test_repo_launcher_exposes_alive_help():
    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [str(repo_root / "alive"), "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "chat" in result.stdout
    assert "repl" in result.stdout
