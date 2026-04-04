import json
import subprocess
from pathlib import Path

from typer.testing import CliRunner

from nalr.cli.app import app
from nalr.cil.runtime import CommandInterfaceLayer
from nalr.runtime.controller import RuntimeController


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


def test_cil_parse_populates_command_id_for_legacy_cli_commands(tmp_path):
    cil = CommandInterfaceLayer(RuntimeController(project_root=tmp_path, config_root=Path(__file__).resolve().parents[2] / "config"))

    envelope = cil._parse("safe on")

    assert envelope.command_id
    assert envelope.domain == "safe"
    assert envelope.verb == "on"
    assert envelope.canonical == "safe on"
    assert envelope.mutation_scope == "runtime"
    assert envelope.rollback_available is True


def test_cil_parse_normalizes_flags_and_args_for_budget_set(tmp_path):
    cil = CommandInterfaceLayer(RuntimeController(project_root=tmp_path, config_root=Path(__file__).resolve().parents[2] / "config"))

    envelope = cil._parse("budget set --cap 50000")

    assert envelope.command_id
    assert envelope.canonical == "budget set 50000"
    assert envelope.flags == {"cap": True}
    assert envelope.parsed_args == {"value": 50000}
    assert envelope.mutation_scope == "resource"


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

    result = RUNNER.invoke(app, ["chat", "帮我记住晚饭想吃面，并规划今晚。", "--name", "阿澜"])

    assert result.exit_code == 0
    assert "Round 1" in result.stdout
    assert "Action:" in result.stdout
    assert "阿澜:" in result.stdout


def test_cli_chat_open_question_avoids_generic_help_fallback_phrase(tmp_path, monkeypatch):
    monkeypatch.setenv("NALR_HOME", str(tmp_path / ".alive"))
    monkeypatch.setenv("NALR_CONFIG_DIR", str(Path(__file__).resolve().parents[2] / "config"))

    result = RUNNER.invoke(app, ["chat", "你可以做什么？", "--name", "阿澜"])

    assert result.exit_code == 0
    assert "我先顺着你刚才提到的内容继续往下接" not in result.stdout
    assert "我先帮你" not in result.stdout


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
        ["repl", "--name", "阿澜"],
        input="帮我记住晚饭想吃面\n/agents\n/skills\n/mode task\n/safe on\n/exit\n",
    )

    assert result.exit_code == 0
    assert "阿澜:" in result.stdout
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


def test_cli_dream_status_trace_and_metrics(tmp_path, monkeypatch):
    monkeypatch.setenv("NALR_HOME", str(tmp_path / ".alive"))
    monkeypatch.setenv("NALR_CONFIG_DIR", str(Path(__file__).resolve().parents[2] / "config"))

    RUNNER.invoke(
        app,
        [
            "chat",
            "Remember that tea helps me slow down before sleep.",
            "--target",
            "user",
            "--cue",
            "tea",
            "--scenario",
            "companion",
        ],
    )
    RUNNER.invoke(app, ["chat", "Sleep reshape around tea.", "--target", "user", "--cue", "tea", "--mode", "sleep"])

    status_result = RUNNER.invoke(app, ["dream", "status"])
    trace_result = RUNNER.invoke(app, ["dream", "trace", "last"])
    metrics_result = RUNNER.invoke(app, ["dream", "metrics"])

    assert status_result.exit_code == 0
    assert trace_result.exit_code == 0
    assert metrics_result.exit_code == 0
    assert '"enabled": true' in status_result.stdout
    assert '"trigger": "sleep_full"' in trace_result.stdout
    assert '"total_runs": 1' in metrics_result.stdout


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


def test_cli_identity_show_and_set_name(tmp_path, monkeypatch):
    monkeypatch.setenv("NALR_HOME", str(tmp_path / ".alive"))
    monkeypatch.setenv("NALR_CONFIG_DIR", str(Path(__file__).resolve().parents[2] / "config"))

    set_result = RUNNER.invoke(app, ["identity", "set-name", "阿澜"])
    show_result = RUNNER.invoke(app, ["identity", "show"])

    assert set_result.exit_code == 0
    assert show_result.exit_code == 0
    payload = json.loads(show_result.stdout)
    assert payload["display_name"] == "阿澜"
    assert payload["internal_handle"].startswith("nalr-")


def test_cli_run_start_status_explain_pause_resume_and_abort(tmp_path, monkeypatch):
    monkeypatch.setenv("NALR_HOME", str(tmp_path / ".alive"))
    monkeypatch.setenv("NALR_CONFIG_DIR", str(Path(__file__).resolve().parents[2] / "config"))

    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "runner.py").write_text("VALUE = 1\n", encoding="utf-8")

    start_result = RUNNER.invoke(app, ["run", "start", "检查 runner.py 并规划下一步"])
    status_result = RUNNER.invoke(app, ["run", "status"])
    explain_result = RUNNER.invoke(app, ["run", "explain"])
    pause_result = RUNNER.invoke(app, ["run", "pause"])
    resume_result = RUNNER.invoke(app, ["run", "resume"])
    abort_result = RUNNER.invoke(app, ["run", "abort"])

    assert start_result.exit_code == 0
    assert status_result.exit_code == 0
    assert explain_result.exit_code == 0
    assert pause_result.exit_code == 0
    assert resume_result.exit_code == 0
    assert abort_result.exit_code == 0

    start_payload = json.loads(start_result.stdout)
    status_payload = json.loads(status_result.stdout)
    explain_payload = json.loads(explain_result.stdout)
    abort_payload = json.loads(abort_result.stdout)

    assert start_payload["status"] == "running"
    assert status_payload["run_id"] == start_payload["run_id"]
    assert explain_payload["current_step"]["tool_choice"] == "repo_scan"
    assert abort_payload["status"] == "aborted"


def test_cli_root_dream_alias_runs_manual_dream_with_optional_cue(tmp_path, monkeypatch):
    monkeypatch.setenv("NALR_HOME", str(tmp_path / ".alive"))
    monkeypatch.setenv("NALR_CONFIG_DIR", str(Path(__file__).resolve().parents[2] / "config"))

    result = RUNNER.invoke(app, ["Dream", "tea"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["trace"]["mode"] == "sleep"
    assert payload["trace"]["cue"] == "tea"
    assert payload["dream_run_id"].startswith("dream-")


def test_cli_supports_trace_compact_and_counterfactual_commands(tmp_path, monkeypatch):
    monkeypatch.setenv("NALR_HOME", str(tmp_path / ".alive"))
    monkeypatch.setenv("NALR_CONFIG_DIR", str(Path(__file__).resolve().parents[2] / "config"))

    RUNNER.invoke(
        app,
        [
            "chat",
            "Help me plan dinner and remember pasta.",
            "--target",
            "friend",
            "--cue",
            "pasta",
            "--scenario",
            "task",
        ],
    )

    compact_result = RUNNER.invoke(app, ["trace", "compact"])
    why_this_result = RUNNER.invoke(app, ["why", "this", "1"])
    why_not_result = RUNNER.invoke(app, ["why", "not", "rest", "1"])
    changed_result = RUNNER.invoke(app, ["what", "changed", "1"])
    longrun_result = RUNNER.invoke(app, ["eval", "longrun", "3"])

    assert compact_result.exit_code == 0
    assert '"parquet_path"' in compact_result.stdout
    assert why_this_result.exit_code == 0
    assert '"round_id": 1' in why_this_result.stdout
    assert why_not_result.exit_code == 0
    assert '"action": "rest"' in why_not_result.stdout
    assert changed_result.exit_code == 0
    assert '"window": 1' in changed_result.stdout
    assert longrun_result.exit_code == 0
    assert '"generated_rounds": 3' in longrun_result.stdout
