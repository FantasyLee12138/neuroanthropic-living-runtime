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
    profile = cil.execute("skill profile generate_candidates")
    relation = cil.execute("relation show user")

    assert stats["total_calls"] > 0
    assert "generate_candidates" in stats["skills"]
    assert stats["skills"]["generate_candidates"]["fallback_count"] >= 1
    assert profile["skill_name"] == "generate_candidates"
    assert profile["skills"]["generate_candidates"]["fallback_count"] >= 1
    assert relation["target"] == "user"
    assert relation["closeness"] >= 0.5


def test_cil_exposes_endogenous_tick_status_and_motivation_views(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    cil = CommandInterfaceLayer(controller)

    tick = cil.endogenous_tick(trigger="idle")
    status = cil.endogenous_status()
    why_motivation = cil.execute(f"why motivation {tick['round_id']}")
    replay_motivation = cil.execute(f"replay motivation {tick['round_id']}")

    assert tick["cause_type"] == "endogenous"
    assert tick["round_id"] == 1
    assert status["latest_trigger"]["trigger_type"] == "idle"
    assert status["latest_endogenous_round_id"] == tick["round_id"]
    assert status["last_trigger"] == "idle"
    assert "suppression_reason" in status
    assert why_motivation["cause_type"] == "endogenous"
    assert why_motivation["storage"]["read_source"] in {"json", "parquet"}
    assert replay_motivation["endogenous_tick_reason"]
    assert replay_motivation["storage"]["read_source"] in {"json", "parquet"}


def test_cil_exposes_initiative_status_distribution_and_trigger(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    cil = CommandInterfaceLayer(controller)
    controller.initiative_update_settings(
        idle_seconds_threshold=0,
        vitality_threshold=0.0,
        relation_strength_threshold=0.0,
        habit_strength_threshold=0.0,
        proposal_posterior_threshold=0.0,
        cooldown_seconds=0,
        hourly_limit=4,
        require_memory_backing=False,
        auto_send_enabled=False,
    )

    status = cil.execute("initiative status")
    distribution = cil.execute("initiative distribution")
    trigger = cil.execute("initiative trigger idle")

    assert "settings" in status
    assert "posterior" in distribution
    assert "proposal" in trigger


def test_cil_initiative_trigger_supports_force_flag(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    cil = CommandInterfaceLayer(controller)

    payload = cil.execute("initiative trigger idle force")

    assert payload["forced"] is True


def test_cil_thought_show_surfaces_snapshot(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    cil = CommandInterfaceLayer(controller)
    controller.tick(
        RoundEvent(
            source="user",
            content="帮我判断现在先做什么。",
            target="user",
            cue="优先级",
        ),
        scenario="companion",
        mode="interactive",
    )

    payload = cil.execute("thought show last")

    assert payload["round_id"] == 1
    assert "thought_summary" in payload
    assert "action_field" in payload


def test_cil_exposes_monologue_status_and_show(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    cil = CommandInterfaceLayer(controller)

    status = cil.execute("monologue status")
    visible = cil.execute("monologue show 4")

    assert status["hidden"] is True
    assert status["generated_total"] >= 1
    assert len(visible["fragments"]) == 4


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
    controller.flush_pending_io(raise_on_error=True)

    trace_path = tmp_path / ".alive" / "traces" / "command_traces.json"
    traces = json.loads(trace_path.read_text(encoding="utf-8"))

    assert payload.operator_level == "ops_admin"
    assert payload.rollback_available is True
    assert payload.command_id
    assert payload.canonical == "safe on"
    assert payload.parsed_args == {}
    assert traces[-1]["operator_level"] == "ops_admin"
    assert traces[-1]["rollback_available"] is True
    assert traces[-1]["command_id"] == payload.command_id
    assert traces[-1]["canonical"] == "safe on"
    assert traces[-1]["parsed_args"] == {}


def test_cil_mutation_records_snapshot_and_can_restore_it(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    cil = CommandInterfaceLayer(controller)

    payload = cil.execute("safe on")
    snapshot_id = payload.snapshot_id
    rollback = payload.rollback

    assert snapshot_id
    assert rollback["strategy"] == "domain_inverse"
    assert rollback["snapshot_id"] == snapshot_id
    assert rollback["command"] == "safe off"

    restored = cil.execute(rollback["command"])
    state = controller.load_runtime_state()
    controller.flush_pending_io(raise_on_error=True)
    traces = json.loads((tmp_path / ".alive" / "traces" / "command_traces.json").read_text(encoding="utf-8"))

    assert restored.applied is True
    assert restored.scope == "runtime"
    assert state.safe_mode is False
    assert traces[-2]["snapshot_id"] == snapshot_id
    assert traces[-1]["command"] == rollback["command"]


def test_cil_snapshot_restore_remains_available_for_snapshot_rollback_commands(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    controller.tick(
        RoundEvent(
            source="user",
            content="Remember that tea helps me slow down.",
            target="user",
            cue="tea",
            valence=0.1,
        ),
        scenario="companion",
        mode="interactive",
    )
    cil = CommandInterfaceLayer(controller)
    before = controller.memory_store.habit_strength("tea")
    payload = cil.execute("habit reset tea")
    rollback = payload.rollback
    restored = cil.execute(rollback["command"])
    after = controller.memory_store.habit_strength("tea")
    traces = json.loads((tmp_path / ".alive" / "traces" / "command_traces.json").read_text(encoding="utf-8"))

    assert restored.scope == "snapshot"
    assert rollback["strategy"] == "snapshot_restore"
    assert before > 0.0
    assert after == before
    assert traces[-2]["rollback"]["strategy"] == "snapshot_restore"
    assert traces[-1]["command"] == rollback["command"]


def test_cil_legacy_mutations_are_boundary_mediated_and_traced(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    controller.tick(
        RoundEvent(
            source="user",
            content="Remember that coffee helps me focus every morning.",
            target="user",
            cue="coffee",
            valence=0.25,
        ),
        scenario="companion",
        mode="interactive",
    )
    cil = CommandInterfaceLayer(controller)

    before_relation = cil.execute("relation show user")
    recall = cil.execute("memory recall coffee")
    reset = cil.execute("habit reset coffee")
    nudge = cil.execute("nudge relation user trust +0.05")
    budget = cil.execute("budget set 50000")
    checkpoint = cil.execute("checkpoint create")
    traces = json.loads((tmp_path / ".alive" / "traces" / "command_traces.json").read_text(encoding="utf-8"))

    assert recall["cue"] == "coffee"
    assert recall["strength"] > 0.0
    assert recall["tier"] in {"hot", "warm", "cold"}
    assert reset.applied is True
    assert reset.boundary_action == "proposal_route"
    assert reset.cause_type == "external_stimulus"
    assert reset.deprecation_warning
    assert controller.memory_store.habit_strength("coffee") == 0.0
    assert nudge.applied is True
    assert nudge.boundary_action == "proposal_route"
    assert nudge.cause_type == "external_stimulus"
    assert nudge.deprecation_warning
    assert controller.relation_show("user")["closeness"] > before_relation["closeness"]
    assert budget.applied is True
    assert budget.boundary_action == "downgrade_to_stimulus"
    assert budget.cause_type == "external_stimulus"
    assert budget.deprecation_warning
    assert controller.load_runtime_state().budget_remaining == 0.5
    assert checkpoint.applied is True
    assert checkpoint.delta["checkpoint_id"].startswith("ckpt-")
    assert traces[-4]["boundary_action"] == "proposal_route"
    assert traces[-3]["boundary_action"] == "proposal_route"
    assert traces[-2]["boundary_action"] == "downgrade_to_stimulus"
    assert any(item["command"] == "checkpoint create" for item in traces)


def test_cil_rejects_identity_set_name_after_subject_bootstrap(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    cil = CommandInterfaceLayer(controller)

    payload = cil.execute("identity set-name 阿澜")
    identity = cil.execute("identity show")
    controller.flush_pending_io(raise_on_error=True)
    traces = json.loads((tmp_path / ".alive" / "traces" / "command_traces.json").read_text(encoding="utf-8"))

    assert payload.applied is False
    assert payload.scope == "identity"
    assert payload.operator_level == "soft_intervene"
    assert payload.boundary_action == "reject"
    assert payload.cause_type == "external_stimulus"
    assert payload.violation_code == "identity_seed_locked"
    assert controller.load_runtime_state().safe_mode is True
    assert identity["display_name"] != "阿澜"
    assert identity["internal_handle"].startswith("nalr-")
    assert traces[-1]["command"] == "identity set-name 阿澜"
    assert traces[-1]["boundary_action"] == "reject"
    assert traces[-1]["violation_code"] == "identity_seed_locked"


def test_cil_supports_run_lifecycle_commands(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "agent.py").write_text("VALUE = 1\n", encoding="utf-8")
    cil = CommandInterfaceLayer(controller)

    started = cil.execute("run start 检查 agent.py")
    status = cil.execute("run status")
    explained = cil.execute("run explain")
    paused = cil.execute("run pause")
    resumed = cil.execute("run resume")
    aborted = cil.execute("run abort")

    assert started["status"] == "running"
    assert status["run_id"] == started["run_id"]
    assert explained["current_step"]["tool_choice"] == "repo_scan"
    assert paused["status"] == "paused"
    assert resumed["status"] == "running"
    assert aborted["status"] == "aborted"
