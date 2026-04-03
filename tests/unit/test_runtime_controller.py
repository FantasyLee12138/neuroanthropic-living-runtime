from pathlib import Path

from nalr.runtime.controller import RuntimeController
from nalr.schemas.models import RoundEvent


CONFIG_ROOT = Path(__file__).resolve().parents[2] / "config"


def test_tick_records_trace_and_top_drivers(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    result = controller.tick(
        RoundEvent(
            source="user",
            content="Please help me plan my day and remember breakfast.",
            target="user",
            valence=0.2,
            energy_delta=-0.1,
        ),
        scenario="task",
        mode="interactive",
    )

    assert result.round_id == 1
    assert result.sampled_action.name in {"respond", "plan", "recall", "rest"}
    assert len(result.trace.top_drivers) == 3
    assert any(item.agent_name == "PFCAgent" for item in result.trace.contributions)


def test_command_safe_mode_and_checkpoint_emit_command_trace(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    safe_result = controller.apply_command("safe on")
    checkpoint = controller.checkpoint()

    assert safe_result.applied is True
    assert safe_result.scope == "runtime"
    assert safe_result.delta["safe_mode"] is True
    assert checkpoint.checkpoint_id.startswith("ckpt-")
    assert controller.load_runtime_state().safe_mode is True


def test_rewind_restores_checkpointed_runtime_state(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    controller.apply_command("safe on")
    checkpoint = controller.checkpoint()
    controller.apply_command("safe off")
    controller.apply_command("mode set idle")

    rewind_result = controller.rewind(checkpoint.checkpoint_id)
    state = controller.load_runtime_state()

    assert rewind_result.applied is True
    assert rewind_result.scope == "checkpoint"
    assert state.safe_mode is True
    assert state.mode == "safe"


def test_why_this_and_metrics_summary_surface_trace_evidence(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    controller.tick(
        RoundEvent(source="user", content="Help me plan lunch and remember noodles.", target="user", cue="noodles"),
        scenario="chat",
        mode="interactive",
    )

    why_payload = controller.why_this(1)
    metrics = controller.metrics_summary()

    assert why_payload["round_id"] == 1
    assert why_payload["top_drivers"]
    assert metrics["total_rounds"] == 1
    assert metrics["sampled_actions"]


def test_tick_exposes_conflict_and_guard_trace_fields(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    result = controller.tick(
        RoundEvent(
            source="user",
            content="Help me plan a careful but emotionally warm reply while also remembering brunch.",
            target="friend",
            cue="brunch",
            valence=0.4,
        ),
        scenario="companion",
        mode="interactive",
    )

    assert "conflict_score" in result.trace.state_snapshot
    assert "plausibility_fail_score" in result.trace.state_snapshot
    assert all(hasattr(item, "confidence") for item in result.trace.contributions)
    assert result.trace.state_snapshot["last_render_provider"]


def test_replay_why_not_and_what_changed_return_counterfactuals(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    controller.tick(
        RoundEvent(source="user", content="Help me plan dinner and remember pasta.", target="friend", cue="pasta"),
        scenario="task",
        mode="interactive",
    )
    controller.tick(
        RoundEvent(source="user", content="Remember pasta again but be careful.", target="friend", cue="pasta"),
        scenario="companion",
        mode="interactive",
    )

    replay_payload = controller.replay(1, seed=7)
    why_not_payload = controller.why_not(2, "rest")
    changed_payload = controller.what_changed(window=2)

    assert replay_payload["round_id"] == 1
    assert "original_action" in replay_payload
    assert "ablations" in replay_payload
    assert why_not_payload["round_id"] == 2
    assert why_not_payload["action"] == "rest"
    assert why_not_payload["blocked_by"]
    assert changed_payload["window"] == 2
    assert changed_payload["action_counts"]


def test_safe_mode_disables_dmn_and_perspective_contributions(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    controller.apply_command("safe on")

    result = controller.tick(
        RoundEvent(source="user", content="I am drifting and need a gentle response.", target="friend"),
        scenario="companion",
        mode="idle",
    )

    names = {item.agent_name for item in result.trace.contributions}
    assert "DMNAgent" not in names
    assert "PerspectiveModel" not in names
