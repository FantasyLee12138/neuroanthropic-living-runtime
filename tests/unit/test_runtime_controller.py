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
