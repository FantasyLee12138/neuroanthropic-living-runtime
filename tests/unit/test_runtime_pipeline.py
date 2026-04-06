from pathlib import Path

from nalr.runtime.controller import ACTION_STAGE_BY_OWNER, PIPELINE_TELEMETRY_STAGES, RuntimeController
from nalr.schemas.models import RoundEvent


CONFIG_ROOT = Path(__file__).resolve().parents[2] / "config"


def test_tick_records_v056_pipeline_stages(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    result = controller.tick(
        RoundEvent(
            source="user",
            content="Help me plan dinner and remember rice.",
            target="user",
            cue="rice",
            valence=0.15,
            energy_delta=-0.05,
        ),
        scenario="task",
        mode="interactive",
    )

    assert result.trace.pipeline_stages == [
        "state_update",
        "salience",
        "body",
        "emotion",
        "relationship",
        "resource",
        "pfc",
        "habit",
        "desire",
        "dmn",
        "hippocampus",
        "perspective",
        "value",
        "unconscious",
        "cerebellar",
        "conflict",
        "thalamus",
        "plausibility_guard",
        "forced_mode_switch",
        "output_gate",
        "late_perspective",
        "renderer",
        "writeback",
    ]
    assert result.trace.proposal_summaries
    assert any(item["stage"] == "pfc" for item in result.trace.proposal_summaries)


def test_pipeline_telemetry_and_action_head_use_separate_orders(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    result = controller.tick(
        RoundEvent(
            source="user",
            content="Help me plan dinner and remember rice.",
            target="user",
            cue="rice",
            valence=0.15,
            energy_delta=-0.05,
        ),
        scenario="task",
        mode="interactive",
    )

    proposal_stages = {item["stage"] for item in result.trace.proposal_summaries}
    assert result.trace.pipeline_stages == list(PIPELINE_TELEMETRY_STAGES)
    assert proposal_stages.issubset(set(ACTION_STAGE_BY_OWNER.values()))
    assert "late_perspective" not in proposal_stages
    assert "renderer" not in proposal_stages
    assert "writeback" not in proposal_stages
