from pathlib import Path

from nalr.runtime.controller import RuntimeController
from nalr.schemas.models import RoundEvent


CONFIG_ROOT = Path(__file__).resolve().parents[2] / "config"


def test_tick_records_formula_distribution_state_and_richer_proposal_fields(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    result = controller.tick(
        RoundEvent(
            source="user",
            content="Help me plan dinner, remember noodles, and keep it calm.",
            target="user",
            cue="noodles",
            valence=-0.15,
            energy_delta=-0.05,
        ),
        scenario="task",
        mode="interactive",
    )

    distribution = result.trace.distribution_state

    assert distribution["p_base"]
    assert distribution["p_raw"]
    assert distribution["p_final"]
    assert distribution["gate"]
    assert distribution["ci"]
    assert abs(sum(distribution["p_final"].values()) - 1.0) < 1e-6
    assert all(0.01 <= value <= 0.85 for value in distribution["p_raw"].values())

    proposal = next(item for item in result.trace.proposal_summaries if item["stage"] == "pfc")
    assert "delta_p" in proposal
    assert "sigma_scale" in proposal
    assert "weight_applied" in proposal
    assert "selected" in proposal


def test_tick_executes_conflict_thalamus_and_output_skills_independently(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    result = controller.tick(
        RoundEvent(
            source="user",
            content="Please help me think carefully, but I also want to wander and rest.",
            target="user",
            cue="rest",
            valence=0.05,
        ),
        scenario="chat",
        mode="interactive",
    )

    skill_names = [item["skill_name"] for item in result.trace.skill_traces]

    assert "score_conflict" in skill_names
    assert "aggregate_proposals" in skill_names
    assert "normalize_distribution" in skill_names
    assert "sample_action" in skill_names
    assert "apply_output_gate" in skill_names
    assert "render_tone_profile" in skill_names
    assert "compute_delay_profile" in skill_names
