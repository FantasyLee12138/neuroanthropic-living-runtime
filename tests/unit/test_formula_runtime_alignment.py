from pathlib import Path

from nalr.runtime.controller import RuntimeController
from nalr.schemas.models import ActionDistributionState, ProposalBundle, RoundEvent


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
    assert distribution["conflict"]["components"]
    assert distribution["conflict"]["passes"]
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
    assert result.trace.distribution_state["conflict"]["total_score"] >= 0.0


def test_distribution_state_uses_sigma_scale_and_utility_shift_in_shifted_utility(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    state = controller.load_runtime_state()
    scenario_cfg = controller.config["scenarios"]["scenarios"]["task"]
    mode_cfg = controller.config["modes"]["modes"]["interactive"]
    relation_state = {
        "closeness": 0.45,
        "boundary_level": 0.82,
        "relationship_risk": 0.58,
        "privacy_level": 0.75,
    }
    context = {
        "closeness": 0.45,
        "habit_strength": 0.10,
        "valence": -0.10,
    }

    baseline = ProposalBundle(
        owner="ValueAgent",
        confidence=0.92,
        action_preferences={"plan": 0.12},
        delta_p={"plan": 0.12},
        sigma_scale=0.70,
        utility_shift={"plan": 0.00},
        reason="baseline",
    )
    shifted = ProposalBundle(
        owner="ValueAgent",
        confidence=0.92,
        action_preferences={"plan": 0.12},
        delta_p={"plan": 0.12},
        sigma_scale=1.35,
        utility_shift={"plan": 0.22},
        reason="shifted",
    )

    baseline_distribution = controller._build_distribution_state(
        [baseline], state, scenario_cfg, mode_cfg, relation_state, context
    )
    shifted_distribution = controller._build_distribution_state(
        [shifted], state, scenario_cfg, mode_cfg, relation_state, context
    )

    assert shifted_distribution.u_base["plan"] == baseline_distribution.u_base["plan"]
    assert shifted_distribution.u_shifted["plan"] > baseline_distribution.u_shifted["plan"]
    assert shifted_distribution.p_raw["plan"] > baseline_distribution.p_raw["plan"]
    assert shifted_distribution.risk_suppressor["connect"] < 1.0


def test_task_profile_reduces_stochastic_noise_when_control_strength_is_higher(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    state = controller.load_runtime_state()
    state.mode = "interactive"
    state.body_energy = 0.68
    deterministic = {
        "respond": 0.34,
        "plan": 0.30,
        "recall": 0.14,
        "rest": 0.10,
        "clarify": 0.07,
        "connect": 0.03,
        "wander": 0.02,
    }
    distribution_state = ActionDistributionState(
        p_base=dict(deterministic),
        p_raw=dict(deterministic),
        p_mix={},
        p_final={},
        ci={action: 0.22 for action in deterministic},
        gate={action: 1.0 for action in deterministic},
        risk_suppressor={action: 1.0 for action in deterministic},
        resample_idx=0,
    )
    event = RoundEvent(
        source="user",
        content="Please help me plan calmly.",
        target="user",
        valence=-0.12,
    )
    relation_state = {
        "closeness": 0.55,
        "boundary_level": 0.35,
        "relationship_risk": 0.28,
        "privacy_level": 0.40,
    }

    chat_cfg = {"output_warmth_variance": 0.18, "pfc_base_share": 0.22}
    task_cfg = {"output_warmth_variance": 0.18, "pfc_base_share": 0.34}

    _, chat_stochastic = controller._apply_stochastic_layer(
        deterministic,
        distribution_state,
        state,
        event,
        chat_cfg,
        relation_state,
        0.05,
        17,
    )
    _, task_stochastic = controller._apply_stochastic_layer(
        deterministic,
        distribution_state,
        state,
        event,
        task_cfg,
        relation_state,
        0.05,
        17,
    )

    assert task_stochastic.lambda_noise < chat_stochastic.lambda_noise
