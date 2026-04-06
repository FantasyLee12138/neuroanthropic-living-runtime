import math
from pathlib import Path

from nalr.runtime.controller import RuntimeController
from nalr.schemas.models import ActionBookkeepingState, ActionEvidenceSignal, RoundEvent, RuntimeState


CONFIG_ROOT = Path(__file__).resolve().parents[2] / "config"


def test_tick_records_formula_action_bookkeeping_and_richer_proposal_fields(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    controller.entropy_pool.ingest_bytes(bytes(range(256)) * 8, source="fixture_qrng", reason="formula alignment")

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

    action_bookkeeping = result.trace.action_bookkeeping
    action_layer = result.trace.probability_field["action"]

    assert action_bookkeeping["p_base"]
    assert action_bookkeeping["gate"]
    assert action_bookkeeping["ci"]
    assert action_bookkeeping["risk_suppressor"]
    assert action_bookkeeping["query_intent"]["posterior"]
    assert action_bookkeeping["disclosure_intent"]["posterior"]
    assert abs(sum(action_layer["winner_posterior"].values()) - 1.0) <= 1e-5
    assert all(0.0 <= value <= 1.0 for value in action_bookkeeping["p_base"].values())
    assert "distribution_state" not in controller.trace_round(result.round_id)
    assert result.trace.stochastic_state["entropy_ref"]["source"]
    assert "batch_id" in result.trace.stochastic_state["entropy_ref"]
    assert 0.0 <= result.trace.stochastic_state["v_t"] <= 1.0
    assert "v_t_components" in result.trace.stochastic_state
    assert "sigma_emo" in result.trace.stochastic_state
    assert "sigma_mood" in result.trace.stochastic_state
    assert "entropy_refs_by_node" in result.trace.stochastic_state
    assert result.trace.stochastic_state["entropy_refs_by_node"]["xi_emo"]["source"]
    assert result.trace.stochastic_state["entropy_refs_by_node"]["action_sample"]["source"]

    proposal = next(item for item in result.trace.proposal_summaries if item["stage"] == "pfc")
    assert "delta_p" in proposal
    assert "sigma_scale" in proposal
    assert "weight_applied" in proposal
    assert "selected" in proposal


def test_tick_executes_conflict_and_output_skills_without_thalamus_sampling(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    controller.entropy_pool.ingest_bytes(bytes(range(256)) * 8, source="fixture_qrng", reason="skill fanout")

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
    assert "apply_output_gate" in skill_names
    assert "render_tone_profile" in skill_names
    assert "compute_delay_profile" in skill_names
    assert "aggregate_proposals" not in skill_names
    assert "normalize_distribution" not in skill_names
    assert "sample_action" not in skill_names
    assert result.trace.conflict_arbitration["total_score"] >= 0.0


def test_action_bookkeeping_uses_sigma_scale_and_utility_shift_in_shifted_utility(tmp_path):
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

    baseline = ActionEvidenceSignal(
        module_name="ValueAgent",
        module_type="value",
        confidence=0.92,
        action_delta={"plan": 0.12},
        sigma_scale=0.70,
        utility_shift={"plan": 0.00},
        trace_reason="baseline",
    )
    shifted = ActionEvidenceSignal(
        module_name="ValueAgent",
        module_type="value",
        confidence=0.92,
        action_delta={"plan": 0.12},
        sigma_scale=1.35,
        utility_shift={"plan": 0.22},
        trace_reason="shifted",
    )

    baseline_bookkeeping = controller._build_action_bookkeeping(
        [baseline], state, scenario_cfg, mode_cfg, relation_state, context
    )
    shifted_bookkeeping = controller._build_action_bookkeeping(
        [shifted], state, scenario_cfg, mode_cfg, relation_state, context
    )

    assert shifted_bookkeeping.u_base["plan"] == baseline_bookkeeping.u_base["plan"]
    assert shifted_bookkeeping.p_base["plan"] == baseline_bookkeeping.p_base["plan"]
    assert shifted_bookkeeping.risk_suppressor["connect"] < 1.0


def test_task_profile_reduces_stochastic_noise_when_control_strength_is_higher(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    controller.entropy_pool.ingest_bytes(bytes(range(256)) * 4, source="fixture_qrng", reason="stochastic test")
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
    action_bookkeeping = ActionBookkeepingState(
        p_base=dict(deterministic),
        ci={action: 0.22 for action in deterministic},
        gate={action: 1.0 for action in deterministic},
        risk_suppressor={action: 1.0 for action in deterministic},
    )
    control_ledger = {
        "ci": dict(action_bookkeeping.ci),
        "gate": dict(action_bookkeeping.gate),
        "risk_suppressor": dict(action_bookkeeping.risk_suppressor),
    }
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
        action_bookkeeping,
        control_ledger,
        state,
        event,
        chat_cfg,
        relation_state,
        0.05,
        17,
        action_energy={action: math.log(max(value, 1e-9)) for action, value in deterministic.items()},
    )
    _, task_stochastic = controller._apply_stochastic_layer(
        deterministic,
        action_bookkeeping,
        control_ledger,
        state,
        event,
        task_cfg,
        relation_state,
        0.05,
        17,
        action_energy={action: math.log(max(value, 1e-9)) for action, value in deterministic.items()},
    )

    assert task_stochastic.lambda_noise < chat_stochastic.lambda_noise


def test_tick_records_resource_biases_and_temperament_drift_state(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    controller.entropy_pool.ingest_bytes(bytes(range(256)) * 8, source="fixture_qrng", reason="slow state trace")
    state = controller.load_runtime_state()
    state.budget_remaining = 0.18
    controller._save_state(state)

    controller.tick(
        RoundEvent(
            source="user",
            content="Please help me plan carefully even though I am drained and low on budget.",
            target="user",
            cue="budget",
            valence=-0.35,
            energy_delta=-0.12,
        ),
        scenario="task",
        mode="interactive",
    )

    updated = controller.load_runtime_state()
    resource_state = updated.resource_state
    temperament_state = updated.temperament_state

    assert resource_state["scarcity_index"] > 0.0
    assert resource_state["body_hunger_bias"] > 0.10
    assert resource_state["effort_avoidance_bias"] > 0.05
    assert resource_state["deliberation_compress"] < 1.0
    assert resource_state["action_shrink_scale"] < 1.0
    assert "scarcity_pressure" in resource_state
    assert "baseline" in temperament_state
    assert "drift" in temperament_state
    assert "current" in temperament_state
    assert "drift_diagnostics" in temperament_state
    assert all(0.0 <= value <= 1.0 for value in temperament_state["current"].values())


def test_runtime_state_wraps_legacy_flat_temperament_state():
    state = RuntimeState(temperament_state={"warmth_bias": 0.5, "directness_bias": 0.55})

    assert state.temperament_state["baseline"] == {"warmth_bias": 0.5, "directness_bias": 0.55}
    assert state.temperament_state["drift"] == {"warmth_bias": 0.0, "directness_bias": 0.0}
    assert state.temperament_state["current"] == {"warmth_bias": 0.5, "directness_bias": 0.55}


def test_tick_supports_internal_short_reply_action_when_resources_are_tight(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    state = controller.load_runtime_state()
    state.budget_remaining = 0.05
    state.body_energy = 0.22
    controller._save_state(state)

    result = controller.tick(
        RoundEvent(
            source="user",
            content="Give me the quickest possible answer while you are low on budget.",
            target="user",
            cue="quick",
            valence=-0.05,
        ),
        scenario="task",
        mode="interactive",
    )

    assert "short_reply" in result.trace.probability_field["action"]["base_energy"]
    assert result.trace.probability_field["action"]["winner_posterior"]["short_reply"] >= 0.0
    assert result.sampled_action.name == "respond"
    assert result.trace.render_plan["action"] == "respond"
