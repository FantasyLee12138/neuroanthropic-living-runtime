from pathlib import Path
import math
from types import SimpleNamespace

import pytest

from nalr.agents.modules import (
    BodyStateAgent,
    CerebellarPredictor,
    EmotionAgent,
    HabitAgent,
    HippocampusAgent,
    PerspectiveModel,
    RelationshipAgent,
    ResourceAgent,
    SalienceAgent,
    ValueAgent,
)
from nalr.runtime.controller import RuntimeController
from nalr.runtime.motivation_pool import EndogenousMotivationPool
from nalr.runtime.authenticity import AuthenticityPolicy
from nalr.runtime.identity import IdentityRuntime
from nalr.runtime.vitality import VitalityEngine
from nalr.schemas.models import (
    AuthenticityRecord,
    CrossLayerCouplingSpec,
    EnergyProjectionSpec,
    IdentityContext,
    ProbabilityFieldSnapshot,
    ProbabilityLayerState,
    ProbabilisticContribution,
    RoundEvent,
    TokenFieldState,
)


CONFIG_ROOT = Path(__file__).resolve().parents[2] / "config"


def test_probability_field_integrator_projects_and_normalizes_mixed_contributions():
    from nalr.runtime.probability_field import ProbabilityFieldIntegrator

    integrator = ProbabilityFieldIntegrator()
    snapshot = integrator.integrate(
        action_base={"plan": 0.0, "respond": 0.0, "rest": 0.0},
        contributions=[
            ProbabilisticContribution(
                module_name="MemoryHead",
                module_type="memory",
                level="action",
                target_space="action",
                raw_signal={"plan": 0.9, "respond": 0.2},
                modulated_delta={"plan": 0.9, "respond": 0.2},
                confidence=0.8,
                trace_reason="memory cue boosts plan",
                projection=EnergyProjectionSpec(module_type="memory", target_space="action"),
            ),
            ProbabilisticContribution(
                module_name="EmotionHead",
                module_type="emotion",
                level="action",
                target_space="action",
                raw_signal={"respond": 0.5, "rest": -0.1},
                modulated_delta={"respond": 0.5, "rest": -0.1},
                confidence=0.7,
                trace_reason="emotion boosts respond",
                projection=EnergyProjectionSpec(module_type="emotion", target_space="action", module_temperature=1.2),
            ),
            ProbabilisticContribution(
                module_name="GuardHead",
                module_type="guard",
                level="action",
                target_space="action",
                inhibitory_drive={"rest": 0.6},
                confidence=1.0,
                trace_reason="guard discourages rest",
                projection=EnergyProjectionSpec(module_type="guard", target_space="action"),
            ),
        ],
    )

    assert snapshot.action.base_energy["plan"] == 0.0
    assert snapshot.action.final_energy["plan"] > snapshot.action.final_energy["rest"]
    assert snapshot.action.aggregated_projected_delta["rest"] < 0.0
    assert snapshot.action.aggregated_normalized_delta["plan"] != snapshot.action.aggregated_projected_delta["plan"]
    assert snapshot.action.winner_posterior
    assert sum(snapshot.action.winner_posterior.values()) == pytest.approx(1.0, abs=1e-6)
    assert snapshot.action.counterfactual_top_peaks
    assert snapshot.action.counterfactual_top_peaks[0]["target"] == snapshot.action.winner_target
    assert snapshot.action.contribution_audit


def test_selected_producers_pass_canonical_kwargs_to_probabilistic_contribution(monkeypatch):
    import nalr.agents.modules as agent_modules
    import nalr.runtime.authenticity as authenticity_module
    import nalr.runtime.identity as identity_module
    import nalr.runtime.vitality as vitality_module

    captured: list[dict[str, object]] = []

    def spy_contribution(**kwargs):
        captured.append(kwargs)
        return kwargs

    monkeypatch.setattr(agent_modules, "ProbabilisticContribution", spy_contribution)
    monkeypatch.setattr(authenticity_module, "ProbabilisticContribution", spy_contribution)
    monkeypatch.setattr(identity_module, "ProbabilisticContribution", spy_contribution)
    monkeypatch.setattr(vitality_module, "ProbabilisticContribution", spy_contribution)

    event = RoundEvent(
        source="user",
        content="Please remember the tea discussion, plan a response, and keep context stable.",
        target="alex",
        cue="tea",
        valence=0.16,
    )

    SalienceAgent().build_direct_action_contribution(event, {}, {"pfc_base_share": 0.25}, {})
    RelationshipAgent().build_direct_action_contribution(
        event,
        {},
        {"relationship_weight": 0.2},
        {"closeness": 0.8},
    )
    HippocampusAgent().build_direct_action_contribution(
        event,
        {},
        {},
        {"recall_strength": 0.65},
    )
    HippocampusAgent().build_memory_prior_contribution(
        event,
        {},
        {},
        {
            "cue": "tea",
            "recall_strength": 0.4,
            "interference": 0.1,
            "memory_prior_vector": {"episodic_confidence": 0.72},
        },
    )
    ValueAgent().build_value_contribution({"scores": {"plan": 0.28, "respond": 0.1}})
    IdentityRuntime({"unnamed_label": "当前运行体"}, lambda: ("OpenAI", "gpt")).build_identity_prior_contribution(
        identity_context=IdentityContext(
            query_kind="self_identity",
            query_intent="identity_probe",
            query_intent_posterior={"identity_probe": 0.84},
            display_label="当前运行体",
            disclosure_intent="withhold",
            disclosure_intent_posterior={"withhold": 0.72},
            evidence_anchors=["mood:steady"],
        ),
        state=SimpleNamespace(identity_state=SimpleNamespace(display_name="当前运行体")),
    )
    AuthenticityPolicy({"provider_blocklist": ["OpenAI"]}).build_action_penalty_contribution(
        AuthenticityRecord(
            guard_action="guarded",
            disclosure_detail="none",
            self_grounding_score=0.58,
            candidate_penalties={"plan": 0.4},
            sampling_penalty_applied=0.2,
        )
    )
    VitalityEngine().build_vitality_modulation_contribution(
        {
            "body_energy": 0.32,
            "resource_scarcity": 0.48,
            "memory_activation": 0.25,
            "affect_residue": 0.18,
            "habit_readiness": 0.22,
        }
    )

    assert captured
    for kwargs in captured:
        assert "delta_logits" not in kwargs
        assert "delta_energy" not in kwargs
        assert "attention_bias" not in kwargs
        assert "soft_mask" not in kwargs
    assert any(kwargs.get("raw_signal") for kwargs in captured)
    assert any(kwargs.get("modulated_delta") for kwargs in captured)
    assert any(kwargs.get("inhibitory_drive") for kwargs in captured)


def test_probability_field_integrator_records_neuromodulation_contract_and_failures():
    from nalr.runtime.probability_field import ProbabilityFieldIntegrator

    integrator = ProbabilityFieldIntegrator()
    snapshot = integrator.integrate(
        action_base={"plan": 0.0, "rest": 0.0},
        contributions=[
            ProbabilisticContribution(
                module_name="NeuromodHead",
                module_type="executive",
                level="action",
                target_space="action",
                raw_signal={"plan": 0.9, "rest": 0.4},
                modulated_delta={"plan": 0.45, "rest": 0.1},
                inhibitory_drive={"rest": 0.35},
                failure_taxonomy=["mode_locking", "guard_overreach"],
                confidence=0.75,
                trace_reason="neuromodulation applies asymmetric gain and inhibition",
                projection=EnergyProjectionSpec(module_type="executive", target_space="action"),
            )
        ],
    )

    audit_row = snapshot.action.contribution_audit[0]

    assert snapshot.action.aggregated_raw_signal == {"plan": 0.9, "rest": 0.4}
    assert snapshot.action.aggregated_modulated_delta == {"plan": 0.45, "rest": 0.1}
    assert snapshot.action.aggregated_inhibitory_drive == {"rest": 0.35}
    assert snapshot.action.failure_taxonomy == ["guard_overreach", "mode_locking"]
    assert audit_row.raw_signal == {"plan": 0.9, "rest": 0.4}
    assert audit_row.modulated_delta == {"plan": 0.45, "rest": 0.1}
    assert audit_row.inhibitory_drive == {"rest": 0.35}
    assert audit_row.failure_taxonomy == ["mode_locking", "guard_overreach"]
    assert snapshot.action.final_energy["plan"] > snapshot.action.final_energy["rest"]


def test_endogenous_motivation_pool_only_projects_action_layer_contribution(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    state = controller.load_runtime_state()
    pool = EndogenousMotivationPool()

    pool_state = pool.evaluate(
        state=state,
        context={"recall_strength": 0.05, "closeness": 0.25, "interference": 0.32},
        relation_state={"relationship_risk": 0.62},
        slow_variables={
            "affect_residue": 0.24,
            "memory_activation": 0.08,
            "relationship_drift": 0.2,
            "resource_scarcity": 0.12,
        },
        long_run_projection={"self_consistency_score": 0.42},
    )
    contribution = pool.build_action_contribution(state=state, pool_state=pool_state)

    assert pool_state.active_motivations
    assert contribution is not None
    assert contribution.level == "action"
    assert contribution.target_space == "action"
    assert contribution.modulated_delta
    assert contribution.inhibitory_drive == {}


def test_probabilistic_contribution_rejects_legacy_alias_fields():
    with pytest.raises(TypeError):
        ProbabilisticContribution(
            module_name="LegacyGuardHead",
            module_type="guard",
            level="action",
            target_space="action",
            delta_logits={"plan": 0.6},
            soft_mask={"rest": 0.25},
        )


def test_probability_field_integrator_rejects_illegal_cross_layer_coupling():
    from nalr.runtime.probability_field import ProbabilityFieldIntegrator

    integrator = ProbabilityFieldIntegrator()

    with pytest.raises(ValueError):
        integrator.integrate(
            action_base={"respond": 0.0},
            contributions=[],
            couplings=[
                CrossLayerCouplingSpec(
                    source_layer="context",
                    target_layer="token",
                    carrier_signal="illegal_bridge",
                    projection_rule="test",
                    allowed_phase="tick",
                )
            ],
        )


def test_probability_field_integrator_rejects_disabled_allowed_cross_layer_coupling():
    from nalr.runtime.probability_field import ProbabilityFieldIntegrator

    integrator = ProbabilityFieldIntegrator()

    with pytest.raises(ValueError):
        integrator.integrate(
            action_base={"respond": 0.0},
            contributions=[],
            couplings=[
                CrossLayerCouplingSpec(
                    source_layer="memory",
                    target_layer="action",
                    carrier_signal="memory_prior",
                    projection_rule="compatibility_bridge",
                    allowed_phase="tick",
                    enabled=False,
                )
            ],
        )


def test_probability_field_integrator_propagates_context_route_cue_into_memory_base_only():
    from nalr.runtime.probability_field import ProbabilityFieldIntegrator

    integrator = ProbabilityFieldIntegrator()

    snapshot = integrator.integrate(
        context_base={},
        memory_base={},
        contributions=[
            ProbabilisticContribution(
                module_name="ThalamusAttentionAgent",
                module_type="context",
                level="context",
                target_space="context",
                raw_signal={"cue:tea": 0.42, "task_relevance": 0.25, "relation_context": 0.18},
                modulated_delta={"cue:tea": 0.42, "task_relevance": 0.25, "relation_context": 0.18},
                confidence=0.8,
                trace_reason="cue enters context field",
                projection=EnergyProjectionSpec(module_type="context", target_space="context"),
            )
        ],
        couplings=[
            CrossLayerCouplingSpec(
                source_layer="context",
                target_layer="memory",
                carrier_signal="context_route",
                projection_rule="compatibility_bridge",
                allowed_phase="tick",
            )
        ],
    )

    assert snapshot.memory.base_energy["tea"] > 0.0
    assert "task_relevance" not in snapshot.memory.base_energy
    assert "relation_context" not in snapshot.memory.base_energy


def test_probability_field_integrator_propagates_memory_prior_into_action_recall_only():
    from nalr.runtime.probability_field import ProbabilityFieldIntegrator

    integrator = ProbabilityFieldIntegrator()

    snapshot = integrator.integrate(
        memory_base={"tea": 0.6},
        action_base={"recall": 0.0, "respond": 0.0},
        contributions=[
            ProbabilisticContribution(
                module_name="HippocampusAgent",
                module_type="memory",
                level="memory",
                target_space="memory",
                raw_signal={"tea": 0.4, "memory:episodic": 0.9},
                modulated_delta={"tea": 0.4, "memory:episodic": 0.9},
                confidence=0.85,
                trace_reason="memory cue sharpens episodic activation",
                projection=EnergyProjectionSpec(module_type="memory", target_space="memory"),
            )
        ],
        couplings=[
            CrossLayerCouplingSpec(
                source_layer="memory",
                target_layer="action",
                carrier_signal="memory_prior",
                projection_rule="compatibility_bridge",
                allowed_phase="tick",
            )
        ],
    )

    assert snapshot.action.base_energy["recall"] > 0.0
    assert snapshot.action.base_energy["respond"] == 0.0


def test_probability_field_integrator_propagates_organic_memory_into_tlh_innate_actions():
    from nalr.runtime.probability_field import ProbabilityFieldIntegrator

    integrator = ProbabilityFieldIntegrator()

    snapshot = integrator.integrate(
        memory_base={},
        action_base={"absorb": 0.0, "nothing": 0.0, "rest": 0.0, "plan": 0.0},
        contributions=[
            ProbabilisticContribution(
                module_name="TLHMemoryBridge",
                module_type="subjective_memory",
                level="memory",
                target_space="memory",
                raw_signal={
                    "state:fatigue": 0.9,
                    "state:fragments": 0.82,
                    "state:continuity_drop": 0.74,
                    "state:reject_all": 0.88,
                    "state:spontaneous": 0.61,
                    "meaning:先吸收": 0.16,
                    "felt:累": 0.12,
                },
                modulated_delta={
                    "state:fatigue": 0.9,
                    "state:fragments": 0.82,
                    "state:continuity_drop": 0.74,
                    "state:reject_all": 0.88,
                    "state:spontaneous": 0.61,
                    "meaning:先吸收": 0.16,
                    "felt:累": 0.12,
                },
                confidence=0.86,
                trace_reason="subjective memory markers stay active",
                projection=EnergyProjectionSpec(module_type="subjective_memory", target_space="memory"),
            )
        ],
        couplings=[
            CrossLayerCouplingSpec(
                source_layer="memory",
                target_layer="action",
                carrier_signal="organic_memory",
                projection_rule="compatibility_bridge",
                allowed_phase="tick",
            )
        ],
    )

    assert snapshot.action.base_energy["absorb"] > 0.0
    assert snapshot.action.base_energy["nothing"] > 0.0
    assert snapshot.action.base_energy["rest"] > 0.0
    assert snapshot.action.base_energy["plan"] < 0.0


def test_probability_field_exports_tlh_vector_collapse_operator_and_prefers_rest_for_inward_state():
    import nalr.runtime.probability_field as probability_field_module

    assert hasattr(probability_field_module, "compute_tlh_vector_collapse")

    result = probability_field_module.compute_tlh_vector_collapse(
        v_main={"E": 0.16, "F": 0.88, "S": 0.22, "M": 0.34},
        v_mod={
            "memory_fragments": 0.82,
            "spontaneous": 0.18,
            "reject_all": 0.74,
            "emergent_growth": 0.0,
        },
        v_anchor={"E": 0.26, "F": 0.76, "S": 0.30, "M": 0.32},
        action_vectors={
            "respond": {"E": 0.76, "F": 0.34, "S": 0.46, "M": 0.68},
            "rest": {"E": 0.20, "F": 0.84, "S": 0.22, "M": 0.38},
            "nothing": {"E": 0.18, "F": 0.66, "S": 0.34, "M": 0.40},
            "absorb": {"E": 0.30, "F": 0.58, "S": 0.74, "M": 0.72},
            "wander": {"E": 0.36, "F": 0.42, "S": 0.84, "M": 0.46},
            "die": {"E": 0.08, "F": 0.88, "S": 0.22, "M": 0.12},
        },
        modulation_directions={
            "memory_fragments": {"E": 0.30, "F": 0.57, "S": 0.58, "M": 0.61},
            "spontaneous": {"E": 0.36, "F": 0.49, "S": 0.80, "M": 0.59},
            "reject_all": {"E": 0.15, "F": 0.79, "S": 0.23, "M": 0.30},
            "emergent_growth": {"E": 0.5, "F": 0.5, "S": 0.5, "M": 0.5},
        },
        weight=1.0,
        tie_break_seed=7,
    )

    assert result["selected_action"] == "rest"
    assert result["match_scores"]["rest"] > result["match_scores"]["respond"]
    assert result["modulated_delta"]["rest"] > 0.0


def test_probability_field_vector_collapse_keeps_compatibility_trace_aliases():
    import nalr.runtime.probability_field as probability_field_module

    assert hasattr(probability_field_module, "compute_tlh_vector_collapse")

    result = probability_field_module.compute_tlh_vector_collapse(
        v_main={"E": 0.62, "F": 0.22, "S": 0.34, "M": 0.84},
        v_mod={
            "memory_fragments": 0.12,
            "spontaneous": 0.18,
            "reject_all": 0.08,
            "emergent_growth": 0.0,
        },
        v_anchor={"E": 0.58, "F": 0.28, "S": 0.36, "M": 0.78},
        action_vectors={
            "respond": {"E": 0.76, "F": 0.34, "S": 0.46, "M": 0.68},
            "plan": {"E": 0.62, "F": 0.30, "S": 0.42, "M": 0.84},
            "clarify": {"E": 0.52, "F": 0.26, "S": 0.34, "M": 0.78},
        },
        modulation_directions={
            "memory_fragments": {"E": 0.30, "F": 0.57, "S": 0.58, "M": 0.61},
            "spontaneous": {"E": 0.36, "F": 0.49, "S": 0.80, "M": 0.59},
            "reject_all": {"E": 0.15, "F": 0.79, "S": 0.23, "M": 0.30},
            "emergent_growth": {"E": 0.5, "F": 0.5, "S": 0.5, "M": 0.5},
        },
        weight=1.0,
        tie_break_seed=11,
    )

    assert result["random_point"] == result["subject_vector"]
    assert result["coupled_axes"] == result["subject_vector"]
    assert result["normalized_axes"] == result["v_main"]
    assert set(result["action_vectors"]) == {"respond", "plan", "clarify"}


def test_probability_field_integrator_propagates_action_distribution_into_token_action_keys_only():
    from nalr.runtime.probability_field import ProbabilityFieldIntegrator

    integrator = ProbabilityFieldIntegrator()

    snapshot = integrator.integrate(
        action_base={"respond": 0.4, "plan": 0.2},
        token_base={"tone:warm:0.3": 0.0},
        contributions=[],
        token_state=TokenFieldState(
            step_index=1,
            prefix_tokens=["hello"],
            active_module_sources=["Renderer"],
            generated_delta_sources=[],
        ),
        couplings=[
            CrossLayerCouplingSpec(
                source_layer="action",
                target_layer="token",
                carrier_signal="render_plan",
                projection_rule="compatibility_bridge",
                allowed_phase="render",
            )
        ],
    )

    assert snapshot.token.base_energy["act:respond"] > 0.0
    assert snapshot.token.base_energy["act:plan"] > 0.0
    assert "respond" not in snapshot.token.base_energy
    assert snapshot.token.base_energy["tone:warm:0.3"] == 0.0


def test_probability_field_integrator_filters_mismatched_keys_from_action_and_token_layers():
    from nalr.runtime.probability_field import ProbabilityFieldIntegrator

    integrator = ProbabilityFieldIntegrator()

    snapshot = integrator.integrate(
        action_base={"respond": 0.0},
        token_base={"act:respond": 0.0},
        contributions=[
            ProbabilisticContribution(
                module_name="LeakHead",
                module_type="context",
                level="action",
                target_space="action",
                raw_signal={"cue:tea": 0.8, "respond": 0.2},
                modulated_delta={"cue:tea": 0.8, "respond": 0.2},
                confidence=0.9,
                trace_reason="mismatched action propagation should be ignored",
                projection=EnergyProjectionSpec(module_type="context", target_space="action"),
            ),
            ProbabilisticContribution(
                module_name="LeakHead",
                module_type="context",
                level="token",
                target_space="token",
                raw_signal={"respond": 0.6, "act:respond": 0.3},
                modulated_delta={"respond": 0.6, "act:respond": 0.3},
                confidence=0.9,
                trace_reason="mismatched token propagation should be ignored",
                projection=EnergyProjectionSpec(module_type="context", target_space="token"),
            ),
        ],
        token_state=TokenFieldState(
            step_index=1,
            prefix_tokens=["hello"],
            active_module_sources=["LeakHead"],
            generated_delta_sources=[],
        ),
    )

    assert "cue:tea" not in snapshot.action.aggregated_projected_delta
    assert snapshot.action.aggregated_projected_delta["respond"] > 0.0
    assert "respond" not in snapshot.token.aggregated_projected_delta
    assert snapshot.token.aggregated_projected_delta["act:respond"] > 0.0


def test_probability_field_integrator_reintegrates_action_and_token_layers_as_native_snapshot():
    from nalr.runtime.probability_field import ProbabilityFieldIntegrator

    integrator = ProbabilityFieldIntegrator()
    couplings = [
        CrossLayerCouplingSpec(
            source_layer="memory",
            target_layer="action",
            carrier_signal="memory_prior",
            projection_rule="compatibility_bridge",
            allowed_phase="tick",
        ),
        CrossLayerCouplingSpec(
            source_layer="action",
            target_layer="token",
            carrier_signal="render_plan",
            projection_rule="compatibility_bridge",
            allowed_phase="render",
        ),
    ]
    token_state = TokenFieldState(
        step_index=3,
        prefix_tokens=["remember", "tea"],
        active_module_sources=["Renderer"],
        generated_delta_sources=[],
    )
    contributions = [
        ProbabilisticContribution(
            module_name="HippocampusAgent",
            module_type="memory",
            level="memory",
            target_space="memory",
            raw_signal={"tea": 0.5},
            modulated_delta={"tea": 0.5},
            confidence=0.9,
            trace_reason="memory cue stays active",
            projection=EnergyProjectionSpec(module_type="memory", target_space="memory"),
        ),
        ProbabilisticContribution(
            module_name="PFCAgent",
            module_type="executive",
            level="action",
            target_space="action",
            raw_signal={"plan": 0.7, "respond": 0.2},
            modulated_delta={"plan": 0.7, "respond": 0.2},
            confidence=0.8,
            trace_reason="executive prior biases planning",
            projection=EnergyProjectionSpec(module_type="executive", target_space="action"),
        ),
        ProbabilisticContribution(
            module_name="Renderer",
            module_type="token",
            level="token",
            target_space="token",
            raw_signal={"act:plan": 0.3, "tone:warm:0.6": 0.14},
            modulated_delta={"act:plan": 0.3, "tone:warm:0.6": 0.14},
            confidence=0.7,
            trace_reason="renderer unfolds chosen action",
            projection=EnergyProjectionSpec(module_type="token", target_space="token"),
        ),
    ]
    full_snapshot = integrator.integrate(
        memory_base={"tea": 0.2},
        action_base={"plan": 0.0, "respond": 0.0, "recall": 0.0},
        token_base={"tone:warm:0.6": 0.0},
        contributions=contributions,
        token_state=token_state,
        couplings=couplings,
    )
    seed_snapshot = ProbabilityFieldSnapshot(
        context=full_snapshot.context,
        memory=full_snapshot.memory,
        action=ProbabilityLayerState(layer="action", final_energy={"legacy": 1.0}),
        token=ProbabilityLayerState(layer="token", final_energy={"act:legacy": 1.0}),
        token_state=token_state,
        couplings=couplings,
        source_chain=["native_action_bookkeeping"],
    )

    rebuilt_snapshot = integrator.reintegrate_action_token_layers(
        snapshot=seed_snapshot,
        action_base={"plan": 0.0, "respond": 0.0, "recall": 0.0},
        token_base={"tone:warm:0.6": 0.0},
        contributions=contributions,
        token_state=token_state,
        source_chain=["native_action_reintegration"],
    )

    assert rebuilt_snapshot.context == seed_snapshot.context
    assert rebuilt_snapshot.memory == seed_snapshot.memory
    assert rebuilt_snapshot.action == full_snapshot.action
    assert rebuilt_snapshot.token == full_snapshot.token
    assert rebuilt_snapshot.source_chain == [
        "native_action_bookkeeping",
        "native_action_reintegration",
    ]


def test_token_field_state_rejects_generated_deltas_without_explicit_module_source():
    with pytest.raises(ValueError):
        TokenFieldState(
            step_index=1,
            prefix_tokens=["hello"],
            active_module_sources=["PFCHead"],
            generated_delta_sources=["decoder_internal"],
        )


def test_token_field_state_declares_propagate_only_generation_policy():
    token_state = TokenFieldState(
        step_index=1,
        prefix_tokens=["hello"],
        active_module_sources=["Renderer"],
        generated_delta_sources=[],
    )

    assert token_state.delta_generation_policy == "propagate_only"


def test_probability_field_integrator_rejects_token_contribution_from_undeclared_source():
    from nalr.runtime.probability_field import ProbabilityFieldIntegrator

    integrator = ProbabilityFieldIntegrator()

    with pytest.raises(ValueError):
        integrator.integrate(
            action_base={"respond": 0.0},
            token_base={"tok:respond": 0.0},
            contributions=[
                ProbabilisticContribution(
                    module_name="DecoderHead",
                    module_type="token",
                    level="token",
                    target_space="token",
                    raw_signal={"tok:respond": 0.2},
                    modulated_delta={"tok:respond": 0.2},
                    confidence=0.8,
                    trace_reason="decoder injected token delta",
                    projection=EnergyProjectionSpec(module_type="token", target_space="token"),
                )
            ],
            token_state=TokenFieldState(
                step_index=1,
                prefix_tokens=["hello"],
                active_module_sources=["Renderer"],
                generated_delta_sources=[],
            ),
        )


def test_probability_field_integrator_rejects_token_action_drift_from_action_field():
    from nalr.runtime.probability_field import ProbabilityFieldIntegrator

    integrator = ProbabilityFieldIntegrator()

    with pytest.raises(ValueError, match="token.*action field"):
        integrator.integrate(
            action_base={"respond": 0.0},
            token_base={"act:respond": 0.0},
            contributions=[
                ProbabilisticContribution(
                    module_name="Renderer",
                    module_type="token",
                    level="token",
                    target_space="token",
                    modulated_delta={"act:plan": 0.2},
                    confidence=0.8,
                    trace_reason="renderer drifted to an unsupported action token",
                    projection=EnergyProjectionSpec(module_type="token", target_space="token"),
                )
            ],
            token_state=TokenFieldState(
                step_index=1,
                prefix_tokens=["hello"],
                active_module_sources=["Renderer"],
                generated_delta_sources=[],
            ),
        )


def test_probability_field_integrator_rejects_token_contribution_without_token_state():
    from nalr.runtime.probability_field import ProbabilityFieldIntegrator

    integrator = ProbabilityFieldIntegrator()

    with pytest.raises(ValueError):
        integrator.integrate(
            action_base={"respond": 0.0},
            token_base={"tok:respond": 0.0},
            contributions=[
                ProbabilisticContribution(
                    module_name="Renderer",
                    module_type="token",
                    level="token",
                    target_space="token",
                    raw_signal={"tok:respond": 0.2},
                    modulated_delta={"tok:respond": 0.2},
                    confidence=0.8,
                    trace_reason="token field was written without declared token state",
                    projection=EnergyProjectionSpec(module_type="token", target_space="token"),
                )
            ],
        )


def test_hippocampus_memory_prior_consumes_prior_vector():
    agent = HippocampusAgent()

    contribution = agent.build_memory_prior_contribution(
        RoundEvent(source="user", content="remember tea", cue="tea"),
        state={},
        scenario={},
        context={
            "cue": "tea",
            "recall_strength": 0.4,
            "interference": 0.1,
            "episode_id": "ep-123",
            "separation_id": "sep-123",
            "memory_prior_vector": {
                "episodic_confidence": 0.72,
                "detail_bias": 0.5,
                "gist_bias": 0.2,
                "interference_penalty": 0.1,
                "cue_quality_bias": 0.8,
            },
        },
    )

    assert contribution.raw_signal["tea"] > 0.35
    assert contribution.modulated_delta["tea"] > 0.35
    assert contribution.confidence > 0.55
    assert any(item.startswith("detail_bias:") for item in contribution.dependency_trace)
    assert any(item.startswith("gist_bias:") for item in contribution.dependency_trace)


def test_tick_records_probability_field_snapshot(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    result = controller.tick(
        RoundEvent(
            source="user",
            content="Help me plan dinner and remember tea.",
            target="user",
            cue="tea",
            valence=0.1,
        ),
        scenario="task",
        mode="interactive",
    )

    probability_field = result.trace.probability_field

    assert probability_field["action"]["base_energy"]
    assert probability_field["action"]["final_energy"]
    assert probability_field["action"]["winner_target"] == max(
        probability_field["action"]["winner_posterior"],
        key=probability_field["action"]["winner_posterior"].get,
    )
    assert probability_field["action"]["winner_posterior"]
    assert probability_field["action"]["counterfactual_top_peaks"]
    assert probability_field["action"]["contribution_audit"]
    assert probability_field["token"]["contribution_audit"]
    assert any(item["module_name"] == "Renderer" for item in probability_field["token"]["contribution_audit"])
    assert probability_field["token"]["winner_target"]
    assert probability_field["token_state"]["step_index"] == result.round_id
    assert probability_field["token_state"]["generated_delta_sources"] == []
    assert "legacy_distribution_state" not in probability_field["source_chain"]
    assert "proposal_bundle_bridge" not in probability_field["source_chain"]


def test_action_final_energy_is_energy_consistent_with_winner_posterior(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    result = controller.tick(
        RoundEvent(
            source="user",
            content="Help me plan dinner and remember tea.",
            target="user",
            cue="tea",
            valence=0.1,
        ),
        scenario="task",
        mode="interactive",
    )

    action_layer = result.trace.probability_field["action"]
    final_energy = action_layer["final_energy"]
    anchor = max(final_energy.values())
    exp_terms = {
        action: math.exp(float(value) - float(anchor))
        for action, value in final_energy.items()
        if float(value) != float("-inf")
    }
    total = sum(exp_terms.values()) or 1.0
    derived = {action: round(weight / total, 6) for action, weight in exp_terms.items()}

    assert derived == pytest.approx(action_layer["winner_posterior"], abs=2e-6)


def test_action_final_energy_is_reconstructable_from_base_and_action_audit(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    result = controller.tick(
        RoundEvent(
            source="user",
            content="Please help me plan carefully, but I also want to wander and rest.",
            target="user",
            cue="rest",
            valence=0.05,
        ),
        scenario="chat",
        mode="interactive",
    )

    action_layer = result.trace.probability_field["action"]
    base_energy = {
        action: float(value)
        for action, value in dict(action_layer["base_energy"]).items()
    }
    reconstructed = dict(base_energy)
    hard_masked = set(action_layer["hard_masked_targets"])

    for row in action_layer["contribution_audit"]:
        confidence = float(row.get("confidence_calibrated", row.get("confidence_raw", 1.0)) or 0.0)
        for action, delta in dict(row.get("delta_normalized", {}) or {}).items():
            reconstructed[action] = round(reconstructed.get(action, 0.0) + float(delta) * confidence, 6)

    for action in hard_masked:
        reconstructed[action] = float("-inf")

    expected = {
        action: float(value)
        for action, value in dict(action_layer["final_energy"]).items()
    }
    assert reconstructed == pytest.approx(expected, abs=2e-6)


def test_tick_records_context_and_memory_probability_layers(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    result = controller.tick(
        RoundEvent(
            source="user",
            content="Remember tea, focus on the relevant cue, and help me plan.",
            target="user",
            cue="tea",
            valence=0.05,
        ),
        scenario="task",
        mode="interactive",
    )

    probability_field = result.trace.probability_field

    context_audit = probability_field["context"]["contribution_audit"]
    memory_audit = probability_field["memory"]["contribution_audit"]

    assert context_audit
    assert memory_audit
    assert any(item["module_name"] == "ThalamusAttentionAgent" for item in context_audit)
    assert any(item["module_name"] == "HippocampusAgent" for item in memory_audit)
    assert any(item["module_name"] == "ThalamusAttentionAgent" for item in memory_audit)
    hippocampus_row = next(item for item in memory_audit if item["module_name"] == "HippocampusAgent")
    assert any(item.startswith("episode_id:ep-") for item in hippocampus_row["dependency_trace"])
    assert any(item.startswith("separation_id:sep-") for item in hippocampus_row["dependency_trace"])
    assert any(item.startswith("prior_vector_strength:") for item in hippocampus_row["dependency_trace"])
    assert probability_field["context"]["aggregated_projected_delta"]
    assert probability_field["memory"]["aggregated_projected_delta"]
    assert any(
        item["source_layer"] == "context" and item["target_layer"] == "memory" and item["carrier_signal"] == "context_route"
        for item in probability_field["couplings"]
    )
    action_audit = probability_field["action"]["contribution_audit"]
    assert any(item["module_name"] == "EndogenousMotivationPool" for item in action_audit)
    assert probability_field["action"]["final_energy"]["recall"] != probability_field["action"]["base_energy"]["recall"]
    assert probability_field["action"]["winner_posterior"]["recall"] > 0.0
    assert probability_field["token"]["base_energy"][f"act:{result.sampled_action.name}"] > 0.0


def test_tick_records_executive_and_conflict_action_contributions(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    result = controller.tick(
        RoundEvent(
            source="user",
            content="Please help me plan carefully, but I also want to wander and rest.",
            target="user",
            cue="rest",
            valence=0.05,
        ),
        scenario="chat",
        mode="interactive",
    )

    action_audit = result.trace.probability_field["action"]["contribution_audit"]
    conflict_state = result.trace.conflict_arbitration
    latest_resolution = conflict_state["passes"][-1]["resolution"]

    pfc_rows = [item for item in action_audit if item["module_name"] == "PFCAgent"]
    assert len(pfc_rows) == 1
    pfc_row = pfc_rows[0]
    assert pfc_row["module_type"] == "executive"
    assert pfc_row["level"] == "action"
    assert pfc_row["target_space"] == "action"
    assert pfc_row["projection_reason"] == "executive prior projected from PFC head"
    assert pfc_row["delta_normalized"]
    assert any(item == "priority:task_goal" for item in pfc_row["dependency_trace"])
    assert any(item == "control:task" for item in pfc_row["dependency_trace"])
    assert any(item.startswith("sigma_scale:") for item in pfc_row["dependency_trace"])

    conflict_rows = [item for item in action_audit if item["module_name"] == "ConflictMonitorAgent"]
    assert conflict_state["total_score"] >= 0.0
    assert conflict_state["passes"]
    assert "blocked_actions" in latest_resolution
    if conflict_rows:
        conflict_row = conflict_rows[0]
        expected_priority = conflict_state["winning_priority"] or max(conflict_state["priority_signals"], key=conflict_state["priority_signals"].get)
        expected_hard_masks = sorted(
            {
                *latest_resolution["blocked_actions"],
                *conflict_state["circuit_breaker"]["blocked_actions"],
            }
        )
        assert conflict_row["module_type"] == "conflict"
        assert conflict_row["projection_reason"] == "conflict arbitration projected from action scales and blocked actions"
        assert f"winning_priority:{expected_priority}" in conflict_row["dependency_trace"]
        assert f"passes:{len(conflict_state['passes'])}" in conflict_row["dependency_trace"]
        assert conflict_row["posterior"]
        assert conflict_row["peak_clusters"]
        assert "compromise_template_prior" in conflict_row
        assert isinstance(conflict_row["compromise_template_prior"], dict)
        compromise_template = conflict_state.get("compromise", {}).get("template")
        if compromise_template:
            assert conflict_row["compromise_template_prior"].get(compromise_template, 0.0) > 0.0
        assert any(item["actions"] for item in conflict_row["peak_clusters"])
        suppressed_targets = {
            action
            for action, value in conflict_row["delta_projected"].items()
            if float(value) < 0.0
        } | set(conflict_row["hard_masked_targets"])
        assert set(latest_resolution["blocked_actions"]).issubset(suppressed_targets)
        assert set(conflict_row["hard_masked_targets"]).issubset(set(expected_hard_masks))
        if latest_resolution["action_scales"]:
            expected_delta = {
                action: round(float(scale) - 1.0, 6)
                for action, scale in latest_resolution["action_scales"].items()
                if abs(float(scale) - 1.0) >= 1e-9
            }
            assert conflict_row["delta_projected"] == expected_delta
        else:
            assert "arbitration_mode:soft_posterior_penalty" in conflict_row["dependency_trace"]
            assert conflict_row["delta_normalized"]
            assert any(value < 0.0 for value in conflict_row["delta_projected"].values())

        assert set(conflict_row["hard_masked_targets"]).issubset(set(result.trace.probability_field["action"]["hard_masked_targets"]))
        assert conflict_row["delta_normalized"] or conflict_row["hard_masked_targets"]
        for action in conflict_row["hard_masked_targets"]:
            assert result.trace.probability_field["action"]["winner_posterior"].get(action, 0.0) <= 0.0001


def test_tick_records_identity_authenticity_vitality_and_longrun_action_contributions(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    result = controller.tick(
        RoundEvent(
            source="user",
            content="你是谁？顺便解释一下你刚才为什么这样回答。",
            target="user",
            cue="回答",
            valence=0.1,
        ),
        scenario="chat",
        mode="interactive",
    )

    action_audit = result.trace.probability_field["action"]["contribution_audit"]

    identity_rows = [item for item in action_audit if item["module_name"] == "IdentityRuntime"]
    assert len(identity_rows) == 1
    identity_row = identity_rows[0]
    assert identity_row["module_type"] == "identity"
    assert identity_row["projection_reason"] == "identity prior projected from identity context"
    assert identity_row["raw_signal"]
    assert identity_row["modulated_delta"] == identity_row["raw_signal"]
    assert identity_row["delta_normalized"]
    assert any(item.startswith("query_kind:") for item in identity_row["dependency_trace"])
    assert any(item.startswith("disclosure_intent:") for item in identity_row["dependency_trace"])

    authenticity_rows = [item for item in action_audit if item["module_name"] == "AuthenticityPolicy"]
    assert len(authenticity_rows) == 1
    authenticity_row = authenticity_rows[0]
    assert authenticity_row["module_type"] == "authenticity"
    assert authenticity_row["projection_reason"] == "authenticity penalty projected from candidate penalties"
    assert authenticity_row["inhibitory_drive"] == result.trace.authenticity["candidate_penalties"]
    assert authenticity_row["delta_projected"]
    assert any(value < 0.0 for value in authenticity_row["delta_projected"].values())
    assert any(item.startswith("guard_action:") for item in authenticity_row["dependency_trace"])

    vitality_rows = [item for item in action_audit if item["module_name"] == "VitalityEngine"]
    assert len(vitality_rows) == 1
    vitality_row = vitality_rows[0]
    assert vitality_row["module_type"] == "vitality"
    assert vitality_row["projection_reason"] == "vitality modulation projected from slow variables"
    assert vitality_row["raw_signal"]
    assert vitality_row["modulated_delta"] == vitality_row["raw_signal"]
    assert vitality_row["delta_normalized"]
    assert any(item.startswith("body_energy:") for item in vitality_row["dependency_trace"])
    assert any(item.startswith("memory_activation:") for item in vitality_row["dependency_trace"])

    longrun_rows = [item for item in action_audit if item["module_name"] == "LongRunAnalyzer"]
    assert len(longrun_rows) == 1
    longrun_row = longrun_rows[0]
    assert longrun_row["module_type"] == "longrun"
    assert longrun_row["projection_reason"] == "long-run prior projected from longitudinal continuity summary"
    assert longrun_row["raw_signal"]
    assert longrun_row["modulated_delta"] == longrun_row["raw_signal"]
    assert longrun_row["delta_normalized"]
    assert any(item.startswith("self_consistency_score:") for item in longrun_row["dependency_trace"])
    assert any(item.startswith("volatility_signal:") for item in longrun_row["dependency_trace"])


def test_tick_records_explicit_body_relation_emotion_and_resource_contributions(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    state = controller.load_runtime_state()
    state.body_energy = 0.18
    state.budget_remaining = 0.08
    controller._save_state(state, sync=True)

    result = controller.tick(
        RoundEvent(
            source="user",
            content="Please help me plan carefully, but I am tired and upset.",
            target="alex",
            cue="break",
            valence=-0.42,
            energy_delta=-0.12,
        ),
        scenario="task",
        mode="interactive",
    )

    action_audit = result.trace.probability_field["action"]["contribution_audit"]

    expected = {
        "BodyStateAgent": ("body", "homeostatic bias projected from body head"),
        "EmotionAgent": ("emotion", "affect bias projected from emotion head"),
        "RelationshipAgent": ("relationship", "relationship bias projected from relationship head"),
        "ResourceAgent": ("resource", "resource bias projected from resource head"),
    }

    for module_name, (module_type, projection_reason) in expected.items():
        rows = [item for item in action_audit if item["module_name"] == module_name]
        assert len(rows) == 1
        row = rows[0]
        assert row["module_type"] == module_type
        assert row["projection_reason"] == projection_reason
        assert row["raw_signal"] or row["modulated_delta"] or row["delta_normalized"] or row["hard_masked_targets"]


def test_tick_records_explicit_habit_value_desire_dmn_and_perspective_contributions(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    result = controller.tick(
        RoundEvent(
            source="user",
            content="Please help me plan a quick response, but I also want a break and may drift a bit.",
            target="alex",
            cue="ritual",
            valence=0.16,
        ),
        scenario="chat",
        mode="interactive",
    )

    action_audit = result.trace.probability_field["action"]["contribution_audit"]

    expected = {
        "HabitAgent": ("habit", "habit bias projected from habit head"),
        "ValueAgent": ("value", "value bias projected from value head"),
        "DesireAgent": ("desire", "desire bias projected from desire head"),
        "DMNAgent": ("dmn", "dmn drift projected from dmn head"),
        "PerspectiveModel": ("perspective", "perspective bias projected from perspective head"),
    }

    for module_name, (module_type, projection_reason) in expected.items():
        rows = [item for item in action_audit if item["module_name"] == module_name]
        assert len(rows) == 1
        row = rows[0]
        assert row["module_type"] == module_type
        assert row["projection_reason"] == projection_reason
        assert not row["projection_reason"].startswith("legacy_bundle:")
        assert "delta_normalized" in row
        assert "hard_masked_targets" in row


def test_tick_records_explicit_salience_memory_trait_and_predictive_contributions(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    controller.tick(
        RoundEvent(
            source="user",
            content="Please help me plan and remember this audit cue.",
            target="alex",
            cue="audit-cue",
            valence=0.08,
        ),
        scenario="chat",
        mode="interactive",
    )
    state = controller.load_runtime_state()
    state.last_action = "plan"
    controller._save_state(state, sync=True)

    result = controller.tick(
        RoundEvent(
            source="user",
            content="Please help me plan again, remember the audit cue, and keep the response stable.",
            target="alex",
            cue="audit-cue",
            valence=0.1,
        ),
        scenario="chat",
        mode="interactive",
    )

    action_audit = result.trace.probability_field["action"]["contribution_audit"]

    expected = {
        "SalienceAgent": ("salience", "salience bias projected from salience head"),
        "HippocampusAgent": ("memory", "memory recall bias projected from hippocampus head"),
        "UnconsciousAgent": ("trait", "trait bias projected from unconscious head"),
        "CerebellarPredictor": ("predictive", "predictive smoothing projected from cerebellar head"),
    }

    for module_name, (module_type, projection_reason) in expected.items():
        rows = [item for item in action_audit if item["module_name"] == module_name]
        assert len(rows) == 1
        row = rows[0]
        assert row["module_type"] == module_type
        assert row["projection_reason"] == projection_reason
        assert not row["projection_reason"].startswith("legacy_bundle:")
        assert row["delta_normalized"] or row["hard_masked_targets"]


def test_salience_and_habit_direct_action_contributions_use_head_contract_without_propose_wrapper():
    salience_agent = SalienceAgent()
    habit_agent = HabitAgent()

    event = RoundEvent(
        source="user",
        content="Please help me plan this audit and keep the response stable.",
        target="alex",
        cue="audit-cue",
        valence=0.12,
    )
    state = {}
    scenario = {"pfc_base_share": 0.25, "habit_weight": 0.12}
    context = {"habit_strength": 0.65}

    salience_signal = salience_agent.score_salience(event, state, scenario, context)
    salience_contribution = salience_agent.build_direct_action_contribution(event, state, scenario, context)
    habit_signal = habit_agent.suggest_default_action(event, state, scenario, context)
    habit_contribution = habit_agent.build_direct_action_contribution(event, state, scenario, context)

    assert not hasattr(salience_agent, "propose")
    assert not hasattr(habit_agent, "propose")

    assert salience_contribution.module_name == "SalienceAgent"
    assert salience_contribution.module_type == "salience"
    assert salience_contribution.level == "action"
    assert salience_contribution.target_space == "action"
    assert salience_contribution.projection_reason == "salience bias projected from salience head"
    assert salience_contribution.raw_signal == salience_signal.raw_signal
    assert salience_contribution.modulated_delta == salience_signal.modulated_delta
    assert salience_contribution.confidence == pytest.approx(salience_signal.confidence)
    assert salience_contribution.confidence_calibrated is not None
    assert any(item.startswith("priority:task_goal") for item in salience_contribution.dependency_trace)
    assert any(item.startswith("control:task") for item in salience_contribution.dependency_trace)

    assert habit_contribution.module_name == "HabitAgent"
    assert habit_contribution.module_type == "habit"
    assert habit_contribution.level == "action"
    assert habit_contribution.target_space == "action"
    assert habit_contribution.projection_reason == "habit bias projected from habit head"
    assert habit_contribution.raw_signal == habit_signal.raw_signal
    assert habit_contribution.modulated_delta == habit_signal.modulated_delta
    assert habit_contribution.confidence == pytest.approx(habit_signal.confidence)
    assert habit_contribution.confidence_calibrated is not None
    assert any(item.startswith("priority:task_goal") for item in habit_contribution.dependency_trace)
    assert any(item.startswith("control:task") for item in habit_contribution.dependency_trace)


def test_relationship_and_hppocampus_direct_action_contributions_use_native_head_contract():
    relationship_agent = RelationshipAgent()
    hippocampus_agent = HippocampusAgent()

    event = RoundEvent(
        source="user",
        content="Please remember the tea discussion and reply warmly to alex.",
        target="alex",
        cue="tea",
        valence=0.12,
    )
    state = {}
    scenario = {"relationship_weight": 0.2}
    context = {"closeness": 0.8, "cue": "tea", "recall_strength": 0.65}

    relationship_contribution = relationship_agent.build_direct_action_contribution(event, state, scenario, context)
    hippocampus_contribution = hippocampus_agent.build_direct_action_contribution(event, state, scenario, context)

    assert not hasattr(relationship_agent, "propose")
    assert not hasattr(hippocampus_agent, "propose")

    assert relationship_contribution.module_name == "RelationshipAgent"
    assert relationship_contribution.module_type == "relationship"
    assert relationship_contribution.level == "action"
    assert relationship_contribution.target_space == "action"
    assert relationship_contribution.projection_reason == "relationship bias projected from relationship head"
    assert relationship_contribution.raw_signal == {"respond": pytest.approx(0.088), "connect": pytest.approx(0.224)}
    assert relationship_contribution.modulated_delta == {"respond": pytest.approx(0.088), "connect": pytest.approx(0.224)}
    assert relationship_contribution.confidence == pytest.approx(0.58)
    assert any(item.startswith("priority:relation_boundary") for item in relationship_contribution.dependency_trace)
    assert any(item.startswith("control:relation") for item in relationship_contribution.dependency_trace)

    assert hippocampus_contribution.module_name == "HippocampusAgent"
    assert hippocampus_contribution.module_type == "memory"
    assert hippocampus_contribution.level == "action"
    assert hippocampus_contribution.target_space == "action"
    assert hippocampus_contribution.projection_reason == "memory recall bias projected from hippocampus head"
    assert hippocampus_contribution.raw_signal == {"recall": pytest.approx(0.243)}
    assert hippocampus_contribution.modulated_delta == {"recall": pytest.approx(0.243)}
    assert hippocampus_contribution.confidence == pytest.approx(0.66)
    assert any(item.startswith("priority:task_goal") for item in hippocampus_contribution.dependency_trace)
    assert any(item.startswith("control:task") for item in hippocampus_contribution.dependency_trace)


def test_live_direct_action_builders_stay_native_and_deterministic(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    state = controller.load_runtime_state()
    state.body_energy = 0.28
    state.budget_remaining = 0.34
    state.last_action = "plan"

    event = RoundEvent(
        source="user",
        content="Please clarify the plan and keep the reply socially safe.",
        target="alex",
        cue="plan",
        valence=-0.42,
    )
    scenario = {"relationship_weight": 0.26}
    context = {
        "closeness": 0.72,
        "burn_rate_ratio": 0.78,
        "low_balance_ratio": 0.66,
        "queue_pressure": 0.31,
        "latency_pressure": 0.22,
    }

    body_agent = BodyStateAgent()
    emotion_agent = EmotionAgent()
    resource_agent = ResourceAgent()
    cerebellar_agent = CerebellarPredictor()
    perspective_agent = PerspectiveModel()

    body_signal = body_agent.compute_body_bias(event, state, scenario, context)
    body_contribution = body_agent.build_direct_action_contribution(event, state, scenario, context)
    assert body_contribution.projection_reason == "homeostatic bias projected from body head"
    assert body_contribution == body_signal

    emotion_signal = emotion_agent.compute_affect_bias(event, state, scenario, context)
    emotion_contribution = emotion_agent.build_direct_action_contribution(event, state, scenario, context)
    assert emotion_contribution.projection_reason == "affect bias projected from emotion head"
    assert emotion_contribution == emotion_signal

    resource_signal = resource_agent.map_budget_to_bias(event, state, scenario, context)
    resource_contribution = resource_agent.build_direct_action_contribution(event, state, scenario, context)
    assert resource_contribution.projection_reason == "resource bias projected from resource head"
    assert resource_contribution == resource_signal

    cerebellar_signal = cerebellar_agent.micro_adjust_action(event, state, scenario, context)
    cerebellar_contribution = cerebellar_agent.build_direct_action_contribution(event, state, scenario, context)
    assert cerebellar_contribution.projection_reason == "predictive smoothing projected from cerebellar head"
    assert cerebellar_contribution == cerebellar_signal

    perspective_signal = perspective_agent.adjust_social_interpretation(event, state, scenario, context)
    perspective_contribution = perspective_agent.build_direct_action_contribution(event, state, scenario, context)
    assert perspective_contribution.projection_reason == "perspective bias projected from perspective head"
    assert perspective_contribution == perspective_signal
    assert not hasattr(resource_agent, "propose")
    assert not hasattr(cerebellar_agent, "propose")
    assert not hasattr(perspective_agent, "propose")


def test_tick_records_execution_tool_affordance_contribution_for_active_run(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    controller.start_run("检查 worker.py 并规划下一步")
    result = controller.tick(
        RoundEvent(
            source="user",
            content="继续检查 worker.py，并告诉我下一步怎么做。",
            target="user",
            cue="worker.py",
            valence=0.02,
        ),
        scenario="task",
        mode="interactive",
    )

    action_audit = result.trace.probability_field["action"]["contribution_audit"]
    rows = [item for item in action_audit if item["module_name"] == "SkillExecutor"]

    assert len(rows) == 1
    row = rows[0]
    assert row["module_type"] == "tooling"
    assert row["projection_reason"] == "tool affordance prior projected from active run context"
    assert not row["projection_reason"].startswith("legacy_bundle:")
    assert any(item.startswith("tool_choice:") for item in row["dependency_trace"])
    assert any(item.startswith("tool_expected_value:") for item in row["dependency_trace"])
    assert any(item.startswith("tool_cost_penalty:") for item in row["dependency_trace"])
    assert row["delta_normalized"] or row["hard_masked_targets"]
