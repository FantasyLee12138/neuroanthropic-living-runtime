from nalr.runtime.probability_field import ProbabilityFieldIntegrator
from nalr.schemas import models
from nalr.schemas.models import (
    PeakArbitrationRecord,
    ProbabilisticContribution,
    ProbabilityFieldSnapshot,
    ProbabilityLayerState,
    TokenContributionTrace,
)


def test_probability_contribution_normalizes_layer_payloads():
    contribution = ProbabilisticContribution(
        module_name="IdentityRuntime",
        layer="executive",
        level="global",
        confidence=1.2,
        delta_logits={"respond": 0.6, "plan": -0.2},
        soft_mask={"wander": -3.0},
        hard_mask=["connect"],
        posterior={"identity_alignment": 0.9},
        trace_reason="stabilize self prior",
    )

    assert contribution.confidence == 1.0
    assert contribution.layer == "executive"
    assert contribution.hard_mask == ["connect"]
    assert contribution.delta_logits["respond"] == 0.6


def test_probability_field_integrator_accepts_projection_based_contributions():
    integrator = ProbabilityFieldIntegrator()
    bundle = models.ProposalBundle(
        owner="PFCAgent",
        confidence=0.8,
        action_preferences={"plan": 0.7},
        delta_p={"plan": 0.7},
        reason="goal prior reweight",
    )
    contribution = ProbabilisticContribution.from_proposal_bundle(
        bundle,
        module_name="PFCAgent",
        layer="action",
        level="action",
        trace_reason="goal prior reweight",
    )

    snapshot = integrator.integrate(
        base_action_logits={"respond": 0.1, "plan": 0.2},
        base_token_logits={"我": 0.1},
        context_contributions=[],
        memory_contributions=[],
        action_contributions=[contribution],
        token_contributions=[],
    )

    assert snapshot.winner_posterior["top_action"] == "plan"
    assert snapshot.action_logits_final["plan"] > snapshot.action_logits_final["respond"]


def test_probability_field_integrator_merges_context_memory_action_and_token_layers():
    integrator = ProbabilityFieldIntegrator()

    context = [
        ProbabilisticContribution(
            module_name="ThalamusAttentionAgent",
            layer="context",
            level="context",
            confidence=0.8,
            attention_bias={"recent_task": 0.7, "small_talk": -0.4},
            trace_reason="surprisal and relevance routing",
        )
    ]
    memory = [
        ProbabilisticContribution(
            module_name="HippocampusAgent",
            layer="memory",
            level="memory",
            confidence=0.75,
            posterior={"rice_memory": 0.82},
            delta_logits={"recall": 0.4},
            trace_reason="episodic reactivation",
        )
    ]
    action = [
        ProbabilisticContribution(
            module_name="PFCAgent",
            layer="action",
            level="action",
            confidence=0.9,
            delta_logits={"plan": 0.6, "respond": 0.2},
            trace_reason="goal prior reweight",
        ),
        ProbabilisticContribution(
            module_name="AuthenticityPolicy",
            layer="executive",
            level="global",
            confidence=0.7,
            soft_mask={"wander": -1.5},
            trace_reason="inauthentic drift penalty",
        ),
    ]
    token = [
        ProbabilisticContribution(
            module_name="IdentityRuntime",
            layer="token",
            level="token",
            confidence=0.85,
            delta_logits={"我": 0.3, "我们": 0.1},
            trace_reason="self prior token lift",
        )
    ]

    snapshot = integrator.integrate(
        base_action_logits={"respond": 0.5, "plan": 0.4, "wander": 0.1},
        base_token_logits={"我": 0.2, "我们": 0.15, "你": 0.1},
        context_contributions=context,
        memory_contributions=memory,
        action_contributions=action,
        token_contributions=token,
    )

    assert isinstance(snapshot, ProbabilityFieldSnapshot)
    assert snapshot.context_attn_final["recent_task"] > snapshot.context_attn_final["small_talk"]
    assert snapshot.memory_prior_final["rice_memory"] == 0.82
    assert snapshot.action_logits_final["plan"] > snapshot.action_logits_final["respond"]
    assert snapshot.action_logits_final["wander"] < 0.1
    assert snapshot.token_logits_final["我"] > snapshot.token_logits_final["你"]
    assert snapshot.winner_posterior["top_action"] == "plan"
    assert snapshot.counterfactual_top_peaks


def test_probability_field_snapshot_captures_token_trace_and_peak_arbitration():
    snapshot = ProbabilityFieldSnapshot(
        layers=[
            ProbabilityLayerState(
                layer="token",
                contributions=["IdentityRuntime"],
                combined_bias={"我": 0.3},
                suppressed_targets=["connect"],
            )
        ],
        context_attn_final={"recent_task": 0.8},
        memory_prior_final={"rice_memory": 0.7},
        action_logits_final={"plan": 0.9},
        token_logits_final={"我": 0.5},
        winner_posterior={"top_action": "plan", "probability": 0.9},
        counterfactual_top_peaks=[{"action": "respond", "score": 0.6}],
        token_traces=[
            TokenContributionTrace(
                token="我",
                final_logit=0.5,
                module_deltas={"IdentityRuntime": 0.3},
                suppressed_by=[],
                reason="self prior token lift",
            )
        ],
        peak_arbitration=PeakArbitrationRecord(
            winning_peak="plan",
            winning_score=0.9,
            competing_peaks={"respond": 0.6},
            suppression_reasons={"respond": "lower long-run value"},
            compromise_applied=False,
        ),
    )

    assert snapshot.token_traces[0].module_deltas["IdentityRuntime"] == 0.3
    assert snapshot.peak_arbitration.winning_peak == "plan"
    assert snapshot.peak_arbitration.suppression_reasons["respond"] == "lower long-run value"
