from nalr.output.renderer import fallback_render_text
from nalr.output.style import build_expression_profile
from nalr.schemas.models import ExpressionProfile, RenderPlan, StochasticState


def test_expression_profile_maps_intensity_to_delay_disclosure_and_repair():
    low = build_expression_profile(
        sampled_action="respond",
        state={"body_energy": 0.8, "mood": 0.6},
        scenario="companion",
        scenario_config={"output_warmth_variance": 0.30, "delay_tolerance": 0.28},
        relation_state={"closeness": 0.9, "relationship_risk": 0.1, "boundary_level": 0.2, "privacy_level": 0.8},
        stochastic=StochasticState(
            emo_channel="warm",
            xi_emo=0.08,
            xi_mood=0.04,
            lambda_noise=0.22,
            r_intensity=0.25,
            noise_guard_triggered=False,
            round_seed=7,
        ),
    )
    high = build_expression_profile(
        sampled_action="respond",
        state={"body_energy": 0.8, "mood": 0.6},
        scenario="companion",
        scenario_config={"output_warmth_variance": 0.30, "delay_tolerance": 0.28},
        relation_state={"closeness": 0.9, "relationship_risk": 0.1, "boundary_level": 0.2, "privacy_level": 0.8},
        stochastic=StochasticState(
            emo_channel="warm",
            xi_emo=0.08,
            xi_mood=0.04,
            lambda_noise=0.22,
            r_intensity=0.85,
            noise_guard_triggered=False,
            round_seed=8,
        ),
    )

    assert isinstance(low, ExpressionProfile)
    assert high.reply_delay > low.reply_delay
    assert high.self_disclosure > low.self_disclosure
    assert high.tone_sharpness > low.tone_sharpness
    assert high.repair_tendency < low.repair_tendency
    assert -0.08 <= low.timing_jitter <= 0.12
    assert -0.05 <= high.fragmentation_jitter <= 0.10


def test_fallback_renderer_consumes_expression_profile_and_safety_constraints():
    restrained = RenderPlan(
        action="respond",
        expression=ExpressionProfile(
            reply_delay=0.18,
            latency_style=0.26,
            sentence_fragmentation=0.08,
            hedging_level=0.12,
            warmth_level=0.68,
            directness_level=0.82,
            self_disclosure=0.18,
            tone_sharpness=0.22,
            repair_tendency=0.18,
            timing_jitter=0.0,
            fragmentation_jitter=0.0,
        ),
        safety_constraints={"conflict_hot": False},
        event_summary="帮我直接回复 Alex",
        scenario="task",
        target="Alex",
        relation_state={"relationship_risk": 0.18},
        perspective={"reaction_hypothesis": {"risk": 0.15}},
    )
    careful = RenderPlan(
        action="respond",
        expression=ExpressionProfile(
            reply_delay=0.44,
            latency_style=0.60,
            sentence_fragmentation=0.20,
            hedging_level=0.62,
            warmth_level=0.46,
            directness_level=0.34,
            self_disclosure=0.10,
            tone_sharpness=0.18,
            repair_tendency=0.74,
            timing_jitter=0.0,
            fragmentation_jitter=0.0,
        ),
        safety_constraints={"conflict_hot": True},
        event_summary="帮我直接回复 Alex",
        scenario="chat",
        target="Alex",
        relation_state={"relationship_risk": 0.72},
        perspective={"reaction_hypothesis": {"risk": 0.78}},
    )

    restrained_text = fallback_render_text(restrained)
    careful_text = fallback_render_text(careful)

    assert restrained_text != careful_text
    assert "先说重点" in restrained_text
    assert "如果你愿意" in careful_text
