from nalr.output.style import build_expression_profile
from nalr.schemas.models import ExpressionProfile, StochasticState


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
