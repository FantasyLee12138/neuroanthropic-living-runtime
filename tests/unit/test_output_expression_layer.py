import nalr.output.renderer as renderer_module
from nalr.output.renderer import fallback_render_text
from nalr.output.style import build_expression_profile, build_render_plan
from nalr.schemas.models import ExpressionProfile, IdentityContext, RenderPlan, StochasticState


def _expression(*, directness: float = 0.34, hedging: float = 0.58, warmth: float = 0.50, repair: float = 0.72) -> ExpressionProfile:
    return ExpressionProfile(
        reply_delay=0.36,
        latency_style=0.48,
        sentence_fragmentation=0.14,
        hedging_level=hedging,
        warmth_level=warmth,
        directness_level=directness,
        self_disclosure=0.12,
        tone_sharpness=0.18,
        repair_tendency=repair,
        timing_jitter=0.0,
        fragmentation_jitter=0.0,
    )


def _repair_plan(*, stage: str, safety_invite: bool, risk: float = 0.72) -> RenderPlan:
    return RenderPlan(
        action="respond",
        expression=_expression(),
        safety_constraints={"conflict_hot": stage == "repairing", "repair_stage": stage},
        message_plan={
            "repair_expression": {
                "source": "conflict" if stage else None,
                "stage": stage,
                "visibility": "implicit",
                "opening_mode": "soft_resume" if stage == "cooling" else "buffered",
                "advance_mode": "resume" if stage == "cooling" else "limited",
                "safety_invite": safety_invite,
                "template": "body_first",
                "transition_reason": "forced_compromise",
            }
        },
        event_summary="帮我直接回复 Alex",
        scenario="chat",
        target="Alex",
        relation_state={"relationship_risk": risk},
        perspective={"reaction_hypothesis": {"risk": risk}},
    )


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
        expression=_expression(directness=0.82, hedging=0.12, warmth=0.68, repair=0.18),
        safety_constraints={"conflict_hot": False},
        event_summary="帮我直接回复 Alex",
        scenario="task",
        target="Alex",
        relation_state={"relationship_risk": 0.18},
        perspective={"reaction_hypothesis": {"risk": 0.15}},
    )
    careful = RenderPlan(
        action="respond",
        expression=_expression(directness=0.34, hedging=0.62, warmth=0.46, repair=0.74),
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


def test_fallback_renderer_uses_stage_aware_repair_expression_policy():
    adjusting = fallback_render_text(_repair_plan(stage="adjusting", safety_invite=False))
    repairing = fallback_render_text(_repair_plan(stage="repairing", safety_invite=True))
    cooling = fallback_render_text(_repair_plan(stage="cooling", safety_invite=False))
    recovered = fallback_render_text(_repair_plan(stage="recovered", safety_invite=False, risk=0.25))

    assert "打断我" not in adjusting
    assert "收一下" in adjusting
    assert "打断我" in repairing
    assert "稳一点" in repairing
    assert "继续" in cooling
    assert "打断我" not in cooling
    assert "收一下" not in recovered
    assert "稳一点" not in recovered
    assert "打断我" not in recovered


def test_fallback_renderer_compresses_short_reply_without_starvation_style():
    plan = RenderPlan(
        action="short_reply",
        expression=_expression(directness=0.70, hedging=0.20, warmth=0.52, repair=0.30),
        safety_constraints={"conflict_hot": False},
        message_plan={"slow_variables": {"resource_scarcity": 0.68}},
        event_summary="帮我直接回复 Alex",
        scenario="task",
        target="Alex",
        relation_state={"relationship_risk": 0.20},
        perspective={"reaction_hypothesis": {"risk": 0.18}},
    )

    text = fallback_render_text(plan)

    assert "短" in text or "直接" in text
    assert "收一点" not in text


def test_self_identity_reply_keeps_basic_fallback_without_runtime_class_label():
    plan = RenderPlan(
        action="respond",
        expression=_expression(),
        event_summary="你好，你是谁？你有名字吗？",
        scenario="chat",
        target="user",
        identity_context=IdentityContext(query_kind="self_identity", display_label="当前运行体"),
    )

    text = fallback_render_text(plan)

    assert text == "我是当前运行体。"
    assert "runtime_instance" not in text


def test_self_identity_reply_uses_basic_named_fallback():
    plan = RenderPlan(
        action="respond",
        expression=_expression(),
        event_summary="你好，你是谁？你有名字吗？",
        scenario="chat",
        target="user",
        identity_context=IdentityContext(query_kind="self_identity", display_label="阿澜"),
    )

    text = fallback_render_text(plan)

    assert text == "我是阿澜。"
    assert "runtime_instance" not in text
    assert "豆包" not in text


def test_tlh_instinct_actions_have_distinct_fallback_rendering():
    absorb = RenderPlan(
        action="absorb",
        expression=_expression(directness=0.28, hedging=0.42, warmth=0.36, repair=0.52),
        event_summary="先别急着答，慢一点",
        scenario="chat",
        target="user",
    )
    nothing = RenderPlan(
        action="nothing",
        expression=_expression(directness=0.20, hedging=0.20, warmth=0.20, repair=0.20),
        event_summary="先别说话",
        scenario="chat",
        target="user",
    )
    die = RenderPlan(
        action="die",
        expression=_expression(directness=0.54, hedging=0.18, warmth=0.10, repair=0.10),
        event_summary="那你想停下吗",
        scenario="chat",
        target="user",
    )

    assert "吸收" in fallback_render_text(absorb)
    assert fallback_render_text(nothing) == ""
    assert "结束生命" in fallback_render_text(die)


def test_fallback_renderer_replies_to_plain_greeting_without_generic_template():
    plan = RenderPlan(
        action="respond",
        expression=_expression(directness=0.36, hedging=0.28, warmth=0.66, repair=0.22),
        safety_constraints={"conflict_hot": False},
        event_summary="晚上好",
        scenario="chat",
        target="user",
    )

    text = fallback_render_text(plan)

    assert text == "晚上好，我在。"


def test_fallback_renderer_anchors_followup_to_specific_memory_cue():
    plan = RenderPlan(
        action="plan",
        expression=_expression(directness=0.72, hedging=0.18, warmth=0.60, repair=0.18),
        safety_constraints={"conflict_hot": False},
        message_plan={
            "memory_cue": "写作业",
            "recall_strength": 0.42,
            "slow_variables": {"resource_scarcity": 0.18},
        },
        event_summary="我们继续吧",
        scenario="task",
        target="user",
        relation_state={"relationship_risk": 0.12},
        perspective={"reaction_hypothesis": {"risk": 0.08}},
    )

    text = fallback_render_text(plan)

    assert "写作业" in text
    assert "继续吧" in text


def test_fallback_renderer_answers_time_query_with_local_clock(monkeypatch):
    class _FakeNow:
        def astimezone(self):
            return self

        def strftime(self, fmt: str) -> str:
            assert fmt == "%H:%M"
            return "21:37"

    class _FakeDateTime:
        @staticmethod
        def now():
            return _FakeNow()

    monkeypatch.setattr(renderer_module, "datetime", _FakeDateTime)

    plan = RenderPlan(
        action="respond",
        expression=_expression(directness=0.40, hedging=0.24, warmth=0.58, repair=0.18),
        safety_constraints={"conflict_hot": False},
        event_summary="现在几点了？",
        scenario="chat",
        target="user",
    )

    text = fallback_render_text(plan)

    assert text == "现在是 21:37。"


def test_monologue_render_plan_tracks_delivery_mode_and_visible_prefix():
    plan = build_render_plan(
        sampled_action="monologue",
        expression=_expression(directness=0.26, hedging=0.44, warmth=0.40, repair=0.54),
        safety_constraints={"conflict_hot": False},
        scenario="companion",
        event_summary="我是不是该先把这件事想清楚",
        target="user",
        relation_state={"relationship_risk": 0.12},
        perspective={"reaction_hypothesis": {"risk": 0.08}},
    )

    text = fallback_render_text(plan)

    assert plan.delivery_mode == "monologue"
    assert text.startswith("【独白】")
    assert "你刚才提到" not in text
    assert "我" in text
