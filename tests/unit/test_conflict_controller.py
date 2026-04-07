from pathlib import Path

from nalr.agents.modules import ConflictMonitorAgent, ThalamusAttentionAgent
from nalr.runtime.controller import RuntimeController
from nalr.schemas.models import ActionEvidenceSignal, ConflictRepairState, RoundEvent


CONFIG_ROOT = Path(__file__).resolve().parents[2] / "config"


def _signal(
    module_name: str,
    *,
    confidence: float,
    action_delta: dict[str, float],
    priority_bucket: str,
    control_domain: str,
    veto: bool = False,
    gated_actions: list[str] | None = None,
    risk_hints: dict | None = None,
) -> ActionEvidenceSignal:
    return ActionEvidenceSignal(
        module_name=module_name,
        module_type=control_domain,
        confidence=confidence,
        action_delta=dict(action_delta),
        veto=veto,
        trace_reason=f"{module_name} signal",
        sigma_scale=1.0,
        priority_bucket=priority_bucket,
        control_domain=control_domain,
        gated_actions=list(gated_actions or []),
        risk_hints=dict(risk_hints or {}),
    )


def _action_layer(
    winner_posterior: dict[str, float],
    *,
    hard_masked_targets: list[str] | None = None,
    final_energy: dict[str, float] | None = None,
) -> dict:
    return {
        "winner_posterior": dict(winner_posterior),
        "final_energy": dict(final_energy or winner_posterior),
        "hard_masked_targets": list(hard_masked_targets or []),
        "contribution_audit": [],
    }


def _probability_field(action_layer: dict) -> dict:
    return {
        "action": dict(action_layer),
        "source_chain": ["field_native_test"],
    }


def _prime_conflict_state(
    controller: RuntimeController,
    *,
    body_energy: float = 0.08,
    budget_remaining: float = 0.04,
    mood: float = 0.2,
) -> None:
    state = controller.load_runtime_state()
    state.body_energy = body_energy
    state.budget_remaining = budget_remaining
    state.mood = mood
    controller._save_state(state)


def test_score_conflict_reports_full_component_breakdown():
    agent = ConflictMonitorAgent()
    signals = [
        _signal(
            "BodyStateAgent",
            confidence=0.96,
            action_delta={"rest": 0.32, "respond": 0.04},
            priority_bucket="body_safety",
            control_domain="body",
            gated_actions=["connect", "plan"],
            risk_hints={"body_load": 0.95},
        ),
        _signal(
            "ResourceAgent",
            confidence=0.90,
            action_delta={"rest": 0.18, "respond": 0.06},
            priority_bucket="budget_overload",
            control_domain="resource",
            gated_actions=["plan"],
            risk_hints={"overload": 0.88},
        ),
        _signal(
            "PFCAgent",
            confidence=0.91,
            action_delta={"plan": 0.30, "respond": 0.08},
            priority_bucket="task_goal",
            control_domain="task",
            risk_hints={"goal_pressure": 0.72},
        ),
        _signal(
            "RelationshipAgent",
            confidence=0.87,
            action_delta={"connect": 0.24, "clarify": 0.10},
            priority_bucket="relation_boundary",
            control_domain="relation",
            gated_actions=["connect"],
            risk_hints={"relationship_risk": 0.83},
        ),
        _signal(
            "DesireAgent",
            confidence=0.82,
            action_delta={"rest": 0.22, "wander": 0.12},
            priority_bucket="immediate_desire",
            control_domain="desire",
            risk_hints={"comfort_pull": 0.76},
        ),
        _signal(
            "DMNAgent",
            confidence=0.78,
            action_delta={"wander": 0.28},
            priority_bucket="roaming",
            control_domain="dmn",
            risk_hints={"roam_pull": 0.94},
        ),
    ]
    action_view = _action_layer(
        {
            "rest": 0.31,
            "plan": 0.27,
            "connect": 0.17,
            "wander": 0.15,
            "respond": 0.10,
        }
    )

    assessment = agent.score_conflict(signals, action_view)

    assert assessment["score"] >= 0.82
    assert set(assessment["components"]) == {
        "proposal_divergence",
        "veto_tension",
        "value_gap",
        "relation_risk_gap",
        "body_gap",
    }
    assert assessment["components"]["proposal_divergence"] > 0.0
    assert assessment["components"]["value_gap"] > 0.0
    assert assessment["components"]["relation_risk_gap"] > 0.0
    assert assessment["components"]["body_gap"] > 0.0
    assert assessment["priority_signals"]["body_safety"] >= assessment["priority_signals"]["task_goal"]
    assert assessment["critical_conflict"] is True


def test_conflict_resolution_prefers_body_before_task_and_roaming():
    agent = ConflictMonitorAgent()
    signals = [
        _signal(
            "BodyStateAgent",
            confidence=0.97,
            action_delta={"rest": 0.34},
            priority_bucket="body_safety",
            control_domain="body",
            gated_actions=["plan", "connect"],
            risk_hints={"body_load": 0.96},
        ),
        _signal(
            "PFCAgent",
            confidence=0.90,
            action_delta={"plan": 0.28},
            priority_bucket="task_goal",
            control_domain="task",
            risk_hints={"goal_pressure": 0.85},
        ),
        _signal(
            "DMNAgent",
            confidence=0.88,
            action_delta={"wander": 0.26},
            priority_bucket="roaming",
            control_domain="dmn",
            risk_hints={"roam_pull": 0.93},
        ),
    ]
    action_view = _action_layer(
        {
            "rest": 0.33,
            "plan": 0.31,
            "wander": 0.24,
            "respond": 0.12,
        }
    )
    assessment = agent.score_conflict(signals, action_view)

    resolution = agent.trigger_control_escalation(assessment, action_view, 0)

    assert resolution["winning_priority"] == "body_safety"
    assert "plan" in resolution["blocked_actions"]
    assert "wander" in resolution["blocked_actions"]
    assert resolution["applied_template"] in {None, "body_first"}


def test_conflict_resolution_prefers_strongest_peak_pressure_over_fixed_bucket_order():
    agent = ConflictMonitorAgent()
    assessment = {
        "score": 0.86,
        "priority_signals": {
            "body_safety": 0.61,
            "budget_overload": 0.12,
            "relation_boundary": 0.18,
            "task_goal": 0.93,
            "immediate_desire": 0.24,
            "roaming": 0.22,
        },
        "critical_conflict": True,
    }
    action_view = _action_layer(
        {
            "plan": 0.42,
            "rest": 0.25,
            "respond": 0.18,
            "wander": 0.15,
        }
    )

    resolution = agent.trigger_control_escalation(assessment, action_view, 0)

    assert resolution["winning_priority"] == "task_goal"
    assert "wander" in resolution["blocked_actions"]
    assert "plan" not in resolution["blocked_actions"]


def test_conflict_and_thalamus_producers_pass_canonical_kwargs(monkeypatch):
    import nalr.agents.modules as agent_modules

    captured: list[dict[str, object]] = []

    def spy_contribution(**kwargs):
        captured.append(kwargs)
        return kwargs

    monkeypatch.setattr(agent_modules, "ProbabilisticContribution", spy_contribution)

    conflict_agent = ConflictMonitorAgent()
    conflict_state = {
        "score": 0.84,
        "priority_signals": {"body_safety": 0.92, "task_goal": 0.48},
        "winning_priority": "body_safety",
        "passes": [
            {
                "resolution": {
                    "flag": True,
                    "blocked_actions": ["plan", "wander"],
                    "action_scales": {"plan": 0.2, "wander": 0.3},
                    "applied_template": "body_first",
                }
            }
        ],
        "critical_conflict": True,
        "compromise": {"template": "body_first", "triggered": True},
    }
    action_view = _action_layer({"rest": 0.54, "plan": 0.28, "wander": 0.18})
    conflict_agent.build_arbitration_contribution(conflict_state, action_view)

    event = RoundEvent(source="user", content="remember tea and plan carefully", target="alex", cue="tea")
    thalamus_agent = ThalamusAttentionAgent()
    thalamus_agent.build_context_routing_contribution(
        event,
        {},
        {},
        {"cue": "tea", "closeness": 0.7, "recall_strength": 0.55},
    )
    thalamus_agent.build_memory_routing_contribution(
        event,
        {},
        {},
        {"cue": "tea", "closeness": 0.7, "recall_strength": 0.55},
    )

    assert len(captured) == 3
    for kwargs in captured:
        assert "delta_energy" not in kwargs
        assert "attention_bias" not in kwargs
    assert set(captured[0]["hard_mask"]).issubset({"plan", "wander"})
    assert captured[0]["modulated_delta"]["plan"] < 0.0
    assert captured[1]["raw_signal"]["cue:tea"] > 0.0
    assert captured[1]["modulated_delta"]["task_relevance"] > 0.0
    assert captured[2]["raw_signal"] == {"tea": captured[2]["modulated_delta"]["tea"]}


def test_score_conflict_accepts_field_native_action_layer_truth():
    agent = ConflictMonitorAgent()
    signals = [
        _signal(
            "BodyStateAgent",
            confidence=0.96,
            action_delta={"rest": 0.32, "respond": 0.04},
            priority_bucket="body_safety",
            control_domain="body",
            gated_actions=["connect", "plan"],
            risk_hints={"body_load": 0.95},
        ),
        _signal(
            "PFCAgent",
            confidence=0.91,
            action_delta={"plan": 0.30, "respond": 0.08},
            priority_bucket="task_goal",
            control_domain="task",
            risk_hints={"goal_pressure": 0.72},
        ),
        _signal(
            "RelationshipAgent",
            confidence=0.87,
            action_delta={"connect": 0.24, "clarify": 0.10},
            priority_bucket="relation_boundary",
            control_domain="relation",
            gated_actions=["connect"],
            risk_hints={"relationship_risk": 0.83},
        ),
        _signal(
            "DMNAgent",
            confidence=0.78,
            action_delta={"wander": 0.28},
            priority_bucket="roaming",
            control_domain="dmn",
            risk_hints={"roam_pull": 0.94},
        ),
    ]
    action_layer = _action_layer(
        {
            "rest": 0.30,
            "plan": 0.28,
            "connect": 0.18,
            "wander": 0.14,
            "respond": 0.10,
        },
        hard_masked_targets=["connect"],
    )

    assessment = agent.score_conflict(signals, action_layer)
    resolution = agent.trigger_control_escalation(assessment, action_layer, 0)

    assert assessment["score"] >= 0.75
    assert assessment["components"]["proposal_divergence"] > 0.0
    assert assessment["components"]["body_gap"] > 0.0
    assert assessment["priority_signals"]["body_safety"] >= assessment["priority_signals"]["task_goal"]
    assert assessment["critical_conflict"] is (assessment["score"] >= agent.conflict_critical)
    assert resolution["winning_priority"] == "body_safety"
    assert "plan" in resolution["blocked_actions"]
    assert "wander" in resolution["blocked_actions"]


def test_build_arbitration_contribution_accepts_probability_field_action_view():
    agent = ConflictMonitorAgent()
    probability_field = _probability_field(
        _action_layer(
            {
                "rest": 0.34,
                "plan": 0.30,
                "wander": 0.22,
                "respond": 0.14,
            },
            hard_masked_targets=["wander"],
        )
    )
    conflict_state = {
        "score": 0.84,
        "priority_signals": {
            "body_safety": 0.88,
            "budget_overload": 0.20,
            "relation_boundary": 0.12,
            "task_goal": 0.61,
            "immediate_desire": 0.18,
            "roaming": 0.72,
        },
        "winning_priority": "body_safety",
        "dominant_conflicts": ["proposal_divergence", "body_gap"],
        "critical_conflict": True,
        "allowed_resamples": 2,
        "compromise": {"triggered": True, "template": "body_first"},
        "passes": [
            {
                "resolution": {
                    "flag": True,
                    "winning_priority": "body_safety",
                    "blocked_actions": ["plan", "wander"],
                    "action_scales": {"plan": 0.2, "wander": 0.0, "rest": 1.3},
                    "applied_template": "body_first",
                }
            }
        ],
    }

    contribution = agent.build_arbitration_contribution(conflict_state, probability_field)

    assert contribution.module_name == "ConflictMonitorAgent"
    assert contribution.native_operator == "conflict_arbitration"
    assert contribution.level == "action"
    assert contribution.target_space == "action"
    assert contribution.modulated_delta["plan"] < 0.0
    assert contribution.modulated_delta["wander"] < 0.0
    assert contribution.posterior["rest"] > contribution.posterior["plan"]
    assert set(contribution.hard_mask).issubset({"plan", "wander"})
    assert contribution.compromise_template_prior["body_first"] > 0.0
    assert "winning_priority:body_safety" in contribution.dependency_trace


def test_thalamus_aggregate_proposals_accepts_field_native_action_truth():
    agent = ThalamusAttentionAgent()

    action_truth = {
        "winner_posterior": {
            "rest": 0.64,
            "plan": 0.36,
        },
        "final_energy": {
            "rest": 0.64,
            "plan": 0.36,
        },
        "hard_masked_targets": ["plan"],
        "contribution_audit": [],
    }

    assert agent.aggregate_proposals(action_truth) == {"distribution": {"rest": 0.64, "plan": 0.36}}


def test_request_resample_uses_configured_thresholds_and_critical_override():
    agent = ConflictMonitorAgent()

    low = agent.request_resample({"score": 0.40, "critical_conflict": False}, 0)
    below_high = agent.request_resample({"score": 0.70, "critical_conflict": False}, 0)
    high = agent.request_resample({"score": 0.75, "critical_conflict": False}, 0)
    high_exhausted = agent.request_resample({"score": 0.79, "critical_conflict": False}, 1)
    critical = agent.request_resample({"score": 0.91, "critical_conflict": True}, 1)
    exhausted = agent.request_resample({"score": 0.91, "critical_conflict": True}, 2)

    assert low["flag"] is False
    assert low["allowed_resamples"] == 0
    assert below_high["flag"] is False
    assert below_high["allowed_resamples"] == 0
    assert high["flag"] is True
    assert high["allowed_resamples"] == 1
    assert high["force_compromise"] is False
    assert high_exhausted["flag"] is False
    assert high_exhausted["allowed_resamples"] == 1
    assert critical["flag"] is True
    assert critical["allowed_resamples"] == 2
    assert critical["force_compromise"] is True
    assert exhausted["flag"] is False
    assert exhausted["force_compromise"] is True


def test_runtime_controller_builds_conflict_agent_with_threshold_config(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    agent = controller.agent_map["ConflictMonitorAgent"]

    assert agent.conflict_high == 0.75
    assert agent.conflict_critical == 0.82
    assert agent.max_resample_rounds == 1


def test_mark_post_error_adjustment_returns_shift_adjustment_contract():
    agent = ConflictMonitorAgent()

    result = agent.mark_post_error_adjustment(
        state={
            "repair_state": ConflictRepairState(stage="idle"),
            "repair_ledger": [],
            "conflict_recovery_rounds": 0,
        },
        round_id=7,
        resolution={"winning_priority": "task_goal"},
        compromise={"triggered": False, "template": None},
        deadlock_fuse_triggered=False,
        top_action_before="wander",
        top_action_after="plan",
        blocked_actions=["wander"],
        safe_mode_before=False,
    )

    assert result["repair_mode"] == "post_error_adjustment"
    assert result["post_error_adjustment"]["triggered"] is True
    assert result["post_error_adjustment"]["reason"] == "top_action_shift"
    assert result["repair_transition"]["to_stage"] == "adjusting"
    assert result["repair_ledger_append"]["repair_stage_after"] == "adjusting"


def test_mark_post_error_adjustment_returns_forced_compromise_contract():
    agent = ConflictMonitorAgent()

    result = agent.mark_post_error_adjustment(
        state={
            "repair_state": ConflictRepairState(stage="idle"),
            "repair_ledger": [],
            "conflict_recovery_rounds": 0,
        },
        round_id=8,
        resolution={"winning_priority": "body_safety", "applied_template": "body_first"},
        compromise={"triggered": True, "template": "body_first"},
        deadlock_fuse_triggered=False,
        top_action_before="plan",
        top_action_after="rest",
        blocked_actions=["plan", "connect"],
        safe_mode_before=False,
    )

    assert result["repair_mode"] == "post_error_adjustment"
    assert result["post_error_adjustment"]["reason"] == "forced_compromise"
    assert result["repair_transition"]["to_stage"] == "adjusting"
    assert result["repair_ledger_append"]["template"] == "body_first"


def test_mark_post_error_adjustment_returns_deadlock_fuse_contract():
    agent = ConflictMonitorAgent()

    result = agent.mark_post_error_adjustment(
        state={
            "repair_state": ConflictRepairState(stage="adjusting"),
            "repair_ledger": [{"round_id": 1, "reason": "forced_compromise"}],
            "conflict_recovery_rounds": 5,
        },
        round_id=9,
        resolution={"winning_priority": "body_safety", "applied_template": "body_first"},
        compromise={"triggered": True, "template": "body_first"},
        deadlock_fuse_triggered=True,
        top_action_before="plan",
        top_action_after="rest",
        blocked_actions=["plan", "connect", "wander"],
        safe_mode_before=False,
    )

    assert result["repair_mode"] == "deadlock_fuse"
    assert result["post_error_adjustment"]["reason"] == "deadlock_fuse"
    assert result["repair_transition"]["to_stage"] == "repairing"
    assert result["conflict_safe_mode_owner"] == "conflict"
    assert result["safe_mode_patch"] is True
    assert result["repair_cooldown_rounds_patch"] >= 2


def test_runtime_sets_conflict_hot_after_three_critical_rounds_and_recovers(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    controller.entropy_pool.ingest_bytes(bytes(range(256)) * 20, source="fixture_qrng", reason="critical conflict")
    _prime_conflict_state(controller)

    critical_event = RoundEvent(
        source="user",
        content="I need a quick easy break, but help me plan carefully, reply to Alex, and let me drift.",
        target="alex",
        cue="break",
        valence=-0.40,
        energy_delta=-0.18,
    )

    for _ in range(3):
        result = controller.tick(critical_event, scenario="task", mode="interactive")

    hot_state = controller.load_runtime_state()
    hot_conflict = result.trace.conflict_arbitration

    assert hot_conflict["critical_conflict"] is True
    assert hot_state.critical_conflict_streak == 3
    assert hot_state.conflict_hot_rounds == 5
    assert hot_conflict["circuit_breaker"]["triggered"] is True
    assert hot_conflict["repair_state_snapshot"]["stage"] == "repairing"
    assert hot_state.repair_state.stage == "repairing"
    assert hot_state.conflict_safe_mode_owner == "conflict"
    assert hot_state.repair_ledger
    assert hot_state.conflict_learning_state["adjustment_reasons"]
    assert result.trace.render_plan["safety_constraints"]["conflict_hot"] is True

    calm_event = RoundEvent(
        source="user",
        content="Please answer directly with one clear next step.",
        target="user",
        cue="step",
        valence=0.05,
        energy_delta=0.08,
    )

    for _ in range(5):
        controller.tick(calm_event, scenario="task", mode="interactive")

    recovered_state = controller.load_runtime_state()
    assert recovered_state.conflict_hot_rounds == 0
    assert recovered_state.critical_conflict_streak == 0
    assert recovered_state.safe_mode is False
    assert recovered_state.repair_mode is None
    assert recovered_state.conflict_safe_mode_owner is None
    assert recovered_state.repair_state.stage == "recovered"
    assert recovered_state.last_compromise_template is None


def test_repair_ledger_entry_records_learning_and_conflict_context(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    controller.entropy_pool.ingest_bytes(bytes(range(256)) * 12, source="fixture_qrng", reason="repair ledger")
    _prime_conflict_state(controller)

    controller.tick(
        RoundEvent(
            source="user",
            content="I need a quick easy break, but help me plan carefully, reply to Alex, and let me drift.",
            target="alex",
            cue="break",
            valence=-0.40,
            energy_delta=-0.18,
        ),
        scenario="task",
        mode="interactive",
    )

    state = controller.load_runtime_state()
    latest = state.repair_ledger[-1]

    assert latest.conflict_score > 0.0
    assert latest.pass_count >= 1
    assert latest.resample_count >= 0
    assert isinstance(latest.dominant_conflicts, list)
    assert latest.action_delta_summary
    assert latest.learning_signal
    assert latest.stage_before in {"idle", "adjusting", "repairing", "cooling", "recovered"}


def test_deadlock_fuse_triggers_safe_mode_and_repair_mode(tmp_path):
    """§8.5: 3 consecutive critical conflicts trigger deadlock fuse -> safe_mode + repair_mode."""
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    _prime_conflict_state(controller, body_energy=0.06, budget_remaining=0.03, mood=0.15)

    critical_event = RoundEvent(
        source="user",
        content="I need a quick easy break, but help me plan carefully, reply to Alex, and let me drift.",
        target="alex",
        cue="break",
        valence=-0.45,
        energy_delta=-0.20,
    )

    for _ in range(3):
        result = controller.tick(critical_event, scenario="task", mode="interactive")

    fuse_state = controller.load_runtime_state()
    fuse_conflict = result.trace.conflict_arbitration

    assert fuse_conflict["deadlock_fuse_triggered"] is True
    assert fuse_state.safe_mode is True
    assert fuse_state.repair_mode == "deadlock_fuse"
    assert fuse_state.conflict_safe_mode_owner == "conflict"
    assert fuse_conflict["repair_transition"]["to_stage"] == "repairing"
    assert fuse_conflict["repair_state_snapshot"]["stage"] == "repairing"
    assert fuse_conflict["repair_ledger_tail"]

    blocked = fuse_conflict["circuit_breaker"]["blocked_actions"]
    for action in ("connect", "plan", "wander"):
        assert action in blocked, f"{action} should be blocked by deadlock fuse"


def test_high_conflict_records_shift_adjustment_and_ledger_entry(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    _prime_conflict_state(controller)

    result = controller.tick(
        RoundEvent(
            source="user",
            content="I need a quick easy break, but help me plan carefully, reply to Alex, and let me drift.",
            target="alex",
            cue="break",
            valence=-0.35,
            energy_delta=-0.16,
        ),
        scenario="task",
        mode="interactive",
    )

    state = controller.load_runtime_state()
    conflict = result.trace.conflict_arbitration
    adjustment = conflict["post_error_adjustment"]

    assert conflict["compromise"]["triggered"] is True
    assert adjustment["triggered"] is True
    assert adjustment["reason"] == "forced_compromise"
    assert adjustment["top_action_before"]
    assert adjustment["top_action_after"]
    assert conflict["repair_transition"]["to_stage"] == "adjusting"
    assert conflict["repair_state_snapshot"]["stage"] == "adjusting"
    assert state.repair_state.stage == "adjusting"
    assert state.last_post_error_adjustment.reason == "forced_compromise"
    assert len(state.repair_ledger) == 1


def test_high_conflict_appends_shift_repair_trace_entry(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    _prime_conflict_state(controller)

    controller.tick(
        RoundEvent(
            source="user",
            content="I need a quick easy break, but help me plan carefully, reply to Alex, and let me drift.",
            target="alex",
            cue="break",
            valence=-0.35,
            energy_delta=-0.16,
        ),
        scenario="task",
        mode="interactive",
    )

    repair_entries = controller.trace_store.list_repair_entries()

    assert len(repair_entries) == 1
    assert repair_entries[0]["round_id"] == 1
    assert repair_entries[0]["reason"] == "forced_compromise"
    assert repair_entries[0]["top_action_before"]
    assert repair_entries[0]["top_action_after"]
    assert repair_entries[0]["session_id"] == controller.load_runtime_state().session_id


def test_conflict_recovery_does_not_clear_non_conflict_safe_mode(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    _prime_conflict_state(controller)

    critical_event = RoundEvent(
        source="user",
        content="I need a quick easy break, but help me plan carefully, reply to Alex, and let me drift.",
        target="alex",
        cue="break",
        valence=-0.45,
        energy_delta=-0.20,
    )
    for _ in range(3):
        controller.tick(critical_event, scenario="task", mode="interactive")

    state = controller.load_runtime_state()
    state.conflict_hot_rounds = 1
    state.conflict_recovery_rounds = 1
    state.safe_mode = True
    state.conflict_safe_mode_owner = None
    state.repair_state.stage = "cooling"
    controller._save_state(state)

    controller.tick(
        RoundEvent(
            source="user",
            content="Please answer directly with one clear next step.",
            target="user",
            cue="step",
            valence=0.05,
            energy_delta=0.08,
        ),
        scenario="task",
        mode="interactive",
    )

    recovered = controller.load_runtime_state()
    assert recovered.conflict_hot_rounds == 0
    assert recovered.safe_mode is True
    assert recovered.conflict_safe_mode_owner is None
