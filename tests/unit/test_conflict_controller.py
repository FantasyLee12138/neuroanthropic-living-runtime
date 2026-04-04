from pathlib import Path

from nalr.agents.modules import ConflictMonitorAgent
from nalr.runtime.controller import RuntimeController
from nalr.schemas.models import ActionDistributionState, ConflictRepairState, ProposalBundle, RoundEvent


CONFIG_ROOT = Path(__file__).resolve().parents[2] / "config"


def _proposal(
    owner: str,
    *,
    confidence: float,
    delta_p: dict[str, float],
    priority_bucket: str,
    control_domain: str,
    veto: bool = False,
    gated_actions: list[str] | None = None,
    risk_hints: dict | None = None,
) -> ProposalBundle:
    bundle = ProposalBundle(
        owner=owner,
        confidence=confidence,
        action_preferences=dict(delta_p),
        delta_p=dict(delta_p),
        veto=veto,
        reason=f"{owner} proposal",
    )
    bundle.priority_bucket = priority_bucket
    bundle.control_domain = control_domain
    bundle.gated_actions = list(gated_actions or [])
    bundle.risk_hints = dict(risk_hints or {})
    return bundle


def _distribution(p_raw: dict[str, float]) -> ActionDistributionState:
    return ActionDistributionState(
        p_base={action: 0.05 for action in p_raw},
        p_raw=dict(p_raw),
        p_mix={},
        p_final={},
        ci={action: 0.2 for action in p_raw},
        gate={action: 1.0 for action in p_raw},
        risk_suppressor={action: 1.0 for action in p_raw},
        resample_idx=0,
    )


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
    proposals = [
        _proposal(
            "BodyStateAgent",
            confidence=0.96,
            delta_p={"rest": 0.32, "respond": 0.04},
            priority_bucket="body_safety",
            control_domain="body",
            gated_actions=["connect", "plan"],
            risk_hints={"body_load": 0.95},
        ),
        _proposal(
            "ResourceAgent",
            confidence=0.90,
            delta_p={"rest": 0.18, "respond": 0.06},
            priority_bucket="budget_overload",
            control_domain="resource",
            gated_actions=["plan"],
            risk_hints={"overload": 0.88},
        ),
        _proposal(
            "PFCAgent",
            confidence=0.91,
            delta_p={"plan": 0.30, "respond": 0.08},
            priority_bucket="task_goal",
            control_domain="task",
            risk_hints={"goal_pressure": 0.72},
        ),
        _proposal(
            "RelationshipAgent",
            confidence=0.87,
            delta_p={"connect": 0.24, "clarify": 0.10},
            priority_bucket="relation_boundary",
            control_domain="relation",
            gated_actions=["connect"],
            risk_hints={"relationship_risk": 0.83},
        ),
        _proposal(
            "DesireAgent",
            confidence=0.82,
            delta_p={"rest": 0.22, "wander": 0.12},
            priority_bucket="immediate_desire",
            control_domain="desire",
            risk_hints={"comfort_pull": 0.76},
        ),
        _proposal(
            "DMNAgent",
            confidence=0.78,
            delta_p={"wander": 0.28},
            priority_bucket="roaming",
            control_domain="dmn",
            risk_hints={"roam_pull": 0.94},
        ),
    ]
    distribution_state = _distribution(
        {
            "rest": 0.31,
            "plan": 0.27,
            "connect": 0.17,
            "wander": 0.15,
            "respond": 0.10,
        }
    )

    assessment = agent.score_conflict(proposals, distribution_state)

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
    proposals = [
        _proposal(
            "BodyStateAgent",
            confidence=0.97,
            delta_p={"rest": 0.34},
            priority_bucket="body_safety",
            control_domain="body",
            gated_actions=["plan", "connect"],
            risk_hints={"body_load": 0.96},
        ),
        _proposal(
            "PFCAgent",
            confidence=0.90,
            delta_p={"plan": 0.28},
            priority_bucket="task_goal",
            control_domain="task",
            risk_hints={"goal_pressure": 0.85},
        ),
        _proposal(
            "DMNAgent",
            confidence=0.88,
            delta_p={"wander": 0.26},
            priority_bucket="roaming",
            control_domain="dmn",
            risk_hints={"roam_pull": 0.93},
        ),
    ]
    distribution_state = _distribution(
        {
            "rest": 0.33,
            "plan": 0.31,
            "wander": 0.24,
            "respond": 0.12,
        }
    )
    assessment = agent.score_conflict(proposals, distribution_state)

    resolution = agent.trigger_control_escalation(assessment, distribution_state, 0)

    assert resolution["winning_priority"] == "body_safety"
    assert "plan" in resolution["blocked_actions"]
    assert "wander" in resolution["blocked_actions"]
    assert resolution["applied_template"] in {None, "body_first"}


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
    hot_conflict = result.trace.distribution_state["conflict"]

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
    fuse_conflict = result.trace.distribution_state["conflict"]

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
    conflict = result.trace.distribution_state["conflict"]
    adjustment = conflict["post_error_adjustment"]

    assert conflict["compromise"]["triggered"] is False
    assert adjustment["triggered"] is True
    assert adjustment["reason"] == "top_action_shift"
    assert adjustment["top_action_before"] != adjustment["top_action_after"]
    assert conflict["repair_transition"]["to_stage"] == "adjusting"
    assert conflict["repair_state_snapshot"]["stage"] == "adjusting"
    assert state.repair_state.stage == "adjusting"
    assert state.last_post_error_adjustment.reason == "top_action_shift"
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
    assert repair_entries[0]["reason"] == "top_action_shift"
    assert repair_entries[0]["top_action_before"] != repair_entries[0]["top_action_after"]
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
