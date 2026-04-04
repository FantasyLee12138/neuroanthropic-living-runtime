from pathlib import Path

from nalr.agents.modules import ConflictMonitorAgent
from nalr.runtime.controller import RuntimeController
from nalr.schemas.models import ActionDistributionState, ProposalBundle, RoundEvent


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


def test_request_resample_uses_v056_thresholds_and_compromise_gate():
    agent = ConflictMonitorAgent()

    low = agent.request_resample({"score": 0.40, "critical_conflict": False}, 0)
    medium = agent.request_resample({"score": 0.70, "critical_conflict": False}, 0)
    critical = agent.request_resample({"score": 0.91, "critical_conflict": True}, 1)
    exhausted = agent.request_resample({"score": 0.91, "critical_conflict": True}, 2)

    assert low["flag"] is False
    assert low["allowed_resamples"] == 0
    assert medium["flag"] is True
    assert medium["allowed_resamples"] == 1
    assert medium["force_compromise"] is False
    assert critical["flag"] is True
    assert critical["allowed_resamples"] == 2
    assert critical["force_compromise"] is True
    assert exhausted["flag"] is False
    assert exhausted["force_compromise"] is True


def test_runtime_sets_conflict_hot_after_three_critical_rounds_and_recovers(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
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
    assert recovered_state.last_compromise_template is None


def test_deadlock_fuse_triggers_safe_mode_and_repair_mode(tmp_path):
    """§8.5: 3 consecutive critical conflicts trigger deadlock fuse → safe_mode + repair_mode."""
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

    # Deadlock fuse should have triggered
    assert fuse_conflict["deadlock_fuse_triggered"] is True
    assert fuse_state.safe_mode is True
    assert fuse_state.repair_mode == "deadlock_fuse"

    # High-risk actions should be gated by circuit breaker
    blocked = fuse_conflict["circuit_breaker"]["blocked_actions"]
    for action in ("connect", "plan", "wander"):
        assert action in blocked, f"{action} should be blocked by deadlock fuse"
