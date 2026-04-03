import pytest

from nalr.agents.registry import build_agent_registry
from nalr.schemas.models import CircuitBreakerPolicy, FallbackRoute, SkillPermissionProfile, SkillSpec
from nalr.skills.registry import build_skill_registry, serialize_contract


def test_agent_registry_exposes_v056_contracts():
    registry = build_agent_registry()

    assert {
        "BodyStateAgent",
        "RelationshipAgent",
        "DesireAgent",
        "EmotionAgent",
        "DMNAgent",
        "PFCAgent",
        "UnconsciousAgent",
        "HippocampusAgent",
        "ThalamusAttentionAgent",
        "SalienceAgent",
        "HabitAgent",
        "ValueAgent",
        "ConflictMonitorAgent",
        "ResourceAgent",
        "CerebellarPredictor",
        "PerspectiveModel",
        "BehaviorPlausibilityGuard",
        "ForcedModeSwitch",
        "OutputGate",
    }.issubset(registry.keys())

    spec = registry["PFCAgent"]
    assert spec.class_kind == "explicit"
    assert spec.wakeup_rule == "task_or_explanation_or_execution"
    assert "generate_candidates" in spec.owned_skills
    assert spec.budget_class == "H"
    assert spec.fallback_policy == "fallback_to_rules"

    conflict_spec = registry["ConflictMonitorAgent"]
    assert conflict_spec.owned_skills == [
        "score_conflict",
        "trigger_control_escalation",
        "request_resample",
        "mark_post_error_adjustment",
    ]

    plausibility_spec = registry["BehaviorPlausibilityGuard"]
    assert "check_behavior_plausibility" in plausibility_spec.owned_skills


def test_skill_registry_exposes_typed_skill_specs():
    registry = build_skill_registry()

    spec = registry["generate_candidates"]
    assert spec.owner_module == "PFCAgent"
    assert spec.skill_kind == "planning"
    assert spec.sync_mode == "sync"
    assert spec.cost_class == "H"
    assert spec.failure_policy == "fallback_to_rules"
    assert spec.policy_check is True
    assert spec.idempotent is True
    assert spec.output_kind == "candidate_actions"
    assert "proposal" in spec.trace_tags
    assert spec.permission.external_io is True
    assert spec.fallback_route is not None
    assert spec.fallback_route.strategy == "fallback_to_rules"
    assert spec.fallback_route.cost_class == "L"
    assert spec.breaker_policy is not None
    assert spec.breaker_policy.failure_threshold == 3
    assert spec.breaker_policy.cooldown_rounds == 5
    assert serialize_contract(spec.input_schema)["event"] == "RoundEvent"
    assert serialize_contract(spec.output_schema) == "ProposalBundle"

    async_spec = registry["flush_trace_batch"]
    assert async_spec.sync_mode == "async"
    assert async_spec.skill_kind == "ops"
    assert async_spec.permission.scope == "trace_append"

    repair_spec = registry["mark_post_error_adjustment"]
    assert repair_spec.owner_module == "ConflictMonitorAgent"
    assert repair_spec.skill_kind == "guarding"
    assert repair_spec.output_kind == "flag"
    assert serialize_contract(repair_spec.input_schema)["state"] == "RuntimeState"
    assert serialize_contract(repair_spec.output_schema)["repair_mode"] == "optional[str]"


def test_skill_spec_rejects_invalid_runtime_metadata():
    with pytest.raises(ValueError, match="cost_class"):
        SkillSpec(
            name="bad_cost",
            owner_module="demo",
            input_schema={"value": int},
            output_schema={"ok": bool},
            timeout_ms=10,
            cost_class="Z",
            failure_policy="return_neutral",
            trace_tags=["demo"],
        )

    with pytest.raises(ValueError, match="fallback route"):
        SkillSpec(
            name="bad_breaker",
            owner_module="demo",
            input_schema={"value": int},
            output_schema={"ok": bool},
            timeout_ms=10,
            cost_class="H",
            failure_policy="fallback_to_rules",
            trace_tags=["demo"],
            policy_check=True,
            permission=SkillPermissionProfile(external_io=True),
            breaker_policy=CircuitBreakerPolicy(failure_threshold=3, cooldown_rounds=5),
        )

    with pytest.raises(ValueError, match="policy_check"):
        SkillSpec(
            name="bad_policy",
            owner_module="demo",
            input_schema={"value": int},
            output_schema={"ok": bool},
            timeout_ms=10,
            cost_class="L",
            failure_policy="return_neutral",
            trace_tags=["demo"],
            permission=SkillPermissionProfile(external_io=True),
        )

    valid = SkillSpec(
        name="good",
        owner_module="demo",
        input_schema={"value": int},
        output_schema={"ok": bool},
        timeout_ms=10,
        cost_class="H",
        failure_policy="fallback_to_rules",
        trace_tags=["Demo", " demo "],
        policy_check=True,
        permission=SkillPermissionProfile(external_io=True),
        fallback_route=FallbackRoute(strategy="fallback_to_rules", target="demo.fallback", cost_class="L"),
        breaker_policy=CircuitBreakerPolicy(failure_threshold=3, cooldown_rounds=5),
    )
    assert valid.trace_tags == ["demo"]
