from nalr.agents.registry import build_agent_registry
from nalr.skills.registry import build_skill_registry


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


def test_skill_registry_exposes_typed_skill_specs():
    registry = build_skill_registry()

    spec = registry["generate_candidates"]
    assert spec.owner_module == "PFCAgent"
    assert spec.skill_kind == "planning"
    assert spec.sync_mode == "sync"
    assert spec.cost_class == "H"
    assert spec.failure_policy == "fallback_to_rules"
    assert spec.policy_check is False
    assert spec.idempotent is True
    assert spec.output_kind == "candidate_actions"
    assert "proposal" in spec.trace_tags

    async_spec = registry["flush_trace_batch"]
    assert async_spec.sync_mode == "async"
    assert async_spec.skill_kind == "ops"
