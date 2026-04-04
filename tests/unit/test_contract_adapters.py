from dataclasses import dataclass, field

from nalr.runtime.adapters import adapt_proposal, adapt_skill_spec


@dataclass
class _ThinProposal:
    agent_name: str
    action_preferences: dict[str, float]
    confidence: float
    sigma_scale: float = 1.0
    veto: bool = False
    trace_tags: list[str] = field(default_factory=list)
    reason: str = ""


@dataclass
class _ThinSkill:
    name: str
    owner_module: str
    input_schema: dict[str, str]
    output_schema: dict[str, str]
    timeout_ms: int
    cost_class: str
    failure_policy: str
    trace_tags: list[str]


def test_adapt_proposal_supplies_runtime_defaults_for_upstream_contract():
    proposal = _ThinProposal(
        agent_name="ThinAgent",
        action_preferences={"respond": 0.4, "plan": 0.2},
        confidence=0.61,
        trace_tags=["thin"],
        reason="upstream contract",
    )

    adapted = adapt_proposal(proposal, weight=0.5, resample_idx=2)

    assert adapted["provider"] == "upstream_contract"
    assert adapted["model"] == "pending-merge"
    assert adapted["latency_ms"] == 0
    assert adapted["top_action"] == "respond"
    assert adapted["top_score"] == 0.2
    assert adapted["resample_idx"] == 2


def test_adapt_skill_spec_exposes_registry_metadata_for_cil():
    spec = _ThinSkill(
        name="generate_candidates",
        owner_module="PFCAgent",
        input_schema={"event": "RoundEvent"},
        output_schema={"action_preferences": "dict[str, float]"},
        timeout_ms=300,
        cost_class="H",
        failure_policy="fallback_to_rules",
        trace_tags=["proposal", "pfc"],
    )

    adapted = adapt_skill_spec(spec)

    assert adapted["name"] == "generate_candidates"
    assert adapted["owner_module"] == "PFCAgent"
    assert adapted["trace_tags"] == ["proposal", "pfc"]
    assert adapted["timeout_ms"] == 300
