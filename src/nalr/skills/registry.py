from __future__ import annotations

from nalr.schemas.models import SkillSpec


def build_skill_registry() -> dict[str, SkillSpec]:
    return {
        "compute_body_bias": SkillSpec(
            name="compute_body_bias",
            owner_module="BodyStateAgent",
            input_schema={"body_energy": "float", "event": "RoundEvent"},
            output_schema={"action_preferences": "dict[str, float]"},
            timeout_ms=20,
            cost_class="L",
            failure_policy="neutral_body_bias",
            trace_tags=["proposal", "body"],
        ),
        "compute_scarcity_index": SkillSpec(
            name="compute_scarcity_index",
            owner_module="ResourceAgent",
            input_schema={"budget_remaining": "float"},
            output_schema={"action_preferences": "dict[str, float]"},
            timeout_ms=20,
            cost_class="L",
            failure_policy="stable_mode_budget",
            trace_tags=["proposal", "resource"],
        ),
        "generate_candidates": SkillSpec(
            name="generate_candidates",
            owner_module="PFCAgent",
            input_schema={"scenario": "str", "event": "RoundEvent"},
            output_schema={"action_preferences": "dict[str, float]"},
            timeout_ms=300,
            cost_class="H",
            failure_policy="rule_fallback_planner",
            trace_tags=["proposal", "pfc"],
        ),
        "aggregate_proposals": SkillSpec(
            name="aggregate_proposals",
            owner_module="ThalamusAttentionAgent",
            input_schema={"proposals": "list[Proposal]"},
            output_schema={"sampled_action": "ActionCandidate"},
            timeout_ms=40,
            cost_class="M",
            failure_policy="baseline_sampler",
            trace_tags=["sampling"],
        ),
    }

