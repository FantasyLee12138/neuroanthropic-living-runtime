from pathlib import Path

from nalr.runtime.controller import RuntimeController
from nalr.schemas.models import RoundEvent


CONFIG_ROOT = Path(__file__).resolve().parents[2] / "config"


def test_new_agents_emit_stage_level_proposals(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    result = controller.tick(
        RoundEvent(
            source="user",
            content="I want a quick easy coffee break, but also help me explain the plan clearly.",
            target="user",
            cue="coffee",
            valence=-0.45,
            energy_delta=-0.10,
        ),
        scenario="task",
        mode="interactive",
    )

    by_stage = {item["stage"]: item for item in result.trace.proposal_summaries}
    assert by_stage["salience"]["top_action"] is not None
    assert by_stage["emotion"]["top_action"] is not None
    assert by_stage["desire"]["top_action"] is not None
    assert by_stage["value"]["top_action"] is not None


def test_salience_stage_promotes_non_fallback_action_for_clearly_salient_input(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    result = controller.tick(
        RoundEvent(
            source="user",
            content="Help me check the code files, remember the failing test, and summarize the plan.",
            target="user",
            cue="tests",
            valence=0.1,
        ),
        scenario="task",
        mode="interactive",
    )

    by_stage = {item["stage"]: item for item in result.trace.proposal_summaries}

    assert by_stage["salience"]["top_action"] == "plan"
    assert result.sampled_action.name in {"plan", "recall", "clarify"}


def test_plausibility_guard_blocks_wander_in_task_runtime(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    for agent_name in [
        "BodyStateAgent",
        "ResourceAgent",
        "PFCAgent",
        "RelationshipAgent",
        "SalienceAgent",
        "ValueAgent",
        "HabitAgent",
        "HippocampusAgent",
    ]:
        controller.apply_command(f"agent disable {agent_name}")

    result = controller.tick(
        RoundEvent(
            source="user",
            content="drift soft",
            valence=0.0,
        ),
        scenario="task",
        mode="idle",
    )

    assert result.sampled_action.name != "wander"
    assert any(
        item["stage"] == "plausibility_guard" and item["requires_resample"] is True and "wander" in item["reason"]
        for item in result.trace.gate_decisions
    )
    assert result.trace.conflict_arbitration["total_score"] >= 0.0
