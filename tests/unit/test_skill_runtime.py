import json
from pathlib import Path

from nalr.runtime.controller import RuntimeController
from nalr.schemas.models import RoundEvent
from nalr.skills.executor import SkillExecutor
from nalr.skills.registry import build_skill_registry


CONFIG_ROOT = Path(__file__).resolve().parents[2] / "config"


def test_skill_executor_applies_fallback_policy():
    executor = SkillExecutor(build_skill_registry())

    result = executor.execute(
        round_id=1,
        skill_name="generate_candidates",
        inputs={"scenario": "task"},
        provider=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    assert result.degraded is True
    assert result.failure_policy_applied == "fallback_to_rules"
    assert result.output


def test_skill_executor_rejects_invalid_output_schema():
    executor = SkillExecutor(build_skill_registry())

    result = executor.execute(
        round_id=1,
        skill_name="generate_candidates",
        inputs={"scenario": "task", "event": {"content": "plan"}},
        provider=lambda: {"unexpected": "shape"},
    )

    assert result.degraded is True
    assert result.failure_policy_applied == "output_validation_failed"


def test_skill_executor_trips_circuit_breaker_after_repeated_failures():
    executor = SkillExecutor(build_skill_registry())

    for _ in range(3):
        result = executor.execute(
            round_id=1,
            skill_name="generate_candidates",
            inputs={},
            provider=lambda: {"unexpected": "shape"},
        )
        assert result.degraded is True

    breaker_result = executor.execute(
        round_id=2,
        skill_name="generate_candidates",
        inputs={"scenario": "task", "event": {"content": "plan"}},
        provider=lambda: {"action_preferences": {"plan": 0.4}},
    )

    assert breaker_result.degraded is True
    assert breaker_result.failure_policy_applied == "trip_circuit_breaker"


def test_tick_records_skill_level_trace_and_files(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    result = controller.tick(
        RoundEvent(
            source="user",
            content="Help me remember tea and plan tomorrow.",
            target="user",
            cue="tea",
            valence=0.2,
        ),
        scenario="task",
        mode="interactive",
    )

    assert result.trace.skill_traces
    skills_path = tmp_path / ".alive" / "traces" / "skills" / "skill_traces.jsonl"
    assert skills_path.exists()
    rows = [json.loads(line) for line in skills_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert any(row["round_id"] == result.round_id for row in rows)
