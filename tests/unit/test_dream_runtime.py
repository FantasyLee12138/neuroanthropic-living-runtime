from pathlib import Path

from nalr.runtime.controller import RuntimeController
from nalr.schemas.models import RoundEvent


CONFIG_ROOT = Path(__file__).resolve().parents[2] / "config"


def test_idle_and_sleep_rounds_record_dream_artifacts_and_effects(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    controller.tick(
        RoundEvent(
            source="user",
            content="Remember that tea helps me slow down before sleep.",
            target="user",
            cue="tea",
            valence=0.25,
        ),
        scenario="companion",
        mode="interactive",
    )

    idle_result = controller.tick(
        RoundEvent(
            source="system",
            content="Background idle reshaping pass.",
            target="user",
            cue="tea",
            valence=0.0,
        ),
        scenario="companion",
        mode="idle",
    )
    sleep_result = controller.tick(
        RoundEvent(
            source="system",
            content="Night sleep consolidation pass.",
            target="user",
            cue="tea",
            valence=0.0,
        ),
        scenario="companion",
        mode="sleep",
    )

    idle_trace = controller.trace_round(idle_result.round_id)
    sleep_trace = controller.trace_round(sleep_result.round_id)
    dream_metrics = controller.dream_metrics()

    assert idle_trace["dream_run_id"]
    assert idle_trace["dream_trigger"] == "idle_light"
    assert idle_trace["dream_trace_ref"].startswith("dream://")
    assert idle_trace["dream_guard_summary"]["approved"] is True
    assert set(idle_trace["dream_guard_summary"]["allowed_types"]) == {
        "memory_consolidation",
        "emotion_adjustments",
        "habit_adjustments",
    }
    assert "limited_identity_drift" not in idle_trace["dream_effect_summary"]["applied_types"]

    assert sleep_trace["dream_run_id"]
    assert sleep_trace["dream_trigger"] == "sleep_full"
    assert sleep_trace["dream_guard_summary"]["approved"] is True
    assert "limited_identity_drift" in sleep_trace["dream_guard_summary"]["evaluated_types"]
    assert "limited_identity_drift" in sleep_trace["dream_effect_summary"]["applied_types"]

    assert dream_metrics["total_runs"] == 2
    assert dream_metrics["runs_by_trigger"]["idle_light"] == 1
    assert dream_metrics["runs_by_trigger"]["sleep_full"] == 1
    assert dream_metrics["identity_proposal_approvals"] >= 1


def test_state_payload_and_why_surface_dream_summary(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    result = controller.tick(
        RoundEvent(
            source="system",
            content="Sleep reshape around noodles.",
            target="user",
            cue="noodles",
            valence=0.0,
        ),
        scenario="chat",
        mode="sleep",
    )

    state_payload = controller.state_payload()
    why_payload = controller.why_this(result.round_id)

    assert state_payload["dream"]["enabled"] is True
    assert state_payload["dream"]["latest_run_id"] == why_payload["dream"]["run_id"]
    assert why_payload["dream"]["trigger"] == "sleep_full"
    assert why_payload["dream"]["guard_summary"]["approved"] is True
