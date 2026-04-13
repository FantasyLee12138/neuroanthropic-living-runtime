from __future__ import annotations

from nalr.runtime.feedback_loop import FeedbackLoopRuntime


def test_feedback_loop_runtime_derives_metrics_and_memory_writeback_gate():
    runtime = FeedbackLoopRuntime()

    payload = runtime.evaluate(
        trace_payload={
            "round_id": 7,
            "cause_type": "external_stimulus",
            "sampled_action": "respond",
            "rendered_expression": {"text": "Tea sounds good.", "degraded": False},
            "state_delta_after_clip": {"mood": 0.12, "body_energy": -0.05},
            "motivation_feedback": {"reward_signal": 0.28},
            "memory_write_gate": {"suppressed": False},
        },
        control_events=[{"event_type": "proposal_applied"}],
        layer_fuses={"feedback": {"mode": "normal", "throttle": 1.0, "muted": False}},
    )

    assert payload["feedback_metrics"]["behavior_effectiveness"] == 1.0
    assert payload["feedback_metrics"]["user_feedback_rate"] == 1.0
    assert payload["feedback_metrics"]["feedback_reward"] == 0.28
    assert payload["memory_writeback"]["eligible"] is True
    assert payload["control_summary"]["control_event_count"] == 1


def test_feedback_loop_runtime_counts_fuse_hits_and_respects_suppressed_memory_gate():
    runtime = FeedbackLoopRuntime()

    payload = runtime.evaluate(
        trace_payload={
            "round_id": 8,
            "cause_type": "endogenous",
            "sampled_action": "monologue",
            "rendered_expression": {"text": "", "degraded": True},
            "state_delta_after_clip": {"mood": 0.0},
            "motivation_feedback": {"reward_signal": -0.12},
            "memory_write_gate": {"suppressed": True, "reason": "pollution_guard"},
        },
        control_events=[{"event_type": "fuse_engaged"}, {"event_type": "alert_triggered"}],
        layer_fuses={
            "feedback": {"mode": "degraded", "throttle": 0.4, "muted": False},
            "cognition": {"mode": "normal", "throttle": 1.0, "muted": False},
        },
    )

    assert payload["feedback_metrics"]["behavior_effectiveness"] == 0.0
    assert payload["feedback_metrics"]["fuse_hit_rate"] == 0.5
    assert payload["memory_writeback"]["eligible"] is False
    assert payload["memory_writeback"]["reason"] == "pollution_guard"
