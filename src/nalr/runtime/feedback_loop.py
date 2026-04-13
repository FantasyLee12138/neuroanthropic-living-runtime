from __future__ import annotations

from typing import Any


def _clip_unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


class FeedbackLoopRuntime:
    def evaluate(
        self,
        *,
        trace_payload: dict[str, Any],
        control_events: list[dict[str, Any]] | None = None,
        layer_fuses: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        rendered_expression = dict(trace_payload.get("rendered_expression", {}) or {})
        motivation_feedback = dict(trace_payload.get("motivation_feedback", {}) or {})
        memory_write_gate = dict(trace_payload.get("memory_write_gate", {}) or {})
        state_delta = dict(trace_payload.get("state_delta_after_clip", {}) or {})
        cause_type = str(trace_payload.get("cause_type") or "")
        sampled_action = str(trace_payload.get("sampled_action") or "")
        output_text = str(rendered_expression.get("text") or "")
        degraded = bool(rendered_expression.get("degraded", False))
        events = [item for item in list(control_events or []) if isinstance(item, dict)]
        fuse_payload = dict(layer_fuses or {})

        behavior_effectiveness = 1.0 if output_text or sampled_action == "nothing" else 0.0
        if degraded and not output_text:
            behavior_effectiveness = 0.0
        user_feedback_rate = 1.0 if cause_type == "external_stimulus" else 0.0
        state_correction_magnitude = round(sum(abs(float(value or 0.0)) for value in state_delta.values()), 4)
        feedback_reward = round(
            float(motivation_feedback.get("reward", motivation_feedback.get("reward_signal", 0.0)) or 0.0),
            4,
        )

        fuse_rows = [item for item in fuse_payload.values() if isinstance(item, dict)]
        fuse_hits = sum(
            1
            for item in fuse_rows
            if bool(item.get("muted", False)) or float(item.get("throttle", 1.0) or 1.0) < 1.0 or str(item.get("mode") or "normal") != "normal"
        )
        fuse_hit_rate = round(fuse_hits / max(len(fuse_rows), 1), 4)

        memory_gate_reason = str(memory_write_gate.get("reason") or "")
        memory_writeback_eligible = not bool(memory_write_gate.get("suppressed", False)) and (
            behavior_effectiveness > 0.0 or abs(feedback_reward) >= 0.1 or state_correction_magnitude > 0.0
        )

        return {
            "feedback_metrics": {
                "behavior_effectiveness": round(_clip_unit(behavior_effectiveness), 4),
                "user_feedback_rate": round(_clip_unit(user_feedback_rate), 4),
                "state_correction_magnitude": state_correction_magnitude,
                "feedback_reward": feedback_reward,
                "fuse_hit_rate": fuse_hit_rate,
            },
            "memory_writeback": {
                "eligible": memory_writeback_eligible,
                "reason": memory_gate_reason or ("feedback_loop_ok" if memory_writeback_eligible else "insufficient_feedback_signal"),
            },
            "control_summary": {
                "control_event_count": len(events),
                "fuse_hits": fuse_hits,
                "latest_event_type": str(events[-1].get("event_type") or "") if events else "",
            },
        }
