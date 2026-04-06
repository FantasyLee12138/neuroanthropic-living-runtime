from __future__ import annotations

from dataclasses import asdict
from typing import Any

from nalr.schemas.models import MotivationFeedbackRecord, MotivationLearningState, MotivationPoolState


def _clip(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


class MotivationFeedbackUpdater:
    def update(
        self,
        *,
        learning_state: MotivationLearningState,
        pool_state: MotivationPoolState,
        trace_payload: dict[str, Any],
    ) -> tuple[MotivationLearningState, dict[str, Any]]:
        active = list(pool_state.active_motivations)
        if not active:
            empty_summary = {
                "reward_signal": 0.0,
                "records": [],
                "endogenous_policy_shift": dict(learning_state.endogenous_policy_shift),
            }
            return learning_state, empty_summary

        vitality = dict(trace_payload.get("vitality_snapshot", {}) or {})
        authenticity = dict(trace_payload.get("authenticity", {}) or {})
        sampled_action = str(trace_payload.get("sampled_action") or "")
        round_id = str(trace_payload.get("round_id") or "")
        memory_activation = float(vitality.get("memory_activation", 0.0) or 0.0)
        relation_drift = float(vitality.get("relationship_drift", 0.0) or 0.0)
        affect_residue = float(vitality.get("affect_residue", 0.0) or 0.0)
        body_energy = float(vitality.get("body_energy", 0.0) or 0.0)
        grounding = float(authenticity.get("self_grounding_score", 0.0) or 0.0)
        penalty = float(authenticity.get("sampling_penalty_applied", 0.0) or 0.0)

        reward_signal = _clip(
            memory_activation * 0.28
            + max(0.0, grounding - 0.45) * 0.35
            + body_energy * 0.08
            - relation_drift * 0.22
            - affect_residue * 0.16
            - penalty * 0.25
        )

        next_weights = dict(learning_state.motivation_weights)
        policy_shift = dict(learning_state.endogenous_policy_shift)
        records = list(learning_state.recent_feedback)
        new_records: list[dict[str, Any]] = []
        for item in active:
            current_weight = float(next_weights.get(item.motivation_type, 1.0) or 1.0)
            updated_weight = max(0.1, min(2.0, current_weight + reward_signal * 0.12))
            next_weights[item.motivation_type] = round(updated_weight, 6)
            for action, bias in item.target_actions.items():
                policy_shift[action] = round(float(policy_shift.get(action, 0.0) or 0.0) + bias * reward_signal * 0.5, 6)

            record = MotivationFeedbackRecord(
                round_id=round_id,
                motivation_id=item.motivation_id,
                sampled_action=sampled_action,
                user_response=None,
                response_latency_ms=None,
                affect_delta={
                    "affect_residue": round(-affect_residue, 6),
                    "body_energy": round(body_energy, 6),
                },
                memory_activation_delta=memory_activation,
                relation_delta=round(-relation_drift, 6),
                reward_signal=reward_signal,
                update_reason=f"memory={memory_activation:.2f}; grounding={grounding:.2f}; penalty={penalty:.2f}",
            )
            records.append(record)
            new_records.append(asdict(record))

        next_state = MotivationLearningState(
            motivation_weights=next_weights,
            recent_feedback=records[-20:],
            endogenous_policy_shift=policy_shift,
        )
        return next_state, {
            "reward_signal": round(reward_signal, 6),
            "records": new_records,
            "endogenous_policy_shift": dict(next_state.endogenous_policy_shift),
        }
