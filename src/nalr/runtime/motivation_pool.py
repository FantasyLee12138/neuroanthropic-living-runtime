from __future__ import annotations

from dataclasses import asdict
from typing import Any

from nalr.schemas.models import (
    EndogenousMotivationSignal,
    MotivationLearningState,
    MotivationPoolState,
    ProbabilisticContribution,
)


DEFAULT_MOTIVATION_WEIGHTS: dict[str, float] = {
    "memory_exploration": 1.0,
    "behavior_exploration": 1.0,
    "affect_regulation": 1.0,
    "relation_calibration": 1.0,
    "internal_replay": 1.0,
}

MOTIVATION_ACTION_TARGETS: dict[str, dict[str, float]] = {
    "memory_exploration": {"recall": 0.24, "clarify": 0.08, "plan": 0.06},
    "behavior_exploration": {"plan": 0.08, "clarify": 0.12, "wander": 0.18},
    "affect_regulation": {"rest": 0.2, "respond": -0.05, "clarify": 0.04},
    "relation_calibration": {"connect": 0.12, "clarify": 0.09, "respond": 0.05},
    "internal_replay": {"recall": 0.12, "rest": 0.08, "wander": 0.1},
}


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


class EndogenousMotivationPool:
    def _weight_snapshot(self, learning_state: MotivationLearningState) -> dict[str, float]:
        snapshot = dict(DEFAULT_MOTIVATION_WEIGHTS)
        snapshot.update({key: max(0.1, float(value)) for key, value in learning_state.motivation_weights.items()})
        return snapshot

    def _signal(
        self,
        *,
        motivation_type: str,
        raw_drive: float,
        source_features: dict[str, float],
        state_tags: list[str],
        audit_reason: str,
    ) -> EndogenousMotivationSignal | None:
        if raw_drive <= 0.12:
            return None
        return EndogenousMotivationSignal(
            motivation_id=motivation_type,
            motivation_type=motivation_type,
            raw_drive=round(raw_drive, 6),
            source_features=source_features,
            target_actions=MOTIVATION_ACTION_TARGETS[motivation_type],
            state_tags=state_tags,
            audit_reason=audit_reason,
        )

    def evaluate(
        self,
        *,
        state,
        context: dict[str, Any],
        relation_state: dict[str, Any],
        slow_variables: dict[str, Any],
        long_run_projection: dict[str, Any],
    ) -> MotivationPoolState:
        weights = self._weight_snapshot(state.motivation_learning_state)
        recall_strength = float(context.get("recall_strength", 0.0) or 0.0)
        closeness = float(context.get("closeness", 0.5) or 0.5)
        interference = float(context.get("interference", 0.0) or 0.0)
        affect_residue = float(slow_variables.get("affect_residue", state.affect_residue) or 0.0)
        memory_activation = float(slow_variables.get("memory_activation", 0.0) or 0.0)
        relation_drift = float(slow_variables.get("relationship_drift", 0.0) or 0.0)
        scarcity = float(slow_variables.get("resource_scarcity", 0.0) or 0.0)
        continuity = float(long_run_projection.get("self_consistency_score", 0.5) or 0.5)
        focus_lock = float(state.focus_lock_count or 0.0)
        emotion_state = getattr(state, "emotion_state", None)
        desire_state = getattr(state, "desire_state", None)
        emotion_arousal = float(getattr(emotion_state, "arousal", 0.0) or 0.0)
        emotion_residue = float(getattr(emotion_state, "residue", affect_residue) or affect_residue)
        latent_drives = dict(getattr(desire_state, "latent_drives", {}) or {})
        comfort_drive = float(latent_drives.get("comfort", 0.0) or 0.0)
        meaning_drive = float(latent_drives.get("meaning", 0.0) or 0.0)
        relation_drive = float(latent_drives.get("relation", 0.0) or 0.0)
        exploration_drive = float(latent_drives.get("exploration", 0.0) or 0.0)
        completion_drive = float(latent_drives.get("completion", 0.0) or 0.0)

        active: list[EndogenousMotivationSignal] = []
        active_candidates = (
            self._signal(
                motivation_type="memory_exploration",
                raw_drive=_clip((0.62 - recall_strength) * 0.8 + memory_activation * 0.15 + interference * 0.3 + meaning_drive * 0.12 + exploration_drive * 0.1),
                source_features={
                    "recall_strength": recall_strength,
                    "memory_activation": memory_activation,
                    "interference": interference,
                    "meaning_drive": meaning_drive,
                    "exploration_drive": exploration_drive,
                },
                state_tags=["memory", "exploration"],
                audit_reason="memory prior weak or interference remains elevated",
            ),
            self._signal(
                motivation_type="behavior_exploration",
                raw_drive=_clip(focus_lock / 6.0 + max(0.0, 0.58 - continuity) * 0.45 + exploration_drive * 0.2 + emotion_arousal * 0.08),
                source_features={
                    "focus_lock_count": focus_lock,
                    "self_consistency_score": continuity,
                    "exploration_drive": exploration_drive,
                    "emotion_arousal": emotion_arousal,
                },
                state_tags=["action", "variation"],
                audit_reason="action field shows lock-in or low novelty tolerance",
            ),
            self._signal(
                motivation_type="affect_regulation",
                raw_drive=_clip(max(0.0, 0.52 - float(state.body_energy)) * 0.55 + affect_residue * 0.55 + emotion_residue * 0.18 + scarcity * 0.2 + comfort_drive * 0.22),
                source_features={
                    "body_energy": float(state.body_energy),
                    "affect_residue": affect_residue,
                    "resource_scarcity": scarcity,
                    "comfort_drive": comfort_drive,
                    "emotion_residue": emotion_residue,
                },
                state_tags=["affect", "repair"],
                audit_reason="vitality indicates recovery pressure",
            ),
            self._signal(
                motivation_type="relation_calibration",
                raw_drive=_clip(float(relation_state.get("relationship_risk", 0.0) or 0.0) * 0.6 + (1.0 - closeness) * 0.18 + relation_drive * 0.22),
                source_features={
                    "relationship_risk": float(relation_state.get("relationship_risk", 0.0) or 0.0),
                    "closeness": closeness,
                    "relation_drive": relation_drive,
                },
                state_tags=["relation", "boundary"],
                audit_reason="relation risk remains elevated and needs calibration",
            ),
            self._signal(
                motivation_type="internal_replay",
                raw_drive=_clip(interference * 0.45 + affect_residue * 0.25 + relation_drift * 0.25 + meaning_drive * 0.12 + completion_drive * 0.1 + max(0.0, 0.5 - continuity) * 0.14),
                source_features={
                    "interference": interference,
                    "affect_residue": affect_residue,
                    "relationship_drift": relation_drift,
                    "meaning_drive": meaning_drive,
                    "completion_drive": completion_drive,
                },
                state_tags=["replay", "internal"],
                audit_reason="residual conflict and replay pressure stay active",
            ),
        )
        for item in active_candidates:
            if item is not None:
                active.append(item)

        weighted_score = 0.0
        total_weight = 0.0
        for item in active:
            weight = weights.get(item.motivation_type, 1.0)
            weighted_score += item.raw_drive * weight
            total_weight += weight

        return MotivationPoolState(
            active_motivations=active,
            pool_weight_snapshot=weights,
            endogenous_activation_score=round(weighted_score / max(total_weight, 1.0), 6),
            last_feedback_update_at=state.motivation_pool_state.last_feedback_update_at,
        )

    def build_action_contribution(
        self,
        *,
        state,
        pool_state: MotivationPoolState,
    ) -> ProbabilisticContribution | None:
        if not pool_state.active_motivations:
            return None

        modulated_delta: dict[str, float] = {}
        dependency_trace: list[str] = []
        for item in pool_state.active_motivations:
            weight = float(pool_state.pool_weight_snapshot.get(item.motivation_type, 1.0) or 1.0)
            drive = item.raw_drive * min(weight, 1.5)
            dependency_trace.append(f"{item.motivation_type}:{round(drive, 4)}")
            for action, bias in item.target_actions.items():
                modulated_delta[action] = round(modulated_delta.get(action, 0.0) + bias * drive, 6)

        if not modulated_delta:
            return None

        return ProbabilisticContribution(
            module_name="EndogenousMotivationPool",
            module_type="motivation",
            level="action",
            target_space="action",
            raw_signal={item.motivation_type: item.raw_drive for item in pool_state.active_motivations},
            modulated_delta=dict(modulated_delta),
            confidence=round(_clip(0.35 + pool_state.endogenous_activation_score * 0.5), 4),
            trace_reason="endogenous motivations inject first-class action bias",
            projection_reason="motivation pool projected from endogenous drive state",
            applied_at_stage="endogenous_motivation_pool",
            native_operator="endogenous_bias",
            dependency_trace=dependency_trace + [f"round:{int(state.round_count)}"],
        )

    def trace_payload(self, pool_state: MotivationPoolState) -> dict[str, Any]:
        return asdict(pool_state)
