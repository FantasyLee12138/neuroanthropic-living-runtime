from __future__ import annotations

from dataclasses import asdict
from typing import Any

from nalr.schemas.models import EndogenousSchedulerState, EndogenousTickTrigger, MotivationPoolState


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


class EndogenousTickScheduler:
    def build_trigger(
        self,
        *,
        state,
        context: dict[str, Any],
        relation_state: dict[str, Any],
        slow_variables: dict[str, Any],
        pool_state: MotivationPoolState,
    ) -> EndogenousTickTrigger | None:
        focus_lock = float(state.focus_lock_count or 0.0)
        affect_residue = float(slow_variables.get("affect_residue", state.affect_residue) or 0.0)
        memory_activation = float(slow_variables.get("memory_activation", 0.0) or 0.0)
        relation_risk = float(relation_state.get("relationship_risk", 0.0) or 0.0)
        activation = float(pool_state.endogenous_activation_score or 0.0)
        no_external_target = 1.0 if not context.get("cue") else 0.0

        trigger_specs = [
            (
                "field_imbalance",
                _clip(focus_lock / 6.0 + affect_residue * 0.8),
                "endogenous_regulation" if affect_residue >= 0.16 else "endogenous_light",
                "focus lock or affect residue remains elevated",
            ),
            (
                "motivation_sum_high",
                activation,
                "endogenous_light",
                "motivation pool activation exceeds internal trigger threshold",
            ),
            (
                "silent_but_active",
                _clip(no_external_target * 0.25 + activation * 0.8 + memory_activation * 0.3),
                "endogenous_replay",
                "no fresh external cue but endogenous activation stays live",
            ),
            (
                "endogenous_wake",
                _clip(memory_activation * 0.55 + relation_risk * 0.45 + affect_residue * 0.4),
                "endogenous_replay" if memory_activation >= relation_risk else "endogenous_regulation",
                "memory burst or relation pressure reactivates internal processing",
            ),
        ]
        best_type, best_score, selected_mode, reason = max(trigger_specs, key=lambda item: item[1])
        if best_score < 0.18:
            return None
        return EndogenousTickTrigger(
            trigger_type=best_type,
            trigger_score=round(best_score, 6),
            source_metrics={
                "focus_lock_count": focus_lock,
                "affect_residue": affect_residue,
                "memory_activation": memory_activation,
                "relationship_risk": relation_risk,
                "motivation_activation": activation,
            },
            selected_mode=selected_mode,
            audit_reason=reason,
        )

    def should_trigger(
        self,
        *,
        state,
        context: dict[str, Any],
        relation_state: dict[str, Any],
        slow_variables: dict[str, Any],
        pool_state: MotivationPoolState,
    ) -> bool:
        return self.build_trigger(
            state=state,
            context=context,
            relation_state=relation_state,
            slow_variables=slow_variables,
            pool_state=pool_state,
        ) is not None

    def update_state(
        self,
        *,
        scheduler_state: EndogenousSchedulerState,
        trigger: EndogenousTickTrigger | None,
        recorded_at: str | None = None,
        suppression_reason: str | None = None,
    ) -> EndogenousSchedulerState:
        recent = list(scheduler_state.recent_triggers)
        if trigger is not None:
            recent.append(trigger)
        return EndogenousSchedulerState(
            last_endogenous_tick_at=recorded_at or scheduler_state.last_endogenous_tick_at,
            recent_triggers=recent[-20:],
            suppression_reason=suppression_reason,
        )

    def trace_payload(self, scheduler_state: EndogenousSchedulerState, trigger: EndogenousTickTrigger | None) -> dict[str, Any]:
        payload = asdict(scheduler_state)
        payload["last_trigger"] = asdict(trigger) if trigger is not None else {}
        return payload
