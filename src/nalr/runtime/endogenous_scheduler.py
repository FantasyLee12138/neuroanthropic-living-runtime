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
        subjective = getattr(state, "subjective_state", None)
        body_state = getattr(state, "body_state", None)
        organic_mode = getattr(state, "organic_mode", None)
        spontaneous = _clip(float(getattr(subjective, "spontaneous", 0.0) or 0.0))
        reject_all = _clip(float(getattr(subjective, "reject_all", 0.0) or 0.0))
        meaning_density = _clip(len(list(getattr(subjective, "meaning_made", []) or [])) * 0.2)
        fatigue = _clip(float(getattr(state, "fatigue", getattr(body_state, "fatigue", 0.0)) or 0.0))
        fragments = _clip(float(getattr(state, "memory_fragments", getattr(body_state, "memory_fragments", 0.0)) or 0.0))
        continuity = _clip(float(getattr(state, "self_continuity", getattr(body_state, "self_continuity", 1.0)) or 0.0))
        continuity_drop = round(1.0 - continuity, 6)
        organic_gain = 1.0
        if organic_mode is not None and bool(getattr(organic_mode, "enabled", False)):
            organic_gain += max(
                0.0,
                (
                    float(getattr(organic_mode, "body_weight", 1.0) or 1.0)
                    + float(getattr(organic_mode, "subjective_weight", 1.0) or 1.0)
                )
                / 2.0
                - 1.0,
            ) * 0.18
        subjective_pressure = _clip(
            (
                spontaneous * 0.22
                + reject_all * 0.24
                + meaning_density * 0.16
                + fatigue * 0.15
                + fragments * 0.13
                + continuity_drop * 0.1
            )
            * organic_gain
        )

        trigger_specs = [
            (
                "field_imbalance",
                _clip(focus_lock / 6.0 + affect_residue * 0.8 + subjective_pressure * 0.28),
                "endogenous_regulation" if affect_residue >= 0.16 else "endogenous_light",
                "focus lock or affect residue remains elevated",
            ),
            (
                "motivation_sum_high",
                _clip(activation + subjective_pressure * 0.18),
                "endogenous_light",
                "motivation pool activation exceeds internal trigger threshold",
            ),
            (
                "silent_but_active",
                _clip(no_external_target * 0.25 + activation * 0.8 + memory_activation * 0.3 + subjective_pressure * 0.42),
                "endogenous_replay",
                "no fresh external cue but endogenous activation stays live",
            ),
            (
                "endogenous_wake",
                _clip(memory_activation * 0.55 + relation_risk * 0.45 + affect_residue * 0.4 + subjective_pressure * 0.24),
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
                "subjective_pressure": subjective_pressure,
                "spontaneous": spontaneous,
                "reject_all": reject_all,
                "meaning_density": meaning_density,
                "fatigue": fatigue,
                "memory_fragments": fragments,
                "continuity_drop": continuity_drop,
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
