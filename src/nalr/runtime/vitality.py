from __future__ import annotations

from typing import Any

from nalr.schemas.models import ActionCandidate, AgentContribution, RuntimeState


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


class VitalityEngine:
    def apply_noninteractive_shaping(
        self,
        state: RuntimeState,
        requested_mode: str,
        cue: str | None,
        memory_store,
        *,
        proposal: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if requested_mode not in {"idle", "sleep"}:
            return []

        if proposal is None:
            shaping_events = [memory_store.shape_noninteractive(mode=requested_mode, cue=cue)]
        else:
            shaping_events = [memory_store.apply_noninteractive_proposal(proposal)]
        previous_focus = state.focus
        resource_scarcity = float(state.resource_state.get("scarcity_index", _clip(1.0 - state.budget_remaining)))

        if requested_mode == "idle":
            state.focus = "wander" if state.focus != "rest" else state.focus
            state.mood = round(_clip(state.mood * 0.97 + (0.52 - state.affect_residue * 0.08) * 0.03), 4)
            state.body_energy = round(_clip(state.body_energy + 0.01), 4)
            shaping_detail = "memory_rebalance+habit_decay+salience_replay"
        else:
            state.focus = "rest"
            state.mood = round(_clip(state.mood * 0.90 + 0.55 * 0.10 - state.affect_residue * 0.03), 4)
            state.body_energy = round(_clip(state.body_energy + 0.05), 4)
            state.affect_residue = round(_clip(state.affect_residue * 0.82), 4)
            shaping_detail = "memory_consolidation+habit_consolidation+affect_falloff"

        shaping_events.append(
            {
                "source": requested_mode,
                "non_interactive": True,
                "focus_from": previous_focus,
                "focus_to": state.focus,
                "mood_level": round(state.mood, 4),
                "body_energy": round(state.body_energy, 4),
                "resource_scarcity": round(resource_scarcity, 4),
                "shaping_detail": shaping_detail,
            }
        )
        state.session_metadata["last_noninteractive_mode"] = requested_mode
        state.session_metadata["last_noninteractive_round"] = state.round_count
        return shaping_events

    def build_slow_variable_payload(
        self,
        *,
        state: RuntimeState,
        context: dict[str, Any],
        relation_state: dict[str, float],
        prior_closeness: float,
    ) -> dict[str, Any]:
        resource_scarcity = float(state.resource_state.get("scarcity_index", _clip(1.0 - state.budget_remaining)))
        relationship_drift = abs(float(relation_state.get("closeness", 0.5)) - float(prior_closeness))
        memory_activation = float(context.get("recall_strength", 0.0))
        habit_readiness = float(context.get("habit_strength", 0.0))
        salience_shift = max(abs(float(context.get("valence", 0.0))), memory_activation, habit_readiness, relationship_drift)
        return {
            "affect_residue": round(state.affect_residue, 4),
            "mood_level": round(state.mood, 4),
            "memory_activation": round(memory_activation, 4),
            "memory_interference": round(float(context.get("interference", 0.0)), 4),
            "habit_readiness": round(habit_readiness, 4),
            "resource_scarcity": round(resource_scarcity, 4),
            "relationship_drift": round(relationship_drift, 4),
            "relationship_closeness": round(float(relation_state.get("closeness", 0.5)), 4),
            "body_energy": round(state.body_energy, 4),
            "salience_shift": round(salience_shift, 4),
            "trigger_valence": round(float(context.get("valence", 0.0)), 4),
            "cue_present": bool(context.get("cue")),
            "detail_available": float(context.get("recall_strength", 0.0)) >= float(context.get("detail_threshold", 0.5)),
        }

    def build_vitality_snapshot(
        self,
        *,
        state: RuntimeState,
        context: dict[str, Any],
        relation_state: dict[str, float],
        prior_closeness: float,
        scenario: str,
        sampled_action: ActionCandidate,
        contributions: list[AgentContribution],
        gate_decisions: list[dict[str, Any]],
        shaping_events: list[dict[str, Any]],
    ) -> dict[str, Any]:
        snapshot = self.build_slow_variable_payload(
            state=state,
            context=context,
            relation_state=relation_state,
            prior_closeness=prior_closeness,
        )
        habit_takeover = any(
            row.agent_name == "HabitAgent" and row.action_name == sampled_action.name and row.score >= 0.03
            for row in contributions
        )
        snapshot.update(
            {
                "scenario": scenario,
                "mode": state.mode,
                "cue": context.get("cue"),
                "sampled_action": sampled_action.name,
                "habit_takeover": habit_takeover,
                "forced_recovery": any(item.get("stage") == "forced_mode_switch" for item in gate_decisions),
                "non_interactive_event_count": len([item for item in shaping_events if item.get("non_interactive")]),
            }
        )
        return snapshot
