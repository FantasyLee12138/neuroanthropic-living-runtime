from __future__ import annotations

from typing import Any

from nalr.schemas.models import ActionCandidate, AgentContribution, EnergyProjectionSpec, ProbabilisticContribution, RuntimeState


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
        previous_fatigue = float(state.fatigue)
        resource_scarcity = float(state.resource_state.get("scarcity_index", _clip(1.0 - state.budget_remaining)))

        if requested_mode == "idle":
            state.focus = "wander" if state.focus != "rest" else state.focus
            state.mood = round(_clip(state.mood * 0.97 + (0.52 - state.affect_residue * 0.08) * 0.03), 4)
            state.body_energy = round(_clip(state.body_energy + 0.01), 4)
            embodied_fatigue = _clip((1.0 - state.body_energy) * 0.78 + state.affect_residue * 0.18 + resource_scarcity * 0.04)
            state.fatigue = round(_clip(previous_fatigue * 0.80 + embodied_fatigue * 0.20 - 0.035), 4)
            shaping_detail = "memory_rebalance+habit_decay+salience_replay"
        else:
            state.focus = "rest"
            state.mood = round(_clip(state.mood * 0.90 + 0.55 * 0.10 - state.affect_residue * 0.03), 4)
            state.body_energy = round(_clip(state.body_energy + 0.05), 4)
            state.affect_residue = round(_clip(state.affect_residue * 0.82), 4)
            embodied_fatigue = _clip((1.0 - state.body_energy) * 0.72 + state.affect_residue * 0.24 + resource_scarcity * 0.04)
            state.fatigue = round(_clip(previous_fatigue * 0.68 + embodied_fatigue * 0.32 - 0.09), 4)
            shaping_detail = "memory_consolidation+habit_consolidation+affect_falloff"

        shaping_events.append(
            {
                "source": requested_mode,
                "non_interactive": True,
                "focus_from": previous_focus,
                "focus_to": state.focus,
                "mood_level": round(state.mood, 4),
                "body_energy": round(state.body_energy, 4),
                "fatigue_from": round(previous_fatigue, 4),
                "fatigue_to": round(state.fatigue, 4),
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

    def build_vitality_modulation_contribution(self, vitality_snapshot: dict[str, Any]) -> ProbabilisticContribution:
        body_energy = float(vitality_snapshot.get("body_energy", 0.0) or 0.0)
        resource_scarcity = float(vitality_snapshot.get("resource_scarcity", 0.0) or 0.0)
        memory_activation = float(vitality_snapshot.get("memory_activation", 0.0) or 0.0)
        affect_residue = float(vitality_snapshot.get("affect_residue", 0.0) or 0.0)
        habit_readiness = float(vitality_snapshot.get("habit_readiness", 0.0) or 0.0)

        modulated_delta: dict[str, float] = {
            "respond": round(0.03 + habit_readiness * 0.05, 6),
        }
        if body_energy < 0.55:
            deficit = 0.55 - body_energy
            modulated_delta["rest"] = round(0.10 + deficit * 0.60 + resource_scarcity * 0.12, 6)
            modulated_delta["plan"] = round(-(0.06 + deficit * 0.45 + resource_scarcity * 0.10), 6)
            modulated_delta["connect"] = round(-(0.03 + deficit * 0.20), 6)
        if memory_activation > 0.0:
            modulated_delta["recall"] = round(memory_activation * 0.24, 6)
        if affect_residue > 0.12:
            modulated_delta["clarify"] = round(affect_residue * 0.16, 6)
        if resource_scarcity > 0.35:
            modulated_delta["wander"] = round(-(resource_scarcity * 0.18), 6)

        confidence = _clip(
            0.30 + max(body_energy < 0.55 and (0.55 - body_energy), 0.0, memory_activation, affect_residue, resource_scarcity),
            0.0,
            1.0,
        )
        dependency_trace = [
            f"body_energy:{round(body_energy, 4)}",
            f"memory_activation:{round(memory_activation, 4)}",
            f"affect_residue:{round(affect_residue, 4)}",
            f"resource_scarcity:{round(resource_scarcity, 4)}",
            f"habit_readiness:{round(habit_readiness, 4)}",
        ]
        return ProbabilisticContribution(
            module_name="VitalityEngine",
            module_type="vitality",
            level="action",
            target_space="action",
            raw_signal=dict(modulated_delta),
            modulated_delta=modulated_delta,
            confidence=confidence,
            confidence_calibrated=round(_clip(confidence * 0.92, 0.0, 1.0), 4),
            trace_reason="vitality modulation from slow variables",
            projection_reason="vitality modulation projected from slow variables",
            applied_at_stage="vitality_modulation",
            native_operator="slow_variable_modulation",
            dependency_trace=dependency_trace,
            projection=EnergyProjectionSpec(module_type="vitality", target_space="action", module_temperature=0.95),
        )
