from __future__ import annotations

from typing import Any

from nalr.schemas.models import ActionCandidate, AgentContribution, RuntimeState


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


class VitalityEngine:
    def build_vitality_modulation_payload(
        self,
        *,
        state: RuntimeState,
        context: dict[str, Any],
        relation_state: dict[str, float],
        prior_closeness: float,
        requested_mode: str | None = None,
    ) -> dict[str, Any]:
        slow_variables = self.build_slow_variable_payload(
            state=state,
            context=context,
            relation_state=relation_state,
            prior_closeness=prior_closeness,
        )
        resource_scarcity = float(slow_variables["resource_scarcity"])
        body_energy = float(slow_variables["body_energy"])
        affect_residue = float(slow_variables["affect_residue"])
        memory_activation = float(slow_variables["memory_activation"])
        habit_readiness = float(slow_variables["habit_readiness"])
        relationship_drift = float(slow_variables["relationship_drift"])

        temperature_scale = _clip(1.0 + resource_scarcity * 0.14 + affect_residue * 0.10 - body_energy * 0.12, 0.65, 1.25)
        search_scale = _clip(1.0 + memory_activation * 0.10 + habit_readiness * 0.06 - resource_scarcity * 0.08, 0.55, 1.30)
        tool_budget_scale = _clip(1.0 - resource_scarcity * 0.30 - (1.0 - body_energy) * 0.10, 0.35, 1.05)
        exploration_bias = _clip(body_energy * 0.30 + memory_activation * 0.22 - resource_scarcity * 0.18 + relationship_drift * 0.06, 0.0, 1.0)
        conservatism_bias = _clip(1.0 - exploration_bias * 0.80 + resource_scarcity * 0.10, 0.0, 1.0)
        vitality_score = _clip(0.40 + body_energy * 0.25 + conservatism_bias * 0.15 - resource_scarcity * 0.10)
        vitality_penalty = _clip(resource_scarcity * 0.22 + (1.0 - body_energy) * 0.18 + affect_residue * 0.08)

        action_bias = {
            "rest": round(_clip(0.12 + (1.0 - body_energy) * 0.22 + resource_scarcity * 0.08), 4),
            "respond": round(_clip(0.08 + float(slow_variables["relationship_closeness"]) * 0.10 + body_energy * 0.04), 4),
            "plan": round(_clip(0.06 + memory_activation * 0.08 - resource_scarcity * 0.04), 4),
            "clarify": round(_clip(0.05 + float(slow_variables["relationship_closeness"]) * 0.06 + habit_readiness * 0.04), 4),
            "recall": round(_clip(0.05 + memory_activation * 0.14), 4),
            "connect": round(_clip(0.03 + float(slow_variables["relationship_closeness"]) * 0.08 - resource_scarcity * 0.03), 4),
            "wander": round(_clip(0.04 + exploration_bias * 0.12 - resource_scarcity * 0.06), 4),
        }
        if requested_mode == "idle":
            action_bias["wander"] = round(_clip(action_bias["wander"] + 0.04), 4)
        elif requested_mode == "sleep":
            action_bias["rest"] = round(_clip(action_bias["rest"] + 0.08), 4)
            action_bias["wander"] = round(_clip(action_bias["wander"] - 0.04), 4)

        return {
            "module_name": "VitalityEngine",
            "layer": "global",
            "kind": "vitality_modulation",
            "prior_role": "stable_prior",
            "mode": requested_mode or state.mode,
            "slow_variables": slow_variables,
            "modulation": {
                "temperature_scale": round(temperature_scale, 4),
                "search_scale": round(search_scale, 4),
                "tool_budget_scale": round(tool_budget_scale, 4),
                "exploration_bias": round(exploration_bias, 4),
                "conservatism_bias": round(conservatism_bias, 4),
            },
            "delta_logits": action_bias,
            "attention_bias": {
                "memory": round(_clip(memory_activation + habit_readiness * 0.4), 4),
                "stability": round(_clip(conservatism_bias), 4),
                "exploration": round(_clip(exploration_bias), 4),
            },
            "vitality_score": round(vitality_score, 4),
            "vitality_penalty": round(vitality_penalty, 4),
            "soft_mask": {
                "wander": round(_clip(resource_scarcity * 0.14 + (1.0 - body_energy) * 0.05), 4),
            },
            "hard_mask": {},
            "confidence": round(vitality_score, 4),
            "trace_reason": (
                f"mode={requested_mode or state.mode}; energy={body_energy:.2f}; "
                f"scarcity={resource_scarcity:.2f}; memory={memory_activation:.2f}"
            ),
        }

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
        modulation_payload = self.build_vitality_modulation_payload(
            state=state,
            context=context,
            relation_state=relation_state,
            prior_closeness=prior_closeness,
            requested_mode=state.mode,
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
                "vitality_score": modulation_payload["vitality_score"],
                "vitality_penalty": modulation_payload["vitality_penalty"],
                "vitality_modulation": modulation_payload["modulation"],
                "vitality_bias": modulation_payload["delta_logits"],
            }
        )
        return snapshot
