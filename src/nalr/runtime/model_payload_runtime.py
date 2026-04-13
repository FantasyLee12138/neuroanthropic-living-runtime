from __future__ import annotations

from typing import TYPE_CHECKING, Any

from nalr.schemas.models import RenderPlan, RoundEvent, RuntimeState

if TYPE_CHECKING:
    from nalr.runtime.controller import RuntimeController


class _ControllerBackedRuntime:
    def __init__(self, controller: "RuntimeController") -> None:
        self.controller = controller

    def __getattr__(self, name: str):
        return getattr(self.controller, name)


class ModelPayloadRuntime(_ControllerBackedRuntime):
    def _compact_float(self, value: Any) -> float:
        return round(float(value or 0.0), 4)

    def _build_state_summary(self, state: RuntimeState) -> dict[str, Any]:
        return {
            "mode": state.mode,
            "safe_mode": state.safe_mode,
            "focus": state.focus,
            "body_energy": self._compact_float(state.body_energy),
            "mood": self._compact_float(state.mood),
            "affect_residue": self._compact_float(state.affect_residue),
            "budget_remaining": self._compact_float(state.budget_remaining),
            "run_status": state.run_status,
            "current_goal": state.current_goal,
        }

    def _build_context_summary(self, context: dict[str, Any]) -> dict[str, Any]:
        keys = (
            "cue",
            "closeness",
            "recall_strength",
            "interference",
            "habit_strength",
            "burn_rate_ratio",
            "low_balance_ratio",
            "queue_pressure",
            "latency_pressure",
            "detail_threshold",
        )
        summary: dict[str, Any] = {}
        for key in keys:
            value = context.get(key)
            if isinstance(value, float):
                summary[key] = self._compact_float(value)
            elif isinstance(value, int):
                summary[key] = value
            elif value not in {None, ""}:
                summary[key] = value
        return summary

    def _build_relation_summary(self, relation_state: dict[str, Any]) -> dict[str, Any]:
        return {
            "closeness": self._compact_float(relation_state.get("closeness", 0.0)),
            "boundary_level": self._compact_float(relation_state.get("boundary_level", 0.0)),
            "relationship_risk": self._compact_float(relation_state.get("relationship_risk", 0.0)),
            "privacy_level": self._compact_float(relation_state.get("privacy_level", 0.0)),
        }

    def _build_event_summary(self, event: RoundEvent) -> dict[str, Any]:
        return {
            "content": event.content,
            "target": event.target,
            "cue": event.cue,
            "valence": self._compact_float(event.valence),
            "energy_delta": self._compact_float(event.energy_delta),
        }

    def _build_pfc_model_payload(
        self,
        event: RoundEvent,
        state: RuntimeState,
        scenario: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "event": self._build_event_summary(event),
            "state_summary": self._build_state_summary(state),
            "scenario_summary": {
                "name": scenario.get("name", ""),
                "pfc_base_share": self._compact_float(scenario.get("pfc_base_share", 0.0)),
                "delay_tolerance": self._compact_float(scenario.get("delay_tolerance", 0.0)),
            },
            "context_summary": self._build_context_summary(context),
        }

    def _build_perspective_model_payload(
        self,
        event: RoundEvent,
        state: RuntimeState,
        scenario: dict[str, Any],
        context: dict[str, Any],
        relation_state: dict[str, Any],
        *,
        action_name: str,
        output_key: str,
    ) -> dict[str, Any]:
        return {
            "event": self._build_event_summary(event),
            "state_summary": self._build_state_summary(state),
            "scenario_summary": {"name": scenario.get("name", "")},
            "context_summary": self._build_context_summary(context),
            "relation_state": self._build_relation_summary(relation_state),
            output_key: action_name,
        }

    def _build_render_model_payload(self, render_plan: RenderPlan) -> dict[str, Any]:
        identity = render_plan.identity_context
        return {
            "action": render_plan.action,
            "event_summary": render_plan.event_summary,
            "target": render_plan.target,
            "identity": {
                "display_label": identity.display_label,
                "class_label": identity.class_label,
                "query_kind": identity.query_kind,
                "query_intent": identity.query_intent,
                "disclosure_detail": identity.disclosure_detail,
                "disclosure_intent": identity.disclosure_intent,
            },
            "relation_state": self._build_relation_summary(render_plan.relation_state),
            "slow_variables": {
                key: self._compact_float(value) if isinstance(value, float) else value
                for key, value in dict(render_plan.message_plan.get("slow_variables", {})).items()
                if key in {"resource_scarcity", "memory_activation", "relationship_heat", "identity_salience"}
            },
            "repair_expression": dict(render_plan.message_plan.get("repair_expression", {})),
            "safety_constraints": {
                "gate": self._compact_float(render_plan.safety_constraints.get("gate", 0.0)),
                "conflict_hot": bool(render_plan.safety_constraints.get("conflict_hot", False)),
                "winning_priority": render_plan.safety_constraints.get("winning_priority"),
                "compromise_template": render_plan.safety_constraints.get("compromise_template"),
                "repair_stage": render_plan.safety_constraints.get("repair_stage"),
            },
        }
