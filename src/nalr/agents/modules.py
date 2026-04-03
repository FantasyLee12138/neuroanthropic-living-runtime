from __future__ import annotations

from dataclasses import dataclass

from nalr.schemas.models import Proposal, RoundEvent, RuntimeState


def _top_reason(action_preferences: dict[str, float], reason: str) -> str:
    if not action_preferences:
        return reason
    action = max(action_preferences, key=action_preferences.get)
    return f"{reason}; top_action={action}"


@dataclass
class BaseAgent:
    name: str

    def propose(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> Proposal:
        raise NotImplementedError


class BodyStateAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="BodyStateAgent")

    def propose(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> Proposal:
        energy = max(0.0, min(1.0, state.body_energy + event.energy_delta))
        prefs = {"respond": 0.12}
        if energy < 0.45:
            prefs["rest"] = 0.42 + (0.45 - energy)
        return Proposal(self.name, prefs, confidence=0.62, trace_tags=["body"], reason=_top_reason(prefs, f"energy={energy:.2f}"))


class ResourceAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="ResourceAgent")

    def propose(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> Proposal:
        prefs = {"respond": 0.10}
        if state.budget_remaining < 0.35:
            prefs["rest"] = 0.30
        if state.safe_mode:
            prefs["respond"] = 0.40
        return Proposal(self.name, prefs, confidence=0.58, trace_tags=["resource"], reason=_top_reason(prefs, f"budget={state.budget_remaining:.2f}"))


class PFCAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="PFCAgent")

    def propose(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> Proposal:
        content = event.content.lower()
        prefs = {"respond": 0.20}
        if scenario.get("pfc_base_share", 0.2) >= 0.30 or "plan" in content or "help" in content:
            prefs["plan"] = 0.55
        if "remember" in content or context["cue"]:
            prefs["recall"] = max(prefs.get("recall", 0.0), 0.32)
        return Proposal(self.name, prefs, confidence=0.83, trace_tags=["pfc"], reason=_top_reason(prefs, "deliberate planner"))


class HippocampusAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="HippocampusAgent")

    def propose(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> Proposal:
        recall_strength = context["recall_strength"]
        prefs = {}
        if recall_strength > 0:
            prefs["recall"] = 0.20 + recall_strength * 0.4
        return Proposal(self.name, prefs, confidence=0.66, trace_tags=["memory"], reason=_top_reason(prefs or {"respond": 0.0}, f"recall={recall_strength:.2f}"))


class HabitAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="HabitAgent")

    def propose(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> Proposal:
        habit_strength = context["habit_strength"]
        prefs = {}
        if habit_strength > 0:
            prefs["respond"] = 0.10 + habit_strength * scenario.get("habit_weight", 0.1)
            prefs["recall"] = 0.06 + habit_strength * 0.2
        return Proposal(self.name, prefs, confidence=0.60, trace_tags=["habit"], reason=_top_reason(prefs or {"respond": 0.0}, f"habit={habit_strength:.2f}"))


class RelationshipAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="RelationshipAgent")

    def propose(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> Proposal:
        closeness = context["closeness"]
        prefs = {"respond": 0.08 + scenario.get("relationship_weight", 0.1) * closeness}
        if closeness > 0.55 and event.target:
            prefs["connect"] = 0.15 + closeness * 0.2
        return Proposal(self.name, prefs, confidence=0.57, trace_tags=["relationship"], reason=_top_reason(prefs, f"closeness={closeness:.2f}"))


class DMNAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="DMNAgent")

    def propose(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> Proposal:
        prefs = {}
        if state.mode == "idle":
            prefs["wander"] = 0.12 + scenario.get("dmn_weight", 0.05)
        return Proposal(self.name, prefs, confidence=0.45, trace_tags=["dmn"], reason=_top_reason(prefs or {"respond": 0.0}, "background drift"))


class PerspectiveModel(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="PerspectiveModel")

    def propose(self, event: RoundEvent, state: RuntimeState, scenario: dict, context: dict) -> Proposal:
        prefs = {}
        if scenario.get("relationship_weight", 0.0) >= 0.22 and event.target:
            prefs["clarify"] = 0.12 + context["closeness"] * 0.1
        return Proposal(self.name, prefs, confidence=0.49, trace_tags=["perspective"], reason=_top_reason(prefs or {"respond": 0.0}, "social interpretation"))


def build_agents() -> list[BaseAgent]:
    return [
        BodyStateAgent(),
        ResourceAgent(),
        PFCAgent(),
        HippocampusAgent(),
        HabitAgent(),
        RelationshipAgent(),
        DMNAgent(),
        PerspectiveModel(),
    ]

