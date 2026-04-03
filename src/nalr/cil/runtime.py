from __future__ import annotations

import os

from nalr.runtime.controller import RuntimeController
from nalr.schemas.models import RoundEvent


class CommandInterfaceLayer:
    def __init__(self, controller: RuntimeController) -> None:
        self.controller = controller

    def show_state(self) -> dict:
        return self.controller.state_payload()

    def show_body(self) -> dict:
        return {"body_energy": self.controller.load_runtime_state().body_energy}

    def show_mood(self) -> dict:
        return {"mood": self.controller.load_runtime_state().mood}

    def show_focus(self) -> dict:
        if self.controller.load_runtime_state().round_count == 0:
            self.controller.tick(
                RoundEvent(source="system", content="focus probe"),
                scenario=os.environ.get("NALR_SCENARIO", "chat"),
                mode=os.environ.get("NALR_MODE", "interactive"),
            )
        return {"focus": self.controller.load_runtime_state().focus}

    def show_relation(self, target: str) -> dict:
        return self.controller.relation_show(target)

    def memory_top(self, limit: int = 5) -> list[dict]:
        return self.controller.memory_top(limit)

    def habit_top(self, limit: int = 5) -> list[dict]:
        return self.controller.habit_top(limit)

    def skill_stats(self) -> dict:
        return self.controller.skill_stats()

    def skill_profile(self, skill_name: str) -> dict:
        return self.controller.skill_profile(skill_name)

    def apply(self, command: str):
        return self.controller.apply_command(command)

    def relation_alias_rest(self):
        return self.controller.apply_command("body rest")

    def relation_alias_calm(self):
        return self.controller.apply_command("mood calm")
