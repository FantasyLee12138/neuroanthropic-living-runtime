from __future__ import annotations

from nalr.runtime.controller import RuntimeController
from nalr.schemas.models import CommandEnvelope


class CommandInterfaceLayer:
    def __init__(self, controller: RuntimeController) -> None:
        self.controller = controller

    def _parse(self, command: str) -> CommandEnvelope:
        alias_map = {
            "rest": "body rest",
            "calm": "mood calm",
            "explain current": "explain current",
        }
        resolved = alias_map.get(command, command)
        parts = resolved.split()
        domain = parts[0]
        verb = parts[1] if len(parts) > 1 else "show"
        target = parts[2] if len(parts) > 2 else None
        operator_level = "read_only"
        if domain in {"body", "mood", "nudge", "mode"}:
            operator_level = "soft_intervene"
        if domain in {"agent", "debug", "suppress"}:
            operator_level = "debug_control"
        if domain in {"safe", "checkpoint", "budget"}:
            operator_level = "ops_admin"
        return CommandEnvelope(domain=domain, verb=verb, target=target, operator_level=operator_level)

    def execute(self, command: str):
        envelope = self._parse(command)

        if command == "explain current":
            state = self.controller.load_runtime_state()
            if state.round_count == 0:
                return {"round_id": 0, "sampled_action": None, "top_drivers": []}
            return self.controller.why_this(state.round_count)

        if envelope.domain == "state" and envelope.verb == "show":
            return self.controller.state_payload()
        if envelope.domain == "body" and envelope.verb == "show":
            return {"body_energy": self.controller.load_runtime_state().body_energy}
        if envelope.domain == "body" and envelope.verb == "rest":
            return self.controller.apply_command("body rest", envelope=envelope)
        if envelope.domain == "mood" and envelope.verb == "show":
            return {"mood": self.controller.load_runtime_state().mood}
        if envelope.domain == "mood" and envelope.verb == "calm":
            return self.controller.apply_command("mood calm", envelope=envelope)
        if envelope.domain == "focus" and envelope.verb == "show":
            return {"focus": self.controller.load_runtime_state().focus}
        if envelope.domain == "relation" and envelope.verb == "show" and envelope.target:
            return self.controller.relation_show(envelope.target)
        if envelope.domain == "memory" and envelope.verb == "top":
            return self.controller.memory_top()
        if envelope.domain == "habit" and envelope.verb == "top":
            return self.controller.habit_top()
        if envelope.domain == "trace" and envelope.verb == "round" and envelope.target:
            return self.controller.trace_round(int(envelope.target))
        if envelope.domain == "trace" and envelope.verb == "why" and envelope.target:
            return self.controller.why_this(int(envelope.target))
        if envelope.domain == "trace" and envelope.verb == "contribution" and envelope.target:
            return self.controller.contribution_breakdown(int(envelope.target))
        if envelope.domain == "agent" and envelope.verb == "list":
            return self.controller.agent_list()
        if envelope.domain == "agent" and envelope.verb in {"disable", "enable"} and envelope.target:
            return self.controller.apply_command(f"agent {envelope.verb} {envelope.target}", envelope=envelope)
        if envelope.domain == "skill" and envelope.verb == "stats":
            return self.controller.skill_stats()
        if envelope.domain == "skill" and envelope.verb == "profile" and envelope.target:
            return self.controller.skill_profile(envelope.target)
        if envelope.domain == "debug" and envelope.verb == "weight" and len(command.split()) == 4:
            _, _, agent_name, weight = command.split()
            return self.controller.apply_command(f"debug weight {agent_name} {weight}", envelope=envelope)
        if envelope.domain == "nudge" and envelope.verb == "focus" and envelope.target:
            return self.controller.apply_command(f"nudge focus {envelope.target}", envelope=envelope)
        if envelope.domain == "suppress" and envelope.verb == "dmn":
            ttl = command.split()[2] if len(command.split()) >= 3 else "temporary"
            return self.controller.apply_command(f"suppress dmn {ttl}", envelope=envelope)
        if envelope.domain == "mode" and envelope.verb == "set" and envelope.target:
            return self.controller.apply_command(f"mode set {envelope.target}", envelope=envelope)
        if envelope.domain == "safe" and envelope.verb in {"on", "off"}:
            return self.controller.apply_command(f"safe {envelope.verb}", envelope=envelope)
        if envelope.domain == "checkpoint" and envelope.verb == "create":
            return self.controller.checkpoint()
        if envelope.domain == "checkpoint" and envelope.verb == "rewind" and envelope.target:
            return self.controller.rewind(envelope.target)
        if envelope.domain == "budget" and envelope.verb == "show":
            return {"budget_remaining": self.controller.load_runtime_state().budget_remaining}
        if envelope.domain == "replay" and envelope.verb == "round" and envelope.target:
            parts = command.split()
            seed = int(parts[3]) if len(parts) >= 4 else None
            return self.controller.replay_round(int(envelope.target), seed=seed)
        raise ValueError(f"unsupported command: {command}")
