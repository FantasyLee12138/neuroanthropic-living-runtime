from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from nalr.runtime.controller import RuntimeController
from nalr.schemas.models import CommandEnvelope


@dataclass(frozen=True)
class CommandSpec:
    domain: str
    verb: str
    operator_level: str
    mutable: bool = False
    mutation_scope: str | None = None
    rollback_available: bool = False


COMMAND_SPECS: dict[tuple[str, str], CommandSpec] = {
    ("state", "show"): CommandSpec("state", "show", "read_only"),
    ("body", "show"): CommandSpec("body", "show", "read_only"),
    ("body", "rest"): CommandSpec("body", "rest", "soft_intervene", mutable=True, mutation_scope="body", rollback_available=True),
    ("mood", "show"): CommandSpec("mood", "show", "read_only"),
    ("mood", "calm"): CommandSpec("mood", "calm", "soft_intervene", mutable=True, mutation_scope="mood", rollback_available=True),
    ("focus", "show"): CommandSpec("focus", "show", "read_only"),
    ("identity", "show"): CommandSpec("identity", "show", "read_only"),
    ("identity", "set-name"): CommandSpec("identity", "set-name", "soft_intervene", mutable=True, mutation_scope="identity", rollback_available=True),
    ("run", "start"): CommandSpec("run", "start", "soft_intervene"),
    ("run", "status"): CommandSpec("run", "status", "soft_intervene"),
    ("run", "pause"): CommandSpec("run", "pause", "soft_intervene"),
    ("run", "resume"): CommandSpec("run", "resume", "soft_intervene"),
    ("run", "abort"): CommandSpec("run", "abort", "soft_intervene"),
    ("run", "explain"): CommandSpec("run", "explain", "soft_intervene"),
    ("relation", "show"): CommandSpec("relation", "show", "read_only"),
    ("memory", "top"): CommandSpec("memory", "top", "read_only"),
    ("memory", "recall"): CommandSpec("memory", "recall", "read_only"),
    ("habit", "top"): CommandSpec("habit", "top", "read_only"),
    ("habit", "reset"): CommandSpec("habit", "reset", "soft_intervene", mutable=True, mutation_scope="habit", rollback_available=True),
    ("trace", "round"): CommandSpec("trace", "round", "read_only"),
    ("trace", "why"): CommandSpec("trace", "why", "read_only"),
    ("trace", "contribution"): CommandSpec("trace", "contribution", "read_only"),
    ("trace", "compact"): CommandSpec("trace", "compact", "debug_control"),
    ("agent", "list"): CommandSpec("agent", "list", "read_only"),
    ("agent", "disable"): CommandSpec("agent", "disable", "debug_control", mutable=True, mutation_scope="agent", rollback_available=True),
    ("agent", "enable"): CommandSpec("agent", "enable", "debug_control", mutable=True, mutation_scope="agent", rollback_available=True),
    ("skill", "stats"): CommandSpec("skill", "stats", "read_only"),
    ("skill", "profile"): CommandSpec("skill", "profile", "read_only"),
    ("debug", "weight"): CommandSpec("debug", "weight", "debug_control", mutable=True, mutation_scope="agent", rollback_available=True),
    ("nudge", "focus"): CommandSpec("nudge", "focus", "soft_intervene", mutable=True, mutation_scope="focus", rollback_available=True),
    ("nudge", "relation"): CommandSpec("nudge", "relation", "soft_intervene", mutable=True, mutation_scope="relation", rollback_available=True),
    ("suppress", "dmn"): CommandSpec("suppress", "dmn", "debug_control", mutable=True, mutation_scope="DMNAgent", rollback_available=True),
    ("mode", "set"): CommandSpec("mode", "set", "soft_intervene", mutable=True, mutation_scope="runtime", rollback_available=True),
    ("safe", "on"): CommandSpec("safe", "on", "ops_admin", mutable=True, mutation_scope="runtime", rollback_available=True),
    ("safe", "off"): CommandSpec("safe", "off", "ops_admin", mutable=True, mutation_scope="runtime", rollback_available=True),
    ("checkpoint", "create"): CommandSpec("checkpoint", "create", "ops_admin", mutable=True, mutation_scope="checkpoint", rollback_available=True),
    ("checkpoint", "rewind"): CommandSpec("checkpoint", "rewind", "ops_admin", mutable=True, mutation_scope="checkpoint", rollback_available=True),
    ("budget", "show"): CommandSpec("budget", "show", "read_only"),
    ("budget", "set"): CommandSpec("budget", "set", "ops_admin", mutable=True, mutation_scope="resource", rollback_available=True),
    ("replay", "round"): CommandSpec("replay", "round", "debug_control"),
    ("why", "this"): CommandSpec("why", "this", "debug_control"),
    ("why", "not"): CommandSpec("why", "not", "debug_control"),
    ("what", "changed"): CommandSpec("what", "changed", "debug_control"),
    ("eval", "longrun"): CommandSpec("eval", "longrun", "debug_control"),
    ("snapshot", "restore"): CommandSpec("snapshot", "restore", "ops_admin", mutable=True, mutation_scope="snapshot", rollback_available=True),
    ("dream", "status"): CommandSpec("dream", "status", "read_only"),
    ("dream", "trace"): CommandSpec("dream", "trace", "read_only"),
    ("dream", "proposals"): CommandSpec("dream", "proposals", "read_only"),
    ("dream", "metrics"): CommandSpec("dream", "metrics", "read_only"),
    ("dream", "run"): CommandSpec("dream", "run", "ops_admin"),
    ("dream", "on"): CommandSpec("dream", "on", "ops_admin"),
    ("dream", "off"): CommandSpec("dream", "off", "ops_admin"),
}


class CommandInterfaceLayer:
    def __init__(self, controller: RuntimeController) -> None:
        self.controller = controller

    def _parse(self, command: str) -> CommandEnvelope:
        alias_map = {
            "rest": "body rest",
            "calm": "mood calm",
            "explain current": "explain current",
        }
        resolved = alias_map.get(command.strip(), command.strip())
        parts = resolved.split()
        if not parts:
            raise ValueError("empty command")

        if resolved == "explain current":
            return CommandEnvelope(
                command_id=uuid4().hex,
                domain="explain",
                verb="current",
                canonical=resolved,
                operator_level="read_only",
            )

        domain = parts[0]
        verb = parts[1] if len(parts) > 1 else "show"
        target = parts[2] if len(parts) > 2 else None
        key = (domain, verb)
        if key not in COMMAND_SPECS and not resolved.startswith("run start "):
            raise ValueError(f"unsupported command: {command}")

        parsed_args: dict[str, object] = {}
        flags: dict[str, object] = {}
        canonical = resolved
        spec = COMMAND_SPECS.get(key, CommandSpec(domain, verb, "soft_intervene"))

        if resolved.startswith("run start "):
            key = ("run", "start")
            spec = COMMAND_SPECS[key]
            goal = resolved[len("run start ") :].strip()
            canonical = f"run start {goal}"
            parsed_args["goal"] = goal
            target = goal or None
        elif key == ("budget", "set"):
            if len(parts) >= 4 and parts[2] == "--cap":
                flags["cap"] = True
                value = int(parts[3])
                canonical = f"budget set {value}"
            else:
                value = int(parts[2])
                canonical = f"budget set {value}"
            parsed_args["value"] = value
            target = str(value)
        elif key == ("identity", "set-name"):
            name = " ".join(parts[2:]).strip()
            parsed_args["name"] = name
            target = name or None
            canonical = f"identity set-name {name}"
        elif key == ("debug", "weight") and len(parts) >= 4:
            parsed_args["agent_name"] = parts[2]
            parsed_args["weight"] = float(parts[3])
            target = parts[2]
        elif key == ("nudge", "focus") and len(parts) >= 3:
            parsed_args["delta"] = float(parts[2])
            target = parts[2]
        elif key == ("nudge", "relation") and len(parts) >= 5:
            if parts[3] != "trust":
                raise ValueError(f"unsupported relation nudge metric: {parts[3]}")
            parsed_args["relation_target"] = parts[2]
            parsed_args["delta"] = float(parts[4])
            target = parts[2]
            canonical = f"nudge relation {parts[2]} trust {parts[4]}"
        elif key == ("habit", "reset") and len(parts) >= 3:
            parsed_args["pattern"] = parts[2]
        elif key == ("mode", "set") and len(parts) >= 3:
            parsed_args["mode"] = parts[2]
        elif key == ("checkpoint", "rewind") and len(parts) >= 3:
            parsed_args["checkpoint_id"] = parts[2]
        elif key == ("replay", "round"):
            if len(parts) >= 3:
                target = parts[2]
            if len(parts) >= 4:
                parsed_args["seed"] = int(parts[3])
        elif key == ("why", "this"):
            if len(parts) >= 3:
                target = parts[2]
        elif key == ("why", "not"):
            if len(parts) >= 3:
                parsed_args["action"] = parts[2]
                target = parts[3] if len(parts) >= 4 else None
        elif key == ("what", "changed") and len(parts) >= 3:
            parsed_args["window"] = int(parts[2])
            target = parts[2]
        elif key == ("eval", "longrun") and len(parts) >= 3:
            parsed_args["rounds"] = int(parts[2])
            target = parts[2]
        elif key == ("snapshot", "restore") and len(parts) >= 3:
            target = parts[2]
        elif key == ("suppress", "dmn"):
            ttl = parts[2] if len(parts) >= 3 else "temporary"
            parsed_args["ttl"] = ttl
            target = ttl
        elif key in {("dream", "trace"), ("dream", "proposals")}:
            target = parts[2] if len(parts) >= 3 else "last"
        elif key == ("dream", "run"):
            parsed_args["mode"] = parts[2] if len(parts) >= 3 else "sleep"
            if len(parts) >= 4:
                parsed_args["cue"] = parts[3]

        return CommandEnvelope(
            command_id=uuid4().hex,
            domain=domain,
            verb=verb,
            target=target,
            canonical=canonical,
            parsed_args=parsed_args,
            flags=flags,
            operator_level=spec.operator_level,
            mutation_scope=spec.mutation_scope,
            rollback_available=spec.rollback_available,
        )

    def execute(self, command: str):
        envelope = self._parse(command)
        key = (envelope.domain, envelope.verb)

        if key == ("explain", "current"):
            state = self.controller.load_runtime_state()
            if state.round_count == 0:
                return {"round_id": 0, "sampled_action": None, "top_drivers": []}
            return self.controller.why_this(state.round_count)

        if key == ("state", "show"):
            return self.controller.state_payload()
        if key == ("body", "show"):
            return {"body_energy": self.controller.load_runtime_state().body_energy}
        if key == ("mood", "show"):
            return {"mood": self.controller.load_runtime_state().mood}
        if key == ("focus", "show"):
            return {"focus": self.controller.load_runtime_state().focus}
        if key == ("identity", "show"):
            return self.controller.identity_payload()
        if key == ("relation", "show") and envelope.target:
            return self.controller.relation_show(envelope.target)
        if key == ("memory", "top"):
            return self.controller.memory_top()
        if key == ("memory", "recall") and envelope.target:
            return self.controller.memory_recall(envelope.target)
        if key == ("habit", "top"):
            return self.controller.habit_top()
        if key == ("trace", "round") and envelope.target:
            return self.controller.trace_round(int(envelope.target))
        if key == ("trace", "why") and envelope.target:
            return self.controller.why_this(int(envelope.target))
        if key == ("trace", "contribution") and envelope.target:
            return self.controller.contribution_breakdown(int(envelope.target))
        if key == ("trace", "compact"):
            return self.controller.compact_traces()
        if key == ("agent", "list"):
            return self.controller.agent_list()
        if key == ("skill", "stats"):
            return self.controller.skill_stats()
        if key == ("skill", "profile") and envelope.target:
            return self.controller.skill_profile(envelope.target)
        if key == ("budget", "show"):
            return {"budget_remaining": self.controller.load_runtime_state().budget_remaining}
        if key == ("dream", "status"):
            return self.controller.dream_status()
        if key == ("dream", "trace"):
            return self.controller.dream_trace(envelope.target or "last")
        if key == ("dream", "proposals"):
            return self.controller.dream_proposals(envelope.target or "last")
        if key == ("dream", "metrics"):
            return self.controller.dream_metrics()
        if key == ("dream", "run"):
            return self.controller.run_dream(
                mode=str(envelope.parsed_args.get("mode", "sleep")),
                cue=envelope.parsed_args.get("cue"),
            )
        if key == ("dream", "on"):
            return self.controller.set_dream_enabled(True)
        if key == ("dream", "off"):
            return self.controller.set_dream_enabled(False)
        if key == ("run", "start"):
            return self.controller.start_run(str(envelope.parsed_args["goal"]), operator_level=envelope.operator_level)
        if key == ("run", "status"):
            return self.controller.run_status()
        if key == ("run", "pause"):
            return self.controller.pause_run()
        if key == ("run", "resume"):
            return self.controller.resume_run()
        if key == ("run", "abort"):
            return self.controller.abort_run()
        if key == ("run", "explain"):
            return self.controller.explain_run()
        if key == ("replay", "round") and envelope.target:
            return self.controller.replay_round(int(envelope.target), seed=envelope.parsed_args.get("seed"))
        if key == ("why", "this"):
            round_id = int(envelope.target) if envelope.target else self.controller.load_runtime_state().round_count
            return self.controller.why_this(round_id)
        if key == ("why", "not"):
            round_id = int(envelope.target) if envelope.target else self.controller.load_runtime_state().round_count
            return self.controller.why_not(round_id, str(envelope.parsed_args["action"]))
        if key == ("what", "changed"):
            return self.controller.what_changed(int(envelope.parsed_args.get("window", 5)))
        if key == ("eval", "longrun"):
            return self.controller.eval_longrun(int(envelope.parsed_args.get("rounds", 1000)))

        spec = COMMAND_SPECS.get(key)
        if spec is not None and spec.mutable:
            return self.controller.execute_command(envelope)

        raise ValueError(f"unsupported command: {command}")
