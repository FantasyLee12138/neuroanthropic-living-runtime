from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4
from typing import Any

from nalr.runtime.controller import RuntimeController
from nalr.schemas.models import CommandEnvelope, to_dict


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
    ("endogenous", "tick"): CommandSpec("endogenous", "tick", "debug_control"),
    ("endogenous", "status"): CommandSpec("endogenous", "status", "read_only"),
    ("initiative", "status"): CommandSpec("initiative", "status", "read_only"),
    ("initiative", "distribution"): CommandSpec("initiative", "distribution", "read_only"),
    ("initiative", "trigger"): CommandSpec("initiative", "trigger", "debug_control"),
    ("initiative", "config"): CommandSpec("initiative", "config", "ops_admin"),
    ("initiative", "why"): CommandSpec("initiative", "why", "read_only"),
    ("monologue", "status"): CommandSpec("monologue", "status", "read_only"),
    ("monologue", "show"): CommandSpec("monologue", "show", "read_only"),
    ("thought", "show"): CommandSpec("thought", "show", "read_only"),
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
    ("replay", "motivation"): CommandSpec("replay", "motivation", "read_only"),
    ("why", "this"): CommandSpec("why", "this", "debug_control"),
    ("why", "not"): CommandSpec("why", "not", "debug_control"),
    ("why", "motivation"): CommandSpec("why", "motivation", "read_only"),
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


def build_endogenous_status_payload(controller: RuntimeController) -> dict[str, Any]:
    state = controller.load_runtime_state()
    scheduler_state = dict(to_dict(getattr(state, "endogenous_scheduler_state", {})) or {})
    endogenous_state = dict(to_dict(getattr(state, "endogenous_state", {})) or {})
    recent_triggers = list(scheduler_state.get("recent_triggers", []) or [])
    latest_trigger = recent_triggers[-1] if recent_triggers else {}
    round_reader = getattr(controller.trace_store, "list_round_summaries", None)
    round_rows = list(round_reader()) if callable(round_reader) else controller.trace_store.list_rounds()
    latest_endogenous_round = next(
        (row for row in reversed(round_rows) if row.get("cause_type") == "endogenous"),
        None,
    )
    return {
        "round_count": int(getattr(state, "round_count", 0) or 0),
        "last_endogenous_tick_at": scheduler_state.get("last_endogenous_tick_at"),
        "suppression_reason": scheduler_state.get("suppression_reason"),
        "last_trigger": endogenous_state.get("last_trigger", ""),
        "latest_trigger": latest_trigger,
        "recent_triggers": recent_triggers[-5:],
        "current_intent": endogenous_state.get("current_intent"),
        "stability": endogenous_state.get("stability", 0),
        "latest_endogenous_round_id": latest_endogenous_round.get("round_id") if latest_endogenous_round else None,
        "storage": controller.trace_storage_status(),
    }


def run_endogenous_tick_payload(
    controller: RuntimeController,
    *,
    trigger: str = "idle",
    mode: str | None = None,
) -> dict[str, Any]:
    return controller.run_endogenous_tick(trigger=trigger, mode=mode)


class CommandInterfaceLayer:
    def __init__(self, controller: RuntimeController) -> None:
        self.controller = controller

    def endogenous_tick(self, *, trigger: str = "idle", mode: str | None = None) -> dict[str, Any]:
        return run_endogenous_tick_payload(self.controller, trigger=trigger, mode=mode)

    def endogenous_status(self) -> dict[str, Any]:
        return build_endogenous_status_payload(self.controller)

    def initiative_status(self) -> dict[str, Any]:
        return self.controller.initiative_status()

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
        elif key == ("replay", "motivation"):
            if len(parts) >= 3:
                target = parts[2]
        elif key == ("why", "motivation"):
            if len(parts) >= 3:
                target = parts[2]
        elif key == ("endogenous", "tick"):
            if len(parts) >= 3:
                parsed_args["trigger"] = parts[2]
                target = parts[2]
            if len(parts) >= 4:
                parsed_args["mode"] = parts[3]
            canonical = " ".join(parts[:4]).strip()
        elif key == ("initiative", "trigger"):
            trigger_args = parts[2:]
            force = any(part == "force" for part in trigger_args)
            trigger_tokens = [part for part in trigger_args if part != "force"]
            if trigger_tokens:
                parsed_args["trigger"] = trigger_tokens[0]
                target = trigger_tokens[0]
            if len(trigger_tokens) >= 2:
                parsed_args["mode"] = trigger_tokens[1]
            if force:
                parsed_args["force"] = True
            canonical = " ".join(parts).strip()
        elif key == ("initiative", "why"):
            target = parts[2] if len(parts) >= 3 else "last"
        elif key == ("monologue", "show"):
            if len(parts) >= 3:
                parsed_args["limit"] = int(parts[2])
                target = parts[2]
        elif key == ("thought", "show"):
            target = parts[2] if len(parts) >= 3 else "last"
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
            return self.controller.trace_round(envelope.target)
        if key == ("trace", "why") and envelope.target:
            return self.controller.why_this(envelope.target)
        if key == ("trace", "contribution") and envelope.target:
            return self.controller.contribution_breakdown(envelope.target)
        if key == ("trace", "compact"):
            return self.controller.compact_traces()
        if key == ("endogenous", "tick"):
            return self.endogenous_tick(
                trigger=str(envelope.parsed_args.get("trigger", "idle")),
                mode=envelope.parsed_args.get("mode"),
            )
        if key == ("endogenous", "status"):
            return self.endogenous_status()
        if key == ("initiative", "status"):
            return self.initiative_status()
        if key == ("initiative", "distribution"):
            return self.controller.initiative_distribution()
        if key == ("initiative", "trigger"):
            return self.controller.initiative_trigger_now(
                trigger=str(envelope.parsed_args.get("trigger", "idle")),
                mode=envelope.parsed_args.get("mode"),
                force=bool(envelope.parsed_args.get("force", False)),
            )
        if key == ("initiative", "config"):
            return self.controller.initiative_status()
        if key == ("initiative", "why"):
            return self.controller.initiative_why(envelope.target or "last")
        if key == ("monologue", "status"):
            return self.controller.monologue_status()
        if key == ("monologue", "show"):
            return self.controller.monologue_show(envelope.parsed_args.get("limit"))
        if key == ("thought", "show"):
            return self.controller.thought_snapshot(envelope.target or "last")
        if key == ("agent", "list"):
            return self.controller.agent_list()
        if key == ("skill", "stats"):
            return self.controller.skill_stats()
        if key == ("skill", "profile") and envelope.target:
            return self.controller.skill_profile(envelope.target)
        if key == ("replay", "motivation"):
            round_id = self.controller.resolve_round_ref(envelope.target or "last")
            return self.controller.replay_motivation(round_id)
        if key == ("why", "motivation"):
            return self.controller.why_motivation(envelope.target or "last")
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
