from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from nalr.agents.modules import build_agents
from nalr.memory.store import MemoryStore
from nalr.output.style import compute_style_profile
from nalr.schemas.models import (
    ActionCandidate,
    AgentContribution,
    CheckpointRef,
    CommandResult,
    HealthEvent,
    Proposal,
    RoundEvent,
    RoundResult,
    RoundTrace,
    RuntimeState,
    to_dict,
)
from nalr.skills.registry import build_skill_registry
from nalr.trace.store import TraceStore


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


class RuntimeController:
    def __init__(self, project_root: Path, config_root: Path | None = None, home_path: Path | None = None) -> None:
        self.project_root = Path(project_root)
        self.config_root = Path(config_root) if config_root else self.project_root / "config"
        self.home_path = Path(home_path) if home_path else self.project_root / ".alive"
        self.runtime_dir = self.home_path / "runtime"
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.runtime_dir / "persona_state.json"
        self.checkpoint_dir = self.runtime_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.config = self._load_config()
        self.trace_store = TraceStore(self.home_path)
        self.memory_store = MemoryStore(self.home_path)
        self.agents = build_agents()
        self.skills = build_skill_registry()

        if not self.state_path.exists():
            initial_state = RuntimeState(
                agents_enabled={
                    name: agent_cfg.get("enabled", True)
                    for name, agent_cfg in self.config["agents"]["agents"].items()
                }
            )
            self._save_state(initial_state)

    def _load_config(self) -> dict[str, Any]:
        def read_yaml(name: str) -> dict[str, Any]:
            path = self.config_root / name
            return yaml.safe_load(path.read_text(encoding="utf-8"))

        return {
            "agents": read_yaml("agents.yaml"),
            "modes": read_yaml("modes.yaml"),
            "scenarios": read_yaml("scenarios.yaml"),
            "thresholds": read_yaml("thresholds.yaml"),
            "temperament": read_yaml("temperament.yaml"),
            "resource_rules": read_yaml("resource_rules.yaml"),
            "output_style": read_yaml("output_style.yaml"),
        }

    def _save_state(self, state: RuntimeState) -> None:
        self.state_path.write_text(json.dumps(to_dict(state), ensure_ascii=False, indent=2), encoding="utf-8")

    def load_runtime_state(self) -> RuntimeState:
        payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        return RuntimeState(**payload)

    def _state_hash(self, state: RuntimeState) -> str:
        return hashlib.sha1(json.dumps(to_dict(state), sort_keys=True).encode("utf-8")).hexdigest()

    def _aggregate(self, proposals: list[Proposal], state: RuntimeState) -> tuple[ActionCandidate, list[AgentContribution]]:
        scores: dict[str, float] = {}
        contributions: list[AgentContribution] = []

        for proposal in proposals:
            if proposal.veto:
                continue
            if not proposal.action_preferences:
                continue
            top_action = max(proposal.action_preferences, key=proposal.action_preferences.get)
            top_score = proposal.action_preferences[top_action]
            contributions.append(
                AgentContribution(
                    agent_name=proposal.agent_name,
                    action_name=top_action,
                    score=round(top_score, 4),
                    reason=proposal.reason,
                )
            )
            for action_name, score in proposal.action_preferences.items():
                scores[action_name] = scores.get(action_name, 0.0) + score

        if not scores:
            scores["respond"] = 0.1

        if state.safe_mode:
            scores["respond"] = scores.get("respond", 0.0) + 0.2

        total = sum(max(score, 0.0) for score in scores.values()) or 1.0
        sampled_name = max(scores, key=scores.get)
        probability = round(scores[sampled_name] / total, 4)
        rationale = "; ".join(item.reason for item in sorted(contributions, key=lambda item: item.score, reverse=True)[:3])
        candidate = ActionCandidate(name=sampled_name, probability=probability, rationale=rationale)
        return candidate, sorted(contributions, key=lambda item: item.score, reverse=True)

    def tick(self, event: RoundEvent, scenario: str, mode: str) -> RoundResult:
        state = self.load_runtime_state()
        cue = self.memory_store.ingest_event(event)

        requested_mode = "safe" if state.safe_mode else mode
        mode_cfg = self.config["modes"]["modes"].get(requested_mode, self.config["modes"]["modes"]["interactive"])
        scenario_cfg = self.config["scenarios"]["scenarios"][scenario]

        state.mode = requested_mode
        state.round_count += 1
        state.body_energy = _clip(state.body_energy + event.energy_delta)
        state.mood = _clip(state.mood + (event.valence * 0.1))
        state.budget_remaining = _clip(state.budget_remaining - 0.002 + (0.01 if requested_mode in {"idle", "sleep"} else 0.0))

        context = {
            "cue": cue,
            "recall_strength": self.memory_store.recall_strength(cue),
            "habit_strength": self.memory_store.habit_strength(cue),
            "closeness": self.memory_store.closeness(event.target),
        }

        proposals = []
        for agent in self.agents:
            if not state.agents_enabled.get(agent.name, True):
                continue
            proposal = agent.propose(event, state, scenario_cfg, context)
            proposals.append(proposal)

        sampled_action, contributions = self._aggregate(proposals, state)
        state.focus = sampled_action.name
        state.last_action = sampled_action.name

        starvation = self.config["resource_rules"]["resource_defaults"]["starvation_threshold"]
        if state.budget_remaining <= starvation / 10:
            state.safe_mode = True
            state.mode = "safe"

        style_profile = compute_style_profile(to_dict(state), scenario, self.config["output_style"]["styles"])
        sampled_action.metadata["style_profile"] = style_profile

        trace = RoundTrace(
            round_id=state.round_count,
            scenario=scenario,
            mode=state.mode,
            sampled_action=sampled_action.name,
            contributions=contributions,
            top_drivers=contributions[:3],
            style_profile=style_profile,
            state_snapshot=to_dict(state),
        )

        health = HealthEvent(event="tick", status="ok", detail=f"budget={state.budget_remaining:.2f}")

        self._save_state(state)
        self.trace_store.write_round(trace)

        return RoundResult(
            round_id=state.round_count,
            sampled_action=sampled_action,
            trace=trace,
            state=state,
            health=health,
        )

    def apply_command(self, command: str) -> CommandResult:
        state = self.load_runtime_state()
        before_hash = self._state_hash(state)
        parts = command.split()
        result = CommandResult(applied=False, scope="runtime", delta={})

        if parts[:2] == ["safe", "on"]:
            state.safe_mode = True
            state.mode = "safe"
            result = CommandResult(
                applied=True,
                scope="runtime",
                delta={"safe_mode": True, "mode": "safe"},
                risk_note="reduces spontaneity",
                rollback_hint="alive safe off",
            )
        elif parts[:2] == ["safe", "off"]:
            state.safe_mode = False
            state.mode = "interactive"
            result = CommandResult(
                applied=True,
                scope="runtime",
                delta={"safe_mode": False, "mode": "interactive"},
                risk_note="restores full runtime variability",
                rollback_hint="alive safe on",
            )
        elif parts[:2] == ["mode", "set"] and len(parts) == 3:
            target_mode = parts[2]
            state.mode = target_mode
            if target_mode != "safe":
                state.safe_mode = False
            result = CommandResult(
                applied=True,
                scope="runtime",
                delta={"mode": target_mode},
                rollback_hint="alive mode set interactive",
            )
        elif parts[:2] == ["agent", "disable"] and len(parts) == 3:
            state.agents_enabled[parts[2]] = False
            result = CommandResult(
                applied=True,
                scope=parts[2],
                delta={"enabled": False},
                ttl="until re-enabled",
                rollback_hint=f"alive agent enable {parts[2]}",
            )
        elif parts[:2] == ["agent", "enable"] and len(parts) == 3:
            state.agents_enabled[parts[2]] = True
            result = CommandResult(
                applied=True,
                scope=parts[2],
                delta={"enabled": True},
                ttl="until changed",
                rollback_hint=f"alive agent disable {parts[2]}",
            )

        self._save_state(state)
        after_hash = self._state_hash(state)
        self.trace_store.append_command(command, result, before_hash, after_hash)
        return result

    def checkpoint(self) -> CheckpointRef:
        state = self.load_runtime_state()
        checkpoint_id = f"ckpt-{state.round_count:04d}"
        path = self.checkpoint_dir / f"{checkpoint_id}.json"
        shutil.copyfile(self.state_path, path)
        state.last_checkpoint_id = checkpoint_id
        self._save_state(state)
        return CheckpointRef(checkpoint_id=checkpoint_id, path=path)

    def rewind(self, checkpoint_id: str) -> CommandResult:
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.json"
        if not checkpoint_path.exists():
            return CommandResult(
                applied=False,
                scope="checkpoint",
                delta={"checkpoint_id": checkpoint_id, "restored": False},
                risk_note="checkpoint not found",
            )

        before_state = self.load_runtime_state()
        before_hash = self._state_hash(before_state)
        shutil.copyfile(checkpoint_path, self.state_path)
        restored_state = self.load_runtime_state()
        after_hash = self._state_hash(restored_state)
        result = CommandResult(
            applied=True,
            scope="checkpoint",
            delta={"checkpoint_id": checkpoint_id, "restored": True, "safe_mode": restored_state.safe_mode, "mode": restored_state.mode},
            rollback_hint="create a fresh checkpoint before further changes",
        )
        self.trace_store.append_command(f"checkpoint rewind {checkpoint_id}", result, before_hash, after_hash)
        return result

    def memory_top(self, limit: int = 5) -> list[dict]:
        return self.memory_store.memory_top(limit=limit)

    def habit_top(self, limit: int = 5) -> list[dict]:
        return self.memory_store.habit_top(limit=limit)

    def state_payload(self) -> dict[str, Any]:
        return to_dict(self.load_runtime_state())

    def trace_round(self, round_id: int) -> dict[str, Any]:
        return self.trace_store.read_round(round_id)

    def why_this(self, round_id: int) -> dict[str, Any]:
        trace = self.trace_round(round_id)
        return {
            "round_id": trace["round_id"],
            "sampled_action": trace["sampled_action"],
            "top_drivers": trace["top_drivers"],
            "style_profile": trace["style_profile"],
            "state_snapshot": {
                "mode": trace["state_snapshot"]["mode"],
                "safe_mode": trace["state_snapshot"]["safe_mode"],
                "focus": trace["state_snapshot"]["focus"],
                "budget_remaining": trace["state_snapshot"]["budget_remaining"],
            },
        }

    def contribution_breakdown(self, round_id: int) -> dict[str, Any]:
        trace = self.trace_round(round_id)
        return {
            "round_id": trace["round_id"],
            "sampled_action": trace["sampled_action"],
            "contributions": trace["contributions"],
        }

    def metrics_summary(self) -> dict[str, Any]:
        rounds = self.trace_store.list_rounds()
        sampled_actions: dict[str, int] = {}
        modes: dict[str, int] = {}
        driver_counts: dict[str, int] = {}
        budget_values: list[float] = []
        safe_mode_rounds = 0

        for trace in rounds:
            sampled_actions[trace["sampled_action"]] = sampled_actions.get(trace["sampled_action"], 0) + 1
            modes[trace["mode"]] = modes.get(trace["mode"], 0) + 1
            budget_values.append(trace["state_snapshot"]["budget_remaining"])
            if trace["state_snapshot"]["safe_mode"]:
                safe_mode_rounds += 1
            for driver in trace["top_drivers"]:
                driver_counts[driver["agent_name"]] = driver_counts.get(driver["agent_name"], 0) + 1

        top_agents = [
            {"agent_name": name, "count": count}
            for name, count in sorted(driver_counts.items(), key=lambda item: item[1], reverse=True)[:5]
        ]
        avg_budget = round(sum(budget_values) / len(budget_values), 4) if budget_values else 0.0
        return {
            "total_rounds": len(rounds),
            "sampled_actions": sampled_actions,
            "mode_counts": modes,
            "safe_mode_rounds": safe_mode_rounds,
            "average_budget_remaining": avg_budget,
            "top_agents": top_agents,
        }

    def agent_list(self) -> list[dict[str, Any]]:
        state = self.load_runtime_state()
        agent_cfg = self.config["agents"]["agents"]
        return [
            {
                "name": name,
                "enabled": state.agents_enabled.get(name, cfg.get("enabled", True)),
                "weight": cfg.get("weight", 1.0),
            }
            for name, cfg in agent_cfg.items()
        ]

    def skill_list(self) -> list[dict[str, Any]]:
        return [asdict(spec) for spec in self.skills.values()]
