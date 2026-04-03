from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import yaml

from nalr.agents.modules import build_agents
from nalr.memory.store import MemoryStore
from nalr.output.style import compute_style_profile
from nalr.runtime.adapters import adapt_proposal, adapt_skill_spec
from nalr.runtime.decision import (
    BehaviorPlausibilityGuard,
    ConflictMonitorAgent,
    ForcedModeSwitch,
    ValueAgent,
    sample_action,
    summarize_counts,
)
from nalr.runtime.model_gateway import ModelGateway
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


def _operator_level(command: str) -> str:
    if command.startswith(("safe ", "checkpoint ", "mode set")):
        return "ops_admin"
    if command.startswith(("agent ", "debug ")):
        return "debug_control"
    if command.startswith(("body rest", "mood calm")):
        return "soft_intervene"
    return "read_only"


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
        self.model_gateway = ModelGateway.from_config(self.config["models"])
        self.value_agent = ValueAgent()
        self.conflict_agent = ConflictMonitorAgent()
        self.plausibility_guard = BehaviorPlausibilityGuard()
        self.forced_mode_switch = ForcedModeSwitch()

        if not self.state_path.exists():
            initial_state = RuntimeState(
                agents_enabled={
                    name: agent_cfg.get("enabled", True)
                    for name, agent_cfg in self.config["agents"]["agents"].items()
                },
                agent_weights={
                    name: agent_cfg.get("weight", 1.0)
                    for name, agent_cfg in self.config["agents"]["agents"].items()
                },
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
            "models": read_yaml("models.yaml")["models"],
        }

    def _save_state(self, state: RuntimeState) -> None:
        self.state_path.write_text(json.dumps(to_dict(state), ensure_ascii=False, indent=2), encoding="utf-8")

    def load_runtime_state(self) -> RuntimeState:
        payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        return RuntimeState(**payload)

    def _state_hash(self, state: RuntimeState) -> str:
        return hashlib.sha1(json.dumps(to_dict(state), sort_keys=True).encode("utf-8")).hexdigest()

    def _candidate_from_state(self, state_payload: dict[str, Any]) -> RuntimeState:
        return RuntimeState(**state_payload)

    def _append_history(self, items: list[str], value: str, limit: int = 8) -> list[str]:
        updated = [*items, value]
        return updated[-limit:]

    def _build_context(self, event: RoundEvent, state: RuntimeState, scenario_cfg: dict[str, Any], allow_detail: bool = True) -> dict[str, Any]:
        cue = self.memory_store.ingest_event(event)
        recall = self.memory_store.recall(cue, allow_detail=allow_detail)
        model_plan = self.model_gateway.plan(f"Scenario={scenario_cfg}; user={event.content}")
        return {
            "cue": cue,
            "recall_strength": self.memory_store.recall_strength(cue),
            "recall": recall,
            "habit_strength": self.memory_store.habit_strength(cue),
            "closeness": self.memory_store.closeness(event.target),
            "body_energy": state.body_energy,
            "pfc_model_text": model_plan["text"],
            "pfc_model_provider": model_plan["provider"],
            "pfc_model_name": model_plan["model"],
        }

    def _collect_proposals(self, event: RoundEvent, state: RuntimeState, scenario_cfg: dict[str, Any], context: dict[str, Any], resample_idx: int = 0) -> tuple[dict[str, float], list[AgentContribution], list[dict[str, Any]]]:
        scores: dict[str, float] = {}
        contributions: list[AgentContribution] = []
        raw: list[dict[str, Any]] = []
        safe_mode_blocked = {"DMNAgent", "PerspectiveModel"}
        for agent in self.agents:
            if not state.agents_enabled.get(agent.name, True):
                continue
            if state.safe_mode and agent.name in safe_mode_blocked:
                continue
            proposal = agent.propose(event, state, scenario_cfg, context)
            weight = state.agent_weights.get(agent.name, 1.0)
            adapted = adapt_proposal(proposal, weight=weight, resample_idx=resample_idx)
            raw.append(adapted)
            if adapted["veto"] or not adapted["action_preferences"]:
                continue
            contributions.append(
                AgentContribution(
                    agent_name=adapted["agent_name"],
                    action_name=adapted["top_action"],
                    score=adapted["top_score"],
                    reason=adapted["reason"],
                    delta_p=adapted["top_score"],
                    sigma_scale=adapted["sigma_scale"],
                    confidence=adapted["confidence"],
                    weight_applied=adapted["weight_applied"],
                    resample_idx=adapted["resample_idx"],
                    selected=False,
                    latency_ms=adapted["latency_ms"],
                    provider=adapted["provider"],
                    model=adapted["model"],
                    tags=list(adapted["trace_tags"]),
                )
            )
            for action_name, score in adapted["action_preferences"].items():
                scores[action_name] = scores.get(action_name, 0.0) + (score * weight)
        return scores, contributions, raw

    def _run_decision_loop(
        self,
        event: RoundEvent,
        state: RuntimeState,
        scenario_name: str,
        scenario_cfg: dict[str, Any],
        context: dict[str, Any],
        seed: int | None = None,
    ) -> dict[str, Any]:
        scores, contributions, raw_proposals = self._collect_proposals(event, state, scenario_cfg, context)
        scores, value_reason = self.value_agent.apply(scores, scenario_name, to_dict(state), context)
        contributions.append(
            AgentContribution(
                agent_name="ValueAgent",
                action_name=max(scores, key=scores.get) if scores else "respond",
                score=round(max(scores.values()) if scores else 0.0, 4),
                reason=value_reason,
                confidence=0.72,
                weight_applied=state.agent_weights.get("ValueAgent", 0.9),
                provider="rule",
                model="value-heuristic",
                tags=["value"],
            )
        )

        scores, forced_switch = self.forced_mode_switch.apply(
            scores,
            state.recent_actions,
            self.config["thresholds"]["thresholds"]["max_mode_lock_rounds"],
        )
        if forced_switch:
            contributions.append(
                AgentContribution(
                    agent_name="ForcedModeSwitch",
                    action_name=max(scores, key=scores.get),
                    score=round(scores[max(scores, key=scores.get)], 4),
                    reason="forced focus switch",
                    confidence=0.65,
                    weight_applied=state.agent_weights.get("ForcedModeSwitch", 0.8),
                    provider="rule",
                    model="forced-mode",
                    tags=["guard"],
                )
            )

        conflict_score, resample_limit = self.conflict_agent.score(scores, raw_proposals, context)
        if state.safe_mode:
            resample_limit = 0
        candidate_name, distribution = sample_action(scores, seed=seed)
        plausibility_fail = self.plausibility_guard.score(candidate_name, scenario_name, to_dict(state), {**context, "style_name": "neutral"})
        resample_count = 0

        while resample_count < resample_limit and plausibility_fail >= 0.70:
            scores, _ = self.plausibility_guard.apply(scores, candidate_name, plausibility_fail)
            resample_count += 1
            candidate_name, distribution = sample_action(scores, seed=(None if seed is None else seed + resample_count))
            plausibility_fail = self.plausibility_guard.score(candidate_name, scenario_name, to_dict(state), {**context, "style_name": "neutral"})

        scores, guard_reason = self.plausibility_guard.apply(scores, candidate_name, plausibility_fail)
        candidate_name, distribution = sample_action(scores, seed=(None if seed is None else seed + 99))
        contributions.append(
            AgentContribution(
                agent_name="ConflictMonitorAgent",
                action_name=candidate_name,
                score=conflict_score,
                reason=f"conflict={conflict_score}",
                confidence=0.70,
                weight_applied=state.agent_weights.get("ConflictMonitorAgent", 0.9),
                provider="rule",
                model="conflict-heuristic",
                tags=["conflict"],
            )
        )
        contributions.append(
            AgentContribution(
                agent_name="BehaviorPlausibilityGuard",
                action_name=candidate_name,
                score=plausibility_fail,
                reason=guard_reason,
                confidence=0.76,
                weight_applied=state.agent_weights.get("BehaviorPlausibilityGuard", 1.0),
                provider="rule",
                model="guard-heuristic",
                tags=["guard"],
            )
        )
        for item in contributions:
            if item.action_name == candidate_name:
                item.selected = True

        probability = round(distribution.get(candidate_name, 0.0), 4)
        rationale = "; ".join(item.reason for item in sorted(contributions, key=lambda item: item.score, reverse=True)[:3])
        return {
            "candidate": ActionCandidate(name=candidate_name, probability=probability, rationale=rationale, metadata={}),
            "distribution": distribution,
            "contributions": sorted(contributions, key=lambda item: item.score, reverse=True),
            "conflict_score": conflict_score,
            "plausibility_fail_score": plausibility_fail,
            "resample_count": resample_count,
        }

    def tick(self, event: RoundEvent, scenario: str, mode: str) -> RoundResult:
        state = self.load_runtime_state()
        pre_state = to_dict(state)
        requested_mode = "safe" if state.safe_mode else mode
        scenario_cfg = self.config["scenarios"]["scenarios"][scenario]

        state.mode = requested_mode
        state.round_count += 1
        state.body_energy = _clip(state.body_energy + event.energy_delta)
        state.mood = _clip(state.mood + (event.valence * 0.1))
        state.budget_remaining = _clip(state.budget_remaining - 0.002 + (0.01 if requested_mode in {"idle", "sleep"} else 0.0))
        context = self._build_context(event, state, scenario_cfg)
        decision = self._run_decision_loop(event, state, scenario, scenario_cfg, context)
        sampled_action = decision["candidate"]
        contributions = decision["contributions"]
        state.focus = sampled_action.name
        state.last_action = sampled_action.name
        state.recent_actions = self._append_history(state.recent_actions, sampled_action.name)
        state.recent_modes = self._append_history(state.recent_modes, state.mode)
        state.last_conflict_score = decision["conflict_score"]
        state.last_plausibility_fail_score = decision["plausibility_fail_score"]

        starvation = self.config["resource_rules"]["resource_defaults"]["starvation_threshold"]
        if state.budget_remaining <= starvation / 10:
            state.safe_mode = True
            state.mode = "safe"

        style_profile = compute_style_profile(to_dict(state), scenario, self.config["output_style"]["styles"])
        sampled_action.metadata["style_profile"] = style_profile
        render_result = self.model_gateway.render(sampled_action.name, style_profile, event.content)
        sampled_action.metadata["rendered_output"] = render_result["text"]
        sampled_action.metadata["provider"] = render_result["provider"]
        sampled_action.metadata["model"] = render_result["model"]
        state.last_render_provider = render_result["provider"]
        state.last_render_model = render_result["model"]

        trace = RoundTrace(
            round_id=state.round_count,
            scenario=scenario,
            mode=state.mode,
            sampled_action=sampled_action.name,
            contributions=contributions,
            top_drivers=contributions[:3],
            style_profile=style_profile,
            state_snapshot=to_dict(state),
            pre_state_snapshot=pre_state,
            event_payload=to_dict(event),
            decision_context=context,
            candidate_distribution=decision["distribution"],
            conflict_score=decision["conflict_score"],
            plausibility_fail_score=decision["plausibility_fail_score"],
            resample_count=decision["resample_count"],
            rendered_output=render_result["text"],
            provider=render_result["provider"],
            model=render_result["model"],
        )

        trace.state_snapshot["conflict_score"] = decision["conflict_score"]
        trace.state_snapshot["plausibility_fail_score"] = decision["plausibility_fail_score"]
        trace.state_snapshot["last_render_provider"] = render_result["provider"]
        trace.state_snapshot["last_render_model"] = render_result["model"]

        health = HealthEvent(event="tick", status="ok", detail=f"budget={state.budget_remaining:.2f}; provider={render_result['provider']}")

        self._save_state(state)
        self.trace_store.write_round(trace)

        return RoundResult(
            round_id=state.round_count,
            sampled_action=sampled_action,
            trace=trace,
            state=state,
            health=health,
            rendered_output=render_result["text"],
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
        elif parts[:2] == ["debug", "weight"] and len(parts) == 4:
            weight = float(parts[3])
            state.agent_weights[parts[2]] = weight
            result = CommandResult(
                applied=True,
                scope=parts[2],
                delta={"weight": weight},
                ttl="until changed",
                rollback_hint=f"alive debug weight {parts[2]} 1.0",
            )
        elif parts[:2] == ["body", "rest"]:
            state.body_energy = _clip(state.body_energy + 0.15)
            state.budget_remaining = _clip(state.budget_remaining + 0.03)
            result = CommandResult(
                applied=True,
                scope="body",
                delta={"body_energy": state.body_energy, "budget_remaining": state.budget_remaining},
                ttl="1 round",
                risk_note="may reduce responsiveness",
                rollback_hint="alive mode set interactive",
            )
        elif parts[:2] == ["mood", "calm"]:
            state.mood = round(state.mood + ((0.55 - state.mood) * 0.35), 4)
            result = CommandResult(
                applied=True,
                scope="mood",
                delta={"mood": state.mood},
                ttl="1 round",
                risk_note="may reduce emotional variance",
                rollback_hint="allow a few rounds of natural drift",
            )

        self._save_state(state)
        after_hash = self._state_hash(state)
        self.trace_store.append_command(command, result, before_hash, after_hash, operator_level=_operator_level(command))
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
        self.trace_store.append_command(
            f"checkpoint rewind {checkpoint_id}",
            result,
            before_hash,
            after_hash,
            operator_level="ops_admin",
        )
        return result

    def memory_top(self, limit: int = 5) -> list[dict]:
        return self.memory_store.memory_top(limit=limit)

    def habit_top(self, limit: int = 5) -> list[dict]:
        return self.memory_store.habit_top(limit=limit)

    def relation_show(self, target: str) -> dict[str, Any]:
        return self.memory_store.relation_state(target)

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
            "rendered_output": trace.get("rendered_output", ""),
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

    def why_not(self, round_id: int, action: str) -> dict[str, Any]:
        trace = self.trace_round(round_id)
        candidate_distribution = trace.get("candidate_distribution", {})
        blocked_by = []
        if action not in candidate_distribution:
            blocked_by.append("not_proposed")
        if trace.get("plausibility_fail_score", 0.0) >= self.config["thresholds"]["thresholds"]["plausibility_fail"]:
            blocked_by.append("plausibility_guard")
        blocked_by.extend(item["agent_name"] for item in trace.get("top_drivers", [])[:2])
        return {
            "round_id": round_id,
            "action": action,
            "selected_action": trace["sampled_action"],
            "candidate_score": candidate_distribution.get(action, 0.0),
            "blocked_by": blocked_by,
        }

    def replay(self, round_id: int, seed: int = 0) -> dict[str, Any]:
        trace = self.trace_round(round_id)
        event = RoundEvent(**trace["event_payload"])
        replay_state = self._candidate_from_state(trace["pre_state_snapshot"])
        scenario_cfg = self.config["scenarios"]["scenarios"][trace["scenario"]]
        context = dict(trace.get("decision_context", {}))
        decision = self._run_decision_loop(event, replay_state, trace["scenario"], scenario_cfg, context, seed=seed)
        ablations = []
        for agent_name in ("DMNAgent", "HippocampusAgent", "PerspectiveModel"):
            replay_state_ablate = self._candidate_from_state(trace["pre_state_snapshot"])
            replay_state_ablate.agents_enabled[agent_name] = False
            ablated = self._run_decision_loop(event, replay_state_ablate, trace["scenario"], scenario_cfg, context, seed=seed)
            ablations.append(
                {
                    "agent": agent_name,
                    "action": ablated["candidate"].name,
                    "delta": round(
                        ablated["distribution"].get(ablated["candidate"].name, 0.0)
                        - trace.get("candidate_distribution", {}).get(trace["sampled_action"], 0.0),
                        4,
                    ),
                }
            )
        return {
            "round_id": round_id,
            "original_action": trace["sampled_action"],
            "replayed_action": decision["candidate"].name,
            "candidate_distribution": decision["distribution"],
            "ablations": ablations,
        }

    def what_changed(self, window: int = 5) -> dict[str, Any]:
        rounds = self.trace_store.list_rounds()[-window:]
        if not rounds:
            return {"window": window, "action_counts": {}, "mode_counts": {}, "budget_delta": 0.0}
        budget_delta = rounds[-1]["state_snapshot"]["budget_remaining"] - rounds[0]["state_snapshot"]["budget_remaining"]
        return {
            "window": window,
            "action_counts": summarize_counts([item["sampled_action"] for item in rounds]),
            "mode_counts": summarize_counts([item["mode"] for item in rounds]),
            "budget_delta": round(budget_delta, 4),
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

    def metrics_timeline(self) -> dict[str, Any]:
        rounds = self.trace_store.list_rounds()
        return {
            "rounds": [
                {
                    "round_id": item["round_id"],
                    "sampled_action": item["sampled_action"],
                    "mode": item["mode"],
                    "conflict_score": item.get("conflict_score", 0.0),
                    "plausibility_fail_score": item.get("plausibility_fail_score", 0.0),
                    "budget_remaining": item["state_snapshot"].get("budget_remaining", 0.0),
                }
                for item in rounds
            ]
        }

    def metrics_heatmap(self) -> dict[str, Any]:
        rounds = self.trace_store.list_rounds()
        actions = sorted({item["sampled_action"] for item in rounds} or {"respond"})
        matrix: dict[str, dict[str, int]] = {}
        for item in rounds:
            for contribution in item.get("contributions", []):
                matrix.setdefault(contribution["agent_name"], {action: 0 for action in actions})
                matrix[contribution["agent_name"]].setdefault(contribution["action_name"], 0)
                matrix[contribution["agent_name"]][contribution["action_name"]] += 1
        return {"actions": actions, "matrix": matrix}

    def compact_traces(self) -> dict[str, Any]:
        return self.trace_store.compact_rounds()

    def eval_longrun(self, rounds: int = 1000) -> dict[str, Any]:
        start_round = self.load_runtime_state().round_count
        for idx in range(rounds):
            self.tick(
                RoundEvent(
                    source="simulation",
                    content=f"synthetic round {idx} task update",
                    target="sim-user",
                    cue=f"topic-{idx % 7}",
                    valence=0.05 if idx % 2 == 0 else -0.02,
                ),
                scenario="task" if idx % 3 == 0 else "chat",
                mode="interactive",
            )
        generated_rounds = self.trace_store.list_rounds()[start_round:]
        summary = self.metrics_summary()
        total = len(generated_rounds) or 1
        critical_conflicts = [
            item for item in generated_rounds if item.get("conflict_score", 0.0) >= self.config["thresholds"]["thresholds"]["conflict_critical"]
        ]
        safe_mode_rounds = sum(1 for item in generated_rounds if item["state_snapshot"].get("safe_mode"))
        scarce_threshold = self.config["resource_rules"]["resource_defaults"]["scarce_threshold"]
        scarce_round = next(
            (idx for idx, item in enumerate(generated_rounds) if item["state_snapshot"].get("budget_remaining", 1.0) <= scarce_threshold),
            None,
        )
        scarcity_burn_drop = 0.0
        if scarce_round is not None and scarce_round >= 3 and scarce_round + 3 < len(generated_rounds):
            before = generated_rounds[scarce_round - 3 : scarce_round]
            after = generated_rounds[scarce_round : scarce_round + 3]
            before_burn = before[0]["state_snapshot"]["budget_remaining"] - before[-1]["state_snapshot"]["budget_remaining"]
            after_burn = after[0]["state_snapshot"]["budget_remaining"] - after[-1]["state_snapshot"]["budget_remaining"]
            if before_burn > 0:
                scarcity_burn_drop = round(max(0.0, (before_burn - after_burn) / before_burn), 4)
        recall_gist = sum(1 for item in generated_rounds if item.get("decision_context", {}).get("recall", {}).get("mode") == "gist")
        recall_detail = sum(1 for item in generated_rounds if item.get("decision_context", {}).get("recall", {}).get("mode") == "detail")
        task_rounds = [item for item in generated_rounds if item["scenario"] == "task"]
        task_successes = sum(1 for item in task_rounds if item["sampled_action"] in {"respond", "plan", "recall", "clarify"})
        relation_checks = []
        for item in generated_rounds:
            target = item.get("event_payload", {}).get("target")
            if not target:
                continue
            closeness = item.get("decision_context", {}).get("closeness", 0.5)
            action = item["sampled_action"]
            if closeness >= 0.55:
                relation_checks.append(action in {"connect", "clarify", "respond", "recall"})
            else:
                relation_checks.append(action != "connect")
        habit_strengths = [item["strength"] for item in self.habit_top(limit=10)]
        summary["generated_rounds"] = total
        summary["crash_rate"] = 0.0
        summary["safe_mode_rate"] = round(safe_mode_rounds / total, 4)
        summary["conflict_deadloop_rate"] = round(len(critical_conflicts) / total, 4)
        summary["scarcity_burn_drop"] = scarcity_burn_drop
        summary["habit_gradient"] = round((sum(habit_strengths) / max(len(habit_strengths), 1)) / total, 4)
        summary["gist_detail_ratio"] = round(recall_gist / max(recall_detail, 1), 4)
        summary["relation_consistency"] = round(sum(1 for item in relation_checks if item) / max(len(relation_checks), 1), 4)
        summary["task_success_rate"] = round(task_successes / max(len(task_rounds), 1), 4)
        summary["top_driver_coverage"] = round(sum(1 for item in generated_rounds if len(item.get("top_drivers", [])) >= 3) / total, 4)
        return summary

    def agent_list(self) -> list[dict[str, Any]]:
        state = self.load_runtime_state()
        agent_cfg = self.config["agents"]["agents"]
        return [
            {
                "name": name,
                "enabled": state.agents_enabled.get(name, cfg.get("enabled", True)),
                "weight": state.agent_weights.get(name, cfg.get("weight", 1.0)),
            }
            for name, cfg in agent_cfg.items()
        ]

    def skill_list(self) -> list[dict[str, Any]]:
        return [adapt_skill_spec(spec) for spec in self.skills.values()]

    def skill_stats(self) -> dict[str, Any]:
        rounds = self.trace_store.list_rounds()
        skill_rows = []
        for spec in self.skills.values():
            adapted = adapt_skill_spec(spec)
            observed = []
            observed_rounds = set()
            for trace in rounds:
                for contribution in trace.get("contributions", []):
                    if contribution["agent_name"] != adapted["owner_module"]:
                        continue
                    observed.append(contribution)
                    observed_rounds.add(trace["round_id"])
            skill_rows.append(
                {
                    "skill_name": adapted["name"],
                    "owner_module": adapted["owner_module"],
                    "timeout_ms": adapted["timeout_ms"],
                    "cost_class": adapted["cost_class"],
                    "failure_policy": adapted["failure_policy"],
                    "trace_tags": adapted["trace_tags"],
                    "observed_rounds": len(observed_rounds),
                    "contribution_hits": len(observed),
                    "selected_hits": sum(1 for item in observed if item.get("selected")),
                    "average_latency_ms": round(
                        sum(item.get("latency_ms", 0) for item in observed) / max(len(observed), 1),
                        4,
                    ),
                }
            )
        return {"skills": skill_rows, "total_skills": len(skill_rows)}

    def skill_profile(self, skill_name: str) -> dict[str, Any]:
        if skill_name not in self.skills:
            raise FileNotFoundError(f"skill {skill_name} not found")
        adapted = adapt_skill_spec(self.skills[skill_name])
        observed = []
        observed_rounds = set()
        for trace in self.trace_store.list_rounds():
            for contribution in trace.get("contributions", []):
                if contribution["agent_name"] != adapted["owner_module"]:
                    continue
                observed.append(contribution)
                observed_rounds.add(trace["round_id"])
        return {
            **adapted,
            "observed_rounds": len(observed_rounds),
            "selected_hits": sum(1 for item in observed if item.get("selected")),
            "providers": sorted({item.get("provider", "upstream_contract") for item in observed} or {"upstream_contract"}),
        }
