from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from nalr.dream.engine import OneiroiAgent
from nalr.dream.store import DreamStore
from nalr.schemas.models import DreamGuardDecision, DreamRunRequest, DreamSnapshot, to_dict


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


class DreamOrchestrator:
    def __init__(self, *, project_root: Path, home_path: Path, config: dict[str, Any], memory_store, vitality_engine) -> None:
        self.project_root = Path(project_root)
        self.home_path = Path(home_path)
        self.config = config
        self.memory_store = memory_store
        self.vitality_engine = vitality_engine
        self.store = DreamStore(self.home_path)

    def enabled(self, state) -> bool:
        override = state.session_metadata.get("dream_enabled_override")
        if override is not None:
            return bool(override)
        return bool(self.config.get("enabled", True))

    def status(self, state) -> dict[str, Any]:
        metrics = self.store.metrics()
        latest = metrics.get("latest_run_id")
        return {
            "enabled": self.enabled(state),
            "latest_run_id": latest,
            "total_runs": metrics["total_runs"],
            "runs_by_trigger": metrics["runs_by_trigger"],
        }

    def metrics(self) -> dict[str, Any]:
        return self.store.metrics()

    def list_runs(self) -> list[dict[str, Any]]:
        return self.store.list_runs()

    def read_run(self, run_ref: str | None = None) -> dict[str, Any]:
        return self.store.read_run(run_ref)

    def build_snapshot(self, *, state, mode: str, trigger: str, cue: str | None, relation_state: dict[str, Any]) -> DreamSnapshot:
        identity_evidence = self.memory_store.identity_evidence()
        top_memories = self.memory_store.memory_top(limit=5)
        hot = [item.get("cue") for item in top_memories if item.get("tier") == "hot" and item.get("cue")]
        warm = [item.get("cue") for item in top_memories if item.get("tier") == "warm" and item.get("cue")]
        archive = [item.get("cue") for item in top_memories if item.get("tier") == "archive" and item.get("cue")]
        effective_cue = cue or (hot[0] if hot else None) or (warm[0] if warm else None)
        return DreamSnapshot(
            sleep_session_id=f"{mode}-{state.session_id[:8]}-{state.round_count}",
            mode=mode,
            trigger=trigger,
            memory_refs={
                "hot": [item for item in [effective_cue, *hot] if item][:3],
                "warm": warm[:3],
                "archive_sample_pool": archive[:3],
            },
            resource_state={
                "daily_token_surplus_rate": round(float(state.budget_remaining), 4),
                "fatigue_level": round(1.0 - float(state.body_energy), 4),
                "resource_scarcity": round(float(state.resource_state.get("scarcity_index", 1.0 - state.budget_remaining)), 4),
            },
            emotional_baseline={
                "valence": round(float(state.mood) - 0.5, 4),
                "arousal": round(float(state.affect_residue), 4),
                "dominant_emotion": "residue" if state.affect_residue > 0.18 else "steady",
            },
            relationship_state={
                "closeness": round(float(relation_state.get("closeness", 0.5)), 4),
                "trust": round(float(relation_state.get("closeness", 0.5)), 4),
                "boundary_tension": round(float(relation_state.get("boundary_level", 0.0)), 4),
            },
            identity_evidence_summary={
                "anchors": list(identity_evidence.get("anchors", [])),
                "signature": identity_evidence.get("signature", ""),
            },
            conflict_residue={
                "repair_stage": getattr(state.repair_state, "stage", "idle"),
                "critical_conflict": bool(state.conflict_hot_rounds > 0),
            },
        )

    def _allowed_types(self, trigger: str) -> list[str]:
        return list(self.config.get("allowed_types", {}).get(trigger, []))

    def _invoke_sidecar(self, request: DreamRunRequest) -> dict[str, Any]:
        env = os.environ.copy()
        env["NALR_HOME"] = str(self.home_path)
        config_dir = os.environ.get("NALR_CONFIG_DIR")
        if config_dir:
            env["NALR_CONFIG_DIR"] = config_dir
        payload = json.dumps({"type": "dream.start", "request": to_dict(request)}, ensure_ascii=False) + "\n"
        try:
            completed = subprocess.run(
                [sys.executable, "-m", "nalr.dream.bridge"],
                cwd=self.project_root,
                input=payload,
                capture_output=True,
                text=True,
                env=env,
                timeout=float(self.config.get("rpc_timeout_seconds", 5)),
            )
            stdout = completed.stdout.strip().splitlines()
            if completed.returncode != 0 or not stdout:
                raise RuntimeError(completed.stderr.strip() or "dream bridge failed")
            response = json.loads(stdout[0])
            if response.get("type") != "dream.result":
                raise RuntimeError(response.get("message", "dream bridge returned error"))
            return response["result"]
        except Exception:
            return OneiroiAgent().run(request)

    def _guard_bundle(self, *, trigger: str, proposal_bundle: dict[str, Any], snapshot: DreamSnapshot) -> DreamGuardDecision:
        allowed = set(self._allowed_types(trigger))
        evaluated = [
            "memory_consolidation",
            "emotion_adjustments",
            "habit_adjustments",
            "dream_memory_write",
            "relationship_adjustments",
            "limited_identity_drift",
        ]
        rejected: list[str] = []
        reasons: list[str] = []
        if proposal_bundle.get("limited_identity_drift"):
            anchors = list(snapshot.identity_evidence_summary.get("anchors", []))
            if "limited_identity_drift" not in allowed:
                rejected.append("limited_identity_drift")
                reasons.append("identity_drift_not_allowed_for_trigger")
            elif not anchors:
                rejected.append("limited_identity_drift")
                reasons.append("identity_drift_requires_evidence")
        return DreamGuardDecision(
            approved=bool(allowed),
            allowed_types=list(allowed),
            evaluated_types=evaluated,
            rejected_types=rejected,
            reasons=reasons,
        )

    def build_recalibration_payload(
        self,
        *,
        mode: str,
        trigger: str,
        snapshot: DreamSnapshot,
        proposal_bundle: dict[str, Any],
        guard: DreamGuardDecision,
        effect_summary: dict[str, Any],
        trace: dict[str, Any],
        trace_ref: str,
    ) -> dict[str, Any]:
        allowed_types = set(guard.allowed_types)
        applied_prior_updates = {
            "memory_prior_updates": list(proposal_bundle.get("memory_consolidation", [])),
            "emotion_prior_updates": list(proposal_bundle.get("emotion_adjustments", []))
            if "emotion_adjustments" in allowed_types
            else [],
            "habit_prior_updates": list(proposal_bundle.get("habit_adjustments", []))
            if "habit_adjustments" in allowed_types
            else [],
            "relationship_prior_updates": list(proposal_bundle.get("relationship_adjustments", []))
            if "relationship_adjustments" in allowed_types
            else [],
            "identity_prior_updates": list(proposal_bundle.get("limited_identity_drift", []))
            if "limited_identity_drift" in allowed_types and "limited_identity_drift" not in guard.rejected_types
            else [],
            "dream_memory_write": list(proposal_bundle.get("dream_memory_write", []))
            if "dream_memory_write" in allowed_types
            else [],
        }
        proposed_prior_updates = {
            "memory_prior_updates": list(proposal_bundle.get("memory_consolidation", [])),
            "emotion_prior_updates": list(proposal_bundle.get("emotion_adjustments", [])),
            "habit_prior_updates": list(proposal_bundle.get("habit_adjustments", [])),
            "relationship_prior_updates": list(proposal_bundle.get("relationship_adjustments", [])),
            "identity_prior_updates": list(proposal_bundle.get("limited_identity_drift", [])),
            "dream_memory_write": list(proposal_bundle.get("dream_memory_write", [])),
        }
        prior_summary = {
            key: len(value)
            for key, value in applied_prior_updates.items()
        }
        recalibration_strength = _clip(
            0.25
            + len(applied_prior_updates["memory_prior_updates"]) * 0.10
            + len(applied_prior_updates["habit_prior_updates"]) * 0.06
            + len(applied_prior_updates["identity_prior_updates"]) * 0.12
            + float(effect_summary.get("applied", False)) * 0.08
        )
        confidence = _clip(
            recalibration_strength
            + len(applied_prior_updates["identity_prior_updates"]) * 0.05
            + len(applied_prior_updates["habit_prior_updates"]) * 0.03
        )
        return {
            "module_name": "DreamOrchestrator",
            "layer": "offline",
            "kind": "recalibration",
            "prior_role": "stable_prior",
            "mode": mode,
            "trigger": trigger,
            "sleep_session_id": snapshot.sleep_session_id,
            "trace_ref": trace_ref,
            "dream_run_id": trace.get("dream_run_id", ""),
            "proposal_counts": dict(trace.get("proposal_counts", {})),
            "guard_summary": to_dict(guard),
            "effect_summary": dict(effect_summary),
            "proposed_prior_updates": proposed_prior_updates,
            "applied_prior_updates": applied_prior_updates,
            "prior_summary": prior_summary,
            "recalibration_strength": round(recalibration_strength, 4),
            "confidence": round(confidence, 4),
            "dream_score": round(confidence, 4),
            "trace_reason": (
                f"mode={mode}; trigger={trigger}; approved={guard.approved}; "
                f"identity_updates={prior_summary['identity_prior_updates']}"
            ),
        }

    def _memory_payload(self, *, mode: str, cue: str | None, bundle: dict[str, Any]) -> dict[str, Any]:
        return {
            "mode": mode,
            "source": mode,
            "cue": cue,
            "replayed_anchor": cue,
            "memory_consolidation": bundle.get("memory_consolidation", []),
            "emotion_adjustments": [],
            "habit_adjustments": bundle.get("habit_adjustments", []),
            "dream_memory_write": bundle.get("dream_memory_write", []),
            "relationship_adjustments": [],
        }

    def _apply_bundle(self, *, state, mode: str, cue: str | None, bundle: dict[str, Any], guard: DreamGuardDecision) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        applied_types: list[str] = []

        for op in bundle.get("emotion_adjustments", []):
            if "emotion_adjustments" not in guard.allowed_types:
                continue
            field = op.get("field")
            if field == "affect_residue":
                state.affect_residue = round(_clip(float(state.affect_residue) * float(op.get("multiplier", 1.0))), 4)
            elif field == "mood":
                target = float(op.get("target", state.mood))
                mix = float(op.get("mix", 0.0))
                state.mood = round(_clip(float(state.mood) * (1 - mix) + target * mix), 4)
            applied_types.append("emotion_adjustments")

        for op in bundle.get("relationship_adjustments", []):
            if "relationship_adjustments" not in guard.allowed_types:
                continue
            target = str(op.get("target", "user"))
            delta = float(op.get("trust_delta", 0.0))
            self.memory_store.nudge_relation(target, delta)
            applied_types.append("relationship_adjustments")

        for op in bundle.get("limited_identity_drift", []):
            if "limited_identity_drift" not in guard.allowed_types or "limited_identity_drift" in guard.rejected_types:
                continue
            credit_delta = float(op.get("credit_delta", 0.0))
            state.identity_state.drift_credit = round(_clip(float(state.identity_state.drift_credit) + credit_delta, 0.0, 2.0), 4)
            alias = str(op.get("proposed_alias") or "").strip()
            if alias and alias not in state.identity_state.aliases and alias != state.identity_state.display_name:
                state.identity_state.aliases = (state.identity_state.aliases + [alias])[-5:]
            applied_types.append("limited_identity_drift")

        memory_payload = self._memory_payload(mode=mode, cue=cue, bundle=bundle)
        shaping_events = self.vitality_engine.apply_noninteractive_shaping(
            state,
            mode,
            cue,
            self.memory_store,
            proposal=memory_payload,
        )
        if shaping_events:
            applied_types.extend(list(shaping_events[0].get("applied_types", [])))
        identity_state = getattr(state, "identity_state", None)
        drift_credit = round(float(getattr(identity_state, "drift_credit", 0.0)), 4) if identity_state is not None else 0.0
        aliases = list(getattr(identity_state, "aliases", [])) if identity_state is not None else []
        effect_summary = {
            "applied": bool(applied_types),
            "applied_types": sorted(set(applied_types)),
            "cue": cue,
            "prior_updates": sorted(set(applied_types)),
            "identity_prior": {
                "drift_credit": drift_credit,
                "aliases": aliases,
            },
            "habit_prior": {
                "updated": "habit_adjustments" in applied_types,
                "sources": [item for item in bundle.get("habit_adjustments", [])],
            },
            "prior_role": "stable_prior",
        }
        return shaping_events, effect_summary

    def run(self, *, state, mode: str, cue: str | None, relation_state: dict[str, Any]) -> dict[str, Any]:
        if mode not in {"idle", "sleep"}:
            return {
                "shaping_events": [],
                "run_id": None,
                "trigger": None,
                "trace_ref": None,
                "guard_summary": {},
                "effect_summary": {},
                "recalibration": {},
            }
        if not self.enabled(state):
            shaping_events = self.vitality_engine.apply_noninteractive_shaping(state, mode, cue, self.memory_store)
            return {
                "shaping_events": shaping_events,
                "run_id": None,
                "trigger": None,
                "trace_ref": None,
                "guard_summary": {"approved": False, "allowed_types": [], "evaluated_types": [], "rejected_types": [], "reasons": ["dream_disabled"]},
                "effect_summary": shaping_events[0] if shaping_events else {},
                "recalibration": {},
            }

        trigger = "sleep_full" if mode == "sleep" else "idle_light"
        snapshot = self.build_snapshot(state=state, mode=mode, trigger=trigger, cue=cue, relation_state=relation_state)
        request = DreamRunRequest(
            snapshot=snapshot,
            budget={
                "max_tokens": self.config.get("budgets", {}).get(f"{trigger}_tokens", 400),
                "allow_identity_drift": trigger == "sleep_full",
            },
            policy={"allowed_types": self._allowed_types(trigger)},
            trace_context={"session_id": state.session_id, "round_id": state.round_count},
        )
        result = self._invoke_sidecar(request)
        guard = self._guard_bundle(trigger=trigger, proposal_bundle=result["proposal_bundle"], snapshot=snapshot)
        shaping_events, effect_summary = self._apply_bundle(
            state=state,
            mode=mode,
            cue=cue,
            bundle=result["proposal_bundle"],
            guard=guard,
        )
        trace_ref = f"dream://runs/{result['trace']['dream_run_id']}"
        recalibration = self.build_recalibration_payload(
            mode=mode,
            trigger=trigger,
            snapshot=snapshot,
            proposal_bundle=result["proposal_bundle"],
            guard=guard,
            effect_summary=effect_summary,
            trace=result["trace"],
            trace_ref=trace_ref,
        )
        effect_summary = {
            **effect_summary,
            "prior_updates": dict(recalibration.get("applied_prior_updates", {})),
            "identity_prior_updates": list(recalibration.get("applied_prior_updates", {}).get("identity_prior_updates", [])),
            "habit_prior_updates": list(recalibration.get("applied_prior_updates", {}).get("habit_prior_updates", [])),
            "confidence": float(recalibration.get("confidence", recalibration.get("recalibration_strength", 0.0))),
            "dream_score": float(recalibration.get("dream_score", recalibration.get("recalibration_strength", 0.0))),
        }
        payload = {
            "trace": result["trace"],
            "proposal_bundle": result["proposal_bundle"],
            "guard_summary": to_dict(guard),
            "effect_summary": effect_summary,
            "recalibration": recalibration,
        }
        trace_ref = self.store.write_run(payload)
        return {
            "shaping_events": shaping_events,
            "run_id": result["trace"]["dream_run_id"],
            "trigger": trigger,
            "trace_ref": trace_ref,
            "guard_summary": to_dict(guard),
            "effect_summary": effect_summary,
            "recalibration": recalibration,
        }
