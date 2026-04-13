from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from nalr.schemas.models import to_dict


INITIATIVE_INTENTS = (
    "share_memory",
    "check_relation",
    "express_state",
    "follow_up_task",
    "stay_silent",
)


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _normalize(scores: dict[str, float]) -> dict[str, float]:
    positives = {key: max(float(value), 0.0) for key, value in scores.items()}
    total = sum(positives.values()) or 1.0
    return {key: round(value / total, 6) for key, value in positives.items()}


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = str(value).strip()
    if not normalized:
        return None
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


class InitiativeRuntime:
    def __init__(self, default_settings: dict[str, Any] | None = None) -> None:
        self._defaults = self.normalize_settings(default_settings or {})

    def default_settings(self) -> dict[str, Any]:
        return dict(self._defaults)

    def normalize_settings(self, payload: dict[str, Any]) -> dict[str, Any]:
        defaults = {
            "idle_seconds_threshold": 300,
            "vitality_threshold": 0.8,
            "relation_strength_threshold": 0.6,
            "habit_strength_threshold": 0.45,
            "proposal_posterior_threshold": 0.7,
            "hourly_limit": 3,
            "cooldown_seconds": 600,
            "require_memory_backing": True,
            "auto_send_enabled": True,
        }
        defaults.update(dict(payload or {}))
        return {
            "idle_seconds_threshold": max(0, int(defaults.get("idle_seconds_threshold", 300) or 0)),
            "vitality_threshold": round(_clip(float(defaults.get("vitality_threshold", 0.8) or 0.0)), 4),
            "relation_strength_threshold": round(_clip(float(defaults.get("relation_strength_threshold", 0.6) or 0.0)), 4),
            "habit_strength_threshold": round(_clip(float(defaults.get("habit_strength_threshold", 0.45) or 0.0)), 4),
            "proposal_posterior_threshold": round(_clip(float(defaults.get("proposal_posterior_threshold", 0.7) or 0.0)), 4),
            "hourly_limit": max(0, int(defaults.get("hourly_limit", 3) or 0)),
            "cooldown_seconds": max(0, int(defaults.get("cooldown_seconds", 600) or 0)),
            "require_memory_backing": bool(defaults.get("require_memory_backing", True)),
            "auto_send_enabled": bool(defaults.get("auto_send_enabled", True)),
        }

    def merge_settings(self, *payloads: dict[str, Any] | None) -> dict[str, Any]:
        merged = self.default_settings()
        for payload in payloads:
            if isinstance(payload, dict):
                merged.update(payload)
        return self.normalize_settings(merged)

    def idle_seconds(self, latest_recorded_at: str | None, *, now_iso: str) -> int:
        now_dt = _parse_iso(now_iso) or datetime.now(timezone.utc)
        latest_dt = _parse_iso(latest_recorded_at)
        if latest_dt is None:
            return 0
        if latest_dt.tzinfo is None:
            latest_dt = latest_dt.replace(tzinfo=timezone.utc)
        return max(0, int((now_dt - latest_dt).total_seconds()))

    def _normalize_active_run_payload(self, active_run: dict[str, Any] | Any | None) -> dict[str, Any] | None:
        if active_run is None:
            return None
        payload = dict(active_run) if isinstance(active_run, dict) else dict(to_dict(active_run))
        if not payload:
            return None
        status = str(payload.get("status") or payload.get("run_status") or "").strip().lower()
        if status:
            payload["status"] = status
        payload["dirty_worktree_detected"] = bool(payload.get("dirty_worktree_detected", False))
        if "commit_permission_required" in payload:
            payload["commit_permission_required"] = bool(payload.get("commit_permission_required"))
        return payload

    def _active_run_blocks_outward_dispatch(self, active_run: dict[str, Any] | None) -> bool:
        if not isinstance(active_run, dict):
            return False
        status = str(active_run.get("status") or active_run.get("run_status") or "").strip().lower()
        if status in {"running", "awaiting_approval"}:
            return True
        if str(active_run.get("approval_state") or "").strip().lower() == "awaiting_approval":
            return True
        if bool(active_run.get("awaiting_approval")):
            return True
        return False

    def prepare(
        self,
        *,
        settings: dict[str, Any],
        idle_seconds: int,
        vitality_snapshot: dict[str, Any],
        relation_state: dict[str, Any],
        subjective_state: dict[str, Any] | None = None,
        temperament_state: dict[str, Any] | None = None,
        habit_strength: float,
        memory_backing: dict[str, Any] | None,
        active_session: dict[str, Any] | None,
        active_run: dict[str, Any] | Any | None = None,
        pending_approval: bool = False,
        active_run_blocked: bool | None = None,
        safe_mode: bool = False,
        recent_history: list[dict[str, Any]] | None = None,
        current_goal: str | None = None,
    ) -> dict[str, Any]:
        normalized_active_run = self._normalize_active_run_payload(active_run)
        derived_active_run_blocked = self._active_run_blocks_outward_dispatch(normalized_active_run)
        prepared = {
            "settings": self.merge_settings(settings),
            "idle_seconds": int(idle_seconds),
            "vitality_snapshot": dict(vitality_snapshot or {}),
            "relation_state": dict(relation_state or {}),
            "subjective_state": dict(subjective_state or {}),
            "temperament_state": dict(temperament_state or {}),
            "habit_strength": float(habit_strength),
            "memory_backing": dict(memory_backing or {}),
            "active_session": dict(active_session) if isinstance(active_session, dict) else None,
            "active_run": normalized_active_run,
            "pending_approval": bool(pending_approval),
            "active_run_blocked": derived_active_run_blocked if active_run_blocked is None else bool(active_run_blocked),
            "safe_mode": bool(safe_mode),
            "recent_history": list(recent_history or []),
            "current_goal": current_goal,
        }
        return prepared

    def _evaluate_core(
        self,
        *,
        settings: dict[str, Any],
        idle_seconds: int,
        vitality_snapshot: dict[str, Any],
        relation_state: dict[str, Any],
        subjective_state: dict[str, Any] | None,
        temperament_state: dict[str, Any] | None,
        habit_strength: float,
        memory_backing: dict[str, Any] | None,
        active_session: dict[str, Any] | None,
        pending_approval: bool,
        active_run_blocked: bool,
        safe_mode: bool,
        recent_history: list[dict[str, Any]] | None = None,
        current_goal: str | None = None,
    ) -> dict[str, Any]:
        settings = self.merge_settings(settings)
        recent_history = list(recent_history or [])
        vitality_level = _clip(
            float(vitality_snapshot.get("body_energy", vitality_snapshot.get("vitality", 0.0)) or 0.0)
        )
        relation_strength = _clip(float(relation_state.get("closeness", relation_state.get("trust", 0.0)) or 0.0))
        boundary_level = _clip(float(relation_state.get("boundary_level", 0.0) or 0.0))
        subjective_state = dict(subjective_state or {})
        spontaneous = _clip(float(subjective_state.get("spontaneous", 0.0) or 0.0))
        reject_all = _clip(float(subjective_state.get("reject_all", 0.0) or 0.0))
        meaning_density = _clip(len(list(subjective_state.get("meaning_made", []) or [])) * 0.2)
        temperament_state = dict(temperament_state or {})
        temperament_current = dict(temperament_state.get("current", temperament_state) or {})
        attachment_need = _clip(float(temperament_current.get("attachment_need", 0.5) or 0.0))
        boundary_softness = _clip(float(temperament_current.get("boundary_softness", 0.5) or 0.0))
        extraversion = _clip(float(temperament_current.get("extraversion", 0.5) or 0.0))
        cognitive_bandwidth = _clip(float(temperament_current.get("cognitive_bandwidth", 0.5) or 0.0))
        approach_tension = _clip(attachment_need * (1.0 - boundary_softness))
        social_output_capacity = _clip(extraversion * cognitive_bandwidth)
        memory_strength = _clip(float((memory_backing or {}).get("strength", 0.0) or 0.0))
        memory_cue = str((memory_backing or {}).get("cue") or "").strip()
        topic_relevance = _clip(float((memory_backing or {}).get("topic_relevance", 0.0) or 0.0))
        memory_is_endogenous = memory_cue.startswith("endogenous:")
        recent_outbound = [item for item in recent_history[-6:] if bool(item.get("auto_sent"))]
        ignored_ratio = (
            sum(1 for item in recent_outbound if not bool(item.get("feedback_recorded"))) / max(len(recent_outbound), 1)
            if recent_outbound
            else 0.0
        )
        recent_auto_sent = sum(1 for item in recent_history[-6:] if bool(item.get("auto_sent")))
        frequency_density = _clip(recent_auto_sent / max(int(settings["hourly_limit"]) or 3, 3))
        feedback_penalty = _clip(ignored_ratio * 0.18)
        interruption_cost = _clip(
            (0.30 if pending_approval else 0.0)
            + (0.22 if active_run_blocked else 0.0)
            + (0.10 if active_session is None else 0.0)
        )
        social_risk = _clip(boundary_level * 0.58 + max(0.0, 0.42 - relation_strength) * 0.24)
        self_expression_drive = _clip(spontaneous * 0.75 + meaning_density * 0.2 - reject_all * 0.15)
        externalized_memory_strength = memory_strength
        if settings["require_memory_backing"] and memory_is_endogenous and not current_goal:
            externalized_memory_strength = _clip(
                memory_strength * (0.18 + self_expression_drive * 1.7 + social_output_capacity * 0.25)
            )
        endogenous_expression_penalty = (
            _clip((0.16 - spontaneous) * 1.25 + (0.14 - self_expression_drive) * 0.85)
            if settings["require_memory_backing"] and memory_is_endogenous and not current_goal
            else 0.0
        )
        irrelevance_penalty = _clip(
            0.03
            + (0.12 if not current_goal else 0.0)
            + (0.08 if memory_strength <= 0.0 else 0.0)
            - topic_relevance * 0.18
            - self_expression_drive * 0.10
            + endogenous_expression_penalty * 0.18
        )
        grounding_score = _clip(
            0.45 * memory_strength
            + 0.18 * topic_relevance
            + 0.20 * relation_strength
            + 0.20 * vitality_level
            + 0.15 * (1.0 if current_goal else 0.0)
        )
        grounding_penalty = _clip(max(0.0, 0.45 - grounding_score))
        uncertainty_reduction = _clip(
            0.18
            + (0.32 if current_goal else 0.0)
            + memory_strength * 0.16
            + topic_relevance * 0.14
            + meaning_density * 0.08
        )
        goal_alignment = _clip((0.40 if current_goal else 0.08) + _clip(float(habit_strength)) * 0.12 + topic_relevance * 0.16)
        relational_gain = _clip(relation_strength * 0.55 + (0.12 if active_session is not None else 0.0) + approach_tension * 0.10 + social_output_capacity * 0.08)
        emotional_release = _clip(float(vitality_snapshot.get("affect_residue", 0.0) or 0.0) * 0.60 + vitality_level * 0.20)
        intrinsic_value = _clip(
            0.32 * uncertainty_reduction
            + 0.28 * goal_alignment
            + 0.24 * relational_gain
            + 0.16 * emotional_release
            + grounding_score * 0.10
            + self_expression_drive * 0.18
            + social_output_capacity * 0.08
        )
        speech_cost = _clip(
            interruption_cost
            + social_risk
            + frequency_density * 0.32
            + irrelevance_penalty
            + feedback_penalty
            + grounding_penalty * 0.45
            + endogenous_expression_penalty * 0.30
            + max(0.0, 0.42 - social_output_capacity) * 0.08
        )
        readiness = _clip(
            0.30 * (1.0 if idle_seconds >= int(settings["idle_seconds_threshold"]) else idle_seconds / max(int(settings["idle_seconds_threshold"]) or 1, 1))
            + 0.25 * vitality_level
            + 0.20 * relation_strength
            + 0.10 * _clip(float(habit_strength))
            + 0.15 * externalized_memory_strength
            + spontaneous * 0.12
            + social_output_capacity * 0.08
            - 0.15 * ignored_ratio
            - 0.20 * boundary_level,
            0.0,
            1.0,
        )
        scores = {
            "share_memory": 0.12 + externalized_memory_strength * 0.85 + relation_strength * 0.15 + spontaneous * 0.08 + meaning_density * 0.08 + approach_tension * 0.06 + topic_relevance * 0.06,
            "check_relation": 0.10 + relation_strength * 0.65 + max(0.0, idle_seconds - int(settings["idle_seconds_threshold"])) / max(int(settings["idle_seconds_threshold"]) or 1, 1) * 0.10 + approach_tension * 0.22 + social_output_capacity * 0.10 + topic_relevance * 0.04,
            "express_state": 0.08 + vitality_level * 0.45 + float(vitality_snapshot.get("affect_residue", 0.0) or 0.0) * 0.20 + spontaneous * 0.60 + meaning_density * 0.12 + social_output_capacity * 0.14,
            "follow_up_task": 0.08 + (0.35 if current_goal else 0.0) + _clip(float(habit_strength)) * 0.18 + topic_relevance * (0.18 if current_goal else 0.08),
            "stay_silent": 0.10 + boundary_level * 0.60 + (0.25 if safe_mode else 0.0) + ignored_ratio * 0.25 + reject_all * 0.18 + endogenous_expression_penalty * 0.65 + max(0.0, 0.45 - social_output_capacity) * 0.12,
        }
        if settings["require_memory_backing"] and memory_is_endogenous and not current_goal:
            outward_scale = _clip(0.10 + self_expression_drive * 2.2 + social_output_capacity * 0.24)
            scores["share_memory"] *= outward_scale
            scores["check_relation"] *= outward_scale
            scores["express_state"] *= max(outward_scale, _clip(0.22 + self_expression_drive * 2.4))
            scores["stay_silent"] += _clip(0.18 + endogenous_expression_penalty * 0.45)
        if settings["require_memory_backing"] and memory_strength <= 0.0:
            scores["stay_silent"] += 0.50
            scores["share_memory"] *= 0.25
        if pending_approval or active_run_blocked:
            scores["stay_silent"] += 0.35
        if safe_mode:
            scores["stay_silent"] += 0.45
        if relation_strength < float(settings["relation_strength_threshold"]):
            scores["check_relation"] *= 0.4
        if vitality_level < float(settings["vitality_threshold"]):
            scores["express_state"] *= 0.4
        if float(habit_strength) < float(settings["habit_strength_threshold"]):
            scores["follow_up_task"] *= 0.55

        posterior = _normalize(scores)
        top_intent = max(posterior, key=posterior.get)
        top_score = round(float(scores[top_intent]), 6)
        expression_mode = "external"
        if active_session is None or pending_approval or active_run_blocked or safe_mode:
            expression_mode = "silent"
        if top_intent == "stay_silent" or (intrinsic_value + readiness * 0.35 + grounding_score * 0.15) < speech_cost * 0.8:
            expression_mode = "silent"
        if boundary_level >= 0.82 or top_score < float(settings["proposal_posterior_threshold"]) or readiness <= 0.0:
            expression_mode = "silent"
        suppression_reason = ""
        should_send = expression_mode == "external"
        if safe_mode:
            should_send = False
            suppression_reason = "safe_mode"
        elif boundary_level >= 0.82:
            should_send = False
            suppression_reason = "boundary_high"
        elif pending_approval:
            should_send = False
            suppression_reason = "pending_approval"
        elif active_run_blocked:
            should_send = False
            suppression_reason = "active_run_blocked"
        elif active_session is None:
            should_send = False
            suppression_reason = "no_active_session"
        elif top_intent == "stay_silent":
            should_send = False
            suppression_reason = "stay_silent_peak"
        elif top_score < float(settings["proposal_posterior_threshold"]):
            should_send = False
            suppression_reason = "posterior_below_threshold"
        elif readiness <= 0.0:
            should_send = False
            suppression_reason = "readiness_low"
        expression_mode = "external" if should_send else "silent"

        context_refs: list[str] = []
        if idle_seconds:
            context_refs.append(f"idle_seconds:{int(idle_seconds)}")
        if current_goal:
            context_refs.append(f"goal:{current_goal}")
        memory_refs: list[str] = []
        if memory_backing and memory_backing.get("cue"):
            memory_refs.append(str(memory_backing.get("cue")))
        state_refs = [
            f"body_energy:{round(vitality_level, 4)}",
            f"relation_strength:{round(relation_strength, 4)}",
            f"boundary_level:{round(boundary_level, 4)}",
        ]

        return {
            "proposal_type": "speak",
            "expression_mode": expression_mode,
            "readiness": round(readiness, 6),
            "posterior": posterior,
            "top_intent": top_intent,
            "top_intent_score": top_score,
            "memory_backing": memory_backing or {},
            "grounded_in": {
                "context": context_refs,
                "memory": memory_refs,
                "state": state_refs,
            },
            "grounding_score": round(grounding_score, 6),
            "intrinsic_value": round(intrinsic_value, 6),
            "speech_cost": round(speech_cost, 6),
            "metrics": {
                "idle_seconds": int(idle_seconds),
                "vitality": round(vitality_level, 4),
                "relation_strength": round(relation_strength, 4),
                "habit_strength": round(_clip(float(habit_strength)), 4),
                "boundary_level": round(boundary_level, 4),
                "spontaneous": round(spontaneous, 4),
                "reject_all": round(reject_all, 4),
                "meaning_density": round(meaning_density, 4),
                "attachment_need": round(attachment_need, 4),
                "boundary_softness": round(boundary_softness, 4),
                "extraversion": round(extraversion, 4),
                "cognitive_bandwidth": round(cognitive_bandwidth, 4),
                "approach_tension": round(approach_tension, 4),
                "social_output_capacity": round(social_output_capacity, 4),
                "memory_is_endogenous": bool(memory_is_endogenous),
                "topic_relevance": round(topic_relevance, 4),
                "ignored_ratio": round(float(ignored_ratio), 4),
                "frequency_density": round(float(frequency_density), 4),
                "feedback_penalty": round(float(feedback_penalty), 4),
                "interruption_cost": round(float(interruption_cost), 4),
                "social_risk": round(float(social_risk), 4),
                "irrelevance_penalty": round(float(irrelevance_penalty), 4),
            },
            "should_send": bool(should_send),
            "suppression_reason": suppression_reason,
            "target_session_id": active_session.get("session_id") if isinstance(active_session, dict) else None,
            "proposal_id": f"initiative-{uuid4().hex[:12]}",
        }

    def evaluate(
        self,
        prepared: dict[str, Any] | None = None,
        *,
        settings: dict[str, Any] | None = None,
        idle_seconds: int | None = None,
        vitality_snapshot: dict[str, Any] | None = None,
        relation_state: dict[str, Any] | None = None,
        subjective_state: dict[str, Any] | None = None,
        temperament_state: dict[str, Any] | None = None,
        habit_strength: float = 0.0,
        memory_backing: dict[str, Any] | None = None,
        active_session: dict[str, Any] | None = None,
        active_run: dict[str, Any] | Any | None = None,
        pending_approval: bool = False,
        active_run_blocked: bool | None = None,
        safe_mode: bool = False,
        recent_history: list[dict[str, Any]] | None = None,
        current_goal: str | None = None,
    ) -> dict[str, Any]:
        if prepared is not None and all(
            value is None
            for value in (settings, idle_seconds, vitality_snapshot, relation_state, subjective_state, temperament_state, memory_backing, active_session, active_run, recent_history, current_goal)
        ):
            return self._evaluate_prepared(prepared)
        if settings is None or idle_seconds is None or vitality_snapshot is None or relation_state is None:
            raise TypeError(
                "evaluate() requires either a prepared payload or settings, idle_seconds, vitality_snapshot, and relation_state"
            )
        prepared_payload = self.prepare(
            settings=settings,
            idle_seconds=idle_seconds,
            vitality_snapshot=vitality_snapshot,
            relation_state=relation_state,
            subjective_state=subjective_state,
            temperament_state=temperament_state,
            habit_strength=habit_strength,
            memory_backing=memory_backing,
            active_session=active_session,
            active_run=active_run,
            pending_approval=pending_approval,
            active_run_blocked=active_run_blocked,
            safe_mode=safe_mode,
            recent_history=recent_history,
            current_goal=current_goal,
        )
        return self._evaluate_prepared(prepared_payload)

    def _evaluate_prepared(self, prepared: dict[str, Any]) -> dict[str, Any]:
        active_session = prepared.get("active_session")
        if not isinstance(active_session, dict):
            active_session = None
        return self._evaluate_core(
            settings=dict(prepared.get("settings") or {}),
            idle_seconds=int(prepared.get("idle_seconds") or 0),
            vitality_snapshot=dict(prepared.get("vitality_snapshot") or {}),
            relation_state=dict(prepared.get("relation_state") or {}),
            subjective_state=dict(prepared.get("subjective_state") or {}),
            temperament_state=dict(prepared.get("temperament_state") or {}),
            habit_strength=float(prepared.get("habit_strength") or 0.0),
            memory_backing=dict(prepared.get("memory_backing") or {}),
            active_session=active_session,
            pending_approval=bool(prepared.get("pending_approval")),
            active_run_blocked=bool(prepared.get("active_run_blocked")),
            safe_mode=bool(prepared.get("safe_mode")),
            recent_history=list(prepared.get("recent_history") or []),
            current_goal=prepared.get("current_goal"),
        )

    def commit(self, prepared: dict[str, Any], evaluation: dict[str, Any]) -> dict[str, Any]:
        committed = dict(evaluation)
        committed.setdefault("proposal_id", f"initiative-{uuid4().hex[:12]}")
        committed["producer"] = "initiative"
        committed["producer_phase"] = "commit"
        committed["producer_context"] = {
            "active_run_blocked": bool(prepared.get("active_run_blocked")),
            "pending_approval": bool(prepared.get("pending_approval")),
            "safe_mode": bool(prepared.get("safe_mode")),
        }
        return committed
