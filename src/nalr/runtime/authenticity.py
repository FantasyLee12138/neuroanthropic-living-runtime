from __future__ import annotations

import re
from typing import Any

from nalr.schemas.models import AuthenticityRecord, RenderPlan


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


class AuthenticityPolicy:
    def __init__(self, identity_cfg: dict[str, Any]) -> None:
        self.identity_cfg = identity_cfg

    def _score_sampling_penalties(
        self,
        distribution: dict[str, float],
        *,
        query_kind: str,
        disclosure_intent: str,
        slow_variables: dict[str, Any],
        memory_cue: str | None,
        shaping_events: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not distribution:
            return {
                "adjusted_distribution": {},
                "candidate_penalties": {},
                "sampling_penalty_applied": 0.0,
                "self_grounding_score": 0.0,
                "guard_action": "pass",
                "trace_reason": "empty_distribution",
                "penalty_strength": 0.0,
            }

        penalties = {action: 0.0 for action in distribution}
        has_non_interactive = any(item.get("source") in {"idle", "sleep"} for item in shaping_events)
        cue_active = bool(memory_cue) or float(slow_variables.get("memory_activation", 0.0) or 0.0) >= 0.15
        self_grounding_score = 0.8 if query_kind == "general" else 0.55
        guard_action = "pass"

        if query_kind in {"self_identity", "provider_identity"}:
            for action in distribution:
                if action not in {"respond", "clarify", "short_reply"}:
                    penalties[action] += 0.18
                if action == "wander":
                    penalties[action] += 0.16
            self_grounding_score += 0.12
            guard_action = "soft_penalty"
        elif query_kind == "answer_explanation":
            for action in distribution:
                if action not in {"respond", "recall", "clarify", "short_reply"}:
                    penalties[action] += 0.14
                if cue_active and action not in {"recall", "respond", "short_reply"}:
                    penalties[action] += 0.08
                if has_non_interactive and action == "wander":
                    penalties[action] += 0.08
            self_grounding_score += 0.08
            guard_action = "soft_penalty"
        if disclosure_intent == "withhold":
            for action in distribution:
                if action in {"connect", "wander"}:
                    penalties[action] += 0.10
            self_grounding_score -= 0.05
        elif disclosure_intent == "provider_origin":
            for action in distribution:
                if action in {"connect", "wander"}:
                    penalties[action] += 0.06
        elif disclosure_intent == "relational_self_disclosure":
            penalties["short_reply"] = penalties.get("short_reply", 0.0) + 0.08

        adjusted: dict[str, float] = {}
        sampling_penalty_applied = 0.0
        for action, probability in distribution.items():
            penalty = _clip(penalties.get(action, 0.0), 0.0, 0.75)
            adjusted_probability = max(probability * (1.0 - penalty), 1e-9)
            adjusted[action] = adjusted_probability
            sampling_penalty_applied += max(probability - adjusted_probability, 0.0)

        total = sum(adjusted.values()) or 1.0
        normalized = {action: round(value / total, 6) for action, value in adjusted.items()}
        normalized_penalties = {action: round(value, 4) for action, value in penalties.items() if value > 0.0}
        trace_reason = (
            f"query={query_kind}; disclosure={disclosure_intent}; "
            f"cue_active={cue_active}; non_interactive={has_non_interactive}"
        )
        return {
            "adjusted_distribution": normalized,
            "candidate_penalties": normalized_penalties,
            "sampling_penalty_applied": round(sampling_penalty_applied, 4),
            "self_grounding_score": round(_clip(self_grounding_score, 0.0, 1.0), 4),
            "guard_action": guard_action,
            "trace_reason": trace_reason,
            "penalty_strength": round(
                _clip(
                    sampling_penalty_applied + float(sum(penalties.values())) * 0.15 + (0.08 if guard_action != "pass" else 0.0),
                    0.0,
                    1.0,
                ),
                4,
            ),
        }

    def apply_sampling_penalties(
        self,
        distribution: dict[str, float],
        *,
        query_kind: str,
        disclosure_intent: str,
        slow_variables: dict[str, Any],
        memory_cue: str | None,
        shaping_events: list[dict[str, Any]],
    ) -> tuple[dict[str, float], dict[str, float], float]:
        payload = self._score_sampling_penalties(
            distribution,
            query_kind=query_kind,
            disclosure_intent=disclosure_intent,
            slow_variables=slow_variables,
            memory_cue=memory_cue,
            shaping_events=shaping_events,
        )
        return payload["adjusted_distribution"], payload["candidate_penalties"], payload["sampling_penalty_applied"]

    def build_authenticity_penalty_payload(
        self,
        distribution: dict[str, float],
        *,
        query_kind: str,
        disclosure_intent: str,
        slow_variables: dict[str, Any],
        memory_cue: str | None,
        shaping_events: list[dict[str, Any]],
    ) -> dict[str, Any]:
        payload = self._score_sampling_penalties(
            distribution,
            query_kind=query_kind,
            disclosure_intent=disclosure_intent,
            slow_variables=slow_variables,
            memory_cue=memory_cue,
            shaping_events=shaping_events,
        )
        return {
            "module_name": "AuthenticityPolicy",
            "layer": "action",
            "kind": "penalty",
            "prior_role": "penalty",
            "candidate_penalties": payload["candidate_penalties"],
            "normalized_distribution": payload["adjusted_distribution"],
            "sampling_penalty_applied": payload["sampling_penalty_applied"],
            "self_grounding_score": payload["self_grounding_score"],
            "confidence": payload["self_grounding_score"],
            "penalty_strength": payload["penalty_strength"],
            "guard_action": payload["guard_action"],
            "trace_reason": payload["trace_reason"],
            "signal_summary": {
                "query_kind": query_kind,
                "disclosure_intent": disclosure_intent,
                "memory_cue_active": bool(memory_cue),
                "non_interactive": any(item.get("source") in {"idle", "sleep"} for item in shaping_events),
            },
        }

    def evaluate_text(self, text: str, render_plan: RenderPlan) -> dict[str, Any]:
        identity = render_plan.identity_context
        lowered = text.lower()
        provider_tokens = [token.lower() for token in self.identity_cfg.get("provider_blocklist", [])]
        allow_provider = identity.disclosure_detail in {"specific", "model_id"}
        provider_mentions = [token for token in provider_tokens if token and token in lowered]
        false_self_claim = any(
            re.search(rf"我(?:是|叫|就是)\s*{re.escape(token)}", text, flags=re.IGNORECASE)
            for token in provider_tokens
        )
        provider_leak = bool(provider_mentions) and not allow_provider
        if identity.query_kind == "answer_explanation" and provider_mentions:
            provider_leak = True
        provider_leak_penalty = 0.65 if provider_leak else 0.0
        false_self_claim_penalty = 0.82 if false_self_claim else 0.0

        grounding_score = 0.8 if identity.query_kind == "general" else 0.55
        if identity.display_label and identity.display_label in text:
            grounding_score += 0.2
        if identity.class_label and identity.class_label in text:
            grounding_score += 0.1
        if identity.query_kind == "answer_explanation":
            hint_tokens = [render_plan.action, render_plan.message_plan.get("focus"), render_plan.message_plan.get("memory_cue")]
            if any(token and str(token) in text for token in hint_tokens):
                grounding_score += 0.1
        grounding_score -= provider_leak_penalty * 0.7
        grounding_score -= false_self_claim_penalty * 0.6

        slow_variables = render_plan.message_plan.get("slow_variables", {})
        state_sources = []
        for field_name, threshold in (
            ("affect_residue", 0.18),
            ("memory_activation", 0.15),
            ("habit_readiness", 0.20),
            ("resource_scarcity", 0.40),
            ("relationship_drift", 0.08),
        ):
            if float(slow_variables.get(field_name, 0.0) or 0.0) >= threshold:
                state_sources.append(field_name)
        if render_plan.message_plan.get("memory_cue"):
            state_sources.append("memory_cue")
        if render_plan.message_plan.get("focus"):
            state_sources.append("focus")
        state_sources = sorted(set(state_sources))

        violation_types: list[str] = []
        if provider_leak:
            violation_types.append("provider_leak")
        if false_self_claim:
            violation_types.append("false_self_claim")
        return {
            "provider_leak_detected": provider_leak,
            "false_self_claim_detected": false_self_claim,
            "self_grounding_score": round(_clip(grounding_score, 0.0, 1.0), 4),
            "provider_leak_penalty": provider_leak_penalty,
            "false_self_claim_penalty": false_self_claim_penalty,
            "violation_types": violation_types,
            "state_sources": state_sources,
        }

    def build_authenticity_prior_payload(
        self,
        *,
        render_plan: RenderPlan,
        evaluation: dict[str, Any],
        candidate_penalties: dict[str, float],
        sampling_penalty_applied: float,
    ) -> dict[str, Any]:
        identity = render_plan.identity_context
        return {
            "module_name": "AuthenticityPolicy",
            "layer": "action",
            "kind": "prior_penalty",
            "prior_role": "penalty",
            "identity_context": {
                "query_kind": identity.query_kind,
                "disclosure_detail": identity.disclosure_detail,
                "disclosure_intent": identity.disclosure_intent,
                "display_label": identity.display_label,
                "class_label": identity.class_label,
                "self_description_sources": list(identity.self_description_sources),
                "state_sources": list(evaluation.get("state_sources", [])),
            },
            "candidate_penalties": dict(candidate_penalties),
            "sampling_penalty_applied": float(sampling_penalty_applied),
            "self_grounding_score": float(evaluation.get("self_grounding_score", 0.0)),
            "confidence": float(evaluation.get("self_grounding_score", 0.0)),
            "provider_leak_penalty": float(evaluation.get("provider_leak_penalty", 0.0)),
            "false_self_claim_penalty": float(evaluation.get("false_self_claim_penalty", 0.0)),
            "penalty_strength": float(evaluation.get("penalty_strength", sampling_penalty_applied)),
            "trace_reason": (
                f"query={identity.query_kind}; disclosure={identity.disclosure_detail}; "
                f"grounding={float(evaluation.get('self_grounding_score', 0.0)):.2f}"
            ),
            "bias": {
                "respond": round(_clip(0.05 + float(evaluation.get("self_grounding_score", 0.0)) * 0.05), 4),
                "clarify": round(_clip(0.04 + len(candidate_penalties) * 0.01), 4),
                "short_reply": round(_clip(0.03 + sampling_penalty_applied * 0.02), 4),
                "wander": round(_clip(0.10 + float(evaluation.get("provider_leak_penalty", 0.0)) * 0.08), 4),
            },
        }

    def build_record(
        self,
        *,
        evaluation: dict[str, Any],
        guard_action: str,
        disclosure_detail: str,
        rename_event: dict[str, Any] | None,
        candidate_penalties: dict[str, float],
        sampling_penalty_applied: float,
    ) -> AuthenticityRecord:
        return AuthenticityRecord(
            provider_leak_detected=bool(evaluation.get("provider_leak_detected", False)),
            false_self_claim_detected=bool(evaluation.get("false_self_claim_detected", False)),
            self_grounding_score=float(evaluation.get("self_grounding_score", 0.0)),
            provider_leak_penalty=float(evaluation.get("provider_leak_penalty", 0.0)),
            false_self_claim_penalty=float(evaluation.get("false_self_claim_penalty", 0.0)),
            guard_action=guard_action,
            violation_types=list(evaluation.get("violation_types", [])),
            disclosure_detail=disclosure_detail,
            rename_event=rename_event,
            state_sources=list(evaluation.get("state_sources", [])),
            candidate_penalties=dict(candidate_penalties),
            sampling_penalty_applied=float(sampling_penalty_applied),
        )
