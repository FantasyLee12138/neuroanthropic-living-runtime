from __future__ import annotations

import hashlib
from typing import Any, Callable

from nalr.runtime.intent import IntentRuntime
from nalr.schemas.models import (
    DisclosureIntentState,
    EnergyProjectionSpec,
    IdentityContext,
    ProbabilisticContribution,
    QueryIntentState,
    RuntimeState,
)


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


class IdentityRuntime:
    def __init__(self, identity_cfg: dict[str, Any], provider_descriptor: Callable[[], tuple[str, str]]) -> None:
        self.identity_cfg = identity_cfg
        self.provider_descriptor = provider_descriptor
        self.intent_runtime = IntentRuntime()

    def unnamed_label(self) -> str:
        return str(self.identity_cfg.get("unnamed_label", "当前运行体"))

    def normalize_name(self, name: str | None) -> str:
        if not name:
            return ""
        return " ".join(name.strip().split())

    def display_label(self, state: RuntimeState) -> str:
        return state.identity_state.display_name or self.unnamed_label()

    def initial_threshold_met(self, evidence: dict[str, Any]) -> bool:
        thresholds = self.identity_cfg.get("initial_thresholds", {})
        stable_priors = evidence.get("stable_priors", [])
        top_stable = float(stable_priors[0].get("weight", 0.0)) if stable_priors else 0.0
        second_stable = float(stable_priors[1].get("weight", 0.0)) if len(stable_priors) > 1 else 0.0
        top_habit = float(evidence.get("habits", [{}])[0].get("strength", 0.0)) if evidence.get("habits") else 0.0
        top_relation = float(evidence.get("relations", [{}])[0].get("closeness", 0.0)) if evidence.get("relations") else 0.0
        identity_score = float(evidence.get("identity_score", 0.0) or 0.0)
        naming_signal = float(evidence.get("naming_signal", 0.0) or 0.0)
        continuity_signal = float(evidence.get("continuity_signal", 0.0) or 0.0)
        if naming_signal >= 0.38 and continuity_signal >= 0.60 and top_relation >= 0.45:
            return True
        if identity_score >= 0.52 and naming_signal >= 0.28 and continuity_signal >= 0.48:
            return True
        if top_stable < float(thresholds.get("top_stable_prior", 0.18)):
            return False
        return (
            second_stable >= float(thresholds.get("second_stable_prior", 0.12))
            or top_habit >= float(thresholds.get("top_habit", 0.22))
            or top_relation >= float(thresholds.get("top_relation", 0.65))
        )

    def generate_identity_name(self, state: RuntimeState, evidence_signature: str) -> str:
        pool = list(self.identity_cfg.get("display_name_pool", []))
        if not pool:
            return self.unnamed_label()
        seed = evidence_signature or state.session_id
        start = int(hashlib.sha1(seed.encode("utf-8")).hexdigest()[:8], 16) % len(pool)
        reserved = set(state.identity_state.aliases)
        if state.identity_state.display_name:
            reserved.add(state.identity_state.display_name)
        for offset in range(len(pool)):
            candidate = pool[(start + offset) % len(pool)]
            if candidate not in reserved:
                return candidate
        return pool[start]

    def anchor_drift_score(self, previous_anchors: list[str], current_anchors: list[str]) -> float:
        previous = set(previous_anchors)
        current = set(current_anchors)
        if not previous or not current:
            return 0.0
        union = previous | current
        overlap = previous & current
        if not union:
            return 0.0
        return round(1.0 - (len(overlap) / len(union)), 4)

    def augment_identity_evidence(self, state: RuntimeState, evidence: dict[str, Any]) -> dict[str, Any]:
        anchors = list(evidence.get("anchors", []))
        mood_band = "low" if state.mood < 0.42 else "high" if state.mood > 0.68 else "steady"
        energy_band = "low" if state.body_energy < 0.38 else "high" if state.body_energy > 0.72 else "steady"
        affect_band = "hot" if state.affect_residue > 0.45 else "residue" if state.affect_residue > 0.18 else "quiet"
        scarcity = float(state.resource_state.get("scarcity_index", _clip(1.0 - state.budget_remaining)))
        resource_band = "scarce" if scarcity > 0.55 else "tight" if scarcity > 0.25 else "open"
        anchors.extend(
            [
                f"mood:{mood_band}",
                f"energy:{energy_band}",
                f"affect:{affect_band}",
                f"resource:{resource_band}",
            ]
        )
        anchors = sorted(set(anchors))
        signature = hashlib.sha1("|".join(anchors).encode("utf-8")).hexdigest() if anchors else ""
        return {**evidence, "anchors": anchors, "signature": signature}

    def maybe_update_from_evidence(
        self,
        state: RuntimeState,
        memory_store,
        *,
        round_id: int,
        set_identity_name: Callable[..., dict[str, Any] | None],
    ) -> dict[str, Any] | None:
        evidence = self.augment_identity_evidence(state, memory_store.identity_evidence())
        anchors = list(evidence.get("anchors", []))
        signature = str(evidence.get("signature", ""))
        identity = state.identity_state

        if not identity.display_name and self.initial_threshold_met(evidence):
            generated_name = self.generate_identity_name(state, signature)
            rename_event = set_identity_name(state, generated_name, source_hint="generated", round_id=round_id)
            identity.evidence_signature = signature
            identity.evidence_anchors = anchors
            identity.drift_credit = 0.0
            return rename_event

        if not identity.display_name:
            return None

        if not identity.evidence_anchors and anchors:
            identity.evidence_signature = signature
            identity.evidence_anchors = anchors
            return None

        drift_cfg = self.identity_cfg.get("drift", {})
        drift_score = self.anchor_drift_score(identity.evidence_anchors, anchors)
        identity.drift_credit = round(
            max(
                0.0,
                identity.drift_credit * float(drift_cfg.get("momentum", 0.82))
                + drift_score
                - float(drift_cfg.get("base_decay", 0.25)),
            ),
            4,
        )
        if drift_score >= float(drift_cfg.get("trigger_score", 0.6)) and identity.drift_credit >= float(drift_cfg.get("trigger_credit", 1.4)):
            generated_name = self.generate_identity_name(state, signature)
            rename_event = set_identity_name(state, generated_name, source_hint="generated", round_id=round_id)
            identity.evidence_signature = signature
            identity.evidence_anchors = anchors
            identity.drift_credit = 0.0
            return rename_event
        return None

    def classify_query(self, text: str) -> str:
        return self.intent_runtime.classify_query(text)

    def disclosure_detail(
        self,
        *,
        disclosure_state: DisclosureIntentState,
        scenario: str,
        state: RuntimeState,
    ) -> str:
        return self.intent_runtime._legacy_disclosure_detail(
            disclosure_state.posterior.top_intent,
            scenario=scenario,
            state=state,
        )

    def shaping_sources(self, shaping_events: list[dict[str, Any]], slow_variables: dict[str, Any]) -> list[str]:
        sources: list[str] = []
        for field_name, threshold in (
            ("affect_residue", 0.18),
            ("memory_activation", 0.15),
            ("habit_readiness", 0.20),
            ("resource_scarcity", 0.40),
            ("relationship_drift", 0.08),
        ):
            if float(slow_variables.get(field_name, 0.0) or 0.0) >= threshold:
                sources.append(field_name)
        sources.extend(str(item.get("source")) for item in shaping_events if item.get("source") in {"idle", "sleep"})
        return sorted(set(item for item in sources if item))

    def build_identity_context(
        self,
        *,
        query_state: QueryIntentState,
        disclosure_state: DisclosureIntentState,
        scenario: str,
        state: RuntimeState,
        shaping_events: list[dict[str, Any]],
        slow_variables: dict[str, Any],
        state_sources: list[str],
        rename_event: dict[str, Any] | None,
    ) -> IdentityContext:
        query_kind = query_state.legacy_query_kind
        disclosure_detail = self.disclosure_detail(disclosure_state=disclosure_state, scenario=scenario, state=state)
        provider_label, model_label = self.provider_descriptor()
        evolution_reason = ""
        if rename_event is not None:
            evolution_reason = str(rename_event.get("source", "rename"))
        elif state.identity_state.display_name:
            evolution_reason = "stable_continuity"
        non_interactive_sources = sorted(
            {
                str(item.get("source"))
                for item in shaping_events
                if item.get("non_interactive") and item.get("source") in {"idle", "sleep"}
            }
        )
        self_description_sources = sorted(set(state_sources + self.shaping_sources(shaping_events, slow_variables)))
        return IdentityContext(
            query_kind=query_kind,
            query_intent=query_state.posterior.top_intent,
            query_intent_posterior=dict(query_state.posterior.posterior),
            display_label=self.display_label(state),
            class_label=str(self.identity_cfg.get("class_label", "runtime_instance")),
            internal_handle=state.identity_state.internal_handle,
            aliases=list(state.identity_state.aliases),
            disclosure_detail=disclosure_detail,
            disclosure_intent=disclosure_state.posterior.top_intent,
            disclosure_intent_posterior=dict(disclosure_state.posterior.posterior),
            disclosure_clipped=list(disclosure_state.posterior.clipped),
            provider_label=provider_label,
            model_label=model_label,
            evolution_reason=evolution_reason,
            non_interactive_sources=non_interactive_sources,
            evidence_anchors=list(state.identity_state.evidence_anchors),
            self_description_sources=self_description_sources,
        )

    def build_identity_evolution_payload(
        self,
        *,
        state: RuntimeState,
        rename_event: dict[str, Any] | None,
        shaping_events: list[dict[str, Any]],
        slow_variables: dict[str, Any],
        identity_context: IdentityContext,
    ) -> dict[str, Any]:
        rename_reason = identity_context.evolution_reason or ("unnamed" if not state.identity_state.display_name else "stable_continuity")
        return {
            "internal_handle": state.identity_state.internal_handle,
            "current_display_name": self.display_label(state),
            "aliases": list(state.identity_state.aliases),
            "rename_event": rename_event,
            "rename_reason": rename_reason,
            "evidence_anchors": list(state.identity_state.evidence_anchors),
            "identity_shaping_sources": self.shaping_sources(shaping_events, slow_variables),
            "non_interactive_sources": list(identity_context.non_interactive_sources),
        }

    def build_identity_prior_contribution(
        self,
        *,
        identity_context: IdentityContext,
        state: RuntimeState,
    ) -> ProbabilisticContribution:
        modulated_delta: dict[str, float] = {}
        query_kind = str(identity_context.query_kind or "general")
        disclosure_intent = str(identity_context.disclosure_intent or "withhold")
        affect_residue = float(getattr(state, "affect_residue", 0.0) or 0.0)
        resource_state = getattr(state, "resource_state", {}) or {}
        if not isinstance(resource_state, dict):
            resource_state = {}
        budget_remaining = float(getattr(state, "budget_remaining", 1.0) or 1.0)
        body_energy = float(getattr(state, "body_energy", 0.55) or 0.55)
        identity_state = getattr(state, "identity_state", None)
        display_name = str(getattr(identity_state, "display_name", "") or "")
        stress_load = _clip(
            max(
                affect_residue,
                float(resource_state.get("scarcity_index", _clip(1.0 - budget_remaining))),
                max(0.0, 0.55 - body_energy),
            ),
            0.0,
            1.0,
        )
        prior_scale = _clip(0.80 - stress_load * 0.45, 0.25, 0.80)

        if query_kind in {"self_identity", "provider_identity"}:
            modulated_delta["respond"] = modulated_delta.get("respond", 0.0) + 0.16 * prior_scale
            modulated_delta["clarify"] = modulated_delta.get("clarify", 0.0) + 0.06 * prior_scale
            modulated_delta["wander"] = modulated_delta.get("wander", 0.0) - 0.06 * prior_scale
        elif query_kind == "answer_explanation":
            modulated_delta["respond"] = modulated_delta.get("respond", 0.0) + 0.10 * prior_scale
            modulated_delta["recall"] = modulated_delta.get("recall", 0.0) + 0.08 * prior_scale
            modulated_delta["wander"] = modulated_delta.get("wander", 0.0) - 0.04 * prior_scale
        else:
            modulated_delta["respond"] = modulated_delta.get("respond", 0.0) + 0.02 * prior_scale

        if disclosure_intent == "withhold":
            modulated_delta["connect"] = modulated_delta.get("connect", 0.0) - 0.05 * prior_scale
            modulated_delta["wander"] = modulated_delta.get("wander", 0.0) - 0.02 * prior_scale
        elif disclosure_intent == "provider_origin":
            modulated_delta["clarify"] = modulated_delta.get("clarify", 0.0) + 0.04 * prior_scale
        elif disclosure_intent == "relational_self_disclosure":
            modulated_delta["connect"] = modulated_delta.get("connect", 0.0) + 0.04 * prior_scale

        if display_name:
            modulated_delta["respond"] = modulated_delta.get("respond", 0.0) + 0.015 * prior_scale

        query_confidence = max(identity_context.query_intent_posterior.values(), default=0.0)
        disclosure_confidence = max(identity_context.disclosure_intent_posterior.values(), default=0.0)
        confidence = _clip(0.20 + max(query_confidence, disclosure_confidence) * 0.40 + prior_scale * 0.15, 0.0, 1.0)

        dependency_trace = [
            f"query_kind:{query_kind}",
            f"query_intent:{identity_context.query_intent}",
            f"disclosure_intent:{disclosure_intent}",
            f"display_label:{identity_context.display_label or self.unnamed_label()}",
        ]
        dependency_trace.extend(f"anchor:{anchor}" for anchor in identity_context.evidence_anchors[:3])

        return ProbabilisticContribution(
            module_name="IdentityRuntime",
            module_type="identity",
            level="action",
            target_space="action",
            raw_signal={action: round(value, 6) for action, value in modulated_delta.items() if abs(value) > 1e-9},
            modulated_delta={action: round(value, 6) for action, value in modulated_delta.items() if abs(value) > 1e-9},
            confidence=confidence,
            confidence_calibrated=round(_clip(confidence * 0.95, 0.0, 1.0), 4),
            trace_reason=f"identity prior query={query_kind} disclosure={disclosure_intent}",
            projection_reason="identity prior projected from identity context",
            applied_at_stage="identity_prior",
            native_operator="identity_prior",
            dependency_trace=dependency_trace,
            projection=EnergyProjectionSpec(module_type="identity", target_space="action", module_temperature=0.9),
        )
