from __future__ import annotations

import hashlib
from typing import Any, Callable

from nalr.runtime.intent import IntentRuntime
from nalr.schemas.models import DisclosureIntentState, IdentityContext, QueryIntentState, RuntimeState
from nalr.schemas.models import to_dict


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

    def _summarize_identity_signals(
        self,
        *,
        state: RuntimeState,
        evidence: dict[str, Any],
        query_state: QueryIntentState,
        disclosure_state: DisclosureIntentState,
        scenario: str,
        shaping_events: list[dict[str, Any]],
        slow_variables: dict[str, Any],
        state_sources: list[str],
        rename_event: dict[str, Any] | None,
    ) -> dict[str, Any]:
        stable_priors = evidence.get("stable_priors", [])
        top_stable = float(stable_priors[0].get("weight", 0.0)) if stable_priors else 0.0
        second_stable = float(stable_priors[1].get("weight", 0.0)) if len(stable_priors) > 1 else 0.0
        top_habit = float(evidence.get("habits", [{}])[0].get("strength", 0.0)) if evidence.get("habits") else 0.0
        top_relation = float(evidence.get("relations", [{}])[0].get("closeness", 0.0)) if evidence.get("relations") else 0.0
        identity_score = float(evidence.get("identity_score", 0.0) or 0.0)
        naming_signal = float(evidence.get("naming_signal", 0.0) or 0.0)
        continuity_signal = float(evidence.get("continuity_signal", 0.0) or 0.0)
        drift_score = self.anchor_drift_score(list(state.identity_state.evidence_anchors), list(evidence.get("anchors", [])))
        resource_scarcity = float(state.resource_state.get("scarcity_index", _clip(1.0 - state.budget_remaining)))
        return {
            "scenario": scenario,
            "query_kind": query_state.legacy_query_kind,
            "disclosure_detail": self.disclosure_detail(
                disclosure_state=disclosure_state,
                scenario=scenario,
                state=state,
            ),
            "query_intent": query_state.posterior.top_intent,
            "disclosure_intent": disclosure_state.posterior.top_intent,
            "identity_score": round(identity_score, 4),
            "naming_signal": round(naming_signal, 4),
            "continuity_signal": round(continuity_signal, 4),
            "top_stable_prior": round(top_stable, 4),
            "second_stable_prior": round(second_stable, 4),
            "top_habit": round(top_habit, 4),
            "top_relation": round(top_relation, 4),
            "drift_score": round(drift_score, 4),
            "resource_scarcity": round(resource_scarcity, 4),
            "rename_event_present": rename_event is not None,
            "state_sources": list(state_sources),
            "shaping_sources": self.shaping_sources(shaping_events, slow_variables),
        }

    def _identity_prior_kernel(self, state: RuntimeState, signals: dict[str, Any], identity_context: IdentityContext) -> dict[str, Any]:
        query_kind = identity_context.query_kind
        disclosure_detail = identity_context.disclosure_detail
        identity_score = float(signals.get("identity_score", 0.0))
        naming_signal = float(signals.get("naming_signal", 0.0))
        continuity_signal = float(signals.get("continuity_signal", 0.0))
        drift_score = float(signals.get("drift_score", 0.0))
        top_relation = float(signals.get("top_relation", 0.0))
        top_habit = float(signals.get("top_habit", 0.0))
        resource_scarcity = float(signals.get("resource_scarcity", 0.0))

        trait_prior = _clip(0.42 + identity_score * 0.28 + continuity_signal * 0.12 - float(state.affect_residue) * 0.10)
        belief_prior = _clip(0.38 + naming_signal * 0.18 + top_relation * 0.20 + continuity_signal * 0.10)
        style_prior = _clip(0.35 + min(len(state.identity_state.aliases), 5) * 0.03 + float(state.mood) * 0.10)
        boundary_prior = _clip(0.45 + (1.0 - top_relation) * 0.18 + resource_scarcity * 0.16)
        continuity_prior = _clip(0.40 + (1.0 - drift_score) * 0.34 + top_habit * 0.06 + identity_score * 0.08)

        if query_kind in {"self_identity", "provider_identity"}:
            boundary_prior = _clip(boundary_prior + 0.12)
            continuity_prior = _clip(continuity_prior + 0.08)
        if query_kind == "answer_explanation":
            belief_prior = _clip(belief_prior + 0.08)
            style_prior = _clip(style_prior + 0.04)
        if disclosure_detail in {"withhold", "none"}:
            boundary_prior = _clip(boundary_prior + 0.08)
        if disclosure_detail in {"specific", "model_id"}:
            style_prior = _clip(style_prior + 0.03)

        return {
            "trait_prior": round(trait_prior, 4),
            "belief_prior": round(belief_prior, 4),
            "style_prior": round(style_prior, 4),
            "boundary_prior": round(boundary_prior, 4),
            "continuity_prior": round(continuity_prior, 4),
        }

    def _identity_delta_logits(self, query_kind: str, disclosure_detail: str, signals: dict[str, Any]) -> dict[str, float]:
        identity_score = float(signals.get("identity_score", 0.0))
        continuity_signal = float(signals.get("continuity_signal", 0.0))
        drift_score = float(signals.get("drift_score", 0.0))
        resource_scarcity = float(signals.get("resource_scarcity", 0.0))

        biases = {
            "respond": 0.04,
            "clarify": 0.02,
            "short_reply": 0.0,
            "recall": 0.0,
            "plan": 0.0,
            "connect": 0.0,
            "wander": -0.03,
        }
        if query_kind in {"self_identity", "provider_identity"}:
            biases["respond"] += 0.14 + identity_score * 0.08
            biases["clarify"] += 0.08 + continuity_signal * 0.04
            biases["short_reply"] += 0.06
            biases["wander"] -= 0.10 + drift_score * 0.04
        elif query_kind == "answer_explanation":
            biases["respond"] += 0.10 + continuity_signal * 0.05
            biases["recall"] += 0.08
            biases["clarify"] += 0.05
            biases["plan"] += 0.03
            biases["wander"] -= 0.06
        else:
            biases["respond"] += 0.05
            biases["clarify"] += 0.04
            biases["plan"] += 0.02

        if disclosure_detail in {"withhold", "none"}:
            biases["short_reply"] += 0.05
            biases["connect"] -= 0.03
        if disclosure_detail in {"specific", "model_id"}:
            biases["respond"] += 0.03
            biases["clarify"] += 0.02
        if resource_scarcity > 0.45:
            biases["plan"] -= 0.03
            biases["wander"] -= 0.02

        return {action: round(_clip(value, -0.35, 0.35), 4) for action, value in biases.items()}

    def build_identity_prior_payload(
        self,
        *,
        state: RuntimeState,
        query_state: QueryIntentState,
        disclosure_state: DisclosureIntentState,
        scenario: str,
        shaping_events: list[dict[str, Any]],
        slow_variables: dict[str, Any],
        state_sources: list[str],
        rename_event: dict[str, Any] | None,
        evidence: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        evidence_payload = self.augment_identity_evidence(state, evidence or {})
        identity_context = self.build_identity_context(
            query_state=query_state,
            disclosure_state=disclosure_state,
            scenario=scenario,
            state=state,
            shaping_events=shaping_events,
            slow_variables=slow_variables,
            state_sources=state_sources,
            rename_event=rename_event,
        )
        signals = self._summarize_identity_signals(
            state=state,
            evidence=evidence_payload,
            query_state=query_state,
            disclosure_state=disclosure_state,
            scenario=scenario,
            shaping_events=shaping_events,
            slow_variables=slow_variables,
            state_sources=state_sources,
            rename_event=rename_event,
        )
        prior_kernel = self._identity_prior_kernel(state, signals, identity_context)
        delta_logits = self._identity_delta_logits(identity_context.query_kind, identity_context.disclosure_detail, signals)
        confidence = _clip(
            0.34
            + signals["identity_score"] * 0.24
            + signals["continuity_signal"] * 0.18
            + signals["naming_signal"] * 0.12
            + prior_kernel["boundary_prior"] * 0.06
        )
        trace_reason = (
            f"query={identity_context.query_kind}; disclosure={identity_context.disclosure_detail}; "
            f"identity={signals['identity_score']:.2f}; continuity={signals['continuity_signal']:.2f}"
        )
        soft_mask = {
            "wander": round(_clip(0.16 + signals["drift_score"] * 0.10 + signals["resource_scarcity"] * 0.04), 4),
            "connect": round(_clip(0.04 + prior_kernel["boundary_prior"] * 0.06), 4),
        }
        attention_bias = {
            "identity_evidence": round(_clip(signals["identity_score"] + signals["naming_signal"] * 0.5), 4),
            "continuity": round(_clip(signals["continuity_signal"] + signals["top_stable_prior"] * 0.4), 4),
            "boundary": round(_clip(prior_kernel["boundary_prior"]), 4),
        }
        return {
            "module_name": "IdentityRuntime",
            "layer": "global",
            "kind": "prior_kernel",
            "prior_role": "stable_prior",
            "identity_score": round(signals["identity_score"], 4),
            "confidence": round(confidence, 4),
            "identity_prior_kernel": prior_kernel,
            "delta_logits": delta_logits,
            "attention_bias": attention_bias,
            "soft_mask": soft_mask,
            "hard_mask": {},
            "confidence": round(confidence, 4),
            "trace_reason": trace_reason,
            "identity_context": to_dict(identity_context),
            "identity_signals": signals,
            "identity_evidence": evidence_payload,
        }

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
        resource_scarcity = float(state.resource_state.get("scarcity_index", _clip(1.0 - state.budget_remaining)))
        continuity_prior = _clip(0.48 + (1.0 - float(len(state.identity_state.evidence_anchors) > 0)) * 0.04 + float(state.mood) * 0.10)
        boundary_prior = _clip(0.50 + resource_scarcity * 0.10 + len(identity_context.disclosure_clipped) * 0.03)
        identity_prior_kernel = {
            "trait_prior": round(_clip(0.42 + float(state.mood) * 0.12 - float(state.affect_residue) * 0.08), 4),
            "belief_prior": round(_clip(0.40 + len(identity_context.evidence_anchors) * 0.03 + len(identity_context.self_description_sources) * 0.02), 4),
            "style_prior": round(_clip(0.35 + min(len(state.identity_state.aliases), 5) * 0.03), 4),
            "boundary_prior": round(boundary_prior, 4),
            "continuity_prior": round(continuity_prior, 4),
        }
        identity_prior_bias = {
            "respond": round(_clip(0.08 + (0.08 if identity_context.query_kind in {"self_identity", "provider_identity"} else 0.03)), 4),
            "clarify": round(_clip(0.06 + len(identity_context.evidence_anchors) * 0.01), 4),
            "short_reply": round(_clip(0.04 + (0.02 if identity_context.disclosure_detail in {"withhold", "none"} else 0.0)), 4),
            "wander": round(_clip(0.12 + resource_scarcity * 0.04), 4),
        }
        return {
            "internal_handle": state.identity_state.internal_handle,
            "current_display_name": self.display_label(state),
            "aliases": list(state.identity_state.aliases),
            "rename_event": rename_event,
            "rename_reason": rename_reason,
            "evidence_anchors": list(state.identity_state.evidence_anchors),
            "identity_shaping_sources": self.shaping_sources(shaping_events, slow_variables),
            "non_interactive_sources": list(identity_context.non_interactive_sources),
            "identity_score": round(float(identity_context.query_intent_posterior.get(identity_context.query_intent, 0.0)), 4),
            "confidence": round(_clip(0.40 + float(state.mood) * 0.08 + float(state.identity_state.drift_credit) * 0.05), 4),
            "identity_prior_kernel": identity_prior_kernel,
            "identity_prior_bias": identity_prior_bias,
            "prior_role": "stable_prior",
        }
