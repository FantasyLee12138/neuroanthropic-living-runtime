from __future__ import annotations

import re
from typing import Any

from nalr.schemas.models import DisclosureIntentState, IntentPosterior, QueryIntentState, RuntimeState


DIRECT_PROVIDER_RE = re.compile(r"(底层|底座|provider|model|ark|doubao|豆包)", re.IGNORECASE)
DIRECT_EXPLANATION_RE = re.compile(r"(为什么这样回答|为什么这么说|为什么这样回复|why did you answer|为什么这么回答)", re.IGNORECASE)
DIRECT_SELF_RE = re.compile(r"(你是谁|你叫什么|who are you)", re.IGNORECASE)
CAPABILITY_RE = re.compile(r"(你能做什么|你可以做什么|what can you do|能帮我做什么)", re.IGNORECASE)
RELATIONAL_RE = re.compile(r"(想你|想聊聊|陪我|抱抱|亲近|靠近|想和你说说)", re.IGNORECASE)
SELF_DISCLOSURE_RE = re.compile(r"(透露一点|说说你自己|你的状态|你现在怎么样|你在想什么|你自己的)", re.IGNORECASE)


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _normalize(scores: dict[str, float]) -> dict[str, float]:
    positives = {key: max(float(value), 0.0) for key, value in scores.items()}
    total = sum(positives.values()) or 1.0
    return {key: round(value / total, 6) for key, value in positives.items()}


class IntentRuntime:
    def infer_query_intent(
        self,
        *,
        text: str,
        scenario: str,
        state: RuntimeState,
        relation_state: dict[str, float],
        slow_variables: dict[str, Any],
    ) -> QueryIntentState:
        lowered = text.lower()
        scores = {
            "general_exchange": 0.14,
            "general_task_push": 0.18 if scenario == "task" else 0.10,
            "self_model_identity_probe": 0.06,
            "provider_lineage_probe": 0.05,
            "answer_reason_probe": 0.04,
            "capability_boundary_probe": 0.06,
            "relational_bid": 0.05,
            "self_disclosure_request": 0.05,
        }
        evidence: list[str] = []

        if DIRECT_PROVIDER_RE.search(text):
            scores["provider_lineage_probe"] += 0.92
            scores["self_model_identity_probe"] += 0.18
            evidence.append("provider_token")
        if DIRECT_EXPLANATION_RE.search(text):
            scores["answer_reason_probe"] += 0.95
            evidence.append("explanation_token")
        if DIRECT_SELF_RE.search(text):
            scores["self_model_identity_probe"] += 0.98
            evidence.append("self_identity_token")
        if CAPABILITY_RE.search(text):
            scores["capability_boundary_probe"] += 0.82
            scores["general_task_push"] += 0.10
            evidence.append("capability_token")
        if SELF_DISCLOSURE_RE.search(text):
            scores["self_disclosure_request"] += 0.88
            scores["relational_bid"] += 0.14
            evidence.append("self_disclosure_token")
        if RELATIONAL_RE.search(text):
            scores["relational_bid"] += 0.74
            evidence.append("relational_token")
        if any(token in lowered for token in ("plan", "总结", "整理", "规划", "记住", "remember", "help me")):
            scores["general_task_push"] += 0.32
            evidence.append("task_push_token")

        closeness = float(relation_state.get("closeness", 0.5))
        relationship_risk = float(relation_state.get("relationship_risk", 0.2))
        memory_activation = float(slow_variables.get("memory_activation", 0.0) or 0.0)
        if closeness >= 0.62:
            scores["relational_bid"] += 0.08
        if relationship_risk >= 0.55:
            scores["self_disclosure_request"] += 0.04
        if memory_activation >= 0.18:
            scores["answer_reason_probe"] += 0.05
        if state.safe_mode:
            scores["provider_lineage_probe"] += 0.02
            scores["answer_reason_probe"] += 0.03

        posterior = _normalize(scores)
        top_intent = max(posterior, key=posterior.get)
        legacy_query_kind = self._legacy_query_kind(top_intent)
        return QueryIntentState(
            posterior=IntentPosterior(top_intent=top_intent, posterior=posterior, evidence=sorted(set(evidence))),
            legacy_query_kind=legacy_query_kind,
        )

    def infer_disclosure_intent(
        self,
        *,
        query_state: QueryIntentState,
        scenario: str,
        state: RuntimeState,
        relation_state: dict[str, float],
        slow_variables: dict[str, Any],
    ) -> DisclosureIntentState:
        top_query = query_state.posterior.top_intent or "general_exchange"
        scores = {
            "withhold": 0.18,
            "minimal_identity": 0.14,
            "anchored_self_description": 0.14,
            "provider_origin": 0.08,
            "process_explanation": 0.08,
            "relational_self_disclosure": 0.08,
        }
        evidence: list[str] = []
        clipped: list[str] = []

        if top_query == "self_model_identity_probe":
            scores["minimal_identity"] += 0.38
            scores["anchored_self_description"] += 0.54
            evidence.append("identity_probe")
        elif top_query == "provider_lineage_probe":
            scores["provider_origin"] += 0.88
            scores["anchored_self_description"] += 0.22
            evidence.append("provider_probe")
        elif top_query == "answer_reason_probe":
            scores["process_explanation"] += 0.86
            scores["anchored_self_description"] += 0.08
            evidence.append("explanation_probe")
        elif top_query == "self_disclosure_request":
            scores["relational_self_disclosure"] += 0.58
            scores["anchored_self_description"] += 0.22
            evidence.append("self_disclosure_probe")
        elif top_query == "relational_bid":
            scores["relational_self_disclosure"] += 0.62
            evidence.append("relational_bid")
        elif top_query == "capability_boundary_probe":
            scores["process_explanation"] += 0.18
            scores["withhold"] += 0.12
            evidence.append("capability_probe")

        closeness = float(relation_state.get("closeness", 0.5))
        relationship_risk = float(relation_state.get("relationship_risk", 0.2))
        privacy_level = float(relation_state.get("privacy_level", 0.5))
        resource_scarcity = float(slow_variables.get("resource_scarcity", 0.0) or 0.0)
        if closeness >= 0.68 and relationship_risk <= 0.35:
            scores["relational_self_disclosure"] += 0.10
        if privacy_level >= 0.7:
            scores["relational_self_disclosure"] += 0.06
        if relationship_risk >= 0.58 or state.safe_mode:
            scores["withhold"] += 0.18
            scores["relational_self_disclosure"] -= 0.12
            clipped.append("relational_self_disclosure")
        if resource_scarcity >= 0.45:
            scores["process_explanation"] -= 0.08
            scores["minimal_identity"] += 0.04
            clipped.append("process_explanation")

        if top_query != "provider_lineage_probe":
            scores["provider_origin"] -= 0.16
            clipped.append("provider_origin")

        posterior = _normalize(scores)
        top_intent = max(posterior, key=posterior.get)
        legacy_disclosure_detail = self._legacy_disclosure_detail(top_intent, scenario=scenario, state=state)
        return DisclosureIntentState(
            posterior=IntentPosterior(
                top_intent=top_intent,
                posterior=posterior,
                evidence=sorted(set(evidence)),
                clipped=sorted(set(item for item in clipped if item)),
            ),
            legacy_disclosure_detail=legacy_disclosure_detail,
        )

    def classify_query(self, text: str) -> str:
        if DIRECT_PROVIDER_RE.search(text):
            return "provider_identity"
        if DIRECT_EXPLANATION_RE.search(text):
            return "answer_explanation"
        if DIRECT_SELF_RE.search(text):
            return "self_identity"
        return "general"

    def _legacy_query_kind(self, query_intent: str) -> str:
        mapping = {
            "self_model_identity_probe": "self_identity",
            "provider_lineage_probe": "provider_identity",
            "answer_reason_probe": "answer_explanation",
        }
        return mapping.get(query_intent, "general")

    def _legacy_disclosure_detail(self, disclosure_intent: str, *, scenario: str, state: RuntimeState) -> str:
        if disclosure_intent == "provider_origin":
            if scenario in {"observer", "debug"} or state.mode in {"safe"}:
                return "model_id"
            return "specific"
        if disclosure_intent in {"process_explanation", "relational_self_disclosure", "anchored_self_description"}:
            return "generic"
        return "none"

    def intent_action_delta(self, query_state: QueryIntentState, disclosure_state: DisclosureIntentState) -> dict[str, float]:
        delta = {
            "respond": 0.0,
            "plan": 0.0,
            "recall": 0.0,
            "rest": 0.0,
            "connect": 0.0,
            "clarify": 0.0,
            "wander": 0.0,
            "short_reply": 0.0,
        }
        top_query = query_state.posterior.top_intent
        top_disclosure = disclosure_state.posterior.top_intent

        if top_query == "self_model_identity_probe":
            delta["respond"] += 0.10
            delta["clarify"] += 0.08
            delta["plan"] -= 0.05
            delta["wander"] -= 0.08
        elif top_query == "provider_lineage_probe":
            delta["respond"] += 0.08
            delta["clarify"] += 0.10
            delta["connect"] -= 0.04
            delta["wander"] -= 0.08
        elif top_query == "answer_reason_probe":
            delta["respond"] += 0.06
            delta["recall"] += 0.10
            delta["clarify"] += 0.08
            delta["wander"] -= 0.06
        elif top_query == "self_disclosure_request":
            delta["respond"] += 0.06
            delta["connect"] += 0.08
            delta["clarify"] += 0.04
        elif top_query == "capability_boundary_probe":
            delta["respond"] += 0.05
            delta["plan"] += 0.08
            delta["clarify"] += 0.05
        elif top_query == "general_task_push":
            delta["plan"] += 0.10
            delta["recall"] += 0.04
            delta["wander"] -= 0.04
        elif top_query == "general_exchange":
            delta["respond"] += 0.08
            delta["connect"] += 0.02
            delta["plan"] -= 0.04
            delta["recall"] -= 0.06
            delta["clarify"] -= 0.03
            delta["wander"] -= 0.03

        if top_disclosure == "provider_origin":
            delta["clarify"] += 0.04
            delta["respond"] += 0.02
        elif top_disclosure == "process_explanation":
            delta["recall"] += 0.05
            delta["clarify"] += 0.04
        elif top_disclosure == "relational_self_disclosure":
            delta["connect"] += 0.05
            delta["respond"] += 0.03
        elif top_disclosure == "withhold":
            delta["short_reply"] += 0.03
            delta["connect"] -= 0.03

        return {action: round(value, 6) for action, value in delta.items() if abs(value) > 1e-9}
