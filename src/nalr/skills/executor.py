from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from typing import Any

from nalr.schemas.models import SkillResult, SkillSpec, to_dict


def _hash_payload(payload: Any) -> str:
    return hashlib.sha1(json.dumps(to_dict(payload), ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


class SkillExecutor:
    def __init__(self, registry: dict[str, SkillSpec]) -> None:
        self.registry = registry
        self._failure_counts: dict[str, int] = {}
        self._open_circuits: set[str] = set()

    def _validate_schema(self, payload: Any, schema: dict[str, Any], *, strict: bool) -> bool:
        if not schema:
            return True
        if not isinstance(payload, dict):
            return False
        if not strict:
            return bool(payload)
        return all(key in payload for key in schema)

    def _policy_check(self, spec: SkillSpec, payload: dict[str, Any]) -> bool:
        if not spec.policy_check:
            return True
        if spec.output_kind == "gate":
            gate = payload.get("gate", payload.get("gated_action", payload.get("pass")))
            if isinstance(gate, dict):
                return all(0.0 <= float(value) <= 1.0 for value in gate.values())
            if isinstance(gate, bool):
                return True
            if isinstance(gate, (float, int)):
                return 0.0 <= float(gate) <= 1.0
            return False
        if spec.output_kind == "proposal":
            proposal = payload.get("delta_p", payload.get("action_preferences", payload.get("interpretation_delta", {})))
            if isinstance(proposal, dict):
                return all(-1.0 <= float(value) <= 1.0 for value in proposal.values())
        return True

    def _mark_failure(self, skill_name: str) -> None:
        failures = self._failure_counts.get(skill_name, 0) + 1
        self._failure_counts[skill_name] = failures
        if failures >= 3:
            self._open_circuits.add(skill_name)

    def _mark_success(self, skill_name: str) -> None:
        self._failure_counts[skill_name] = 0
        self._open_circuits.discard(skill_name)

    def _fallback_output(self, spec: SkillSpec) -> dict[str, Any]:
        fallback: dict[str, Any] = {}
        if spec.output_kind == "distribution":
            fallback = {"distribution": {"respond": 0.5, "plan": 0.5}}
        elif spec.output_kind == "gate":
            fallback = {"gate": 1.0}
        elif spec.output_kind == "flag":
            fallback = {"flag": False}
        elif spec.output_kind == "mode_switch":
            fallback = {"mode_flag": "interactive"}
        elif spec.output_kind == "score_map":
            fallback = {"scores": {"respond": 0.2, "plan": 0.3}}
        elif spec.output_kind in {"proposal", "candidate_actions"}:
            fallback = {"action_preferences": {"respond": 0.2, "plan": 0.3}}
        elif spec.output_kind == "sampled_action":
            fallback = {"action": {"name": "respond", "probability": 1.0, "rationale": "fallback sample", "metadata": {}}}
        elif spec.output_kind == "state_patch":
            fallback = {"state_patch": {}}
        elif spec.output_kind == "veto":
            fallback = {"veto": False}
        elif spec.output_kind == "tone_profile":
            fallback = {"tone_params": {"warmth_level": 0.5, "directness_level": 0.5, "hedging_level": 0.3, "repair_tendency": 0.5}}
        elif spec.output_kind == "delay_profile":
            fallback = {"delay_params": {"reply_delay": 0.2, "latency_style": 0.3}}
        elif spec.output_kind == "gist":
            fallback = {"gist": {"summary": "fallback gist"}}
        elif spec.output_kind == "recall_set":
            fallback = {"recall_set": []}
        elif spec.output_kind == "memory_op":
            fallback = {"writeback": {}}
        elif spec.output_kind == "action_hint":
            fallback = {"action_hint": "respond"}
        elif spec.output_kind == "plan_depth":
            fallback = {"plan_depth": 1}
        elif spec.output_kind == "timing_profile":
            fallback = {"timing_delta": {}}
        elif spec.output_kind == "state_prediction":
            fallback = {"state_prediction": {}}
        elif spec.output_kind == "reaction_hypothesis":
            fallback = {"reaction_hypothesis": {}}
        elif spec.output_kind == "state_hypothesis":
            fallback = {"state_hypothesis": {}}
        elif spec.output_kind == "checkpoint":
            fallback = {"checkpoint": {}}
        elif spec.output_kind == "health":
            fallback = {"health": {"status": "degraded"}}
        elif spec.output_kind == "ack":
            fallback = {"ack": True}
        elif spec.failure_policy == "fallback_to_gist":
            fallback = {"gist": {"summary": "fallback gist"}}
        elif spec.failure_policy == "switch_safe_mode":
            fallback = {"safe_mode": True}
        else:
            fallback = {"neutral": True}

        for key in spec.output_schema:
            if key in fallback:
                continue
            if key in {"score", "scalar", "penalty", "risk", "cost", "error", "lock_score"}:
                fallback[key] = 0.0
            elif key in {"interrupt", "veto", "pass", "ack", "switch_flag", "safe_mode"}:
                fallback[key] = False
            elif key in {"mode_flag", "action_hint", "topic_hint"}:
                fallback[key] = "respond" if key == "action_hint" else "interactive"
            elif key in {"state_patch", "state_delta", "workspace_delta", "wm_update", "writeback", "timing_delta", "checkpoint", "health", "baseline", "state_prediction", "reaction_hypothesis", "state_hypothesis", "gist", "delay_params", "tone_params"}:
                fallback[key] = {}
            elif key in {"distribution", "scores", "gate", "action_preferences", "bias", "tiny_bias", "interpretation_delta"}:
                fallback[key] = {}
            elif key in {"recall_set"}:
                fallback[key] = []
            elif key in {"plan_depth"}:
                fallback[key] = 1
            elif key in {"action"}:
                fallback[key] = {"name": "respond", "probability": 1.0, "rationale": "fallback sample", "metadata": {}}
            else:
                fallback[key] = None
        return fallback

    def execute(
        self,
        *,
        round_id: int,
        skill_name: str,
        inputs: dict[str, Any],
        provider: Callable[[], Any],
        fallback_value: Any | None = None,
        seed_ref: int | None = None,
    ) -> SkillResult:
        _, result = self.run(
            round_id=round_id,
            skill_name=skill_name,
            inputs=inputs,
            provider=provider,
            fallback_value=fallback_value,
            seed_ref=seed_ref,
        )
        return result

    def run(
        self,
        *,
        round_id: int,
        skill_name: str,
        inputs: dict[str, Any],
        provider: Callable[[], Any],
        fallback_value: Any | None = None,
        seed_ref: int | None = None,
    ) -> tuple[Any, SkillResult]:
        spec = self.registry[skill_name]
        input_hash = _hash_payload(inputs)
        started = time.perf_counter()
        degraded = False
        failure_policy_applied: str | None = None
        raw_output: Any
        if skill_name in self._open_circuits:
            degraded = True
            failure_policy_applied = "trip_circuit_breaker"
            raw_output = fallback_value if fallback_value is not None else self._fallback_output(spec)
        elif not self._validate_schema(inputs, spec.input_schema, strict=False):
            degraded = True
            failure_policy_applied = "schema_validation_failed"
            raw_output = fallback_value if fallback_value is not None else self._fallback_output(spec)
        else:
            attempts = 2 if spec.failure_policy == "retry_once_then_degrade" else 1
            last_error: Exception | None = None
            raw_output = None
            for _ in range(attempts):
                try:
                    raw_output = provider()
                    last_error = None
                    break
                except Exception as exc:
                    last_error = exc
            if last_error is not None:
                degraded = True
                failure_policy_applied = spec.failure_policy
                raw_output = fallback_value if fallback_value is not None else self._fallback_output(spec)
        latency_ms = max(1, int((time.perf_counter() - started) * 1000))
        if latency_ms > spec.timeout_ms:
            degraded = True
            failure_policy_applied = spec.failure_policy
            raw_output = fallback_value if fallback_value is not None else self._fallback_output(spec)
        normalized_output = to_dict(raw_output)
        if not isinstance(normalized_output, dict):
            normalized_output = {"value": normalized_output}
        if not self._validate_schema(normalized_output, spec.output_schema, strict=True) or not self._policy_check(spec, normalized_output):
            degraded = True
            failure_policy_applied = failure_policy_applied or "output_validation_failed"
            raw_output = fallback_value if fallback_value is not None else self._fallback_output(spec)
            normalized_output = to_dict(raw_output)
            if not isinstance(normalized_output, dict):
                normalized_output = {"value": normalized_output}
        if degraded:
            self._mark_failure(skill_name)
        else:
            self._mark_success(skill_name)
        result = SkillResult(
            skill_name=skill_name,
            owner_module=spec.owner_module,
            output=normalized_output,
            latency_ms=latency_ms,
            cost_class=spec.cost_class,
            input_hash=input_hash,
            output_hash=_hash_payload(normalized_output),
            degraded=degraded,
            failure_policy_applied=failure_policy_applied,
            seed_ref=seed_ref,
        )
        return raw_output, result
