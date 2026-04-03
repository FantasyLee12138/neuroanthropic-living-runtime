from __future__ import annotations

import hashlib
import inspect
import json
import time
from dataclasses import MISSING, fields, is_dataclass
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

from nalr.schemas.models import (
    ActionCandidate,
    CircuitBreakerState,
    ProposalBundle,
    SkillResult,
    SkillRuntimeContext,
    SkillSpec,
    to_dict,
)
from nalr.skills.contracts import coerce_contract


def _hash_payload(payload: Any) -> str:
    return hashlib.sha1(json.dumps(to_dict(payload), ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


class PolicyViolation(Exception):
    pass


class SkillExecutor:
    def __init__(self, registry: dict[str, SkillSpec], circuit_breaker_path: Path | None = None) -> None:
        self.registry = registry
        self.circuit_breaker_path = Path(circuit_breaker_path) if circuit_breaker_path else None
        self._breaker_cache: dict[str, CircuitBreakerState] = {}

    def _load_breakers(self) -> dict[str, CircuitBreakerState]:
        if self.circuit_breaker_path is None:
            return self._breaker_cache
        if not self.circuit_breaker_path.exists():
            self.circuit_breaker_path.write_text("{}", encoding="utf-8")
        payload = json.loads(self.circuit_breaker_path.read_text(encoding="utf-8") or "{}")
        if isinstance(payload, list):
            payload = {}
        self._breaker_cache = {name: CircuitBreakerState(**state) for name, state in payload.items()}
        return self._breaker_cache

    def _save_breakers(self) -> None:
        if self.circuit_breaker_path is None:
            return
        payload = {name: to_dict(state) for name, state in self._breaker_cache.items()}
        self.circuit_breaker_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _breaker_for(self, skill_name: str) -> CircuitBreakerState:
        return self._load_breakers().setdefault(skill_name, CircuitBreakerState())

    def _breaker_is_open(self, spec: SkillSpec, state: CircuitBreakerState, round_id: int) -> bool:
        if spec.breaker_policy is None or state.open_until_round is None:
            return False
        return round_id <= state.open_until_round

    def _record_failure(self, spec: SkillSpec, skill_name: str, round_id: int, reason: str) -> CircuitBreakerState:
        state = self._breaker_for(skill_name)
        state.failure_count += 1
        state.last_failure_round = round_id
        state.last_failure_reason = reason
        state.fallback_route = spec.fallback_route.target if spec.fallback_route else None
        if spec.breaker_policy and state.failure_count >= spec.breaker_policy.failure_threshold:
            state.open_until_round = round_id + spec.breaker_policy.cooldown_rounds
        self._save_breakers()
        return state

    def _record_success(self, skill_name: str) -> CircuitBreakerState:
        state = self._breaker_for(skill_name)
        state.failure_count = 0
        state.open_until_round = None
        state.last_failure_round = None
        state.last_failure_reason = None
        state.fallback_route = None
        self._save_breakers()
        return state

    def _invoke_callable(self, provider: Any, validated_inputs: dict[str, Any]) -> Any:
        signature = inspect.signature(provider)
        params = list(signature.parameters.values())
        expects_kwargs = any(
            parameter.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.VAR_KEYWORD)
            for parameter in params
        )
        if expects_kwargs:
            return provider(**validated_inputs)
        return provider()

    def _policy_precheck(self, spec: SkillSpec, runtime_context: SkillRuntimeContext) -> str | None:
        if spec.permission.allowed_operator_levels and runtime_context.operator_level not in spec.permission.allowed_operator_levels:
            return "operator_level_denied"
        if spec.permission.external_io and runtime_context.policy_flags.get("deny_external_io"):
            return "external_io_denied"
        if spec.permission.social_risk and runtime_context.policy_flags.get("deny_social_inference"):
            return "social_inference_denied"
        return None

    def _policy_postcheck(self, spec: SkillSpec, output: Any) -> str | None:
        if not spec.policy_check:
            return None
        if isinstance(output, ProposalBundle):
            if not 0.0 <= float(output.confidence) <= 1.0:
                return "proposal_confidence_out_of_range"
            if not 0.60 <= float(output.sigma_scale) <= 1.60:
                return "proposal_sigma_scale_out_of_range"
            if any(abs(float(value)) > 1.0 for value in output.action_preferences.values()):
                return "proposal_action_preferences_out_of_range"
            if any(abs(float(value)) > 1.0 for value in output.delta_p.values()):
                return "proposal_delta_p_out_of_range"
            return None

        payload = to_dict(output)
        if spec.output_kind == "gate":
            gate = payload.get("gate", payload.get("pass"))
            if isinstance(gate, dict):
                if any(not 0.0 <= float(value) <= 1.0 for value in gate.values()):
                    return "gate_out_of_range"
            elif isinstance(gate, (int, float)):
                if not 0.0 <= float(gate) <= 1.0:
                    return "gate_out_of_range"
        if spec.output_kind == "reaction_hypothesis":
            risk = payload.get("reaction_hypothesis", {}).get("risk")
            if risk is not None and not 0.0 <= float(risk) <= 1.0:
                return "reaction_risk_out_of_range"
        if spec.output_kind == "rendered_expression":
            if not payload.get("text", "").strip():
                return "render_text_empty"
        return None

    def _default_value(self, contract: Any, spec: SkillSpec) -> Any:
        origin = get_origin(contract)
        if isinstance(contract, dict):
            return {key: self._default_value(value, spec) for key, value in contract.items()}
        if contract is Any:
            return {}
        if origin is list:
            return []
        if origin is dict:
            return {}
        if origin is not None and type(None) in get_args(contract):
            return None
        if contract is str:
            if spec.output_kind == "action_hint":
                return "respond"
            if spec.output_kind == "mode_switch":
                return "interactive"
            return ""
        if contract is int:
            return 0
        if contract is float:
            return 0.0
        if contract is bool:
            return False
        if contract is ProposalBundle:
            return ProposalBundle(owner=spec.owner_module, action_preferences={"respond": 0.0}, delta_p={"respond": 0.0}, reason="typed fallback")
        if contract is ActionCandidate:
            return ActionCandidate(name="respond", probability=1.0, rationale="typed fallback")
        if isinstance(contract, type) and is_dataclass(contract):
            values: dict[str, Any] = {}
            type_hints = get_type_hints(contract)
            for field in fields(contract):
                field_contract = type_hints.get(field.name, field.type)
                if field.default is not MISSING or field.default_factory is not MISSING:
                    continue
                if field.name == "owner":
                    values[field.name] = spec.owner_module
                elif field.name == "name":
                    values[field.name] = "respond"
                elif field.name == "route":
                    values[field.name] = "fallback"
                elif field.name == "model":
                    values[field.name] = "fallback"
                elif field.name == "text":
                    values[field.name] = ""
                elif field.name == "rationale":
                    values[field.name] = "typed fallback"
                elif field.name == "probability":
                    values[field.name] = 1.0
                else:
                    values[field.name] = self._default_value(field_contract, spec)
            return contract(**values)
        return {}

    def _coerce_output(self, output: Any, spec: SkillSpec) -> Any:
        return coerce_contract(output, spec.output_schema, path=f"{spec.name}.output")

    def _apply_fallback(
        self,
        *,
        spec: SkillSpec,
        validated_inputs: dict[str, Any] | None,
        fallback_provider: Any | None,
        fallback_value: Any | None,
    ) -> tuple[Any, str | None, str | None]:
        candidate = None
        if fallback_provider is not None and validated_inputs is not None:
            try:
                candidate = self._invoke_callable(fallback_provider, validated_inputs)
            except Exception:
                candidate = None
        if candidate is None and fallback_value is not None:
            candidate = fallback_value
        if candidate is None:
            candidate = self._default_value(spec.output_schema, spec)
        try:
            validated = self._coerce_output(candidate, spec)
        except ValueError:
            validated = self._coerce_output(self._default_value(spec.output_schema, spec), spec)
        return validated, spec.fallback_route.target if spec.fallback_route else None, spec.fallback_route.cost_class if spec.fallback_route else None

    def execute(
        self,
        *,
        round_id: int,
        skill_name: str,
        inputs: dict[str, Any],
        provider: Any,
        fallback_provider: Any | None = None,
        fallback_value: Any | None = None,
        runtime_context: SkillRuntimeContext | None = None,
        seed_ref: int | None = None,
    ) -> SkillResult:
        _, result = self.run(
            round_id=round_id,
            skill_name=skill_name,
            inputs=inputs,
            provider=provider,
            fallback_provider=fallback_provider,
            fallback_value=fallback_value,
            runtime_context=runtime_context,
            seed_ref=seed_ref,
        )
        return result

    def run(
        self,
        *,
        round_id: int,
        skill_name: str,
        inputs: dict[str, Any],
        provider: Any,
        fallback_provider: Any | None = None,
        fallback_value: Any | None = None,
        runtime_context: SkillRuntimeContext | None = None,
        seed_ref: int | None = None,
    ) -> tuple[Any, SkillResult]:
        spec = self.registry[skill_name]
        context = runtime_context or SkillRuntimeContext(round_id=round_id, scenario="", mode="interactive")
        input_hash = _hash_payload(inputs)
        started = time.perf_counter()

        degraded = False
        failure_policy_applied: str | None = None
        fallback_route: str | None = None
        fallback_cost_class: str | None = None
        policy_rejection_reason: str | None = None
        validated_inputs: dict[str, Any] | None = None

        try:
            validated_inputs = coerce_contract(inputs, spec.input_schema, path=f"{skill_name}.inputs")
        except ValueError:
            degraded = True
            failure_policy_applied = "input_validation_failed"
            raw_output, fallback_route, fallback_cost_class = self._apply_fallback(
                spec=spec,
                validated_inputs=None,
                fallback_provider=fallback_provider,
                fallback_value=fallback_value,
            )
        else:
            breaker_state = self._breaker_for(skill_name)
            if self._breaker_is_open(spec, breaker_state, round_id):
                degraded = True
                failure_policy_applied = "trip_circuit_breaker"
                raw_output, fallback_route, fallback_cost_class = self._apply_fallback(
                    spec=spec,
                    validated_inputs=validated_inputs,
                    fallback_provider=fallback_provider,
                    fallback_value=fallback_value,
                )
            else:
                policy_rejection_reason = self._policy_precheck(spec, context)
                if policy_rejection_reason is not None:
                    degraded = True
                    failure_policy_applied = spec.failure_policy
                    raw_output, fallback_route, fallback_cost_class = self._apply_fallback(
                        spec=spec,
                        validated_inputs=validated_inputs,
                        fallback_provider=fallback_provider,
                        fallback_value=fallback_value,
                    )
                else:
                    attempts = 2 if spec.failure_policy == "retry_once_then_degrade" else 1
                    last_error: Exception | None = None
                    raw_output = None
                    for _ in range(attempts):
                        try:
                            candidate = self._invoke_callable(provider, validated_inputs)
                            validated_output = self._coerce_output(candidate, spec)
                            policy_failure = self._policy_postcheck(spec, validated_output)
                            if policy_failure is not None:
                                raise PolicyViolation(policy_failure)
                            raw_output = validated_output
                            last_error = None
                            break
                        except (PolicyViolation, ValueError, Exception) as exc:  # noqa: PERF203
                            last_error = exc
                            if isinstance(exc, (PolicyViolation, ValueError)):
                                break
                    if last_error is not None:
                        degraded = True
                        if isinstance(last_error, PolicyViolation):
                            failure_policy_applied = spec.failure_policy
                            policy_rejection_reason = str(last_error)
                        elif isinstance(last_error, ValueError):
                            failure_policy_applied = "output_validation_failed"
                        else:
                            failure_policy_applied = spec.failure_policy
                        raw_output, fallback_route, fallback_cost_class = self._apply_fallback(
                            spec=spec,
                            validated_inputs=validated_inputs,
                            fallback_provider=fallback_provider,
                            fallback_value=fallback_value,
                        )

        latency_ms = max(1, int((time.perf_counter() - started) * 1000))
        if latency_ms > spec.timeout_ms:
            degraded = True
            failure_policy_applied = spec.failure_policy
            raw_output, fallback_route, fallback_cost_class = self._apply_fallback(
                spec=spec,
                validated_inputs=validated_inputs,
                fallback_provider=fallback_provider,
                fallback_value=fallback_value,
            )

        if degraded and failure_policy_applied not in {"input_validation_failed", "trip_circuit_breaker"} and policy_rejection_reason is None:
            breaker_state = self._record_failure(spec, skill_name, round_id, failure_policy_applied or "runtime_failure")
        elif degraded and failure_policy_applied == "trip_circuit_breaker":
            breaker_state = self._breaker_for(skill_name)
        elif degraded and policy_rejection_reason is not None:
            breaker_state = self._breaker_for(skill_name)
        else:
            breaker_state = self._record_success(skill_name)

        normalized_output = to_dict(raw_output)
        result = SkillResult(
            skill_name=skill_name,
            owner_module=spec.owner_module,
            output=normalized_output if isinstance(normalized_output, dict) else {"value": normalized_output},
            latency_ms=latency_ms,
            cost_class=spec.cost_class,
            input_hash=input_hash,
            output_hash=_hash_payload(normalized_output),
            degraded=degraded,
            failure_policy_applied=failure_policy_applied,
            seed_ref=seed_ref,
            fallback_route=fallback_route,
            fallback_cost_class=fallback_cost_class,
            policy_rejection_reason=policy_rejection_reason,
            breaker_state={
                "failure_count": breaker_state.failure_count,
                "open_until_round": breaker_state.open_until_round,
                "open": self._breaker_is_open(spec, breaker_state, round_id),
                "last_failure_reason": breaker_state.last_failure_reason,
            },
        )
        return raw_output, result
