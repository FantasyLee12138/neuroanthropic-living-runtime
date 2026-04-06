from __future__ import annotations

from dataclasses import MISSING, fields, is_dataclass
from types import UnionType
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

from nalr.schemas.models import (
    ActionCandidate,
    ActionEvidenceSignal,
    CrossLayerCouplingSpec,
    DisclosureIntentState,
    EnergyProjectionSpec,
    ExpressionProfile,
    IntentPosterior,
    IntentTraceRecord,
    ProbabilityFieldSnapshot,
    ProbabilityLayerState,
    ProbabilisticContribution,
    QuantumEntropyBatch,
    QuantumEntropyRef,
    QueryIntentState,
    RenderPlan,
    RenderedExpression,
    RoundEvent,
    RoundTrace,
    RuntimeState,
    TokenFieldState,
    to_dict,
)


TYPE_REF_REGISTRY: dict[str, Any] = {
    "Any": Any,
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "dict": dict[str, Any],
    "list": list[Any],
    "RoundEvent": RoundEvent,
    "RuntimeState": RuntimeState,
    "ProbabilisticContribution": ProbabilisticContribution,
    "ActionCandidate": ActionCandidate,
    "ActionEvidenceSignal": ActionEvidenceSignal,
    "ProbabilityLayerState": ProbabilityLayerState,
    "ProbabilityFieldSnapshot": ProbabilityFieldSnapshot,
    "EnergyProjectionSpec": EnergyProjectionSpec,
    "CrossLayerCouplingSpec": CrossLayerCouplingSpec,
    "TokenFieldState": TokenFieldState,
    "QueryIntentState": QueryIntentState,
    "DisclosureIntentState": DisclosureIntentState,
    "IntentPosterior": IntentPosterior,
    "IntentTraceRecord": IntentTraceRecord,
    "QuantumEntropyRef": QuantumEntropyRef,
    "QuantumEntropyBatch": QuantumEntropyBatch,
    "RoundTrace": RoundTrace,
    "RenderPlan": RenderPlan,
    "RenderedExpression": RenderedExpression,
    "ExpressionProfile": ExpressionProfile,
}


UNION_ORIGINS = {Union, UnionType}
def _split_generic_args(expr: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    start = 0
    for idx, char in enumerate(expr):
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(expr[start:idx].strip())
            start = idx + 1
    parts.append(expr[start:].strip())
    return [part for part in parts if part]


def resolve_contract(contract: Any) -> Any:
    if isinstance(contract, dict):
        return {key: resolve_contract(value) for key, value in contract.items()}
    if not isinstance(contract, str):
        return contract

    expr = contract.strip()
    if expr in TYPE_REF_REGISTRY:
        return TYPE_REF_REGISTRY[expr]
    if expr.startswith("list[") and expr.endswith("]"):
        inner = resolve_contract(expr[5:-1])
        return list[inner]
    if expr.startswith("dict[") and expr.endswith("]"):
        key_expr, value_expr = _split_generic_args(expr[5:-1])
        return dict[resolve_contract(key_expr), resolve_contract(value_expr)]
    if expr.startswith("optional[") and expr.endswith("]"):
        return resolve_contract(expr[9:-1]) | None
    raise ValueError(f"unknown contract expression: {contract}")


def serialize_contract(contract: Any) -> Any:
    if isinstance(contract, dict):
        return {key: serialize_contract(value) for key, value in contract.items()}
    if contract is Any:
        return "Any"

    origin = get_origin(contract)
    if origin in UNION_ORIGINS:
        args = get_args(contract)
        non_none = [arg for arg in args if arg is not type(None)]
        if len(non_none) == 1 and len(non_none) != len(args):
            return f"optional[{serialize_contract(non_none[0])}]"
        return " | ".join(serialize_contract(arg) for arg in args)
    if origin is list:
        return f"list[{serialize_contract(get_args(contract)[0])}]"
    if origin is dict:
        key_type, value_type = get_args(contract)
        return f"dict[{serialize_contract(key_type)}, {serialize_contract(value_type)}]"
    if isinstance(contract, type):
        return contract.__name__
    return str(contract)


def serialize_contract_value(value: Any) -> Any:
    if isinstance(value, ProbabilisticContribution):
        return to_dict(value)
    if isinstance(value, dict):
        return {key: serialize_contract_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [serialize_contract_value(item) for item in value]
    if isinstance(value, tuple):
        return [serialize_contract_value(item) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return serialize_contract_value(to_dict(value))
    return value


def _coerce_primitive(value: Any, contract: type, path: str) -> Any:
    if contract is Any:
        return value
    if contract is bool:
        if isinstance(value, bool):
            return value
        raise ValueError(f"{path} expected bool")
    if contract is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{path} expected int")
        return value
    if contract is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{path} expected float")
        return float(value)
    if contract is str:
        if not isinstance(value, str):
            raise ValueError(f"{path} expected str")
        return value
    raise ValueError(f"{path} unsupported primitive contract {contract}")


def coerce_contract(value: Any, contract: Any, *, path: str = "value") -> Any:
    if contract is Any:
        return value
    if isinstance(contract, dict):
        if not isinstance(value, dict):
            raise ValueError(f"{path} expected object")
        missing = [key for key in contract if key not in value]
        if missing:
            raise ValueError(f"{path} missing keys: {', '.join(missing)}")
        extras = [key for key in value if key not in contract]
        if extras:
            raise ValueError(f"{path} unexpected keys: {', '.join(extras)}")
        return {key: coerce_contract(value[key], field_contract, path=f"{path}.{key}") for key, field_contract in contract.items()}

    origin = get_origin(contract)
    if origin in UNION_ORIGINS:
        last_error: ValueError | None = None
        for option in get_args(contract):
            if option is type(None) and value is None:
                return None
            try:
                return coerce_contract(value, option, path=path)
            except ValueError as exc:
                last_error = exc
        raise last_error or ValueError(f"{path} did not match union contract")
    if origin is list:
        if not isinstance(value, list):
            raise ValueError(f"{path} expected list")
        inner = get_args(contract)[0]
        return [coerce_contract(item, inner, path=f"{path}[{idx}]") for idx, item in enumerate(value)]
    if origin is Literal:
        allowed = get_args(contract)
        if value not in allowed:
            joined = ", ".join(repr(item) for item in allowed)
            raise ValueError(f"{path} expected one of: {joined}")
        return value
    if origin is dict:
        if not isinstance(value, dict):
            raise ValueError(f"{path} expected dict")
        key_type, value_type = get_args(contract)
        coerced: dict[Any, Any] = {}
        for key, item in value.items():
            coerced_key = coerce_contract(key, key_type, path=f"{path}.<key>")
            coerced[coerced_key] = coerce_contract(item, value_type, path=f"{path}.{key}")
        return coerced

    if isinstance(contract, type) and contract in {str, int, float, bool}:
        return _coerce_primitive(value, contract, path)

    if isinstance(contract, type) and is_dataclass(contract):
        if isinstance(value, contract):
            return value
        if not isinstance(value, dict):
            raise ValueError(f"{path} expected {contract.__name__}")
        field_values: dict[str, Any] = {}
        known_fields = {field.name: field for field in fields(contract)}
        type_hints = get_type_hints(contract)
        extras = [key for key in value if key not in known_fields]
        if extras:
            raise ValueError(f"{path} unexpected keys: {', '.join(extras)}")
        for field in fields(contract):
            field_contract = type_hints.get(field.name, field.type)
            if field.name in value:
                field_values[field.name] = coerce_contract(value[field.name], field_contract, path=f"{path}.{field.name}")
            elif field.default is not MISSING or field.default_factory is not MISSING:
                continue
            else:
                raise ValueError(f"{path} missing field: {field.name}")
        return contract(**field_values)

    if contract is dict:
        if not isinstance(value, dict):
            raise ValueError(f"{path} expected dict")
        return value

    raise ValueError(f"{path} unsupported contract {contract!r}")
