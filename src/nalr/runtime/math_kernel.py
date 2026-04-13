from __future__ import annotations

import math
from typing import Any

try:
    import numpy as _np
except ImportError:  # pragma: no cover - optional acceleration path
    _np = None


def normalize_distribution(distribution: dict[str, float]) -> dict[str, float]:
    if not distribution:
        return {}
    if _np is None:
        total = sum(max(float(value), 0.0) for value in distribution.values()) or 1.0
        return {action: max(float(value), 0.0) / total for action, value in distribution.items()}

    actions = list(distribution)
    values = _np.array([max(float(distribution[action]), 0.0) for action in actions], dtype=float)
    total = float(values.sum()) or 1.0
    normalized = values / total
    return {action: float(normalized[index]) for index, action in enumerate(actions)}


def softmax_distribution(utilities: dict[str, float]) -> dict[str, float]:
    if not utilities:
        return {}
    if _np is None:
        max_utility = max(float(value) for value in utilities.values())
        weights = {action: math.exp(float(value) - max_utility) for action, value in utilities.items()}
        return normalize_distribution(weights)

    actions = list(utilities)
    values = _np.array([float(utilities[action]) for action in actions], dtype=float)
    anchor = float(values.max(initial=0.0))
    weights = _np.exp(values - anchor)
    total = float(weights.sum()) or 1.0
    posterior = weights / total
    return {action: float(posterior[index]) for index, action in enumerate(actions)}


def kl_divergence(q_dist: dict[str, float], p_dist: dict[str, float]) -> float:
    if not q_dist:
        return 0.0
    if _np is None:
        kl = 0.0
        for action, q_value in q_dist.items():
            p_value = max(float(p_dist.get(action, 1e-9) or 0.0), 1e-9)
            q_value = max(float(q_value or 0.0), 1e-9)
            kl += q_value * math.log(q_value / p_value)
        return kl

    actions = list(q_dist)
    q_values = _np.array([max(float(q_dist[action]), 1e-9) for action in actions], dtype=float)
    p_values = _np.array([max(float(p_dist.get(action, 1e-9) or 0.0), 1e-9) for action in actions], dtype=float)
    return float(_np.sum(q_values * _np.log(q_values / p_values)))


def sample_action_name(distribution: dict[str, float], sample_value: float = 0.5) -> str:
    if not distribution:
        return ""
    threshold = max(0.0, min(1.0, float(sample_value)))
    cumulative = 0.0
    sampled_name = max(distribution, key=distribution.get)
    for action_name, probability in sorted(distribution.items()):
        cumulative += max(float(probability), 0.0)
        if threshold <= cumulative:
            sampled_name = action_name
            break
    return sampled_name


def numpy_enabled() -> bool:
    return _np is not None


def numpy_backend() -> str:
    return "numpy" if _np is not None else "python"


def vector_debug_payload() -> dict[str, Any]:
    return {
        "backend": numpy_backend(),
        "numpy_enabled": numpy_enabled(),
    }
