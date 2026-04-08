from pathlib import Path

from nalr.runtime.controller import RuntimeController
from nalr.runtime.math_kernel import (
    kl_divergence,
    normalize_distribution,
    sample_action_name,
    softmax_distribution,
)


CONFIG_ROOT = Path(__file__).resolve().parents[2] / "config"


def test_math_kernel_matches_controller_scalar_helpers(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    distribution = {"plan": 0.4, "respond": 0.2, "rest": -0.1}
    utilities = {"plan": 1.2, "respond": 0.5, "rest": -0.8}
    q_dist = {"plan": 0.7, "respond": 0.3}
    p_dist = {"plan": 0.5, "respond": 0.5}

    assert normalize_distribution(distribution) == controller._normalize(distribution)
    assert softmax_distribution(utilities) == controller._softmax(utilities)
    assert kl_divergence(q_dist, p_dist) == controller._kl_divergence(q_dist, p_dist)
    assert sample_action_name({"plan": 0.7, "respond": 0.3}, 0.4) == controller._sample_action_from_distribution({"plan": 0.7, "respond": 0.3}, 0.4).name
