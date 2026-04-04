from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path

from nalr.dream.orchestrator import DreamOrchestrator
from nalr.memory.store import MemoryStore
from nalr.runtime.controller import RuntimeController
from nalr.runtime.vitality import VitalityEngine
from nalr.schemas import models
from nalr.schemas.models import RoundEvent
from nalr.trace import store as trace_store


CONFIG_ROOT = Path(__file__).resolve().parents[2] / "config"


def test_probability_field_schema_exports_required_types():
    required_types = [
        "ProbabilisticContribution",
        "ProbabilityLayerState",
        "ProbabilityFieldSnapshot",
        "TokenContributionTrace",
        "PeakArbitrationRecord",
    ]

    missing = [name for name in required_types if not hasattr(models, name)]

    assert missing == [], f"missing probability-field schema types: {missing}"


def test_probability_field_integrator_module_is_available():
    spec = importlib.util.find_spec("nalr.runtime.probability_field")

    assert spec is not None, "expected nalr.runtime.probability_field to exist"

    module = importlib.import_module("nalr.runtime.probability_field")
    assert hasattr(module, "ProbabilityFieldIntegrator"), "expected ProbabilityFieldIntegrator export"


def test_probability_field_projection_helper_projects_proposal_bundle_semantics():
    bundle = models.ProposalBundle(
        owner="PFCAgent",
        confidence=0.84,
        action_preferences={"plan": 0.55},
        delta_p={"plan": 0.55},
        trace_tags=["pfc", "proposal"],
        reason="goal prior reweight",
    )

    contribution = models.ProbabilisticContribution.from_proposal_bundle(
        bundle,
        module_name="PFCAgent",
        layer="action",
        level="action",
        trace_reason="goal prior reweight",
    )

    assert contribution.module_name == "PFCAgent"
    assert contribution.owner == "PFCAgent"
    assert contribution.delta_logits["plan"] == 0.55
    assert contribution.action_preferences["plan"] == 0.55
    assert contribution.trace_reason == "goal prior reweight"


def test_round_trace_schema_includes_probability_field_columns():
    required_columns = {
        "probability_field_json",
        "context_attn_json",
        "memory_prior_json",
        "action_logits_json",
        "token_logits_json",
        "winner_posterior_json",
        "counterfactual_top_peaks_json",
    }

    missing = sorted(required_columns - set(trace_store.ROUND_TRACE_SCHEMA))

    assert missing == [], f"missing probability-field trace columns: {missing}"


def test_tick_populates_probability_field_snapshot(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    result = controller.tick(
        RoundEvent(
            source="user",
            content="Please plan dinner and remember rice.",
            target="user",
            cue="rice",
            valence=0.15,
            energy_delta=-0.05,
        ),
        scenario="task",
        mode="interactive",
    )

    probability_field = result.trace.distribution_state.get("probability_field", {})

    assert probability_field, "expected a probability_field snapshot in trace.distribution_state"
    assert "context_attn_final" in probability_field
    assert "memory_prior_final" in probability_field
    assert "action_logits_final" in probability_field
    assert "token_logits_final" in probability_field
    assert "winner_posterior" in probability_field
    assert "counterfactual_top_peaks" in probability_field


def test_dream_orchestrator_run_returns_prior_recalibration_payload(tmp_path):
    class _DummyRepairState:
        stage = "idle"

    class _DummyState:
        session_id = "sess-v057"
        round_count = 3
        budget_remaining = 0.55
        body_energy = 0.72
        focus = "rest"
        resource_state = {"scarcity_index": 0.23}
        mood = 0.61
        affect_residue = 0.14
        repair_state = _DummyRepairState()
        conflict_hot_rounds = 0
        session_metadata = {}

    memory_store = MemoryStore(tmp_path)
    vitality_engine = VitalityEngine()
    orchestrator = DreamOrchestrator(
        project_root=tmp_path,
        home_path=tmp_path,
        config={
            "enabled": True,
            "rpc_timeout_seconds": 0.01,
            "allowed_types": {
                "sleep_full": ["memory_consolidation", "habit_adjustments", "limited_identity_drift"],
                "idle_light": ["memory_consolidation"],
            },
        },
        memory_store=memory_store,
        vitality_engine=vitality_engine,
    )

    payload = orchestrator.run(
        state=_DummyState(),
        mode="sleep",
        cue="tea",
        relation_state={"closeness": 0.62, "boundary_level": 0.14},
    )

    assert "prior_updates" in payload.get("effect_summary", {}), "expected dream prior recalibration details"
    assert "identity_prior" in payload.get("effect_summary", {})
    assert "habit_prior" in payload.get("effect_summary", {})
