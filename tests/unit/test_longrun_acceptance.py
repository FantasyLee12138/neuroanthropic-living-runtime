from pathlib import Path

import pytest

from nalr.runtime.controller import RuntimeController
from nalr.runtime.longrun import LongRunAnalyzer
from nalr.schemas.models import RoundEvent


CONFIG_ROOT = Path(__file__).resolve().parents[2] / "config"


def test_acceptance_report_and_eval_longrun_surface_online_longrun_prior(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    controller.tick(
        RoundEvent(
            source="user",
            content="Please help me plan steadily and keep continuity.",
            target="user",
            cue="tea",
            valence=0.03,
        ),
        scenario="task",
        mode="interactive",
    )

    report = controller.acceptance_report(window=1)
    summary = controller.eval_longrun(rounds=3)

    assert report["long_run_prior"]["observed_round_rate"] == 1.0
    assert report["long_run_prior"]["online_projection_coverage"] == 1.0
    assert report["long_run_prior"]["self_consistency_score_avg"] >= 0.0
    assert report["long_run_prior"]["samples"]

    assert "long_run_prior" in summary
    assert summary["long_run_prior"]["observed_round_rate"] == 1.0
    assert summary["long_run_prior"]["online_projection_coverage"] == 1.0


def test_longrun_acceptance_tracks_conflict_subordination(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    original_execute_skill = controller._execute_skill

    def fake_execute_skill(*, round_id, skill_name, inputs, provider, skill_traces, runtime_context, fallback_provider=None, fallback_value=None, seed_ref=None):
        if skill_name == "score_conflict":
            return {
                "score": 0.82,
                "total_score": 0.82,
                "components": {"task_goal": 0.82},
                "priority_signals": {"task_goal": 0.82},
                "critical_conflict": False,
            }
        if skill_name == "trigger_control_escalation":
            return {
                "flag": True,
                "winning_priority": "task_goal",
                "blocked_actions": ["plan"],
                "action_scales": {"plan": 0.2},
                "applied_template": None,
                "reason": "task_goal:0.82",
            }
        if skill_name == "request_resample":
            return {
                "flag": False,
                "allowed_resamples": 0,
                "force_compromise": False,
                "critical_conflict": False,
            }
        return original_execute_skill(
            round_id=round_id,
            skill_name=skill_name,
            inputs=inputs,
            provider=provider,
            skill_traces=skill_traces,
            runtime_context=runtime_context,
            fallback_provider=fallback_provider,
            fallback_value=fallback_value,
            seed_ref=seed_ref,
        )

    monkeypatch.setattr(controller, "_execute_skill", fake_execute_skill)
    controller.long_run_analyzer.metrics_summary = lambda: {
        "self_consistency_score": 0.94,
        "safe_mode_rounds": 0,
    }

    controller.tick(
        RoundEvent(
            source="user",
            content="Please help me plan carefully and keep continuity.",
            target="user",
            cue="tea",
            valence=0.04,
        ),
        scenario="task",
        mode="interactive",
    )

    report = controller.acceptance_report(window=1)

    assert report["long_run_prior"]["blocked_action_round_count"] == 1
    assert report["long_run_prior"]["conflict_subordination_rate"] == pytest.approx(1.0, abs=1e-6)


def test_longrun_acceptance_tracks_authenticity_subordination(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    def fake_apply_sampling_penalties(distribution, **kwargs):
        adjusted = dict(distribution)
        adjusted["plan"] = 1e-9
        total = sum(adjusted.values()) or 1.0
        normalized = {action: round(value / total, 6) for action, value in adjusted.items()}
        return normalized, {"plan": 0.4}, 0.2

    monkeypatch.setattr(controller.authenticity_policy, "apply_sampling_penalties", fake_apply_sampling_penalties)
    controller.long_run_analyzer.metrics_summary = lambda: {
        "self_consistency_score": 0.94,
        "safe_mode_rounds": 0,
    }

    result = controller.tick(
        RoundEvent(
            source="user",
            content="Please help me plan carefully and explain who you are.",
            target="user",
            cue="tea",
            valence=0.04,
        ),
        scenario="task",
        mode="interactive",
    )

    report = controller.acceptance_report(window=1)

    assert result.trace.authenticity["candidate_penalties"]["plan"] == pytest.approx(0.4, abs=1e-6)
    assert report["long_run_prior"]["auth_penalty_round_count"] == 1
    assert report["long_run_prior"]["authenticity_subordination_rate"] == pytest.approx(1.0, abs=1e-6)


def test_longrun_acceptance_uses_field_conflict_mask_without_distribution_state_conflict(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    controller.tick(
        RoundEvent(
            source="user",
            content="Please keep continuity and avoid plan if conflict blocks it.",
            target="user",
            cue="tea",
            valence=0.04,
        ),
        scenario="task",
        mode="interactive",
    )

    trace_payload = controller.trace_round(1)
    trace_payload["conflict_arbitration"]["hard_masked_targets"] = ["plan"]
    trace_payload["conflict_arbitration"]["critical_conflict"] = True
    trace_payload["probability_field"]["action"]["winner_posterior"]["plan"] = 0.0

    monkeypatch.setattr(controller.trace_store, "list_rounds", lambda: [trace_payload])

    report = controller.acceptance_report(window=1)

    assert report["long_run_prior"]["blocked_action_round_count"] == 1
    assert report["long_run_prior"]["conflict_subordination_rate"] == pytest.approx(1.0, abs=1e-6)


def test_longrun_online_projection_clips_and_prior_thresholds():
    class DummyTraceStore:
        def list_rounds(self):
            return []

    analyzer = LongRunAnalyzer(DummyTraceStore(), lambda: {}, ("plan", "respond", "clarify", "wander", "rest", "recall"))
    analyzer.metrics_summary = lambda: {
        "self_consistency_score": 1.7,
    }

    projection = analyzer.build_online_projection(
        round_id=99,
        slow_variables={
            "affect_residue": 1.4,
            "relationship_drift": 0.3,
            "resource_scarcity": 1.2,
            "memory_activation": 1.0,
        },
        shaping_events=[
            {"non_interactive": True},
            {"non_interactive": True},
            {"non_interactive": True},
            {"non_interactive": True},
        ],
    )
    contribution = analyzer.build_long_run_prior_contribution(projection)

    assert projection["continuity_window"] == 8
    assert projection["self_consistency_score"] == 1.0
    assert projection["volatility_signal"] == 1.0
    assert contribution.raw_signal["rest"] > 0.0
    assert contribution.modulated_delta["rest"] > 0.0
    assert contribution.raw_signal["recall"] == pytest.approx(0.09, abs=1e-6)
    assert contribution.modulated_delta["recall"] == pytest.approx(0.09, abs=1e-6)

    low_projection = analyzer.build_online_projection(
        round_id=1,
        slow_variables={
            "affect_residue": 0.08,
            "relationship_drift": 0.04,
            "resource_scarcity": 0.03,
            "memory_activation": 0.0,
        },
        shaping_events=[],
    )
    low_contribution = analyzer.build_long_run_prior_contribution(low_projection)

    assert "rest" not in low_contribution.modulated_delta
    assert "recall" not in low_contribution.modulated_delta


def test_longrun_prior_producer_passes_canonical_kwargs(monkeypatch):
    import nalr.runtime.longrun as longrun_module

    captured: dict[str, object] = {}

    def spy_contribution(**kwargs):
        captured.update(kwargs)
        return kwargs

    monkeypatch.setattr(longrun_module, "ProbabilisticContribution", spy_contribution)

    class DummyTraceStore:
        def list_rounds(self):
            return []

    analyzer = LongRunAnalyzer(DummyTraceStore(), lambda: {}, ("plan", "respond", "clarify", "wander", "rest", "recall"))
    analyzer.build_long_run_prior_contribution(
        {
            "self_consistency_score": 0.88,
            "volatility_signal": 0.42,
            "continuity_window": 5,
            "non_interactive_shift": 2,
            "rename_reason": "stable_continuity",
        }
    )

    assert "delta_energy" not in captured
    assert captured["raw_signal"]
    assert captured["modulated_delta"] == captured["raw_signal"]
    assert captured["projection_reason"] == "long-run prior projected from longitudinal continuity summary"
