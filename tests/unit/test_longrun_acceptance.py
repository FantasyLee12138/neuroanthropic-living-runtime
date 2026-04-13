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


def test_longrun_online_projection_prefers_latest_round_projection_over_full_metrics_scan():
    class DummyTraceStore:
        def list_rounds(self):
            return [
                {
                    "long_run_projection": {"self_consistency_score": 0.73},
                    "authenticity": {"self_grounding_score": 0.11},
                }
            ]

    analyzer = LongRunAnalyzer(DummyTraceStore(), lambda: {}, ("plan", "respond", "clarify", "wander", "rest", "recall"))
    analyzer.metrics_summary = lambda: (_ for _ in ()).throw(AssertionError("metrics_summary should not be called when latest round exists"))

    projection = analyzer.build_online_projection(
        round_id=2,
        slow_variables={"affect_residue": 0.02, "relationship_drift": 0.01, "resource_scarcity": 0.0, "memory_activation": 0.0},
        shaping_events=[],
    )

    assert projection["self_consistency_score"] == 0.73


def test_longrun_projection_surfaces_anchor_summary_after_real_rounds(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    state = controller.load_runtime_state()
    state.body_energy = 0.14
    state.fatigue = 0.82
    state.memory_fragments = 0.74
    state.self_continuity = 0.38
    state.meaning_strength = 0.41
    state.subjective_state.spontaneous = 0.69
    state.subjective_state.reject_all = 0.44
    state.subjective_state.meaning_made = ["keep continuity through inward integration"]
    controller._save_state(state, sync=True)

    for cue in ("pause", "integrate", "continue"):
        controller.tick(
            RoundEvent(
                source="user",
                content=f"Please {cue} and keep continuity stable.",
                target="user",
                cue=cue,
                valence=0.04,
            ),
            scenario="task",
            mode="interactive",
        )

    trace = controller.trace_round("latest")
    projection = trace["long_run_projection"]

    assert "personality_anchor_summary" in projection
    assert projection["personality_anchor_summary"]["axis_baseline"]
    assert projection["personality_anchor_summary"]["driver_signature"]
    assert projection["personality_anchor_summary"]["anchor_alignment"] >= 0.0


def test_longrun_projection_uses_recent_round_window_without_full_history_scan():
    class DummyTraceStore:
        def recent_rounds(self, *, limit: int = 20):
            assert limit == 12
            return [
                {
                    "top_drivers": [
                        {"action_name": "plan", "score": 0.6},
                        {"action_name": "respond", "score": 0.2},
                    ]
                },
                {
                    "top_drivers": [
                        {"action_name": "respond", "score": 0.7},
                        {"action_name": "rest", "score": 0.1},
                    ]
                },
            ]

        def list_rounds(self):
            raise AssertionError("build_round_projection should not scan full history")

    analyzer = LongRunAnalyzer(DummyTraceStore(), lambda: {}, ("plan", "respond", "clarify", "wander", "rest", "recall"))

    projection = analyzer.build_round_projection(
        round_id=7,
        vitality_snapshot={"affect_residue": 0.08, "relationship_drift": 0.02, "resource_scarcity": 0.01},
        authenticity={"self_grounding_score": 0.82},
        identity_evolution={},
        shaping_events=[],
        personality_anchor={"axis_baseline": {"E": 0.5, "F": 0.5, "S": 0.5, "M": 0.5}, "alignment": 0.9, "stability": 0.8},
        top_drivers=[{"action_name": "plan", "score": 0.9}],
    )

    summary = projection["personality_anchor_summary"]
    assert summary["dominant_actions"][0] == "plan"
    assert "respond" in summary["dominant_actions"]


def test_personality_anchor_accumulates_recent_driver_bias(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    state = controller.load_runtime_state()
    state.body_energy = 0.12
    state.fatigue = 0.86
    state.memory_fragments = 0.8
    state.self_continuity = 0.34
    state.meaning_strength = 0.35
    state.subjective_state.spontaneous = 0.76
    state.subjective_state.reject_all = 0.63
    state.subjective_state.meaning_made = ["absorb first, answer later"]
    controller._save_state(state, sync=True)

    for cue in ("quiet", "pause", "absorb", "settle"):
        controller.tick(
            RoundEvent(
                source="user",
                content=f"Stay with the {cue} pull and absorb before replying.",
                target="user",
                cue=cue,
            ),
            scenario="chat",
            mode="interactive",
        )

    updated = controller.load_runtime_state()

    assert updated.personality_anchor.action_bias
    assert any(action in updated.personality_anchor.action_bias for action in ("absorb", "rest", "nothing"))
    assert updated.personality_anchor.anchor_signature
