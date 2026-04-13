import json
from pathlib import Path

from typer.testing import CliRunner

from nalr.cli.app import app
from nalr.runtime.controller import RuntimeController
from nalr.schemas.models import RoundEvent, RuntimeState


CONFIG_ROOT = Path(__file__).resolve().parents[2] / "config"
RUNNER = CliRunner()


def _good_history_burden(*, continuity_nonce: str = "nonce-1", checkpoint_id: str = "ckpt-1") -> dict[str, object]:
    return {
        "repair_scars": [
            {
                "round_id": 1,
                "reason": "task_goal_overrode_wander",
                "winning_priority": "task_goal",
                "template": "hold_line",
                "repair_stage_after": "stabilized",
                "blocked_actions": ["wander"],
                "conflict_score": 0.82,
            }
        ],
        "suppressed_but_recoverable_patterns": [
            {
                "pattern": "doomscroll",
                "suppressed_by": "repair_ledger",
                "status": "suppressed_recoverable",
                "suppressed_at_round": 1,
                "strength": 0.64,
            }
        ],
        "unfinished_commitments": [
            {
                "kind": "task",
                "summary": "finish subjectivity gate",
                "status": "active",
                "priority": 0.9,
            }
        ],
        "meaning_debts": [
            {
                "summary": "prove recovery continuity after restart",
                "status": "open",
            }
        ],
        "conflict_residues": {
            "repair_mode": "stabilize",
            "safe_mode_owner": "task_goal",
            "critical_conflict_streak": 1,
            "conflict_hot_rounds": 1,
            "conflict_recovery_rounds": 0,
            "last_conflict_priority": "task_goal",
            "last_compromise_template": "hold_line",
            "repair_stage": "stabilized",
        },
        "continuity_residues": {
            "continuity_nonce": continuity_nonce,
            "self_continuity": 0.84,
            "meaning_strength": 0.74,
            "memory_fragments": 0.21,
            "last_checkpoint_id": checkpoint_id,
            "active_run_id": None,
            "run_status": "idle",
        },
        "history_inertia_score": 0.58,
    }


def _good_history_burden_delta() -> dict[str, object]:
    return {
        "repair_scars_added": 1,
        "suppressed_patterns_delta": 1,
        "unfinished_commitments_delta": 0,
        "meaning_debts_delta": 1,
        "continuity_shift": -0.04,
        "history_inertia_score": 0.58,
    }


def _good_memory_evidence() -> dict[str, object]:
    return {
        "event_log_ref": {"round_trace_ref": "round://1"},
        "cue": "主体性",
        "hit": False,
        "miss": True,
        "mode": "gist",
        "detail": False,
        "cue_rescue": True,
        "interference": 0.44,
        "contamination": {"detected": True, "source": "summary_bias", "kind": "gist_override"},
        "latency_cost": {
            "latency_ms": 46,
            "cost_tier": "episodic",
            "search_trace": {"tiers": ["gist", "episodic"]},
        },
        "behavioral_consequence": {
            "selected_action": "clarify",
            "recall_strength": 0.36,
            "initiative_top_intent": "repair",
            "initiative_memory_backing": {"gist": ["主体性"], "cue": ["1.6"]},
            "probability_peak": {"clarify": 0.41},
        },
    }


def _good_arbitration_record() -> dict[str, object]:
    return {
        "competing_proposals": [
            {
                "agent_name": "PFCAgent",
                "stage": "pfc",
                "top_action": "clarify",
                "selected": True,
                "gated_actions": [],
            },
            {
                "agent_name": "HippocampusAgent",
                "stage": "hippocampus",
                "top_action": "recall",
                "selected": False,
                "gated_actions": ["wander"],
            },
        ],
        "evidence_sources": ["PFCAgent", "HippocampusAgent", "ConflictMonitorAgent"],
        "priority_scores": {"PFCAgent": 0.77, "HippocampusAgent": 0.61},
        "budget_consumption": {
            "budget_remaining": 0.54,
            "resample_count": 1,
            "window_tool_actions": 1,
            "window_endogenous_rounds": 0,
        },
        "selected_winner": "clarify",
        "rejected_options": [
            {
                "agent_name": "HippocampusAgent",
                "top_action": "recall",
                "reason": "not_selected",
            }
        ],
        "hard_masks": ["wander"],
        "consequence_patch": {
            "repair_state": {"stage": "stabilized"},
            "suppressed_actions": [{"kind": "habit", "summary": "wander"}],
            "budget_residue": 0.54,
            "scheduled_task_residue": 1,
            "history_inertia_score": 0.58,
            "safe_mode_owner": "task_goal",
        },
    }


def _good_snapshot_continuity(*, continuity_nonce: str = "nonce-1", checkpoint_id: str = "ckpt-1") -> dict[str, object]:
    return {
        "continuity_nonce": continuity_nonce,
        "runtime_revision": 4,
        "last_checkpoint_id": checkpoint_id,
        "rewind_supported": True,
        "snapshot_dir": "/tmp/snapshots",
        "self_continuity": 0.84,
        "repair_stage": "stabilized",
        "history_burden_delta": _good_history_burden_delta(),
    }


def _good_state_truth() -> dict[str, object]:
    return {
        "event_log": {
            "round_trace_ref": "round://1",
            "round_trace_jsonl": "/tmp/rounds.jsonl",
            "command_trace_jsonl": "/tmp/commands.jsonl",
            "memory_events_jsonl": "/tmp/memory.jsonl",
            "append_only": True,
        },
        "authoritative_state": {
            "runtime_revision": 4,
            "session_id": "sess-1",
            "last_mutation_at": "2026-04-13T10:00:00Z",
            "runtime_state_path": "/tmp/runtime.parquet",
            "memory_state_root": "/tmp/memory",
            "scheduled_task_state_path": "/tmp/tasks.json",
        },
        "snapshot": {
            "last_checkpoint_id": "ckpt-1",
            "checkpoint_dir": "/tmp/checkpoints",
            "snapshot_dir": "/tmp/snapshots",
            "rewind_supported": True,
        },
        "projection": {
            "terminal_view": "derived",
            "observer_status": "derived",
            "workbench_read_model": "derived",
            "round_detail": "derived",
        },
        "consistency_checks": {
            "subject_projection_from_state": True,
            "meaning_projection_from_state": True,
            "agency_projection_from_state": True,
            "projection_only_truth_fields": [],
            "session_cache_is_projection_only": True,
        },
    }


def _good_long_run_prior() -> dict[str, object]:
    return {
        "observed_round_rate": 1.0,
        "online_projection_coverage": 1.0,
        "blocked_action_round_count": 1,
        "conflict_subordination_rate": 1.0,
        "auth_penalty_round_count": 1,
        "authenticity_subordination_rate": 1.0,
        "self_consistency_score_avg": 0.88,
        "volatility_signal_avg": 0.22,
        "continuity_window_avg": 4.0,
        "samples": [
            {
                "round_id": 1,
                "self_consistency_score": 0.88,
                "volatility_signal": 0.22,
                "continuity_window": 4,
                "blocked_actions": ["wander"],
                "candidate_penalties": {"wander": 0.1},
            }
        ],
    }


def _baseline_state(*, continuity_nonce: str = "nonce-1", checkpoint_id: str = "ckpt-1") -> RuntimeState:
    state = RuntimeState(
        session_id="sess-1",
        round_count=1,
        runtime_revision=4,
        last_mutation_at="2026-04-13T10:00:00Z",
        self_continuity=0.84,
        meaning_strength=0.74,
        memory_fragments=0.21,
        last_checkpoint_id=checkpoint_id,
    )
    state.subject_core.continuity_nonce = continuity_nonce
    state.history_burden = _good_history_burden(continuity_nonce=continuity_nonce, checkpoint_id=checkpoint_id)
    return state


def _baseline_round() -> dict[str, object]:
    return {
        "round_id": 1,
        "sampled_action": "clarify",
        "mode": "interactive",
        "probability_field": {},
        "event_log_ref": {"round_trace_ref": "round://1"},
        "memory_evidence": _good_memory_evidence(),
        "arbitration_record": _good_arbitration_record(),
        "history_burden_delta": _good_history_burden_delta(),
        "snapshot_continuity": _good_snapshot_continuity(),
    }


def _configure_subjectivity_baseline(monkeypatch, controller, *, rounds=None, state=None, state_truth=None, long_run_prior=None):
    monkeypatch.setattr(controller.trace_store, "list_rounds", lambda: list(rounds or [_baseline_round()]))
    monkeypatch.setattr(controller, "load_runtime_state", lambda: state or _baseline_state())
    monkeypatch.setattr(controller, "_sync_plan16_state", lambda current_state: None)
    monkeypatch.setattr(controller, "state_truth_payload", lambda current_state, trace=None: state_truth or _good_state_truth())
    monkeypatch.setattr(controller, "_long_run_prior_summary", lambda observed_rounds: long_run_prior or _good_long_run_prior())


def test_acceptance_report_surfaces_15_controlled_learning_summary(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    payload = controller.acceptance_report(window=1)
    summary = payload["release_15"]["controlled_learning"]

    assert summary["learning_mode"] == "guided-learn"
    assert summary["trace_external_learning"] is True
    assert "docs.python.org" in summary["allowed_network_domains"]
    assert summary["learning_log_dir"] == str(tmp_path / ".alive" / "learning")
    assert str(tmp_path) in summary["writable_roots"]


def test_acceptance_report_uses_current_observer_learning_preferences(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    controller.update_observer_settings(
        {
            "autonomy": {
                "learning_mode": "active-learn",
                "allowed_network_domains": ["example.com"],
                "writable_roots": [str(tmp_path / "workspace")],
                "knowledge_roots": [str(tmp_path / "docs")],
                "learning_log_dir": str(tmp_path / ".alive" / "learning-cache"),
                "trace_external_learning": False,
            }
        }
    )

    payload = controller.acceptance_report(window=1)
    summary = payload["release_15"]["controlled_learning"]

    assert summary["learning_mode"] == "active-learn"
    assert summary["allowed_network_domains"] == ["example.com"]
    assert summary["writable_roots"] == [str(tmp_path / "workspace")]
    assert summary["knowledge_roots"] == [str(tmp_path / "docs")]
    assert summary["learning_log_dir"] == str(tmp_path / ".alive" / "learning-cache")
    assert summary["trace_external_learning"] is False


def test_acceptance_report_exposes_bounded_long_run_release_surface(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    payload = controller.acceptance_report(window=3)
    bounded = payload["release_15"]["bounded_long_run"]

    assert bounded["window"] == 3
    assert bounded["rounds_considered"] == payload["rounds_considered"]
    assert bounded["long_run_prior"] == payload["long_run_prior"]
    assert bounded["acceptance_report_window"] == payload["window"]


def test_eval_acceptance_report_command_surfaces_release_15_controlled_learning_summary(tmp_path, monkeypatch):
    monkeypatch.setenv("NALR_HOME", str(tmp_path / ".alive"))
    monkeypatch.setenv("NALR_CONFIG_DIR", str(CONFIG_ROOT))

    result = RUNNER.invoke(app, ["eval", "acceptance-report", "--window", "1"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    summary = payload["release_15"]["controlled_learning"]
    bounded = payload["release_15"]["bounded_long_run"]

    assert summary["learning_mode"] == "guided-learn"
    assert summary["trace_external_learning"] is True
    assert "docs.python.org" in summary["allowed_network_domains"]
    assert bounded["window"] == 1
    assert bounded["long_run_prior"] == payload["long_run_prior"]


def test_acceptance_report_surfaces_release_16_subjectivity_gate(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    controller.tick(
        RoundEvent(
            source="user",
            content="请证明记忆、竞争和连续性都会留下真实后果。",
            target="user",
            cue="主体性",
        ),
        scenario="chat",
        mode="interactive",
    )

    payload = controller.acceptance_report(window=1)
    release_16 = payload["release_16"]

    assert release_16["release_frontdoor"]["status"] in {"pass", "partial"}
    assert release_16["release_subjectivity"]["status"] in {"pass", "partial"}
    assert "single_truth" in release_16["subjectivity"]
    assert "memory" in release_16["subjectivity"]
    assert "competition" in release_16["subjectivity"]
    assert "history" in release_16["subjectivity"]
    assert "recovery" in release_16["subjectivity"]
    assert "drift" in release_16["subjectivity"]


def test_acceptance_report_passes_release_16_subjectivity_with_complete_evidence(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    _configure_subjectivity_baseline(monkeypatch, controller)

    payload = controller.acceptance_report(window=1)
    release_16 = payload["release_16"]

    assert release_16["release_subjectivity"]["status"] == "pass"
    assert release_16["release_subjectivity"]["blocking_items"] == []
    assert all(section["status"] == "pass" for section in release_16["subjectivity"].values())


def test_acceptance_report_counts_zero_ms_latency_with_explicit_cost_tier_as_memory_evidence(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    good_round = _baseline_round()
    good_round["memory_evidence"] = {
        **_good_memory_evidence(),
        "latency_cost": {
            "latency_ms": 0,
            "cost_tier": "hot",
            "search_trace": {"tiers": ["hot"]},
        },
    }
    _configure_subjectivity_baseline(monkeypatch, controller, rounds=[good_round])

    payload = controller.acceptance_report(window=1)

    assert payload["release_16"]["subjectivity"]["memory"]["status"] == "pass"
    assert "missing_latency_cost" not in payload["release_16"]["subjectivity"]["memory"]["missing_evidence"]


def test_acceptance_report_fails_single_truth_when_projection_grows_shadow_truth(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    state_truth = _good_state_truth()
    state_truth["consistency_checks"] = {
        **state_truth["consistency_checks"],
        "agency_projection_from_state": False,
        "projection_only_truth_fields": ["web_session.summary"],
        "session_cache_is_projection_only": False,
    }
    _configure_subjectivity_baseline(monkeypatch, controller, state_truth=state_truth)

    payload = controller.acceptance_report(window=1)

    assert payload["release_16"]["subjectivity"]["single_truth"]["status"] == "fail"
    assert "single_truth" in payload["release_16"]["release_subjectivity"]["blocking_items"]
    assert payload["release_16"]["release_subjectivity"]["status"] == "fail"


def test_acceptance_report_fails_memory_when_recall_looks_like_perfect_retrieval(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    bad_round = _baseline_round()
    bad_round["memory_evidence"] = {
        "event_log_ref": {"round_trace_ref": "round://1"},
        "cue": "主体性",
        "hit": True,
        "miss": False,
        "mode": "detail",
        "detail": True,
        "cue_rescue": False,
        "interference": 0.0,
        "contamination": {"detected": False},
        "latency_cost": {"latency_ms": 0, "cost_tier": "", "search_trace": {}},
        "behavioral_consequence": {},
    }
    _configure_subjectivity_baseline(monkeypatch, controller, rounds=[bad_round])

    payload = controller.acceptance_report(window=1)

    assert payload["release_16"]["subjectivity"]["memory"]["status"] == "fail"


def test_acceptance_report_fails_competition_when_no_consequence_is_written_back(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    bad_round = _baseline_round()
    bad_round["arbitration_record"] = {
        "competing_proposals": [
            {"agent_name": "PFCAgent", "stage": "pfc", "top_action": "clarify", "selected": True, "gated_actions": []},
            {"agent_name": "HippocampusAgent", "stage": "hippocampus", "top_action": "recall", "selected": False, "gated_actions": []},
        ],
        "evidence_sources": ["PFCAgent", "HippocampusAgent"],
        "priority_scores": {"PFCAgent": 0.7, "HippocampusAgent": 0.6},
        "budget_consumption": {"budget_remaining": 0.5},
        "selected_winner": "clarify",
        "rejected_options": [{"agent_name": "HippocampusAgent", "top_action": "recall", "reason": "not_selected"}],
        "hard_masks": [],
        "consequence_patch": {},
    }
    _configure_subjectivity_baseline(monkeypatch, controller, rounds=[bad_round])

    payload = controller.acceptance_report(window=1)

    assert payload["release_16"]["subjectivity"]["competition"]["status"] == "fail"


def test_acceptance_report_fails_history_when_no_burden_or_inertia_is_left(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    empty_state = _baseline_state()
    empty_state.history_burden = {
        "repair_scars": [],
        "suppressed_but_recoverable_patterns": [],
        "unfinished_commitments": [],
        "meaning_debts": [],
        "conflict_residues": {
            "repair_mode": "",
            "safe_mode_owner": "",
            "critical_conflict_streak": 0,
            "conflict_hot_rounds": 0,
            "conflict_recovery_rounds": 0,
            "last_conflict_priority": "",
            "last_compromise_template": "",
            "repair_stage": "",
        },
        "continuity_residues": {
            "continuity_nonce": "",
            "self_continuity": 0.0,
            "meaning_strength": 0.0,
            "memory_fragments": 0.0,
            "last_checkpoint_id": None,
            "active_run_id": None,
            "run_status": "idle",
        },
        "history_inertia_score": 0.0,
    }
    bad_round = _baseline_round()
    bad_round["history_burden_delta"] = {
        "repair_scars_added": 0,
        "suppressed_patterns_delta": 0,
        "unfinished_commitments_delta": 0,
        "meaning_debts_delta": 0,
        "continuity_shift": 0.0,
        "history_inertia_score": 0.0,
    }
    _configure_subjectivity_baseline(monkeypatch, controller, rounds=[bad_round], state=empty_state)

    payload = controller.acceptance_report(window=1)

    assert payload["release_16"]["subjectivity"]["history"]["status"] == "fail"


def test_acceptance_report_fails_recovery_when_snapshot_continuity_breaks_nonce_chain(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    bad_round = _baseline_round()
    bad_round["snapshot_continuity"] = {
        **_good_snapshot_continuity(),
        "continuity_nonce": "foreign-nonce",
    }
    _configure_subjectivity_baseline(monkeypatch, controller, rounds=[bad_round])

    payload = controller.acceptance_report(window=1)

    assert payload["release_16"]["subjectivity"]["recovery"]["status"] == "fail"


def test_acceptance_report_fails_drift_when_long_run_stability_collapses(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    degraded_state = _baseline_state()
    degraded_state.self_continuity = 0.22
    degraded_state.history_burden["history_inertia_score"] = 0.02
    degraded_prior = {
        "observed_round_rate": 1.0,
        "online_projection_coverage": 0.12,
        "blocked_action_round_count": 1,
        "conflict_subordination_rate": 0.0,
        "auth_penalty_round_count": 1,
        "authenticity_subordination_rate": 0.0,
        "self_consistency_score_avg": 0.21,
        "volatility_signal_avg": 0.93,
        "continuity_window_avg": 1.0,
        "samples": [{"round_id": 1, "self_consistency_score": 0.21, "volatility_signal": 0.93, "continuity_window": 1}],
    }
    _configure_subjectivity_baseline(monkeypatch, controller, state=degraded_state, long_run_prior=degraded_prior)

    payload = controller.acceptance_report(window=1)

    assert payload["release_16"]["subjectivity"]["drift"]["status"] == "fail"
