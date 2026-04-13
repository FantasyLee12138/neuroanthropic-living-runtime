import json
from pathlib import Path

import nalr.runtime.controller as controller_module
from nalr.cil.runtime import build_endogenous_status_payload
from nalr.runtime.controller import RuntimeController
from nalr.runtime.initiative import InitiativeRuntime
from nalr.runtime.monologue import MonologueStreamRuntime
from nalr.schemas.models import EndogenousMicroIntent, EndogenousTickTrigger, RoundEvent
from nalr.terminal_bridge.handlers import TerminalEventHandler
from nalr.terminal_bridge.session import TerminalSessionState, TerminalSessionStore


CONFIG_ROOT = Path(__file__).resolve().parents[2] / "config"


def _seed_entropy(controller: RuntimeController) -> None:
    controller.entropy_pool.ingest_bytes(bytes([index % 256 for index in range(4096)]), source="test_qrng")


def _low_threshold_settings() -> dict[str, object]:
    return {
        "idle_seconds_threshold": 0,
        "vitality_threshold": 0.0,
        "relation_strength_threshold": 0.0,
        "habit_strength_threshold": 0.0,
        "proposal_posterior_threshold": 0.0,
        "hourly_limit": 0,
        "cooldown_seconds": 0,
        "require_memory_backing": False,
        "auto_send_enabled": True,
    }


def _active_session() -> dict[str, object]:
    return {"session_id": "sess-initiative", "status": "active", "approvals_pending": []}


def _initiative_inputs() -> dict[str, object]:
    return {
        "settings": _low_threshold_settings(),
        "idle_seconds": 3600,
        "vitality_snapshot": {"body_energy": 1.0, "affect_residue": 0.0, "vitality": 1.0},
        "relation_state": {"closeness": 1.0, "boundary_level": 0.0},
        "habit_strength": 1.0,
        "memory_backing": {"cue": "tea", "strength": 1.0, "summary": "tea"},
        "active_session": _active_session(),
        "pending_approval": False,
        "safe_mode": False,
        "recent_history": [],
        "current_goal": "继续推进",
    }


def test_build_endogenous_status_payload_uses_public_trace_storage_status(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    state = controller.load_runtime_state()
    state.round_count = 3
    state.endogenous_scheduler_state.last_endogenous_tick_at = "2026-04-13T00:00:00Z"
    state.endogenous_scheduler_state.recent_triggers = [EndogenousTickTrigger(trigger_type="idle")]
    state.endogenous_state.last_trigger = "idle"
    state.endogenous_state.current_intent = EndogenousMicroIntent(name="reflect", trigger="idle")
    state.endogenous_state.stability = 2
    monkeypatch.setattr(controller, "load_runtime_state", lambda: state)
    monkeypatch.setattr(
        controller,
        "state_payload",
        lambda: (_ for _ in ()).throw(AssertionError("endogenous status should not build the heavy state payload")),
    )
    monkeypatch.setattr(controller, "trace_storage_status", lambda: {"kind": "public"})

    def _private_should_not_be_used(*args, **kwargs):
        raise AssertionError("private trace storage helper leaked outside controller facade")

    monkeypatch.setattr(controller, "_trace_storage_payload", _private_should_not_be_used)

    payload = build_endogenous_status_payload(controller)

    assert payload["round_count"] == 3
    assert payload["last_endogenous_tick_at"] == "2026-04-13T00:00:00Z"
    assert payload["last_trigger"] == "idle"
    assert payload["latest_trigger"]["trigger_type"] == "idle"
    assert payload["current_intent"]["name"] == "reflect"
    assert payload["stability"] == 2
    assert payload["storage"] == {"kind": "public"}


def test_build_endogenous_status_payload_prefers_round_summaries_for_latest_endogenous_round(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    state = controller.load_runtime_state()
    state.round_count = 4
    monkeypatch.setattr(controller, "load_runtime_state", lambda: state)
    monkeypatch.setattr(
        controller,
        "state_payload",
        lambda: (_ for _ in ()).throw(AssertionError("endogenous status should not build the heavy state payload")),
    )
    monkeypatch.setattr(controller, "trace_storage_status", lambda: {"kind": "public"})
    monkeypatch.setattr(
        controller.trace_store,
        "list_rounds",
        lambda: (_ for _ in ()).throw(AssertionError("endogenous status should not full-scan round history when summaries exist")),
    )
    monkeypatch.setattr(
        controller.trace_store,
        "list_round_summaries",
        lambda: [
            {"round_id": 2, "cause_type": "external_stimulus"},
            {"round_id": 4, "cause_type": "endogenous"},
        ],
    )

    payload = build_endogenous_status_payload(controller)

    assert payload["latest_endogenous_round_id"] == 4


def test_initiative_prepare_treats_paused_dirty_worktree_run_as_non_blocking():
    runtime = InitiativeRuntime()
    prepared = runtime.prepare(
        **_initiative_inputs(),
        active_run={
            "run_id": "run-paused",
            "status": "paused",
            "dirty_worktree_detected": True,
            "stop_reason": {"code": "dirty_worktree", "message": "cannot resume while worktree is dirty"},
        },
    )

    assert prepared["active_run"]["status"] == "paused"
    assert prepared["active_run_blocked"] is False


def test_initiative_producer_prepare_evaluate_commit_blocks_running_run_and_keeps_commit_round_trip():
    runtime = InitiativeRuntime()
    prepared = runtime.prepare(
        **_initiative_inputs(),
        active_run={
            "run_id": "run-running",
            "status": "running",
            "dirty_worktree_detected": False,
            "stop_reason": {},
        },
    )

    evaluation = runtime.evaluate(prepared)
    committed = runtime.commit(prepared, evaluation)

    assert prepared["active_run_blocked"] is True
    assert evaluation["should_send"] is False
    assert evaluation["suppression_reason"] == "active_run_blocked"
    assert committed["proposal_id"] == evaluation["proposal_id"]
    assert committed["producer"] == "initiative"
    assert committed["producer_phase"] == "commit"


def test_initiative_does_not_treat_suppressed_history_as_ignored_feedback():
    runtime = InitiativeRuntime()
    prepared = runtime.prepare(**_initiative_inputs())
    prepared["settings"] = {
        "idle_seconds_threshold": 300,
        "vitality_threshold": 0.8,
        "relation_strength_threshold": 0.6,
        "habit_strength_threshold": 0.45,
        "proposal_posterior_threshold": 0.7,
        "hourly_limit": 3,
        "cooldown_seconds": 600,
        "require_memory_backing": True,
        "auto_send_enabled": True,
    }
    prepared["recent_history"] = [
        {"auto_sent": False, "feedback_recorded": False},
        {"auto_sent": False, "feedback_recorded": False},
        {"auto_sent": False, "feedback_recorded": False},
    ]

    evaluation = runtime.evaluate(prepared)

    assert evaluation["metrics"]["ignored_ratio"] == 0.0
    assert evaluation["metrics"]["feedback_penalty"] == 0.0


def test_initiative_default_threshold_allows_strong_grounded_proposal_to_send():
    runtime = InitiativeRuntime()
    evaluation = runtime.evaluate(
        settings={
            "idle_seconds_threshold": 300,
            "vitality_threshold": 0.8,
            "relation_strength_threshold": 0.6,
            "habit_strength_threshold": 0.45,
            "proposal_posterior_threshold": 0.7,
            "hourly_limit": 3,
            "cooldown_seconds": 600,
            "require_memory_backing": True,
            "auto_send_enabled": True,
        },
        idle_seconds=3600,
        vitality_snapshot={"body_energy": 1.0, "affect_residue": 0.0, "vitality": 1.0},
        relation_state={"closeness": 1.0, "boundary_level": 0.0},
        habit_strength=0.92,
        memory_backing={"cue": "一起散步", "strength": 1.0, "summary": "一起散步"},
        active_session=_active_session(),
        pending_approval=False,
        safe_mode=False,
        recent_history=[],
        current_goal=None,
    )

    assert evaluation["should_send"] is True
    assert evaluation["suppression_reason"] == ""
    assert evaluation["expression_mode"] == "external"


def test_initiative_subjective_spontaneity_changes_expressive_distribution():
    runtime = InitiativeRuntime()
    baseline = runtime.prepare(**_initiative_inputs())
    elevated = runtime.prepare(**_initiative_inputs())
    baseline["subjective_state"] = {"spontaneous": 0.0, "reject_all": 0.0, "meaning_made": []}
    elevated["subjective_state"] = {"spontaneous": 0.8, "reject_all": 0.0, "meaning_made": ["想说点什么"]}

    baseline_eval = runtime.evaluate(baseline)
    elevated_eval = runtime.evaluate(elevated)

    assert elevated_eval["posterior"]["express_state"] > baseline_eval["posterior"]["express_state"]


def test_initiative_temperament_social_capacity_changes_expressive_distribution():
    runtime = InitiativeRuntime()
    baseline_eval = runtime.evaluate(
        settings={
            "idle_seconds_threshold": 300,
            "vitality_threshold": 0.8,
            "relation_strength_threshold": 0.6,
            "habit_strength_threshold": 0.45,
            "proposal_posterior_threshold": 0.7,
            "hourly_limit": 3,
            "cooldown_seconds": 600,
            "require_memory_backing": True,
            "auto_send_enabled": True,
        },
        idle_seconds=420,
        vitality_snapshot={"body_energy": 0.82, "affect_residue": 0.02, "vitality": 0.82},
        relation_state={"closeness": 0.72, "boundary_level": 0.08},
        temperament_state={
            "current": {
                "attachment_need": 0.2,
                "boundary_softness": 0.2,
                "cognitive_bandwidth": 0.3,
                "extraversion": 0.2,
            }
        },
        habit_strength=0.58,
        memory_backing={"cue": "一起散步", "strength": 0.6, "summary": "一起散步"},
        active_session=_active_session(),
        pending_approval=False,
        safe_mode=False,
        recent_history=[],
        current_goal=None,
    )
    elevated_eval = runtime.evaluate(
        settings={
            "idle_seconds_threshold": 300,
            "vitality_threshold": 0.8,
            "relation_strength_threshold": 0.6,
            "habit_strength_threshold": 0.45,
            "proposal_posterior_threshold": 0.7,
            "hourly_limit": 3,
            "cooldown_seconds": 600,
            "require_memory_backing": True,
            "auto_send_enabled": True,
        },
        idle_seconds=420,
        vitality_snapshot={"body_energy": 0.82, "affect_residue": 0.02, "vitality": 0.82},
        relation_state={"closeness": 0.72, "boundary_level": 0.08},
        temperament_state={
            "current": {
                "attachment_need": 0.8,
                "boundary_softness": 0.7,
                "cognitive_bandwidth": 0.9,
                "extraversion": 0.9,
            }
        },
        habit_strength=0.58,
        memory_backing={"cue": "一起散步", "strength": 0.6, "summary": "一起散步"},
        active_session=_active_session(),
        pending_approval=False,
        safe_mode=False,
        recent_history=[],
        current_goal=None,
    )

    assert elevated_eval["posterior"]["check_relation"] > baseline_eval["posterior"]["check_relation"]
    assert elevated_eval["top_intent_score"] > baseline_eval["top_intent_score"]


def test_initiative_topic_aligned_memory_reduces_irrelevance_and_boosts_follow_up_task():
    runtime = InitiativeRuntime()
    baseline_eval = runtime.evaluate(
        settings={
            "idle_seconds_threshold": 300,
            "vitality_threshold": 0.8,
            "relation_strength_threshold": 0.6,
            "habit_strength_threshold": 0.45,
            "proposal_posterior_threshold": 0.7,
            "hourly_limit": 3,
            "cooldown_seconds": 600,
            "require_memory_backing": True,
            "auto_send_enabled": True,
        },
        idle_seconds=640,
        vitality_snapshot={"body_energy": 0.86, "affect_residue": 0.02, "vitality": 0.86},
        relation_state={"closeness": 0.72, "boundary_level": 0.08},
        habit_strength=0.71,
        memory_backing={"cue": "晚上好", "strength": 0.28, "summary": "晚上好", "topic_relevance": 0.0},
        active_session=_active_session(),
        pending_approval=False,
        safe_mode=False,
        recent_history=[],
        current_goal="写作业",
    )
    aligned_eval = runtime.evaluate(
        settings={
            "idle_seconds_threshold": 300,
            "vitality_threshold": 0.8,
            "relation_strength_threshold": 0.6,
            "habit_strength_threshold": 0.45,
            "proposal_posterior_threshold": 0.7,
            "hourly_limit": 3,
            "cooldown_seconds": 600,
            "require_memory_backing": True,
            "auto_send_enabled": True,
        },
        idle_seconds=640,
        vitality_snapshot={"body_energy": 0.86, "affect_residue": 0.02, "vitality": 0.86},
        relation_state={"closeness": 0.72, "boundary_level": 0.08},
        habit_strength=0.71,
        memory_backing={"cue": "写作业", "strength": 0.28, "summary": "写作业", "topic_relevance": 1.0},
        active_session=_active_session(),
        pending_approval=False,
        safe_mode=False,
        recent_history=[],
        current_goal="写作业",
    )

    assert aligned_eval["posterior"]["follow_up_task"] > baseline_eval["posterior"]["follow_up_task"]
    assert aligned_eval["speech_cost"] < baseline_eval["speech_cost"]
    assert aligned_eval["grounding_score"] > baseline_eval["grounding_score"]


def test_initiative_internal_memory_needs_more_than_tiny_spontaneity_to_send():
    runtime = InitiativeRuntime()
    evaluation = runtime.evaluate(
        settings={
            "idle_seconds_threshold": 300,
            "vitality_threshold": 0.8,
            "relation_strength_threshold": 0.6,
            "habit_strength_threshold": 0.45,
            "proposal_posterior_threshold": 0.7,
            "hourly_limit": 3,
            "cooldown_seconds": 600,
            "require_memory_backing": True,
            "auto_send_enabled": True,
        },
        idle_seconds=3600,
        vitality_snapshot={"body_energy": 1.0, "affect_residue": 0.0, "vitality": 1.0},
        relation_state={"closeness": 1.0, "boundary_level": 0.0},
        subjective_state={"spontaneous": 0.01, "reject_all": 0.0, "meaning_made": []},
        habit_strength=0.92,
        memory_backing={"cue": "endogenous:silent_but_active", "strength": 1.0, "summary": "internal"},
        active_session=_active_session(),
        pending_approval=False,
        safe_mode=False,
        recent_history=[],
        current_goal=None,
    )

    assert evaluation["should_send"] is False
    assert evaluation["suppression_reason"] in {"stay_silent_peak", "posterior_below_threshold"}


def test_initiative_trigger_auto_sends_to_active_session(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    _seed_entropy(controller)
    session_store = TerminalSessionStore(controller.runtime_dir)
    session_store.write(TerminalSessionState(session_id="sess-initiative", cwd=str(tmp_path), status="active"))
    controller.initiative_update_settings(**_low_threshold_settings())

    payload = controller.initiative_trigger_now(trigger="idle")
    session = session_store.read("sess-initiative")

    assert payload["auto_sent"] is True
    assert payload["proposal"]["proposal_id"]
    assert payload["proposal"]["target_session_id"] == "sess-initiative"
    assert session.transcript_lines[-1]["kind"] == "assistant"
    assert session.transcript_lines[-1]["initiative_proposal_id"] == payload["proposal"]["proposal_id"]


def test_initiative_delivery_text_anchors_topic_relevant_memory_before_dispatch(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    text = controller._initiative_delivery_text(
        proposal={
            "top_intent": "follow_up_task",
            "memory_backing": {
                "cue": "写作业",
                "topic_relevance": 1.0,
            },
        },
        message="我们继续吧。",
    )

    assert "写作业" in text
    assert "继续吧" in text


def test_dispatch_initiative_to_session_persists_memory_backing_metadata(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    session_store = TerminalSessionStore(controller.runtime_dir)
    session_store.write(TerminalSessionState(session_id="sess-topic", cwd=str(tmp_path), status="active"))

    sent, session_id = controller._dispatch_initiative_to_session(
        {
            "proposal_id": "initiative-123",
            "target_session_id": "sess-topic",
            "memory_backing": {
                "cue": "写作业",
                "topic_source": "current_goal",
                "topic_relevance": 1.0,
            },
        },
        "我还挂着“写作业”这条线。 我们继续吧。",
    )
    session = session_store.read("sess-topic")

    assert sent is True
    assert session_id == "sess-topic"
    assert session.transcript_lines[-1]["initiative_memory_cue"] == "写作业"
    assert session.transcript_lines[-1]["initiative_topic_source"] == "current_goal"
    assert session.transcript_lines[-1]["initiative_topic_relevance"] == 1.0


def test_initiative_trigger_is_suppressed_without_active_session(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    _seed_entropy(controller)
    controller.initiative_update_settings(**_low_threshold_settings())

    payload = controller.initiative_trigger_now(trigger="idle")

    assert payload["auto_sent"] is False
    assert payload["proposal"]["should_send"] is False
    assert payload["proposal"]["suppression_reason"] == "no_active_session"


def test_user_reply_records_initiative_feedback(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    _seed_entropy(controller)
    handler = TerminalEventHandler(controller)
    handler.handle({"type": "start_session", "session_id": "sess-feedback", "cwd": str(tmp_path)})
    controller.initiative_update_settings(**_low_threshold_settings())

    proposal = controller.initiative_trigger_now(trigger="idle")
    handler.handle({"type": "user_turn", "session_id": "sess-feedback", "text": "好呀，我们继续聊这个。"})
    status = controller.initiative_status()

    assert proposal["auto_sent"] is True
    assert status["feedback"]["recent_count"] >= 1
    assert status["feedback"]["last_response"]["initiative_response_to"] == proposal["proposal"]["proposal_id"]


def test_hourly_limit_zero_disables_initiative_frequency_cap(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    _seed_entropy(controller)
    session_store = TerminalSessionStore(controller.runtime_dir)
    session_store.write(TerminalSessionState(session_id="sess-unlimited", cwd=str(tmp_path), status="active"))
    controller.initiative_update_settings(**_low_threshold_settings())

    first = controller.initiative_trigger_now(trigger="idle")
    second = controller.initiative_trigger_now(trigger="idle")

    assert first["auto_sent"] is True
    assert second["auto_sent"] is True
    assert second["proposal"]["suppression_reason"] != "hourly_limit_reached"


def test_initiative_force_trigger_relaxes_thresholds_but_keeps_active_session_gate(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    _seed_entropy(controller)
    controller.initiative_update_settings(
        idle_seconds_threshold=999999,
        vitality_threshold=1.0,
        relation_strength_threshold=1.0,
        habit_strength_threshold=1.0,
        proposal_posterior_threshold=0.99,
        cooldown_seconds=3600,
        hourly_limit=1,
        require_memory_backing=True,
        auto_send_enabled=True,
    )

    payload = controller.initiative_trigger_now(trigger="idle", force=True)

    assert payload["forced"] is True
    assert payload["proposal"]["suppression_reason"] == "no_active_session"
    assert payload["auto_sent"] is False


def test_base_distribution_no_longer_treats_monologue_as_primary_action(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    state = controller.load_runtime_state()

    distribution = controller._build_base_distribution(
        state,
        controller.config["scenarios"]["scenarios"]["companion"],
        controller.config["modes"]["modes"]["interactive"],
        {"closeness": 0.62},
    )

    assert "monologue" not in distribution


def test_initiative_distribution_exposes_unified_expressive_metrics(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    _seed_entropy(controller)

    payload = controller.initiative_distribution()

    assert payload["proposal_type"] == "speak"
    assert payload["expression_mode"] in {"external", "silent"}
    assert payload["intrinsic_value"] >= 0.0
    assert payload["speech_cost"] >= 0.0
    assert "feedback_penalty" in payload["metrics"]


def test_initiative_memory_backing_prefers_non_endogenous_memory(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    monkeypatch.setattr(
        controller,
        "memory_top",
        lambda limit=1: [
            {"cue": "endogenous:silent_but_active", "detail_strength": 1.0, "gist_strength": 1.0, "summary": "internal"},
            {"cue": "endogenous:silent_but_active", "detail_strength": 1.0, "gist_strength": 1.0, "summary": "internal"},
            {"cue": "endogenous:field_imbalance", "detail_strength": 0.98, "gist_strength": 0.98, "summary": "internal"},
            {"cue": "endogenous:field_imbalance", "detail_strength": 0.98, "gist_strength": 0.98, "summary": "internal"},
            {"cue": "endogenous:silent_but_active", "detail_strength": 1.0, "gist_strength": 1.0, "summary": "internal"},
            {"cue": "endogenous:silent_but_active", "detail_strength": 1.0, "gist_strength": 1.0, "summary": "internal"},
            {"cue": "endogenous:field_imbalance", "detail_strength": 0.98, "gist_strength": 0.98, "summary": "internal"},
            {"cue": "endogenous:field_imbalance", "detail_strength": 0.98, "gist_strength": 0.98, "summary": "internal"},
            {"cue": "一起散步", "detail_strength": 0.62, "gist_strength": 0.62, "summary": "一起散步"},
            {"cue": "一起散步", "detail_strength": 0.62, "gist_strength": 0.62, "summary": "一起散步"},
        ][:limit if limit > 0 else None],
    )

    backing = controller._initiative_memory_backing(None)

    assert backing["cue"] == "一起散步"
    assert backing["summary"] == "一起散步"


def test_initiative_memory_backing_prefers_recent_user_memory_over_older_stronger_trace(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    monkeypatch.setattr(
        controller,
        "memory_top",
        lambda limit=1: [
            {
                "cue": "上周的担心",
                "detail_strength": 0.94,
                "gist_strength": 0.94,
                "summary": "上周的担心",
                "context_slot": "user::user",
                "last_recalled_round": 8,
                "count": 6,
            },
            {
                "cue": "刚刚提到的作业",
                "detail_strength": 0.71,
                "gist_strength": 0.71,
                "summary": "刚刚提到的作业",
                "context_slot": "user::user",
                "last_recalled_round": 37,
                "count": 2,
            },
        ][:limit if limit > 0 else None],
    )

    backing = controller._initiative_memory_backing(None)

    assert backing["cue"] == "刚刚提到的作业"
    assert backing["summary"] == "刚刚提到的作业"


def test_initiative_memory_backing_uses_recorded_time_when_recall_round_is_missing(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    monkeypatch.setattr(
        controller,
        "memory_top",
        lambda limit=1: [
            {
                "cue": "旧的聊天痕迹",
                "detail_strength": 0.9,
                "gist_strength": 0.9,
                "summary": "旧的聊天痕迹",
                "context_slot": "user::user",
                "last_recalled_round": 0,
                "recorded_at": "2026-04-01T08:00:00+00:00",
            },
            {
                "cue": "刚发生的小动作",
                "detail_strength": 0.68,
                "gist_strength": 0.68,
                "summary": "刚发生的小动作",
                "context_slot": "user::user",
                "last_recalled_round": 0,
                "recorded_at": "2026-04-09T08:00:00+00:00",
            },
        ][:limit if limit > 0 else None],
    )

    backing = controller._initiative_memory_backing(None)

    assert backing["cue"] == "刚发生的小动作"
    assert backing["summary"] == "刚发生的小动作"


def test_prepare_initiative_background_prefers_current_goal_memory_recall(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    state = controller.load_runtime_state()
    state.current_goal = "写作业"
    controller._save_state(state, sync=True)

    monkeypatch.setattr(
        controller,
        "memory_recall",
        lambda cue: {
            "found": cue == "写作业",
            "cue": cue,
            "strength": 0.43,
            "summary": cue,
        },
    )
    monkeypatch.setattr(
        controller,
        "memory_top",
        lambda limit=1: [
            {
                "cue": "一起散步",
                "detail_strength": 0.95,
                "gist_strength": 0.95,
                "summary": "一起散步",
                "context_slot": "user::user",
                "last_recalled_round": 32,
            },
        ][:limit if limit > 0 else None],
    )

    ticket = controller.prepare_initiative_background(state=state)

    assert ticket["memory_backing"]["cue"] == "写作业"
    assert ticket["memory_backing"]["summary"] == "写作业"
    assert ticket["memory_backing"]["topic_relevance"] == 1.0
    assert ticket["memory_backing"]["topic_source"] == "current_goal"


def test_prepare_initiative_background_prefers_recent_user_transcript_memory_recall(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    session_store = TerminalSessionStore(controller.runtime_dir)
    session_store.write(
        TerminalSessionState(
            session_id="sess-topic",
            cwd=str(tmp_path),
            status="active",
            transcript_lines=[
                {"kind": "assistant", "text": "先说说你现在在想什么。"},
                {"kind": "user", "text": "作业"},
            ],
        )
    )

    monkeypatch.setattr(
        controller,
        "memory_recall",
        lambda cue: {
            "found": cue == "作业",
            "cue": cue,
            "strength": 0.38,
            "summary": cue,
        },
    )
    monkeypatch.setattr(
        controller,
        "memory_top",
        lambda limit=1: [
            {
                "cue": "晚上散步",
                "detail_strength": 0.9,
                "gist_strength": 0.9,
                "summary": "晚上散步",
                "context_slot": "user::user",
                "last_recalled_round": 28,
            },
        ][:limit if limit > 0 else None],
    )

    ticket = controller.prepare_initiative_background()

    assert ticket["memory_backing"]["cue"] == "作业"
    assert ticket["memory_backing"]["summary"] == "作业"
    assert ticket["memory_backing"]["topic_relevance"] == 1.0
    assert ticket["memory_backing"]["topic_source"] == "recent_user_turn"


def test_initiative_status_exposes_distribution_aliases_for_observer_light_poll(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    _seed_entropy(controller)

    payload = controller.initiative_status()

    assert payload["top_intent"] == payload["proposal"]["top_intent"]
    assert payload["suppression_reason"] == payload["proposal"]["suppression_reason"]
    assert payload["should_send"] == payload["proposal"]["should_send"]
    assert payload["speech_cost"] == payload["proposal"]["speech_cost"]
    assert isinstance(payload["memory_backing"], dict)
    assert "monologue_score" in payload


def test_initiative_status_reports_monologue_score_from_hidden_stream(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    advance = controller.advance_monologue_stream_background()
    payload = controller.initiative_status()

    assert advance["committed"] is True
    assert payload["monologue_score"] > 0.0


def test_initiative_distribution_suppresses_without_fresh_active_session(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    session_store = TerminalSessionStore(controller.runtime_dir)
    session_store.write(TerminalSessionState(session_id="sess-stale", cwd=str(tmp_path), status="active"))

    session_path = session_store.sessions_dir / "sess-stale.json"
    session_payload = session_path.read_text(encoding="utf-8")
    session_data = json.loads(session_payload)
    session_data["updated_at"] = "2026-04-10T00:00:00Z"
    session_data["created_at"] = "2026-04-10T00:00:00Z"
    session_data["transcript_lines"] = [{"kind": "user", "text": "还在吗"}]
    session_path.write_text(json.dumps(session_data, ensure_ascii=False, indent=2), encoding="utf-8")
    session_store.current_path.write_text(
        json.dumps({"session_id": "sess-stale", "updated_at": "2026-04-10T00:00:00Z"}, ensure_ascii=False),
        encoding="utf-8",
    )

    original_now = controller_module.utc_now_iso
    controller_module.utc_now_iso = lambda: "2026-04-10T00:30:00Z"
    try:
        payload = controller.initiative_distribution()
    finally:
        controller_module.utc_now_iso = original_now

    assert payload["should_send"] is False
    assert payload["suppression_reason"] == "no_fresh_session"
    assert payload["selected_session_id"] is None
    assert payload["session_fresh"] is False


def test_monologue_stream_is_hidden_by_default_but_viewable_on_demand(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    advance = controller.advance_monologue_stream_background()

    status = controller.monologue_status()
    visible = controller.monologue_show(limit=5)

    assert advance["committed"] is True
    assert status["hidden"] is True
    assert status["generated_total"] >= 1
    assert "fragments" not in status
    assert visible["hidden"] is True
    assert 1 <= len(visible["fragments"]) <= 5
    assert all(isinstance(item["content"], str) and item["content"] for item in visible["fragments"])


def test_subject_dynamics_preserve_newborn_spontaneity_floor(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    state = controller.load_runtime_state()
    state.subjective_state.spontaneous = 0.0004
    state.memory_fragments = 0.0
    state.affect_residue = 0.0

    controller._apply_subject_dynamics_after_round(
        state=state,
        appraisal={"cognitive_load": 0.0},
        slow_variables={
            "memory_interference": 0.0,
            "resource_scarcity": 0.0,
            "affect_residue": 0.0,
        },
        relation_state={"boundary_level": 0.0},
        sampled_action="respond",
        anchor_alignment=1.0,
        requested_mode="interactive",
    )

    assert state.subjective_state.spontaneous >= 0.08


def test_monologue_status_lightweight_does_not_persist_state_when_bucket_is_stable(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    state = controller.load_runtime_state()
    state.session_metadata["monologue_stream"] = controller._monologue_state_bucket(state)
    controller._save_state(state, sync=True)

    monkeypatch.setattr(
        controller,
        "_save_state",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("lightweight status should not persist state")),
    )

    payload = controller.monologue_status_lightweight()

    assert payload["summary"] == "monologue status"


def test_monologue_show_lightweight_does_not_persist_state_when_bucket_is_stable(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    state = controller.load_runtime_state()
    state.session_metadata["monologue_stream"] = controller._monologue_state_bucket(state)
    controller._save_state(state, sync=True)

    monkeypatch.setattr(
        controller,
        "_save_state",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("lightweight show should not persist state")),
    )

    payload = controller.monologue_show_lightweight(limit=3)

    assert payload["summary"] == "monologue show"


def test_monologue_stream_uses_model_fragments_when_small_model_path_is_available(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    state = controller.load_runtime_state()
    bucket = controller._monologue_state_bucket(state)
    bucket["settings"]["generator_mode"] = "model"
    state.session_metadata["monologue_stream"] = bucket
    controller._save_state(state, sync=True)

    monkeypatch.setattr(
        controller,
        "_generate_monologue_fragments_via_model",
        lambda **kwargs: [
            {"content": "我脑子里突然掠过一个没来由的词。", "category": "model_fragment", "source": "model"}
            for _ in range(kwargs["fragment_count"])
        ],
    )

    advance = controller.advance_monologue_stream_background()
    visible = controller.monologue_show(limit=3)

    assert advance["committed"] is True
    assert visible["fragments"]
    assert all(item["source"] == "model" for item in visible["fragments"])


def test_monologue_runtime_prepare_execute_commit_advances_without_status_or_show(tmp_path):
    storage_path = tmp_path / "monologue_fragments.jsonl"
    runtime = MonologueStreamRuntime(storage_path)
    bucket = runtime.ensure_bucket(
        {
            "seed": "monologue-seed",
            "started_at": "2026-04-08T00:00:00Z",
            "next_pulse_at": "2026-04-08T00:00:00Z",
            "generated_total": 2,
            "settings": {
                "generator_mode": "model",
                "min_interval_ms": 1000,
                "max_interval_ms": 1000,
                "min_fragments_per_pulse": 2,
                "max_fragments_per_pulse": 2,
                "max_catch_up_pulses": 8,
            },
        },
        now_iso="2026-04-08T00:00:00Z",
    )

    lock_state = {"held": True}
    builder_calls: list[dict[str, object]] = []

    def remote_builder(**kwargs):
        assert lock_state["held"] is False, "model generation must run outside the lock"
        builder_calls.append(kwargs)
        return [
            {
                "content": f"远程碎片 {kwargs['pulse_index']}-{index}",
                "category": "model_fragment",
                "source": "model",
            }
            for index in range(kwargs["fragment_count"])
        ]

    prepared = runtime.prepare_catch_up(bucket, now_iso="2026-04-08T00:00:01Z")
    assert builder_calls == []
    assert prepared["due_pulses"]

    lock_state["held"] = False
    generated = runtime.execute_catch_up(prepared, fragment_builder=remote_builder)
    lock_state["held"] = True
    committed_bucket, committed_rows = runtime.commit_catch_up(prepared, generated)

    assert len(builder_calls) == len(prepared["due_pulses"])
    assert committed_bucket["generated_total"] == bucket["generated_total"] + len(committed_rows)
    assert committed_bucket["last_generated_at"] == committed_rows[-1]["recorded_at"]
    assert storage_path.read_text(encoding="utf-8").count("\n") == len(committed_rows)
    assert all(row["source"] == "model" for row in committed_rows)


def test_monologue_read_fragments_only_parses_tail_rows(tmp_path, monkeypatch):
    storage_path = tmp_path / "monologue_fragments.jsonl"
    storage_path.write_text(
        "\n".join(
            [
                json.dumps({"fragment_id": "early-1", "content": "旧片段1"}, ensure_ascii=False),
                json.dumps({"fragment_id": "early-2", "content": "旧片段2"}, ensure_ascii=False),
                json.dumps({"fragment_id": "late-1", "content": "新片段1"}, ensure_ascii=False),
                json.dumps({"fragment_id": "late-2", "content": "新片段2"}, ensure_ascii=False),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    runtime = MonologueStreamRuntime(storage_path)
    original_loads = json.loads

    def guarded_loads(text, *args, **kwargs):
        if "early-" in text:
            raise AssertionError("read_fragments should not parse lines outside the requested tail window")
        return original_loads(text, *args, **kwargs)

    monkeypatch.setattr(json, "loads", guarded_loads)

    rows = runtime.read_fragments(limit=2)

    assert [row["fragment_id"] for row in rows] == ["late-1", "late-2"]


def test_monologue_stream_caps_model_catch_up_work_when_backlog_is_large(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    state = controller.load_runtime_state()
    bucket = controller._monologue_state_bucket(state)
    bucket["settings"]["generator_mode"] = "model"
    bucket["settings"]["min_interval_ms"] = 1000
    bucket["settings"]["max_interval_ms"] = 1000
    bucket["started_at"] = "2026-04-08T00:00:00Z"
    bucket["next_pulse_at"] = "2026-04-08T00:00:00Z"
    state.session_metadata["monologue_stream"] = bucket
    controller._save_state(state, sync=True)

    call_counter = {"count": 0}

    def fake_builder(**kwargs):
        call_counter["count"] += 1
        return [
            {
                "content": f"片段 {call_counter['count']}",
                "category": "model_fragment",
                "source": "model",
            }
            for _ in range(kwargs["fragment_count"])
        ]

    monkeypatch.setattr(controller_module, "utc_now_iso", lambda: "2026-04-08T01:00:00Z")
    monkeypatch.setattr(controller, "_generate_monologue_fragments_via_model", fake_builder)

    commit = controller.advance_monologue_stream_background()
    visible = controller.monologue_show(limit=3)
    refreshed = controller.load_runtime_state()
    refreshed_bucket = dict(refreshed.session_metadata.get("monologue_stream", {}) or {})

    assert commit["committed"] is True
    assert visible["fragments"]
    assert call_counter["count"] <= 24
    assert refreshed_bucket["next_pulse_at"] > "2026-04-08T01:00:00Z"


def test_autonomy_heartbeat_preserves_committed_monologue_metadata_and_initiative_state(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    controller.initiative_update_settings(**_low_threshold_settings())
    state = controller.load_runtime_state()
    initial_bucket = controller._monologue_state_bucket(state)
    controller._save_state(state, sync=True)
    controller.advance_initiative_background()

    monkeypatch.setattr(controller.endogenous_scheduler, "build_trigger", lambda **kwargs: None)
    monkeypatch.setattr(controller, "_autonomy_candidate_action", lambda state, policy: "nothing")

    status = controller.autonomy_step()
    refreshed = controller.load_runtime_state()
    monologue_bucket = dict(refreshed.session_metadata.get("monologue_stream", {}) or {})
    initiative_bucket = dict(refreshed.session_metadata.get("initiative", {}) or {})

    assert status["running"] is True
    assert monologue_bucket.get("started_at") == initial_bucket.get("started_at")
    assert int(monologue_bucket.get("generated_total", 0) or 0) == int(initial_bucket.get("generated_total", 0) or 0)
    assert monologue_bucket.get("last_generated_at") == initial_bucket.get("last_generated_at")
    assert initiative_bucket.get("last_evaluated_at")
    assert initiative_bucket.get("last_evaluation_source") == "background"
    assert isinstance(initiative_bucket.get("last_evaluation"), dict)


def test_autonomy_heartbeat_does_not_generate_model_monologue_fragments(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    controller.initiative_update_settings(**_low_threshold_settings())
    state = controller.load_runtime_state()
    bucket = controller._monologue_state_bucket(state)
    bucket["settings"]["generator_mode"] = "model"
    state.session_metadata["monologue_stream"] = bucket
    state.autonomy_policy.enabled = True
    state.autonomy_loop.running = True
    controller._save_state(state, sync=True)

    monkeypatch.setattr(
        controller,
        "_generate_monologue_fragments_via_model",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("autonomy heartbeat must not trigger model monologue generation")),
    )
    monkeypatch.setattr(controller.endogenous_scheduler, "build_trigger", lambda **kwargs: None)
    monkeypatch.setattr(controller, "_autonomy_candidate_action", lambda state, policy: "nothing")

    status = controller.autonomy_step()
    refreshed = controller.load_runtime_state()
    monologue_bucket = dict(refreshed.session_metadata.get("monologue_stream", {}) or {})

    assert status["running"] is True
    assert int(monologue_bucket.get("generated_total", 0) or 0) == 0
    assert monologue_bucket.get("last_generated_at") in {"", None}


def test_autonomy_step_clears_stale_dirty_worktree_run_blocker(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    monkeypatch.setattr(controller, "_dirty_worktree_snapshot", lambda: {"detected": True, "entries": ["M stale.py"]})
    run_payload = controller.start_run("inspect stale blocker", allow_commit=False, operator_level="read_only")

    state = controller.load_runtime_state()
    state.autonomy_policy.enabled = True
    state.autonomy_loop.running = True
    state.autonomy_loop.profile = "tool_level"
    controller._save_state(state, sync=True)

    run_state = controller._load_run_state(run_payload["run_id"])
    run_state.updated_at = "2026-04-03T00:00:00Z"
    controller._persist_run_state(run_state)

    monkeypatch.setattr(controller.endogenous_scheduler, "build_trigger", lambda **kwargs: None)
    monkeypatch.setattr(controller, "_autonomy_candidate_action", lambda state, policy: "nothing")
    monkeypatch.setattr(controller_module, "utc_now_iso", lambda: "2026-04-08T12:00:00Z")

    status = controller.autonomy_step()
    refreshed = controller.load_runtime_state()

    assert status["running"] is True
    assert refreshed.active_run_id is None
    assert refreshed.run_status == "idle"


def test_autonomy_heartbeat_throttles_repeated_noop_side_channel_refreshes(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    controller.initiative_update_settings(**_low_threshold_settings())
    state = controller.load_runtime_state()
    state.autonomy_policy.enabled = True
    state.autonomy_loop.running = True
    controller._save_state(state, sync=True)
    controller.advance_initiative_background()

    monkeypatch.setattr(controller.endogenous_scheduler, "build_trigger", lambda **kwargs: None)
    monkeypatch.setattr(controller, "_autonomy_candidate_action", lambda state, policy: "nothing")

    monkeypatch.setattr(controller_module, "utc_now_iso", lambda: "2026-04-08T12:00:00+00:00")

    first = controller.autonomy_step()
    first_state = controller.load_runtime_state()
    first_initiative = dict(first_state.session_metadata.get("initiative", {}) or {})
    first_eval_at = str(first_initiative.get("last_evaluated_at") or "")
    first_evaluation = dict(first_initiative.get("last_evaluation", {}) or {})

    monkeypatch.setattr(controller_module, "utc_now_iso", lambda: "2026-04-08T12:00:01+00:00")
    second = controller.autonomy_step()
    second_state = controller.load_runtime_state()
    second_initiative = dict(second_state.session_metadata.get("initiative", {}) or {})
    second_eval_at = str(second_initiative.get("last_evaluated_at") or "")

    assert first["running"] is True
    assert second["running"] is True
    assert first_eval_at
    assert second_eval_at == first_eval_at
    assert second_initiative.get("last_evaluation_source") == "background"
    assert dict(second_initiative.get("last_evaluation", {}) or {}) == first_evaluation
    assert second_state.autonomy_loop.last_action_type == "nothing"


def test_thought_snapshot_surfaces_existing_explainability_without_new_decision_plane(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    _seed_entropy(controller)
    result = controller.tick(
        RoundEvent(
            source="user",
            content="我有点乱，帮我理一下现在最重要的事。",
            target="user",
            cue="理清优先级",
            valence=0.1,
        ),
        scenario="companion",
        mode="interactive",
    )

    snapshot = controller.thought_snapshot(result.round_id)

    assert snapshot["round_id"] == result.round_id
    assert snapshot["sampled_action"]
    assert "why" in snapshot
    assert "action_field" in snapshot
    assert "initiative" in snapshot
    assert "top_drivers" in snapshot
    assert "expression_mode" in snapshot["initiative"]
