from pathlib import Path

from nalr.cil.runtime import build_endogenous_status_payload
from nalr.runtime.controller import RuntimeController
from nalr.schemas.models import RoundEvent
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


def test_build_endogenous_status_payload_uses_public_trace_storage_status(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    monkeypatch.setattr(
        controller,
        "state_payload",
        lambda: {
            "round_count": 0,
            "endogenous_scheduler_state": {},
            "endogenous_state": {},
        },
    )
    monkeypatch.setattr(controller, "trace_storage_status", lambda: {"kind": "public"})

    def _private_should_not_be_used(*args, **kwargs):
        raise AssertionError("private trace storage helper leaked outside controller facade")

    monkeypatch.setattr(controller, "_trace_storage_payload", _private_should_not_be_used)

    payload = build_endogenous_status_payload(controller)

    assert payload["storage"] == {"kind": "public"}


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


def test_monologue_stream_is_hidden_by_default_but_viewable_on_demand(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    status = controller.monologue_status()
    visible = controller.monologue_show(limit=5)

    assert status["hidden"] is True
    assert status["generated_total"] >= 1
    assert "fragments" not in status
    assert visible["hidden"] is True
    assert 1 <= len(visible["fragments"]) <= 5
    assert all(isinstance(item["content"], str) and item["content"] for item in visible["fragments"])


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

    visible = controller.monologue_show(limit=3)

    assert visible["fragments"]
    assert all(item["source"] == "model" for item in visible["fragments"])


def test_autonomy_heartbeat_refreshes_hidden_monologue_metadata_and_initiative_state(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    controller.initiative_update_settings(**_low_threshold_settings())
    state = controller.load_runtime_state()
    initial_bucket = controller._monologue_state_bucket(state)
    controller._save_state(state, sync=True)

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
    assert initiative_bucket.get("last_evaluation_source") == "heartbeat"
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
