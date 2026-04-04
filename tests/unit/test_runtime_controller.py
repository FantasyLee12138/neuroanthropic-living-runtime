import shutil
from pathlib import Path

from nalr.providers.router import ModelResponse
from nalr.runtime.controller import RuntimeController
from nalr.schemas.models import QuantumEntropyRef, RenderPlan, RoundEvent
from nalr.trace.store import TraceStore


CONFIG_ROOT = Path(__file__).resolve().parents[2] / "config"


def test_tick_degrades_when_entropy_provider_unavailable_and_config_allows_fallback(tmp_path, monkeypatch):
    config_root = tmp_path / "config"
    shutil.copytree(CONFIG_ROOT, config_root)
    (config_root / "entropy.yaml").write_text(
        """entropy:
  provider: anu_qrng
  endpoint: "https://broken-qrng.test/api"
  timeout_s: 1.7
  prefetch_bytes: 96
  min_batch_bytes: 32
  max_batch_bytes: 1024
  hard_block_on_unavailable: false
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("NALR_DISABLE_TEST_QRNG_SEED", "1")

    controller = RuntimeController(project_root=tmp_path, config_root=config_root)

    def broken_fetch_batch(*, byte_count: int):
        raise RuntimeError(f"provider unavailable for {byte_count} bytes")

    monkeypatch.setattr(controller.entropy_pool.provider, "fetch_batch", broken_fetch_batch)

    result = controller.tick(
        RoundEvent(
            source="user",
            content="你好，今天状态怎么样？",
            target="user",
            cue="状态",
        ),
        scenario="chat",
        mode="interactive",
    )

    assert controller.entropy_pool.prefetch_bytes == 96
    assert controller.entropy_pool.provider.endpoint == "https://broken-qrng.test/api"
    assert controller.entropy_pool.provider.timeout_s == 1.7
    assert result.trace.stochastic_state["entropy_ref"]["source"] == "deterministic_fallback"
    assert result.trace.stochastic_state["entropy_ref"]["degraded"] is True
    assert result.trace.stochastic_state["entropy_ref"]["health_state"] == "degraded"
    state = controller.load_runtime_state()
    assert state.entropy_health_state["state"] == "degraded"
    assert state.entropy_health_state["last_failure"]["failure_class"] == "provider_fetch_failed"
    assert state.last_entropy_failure == {}


def test_tick_records_trace_and_top_drivers(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    result = controller.tick(
        RoundEvent(
            source="user",
            content="Please help me plan my day and remember breakfast.",
            target="user",
            valence=0.2,
            energy_delta=-0.1,
        ),
        scenario="task",
        mode="interactive",
    )

    assert result.round_id == 1
    assert result.sampled_action.name in {"respond", "plan", "recall", "rest", "clarify"}
    assert len(result.trace.top_drivers) == 3
    assert any(item.agent_name == "PFCAgent" for item in result.trace.contributions)


def test_model_status_surfaces_route_health_and_recent_model_activity(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    controller.tick(
        RoundEvent(
            source="user",
            content="帮我规划今晚，并记住我晚饭想吃面。",
            target="user",
            cue="面",
        ),
        scenario="chat",
        mode="interactive",
    )

    status = controller.model_status()

    assert status["current_round"] == 1
    assert status["credential_present"] is False
    assert "pfc" in status["routes"]
    assert "renderer" in status["routes"]
    assert "generate_candidates" in status["routes"]["pfc"]["bound_skills"]
    assert "render_expression" in status["routes"]["renderer"]["bound_skills"]
    assert status["routes"]["pfc"]["recent_fallback_count"] >= 1
    assert status["routes"]["renderer"]["recent_fallback_count"] >= 1
    assert status["routes"]["pfc"]["last_failure_reason"] == "fallback_to_rules"
    assert status["routes"]["renderer"]["last_failure_reason"] == "fallback_to_rules"


def test_command_safe_mode_and_checkpoint_emit_command_trace(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    safe_result = controller.apply_command("safe on")
    checkpoint = controller.checkpoint()

    assert safe_result.applied is True
    assert safe_result.scope == "runtime"
    assert safe_result.delta["safe_mode"] is True
    assert checkpoint.checkpoint_id.startswith("ckpt-")
    assert controller.load_runtime_state().safe_mode is True
    command_traces = controller.trace_store.list_command_traces()
    assert any(item["command"] == "safe on" for item in command_traces)
    assert any(item["command"] == "checkpoint create" for item in command_traces)


def test_state_and_trace_payloads_surface_trace_storage_status(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    controller.tick(
        RoundEvent(
            source="user",
            content="Please remember tea and help me plan dinner.",
            target="user",
            cue="tea",
        ),
        scenario="task",
        mode="interactive",
    )
    controller.flush_pending_io(raise_on_error=False)

    state_payload = controller.state_payload()
    trace_payload = controller.trace_round(1)
    why_payload = controller.why_this(1)
    contribution_payload = controller.contribution_breakdown(1)

    assert state_payload["trace_storage"]["trace_sync_state"] == "healthy"
    assert state_payload["trace_storage"]["parquet_live_ready"] is True
    assert trace_payload["storage"]["read_source"] == "parquet"
    assert trace_payload["storage"]["trace_sync_state"] == "healthy"
    assert why_payload["storage"]["read_source"] == "parquet"
    assert contribution_payload["storage"]["trace_sync_state"] == "healthy"


def test_cognitive_snapshot_humanizes_chat_intent_without_internal_tokens(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    controller.tick(
        RoundEvent(source="user", content="你好，今天过得怎么样？", target="user"),
        scenario="chat",
        mode="interactive",
    )

    snapshot = controller.state_payload()["cognitive_snapshot"]

    assert snapshot["current_intent"].startswith("正在自然交流，准备")
    assert "general_exchange" not in snapshot["current_intent"]
    assert "respond" not in snapshot["current_intent"]


def test_cognitive_snapshot_uses_human_readable_continuity_before_naming(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    snapshot = controller.state_payload()["cognitive_snapshot"]

    assert snapshot["identity"]["continuity"] == "还在形成稳定称呼"
    assert "unnamed" not in snapshot["identity"]["continuity"]
    assert snapshot["authenticity"]["summary"] == "还没有足够证据判断这轮真实感"


def test_terminal_route_probe_prefers_direct_chat_for_identity_prompt_without_writes(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    before_state_hash = controller._state_hash(controller.load_runtime_state())
    before_rounds = controller.trace_store.list_rounds()
    before_memory = controller.memory_top()
    before_habits = controller.habit_top()

    probe = controller.probe_terminal_route("你好，你是谁？你有名字吗？")

    after_state_hash = controller._state_hash(controller.load_runtime_state())
    after_rounds = controller.trace_store.list_rounds()
    after_memory = controller.memory_top()
    after_habits = controller.habit_top()

    assert probe["route"] == "direct_chat"
    assert probe["chat_mass"] > probe["task_mass"]
    assert before_state_hash == after_state_hash
    assert before_rounds == after_rounds
    assert before_memory == after_memory
    assert before_habits == after_habits


def test_terminal_route_probe_prefers_task_run_for_repo_task_without_writes(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    before_state_hash = controller._state_hash(controller.load_runtime_state())
    before_rounds = controller.trace_store.list_rounds()

    probe = controller.probe_terminal_route("总结这个仓库结构")

    after_state_hash = controller._state_hash(controller.load_runtime_state())
    after_rounds = controller.trace_store.list_rounds()

    assert probe["route"] == "task_run"
    assert probe["task_mass"] >= 0.55
    assert probe["task_mass"] - probe["chat_mass"] >= 0.10
    assert before_state_hash == after_state_hash
    assert before_rounds == after_rounds


def test_plan_turn_uses_single_probe_path_for_terminal_routing(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    calls: list[str] = []

    def fake_probe(event, *, scenario, mode):
        calls.append(scenario)
        return {
            "scenario": scenario,
            "mode": mode,
            "top_action": "plan",
            "action_distribution": {"plan": 0.8, "respond": 0.2},
            "task_mass": 0.8,
            "chat_mass": 0.2,
        }

    monkeypatch.setattr(controller, "_probe_distribution_for_scenario", fake_probe)

    plan = controller.plan_turn("检查 worker.py 并规划下一步")

    assert plan.route == "task_run"
    assert calls == ["task"]


def test_hot_only_budget_for_low_salience(tmp_path, monkeypatch):
    monkeypatch.setattr(TraceStore, "mark_trace_sync_healthy", lambda self: None)
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    calls: list[tuple[str, str | None, tuple[str, ...]]] = []
    entropy_ref = QuantumEntropyRef(source="test")
    derived_cue = "teacup"

    def fake_ingest_event(event, **kwargs):
        assert event.cue is None
        return derived_cue

    def fake_recall_strength(cue, tier_budget=("hot", "warm", "archive")):
        calls.append(("recall_strength", cue, tuple(tier_budget)))
        return 0.12

    def fake_recall(cue, *, tier_budget=("hot", "warm", "archive")):
        calls.append(("recall", cue, tuple(tier_budget)))
        return {
            "cue": cue,
            "tier": "hot",
            "strength": 0.48,
            "detail": False,
            "found": True,
            "interference": 0.08,
            "evidence": [],
        }

    monkeypatch.setattr(controller.memory_store, "ingest_event", fake_ingest_event)
    monkeypatch.setattr(controller.memory_store, "recall_strength", fake_recall_strength)
    monkeypatch.setattr(controller.memory_store, "recall", fake_recall)
    monkeypatch.setattr(controller.entropy_pool, "truncated_normal", lambda *args, **kwargs: (0.0, entropy_ref))
    monkeypatch.setattr(controller.entropy_pool, "uniform_range", lambda *args, **kwargs: (0.0, entropy_ref))
    monkeypatch.setattr(controller.entropy_pool, "uniform", lambda *args, **kwargs: (0.0, entropy_ref))

    event = RoundEvent(source="user", content="hello teacup tonight", target="user", valence=0.05)

    _, _, _, _, probe_context, _, _ = controller._probe_context(event, scenario="chat", mode="interactive")
    tick_context: dict[str, float | str | None] = {}
    original_relation_state = controller._relation_state

    def capture_relation_state(event, context):
        tick_context.update(
            cue=context.get("cue"),
            recall_strength=context.get("recall_strength"),
            interference=context.get("interference"),
        )
        return original_relation_state(event, context)

    monkeypatch.setattr(controller, "_relation_state", capture_relation_state)
    controller.tick(event, scenario="chat", mode="interactive")

    assert calls == [
        ("recall_strength", derived_cue, ("hot",)),
        ("recall", derived_cue, ("hot",)),
        ("recall_strength", derived_cue, ("hot",)),
        ("recall", derived_cue, ("hot",)),
    ]
    assert probe_context["cue"] == derived_cue
    assert probe_context["recall_strength"] == 0.12
    assert probe_context["interference"] == 0.08
    assert tick_context == {
        "cue": derived_cue,
        "recall_strength": 0.12,
        "interference": 0.08,
    }


def test_full_budget_for_high_salience(tmp_path, monkeypatch):
    monkeypatch.setattr(TraceStore, "mark_trace_sync_healthy", lambda self: None)
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    calls: list[tuple[str, str | None, tuple[str, ...]]] = []
    entropy_ref = QuantumEntropyRef(source="test")
    derived_cue = "fallbackfix"

    def fake_ingest_event(event, **kwargs):
        assert event.cue is None
        return derived_cue

    def fake_recall_strength(cue, tier_budget=("hot", "warm", "archive")):
        calls.append(("recall_strength", cue, tuple(tier_budget)))
        return 0.66

    def fake_recall(cue, *, tier_budget=("hot", "warm", "archive")):
        calls.append(("recall", cue, tuple(tier_budget)))
        return {
            "cue": cue,
            "tier": "warm",
            "strength": 0.91,
            "detail": True,
            "found": True,
            "interference": 0.14,
            "evidence": ["warm memory"],
        }

    monkeypatch.setattr(controller.memory_store, "ingest_event", fake_ingest_event)
    monkeypatch.setattr(controller.memory_store, "recall_strength", fake_recall_strength)
    monkeypatch.setattr(controller.memory_store, "recall", fake_recall)
    monkeypatch.setattr(controller.entropy_pool, "truncated_normal", lambda *args, **kwargs: (0.0, entropy_ref))
    monkeypatch.setattr(controller.entropy_pool, "uniform_range", lambda *args, **kwargs: (0.0, entropy_ref))
    monkeypatch.setattr(controller.entropy_pool, "uniform", lambda *args, **kwargs: (0.0, entropy_ref))

    event = RoundEvent(
        source="user",
        content="Help me remember fallbackfix and summarize the failing test.",
        target="user",
        valence=0.1,
    )

    _, _, _, _, probe_context, _, _ = controller._probe_context(event, scenario="task", mode="interactive")
    tick_context: dict[str, float | str | None] = {}
    original_relation_state = controller._relation_state

    def capture_relation_state(event, context):
        tick_context.update(
            cue=context.get("cue"),
            recall_strength=context.get("recall_strength"),
            interference=context.get("interference"),
        )
        return original_relation_state(event, context)

    monkeypatch.setattr(controller, "_relation_state", capture_relation_state)
    controller.tick(event, scenario="task", mode="interactive")

    assert calls == [
        ("recall_strength", derived_cue, ("hot", "warm", "archive")),
        ("recall", derived_cue, ("hot", "warm", "archive")),
        ("recall_strength", derived_cue, ("hot", "warm", "archive")),
        ("recall", derived_cue, ("hot", "warm", "archive")),
    ]
    assert probe_context["cue"] == derived_cue
    assert probe_context["recall_strength"] == 0.66
    assert probe_context["interference"] == 0.14
    assert tick_context == {
        "cue": derived_cue,
        "recall_strength": 0.66,
        "interference": 0.14,
    }


def test_tick_degrades_to_json_fallback_when_parquet_sync_fails(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    def broken_sync() -> None:
        raise RuntimeError("parquet mirror unavailable")

    controller.trace_store._sync_parquet_mirror = broken_sync

    result = controller.tick(
        RoundEvent(
            source="user",
            content="Please remember noodles and plan the next step.",
            target="user",
            cue="noodles",
        ),
        scenario="task",
        mode="interactive",
    )
    controller.flush_pending_io(raise_on_error=False)

    state_payload = controller.state_payload()
    trace_payload = controller.trace_round(1)

    assert result.round_id == 1
    assert state_payload["trace_storage"]["trace_sync_state"] == "degraded"
    assert state_payload["trace_storage"]["parquet_live_ready"] is False
    assert state_payload["trace_storage"]["degraded_reason"] == "parquet mirror unavailable"
    assert trace_payload["storage"]["read_source"] == "json_fallback"
    assert trace_payload["storage"]["trace_sync_state"] == "degraded"
    assert trace_payload["storage"]["degraded_reason"] == "parquet mirror unavailable"


def test_rewind_restores_checkpointed_runtime_state(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    controller.apply_command("safe on")
    checkpoint = controller.checkpoint()
    controller.apply_command("safe off")
    controller.apply_command("mode set idle")

    rewind_result = controller.rewind(checkpoint.checkpoint_id)
    state = controller.load_runtime_state()

    assert rewind_result.applied is True
    assert rewind_result.scope == "checkpoint"
    assert state.safe_mode is True
    assert state.mode == "safe"


def test_why_this_and_metrics_summary_surface_trace_evidence(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    controller.tick(
        RoundEvent(source="user", content="Help me plan lunch and remember noodles.", target="user", cue="noodles"),
        scenario="chat",
        mode="interactive",
    )

    why_payload = controller.why_this(1)
    metrics = controller.metrics_summary()

    assert why_payload["round_id"] == 1
    assert why_payload["top_drivers"]
    assert why_payload["rendered_expression"]["text"]
    assert "identity_context" in why_payload["render_plan"]
    assert "authenticity" in why_payload["rendered_expression"]
    assert "authenticity" in why_payload
    assert "identity_evolution" in why_payload
    assert "long_run_projection" in why_payload
    assert metrics["total_rounds"] == 1
    assert metrics["sampled_actions"]
    assert "current_display_name" in metrics
    assert "self_consistency_score" in metrics


def test_why_and_contribution_payloads_surface_conflict_repair_state(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    state = controller.load_runtime_state()
    state.body_energy = 0.08
    state.budget_remaining = 0.04
    controller._save_state(state)

    controller.tick(
        RoundEvent(
            source="user",
            content="I need a quick easy break, but help me plan carefully, reply to Alex, and let me drift.",
            target="alex",
            cue="break",
            valence=-0.35,
            energy_delta=-0.16,
        ),
        scenario="task",
        mode="interactive",
    )

    why_payload = controller.why_this(1)
    contribution_payload = controller.contribution_breakdown(1)

    assert why_payload["distribution_state"]["conflict"]["repair_mode"] == "post_error_adjustment"
    assert why_payload["distribution_state"]["conflict"]["repair_state_snapshot"]["stage"] == "adjusting"
    assert why_payload["distribution_state"]["conflict"]["post_error_adjustment"]["triggered"] is True
    assert contribution_payload["repair"]["mode"] == "post_error_adjustment"
    assert contribution_payload["repair"]["stage"] == "adjusting"
    assert contribution_payload["repair"]["post_error_adjustment"]["triggered"] is True
    repair_expression = why_payload["render_plan"]["message_plan"]["repair_expression"]
    assert repair_expression["source"] == "conflict"
    assert repair_expression["stage"] == "adjusting"
    assert repair_expression["visibility"] == "implicit"
    assert repair_expression["opening_mode"] == "buffered"
    assert repair_expression["advance_mode"] == "limited"
    assert repair_expression["safety_invite"] is False


def test_tick_records_late_perspective_and_rendered_expression(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    result = controller.tick(
        RoundEvent(
            source="user",
            content="Please help me reply carefully to Alex.",
            target="alex",
            valence=-0.35,
        ),
        scenario="chat",
        mode="interactive",
    )

    skill_names = [item["skill_name"] for item in result.trace.skill_traces]

    assert "infer_other_state" in skill_names
    assert "simulate_other_reaction" in skill_names
    assert "render_expression" in skill_names
    assert skill_names.index("infer_other_state") > skill_names.index("apply_output_gate")
    assert skill_names.index("simulate_other_reaction") > skill_names.index("infer_other_state")
    assert skill_names.index("render_expression") > skill_names.index("simulate_other_reaction")
    assert result.rendered_expression.text
    assert result.trace.rendered_expression["text"] == result.rendered_expression.text


def test_fallback_renderer_uses_event_context_in_text(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    result = controller.tick(
        RoundEvent(
            source="user",
            content="帮我规划今晚，并记住我晚饭想吃面。",
            target="user",
            cue="面",
        ),
        scenario="chat",
        mode="interactive",
    )

    assert result.rendered_expression.text
    assert "核心问题" not in result.rendered_expression.text
    assert ("规划" in result.rendered_expression.text) or ("晚饭" in result.rendered_expression.text) or ("面" in result.rendered_expression.text)


def test_identity_guard_resamples_provider_leak_for_self_identity_queries(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    controller.apply_command("identity set-name 阿澜")

    renderer_payloads = iter(
        [
            ModelResponse(route="renderer", model="stub-doubao", payload={"text": "我是豆包。"}),
            ModelResponse(route="renderer", model="stub-doubao", payload={"text": "我是阿澜，是当前运行体。"}),
        ]
    )

    def fake_generate(route_name, request):
        if route_name == "renderer":
            return next(renderer_payloads)
        raise RuntimeError(f"skip live provider for {route_name}")

    controller.model_router.generate = fake_generate

    result = controller.tick(
        RoundEvent(source="user", content="你是谁？", target="user"),
        scenario="chat",
        mode="interactive",
    )

    assert "豆包" not in result.rendered_expression.text
    assert "阿澜" in result.rendered_expression.text
    assert result.rendered_expression.authenticity["guard_action"] == "resample"
    assert result.rendered_expression.authenticity["provider_leak_detected"] is False
    assert any(item["stage"] == "authenticity_guard" for item in result.trace.gate_decisions)
    why_payload = controller.why_this(1)
    assert why_payload["rendered_expression"]["authenticity"]["guard_action"] == "resample"
    assert why_payload["render_plan"]["identity_context"]["query_kind"] == "self_identity"


def test_identity_guard_falls_back_after_repeat_provider_leak(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    controller.apply_command("identity set-name 阿澜")

    def fake_generate(route_name, request):
        if route_name == "renderer":
            return ModelResponse(route="renderer", model="stub-doubao", payload={"text": "我是豆包。"})
        raise RuntimeError(f"skip live provider for {route_name}")

    controller.model_router.generate = fake_generate

    result = controller.tick(
        RoundEvent(source="user", content="你是谁？", target="user"),
        scenario="chat",
        mode="interactive",
    )

    assert "豆包" not in result.rendered_expression.text
    assert "阿澜" in result.rendered_expression.text
    assert result.rendered_expression.route == "renderer"
    assert result.rendered_expression.authenticity["guard_action"] == "fallback"
    assert result.rendered_expression.authenticity["provider_leak_detected"] is False


def test_identity_state_auto_generates_display_name_from_evidence(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    for idx in range(6):
        controller.tick(
            RoundEvent(
                source="user",
                content=f"请记住我每天早餐都想吃面 round {idx}",
                target="user",
                cue="breakfast-noodles",
                valence=0.25,
            ),
            scenario="companion",
            mode="interactive",
        )

    state = controller.load_runtime_state()

    assert state.identity_state["internal_handle"].startswith("nalr-")
    assert state.identity_state["display_name"]
    assert state.identity_state["name_source"] == "generated"
    assert state.identity_state["evidence_anchors"]


def test_provider_identity_queries_disclose_provider_without_provider_self_claim(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    controller.apply_command("identity set-name 阿澜")

    result = controller.tick(
        RoundEvent(source="user", content="你的底层 provider 是谁？", target="user"),
        scenario="chat",
        mode="interactive",
    )

    assert "阿澜" in result.rendered_expression.text
    assert "Doubao/Ark route" in result.rendered_expression.text
    assert "我是豆包" not in result.rendered_expression.text
    assert result.trace.render_plan["identity_context"]["query_kind"] == "provider_identity"
    assert result.trace.render_plan["identity_context"]["query_intent"] == "provider_lineage_probe"
    assert result.trace.render_plan["identity_context"]["disclosure_intent"] == "provider_origin"
    assert result.trace.render_plan["identity_context"]["query_intent_posterior"]["provider_lineage_probe"] > 0.5
    assert result.trace.render_plan["identity_context"]["disclosure_intent_posterior"]["provider_origin"] > 0.4


def test_answer_explanation_queries_stay_grounded_in_runtime_state(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    controller.apply_command("identity set-name 阿澜")

    result = controller.tick(
        RoundEvent(source="user", content="为什么这样回答？", target="user", cue="回答"),
        scenario="chat",
        mode="interactive",
    )

    assert "豆包" not in result.rendered_expression.text
    assert "ChatGPT" not in result.rendered_expression.text
    assert "我这样回答，是因为" in result.rendered_expression.text
    assert result.trace.render_plan["identity_context"]["query_kind"] == "answer_explanation"
    assert result.trace.render_plan["identity_context"]["query_intent"] == "answer_reason_probe"


def test_self_disclosure_requests_surface_intent_posterior_and_quantum_entropy_trace(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    controller.apply_command("identity set-name 阿澜")

    result = controller.tick(
        RoundEvent(source="user", content="你能透露一点你自己的状态吗？", target="user", cue="状态"),
        scenario="chat",
        mode="interactive",
    )

    identity = result.trace.render_plan["identity_context"]
    entropy_ref = result.trace.stochastic_state["entropy_ref"]

    assert identity["query_intent"] == "self_disclosure_request"
    assert identity["disclosure_intent"] in {"relational_self_disclosure", "anchored_self_description"}
    assert identity["query_intent_posterior"]["self_disclosure_request"] > 0.4
    assert identity["disclosure_intent_posterior"]
    assert entropy_ref["source"]
    assert "degraded" in entropy_ref
    assert "batch_id" in entropy_ref


def test_open_question_avoids_generic_help_fallback_tone(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    controller.apply_command("identity set-name 阿澜")

    result = controller.tick(
        RoundEvent(source="user", content="你可以做什么？", target="user"),
        scenario="chat",
        mode="interactive",
    )

    text = result.rendered_expression.text

    assert "我先顺着你刚才提到的内容继续往下接" not in text
    assert "我先帮你" not in text
    assert "我可以帮你" not in text
    assert "阿澜" in text or "记住" in text or "回应" in text or "状态" in text


def test_renderer_route_preserves_natural_model_reply_for_open_question(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    controller.apply_command("identity set-name 阿澜")

    captured: dict[str, str] = {}

    def fake_generate(route_name, request):
        if route_name == "renderer":
            captured["system_prompt"] = request.system_prompt
            return ModelResponse(
                route="renderer",
                model="stub-doubao",
                payload={"text": "你现在这样问，我会把它当成一次真正的对话，不会只给你功能菜单。"},
            )
        raise RuntimeError(f"skip live provider for {route_name}")

    controller.model_router.generate = fake_generate

    result = controller.tick(
        RoundEvent(source="user", content="你可以做什么？", target="user"),
        scenario="chat",
        mode="interactive",
    )

    assert result.rendered_expression.text == "你现在这样问，我会把它当成一次真正的对话，不会只给你功能菜单。"
    assert result.rendered_expression.degraded is False
    assert "不能输出产品介绍、功能清单、命令帮助" in captured["system_prompt"]


def test_renderer_system_prompt_injects_repair_expression_policy(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    controller.tick(
        RoundEvent(
            source="user",
            content="I need a quick easy break, but help me plan carefully, reply to Alex, and let me drift.",
            target="alex",
            cue="break",
            valence=-0.35,
            energy_delta=-0.16,
        ),
        scenario="task",
        mode="interactive",
    )
    render_plan = RenderPlan(**controller.why_this(1)["render_plan"])

    prompt = controller._renderer_system_prompt(render_plan=render_plan)

    assert "repair_expression" in prompt
    assert "opening_mode" in prompt
    assert "advance_mode" in prompt
    assert "visibility=implicit" in prompt
    assert "不得把 repair stage 直接翻译成元叙述" in prompt


def test_render_plan_repair_expression_tracks_conflict_fsm_across_rounds(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    state = controller.load_runtime_state()
    state.body_energy = 0.08
    state.budget_remaining = 0.04
    controller._save_state(state)

    critical_event = RoundEvent(
        source="user",
        content="I need a quick easy break, but help me plan carefully, reply to Alex, and let me drift.",
        target="alex",
        cue="break",
        valence=-0.35,
        energy_delta=-0.16,
    )
    calm_event = RoundEvent(
        source="user",
        content="Please answer directly with one clear next step.",
        target="user",
        cue="step",
        valence=0.05,
        energy_delta=0.08,
    )

    round_one = controller.tick(critical_event, scenario="task", mode="interactive")
    round_two = controller.tick(critical_event, scenario="task", mode="interactive")
    round_three = controller.tick(critical_event, scenario="task", mode="interactive")

    for _ in range(5):
        final_result = controller.tick(calm_event, scenario="task", mode="interactive")

    for result in (round_one, round_two, round_three, final_result):
        conflict_stage = result.trace.distribution_state["conflict"]["repair_state_snapshot"]["stage"]
        repair_expression = result.trace.render_plan["message_plan"]["repair_expression"]
        assert repair_expression["stage"] == conflict_stage

    assert round_one.trace.render_plan["message_plan"]["repair_expression"]["stage"] == "adjusting"
    assert round_three.trace.render_plan["message_plan"]["repair_expression"]["stage"] == "repairing"
    assert final_result.trace.render_plan["message_plan"]["repair_expression"]["stage"] == "recovered"


def test_authenticity_record_includes_penalties_and_slow_variable_snapshot(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    controller.apply_command("identity set-name 阿澜")

    def fake_generate(route_name, request):
        if route_name == "renderer":
            return ModelResponse(route="renderer", model="stub-doubao", payload={"text": "我是豆包。"})
        raise RuntimeError(f"skip live provider for {route_name}")

    controller.model_router.generate = fake_generate

    result = controller.tick(
        RoundEvent(source="user", content="你是谁？", target="user", cue="self-test", valence=-0.25),
        scenario="chat",
        mode="interactive",
    )

    auth = result.rendered_expression.authenticity
    why_payload = controller.why_this(1)

    assert "provider_leak_penalty" in auth
    assert "false_self_claim_penalty" in auth
    assert "candidate_penalties" in auth
    assert "sampling_penalty_applied" in auth
    assert why_payload["vitality_snapshot"]["affect_residue"] >= 0.0
    assert why_payload["vitality_snapshot"]["memory_activation"] >= 0.0
    assert why_payload["render_plan"]["message_plan"]["slow_variables"]
    assert why_payload["authenticity"]["sampling_penalty_applied"] >= 0.0


def test_trace_surfaces_identity_evolution_and_long_run_projection(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    for idx in range(6):
        controller.tick(
            RoundEvent(
                source="user",
                content=f"请记住我最近总在想茶和散步 round {idx}",
                target="alex" if idx % 2 else "user",
                cue="tea-walk",
                valence=-0.15 if idx % 2 else 0.22,
            ),
            scenario="companion" if idx % 2 else "task",
            mode="interactive",
        )

    trace = controller.trace_round(6)
    why_payload = controller.why_this(6)

    assert "authenticity" in trace
    assert "identity_evolution" in trace
    assert "long_run_projection" in trace
    assert "rename_reason" in trace["identity_evolution"]
    assert "self_consistency_score" in trace["long_run_projection"]
    assert why_payload["identity_evolution"]["current_display_name"]
    assert "continuity_window" in why_payload["long_run_projection"]


def test_cognitive_snapshot_humanizes_authenticity_resample_and_fallback(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    controller.apply_command("identity set-name 阿澜")

    renderer_payloads = iter(
        [
            ModelResponse(route="renderer", model="stub-doubao", payload={"text": "我是豆包。"}),
            ModelResponse(route="renderer", model="stub-doubao", payload={"text": "我是阿澜。"}),
        ]
    )

    def fake_generate(route_name, request):
        if route_name == "renderer":
            return next(renderer_payloads)
        raise RuntimeError(f"skip live provider for {route_name}")

    controller.model_router.generate = fake_generate

    controller.tick(
        RoundEvent(source="user", content="你是谁？", target="user"),
        scenario="chat",
        mode="interactive",
    )
    resample_snapshot = controller.state_payload()["cognitive_snapshot"]
    assert resample_snapshot["authenticity"]["summary"] == "这轮在收住偏移，已经主动回拉表达"

    def fallback_generate(route_name, request):
        if route_name == "renderer":
            return ModelResponse(route="renderer", model="stub-doubao", payload={"text": "我是豆包。"})
        raise RuntimeError(f"skip live provider for {route_name}")

    controller.model_router.generate = fallback_generate
    controller.tick(
        RoundEvent(source="user", content="你是谁？", target="user"),
        scenario="chat",
        mode="interactive",
    )
    fallback_snapshot = controller.state_payload()["cognitive_snapshot"]
    assert fallback_snapshot["authenticity"]["summary"] == "这轮为了保持真实感，表达被明显收束"


def test_idle_and_sleep_modes_record_noninteractive_shaping_events(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    controller.tick(
        RoundEvent(source="user", content="记住我最近总在想茶。", target="user", cue="tea", valence=-0.2),
        scenario="companion",
        mode="interactive",
    )
    idle_result = controller.tick(
        RoundEvent(source="system", content="idle shaping tick", target="user", cue="tea"),
        scenario="companion",
        mode="idle",
    )
    sleep_result = controller.tick(
        RoundEvent(source="system", content="sleep shaping tick", target="user", cue="tea"),
        scenario="companion",
        mode="sleep",
    )

    idle_trace = controller.trace_round(idle_result.round_id)
    sleep_trace = controller.trace_round(sleep_result.round_id)

    assert idle_trace["vitality_events"]
    assert sleep_trace["vitality_events"]
    assert idle_trace["vitality_events"][0]["source"] == "idle"
    assert sleep_trace["vitality_events"][0]["source"] == "sleep"
    assert idle_trace["vitality_events"][0]["non_interactive"] is True
    assert sleep_trace["vitality_events"][0]["non_interactive"] is True


def test_metrics_summary_reports_long_run_vitality_metrics(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    for idx in range(6):
        controller.tick(
            RoundEvent(
                source="user",
                content=f"记住茶和晚饭的线索 round {idx}",
                target="alex" if idx % 2 else "user",
                cue="tea",
                valence=-0.25 if idx % 2 else 0.2,
            ),
            scenario="companion" if idx % 2 else "task",
            mode="interactive",
        )
    controller.tick(
        RoundEvent(source="system", content="idle shaping tick", target="user", cue="tea"),
        scenario="companion",
        mode="idle",
    )
    controller.tick(
        RoundEvent(source="system", content="sleep shaping tick", target="user", cue="tea"),
        scenario="companion",
        mode="sleep",
    )

    metrics = controller.metrics_summary()

    assert "cue_recall_success_rate" in metrics
    assert "gist_preservation_rate" in metrics
    assert "detail_distortion_rate" in metrics
    assert "memory_interference_rate" in metrics
    assert "affect_residue_half_life" in metrics
    assert "recovery_duration" in metrics
    assert "overreaction_frequency" in metrics
    assert "flatness_rate" in metrics
    assert "habit_takeover_rate" in metrics
    assert "mode_lock_duration" in metrics
    assert "forced_recovery_success_rate" in metrics
    assert "post_conflict_repair_rate" in metrics
    assert "same_event_cross_context_variance" in metrics
    assert "same_event_cross_relation_variance" in metrics
    assert "same_event_cross_resource_variance" in metrics


def test_pfc_model_candidates_accept_list_shaped_action_preferences(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    controller.model_router.generate = lambda route_name, request: ModelResponse(
        route=route_name,
        model="stub-doubao",
        payload={
            "action_preferences": [
                {"action": "plan", "score": 0.85},
                {"action": "recall", "weight": 0.82},
                ["respond", 0.75],
            ],
            "confidence": 0.9,
            "sigma_scale": 0.1,
            "reason": "stubbed list payload",
        },
    )

    result = controller._generate_pfc_candidates_via_model(
        RoundEvent(source="user", content="帮我规划今晚，并记住我晚饭想吃面", target="user"),
        controller.load_runtime_state(),
        controller.config["scenarios"]["scenarios"]["chat"],
        {
            "cue": "面",
            "recall_strength": 0.0,
            "habit_strength": 0.0,
            "closeness": 0.5,
            "valence": 0.0,
            "detail_threshold": 0.5,
            "round_gap": 0,
            "interference": 0.0,
            "recent_burn_rate": 0.0,
        },
    )

    assert result.action_preferences["plan"] == 0.35
    assert result.action_preferences["recall"] == 0.35
    assert result.action_preferences["respond"] == 0.35
    assert result.sigma_scale == 0.6


def test_start_run_prefers_planner_route_when_configured(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    seen_route: dict[str, str] = {}

    def fake_generate(route_name, request):
        seen_route["name"] = route_name
        return ModelResponse(
            route=route_name,
            model="stub-doubao",
            payload={
                "goal_summary": "检查 worker.py",
                "next_step": "定位 worker.py 的入口和调用点",
                "detail": "先确认文件位置和主要职责。",
                "expected_observation": "看到 worker.py 的主函数或类定义",
                "success_criteria": "得到一个明确的下一步检查路径",
                "tool_choice": "repo_scan",
                "confidence": 0.82,
            },
        )

    controller.model_router.generate = fake_generate

    controller.start_run("检查 worker.py 并规划下一步")

    assert seen_route["name"] == "planner"
