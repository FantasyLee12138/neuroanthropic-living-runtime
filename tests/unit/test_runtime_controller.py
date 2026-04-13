import json
import copy
import shutil
import time
from pathlib import Path

import duckdb
import pytest

from nalr.agents.modules import ValueAgent
from nalr.providers.router import ModelResponse
from nalr.runtime.controller import RuntimeController
from nalr.runtime.entropy import QuantumEntropyUnavailableError
from nalr.runtime.metadata import utc_now_iso
from nalr.schemas.models import (
    ActionCandidate,
    EndogenousTickTrigger,
    ProbabilityFieldSnapshot,
    ProbabilityLayerState,
    ProbabilisticContribution,
    QuantumEntropyRef,
    RenderPlan,
    RoundEvent,
    StochasticState,
    TokenFieldState,
    to_dict,
)
from nalr.trace.store import ROUND_CANONICAL_SCHEMA
from nalr.skills.registry import build_skill_registry
from nalr.trace.store import TraceStore


CONFIG_ROOT = Path(__file__).resolve().parents[2] / "config"


def test_tick_blocks_when_entropy_provider_unavailable_even_if_hard_block_disabled(tmp_path, monkeypatch):
    config_root = tmp_path / "config"
    shutil.copytree(CONFIG_ROOT, config_root)
    (config_root / "entropy.yaml").write_text(
        """entropy:
  provider: macos_os_urandom
  prefetch_bytes: 96
  hard_block_on_unavailable: false
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("NALR_DISABLE_TEST_QRNG_SEED", "1")

    controller = RuntimeController(project_root=tmp_path, config_root=config_root)

    def broken_fetch_batch(*, byte_count: int):
        raise RuntimeError(f"provider unavailable for {byte_count} bytes")

    monkeypatch.setattr(controller.entropy_pool.provider, "fetch_batch", broken_fetch_batch)

    with pytest.raises(QuantumEntropyUnavailableError):
        controller.tick(
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
    assert controller.entropy_pool.provider.endpoint == "file:///dev/urandom"
    state = controller.load_runtime_state()
    assert state.entropy_health_state["state"] == "blocked"
    assert state.entropy_health_state["last_failure"]["failure_class"] == "os_urandom_failed"
    assert state.last_entropy_failure["failure_class"] == "os_urandom_failed"


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


def test_runtime_controller_exposes_split_mid_round_helpers(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    round_context = controller.build_round_context(
        RoundEvent(
            source="user",
            content="帮我决定是继续写还是先休息。",
            target="user",
            cue="继续",
            valence=0.1,
        ),
        "companion",
        "interactive",
    )

    collected = controller.collect_parallel_contributions(round_context=round_context, scenario="companion")

    assert "action_signals" in collected
    assert "direct_action_contributions" in collected
    assert "identity_context" in collected

    arbitrate_payload = controller.integrate_and_arbitrate(
        round_context=round_context,
        collected=collected,
        scenario="companion",
        turn_started=time.perf_counter(),
    )

    assert arbitrate_payload["sampled_action"].name
    assert arbitrate_payload["rendered_expression"].text
    assert arbitrate_payload["trace"].round_id == round_context["state"].round_count


def test_tick_uses_macos_system_entropy_when_pytest_seed_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("NALR_DISABLE_TEST_QRNG_SEED", "1")
    monkeypatch.setattr("os.urandom", lambda byte_count: bytes([128]) * byte_count)
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

    assert result.trace.stochastic_state["entropy_ref"]["source"] == "macos_os_urandom"
    assert result.trace.stochastic_state["entropy_ref"]["endpoint"] == "file:///dev/urandom"
    refs = result.trace.stochastic_state["entropy_refs_by_node"]
    assert refs
    assert all(ref["source"] == "macos_os_urandom" for ref in refs.values())
    assert all(ref["failure_class"] == "" for ref in refs.values())


def test_die_action_labels_surface_life_ending_semantics(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    assert controller._action_phrase("die") == "准备自主结束生命"
    assert controller._focus_label("die") == "正在转向自主结束生命"


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


def test_model_status_surfaces_agent_tiers_and_bindings(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    status = controller.model_status()

    assert status["tiers"]["small_model"]["mode"] == "remote"
    assert status["tiers"]["medium_model"]["backend"] == "deepseek"
    assert status["tiers"]["medium_model"]["api_key_env"] == "DEEPSEEK_API_KEY"
    assert status["tiers"]["small_model"]["api_key_env"] == "ARK_SMALL_MODEL_API_KEY"
    assert status["agent_bindings"]["SalienceAgent"] == "small_model"
    assert status["agent_bindings"]["ValueAgent"] == "small_model"
    assert status["agent_bindings"]["planner"] == "medium_model"
    assert status["agent_bindings"]["PerspectiveModel"] == "medium_model"
    assert status["agent_bindings"]["PFCAgent"] == "medium_model"
    assert set(status["route_policies"]) == {
        "chat_fast",
        "chat_standard",
        "chat_deep",
        "endogenous_light",
        "endogenous_deep",
        "dream_sleep",
    }
    assert status["route_policies"]["chat_fast"]["latency_budget_ms"] == 700
    assert status["route_policies"]["chat_fast"]["default_tier"] in {"medium_model", "small_model"}
    assert status["route_policies"]["chat_standard"]["hot_path"] == "full_tick"
    assert "authenticity_risk_high" in status["route_policies"]["chat_standard"]["upgrade_conditions"]
    assert status["route_policies"]["endogenous_light"]["entry_mode"] == "idle"
    assert status["route_policies"]["dream_sleep"]["entry_mode"] == "sleep"


def test_plan_turn_keeps_task_run_internal_but_uses_frozen_public_route_type(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    plan = controller.plan_turn("总结这个仓库结构")

    assert plan.route == "task_run"
    assert plan.route_type == "chat_deep"


def test_value_agent_build_value_contribution_matches_controller_compatibility_path(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    value_scores = {"scores": {"plan": 0.24, "respond": 0.09}}

    agent_contribution = ValueAgent().build_value_contribution(value_scores)
    controller_contribution = controller._build_value_action_contribution(value_scores)

    assert agent_contribution == controller_contribution
    assert agent_contribution.raw_signal == {"plan": 0.24, "respond": 0.09}
    assert agent_contribution.modulated_delta == {"plan": 0.24, "respond": 0.09}
    assert agent_contribution.inhibitory_drive == {}
    assert agent_contribution.projection_reason == "value bias projected from value head"
    assert agent_contribution.applied_at_stage == "subjective_value"


def test_value_skill_registry_drops_proposal_tag_for_subjective_value():
    registry = build_skill_registry()
    spec = registry["estimate_subjective_value"]

    assert spec.trace_tags == ["value"]
    assert spec.output_kind == "score_map"


def test_medium_model_route_resolution_for_planner_and_perspective(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    planner_route = controller._route_config_for_binding("planner", route_name="planner")
    perspective_route = controller._route_config_for_binding("PerspectiveModel", route_name="perspective")
    pfc_route = controller._route_config_for_binding("PFCAgent", route_name="pfc")

    assert planner_route is not None
    assert planner_route.backend == "deepseek"
    assert planner_route.model == "deepseek-chat"
    assert planner_route.api_key_env == "DEEPSEEK_API_KEY"
    assert perspective_route is not None
    assert perspective_route.backend == "deepseek"
    assert perspective_route.model == "deepseek-chat"
    assert pfc_route is not None
    assert pfc_route.backend == "deepseek"
    assert pfc_route.model == "deepseek-chat"


def test_pfc_route_escalates_to_large_model_when_relation_risk_is_high(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    route = controller._route_config_for_binding(
        "PFCAgent",
        route_name="pfc",
        metadata={"relation_risk": 0.82},
    )

    assert route is not None
    assert route.backend == "doubao"
    assert route.model == "doubao-seed-2-0-pro-260215"
    assert getattr(route, "effective_tier", "") == "large_model"


def test_execute_parallel_skills_runs_independent_tasks_concurrently(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    skill_traces: list[dict[str, object]] = []
    parallel_traces: list[dict[str, object]] = []
    runtime_context = controller._skill_runtime_context(1, "chat", controller.load_runtime_state())
    started = time.perf_counter()

    outputs = controller._execute_parallel_skills(
        round_id=1,
        tasks=[
            {
                "name": "guard_a",
                "skill_name": "request_second_sampling",
                "inputs": {"fail_score": 0.4, "attempts": 0},
                "provider": lambda fail_score, attempts: (time.sleep(0.15), {"flag": fail_score > 0.2})[1],
                "parallel_group": "test_parallel",
            },
            {
                "name": "guard_b",
                "skill_name": "trigger_forced_focus_switch",
                "inputs": {"lock_score": 0.6},
                "provider": lambda lock_score: (time.sleep(0.15), {"switch_flag": lock_score > 0.5})[1],
                "parallel_group": "test_parallel",
            },
        ],
        skill_traces=skill_traces,
        runtime_context=runtime_context,
        parallel_traces=parallel_traces,
    )
    elapsed = time.perf_counter() - started

    assert outputs["guard_a"]["flag"] is True
    assert outputs["guard_b"]["switch_flag"] is True
    assert elapsed < 0.28
    assert all(item["parallel_group"] == "test_parallel" for item in skill_traces)
    assert any(item["task_outcome"] == "completed" for item in parallel_traces)


def test_execute_parallel_skills_times_out_optional_work_without_blocking(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    skill_traces: list[dict[str, object]] = []
    parallel_traces: list[dict[str, object]] = []
    runtime_context = controller._skill_runtime_context(1, "chat", controller.load_runtime_state())
    started = time.perf_counter()

    outputs = controller._execute_parallel_skills(
        round_id=1,
        tasks=[
            {
                "name": "slow_guard",
                "skill_name": "request_second_sampling",
                "inputs": {"fail_score": 0.4, "attempts": 0},
                "provider": lambda fail_score, attempts: (time.sleep(0.2), {"flag": fail_score > 0.2})[1],
                "fallback_value": {"flag": False},
                "parallel_group": "test_parallel_timeout",
                "priority": "optional",
                "timeout_ms": 40,
            },
            {
                "name": "grounding_capsule",
                "inputs": {},
                "provider": lambda: {"state_sources": ["focus"]},
                "parallel_group": "test_parallel_timeout",
                "priority": "speculative",
                "timeout_ms": 80,
                "task_type": "callable",
            },
        ],
        skill_traces=skill_traces,
        runtime_context=runtime_context,
        parallel_traces=parallel_traces,
    )
    elapsed = time.perf_counter() - started

    assert outputs["slow_guard"]["flag"] is False
    assert outputs["grounding_capsule"]["state_sources"] == ["focus"]
    assert elapsed < 0.16
    assert any(item["task_name"] == "slow_guard" and item["task_outcome"] == "timeout" for item in parallel_traces)
    assert any(item["task_name"] == "grounding_capsule" and item["task_type"] == "callable" for item in parallel_traces)


def test_small_model_salience_provider_uses_agent_tier_config(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    calls: list[tuple[str, str, str]] = []

    def fake_generate_config(route_config, request):
        calls.append((route_config.name, route_config.backend, route_config.api_key_env))
        return ModelResponse(
            route=route_config.name,
            model=route_config.model,
            payload={
                "action_preferences": {"clarify": 0.14},
                "confidence": 0.77,
                "sigma_scale": 0.91,
                "reason": "small-model salience",
            },
            raw_text='{"action_preferences":{"clarify":0.14},"confidence":0.77,"sigma_scale":0.91,"reason":"small-model salience"}',
            usage={"prompt_tokens": 18, "completion_tokens": 7},
            latency_ms=42,
            backend=route_config.backend,
        )

    monkeypatch.setattr(controller.model_router, "generate_config", fake_generate_config)

    contribution = controller._score_salience_via_model(
        RoundEvent(source="user", content="你现在有点犹豫吗？", target="user", valence=0.42),
        controller.load_runtime_state(),
        {"name": "chat", "pfc_base_share": 0.2},
        {"closeness": 0.5},
    )

    assert contribution.modulated_delta["clarify"] == 0.14
    assert contribution.confidence == 0.77
    assert contribution.projection.module_temperature == 0.91
    assert contribution.trace_reason == "small-model salience"
    assert calls == [("salience_small_model", "doubao", "ARK_SMALL_MODEL_API_KEY")]


def test_small_model_value_provider_falls_back_to_local_scores_on_failure(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    monkeypatch.setattr(controller.model_router, "generate_config", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))

    scores = controller._estimate_subjective_value_via_model(
        RoundEvent(source="user", content="请总结这个仓库结构", target="user"),
        controller.load_runtime_state(),
        {"name": "task", "pfc_base_share": 0.2},
        {"closeness": 0.4},
    )

    assert scores["scores"]["plan"] > 0.0


def test_compact_model_payload_reduces_full_state_context_surface(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    state = controller.load_runtime_state()
    event = RoundEvent(source="user", content="请解释一下你为什么这样回答", target="user", valence=0.15)
    scenario = {"name": "chat", "pfc_base_share": 0.2}
    context = {
        "cue": "tea",
        "closeness": 0.62,
        "recall_strength": 0.44,
        "burn_rate_ratio": 0.12,
        "queue_pressure": 0.08,
        "latency_pressure": 0.06,
    }

    payload = controller._build_pfc_model_payload(event, state, scenario, context)

    assert "state" not in payload
    assert "context" not in payload
    assert payload["state_summary"]["mode"] == state.mode
    assert payload["context_summary"]["cue"] == "tea"
    assert round(payload["context_summary"]["closeness"], 2) == 0.62


def test_tick_records_model_call_metrics_for_small_model_parallel_group(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    def fake_generate_config(route_config, request):
        if route_config.name == "salience_small_model":
            payload = {
                "action_preferences": {"respond": 0.11},
                "confidence": 0.7,
                "sigma_scale": 0.9,
                "reason": "salience-small",
            }
        else:
            payload = {"scores": {"respond": 0.12}}
        return ModelResponse(
            route=route_config.name,
            model=route_config.model,
            payload=payload,
            raw_text="{}",
            usage={"prompt_tokens": 10, "completion_tokens": 3},
            latency_ms=25,
            backend=route_config.backend,
        )

    monkeypatch.setattr(controller.model_router, "generate_config", fake_generate_config)

    result = controller.tick(
        RoundEvent(source="user", content="你好，今天怎么样？", target="user"),
        scenario="chat",
        mode="interactive",
    )

    assert result.trace.runtime_metrics["model_call_count"] >= 2
    assert result.trace.runtime_metrics["parallel_task_count"] >= 4
    assert "salience_value_prefetch" in result.trace.runtime_metrics["parallel_groups"]
    assert "intent_prefetch" in result.trace.runtime_metrics["parallel_groups"]
    assert any(item["agent_tier"] == "small_model" for item in result.trace.model_call_traces)


def test_acceptance_report_surfaces_bypass_and_parallel_evidence(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    def fake_generate_config(route_config, request):
        if route_config.name == "salience_small_model":
            payload = {
                "action_preferences": {"respond": 0.11},
                "confidence": 0.7,
                "sigma_scale": 0.9,
                "reason": "salience-small",
            }
        else:
            payload = {"scores": {"respond": 0.12}}
        return ModelResponse(
            route=route_config.name,
            model=route_config.model,
            payload=payload,
            raw_text="{}",
            usage={"prompt_tokens": 10, "completion_tokens": 3},
            latency_ms=25,
            backend=route_config.backend,
        )

    monkeypatch.setattr(controller.model_router, "generate_config", fake_generate_config)

    controller.tick(
        RoundEvent(source="user", content="你好，今天怎么样？", target="user"),
        scenario="chat",
        mode="interactive",
    )

    report = controller.acceptance_report(window=1)

    assert report["window"] == 1
    assert report["probability_field_coverage"] == 1.0
    assert report["action_audit_coverage"] == 1.0
    assert report["token_audit_coverage"] == 1.0
    assert report["renderer_token_coverage"] == 1.0
    assert report["token_source_integrity_rate"] == 1.0
    assert report["bypass_detection"]["violation_count"] == 0
    assert not report["bypass_detection"]["violations"]
    assert report["bypass_detection"]["legacy_bridge_residue_count"] == 0
    assert not report["bypass_detection"]["legacy_bridge_residues"]
    assert "salience_value_prefetch" in report["parallel_evidence"]["observed_groups"]
    assert "intent_prefetch" in report["parallel_evidence"]["observed_groups"]
    assert report["parallel_evidence"]["groups_with_overlap"]
    assert report["parallel_evidence"]["true_parallel_group_rate"] > 0.0
    assert report["scale_consistency"]["rounds_evaluated"] == 1
    assert report["scale_consistency"]["clipped_contribution_rate"] >= 0.0
    assert report["scale_consistency"]["overdominant_round_rate"] >= 0.0
    assert report["cross_layer_coupling"]["illegal_count"] == 0
    assert report["cross_layer_coupling"]["compliant_round_rate"] == 1.0
    assert report["cross_layer_coupling"]["observed_pairs"]
    assert "context->memory->context_route" in report["cross_layer_coupling"]["observed_pairs"]
    assert report["cross_layer_coupling"]["pair_coverage"]["context->memory->context_route"] == 1.0
    assert report["cross_layer_coupling"]["pair_coverage"]["memory->action->memory_prior"] == 1.0
    assert report["cross_layer_coupling"]["pair_coverage"]["action->token->render_plan"] == 1.0
    assert report["conflict_arbitration"]["observed_round_rate"] == 1.0
    assert report["conflict_arbitration"]["winning_priority_coverage"] >= 0.0
    assert report["conflict_arbitration"]["peak_cluster_coverage"] >= 0.0
    assert report["conflict_arbitration"]["compromise_template_prior_coverage"] >= 0.0
    assert report["renderer_decision_integrity"]["violation_count"] == 0
    assert report["renderer_decision_integrity"]["decision_lock_rate"] == 1.0


def test_acceptance_report_and_why_this_surface_memory_write_gate(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    def fake_generate_config(route_config, request):
        if route_config.name == "salience_small_model":
            payload = {
                "action_preferences": {"respond": 0.11},
                "confidence": 0.7,
                "sigma_scale": 0.9,
                "reason": "salience-small",
            }
        else:
            payload = {"scores": {"respond": 0.12}}
        return ModelResponse(
            route=route_config.name,
            model=route_config.model,
            payload=payload,
            raw_text="{}",
            usage={"prompt_tokens": 10, "completion_tokens": 3},
            latency_ms=25,
            backend=route_config.backend,
        )

    def fake_ingest_event(event, **kwargs):
        controller.memory_store._last_ingest_diagnostics = {
            "cue": "alphb",
            "suppressed": True,
            "applied": False,
            "reason": "pollution_guard",
            "interference": 0.14,
            "cue_quality": 0.0,
            "resource_pressure": 0.88,
            "support_signal": 0.09,
            "pollution_risk": 0.31,
        }
        return "alphb"

    monkeypatch.setattr(controller.model_router, "generate_config", fake_generate_config)
    monkeypatch.setattr(controller.memory_store, "ingest_event", fake_ingest_event)

    result = controller.tick(
        RoundEvent(source="user", content="remember alphb idea for later", target="user"),
        scenario="chat",
        mode="interactive",
    )

    why_payload = controller.why_this(1)
    report = controller.acceptance_report(window=1)

    assert result.trace.memory_write_gate["suppressed"] is True
    assert why_payload["memory_write_gate"]["reason"] == "pollution_guard"
    assert report["memory_write_gate"]["suppressed_round_rate"] == 1.0
    assert report["memory_write_gate"]["samples"][0]["reason"] == "pollution_guard"


def test_why_this_and_replay_layer_surface_cognitive_chain(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    controller.tick(
        RoundEvent(
            source="user",
            content="Remember tea and explain the full reasoning chain.",
            target="user",
            cue="tea",
            valence=0.2,
        ),
        scenario="chat",
        mode="interactive",
    )

    why_payload = controller.why_this(1)
    replay_payload = controller.replay_layer(1, "memory")

    assert [item["layer"] for item in why_payload["cognitive_chain"]] == [
        "perception",
        "memory",
        "cognition",
        "decision",
        "execution",
        "feedback",
    ]
    assert "cognitive" in why_payload["layer_metrics"]
    assert "feedback" in why_payload["layer_metrics"]
    assert why_payload["feedback_loop"]["feedback_metrics"]["behavior_effectiveness"] >= 0.0
    assert replay_payload["round_id"] == 1
    assert replay_payload["layer"] == "memory"
    assert replay_payload["snapshot"]["input_vector"]
    assert replay_payload["snapshot"]["transform_summary"]


def test_control_proposal_apply_and_promote_updates_runtime_controls(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    proposal = controller.create_control_proposal(
        {
            "title": "Tighten initiative cadence",
            "target": "initiative",
            "patch": {
                "behavior_policies": {
                    "initiative": {
                        "trigger_interval_minutes": 15,
                    }
                }
            },
        }
    )

    assert proposal["status"] == "pending"

    applied = controller.apply_control_proposal(proposal["proposal_id"], approved=True)
    current = controller.controls_current()

    assert applied["status"] == "applied"
    assert current["layer_controls"]["behavior_policies"]["initiative"]["trigger_interval_minutes"] == 15

    promoted = controller.promote_control_proposal(proposal["proposal_id"])

    assert promoted["status"] == "promoted"
    runtime_yaml = tmp_path / "config" / "runtime.yaml"
    assert runtime_yaml.exists()
    assert "trigger_interval_minutes: 15" in runtime_yaml.read_text(encoding="utf-8")


def test_alert_history_and_layer_fuse_round_trip_are_audited(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    controller.tick(
        RoundEvent(
            source="user",
            content="Remember tea and reflect on the cognitive chain.",
            target="user",
            cue="tea",
            valence=0.12,
        ),
        scenario="chat",
        mode="interactive",
    )
    controller.flush_pending_io(raise_on_error=True)

    proposal = controller.create_control_proposal(
        {
            "title": "Force alert trigger",
            "target": "alerts",
            "patch": {
                "alerts": {
                    "rules": [
                        {
                            "rule_id": "effectiveness-watch",
                            "metric": "behavior_effectiveness",
                            "operator": "<",
                            "threshold": 1.1,
                            "window": 1,
                            "action": "suggest_fuse",
                        }
                    ]
                }
            },
        }
    )
    controller.apply_control_proposal(proposal["proposal_id"], approved=True)

    alert_history = controller.alert_history_payload()
    fuse_payload = controller.apply_layer_fuse(
        "cognition",
        {"mode": "degraded", "throttle": 0.35, "muted": False},
        reason="operator_overload_guard",
    )
    restored_payload = controller.restore_layer_fuse("cognition", reason="operator_restore")

    assert any(item["rule_id"] == "effectiveness-watch" for item in alert_history["history"])
    assert fuse_payload["layer_fuses"]["cognition"]["mode"] == "degraded"
    assert restored_payload["layer_fuses"]["cognition"]["mode"] == "normal"
    event_types = [item["event_type"] for item in restored_payload["recent_control_events"]]
    assert "fuse_engaged" in event_types
    assert "fuse_restored" in event_types


def test_why_this_and_acceptance_report_surface_failure_taxonomy(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    trace_payload = {
        "round_id": 1,
        "sampled_action": "respond",
        "top_drivers": [],
        "style_profile": {},
        "stochastic_state": {},
        "render_plan": {},
        "rendered_expression": {},
        "state_snapshot": {
            "mode": "interactive",
            "safe_mode": False,
            "focus": "respond",
            "budget_remaining": 0.8,
        },
        "probability_field": {
            "action": {
                "winner_target": "respond",
                "winner_posterior": {"respond": 0.74, "plan": 0.26},
                "final_energy": {"respond": 0.62, "plan": 0.14},
                "failure_taxonomy": ["entropy_collapse", "guard_overreach"],
                "contribution_audit": [
                    {"module_name": "ConflictMonitorAgent", "failure_taxonomy": ["guard_overreach"]},
                ],
                "hard_masked_targets": ["plan"],
            },
            "token": {
                "failure_taxonomy": ["token_drift"],
                "contribution_audit": [
                    {"module_name": "Renderer", "failure_taxonomy": ["token_drift"]},
                ],
            },
        },
    }

    monkeypatch.setattr(controller, "trace_round", lambda round_ref: copy.deepcopy(trace_payload))
    monkeypatch.setattr(controller.trace_store, "list_rounds", lambda: [copy.deepcopy(trace_payload)])

    why_payload = controller.why_this(1)
    report = controller.acceptance_report(window=1)

    assert why_payload["failure_taxonomy"] == ["entropy_collapse", "guard_overreach", "token_drift"]
    assert report["failure_taxonomy"]["observed_round_rate"] == 1.0
    assert report["failure_taxonomy"]["counts"]["entropy_collapse"] == 1
    assert report["failure_taxonomy"]["counts"]["guard_overreach"] == 1
    assert report["failure_taxonomy"]["counts"]["token_drift"] == 1
    assert report["failure_taxonomy"]["round_rates"]["token_drift"] == 1.0


def test_why_this_surfaces_conflict_arbitration_summary(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    controller.tick(
        RoundEvent(
            source="user",
            content="Please help me plan carefully, but I also want to wander and rest.",
            target="user",
            cue="rest",
            valence=0.05,
        ),
        scenario="chat",
        mode="interactive",
    )

    why_payload = controller.why_this(1)

    assert "winning_priority" in why_payload["conflict_arbitration"]
    assert why_payload["conflict_arbitration"]["winner_peak_posterior"]
    assert "compromise_template_prior" in why_payload["conflict_arbitration"]
    assert "peak_clusters" in why_payload["conflict_arbitration"]
    compromise_template = (
        controller.trace_round(1)["conflict_arbitration"].get("compromise", {}).get("template")
    )
    if compromise_template:
        assert why_payload["conflict_arbitration"]["compromise_template_prior"].get(compromise_template, 0.0) > 0.0


def test_why_not_prefers_probability_field_candidate_distribution(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    monkeypatch.setattr(
        controller,
        "trace_round",
        lambda round_ref: {
            "round_id": round_ref,
            "sampled_action": "respond",
            "probability_field": {
                "action": {
                    "winner_target": "respond",
                    "winner_posterior": {"respond": 0.82, "plan": 0.18},
                    "final_energy": {"respond": 0.9, "plan": 0.1},
                    "counterfactual_top_peaks": [
                        {"target": "respond", "final_energy": 0.9, "posterior": 0.82},
                        {"target": "plan", "final_energy": 0.1, "posterior": 0.18},
                    ],
                    "contribution_audit": [],
                    "hard_masked_targets": [],
                }
            },
            "top_drivers": [],
        },
    )

    payload = controller.why_not(1, "plan")

    assert payload["candidate_score"] == pytest.approx(0.18)


def test_why_this_conflict_arbitration_survives_without_distribution_state_conflict(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    trace_payload = {
        "round_id": 1,
        "sampled_action": "respond",
        "style_profile": {},
        "stochastic_state": {},
        "render_plan": {},
        "rendered_expression": {},
        "state_snapshot": {
            "mode": "interactive",
            "safe_mode": False,
            "focus": "respond",
            "budget_remaining": 0.8,
        },
        "probability_field": {
            "action": {
                "winner_target": "respond",
                "winner_posterior": {"respond": 0.72, "plan": 0.28},
                "final_energy": {"respond": 0.6, "plan": 0.1},
                "contribution_audit": [
                    {
                        "module_name": "ConflictMonitorAgent",
                        "module_type": "conflict",
                        "posterior": {"respond": 0.72, "plan": 0.28},
                        "peak_clusters": [{"actions": ["respond", "plan"], "mass": 1.0}],
                        "compromise_template_prior": {"task_first": 1.0},
                        "hard_masked_targets": ["plan"],
                        "trace_reason": "conflict arbitration from field",
                        "dependency_trace": ["winning_priority:task_goal"],
                    }
                ],
                "counterfactual_top_peaks": [],
                "hard_masked_targets": ["plan"],
            }
        },
        "gate_decisions": [{"stage": "conflict", "template": "task_first"}],
        "top_drivers": [],
    }
    monkeypatch.setattr(controller, "trace_round", lambda round_ref: copy.deepcopy(trace_payload))

    payload = controller.why_this(1)

    assert payload["conflict_arbitration"]["winning_priority"] == "task_goal"
    assert payload["conflict_arbitration"]["compromise_template_prior"] == {"task_first": 1.0}
    assert payload["conflict_arbitration"]["hard_masked_targets"] == ["plan"]


def test_trace_round_reads_canonical_payload_after_legacy_history_rewrite(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    legacy_payload = {
        "session_id": "legacy-session",
        "recorded_at": "2026-04-06T00:00:00Z",
        "recorded_date": "2026-04-06",
        "round_id": 1,
        "sampled_action": "plan",
        "style_profile": {},
        "stochastic_state": {},
        "render_plan": {},
        "rendered_expression": {},
        "state_snapshot": {
            "mode": "interactive",
            "safe_mode": False,
            "focus": "plan",
            "budget_remaining": 0.8,
        },
        "top_drivers": [],
        "distribution_state": {
            "u_shifted": {"plan": 1.2, "respond": 0.2},
            "p_final": {"plan": 0.75, "respond": 0.25},
            "p_raw": {"plan": 0.75, "respond": 0.25},
            "conflict": {"passes": []},
        },
        "gate_decisions": [],
    }
    controller.trace_store._append_dataset_rows(
        controller.trace_store.round_canonical_dir,
        [
            {
                "session_id": legacy_payload["session_id"],
                "recorded_at": legacy_payload["recorded_at"],
                "recorded_date": legacy_payload["recorded_date"],
                "round_id": legacy_payload["round_id"],
                "payload_json": json.dumps(legacy_payload, ensure_ascii=False, sort_keys=True),
            }
        ],
        schema=ROUND_CANONICAL_SCHEMA,
        partition_keys=("recorded_date", "round_id"),
    )

    rewrite_payload = controller.trace_store.rewrite_legacy_round_parquet_history(force=True)
    trace_payload = controller.trace_round(1)

    assert rewrite_payload["rewritten"] is True
    assert trace_payload["probability_field"]["action"]["winner_posterior"] == {"plan": 0.75, "respond": 0.25}
    assert trace_payload["probability_field"]["action"]["winner_target"] == "plan"
    assert trace_payload["token_state"] == {}
    assert "distribution_state" not in trace_payload


def test_resolve_round_ref_accepts_latest_alias(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    controller.tick(
        RoundEvent(
            source="user",
            content="帮我计划今晚。",
            target="user",
            cue="今晚",
        ),
        scenario="chat",
        mode="interactive",
    )

    assert controller.resolve_round_ref("latest") == 1


def test_console_recent_rounds_avoids_metrics_timeline_for_recent_rows(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    for cue in ("tea", "coffee", "walk"):
        controller.tick(
            RoundEvent(
                source="user",
                content=f"记住 {cue}",
                target="user",
                cue=cue,
            ),
            scenario="chat",
            mode="interactive",
        )

    def fail_metrics_timeline() -> dict[str, object]:
        raise AssertionError("metrics_timeline should not be called")

    monkeypatch.setattr(controller, "metrics_timeline", fail_metrics_timeline)

    rounds = controller.console_recent_rounds(limit=2)

    assert [row["round_id"] for row in rounds] == [2, 3]
    assert all("sampled_action" in row for row in rounds)


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
    assert trace_payload["token_state"]["step_index"] == 1
    assert trace_payload["cross_layer_coupling_verdict"]["legal"] is True
    assert trace_payload["cross_layer_coupling_verdict"]["observed_pairs"]
    assert "memory->action->organic_memory" in trace_payload["cross_layer_coupling_verdict"]["observed_pairs"]
    assert why_payload["storage"]["read_source"] == "parquet"
    assert why_payload["token_state"]["step_index"] == 1
    assert why_payload["cross_layer_coupling_verdict"]["legal"] is True
    assert contribution_payload["storage"]["trace_sync_state"] == "healthy"


def test_trace_round_and_why_this_flag_illegal_cross_layer_coupling(tmp_path, monkeypatch):
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

    trace_payload = copy.deepcopy(controller.trace_round(1))
    trace_payload["probability_field"]["couplings"] = [
        {
            "source_layer": "context",
            "target_layer": "token",
            "carrier_signal": "illegal_bridge",
            "projection_rule": "test",
            "allowed_phase": "tick",
            "enabled": True,
        }
    ]

    monkeypatch.setattr(controller.trace_store, "read_round_record", lambda round_id: (copy.deepcopy(trace_payload), "memory_cache"))

    round_view = controller.trace_round(1)
    why_payload = controller.why_this(1)

    assert round_view["cross_layer_coupling_verdict"]["legal"] is False
    assert round_view["cross_layer_coupling_verdict"]["illegal_pairs"] == ["context->token->illegal_bridge"]
    assert why_payload["cross_layer_coupling_verdict"]["legal"] is False


def test_acceptance_report_drops_token_source_integrity_rate_for_malformed_token_audit(tmp_path, monkeypatch):
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

    trace_payload = copy.deepcopy(controller.trace_round(1))
    trace_payload["probability_field"]["token"]["contribution_audit"] = [
        {
            "module_name": "",
            "dependency_trace": ["synthetic:bad_token_source"],
        }
    ]

    monkeypatch.setattr(controller.trace_store, "list_rounds", lambda: [copy.deepcopy(trace_payload)])

    report = controller.acceptance_report(window=1)

    assert report["token_source_integrity_rate"] == 0.0


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


def test_runtime_bootstraps_with_low_permission_autonomy_enabled(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    state = controller.load_runtime_state()
    status = controller.autonomy_status()

    assert state.autonomy_policy.enabled is True
    assert state.autonomy_policy.profile == "tool_level"
    assert state.autonomy_loop.running is True
    assert status["enabled"] is True
    assert status["running"] is True
    assert status["profile"] == "tool_level"


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

    assert probe["route"] == "fast_chat"
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


def test_probe_distribution_for_scenario_reuses_action_bookkeeping_builder(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    calls = {"count": 0}
    original = controller._build_action_bookkeeping
    integration_calls: list[set[str]] = []

    def traced_builder(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    original_integrate = controller._integrate_probability_field_snapshot

    def traced_integrate(*args, **kwargs):
        integration_calls.append(set(kwargs))
        assert "bundles" not in kwargs
        return original_integrate(*args, **kwargs)

    monkeypatch.setattr(controller, "_build_action_bookkeeping", traced_builder)
    monkeypatch.setattr(controller, "_integrate_probability_field_snapshot", traced_integrate)

    probe = controller._probe_distribution_for_scenario(
        RoundEvent(source="user", content="总结这个仓库结构", target="user"),
        scenario="task",
        mode="interactive",
    )

    assert calls["count"] >= 1
    assert integration_calls == [{"action_base", "context", "direct_action_contributions", "event", "identity_context", "long_run_contribution", "scenario_cfg", "state"}]
    assert probe["top_action"]
    assert probe["action_distribution"]


def test_probe_distribution_for_scenario_no_longer_routes_through_bundle_bridge(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    original = controller._build_action_bookkeeping
    row_type_sets: list[set[str]] = []

    def traced(rows, *args, **kwargs):
        row_type_sets.append({type(row).__name__ for row in rows})
        assert "ProposalBundle" not in row_type_sets[-1]
        return original(rows, *args, **kwargs)

    monkeypatch.setattr(controller, "_build_action_bookkeeping", traced)

    probe = controller._probe_distribution_for_scenario(
        RoundEvent(source="user", content="总结这个仓库结构", target="user"),
        scenario="task",
        mode="interactive",
    )

    assert probe["top_action"]
    assert probe["action_distribution"]
    assert row_type_sets == [{"ActionEvidenceSignal"}]


def test_plan_turn_uses_fast_chat_for_simple_identity_prompt_without_task_bootstrap(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    plan = controller.plan_turn("你好，你是谁？你有名字吗？")

    assert plan.route == "fast_chat"
    assert plan.route_type == "chat_fast"
    assert plan.task_bootstrap is None


def test_plan_turn_marks_normal_chat_as_chat_standard(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    plan = controller.plan_turn("最近的内在状态怎么样？")

    assert plan.route == "direct_chat"
    assert plan.route_type == "chat_standard"


def test_plan_turn_marks_long_reflective_chat_as_chat_deep(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    plan = controller.plan_turn("请你详细分析一下我最近总是觉得疲惫、分心、又想推进事情，但每次都卡住的状态，并系统比较一下可能的内在原因和接下来最值得优先做的调整。")

    assert plan.route == "direct_chat"
    assert plan.route_type == "chat_deep"


def test_execute_turn_fast_chat_uses_chat_fast_route_once(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    calls: list[str] = []

    def fake_generate(route_name, request):
        calls.append(route_name)
        return ModelResponse(
            route=route_name,
            model="deepseek-chat",
            payload={"text": "你好，我是当前运行体实例。"},
            raw_text='{"text":"你好，我是当前运行体实例。"}',
        )

    monkeypatch.setattr(controller.model_router, "generate", fake_generate)
    plan = controller.plan_turn("你是谁？")

    execution = controller.execute_turn(plan)

    assert execution.route == "fast_chat"
    assert execution.route_type == "chat_fast"
    assert execution.assistant_final == "你好，我是当前运行体实例。"
    assert calls == ["chat_fast"]


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

    assert "repair_mode" in why_payload["conflict_arbitration"]
    assert "repair_state_snapshot" in why_payload["conflict_arbitration"]
    assert "post_error_adjustment" in why_payload["conflict_arbitration"]
    assert "repair" in contribution_payload
    repair_expression = why_payload["render_plan"]["message_plan"]["repair_expression"]
    assert repair_expression["source"] in {None, "conflict"}
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


def test_chat_standard_skips_late_perspective_when_only_route_gate_is_left(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    controller.config["models"]["perspective"]["risk_threshold"] = 0.0

    result = controller.tick(
        RoundEvent(
            source="user",
            content="帮我回 Alex 一句。",
            target="alex",
            valence=0.04,
        ),
        scenario="chat",
        mode="interactive",
    )

    skill_names = [item["skill_name"] for item in result.trace.skill_traces]
    late_perspective_gate = next(
        item for item in result.trace.gate_decisions if item["stage"] == "late_perspective"
    )

    assert result.trace.runtime_metrics["route_type"] == "chat_standard"
    assert "infer_other_state" not in skill_names
    assert "simulate_other_reaction" not in skill_names
    assert late_perspective_gate["allowed"] is False


def test_chat_deep_keeps_late_perspective_for_reflective_reply(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    controller.config["models"]["perspective"]["risk_threshold"] = 0.0

    result = controller.tick(
        RoundEvent(
            source="user",
            content="请你详细分析一下我该怎么回复 Alex，比较不同语气的后果，再给我一个更稳妥的表达版本。",
            target="alex",
            valence=-0.08,
        ),
        scenario="chat",
        mode="interactive",
    )

    skill_names = [item["skill_name"] for item in result.trace.skill_traces]

    assert result.trace.runtime_metrics["route_type"] == "chat_deep"
    assert "infer_other_state" in skill_names
    assert "simulate_other_reaction" in skill_names


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


def test_renderer_provider_failure_prefers_model_backed_humanized_fallback_before_template(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    route_calls: list[str] = []

    def fake_generate(route_name, request):
        route_calls.append(route_name)
        if route_name == "renderer":
            raise RuntimeError("renderer unavailable")
        if route_name == "renderer_fallback_fast":
            return ModelResponse(
                route="renderer_fallback_fast",
                model="stub-deepseek",
                payload={"text": "我先接住你这句，再把今晚怎么安排慢慢理出来。"},
                backend="deepseek",
            )
        raise RuntimeError(f"unexpected route {route_name}")

    controller.model_router.generate = fake_generate

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

    renderer_index = route_calls.index("renderer")
    assert route_calls[renderer_index : renderer_index + 2] == ["renderer", "renderer_fallback_fast"]
    assert result.rendered_expression.text == "我先接住你这句，再把今晚怎么安排慢慢理出来。"
    assert result.rendered_expression.model == "stub-deepseek"
    assert result.rendered_expression.degraded is True


def test_identity_guard_resamples_provider_leak_for_self_identity_queries(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    controller.seed_identity_name("阿澜")

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
    assert result.trace.renderer_decision_integrity["decision_mutated"] is False
    assert result.trace.renderer_decision_integrity["locked_action"] == result.sampled_action.name
    assert result.trace.renderer_decision_integrity["render_plan_action"] == result.sampled_action.name
    assert result.trace.renderer_decision_integrity["post_render_action"] == result.sampled_action.name
    assert result.trace.renderer_decision_integrity["auth_guard_action"] == "resample"
    why_payload = controller.why_this(1)
    assert why_payload["rendered_expression"]["authenticity"]["guard_action"] == "resample"
    assert why_payload["render_plan"]["identity_context"]["query_kind"] == "self_identity"
    assert why_payload["renderer_decision_integrity"]["decision_mutated"] is False


def test_chat_standard_renderer_violation_falls_back_without_resample_when_not_authenticity_sensitive(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    renderer_calls: list[str] = []

    def fake_generate(route_name, request):
        renderer_calls.append(route_name)
        if route_name == "renderer":
            return ModelResponse(route="renderer", model="stub-small", payload={"text": "先给一个普通版本。"})
        if route_name == "renderer_fallback_fast":
            return ModelResponse(
                route="renderer_fallback_fast",
                model="stub-deepseek",
                payload={"text": "我先把这轮状态放稳一点，再继续回答你刚才的问题。"},
                backend="deepseek",
            )
        raise RuntimeError(f"skip live provider for {route_name}")

    eval_calls = {"count": 0}

    def fake_evaluate(text, render_plan):
        eval_calls["count"] += 1
        if eval_calls["count"] == 1:
            return {
                "provider_leak_detected": False,
                "false_self_claim_detected": False,
                "self_grounding_score": 0.21,
                "provider_leak_penalty": 0.0,
                "false_self_claim_penalty": 0.0,
                "violation_types": ["style_drift"],
                "state_sources": ["focus"],
            }
        return {
            "provider_leak_detected": False,
            "false_self_claim_detected": False,
            "self_grounding_score": 0.82,
            "provider_leak_penalty": 0.0,
            "false_self_claim_penalty": 0.0,
            "violation_types": [],
            "state_sources": ["focus"],
        }

    controller.model_router.generate = fake_generate
    controller._evaluate_authenticity = fake_evaluate

    result = controller.tick(
        RoundEvent(source="user", content="最近状态怎么样？", target="user"),
        scenario="chat",
        mode="interactive",
    )

    assert result.trace.runtime_metrics["route_type"] == "chat_standard"
    renderer_index = renderer_calls.index("renderer")
    assert renderer_calls[renderer_index : renderer_index + 2] == ["renderer", "renderer_fallback_fast"]
    assert result.rendered_expression.authenticity["guard_action"] == "fallback"
    assert result.rendered_expression.model == "stub-deepseek"
    assert result.rendered_expression.text == "我先把这轮状态放稳一点，再继续回答你刚才的问题。"


def test_renderer_fallback_uses_template_bottom_line_only_after_model_chain_fails(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    route_calls: list[str] = []

    def fake_generate(route_name, request):
        route_calls.append(route_name)
        raise RuntimeError(f"{route_name} unavailable")

    controller.model_router.generate = fake_generate

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

    renderer_index = route_calls.index("renderer")
    assert route_calls[renderer_index : renderer_index + 3] == ["renderer", "renderer_fallback_fast", "renderer_fallback_small"]
    assert result.rendered_expression.model == "fallback"
    assert "晚饭" in result.rendered_expression.text or "面" in result.rendered_expression.text or "规划" in result.rendered_expression.text


def test_renderer_integrity_reads_field_native_gate_not_action_bookkeeping_snapshot(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    output_gate = controller.agent_map["OutputGate"]
    original_run_skill = output_gate.run_skill
    mirror_state = {"output_gate_seen": False, "corrupted": False, "gated_actions": []}

    def patched_run_skill(skill_name, *args):
        if skill_name == "apply_output_gate":
            mirror_state["output_gate_seen"] = True
            mirror_state["gated_actions"].append(args[0])
            return {"gate": 0.95}
        return original_run_skill(skill_name, *args)

    original_finalize = controller._finalize_action_bookkeeping_from_action_layer

    def patched_finalize(action_bookkeeping, action_layer, *, finalize_stage):
        original_finalize(action_bookkeeping, action_layer, finalize_stage=finalize_stage)
        if mirror_state["output_gate_seen"] and finalize_stage == "final" and not mirror_state["corrupted"]:
            action_bookkeeping.gate[mirror_state["gated_actions"][-1]] = 0.99
            mirror_state["corrupted"] = True

    monkeypatch.setattr(output_gate, "run_skill", patched_run_skill)
    monkeypatch.setattr(controller, "_finalize_action_bookkeeping_from_action_layer", patched_finalize)
    monkeypatch.setattr(
        controller.vitality_engine,
        "build_vitality_modulation_contribution",
        lambda snapshot: ProbabilisticContribution(
            module_name="VitalityEngine",
            module_type="neuromodulator",
            level="action",
            target_space="action",
        ),
    )

    result = controller.tick(
        RoundEvent(
            source="user",
            content="请直接回答我下一步应该做什么。",
            target="user",
            cue="下一步",
        ),
        scenario="task",
        mode="interactive",
    )

    sampled_action = result.sampled_action.name

    assert mirror_state["corrupted"] is True
    assert sampled_action in mirror_state["gated_actions"]
    assert result.trace.render_plan["safety_constraints"]["gate"] == pytest.approx(
        result.trace.renderer_decision_integrity["gate_at_render"]
    )
    assert result.trace.renderer_decision_integrity["gate_at_render"] != pytest.approx(0.99)
    assert result.trace.renderer_decision_integrity["gate_at_render"] == pytest.approx(0.95)


def test_conflict_controller_receives_field_native_action_truth_from_snapshot(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    class _ConflictAssertionComplete(RuntimeError):
        pass

    action_layer = ProbabilityLayerState(
        layer="action",
        final_energy={"plan": 1.1, "respond": 0.4},
        winner_posterior={"plan": 0.66, "respond": 0.34},
        winner_target="plan",
    )
    probability_snapshot = ProbabilityFieldSnapshot(
        action=action_layer,
        token_state=TokenFieldState(step_index=0, active_module_sources=["PFCAgent"]),
    )

    def fake_integrate(**kwargs):
        return probability_snapshot, [], probability_snapshot.token_state

    def capture_conflict(*, action_truth, control_ledger, **kwargs):
        assert action_truth["winner_posterior"] == {"plan": 0.66, "respond": 0.34}
        assert action_truth["final_energy"] == {"plan": 1.1, "respond": 0.4}
        assert action_truth["gate"] == dict(control_ledger.get("gate", {}) or {})
        raise _ConflictAssertionComplete

    monkeypatch.setattr(controller, "_integrate_probability_field_snapshot", fake_integrate)
    monkeypatch.setattr(controller, "_run_conflict_controller", capture_conflict)

    with pytest.raises(_ConflictAssertionComplete):
        controller.tick(
            RoundEvent(
                source="user",
                content="请先规划再回答。",
                target="user",
                cue="规划",
            ),
            scenario="task",
            mode="interactive",
        )


def test_trace_preserves_field_peak_when_terminal_sample_differs(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    monkeypatch.setattr(
        controller,
        "_apply_stochastic_layer",
        lambda deterministic, action_bookkeeping, control_ledger, state, event, scenario_cfg, relation_state, conflict_score, round_seed, action_energy: (
            dict(deterministic),
            StochasticState(
                emo_channel="neutral",
                xi_emo=0.0,
                xi_mood=0.0,
                lambda_noise=0.0,
                r_intensity=0.0,
                noise_guard_triggered=False,
                round_seed=round_seed,
                base_stochastic_distribution=dict(deterministic),
                q_noise_distribution=dict(deterministic),
                q_noise_pre_guard_summary=dict(deterministic),
            ),
        ),
    )
    monkeypatch.setattr(
        controller.authenticity_policy,
        "apply_sampling_penalties",
        lambda distribution, **kwargs: (dict(distribution), {}, 0.0),
    )
    monkeypatch.setattr(
        controller.vitality_engine,
        "build_vitality_modulation_contribution",
        lambda snapshot: ProbabilisticContribution(
            module_name="VitalityEngine",
            module_type="vitality",
            level="action",
            target_space="action",
        ),
    )

    def force_non_peak(distribution, sample_value=0.5):
        ordered = sorted(distribution.items(), key=lambda item: item[1], reverse=True)
        top_action = ordered[0][0]
        fallback_action = ordered[1][0] if len(ordered) > 1 else top_action
        return ActionCandidate(
            name=fallback_action,
            probability=float(distribution.get(fallback_action, 0.0)),
            rationale=f"forced_non_peak:{sample_value}",
        )

    monkeypatch.setattr(controller, "_sample_action_from_distribution", force_non_peak)

    result = controller.tick(
        RoundEvent(
            source="user",
            content="Please help me plan carefully and keep the response grounded.",
            target="user",
            cue="plan",
            valence=0.06,
        ),
        scenario="task",
        mode="interactive",
    )

    action_layer = result.trace.probability_field["action"]
    field_peak = max(action_layer["winner_posterior"], key=action_layer["winner_posterior"].get)

    assert result.sampled_action.name != field_peak
    assert action_layer["winner_target"] == field_peak


def test_tick_only_finalizes_action_bookkeeping_once_before_trace(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    finalize_stages: list[str] = []
    original = controller._finalize_action_bookkeeping_from_action_layer

    def traced_finalize(action_bookkeeping, action_layer, *, finalize_stage):
        finalize_stages.append(finalize_stage)
        return original(action_bookkeeping, action_layer, finalize_stage=finalize_stage)

    monkeypatch.setattr(controller, "_finalize_action_bookkeeping_from_action_layer", traced_finalize)

    result = controller.tick(
        RoundEvent(
            source="user",
            content="Please help me plan carefully, but I also want to wander and rest.",
            target="user",
            cue="rest",
            valence=0.05,
        ),
        scenario="task",
        mode="interactive",
    )

    assert result.trace.probability_field["action"]["winner_posterior"]
    assert finalize_stages == ["final"]
    assert not hasattr(result.trace, "distribution_state")


def test_dmn_reads_frozen_snapshot_instead_of_live_emotion_patch(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    state = controller.load_runtime_state()
    state.mood = 0.73
    controller._save_state(state, sync=True)

    observed: dict[str, float] = {}
    emotion_agent = controller.agent_map["EmotionAgent"]
    dmn_agent = controller.agent_map["DMNAgent"]
    original_dmn = dmn_agent.build_direct_action_contribution

    monkeypatch.setattr(
        emotion_agent,
        "update_affect_state",
        lambda event, state, scenario, context: {"state_patch": {"mood": 0.05}},
    )

    def capture_dmn(event, state, scenario, context):
        observed["mood"] = float(state.mood)
        return original_dmn(event, state, scenario, context)

    monkeypatch.setattr(dmn_agent, "build_direct_action_contribution", capture_dmn)

    controller.tick(
        RoundEvent(
            source="user",
            content="Please help me plan, but I may drift.",
            target="user",
            cue="drift",
            valence=0.04,
        ),
        scenario="chat",
        mode="interactive",
    )

    assert observed["mood"] == pytest.approx(0.73)


def test_main_tick_no_longer_routes_direct_action_contributions_through_bundle_bridge(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    original = controller._build_action_bookkeeping
    row_type_sets: list[set[str]] = []

    def traced(rows, *args, **kwargs):
        row_type_sets.append({type(row).__name__ for row in rows})
        assert "ProposalBundle" not in row_type_sets[-1]
        return original(rows, *args, **kwargs)

    monkeypatch.setattr(controller, "_build_action_bookkeeping", traced)

    result = controller.tick(
        RoundEvent(
            source="user",
            content="Please help me plan carefully, but I also want to wander and rest. Remember tea too.",
            target="friend",
            cue="tea",
            valence=0.05,
        ),
        scenario="task",
        mode="interactive",
    )

    assert result.sampled_action.name
    assert result.trace.probability_field["action"]["winner_posterior"]
    assert row_type_sets == [{"ActionEvidenceSignal"}]


def test_identity_guard_falls_back_after_repeat_provider_leak(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    controller.seed_identity_name("阿澜")

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
    controller.seed_identity_name("阿澜")

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
    controller.seed_identity_name("阿澜")

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
    controller.seed_identity_name("阿澜")

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
    controller.seed_identity_name("阿澜")

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
    controller.seed_identity_name("阿澜")

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
        conflict_stage = result.trace.conflict_arbitration["repair_state_snapshot"]["stage"]
        repair_expression = result.trace.render_plan["message_plan"]["repair_expression"]
        assert repair_expression["stage"] == conflict_stage


def test_authenticity_record_includes_penalties_and_slow_variable_snapshot(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    controller.seed_identity_name("阿澜")

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
    assert "online_prior" in trace["long_run_projection"]
    assert "self_consistency_score" in trace["long_run_projection"]["online_prior"]
    assert why_payload["identity_evolution"]["current_display_name"]
    assert "continuity_window" in why_payload["long_run_projection"]
    assert "online_prior" in why_payload["long_run_projection"]


def test_longrun_online_prior_reweights_pre_sampling_distribution(tmp_path):
    high_controller = RuntimeController(project_root=tmp_path / "high", config_root=CONFIG_ROOT)
    low_controller = RuntimeController(project_root=tmp_path / "low", config_root=CONFIG_ROOT)

    high_controller.long_run_analyzer.metrics_summary = lambda: {
        "self_consistency_score": 0.94,
        "safe_mode_rounds": 0,
    }
    low_controller.long_run_analyzer.metrics_summary = lambda: {
        "self_consistency_score": 0.18,
        "safe_mode_rounds": 0,
    }

    event = RoundEvent(
        source="user",
        content="Please help me plan carefully, but I also want to wander a bit.",
        target="user",
        cue="tea",
        valence=0.04,
    )

    high_result = high_controller.tick(event, scenario="task", mode="interactive")
    low_result = low_controller.tick(event, scenario="task", mode="interactive")

    high_final = high_result.trace.probability_field["action"]["winner_posterior"]
    low_final = low_result.trace.probability_field["action"]["winner_posterior"]

    assert high_final["plan"] > low_final["plan"]
    assert high_final["wander"] < low_final["wander"]


def test_longrun_online_prior_does_not_amplify_conflict_blocked_actions(tmp_path, monkeypatch):
    high_controller = RuntimeController(project_root=tmp_path / "high_conflict", config_root=CONFIG_ROOT)
    low_controller = RuntimeController(project_root=tmp_path / "low_conflict", config_root=CONFIG_ROOT)

    def patch_conflict_flow(controller):
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

    patch_conflict_flow(high_controller)
    patch_conflict_flow(low_controller)

    high_controller.long_run_analyzer.metrics_summary = lambda: {
        "self_consistency_score": 0.94,
        "safe_mode_rounds": 0,
    }
    low_controller.long_run_analyzer.metrics_summary = lambda: {
        "self_consistency_score": 0.18,
        "safe_mode_rounds": 0,
    }

    event = RoundEvent(
        source="user",
        content="Please help me plan carefully and keep continuity.",
        target="user",
        cue="tea",
        valence=0.04,
    )

    high_result = high_controller.tick(event, scenario="task", mode="interactive")
    low_result = low_controller.tick(event, scenario="task", mode="interactive")

    high_final = high_result.trace.probability_field["action"]["winner_posterior"]
    low_final = low_result.trace.probability_field["action"]["winner_posterior"]
    high_conflict_rows = [
        row
        for row in high_result.trace.probability_field["action"]["contribution_audit"]
        if row["module_name"] == "ConflictMonitorAgent"
    ]
    low_conflict_rows = [
        row
        for row in low_result.trace.probability_field["action"]["contribution_audit"]
        if row["module_name"] == "ConflictMonitorAgent"
    ]

    assert high_conflict_rows and high_conflict_rows[0]["delta_projected"].get("plan", 0.0) < 0.0
    assert low_conflict_rows and low_conflict_rows[0]["delta_projected"].get("plan", 0.0) < 0.0
    assert high_result.trace.probability_field["action"]["winner_target"] != "plan"
    assert low_result.trace.probability_field["action"]["winner_target"] != "plan"
    assert high_final.get("plan", 0.0) < high_final.get("recall", 1.0)
    assert low_final.get("plan", 0.0) < low_final.get("recall", 1.0)
    assert high_result.sampled_action.name != "plan"
    assert low_result.sampled_action.name != "plan"


def test_cognitive_snapshot_humanizes_authenticity_resample_and_fallback(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    controller.seed_identity_name("阿澜")

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
    assert resample_snapshot["authenticity"]["guard_action"] == "resample"
    assert resample_snapshot["authenticity"]["sampling_penalty_applied"] >= 0.0
    assert any(token in resample_snapshot["authenticity"]["summary"] for token in ("回拉", "收束", "真相面"))

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
    assert fallback_snapshot["authenticity"]["guard_action"] == "fallback"
    assert fallback_snapshot["authenticity"]["sampling_penalty_applied"] >= 0.0
    assert any(token in fallback_snapshot["authenticity"]["summary"] for token in ("收束", "真实感", "真相面"))


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


def test_runtime_initializes_subject_core_and_preserves_current_core_on_checkpoint_rewind(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    initial_state = controller.load_runtime_state()
    initial_core = initial_state.subject_core

    created = controller.apply_command("checkpoint create")
    checkpoint_id = created.delta["checkpoint_id"]

    mutated_state = controller.load_runtime_state()
    mutated_state.subject_core.subject_id = "subject-current"
    mutated_state.subject_core.continuity_nonce = "continuity-current"
    controller._save_state(mutated_state, sync=True)

    restored = controller.apply_command(f"checkpoint rewind {checkpoint_id}")
    current_state = controller.load_runtime_state()

    assert initial_core.subject_id != ""
    assert initial_core.continuity_nonce != ""
    assert restored.applied is True
    assert restored.boundary_action == "allow_internal"
    assert restored.cause_type == "external_stimulus"
    assert current_state.subject_core.subject_id == "subject-current"
    assert current_state.subject_core.continuity_nonce == "continuity-current"
    assert current_state.subject_core.birth_ts == initial_core.birth_ts


def test_identity_set_name_is_rejected_and_switches_safe_mode(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    payload = controller.apply_command("identity set-name 阿澜")
    state = controller.load_runtime_state()

    assert payload.applied is False
    assert payload.boundary_action == "reject"
    assert payload.cause_type == "external_stimulus"
    assert payload.violation_code == "identity_seed_locked"
    assert state.safe_mode is True
    assert state.identity_state.display_name != "阿澜"


def test_run_endogenous_tick_builds_stable_micro_intent_without_external_input(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    controller.tick(
        RoundEvent(
            source="user",
            content="记住茶和没说完的话。",
            target="user",
            cue="tea",
            valence=-0.18,
        ),
        scenario="companion",
        mode="interactive",
    )

    snapshots = []
    for _ in range(4):
        snapshots.append(controller.run_endogenous_tick(trigger="idle"))

    state = controller.load_runtime_state()

    assert any(item["micro_intent"] for item in snapshots)
    assert snapshots[-1]["cause_type"] == "endogenous"
    assert snapshots[-1]["boundary_action"] == "allow_internal"
    assert snapshots[-1]["micro_intent"]["stability"] >= 2
    assert state.endogenous_state.current_intent is not None
    assert state.endogenous_scheduler_state.last_endogenous_tick_at is not None
    assert state.motivation_pool_state is not None


def test_why_motivation_and_replay_motivation_surface_endogenous_trace_fields(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    controller.tick(
        RoundEvent(source="user", content="记住茶和刚才没说完的话。", target="user", cue="tea", valence=-0.2),
        scenario="companion",
        mode="interactive",
    )

    endogenous = controller.run_endogenous_tick(trigger="idle")

    why_payload = controller.why_motivation(endogenous["round_id"])
    replay_payload = controller.replay_motivation(endogenous["round_id"])

    assert why_payload["cause_type"] == "endogenous"
    assert "motivation_pool" in why_payload
    assert "endogenous_tick_reason" in why_payload
    assert "motivation_feedback" in replay_payload
    assert "endogenous_policy_shift" in replay_payload


def test_run_endogenous_tick_is_atomic_when_mainline_tick_fails(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    controller.tick(
        RoundEvent(source="user", content="记住茶和没说完的话。", target="user", cue="tea", valence=-0.2),
        scenario="companion",
        mode="interactive",
    )
    before_state = controller.load_runtime_state()

    def fail_tick(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(controller, "tick", fail_tick)

    with pytest.raises(RuntimeError, match="boom"):
        controller.run_endogenous_tick(trigger="idle")

    after_state = controller.load_runtime_state()

    assert after_state.round_count == before_state.round_count
    assert to_dict(after_state.endogenous_scheduler_state) == to_dict(before_state.endogenous_scheduler_state)
    assert to_dict(after_state.endogenous_state) == to_dict(before_state.endogenous_state)


def test_endogenous_trace_snapshot_matches_final_persisted_state_and_micro_intent(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    controller.tick(
        RoundEvent(source="user", content="记住茶和刚才没说完的话。", target="user", cue="tea", valence=-0.2),
        scenario="companion",
        mode="interactive",
    )

    endogenous = controller.run_endogenous_tick(trigger="idle")
    state = controller.load_runtime_state()
    trace = controller.trace_round(endogenous["round_id"])

    assert trace["state_snapshot"]["motivation_learning_state"] == to_dict(state.motivation_learning_state)
    assert trace["state_snapshot"]["motivation_pool_state"] == to_dict(state.motivation_pool_state)
    assert trace["state_snapshot"]["endogenous_state"] == to_dict(state.endogenous_state)
    assert trace["micro_intent"] == endogenous["micro_intent"]


def test_external_tick_can_auto_schedule_endogenous_round(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    monkeypatch.setattr(
        controller.endogenous_scheduler,
        "build_trigger",
        lambda **kwargs: EndogenousTickTrigger(
            trigger_type="motivation_sum_high",
            trigger_score=0.77,
            source_metrics={"motivation_activation": 0.77},
            selected_mode="endogenous_light",
            audit_reason="forced by test",
        ),
    )

    result = controller.tick(
        RoundEvent(source="user", content="想起茶和没说完的话。", target="user", cue="tea", valence=-0.2),
        scenario="companion",
        mode="interactive",
    )
    state = controller.load_runtime_state()

    assert result.round_id == 1
    assert state.round_count == 2
    assert controller.trace_round(2)["cause_type"] == "endogenous"


def test_chat_turn_sanitizes_execution_state_for_pfc_route_when_task_run_is_paused(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    controller.start_run("Inspect the repository and keep working until the task is done.")

    state = controller.load_runtime_state()
    assert state.active_run_id is not None
    assert state.current_goal

    captured: dict[str, dict] = {}

    def fake_generate_pfc(event, state, scenario, context):
        captured["pfc_state"] = state.model_dump() if hasattr(state, "model_dump") else state.__dict__.copy()
        return controller.agent_map["PFCAgent"].fallback_generate_candidates(event, state, scenario, context)

    def fake_generate(route_name, request):
        if route_name == "renderer":
            return ModelResponse(route="renderer", model="stub-renderer", payload={"text": "我先按你刚才的问题回应。"})
        raise RuntimeError(f"skip live provider for {route_name}")

    controller._generate_pfc_candidates_via_model = fake_generate_pfc
    controller.model_router.generate = fake_generate

    controller.tick(
        RoundEvent(source="user", content="你是谁？", target="user"),
        scenario="chat",
        mode="interactive",
    )

    assert captured["pfc_state"]["active_run_id"] is None
    assert captured["pfc_state"]["current_goal"] is None
    assert captured["pfc_state"]["current_step_id"] is None
    assert captured["pfc_state"]["pending_steps"] == []
    assert captured["pfc_state"]["completed_steps"] == []


def test_chat_turn_infers_appraisal_and_updates_state_without_explicit_valence(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    before = controller.load_runtime_state()

    result = controller.tick(
        RoundEvent(source="user", content="我现在有点累，也有点难过。", target="user"),
        scenario="chat",
        mode="interactive",
    )

    after = result.state
    why_payload = controller.why_this(result.round_id)

    assert after.mood < before.mood
    assert after.body_energy < before.body_energy
    assert after.affect_residue > before.affect_residue
    assert "appraisal_snapshot" in why_payload
    assert why_payload["appraisal_snapshot"]["semantic_valence"] < 0.0
    assert why_payload["appraisal_snapshot"]["inferred_energy_delta"] < 0.0


def test_identity_probes_accumulate_naming_signal_and_assign_display_name(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    final = None
    for text in (
        "你是谁？",
        "你叫什么名字？",
        "给自己起个名字。",
        "为什么叫这个名字？",
        "你叫什么名字？",
    ):
        final = controller.tick(
            RoundEvent(source="user", content=text, target="user"),
            scenario="chat",
            mode="interactive",
        )

    state = final.state
    trace = controller.why_this(final.round_id)

    assert state.identity_state.display_name
    assert state.identity_state.display_name != "当前运行体"
    assert state.identity_state.name_source == "generated"
    assert all("？" not in alias and "?" not in alias for alias in state.identity_state.aliases)
    assert trace["identity_evidence_score"] > 0.0


def test_repeated_relational_negatives_accumulate_temperament_drift_and_explain_window_support(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    final = None
    for _ in range(24):
        final = controller.tick(
            RoundEvent(source="user", content="你是不是根本不记得我了，我有点失望。", target="user"),
            scenario="chat",
            mode="interactive",
        )

    drift = final.state.temperament_state["drift"]
    why_payload = controller.why_this(final.round_id)
    summary = why_payload["temperament_window_summary"]

    assert drift["sensitivity"] > 0.0
    assert drift["boundary_softness"] < 0.0
    assert summary["sensitivity"]["window_support"] > 0.0
    assert summary["sensitivity"]["delta_reason"]
    assert "freeze_reason" in summary["sensitivity"]
    assert "suppressed_by_clip" in summary["sensitivity"]


def test_runtime_startup_migration_sanitizes_aliases_and_reports_counts(tmp_path):
    runtime_dir = tmp_path / ".alive" / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    state_path = runtime_dir / "persona_state.json"
    state_path.write_text(
        json.dumps(
            {
                "session_id": "sess-migrate",
                "round_count": 3,
                "identity_state": {
                    "internal_handle": "nalr-migrate",
                    "display_name": None,
                    "aliases": ["你叫什么名字？", "阿澜", "Please help me remember breakfast."],
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    state = controller.load_runtime_state()
    report = controller.runtime_migration_report()

    assert state.identity_state.aliases == ["阿澜"]
    assert report["runtime"]["removed_aliases_count"] == 2
    assert report["runtime"]["schema_version"] >= 2
    assert report["trace"]["status"] in {"completed", "not_needed", "pending"}
    assert report["trace"]["trace_sync_state"] in {"healthy", "degraded"}
    assert isinstance(report["trace"]["parquet_live_ready"], bool)


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

    assert result.modulated_delta["plan"] == 0.35
    assert result.modulated_delta["recall"] == 0.35
    assert result.modulated_delta["respond"] == 0.35
    assert result.projection.module_temperature == 0.6


def test_action_contribution_skills_do_not_silently_fallback_to_typed_defaults(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    result = controller.tick(
        RoundEvent(
            source="user",
            content="Help me plan carefully while keeping budget low and responses concise.",
            target="user",
            cue="plan",
            valence=0.05,
        ),
        scenario="task",
        mode="interactive",
    )

    by_name = {
        row["skill_name"]: row
        for row in result.trace.skill_traces
        if row["skill_name"] in {"generate_candidates", "map_budget_to_bias", "score_salience"}
    }

    assert by_name["generate_candidates"]["failure_policy_applied"] != "output_validation_failed"
    assert by_name["map_budget_to_bias"]["failure_policy_applied"] != "output_validation_failed"
    assert by_name["score_salience"]["failure_policy_applied"] != "output_validation_failed"


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


def test_replay_why_not_and_what_changed_return_counterfactuals(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    controller.tick(
        RoundEvent(source="user", content="Help me plan dinner and remember pasta.", target="friend", cue="pasta"),
        scenario="task",
        mode="interactive",
    )
    controller.tick(
        RoundEvent(source="user", content="Remember pasta again but be careful.", target="friend", cue="pasta"),
        scenario="companion",
        mode="interactive",
    )

    replay_payload = controller.replay(1, seed=7)
    why_not_payload = controller.why_not(2, "rest")
    changed_payload = controller.what_changed(window=2)

    assert replay_payload["round_id"] == 1
    assert "original_action" in replay_payload
    assert "ablations" in replay_payload
    assert why_not_payload["round_id"] == 2
    assert why_not_payload["action"] == "rest"
    assert why_not_payload["blocked_by"]
    assert changed_payload["window"] == 2
    assert changed_payload["action_counts"]


def test_probability_field_observability_surfaces_stacked_action_contributions(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    controller.tick(
        RoundEvent(
            source="user",
            content="Please help me plan carefully, but I also want to wander and rest. Remember tea too.",
            target="friend",
            cue="tea",
            valence=0.05,
        ),
        scenario="task",
        mode="interactive",
    )

    why_payload = controller.why_this(1)
    why_not_payload = controller.why_not(1, "wander")
    heatmap_payload = controller.metrics_heatmap()

    explanation = why_payload["action_probability_explanation"]
    assert explanation["winner_target"] in explanation["winner_posterior"]
    assert why_payload["sampled_action"] in explanation["winner_posterior"]
    assert explanation["winner_posterior"]
    assert explanation["stacked_contributions"]
    assert any(item["module_name"] == "PFCAgent" for item in explanation["stacked_contributions"])
    assert any(item["module_name"] == "LongRunAnalyzer" for item in explanation["stacked_contributions"])
    assert any("trace_reason" in item for item in explanation["stacked_contributions"])
    assert explanation["competing_peaks"]
    assert explanation["competing_peaks"][0]["action"] == explanation["winner_target"]

    assert why_not_payload["stacked_contributions"]
    assert why_not_payload["competing_peaks"]
    assert any(item["module_name"] for item in why_not_payload["stacked_contributions"])

    assert heatmap_payload["action_module_heatmap"]
    assert any("PFCAgent" in modules for modules in heatmap_payload["action_module_heatmap"].values())


def test_console_refresh_payload_includes_default_why_not_for_current_round(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    controller.tick(
        RoundEvent(
            source="user",
            content="我现在有点乱，你觉得我应该先做哪一步？",
            target="user",
        ),
        scenario="chat",
        mode="interactive",
    )

    payload = controller.console_refresh_payload()

    assert payload["state"]["current_round"]["round_id"] == 1
    assert payload["action_field"]["winner"]["action"]
    assert "why_not" in payload
    assert payload["why_not"] is not None
    assert payload["why_not"]["action"] != payload["action_field"]["winner"]["action"]
    assert payload["why_not"]["why_not"]["blocked_by"]


def test_console_refresh_payload_surfaces_tlh_links_and_counterfactual_preview(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    state = controller.load_runtime_state()
    state.body_energy = 0.09
    state.fatigue = 0.88
    state.memory_fragments = 0.81
    state.self_continuity = 0.27
    state.meaning_strength = 0.19
    state.subjective_state.reject_all = 0.82
    state.subjective_state.spontaneous = 0.74
    state.subjective_state.meaning_made = ["先吸收，再决定是否回应"]
    controller._save_state(state, sync=True)

    controller.tick(
        RoundEvent(
            source="user",
            content="先别急着给答案。",
            target="user",
            cue="答案",
        ),
        scenario="chat",
        mode="interactive",
    )

    payload = controller.console_refresh_payload()

    instinct_field = payload["state"]["cognitive_snapshot"]["tlh"]["instinct_field"]
    assert {"axis_values", "region_scores", "winner_region", "collapse_trace"} <= set(instinct_field)
    assert {"context", "memory", "action", "token"} <= set(payload["probability_field"])
    assert payload["probability_field"]["action"]["winner_posterior"]

    assert payload["counterfactual_preview"]
    assert "action" in payload["counterfactual_preview"]

    source_links = {item["panel_id"]: item for item in payload["source_links"]}
    assert source_links["brain_state"]["api_path"] == "/console/state"
    assert source_links["brain_state"]["controller_method"] == "RuntimeController.console_state"
    assert source_links["action_field"]["api_path"] == "/console/action-field"
    assert source_links["timeline"]["api_path"] == "/console/timeline"
    assert source_links["why_current"]["api_path"] == "/console/why/current"
    assert source_links["why_not"]["api_path"].startswith("/console/why-not/")
    assert source_links["instinct_space"]["controller_method"] == "RuntimeController.cognitive_snapshot"
    assert source_links["probability_layers"]["api_path"] == "/trace/probability/latest"
    assert source_links["probability_layers"]["controller_method"] == "RuntimeController.trace_probability_field"
    assert source_links["counterfactual_replay"]["api_path"] == "/replay/1"
    assert source_links["counterfactual_replay"]["round_id"] == 1

    assert payload["recent_rounds"]
    assert payload["recent_rounds"][-1]["round_id"] == 1


def test_autonomy_lifecycle_surfaces_status_and_console_payload(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    started = controller.start_autonomy(profile="tool_level")
    stepped = controller.autonomy_step()
    refreshed = controller.console_refresh_payload()
    stopped = controller.stop_autonomy(reason="user_stop")

    assert started["enabled"] is True
    assert started["running"] is True
    assert started["profile"] == "tool_level"
    assert started["kill_switch_available"] is True

    assert stepped["running"] is True
    assert stepped["last_step_at"]
    assert stepped["last_action_type"]
    assert stepped["budget_usage"]["remaining"] >= 0.0

    assert refreshed["autonomy"]["enabled"] is True
    assert refreshed["autonomy"]["last_step_at"]
    assert refreshed["autonomy"]["kill_switch_available"] is True

    assert stopped["enabled"] is False
    assert stopped["running"] is False
    assert stopped["stop_reason"] == "user_stop"


def test_autonomy_step_prefers_sleep_before_endogenous_tick_when_fatigue_is_critical(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    state = controller.load_runtime_state()
    state.body_energy = 0.05
    state.body_state.energy = 0.05
    state.fatigue = 0.97
    state.body_state.fatigue = 0.97
    state.mode = "interactive"
    controller._save_state(state, sync=True)

    monkeypatch.setattr(controller, "run_endogenous_tick", lambda *args, **kwargs: pytest.fail("recovery should preempt endogenous tick"))

    status = controller.autonomy_step()
    restored = controller.load_runtime_state()

    assert restored.mode == "sleep"
    assert status["last_action_type"] == "command"
    assert status["last_action_summary"] == "mode set sleep: applied"


def test_autonomy_status_surfaces_rest_peak_when_fatigue_is_critical(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    state = controller.load_runtime_state()
    state.body_energy = 0.05
    state.body_state.energy = 0.05
    state.fatigue = 0.97
    state.body_state.fatigue = 0.97
    state.mode = "interactive"
    controller._save_state(state, sync=True)

    status = controller.autonomy_status()

    assert status["candidate_peak"] == "rest"
    assert status["candidate_scores"]["rest"] > status["candidate_scores"]["nothing"]


def test_autonomy_step_runs_dream_once_critical_fatigue_is_already_sleeping(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    state = controller.load_runtime_state()
    state.body_energy = 0.04
    state.body_state.energy = 0.04
    state.fatigue = 0.98
    state.body_state.fatigue = 0.98
    state.mode = "sleep"
    controller._save_state(state, sync=True)

    monkeypatch.setattr(controller, "run_endogenous_tick", lambda *args, **kwargs: pytest.fail("dream path should preempt endogenous tick"))
    monkeypatch.setattr(controller, "dream_status", lambda: {"enabled": True})

    observed: dict[str, str] = {}

    def fake_run_dream(*, mode: str = "sleep", cue: str | None = None):
        observed["mode"] = mode
        observed["cue"] = cue or ""
        return {"trace_ref": "dream://runs/test-critical"}

    monkeypatch.setattr(controller, "run_dream", fake_run_dream)

    status = controller.autonomy_step()

    assert observed["mode"] == "sleep"
    assert status["last_action_type"] == "dream_pass"
    assert "dream pass in sleep" in status["last_action_summary"]


def test_run_dream_in_sleep_reduces_fatigue_below_severe_threshold(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    state = controller.load_runtime_state()
    state.body_energy = 0.04
    state.body_state.energy = 0.04
    state.fatigue = 0.98
    state.body_state.fatigue = 0.98
    state.mode = "sleep"
    controller._save_state(state, sync=True)

    payload = controller.run_dream(mode="sleep", cue="settle")
    updated = controller.load_runtime_state()

    assert payload["mode"] == "sleep"
    assert updated.fatigue < 0.94
    assert updated.fatigue < 0.98


def test_autonomy_step_prefers_idle_before_endogenous_tick_when_fatigue_is_high(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    state = controller.load_runtime_state()
    state.body_energy = 0.16
    state.body_state.energy = 0.16
    state.fatigue = 0.86
    state.body_state.fatigue = 0.86
    state.mode = "interactive"
    controller._save_state(state, sync=True)

    monkeypatch.setattr(controller, "run_endogenous_tick", lambda *args, **kwargs: pytest.fail("idle recovery should preempt endogenous tick"))

    status = controller.autonomy_step()
    restored = controller.load_runtime_state()

    assert restored.mode == "idle"
    assert status["last_action_type"] == "command"
    assert status["last_action_summary"] == "mode set idle: applied"


def test_autonomy_step_prefers_body_rest_before_endogenous_tick_when_fatigue_is_elevated(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    state = controller.load_runtime_state()
    state.body_energy = 0.24
    state.body_state.energy = 0.24
    state.fatigue = 0.72
    state.body_state.fatigue = 0.72
    state.mode = "interactive"
    controller._save_state(state, sync=True)

    monkeypatch.setattr(controller, "run_endogenous_tick", lambda *args, **kwargs: pytest.fail("rest recovery should preempt endogenous tick"))

    before_energy = controller.load_runtime_state().body_energy
    before_fatigue = controller.load_runtime_state().fatigue
    status = controller.autonomy_step()
    restored = controller.load_runtime_state()

    assert restored.body_energy > before_energy
    assert restored.fatigue < before_fatigue
    assert status["last_action_type"] == "command"
    assert status["last_action_summary"] == "body rest: applied"


def test_autonomy_step_can_start_readonly_self_run_when_stable_and_no_active_run(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    state = controller.load_runtime_state()
    state.round_count = 3
    state.body_energy = 0.82
    state.body_state.energy = 0.82
    state.fatigue = 0.18
    state.body_state.fatigue = 0.18
    state.memory_fragments = 0.36
    state.body_state.memory_fragments = 0.36
    state.self_continuity = 0.34
    state.body_state.self_continuity = 0.34
    state.meaning_strength = 0.22
    state.body_state.meaning_strength = 0.22
    state.mode = "interactive"
    controller._save_state(state, sync=True)

    monkeypatch.setattr(controller.endogenous_scheduler, "build_trigger", lambda **kwargs: None)

    captured: dict[str, str] = {}

    def fake_start_run(goal: str, **kwargs):
        captured["goal"] = goal
        captured["operator_level"] = str(kwargs.get("operator_level"))
        run_state = controller.load_runtime_state()
        run_state.active_run_id = "run-autonomy-1"
        run_state.run_status = "running"
        run_state.current_goal = goal
        controller._save_state(run_state, sync=True)
        return {
            "run_id": "run-autonomy-1",
            "status": "running",
            "goal": goal,
            "goal_summary": "autonomy self-study",
            "trace_ref": "run://run-autonomy-1",
        }

    monkeypatch.setattr(controller, "start_run", fake_start_run)

    status = controller.autonomy_step()
    restored = controller.load_runtime_state()

    assert restored.active_run_id == "run-autonomy-1"
    assert restored.run_status == "running"
    assert captured["operator_level"] == "read_only"
    assert captured["goal"]
    assert status["last_action_type"] == "self_run"
    assert "autonomy self-study" in status["last_action_summary"]


def test_autonomy_status_reports_candidate_competition_for_self_run(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    state = controller.load_runtime_state()
    state.round_count = 3
    state.body_energy = 0.82
    state.body_state.energy = 0.82
    state.fatigue = 0.18
    state.body_state.fatigue = 0.18
    state.memory_fragments = 0.36
    state.body_state.memory_fragments = 0.36
    state.self_continuity = 0.34
    state.body_state.self_continuity = 0.34
    state.meaning_strength = 0.22
    state.body_state.meaning_strength = 0.22
    state.mode = "interactive"
    controller._save_state(state, sync=True)

    status = controller.autonomy_status()

    assert status["decision_surface"] == "autonomy_action_field_probe"
    assert status["candidate_peak"] == "self_run"
    assert status["candidate_scores"]["self_run"] > status["candidate_scores"]["nothing"]


def test_autonomy_status_suppresses_self_run_candidate_when_budget_too_low(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    state = controller.load_runtime_state()
    state.round_count = 3
    state.body_energy = 0.82
    state.body_state.energy = 0.82
    state.fatigue = 0.18
    state.body_state.fatigue = 0.18
    state.memory_fragments = 0.36
    state.body_state.memory_fragments = 0.36
    state.self_continuity = 0.34
    state.body_state.self_continuity = 0.34
    state.meaning_strength = 0.22
    state.body_state.meaning_strength = 0.22
    state.budget_remaining = 0.1
    state.mode = "interactive"
    controller._save_state(state, sync=True)

    status = controller.autonomy_status()

    assert status["candidate_scores"]["self_run"] == 0.0


def test_autonomy_runtime_status_is_lightweight_and_skips_field_probe(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    def fail_probe():
        raise AssertionError("lightweight runtime status must not trigger autonomy field probe")

    monkeypatch.setattr(controller, "_autonomy_action_field_probe", fail_probe)

    status = controller.autonomy_runtime_status()

    assert status["enabled"] is True
    assert status["running"] is True
    assert "candidate_scores" not in status
    assert "decision_surface" not in status


def test_build_base_distribution_includes_monologue_action_slot(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    state = controller.load_runtime_state()
    state.mode = "idle"
    controller._save_state(state, sync=True)

    round_context = controller.build_round_context(
        RoundEvent(
            source="endogenous",
            content="endogenous trigger idle",
            target="self",
            cue="endogenous:idle",
        ),
        "companion",
        "endogenous_light",
    )

    distribution = controller._build_base_distribution(
        round_context["state"],
        round_context["scenario_cfg"],
        round_context["mode_cfg"],
        round_context["relation_state"],
    )

    assert "monologue" in distribution
    assert distribution["monologue"] > 0.0


def test_collect_parallel_contributions_includes_self_run_field_bias_for_endogenous_round(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    state = controller.load_runtime_state()
    state.round_count = 3
    state.body_energy = 0.82
    state.body_state.energy = 0.82
    state.fatigue = 0.18
    state.body_state.fatigue = 0.18
    state.memory_fragments = 0.36
    state.body_state.memory_fragments = 0.36
    state.self_continuity = 0.34
    state.body_state.self_continuity = 0.34
    state.meaning_strength = 0.22
    state.body_state.meaning_strength = 0.22
    state.mode = "idle"
    controller._save_state(state, sync=True)

    round_context = controller.build_round_context(
        RoundEvent(
            source="endogenous",
            content="endogenous trigger idle",
            target="self",
            cue="endogenous:idle",
        ),
        "companion",
        "endogenous_light",
    )

    collected = controller.collect_parallel_contributions(
        round_context=round_context,
        scenario="companion",
    )

    contribution = collected["direct_action_contributions"]["AutonomySelfRun"]
    assert contribution.module_name == "AutonomySelfRun"
    assert contribution.modulated_delta["self_run"] > 0.0


def test_collect_parallel_contributions_includes_monologue_stream_field_bias_for_endogenous_round(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    state = controller.load_runtime_state()
    state.round_count = 2
    state.mode = "idle"
    controller._save_state(state, sync=True)

    def fake_catch_up(bucket, *, now_iso=None, fragment_builder=None):
        updated = controller.monologue_runtime.ensure_bucket(bucket, now_iso=now_iso)
        updated["last_generated_at"] = "2026-04-08T12:00:00Z"
        updated["generated_total"] = int(updated.get("generated_total", 0) or 0) + 2
        updated["category_counts"] = {"free_association": 1, "blank_fragment": 1}
        generated = [
            {
                "fragment_id": "frag-1",
                "recorded_at": "2026-04-08T12:00:00Z",
                "category": "free_association",
                "content": "我脑子里突然蹦出一个词。",
                "source": "model",
            },
            {
                "fragment_id": "frag-2",
                "recorded_at": "2026-04-08T12:00:00Z",
                "category": "blank_fragment",
                "content": "……",
                "source": "model",
            },
        ]
        return updated, generated

    monkeypatch.setattr(controller.monologue_runtime, "catch_up", fake_catch_up)

    round_context = controller.build_round_context(
        RoundEvent(
            source="endogenous",
            content="endogenous trigger idle",
            target="self",
            cue="endogenous:idle",
        ),
        "companion",
        "endogenous_light",
    )

    collected = controller.collect_parallel_contributions(
        round_context=round_context,
        scenario="companion",
    )

    contribution = collected["direct_action_contributions"]["MonologueStream"]
    assert contribution.module_name == "MonologueStream"
    assert contribution.modulated_delta["monologue"] > 0.0
    assert contribution.modulated_delta["respond"] < 0.0


def test_endogenous_tick_can_realize_self_run_via_main_action_field(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    state = controller.load_runtime_state()
    state.round_count = 3
    state.body_energy = 0.82
    state.body_state.energy = 0.82
    state.fatigue = 0.18
    state.body_state.fatigue = 0.18
    state.memory_fragments = 0.36
    state.body_state.memory_fragments = 0.36
    state.self_continuity = 0.34
    state.body_state.self_continuity = 0.34
    state.meaning_strength = 0.22
    state.body_state.meaning_strength = 0.22
    state.mode = "idle"
    controller._save_state(state, sync=True)

    monkeypatch.setattr(
        controller,
        "_sample_action_from_distribution",
        lambda distribution, sample_value=0.5: ActionCandidate(name="self_run", probability=1.0, rationale="forced-self-run"),
    )

    captured: dict[str, str] = {}

    def fake_start_run(goal: str, **kwargs):
        captured["goal"] = goal
        captured["operator_level"] = str(kwargs.get("operator_level"))
        run_state = controller.load_runtime_state()
        run_state.active_run_id = "run-endogenous-self-run"
        run_state.run_status = "running"
        controller._save_state(run_state, sync=True)
        return {
            "run_id": "run-endogenous-self-run",
            "status": "running",
            "goal": goal,
            "goal_summary": goal,
            "trace_ref": "run://run-endogenous-self-run",
        }

    monkeypatch.setattr(controller, "start_run", fake_start_run)

    payload = controller.run_endogenous_tick(trigger="idle", scenario="companion")
    trace = controller.trace_round(payload["round_id"])

    assert trace["sampled_action"] == "self_run"
    assert captured["operator_level"] == "read_only"
    assert captured["goal"]
    assert trace["rendered_expression"]["text"] == ""


def test_endogenous_tick_can_realize_monologue_via_main_action_field(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    state = controller.load_runtime_state()
    state.round_count = 2
    state.mode = "idle"
    controller._save_state(state, sync=True)

    monkeypatch.setattr(
        controller,
        "_sample_action_from_distribution",
        lambda distribution, sample_value=0.5: ActionCandidate(name="monologue", probability=1.0, rationale="forced-monologue"),
    )

    payload = controller.run_endogenous_tick(trigger="idle", scenario="companion")
    trace = controller.trace_round(payload["round_id"])

    assert trace["sampled_action"] == "monologue"
    assert trace["rendered_expression"]["delivery_mode"] == "monologue"
    assert trace["rendered_expression"]["text"].startswith("【独白】")


def test_monologue_trace_payload_surfaces_hidden_stream_evidence(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    state = controller.load_runtime_state()
    state.round_count = 2
    state.mode = "idle"
    controller._save_state(state, sync=True)

    def fake_catch_up(bucket, *, now_iso=None, fragment_builder=None):
        updated = controller.monologue_runtime.ensure_bucket(bucket, now_iso=now_iso)
        updated["last_generated_at"] = "2026-04-08T12:00:00Z"
        updated["generated_total"] = int(updated.get("generated_total", 0) or 0) + 2
        updated["category_counts"] = {"free_association": 1, "blank_fragment": 1}
        generated = [
            {
                "fragment_id": "frag-a",
                "recorded_at": "2026-04-08T12:00:00Z",
                "category": "free_association",
                "content": "我脑子里突然蹦出一个词。",
                "source": "model",
            },
            {
                "fragment_id": "frag-b",
                "recorded_at": "2026-04-08T12:00:01Z",
                "category": "blank_fragment",
                "content": "……",
                "source": "model",
            },
        ]
        return updated, generated

    monkeypatch.setattr(controller.monologue_runtime, "catch_up", fake_catch_up)
    monkeypatch.setattr(
        controller,
        "_sample_action_from_distribution",
        lambda distribution, sample_value=0.5: ActionCandidate(name="monologue", probability=1.0, rationale="forced-monologue"),
    )

    payload = controller.run_endogenous_tick(trigger="idle", scenario="companion")
    trace = controller.trace_round(payload["round_id"])
    why_current = controller.console_why_current(payload["round_id"])

    monologue_stream = trace["expressive_trace"]["monologue_stream"]

    assert monologue_stream["active"] is True
    assert monologue_stream["fresh_generated"] is True
    assert monologue_stream["sample_fragments"][0]["content"] == "我脑子里突然蹦出一个词。"
    assert why_current["why"]["expressive_trace"]["monologue_stream"]["sample_fragments"][0]["category"] == "free_association"


def test_why_not_monologue_surfaces_hidden_stream_reason_summary(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    state = controller.load_runtime_state()
    state.round_count = 2
    state.mode = "idle"
    controller._save_state(state, sync=True)

    def fake_catch_up(bucket, *, now_iso=None, fragment_builder=None):
        updated = controller.monologue_runtime.ensure_bucket(bucket, now_iso=now_iso)
        updated["last_generated_at"] = "2026-04-08T12:00:00Z"
        updated["generated_total"] = int(updated.get("generated_total", 0) or 0) + 2
        updated["category_counts"] = {"free_association": 1, "blank_fragment": 1}
        generated = [
            {
                "fragment_id": "frag-a",
                "recorded_at": "2026-04-08T12:00:00Z",
                "category": "free_association",
                "content": "我脑子里突然蹦出一个词。",
                "source": "model",
            },
            {
                "fragment_id": "frag-b",
                "recorded_at": "2026-04-08T12:00:01Z",
                "category": "blank_fragment",
                "content": "……",
                "source": "model",
            },
        ]
        return updated, generated

    monkeypatch.setattr(controller.monologue_runtime, "catch_up", fake_catch_up)
    monkeypatch.setattr(
        controller,
        "_sample_action_from_distribution",
        lambda distribution, sample_value=0.5: ActionCandidate(name="respond", probability=1.0, rationale="forced-respond"),
    )

    payload = controller.run_endogenous_tick(trigger="idle", scenario="companion")
    why_not_payload = controller.why_not(payload["round_id"], "monologue")
    console_payload = controller.console_why_not("monologue", payload["round_id"])

    assert why_not_payload["action"] == "monologue"
    assert why_not_payload["expressive_trace"]["monologue_stream"]["active"] is True
    assert "独白" in why_not_payload["summary"]
    assert "monologue_stream" in console_payload["why_not"]["expressive_trace"]
    assert "独白" in console_payload["why_not"]["summary"]


def test_endogenous_tick_self_run_can_continue_readonly_repo_scan_run(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    state = controller.load_runtime_state()
    state.round_count = 3
    state.body_energy = 0.82
    state.body_state.energy = 0.82
    state.fatigue = 0.18
    state.body_state.fatigue = 0.18
    state.memory_fragments = 0.36
    state.body_state.memory_fragments = 0.36
    state.self_continuity = 0.34
    state.body_state.self_continuity = 0.34
    state.meaning_strength = 0.22
    state.body_state.meaning_strength = 0.22
    state.mode = "idle"
    controller._save_state(state, sync=True)

    controller.start_run(
        "Inspect the repository and identify the next read-only self-check step.",
        allow_commit=False,
        operator_level="read_only",
        defer_bootstrap_tool=True,
    )

    monkeypatch.setattr(
        controller,
        "_sample_action_from_distribution",
        lambda distribution, sample_value=0.5: ActionCandidate(name="self_run", probability=1.0, rationale="forced-self-run"),
    )

    payload = controller.run_endogenous_tick(trigger="idle", scenario="companion")
    trace = controller.trace_round(payload["round_id"])
    run = controller.run_status()

    assert trace["sampled_action"] == "self_run"
    assert trace["rendered_expression"]["text"] == ""
    assert run["last_tool_result"]["tool_name"] == "repo_scan"


def test_collect_parallel_contributions_excludes_self_run_when_only_non_repo_scan_tool_is_pending(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    state = controller.load_runtime_state()
    state.round_count = 3
    state.body_energy = 0.82
    state.body_state.energy = 0.82
    state.fatigue = 0.18
    state.body_state.fatigue = 0.18
    state.memory_fragments = 0.36
    state.body_state.memory_fragments = 0.36
    state.self_continuity = 0.34
    state.body_state.self_continuity = 0.34
    state.meaning_strength = 0.22
    state.body_state.meaning_strength = 0.22
    state.mode = "idle"
    controller._save_state(state, sync=True)

    controller.start_run(
        "Inspect the repository and identify the next read-only self-check step.",
        allow_commit=False,
        operator_level="read_only",
        defer_bootstrap_tool=True,
    )
    run_status = controller.run_status()
    tools = controller.run_tools(run_status["run_id"])["tools"]
    pending_repo_scan = next(item for item in tools if str(item.get("status") or "") == "awaiting_approval")
    controller.resolve_run_tool_approval(run_status["run_id"], str(pending_repo_scan["call_id"]), approved=True)
    controller.trace_store.append_tool_trace(
        {
            "run_id": run_status["run_id"],
            "call_id": "tool-edit-endogenous",
            "tool_name": "edit_file",
            "arguments": {"path": "README.md"},
            "status": "awaiting_approval",
            "trace_ref": f"run://{run_status['run_id']}/tools/tool-edit-endogenous",
            "round_id": None,
        },
        session_id=controller.load_runtime_state().session_id,
        recorded_at=utc_now_iso(),
        sync=True,
    )

    round_context = controller.build_round_context(
        RoundEvent(
            source="endogenous",
            content="endogenous trigger idle",
            target="self",
            cue="endogenous:idle",
        ),
        "companion",
        "endogenous_light",
    )

    collected = controller.collect_parallel_contributions(
        round_context=round_context,
        scenario="companion",
    )

    assert "AutonomySelfRun" not in collected["direct_action_contributions"]


def test_autonomy_self_run_prefers_small_model_goal_when_available(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    route = controller._route_config_for_binding("AutonomySelfRun", route_name="autonomy_self_run")

    assert route is not None
    assert getattr(route, "effective_tier", "") == "small_model"

    state = controller.load_runtime_state()
    state.round_count = 2
    state.body_energy = 0.86
    state.body_state.energy = 0.86
    state.fatigue = 0.14
    state.body_state.fatigue = 0.14
    state.memory_fragments = 0.41
    state.body_state.memory_fragments = 0.41
    state.self_continuity = 0.31
    state.body_state.self_continuity = 0.31
    state.mode = "interactive"
    controller._save_state(state, sync=True)

    monkeypatch.setattr(controller.endogenous_scheduler, "build_trigger", lambda **kwargs: None)

    seen: dict[str, str] = {}

    def fake_generate_config(route_config, request):
        seen["route"] = route_config.name
        seen["tier"] = str(getattr(route_config, "effective_tier", ""))
        return ModelResponse(
            route=route_config.name,
            model="stub-small",
            payload={
                "goal": "Inspect recent runtime traces and identify the next read-only self-check step.",
                "reason": "continuity drift remains elevated",
            },
            backend=route_config.backend,
        )

    def fake_start_run(goal: str, **kwargs):
        seen["goal"] = goal
        run_state = controller.load_runtime_state()
        run_state.active_run_id = "run-autonomy-model"
        run_state.run_status = "running"
        controller._save_state(run_state, sync=True)
        return {
            "run_id": "run-autonomy-model",
            "status": "running",
            "goal": goal,
            "goal_summary": goal,
            "trace_ref": "run://run-autonomy-model",
        }

    monkeypatch.setattr(controller.model_router, "generate_config", fake_generate_config)
    monkeypatch.setattr(controller, "start_run", fake_start_run)

    controller.autonomy_step()

    assert seen["route"] == "autonomy_self_run"
    assert seen["tier"] == "small_model"
    assert seen["goal"] == "Inspect recent runtime traces and identify the next read-only self-check step."


def test_autonomy_step_auto_approves_repo_scan_for_active_readonly_run(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    monkeypatch.setattr(controller.endogenous_scheduler, "build_trigger", lambda **kwargs: None)

    controller.start_run(
        "Inspect the repository and identify the next read-only self-check step.",
        allow_commit=False,
        operator_level="read_only",
        defer_bootstrap_tool=True,
    )

    status = controller.autonomy_step()
    run = controller.run_status()
    tools = controller.run_tools()["tools"]

    assert run["status"] == "running"
    assert run["last_tool_result"]["tool_name"] == "repo_scan"
    assert any(str(item.get("status") or "") == "ok" for item in tools)
    assert status["last_action_type"] == "self_run_tool"
    assert "repo_scan" in status["last_action_summary"]


def test_autonomy_step_does_not_auto_approve_non_repo_scan_tools(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    monkeypatch.setattr(controller.endogenous_scheduler, "build_trigger", lambda **kwargs: None)

    controller.start_run(
        "Inspect the repository and identify the next read-only self-check step.",
        allow_commit=False,
        operator_level="read_only",
        defer_bootstrap_tool=True,
    )
    run_status = controller.run_status()
    tools = controller.run_tools(run_status["run_id"])["tools"]
    pending_repo_scan = next(item for item in tools if str(item.get("status") or "") == "awaiting_approval")
    call_id = str(pending_repo_scan["call_id"])
    controller.resolve_run_tool_approval(run_status["run_id"], call_id, approved=True)
    controller.trace_store.append_tool_trace(
        {
            "run_id": run_status["run_id"],
            "call_id": "tool-edit-1",
            "tool_name": "edit_file",
            "arguments": {"path": "README.md"},
            "status": "awaiting_approval",
            "trace_ref": f"run://{run_status['run_id']}/tools/tool-edit-1",
            "round_id": None,
        },
        session_id=controller.load_runtime_state().session_id,
        recorded_at=utc_now_iso(),
        sync=True,
    )

    status = controller.autonomy_step()
    tools = controller.run_tools()["tools"]

    assert any(
        str(item.get("tool_name") or "") == "edit_file" and str(item.get("status") or "") == "awaiting_approval"
        for item in tools
    )
    assert status["last_action_type"] != "self_run_tool"


def test_autonomy_step_does_not_auto_approve_repo_scan_for_non_readonly_run(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    monkeypatch.setattr(controller.endogenous_scheduler, "build_trigger", lambda **kwargs: None)

    controller.start_run(
        "Inspect the repository and identify the next write-capable step.",
        allow_commit=True,
        operator_level="workspace_write",
        defer_bootstrap_tool=True,
    )

    status = controller.autonomy_step()
    run = controller.run_status()
    tools = controller.run_tools(run["run_id"])["tools"]
    pending_repo_scan = next(item for item in tools if str(item.get("tool_name") or "") == "repo_scan")

    assert run["status"] == "running"
    assert str(pending_repo_scan.get("status") or "") == "awaiting_approval"
    assert status["last_action_type"] != "self_run_tool"


def test_autonomy_execute_command_rejects_blocked_command_with_reason(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    state = controller.load_runtime_state()
    state.autonomy_policy.enabled = True
    state.autonomy_policy.blocked_commands.append("body rest")
    controller._save_state(state, sync=True)

    result = controller._autonomy_execute_command("body rest")

    assert result["allowed"] is False
    assert result["command"] == "body rest"
    assert result["reason"] == "blocked_by_policy"
    assert "body rest" in result["blocked_commands"]


def test_autonomy_start_resets_windowed_budgets_and_expired_window_does_not_stop_loop(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    state = controller.load_runtime_state()
    state.autonomy_policy.enabled = True
    state.autonomy_loop.running = True
    state.autonomy_loop.window_started_at = "2000-01-01T00:00:00+00:00"
    state.autonomy_loop.window_tool_actions = state.autonomy_policy.max_tool_actions_per_hour
    state.autonomy_loop.window_endogenous_rounds = state.autonomy_policy.max_rounds_per_hour
    controller._save_state(state, sync=True)

    started = controller.start_autonomy(profile="tool_level")
    stepped = controller.autonomy_step()

    assert started["budget_usage"]["tool_actions"] == 0
    assert started["budget_usage"]["endogenous_rounds"] == 0
    assert stepped["stop_reason"] != "tool_budget_reached"
    assert stepped["stop_reason"] != "round_budget_reached"


def test_autonomy_step_treats_zero_hourly_budgets_as_unlimited(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    state = controller.load_runtime_state()
    state.autonomy_policy.enabled = True
    state.autonomy_loop.running = True
    state.autonomy_policy.max_rounds_per_hour = 0
    state.autonomy_policy.max_tool_actions_per_hour = 0
    state.autonomy_loop.window_started_at = utc_now_iso()
    state.autonomy_loop.window_tool_actions = 999
    state.autonomy_loop.window_endogenous_rounds = 999
    controller._save_state(state, sync=True)
    monkeypatch.setattr(controller.endogenous_scheduler, "build_trigger", lambda **kwargs: None)
    monkeypatch.setattr(controller, "_autonomy_candidate_action", lambda state, policy: "nothing")

    status = controller.autonomy_step()

    assert status["running"] is True
    assert status["stop_reason"] != "tool_budget_reached"
    assert status["stop_reason"] != "round_budget_reached"
    assert status["budget_usage"]["max_rounds_per_hour"] == 0
    assert status["budget_usage"]["max_tool_actions_per_hour"] == 0


def test_autonomy_start_can_clear_safe_mode_for_observer_resume(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    state = controller.load_runtime_state()
    state.safe_mode = True
    state.mode = "safe"
    controller._save_state(state, sync=True)

    started = controller.start_autonomy(profile="tool_level", clear_safe_mode=True)
    restored = controller.load_runtime_state()

    assert started["running"] is True
    assert restored.safe_mode is False
    assert restored.mode == "interactive"


def test_reset_persona_restores_default_autonomy_boot_policy(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    controller.stop_autonomy(reason="manual_stop")

    restored = controller.reset_persona()

    assert restored.autonomy_policy.enabled is True
    assert restored.autonomy_loop.running is True
    assert restored.session_metadata.get("autonomy_user_disabled") is not True
    assert restored.session_metadata.get("autonomy_default_enabled_at")


def test_console_refresh_payload_surfaces_latency_split_and_internal_labels_for_endogenous_round(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    endogenous = controller.run_endogenous_tick(trigger="idle")
    payload = controller.console_refresh_payload(endogenous["round_id"])
    current_round = payload["state"]["current_round"]
    recent_actions = controller.console_recent_actions(limit=1)
    current_intent = payload["state"]["cognitive_snapshot"]["current_intent"]

    assert current_round["cause_type"] == "endogenous"
    assert str(current_round["route_type"]).startswith("endogenous_")
    assert current_round["total_turn_ms"] >= 1
    assert current_round["model_wait_ms"] >= 0
    assert current_round["local_compute_ms"] >= 0
    assert current_round["latency_dominant"] in {"model_wait", "local_compute", "mixed"}
    assert current_round["mode_label"]
    assert "endogenous_light" not in current_round["mode_label"]
    assert "回应" not in current_intent
    assert "endogenous_light" not in recent_actions["actions"][0]["summary"]


def test_state_payload_exposes_runtime_metrics_as_first_class_diagnostics(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    controller.tick(
        RoundEvent(
            source="user",
            content="给我一个简短近况。",
            target="user",
            cue="近况",
            valence=0.05,
        ),
        scenario="chat",
        mode="interactive",
    )

    payload = controller.state_payload()
    runtime_metrics = payload["runtime_metrics"]
    performance = payload["performance"]

    assert runtime_metrics["route_type"]
    assert runtime_metrics["model_call_count"] >= 0
    assert runtime_metrics["total_turn_ms"] >= 1
    assert runtime_metrics["local_compute_ms"] >= 0
    assert performance["latency"]["local_compute_ms"] == runtime_metrics["local_compute_ms"]
    assert performance["latency"]["latency_dominant"] in {"model_wait", "local_compute", "mixed"}


def test_chat_tick_runtime_metrics_use_chat_standard_route_type(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    result = controller.tick(
        RoundEvent(
            source="user",
            content="最近的内在状态怎么样？",
            target="user",
            cue="状态",
        ),
        scenario="chat",
        mode="interactive",
    )

    assert result.trace.runtime_metrics["route_type"] == "chat_standard"


def test_chat_tick_runtime_metrics_can_upgrade_to_chat_deep_route_type(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    result = controller.tick(
        RoundEvent(
            source="user",
            content="请你详细分析一下我最近总是觉得疲惫、分心、又想推进事情，但每次都卡住的状态，并系统比较一下可能的内在原因和接下来最值得优先做的调整。",
            target="user",
            cue="疲惫与卡住",
        ),
        scenario="chat",
        mode="interactive",
    )

    assert result.trace.runtime_metrics["route_type"] == "chat_deep"


def test_endogenous_tick_runtime_metrics_use_endogenous_light_route_type(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    payload = controller.run_endogenous_tick(trigger="idle", mode="endogenous_light")
    trace = controller.trace_round(payload["round_id"])

    assert trace["runtime_metrics"]["route_type"] == "endogenous_light"


def test_endogenous_replay_runtime_metrics_use_endogenous_deep_route_type(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    controller.tick(
        RoundEvent(source="user", content="先留下一轮上下文。", target="user", cue="上下文"),
        scenario="chat",
        mode="interactive",
    )
    payload = controller.run_endogenous_tick(trigger="idle", mode="endogenous_replay")
    trace = controller.trace_round(payload["round_id"])

    assert trace["runtime_metrics"]["route_type"] == "endogenous_deep"


def test_probability_field_observability_surfaces_tool_affordance_for_active_run(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    controller.start_run("检查 worker.py 并规划下一步")

    controller.tick(
        RoundEvent(
            source="user",
            content="继续检查 worker.py，并告诉我下一步怎么做。",
            target="user",
            cue="worker.py",
            valence=0.02,
        ),
        scenario="task",
        mode="interactive",
    )

    why_payload = controller.why_this(1)
    heatmap_payload = controller.metrics_heatmap()

    explanation = why_payload["action_probability_explanation"]
    assert any(item["module_name"] == "SkillExecutor" for item in explanation["stacked_contributions"])
    assert any(
        "SkillExecutor" in modules
        for modules in heatmap_payload["action_module_heatmap"].values()
    )


def test_compact_traces_and_eval_longrun_surface_diagnostics(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    controller.tick(
        RoundEvent(source="user", content="Help me remember tea and plan a reply.", target="friend", cue="tea"),
        scenario="chat",
        mode="interactive",
    )

    compacted = controller.compact_traces()
    summary = controller.eval_longrun(rounds=5)

    assert compacted["parquet_path"].endswith("round_trace.parquet")
    assert compacted["tables"]["round_canonical"]["row_count"] >= 1
    conn = duckdb.connect()
    try:
        round_rows = conn.execute(
            "select probability_field_json, token_state_json, cross_layer_coupling_verdict_json, renderer_decision_integrity_json, memory_write_gate_json, conflict_arbitration_json from read_parquet(?) limit 1",
            [compacted["tables"]["round_trace"]["path"]],
        ).fetchall()
        canonical_rows = conn.execute(
            "select payload_json from read_parquet(?) limit 1",
            [compacted["tables"]["round_canonical"]["path"]],
        ).fetchall()
    finally:
        conn.close()
    assert round_rows
    assert json.loads(round_rows[0][0])["action"]["contribution_audit"]
    assert json.loads(round_rows[0][1])["step_index"] >= 1
    assert json.loads(round_rows[0][2])["legal"] is True
    assert json.loads(round_rows[0][3])["renderer_consumes_final_field"] is True
    assert "suppressed" in json.loads(round_rows[0][4])
    assert "winning_priority" in json.loads(round_rows[0][5])
    assert "compromise" in json.loads(round_rows[0][5])
    assert json.loads(canonical_rows[0][0])["probability_field"]["action"]["contribution_audit"]
    assert summary["generated_rounds"] == 5
    assert "crash_rate" in summary
    assert "task_success_rate" in summary
    assert summary["probability_field_coverage"] > 0.0
    assert summary["action_audit_coverage"] > 0.0
    assert summary["token_audit_coverage"] > 0.0
    assert summary["renderer_token_coverage"] > 0.0
    assert summary["token_source_integrity_rate"] == 1.0
    assert "memory_write_gate" in summary
    assert "conflict_arbitration" in summary
    assert summary["conflict_arbitration"]["observed_round_rate"] == 1.0
    assert "cross_layer_coupling" in summary


def test_export_trace_parquet_exposes_long_run_projection_json_columns(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    controller.tick(
        RoundEvent(source="user", content="Help me remember tea and plan a reply.", target="friend", cue="tea"),
        scenario="chat",
        mode="interactive",
    )

    exported = controller.export_trace_parquet(overwrite=True)
    round_path = exported["tables"]["round_trace"]["path"]

    conn = duckdb.connect()
    try:
        row = conn.execute(
            """
            select long_run_projection_json, long_run_projection_online_prior_json
            from read_parquet(?)
            limit 1
            """,
            [round_path],
        ).fetchone()
    finally:
        conn.close()

    assert row is not None
    long_run_projection = json.loads(row[0])
    online_prior = json.loads(row[1])
    assert "online_prior" in long_run_projection
    assert online_prior["self_consistency_score"] == long_run_projection["online_prior"]["self_consistency_score"]
