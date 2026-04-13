from pathlib import Path

from nalr.runtime.controller import RuntimeController
from nalr.schemas.models import CommandResult, RoundEvent


CONFIG_ROOT = Path(__file__).resolve().parents[2] / "config"


def test_runtime_controller_initializes_domain_runtimes(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    assert controller.round_pipeline.controller is controller
    assert controller.run_runtime.controller is controller
    assert controller.observer_runtime.controller is controller
    assert controller.model_routing_runtime.controller is controller
    assert controller.model_payload_runtime.controller is controller
    assert controller.observer_settings_runtime.project_root == tmp_path


def test_tick_delegates_to_round_pipeline(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    sentinel = object()
    seen: dict[str, object] = {}

    class FakeRoundPipeline:
        def tick(self, event, scenario: str, mode: str):
            seen["event"] = event
            seen["scenario"] = scenario
            seen["mode"] = mode
            return sentinel

    monkeypatch.setattr(controller, "round_pipeline", FakeRoundPipeline(), raising=False)
    event = RoundEvent(source="user", content="delegate tick", target="user", cue="delegate")

    result = controller.tick(event, scenario="chat", mode="interactive")

    assert result is sentinel
    assert seen == {
        "event": event,
        "scenario": "chat",
        "mode": "interactive",
    }


def test_start_run_delegates_to_run_runtime(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    sentinel = {"run_id": "delegated", "status": "running"}
    seen: dict[str, object] = {}

    class FakeRunRuntime:
        def start_run(self, goal: str, **kwargs):
            seen["goal"] = goal
            seen.update(kwargs)
            return sentinel

    monkeypatch.setattr(controller, "run_runtime", FakeRunRuntime(), raising=False)

    payload = controller.start_run("delegate run", operator_level="debug_control")

    assert payload is sentinel
    assert seen["goal"] == "delegate run"
    assert seen["operator_level"] == "debug_control"
    assert seen["allow_commit"] is False


def test_observer_settings_facades_delegate_to_observer_settings_runtime(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    seen: dict[str, object] = {}
    defaults = {"defaults": True}
    loaded = {"loaded": True}
    normalized = {"normalized": True}
    model_config = {"model": True}
    tiers = {"medium_model", "state_machine"}
    string_list = ["alpha", "beta"]
    path_value = str(tmp_path / "workspace")
    path_list = [path_value]
    merged_models = {"model_routes": {"renderer": {"model": "delegated"}}}
    current = {"current": True}

    class FakeObserverSettingsRuntime:
        project_root = tmp_path

        def default_settings(self):
            seen["default_settings"] = True
            return defaults

        def load_settings_file(self):
            seen["load_settings_file"] = True
            return loaded

        def normalize_settings(self, payload, *, base=None, current_config=None):
            seen["normalize_settings"] = {
                "payload": payload,
                "base": base,
                "current_config": current_config,
            }
            return normalized

        def normalize_model_config(self, payload, *, tier_mode):
            seen["normalize_model_config"] = {
                "payload": payload,
                "tier_mode": tier_mode,
            }
            return model_config

        def known_model_tier_names(self, *, observer_model_tiers=None, current_config=None):
            seen["known_model_tier_names"] = {
                "observer_model_tiers": observer_model_tiers,
                "current_config": current_config,
            }
            return tiers

        def normalize_string_list(self, values):
            seen["normalize_string_list"] = values
            return string_list

        def normalize_path(self, value):
            seen["normalize_path"] = value
            return path_value

        def normalize_path_list(self, values):
            seen["normalize_path_list"] = values
            return path_list

        def apply_settings_to_models(self, models_cfg, observer_settings):
            seen["apply_settings_to_models"] = {
                "models_cfg": models_cfg,
                "observer_settings": observer_settings,
            }
            return merged_models

        def current_settings(self, config=None):
            seen["current_settings"] = config
            return current

    monkeypatch.setattr(controller, "observer_settings_runtime", FakeObserverSettingsRuntime(), raising=False)

    payload = {"models": {"module_model_bindings": {"deep_renderer": "medium_model"}}}
    base = {"models": {"module_model_bindings": {}}}
    model_payload = {"backend": "openai_compatible"}
    observer_model_tiers = {"medium_model": {"enabled": True}}
    models_cfg = {"model_routes": {}}

    assert controller._default_observer_settings() is defaults
    assert controller._load_observer_settings_file() is loaded
    assert controller._normalize_observer_settings(payload, base=base) is normalized
    assert controller._normalize_model_config(model_payload, tier_mode=True) is model_config
    assert controller._known_model_tier_names(observer_model_tiers=observer_model_tiers) == tiers
    assert controller._normalize_observer_string_list([" alpha ", "beta"]) == string_list
    assert controller._normalize_observer_path("workspace") == path_value
    assert controller._normalize_observer_path_list(["workspace"]) == path_list
    assert controller._apply_observer_settings_to_models(models_cfg, payload) is merged_models
    assert controller._observer_settings() is current
    assert controller.observer_settings_current() is current
    assert controller.normalize_observer_settings_payload(payload, base=base) is normalized

    assert seen["default_settings"] is True
    assert seen["load_settings_file"] is True
    assert seen["normalize_settings"] == {
        "payload": payload,
        "base": base,
        "current_config": controller.config,
    }
    assert seen["normalize_model_config"] == {
        "payload": model_payload,
        "tier_mode": True,
    }
    assert seen["known_model_tier_names"] == {
        "observer_model_tiers": observer_model_tiers,
        "current_config": controller.config,
    }
    assert seen["normalize_string_list"] == [" alpha ", "beta"]
    assert seen["normalize_path"] == "workspace"
    assert seen["normalize_path_list"] == ["workspace"]
    assert seen["apply_settings_to_models"] == {
        "models_cfg": models_cfg,
        "observer_settings": payload,
    }
    assert seen["current_settings"] == controller.config


def test_trace_round_delegates_to_observer_runtime(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    controller.tick(
        RoundEvent(
            source="user",
            content="seed one round before trace delegation",
            target="user",
            cue="seed",
        ),
        scenario="chat",
        mode="interactive",
    )
    sentinel = {"round_id": 1, "delegated": True}
    seen: dict[str, object] = {}

    class FakeObserverRuntime:
        def trace_round(self, round_ref):
            seen["round_ref"] = round_ref
            return sentinel

    monkeypatch.setattr(controller, "observer_runtime", FakeObserverRuntime(), raising=False)

    payload = controller.trace_round(1)

    assert payload is sentinel
    assert seen == {"round_ref": 1}


def test_cognitive_and_storage_facades_delegate_to_observer_runtime(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    seen: dict[str, object] = {}
    migration = {"migration": True}
    storage = {"storage": True}
    snapshot = {"snapshot": True}
    learning = {"learning": True}

    class FakeObserverRuntime:
        def migration_report(self):
            seen["migration_report"] = True
            return migration

        def runtime_storage_status(self):
            seen["runtime_storage_status"] = True
            return storage

        def cognitive_snapshot(self, *, state=None, run_payload=None):
            seen["cognitive_snapshot"] = {
                "state": state,
                "run_payload": run_payload,
            }
            return snapshot

        def internal_learning_summary(self, state=None):
            seen["internal_learning_summary"] = state
            return learning

        def _action_phrase(self, action_name: str):
            seen["_action_phrase"] = action_name
            return "delegated action"

        def _focus_label(self, focus: str):
            seen["_focus_label"] = focus
            return "delegated focus"

    monkeypatch.setattr(controller, "observer_runtime", FakeObserverRuntime(), raising=False)

    assert controller.migration_report() is migration
    assert controller.runtime_storage_status() is storage
    assert controller.cognitive_snapshot(run_payload={"goal": "delegate"}) is snapshot
    assert controller.internal_learning_summary() is learning
    assert controller._action_phrase("die") == "delegated action"
    assert controller._focus_label("die") == "delegated focus"
    assert seen["migration_report"] is True
    assert seen["runtime_storage_status"] is True
    assert seen["cognitive_snapshot"] == {"state": None, "run_payload": {"goal": "delegate"}}
    assert seen["internal_learning_summary"] is None
    assert seen["_action_phrase"] == "die"
    assert seen["_focus_label"] == "die"


def test_state_bootstrap_facades_delegate_to_state_runtime(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    seen: dict[str, object] = {}
    newborn = controller.load_runtime_state()
    restored = controller.load_runtime_state()

    class FakeStateRuntime:
        def newborn_runtime_state(self):
            seen["newborn_runtime_state"] = True
            return newborn

        def rebind_runtime_storage(self):
            seen["rebind_runtime_storage"] = True

        def reset_persona(self):
            seen["reset_persona"] = True
            return restored

    monkeypatch.setattr(controller, "state_runtime", FakeStateRuntime(), raising=False)

    assert controller.newborn_runtime_state() is newborn
    assert controller._rebind_runtime_storage() is None
    assert controller.reset_persona() is restored
    assert seen == {
        "newborn_runtime_state": True,
        "rebind_runtime_storage": True,
        "reset_persona": True,
    }


def test_initiative_and_monologue_bucket_facades_delegate_to_run_runtime(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    state = controller.load_runtime_state()
    seen: dict[str, object] = {}
    initiative_bucket = {"settings": {"enabled": True}}
    monologue_bucket = {"last_viewed_at": "2026-04-11T00:00:00Z"}
    monologue_advance = {"committed": False, "reason": "delegated"}
    initiative_advance = {"committed": True, "reason": ""}

    class FakeRunRuntime:
        def _initiative_state_bucket(self, state_value):
            seen["_initiative_state_bucket"] = state_value
            return initiative_bucket

        def _monologue_state_bucket(self, state_value):
            seen["_monologue_state_bucket"] = state_value
            return monologue_bucket

        def advance_monologue_stream_background(self, *, now_iso=None):
            seen["advance_monologue_stream_background"] = now_iso
            return monologue_advance

        def advance_initiative_background(self, *, latest_recorded_at=None):
            seen["advance_initiative_background"] = latest_recorded_at
            return initiative_advance

    monkeypatch.setattr(controller, "run_runtime", FakeRunRuntime(), raising=False)

    assert controller._initiative_state_bucket(state) is initiative_bucket
    assert controller._monologue_state_bucket(state) is monologue_bucket
    assert controller.advance_monologue_stream_background(now_iso="2026-04-11T00:00:00Z") is monologue_advance
    assert controller.advance_initiative_background(latest_recorded_at="2026-04-11T00:05:00Z") is initiative_advance
    assert seen == {
        "_initiative_state_bucket": state,
        "_monologue_state_bucket": state,
        "advance_monologue_stream_background": "2026-04-11T00:00:00Z",
        "advance_initiative_background": "2026-04-11T00:05:00Z",
    }


def test_initiative_and_monologue_helpers_delegate_to_run_runtime(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    state = controller.load_runtime_state()
    seen: dict[str, object] = {}
    backing = {"cue": "topic"}
    distribution = {"should_send": False}
    delivery = "anchored delivery"
    context_backing = {"cue": "context"}
    pressure = {"score": 0.42}
    sync_payload = ({"generated_total": 1}, [{"fragment_id": "f1"}])
    contribution = (None, {"active": False})

    class FakeRunRuntime:
        def _initiative_memory_backing(self, cue=None, **kwargs):
            seen["_initiative_memory_backing"] = {"cue": cue, **kwargs}
            return backing

        def _initiative_distribution_payload(self, state_value, **kwargs):
            seen["_initiative_distribution_payload"] = {"state": state_value, **kwargs}
            return distribution

        def _initiative_delivery_text(self, *, proposal, message):
            seen["_initiative_delivery_text"] = {"proposal": proposal, "message": message}
            return delivery

        def _initiative_context_memory_backing(self, *, cue, recall_strength, fallback_summary):
            seen["_initiative_context_memory_backing"] = {
                "cue": cue,
                "recall_strength": recall_strength,
                "fallback_summary": fallback_summary,
            }
            return context_backing

        def _monologue_stream_pressure_payload(self, *, recent_fragments, generated):
            seen["_monologue_stream_pressure_payload"] = {
                "recent_fragments": recent_fragments,
                "generated": generated,
            }
            return pressure

        def _monologue_stream_pressure_score(self, state_value):
            seen["_monologue_stream_pressure_score"] = state_value
            return 0.42

        def _sync_monologue_stream(self, state_value, *, mark_viewed=False, generate_if_due=True):
            seen["_sync_monologue_stream"] = {
                "state": state_value,
                "mark_viewed": mark_viewed,
                "generate_if_due": generate_if_due,
            }
            return sync_payload

        def _build_monologue_stream_action_contribution(self, *, state, scenario, endogenous_turn):
            seen["_build_monologue_stream_action_contribution"] = {
                "state": state,
                "scenario": scenario,
                "endogenous_turn": endogenous_turn,
            }
            return contribution

    monkeypatch.setattr(controller, "run_runtime", FakeRunRuntime(), raising=False)

    assert controller._initiative_memory_backing("topic") is backing
    assert controller._initiative_distribution_payload(state, cue="topic") is distribution
    assert controller._initiative_delivery_text(proposal={"memory_backing": {}}, message="hi") == delivery
    assert controller._initiative_context_memory_backing(cue="topic", recall_strength=0.4, fallback_summary="fb") is context_backing
    assert controller._monologue_stream_pressure_payload(recent_fragments=[{"content": "x"}], generated=[]) is pressure
    assert controller._monologue_stream_pressure_score(state) == 0.42
    assert controller._sync_monologue_stream(state, mark_viewed=True, generate_if_due=False) == sync_payload
    assert controller._build_monologue_stream_action_contribution(state=state, scenario="chat", endogenous_turn=True) == contribution
    assert seen["_initiative_memory_backing"]["cue"] == "topic"
    assert seen["_initiative_distribution_payload"]["state"] is state
    assert seen["_initiative_distribution_payload"]["cue"] == "topic"
    assert seen["_initiative_delivery_text"] == {"proposal": {"memory_backing": {}}, "message": "hi"}
    assert seen["_initiative_context_memory_backing"]["cue"] == "topic"
    assert seen["_monologue_stream_pressure_payload"]["recent_fragments"] == [{"content": "x"}]
    assert seen["_monologue_stream_pressure_score"] is state
    assert seen["_sync_monologue_stream"] == {"state": state, "mark_viewed": True, "generate_if_due": False}
    assert seen["_build_monologue_stream_action_contribution"] == {
        "state": state,
        "scenario": "chat",
        "endogenous_turn": True,
    }


def test_expressive_action_helpers_delegate_to_round_pipeline(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    seen: dict[str, object] = {}
    payload = {"top_intent": "share_memory"}
    delta = {"recall": 0.12}
    contribution = object()

    class FakeRoundPipeline:
        def _expressive_action_biases(self, payload_value, *, scenario, endogenous_turn):
            seen["_expressive_action_biases"] = {
                "payload": payload_value,
                "scenario": scenario,
                "endogenous_turn": endogenous_turn,
            }
            return delta

        def _build_expressive_action_contribution(
            self,
            *,
            current_distribution,
            payload,
            scenario,
            endogenous_turn,
        ):
            seen["_build_expressive_action_contribution"] = {
                "current_distribution": current_distribution,
                "payload": payload,
                "scenario": scenario,
                "endogenous_turn": endogenous_turn,
            }
            return contribution

    monkeypatch.setattr(controller, "round_pipeline", FakeRoundPipeline(), raising=False)

    assert controller._expressive_action_biases(payload, scenario="chat", endogenous_turn=False) == delta
    assert controller._build_expressive_action_contribution(
        current_distribution={"respond": 0.5},
        payload=payload,
        scenario="task",
        endogenous_turn=True,
    ) is contribution
    assert seen["_expressive_action_biases"] == {
        "payload": payload,
        "scenario": "chat",
        "endogenous_turn": False,
    }
    assert seen["_build_expressive_action_contribution"] == {
        "current_distribution": {"respond": 0.5},
        "payload": payload,
        "scenario": "task",
        "endogenous_turn": True,
    }


def test_renderer_fallback_helpers_delegate_to_round_pipeline(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    seen: dict[str, object] = {}
    render_plan = object()
    deterministic_output = {"text": "fallback", "route": "renderer", "model": "fallback"}
    payload = {"fallback_reason": "renderer_failure"}
    traces = [{"route": "renderer_fallback_fast"}]

    class FakeRoundPipeline:
        def _fallback_render_route_names(self):
            seen["_fallback_render_route_names"] = True
            return ["renderer_fallback_fast"]

        def _build_deterministic_render_fallback_output(self, render_plan_value, *, model_label):
            seen["_build_deterministic_render_fallback_output"] = {
                "render_plan": render_plan_value,
                "model_label": model_label,
            }
            return deterministic_output

        def _fallback_renderer_system_prompt(
            self,
            render_plan_value,
            *,
            fallback_reason,
            violation_types=None,
        ):
            seen["_fallback_renderer_system_prompt"] = {
                "render_plan": render_plan_value,
                "fallback_reason": fallback_reason,
                "violation_types": violation_types,
            }
            return "delegated system prompt"

        def _build_fallback_render_model_payload(
            self,
            render_plan_value,
            *,
            fallback_reason,
            violation_types=None,
        ):
            seen["_build_fallback_render_model_payload"] = {
                "render_plan": render_plan_value,
                "fallback_reason": fallback_reason,
                "violation_types": violation_types,
            }
            return payload

        def _render_expression_via_humanized_fallback_chain(
            self,
            render_plan_value,
            *,
            fallback_reason,
            deterministic_model,
            violation_types=None,
            model_call_traces=None,
        ):
            seen["_render_expression_via_humanized_fallback_chain"] = {
                "render_plan": render_plan_value,
                "fallback_reason": fallback_reason,
                "deterministic_model": deterministic_model,
                "violation_types": violation_types,
                "model_call_traces": model_call_traces,
            }
            return deterministic_output

    monkeypatch.setattr(controller, "round_pipeline", FakeRoundPipeline(), raising=False)

    assert controller._fallback_render_route_names() == ["renderer_fallback_fast"]
    assert controller._build_deterministic_render_fallback_output(render_plan, model_label="fallback") is deterministic_output
    assert controller._fallback_renderer_system_prompt(
        render_plan,
        fallback_reason="renderer_failure",
        violation_types=["style"],
    ) == "delegated system prompt"
    assert controller._build_fallback_render_model_payload(
        render_plan,
        fallback_reason="auth_guard",
        violation_types=["persona"],
    ) is payload
    assert controller._render_expression_via_humanized_fallback_chain(
        render_plan,
        fallback_reason="renderer_failure",
        deterministic_model="fallback",
        violation_types=["style"],
        model_call_traces=traces,
    ) is deterministic_output

    assert seen["_fallback_render_route_names"] is True
    assert seen["_build_deterministic_render_fallback_output"] == {
        "render_plan": render_plan,
        "model_label": "fallback",
    }
    assert seen["_fallback_renderer_system_prompt"] == {
        "render_plan": render_plan,
        "fallback_reason": "renderer_failure",
        "violation_types": ["style"],
    }
    assert seen["_build_fallback_render_model_payload"] == {
        "render_plan": render_plan,
        "fallback_reason": "auth_guard",
        "violation_types": ["persona"],
    }
    assert seen["_render_expression_via_humanized_fallback_chain"] == {
        "render_plan": render_plan,
        "fallback_reason": "renderer_failure",
        "deterministic_model": "fallback",
        "violation_types": ["style"],
        "model_call_traces": traces,
    }


def test_parallel_skill_helpers_delegate_to_round_pipeline(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    runtime_context = controller._skill_runtime_context(1, "chat", controller.load_runtime_state())
    seen: dict[str, object] = {}
    outputs = {"guard": {"flag": False}}
    timeout_result = object()
    task = {
        "name": "guard",
        "skill_name": "request_second_sampling",
        "inputs": {"fail_score": 0.4, "attempts": 0},
        "provider": lambda fail_score, attempts: {"flag": fail_score > 0.2},
    }
    skill_traces: list[dict[str, object]] = []
    parallel_traces: list[dict[str, object]] = []

    class FakeRoundPipeline:
        def _execute_parallel_skills(
            self,
            *,
            round_id,
            tasks,
            skill_traces,
            runtime_context,
            parallel_traces=None,
        ):
            seen["_execute_parallel_skills"] = {
                "round_id": round_id,
                "tasks": tasks,
                "skill_traces": skill_traces,
                "runtime_context": runtime_context,
                "parallel_traces": parallel_traces,
            }
            return outputs

        def _parallel_fallback_output(self, task_value):
            seen["_parallel_fallback_output"] = task_value
            return {"flag": False}

        def _parallel_timeout_skill_result(
            self,
            *,
            round_id,
            task,
            latency_ms,
            failure_policy="parallel_timeout",
        ):
            seen["_parallel_timeout_skill_result"] = {
                "round_id": round_id,
                "task": task,
                "latency_ms": latency_ms,
                "failure_policy": failure_policy,
            }
            return timeout_result

    monkeypatch.setattr(controller, "round_pipeline", FakeRoundPipeline(), raising=False)

    assert controller._execute_parallel_skills(
        round_id=1,
        tasks=[task],
        skill_traces=skill_traces,
        runtime_context=runtime_context,
        parallel_traces=parallel_traces,
    ) is outputs
    assert controller._parallel_fallback_output(task) == {"flag": False}
    assert controller._parallel_timeout_skill_result(
        round_id=1,
        task=task,
        latency_ms=42,
        failure_policy="parallel_exception_fallback",
    ) is timeout_result

    assert seen["_execute_parallel_skills"] == {
        "round_id": 1,
        "tasks": [task],
        "skill_traces": skill_traces,
        "runtime_context": runtime_context,
        "parallel_traces": parallel_traces,
    }
    assert seen["_parallel_fallback_output"] is task
    assert seen["_parallel_timeout_skill_result"] == {
        "round_id": 1,
        "task": task,
        "latency_ms": 42,
        "failure_policy": "parallel_exception_fallback",
    }


def test_action_signal_helpers_delegate_to_round_pipeline(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    state = controller.load_runtime_state()
    seen: dict[str, object] = {}
    contribution = object()
    signal = object()
    metadata = {"priority_bucket": "task_goal"}
    projected = {"respond": 0.2}

    class FakeRoundPipeline:
        def _action_evidence_from_contribution(
            self,
            contribution_value,
            *,
            priority_bucket=None,
            control_domain=None,
            gated_actions=None,
            risk_hints=None,
            veto=False,
            trace_tags=None,
        ):
            seen["_action_evidence_from_contribution"] = {
                "contribution": contribution_value,
                "priority_bucket": priority_bucket,
                "control_domain": control_domain,
                "gated_actions": gated_actions,
                "risk_hints": risk_hints,
                "veto": veto,
                "trace_tags": trace_tags,
            }
            return signal

        def _projected_action_delta_from_contribution(self, contribution_value):
            seen["_projected_action_delta_from_contribution"] = contribution_value
            return projected

        def _action_signal_metadata(self, **kwargs):
            seen["_action_signal_metadata"] = kwargs
            return metadata

        def _coerce_action_signals(self, rows):
            seen["_coerce_action_signals"] = rows
            return [signal]

        def _build_value_action_contribution(self, value_scores):
            seen["_build_value_action_contribution"] = value_scores
            return signal

    monkeypatch.setattr(controller, "round_pipeline", FakeRoundPipeline(), raising=False)

    assert controller._action_evidence_from_contribution(
        contribution,
        priority_bucket="body_safety",
        control_domain="body",
        gated_actions=["plan"],
        risk_hints={"overload": 0.8},
        veto=True,
        trace_tags=["resource"],
    ) is signal
    assert controller._projected_action_delta_from_contribution(contribution) is projected
    assert controller._action_signal_metadata(
        owner="ResourceAgent",
        state=state,
        relation_state={"relationship_risk": 0.3, "boundary_level": 0.1},
        context={"habit_strength": 0.2},
        module_type="resource",
        priority_bucket="budget_overload",
        control_domain="resource",
        projected_delta={"plan": 0.4},
        utility_shift={"plan": 0.4},
        gated_actions=["plan"],
        risk_hints={"overload": 0.8},
        veto=False,
        trace_tags=["resource"],
    ) is metadata
    assert controller._coerce_action_signals([contribution]) == [signal]
    assert controller._build_value_action_contribution({"scores": {"plan": 0.4}}) is signal

    assert seen["_action_evidence_from_contribution"] == {
        "contribution": contribution,
        "priority_bucket": "body_safety",
        "control_domain": "body",
        "gated_actions": ["plan"],
        "risk_hints": {"overload": 0.8},
        "veto": True,
        "trace_tags": ["resource"],
    }
    assert seen["_projected_action_delta_from_contribution"] is contribution
    assert seen["_action_signal_metadata"]["owner"] == "ResourceAgent"
    assert seen["_action_signal_metadata"]["state"] is state
    assert seen["_coerce_action_signals"] == [contribution]
    assert seen["_build_value_action_contribution"] == {"scores": {"plan": 0.4}}


def test_probability_field_helpers_delegate_to_round_pipeline(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    state = controller.load_runtime_state()
    event = RoundEvent(source="user", content="delegate field", target="user", cue="field")
    seen: dict[str, object] = {}
    contribution = object()
    snapshot = object()
    token_state = object()
    couplings = [object()]
    render_plan = object()

    class FakeRoundPipeline:
        def _candidate_distribution_from_trace(self, trace):
            seen["_candidate_distribution_from_trace"] = trace
            return {"plan": 0.7}

        def _probability_field_couplings(self):
            seen["_probability_field_couplings"] = True
            return couplings

        def _collect_probability_field_contributions(self, **kwargs):
            seen["_collect_probability_field_contributions"] = kwargs
            return [contribution], token_state

        def _integrate_probability_field_snapshot(self, **kwargs):
            seen["_integrate_probability_field_snapshot"] = kwargs
            return snapshot, [contribution], token_state

    monkeypatch.setattr(controller, "round_pipeline", FakeRoundPipeline(), raising=False)

    assert controller._candidate_distribution_from_trace({"winner_posterior": {"plan": 0.7}}) == {"plan": 0.7}
    assert controller._probability_field_couplings() is couplings
    assert controller._collect_probability_field_contributions(
        direct_action_contributions={"PFCAgent": contribution},
        render_plan=render_plan,
        event=event,
        state=state,
        scenario_cfg={"pfc_base_share": 0.2},
        context={"cue": "field"},
        identity_context=None,
        authenticity=None,
        vitality_snapshot={"energy": 0.5},
        long_run_contribution=None,
        include_conflict=True,
        include_token=True,
        control_ledger={"conflict": {}},
        action_truth={"winner_posterior": {"plan": 0.7}},
    ) == ([contribution], token_state)
    assert controller._integrate_probability_field_snapshot(
        direct_action_contributions={"PFCAgent": contribution},
        action_base={"plan": 0.7},
        event=event,
        state=state,
        scenario_cfg={"pfc_base_share": 0.2},
        context={"cue": "field"},
        identity_context=None,
        authenticity=None,
        vitality_snapshot={"energy": 0.5},
        long_run_contribution=None,
        include_conflict=False,
        include_token=False,
        render_plan=render_plan,
        source_chain=["field_native"],
        control_ledger={"gate": {"plan": 1.0}},
        action_truth={"winner_posterior": {"plan": 0.7}},
    ) == (snapshot, [contribution], token_state)

    assert seen["_candidate_distribution_from_trace"] == {"winner_posterior": {"plan": 0.7}}
    assert seen["_probability_field_couplings"] is True
    assert seen["_collect_probability_field_contributions"]["state"] is state
    assert seen["_collect_probability_field_contributions"]["include_conflict"] is True
    assert seen["_integrate_probability_field_snapshot"]["action_base"] == {"plan": 0.7}
    assert seen["_integrate_probability_field_snapshot"]["source_chain"] == ["field_native"]


def test_action_bookkeeping_helpers_delegate_to_round_pipeline(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    seen: dict[str, object] = {}
    bookkeeping = object()
    action_layer = object()
    snapshot = object()
    token_state = object()
    stochastic = object()
    contribution = object()
    control_ledger = {"gate": {"plan": 1.0}}
    action_truth = {"winner_posterior": {"plan": 0.8}}

    class FakeRoundPipeline:
        def _finalize_action_bookkeeping_from_action_layer(self, action_bookkeeping, action_layer_value, *, finalize_stage):
            seen["_finalize_action_bookkeeping_from_action_layer"] = {
                "action_bookkeeping": action_bookkeeping,
                "action_layer": action_layer_value,
                "finalize_stage": finalize_stage,
            }

        def _reintegrate_probability_snapshot(self, **kwargs):
            seen["_reintegrate_probability_snapshot"] = kwargs
            return snapshot

        def _action_truth_from_field(self, action_layer_value, *, gate=None, conflict_mode="field_native"):
            seen["_action_truth_from_field"] = {
                "action_layer": action_layer_value,
                "gate": gate,
                "conflict_mode": conflict_mode,
            }
            return action_truth

        def _refresh_action_truth(self, action_layer_value, current_action_truth=None, *, conflict_mode="field_native"):
            seen["_refresh_action_truth"] = {
                "action_layer": action_layer_value,
                "current_action_truth": current_action_truth,
                "conflict_mode": conflict_mode,
            }
            return action_truth

        def _control_ledger_from_action_bookkeeping(self, action_bookkeeping):
            seen["_control_ledger_from_action_bookkeeping"] = action_bookkeeping
            return control_ledger

        def _merge_control_ledger_into_action_bookkeeping(self, action_bookkeeping, control_ledger_value):
            seen["_merge_control_ledger_into_action_bookkeeping"] = {
                "action_bookkeeping": action_bookkeeping,
                "control_ledger": control_ledger_value,
            }

        def _action_bookkeeping_payload(
            self,
            *,
            action_bookkeeping,
            probability_field_snapshot,
            control_ledger,
            action_truth,
            stochastic_state,
        ):
            seen["_action_bookkeeping_payload"] = {
                "action_bookkeeping": action_bookkeeping,
                "probability_field_snapshot": probability_field_snapshot,
                "control_ledger": control_ledger,
                "action_truth": action_truth,
                "stochastic_state": stochastic_state,
            }
            return {"gate": {"plan": 1.0}}

        def _build_distribution_delta_contribution(self, **kwargs):
            seen["_build_distribution_delta_contribution"] = kwargs
            return contribution

    monkeypatch.setattr(controller, "round_pipeline", FakeRoundPipeline(), raising=False)

    assert controller._finalize_action_bookkeeping_from_action_layer(bookkeeping, action_layer, finalize_stage="final") is None
    assert controller._reintegrate_probability_snapshot(
        snapshot=snapshot,
        contributions=[contribution],
        token_state=token_state,
        source_chain=["after_guard"],
    ) is snapshot
    assert controller._action_truth_from_field(action_layer, gate={"plan": 1.0}, conflict_mode="repair") is action_truth
    assert controller._refresh_action_truth(action_layer, current_action_truth=action_truth, conflict_mode="repair") is action_truth
    assert controller._control_ledger_from_action_bookkeeping(bookkeeping) is control_ledger
    assert controller._merge_control_ledger_into_action_bookkeeping(bookkeeping, control_ledger) is None
    assert controller._action_bookkeeping_payload(
        action_bookkeeping=bookkeeping,
        probability_field_snapshot=snapshot,
        control_ledger=control_ledger,
        action_truth=action_truth,
        stochastic_state=stochastic,
    ) == {"gate": {"plan": 1.0}}
    assert controller._build_distribution_delta_contribution(
        module_name="ProbeGuard",
        module_type="guard",
        from_distribution={"plan": 0.4},
        to_distribution={"plan": 0.8},
        trace_reason="guard",
        projection_reason="guard",
        applied_at_stage="probe_guard",
        native_operator="probe_gate",
        dependency_trace=["scenario:task"],
        hard_mask={"respond": True},
        confidence=0.9,
        module_temperature=0.8,
    ) is contribution

    assert seen["_finalize_action_bookkeeping_from_action_layer"]["finalize_stage"] == "final"
    assert seen["_reintegrate_probability_snapshot"]["source_chain"] == ["after_guard"]
    assert seen["_action_truth_from_field"]["conflict_mode"] == "repair"
    assert seen["_refresh_action_truth"]["current_action_truth"] is action_truth
    assert seen["_control_ledger_from_action_bookkeeping"] is bookkeeping
    assert seen["_merge_control_ledger_into_action_bookkeeping"]["control_ledger"] is control_ledger
    assert seen["_action_bookkeeping_payload"]["stochastic_state"] is stochastic
    assert seen["_build_distribution_delta_contribution"]["module_name"] == "ProbeGuard"


def test_model_decision_and_render_helpers_delegate_to_round_pipeline(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    state = controller.load_runtime_state()
    event = RoundEvent(source="user", content="delegate model", target="user", cue="model")
    render_plan = object()
    seen: dict[str, object] = {}
    contribution = object()
    payload = {"scores": {"plan": 0.4}}
    traces = [{"route": "renderer"}]

    class FakeRoundPipeline:
        def _renderer_system_prompt(self, render_plan_value, *, prompt_mode="base", violation_types=None):
            seen["_renderer_system_prompt"] = {
                "render_plan": render_plan_value,
                "prompt_mode": prompt_mode,
                "violation_types": violation_types,
            }
            return "delegated renderer prompt"

        def _evaluate_authenticity(self, text, render_plan_value):
            seen["_evaluate_authenticity"] = {"text": text, "render_plan": render_plan_value}
            return {"self_grounding_score": 0.9}

        def _coerce_score_map(self, value):
            seen["_coerce_score_map"] = value
            return {"plan": 0.3}

        def _generate_pfc_candidates_via_model(self, event_value, state_value, scenario_value, context_value, *, model_call_traces=None):
            seen["_generate_pfc_candidates_via_model"] = {
                "event": event_value,
                "state": state_value,
                "scenario": scenario_value,
                "context": context_value,
                "model_call_traces": model_call_traces,
            }
            return contribution

        def _invoke_pfc_model_generator(self, event_value, state_value, scenario_value, context_value, *, model_call_traces=None):
            seen["_invoke_pfc_model_generator"] = {
                "event": event_value,
                "state": state_value,
                "scenario": scenario_value,
                "context": context_value,
                "model_call_traces": model_call_traces,
            }
            return contribution

        def _should_run_late_perspective(self, event_value, state_value, relation_state_value, sampled_action, *, route_type=""):
            seen["_should_run_late_perspective"] = {
                "event": event_value,
                "state": state_value,
                "relation_state": relation_state_value,
                "sampled_action": sampled_action,
                "route_type": route_type,
            }
            return True, "risk_gate_open"

        def _infer_other_state_via_model(
            self,
            event_value,
            state_value,
            scenario_value,
            context_value,
            sampled_action,
            relation_state_value,
            *,
            model_call_traces=None,
            parallel_group=None,
        ):
            seen["_infer_other_state_via_model"] = {
                "event": event_value,
                "state": state_value,
                "scenario": scenario_value,
                "context": context_value,
                "sampled_action": sampled_action,
                "relation_state": relation_state_value,
                "model_call_traces": model_call_traces,
                "parallel_group": parallel_group,
            }
            return {"state_hypothesis": {"mood": "focused"}}

        def _simulate_other_reaction_via_model(
            self,
            event_value,
            state_value,
            scenario_value,
            context_value,
            sampled_action,
            relation_state_value,
            *,
            model_call_traces=None,
            parallel_group=None,
        ):
            seen["_simulate_other_reaction_via_model"] = {
                "event": event_value,
                "state": state_value,
                "scenario": scenario_value,
                "context": context_value,
                "sampled_action": sampled_action,
                "relation_state": relation_state_value,
                "model_call_traces": model_call_traces,
                "parallel_group": parallel_group,
            }
            return {"reaction_hypothesis": {"tone": "warm"}}

        def _render_expression_via_model(
            self,
            render_plan_value,
            *,
            prompt_mode="base",
            violation_types=None,
            model_call_traces=None,
        ):
            seen["_render_expression_via_model"] = {
                "render_plan": render_plan_value,
                "prompt_mode": prompt_mode,
                "violation_types": violation_types,
                "model_call_traces": model_call_traces,
            }
            return {"text": "delegated text", "route": "renderer", "model": "delegated-model"}

        def _should_allow_renderer_resample(self, *, route_type, render_plan, violation_types, context):
            seen["_should_allow_renderer_resample"] = {
                "route_type": route_type,
                "render_plan": render_plan,
                "violation_types": violation_types,
                "context": context,
            }
            return True

        def _score_salience_via_model(
            self,
            event_value,
            state_value,
            scenario_value,
            context_value,
            *,
            model_call_traces=None,
            parallel_group=None,
        ):
            seen["_score_salience_via_model"] = {
                "event": event_value,
                "state": state_value,
                "scenario": scenario_value,
                "context": context_value,
                "model_call_traces": model_call_traces,
                "parallel_group": parallel_group,
            }
            return contribution

        def _estimate_subjective_value_via_model(
            self,
            event_value,
            state_value,
            scenario_value,
            context_value,
            *,
            model_call_traces=None,
            parallel_group=None,
        ):
            seen["_estimate_subjective_value_via_model"] = {
                "event": event_value,
                "state": state_value,
                "scenario": scenario_value,
                "context": context_value,
                "model_call_traces": model_call_traces,
                "parallel_group": parallel_group,
            }
            return payload

    monkeypatch.setattr(controller, "round_pipeline", FakeRoundPipeline(), raising=False)

    assert controller._renderer_system_prompt(render_plan, prompt_mode="violation", violation_types=["style"]) == "delegated renderer prompt"
    assert controller._evaluate_authenticity("hello", render_plan) == {"self_grounding_score": 0.9}
    assert controller._coerce_score_map([["plan", 0.3]]) == {"plan": 0.3}
    assert controller._generate_pfc_candidates_via_model(event, state, {"name": "chat"}, {"cue": "model"}, model_call_traces=traces) is contribution
    assert controller._invoke_pfc_model_generator(event, state, {"name": "task"}, {"cue": "model"}, model_call_traces=traces) is contribution
    assert controller._should_run_late_perspective(event, state, {"relationship_risk": 0.5}, "connect", route_type="chat_standard") == (True, "risk_gate_open")
    assert controller._infer_other_state_via_model(
        event,
        state,
        {"name": "chat"},
        {"cue": "model"},
        "connect",
        {"relationship_risk": 0.5},
        model_call_traces=traces,
        parallel_group="g1",
    ) == {"state_hypothesis": {"mood": "focused"}}
    assert controller._simulate_other_reaction_via_model(
        event,
        state,
        {"name": "chat"},
        {"cue": "model"},
        "respond",
        {"relationship_risk": 0.5},
        model_call_traces=traces,
        parallel_group="g2",
    ) == {"reaction_hypothesis": {"tone": "warm"}}
    assert controller._render_expression_via_model(
        render_plan,
        prompt_mode="violation",
        violation_types=["provider_leak"],
        model_call_traces=traces,
    ) == {"text": "delegated text", "route": "renderer", "model": "delegated-model"}
    assert controller._should_allow_renderer_resample(
        route_type="task_run",
        render_plan=render_plan,
        violation_types=["provider_leak"],
        context={"authenticity_risk": 0.5},
    ) is True
    assert controller._score_salience_via_model(
        event,
        state,
        {"name": "chat"},
        {"cue": "model"},
        model_call_traces=traces,
        parallel_group="salience",
    ) is contribution
    assert controller._estimate_subjective_value_via_model(
        event,
        state,
        {"name": "task"},
        {"cue": "model"},
        model_call_traces=traces,
        parallel_group="value",
    ) is payload

    assert seen["_renderer_system_prompt"]["prompt_mode"] == "violation"
    assert seen["_evaluate_authenticity"]["text"] == "hello"
    assert seen["_coerce_score_map"] == [["plan", 0.3]]
    assert seen["_generate_pfc_candidates_via_model"]["model_call_traces"] is traces
    assert seen["_invoke_pfc_model_generator"]["scenario"] == {"name": "task"}
    assert seen["_should_run_late_perspective"]["route_type"] == "chat_standard"
    assert seen["_infer_other_state_via_model"]["parallel_group"] == "g1"
    assert seen["_simulate_other_reaction_via_model"]["parallel_group"] == "g2"
    assert seen["_render_expression_via_model"]["violation_types"] == ["provider_leak"]
    assert seen["_should_allow_renderer_resample"]["route_type"] == "task_run"
    assert seen["_score_salience_via_model"]["parallel_group"] == "salience"
    assert seen["_estimate_subjective_value_via_model"]["parallel_group"] == "value"


def test_action_bookkeeping_builder_helpers_delegate_to_round_pipeline(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    state = controller.load_runtime_state()
    seen: dict[str, object] = {}
    rows = [object()]
    bookkeeping = object()

    class FakeRoundPipeline:
        def _build_base_distribution(self, state_value, scenario_cfg, mode_cfg, relation_state):
            seen["_build_base_distribution"] = {
                "state": state_value,
                "scenario_cfg": scenario_cfg,
                "mode_cfg": mode_cfg,
                "relation_state": relation_state,
            }
            return {"respond": 0.4}

        def _compute_context_delta(self, state_value, scenario_cfg):
            seen["_compute_context_delta"] = {
                "state": state_value,
                "scenario_cfg": scenario_cfg,
            }
            return {"respond": 0.1}

        def _update_ci(self, state_value, actions, context, scenario_cfg, thresholds):
            seen["_update_ci"] = {
                "state": state_value,
                "actions": actions,
                "context": context,
                "scenario_cfg": scenario_cfg,
                "thresholds": thresholds,
            }
            return {"respond": 0.3}

        def _build_action_bookkeeping(
            self,
            rows_value,
            state_value,
            scenario_cfg,
            mode_cfg,
            relation_state,
            context,
            query_state=None,
            disclosure_state=None,
            resample_idx=0,
        ):
            seen["_build_action_bookkeeping"] = {
                "rows": rows_value,
                "state": state_value,
                "scenario_cfg": scenario_cfg,
                "mode_cfg": mode_cfg,
                "relation_state": relation_state,
                "context": context,
                "query_state": query_state,
                "disclosure_state": disclosure_state,
                "resample_idx": resample_idx,
            }
            return bookkeeping

        def _top_action_name(self, distribution):
            seen["_top_action_name"] = distribution
            return "respond"

    monkeypatch.setattr(controller, "round_pipeline", FakeRoundPipeline(), raising=False)

    assert controller._build_base_distribution(state, {"name": "chat"}, {"allow_dmn": True}, {"closeness": 0.5}) == {"respond": 0.4}
    assert controller._compute_context_delta(state, {"delay_tolerance": 0.1}) == {"respond": 0.1}
    assert controller._update_ci(state, ["respond"], {"closeness": 0.5}, {"pfc_base_share": 0.2}, {"ci_min": 0.1}) == {"respond": 0.3}
    assert controller._build_action_bookkeeping(
        rows,
        state,
        {"name": "task"},
        {"allow_dmn": True},
        {"closeness": 0.5},
        {"cue": "tea"},
        resample_idx=2,
    ) is bookkeeping
    assert controller._top_action_name({"respond": 0.7, "plan": 0.3}) == "respond"

    assert seen["_build_base_distribution"]["state"] is state
    assert seen["_compute_context_delta"]["scenario_cfg"] == {"delay_tolerance": 0.1}
    assert seen["_update_ci"]["actions"] == ["respond"]
    assert seen["_build_action_bookkeeping"]["rows"] == rows
    assert seen["_build_action_bookkeeping"]["resample_idx"] == 2
    assert seen["_top_action_name"] == {"respond": 0.7, "plan": 0.3}


def test_conflict_expression_helpers_delegate_to_round_pipeline(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    state = controller.load_runtime_state()
    seen: dict[str, object] = {}
    expression = object()
    adjusted = object()

    class FakeRoundPipeline:
        def _build_repair_expression_policy(self, conflict_state):
            seen["_build_repair_expression_policy"] = conflict_state
            return {"stage": "repairing"}

        def _safe_mode_delta(self, before, after):
            seen["_safe_mode_delta"] = {"before": before, "after": after}
            return "entered"

        def _update_conflict_circuit(self, state_value, *, critical_conflict, winning_priority, compromise_template):
            seen["_update_conflict_circuit"] = {
                "state": state_value,
                "critical_conflict": critical_conflict,
                "winning_priority": winning_priority,
                "compromise_template": compromise_template,
            }
            return {"active": True}

        def _apply_conflict_expression_adjustments(self, expression_value, conflict_state):
            seen["_apply_conflict_expression_adjustments"] = {
                "expression": expression_value,
                "conflict_state": conflict_state,
            }
            return adjusted

    monkeypatch.setattr(controller, "round_pipeline", FakeRoundPipeline(), raising=False)

    assert controller._build_repair_expression_policy({"repair_state_snapshot": {"stage": "repairing"}}) == {"stage": "repairing"}
    assert controller._safe_mode_delta(False, True) == "entered"
    assert controller._update_conflict_circuit(
        state,
        critical_conflict=True,
        winning_priority="body_safety",
        compromise_template="body_first",
    ) == {"active": True}
    assert controller._apply_conflict_expression_adjustments(expression, {"compromise": {"template": "body_first"}}) is adjusted

    assert seen["_build_repair_expression_policy"] == {"repair_state_snapshot": {"stage": "repairing"}}
    assert seen["_safe_mode_delta"] == {"before": False, "after": True}
    assert seen["_update_conflict_circuit"]["state"] is state
    assert seen["_apply_conflict_expression_adjustments"]["expression"] is expression


def test_command_boundary_helpers_delegate_to_run_runtime(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    state = controller.load_runtime_state()
    envelope = controller._legacy_command_envelope("safe on")
    result = controller.apply_command("safe on")
    seen: dict[str, object] = {}
    boundary_result = CommandResult(applied=True, scope="runtime", delta={"safe_mode": True})
    rollback = {"strategy": "domain_inverse", "command": "safe off"}
    snapshot_path = controller.snapshot_dir / "snap-delegated"

    class FakeRunRuntime:
        def _infer_operator_level(self, domain):
            seen["_infer_operator_level"] = domain
            return "ops_admin"

        def _default_mutation_scope(self, domain, target=None):
            seen["_default_mutation_scope"] = {"domain": domain, "target": target}
            return "runtime"

        def _legacy_command_envelope(self, command):
            seen["_legacy_command_envelope"] = command
            return envelope

        def _command_snapshot_path(self, snapshot_id):
            seen["_command_snapshot_path"] = snapshot_id
            return snapshot_path

        def _create_command_snapshot(self, state_value, envelope_value):
            seen["_create_command_snapshot"] = {"state": state_value, "envelope": envelope_value}
            return "snap-delegated"

        def _apply_snapshot_restore(self, snapshot_id, envelope_value):
            seen["_apply_snapshot_restore"] = {"snapshot_id": snapshot_id, "envelope": envelope_value}
            return boundary_result

        def _enrich_command_result(self, result_value, envelope_value):
            seen["_enrich_command_result"] = {"result": result_value, "envelope": envelope_value}
            return result_value

        def _build_rollback(self, envelope_value, before_state, result_value, snapshot_id):
            seen["_build_rollback"] = {
                "envelope": envelope_value,
                "before_state": before_state,
                "result": result_value,
                "snapshot_id": snapshot_id,
            }
            return rollback

        def _mark_boundary_result(
            self,
            result_value,
            *,
            boundary_action,
            cause_type="external_stimulus",
            violation_code="",
            deprecation_warning="",
        ):
            seen["_mark_boundary_result"] = {
                "result": result_value,
                "boundary_action": boundary_action,
                "cause_type": cause_type,
                "violation_code": violation_code,
                "deprecation_warning": deprecation_warning,
            }
            return boundary_result

        def _boundary_deprecation(self, command):
            seen["_boundary_deprecation"] = command
            return "deprecated"

        def _reject_boundary_command(
            self,
            state_value,
            *,
            scope,
            operator_level,
            violation_code,
            risk_note,
        ):
            seen["_reject_boundary_command"] = {
                "state": state_value,
                "scope": scope,
                "operator_level": operator_level,
                "violation_code": violation_code,
                "risk_note": risk_note,
            }
            return boundary_result

    monkeypatch.setattr(controller, "run_runtime", FakeRunRuntime(), raising=False)

    assert controller._infer_operator_level("safe") == "ops_admin"
    assert controller._default_mutation_scope("safe", "on") == "runtime"
    assert controller._legacy_command_envelope("safe on") is envelope
    assert controller._command_snapshot_path("snap-123") == snapshot_path
    assert controller._create_command_snapshot(state, envelope) == "snap-delegated"
    assert controller._apply_snapshot_restore("snap-restore", envelope) is boundary_result
    assert controller._enrich_command_result(result, envelope) is result
    assert controller._build_rollback(envelope, state, result, "snap-789") is rollback
    assert controller._mark_boundary_result(result, boundary_action="allow_internal") is boundary_result
    assert controller._boundary_deprecation("safe on") == "deprecated"
    assert controller._reject_boundary_command(
        state,
        scope="identity",
        operator_level="soft_intervene",
        violation_code="locked",
        risk_note="risk",
    ) is boundary_result

    assert seen["_infer_operator_level"] == "safe"
    assert seen["_default_mutation_scope"] == {"domain": "safe", "target": "on"}
    assert seen["_legacy_command_envelope"] == "safe on"
    assert seen["_command_snapshot_path"] == "snap-123"
    assert seen["_create_command_snapshot"] == {"state": state, "envelope": envelope}
    assert seen["_apply_snapshot_restore"] == {"snapshot_id": "snap-restore", "envelope": envelope}
    assert seen["_enrich_command_result"] == {"result": result, "envelope": envelope}
    assert seen["_build_rollback"]["snapshot_id"] == "snap-789"
    assert seen["_mark_boundary_result"]["boundary_action"] == "allow_internal"
    assert seen["_boundary_deprecation"] == "safe on"
    assert seen["_reject_boundary_command"]["state"] is state


def test_model_routing_facades_delegate_to_model_routing_runtime(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    seen: dict[str, object] = {}
    tiers = {"small_model": {"backend": "openai_compatible"}}
    bindings = {"Renderer": "large_model"}
    route_config = object()
    response = object()
    bucket: list[dict[str, object]] = []
    request = object()

    class FakeModelRoutingRuntime:
        def _provider_descriptor_for_route(self, route_name="renderer"):
            seen["_provider_descriptor_for_route"] = route_name
            return ("delegated provider", "delegated model")

        def _provider_descriptor(self):
            seen["_provider_descriptor"] = True
            return ("provider", "model")

        def _model_tiers(self):
            seen["_model_tiers"] = True
            return tiers

        def _pipeline_model_bindings(self):
            seen["_pipeline_model_bindings"] = True
            return bindings

        def _pipeline_tier(self, binding_key):
            seen["_pipeline_tier"] = binding_key
            return "large_model"

        def _effective_pipeline_tier(self, binding_key, *, metadata=None):
            seen["_effective_pipeline_tier"] = {"binding_key": binding_key, "metadata": metadata}
            return "small_model"

        def _tier_config(self, tier_name):
            seen["_tier_config"] = tier_name
            return {"backend": "openai_compatible"}

        def _infer_tier_name_for_route(self, route_cfg):
            seen["_infer_tier_name_for_route"] = route_cfg
            return "medium_model"

        def _route_config_for_binding(self, binding_key, *, route_name, metadata=None):
            seen["_route_config_for_binding"] = {
                "binding_key": binding_key,
                "route_name": route_name,
                "metadata": metadata,
            }
            return route_config

        def _route_policy_contract(self):
            seen["_route_policy_contract"] = True
            return {"chat_standard": {"decision_mode": "packet_plus_conditional_modules"}}

        def _record_model_call(
            self,
            bucket_value,
            *,
            skill_name,
            binding_key,
            route_config,
            response,
            prompt_chars,
            parallel_group=None,
        ):
            seen["_record_model_call"] = {
                "bucket": bucket_value,
                "skill_name": skill_name,
                "binding_key": binding_key,
                "route_config": route_config,
                "response": response,
                "prompt_chars": prompt_chars,
                "parallel_group": parallel_group,
            }

        def _call_bound_model_route(
            self,
            binding_key,
            *,
            route_name,
            request,
            binding_metadata=None,
            model_call_traces=None,
            skill_name=None,
            parallel_group=None,
        ):
            seen["_call_bound_model_route"] = {
                "binding_key": binding_key,
                "route_name": route_name,
                "request": request,
                "binding_metadata": binding_metadata,
                "model_call_traces": model_call_traces,
                "skill_name": skill_name,
                "parallel_group": parallel_group,
            }
            return response

    monkeypatch.setattr(controller, "model_routing_runtime", FakeModelRoutingRuntime(), raising=False)

    assert controller._provider_descriptor_for_route("planner") == ("delegated provider", "delegated model")
    assert controller._provider_descriptor() == ("provider", "model")
    assert controller.provider_descriptor() == ("provider", "model")
    assert controller._model_tiers() is tiers
    assert controller._pipeline_model_bindings() is bindings
    assert controller._pipeline_tier("Renderer") == "large_model"
    assert controller._effective_pipeline_tier("PFCAgent", metadata={"relation_risk": 0.8}) == "small_model"
    assert controller._tier_config("small_model") == {"backend": "openai_compatible"}
    assert controller._infer_tier_name_for_route(route_config) == "medium_model"
    assert controller._route_policy_contract() == {"chat_standard": {"decision_mode": "packet_plus_conditional_modules"}}
    assert controller._route_config_for_binding("Renderer", route_name="renderer", metadata={"x": 1}) is route_config
    assert controller._call_bound_model_route(
        "Renderer",
        route_name="renderer",
        request=request,
        binding_metadata={"x": 1},
        model_call_traces=bucket,
        skill_name="render_expression",
        parallel_group="g1",
    ) is response
    assert controller._record_model_call(
        bucket,
        skill_name="render_expression",
        binding_key="Renderer",
        route_config=route_config,
        response=response,
        prompt_chars=42,
        parallel_group="g1",
    ) is None

    assert seen["_provider_descriptor_for_route"] == "planner"
    assert seen["_provider_descriptor"] is True
    assert seen["_model_tiers"] is True
    assert seen["_pipeline_model_bindings"] is True
    assert seen["_pipeline_tier"] == "Renderer"
    assert seen["_effective_pipeline_tier"] == {"binding_key": "PFCAgent", "metadata": {"relation_risk": 0.8}}
    assert seen["_tier_config"] == "small_model"
    assert seen["_infer_tier_name_for_route"] is route_config
    assert seen["_route_policy_contract"] is True
    assert seen["_route_config_for_binding"] == {
        "binding_key": "Renderer",
        "route_name": "renderer",
        "metadata": {"x": 1},
    }
    assert seen["_call_bound_model_route"]["parallel_group"] == "g1"
    assert seen["_record_model_call"]["prompt_chars"] == 42


def test_model_payload_facades_delegate_to_model_payload_runtime(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    state = controller.load_runtime_state()
    event = RoundEvent(source="user", content="delegate payload", target="user", cue="tea")
    scenario = {"name": "task", "pfc_base_share": 0.2, "delay_tolerance": 0.1}
    context = {"cue": "tea", "closeness": 0.6}
    relation_state = {"closeness": 0.6, "boundary_level": 0.1}
    render_plan = object()
    seen: dict[str, object] = {}
    state_summary = {"mode": "interactive"}
    context_summary = {"cue": "tea"}
    relation_summary = {"closeness": 0.6}
    event_summary = {"content": "delegate payload"}
    pfc_payload = {"state_summary": state_summary}
    perspective_payload = {"relation_state": relation_summary}
    render_payload = {"action": "respond"}

    class FakeModelPayloadRuntime:
        def _compact_float(self, value):
            seen["_compact_float"] = value
            return 0.42

        def _build_state_summary(self, state_value):
            seen["_build_state_summary"] = state_value
            return state_summary

        def _build_context_summary(self, context_value):
            seen["_build_context_summary"] = context_value
            return context_summary

        def _build_relation_summary(self, relation_state_value):
            seen["_build_relation_summary"] = relation_state_value
            return relation_summary

        def _build_event_summary(self, event_value):
            seen["_build_event_summary"] = event_value
            return event_summary

        def _build_pfc_model_payload(self, event_value, state_value, scenario_value, context_value):
            seen["_build_pfc_model_payload"] = {
                "event": event_value,
                "state": state_value,
                "scenario": scenario_value,
                "context": context_value,
            }
            return pfc_payload

        def _build_perspective_model_payload(
            self,
            event_value,
            state_value,
            scenario_value,
            context_value,
            relation_state_value,
            *,
            action_name,
            output_key,
        ):
            seen["_build_perspective_model_payload"] = {
                "event": event_value,
                "state": state_value,
                "scenario": scenario_value,
                "context": context_value,
                "relation_state": relation_state_value,
                "action_name": action_name,
                "output_key": output_key,
            }
            return perspective_payload

        def _build_render_model_payload(self, render_plan_value):
            seen["_build_render_model_payload"] = render_plan_value
            return render_payload

    monkeypatch.setattr(controller, "model_payload_runtime", FakeModelPayloadRuntime(), raising=False)

    assert controller._compact_float(0.4242) == 0.42
    assert controller._build_state_summary(state) is state_summary
    assert controller._build_context_summary(context) is context_summary
    assert controller._build_relation_summary(relation_state) is relation_summary
    assert controller._build_event_summary(event) is event_summary
    assert controller._build_pfc_model_payload(event, state, scenario, context) is pfc_payload
    assert controller._build_perspective_model_payload(
        event,
        state,
        scenario,
        context,
        relation_state,
        action_name="respond",
        output_key="predicted_action",
    ) is perspective_payload
    assert controller._build_render_model_payload(render_plan) is render_payload

    assert seen["_compact_float"] == 0.4242
    assert seen["_build_state_summary"] is state
    assert seen["_build_context_summary"] == context
    assert seen["_build_relation_summary"] == relation_state
    assert seen["_build_event_summary"] is event
    assert seen["_build_pfc_model_payload"]["scenario"] == scenario
    assert seen["_build_perspective_model_payload"]["action_name"] == "respond"
    assert seen["_build_perspective_model_payload"]["output_key"] == "predicted_action"
    assert seen["_build_render_model_payload"] is render_plan


def test_initiative_session_helpers_delegate_to_run_runtime(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    state = controller.load_runtime_state()
    seen: dict[str, object] = {}
    selection = {"selected_session_id": "sess-1"}
    active = {"session_id": "sess-1"}
    stats = {"sent_last_hour": 1}
    cues = [("topic", "current_goal")]
    run_payload = {"run_id": "run-1"}

    class FakeRunRuntime:
        def _initiative_session_selection(self, *, now_iso=None):
            seen["_initiative_session_selection"] = now_iso
            return selection

        def _initiative_active_session(self):
            seen["_initiative_active_session"] = True
            return active

        def _initiative_has_pending_approval(self, active_session):
            seen["_initiative_has_pending_approval"] = active_session
            return True

        def _initiative_recent_history_stats(self, history, *, now_iso, settings):
            seen["_initiative_recent_history_stats"] = {
                "history": history,
                "now_iso": now_iso,
                "settings": settings,
            }
            return stats

        def _initiative_topic_cues(self, *, current_goal=None, active_session=None):
            seen["_initiative_topic_cues"] = {
                "current_goal": current_goal,
                "active_session": active_session,
            }
            return cues

        def _initiative_active_run_payload(self, state_value):
            seen["_initiative_active_run_payload"] = state_value
            return run_payload

    monkeypatch.setattr(controller, "run_runtime", FakeRunRuntime(), raising=False)

    assert controller._initiative_session_selection(now_iso="2026-04-11T00:00:00Z") is selection
    assert controller._initiative_active_session() is active
    assert controller._initiative_has_pending_approval({"session_id": "sess-1"}) is True
    assert controller._initiative_recent_history_stats([], now_iso="2026-04-11T00:00:00Z", settings={"hourly_limit": 1}) is stats
    assert controller._initiative_topic_cues(current_goal="topic", active_session={"session_id": "sess-1"}) is cues
    assert controller._initiative_active_run_payload(state) is run_payload
    assert seen["_initiative_session_selection"] == "2026-04-11T00:00:00Z"
    assert seen["_initiative_active_session"] is True
    assert seen["_initiative_has_pending_approval"] == {"session_id": "sess-1"}
    assert seen["_initiative_recent_history_stats"] == {
        "history": [],
        "now_iso": "2026-04-11T00:00:00Z",
        "settings": {"hourly_limit": 1},
    }
    assert seen["_initiative_topic_cues"] == {
        "current_goal": "topic",
        "active_session": {"session_id": "sess-1"},
    }
    assert seen["_initiative_active_run_payload"] is state


def test_monologue_model_builder_delegates_to_run_runtime(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    seen: dict[str, object] = {}
    fragments = [{"content": "fragment", "category": "model_fragment", "source": "model"}]

    class FakeRunRuntime:
        def _generate_monologue_fragments_via_model(self, **kwargs):
            seen["kwargs"] = kwargs
            return fragments

    monkeypatch.setattr(controller, "run_runtime", FakeRunRuntime(), raising=False)

    payload = controller._generate_monologue_fragments_via_model(
        seed="seed",
        pulse_index=1,
        pulse_dt="2026-04-11T00:00:00Z",
        fragment_count=2,
    )

    assert payload is fragments
    assert seen["kwargs"] == {
        "seed": "seed",
        "pulse_index": 1,
        "pulse_dt": "2026-04-11T00:00:00Z",
        "fragment_count": 2,
    }


def test_thought_snapshot_facades_delegate_to_observer_runtime(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    seen: dict[str, object] = {}
    snapshot = {"round_id": 7}
    empty = {"round_id": None}

    class FakeObserverRuntime:
        def thought_snapshot(self, round_ref="last"):
            seen["thought_snapshot"] = round_ref
            return snapshot

        def empty_thought_payload(self, round_ref=None):
            seen["empty_thought_payload"] = round_ref
            return empty

    monkeypatch.setattr(controller, "observer_runtime", FakeObserverRuntime(), raising=False)

    assert controller.thought_snapshot(7) is snapshot
    assert controller.empty_thought_payload() is empty
    assert seen["thought_snapshot"] == 7
    assert seen["empty_thought_payload"] is None


def test_initiative_write_helpers_delegate_to_run_runtime(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    proposal = {"proposal_id": "p1"}
    outcome = {"recorded": True}
    entry = {"proposal_id": "p1", "auto_sent": True}
    seen: dict[str, object] = {}

    class FakeRunRuntime:
        def _record_initiative_feedback_delta(self, text: str):
            seen["_record_initiative_feedback_delta"] = text
            return 0.2

        def record_initiative_feedback(self, text: str, *, session_id=None, target="user"):
            seen["record_initiative_feedback"] = {
                "text": text,
                "session_id": session_id,
                "target": target,
            }
            return outcome

        def _dispatch_initiative_to_session(self, proposal_value, text: str):
            seen["_dispatch_initiative_to_session"] = {
                "proposal": proposal_value,
                "text": text,
            }
            return True, "sess-1"

        def _record_initiative_outcome(self, **kwargs):
            seen["_record_initiative_outcome"] = kwargs
            return entry

    monkeypatch.setattr(controller, "run_runtime", FakeRunRuntime(), raising=False)

    assert controller._record_initiative_feedback_delta("ok") == 0.2
    assert controller.record_initiative_feedback("reply", session_id="sess-1") is outcome
    assert controller._dispatch_initiative_to_session(proposal, "hello") == (True, "sess-1")
    assert controller._record_initiative_outcome(
        round_id=1,
        recorded_at="2026-04-11T00:00:00Z",
        proposal=proposal,
        message="hello",
        auto_sent=True,
        session_id="sess-1",
    ) is entry
    assert seen["_record_initiative_feedback_delta"] == "ok"
    assert seen["record_initiative_feedback"] == {
        "text": "reply",
        "session_id": "sess-1",
        "target": "user",
    }
    assert seen["_dispatch_initiative_to_session"] == {
        "proposal": proposal,
        "text": "hello",
    }
    assert seen["_record_initiative_outcome"] == {
        "round_id": 1,
        "recorded_at": "2026-04-11T00:00:00Z",
        "proposal": proposal,
        "message": "hello",
        "auto_sent": True,
        "session_id": "sess-1",
    }


def test_autonomy_status_and_lifecycle_facades_delegate_to_run_runtime(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    seen: dict[str, object] = {}
    detailed = {"running": True, "kind": "detailed"}
    runtime = {"running": True, "kind": "runtime"}
    started = {"running": True, "kind": "started"}
    stopped = {"running": False, "kind": "stopped"}

    class FakeRunRuntime:
        def _autonomy_status_payload(self, *, include_diagnostics: bool):
            seen["_autonomy_status_payload"] = include_diagnostics
            return detailed if include_diagnostics else runtime

        def start_autonomy(self, profile: str = "tool_level", *, clear_safe_mode: bool = False):
            seen["start_autonomy"] = {
                "profile": profile,
                "clear_safe_mode": clear_safe_mode,
            }
            return started

        def stop_autonomy(self, reason: str = "manual_stop"):
            seen["stop_autonomy"] = reason
            return stopped

    monkeypatch.setattr(controller, "run_runtime", FakeRunRuntime(), raising=False)

    assert controller.autonomy_status() is detailed
    assert controller.autonomy_runtime_status() is runtime
    assert controller.start_autonomy(profile="tool_level", clear_safe_mode=True) is started
    assert controller.stop_autonomy(reason="user_stop") is stopped
    assert seen["_autonomy_status_payload"] is False
    assert seen["start_autonomy"] == {
        "profile": "tool_level",
        "clear_safe_mode": True,
    }
    assert seen["stop_autonomy"] == "user_stop"


def test_autonomy_helper_facades_delegate_to_run_runtime(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    state = controller.load_runtime_state()
    policy = state.autonomy_policy
    ticket = {"action": "noop"}
    execution = {"goal": "inspect runtime"}
    command_result = {"status": "ok"}
    dream_result = {"status": "dreamed"}
    readonly_status = {"status": "readonly"}
    seen: dict[str, object] = {}

    class FakeRunRuntime:
        def _prepare_autonomy_step_state(self, state_value=None):
            seen["_prepare_autonomy_step_state"] = state_value
            return state

        def _autonomy_default_self_run_goal(self, state_value):
            seen["_autonomy_default_self_run_goal"] = state_value
            return "default autonomy goal"

        def prepare_autonomy_background(self, *, state=None):
            seen["prepare_autonomy_background"] = state
            return ticket

        def execute_autonomy_background(self, ticket_value):
            seen["execute_autonomy_background"] = ticket_value
            return execution

        def commit_autonomy_background(self, ticket_value, execution_value=None):
            seen["commit_autonomy_background"] = {
                "ticket": ticket_value,
                "execution": execution_value,
            }
            return {"committed": True}

        def autonomy_step(self):
            seen["autonomy_step"] = True
            return {"step": True}

        def _autonomy_after_command_step(self, result):
            seen["_autonomy_after_command_step"] = result
            return command_result

        def _autonomy_run_dream_pass(self, state_value):
            seen["_autonomy_run_dream_pass"] = state_value
            return dream_result

        def _autonomy_self_run_goal(self, state_value):
            seen["_autonomy_self_run_goal"] = state_value
            return "self run goal"

        def _autonomy_continue_readonly_run_directive(self, state_value, policy_value):
            seen["_autonomy_continue_readonly_run_directive"] = {
                "state": state_value,
                "policy": policy_value,
            }
            return {"kind": "repo_scan_approval"}

        def _autonomy_start_readonly_run(self, goal: str):
            seen["_autonomy_start_readonly_run"] = goal
            return readonly_status

        def _autonomy_continue_readonly_run(self, run_id: str, call_id: str):
            seen["_autonomy_continue_readonly_run"] = {
                "run_id": run_id,
                "call_id": call_id,
            }
            return readonly_status

        def _autonomy_rest_realization(self, state_value, policy_value):
            seen["_autonomy_rest_realization"] = {
                "state": state_value,
                "policy": policy_value,
            }
            return {"kind": "command", "command": "body rest"}

    monkeypatch.setattr(controller, "run_runtime", FakeRunRuntime(), raising=False)

    assert controller._prepare_autonomy_step_state() is state
    assert controller._autonomy_default_self_run_goal(state) == "default autonomy goal"
    assert controller.prepare_autonomy_background(state=state) is ticket
    assert controller.execute_autonomy_background(ticket) is execution
    assert controller.commit_autonomy_background(ticket, execution) == {"committed": True}
    assert controller.autonomy_step() == {"step": True}
    assert controller._autonomy_after_command_step({"command": "ok"}) is command_result
    assert controller._autonomy_run_dream_pass(state) is dream_result
    assert controller._autonomy_self_run_goal(state) == "self run goal"
    assert controller._autonomy_continue_readonly_run_directive(state, policy) == {"kind": "repo_scan_approval"}
    assert controller._autonomy_start_readonly_run("inspect repo") is readonly_status
    assert controller._autonomy_continue_readonly_run("run-1", "call-2") is readonly_status
    assert controller._autonomy_rest_realization(state, policy) == {"kind": "command", "command": "body rest"}
    assert seen["_prepare_autonomy_step_state"] is None
    assert seen["_autonomy_default_self_run_goal"] is state
    assert seen["prepare_autonomy_background"] is state
    assert seen["execute_autonomy_background"] is ticket
    assert seen["commit_autonomy_background"] == {
        "ticket": ticket,
        "execution": execution,
    }
    assert seen["autonomy_step"] is True
    assert seen["_autonomy_after_command_step"] == {"command": "ok"}
    assert seen["_autonomy_run_dream_pass"] is state
    assert seen["_autonomy_self_run_goal"] is state
    assert seen["_autonomy_continue_readonly_run_directive"] == {
        "state": state,
        "policy": policy,
    }
    assert seen["_autonomy_start_readonly_run"] == "inspect repo"
    assert seen["_autonomy_continue_readonly_run"] == {
        "run_id": "run-1",
        "call_id": "call-2",
    }
    assert seen["_autonomy_rest_realization"] == {
        "state": state,
        "policy": policy,
    }


def test_run_truth_facades_delegate_to_run_runtime(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    state = controller.load_runtime_state()
    run_state = controller._build_task_bootstrap(
        "inspect facade truth",
        allow_commit=False,
        operator_level="read_only",
        defer_bootstrap_tool=True,
    )[0]
    occupancy = {"run_visible": True, "run_id": "run-1"}
    truth = {"runtime_revision": 7, "run_visible": True}
    fault_guard = {"contract_status": "contract_only"}
    seen: dict[str, object] = {}

    class FakeRunRuntime:
        def _run_stop_reason_code(self, run_state_value):
            seen["_run_stop_reason_code"] = run_state_value
            return "dirty_worktree"

        def _run_has_pending_approval(self, run_id: str):
            seen["_run_has_pending_approval"] = run_id
            return True

        def _active_run_occupancy(self, state_value):
            seen["_active_run_occupancy"] = state_value
            return occupancy

        def runtime_status_truth_payload(self, state_value=None):
            seen["runtime_status_truth_payload"] = state_value
            return truth

        def fault_guard_status(self, state_value=None):
            seen["fault_guard_status"] = state_value
            return fault_guard

    monkeypatch.setattr(controller, "run_runtime", FakeRunRuntime(), raising=False)

    assert controller._run_stop_reason_code(run_state) == "dirty_worktree"
    assert controller._run_has_pending_approval("run-1") is True
    assert controller._active_run_occupancy(state) is occupancy
    assert controller.runtime_status_truth_payload(state) is truth
    assert controller.fault_guard_status(state) is fault_guard
    assert seen["_run_stop_reason_code"] is run_state
    assert seen["_run_has_pending_approval"] == "run-1"
    assert seen["_active_run_occupancy"] is state
    assert seen["runtime_status_truth_payload"] is state
    assert seen["fault_guard_status"] is state


def test_autonomy_run_state_helpers_delegate_to_run_runtime(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    state = controller.load_runtime_state()
    policy = state.autonomy_policy
    run_state = controller._build_task_bootstrap(
        "inspect autonomy helper delegation",
        allow_commit=False,
        operator_level="read_only",
        defer_bootstrap_tool=True,
    )[0]
    stale = {"run_id": "run-1", "age_seconds": 360.0}
    seen: dict[str, object] = {}

    class FakeRunRuntime:
        def _autonomy_pending_readonly_repo_scan(self, state_value, policy_value):
            seen["_autonomy_pending_readonly_repo_scan"] = {
                "state": state_value,
                "policy": policy_value,
            }
            return {"run_id": "run-1", "call_id": "call-1"}

        def _stale_dirty_worktree_run_info(self, run_state_value):
            seen["_stale_dirty_worktree_run_info"] = run_state_value
            return stale

        def _active_run_blocks_background_progress(self, state_value):
            seen["_active_run_blocks_background_progress"] = state_value
            return True

        def _clear_stale_dirty_worktree_run_blocker(self, state_value):
            seen["_clear_stale_dirty_worktree_run_blocker"] = state_value
            return True

        def _autonomy_trace_append(self, state_value, *, action_type, summary, delta=None):
            seen["_autonomy_trace_append"] = {
                "state": state_value,
                "action_type": action_type,
                "summary": summary,
                "delta": delta,
            }

    monkeypatch.setattr(controller, "run_runtime", FakeRunRuntime(), raising=False)

    assert controller._autonomy_pending_readonly_repo_scan(state, policy) == {"run_id": "run-1", "call_id": "call-1"}
    assert controller._stale_dirty_worktree_run_info(run_state) is stale
    assert controller._active_run_blocks_background_progress(state) is True
    assert controller._clear_stale_dirty_worktree_run_blocker(state) is True
    controller._autonomy_trace_append(
        state,
        action_type="self_run",
        summary="delegated trace append",
        delta={"run_id": "run-1"},
    )
    assert seen["_autonomy_pending_readonly_repo_scan"] == {
        "state": state,
        "policy": policy,
    }
    assert seen["_stale_dirty_worktree_run_info"] is run_state
    assert seen["_active_run_blocks_background_progress"] is state
    assert seen["_clear_stale_dirty_worktree_run_blocker"] is state
    assert seen["_autonomy_trace_append"] == {
        "state": state,
        "action_type": "self_run",
        "summary": "delegated trace append",
        "delta": {"run_id": "run-1"},
    }


def test_autonomy_candidate_helpers_delegate_to_run_runtime(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    state = controller.load_runtime_state()
    policy = state.autonomy_policy
    probe = {"action_distribution": {"rest": 0.2, "self_run": 0.8}}
    seen: dict[str, object] = {}

    class FakeRunRuntime:
        def _autonomy_candidate_scores(self, state_value, policy_value):
            seen["_autonomy_candidate_scores"] = {
                "state": state_value,
                "policy": policy_value,
            }
            return {"self_run": 0.7, "rest": 0.3}

        def _autonomy_candidate_action(self, state_value, policy_value):
            seen["_autonomy_candidate_action"] = {
                "state": state_value,
                "policy": policy_value,
            }
            return "self_run"

        def _autonomy_projected_candidate_scores(self, state_value, policy_value, field_probe=None):
            seen["_autonomy_projected_candidate_scores"] = {
                "state": state_value,
                "policy": policy_value,
                "field_probe": field_probe,
            }
            return {"self_run": 0.82, "rest": 0.18}

    monkeypatch.setattr(controller, "run_runtime", FakeRunRuntime(), raising=False)

    assert controller._autonomy_candidate_scores(state, policy) == {"self_run": 0.7, "rest": 0.3}
    assert controller._autonomy_candidate_action(state, policy) == "self_run"
    assert controller._autonomy_projected_candidate_scores(state, policy, probe) == {"self_run": 0.82, "rest": 0.18}
    assert seen["_autonomy_candidate_scores"] == {
        "state": state,
        "policy": policy,
    }
    assert seen["_autonomy_candidate_action"] == {
        "state": state,
        "policy": policy,
    }
    assert seen["_autonomy_projected_candidate_scores"] == {
        "state": state,
        "policy": policy,
        "field_probe": probe,
    }


def test_autonomy_policy_and_execution_primitives_delegate_to_run_runtime(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    state = controller.load_runtime_state()
    policy = state.autonomy_policy
    anchor = object()
    context = ({"cue": "x"}, {"closeness": 0.5}, {"resource_scarcity": 0.1})
    budget = {"remaining": 0.7}
    result = {"allowed": True, "command": "body rest"}
    seen: dict[str, object] = {}

    class FakeRunRuntime:
        def _autonomy_policy_for_profile(self, profile: str = "tool_level"):
            seen["_autonomy_policy_for_profile"] = profile
            return policy

        def _autonomy_matches_prefix(self, command: str, prefix: str):
            seen["_autonomy_matches_prefix"] = {"command": command, "prefix": prefix}
            return True

        def _autonomy_command_allowed(self, command: str, policy_value=None):
            seen["_autonomy_command_allowed"] = {"command": command, "policy": policy_value}
            return True, ""

        def _autonomy_operator_allowed(self, operator_level: str, policy_value=None):
            seen["_autonomy_operator_allowed"] = {
                "operator_level": operator_level,
                "policy": policy_value,
            }
            return True, ""

        def _autonomy_budget_usage(self, state_value):
            seen["_autonomy_budget_usage"] = state_value
            return budget

        def _append_autonomy_recent_action(self, state_value, *, action_type, summary, round_id=None, trace_ref=None):
            seen["_append_autonomy_recent_action"] = {
                "state": state_value,
                "action_type": action_type,
                "summary": summary,
                "round_id": round_id,
                "trace_ref": trace_ref,
            }

        def _autonomy_window_anchor(self, value):
            seen["_autonomy_window_anchor"] = value
            return anchor

        def _ensure_autonomy_window(self, state_value):
            seen["_ensure_autonomy_window"] = state_value

        def _autonomy_step_context(self, state_value):
            seen["_autonomy_step_context"] = state_value
            return context

        def _autonomy_self_run_score(self, state_value, policy_value):
            seen["_autonomy_self_run_score"] = {
                "state": state_value,
                "policy": policy_value,
            }
            return 0.88

        def _autonomy_record_failure(self, state_value, exc):
            seen["_autonomy_record_failure"] = {
                "state": state_value,
                "exc": exc,
            }

        def _autonomy_execute_command(self, command: str):
            seen["_autonomy_execute_command"] = command
            return result

    monkeypatch.setattr(controller, "run_runtime", FakeRunRuntime(), raising=False)

    assert controller._autonomy_policy_for_profile("observer_only") is policy
    assert controller._autonomy_matches_prefix("body rest", "body") is True
    assert controller._autonomy_command_allowed("body rest", policy) == (True, "")
    assert controller._autonomy_operator_allowed("read_only", policy) == (True, "")
    assert controller._autonomy_budget_usage(state) is budget
    controller._append_autonomy_recent_action(
        state,
        action_type="command",
        summary="body rest: applied",
        round_id=1,
        trace_ref="round://1",
    )
    assert controller._autonomy_window_anchor("2026-04-11T00:00:00Z") is anchor
    assert controller._autonomy_step_context(state) == context
    assert controller._autonomy_self_run_score(state, policy) == 0.88
    exc = RuntimeError("boom")
    assert controller._ensure_autonomy_window(state) is None
    assert controller._autonomy_record_failure(state, exc) is None
    assert controller._autonomy_execute_command("body rest") is result
    assert seen["_autonomy_policy_for_profile"] == "observer_only"
    assert seen["_autonomy_matches_prefix"] == {"command": "body rest", "prefix": "body"}
    assert seen["_autonomy_command_allowed"] == {"command": "body rest", "policy": policy}
    assert seen["_autonomy_operator_allowed"] == {"operator_level": "read_only", "policy": policy}
    assert seen["_autonomy_budget_usage"] is state
    assert seen["_append_autonomy_recent_action"] == {
        "state": state,
        "action_type": "command",
        "summary": "body rest: applied",
        "round_id": 1,
        "trace_ref": "round://1",
    }
    assert seen["_autonomy_window_anchor"] == "2026-04-11T00:00:00Z"
    assert seen["_ensure_autonomy_window"] is state
    assert seen["_autonomy_step_context"] is state
    assert seen["_autonomy_self_run_score"] == {"state": state, "policy": policy}
    assert seen["_autonomy_record_failure"] == {"state": state, "exc": exc}
    assert seen["_autonomy_execute_command"] == "body rest"


def test_autonomy_state_management_facades_delegate_to_run_runtime(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    state = controller.load_runtime_state()
    seen: dict[str, object] = {}

    class FakeRunRuntime:
        def _sync_autonomy_state(self, state_value):
            seen["_sync_autonomy_state"] = state_value

        def _ensure_default_autonomy_runtime(self, state_value, *, force=False):
            seen["_ensure_default_autonomy_runtime"] = {
                "state": state_value,
                "force": force,
            }
            return True

    monkeypatch.setattr(controller, "run_runtime", FakeRunRuntime(), raising=False)

    assert controller._sync_autonomy_state(state) is None
    assert controller.sync_autonomy_state(state) is None
    assert controller._ensure_default_autonomy_runtime(state, force=True) is True
    assert seen["_sync_autonomy_state"] is state
    assert seen["_ensure_default_autonomy_runtime"] == {
        "state": state,
        "force": True,
    }


def test_run_planner_facade_delegates_to_run_runtime(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    repo_metadata = {"cwd": str(tmp_path), "dirty_worktree_detected": False, "tracked_file_count": 3}
    seen: dict[str, object] = {}
    payload = {
        "goal_summary": "inspect runtime",
        "next_step": "open controller.py",
        "detail": "",
        "expected_observation": "find bulky methods",
        "success_criteria": "clear extraction target",
        "tool_choice": "read_file",
        "confidence": 0.7,
    }

    class FakeRunRuntime:
        def _plan_run_via_model(self, goal: str, repo_metadata_value: dict[str, object]):
            seen["goal"] = goal
            seen["repo_metadata"] = repo_metadata_value
            return payload

    monkeypatch.setattr(controller, "run_runtime", FakeRunRuntime(), raising=False)

    assert controller._plan_run_via_model("inspect runtime", repo_metadata) is payload
    assert seen["goal"] == "inspect runtime"
    assert seen["repo_metadata"] == repo_metadata


def test_autonomy_preference_and_goal_model_facades_delegate_to_run_runtime(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    state = controller.load_runtime_state()
    policy = state.autonomy_policy
    seen: dict[str, object] = {}

    class FakeRunRuntime:
        def _apply_observer_autonomy_preferences(self, policy_value):
            seen["_apply_observer_autonomy_preferences"] = policy_value
            return policy_value

        def _generate_autonomy_self_run_goal_via_model(self, state_value):
            seen["_generate_autonomy_self_run_goal_via_model"] = state_value
            return "inspect runtime traces"

    monkeypatch.setattr(controller, "run_runtime", FakeRunRuntime(), raising=False)

    assert controller._apply_observer_autonomy_preferences(policy) is policy
    assert controller._generate_autonomy_self_run_goal_via_model(state) == "inspect runtime traces"
    assert seen["_apply_observer_autonomy_preferences"] is policy
    assert seen["_generate_autonomy_self_run_goal_via_model"] is state
