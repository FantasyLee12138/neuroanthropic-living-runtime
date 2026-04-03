from pathlib import Path

from nalr.providers.router import ModelResponse
from nalr.runtime.controller import RuntimeController
from nalr.schemas.models import RoundEvent


CONFIG_ROOT = Path(__file__).resolve().parents[2] / "config"


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
    assert result.sampled_action.name in {"respond", "plan", "recall", "rest"}
    assert len(result.trace.top_drivers) == 3
    assert any(item.agent_name == "PFCAgent" for item in result.trace.contributions)


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
    assert metrics["total_rounds"] == 1
    assert metrics["sampled_actions"]


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
