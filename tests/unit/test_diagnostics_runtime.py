from pathlib import Path

from nalr.providers.router import ModelResponse
from nalr.runtime.controller import RuntimeController
from nalr.schemas.models import RoundEvent


CONFIG_ROOT = Path(__file__).resolve().parents[2] / "config"


def test_console_refresh_payload_reuses_state_hot_payload_without_reentering_console_state(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    controller.tick(
        RoundEvent(
            source="user",
            content="Please produce one round for console diagnostics.",
            target="user",
            cue="console",
        ),
        scenario="chat",
        mode="interactive",
    )

    state_payload_calls = {"count": 0}
    real_state_payload = controller.state_hot_payload

    def wrapped_state_payload():
        state_payload_calls["count"] += 1
        return real_state_payload()

    monkeypatch.setattr(controller, "state_hot_payload", wrapped_state_payload)
    monkeypatch.setattr(
        controller,
        "console_state",
        lambda: (_ for _ in ()).throw(AssertionError("console_refresh_payload should not call console_state")),
    )

    payload = controller.console_refresh_payload()

    assert state_payload_calls["count"] == 1
    assert payload["state"]["current_round"]["round_id"] == 1
    assert payload["state"]["performance"]["runtime_metrics"]["route_type"]


def test_console_probability_space_reuses_latest_trace_without_second_trace_round(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    controller.tick(
        RoundEvent(
            source="user",
            content="Please produce one round for probability diagnostics.",
            target="user",
            cue="probability",
        ),
        scenario="chat",
        mode="interactive",
    )

    real_trace_round = controller.trace_round
    trace_round_calls = {"count": 0}

    def wrapped_trace_round(round_ref):
        trace_round_calls["count"] += 1
        return real_trace_round(round_ref)

    monkeypatch.setattr(controller, "trace_round", wrapped_trace_round)

    payload = controller.console_probability_space("latest")

    assert payload["source_round_id"] == 1
    assert trace_round_calls["count"] == 1


def test_console_recent_actions_reads_from_hot_cache_without_forcing_flush(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    controller.tick(
        RoundEvent(
            source="user",
            content="Please produce one round for recent-actions diagnostics.",
            target="user",
            cue="recent-actions",
        ),
        scenario="chat",
        mode="interactive",
    )

    monkeypatch.setattr(
        controller.trace_store,
        "flush",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("console_recent_actions should not force a flush")),
    )

    payload = controller.console_recent_actions(limit=1)

    assert payload["actions"]
    assert payload["actions"][0]["round_id"] == 1


def test_console_state_surfaces_runtime_truth_payload(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    payload = controller.console_state()

    assert "runtime_revision" in payload
    assert "last_mutation_at" in payload
    assert "run_blocking" in payload
    assert "run_visible" in payload


def test_console_state_surfaces_chat_kernel_v2_round_fields(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    def fake_generate(route_name, request):
        if route_name != "cognitive_packet":
            raise AssertionError(f"unexpected route: {route_name}")
        return ModelResponse(
            route=route_name,
            model="deepseek-chat",
            payload={
                "salience": 0.34,
                "uncertainty": 0.21,
                "memory_need": False,
                "tool_need": False,
                "conflict_need": False,
                "candidate_action_prior": "respond",
                "draft_reply": "我会先给你一个简短、连续的回应。",
                "proposed_state_patch": {
                    "focus": "maintain_sparse_chat_response",
                    "obligations": ["继续沿着当前状态回答这个问题"],
                },
                "deepen_reason": "",
            },
            raw_text="{}",
        )

    monkeypatch.setattr(controller.model_router, "generate", fake_generate)
    controller.execute_turn(controller.plan_turn("最近的内在状态怎么样？"))

    payload = controller.console_state()
    current_round = payload["current_round"]

    assert current_round["route_type"] == "chat_fast"
    assert current_round["route_budget_ms"] == 700
    assert current_round["activation_set"] == [
        "Router",
        "HotStateLoader",
        "BudgetAllocator",
        "PacketAssembler",
        "SafetyGate",
        "StateWriter",
    ]
    assert current_round["memory_tiers_read"] == ["hot"]
    assert current_round["packet_summary"]["candidate_action_prior"] == "respond"
    assert current_round["background_jobs"]
