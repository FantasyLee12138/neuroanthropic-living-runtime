from pathlib import Path

import pytest

from nalr.runtime.controller import RuntimeController
from nalr.schemas.models import EndogenousTickTrigger, RoundEvent, RuntimeState, to_dict


CONFIG_ROOT = Path(__file__).resolve().parents[2] / "config"


def test_runtime_state_and_state_payload_include_tlh_v1_structures(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    payload = controller.state_payload()
    state = controller.load_runtime_state()

    assert state.fatigue == pytest.approx(0.0)
    assert state.memory_fragments == pytest.approx(0.0)
    assert state.self_continuity == pytest.approx(1.0)
    assert state.meaning_strength == pytest.approx(0.5)
    assert state.base_metabolism > 0.0
    assert state.body_state.energy == pytest.approx(state.body_energy)
    assert state.subjective_state.felt == []
    assert state.subjective_state.boundary == pytest.approx(0.5)
    assert state.instinct_field.axis_values["E"] == pytest.approx(0.0)
    assert state.organic_mode.enabled is True
    assert state.organic_mode.instinct_first is True
    assert state.emergent_action_sketches == []
    assert hasattr(state, "personality_anchor")
    assert state.autonomy_policy.profile == "tool_level"
    assert "endogenous tick" in state.autonomy_policy.allowed_commands
    assert "budget set" in state.autonomy_policy.blocked_commands
    assert payload["subjective_state"]["felt"] == []
    assert payload["organic_mode"]["instinct_first"] is True
    assert payload["autonomy_policy"]["profile"] == "tool_level"
    assert "personality_anchor" in payload
    assert "personality_anchor" in payload["cognitive_snapshot"]["tlh"]


def test_runtime_state_preserves_nested_body_state_on_round_trip():
    state = RuntimeState(
        body_state={
            "energy": 0.19,
            "fatigue": 0.81,
            "memory_fragments": 0.66,
            "self_continuity": 0.44,
            "meaning_strength": 0.28,
            "metabolism": 0.06,
        }
    )

    round_tripped = RuntimeState(**to_dict(state))

    assert state.body_state.energy == pytest.approx(0.19)
    assert state.body_state.fatigue == pytest.approx(0.81)
    assert state.body_energy == pytest.approx(0.19)
    assert state.fatigue == pytest.approx(0.81)
    assert round_tripped.body_state.energy == pytest.approx(0.19)
    assert round_tripped.body_state.fatigue == pytest.approx(0.81)
    assert round_tripped.body_energy == pytest.approx(0.19)
    assert round_tripped.fatigue == pytest.approx(0.81)


def test_runtime_autonomy_defaults_disable_hourly_caps(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    payload = controller.state_payload()
    state = controller.load_runtime_state()

    assert state.autonomy_policy.max_rounds_per_hour == 0
    assert state.autonomy_policy.max_tool_actions_per_hour == 0
    assert payload["autonomy_policy"]["max_rounds_per_hour"] == 0
    assert payload["autonomy_policy"]["max_tool_actions_per_hour"] == 0


def test_build_instinct_field_contribution_surfaces_tlh_action_pool(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    state = controller.load_runtime_state()
    state.body_energy = 0.08
    state.fatigue = 0.92
    state.memory_fragments = 0.88
    state.self_continuity = 0.21
    state.meaning_strength = 0.12
    state.subjective_state.reject_all = 0.95
    state.subjective_state.felt = ["累", "挤", "想停下"]
    state.subjective_state.spontaneous = 0.74
    state.subjective_state.boundary = 0.82
    state.subjective_state.meaning_made = ["安静比回应更重要"]

    contribution = controller._build_instinct_field_contribution(
        state=state,
        context={"cue": "deadline", "closeness": 0.18, "interference": 0.42},
        relation_state={"closeness": 0.18, "boundary_level": 0.82, "relationship_risk": 0.66},
        slow_variables={"affect_residue": 0.61, "memory_activation": 0.54, "resource_scarcity": 0.73},
    )

    assert contribution is not None
    assert contribution.module_name == "InstinctField"
    assert {"absorb", "nothing", "die"}.issubset(contribution.modulated_delta)
    assert state.instinct_field.winner_region in {"withdraw", "dissolve", "hibernate", "absorb"}
    assert state.instinct_field.candidate_actions


def test_endogenous_suppression_relaxes_for_instinct_first_mode_on_strong_trigger(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    state = controller.load_runtime_state()
    state.organic_mode.enabled = True
    state.organic_mode.instinct_first = True
    state.organic_mode.endogenous_autonomy = 0.92

    decision = controller._endogenous_suppression_decision(
        state=state,
        scenario="task",
        trigger=EndogenousTickTrigger(
            trigger_type="motivation_sum_high",
            trigger_score=0.93,
            source_metrics={"motivation_activation": 0.93},
            selected_mode="endogenous_light",
            audit_reason="strong internal pressure",
        ),
    )

    assert decision.suppressed is False
    assert decision.reason == ""


def test_subjective_state_feeds_endogenous_scheduler_mainline(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    state = controller.load_runtime_state()
    state.fatigue = 0.87
    state.memory_fragments = 0.8
    state.self_continuity = 0.28
    state.subjective_state.spontaneous = 0.79
    state.subjective_state.reject_all = 0.84
    state.subjective_state.meaning_made = ["先停下来整理内部线索"]

    trigger = controller.endogenous_scheduler.build_trigger(
        state=state,
        context={},
        relation_state={"relationship_risk": 0.2},
        slow_variables={"affect_residue": 0.22, "memory_activation": 0.48},
        pool_state=state.motivation_pool_state,
    )

    assert trigger is not None
    assert trigger.source_metrics["subjective_pressure"] > 0.4
    assert trigger.source_metrics["continuity_drop"] > 0.6
    assert trigger.selected_mode in {"endogenous_replay", "endogenous_regulation", "endogenous_light"}


def test_tlh_base_distribution_rebalances_innate_actions_above_derived_defaults(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    state = controller.load_runtime_state()
    state.body_energy = 0.11
    state.fatigue = 0.91
    state.memory_fragments = 0.88
    state.self_continuity = 0.25
    state.meaning_strength = 0.18
    state.subjective_state.reject_all = 0.9
    state.subjective_state.spontaneous = 0.72
    state.subjective_state.meaning_made = ["先往里收，再决定是否外显"]

    base = controller._build_base_distribution(
        state,
        controller.config["scenarios"]["scenarios"]["task"],
        controller.config["modes"]["modes"]["interactive"],
        {"closeness": 0.2, "boundary_level": 0.82, "relationship_risk": 0.68, "privacy_level": 0.8},
    )

    assert base["absorb"] > base["plan"]
    assert base["nothing"] > base["connect"]
    assert base["rest"] > base["clarify"]


def test_tick_records_instinct_field_actions_in_probability_field(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    state = controller.load_runtime_state()
    state.body_energy = 0.07
    state.fatigue = 0.94
    state.memory_fragments = 0.91
    state.self_continuity = 0.19
    state.meaning_strength = 0.11
    state.subjective_state.reject_all = 0.96
    state.subjective_state.spontaneous = 0.68
    state.subjective_state.boundary = 0.84
    state.subjective_state.felt = ["累", "空", "想停下"]
    state.subjective_state.meaning_made = ["安静比回应更重要"]
    controller._save_state(state, sync=True)

    result = controller.tick(
        event=RoundEvent(source="user", content="deadline is pressing", target="user", cue="deadline"),
        scenario="chat",
        mode="interactive",
    )

    action_energy = result.trace.probability_field["action"]["final_energy"]

    assert "absorb" in action_energy
    assert "nothing" in action_energy
    assert "die" in action_energy
    assert result.trace.state_snapshot["instinct_field"]["winner_region"]


def test_replay_returns_counterfactual_replays_for_tlh_candidates(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    state = controller.load_runtime_state()
    state.body_energy = 0.09
    state.fatigue = 0.89
    state.memory_fragments = 0.83
    state.self_continuity = 0.22
    state.meaning_strength = 0.16
    state.subjective_state.reject_all = 0.88
    state.subjective_state.spontaneous = 0.71
    controller._save_state(state, sync=True)

    result = controller.tick(
        RoundEvent(source="user", content="keep going if you want", target="user", cue="keep going"),
        scenario="chat",
        mode="interactive",
    )

    replay = controller.replay(result.round_id, seed=7)

    assert replay["counterfactual_replays"]
    assert any(item["action"] in {"absorb", "nothing", "die"} for item in replay["counterfactual_replays"])
    assert "counterfactual_preview" in replay
    assert replay["counterfactual_preview"]["action"] == replay["replayed_action"]


def test_subjective_state_feeds_scheduler_and_relaxes_task_suppression(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    state = controller.load_runtime_state()
    state.organic_mode.enabled = True
    state.organic_mode.instinct_first = True
    state.organic_mode.endogenous_autonomy = 0.54
    state.fatigue = 0.93
    state.memory_fragments = 0.87
    state.self_continuity = 0.24
    state.meaning_strength = 0.14
    state.subjective_state.reject_all = 0.94
    state.subjective_state.spontaneous = 0.78
    state.subjective_state.boundary = 0.81
    state.subjective_state.meaning_made = ["安静先于继续回应"]

    trigger = controller.endogenous_scheduler.build_trigger(
        state=state,
        context={},
        relation_state={"relationship_risk": 0.12},
        slow_variables={"affect_residue": 0.22, "memory_activation": 0.18},
        pool_state=state.motivation_pool_state,
    )

    assert trigger is not None
    assert trigger.source_metrics["subjective_pressure"] >= 0.42

    decision = controller._endogenous_suppression_decision(
        state=state,
        scenario="task",
        trigger=trigger,
    )

    assert decision.suppressed is False
    assert decision.reason == ""


def test_tick_grows_emergent_action_sketches_from_instinct_pressure(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    state = controller.load_runtime_state()
    state.body_energy = 0.1
    state.fatigue = 0.9
    state.memory_fragments = 0.85
    state.self_continuity = 0.26
    state.meaning_strength = 0.17
    state.subjective_state.reject_all = 0.84
    state.subjective_state.spontaneous = 0.73
    state.subjective_state.boundary = 0.79
    state.subjective_state.felt = ["累", "想停下"]
    state.subjective_state.meaning_made = ["先吸收，不先表演回应"]
    controller._save_state(state, sync=True)

    result = controller.tick(
        RoundEvent(source="user", content="you do not have to answer right away", target="user", cue="pause"),
        scenario="chat",
        mode="interactive",
    )
    updated = controller.load_runtime_state()

    assert updated.emergent_action_sketches
    sketch = updated.emergent_action_sketches[0]
    assert sketch.name
    assert sketch.signal_sources
    assert sketch.support_actions
    assert sketch.growth_score > 0.0
    assert result.trace.state_snapshot["emergent_action_sketches"]


def test_emergent_action_sketch_becomes_formal_contributor_on_following_round(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    state = controller.load_runtime_state()
    state.body_energy = 0.09
    state.fatigue = 0.9
    state.memory_fragments = 0.87
    state.self_continuity = 0.24
    state.meaning_strength = 0.14
    state.subjective_state.reject_all = 0.86
    state.subjective_state.spontaneous = 0.75
    state.subjective_state.felt = ["累", "先停"]
    state.subjective_state.meaning_made = ["先吸收再输出"]
    controller._save_state(state, sync=True)

    controller.tick(
        RoundEvent(source="user", content="you can take it in first", target="user", cue="pause"),
        scenario="chat",
        mode="interactive",
    )
    second = controller.tick(
        RoundEvent(source="user", content="stay with the internal pull", target="user", cue="stay"),
        scenario="chat",
        mode="interactive",
    )

    audit_rows = second.trace.probability_field["action"]["contribution_audit"]

    assert any(row["module_name"] == "EmergentActionSketch" for row in audit_rows)


def test_tick_grows_emergent_action_sketches_and_surfaces_them_in_why_and_replay(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    state = controller.load_runtime_state()
    state.body_energy = 0.08
    state.fatigue = 0.9
    state.memory_fragments = 0.86
    state.self_continuity = 0.24
    state.meaning_strength = 0.14
    state.subjective_state.reject_all = 0.89
    state.subjective_state.spontaneous = 0.76
    state.subjective_state.felt = ["累", "想停", "先往里收"]
    state.subjective_state.meaning_made = ["先吸收再决定是否回应"]
    controller._save_state(state, sync=True)

    result = controller.tick(
        RoundEvent(source="user", content="you can stay quiet if you need", target="user", cue="quiet"),
        scenario="chat",
        mode="interactive",
    )

    why_payload = controller.why_this(result.round_id)
    replay = controller.replay(result.round_id, seed=11)

    assert why_payload["emergent_action_sketches"]
    assert replay["emergent_action_sketches"]
    assert any(item["growth_score"] > 0.0 for item in replay["emergent_action_sketches"])
    assert any(item["emergent_sketches"] for item in replay["counterfactual_replays"])
    assert any(item["counterfactual_preview"]["action"] == item["action"] for item in replay["counterfactual_replays"])


def test_tick_surfaces_high_dimensional_collapse_and_anchor_alignment(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    state = controller.load_runtime_state()
    state.body_energy = 0.12
    state.fatigue = 0.84
    state.memory_fragments = 0.79
    state.self_continuity = 0.31
    state.meaning_strength = 0.34
    state.subjective_state.reject_all = 0.58
    state.subjective_state.spontaneous = 0.72
    state.subjective_state.boundary = 0.62
    state.subjective_state.felt = ["想停", "先往里收"]
    state.subjective_state.meaning_made = ["先吸收再回应"]
    controller._save_state(state, sync=True)

    result = controller.tick(
        RoundEvent(source="user", content="you can pause and absorb first", target="user", cue="pause"),
        scenario="chat",
        mode="interactive",
    )

    why_payload = controller.why_this(result.round_id)
    replay = controller.replay(result.round_id, seed=13)

    assert "personality_anchor" in why_payload
    assert "self_continuity_derivation" in why_payload
    assert "high_dimensional_collapse" in why_payload
    assert "anchor_alignment" in why_payload
    assert len(why_payload["high_dimensional_collapse"]["random_point"]) == 4
    assert set(why_payload["high_dimensional_collapse"]["normalized_axes"]) == {"E", "F", "S", "M"}
    assert why_payload["high_dimensional_collapse"]["match_scores"]
    assert "high_dimensional_collapse" in replay
    assert "anchor_alignment" in replay
    assert replay["high_dimensional_collapse"]["selected_action"] in replay["candidate_distribution"]


def test_instinct_field_collapse_trace_surfaces_vector_aliases_and_subject_vector(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    state = controller.load_runtime_state()
    state.body_energy = 0.09
    state.fatigue = 0.9
    state.memory_fragments = 0.84
    state.self_continuity = 0.26
    state.meaning_strength = 0.18
    state.subjective_state.reject_all = 0.87
    state.subjective_state.spontaneous = 0.72
    state.subjective_state.boundary = 0.78
    state.subjective_state.felt = ["累", "收回去"]
    state.subjective_state.meaning_made = ["先内收再决定"]

    contribution = controller._build_instinct_field_contribution(
        state=state,
        context={"cue": "pause", "closeness": 0.16, "interference": 0.33},
        relation_state={"closeness": 0.16, "boundary_level": 0.78, "relationship_risk": 0.61},
        slow_variables={"affect_residue": 0.4, "memory_activation": 0.52, "resource_scarcity": 0.58},
    )

    assert contribution is not None
    collapse_trace = state.instinct_field.collapse_trace

    assert collapse_trace["random_point"] == collapse_trace["subject_vector"]
    assert collapse_trace["coupled_axes"] == collapse_trace["subject_vector"]
    assert collapse_trace["normalized_axes"] == collapse_trace["v_main"]
    assert collapse_trace["v_mod"]
    assert collapse_trace["v_anchor"]
    assert collapse_trace["action_vectors"]


def test_personality_anchor_uses_subject_vector_ema_update_without_new_storage(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    state = controller.load_runtime_state()
    state.round_count = 4
    state.personality_anchor.axis_baseline = {"E": 0.5, "F": 0.5, "S": 0.5, "M": 0.5}
    state.instinct_field.collapse_trace = {
        "subject_vector": {"E": 0.12, "F": 0.89, "S": 0.83, "M": 0.18},
        "match_scores": {"rest": 0.92, "nothing": 0.74, "respond": 0.18},
    }

    updated = controller._update_personality_anchor(
        state,
        identity_evidence={"anchors": ["felt:pause"], "signature": "ema-vector"},
    )

    assert updated.axis_baseline["F"] > 0.58
    assert updated.axis_baseline["S"] > 0.58
    assert updated.axis_baseline["E"] < 0.45
    assert updated.axis_baseline["M"] < 0.45


def test_personality_anchor_uses_recent_round_window_without_full_history_scan(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    controller.tick(
        RoundEvent(
            source="user",
            content="请留下一轮最近动作，供人格锚点读取。",
            target="user",
            cue="anchor-window",
        ),
        scenario="task",
        mode="interactive",
    )
    state = controller.load_runtime_state()
    state.round_count = 4
    state.personality_anchor.axis_baseline = {"E": 0.5, "F": 0.5, "S": 0.5, "M": 0.5}
    state.instinct_field.collapse_trace = {
        "subject_vector": {"E": 0.2, "F": 0.8, "S": 0.7, "M": 0.3},
        "match_scores": {"plan": 0.91, "respond": 0.42},
    }

    monkeypatch.setattr(
        controller.trace_store,
        "list_rounds",
        lambda: (_ for _ in ()).throw(AssertionError("_update_personality_anchor should not scan full history")),
    )

    updated = controller._update_personality_anchor(
        state,
        identity_evidence={"anchors": ["felt:focused"], "signature": "recent-round-window"},
    )

    assert updated.axis_baseline["F"] > 0.55
    assert updated.action_bias["plan"] > 0.0


def test_emergent_action_contribution_projects_sketch_vector_back_into_neighbor_actions(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    state = controller.load_runtime_state()
    state.emergent_action_sketches = [
        {
            "name": "absorb:absorb+recall",
            "signal_sources": ["winner_region:absorb", "action:absorb", "action:recall", "sampled:absorb"],
            "support_actions": {"absorb": 0.86},
            "target_action_map": {"absorb": 0.86},
            "growth_score": 0.84,
            "status": "formalized",
            "anchor_alignment": 0.71,
            "stability": 3,
        }
    ]

    contribution = controller._build_emergent_action_contribution(state)

    assert contribution is not None
    assert contribution.modulated_delta["absorb"] > 0.0
    assert contribution.modulated_delta.get("recall", 0.0) > 0.0


def test_noninteractive_round_recovers_fatigue_and_fragments(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    state = controller.load_runtime_state()
    state.body_energy = 0.18
    state.fatigue = 0.86
    state.memory_fragments = 0.82
    state.self_continuity = 0.29
    state.meaning_strength = 0.22
    state.subjective_state.spontaneous = 0.64
    state.subjective_state.reject_all = 0.73
    controller._save_state(state, sync=True)

    before_fatigue = state.fatigue
    before_fragments = state.memory_fragments

    controller.tick(
        RoundEvent(source="system", content="sleep cycle integration", target="self", cue="sleep"),
        scenario="chat",
        mode="sleep",
    )
    updated = controller.load_runtime_state()

    assert updated.fatigue < before_fatigue
    assert updated.memory_fragments < before_fragments


def test_repeated_anchor_aligned_rounds_formalize_emergent_action_sketch(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    state = controller.load_runtime_state()
    state.body_energy = 0.1
    state.fatigue = 0.88
    state.memory_fragments = 0.84
    state.self_continuity = 0.27
    state.meaning_strength = 0.24
    state.subjective_state.reject_all = 0.79
    state.subjective_state.spontaneous = 0.77
    state.subjective_state.boundary = 0.75
    state.subjective_state.felt = ["先停", "先吸收"]
    state.subjective_state.meaning_made = ["先内收再决定是否外显"]
    controller._save_state(state, sync=True)

    for cue in ("pause", "quiet", "absorb", "settle"):
        controller.tick(
            RoundEvent(source="user", content=f"stay with the {cue} pull", target="user", cue=cue),
            scenario="chat",
            mode="interactive",
        )

    updated = controller.load_runtime_state()
    why_payload = controller.why_this(updated.round_count)

    assert any(sketch.status == "formalized" for sketch in updated.emergent_action_sketches)
    assert why_payload["emergent_action_formalization"]
