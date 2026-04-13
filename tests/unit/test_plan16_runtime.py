from pathlib import Path
import json

import nalr.runtime.controller as controller_module

from nalr.runtime.controller import RuntimeController
from nalr.runtime.scheduled_tasks import ScheduledTaskSchedule, ScheduledTaskSpec
from nalr.terminal_bridge.session import TerminalSessionState
from nalr.schemas.models import RoundEvent


CONFIG_ROOT = Path(__file__).resolve().parents[2] / "config"


def test_state_payload_surfaces_plan16_subject_meaning_and_agency_contracts(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    controller.tick(
        RoundEvent(
            source="user",
            content="Please stabilize your current sense of self and explain what matters.",
            target="user",
            cue="selfhood",
        ),
        scenario="chat",
        mode="interactive",
    )

    payload = controller.state_payload()

    assert payload["subject_kernel"]["display_name"]
    assert payload["subject_kernel"]["boundary_principles"]
    assert payload["subject_kernel"]["core_commitments"]
    assert payload["meaning_system"]["survival_narrative"]
    assert isinstance(payload["meaning_system"]["sources"], list)
    assert isinstance(payload["purpose_memory"], list)
    assert isinstance(payload["relationship_commitments"], list)
    assert isinstance(payload["proactive_backlog"], list)
    assert payload["agency_loop"]["summary"]
    assert "suppressed_actions" in payload["agency_loop"]
    assert "budget" in payload["agency_loop"]


def test_state_hot_payload_skips_flush_pending_io(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    monkeypatch.setattr(
        controller,
        "flush_pending_io",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("hot path should not flush pending io")),
    )

    payload = controller.state_hot_payload()

    assert "performance" in payload
    assert "subject_kernel" in payload
    assert "agency_loop" in payload


def test_workbench_round_payload_aggregates_selected_round_views(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    controller.tick(
        RoundEvent(
            source="user",
            content="Please produce one round so the workbench can read it in a single payload.",
            target="user",
            cue="round-read-model",
        ),
        scenario="chat",
        mode="interactive",
    )

    payload = controller.workbench_round_payload(1, action="respond")

    assert payload["trace"]["round_id"] == 1
    assert payload["why"]["round_id"] == 1
    assert payload["contributions"]["round_id"] == 1
    assert payload["probability"]["source_round_id"] == 1
    assert payload["whyNot"]["round_id"] == 1
    assert payload["replay"]["round_id"] == 1
    assert payload["thought"]["round_id"] == 1
    assert "initiative" in payload["initiativeWhy"]
    assert payload["state_truth"]["authoritative_state"]["runtime_revision"] >= 1
    assert payload["memory_evidence"]["behavioral_consequence"]["selected_action"] == payload["trace"]["sampled_action"]
    assert payload["competition_evidence"]["selected_winner"] == payload["trace"]["sampled_action"]
    assert payload["continuity_evidence"]["continuity_nonce"] == payload["trace"]["continuity_nonce"]


def test_state_payload_surfaces_subjectivity_contract_state(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    controller.tick(
        RoundEvent(
            source="user",
            content="请记住我们要补齐 1.6，并继续保持连续性。",
            target="user",
            cue="1.6",
        ),
        scenario="chat",
        mode="interactive",
    )

    payload = controller.state_payload()

    assert payload["history_burden"]["repair_scars"] == []
    assert "continuity_residues" in payload["history_burden"]
    assert "suppressed_but_recoverable_patterns" in payload["history_burden"]
    assert "autonomy" in payload["arbitration_state"]
    assert "round" in payload["arbitration_state"]
    assert payload["state_truth"]["projection"]["workbench_read_model"] == "derived"


def test_state_truth_payload_detects_terminal_session_shadow_truth(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    controller.terminal_sessions.write(TerminalSessionState(session_id="sess-shadow", cwd=str(tmp_path)))

    session_path = controller.terminal_sessions._path_for("sess-shadow")
    payload = json.loads(session_path.read_text(encoding="utf-8"))
    payload["truth_contract"] = {
        "projection_only": False,
        "authoritative": True,
        "source": "shadow_truth_cache",
    }
    session_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    state = controller.load_runtime_state()
    controller.sync_plan16_state(state)
    truth = controller.state_truth_payload(state)

    assert truth["consistency_checks"]["session_cache_is_projection_only"] is False
    assert any("sess-shadow" in item for item in truth["consistency_checks"]["projection_only_truth_fields"])


def test_plan16_subject_and_meaning_reuse_persisted_purpose_memory(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    first_recorded_at = "2026-04-12T08:00:00Z"
    second_recorded_at = "2026-04-12T09:30:00Z"
    goal = "把 1.6 补齐"

    monkeypatch.setattr(controller_module, "utc_now_iso", lambda: first_recorded_at)
    initial_state = controller.load_runtime_state()
    initial_state.current_goal = goal
    controller.sync_plan16_state(initial_state)
    controller._save_state(initial_state, sync=True)

    monkeypatch.setattr(controller_module, "utc_now_iso", lambda: second_recorded_at)
    followup_state = controller.load_runtime_state()
    followup_state.current_goal = None
    controller._save_state(followup_state, sync=True)

    subject_payload = controller.subject_status()["subject_kernel"]
    meaning_payload = controller.meaning_status()
    purpose_memory = meaning_payload["purpose_memory"]
    meaning_sources = meaning_payload["meaning_system"]["sources"]

    assert any(item["summary"] == f"当前目标：{goal}" and item["recorded_at"] == first_recorded_at for item in purpose_memory)
    assert any(item["source"] == "purpose_memory" and goal in item["label"] for item in meaning_sources)
    assert goal in subject_payload["current_narrative"]


def test_plan16_agency_status_surfaces_due_scheduled_tasks_in_backlog(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    controller.scheduled_task_store.upsert_task(
        ScheduledTaskSpec(
            task_id="scheduled-runtime-review",
            skill_name="runtime_review",
            prompt="检查 runtime 闭环还有哪些未收口的点。",
            schedule=ScheduledTaskSchedule(schedule_type="hourly", interval_hours=1),
        ),
        recorded_at="2000-01-01T00:00:00Z",
    )

    payload = controller.agency_status()
    backlog = payload["proactive_backlog"]

    assert any(item["kind"] == "scheduled_task" and "runtime 闭环" in item["summary"] for item in backlog)
    assert payload["agency_loop"]["budget"]["scheduled_due"] == 1
