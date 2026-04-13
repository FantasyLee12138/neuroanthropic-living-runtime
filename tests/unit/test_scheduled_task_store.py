from __future__ import annotations

from nalr.runtime.scheduled_tasks import (
    ScheduledTaskRunRecord,
    ScheduledTaskSchedule,
    ScheduledTaskSpec,
    ScheduledTaskStore,
    compute_next_run_at,
)


def test_compute_next_run_at_supports_disabled_hourly_and_weekly_modes() -> None:
    assert compute_next_run_at(
        ScheduledTaskSchedule(schedule_type="disabled"),
        reference_at="2026-04-12T10:15:00Z",
    ) is None

    assert compute_next_run_at(
        ScheduledTaskSchedule(schedule_type="hourly", interval_hours=6),
        reference_at="2026-04-12T10:15:00Z",
    ) == "2026-04-12T16:15:00Z"

    assert compute_next_run_at(
        ScheduledTaskSchedule(schedule_type="weekly", weekday=0, hour=9, minute=30),
        reference_at="2026-04-12T10:15:00Z",
    ) == "2026-04-13T09:30:00Z"
    assert compute_next_run_at(
        ScheduledTaskSchedule(schedule_type="weekly", weekday=6, hour=9, minute=0),
        reference_at="2026-04-12T10:15:00Z",
    ) == "2026-04-19T09:00:00Z"


def test_upsert_task_persists_spec_and_computed_next_run_at(tmp_path) -> None:
    store = ScheduledTaskStore(tmp_path)

    written = store.upsert_task(
        ScheduledTaskSpec(
            task_id="task-hourly",
            skill_name="digest_skill",
            task_payload={"topic": "hermes"},
            schedule=ScheduledTaskSchedule(schedule_type="hourly", interval_hours=4),
        ),
        recorded_at="2026-04-12T08:00:00Z",
    )

    read_back = store.read_task("task-hourly")
    listed = store.list_tasks()

    assert written.status == "idle"
    assert written.created_at == "2026-04-12T08:00:00Z"
    assert written.updated_at == "2026-04-12T08:00:00Z"
    assert written.next_run_at == "2026-04-12T12:00:00Z"
    assert read_back == written
    assert listed == [written]


def test_append_run_updates_task_state_and_next_run_at(tmp_path) -> None:
    store = ScheduledTaskStore(tmp_path)
    store.upsert_task(
        ScheduledTaskSpec(
            task_id="task-weekly",
            skill_name="weekly_sync",
            schedule=ScheduledTaskSchedule(schedule_type="weekly", weekday=3, hour=14, minute=5),
        ),
        recorded_at="2026-04-12T08:00:00Z",
    )

    running = store.mark_task_running(
        "task-weekly",
        run_id="run-1",
        recorded_at="2026-04-14T14:00:00Z",
    )

    assert running.status == "running"
    assert running.last_run_id == "run-1"
    assert running.next_run_at == "2026-04-16T14:05:00Z"

    recorded = store.append_run(
        ScheduledTaskRunRecord(
            run_id="run-1",
            task_id="task-weekly",
            status="succeeded",
            scheduled_for="2026-04-16T14:05:00Z",
        ),
        recorded_at="2026-04-16T14:07:00Z",
    )

    task = store.read_task("task-weekly")
    runs = store.list_runs(task_id="task-weekly")

    assert recorded.recorded_at == "2026-04-16T14:07:00Z"
    assert task.status == "idle"
    assert task.last_run_at == "2026-04-16T14:07:00Z"
    assert task.last_run_id == "run-1"
    assert task.next_run_at == "2026-04-23T14:05:00Z"
    assert runs == [recorded]


def test_disabled_task_remains_disabled_after_run_append(tmp_path) -> None:
    store = ScheduledTaskStore(tmp_path)
    store.upsert_task(
        ScheduledTaskSpec(
            task_id="task-disabled",
            skill_name="noop",
            schedule=ScheduledTaskSchedule(schedule_type="disabled"),
        ),
        recorded_at="2026-04-12T08:00:00Z",
    )

    store.append_run(
        ScheduledTaskRunRecord(
            run_id="run-disabled",
            task_id="task-disabled",
            status="failed",
            error_message="disabled tasks should not schedule follow-ups",
        ),
        recorded_at="2026-04-12T08:05:00Z",
    )

    task = store.read_task("task-disabled")

    assert task.status == "disabled"
    assert task.next_run_at is None
    assert task.last_run_at == "2026-04-12T08:05:00Z"
