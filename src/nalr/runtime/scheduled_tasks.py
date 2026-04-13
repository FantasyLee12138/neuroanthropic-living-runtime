from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from nalr.runtime.metadata import utc_now_iso
from nalr.schemas.models import to_dict


def _parse_iso(value: str | None) -> datetime:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError("missing ISO timestamp")
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _iso_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass
class ScheduledTaskSchedule:
    schedule_type: Literal["disabled", "hourly", "weekly"] = "disabled"
    interval_hours: int = 1
    weekday: int | None = None
    hour: int | None = None
    minute: int | None = None

    def __post_init__(self) -> None:
        if self.schedule_type == "hourly":
            self.interval_hours = max(1, int(self.interval_hours or 1))
        if self.schedule_type == "weekly":
            if self.weekday is None or self.hour is None or self.minute is None:
                raise ValueError("weekly schedule requires weekday, hour, and minute")
            self.weekday = max(0, min(6, int(self.weekday)))
            self.hour = max(0, min(23, int(self.hour)))
            self.minute = max(0, min(59, int(self.minute)))


@dataclass
class ScheduledTaskSpec:
    task_id: str
    skill_name: str
    prompt: str = ""
    task_payload: dict[str, Any] = field(default_factory=dict)
    toolset_policy: dict[str, Any] = field(default_factory=dict)
    fresh_session: bool = True
    schedule: ScheduledTaskSchedule = field(default_factory=ScheduledTaskSchedule)
    status: Literal["idle", "running", "disabled"] = "idle"
    created_at: str = ""
    updated_at: str = ""
    next_run_at: str | None = None
    last_run_at: str | None = None
    last_run_id: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.schedule, dict):
            self.schedule = ScheduledTaskSchedule(**self.schedule)
        self.prompt = str(self.prompt or "").strip()
        if not isinstance(self.task_payload, dict):
            self.task_payload = {}
        if not isinstance(self.toolset_policy, dict):
            self.toolset_policy = {}
        if self.schedule.schedule_type == "disabled":
            self.status = "disabled"


@dataclass
class ScheduledTaskRunRecord:
    run_id: str
    task_id: str
    status: Literal["queued", "running", "succeeded", "failed"] = "queued"
    scheduled_for: str | None = None
    recorded_at: str = ""
    error_message: str | None = None


def compute_next_run_at(schedule: ScheduledTaskSchedule, *, reference_at: str) -> str | None:
    if schedule.schedule_type == "disabled":
        return None

    reference_dt = _parse_iso(reference_at)
    if schedule.schedule_type == "hourly":
        return _iso_utc(reference_dt + timedelta(hours=max(1, int(schedule.interval_hours or 1))))

    if schedule.schedule_type == "weekly":
        candidate = reference_dt.replace(
            hour=int(schedule.hour or 0),
            minute=int(schedule.minute or 0),
            second=0,
            microsecond=0,
        )
        days_ahead = (int(schedule.weekday or 0) - reference_dt.weekday()) % 7
        candidate = candidate + timedelta(days=days_ahead)
        if candidate <= reference_dt:
            candidate = candidate + timedelta(days=7)
        return _iso_utc(candidate)

    raise ValueError(f"unsupported schedule_type: {schedule.schedule_type}")


class ScheduledTaskStore:
    def __init__(self, runtime_dir: Path) -> None:
        self.runtime_dir = Path(runtime_dir)
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.tasks_path = self.runtime_dir / "scheduled_tasks.json"
        self.runs_path = self.runtime_dir / "scheduled_task_runs.json"
        for path in (self.tasks_path, self.runs_path):
            if not path.exists():
                self._write_json_atomic(path, [])

    def _write_json_atomic(self, path: Path, payload: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temp_path.replace(path)

    def _read_json_list(self, path: Path) -> list[dict[str, Any]]:
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError:
            return []
        if not raw.strip():
            return []
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return []
        return list(payload) if isinstance(payload, list) else []

    def _load_tasks(self) -> list[ScheduledTaskSpec]:
        return [ScheduledTaskSpec(**item) for item in self._read_json_list(self.tasks_path) if isinstance(item, dict)]

    def _write_tasks(self, tasks: list[ScheduledTaskSpec]) -> None:
        ordered = sorted(tasks, key=lambda item: item.task_id)
        self._write_json_atomic(self.tasks_path, [to_dict(item) for item in ordered])

    def _load_runs(self) -> list[ScheduledTaskRunRecord]:
        return [ScheduledTaskRunRecord(**item) for item in self._read_json_list(self.runs_path) if isinstance(item, dict)]

    def _write_runs(self, runs: list[ScheduledTaskRunRecord]) -> None:
        ordered = sorted(runs, key=lambda item: (item.recorded_at, item.run_id))
        self._write_json_atomic(self.runs_path, [to_dict(item) for item in ordered])

    def upsert_task(self, spec: ScheduledTaskSpec, *, recorded_at: str | None = None) -> ScheduledTaskSpec:
        effective_recorded_at = recorded_at or utc_now_iso()
        tasks = self._load_tasks()
        task_by_id = {task.task_id: task for task in tasks}
        existing = task_by_id.get(spec.task_id)
        next_run_at = compute_next_run_at(spec.schedule, reference_at=effective_recorded_at)
        saved = ScheduledTaskSpec(
            task_id=spec.task_id,
            skill_name=spec.skill_name,
            prompt=spec.prompt,
            task_payload=dict(spec.task_payload),
            toolset_policy=dict(spec.toolset_policy),
            fresh_session=bool(spec.fresh_session),
            schedule=spec.schedule,
            status="disabled" if spec.schedule.schedule_type == "disabled" else "idle",
            created_at=existing.created_at if existing is not None and existing.created_at else effective_recorded_at,
            updated_at=effective_recorded_at,
            next_run_at=next_run_at,
            last_run_at=existing.last_run_at if existing is not None else spec.last_run_at,
            last_run_id=existing.last_run_id if existing is not None else spec.last_run_id,
        )
        task_by_id[saved.task_id] = saved
        self._write_tasks(list(task_by_id.values()))
        return saved

    def read_task(self, task_id: str) -> ScheduledTaskSpec:
        for task in self._load_tasks():
            if task.task_id == task_id:
                return task
        raise FileNotFoundError(f"scheduled task {task_id} not found")

    def list_tasks(self) -> list[ScheduledTaskSpec]:
        return self._load_tasks()

    def due_tasks(self, *, reference_at: str | None = None) -> list[ScheduledTaskSpec]:
        reference_dt = _parse_iso(reference_at or utc_now_iso())
        due: list[ScheduledTaskSpec] = []
        for task in self._load_tasks():
            if task.schedule.schedule_type == "disabled" or task.status != "idle" or not task.next_run_at:
                continue
            try:
                next_run_dt = _parse_iso(task.next_run_at)
            except ValueError:
                continue
            if next_run_dt <= reference_dt:
                due.append(task)
        due.sort(key=lambda item: str(item.next_run_at or ""))
        return due

    def mark_task_running(self, task_id: str, *, run_id: str, recorded_at: str | None = None) -> ScheduledTaskSpec:
        effective_recorded_at = recorded_at or utc_now_iso()
        tasks = self._load_tasks()
        updated: list[ScheduledTaskSpec] = []
        written: ScheduledTaskSpec | None = None
        for task in tasks:
            if task.task_id != task_id:
                updated.append(task)
                continue
            written = ScheduledTaskSpec(
                task_id=task.task_id,
                skill_name=task.skill_name,
                prompt=task.prompt,
                task_payload=dict(task.task_payload),
                toolset_policy=dict(task.toolset_policy),
                fresh_session=bool(task.fresh_session),
                schedule=task.schedule,
                status="disabled" if task.schedule.schedule_type == "disabled" else "running",
                created_at=task.created_at,
                updated_at=effective_recorded_at,
                next_run_at=task.next_run_at,
                last_run_at=task.last_run_at,
                last_run_id=run_id,
            )
            updated.append(written)
        if written is None:
            raise FileNotFoundError(f"scheduled task {task_id} not found")
        self._write_tasks(updated)
        return written

    def append_run(self, record: ScheduledTaskRunRecord, *, recorded_at: str | None = None) -> ScheduledTaskRunRecord:
        effective_recorded_at = recorded_at or utc_now_iso()
        saved_record = ScheduledTaskRunRecord(
            run_id=record.run_id,
            task_id=record.task_id,
            status=record.status,
            scheduled_for=record.scheduled_for,
            recorded_at=effective_recorded_at,
            error_message=record.error_message,
        )
        runs = self._load_runs()
        runs.append(saved_record)
        self._write_runs(runs)

        tasks = self._load_tasks()
        updated_tasks: list[ScheduledTaskSpec] = []
        for task in tasks:
            if task.task_id != record.task_id:
                updated_tasks.append(task)
                continue
            keep_running = record.status == "running"
            next_run_at = task.next_run_at if keep_running else compute_next_run_at(task.schedule, reference_at=effective_recorded_at)
            status = "disabled" if task.schedule.schedule_type == "disabled" else ("running" if keep_running else "idle")
            updated_tasks.append(
                ScheduledTaskSpec(
                    task_id=task.task_id,
                    skill_name=task.skill_name,
                    prompt=task.prompt,
                    task_payload=dict(task.task_payload),
                    toolset_policy=dict(task.toolset_policy),
                    fresh_session=bool(task.fresh_session),
                    schedule=task.schedule,
                    status=status,
                    created_at=task.created_at,
                    updated_at=effective_recorded_at,
                    next_run_at=next_run_at,
                    last_run_at=effective_recorded_at if not keep_running else task.last_run_at,
                    last_run_id=record.run_id,
                )
            )
        self._write_tasks(updated_tasks)
        return saved_record

    def list_runs(self, *, task_id: str | None = None) -> list[ScheduledTaskRunRecord]:
        runs = self._load_runs()
        if task_id is None:
            return runs
        return [run for run in runs if run.task_id == task_id]
