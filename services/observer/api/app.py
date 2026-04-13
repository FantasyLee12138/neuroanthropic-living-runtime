from __future__ import annotations

import asyncio
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime
import json
import math
import os
import threading
import time
from pathlib import Path
from uuid import uuid4

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse

from nalr.cil.runtime import build_endogenous_status_payload, run_endogenous_tick_payload
from nalr.runtime.controller import RuntimeController
from nalr.schemas.models import to_dict
from nalr.terminal_bridge.handlers import TerminalEventHandler
from nalr.terminal_bridge.protocol import ProtocolError, build_outbound_event
from nalr.terminal_bridge.session import TerminalSessionState, TerminalSessionStore
from services.observer.api.web_sessions import WebSessionBroker, build_workbench_cards

OBSERVER_MAIN_SESSION_ID = "observer-main"
ORIGINAL_AUTONOMY_STEP = RuntimeController.autonomy_step


def create_app(
    project_root: Path | str | None = None,
    config_root: Path | str | None = None,
    service_metadata: dict | None = None,
) -> FastAPI:
    project_root_path = Path(project_root) if project_root else Path.cwd()
    effective_config_root = Path(config_root) if config_root else Path(os.environ.get("NALR_CONFIG_DIR", project_root_path / "config"))
    controller = RuntimeController(project_root=project_root_path, config_root=effective_config_root)
    terminal_sessions = TerminalSessionStore(controller.runtime_dir)
    terminal_handler = TerminalEventHandler(controller)
    web_broker = WebSessionBroker()
    background_runners_disabled = str(os.environ.get("NALR_OBSERVER_DISABLE_BACKGROUND_RUNNERS") or "").strip() == "1"
    transcript_sync_offsets: dict[str, int] = {}
    autonomy_thread: threading.Thread | None = None
    autonomy_stop_event = threading.Event()
    autonomy_lock = threading.Lock()
    autonomy_worker_lock = threading.Lock()
    autonomy_worker_state_lock = threading.Lock()
    monologue_thread: threading.Thread | None = None
    monologue_stop_event = threading.Event()
    monologue_lock = threading.Lock()
    monologue_worker_lock = threading.Lock()
    initiative_thread: threading.Thread | None = None
    initiative_stop_event = threading.Event()
    initiative_lock = threading.Lock()
    initiative_worker_lock = threading.Lock()
    scheduled_task_thread: threading.Thread | None = None
    scheduled_task_stop_event = threading.Event()
    scheduled_task_lock = threading.Lock()
    autonomy_phase_context = threading.local()
    runtime_lock = threading.RLock()
    session_turn_locks: dict[str, threading.Lock] = {}
    session_turn_locks_guard = threading.Lock()
    session_turn_queues: dict[str, deque[dict[str, object]]] = {}
    session_turn_workers: dict[str, threading.Thread] = {}
    session_turn_workers_guard = threading.Lock()
    service_started_at = str((service_metadata or {}).get("started_at") or time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    service_instance_id = str((service_metadata or {}).get("instance_id") or os.environ.get("NALR_OBSERVER_INSTANCE_ID") or uuid4().hex)
    service_host = str((service_metadata or {}).get("host") or os.environ.get("NALR_OBSERVER_HOST") or "127.0.0.1")
    port_value = (service_metadata or {}).get("port") or os.environ.get("NALR_OBSERVER_PORT") or 8765
    try:
        service_port = int(port_value)
    except (TypeError, ValueError):
        service_port = 8765
    service_url = str((service_metadata or {}).get("url") or f"http://{service_host}:{service_port}/dashboard")
    pid_value = (service_metadata or {}).get("pid")
    try:
        service_pid = int(pid_value) if pid_value is not None else os.getpid()
    except (TypeError, ValueError):
        service_pid = os.getpid()
    service_probe_state: dict[str, object] = {
        "last_http_ok_at": service_started_at,
        "last_probe_error": "",
        "last_runtime_activity_at": service_started_at,
    }
    runtime_serial_depth = 0
    runtime_serial_guard = threading.Lock()
    autonomy_worker_state: dict[str, object] = {
        "phase": "idle",
        "phase_detail": "observer idle",
        "phase_started_at": service_started_at,
        "last_progress_at": service_started_at,
        "last_error": "",
        "updated_at": service_started_at,
    }
    producer_state_lock = threading.Lock()
    monologue_runner_state: dict[str, object] = {
        "runtime_busy": False,
        "last_busy_at": "",
        "busy_skip_count": 0,
        "overdue_seconds": 0.0,
    }
    initiative_runner_state: dict[str, object] = {
        "runtime_busy": False,
        "last_busy_at": "",
        "busy_skip_count": 0,
        "overdue_seconds": 0.0,
    }

    def _now_iso() -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%S%z")

    def _heartbeat_interval_seconds() -> float:
        try:
            return max(0.05, float(os.environ.get("NALR_AUTONOMY_HEARTBEAT_SECONDS", "2.0")))
        except (TypeError, ValueError):
            return 2.0

    def _producer_interval_seconds(env_name: str) -> float:
        try:
            return max(0.05, float(os.environ.get(env_name, str(_heartbeat_interval_seconds()))))
        except (TypeError, ValueError):
            return _heartbeat_interval_seconds()

    def _monologue_interval_seconds() -> float:
        return _producer_interval_seconds("NALR_MONOLOGUE_HEARTBEAT_SECONDS")

    def _initiative_interval_seconds() -> float:
        return _producer_interval_seconds("NALR_INITIATIVE_HEARTBEAT_SECONDS")

    def _scheduled_task_interval_seconds() -> float:
        return _producer_interval_seconds("NALR_SCHEDULED_TASK_HEARTBEAT_SECONDS")

    def _runtime_stall_threshold_seconds() -> float:
        return max(0.2, _heartbeat_interval_seconds() * 3.0)

    def _autonomy_phase_stall_threshold_seconds() -> float:
        return _runtime_stall_threshold_seconds()

    def _seconds_since_iso(timestamp: object) -> float | None:
        raw = str(timestamp or "").strip()
        if not raw:
            return None
        try:
            parsed = datetime.strptime(raw, "%Y-%m-%dT%H:%M:%S%z")
        except ValueError:
            return None
        return max(0.0, time.time() - parsed.timestamp())

    def _mark_runtime_activity() -> None:
        service_probe_state["last_runtime_activity_at"] = _now_iso()

    def _producer_snapshot(kind: str) -> dict[str, object]:
        with producer_state_lock:
            source = monologue_runner_state if kind == "monologue" else initiative_runner_state
            return dict(source)

    def _producer_mark_ready(kind: str, *, overdue_seconds: float = 0.0) -> None:
        with producer_state_lock:
            source = monologue_runner_state if kind == "monologue" else initiative_runner_state
            source["runtime_busy"] = False
            source["overdue_seconds"] = round(float(overdue_seconds or 0.0), 3)

    def _producer_mark_runtime_busy(kind: str, *, overdue_seconds: float = 0.0) -> None:
        with producer_state_lock:
            source = monologue_runner_state if kind == "monologue" else initiative_runner_state
            source["runtime_busy"] = True
            source["last_busy_at"] = _now_iso()
            source["busy_skip_count"] = int(source.get("busy_skip_count", 0) or 0) + 1
            source["overdue_seconds"] = round(float(overdue_seconds or 0.0), 3)

    def _autonomy_worker_set_phase(
        phase: str,
        *,
        detail: str = "",
        progress: bool = False,
        error: str = "",
    ) -> None:
        now = _now_iso()
        with autonomy_worker_state_lock:
            previous_phase = str(autonomy_worker_state.get("phase") or "idle")
            autonomy_worker_state["phase"] = phase
            autonomy_worker_state["phase_detail"] = detail
            autonomy_worker_state["updated_at"] = now
            if progress or phase != previous_phase:
                autonomy_worker_state["last_progress_at"] = now
            if error:
                autonomy_worker_state["last_error"] = error
            elif phase != "backoff":
                autonomy_worker_state["last_error"] = ""
            if phase != previous_phase:
                autonomy_worker_state["phase_started_at"] = now

    def _autonomy_worker_snapshot() -> dict[str, object]:
        with autonomy_worker_state_lock:
            snapshot = dict(autonomy_worker_state)
        phase = str(snapshot.get("phase") or "idle")
        phase_started_at = str(snapshot.get("phase_started_at") or service_started_at)
        last_progress_at = str(snapshot.get("last_progress_at") or phase_started_at)
        phase_waiting = phase in {"idle", "awaiting_model", "backoff", "executing_endogenous_tick"}
        phase_age_seconds = _seconds_since_iso(phase_started_at)
        progress_age_seconds = _seconds_since_iso(last_progress_at)
        phase_stalled = bool(
            _autonomy_runner_alive()
            and not phase_waiting
            and progress_age_seconds is not None
            and progress_age_seconds > _autonomy_phase_stall_threshold_seconds()
        )
        return {
            **snapshot,
            "phase": phase,
            "phase_started_at": phase_started_at,
            "last_progress_at": last_progress_at,
            "phase_waiting": phase_waiting,
            "phase_age_seconds": round(float(phase_age_seconds or 0.0), 3) if phase_age_seconds is not None else None,
            "last_progress_age_seconds": round(float(progress_age_seconds or 0.0), 3) if progress_age_seconds is not None else None,
            "phase_stalled": phase_stalled,
            "stalled": phase_stalled,
            "stall_seconds": round(float(progress_age_seconds or 0.0), 3) if progress_age_seconds is not None else None,
            "stall_threshold_seconds": round(_autonomy_phase_stall_threshold_seconds(), 3),
        }

    def _session_selection_diagnostics() -> dict[str, object]:
        selection = controller._initiative_session_selection()
        return {
            "fresh_session_available": bool(selection.get("fresh_session_available", False)),
            "selected_session_id": selection.get("selected_session_id"),
            "selected_session_age_seconds": selection.get("selected_session_age_seconds"),
            "selected_session_reason": selection.get("selected_session_reason"),
            "session_fresh": bool(selection.get("session_fresh", False)),
        }

    def _monologue_runtime_diagnostics() -> dict[str, object]:
        status = controller.monologue_status_lightweight()
        return {
            "monologue_overdue_seconds": status.get("overdue_seconds"),
            "monologue_next_pulse_due": bool(status.get("next_pulse_due", False)),
            "monologue_stale": bool(status.get("stale", False)),
            "monologue_status": status,
        }

    def _autonomy_worker_busy() -> bool:
        return bool(autonomy_worker_lock.locked())

    def _autonomy_runner_engaged() -> bool:
        if _autonomy_worker_busy():
            return True
        snapshot = _autonomy_worker_snapshot()
        phase = str(snapshot.get("phase") or "idle")
        return bool(_autonomy_runner_alive() and phase not in {"idle", "backoff"})

    def _autonomy_phase_activate(source: str) -> None:
        autonomy_phase_context.active = True
        autonomy_phase_context.source = source
        autonomy_phase_context.model_depth = 0

    def _autonomy_phase_deactivate() -> None:
        autonomy_phase_context.active = False
        autonomy_phase_context.source = ""
        autonomy_phase_context.model_depth = 0

    def _autonomy_model_phase_enter(route_name: str) -> None:
        if not bool(getattr(autonomy_phase_context, "active", False)):
            return
        depth = int(getattr(autonomy_phase_context, "model_depth", 0) or 0)
        autonomy_phase_context.model_depth = depth + 1
        if depth == 0:
            source = str(getattr(autonomy_phase_context, "source", "observer_heartbeat") or "observer_heartbeat")
            _autonomy_worker_set_phase(
                "awaiting_model",
                detail=f"{source}: awaiting {route_name or 'model'}",
                progress=True,
            )

    def _autonomy_model_phase_exit() -> None:
        if not bool(getattr(autonomy_phase_context, "active", False)):
            return
        depth = max(0, int(getattr(autonomy_phase_context, "model_depth", 0) or 0) - 1)
        autonomy_phase_context.model_depth = depth
        if depth == 0:
            source = str(getattr(autonomy_phase_context, "source", "observer_heartbeat") or "observer_heartbeat")
            _autonomy_worker_set_phase(
                "preparing",
                detail=f"{source}: autonomy executing",
                progress=True,
            )

    def _run_serialized(func, /, *args, **kwargs):
        nonlocal runtime_serial_depth
        _mark_runtime_activity()
        with runtime_serial_guard:
            runtime_serial_depth += 1
        try:
            with runtime_lock:
                return func(*args, **kwargs)
        finally:
            _mark_runtime_activity()
            with runtime_serial_guard:
                runtime_serial_depth = max(0, runtime_serial_depth - 1)

    def _try_run_serialized(func, /, *args, **kwargs):
        nonlocal runtime_serial_depth
        _mark_runtime_activity()
        acquired = runtime_lock.acquire(blocking=False)
        if not acquired:
            return False, None
        with runtime_serial_guard:
            runtime_serial_depth += 1
        try:
            return True, func(*args, **kwargs)
        finally:
            _mark_runtime_activity()
            runtime_lock.release()
            with runtime_serial_guard:
                runtime_serial_depth = max(0, runtime_serial_depth - 1)

    async def _run_serialized_async(func, /, *args, **kwargs):
        return await asyncio.to_thread(_run_serialized, func, *args, **kwargs)

    async def _try_run_serialized_async(func, /, *args, **kwargs):
        return await asyncio.to_thread(_try_run_serialized, func, *args, **kwargs)

    async def _run_readonly_async(func, /, *args, **kwargs):
        return await asyncio.to_thread(func, *args, **kwargs)

    def _runtime_serial_active() -> bool:
        with runtime_serial_guard:
            return runtime_serial_depth > 0

    def _install_autonomy_model_phase_hooks() -> None:
        router = controller.model_router
        if bool(getattr(router, "_observer_autonomy_phase_wrapped", False)):
            return

        original_generate = router.generate
        original_generate_config = router.generate_config

        def _wrapped_generate(route_name, request):
            _autonomy_model_phase_enter(str(route_name or "model"))
            try:
                return original_generate(route_name, request)
            finally:
                _autonomy_model_phase_exit()

        def _wrapped_generate_config(route, request):
            route_name = str(getattr(route, "name", "") or "model")
            _autonomy_model_phase_enter(route_name)
            try:
                return original_generate_config(route, request)
            finally:
                _autonomy_model_phase_exit()

        router.generate = _wrapped_generate
        router.generate_config = _wrapped_generate_config
        router._observer_autonomy_phase_wrapped = True

    _install_autonomy_model_phase_hooks()

    def _session_turn_lock(session_id: str) -> threading.Lock:
        with session_turn_locks_guard:
            lock = session_turn_locks.get(session_id)
            if lock is None:
                lock = threading.Lock()
                session_turn_locks[session_id] = lock
            return lock

    def _dispatch_user_turn_payload(session_id: str, payload: dict[str, object]) -> None:
        turn_lock = _session_turn_lock(session_id)
        with turn_lock:
            try:
                events = _run_serialized(terminal_handler.handle, dict(payload))
            except ProtocolError as exc:
                _append_broker_events(session_id, [build_outbound_event("error", session_id=session_id, message=str(exc))])
                return
            except FileNotFoundError as exc:
                _append_broker_events(session_id, [build_outbound_event("error", session_id=session_id, message=str(exc))])
                return
        try:
            _append_broker_events(session_id, events)
        finally:
            if session_id == OBSERVER_MAIN_SESSION_ID:
                _sync_session_transcript_events(session_id)

    def _session_turn_worker_loop(session_id: str) -> None:
        current = threading.current_thread()
        try:
            while True:
                with session_turn_workers_guard:
                    queue = session_turn_queues.get(session_id)
                    if not queue:
                        session_turn_workers.pop(session_id, None)
                        session_turn_queues.pop(session_id, None)
                        return
                    payload = dict(queue.popleft())
                try:
                    _dispatch_user_turn_payload(session_id, payload)
                except Exception as exc:
                    _append_broker_events(
                        session_id,
                        [build_outbound_event("error", session_id=session_id, message=str(exc))],
                    )
        finally:
            with session_turn_workers_guard:
                if session_turn_workers.get(session_id) is current:
                    session_turn_workers.pop(session_id, None)
                queue = session_turn_queues.get(session_id)
                if not queue:
                    session_turn_queues.pop(session_id, None)

    def _enqueue_session_turn(session_id: str, payload: dict[str, object]) -> None:
        worker: threading.Thread | None = None
        with session_turn_workers_guard:
            queue = session_turn_queues.get(session_id)
            if queue is None:
                queue = deque()
                session_turn_queues[session_id] = queue
            queue.append(dict(payload))
            current_worker = session_turn_workers.get(session_id)
            if current_worker is not None and current_worker.is_alive():
                return
            worker = threading.Thread(
                target=_session_turn_worker_loop,
                args=(session_id,),
                name=f"observer-user-turn-{session_id}",
                daemon=True,
            )
            session_turn_workers[session_id] = worker
        if worker is not None:
            worker.start()

    def _session_turn_active(session_id: str) -> bool:
        if _session_turn_lock(session_id).locked():
            return True
        with session_turn_workers_guard:
            queue = session_turn_queues.get(session_id)
            worker = session_turn_workers.get(session_id)
            return bool((queue and len(queue) > 0) or (worker is not None and worker.is_alive()))

    def _observer_turn_active() -> bool:
        return _session_turn_active(OBSERVER_MAIN_SESSION_ID)

    def _active_turn_sessions() -> list[str]:
        active_session_ids: set[str] = set()
        with session_turn_locks_guard:
            active_session_ids.update(session_id for session_id, lock in session_turn_locks.items() if lock.locked())
        with session_turn_workers_guard:
            active_session_ids.update(
                session_id
                for session_id, queue in session_turn_queues.items()
                if queue or (session_turn_workers.get(session_id) is not None and session_turn_workers[session_id].is_alive())
            )
        return sorted(active_session_ids)

    def _mark_transcript_synced(session_id: str) -> None:
        try:
            transcript_sync_offsets[session_id] = len(terminal_sessions.read(session_id).transcript_lines or [])
        except FileNotFoundError:
            transcript_sync_offsets.pop(session_id, None)

    def _append_broker_events(session_id: str, events: list[dict]) -> list[dict]:
        recorded = web_broker.append_many(session_id, events)
        _mark_transcript_synced(session_id)
        return recorded

    def _sync_session_transcript_events(session_id: str) -> list[dict]:
        try:
            session = terminal_sessions.read(session_id)
        except FileNotFoundError:
            transcript_sync_offsets.pop(session_id, None)
            return []
        transcript = list(session.transcript_lines or [])
        start_index = max(0, min(int(transcript_sync_offsets.get(session_id, 0) or 0), len(transcript)))
        pending_lines = transcript[start_index:]
        if not pending_lines:
            return []
        events: list[dict] = []
        for line in pending_lines:
            kind = str(line.get("kind") or "").strip()
            text = str(line.get("text") or "").strip()
            if kind != "assistant" or not text:
                continue
            events.append(
                {
                    "type": "assistant_final",
                    "session_id": session_id,
                    "message": text,
                    "recorded_at": line.get("recorded_at"),
                    "delivery_mode": line.get("delivery_mode"),
                    "initiative_proposal_id": line.get("initiative_proposal_id"),
                    "initiative_memory_cue": line.get("initiative_memory_cue"),
                    "initiative_topic_source": line.get("initiative_topic_source"),
                    "initiative_topic_relevance": line.get("initiative_topic_relevance"),
                    "round_id": line.get("round_id"),
                    "trace_ref": line.get("trace_ref"),
                    "event_log_ref": dict(line.get("event_log_ref") or {}) or None,
                }
            )
        transcript_sync_offsets[session_id] = len(transcript)
        if not events:
            return []
        events.append(terminal_handler._build_sidebar_snapshot_event(session, run_id=session.active_run_id or session.last_run_id))
        return web_broker.append_many(session_id, events)

    def _autonomy_run_cycle(*, source: str, block: bool) -> dict[str, object]:
        if not autonomy_worker_lock.acquire(blocking=block):
            return _autonomy_status_payload()
        try:
            status = controller.autonomy_runtime_status()
            if not status.get("enabled") or not status.get("running"):
                _autonomy_worker_set_phase("idle", detail="autonomy disabled or stopped", progress=True)
                return _autonomy_status_payload()
            if _observer_turn_active():
                _autonomy_worker_set_phase("backoff", detail="observer turn active", progress=True)
                return _autonomy_status_payload()

            _autonomy_worker_set_phase("preparing", detail=f"{source}: preparing autonomy step", progress=True)
            _autonomy_phase_activate(source)
            try:
                if type(controller).autonomy_step is not ORIGINAL_AUTONOMY_STEP:
                    controller.autonomy_step()
                    _autonomy_worker_set_phase("committing", detail=f"{source}: committing autonomy results", progress=True)
                    _run_serialized(_sync_session_transcript_events, OBSERVER_MAIN_SESSION_ID)
                else:
                    ticket = controller.prepare_autonomy_background()
                    execution = controller.execute_autonomy_background(ticket)
                    commit_phase = "executing_endogenous_tick" if str(ticket.get("action") or "") == "endogenous_tick" else "committing"
                    commit_detail = (
                        f"{source}: executing endogenous tick"
                        if commit_phase == "executing_endogenous_tick"
                        else f"{source}: committing autonomy results"
                    )
                    _autonomy_worker_set_phase(commit_phase, detail=commit_detail, progress=True)
                    _run_serialized(_autonomy_commit_cycle, ticket, execution, source)
            except RuntimeError as exc:
                message = str(exc)
                if "interpreter shutdown" in message or "after shutdown" in message:
                    raise
                raise
            finally:
                _autonomy_phase_deactivate()
            _autonomy_worker_set_phase("idle", detail=f"{source}: idle", progress=True)
            return _autonomy_status_payload()
        except Exception as exc:
            try:
                controller.stop_autonomy(reason=f"runner_exception:{type(exc).__name__}")
            except Exception:
                pass
            _autonomy_worker_set_phase("backoff", detail=f"{source}: {type(exc).__name__}", progress=True, error=str(exc))
            return _autonomy_status_payload()
        finally:
            autonomy_worker_lock.release()

    def _autonomy_commit_cycle(ticket: dict[str, object], execution: dict[str, object], source: str) -> dict[str, object]:
        result = dict(controller.commit_autonomy_background(dict(ticket), dict(execution)) or {})
        _autonomy_worker_set_phase("committing", detail=f"{source}: committing autonomy results", progress=True)
        _sync_session_transcript_events(OBSERVER_MAIN_SESSION_ID)
        return result

    def _monologue_run_cycle(*, source: str, block: bool) -> dict[str, object]:
        if not monologue_worker_lock.acquire(blocking=block):
            return {"committed": False, "reason": "monologue_worker_busy"}
        try:
            monologue_status = controller.monologue_status_lightweight()
            overdue_seconds = float(monologue_status.get("overdue_seconds", 0.0) or 0.0)
            ticket = controller.prepare_monologue_stream_advance()
            if not bool(ticket.get("due")):
                _producer_mark_ready("monologue", overdue_seconds=overdue_seconds)
                return {"committed": False, "reason": "not_due"}
            generated = controller.execute_monologue_stream_advance(ticket)
            executed, committed = _try_run_serialized(controller.commit_monologue_stream_advance, ticket, generated)
            if not executed:
                _producer_mark_runtime_busy("monologue", overdue_seconds=overdue_seconds)
                return {"committed": False, "reason": "runtime_busy", "source": source}
            refreshed_status = controller.monologue_status_lightweight()
            _producer_mark_ready("monologue", overdue_seconds=float(refreshed_status.get("overdue_seconds", 0.0) or 0.0))
            return dict(committed or {})
        finally:
            monologue_worker_lock.release()

    def _initiative_run_cycle(*, source: str, block: bool) -> dict[str, object]:
        if not initiative_worker_lock.acquire(blocking=block):
            return {"committed": False, "reason": "initiative_worker_busy"}
        try:
            latest_view = controller.trace_store.recent_round_signal_views(limit=1)
            latest_recorded_at = latest_view[-1].get("recorded_at") if latest_view else None
            ticket = controller.prepare_initiative_background(latest_recorded_at=latest_recorded_at)
            evaluation = controller.evaluate_initiative_background(ticket)
            executed, committed = _try_run_serialized(controller.commit_initiative_background, ticket, evaluation)
            if not executed:
                _producer_mark_runtime_busy("initiative")
                return {"committed": False, "reason": "runtime_busy", "source": source}
            _producer_mark_ready("initiative")
            return dict(committed or {})
        finally:
            initiative_worker_lock.release()

    def _autonomy_loop() -> None:
        nonlocal autonomy_thread
        interval = _heartbeat_interval_seconds()
        try:
            while not autonomy_stop_event.wait(interval):
                status = controller.autonomy_runtime_status()
                if not status.get("enabled") or not status.get("running"):
                    _autonomy_worker_set_phase("idle", detail="autonomy disabled or stopped", progress=True)
                    return
                _autonomy_run_cycle(source="observer_heartbeat", block=False)
        finally:
            with autonomy_lock:
                autonomy_thread = None

    def _ensure_autonomy_runner() -> None:
        nonlocal autonomy_thread
        status = controller.autonomy_runtime_status()
        if not status.get("enabled") or not status.get("running"):
            return
        with autonomy_lock:
            if autonomy_thread is not None and autonomy_thread.is_alive():
                return
            autonomy_stop_event.clear()
            autonomy_thread = threading.Thread(
                target=_autonomy_loop,
                name="nalr-observer-autonomy",
                daemon=True,
            )
            autonomy_thread.start()

    def _stop_autonomy_runner(*, wait: bool = False) -> None:
        nonlocal autonomy_thread
        autonomy_stop_event.set()
        thread = None
        with autonomy_lock:
            if autonomy_thread is not None and autonomy_thread.is_alive():
                thread = autonomy_thread
        if wait and thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(0.5, _heartbeat_interval_seconds() + 0.5))

    def _autonomy_runner_alive() -> bool:
        return bool(autonomy_thread is not None and autonomy_thread.is_alive() and not autonomy_stop_event.is_set())

    def _monologue_loop() -> None:
        nonlocal monologue_thread
        interval = _monologue_interval_seconds()
        try:
            while not monologue_stop_event.wait(interval):
                try:
                    _monologue_run_cycle(source="observer_background", block=False)
                except RuntimeError as exc:
                    message = str(exc)
                    if "interpreter shutdown" in message or "after shutdown" in message:
                        return
                    raise
        finally:
            with monologue_lock:
                monologue_thread = None

    def _initiative_loop() -> None:
        nonlocal initiative_thread
        interval = _initiative_interval_seconds()
        try:
            while not initiative_stop_event.wait(interval):
                try:
                    _initiative_run_cycle(source="observer_background", block=False)
                except RuntimeError as exc:
                    message = str(exc)
                    if "interpreter shutdown" in message or "after shutdown" in message:
                        return
                    raise
        finally:
            with initiative_lock:
                initiative_thread = None

    def _scheduled_task_loop() -> None:
        nonlocal scheduled_task_thread
        interval = _scheduled_task_interval_seconds()
        try:
            while not scheduled_task_stop_event.wait(interval):
                try:
                    due_tasks = controller.scheduled_task_store.due_tasks(reference_at=_now_iso())
                    for task in due_tasks:
                        executed, _ = _try_run_serialized(controller.trigger_scheduled_task, task.task_id)
                        if not executed:
                            break
                except RuntimeError as exc:
                    message = str(exc)
                    if "interpreter shutdown" in message or "after shutdown" in message:
                        return
                    raise
        finally:
            with scheduled_task_lock:
                scheduled_task_thread = None

    def _ensure_monologue_runner() -> None:
        nonlocal monologue_thread
        with monologue_lock:
            if monologue_thread is not None and monologue_thread.is_alive():
                return
            monologue_stop_event.clear()
            monologue_thread = threading.Thread(
                target=_monologue_loop,
                name="nalr-observer-monologue",
                daemon=True,
            )
            monologue_thread.start()

    def _ensure_initiative_runner() -> None:
        nonlocal initiative_thread
        with initiative_lock:
            if initiative_thread is not None and initiative_thread.is_alive():
                return
            initiative_stop_event.clear()
            initiative_thread = threading.Thread(
                target=_initiative_loop,
                name="nalr-observer-initiative",
                daemon=True,
            )
            initiative_thread.start()

    def _ensure_scheduled_task_runner() -> None:
        nonlocal scheduled_task_thread
        with scheduled_task_lock:
            if scheduled_task_thread is not None and scheduled_task_thread.is_alive():
                return
            scheduled_task_stop_event.clear()
            scheduled_task_thread = threading.Thread(
                target=_scheduled_task_loop,
                name="nalr-observer-scheduled-tasks",
                daemon=True,
            )
            scheduled_task_thread.start()

    def _stop_background_runners(*, wait: bool = False) -> None:
        nonlocal monologue_thread, initiative_thread
        monologue_stop_event.set()
        initiative_stop_event.set()
        monologue_current = None
        initiative_current = None
        with monologue_lock:
            if monologue_thread is not None and monologue_thread.is_alive():
                monologue_current = monologue_thread
        with initiative_lock:
            if initiative_thread is not None and initiative_thread.is_alive():
                initiative_current = initiative_thread
        if wait and monologue_current is not None and monologue_current is not threading.current_thread():
            monologue_current.join(timeout=max(0.5, _monologue_interval_seconds() + 0.5))
        if wait and initiative_current is not None and initiative_current is not threading.current_thread():
            initiative_current.join(timeout=max(0.5, _initiative_interval_seconds() + 0.5))

    def _stop_scheduled_task_runner(*, wait: bool = False) -> None:
        nonlocal scheduled_task_thread
        scheduled_task_stop_event.set()
        thread = None
        with scheduled_task_lock:
            if scheduled_task_thread is not None and scheduled_task_thread.is_alive():
                thread = scheduled_task_thread
        if wait and thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(0.5, _scheduled_task_interval_seconds() + 0.5))

    def _autonomy_runner_stall_payload() -> dict[str, object]:
        runner_state = _autonomy_worker_snapshot()
        return {
            "runner_alive": _autonomy_runner_alive(),
            **runner_state,
        }

    def _runtime_truth_from_autonomy_status(status: dict[str, object]) -> dict[str, object]:
        return {
            "runtime_revision": int(status.get("runtime_revision", 0) or 0),
            "last_mutation_at": str(status.get("last_mutation_at") or ""),
            "run_visible": bool(status.get("run_visible", False)),
            "run_id": str(status.get("run_id") or ""),
            "run_status": str(status.get("run_status") or ""),
            "run_blocking": bool(status.get("run_blocking", False)),
            "run_block_reason": str(status.get("run_block_reason") or ""),
            "run_block_run_id": str(status.get("run_block_run_id") or ""),
            "run_pending_approval": bool(status.get("run_pending_approval", False)),
            "run_dirty_worktree": bool(status.get("run_dirty_worktree", False)),
            "run_stop_reason": str(status.get("run_stop_reason") or ""),
            "run_stale_dirty_worktree": bool(status.get("run_stale_dirty_worktree", False)),
            "fault_guard": dict(status.get("fault_guard", {}) or {}),
        }

    def _autonomy_reason_payload(
        *,
        runner_state: dict[str, object] | None = None,
        status: dict[str, object] | None = None,
        project_candidates: bool = False,
    ) -> dict[str, object]:
        current_status = dict(status or controller.autonomy_runtime_status())
        state = controller.load_runtime_state()
        controller._sync_autonomy_state(state)
        controller._ensure_autonomy_window(state)
        policy = state.autonomy_policy
        loop = state.autonomy_loop
        runner = runner_state or _autonomy_runner_stall_payload()
        blocked_commands = {str(item or "").strip() for item in list(policy.blocked_commands or [])}
        candidate_peak = str(current_status.get("last_action_type") or "").strip()
        candidate_scores: dict[str, float] = {}
        if project_candidates:
            candidate_scores = {
                name: round(float(score or 0.0), 6)
                for name, score in controller._autonomy_projected_candidate_scores(state, policy, None).items()
            }
            candidate_peak = max(candidate_scores, key=candidate_scores.get) if candidate_scores else ""

        if bool(runner.get("phase_stalled")):
            return {
                "stall_reason": "runner_blocked",
                "stall_reason_detail": "自治线程仍存活，但当前 phase 长时间没有推进。",
                "candidate_peak": candidate_peak,
            }
        if _observer_turn_active():
            return {
                "stall_reason": "observer_turn_active",
                "stall_reason_detail": "当前 observer 主会话仍有 active turn，自治心跳会暂时让路。",
                "candidate_peak": candidate_peak,
            }
        if state.safe_mode or str(loop.stop_reason or "") == "safe_mode_active":
            return {
                "stall_reason": "safe_mode_active",
                "stall_reason_detail": "系统处于 safe mode，自治已停止外放动作。",
                "candidate_peak": candidate_peak,
            }
        if float(state.budget_remaining or 0.0) <= 0.0 or str(loop.stop_reason or "") == "budget_exhausted":
            return {
                "stall_reason": "budget_exhausted",
                "stall_reason_detail": "预算已耗尽，自治不会继续推进。",
                "candidate_peak": candidate_peak,
            }
        if time.localtime().tm_hour in set(policy.quiet_hours or []):
            return {
                "stall_reason": "quiet_hours",
                "stall_reason_detail": "当前处于 quiet hours，自治只保留轻量心跳。",
                "candidate_peak": candidate_peak,
            }
        if "self_run" in blocked_commands:
            return {
                "stall_reason": "self_run_blocked",
                "stall_reason_detail": "self_run 被阻断，只读自运行路径当前不可用。",
                "candidate_peak": candidate_peak,
            }
        if not bool(current_status.get("enabled")) or not bool(current_status.get("running")):
            stop_reason = str(loop.stop_reason or "idle")
            return {
                "stall_reason": stop_reason,
                "stall_reason_detail": f"自治当前未运行，stop_reason={stop_reason}。",
                "candidate_peak": candidate_peak,
            }
        if project_candidates and (
            not candidate_peak or float(candidate_scores.get(candidate_peak, 0.0) or 0.0) <= 0.0 or candidate_peak == "nothing"
        ):
            return {
                "stall_reason": "no_candidate_action",
                "stall_reason_detail": "当前没有足够强的自治候选动作，系统处于心跳空转。",
                "candidate_peak": candidate_peak,
            }
        if not project_candidates and not candidate_peak:
            return {
                "stall_reason": "running",
                "stall_reason_detail": "自治已启动，当前走轻量状态路径。",
                "candidate_peak": candidate_peak,
            }
        return {
            "stall_reason": "running",
            "stall_reason_detail": f"自治已启动，当前最强候选动作是 {candidate_peak}。",
            "candidate_peak": candidate_peak,
        }

    def _autonomy_status_payload() -> dict:
        status = controller.autonomy_runtime_status()
        runner_state = _autonomy_runner_stall_payload()
        reason_payload = _autonomy_reason_payload(runner_state=runner_state, status=status)
        session_diagnostics = _session_selection_diagnostics()
        monologue_diagnostics = _monologue_runtime_diagnostics()
        monologue_producer = _producer_snapshot("monologue")
        initiative_producer = _producer_snapshot("initiative")
        runner_alive = bool(runner_state["runner_alive"])
        loop_should_run = bool(status.get("enabled") and status.get("running"))
        runtime_truth = _runtime_truth_from_autonomy_status(status)
        return {
            **status,
            "service": _service_status_payload(truth_payload=runtime_truth, autonomy_status=status),
            "phase": runner_state["phase"],
            "phase_detail": runner_state["phase_detail"],
            "phase_started_at": runner_state["phase_started_at"],
            "last_progress_at": runner_state["last_progress_at"],
            "phase_waiting": runner_state["phase_waiting"],
            "phase_age_seconds": runner_state["phase_age_seconds"],
            "last_progress_age_seconds": runner_state["last_progress_age_seconds"],
            "phase_stalled": runner_state["phase_stalled"],
            "running": bool(loop_should_run),
            "desired_running": loop_should_run,
            "loop_should_run": loop_should_run,
            "runner_attached": runner_alive,
            "runner_alive": runner_alive,
            "stalled": bool(runner_state["stalled"]),
            "stall_seconds": runner_state["stall_seconds"],
            "stall_threshold_seconds": runner_state["stall_threshold_seconds"],
            "heartbeat_state": "stalled" if runner_state["stalled"] else ("running" if runner_alive and loop_should_run else "idle"),
            "runner_source": "observer_heartbeat",
            "observer_turn_active": _observer_turn_active(),
            **session_diagnostics,
            "monologue_overdue_seconds": monologue_diagnostics["monologue_overdue_seconds"],
            "initiative_runtime_busy": bool(initiative_producer.get("runtime_busy", False)),
            "monologue_runtime_busy": bool(monologue_producer.get("runtime_busy", False)),
            "initiative_last_busy_at": initiative_producer.get("last_busy_at"),
            "monologue_last_busy_at": monologue_producer.get("last_busy_at"),
            "initiative_busy_skip_count": initiative_producer.get("busy_skip_count"),
            "monologue_busy_skip_count": monologue_producer.get("busy_skip_count"),
            **reason_payload,
        }

    def _service_status_payload(
        *,
        truth_payload: dict[str, object] | None = None,
        autonomy_status: dict[str, object] | None = None,
    ) -> dict:
        now = _now_iso()
        service_probe_state["last_http_ok_at"] = now
        service_probe_state["last_probe_error"] = ""
        current_autonomy_status = dict(autonomy_status or controller.autonomy_runtime_status())
        runtime_truth = dict(truth_payload or _runtime_truth_from_autonomy_status(current_autonomy_status))
        runner_state = _autonomy_runner_stall_payload()
        reason_payload = _autonomy_reason_payload(runner_state=runner_state, status=current_autonomy_status)
        session_diagnostics = _session_selection_diagnostics()
        monologue_diagnostics = _monologue_runtime_diagnostics()
        monologue_producer = _producer_snapshot("monologue")
        initiative_producer = _producer_snapshot("initiative")
        return {
            **runtime_truth,
            "instance_id": service_instance_id,
            "pid": service_pid,
            "host": service_host,
            "port": service_port,
            "url": service_url,
            "started_at": service_started_at,
            "healthy": True,
            "accepting_http": True,
            "http_ready": True,
            "event_loop_alive": True,
            "last_http_ok_at": service_probe_state["last_http_ok_at"],
            "last_probe_error": service_probe_state["last_probe_error"],
            "last_runtime_activity_at": service_probe_state["last_runtime_activity_at"],
            "phase": runner_state["phase"],
            "phase_detail": runner_state["phase_detail"],
            "phase_started_at": runner_state["phase_started_at"],
            "last_progress_at": runner_state["last_progress_at"],
            "phase_waiting": runner_state["phase_waiting"],
            "phase_age_seconds": runner_state["phase_age_seconds"],
            "last_progress_age_seconds": runner_state["last_progress_age_seconds"],
            "phase_stalled": runner_state["phase_stalled"],
            "stalled": bool(runner_state["stalled"]),
            "runtime_serial_active": _runtime_serial_active(),
            "observer_turn_active": _observer_turn_active(),
            "active_turn_sessions": _active_turn_sessions(),
            "autonomy_runner_alive": bool(runner_state["runner_alive"]),
            "autonomy_runner_stalled": bool(runner_state["stalled"]),
            "autonomy_runner_stall_seconds": runner_state["stall_seconds"],
            "autonomy_runner_stall_threshold_seconds": runner_state["stall_threshold_seconds"],
            "stall_reason": reason_payload["stall_reason"],
            "stall_reason_detail": reason_payload["stall_reason_detail"],
            "candidate_peak": reason_payload["candidate_peak"],
            **session_diagnostics,
            "monologue_overdue_seconds": monologue_diagnostics["monologue_overdue_seconds"],
            "initiative_runtime_busy": bool(initiative_producer.get("runtime_busy", False)),
            "monologue_runtime_busy": bool(monologue_producer.get("runtime_busy", False)),
            "initiative_last_busy_at": initiative_producer.get("last_busy_at"),
            "monologue_last_busy_at": monologue_producer.get("last_busy_at"),
            "initiative_busy_skip_count": initiative_producer.get("busy_skip_count"),
            "monologue_busy_skip_count": monologue_producer.get("busy_skip_count"),
            "project_root": str(project_root_path),
            "config_root": str(effective_config_root),
        }

    def _ensure_background_runners() -> None:
        if background_runners_disabled:
            return
        _ensure_monologue_runner()
        _ensure_initiative_runner()

    def _light_mode_label(mode: str, cause_type: str) -> str:
        normalized_mode = str(mode or "").strip()
        normalized_cause = str(cause_type or "").strip()
        if normalized_cause == "endogenous":
            return {
                "endogenous_light": "内生整理",
                "endogenous_regulation": "内在调节",
                "endogenous_replay": "内部回放",
            }.get(normalized_mode, "内部处理")
        return {
            "interactive": "对外互动",
            "idle": "待机",
            "sleep": "睡眠",
            "safe": "安全模式",
        }.get(normalized_mode, normalized_mode or "未知模式")

    def _light_cause_label(cause_type: str, mode: str) -> str:
        normalized_cause = str(cause_type or "").strip()
        if normalized_cause == "endogenous":
            return _light_mode_label(mode, normalized_cause)
        return {
            "external_stimulus": "外界刺激",
            "dream": "梦境加工",
        }.get(normalized_cause, normalized_cause or "外部触发")

    def _light_continuity_label(display_name: str, name_source: str) -> str:
        has_display_name = bool(display_name.strip())
        normalized_source = str(name_source or "").strip()
        if normalized_source == "generated":
            return "当前名称正根据内部证据逐步形成" if has_display_name else "还在形成稳定称呼"
        if normalized_source == "user_seed":
            return "当前名称沿用用户给出的称呼"
        if normalized_source == "manual_override":
            return "当前名称采用人工指定称呼"
        return "名称与身份连续性稳定" if has_display_name else "还在形成稳定称呼"

    def _light_focus_summary(focus: str, mode: str, cause_type: str, current_goal: str) -> str:
        normalized_mode = str(mode or "").strip()
        normalized_focus = str(focus or "").strip()
        goal = str(current_goal or "").strip()
        if goal:
            return f"围绕目标“{goal}”维持当前焦点"
        if cause_type == "endogenous":
            return {
                "endogenous_light": "正在进行内部整理，先在心里消化线索",
                "endogenous_regulation": "正在进行内在调节，让状态先回稳",
                "endogenous_replay": "正在进行内部回放，先把刚才的痕迹过一遍",
            }.get(normalized_mode, "正在进行内部整理，先在内部自我对话")
        return {
            "task": "专心处理眼前的事",
            "respond": "把注意力放在回应上",
            "wander": "思绪有些发散",
            "rest": "慢慢回落和恢复",
            "absorb": "把注意力转向内部吸收",
            "self_run": "把注意力转向自我检查和只读行动",
            "monologue": "把注意力转向内在独白",
            "nothing": "暂时收住外显表达",
        }.get(normalized_focus, "继续根据当前状态调节表达")

    def _load_light_runtime_state() -> dict[str, Any]:
        try:
            raw_text = controller.state_path.read_text(encoding="utf-8")
            payload = json.loads(raw_text)
            if isinstance(payload, dict):
                return payload
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            pass
        cached_state = getattr(controller, "_state_cache", None)
        if cached_state is not None:
            cached_payload = to_dict(cached_state)
            if isinstance(cached_payload, dict):
                return cached_payload
        return {}

    def _runtime_models_light() -> dict[str, Any]:
        return {
            "module_model_bindings": controller._module_model_bindings(),
            "route_policies": controller._route_policy_contract(),
            "failover": controller.model_router.failover_status(),
        }

    def _runtime_summary_payload() -> dict[str, Any]:
        raw = _load_light_runtime_state()
        recent_actions = controller.console_recent_actions(limit=12).get("actions", [])
        recent_rounds = [
            {
                "round_id": int(item.get("round_id", 0) or 0),
                "sampled_action": str(item.get("action") or "nothing"),
                "cause_type": str(item.get("cause_type") or "external_stimulus"),
                "mode": str(item.get("mode") or "interactive"),
                "trace_ref": str(item.get("trace_ref") or ""),
                "recorded_at": item.get("recorded_at"),
            }
            for item in recent_actions
            if int(item.get("round_id", 0) or 0) > 0
        ]
        current_round = recent_rounds[0] if recent_rounds else None
        current_mode = str((current_round or {}).get("mode") or raw.get("mode") or "")
        current_cause = str((current_round or {}).get("cause_type") or "external_stimulus")
        identity_state = dict(raw.get("identity_state", {}) or {})
        display_name = str(identity_state.get("display_name") or identity_state.get("internal_handle") or "当前运行体").strip()
        continuity = _light_continuity_label(display_name, str(identity_state.get("name_source") or ""))
        body_state = dict(raw.get("body_state", {}) or {})
        subjective_state = dict(raw.get("subjective_state", {}) or {})
        organic_mode = dict(raw.get("organic_mode", {}) or {})
        runtime_state = {
            "runtime_revision": int(raw.get("runtime_revision", 0) or 0),
            "last_mutation_at": str(raw.get("last_mutation_at") or ""),
            "mode": str(raw.get("mode") or ""),
            "focus": str(raw.get("focus") or ""),
            "safe_mode": bool(raw.get("safe_mode", False)),
            "session_id": str(raw.get("session_id") or ""),
            "autonomy_policy": dict(raw.get("autonomy_policy", {}) or {}),
            "body_energy": float(raw.get("body_energy", body_state.get("energy", 0.0)) or 0.0),
            "affect_residue": float(raw.get("affect_residue", 0.0) or 0.0),
            "mood": float(raw.get("mood", 0.0) or 0.0),
            "fatigue": float(raw.get("fatigue", body_state.get("fatigue", 0.0)) or 0.0),
            "self_continuity": float(raw.get("self_continuity", body_state.get("self_continuity", 0.0)) or 0.0),
            "meaning_strength": float(raw.get("meaning_strength", body_state.get("meaning_strength", 0.0)) or 0.0),
            "body_state": body_state,
            "subjective_state": subjective_state,
            "organic_mode": organic_mode,
            "run": {
                "goal_summary": str(raw.get("current_goal") or ""),
                "current_step": {"title": str(raw.get("current_step_id") or "")},
            },
            "state": {
                "brain_state": {
                    "mode": current_mode or str(raw.get("mode") or ""),
                    "self_continuity": continuity,
                }
            },
            "cognitive_snapshot": {
                "current_intent": _light_focus_summary(
                    str(raw.get("focus") or ""),
                    current_mode,
                    current_cause,
                    str(raw.get("current_goal") or ""),
                ),
                "vital_signs": {
                    "mood": float(raw.get("mood", 0.0) or 0.0),
                    "body_energy": float(raw.get("body_energy", body_state.get("energy", 0.0)) or 0.0),
                    "affect_residue": float(raw.get("affect_residue", 0.0) or 0.0),
                    "focus": str(raw.get("focus") or ""),
                    "mode": current_mode or str(raw.get("mode") or ""),
                },
                "identity": {
                    "display_name": display_name,
                    "continuity": continuity,
                },
                "authenticity": {
                    "summary": "还没有足够证据判断这轮真实感",
                    "source": "light_runtime_summary",
                    "guard_action": "none",
                    "sampling_penalty_applied": 0.0,
                },
                "tlh": {
                    "body_state": body_state,
                    "subjective_state": subjective_state,
                    "organic_mode": organic_mode,
                },
            },
        }
        console = {
            "state": {
                "current_round": {
                    "round_id": int((current_round or {}).get("round_id", 0) or 0) or None,
                    "sampled_action": str((current_round or {}).get("sampled_action") or ""),
                    "trace_ref": str((current_round or {}).get("trace_ref") or ""),
                    "cause_type": current_cause,
                    "mode": current_mode,
                    "mode_label": _light_mode_label(current_mode, current_cause),
                }
            },
            "recent_rounds": recent_rounds,
        }
        return {
            "runtime_state": runtime_state,
            "models": _runtime_models_light(),
            "console": console,
        }

    def _recent_actions_payload_from_summary(summary_payload: dict[str, Any] | None = None) -> dict[str, Any]:
        console_payload = dict((summary_payload or {}).get("console", {}) or {})
        recent_rounds = list(console_payload.get("recent_rounds", []) or [])
        actions = [
            {
                "round_id": int(item.get("round_id", 0) or 0),
                "trace_ref": str(item.get("trace_ref") or f"round://{int(item.get('round_id', 0) or 0)}"),
                "action": str(item.get("sampled_action") or "nothing"),
                "summary": _light_cause_label(str(item.get("cause_type") or "external_stimulus"), str(item.get("mode") or "interactive")),
                "recorded_at": item.get("recorded_at"),
                "cause_type": str(item.get("cause_type") or "external_stimulus"),
                "mode": str(item.get("mode") or "interactive"),
            }
            for item in recent_rounds
            if int(item.get("round_id", 0) or 0) > 0
        ]
        return {
            "actions": actions,
            "message": "暂无最近动作" if not actions else "",
        }

    def _web_runtime_bootstrap_payload(
        *,
        summary_payload: dict[str, Any] | None = None,
        recent_actions_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            session_payload = _light_session_state_payload(OBSERVER_MAIN_SESSION_ID)
            session_attached = True
        except FileNotFoundError:
            session_payload = None
            session_attached = False
        summary = summary_payload or _runtime_summary_payload()
        recent_actions = recent_actions_payload or _recent_actions_payload_from_summary(summary)
        autonomy_payload = _autonomy_status_payload()
        return _json_safe(
            {
                "service": dict(autonomy_payload.get("service", {}) or {}),
                "autonomy": autonomy_payload,
                "session": session_payload,
                "session_attached": session_attached,
                "recent_actions": recent_actions.get("actions", []),
                "recent_actions_message": recent_actions.get("message", ""),
            }
        )

    def _workbench_read_model_payload(
        *,
        analysis: bool = False,
        inner_space: bool = False,
        settings: bool = False,
        observer_session_id: str = "",
        chat_session_id: str = "",
    ) -> dict[str, Any]:
        summary = _runtime_summary_payload()
        recent_actions_payload = _recent_actions_payload_from_summary(summary)
        bootstrap = _web_runtime_bootstrap_payload(
            summary_payload=summary,
            recent_actions_payload=recent_actions_payload,
        )
        console_payload = controller.console_refresh_payload() if analysis else dict(summary.get("console", {}) or {})
        settings_payload = controller.observer_settings_payload() if settings else None
        agency_payload = controller.agency_status()
        memory_top = controller.memory_top(limit=6) if inner_space else []
        dream_overview = controller.dream_overview(limit=6) if inner_space else None
        monologue_show = controller.monologue_show(limit=12) if inner_space else None
        effective_observer_session_id = str(observer_session_id or "").strip()
        if not effective_observer_session_id:
            effective_observer_session_id = str((((bootstrap.get("session") or {}).get("session") or {}).get("session_id")) or "").strip()
        session_state = None
        if effective_observer_session_id:
            try:
                session_state = _light_session_state_payload(effective_observer_session_id)
            except FileNotFoundError:
                session_state = None
        chat_session_state = None
        if chat_session_id:
            try:
                chat_session_state = _light_session_state_payload(chat_session_id)
            except FileNotFoundError:
                chat_session_state = None
        subject_payload = controller.subject_status()
        meaning_payload = controller.meaning_status()
        performance_payload = controller.runtime_performance_payload() if settings else None
        state = controller.load_runtime_state()
        controller.sync_plan16_state(state)
        latest_trace = controller.trace_round("last") if int(state.round_count or 0) > 0 else None
        return _json_safe(
            {
                "bootstrap": bootstrap,
                "summary": summary,
                "console": console_payload,
                "settings": settings_payload,
                "initiativeStatus": agency_payload.get("initiative"),
                "memoryTop": memory_top,
                "dreamOverview": dream_overview,
                "monologueShow": monologue_show,
                "sessionState": session_state,
                "chatSessionState": chat_session_state,
                "subject": subject_payload,
                "meaning": meaning_payload,
                "agency": agency_payload,
                "performance": performance_payload,
                "state_truth": controller.state_truth_payload(state, trace=latest_trace),
                "memory_evidence": controller.memory_evidence_payload(latest_trace),
                "competition_evidence": controller.competition_evidence_payload(latest_trace),
                "continuity_evidence": controller.continuity_evidence_payload(latest_trace, state=state),
                "inner_space": {
                    "memory_top": memory_top,
                    "dream_overview": dream_overview,
                    "monologue_show": monologue_show,
                },
            }
        )

    def _workbench_round_payload(round_id: int, *, action: str = "respond") -> dict[str, Any]:
        return _json_safe(controller.workbench_round_payload(round_id, action=action))

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            if not background_runners_disabled:
                _ensure_scheduled_task_runner()
            yield
        finally:
            _stop_scheduled_task_runner(wait=True)
            _stop_autonomy_runner(wait=True)
            _stop_background_runners(wait=True)

    app = FastAPI(title="NALR Observer", version="0.6.0", lifespan=lifespan)
    app.state.controller = controller

    def _empty_decision_payload() -> dict:
        return {"message": "无决策记录", "round_id": None}

    def _session_state_payload(session_id: str) -> dict:
        session = terminal_sessions.read(session_id)
        snapshot = terminal_handler.snapshot_session(session_id)
        return _json_safe(_observer_session_contract_payload(session=session, snapshot=snapshot, light=False))

    def _light_session_state_payload(session_id: str) -> dict:
        session = terminal_sessions.read(session_id)
        pending = [item for item in list(session.approvals_pending or []) if item.get("status") == "pending"]
        run_id = str(session.active_run_id or session.last_run_id or "")
        run_status_payload = controller.run_status(run_id) if run_id else None
        snapshot = {
            "goal_summary": str((run_status_payload or {}).get("goal_summary") or (run_status_payload or {}).get("goal") or ""),
            "current_step": str((((run_status_payload or {}).get("current_step") or {}).get("title")) or ""),
            "reason_summary": str(
                (((run_status_payload or {}).get("current_step") or {}).get("expected_observation"))
                or (((run_status_payload or {}).get("current_step") or {}).get("detail"))
                or ""
            ),
            "last_tool": str((((run_status_payload or {}).get("last_tool_result") or {}).get("tool_name")) or ""),
            "run_status": str((run_status_payload or {}).get("status") or session.status or "idle"),
            "permission_mode": session.permission_mode,
            "pending_approval_count": len(pending),
            "status": run_status_payload
            or {
                "run_id": run_id,
                "status": session.status,
            },
            "why": None,
            "steps": [],
            "tools": [],
            "approvals": {
                "pending": pending,
                "pending_count": len(pending),
            },
            "ui_actions": {
                "primary": [],
                "secondary": [],
            },
            "statusline": {
                "cwd": session.cwd,
                "permission_mode": session.permission_mode,
                "run_status": str((run_status_payload or {}).get("status") or session.status or "idle"),
                "session_id": session.session_id,
                "run_id": run_id,
            },
            "cognitive_snapshot": None,
            "console": None,
        }
        return _json_safe(_observer_session_contract_payload(session=session, snapshot=snapshot, light=True))

    def _observer_session_contract_payload(
        *,
        session: TerminalSessionState,
        snapshot: dict[str, Any],
        light: bool,
    ) -> dict[str, Any]:
        runtime_state = controller.load_runtime_state()
        runtime_truth = controller.runtime_status_truth_payload(runtime_state)
        controlled_learning = terminal_handler._controlled_learning_snapshot(runtime_state)
        return {
            "goal_summary": str(snapshot.get("goal_summary") or ""),
            "current_step": str(snapshot.get("current_step") or ""),
            "reason_summary": str(snapshot.get("reason_summary") or ""),
            "last_tool": str(snapshot.get("last_tool") or ""),
            "run_status": str(snapshot.get("run_status") or session.status or "idle"),
            "permission_mode": str(snapshot.get("permission_mode") or session.permission_mode or "acceptEdits"),
            "pending_approval_count": int(snapshot.get("pending_approval_count") or 0),
            "runtime_truth": runtime_truth,
            "controlled_learning": controlled_learning,
            "session": session.__dict__,
            "status": snapshot.get("status"),
            "why": snapshot.get("why"),
            "steps": snapshot.get("steps", []),
            "tools": snapshot.get("tools", []),
            "approvals": snapshot.get("approvals", {"pending": [], "pending_count": 0}),
            "ui_actions": snapshot.get("ui_actions", {"primary": [], "secondary": []}),
            "statusline": snapshot.get("statusline"),
            "cognitive_snapshot": snapshot.get("cognitive_snapshot"),
            "console": snapshot.get("console"),
            "workbench": {"cards": []} if light else build_workbench_cards(snapshot),
        }

    def _ensure_runtime_session(session_id: str, cwd: str) -> TerminalSessionState:
        try:
            session = terminal_sessions.read(session_id)
            session.cwd = cwd
            session.status = "active"
        except FileNotFoundError:
            session = TerminalSessionState(session_id=session_id, cwd=cwd, status="active")
        terminal_sessions.write(session, mark_current=True)
        return session

    def _runtime_payload(session_id: str | None = None, *, light_session: bool = False) -> dict:
        session_payload = None
        if session_id:
            try:
                session_payload = _light_session_state_payload(session_id) if light_session else _session_state_payload(session_id)
            except FileNotFoundError:
                session_payload = None
        recent_actions = controller.console_recent_actions()
        return _json_safe(
            {
                "session": session_payload,
                "state": controller.state_payload(),
                "service": _service_status_payload(),
                "autonomy": _autonomy_status_payload(),
                "recent_actions": recent_actions.get("actions", []),
                "recent_actions_message": recent_actions.get("message", ""),
                "probability_space": controller.console_probability_space(),
            }
        )

    def _busy_runtime_control_payload(reason: str, *, desired_running: bool | None = None) -> dict:
        autonomy = _autonomy_status_payload()
        if desired_running is not None:
            autonomy["desired_running"] = bool(desired_running)
        payload = {
            "service": _service_status_payload(),
            "autonomy": autonomy,
            "session": None,
            "recent_actions": [],
            "recent_actions_message": "",
            "skipped": True,
            "reason": reason,
        }
        try:
            payload["session"] = _light_session_state_payload(OBSERVER_MAIN_SESSION_ID)
        except FileNotFoundError:
            payload["session"] = None
        return _json_safe(payload)

    def _skipped_console_refresh_payload(reason: str) -> dict:
        summary = "前台对话进行中，本次内源刷新已让路。"
        return {
            "state": {"current_round": {"round_id": None}},
            "action_field": {},
            "timeline": {"events": []},
            "why_current": {"why": {"summary": summary}},
            "why_not": {"action": "", "why_not": {"summary": summary}},
            "skipped": True,
            "reason": reason,
        }

    def _json_safe(value):
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        if isinstance(value, dict):
            return {key: _json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [_json_safe(item) for item in value]
        if isinstance(value, tuple):
            return [_json_safe(item) for item in value]
        return value

    @app.get("/state")
    async def state() -> dict:
        payload = controller.state_payload()
        payload["service"] = _service_status_payload()
        return payload

    @app.get("/service/status")
    async def service_status() -> dict:
        return _service_status_payload()

    @app.get("/console/state")
    async def console_state() -> dict:
        return await _run_readonly_async(controller.console_state)

    @app.get("/console/action-field")
    async def console_action_field(round_id: int | None = None) -> dict:
        try:
            return controller.console_action_field(round_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/console/timeline")
    async def console_timeline(round_id: int | None = None) -> dict:
        try:
            return controller.console_timeline(round_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/console/why/current")
    async def console_why_current(round_id: int | None = None) -> dict:
        try:
            return controller.console_why_current(round_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/console/why-not/{action}")
    async def console_why_not(action: str, round_id: int | None = None) -> dict:
        try:
            return controller.console_why_not(action, round_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/console/refresh")
    async def console_refresh(round_id: int | None = None) -> dict:
        return await _run_readonly_async(controller.console_refresh_payload, round_id)

    @app.get("/console/recent-actions")
    async def console_recent_actions(limit: int = 8) -> dict:
        return await _run_readonly_async(controller.console_recent_actions, limit=limit)

    @app.get("/console/probability-space")
    async def console_probability_space(round_ref: str | None = None) -> dict:
        try:
            return await _run_readonly_async(controller.console_probability_space, round_ref)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/console/talk")
    async def console_talk(payload: dict = Body(...)) -> dict:
        session_id = str(payload.get("session_id") or "").strip()
        text = str(payload.get("text") or "").strip()
        cwd = str(payload.get("cwd") or project_root_path).strip()
        if not session_id:
            raise HTTPException(status_code=400, detail="session_id is required")
        if not text:
            raise HTTPException(status_code=400, detail="text is required")
        def _console_talk_payload() -> dict:
            _ensure_background_runners()
            try:
                terminal_sessions.read(session_id)
            except FileNotFoundError:
                terminal_handler.handle({"type": "start_session", "session_id": session_id, "cwd": cwd, "persist_current": False})
            turn_events = terminal_handler.handle({"type": "user_turn", "session_id": session_id, "text": text})
            assistant = next((event.get("message") for event in reversed(turn_events) if event.get("type") == "assistant_final"), "")
            round_id = next(
                (
                    int(event.get("round_id", 0) or 0)
                    for event in reversed(turn_events)
                    if int(event.get("round_id", 0) or 0) > 0
                ),
                None,
            )
            snapshot_console = next(
                (
                    event.get("console")
                    for event in reversed(turn_events)
                    if event.get("type") == "sidebar_snapshot"
                ),
                None,
            )
            try:
                session = terminal_sessions.read(session_id).__dict__
            except FileNotFoundError:
                session = {"session_id": session_id, "cwd": cwd}
            return {
                "assistant": assistant,
                "session": session,
                "console": controller.console_refresh_payload(round_id) if round_id else snapshot_console,
            }

        return await _run_serialized_async(_console_talk_payload)

    @app.get("/web/sessions")
    async def web_sessions() -> dict:
        sessions = [session.__dict__ for session in terminal_sessions.list_sessions()]
        return _json_safe({"sessions": sessions})

    @app.post("/web/session/start")
    async def web_session_start(payload: dict = Body(default={})):  # type: ignore[valid-type]
        session_id = str(payload.get("session_id") or f"web-{uuid4().hex[:12]}").strip()
        cwd = str(payload.get("cwd") or project_root_path).strip()
        if not cwd:
            raise HTTPException(status_code=400, detail="cwd is required")
        def _web_session_start_payload() -> dict:
            _ensure_background_runners()
            try:
                events = terminal_handler.handle(
                    {
                        "type": "start_session",
                        "session_id": session_id,
                        "cwd": cwd,
                        "persist_current": bool(payload.get("persist_current", True)),
                    }
                )
            except ProtocolError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            recorded = _append_broker_events(session_id, events)
            return _json_safe(
                {
                    "session_id": session_id,
                    "session": terminal_sessions.read(session_id).__dict__,
                    "event_count": len(recorded),
                    "events": recorded,
                }
            )

        return await _run_serialized_async(_web_session_start_payload)

    @app.post("/web/runtime/start")
    async def web_runtime_start(payload: dict = Body(default={})):  # type: ignore[valid-type]
        def _web_runtime_start_payload() -> dict:
            session_id = OBSERVER_MAIN_SESSION_ID
            cwd = str(payload.get("cwd") or project_root_path).strip()
            session = _ensure_runtime_session(session_id, cwd)
            if session.permission_mode != "acceptEdits":
                session.permission_mode = "acceptEdits"
                terminal_sessions.write(session, mark_current=True)
            _append_broker_events(
                session_id,
                [
                    build_outbound_event("session_started", session=terminal_sessions.read(session_id).__dict__),
                    build_outbound_event("assistant_final", session_id=session_id, message="permission mode set to acceptEdits"),
                ],
            )
            controller.start_autonomy(
                profile=str(payload.get("profile") or "tool_level"),
                clear_safe_mode=True,
            )
            _ensure_background_runners()
            _ensure_autonomy_runner()
            return _web_runtime_bootstrap_payload()

        return await _run_serialized_async(_web_runtime_start_payload)

    @app.get("/web/runtime/bootstrap")
    async def web_runtime_bootstrap() -> dict:
        return await _run_readonly_async(_web_runtime_bootstrap_payload)

    @app.get("/web/runtime/summary")
    async def web_runtime_summary() -> dict:
        return await _run_readonly_async(_runtime_summary_payload)

    @app.get("/workbench/read-model")
    async def workbench_read_model(
        analysis: bool = False,
        inner_space: bool = False,
        settings: bool = False,
        observer_session_id: str = "",
        chat_session_id: str = "",
    ) -> dict:
        return await _run_readonly_async(
            _workbench_read_model_payload,
            analysis=analysis,
            inner_space=inner_space,
            settings=settings,
            observer_session_id=observer_session_id,
            chat_session_id=chat_session_id,
        )

    @app.get("/workbench/round/{round_id}")
    async def workbench_round(round_id: int, action: str = "respond") -> dict:
        try:
            return await _run_readonly_async(_workbench_round_payload, round_id, action=action)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/performance/hot-path")
    async def performance_hot_path() -> dict:
        return await _run_readonly_async(controller.runtime_performance_payload)

    @app.get("/scheduled-tasks")
    async def scheduled_tasks() -> dict:
        return await _run_readonly_async(controller.list_scheduled_tasks)

    @app.get("/scheduled-tasks/runs")
    async def scheduled_task_runs(task_id: str | None = None) -> dict:
        return await _run_readonly_async(controller.list_scheduled_task_runs, task_id)

    @app.post("/scheduled-tasks")
    async def upsert_scheduled_task(payload: dict = Body(default={})):  # type: ignore[valid-type]
        try:
            return await _run_serialized_async(controller.upsert_scheduled_task, payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/scheduled-tasks/{task_id}/trigger")
    async def trigger_scheduled_task(task_id: str) -> dict:
        try:
            return await _run_serialized_async(controller.trigger_scheduled_task, task_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/web/runtime/pause")
    async def web_runtime_pause(payload: dict = Body(default={})):  # type: ignore[valid-type]
        def _web_runtime_pause_payload() -> dict:
            reason = str(payload.get("reason") or "dashboard_pause")
            controller.stop_autonomy(reason=reason)
            _stop_autonomy_runner(wait=True)
            runtime_payload = _runtime_payload(OBSERVER_MAIN_SESSION_ID, light_session=False)
            runtime_payload["autonomy"] = _autonomy_status_payload()
            return runtime_payload

        if _autonomy_worker_busy():
            return _busy_runtime_control_payload("autonomy_runner_busy", desired_running=False)
        executed, result = await _try_run_serialized_async(_web_runtime_pause_payload)
        if executed:
            return result
        reason = "autonomy_runner_busy" if _autonomy_worker_busy() else ("autonomy_runner_busy" if _autonomy_runner_alive() else "runtime_busy")
        return _busy_runtime_control_payload(reason, desired_running=False)

    @app.post("/web/runtime/resume")
    async def web_runtime_resume(_: dict = Body(default={})):  # type: ignore[valid-type]
        def _web_runtime_resume_payload() -> dict:
            try:
                controller.resume_run()
            except FileNotFoundError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            _ensure_background_runners()
            runtime_payload = _runtime_payload(OBSERVER_MAIN_SESSION_ID, light_session=False)
            runtime_payload["autonomy"] = _autonomy_status_payload()
            return runtime_payload

        return await _run_serialized_async(_web_runtime_resume_payload)

    @app.post("/web/runtime/wake")
    async def web_runtime_wake(_: dict = Body(default={})):  # type: ignore[valid-type]
        def _web_runtime_wake_payload() -> dict:
            controller.apply_command("mode set interactive")
            _ensure_background_runners()
            runtime_payload = _runtime_payload(OBSERVER_MAIN_SESSION_ID, light_session=False)
            runtime_payload["autonomy"] = _autonomy_status_payload()
            return runtime_payload

        return await _run_serialized_async(_web_runtime_wake_payload)

    @app.post("/web/session/event")
    async def web_session_event(payload: dict = Body(...)) -> dict:
        session_id = str(payload.get("session_id") or "").strip()
        if not session_id:
            raise HTTPException(status_code=400, detail="session_id is required")
        event_type = str(payload.get("type") or "").strip()

        if event_type == "user_turn":
            pending = web_broker.read_since(session_id, after_id=0)
            baseline_event_id = int(pending[-1]["event_id"]) if pending else 0
            _enqueue_session_turn(session_id, payload)
            return {
                "accepted": True,
                "session_id": session_id,
                "event_count": 0,
                "last_event_id": baseline_event_id,
                "queued": True,
            }

        def _web_session_event_payload() -> dict:
            try:
                events = terminal_handler.handle(payload)
            except ProtocolError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except FileNotFoundError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            recorded = _append_broker_events(session_id, events)
            return {
                "accepted": True,
                "session_id": session_id,
                "event_count": len(recorded),
                "last_event_id": recorded[-1]["event_id"] if recorded else None,
            }

        return await _run_serialized_async(_web_session_event_payload)

    @app.get("/web/session/events")
    async def web_session_events(session_id: str, after_id: int = 0, once: bool = False) -> StreamingResponse:
        if not session_id.strip():
            raise HTTPException(status_code=400, detail="session_id is required")
        timeout_seconds = 30.0 if once else 15.0
        return StreamingResponse(
            web_broker.stream_sse(session_id, after_id=after_id, once=once, timeout_seconds=timeout_seconds),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/web/session/state")
    async def web_session_state(session_id: str, full: bool = False) -> dict:
        try:
            payload_fn = _session_state_payload if full else _light_session_state_payload
            return await _run_readonly_async(payload_fn, session_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc


    @app.post("/console/endogenous/tick")
    async def console_endogenous_tick(payload: dict = Body(default={})):  # type: ignore[valid-type]
        def _console_endogenous_tick_payload() -> dict:
            trigger = str(payload.get("trigger") or "idle")
            mode = payload.get("mode")
            if _observer_turn_active():
                return {
                    "tick": {
                        "round_id": None,
                        "cause_type": trigger,
                        "mode": mode,
                        "skipped": True,
                        "reason": "interactive_turn_active",
                    },
                    "console": _skipped_console_refresh_payload("interactive_turn_active"),
                }
            tick_payload = run_endogenous_tick_payload(controller, trigger=trigger, mode=mode)
            _sync_session_transcript_events(OBSERVER_MAIN_SESSION_ID)
            round_id = tick_payload.get("round_id")
            return {
                "tick": tick_payload,
                "console": controller.console_refresh_payload(round_id),
            }

        if _autonomy_runner_engaged():
            reason = "autonomy_runner_busy"
            return {
                "tick": {
                    "round_id": None,
                    "cause_type": str(payload.get("trigger") or "idle"),
                    "mode": payload.get("mode"),
                    "skipped": True,
                    "reason": reason,
                },
                "console": _skipped_console_refresh_payload(reason),
            }
        if _observer_turn_active():
            return {
                "tick": {
                    "round_id": None,
                    "cause_type": str(payload.get("trigger") or "idle"),
                    "mode": payload.get("mode"),
                    "skipped": True,
                    "reason": "interactive_turn_active",
                },
                "console": _skipped_console_refresh_payload("interactive_turn_active"),
            }
        executed, result = await _try_run_serialized_async(_console_endogenous_tick_payload)
        if executed:
            return result
        reason = "autonomy_runner_busy" if _autonomy_worker_busy() else ("autonomy_runner_busy" if _autonomy_runner_alive() else "runtime_busy")
        return {
            "tick": {
                "round_id": None,
                "cause_type": str(payload.get("trigger") or "idle"),
                "mode": payload.get("mode"),
                "skipped": True,
                "reason": reason,
            },
            "console": _skipped_console_refresh_payload(reason),
        }

    @app.get("/autonomy/status")
    async def autonomy_status() -> dict:
        def _autonomy_status_response() -> dict:
            _ensure_autonomy_runner()
            return _autonomy_status_payload()

        return await _run_readonly_async(_autonomy_status_response)

    @app.post("/autonomy/start")
    async def autonomy_start(payload: dict = Body(default={})):  # type: ignore[valid-type]
        def _autonomy_start_payload() -> dict:
            profile = str(payload.get("profile") or "tool_level")
            controller.start_autonomy(profile=profile)
            _ensure_autonomy_runner()
            return _autonomy_status_payload()

        return await _run_serialized_async(_autonomy_start_payload)

    @app.post("/autonomy/stop")
    async def autonomy_stop(payload: dict = Body(default={})):  # type: ignore[valid-type]
        def _autonomy_stop_payload() -> dict:
            reason = str(payload.get("reason") or "manual_stop")
            controller.stop_autonomy(reason=reason)
            _stop_autonomy_runner(wait=True)
            return _autonomy_status_payload()

        if _autonomy_worker_busy():
            reason = "autonomy_runner_busy"
            payload = _autonomy_status_payload()
            payload["skipped"] = True
            payload["reason"] = reason
            payload["desired_running"] = False
            return payload
        executed, result = await _try_run_serialized_async(_autonomy_stop_payload)
        if executed:
            return result
        reason = "autonomy_runner_busy" if _autonomy_worker_busy() else ("autonomy_runner_busy" if _autonomy_runner_alive() else "runtime_busy")
        payload = _autonomy_status_payload()
        payload["skipped"] = True
        payload["reason"] = reason
        payload["desired_running"] = False
        return payload

    @app.post("/autonomy/step")
    async def autonomy_step() -> dict:
        if _observer_turn_active():
            payload = _autonomy_status_payload()
            payload["skipped"] = True
            payload["reason"] = "interactive_turn_active"
            return payload

        return await asyncio.to_thread(_autonomy_run_cycle, source="manual_step", block=True)

    @app.post("/persona/reset")
    async def persona_reset(payload: dict = Body(default={})):  # type: ignore[valid-type]
        def _persona_reset_payload() -> dict:
            _stop_autonomy_runner(wait=True)
            _stop_background_runners(wait=True)
            controller.stop_autonomy(reason=str(payload.get("reason") or "persona_reset"))
            controller.reset_persona()
            _ensure_monologue_runner()
            _ensure_initiative_runner()
            _ensure_autonomy_runner()
            terminal_sessions.clear_all()
            web_broker.clear()
            transcript_sync_offsets.clear()
            return _json_safe(
                {
                    "state": controller.state_payload(),
                    "autonomy": _autonomy_status_payload(),
                    "recent_actions": controller.console_recent_actions().get("actions", []),
                    "probability_space": controller.console_probability_space(),
                    "explainability": {
                        "why": controller.empty_why_payload(),
                        "trace": controller.empty_trace_payload(),
                        "replay": controller.empty_replay_payload(),
                    },
                }
            )

        return await _run_serialized_async(_persona_reset_payload)

    @app.get("/identity")
    async def identity() -> dict:
        return await _run_readonly_async(controller.identity_payload)

    @app.get("/subject/status")
    async def subject_status() -> dict:
        return await _run_readonly_async(controller.subject_status)

    @app.get("/meaning/status")
    async def meaning_status() -> dict:
        return await _run_readonly_async(controller.meaning_status)

    @app.get("/agency/status")
    async def agency_status() -> dict:
        return await _run_readonly_async(controller.agency_status)

    @app.get("/models/status")
    async def model_status() -> dict:
        return await _run_readonly_async(controller.model_status)

    @app.get("/settings")
    async def observer_settings() -> dict:
        return await _run_readonly_async(controller.observer_settings_payload)

    @app.post("/settings")
    async def update_observer_settings(payload: dict = Body(default={})):  # type: ignore[valid-type]
        try:
            result = controller.update_observer_settings(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        _install_autonomy_model_phase_hooks()
        return result

    @app.get("/runs/current")
    async def current_run() -> dict:
        try:
            return controller.run_status()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/runs/{run_id}/steps")
    async def run_steps(run_id: str) -> dict:
        try:
            return controller.run_steps(run_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/runs/{run_id}/tools")
    async def run_tools(run_id: str) -> dict:
        try:
            return controller.run_tools(run_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/sessions/current")
    async def current_terminal_session() -> dict:
        try:
            payload = terminal_sessions.read_current().__dict__
            payload["runtime_session_id"] = controller.load_runtime_state().session_id
            return payload
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/sessions/{session_id}")
    async def terminal_session_detail(session_id: str) -> dict:
        try:
            return terminal_sessions.read(session_id).__dict__
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/trace/{round_id}")
    async def trace(round_id: int) -> dict:
        try:
            return controller.trace_round(round_id)
        except FileNotFoundError as exc:
            return controller.empty_trace_payload(round_id)

    @app.get("/why/{round_id}")
    async def why(round_id: int) -> dict:
        try:
            return controller.why_this(round_id)
        except FileNotFoundError as exc:
            return controller.empty_why_payload(round_id)

    @app.get("/contributions/{round_id}")
    async def contributions(round_id: int) -> dict:
        try:
            return controller.contribution_breakdown(round_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/memory/top")
    async def memory_top(limit: int = 5) -> list[dict]:
        return controller.memory_top(limit=limit)

    @app.get("/memory/recall/{cue}")
    async def memory_recall(cue: str) -> dict:
        return controller.memory_recall(cue)

    @app.get("/habit/top")
    async def habit_top(limit: int = 5) -> list[dict]:
        return controller.habit_top(limit=limit)

    @app.get("/metrics/summary")
    async def metrics_summary() -> dict:
        return controller.metrics_summary()

    @app.get("/metrics/authenticity")
    async def authenticity_metrics() -> dict:
        return controller.authenticity_timeline()

    @app.get("/metrics/vitality")
    async def vitality_metrics() -> dict:
        return controller.vitality_timeline()

    @app.get("/metrics/motivation")
    async def motivation_metrics() -> dict:
        return controller.motivation_metrics()

    @app.get("/metrics/endogenous")
    async def endogenous_metrics() -> dict:
        return controller.endogenous_metrics()

    @app.get("/endogenous/status")
    async def endogenous_status() -> dict:
        return build_endogenous_status_payload(controller)

    @app.post("/endogenous/tick")
    async def endogenous_tick(trigger: str = "idle", mode: str | None = None) -> dict:
        if _observer_turn_active():
            return {
                "round_id": None,
                "cause_type": trigger,
                "mode": mode,
                "skipped": True,
                "reason": "interactive_turn_active",
            }
        executed, result = await _try_run_serialized_async(
            run_endogenous_tick_payload,
            controller,
            trigger=trigger,
            mode=mode,
        )
        if executed:
            return result
        return {
            "round_id": None,
            "cause_type": trigger,
            "mode": mode,
            "skipped": True,
            "reason": "autonomy_runner_busy" if _autonomy_runner_alive() else "runtime_busy",
        }

    @app.get("/initiative/status")
    async def initiative_status() -> dict:
        return controller.initiative_status()

    @app.get("/initiative/distribution")
    async def initiative_distribution() -> dict:
        return controller.initiative_distribution()

    @app.get("/initiative/why")
    async def initiative_why(round_ref: str = "last") -> dict:
        return controller.initiative_why(round_ref)

    @app.post("/initiative/trigger")
    async def initiative_trigger(payload: dict = Body(default={})):  # type: ignore[valid-type]
        trigger = str(payload.get("trigger") or "idle")
        mode = payload.get("mode")
        force = bool(payload.get("force", False))
        response = controller.initiative_trigger_now(trigger=trigger, mode=mode, force=force)
        _sync_session_transcript_events(OBSERVER_MAIN_SESSION_ID)
        return response

    @app.get("/thought/{round_ref}")
    async def thought_snapshot(round_ref: str) -> dict:
        try:
            return controller.thought_snapshot(round_ref)
        except FileNotFoundError:
            return controller.empty_thought_payload(round_ref)

    @app.get("/speak/status")
    async def speak_status() -> dict:
        try:
            return controller.expression_channel_snapshot("speak", "last")
        except FileNotFoundError:
            return controller.empty_expression_channel_payload("speak", "last")

    @app.get("/speak/{round_ref}")
    async def speak_snapshot(round_ref: str) -> dict:
        try:
            return controller.expression_channel_snapshot("speak", round_ref)
        except FileNotFoundError:
            return controller.empty_expression_channel_payload("speak", round_ref)

    @app.get("/monologue/status")
    async def monologue_status() -> dict:
        return controller.monologue_status_lightweight()

    @app.get("/monologue/show")
    async def monologue_show(limit: int | None = None) -> dict:
        return controller.monologue_show_lightweight(limit=limit)

    @app.get("/monologue/{round_ref}")
    async def monologue_snapshot(round_ref: str) -> dict:
        try:
            return controller.expression_channel_snapshot("monologue", round_ref)
        except FileNotFoundError:
            return controller.empty_expression_channel_payload("monologue", round_ref)

    @app.get("/dream/status")
    async def dream_status() -> dict:
        return controller.dream_status()

    @app.get("/dream/runs")
    async def dream_runs() -> dict:
        return controller.dream_runs()

    @app.get("/dream/runs/{round_ref}")
    async def dream_trace(round_ref: str) -> dict:
        try:
            return controller.dream_trace(round_ref)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/dream/metrics")
    async def dream_metrics() -> dict:
        return controller.dream_metrics()

    @app.get("/dream/overview")
    async def dream_overview(limit: int = 6) -> dict:
        return controller.dream_overview(limit=limit)

    @app.get("/skills/stats")
    async def skill_stats() -> dict:
        return controller.skill_stats()

    @app.get("/skills/profile/{skill_name}")
    async def skill_profile(skill_name: str) -> dict:
        try:
            return controller.skill_profile(skill_name)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/metrics/entropy")
    async def entropy_metrics() -> dict:
        return controller.entropy_metrics()

    @app.get("/metrics/conflicts")
    async def conflict_timeline() -> dict:
        return controller.conflict_timeline()

    @app.get("/metrics/mode-switches")
    async def mode_switch_timeline() -> dict:
        return controller.mode_switch_timeline()

    @app.get("/analysis/ablation")
    async def ablation_summary() -> dict:
        return controller.ablation_summary()

    @app.get("/metrics/timeline")
    async def metrics_timeline() -> dict:
        return controller.metrics_timeline()

    @app.get("/metrics/heatmap")
    async def metrics_heatmap() -> dict:
        return controller.metrics_heatmap()

    @app.get("/diagnostics/state-delta")
    async def state_delta(window: int = 20) -> dict:
        return controller.state_delta_timeline(window=window)

    @app.get("/diagnostics/identity-blockers")
    async def identity_blockers() -> dict:
        return controller.identity_blockers()

    @app.get("/diagnostics/cue-fragmentation")
    async def cue_fragmentation() -> dict:
        return controller.cue_fragmentation_report()

    @app.get("/diagnostics/run-contamination")
    async def run_contamination(window: int = 20) -> dict:
        return controller.run_contamination_report(window=window)

    @app.get("/diagnostics/why-no-change/{round_ref}")
    async def why_no_change(round_ref: str) -> dict:
        try:
            return controller.why_no_change(round_ref)
        except FileNotFoundError:
            return controller.empty_why_no_change_payload(round_ref)

    @app.get("/diagnostics/migration")
    async def migration() -> dict:
        return controller.migration_report()

    @app.get("/replay/{round_id}")
    async def replay(round_id: int, seed: int = 0) -> dict:
        try:
            return controller.replay(round_id, seed=seed)
        except FileNotFoundError as exc:
            return controller.empty_replay_payload(round_id, seed=seed)

    @app.get("/replay/motivation/{round_id}")
    async def replay_motivation(round_id: int) -> dict:
        try:
            return controller.replay_motivation(round_id)
        except FileNotFoundError as exc:
            return controller.empty_motivation_payload(round_id)

    @app.get("/why-motivation/{round_id}")
    async def why_motivation(round_id: int) -> dict:
        try:
            return controller.why_motivation(round_id)
        except FileNotFoundError as exc:
            return controller.empty_motivation_payload(round_id)

    @app.get("/what-changed")
    async def what_changed(window: int = 5) -> dict:
        return controller.what_changed(window=window)

    @app.get("/why-not/{round_id}/{action}")
    async def why_not(round_id: int, action: str) -> dict:
        try:
            return controller.why_not(round_id, action)
        except FileNotFoundError as exc:
            return controller.empty_why_not_payload(action=action, round_ref=round_id)

    service_metadata_payload = service_metadata or {}
    workbench_dist_override = service_metadata_payload.get("workbench_dist_path")
    legacy_dashboard_override = service_metadata_payload.get("legacy_dashboard_path")
    workbench_dist_path = Path(workbench_dist_override) if workbench_dist_override else project_root_path / "apps" / "workbench" / "dist"
    if not workbench_dist_override and not workbench_dist_path.exists():
        workbench_dist_path = Path(__file__).resolve().parents[3] / "apps" / "workbench" / "dist"
    legacy_dashboard_path = Path(legacy_dashboard_override) if legacy_dashboard_override else project_root_path / "services" / "observer" / "dashboard" / "index.html"
    if not legacy_dashboard_override and not legacy_dashboard_path.exists():
        legacy_dashboard_path = Path(__file__).resolve().parents[1] / "dashboard" / "index.html"
    dashboard_path = workbench_dist_path / "index.html"

    def _missing_workbench_shell() -> HTMLResponse:
        return HTMLResponse(
            """
<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>NALR Workbench Unavailable</title>
    <style>
      body { margin: 0; font-family: "Avenir Next", "PingFang SC", sans-serif; background: #f5f0e8; color: #1d1a17; }
      main { max-width: 760px; margin: 72px auto; padding: 32px; border: 1px solid rgba(49,44,37,0.12); border-radius: 24px; background: rgba(255,255,255,0.9); }
      h1 { margin-top: 0; font-size: 2rem; }
      p { line-height: 1.8; color: #5e564d; }
      a { color: #0d8b8b; }
      code { background: rgba(49,44,37,0.06); padding: 2px 6px; border-radius: 6px; }
    </style>
  </head>
  <body>
    <main>
      <h1>Workbench 前端尚未构建</h1>
      <p>当前 <code>/dashboard</code> 只服务新的 Workbench。检测不到前端产物时，不再静默回退到旧页面。</p>
      <p>请先构建 <code>apps/workbench</code>，或临时访问 <a href="/dashboard-legacy">/dashboard-legacy</a> 查看旧版观察页。</p>
    </main>
  </body>
</html>
            """.strip()
        )

    @app.get("/dashboard-static/{asset_path:path}")
    async def dashboard_static(asset_path: str) -> FileResponse:
        if not workbench_dist_path.exists():
            raise HTTPException(status_code=404, detail="dashboard static assets are unavailable")
        target = (workbench_dist_path / asset_path).resolve()
        if not target.exists() or target.is_dir() or workbench_dist_path.resolve() not in target.parents:
            raise HTTPException(status_code=404, detail="dashboard asset not found")
        return FileResponse(
            target,
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )

    @app.get("/dashboard")
    async def dashboard() -> Response:
        if not dashboard_path.exists():
            return _missing_workbench_shell()
        return FileResponse(
            dashboard_path,
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )

    @app.get("/dashboard-legacy")
    async def dashboard_legacy() -> Response:
        if not legacy_dashboard_path.exists():
            raise HTTPException(status_code=404, detail="legacy dashboard is unavailable")
        legacy_html = legacy_dashboard_path.read_text(encoding="utf-8")
        banner = (
            "<div style=\"position:sticky;top:0;z-index:9999;padding:10px 16px;background:#fff1cd;color:#5b4200;"
            "font:14px/1.5 -apple-system,BlinkMacSystemFont,'PingFang SC',sans-serif;border-bottom:1px solid rgba(91,66,0,0.14);\">"
            "这是旧版观察页，请优先使用 <a href='/dashboard' style='color:#0d8b8b;'>Workbench</a>。"
            "</div>"
        )
        if "<body>" in legacy_html:
            legacy_html = legacy_html.replace("<body>", f"<body>{banner}", 1)
        else:
            legacy_html = banner + legacy_html
        return HTMLResponse(legacy_html)

    return app


class _LazyObserverApp:
    __test__ = False
    title = "NALR Observer"
    version = "0.6.0"

    def __init__(self) -> None:
        self._resolved_app: FastAPI | None = None

    def _app(self) -> FastAPI:
        if self._resolved_app is None:
            self._resolved_app = create_app()
        return self._resolved_app

    async def __call__(self, scope, receive, send) -> None:
        await self._app()(scope, receive, send)

    def __getattr__(self, name: str):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return getattr(self._app(), name)


app = create_app() if os.environ.get("NALR_OBSERVER_EAGER_DEFAULT_APP") == "1" else _LazyObserverApp()
