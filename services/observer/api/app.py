from __future__ import annotations

import asyncio
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
from fastapi.responses import FileResponse, StreamingResponse

from nalr.cil.runtime import build_endogenous_status_payload, run_endogenous_tick_payload
from nalr.runtime.controller import RuntimeController
from nalr.terminal_bridge.handlers import TerminalEventHandler
from nalr.terminal_bridge.protocol import ProtocolError, build_outbound_event
from nalr.terminal_bridge.session import TerminalSessionState, TerminalSessionStore
from services.observer.api.web_sessions import WebSessionBroker, build_workbench_cards

OBSERVER_MAIN_SESSION_ID = "observer-main"


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
    transcript_sync_offsets: dict[str, int] = {}
    autonomy_thread: threading.Thread | None = None
    autonomy_stop_event = threading.Event()
    autonomy_lock = threading.Lock()
    runtime_lock = threading.RLock()
    session_turn_locks: dict[str, threading.Lock] = {}
    session_turn_locks_guard = threading.Lock()
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

    def _now_iso() -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%S%z")

    def _heartbeat_interval_seconds() -> float:
        try:
            return max(0.05, float(os.environ.get("NALR_AUTONOMY_HEARTBEAT_SECONDS", "2.0")))
        except (TypeError, ValueError):
            return 2.0

    def _runtime_stall_threshold_seconds() -> float:
        return max(0.2, _heartbeat_interval_seconds() * 3.0)

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

    async def _run_serialized_async(func, /, *args, **kwargs):
        return await asyncio.to_thread(_run_serialized, func, *args, **kwargs)

    async def _run_readonly_async(func, /, *args, **kwargs):
        return await asyncio.to_thread(func, *args, **kwargs)

    def _runtime_serial_active() -> bool:
        with runtime_serial_guard:
            return runtime_serial_depth > 0

    def _session_turn_lock(session_id: str) -> threading.Lock:
        with session_turn_locks_guard:
            lock = session_turn_locks.get(session_id)
            if lock is None:
                lock = threading.Lock()
                session_turn_locks[session_id] = lock
            return lock

    def _session_turn_active(session_id: str) -> bool:
        return _session_turn_lock(session_id).locked()

    def _observer_turn_active() -> bool:
        return _session_turn_active(OBSERVER_MAIN_SESSION_ID)

    def _active_turn_sessions() -> list[str]:
        with session_turn_locks_guard:
            return sorted(session_id for session_id, lock in session_turn_locks.items() if lock.locked())

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
                }
            )
        transcript_sync_offsets[session_id] = len(transcript)
        if not events:
            return []
        events.append(terminal_handler._build_sidebar_snapshot_event(session, run_id=session.active_run_id or session.last_run_id))
        return web_broker.append_many(session_id, events)

    def _autonomy_loop() -> None:
        nonlocal autonomy_thread
        interval = _heartbeat_interval_seconds()
        try:
            while not autonomy_stop_event.wait(interval):
                status = controller.autonomy_runtime_status()
                if not status.get("enabled") or not status.get("running"):
                    return
                if _observer_turn_active():
                    continue
                try:
                    _run_serialized(controller.autonomy_step)
                    _run_serialized(_sync_session_transcript_events, OBSERVER_MAIN_SESSION_ID)
                except RuntimeError as exc:
                    message = str(exc)
                    if "interpreter shutdown" in message or "after shutdown" in message:
                        return
                    raise
                except Exception as exc:
                    try:
                        controller.stop_autonomy(reason=f"runner_exception:{type(exc).__name__}")
                    except Exception:
                        pass
                    return
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

    def _autonomy_runner_stall_payload() -> dict[str, object]:
        runner_alive = _autonomy_runner_alive()
        stall_seconds = _seconds_since_iso(service_probe_state.get("last_runtime_activity_at"))
        stalled = bool(runner_alive and stall_seconds is not None and stall_seconds > _runtime_stall_threshold_seconds())
        return {
            "runner_alive": runner_alive,
            "stalled": stalled,
            "stall_seconds": round(float(stall_seconds or 0.0), 3) if stall_seconds is not None else None,
            "stall_threshold_seconds": round(_runtime_stall_threshold_seconds(), 3),
        }

    def _autonomy_status_payload() -> dict:
        status = controller.autonomy_runtime_status()
        runner_state = _autonomy_runner_stall_payload()
        runner_alive = bool(runner_state["runner_alive"])
        loop_should_run = bool(status.get("enabled") and status.get("running"))
        return {
            **status,
            "service": _service_status_payload(),
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
        }

    def _service_status_payload() -> dict:
        now = _now_iso()
        service_probe_state["last_http_ok_at"] = now
        service_probe_state["last_probe_error"] = ""
        runner_state = _autonomy_runner_stall_payload()
        return {
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
            "runtime_serial_active": _runtime_serial_active(),
            "observer_turn_active": _observer_turn_active(),
            "active_turn_sessions": _active_turn_sessions(),
            "autonomy_runner_alive": bool(runner_state["runner_alive"]),
            "autonomy_runner_stalled": bool(runner_state["stalled"]),
            "autonomy_runner_stall_seconds": runner_state["stall_seconds"],
            "autonomy_runner_stall_threshold_seconds": runner_state["stall_threshold_seconds"],
            "project_root": str(project_root_path),
            "config_root": str(effective_config_root),
        }

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            yield
        finally:
            _stop_autonomy_runner(wait=True)

    app = FastAPI(title="NALR Observer", version="0.6.0", lifespan=lifespan)

    def _empty_decision_payload() -> dict:
        return {"message": "无决策记录", "round_id": None}

    def _session_state_payload(session_id: str) -> dict:
        session = terminal_sessions.read(session_id)
        snapshot = terminal_handler.snapshot_session(session_id)
        return _json_safe(
            {
                "session": session.__dict__,
                "status": snapshot.get("status"),
                "why": snapshot.get("why"),
                "steps": snapshot.get("steps"),
                "tools": snapshot.get("tools"),
                "approvals": snapshot.get("approvals"),
                "ui_actions": snapshot.get("ui_actions"),
                "statusline": snapshot.get("statusline"),
                "cognitive_snapshot": snapshot.get("cognitive_snapshot"),
                "console": snapshot.get("console"),
                "workbench": build_workbench_cards(snapshot),
            }
        )

    def _light_session_state_payload(session_id: str) -> dict:
        session = terminal_sessions.read(session_id)
        pending = [item for item in list(session.approvals_pending or []) if item.get("status") == "pending"]
        payload = {
            "session": session.__dict__,
            "status": {
                "run_id": session.active_run_id or session.last_run_id,
                "status": session.status,
            },
            "why": None,
            "steps": [],
            "tools": [],
            "approvals": {
                "pending": pending,
                "pending_count": len(pending),
            },
            "ui_actions": [],
            "statusline": {
                "cwd": session.cwd,
                "permission_mode": session.permission_mode,
                "run_status": session.status,
                "session_id": session.session_id,
                "run_id": str(session.active_run_id or session.last_run_id or ""),
            },
            "cognitive_snapshot": None,
            "console": None,
            "workbench": {"cards": []},
        }
        return _json_safe(payload)

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
            try:
                terminal_sessions.read(session_id)
            except FileNotFoundError:
                terminal_handler.handle({"type": "start_session", "session_id": session_id, "cwd": cwd, "persist_current": False})
            turn_events = terminal_handler.handle({"type": "user_turn", "session_id": session_id, "text": text})
            assistant = next((event.get("message") for event in reversed(turn_events) if event.get("type") == "assistant_final"), "")
            try:
                session = terminal_sessions.read(session_id).__dict__
            except FileNotFoundError:
                session = {"session_id": session_id, "cwd": cwd}
            return {
                "assistant": assistant,
                "session": session,
                "console": controller.console_refresh_payload(),
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
            state = controller.load_runtime_state()
            explicit_command_policy = False
            if controller.observer_settings_path.exists():
                try:
                    raw_settings = json.loads(controller.observer_settings_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    raw_settings = {}
                raw_autonomy_settings = raw_settings.get("autonomy", {}) if isinstance(raw_settings, dict) else {}
                explicit_command_policy = isinstance(raw_autonomy_settings, dict) and (
                    "allowed_commands" in raw_autonomy_settings or "blocked_commands" in raw_autonomy_settings
                )
            blocked_commands = list(state.autonomy_policy.blocked_commands or [])
            if not explicit_command_policy and "self_run" not in blocked_commands:
                blocked_commands.append("self_run")
                state.autonomy_policy.blocked_commands = blocked_commands
                controller._save_state(state, sync=True)
            _ensure_autonomy_runner()
            runtime_payload = _runtime_payload(session_id, light_session=False)
            runtime_payload["autonomy"] = _autonomy_status_payload()
            return runtime_payload

        return await _run_serialized_async(_web_runtime_start_payload)

    @app.get("/web/runtime/bootstrap")
    async def web_runtime_bootstrap() -> dict:
        def _web_runtime_bootstrap_payload() -> dict:
            _ensure_autonomy_runner()
            try:
                session_payload = _light_session_state_payload(OBSERVER_MAIN_SESSION_ID)
                session_attached = True
            except FileNotFoundError:
                session_payload = None
                session_attached = False
            recent_actions = controller.console_recent_actions()
            return _json_safe(
                {
                    "service": _service_status_payload(),
                    "autonomy": _autonomy_status_payload(),
                    "session": session_payload,
                    "session_attached": session_attached,
                    "recent_actions": recent_actions.get("actions", []),
                    "recent_actions_message": recent_actions.get("message", ""),
                }
            )

        return await _run_readonly_async(_web_runtime_bootstrap_payload)

    @app.post("/web/runtime/pause")
    async def web_runtime_pause(payload: dict = Body(default={})):  # type: ignore[valid-type]
        def _web_runtime_pause_payload() -> dict:
            reason = str(payload.get("reason") or "dashboard_pause")
            controller.stop_autonomy(reason=reason)
            _stop_autonomy_runner(wait=True)
            runtime_payload = _runtime_payload(OBSERVER_MAIN_SESSION_ID, light_session=False)
            runtime_payload["autonomy"] = _autonomy_status_payload()
            return runtime_payload

        return await _run_serialized_async(_web_runtime_pause_payload)

    @app.post("/web/runtime/resume")
    async def web_runtime_resume(_: dict = Body(default={})):  # type: ignore[valid-type]
        def _web_runtime_resume_payload() -> dict:
            try:
                controller.resume_run()
            except FileNotFoundError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            runtime_payload = _runtime_payload(OBSERVER_MAIN_SESSION_ID, light_session=False)
            runtime_payload["autonomy"] = _autonomy_status_payload()
            return runtime_payload

        return await _run_serialized_async(_web_runtime_resume_payload)

    @app.post("/web/runtime/wake")
    async def web_runtime_wake(_: dict = Body(default={})):  # type: ignore[valid-type]
        def _web_runtime_wake_payload() -> dict:
            controller.apply_command("mode set interactive")
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
            turn_lock = _session_turn_lock(session_id)

            def _dispatch_user_turn() -> None:
                with turn_lock:
                    try:
                        events = terminal_handler.handle(payload)
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

            threading.Thread(
                target=_dispatch_user_turn,
                name=f"observer-user-turn-{session_id}",
                daemon=True,
            ).start()
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
        return StreamingResponse(
            web_broker.stream_sse(session_id, after_id=after_id, once=once),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/web/session/state")
    async def web_session_state(session_id: str) -> dict:
        try:
            return await _run_readonly_async(_session_state_payload, session_id)
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

        return await _run_serialized_async(_console_endogenous_tick_payload)

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

        return await _run_serialized_async(_autonomy_stop_payload)

    @app.post("/autonomy/step")
    async def autonomy_step() -> dict:
        def _autonomy_step_payload() -> dict:
            if _observer_turn_active():
                payload = _autonomy_status_payload()
                payload["skipped"] = True
                payload["reason"] = "interactive_turn_active"
                return payload
            controller.autonomy_step()
            _sync_session_transcript_events(OBSERVER_MAIN_SESSION_ID)
            return _autonomy_status_payload()

        return await _run_serialized_async(_autonomy_step_payload)

    @app.post("/persona/reset")
    async def persona_reset(payload: dict = Body(default={})):  # type: ignore[valid-type]
        def _persona_reset_payload() -> dict:
            _stop_autonomy_runner(wait=True)
            controller.stop_autonomy(reason=str(payload.get("reason") or "persona_reset"))
            controller.reset_persona()
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

    @app.get("/models/status")
    async def model_status() -> dict:
        return await _run_readonly_async(controller.model_status)

    @app.get("/settings")
    async def observer_settings() -> dict:
        return await _run_readonly_async(controller.observer_settings_payload)

    @app.post("/settings")
    async def update_observer_settings(payload: dict = Body(default={})):  # type: ignore[valid-type]
        return controller.update_observer_settings(payload)

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
        return await _run_serialized_async(run_endogenous_tick_payload, controller, trigger=trigger, mode=mode)

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

    dashboard_path = project_root_path / "services" / "observer" / "dashboard" / "index.html"
    if not dashboard_path.exists():
        dashboard_path = Path(__file__).resolve().parents[1] / "dashboard" / "index.html"

    @app.get("/dashboard")
    async def dashboard() -> FileResponse:
        return FileResponse(
            dashboard_path,
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )

    return app


app = create_app()
