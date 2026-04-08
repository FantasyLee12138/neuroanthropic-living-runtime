import inspect
import json
import socket
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from fastapi.testclient import TestClient
import uvicorn

from nalr.runtime.controller import RuntimeController
from nalr.schemas.models import RoundEvent
from nalr.terminal_bridge.protocol import build_outbound_event
from services.observer.api.app import app as default_app
from services.observer.api.app import create_app
import services.observer.api.app as observer_app_module
from nalr.terminal_bridge.handlers import TerminalEventHandler


CONFIG_ROOT = Path(__file__).resolve().parents[2] / "config"


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _wait_until_ready(url: str, timeout_seconds: float = 10.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=0.5) as response:
                if response.status < 500:
                    return
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last_error = exc
            time.sleep(0.05)
    if last_error is not None:
        raise RuntimeError(f"observer app did not become ready: {last_error}") from last_error
    raise RuntimeError("observer app did not become ready before timeout")


def _start_uvicorn_server(app):
    port = _free_tcp_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, name=f"observer-test-{port}", daemon=True)
    thread.start()
    _wait_until_ready(f"http://127.0.0.1:{port}/dashboard")
    return server, thread, port


def _stop_uvicorn_server(server, thread) -> None:
    server.should_exit = True
    thread.join(timeout=10)


def _fetch(url: str, *, timeout: float = 5.0) -> tuple[int, str]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return int(response.status), response.read().decode("utf-8")


def _post_json(url: str, payload: dict, *, timeout: float = 5.0) -> tuple[int, dict]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return int(response.status), json.loads(response.read().decode("utf-8"))


def test_observer_reads_state_and_trace(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    controller.tick(
        RoundEvent(source="user", content="hello runtime and remember coffee", target="user", cue="coffee"),
        scenario="chat",
        mode="interactive",
    )
    controller.flush_pending_io(raise_on_error=True)

    client = TestClient(create_app(project_root=tmp_path, config_root=CONFIG_ROOT))

    state_response = client.get("/state")
    trace_response = client.get("/trace/1")
    why_response = client.get("/why/1")
    metrics_response = client.get("/metrics/summary")
    recall_response = client.get("/memory/recall/coffee")
    dashboard_response = client.get("/dashboard")

    assert state_response.status_code == 200
    assert trace_response.status_code == 200
    assert why_response.status_code == 200
    assert metrics_response.status_code == 200
    assert recall_response.status_code == 200
    assert dashboard_response.status_code == 200
    assert state_response.json()["round_count"] == 1
    assert state_response.json()["trace_storage"]["trace_sync_state"] == "healthy"
    assert trace_response.json()["round_id"] == 1
    assert trace_response.json()["storage"]["read_source"] == "parquet"
    assert why_response.json()["round_id"] == 1
    assert why_response.json()["storage"]["trace_sync_state"] == "healthy"
    assert metrics_response.json()["total_rounds"] == 1
    assert recall_response.json()["cue"] == "coffee"
    assert recall_response.json()["found"] is True
    assert "NALR Observer" in dashboard_response.text


def test_observer_service_status_surfaces_instance_metadata_in_runtime_payload(tmp_path):
    client = TestClient(
        create_app(
            project_root=tmp_path,
            config_root=CONFIG_ROOT,
            service_metadata={
                "instance_id": "observer-test-instance",
                "pid": 4242,
                "host": "127.0.0.1",
                "port": 9876,
                "url": "http://127.0.0.1:9876/dashboard",
                "started_at": "2026-04-08T12:00:00+08:00",
            },
        )
    )

    service_response = client.get("/service/status")
    runtime_response = client.post("/web/runtime/start", json={})

    assert service_response.status_code == 200
    service_payload = service_response.json()
    assert service_payload["healthy"] is True
    assert service_payload["instance_id"] == "observer-test-instance"
    assert service_payload["pid"] == 4242
    assert service_payload["url"] == "http://127.0.0.1:9876/dashboard"

    assert runtime_response.status_code == 200
    runtime_payload = runtime_response.json()
    assert runtime_payload["service"]["instance_id"] == "observer-test-instance"
    assert runtime_payload["service"]["healthy"] is True
    assert runtime_payload["service"]["http_ready"] is True
    assert "event_loop_alive" in runtime_payload["service"]
    assert "last_http_ok_at" in runtime_payload["service"]


def test_observer_service_status_exposes_runtime_diagnostics(tmp_path):
    client = TestClient(
        create_app(
            project_root=tmp_path,
            config_root=CONFIG_ROOT,
            service_metadata={
                "instance_id": "observer-test-instance",
                "pid": 4242,
                "host": "127.0.0.1",
                "port": 9876,
                "url": "http://127.0.0.1:9876/dashboard",
                "started_at": "2026-04-08T12:00:00+08:00",
            },
        )
    )

    response = client.get("/service/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["healthy"] is True
    assert payload["http_ready"] is True
    assert payload["event_loop_alive"] is True
    assert payload["last_probe_error"] == ""
    assert payload["last_http_ok_at"]


def test_observer_service_status_reports_active_turn_diagnostics(tmp_path, monkeypatch):
    app = create_app(project_root=tmp_path, config_root=CONFIG_ROOT)
    client = TestClient(app)
    original_handle = TerminalEventHandler.handle
    started = threading.Event()

    client.post("/web/session/start", json={"session_id": "observer-main", "cwd": str(tmp_path)})

    def slow_handle(self, event):
        if event.get("type") == "user_turn" and event.get("session_id") == "observer-main":
            started.set()
            time.sleep(0.3)
            return [build_outbound_event("assistant_final", session_id=event["session_id"], message="slow observer reply")]
        return original_handle(self, event)

    monkeypatch.setattr(TerminalEventHandler, "handle", slow_handle)

    def dispatch_turn() -> None:
        client.post("/web/session/event", json={"type": "user_turn", "session_id": "observer-main", "text": "继续"})

    worker = threading.Thread(target=dispatch_turn, daemon=True)
    worker.start()
    assert started.wait(timeout=1.0)

    deadline = time.monotonic() + 1.0
    payload = None
    while time.monotonic() < deadline:
        response = client.get("/service/status")
        assert response.status_code == 200
        payload = response.json()
        if payload["observer_turn_active"]:
            break
        time.sleep(0.02)

    worker.join(timeout=1.0)
    assert payload is not None
    assert payload["healthy"] is True
    assert payload["http_ready"] is True
    assert payload["observer_turn_active"] is True
    assert "observer-main" in payload["active_turn_sessions"]
    assert "runtime_serial_active" in payload


def test_dashboard_does_not_fallback_missing_permission_state_to_ask(tmp_path):
    client = TestClient(create_app(project_root=tmp_path, config_root=CONFIG_ROOT))

    dashboard_response = client.get("/dashboard")

    assert dashboard_response.status_code == 200
    assert 'permissionLabel(session.permission_mode || "ask")' not in dashboard_response.text
    assert 'permission_mode: "ask"' not in dashboard_response.text
    assert "接纳: 未附着" in dashboard_response.text
    assert "当前无待确认操作" in dashboard_response.text


def test_dashboard_connection_badge_uses_service_status_not_backend_online_flag(tmp_path):
    client = TestClient(create_app(project_root=tmp_path, config_root=CONFIG_ROOT))

    dashboard_response = client.get("/dashboard")

    assert dashboard_response.status_code == 200
    assert "const connected = Boolean((serviceStatusState || {}).http_ready);" in dashboard_response.text
    assert "&& backendOnline" not in dashboard_response.text
    assert "/console/probability-space?round_ref=" in dashboard_response.text


def test_dashboard_bootstrap_uses_runtime_bootstrap_before_heavy_console_refresh(tmp_path):
    client = TestClient(create_app(project_root=tmp_path, config_root=CONFIG_ROOT))

    dashboard_response = client.get("/dashboard")

    assert dashboard_response.status_code == 200
    assert 'loadJson("/web/runtime/bootstrap")' in dashboard_response.text
    assert 'await Promise.allSettled([refreshSessionState(), refreshConsole({ heavy: false }), refreshSettings()]);' not in dashboard_response.text


def test_console_talk_does_not_block_dashboard_while_turn_executes(tmp_path, monkeypatch):
    app = create_app(project_root=tmp_path, config_root=CONFIG_ROOT)
    original_handle = TerminalEventHandler.handle
    started = threading.Event()

    def slow_handle(self, event):
        if event.get("type") == "user_turn":
            started.set()
            time.sleep(0.4)
            return [build_outbound_event("assistant_final", session_id=event["session_id"], message="slow reply")]
        return original_handle(self, event)

    monkeypatch.setattr(TerminalEventHandler, "handle", slow_handle)
    server, thread, port = _start_uvicorn_server(app)
    result: dict[str, object] = {}
    try:
        def run_talk() -> None:
            status, payload = _post_json(
                f"http://127.0.0.1:{port}/console/talk",
                {"session_id": "slow-console", "cwd": str(tmp_path), "text": "你好"},
                timeout=5.0,
            )
            result["status"] = status
            result["payload"] = payload

        worker = threading.Thread(target=run_talk, daemon=True)
        worker.start()
        assert started.wait(timeout=1.0)
        begin = time.perf_counter()
        dashboard_status, _ = _fetch(f"http://127.0.0.1:{port}/dashboard", timeout=1.0)
        elapsed = time.perf_counter() - begin
        worker.join(timeout=5.0)

        assert dashboard_status == 200
        assert elapsed < 0.2
        assert result["status"] == 200
        assert result["payload"]["assistant"] == "slow reply"
    finally:
        _stop_uvicorn_server(server, thread)


def test_web_session_event_does_not_block_dashboard_while_turn_executes(tmp_path, monkeypatch):
    app = create_app(project_root=tmp_path, config_root=CONFIG_ROOT)
    original_handle = TerminalEventHandler.handle
    started = threading.Event()
    server, thread, port = _start_uvicorn_server(app)
    try:
        status, payload = _post_json(
            f"http://127.0.0.1:{port}/web/session/start",
            {"session_id": "slow-web", "cwd": str(tmp_path)},
        )
        assert status == 200
        assert payload["session"]["session_id"] == "slow-web"

        def slow_handle(self, event):
            if event.get("type") == "user_turn":
                started.set()
                time.sleep(0.4)
                return [build_outbound_event("assistant_final", session_id=event["session_id"], message="slow web reply")]
            return original_handle(self, event)

        monkeypatch.setattr(TerminalEventHandler, "handle", slow_handle)
        result: dict[str, object] = {}

        def run_turn() -> None:
            turn_status, turn_payload = _post_json(
                f"http://127.0.0.1:{port}/web/session/event",
                {"type": "user_turn", "session_id": "slow-web", "text": "你好"},
                timeout=5.0,
            )
            result["status"] = turn_status
            result["payload"] = turn_payload

        worker = threading.Thread(target=run_turn, daemon=True)
        worker.start()
        assert started.wait(timeout=1.0)
        begin = time.perf_counter()
        dashboard_status, _ = _fetch(f"http://127.0.0.1:{port}/dashboard", timeout=1.0)
        elapsed = time.perf_counter() - begin
        worker.join(timeout=0.5)

        assert dashboard_status == 200
        assert elapsed < 0.2
        assert not worker.is_alive()
        assert result["status"] == 200
        assert result["payload"]["accepted"] is True
    finally:
        _stop_uvicorn_server(server, thread)


def test_background_user_turn_does_not_block_console_endogenous_tick(tmp_path, monkeypatch):
    app = create_app(project_root=tmp_path, config_root=CONFIG_ROOT)
    original_handle = TerminalEventHandler.handle
    original_tick_payload = observer_app_module.run_endogenous_tick_payload
    started = threading.Event()

    def slow_handle(self, event):
        if event.get("type") == "user_turn":
            started.set()
            time.sleep(0.4)
            return [build_outbound_event("assistant_final", session_id=event["session_id"], message="slow background reply")]
        return original_handle(self, event)

    def fast_tick_payload(controller, *, trigger="idle", mode=None):
        return {"round_id": None, "cause_type": trigger, "mode": mode}

    monkeypatch.setattr(TerminalEventHandler, "handle", slow_handle)
    monkeypatch.setattr(observer_app_module, "run_endogenous_tick_payload", fast_tick_payload)
    server, thread, port = _start_uvicorn_server(app)
    try:
        status, payload = _post_json(
            f"http://127.0.0.1:{port}/web/session/start",
            {"session_id": "observer-main", "cwd": str(tmp_path)},
        )
        assert status == 200
        assert payload["session"]["session_id"] == "observer-main"

        enqueue_status, enqueue_payload = _post_json(
            f"http://127.0.0.1:{port}/web/session/event",
            {"type": "user_turn", "session_id": "observer-main", "text": "继续"},
            timeout=5.0,
        )
        assert enqueue_status == 200
        assert enqueue_payload["accepted"] is True
        assert enqueue_payload["queued"] is True
        after_id = int(enqueue_payload.get("last_event_id") or 0)
        assert started.wait(timeout=1.0)

        begin = time.perf_counter()
        tick_status, tick_payload = _post_json(
            f"http://127.0.0.1:{port}/console/endogenous/tick",
            {"trigger": "idle", "mode": "endogenous_light"},
            timeout=1.5,
        )
        elapsed = time.perf_counter() - begin

        assert tick_status == 200
        assert elapsed < 0.2
        assert tick_payload["tick"]["cause_type"] == "idle"
        assert tick_payload["tick"]["skipped"] is True

        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/web/session/events?session_id=observer-main&after_id={after_id}&once=true",
            timeout=2.0,
        ) as response:
            event_lines = [line.decode("utf-8") for line in response.readlines() if line.startswith(b"data: ")]

        assert event_lines
        event_payloads = [json.loads(line[len("data: ") :]) for line in event_lines]
        assert event_payloads[-1]["type"] == "assistant_final"
        assert event_payloads[-1]["message"] == "slow background reply"
    finally:
        monkeypatch.setattr(observer_app_module, "run_endogenous_tick_payload", original_tick_payload)
        _stop_uvicorn_server(server, thread)


def test_console_endogenous_tick_does_not_block_dashboard_while_tick_executes(tmp_path, monkeypatch):
    app = create_app(project_root=tmp_path, config_root=CONFIG_ROOT)
    started = threading.Event()
    original_tick_payload = observer_app_module.run_endogenous_tick_payload
    original_console_refresh = RuntimeController.console_refresh_payload

    def slow_tick_payload(controller, *, trigger="idle", mode=None):
        started.set()
        time.sleep(0.4)
        return {"round_id": None, "cause_type": trigger, "mode": mode}

    def fake_console_refresh_payload(self, round_id=None):
        return {"state": {"current_round": {"round_id": round_id}}, "action_field": {}, "timeline": {"events": []}, "why_current": {}, "why_not": {}}

    monkeypatch.setattr(observer_app_module, "run_endogenous_tick_payload", slow_tick_payload)
    monkeypatch.setattr(RuntimeController, "console_refresh_payload", fake_console_refresh_payload)
    server, thread, port = _start_uvicorn_server(app)
    result: dict[str, object] = {}
    try:
        def run_tick() -> None:
            status, payload = _post_json(
                f"http://127.0.0.1:{port}/console/endogenous/tick",
                {"trigger": "idle", "mode": "endogenous_light"},
                timeout=5.0,
            )
            result["status"] = status
            result["payload"] = payload

        worker = threading.Thread(target=run_tick, daemon=True)
        worker.start()
        assert started.wait(timeout=1.0)
        begin = time.perf_counter()
        dashboard_status, _ = _fetch(f"http://127.0.0.1:{port}/dashboard", timeout=1.0)
        elapsed = time.perf_counter() - begin
        worker.join(timeout=5.0)

        assert dashboard_status == 200
        assert elapsed < 0.2
        assert result["status"] == 200
        assert result["payload"]["tick"]["cause_type"] == "idle"
    finally:
        monkeypatch.setattr(observer_app_module, "run_endogenous_tick_payload", original_tick_payload)
        monkeypatch.setattr(RuntimeController, "console_refresh_payload", original_console_refresh)
        _stop_uvicorn_server(server, thread)


def test_web_session_state_does_not_block_dashboard_while_snapshot_executes(tmp_path, monkeypatch):
    app = create_app(project_root=tmp_path, config_root=CONFIG_ROOT)
    original_snapshot_session = TerminalEventHandler.snapshot_session
    started = threading.Event()
    server, thread, port = _start_uvicorn_server(app)
    try:
        status, payload = _post_json(
            f"http://127.0.0.1:{port}/web/session/start",
            {"session_id": "slow-snapshot", "cwd": str(tmp_path)},
        )
        assert status == 200
        assert payload["session"]["session_id"] == "slow-snapshot"

        def slow_snapshot_session(self, session_id):
            if session_id == "slow-snapshot":
                started.set()
                time.sleep(0.4)
            return original_snapshot_session(self, session_id)

        monkeypatch.setattr(TerminalEventHandler, "snapshot_session", slow_snapshot_session)
        result: dict[str, object] = {}

        def run_session_state() -> None:
            status_code, body = _fetch(
                f"http://127.0.0.1:{port}/web/session/state?session_id=slow-snapshot",
                timeout=5.0,
            )
            result["status"] = status_code
            result["payload"] = json.loads(body)

        worker = threading.Thread(target=run_session_state, daemon=True)
        worker.start()
        assert started.wait(timeout=1.0)
        begin = time.perf_counter()
        dashboard_status, _ = _fetch(f"http://127.0.0.1:{port}/dashboard", timeout=1.0)
        elapsed = time.perf_counter() - begin
        worker.join(timeout=5.0)

        assert dashboard_status == 200
        assert elapsed < 0.2
        assert result["status"] == 200
        assert result["payload"]["session"]["session_id"] == "slow-snapshot"
    finally:
        _stop_uvicorn_server(server, thread)


def test_console_state_does_not_block_dashboard_while_console_snapshot_executes(tmp_path, monkeypatch):
    app = create_app(project_root=tmp_path, config_root=CONFIG_ROOT)
    original_console_state = RuntimeController.console_state
    started = threading.Event()
    server, thread, port = _start_uvicorn_server(app)
    try:
        def slow_console_state(self):
            started.set()
            time.sleep(0.4)
            return original_console_state(self)

        monkeypatch.setattr(RuntimeController, "console_state", slow_console_state)
        result: dict[str, object] = {}

        def run_console_state() -> None:
            status_code, body = _fetch(
                f"http://127.0.0.1:{port}/console/state",
                timeout=5.0,
            )
            result["status"] = status_code
            result["payload"] = json.loads(body)

        worker = threading.Thread(target=run_console_state, daemon=True)
        worker.start()
        assert started.wait(timeout=1.0)
        begin = time.perf_counter()
        dashboard_status, _ = _fetch(f"http://127.0.0.1:{port}/dashboard", timeout=1.0)
        elapsed = time.perf_counter() - begin
        worker.join(timeout=5.0)

        assert dashboard_status == 200
        assert elapsed < 0.2
        assert result["status"] == 200
        assert "brain_state" in result["payload"]
    finally:
        _stop_uvicorn_server(server, thread)


def test_observer_readonly_routes_do_not_wait_for_autonomy_step(tmp_path, monkeypatch):
    monkeypatch.setenv("NALR_AUTONOMY_HEARTBEAT_SECONDS", "0.01")
    app = create_app(project_root=tmp_path, config_root=CONFIG_ROOT)
    started = threading.Event()
    release = threading.Event()

    def blocking_autonomy_step(self):
        started.set()
        release.wait(timeout=2.0)
        return None

    monkeypatch.setattr(RuntimeController, "autonomy_step", blocking_autonomy_step)
    server, thread, port = _start_uvicorn_server(app)
    try:
        status, payload = _post_json(f"http://127.0.0.1:{port}/web/runtime/start", {}, timeout=5.0)
        assert status == 200
        assert payload["autonomy"]["running"] is True
        assert started.wait(timeout=1.0)

        identity_status, identity_body = _fetch(f"http://127.0.0.1:{port}/identity", timeout=0.5)
        console_status, console_body = _fetch(f"http://127.0.0.1:{port}/console/state", timeout=0.5)

        assert identity_status == 200
        assert console_status == 200
        assert '"display_name"' in identity_body
        assert '"current_round"' in console_body
    finally:
        release.set()
        _stop_uvicorn_server(server, thread)


def test_observer_exposes_thought_snapshot_endpoint(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    controller.tick(
        RoundEvent(source="user", content="帮我想一下现在先做什么。", target="user", cue="先做什么"),
        scenario="companion",
        mode="interactive",
    )
    controller.flush_pending_io(raise_on_error=True)

    client = TestClient(create_app(project_root=tmp_path, config_root=CONFIG_ROOT))
    response = client.get("/thought/1")

    assert response.status_code == 200
    payload = response.json()
    assert payload["round_id"] == 1
    assert "thought_summary" in payload
    assert "action_field" in payload


def test_observer_exposes_speak_and_monologue_snapshots(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    controller.tick(
        RoundEvent(source="user", content="你好，跟我说说你现在的状态。", target="user", cue="状态"),
        scenario="chat",
        mode="interactive",
    )
    controller.flush_pending_io(raise_on_error=True)

    client = TestClient(create_app(project_root=tmp_path, config_root=CONFIG_ROOT))
    speak_response = client.get("/speak/1")
    monologue_response = client.get("/monologue/1")

    assert speak_response.status_code == 200
    assert monologue_response.status_code == 200
    speak_payload = speak_response.json()
    monologue_payload = monologue_response.json()
    assert speak_payload["channel"] == "speak"
    assert speak_payload["delivery_mode"] == "speech"
    assert "preview" in speak_payload
    assert monologue_payload["channel"] == "monologue"
    assert monologue_payload["delivery_mode"] == "monologue"
    assert "preview" in monologue_payload


def test_observer_exposes_monologue_runtime_endpoints(tmp_path):
    client = TestClient(create_app(project_root=tmp_path, config_root=CONFIG_ROOT))

    status_response = client.get("/monologue/status")
    show_response = client.get("/monologue/show")

    assert status_response.status_code == 200
    assert show_response.status_code == 200
    assert "generated_total" in status_response.json()
    assert "fragments" in show_response.json()


def test_observer_monologue_runtime_endpoints_do_not_generate_model_fragments_for_dashboard(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    state = controller.load_runtime_state()
    bucket = controller._monologue_state_bucket(state)
    bucket["settings"]["generator_mode"] = "model"
    state.session_metadata["monologue_stream"] = bucket
    controller._save_state(state, sync=True)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("dashboard monologue endpoints must not trigger model generation")

    monkeypatch.setattr(RuntimeController, "_generate_monologue_fragments_via_model", fail_if_called)

    client = TestClient(create_app(project_root=tmp_path, config_root=CONFIG_ROOT))
    status_response = client.get("/monologue/status")
    show_response = client.get("/monologue/show")

    assert status_response.status_code == 200
    assert show_response.status_code == 200
    assert "generated_total" in status_response.json()
    assert "fragments" in show_response.json()


def test_observer_cold_start_attaches_default_autonomy_runner_and_surfaces_execution_truth(tmp_path):
    with TestClient(create_app(project_root=tmp_path, config_root=CONFIG_ROOT)) as client:
        response = client.get("/autonomy/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["enabled"] is True
    assert payload["running"] is True
    assert payload["loop_should_run"] is True
    assert payload["runner_attached"] is True
    assert payload["runner_alive"] is True
    assert payload["runner_source"] == "observer_heartbeat"
    assert "candidate_scores" not in payload
    assert "decision_surface" not in payload


def test_observer_autonomy_status_route_is_lightweight_and_skips_field_probe(tmp_path, monkeypatch):
    def fail_probe(self):
        raise AssertionError("/autonomy/status must use lightweight runtime status")

    monkeypatch.setattr(RuntimeController, "_autonomy_action_field_probe", fail_probe)

    with TestClient(create_app(project_root=tmp_path, config_root=CONFIG_ROOT)) as client:
        response = client.get("/autonomy/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["enabled"] is True
    assert "candidate_scores" not in payload
    assert "decision_surface" not in payload


def test_observer_service_status_reports_stalled_autonomy_runner(tmp_path, monkeypatch):
    monkeypatch.setenv("NALR_AUTONOMY_HEARTBEAT_SECONDS", "0.05")
    started = threading.Event()
    release = threading.Event()

    def blocking_autonomy_step(self):
        started.set()
        release.wait(timeout=2.0)
        return self.autonomy_status()

    monkeypatch.setattr(RuntimeController, "autonomy_step", blocking_autonomy_step)
    app = create_app(project_root=tmp_path, config_root=CONFIG_ROOT)
    server, thread, port = _start_uvicorn_server(app)
    try:
        status, payload = _post_json(f"http://127.0.0.1:{port}/web/runtime/start", {}, timeout=5.0)
        assert status == 200
        assert payload["autonomy"]["running"] is True
        assert started.wait(timeout=1.0)
        time.sleep(0.25)

        service_status, service_body = _fetch(f"http://127.0.0.1:{port}/service/status", timeout=1.0)
        autonomy_status, autonomy_body = _fetch(f"http://127.0.0.1:{port}/autonomy/status", timeout=1.0)

        assert service_status == 200
        assert autonomy_status == 200
        service_payload = json.loads(service_body)
        autonomy_payload = json.loads(autonomy_body)
        assert service_payload["autonomy_runner_alive"] is True
        assert service_payload["autonomy_runner_stalled"] is True
        assert autonomy_payload["runner_alive"] is True
        assert autonomy_payload["stalled"] is True
    finally:
        release.set()
        _stop_uvicorn_server(server, thread)


def test_observer_why_no_change_surfaces_summary(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    controller.tick(
        RoundEvent(source="user", content="你好", target="user", cue="你好"),
        scenario="chat",
        mode="interactive",
    )
    controller.flush_pending_io(raise_on_error=True)

    client = TestClient(create_app(project_root=tmp_path, config_root=CONFIG_ROOT))
    response = client.get("/diagnostics/why-no-change/1")

    assert response.status_code == 200
    payload = response.json()
    assert "summary" in payload
    assert payload["summary"]


def test_observer_observability_endpoints_return_empty_payloads_without_rounds(tmp_path):
    client = TestClient(create_app(project_root=tmp_path, config_root=CONFIG_ROOT))

    thought_response = client.get("/thought/last")
    why_no_change_response = client.get("/diagnostics/why-no-change/last")

    assert thought_response.status_code == 200
    assert why_no_change_response.status_code == 200
    assert thought_response.json()["message"] == "无决策记录"
    assert why_no_change_response.json()["message"] == "无决策记录"


def test_observer_runtime_start_syncs_autonomous_speak_into_web_session_events(tmp_path, monkeypatch):
    def fake_autonomy_step(self):
        session = self.terminal_sessions.read("observer-main")
        session.transcript_lines.append(
            {
                "kind": "assistant",
                "text": "我在这里。",
                "recorded_at": "2026-04-08T09:30:00+00:00",
            }
        )
        self.terminal_sessions.write(session)
        state = self.load_runtime_state()
        state.autonomy_policy.enabled = True
        state.autonomy_loop.running = True
        state.autonomy_loop.last_action_type = "respond"
        state.autonomy_loop.last_action_summary = "autonomous speak emitted"
        state.autonomy_loop.last_step_at = "2026-04-08T09:30:00+00:00"
        self._save_state(state, sync=True)
        return self.autonomy_status()

    monkeypatch.setattr(RuntimeController, "autonomy_step", fake_autonomy_step)

    client = TestClient(create_app(project_root=tmp_path, config_root=CONFIG_ROOT))

    start_response = client.post("/web/runtime/start", json={})
    step_response = client.post("/autonomy/step")
    session_response = client.get("/web/session/state", params={"session_id": "observer-main"})

    assert start_response.status_code == 200
    assert step_response.status_code == 200
    assert session_response.status_code == 200
    payload = session_response.json()
    assert payload["session"]["transcript_lines"][-1]["text"] == "我在这里。"

    events_response = client.get("/web/session/events", params={"session_id": "observer-main", "once": True})
    assert events_response.status_code == 200
    assert "assistant_final" in events_response.text
    assert "我在这里。" in events_response.text


def test_observer_state_and_why_surface_tlh_runtime_fields(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    state = controller.load_runtime_state()
    state.body_energy = 0.08
    state.fatigue = 0.9
    state.memory_fragments = 0.86
    state.self_continuity = 0.24
    state.meaning_strength = 0.18
    state.subjective_state.reject_all = 0.9
    state.subjective_state.spontaneous = 0.74
    state.subjective_state.meaning_made = ["安静比回应更重要"]
    controller._save_state(state, sync=True)
    controller.tick(
        RoundEvent(source="user", content="先别急着回答", target="user", cue="回答"),
        scenario="chat",
        mode="interactive",
    )
    controller.flush_pending_io(raise_on_error=True)

    client = TestClient(create_app(project_root=tmp_path, config_root=CONFIG_ROOT))

    state_response = client.get("/state")
    why_response = client.get("/why/1")

    assert state_response.status_code == 200
    assert why_response.status_code == 200
    assert "subjective_state" in state_response.json()
    assert "organic_mode" in state_response.json()
    assert "instinct_field" in state_response.json()["cognitive_snapshot"]["tlh"]
    assert "counterfactual_replays" in why_response.json()


def test_default_observer_app_is_available():
    assert default_app.title == "NALR Observer"
    assert default_app.version == "0.6.0"


def test_observer_core_routes_are_async_handlers(tmp_path):
    observer_app = create_app(project_root=tmp_path, config_root=CONFIG_ROOT)
    route_map = {route.path: route.endpoint for route in observer_app.routes if hasattr(route, "path")}

    assert inspect.iscoroutinefunction(route_map["/state"])
    assert inspect.iscoroutinefunction(route_map["/identity"])
    assert inspect.iscoroutinefunction(route_map["/metrics/summary"])


def test_observer_exposes_current_run_and_step_views(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "planner.py").write_text("VALUE = 1\n", encoding="utf-8")
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    run_payload = controller.start_run("检查 planner.py 并规划下一步")
    controller.flush_pending_io(raise_on_error=True)

    client = TestClient(create_app(project_root=tmp_path, config_root=CONFIG_ROOT))

    current_run = client.get("/runs/current")
    step_trace = client.get(f"/runs/{run_payload['run_id']}/steps")
    tool_trace = client.get(f"/runs/{run_payload['run_id']}/tools")

    assert current_run.status_code == 200
    assert step_trace.status_code == 200
    assert tool_trace.status_code == 200
    assert current_run.json()["run_id"] == run_payload["run_id"]
    assert step_trace.json()["steps"]
    assert tool_trace.json()["tools"][0]["tool_name"] == "repo_scan"


def test_observer_exposes_model_status(tmp_path):
    client = TestClient(create_app(project_root=tmp_path, config_root=CONFIG_ROOT))

    response = client.get("/models/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["tiers"]["small_model"]["model"] == "ep-20260404191810-qfn7s"
    assert payload["tiers"]["medium_model"]["model"] == "deepseek-chat"
    assert payload["agent_bindings"]["SalienceAgent"] == "small_model"
    assert payload["agent_bindings"]["planner"] == "medium_model"
    assert "chat_fast" in payload["routes"]
    assert payload["route_policies"]["chat_fast"]["latency_budget_ms"] == 700
    assert payload["route_policies"]["task_run"]["hot_path"] == "supervisor_run"


def test_observer_settings_can_override_newborn_unlocks_and_local_model(tmp_path):
    client = TestClient(create_app(project_root=tmp_path, config_root=CONFIG_ROOT))

    update_response = client.post(
        "/settings",
        json={
            "newborn": {
                "organic_mode": {
                    "instinct_first": True,
                },
                "subjective_state": {
                    "spontaneous": 0.11,
                    "felt": ["轻微想动"],
                },
            },
            "models": {
                "model_tiers": {
                    "local_model": {
                        "mode": "remote",
                        "backend": "openai_compatible",
                        "base_url": "http://127.0.0.1:11434/v1",
                        "model": "qwen-local",
                        "timeout_ms": 15000,
                        "retries": 0,
                        "api_key_env": "LOCAL_MODEL_API_KEY",
                        "enabled": True,
                    }
                },
                "agent_model_bindings": {
                    "Renderer": "local_model",
                },
            },
        },
    )

    assert update_response.status_code == 200
    settings_payload = update_response.json()
    assert settings_payload["newborn"]["organic_mode"]["instinct_first"] is True
    assert settings_payload["newborn"]["subjective_state"]["spontaneous"] == 0.11
    assert settings_payload["models"]["model_tiers"]["local_model"]["backend"] == "openai_compatible"
    assert settings_payload["models"]["agent_model_bindings"]["Renderer"] == "local_model"

    reset_response = client.post("/persona/reset", json={})
    assert reset_response.status_code == 200
    state_payload = reset_response.json()["state"]
    assert state_payload["organic_mode"]["instinct_first"] is True
    assert state_payload["subjective_state"]["spontaneous"] == 0.11


def test_observer_settings_can_override_autonomy_command_permissions(tmp_path):
    client = TestClient(create_app(project_root=tmp_path, config_root=CONFIG_ROOT))

    update_response = client.post(
        "/settings",
        json={
            "autonomy": {
                "clear_safe_mode_on_start": True,
                "allowed_commands": ["replay", "memory recall", "trace why"],
                "blocked_commands": ["network", "git commit", "dream run"],
            }
        },
    )

    assert update_response.status_code == 200
    settings_payload = update_response.json()
    assert settings_payload["autonomy"]["allowed_commands"] == ["replay", "memory recall", "trace why"]
    assert settings_payload["autonomy"]["blocked_commands"] == ["network", "git commit", "dream run"]

    start_response = client.post("/web/runtime/start", json={})

    assert start_response.status_code == 200
    autonomy_payload = start_response.json()["autonomy"]
    assert autonomy_payload["allowed_commands"] == ["replay", "memory recall", "trace why"]
    assert autonomy_payload["blocked_commands"] == ["network", "git commit", "dream run"]


def test_observer_settings_can_override_autonomy_boundaries_and_budget(tmp_path):
    client = TestClient(create_app(project_root=tmp_path, config_root=CONFIG_ROOT))

    update_response = client.post(
        "/settings",
        json={
            "autonomy": {
                "allowed_operator_levels": ["read_only"],
                "network_enabled": True,
                "external_io_enabled": True,
                "allow_commit": True,
                "max_rounds_per_hour": 12,
                "max_tool_actions_per_hour": 7,
                "failure_trip_threshold": 5,
                "auto_safe_mode": False,
                "quiet_hours": [1, 2, 3],
            }
        },
    )

    assert update_response.status_code == 200
    settings_payload = update_response.json()
    assert settings_payload["autonomy"]["allowed_operator_levels"] == ["read_only"]
    assert settings_payload["autonomy"]["network_enabled"] is True
    assert settings_payload["autonomy"]["external_io_enabled"] is True
    assert settings_payload["autonomy"]["allow_commit"] is True
    assert settings_payload["autonomy"]["max_rounds_per_hour"] == 12
    assert settings_payload["autonomy"]["max_tool_actions_per_hour"] == 7
    assert settings_payload["autonomy"]["failure_trip_threshold"] == 5
    assert settings_payload["autonomy"]["auto_safe_mode"] is False
    assert settings_payload["autonomy"]["quiet_hours"] == [1, 2, 3]

    start_response = client.post("/web/runtime/start", json={})

    assert start_response.status_code == 200
    state_response = client.get("/state")
    assert state_response.status_code == 200
    autonomy_state = state_response.json()["autonomy_policy"]
    budget_usage = start_response.json()["autonomy"]["budget_usage"]

    assert autonomy_state["allowed_operator_levels"] == ["read_only"]
    assert autonomy_state["network_enabled"] is True
    assert autonomy_state["external_io_enabled"] is True
    assert autonomy_state["allow_commit"] is True
    assert autonomy_state["failure_trip_threshold"] == 5
    assert autonomy_state["auto_safe_mode"] is False
    assert autonomy_state["quiet_hours"] == [1, 2, 3]
    assert budget_usage["max_rounds_per_hour"] == 12
    assert budget_usage["max_tool_actions_per_hour"] == 7


def test_observer_settings_allow_zero_autonomy_hourly_caps(tmp_path):
    client = TestClient(create_app(project_root=tmp_path, config_root=CONFIG_ROOT))

    update_response = client.post(
        "/settings",
        json={
            "autonomy": {
                "max_rounds_per_hour": 0,
                "max_tool_actions_per_hour": 0,
            }
        },
    )

    assert update_response.status_code == 200
    settings_payload = update_response.json()
    assert settings_payload["autonomy"]["max_rounds_per_hour"] == 0
    assert settings_payload["autonomy"]["max_tool_actions_per_hour"] == 0

    start_response = client.post("/web/runtime/start", json={})

    assert start_response.status_code == 200
    state_response = client.get("/state")
    assert state_response.status_code == 200
    autonomy_state = state_response.json()["autonomy_policy"]
    budget_usage = start_response.json()["autonomy"]["budget_usage"]

    assert autonomy_state["max_rounds_per_hour"] == 0
    assert autonomy_state["max_tool_actions_per_hour"] == 0
    assert budget_usage["max_rounds_per_hour"] == 0
    assert budget_usage["max_tool_actions_per_hour"] == 0


def test_observer_exposes_terminal_session_mapping(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "agent.py").write_text("VALUE = 1\n", encoding="utf-8")
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    handler = TerminalEventHandler(controller)
    handler.handle({"type": "start_session", "session_id": "sess-observer", "cwd": str(tmp_path)})
    events = handler.handle({"type": "user_turn", "session_id": "sess-observer", "text": "检查 agent.py"})
    run_id = next(item for item in events if item["type"] == "run_status")["run"]["run_id"]

    client = TestClient(create_app(project_root=tmp_path, config_root=CONFIG_ROOT))

    current_session = client.get("/sessions/current")
    session_detail = client.get("/sessions/sess-observer")

    assert current_session.status_code == 200
    assert session_detail.status_code == 200
    assert current_session.json()["session_id"] == "sess-observer"
    assert session_detail.json()["active_run_id"] == run_id


def test_observer_prefers_parquet_round_mirror_when_json_round_files_are_missing(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    controller.tick(
        RoundEvent(
            source="user",
            content="Please remember tea and plan the next step.",
            target="user",
            cue="tea",
            valence=0.1,
        ),
        scenario="task",
        mode="interactive",
    )
    controller.flush_pending_io(raise_on_error=True)

    round_json = tmp_path / ".alive" / "traces" / "rounds" / "round_1.json"
    round_jsonl = tmp_path / ".alive" / "traces" / "round_traces.jsonl"
    assert round_json.exists()
    round_json.unlink()
    round_jsonl.write_text("", encoding="utf-8")

    client = TestClient(create_app(project_root=tmp_path, config_root=CONFIG_ROOT))

    trace_response = client.get("/trace/1")
    metrics_response = client.get("/metrics/summary")

    assert trace_response.status_code == 200
    assert metrics_response.status_code == 200
    assert trace_response.json()["round_id"] == 1
    assert trace_response.json()["storage"]["read_source"] == "parquet"
    assert metrics_response.json()["total_rounds"] == 1


def test_observer_exposes_dream_status_runs_and_metrics(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    controller.tick(
        RoundEvent(
            source="user",
            content="Remember tea for sleep.",
            target="user",
            cue="tea",
            valence=0.2,
        ),
        scenario="companion",
        mode="interactive",
    )
    controller.tick(
        RoundEvent(
            source="system",
            content="Sleep reshape around tea.",
            target="user",
            cue="tea",
        ),
        scenario="companion",
        mode="sleep",
    )
    controller.flush_pending_io(raise_on_error=True)

    client = TestClient(create_app(project_root=tmp_path, config_root=CONFIG_ROOT))

    status_response = client.get("/dream/status")
    runs_response = client.get("/dream/runs")
    trace_response = client.get("/dream/runs/last")
    metrics_response = client.get("/dream/metrics")

    assert status_response.status_code == 200
    assert runs_response.status_code == 200
    assert trace_response.status_code == 200
    assert metrics_response.status_code == 200
    assert status_response.json()["enabled"] is True
    assert runs_response.json()["runs"]
    assert trace_response.json()["trigger"] == "sleep_full"
    assert metrics_response.json()["total_runs"] == 1


def test_observer_exposes_dream_overview_with_latest_and_recent_runs(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    controller.tick(
        RoundEvent(
            source="user",
            content="Remember tea for sleep.",
            target="user",
            cue="tea",
            valence=0.2,
        ),
        scenario="companion",
        mode="interactive",
    )
    controller.tick(
        RoundEvent(
            source="system",
            content="Idle reshape around tea.",
            target="user",
            cue="tea",
        ),
        scenario="companion",
        mode="idle",
    )
    controller.tick(
        RoundEvent(
            source="system",
            content="Sleep reshape around tea.",
            target="user",
            cue="tea",
        ),
        scenario="companion",
        mode="sleep",
    )
    controller.flush_pending_io(raise_on_error=True)

    client = TestClient(create_app(project_root=tmp_path, config_root=CONFIG_ROOT))
    response = client.get("/dream/overview")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"]["enabled"] is True
    assert payload["metrics"]["total_runs"] == 2
    assert payload["latest_run"]["dream_run_id"]
    assert payload["latest_run"]["trigger"] in {"idle_light", "sleep_full"}
    assert payload["latest_run"]["semantic_summary"]["evaluated_types"]
    assert payload["recent_runs"]
    assert len(payload["recent_runs"]) == 2


def test_observer_exposes_console_routes_for_latest_runtime_state(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    controller.tick(
        RoundEvent(
            source="user",
            content="Remember tea and reflect on it.",
            target="user",
            cue="tea",
            valence=0.2,
        ),
        scenario="chat",
        mode="interactive",
    )
    controller.flush_pending_io(raise_on_error=True)

    client = TestClient(create_app(project_root=tmp_path, config_root=CONFIG_ROOT))

    console_state = client.get("/console/state")
    console_action_field = client.get("/console/action-field")
    console_timeline = client.get("/console/timeline")
    console_why = client.get("/console/why/current")
    console_why_not = client.get("/console/why-not/rest")

    assert console_state.status_code == 200
    assert console_action_field.status_code == 200
    assert console_timeline.status_code == 200
    assert console_why.status_code == 200
    assert console_why_not.status_code == 200

    state_payload = console_state.json()
    assert "brain_state" in state_payload
    assert "cognitive_snapshot" in state_payload
    assert "current_round" in state_payload
    assert "trace_ref" in state_payload["current_round"]
    assert "route_type" in state_payload["current_round"]

    action_payload = console_action_field.json()
    assert "top_actions" in action_payload
    assert "winner" in action_payload
    assert "token_field" in action_payload
    assert "contribution_stack" in action_payload

    timeline_payload = console_timeline.json()
    assert "events" in timeline_payload
    assert timeline_payload["events"]
    assert timeline_payload["events"][0]["type"] in {
        "stimulus",
        "memory_activation",
        "motivation_rise",
        "action_arbitration",
        "token_gate",
        "endogenous_tick",
        "dream_effect",
    }

    why_payload = console_why.json()
    assert why_payload["round_id"] == 1
    assert "why" in why_payload
    assert "summary" in why_payload["why"]

    why_not_payload = console_why_not.json()
    assert why_not_payload["round_id"] == 1
    assert why_not_payload["action"] == "rest"
    assert "why_not" in why_not_payload


def test_observer_exposes_console_refresh_route_with_tlh_payload(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    state = controller.load_runtime_state()
    state.body_energy = 0.08
    state.fatigue = 0.91
    state.memory_fragments = 0.84
    state.self_continuity = 0.23
    state.meaning_strength = 0.16
    state.subjective_state.reject_all = 0.86
    state.subjective_state.spontaneous = 0.75
    state.subjective_state.meaning_made = ["先回到内部，再决定是否说话"]
    controller._save_state(state, sync=True)
    controller.tick(
        RoundEvent(
            source="user",
            content="先不要着急回答",
            target="user",
            cue="回答",
        ),
        scenario="chat",
        mode="interactive",
    )
    controller.flush_pending_io(raise_on_error=True)

    client = TestClient(create_app(project_root=tmp_path, config_root=CONFIG_ROOT))

    response = client.get("/console/refresh")

    assert response.status_code == 200
    payload = response.json()
    assert "state" in payload
    assert "action_field" in payload
    assert "timeline" in payload
    assert "why_current" in payload
    assert "why_not" in payload
    assert "probability_field" in payload
    assert "counterfactual_preview" in payload
    assert "source_links" in payload
    assert "recent_rounds" in payload
    assert "instinct_field" in payload["state"]["cognitive_snapshot"]["tlh"]
    assert payload["probability_field"]["action"]["winner_posterior"]
    assert payload["source_links"][0]["panel_id"]


def test_dashboard_shell_surfaces_tlh_observer_sections(tmp_path):
    client = TestClient(create_app(project_root=tmp_path, config_root=CONFIG_ROOT))

    response = client.get("/dashboard")

    assert response.status_code == 200
    assert "主体控制" in response.text
    assert "开始运行" in response.text
    assert "重置人格" in response.text
    assert "主体状态" in response.text
    assert "概率与坍缩" in response.text
    assert "四维空间" in response.text
    assert "四层概率场" in response.text
    assert "表达通道" in response.text
    assert "对外说话" in response.text
    assert "内部独白" in response.text
    assert "运行体简介" in response.text
    assert 'id="observe-dialog"' in response.text
    assert 'id="system-clock"' in response.text
    assert "峰值焦点" in response.text
    assert "外显表达区" in response.text
    assert "解释系统" in response.text
    assert "最近动作" in response.text
    assert "自治状态" in response.text
    assert "最近自治动作" in response.text
    assert "权限边界与预算" in response.text
    assert 'id="language-switch"' in response.text
    assert 'data-mode=' not in response.text


def test_dashboard_shell_surfaces_web_first_workspace_entry(tmp_path):
    client = TestClient(create_app(project_root=tmp_path, config_root=CONFIG_ROOT))

    response = client.get("/dashboard")

    assert response.status_code == 200
    assert "NALR 工作台" in response.text
    assert "开始运行" in response.text
    assert "进入分析" in response.text
    assert "查看回放" in response.text
    assert "认知工作台" in response.text
    assert "开发者明细" in response.text
    assert 'id="command-toggle"' in response.text
    assert 'id="process-toggle"' in response.text
    assert "命令面板" not in response.text
    assert "Token" in response.text
    assert "聊天输入只用于对话内容" in response.text


def test_observer_console_talk_and_endogenous_tick_return_console_refresh_payload(tmp_path):
    client = TestClient(create_app(project_root=tmp_path, config_root=CONFIG_ROOT))

    talk_response = client.post(
        "/console/talk",
        json={
            "session_id": "sess-console",
            "cwd": str(tmp_path),
            "text": "你好",
        },
    )

    assert talk_response.status_code == 200
    talk_payload = talk_response.json()
    assert "assistant" in talk_payload
    assert "console" in talk_payload
    assert "state" in talk_payload["console"]
    assert "action_field" in talk_payload["console"]
    assert "timeline" in talk_payload["console"]
    assert "why_current" in talk_payload["console"]
    assert "why_not" in talk_payload["console"]

    tick_response = client.post("/console/endogenous/tick", json={"trigger": "idle", "mode": "endogenous_light"})

    assert tick_response.status_code == 200
    tick_payload = tick_response.json()
    assert "tick" in tick_payload
    assert "console" in tick_payload
    assert "state" in tick_payload["console"]
    assert "action_field" in tick_payload["console"]
    assert "why_not" in tick_payload["console"]


def test_observer_exposes_autonomy_lifecycle_routes_and_console_payload(tmp_path):
    client = TestClient(create_app(project_root=tmp_path, config_root=CONFIG_ROOT))

    start_response = client.post("/autonomy/start", json={"profile": "tool_level"})
    status_response = client.get("/autonomy/status")
    step_response = client.post("/autonomy/step")
    refresh_response = client.get("/console/refresh")
    stop_response = client.post("/autonomy/stop", json={"reason": "api_stop"})

    assert start_response.status_code == 200
    assert status_response.status_code == 200
    assert step_response.status_code == 200
    assert refresh_response.status_code == 200
    assert stop_response.status_code == 200

    assert start_response.json()["profile"] == "tool_level"
    assert status_response.json()["enabled"] is True
    assert status_response.json()["runner_attached"] is True
    assert step_response.json()["last_action_type"]
    assert "autonomy" in refresh_response.json()
    assert refresh_response.json()["autonomy"]["kill_switch_available"] is True
    assert stop_response.json()["running"] is False
    assert stop_response.json()["stop_reason"] == "api_stop"
    assert stop_response.json()["runner_alive"] is False


def test_observer_web_session_api_streams_chat_events_and_session_state(tmp_path):
    client = TestClient(create_app(project_root=tmp_path, config_root=CONFIG_ROOT))

    start_response = client.post("/web/session/start", json={"session_id": "sess-web", "cwd": str(tmp_path)})
    assert start_response.status_code == 200
    assert start_response.json()["session"]["session_id"] == "sess-web"

    turn_response = client.post(
        "/web/session/event",
        json={"type": "user_turn", "session_id": "sess-web", "text": "你好"},
    )
    assert turn_response.status_code == 200
    assert turn_response.json()["accepted"] is True
    after_id = int(turn_response.json().get("last_event_id") or 0)

    with client.stream("GET", "/web/session/events", params={"session_id": "sess-web", "after_id": after_id, "once": True}) as stream_response:
        assert stream_response.status_code == 200
        event_lines = [line for line in stream_response.iter_lines() if line.startswith("data: ")]

    assert event_lines
    event_payloads = [json.loads(line[len("data: ") :]) for line in event_lines]
    event_types = [payload["type"] for payload in event_payloads]
    assert "assistant_token" in event_types
    assert "sidebar_snapshot" in event_types
    assert event_types[-1] == "assistant_final"

    state_response = client.get("/web/session/state", params={"session_id": "sess-web"})
    assert state_response.status_code == 200
    state_payload = state_response.json()
    assert state_payload["session"]["session_id"] == "sess-web"
    assert state_payload["session"]["transcript_lines"]
    assert state_payload["console"]["state"]
    assert state_payload["workbench"]["cards"]


def test_observer_web_session_api_supports_approval_round_trip(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "worker.py").write_text("VALUE = 1\n", encoding="utf-8")

    client = TestClient(create_app(project_root=tmp_path, config_root=CONFIG_ROOT))

    start_response = client.post("/web/session/start", json={"session_id": "sess-approval", "cwd": str(tmp_path)})
    assert start_response.status_code == 200

    permissions_response = client.post(
        "/web/session/event",
        json={"type": "control_command", "session_id": "sess-approval", "command": "permissions", "value": "ask"},
    )
    assert permissions_response.status_code == 200

    turn_response = client.post(
        "/web/session/event",
        json={"type": "user_turn", "session_id": "sess-approval", "text": "检查 worker.py 并规划下一步"},
    )
    assert turn_response.status_code == 200
    after_id = int(turn_response.json().get("last_event_id") or 0)

    with client.stream("GET", "/web/session/events", params={"session_id": "sess-approval", "after_id": after_id, "once": True}) as stream_response:
        assert stream_response.status_code == 200
        event_lines = [line for line in stream_response.iter_lines() if line.startswith("data: ")]

    assert event_lines
    event_payloads = [json.loads(line[len("data: ") :]) for line in event_lines]
    approval_event = next(item for item in event_payloads if item["type"] == "approval_request")

    approval_state = client.get("/web/session/state", params={"session_id": "sess-approval"})
    assert approval_state.status_code == 200
    assert approval_state.json()["approvals"]["pending_count"] >= 1

    approve_response = client.post(
        "/web/session/event",
        json={
            "type": "approve",
            "session_id": "sess-approval",
            "call_id": approval_event["call_id"],
            "approved": True,
        },
    )
    assert approve_response.status_code == 200

    with client.stream(
        "GET",
        "/web/session/events",
        params={"session_id": "sess-approval", "after_id": approval_event["event_id"], "once": True},
    ) as approval_stream:
        assert approval_stream.status_code == 200
        approval_lines = [line for line in approval_stream.iter_lines() if line.startswith("data: ")]

    assert approval_lines
    approval_payloads = [json.loads(line[len("data: ") :]) for line in approval_lines]
    approval_types = [payload["type"] for payload in approval_payloads]
    assert "tool_result" in approval_types
    assert "assistant_final" in approval_types


def test_observer_exposes_extended_diagnostics_endpoints(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    controller.start_run("Inspect the repository and keep working until the task is done.")
    controller.tick(
        RoundEvent(source="user", content="你是谁？", target="user"),
        scenario="chat",
        mode="interactive",
    )
    for _ in range(5):
        controller.tick(
            RoundEvent(source="user", content="你是不是根本不记得我了，我有点失望。", target="user"),
            scenario="chat",
            mode="interactive",
        )
    controller.flush_pending_io(raise_on_error=True)

    client = TestClient(create_app(project_root=tmp_path, config_root=CONFIG_ROOT))

    timeline_response = client.get("/metrics/timeline")
    heatmap_response = client.get("/metrics/heatmap")
    replay_response = client.get("/replay/1?seed=5")
    why_not_response = client.get("/why-not/1/rest")
    entropy_response = client.get("/metrics/entropy")
    skill_profile_response = client.get("/skills/profile/generate_candidates")
    state_delta_response = client.get("/diagnostics/state-delta")
    identity_blockers_response = client.get("/diagnostics/identity-blockers")
    contamination_response = client.get("/diagnostics/run-contamination")
    why_no_change_response = client.get("/diagnostics/why-no-change/last")
    cue_fragmentation_response = client.get("/diagnostics/cue-fragmentation")
    migration_response = client.get("/diagnostics/migration")

    assert timeline_response.status_code == 200
    assert heatmap_response.status_code == 200
    assert replay_response.status_code == 200
    assert why_not_response.status_code == 200
    assert entropy_response.status_code == 200
    assert skill_profile_response.status_code == 200
    assert state_delta_response.status_code == 200
    assert identity_blockers_response.status_code == 200
    assert contamination_response.status_code == 200
    assert why_no_change_response.status_code == 200
    assert cue_fragmentation_response.status_code == 200
    assert migration_response.status_code == 200
    assert timeline_response.json()["rounds"]
    assert heatmap_response.json()["actions"]
    assert replay_response.json()["round_id"] == 1
    assert why_not_response.json()["round_id"] == 1
    assert "provider_class" in entropy_response.json()
    assert skill_profile_response.json()["name"] == "generate_candidates"
    assert state_delta_response.json()["points"]
    assert "blockers" in identity_blockers_response.json()
    assert contamination_response.json()["points"]
    assert "failure_mode" in why_no_change_response.json()
    assert "families" in cue_fragmentation_response.json()
    assert "runtime" in migration_response.json()


def test_observer_state_and_metrics_expose_subjectivity_boundary_metrics(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    controller.tick(
        RoundEvent(source="user", content="记住茶。", target="user", cue="tea"),
        scenario="chat",
        mode="interactive",
    )
    controller.apply_command("mood calm")
    controller.flush_pending_io(raise_on_error=True)

    client = TestClient(create_app(project_root=tmp_path, config_root=CONFIG_ROOT))

    state_response = client.get("/state")
    metrics_response = client.get("/metrics/summary")
    motivation_response = client.get("/metrics/motivation")
    endogenous_response = client.get("/metrics/endogenous")
    timeline_response = client.get("/metrics/timeline")
    trace_response = client.get("/trace/1")
    endogenous_tick_response = client.post("/endogenous/tick", params={"trigger": "idle"})
    why_motivation_response = client.get("/why-motivation/2")
    replay_motivation_response = client.get("/replay/motivation/2")

    assert state_response.status_code == 200
    assert metrics_response.status_code == 200
    assert motivation_response.status_code == 200
    assert endogenous_response.status_code == 200
    assert timeline_response.status_code == 200
    assert trace_response.status_code == 200
    assert endogenous_tick_response.status_code == 200
    assert why_motivation_response.status_code == 200
    assert replay_motivation_response.status_code == 200
    assert state_response.json()["subjectivity"]["subject_core_integrity"] is True
    assert "boundary_violation_count" in state_response.json()["subjectivity"]
    assert "external_to_internal_ratio" in metrics_response.json()
    assert "endogenous_intent_rate" in metrics_response.json()
    assert "motivation_active_rate" in metrics_response.json()
    assert "active_rate" in motivation_response.json()
    assert "endogenous_round_rate" in endogenous_response.json()
    assert "subjectivity" in timeline_response.json()
    assert "subject_id" in trace_response.json()
    assert endogenous_tick_response.json()["cause_type"] == "endogenous"
    assert why_motivation_response.json()["cause_type"] == "endogenous"
    assert "motivation_pool" in replay_motivation_response.json()


def test_observer_runtime_start_and_pause_use_single_session_entrypoint(tmp_path):
    client = TestClient(create_app(project_root=tmp_path, config_root=CONFIG_ROOT))

    start_response = client.post("/web/runtime/start", json={})

    assert start_response.status_code == 200
    start_payload = start_response.json()
    assert start_payload["session"]["session"]["session_id"] == "observer-main"
    assert start_payload["session"]["session"]["permission_mode"] == "acceptEdits"
    assert start_payload["autonomy"]["running"] is True
    assert "recent_actions" in start_payload
    assert start_payload["session"]["console"]
    assert start_payload["session"]["workbench"]["cards"]

    session_state_response = client.get("/web/session/state", params={"session_id": "observer-main"})

    assert session_state_response.status_code == 200
    session_state_payload = session_state_response.json()
    assert session_state_payload["console"]
    assert session_state_payload["console"]["state"]
    assert session_state_payload["workbench"]["cards"]

    pause_response = client.post("/web/runtime/pause", json={})

    assert pause_response.status_code == 200
    assert pause_response.json()["autonomy"]["running"] is False


def test_observer_runtime_bootstrap_reuses_existing_observer_main_session(tmp_path):
    client = TestClient(create_app(project_root=tmp_path, config_root=CONFIG_ROOT))

    start_response = client.post("/web/runtime/start", json={})
    bootstrap_response = client.get("/web/runtime/bootstrap")

    assert start_response.status_code == 200
    assert bootstrap_response.status_code == 200
    payload = bootstrap_response.json()
    assert payload["session_attached"] is True
    assert payload["session"]["session"]["session_id"] == "observer-main"
    assert payload["session"]["session"]["permission_mode"] == "acceptEdits"
    assert payload["service"]["http_ready"] is True


def test_observer_runtime_resume_resumes_paused_run(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    run_payload = controller.start_run("inspect runtime")
    controller.pause_run(run_payload["run_id"])

    client = TestClient(create_app(project_root=tmp_path, config_root=CONFIG_ROOT))

    resume_response = client.post("/web/runtime/resume", json={})

    assert resume_response.status_code == 200
    current_run = client.get("/runs/current")
    assert current_run.status_code == 200
    assert current_run.json()["status"] == "running"


def test_observer_runtime_wake_switches_back_to_interactive_mode(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    state = controller.load_runtime_state()
    state.mode = "sleep"
    controller._save_state(state, sync=True)

    client = TestClient(create_app(project_root=tmp_path, config_root=CONFIG_ROOT))

    wake_response = client.post("/web/runtime/wake", json={})

    assert wake_response.status_code == 200
    refreshed_state = client.get("/state")
    assert refreshed_state.status_code == 200
    assert refreshed_state.json()["mode"] == "interactive"


def test_observer_runtime_start_does_not_call_autonomy_step_inline(tmp_path, monkeypatch):
    def fail_autonomy_step(self):
        raise AssertionError("autonomy_step should not run inline during /web/runtime/start")

    monkeypatch.setattr(RuntimeController, "autonomy_step", fail_autonomy_step)

    client = TestClient(create_app(project_root=tmp_path, config_root=CONFIG_ROOT))

    start_response = client.post("/web/runtime/start", json={})

    assert start_response.status_code == 200
    assert start_response.json()["autonomy"]["running"] is True


def test_observer_runtime_start_clears_stale_safe_mode_and_enters_running(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    state = controller.load_runtime_state()
    state.safe_mode = True
    state.mode = "safe"
    controller._save_state(state, sync=True)

    client = TestClient(create_app(project_root=tmp_path, config_root=CONFIG_ROOT))

    start_response = client.post("/web/runtime/start", json={})

    assert start_response.status_code == 200
    payload = start_response.json()
    assert payload["autonomy"]["running"] is True
    refreshed_state = client.get("/state").json()
    assert refreshed_state["safe_mode"] is False
    assert refreshed_state["mode"] != "safe"


def test_observer_runtime_start_rehydrates_budget_before_first_autonomy_step(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    state = controller.load_runtime_state()
    state.safe_mode = True
    state.mode = "safe"
    state.budget_remaining = 0.03
    state.body_energy = 0.08
    state.resource_state = {"resource_mode": "starvation", "scarcity_index": 0.92}
    controller._save_state(state, sync=True)

    client = TestClient(create_app(project_root=tmp_path, config_root=CONFIG_ROOT))

    start_response = client.post("/web/runtime/start", json={})

    assert start_response.status_code == 200
    payload = start_response.json()
    assert payload["autonomy"]["running"] is True
    refreshed_state = client.get("/state").json()
    assert refreshed_state["safe_mode"] is False
    assert refreshed_state["budget_remaining"] >= 0.22
    assert refreshed_state["body_energy"] >= 0.32
    assert refreshed_state["resource_state"]["resource_mode"] != "starvation"


def test_observer_runtime_start_blocks_self_run_path_for_workbench_session(tmp_path):
    client = TestClient(create_app(project_root=tmp_path, config_root=CONFIG_ROOT))

    start_response = client.post("/web/runtime/start", json={})

    assert start_response.status_code == 200
    payload = start_response.json()
    assert payload["autonomy"]["running"] is True
    assert "self_run" in payload["autonomy"]["blocked_commands"]


def test_observer_persona_reset_wipes_histories_and_returns_empty_safe_payloads(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    controller.tick(
        RoundEvent(
            source="user",
            content="记住 tea，并给出一个回应。",
            target="user",
            cue="tea",
        ),
        scenario="chat",
        mode="interactive",
    )
    controller.flush_pending_io(raise_on_error=True)

    client = TestClient(create_app(project_root=tmp_path, config_root=CONFIG_ROOT))
    start_response = client.post("/web/runtime/start", json={})
    assert start_response.status_code == 200

    reset_response = client.post("/persona/reset", json={})

    assert reset_response.status_code == 200
    reset_payload = reset_response.json()
    state_payload = reset_payload["state"]
    assert state_payload["round_count"] == 0
    assert state_payload["fatigue"] == 0
    assert state_payload["memory_fragments"] == 0
    assert state_payload["self_continuity"] == 0.5
    assert state_payload["meaning_strength"] == 0
    assert state_payload["base_metabolism"] == 1.0
    assert state_payload["subjective_state"]["felt"] == []
    assert state_payload["subjective_state"]["spontaneous"] == 0
    assert state_payload["subjective_state"]["boundary"] == 0
    assert state_payload["subjective_state"]["reject_all"] == 0
    assert state_payload["subjective_state"]["meaning_made"] == []
    assert state_payload["organic_mode"]["instinct_first"] is False
    assert state_payload["autonomy_loop"]["recent_actions"] == []
    assert reset_payload["autonomy"]["running"] is True
    assert reset_payload["autonomy"]["runner_attached"] is True
    assert state_payload["session_metadata"].get("autonomy_user_disabled") is not True
    assert reset_payload["recent_actions"] == []
    assert reset_payload["explainability"]["why"]["message"] == "无决策记录"
    assert reset_payload["explainability"]["trace"]["message"] == "无决策记录"
    assert state_payload["cognitive_snapshot"]["tlh"]["instinct_field"]["axis_values"] == {"E": 0.0, "F": 0.0, "S": 0.0, "M": 0.0}

    sessions_response = client.get("/web/sessions")
    why_response = client.get("/why/1")
    trace_response = client.get("/trace/1")

    assert sessions_response.status_code == 200
    assert sessions_response.json()["sessions"] == []
    assert why_response.status_code == 200
    assert trace_response.status_code == 200
    assert why_response.json()["message"] == "无决策记录"
    assert trace_response.json()["message"] == "无决策记录"


def test_observer_console_lightweight_routes_surface_recent_actions_and_probability_space(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    controller.tick(
        RoundEvent(
            source="user",
            content="请先休息一下再回应。",
            target="user",
            cue="休息",
        ),
        scenario="chat",
        mode="interactive",
    )
    controller.flush_pending_io(raise_on_error=True)

    client = TestClient(create_app(project_root=tmp_path, config_root=CONFIG_ROOT))

    recent_actions_response = client.get("/console/recent-actions")
    probability_space_response = client.get("/console/probability-space")

    assert recent_actions_response.status_code == 200
    assert probability_space_response.status_code == 200
    recent_actions_payload = recent_actions_response.json()
    probability_space_payload = probability_space_response.json()
    assert recent_actions_payload["actions"]
    assert "round_label" not in recent_actions_payload["actions"][0]
    assert probability_space_payload["plots"]
    assert [item["label"] for item in probability_space_payload["plots"]] == ["E-F", "E-S", "E-M", "F-S"]
    assert probability_space_payload["layers"][0]["label"] == "情境层"
    assert probability_space_payload["layers"][0]["peaks"]
    assert "peak_score" in probability_space_payload["layers"][0]
    assert probability_space_payload["space_3d"]["axes"] == {"x": "E", "y": "F", "z": "S", "intensity": "M"}
    assert "current_point" in probability_space_payload["space_3d"]


def test_console_probability_space_uses_latest_trace_snapshot_axes(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    controller.tick(
        RoundEvent(source="user", content="你好，告诉我你现在倾向做什么。", target="user", cue="倾向"),
        scenario="chat",
        mode="interactive",
    )
    controller.flush_pending_io(raise_on_error=True)
    trace = controller.trace_round(1)
    expected_axes = dict(trace["state_snapshot"]["instinct_field"]["axis_values"])

    state = controller.load_runtime_state()
    state.instinct_field.axis_values = {"E": 0.01, "F": 0.02, "S": 0.03, "M": 0.04}
    controller._save_state(state, sync=True)

    client = TestClient(create_app(project_root=tmp_path, config_root=CONFIG_ROOT))
    response = client.get("/console/probability-space")

    assert response.status_code == 200
    payload = response.json()
    assert payload["source_round_id"] == 1
    assert payload["space_3d"]["current_point"] == {
        "x": round(float(expected_axes["E"]), 4),
        "y": round(float(expected_axes["F"]), 4),
        "z": round(float(expected_axes["S"]), 4),
        "intensity": round(float(expected_axes["M"]), 4),
    }


def test_console_probability_space_accepts_round_ref_for_consistent_snapshot(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    controller.tick(
        RoundEvent(source="user", content="第一轮，先记住茶。", target="user", cue="茶"),
        scenario="chat",
        mode="interactive",
    )
    controller.tick(
        RoundEvent(source="user", content="第二轮，再看看现在更倾向什么。", target="user", cue="倾向"),
        scenario="chat",
        mode="interactive",
    )
    controller.flush_pending_io(raise_on_error=True)

    round_one_trace = controller.trace_round(1)
    expected_axes = dict(round_one_trace["state_snapshot"]["instinct_field"]["axis_values"])

    client = TestClient(create_app(project_root=tmp_path, config_root=CONFIG_ROOT))
    response = client.get("/console/probability-space", params={"round_ref": 1})

    assert response.status_code == 200
    payload = response.json()
    assert payload["source_round_id"] == 1
    assert payload["space_3d"]["current_point"] == {
        "x": round(float(expected_axes["E"]), 4),
        "y": round(float(expected_axes["F"]), 4),
        "z": round(float(expected_axes["S"]), 4),
        "intensity": round(float(expected_axes["M"]), 4),
    }


def test_observer_exposes_endogenous_status_tick_and_what_changed(tmp_path):
    client = TestClient(create_app(project_root=tmp_path, config_root=CONFIG_ROOT))

    tick_response = client.post("/endogenous/tick", params={"trigger": "idle"})
    status_response = client.get("/endogenous/status")
    why_motivation_response = client.get("/why-motivation/1")
    replay_motivation_response = client.get("/replay/motivation/1")
    what_changed_response = client.get("/what-changed", params={"window": 1})

    assert tick_response.status_code == 200
    assert status_response.status_code == 200
    assert why_motivation_response.status_code == 200
    assert replay_motivation_response.status_code == 200
    assert what_changed_response.status_code == 200
    assert tick_response.json()["cause_type"] == "endogenous"
    assert status_response.json()["latest_trigger"]["trigger_type"] == "idle"
    assert status_response.json()["latest_endogenous_round_id"] == 1
    assert why_motivation_response.json()["cause_type"] == "endogenous"
    assert "storage" in replay_motivation_response.json()
    assert what_changed_response.json()["window"] == 1
