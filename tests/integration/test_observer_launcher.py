from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import types
from contextlib import closing
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from nalr import observer_launcher


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "config"


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _launcher_env(service_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    pythonpath_parts = [str(REPO_ROOT), str(REPO_ROOT / "src")]
    if env.get("PYTHONPATH"):
        pythonpath_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    env["NALR_OBSERVER_SERVICE_DIR"] = str(service_dir)
    return env


def _wait_until_unreachable(url: str, timeout_seconds: float = 10.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(url, timeout=0.5)
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            return
        time.sleep(0.1)
    raise RuntimeError(f"{url} still reachable after timeout")


def test_workbench_command_exposes_observer_launcher_help():
    result = subprocess.run(
        ["zsh", str(REPO_ROOT / "Open_NALR_Workbench.command"), "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert result.returncode == 0
    assert "--no-browser" in result.stdout
    assert "--port" in result.stdout


def test_workbench_command_invokes_foreground_by_default(tmp_path):
    command_path = tmp_path / "Open_NALR_Workbench.command"
    alive_path = tmp_path / "alive-observer"

    command_path.write_text((REPO_ROOT / "Open_NALR_Workbench.command").read_text(encoding="utf-8"), encoding="utf-8")
    command_path.chmod(0o755)
    alive_path.write_text("#!/bin/zsh\nprintf '%s\\n' \"$@\"\n", encoding="utf-8")
    alive_path.chmod(0o755)

    result = subprocess.run(
        ["zsh", str(command_path), "--port", "9900"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[:3] == ["foreground", "--port", "9900"]


def test_observer_launcher_requires_restart_when_requested_port_is_already_serving_dashboard(monkeypatch):
    monkeypatch.setattr(observer_launcher, "_can_bind_port", lambda host, port: False)
    monkeypatch.setattr(observer_launcher, "_url_responds", lambda url, timeout_seconds=1.0: True)
    monkeypatch.setattr(observer_launcher, "_find_available_port", lambda host, start_port, max_attempts=50: start_port + 3)

    target = observer_launcher.resolve_launch_target("127.0.0.1", 8765, "/dashboard")

    assert target["mode"] == "restart_required"
    assert target["port"] == 8765
    assert target["url"] == "http://127.0.0.1:8765/dashboard"


def test_observer_launcher_finds_next_available_port_when_requested_port_is_busy():
    occupied = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    occupied.bind(("127.0.0.1", 0))
    occupied.listen(1)
    busy_port = occupied.getsockname()[1]
    free_probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    free_probe.bind(("127.0.0.1", 0))
    free_port = free_probe.getsockname()[1]
    free_probe.close()
    try:
        original_url_responds = observer_launcher._url_responds
        original_find_available_port = observer_launcher._find_available_port
        observer_launcher._url_responds = lambda url, timeout_seconds=1.0: False
        observer_launcher._find_available_port = lambda host, start_port, max_attempts=50: free_port
        target = observer_launcher.resolve_launch_target("127.0.0.1", busy_port, "/dashboard")
        assert target["mode"] == "start"
        assert target["port"] == free_port
    finally:
        observer_launcher._url_responds = original_url_responds
        observer_launcher._find_available_port = original_find_available_port
        occupied.close()


def test_observer_launcher_autoloads_repo_env_file(tmp_path, monkeypatch):
    env_path = tmp_path / ".env.local"
    env_path.write_text("ARK_API_KEY=ark-test\nDEEPSEEK_API_KEY=deepseek-test\n", encoding="utf-8")
    monkeypatch.delenv("ARK_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    loaded = observer_launcher.autoload_repo_env(tmp_path)

    assert loaded == env_path
    assert os.environ["ARK_API_KEY"] == "ark-test"
    assert os.environ["DEEPSEEK_API_KEY"] == "deepseek-test"


def test_observer_launcher_env_autoload_preserves_existing_values(tmp_path, monkeypatch):
    env_path = tmp_path / ".env.local"
    env_path.write_text("ARK_API_KEY=ark-from-file\n", encoding="utf-8")
    monkeypatch.setenv("ARK_API_KEY", "ark-from-env")

    loaded = observer_launcher.autoload_repo_env(tmp_path)

    assert loaded == env_path
    assert os.environ["ARK_API_KEY"] == "ark-from-env"


def test_observer_launcher_help_lists_public_foreground_command():
    parser = observer_launcher.build_parser()

    help_text = parser.format_help()

    assert "foreground" in help_text
    assert "Start the observer service in the foreground." in help_text


def test_observer_launcher_normalize_argv_preserves_foreground_command():
    assert observer_launcher._normalize_argv(["foreground"]) == ["foreground"]


def test_observer_launcher_main_dispatches_public_foreground_command(monkeypatch):
    observed: dict[str, object] = {}

    def fake_serve_foreground(args):
        observed["command"] = args.command
        observed["host"] = args.host
        observed["port"] = args.port
        observed["instance_id"] = getattr(args, "instance_id", "")
        observed["started_at"] = getattr(args, "started_at", "")
        return 0

    monkeypatch.setattr(observer_launcher, "_serve_foreground", fake_serve_foreground)

    with pytest.raises(SystemExit) as exc_info:
        observer_launcher.main(["foreground", "--host", "127.0.0.1", "--port", "9876"])

    assert exc_info.value.code == 0
    assert observed["command"] == "foreground"
    assert observed["host"] == "127.0.0.1"
    assert observed["port"] == 9876
    assert observed["instance_id"]
    assert observed["started_at"]


def test_observer_launcher_main_foreground_replaces_existing_same_project_instance(monkeypatch, tmp_path):
    observed: dict[str, object] = {}

    monkeypatch.setattr(
        observer_launcher,
        "resolve_launch_target",
        lambda host, port, open_path: {"mode": "restart_required", "port": port, "url": f"http://{host}:{port}{open_path}"},
    )
    monkeypatch.setattr(
        observer_launcher,
        "_load_json_url",
        lambda url, timeout_seconds=2.0: {"project_root": str(tmp_path), "pid": 45678},
    )

    def fake_stop_pid(pid, timeout_seconds=10.0):
        observed["stopped_pid"] = pid
        observed["stop_timeout"] = timeout_seconds
        return True

    def fake_serve_foreground(args):
        observed["served_port"] = args.port
        observed["served_project_root"] = args.project_root
        return 0

    monkeypatch.setattr(observer_launcher, "_stop_pid", fake_stop_pid)
    monkeypatch.setattr(observer_launcher, "_serve_foreground", fake_serve_foreground)

    with pytest.raises(SystemExit) as exc_info:
        observer_launcher.main(["foreground", "--project-root", str(tmp_path), "--config-root", str(CONFIG_ROOT)])

    assert exc_info.value.code == 0
    assert observed["stopped_pid"] == 45678
    assert observed["stop_timeout"] == 2.0
    assert observed["served_port"] == 8765
    assert observed["served_project_root"] == str(tmp_path)


def test_serve_foreground_schedules_browser_open(monkeypatch, tmp_path):
    observed: dict[str, object] = {}

    monkeypatch.setattr(observer_launcher, "_ensure_repo_imports", lambda: tmp_path)
    monkeypatch.setattr(observer_launcher, "autoload_repo_env", lambda repo_root: None)
    monkeypatch.setattr(
        observer_launcher,
        "_sync_workbench_dist",
        lambda repo_root: observed.setdefault("workbench_sync", {"repo_root": str(repo_root), "synced": True}),
    )
    monkeypatch.setattr(
        observer_launcher,
        "_open_browser_when_ready",
        lambda url, no_browser: observed.setdefault("browser", (url, no_browser)),
        raising=False,
    )

    fake_services = types.ModuleType("services")
    fake_observer = types.ModuleType("services.observer")
    fake_api = types.ModuleType("services.observer.api")
    fake_app = types.ModuleType("services.observer.api.app")

    def fake_create_app(*, project_root, config_root, service_metadata):
        observed["service_metadata"] = service_metadata
        return "fake-app"

    fake_app.create_app = fake_create_app

    fake_uvicorn = types.ModuleType("uvicorn")

    def fake_run(app, host, port, log_level):
        observed["run"] = (app, host, port, log_level)

    fake_uvicorn.run = fake_run

    monkeypatch.setitem(sys.modules, "services", fake_services)
    monkeypatch.setitem(sys.modules, "services.observer", fake_observer)
    monkeypatch.setitem(sys.modules, "services.observer.api", fake_api)
    monkeypatch.setitem(sys.modules, "services.observer.api.app", fake_app)
    monkeypatch.setitem(sys.modules, "uvicorn", fake_uvicorn)

    args = type(
        "Args",
        (),
        {
            "host": "127.0.0.1",
            "port": 8765,
            "open_path": "/dashboard",
            "project_root": str(tmp_path),
            "config_root": str(CONFIG_ROOT),
            "instance_id": "foreground-instance",
            "started_at": "2026-04-11T03:55:00+0800",
            "no_browser": False,
        },
    )()

    result = observer_launcher._serve_foreground(args)

    assert result == 0
    assert observed["workbench_sync"] == {"repo_root": str(tmp_path), "synced": True}
    assert observed["browser"] == ("http://127.0.0.1:8765/dashboard", False)
    assert observed["run"] == ("fake-app", "127.0.0.1", 8765, "warning")


def test_sync_workbench_dist_copies_src_when_dist_is_stale(tmp_path):
    workbench_root = tmp_path / "apps" / "workbench"
    src = workbench_root / "src"
    dist = workbench_root / "dist"
    src.mkdir(parents=True)
    dist.mkdir(parents=True)
    (src / "index.html").write_text("<title>new shell</title>", encoding="utf-8")
    (dist / "index.html").write_text("<title>old shell</title>", encoding="utf-8")

    time.sleep(0.02)
    (src / "main.js").write_text("console.log('fresh');", encoding="utf-8")

    result = observer_launcher._sync_workbench_dist(tmp_path)

    assert result["synced"] is True
    assert result["reason"] == "copied"
    assert (dist / "index.html").read_text(encoding="utf-8") == "<title>new shell</title>"
    assert (dist / "main.js").read_text(encoding="utf-8") == "console.log('fresh');"


def test_open_browser_when_ready_opens_after_tcp_accepts(monkeypatch):
    observed: dict[str, object] = {}

    class FakeThread:
        def __init__(self, *, target, name, daemon):
            self._target = target
            observed["thread_name"] = name
            observed["thread_daemon"] = daemon

        def start(self):
            self._target()

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = int(listener.getsockname()[1])

    monkeypatch.setattr(observer_launcher.threading, "Thread", FakeThread)
    monkeypatch.setattr(observer_launcher, "_observer_ready_timeout_seconds", lambda: 0.2)
    monkeypatch.setattr(observer_launcher, "_open_browser", lambda url, no_browser: observed.setdefault("browser", (url, no_browser)))

    try:
        observer_launcher._open_browser_when_ready(f"http://127.0.0.1:{port}/dashboard", False)
    finally:
        listener.close()

    assert observed["thread_name"] == "nalr-observer-browser"
    assert observed["thread_daemon"] is True
    assert observed["browser"] == (f"http://127.0.0.1:{port}/dashboard", False)


def test_alive_observer_start_status_and_stop_manage_background_service(tmp_path):
    service_dir = tmp_path / "observer-service"
    port = _free_tcp_port()
    env = _launcher_env(service_dir)

    start = subprocess.run(
        [
            sys.executable,
            "-m",
            "nalr.observer_launcher",
            "start",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--project-root",
            str(tmp_path),
            "--config-root",
            str(CONFIG_ROOT),
            "--no-browser",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=20,
        env=env,
    )
    assert start.returncode == 0, start.stderr
    start_payload = json.loads(start.stdout)
    assert start_payload["healthy"] is True
    assert start_payload["pid"] > 0
    assert start_payload["url"] == f"http://127.0.0.1:{port}/dashboard"

    status = subprocess.run(
        [sys.executable, "-m", "nalr.observer_launcher", "status"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=10,
        env=env,
    )
    assert status.returncode == 0, status.stderr
    status_payload = json.loads(status.stdout)
    assert status_payload["healthy"] is True
    assert status_payload["pid"] == start_payload["pid"]
    assert status_payload["instance_id"] == start_payload["instance_id"]

    stop = subprocess.run(
        [sys.executable, "-m", "nalr.observer_launcher", "stop"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=20,
        env=env,
    )
    assert stop.returncode == 0, stop.stderr
    stop_payload = json.loads(stop.stdout)
    assert stop_payload["stopped"] is True
    _wait_until_unreachable(start_payload["url"])


def test_alive_observer_start_reuses_existing_managed_instance(tmp_path):
    service_dir = tmp_path / "observer-service"
    port = _free_tcp_port()
    env = _launcher_env(service_dir)

    try:
        first = subprocess.run(
            [
                sys.executable,
                "-m",
                "nalr.observer_launcher",
                "start",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--project-root",
                str(tmp_path),
                "--config-root",
                str(CONFIG_ROOT),
                "--no-browser",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=20,
            env=env,
        )
        assert first.returncode == 0, first.stderr
        first_payload = json.loads(first.stdout)

        second = subprocess.run(
            [
                sys.executable,
                "-m",
                "nalr.observer_launcher",
                "start",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--project-root",
                str(tmp_path),
                "--config-root",
                str(CONFIG_ROOT),
                "--no-browser",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=20,
            env=env,
        )
        assert second.returncode == 0, second.stderr
        second_payload = json.loads(second.stdout)
        assert second_payload["reused"] is True
        assert second_payload["instance_id"] == first_payload["instance_id"]
        assert second_payload["pid"] == first_payload["pid"]
    finally:
        subprocess.run(
            [sys.executable, "-m", "nalr.observer_launcher", "stop"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=20,
            env=env,
        )


def test_observer_launcher_status_recomputes_health_instead_of_echoing_cached_state(tmp_path):
    service_dir = tmp_path / "observer-service"
    service_dir.mkdir(parents=True, exist_ok=True)
    state_path = service_dir / "service.json"
    port = _free_tcp_port()
    state_path.write_text(
        json.dumps(
            {
                "instance_id": "stale-instance",
                "pid": os.getpid(),
                "host": "127.0.0.1",
                "port": port,
                "open_path": "/dashboard",
                "url": f"http://127.0.0.1:{port}/dashboard",
                "started_at": "2026-04-08T12:00:00+0800",
                "healthy": True,
                "reachable": True,
                "remote_status": {"healthy": True},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    env = _launcher_env(service_dir)

    result = subprocess.run(
        [sys.executable, "-m", "nalr.observer_launcher", "status"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=10,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["healthy"] is False
    assert payload["reachable"] is False
    assert payload["http_ready"] is False
    assert payload["last_probe_error"]


def test_observer_launcher_status_does_not_require_uvicorn_for_read_only_commands(tmp_path):
    service_dir = tmp_path / "observer-service"
    env = _launcher_env(service_dir)

    result = subprocess.run(
        ["/opt/homebrew/opt/python@3.14/bin/python3.14", "-m", "nalr.observer_launcher", "status"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=10,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["healthy"] is False


def test_observer_launcher_prefers_repo_python_before_global_python(tmp_path, monkeypatch):
    repo_root = tmp_path
    preferred = repo_root / ".venv" / "bin" / "python"
    preferred.parent.mkdir(parents=True, exist_ok=True)
    preferred.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    preferred.chmod(0o755)
    monkeypatch.setattr(observer_launcher, "_python_bin_supports_observer_service", lambda candidate: True)
    monkeypatch.setattr(
        observer_launcher,
        "PREFERRED_MANAGED_PYTHON_CANDIDATES",
        (
            ".venv/bin/python",
            "/tmp/global-python3.11",
        ),
    )

    resolved = observer_launcher._preferred_managed_python_bin(repo_root)

    assert resolved == preferred.absolute()


def test_alive_observer_script_prefers_repo_venv_python_before_global_python():
    script = (REPO_ROOT / "alive-observer").read_text(encoding="utf-8")

    repo_python_branch = 'elif supports_observer_service "$SCRIPT_DIR/.venv/bin/python"; then'
    global_python_branch = 'elif supports_observer_service "/opt/homebrew/opt/python@3.11/bin/python3.11"; then'

    assert repo_python_branch in script
    assert global_python_branch in script
    assert script.index(repo_python_branch) < script.index(global_python_branch)


def test_observer_launcher_start_uses_preferred_managed_python_bin_for_serve_process(tmp_path, monkeypatch):
    launched: dict[str, object] = {}

    class FakePopen:
        def __init__(self, argv, **kwargs):
            launched["argv"] = list(argv)
            launched["kwargs"] = kwargs
            self.pid = 43210

    monkeypatch.setattr(observer_launcher, "_preferred_managed_python_bin", lambda repo_root: Path("/tmp/python3.11"))
    monkeypatch.setattr(observer_launcher, "resolve_launch_target", lambda host, port, open_path: {"mode": "start", "port": port, "url": f"http://{host}:{port}{open_path}"})
    monkeypatch.setattr(observer_launcher, "_wait_until_ready", lambda url, timeout_seconds=20.0: None)
    monkeypatch.setattr(
        observer_launcher,
        "_load_json_url_with_error",
        lambda url, timeout_seconds=2.0: ({"instance_id": "managed-instance", "healthy": True}, ""),
    )
    monkeypatch.setattr(observer_launcher, "_probe_service", lambda **kwargs: {
        "accepting_http": True,
        "http_ready": True,
        "healthy": True,
        "last_probe_error": "",
        "remote_status": {
            "instance_id": "managed-instance",
            "healthy": True,
            "event_loop_alive": True,
            "last_http_ok_at": "2026-04-08T12:00:00+0800",
        },
        "reachable": True,
        "event_loop_alive": True,
        "last_http_ok_at": "2026-04-08T12:00:00+0800",
    })
    monkeypatch.setattr(observer_launcher, "_pid_is_running", lambda pid: int(pid or 0) == 43210)
    monkeypatch.setattr(observer_launcher, "_stop_pid", lambda pid, timeout_seconds=10.0: True)
    monkeypatch.setattr(observer_launcher.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(observer_launcher.uuid, "uuid4", lambda: type("U", (), {"hex": "managed-instance"})())

    args = type(
        "Args",
        (),
        {
            "host": "127.0.0.1",
            "port": 8766,
            "open_path": "/dashboard",
            "project_root": str(tmp_path),
            "config_root": str(CONFIG_ROOT),
            "no_browser": True,
        },
    )()

    payload = observer_launcher._launch_service(args, tmp_path)

    assert launched["argv"][0] == "/tmp/python3.11"
    assert launched["argv"][1:4] == ["-m", "nalr.observer_launcher", "serve"]
    assert Path(str(payload["python_bin"])) == Path("/tmp/python3.11")


def test_observer_launcher_ready_timeout_is_configurable(monkeypatch):
    monkeypatch.delenv("NALR_OBSERVER_READY_TIMEOUT_SECONDS", raising=False)
    assert observer_launcher._observer_ready_timeout_seconds() == 90.0

    monkeypatch.setenv("NALR_OBSERVER_READY_TIMEOUT_SECONDS", "135")
    assert observer_launcher._observer_ready_timeout_seconds() == 135.0

    monkeypatch.setenv("NALR_OBSERVER_READY_TIMEOUT_SECONDS", "0")
    assert observer_launcher._observer_ready_timeout_seconds() == 90.0
