from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

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
    monkeypatch.setattr(observer_launcher, "_wait_until_ready", lambda url: None)
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
