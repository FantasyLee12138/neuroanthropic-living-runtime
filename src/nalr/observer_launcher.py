from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
import webbrowser
from pathlib import Path


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_OPEN_PATH = "/dashboard"
SERVICE_STATE_FILENAME = "service.json"
SERVICE_LOG_FILENAME = "observer.log"
PREFERRED_MANAGED_PYTHON_CANDIDATES = (
    ".venv/bin/python",
    ".venv/bin/python3.11",
    "/opt/homebrew/opt/python@3.11/bin/python3.11",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _service_dir(repo_root: Path | None = None) -> Path:
    override = os.environ.get("NALR_OBSERVER_SERVICE_DIR", "").strip()
    if override:
        return Path(override)
    root = Path(repo_root) if repo_root else _repo_root()
    return root / ".alive" / "observer-service"


def _service_state_path(repo_root: Path | None = None) -> Path:
    return _service_dir(repo_root) / SERVICE_STATE_FILENAME


def _service_log_path(repo_root: Path | None = None) -> Path:
    return _service_dir(repo_root) / SERVICE_LOG_FILENAME


def _parse_env_line(raw_line: str) -> tuple[str, str] | None:
    line = raw_line.strip()
    if not line or line.startswith("#"):
        return None
    if line.startswith("export "):
        line = line[7:].strip()
    if "=" not in line:
        return None
    key, value = line.split("=", 1)
    key = key.strip()
    value = value.strip()
    if not key:
        return None
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return key, value


def autoload_repo_env(repo_root: Path | None = None) -> Path | None:
    if os.environ.get("NALR_SKIP_ENV_AUTOLOAD") == "1":
        return None
    root = Path(repo_root) if repo_root else _repo_root()
    for candidate in (root / ".env.local", root / ".env"):
        if not candidate.exists():
            continue
        for raw_line in candidate.read_text(encoding="utf-8").splitlines():
            parsed = _parse_env_line(raw_line)
            if parsed is None:
                continue
            key, value = parsed
            os.environ.setdefault(key, value)
        return candidate
    return None


def _ensure_repo_imports() -> Path:
    repo_root = _repo_root()
    for path in (repo_root, repo_root / "src"):
        path_text = str(path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)
    return repo_root


def _emit_json(payload: dict[str, object]) -> None:
    print(json.dumps(payload, ensure_ascii=False))


def _compose_dashboard_url(host: str, port: int, open_path: str = DEFAULT_OPEN_PATH) -> str:
    return f"http://{host}:{port}{open_path}"


def _compose_service_status_url(host: str, port: int) -> str:
    return f"http://{host}:{port}/service/status"


def _wait_until_ready(url: str, timeout_seconds: float = 20.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.5) as response:
                if response.status < 500:
                    return
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last_error = exc
            time.sleep(0.25)
    if last_error is not None:
        raise RuntimeError(f"observer app did not become ready: {last_error}") from last_error
    raise RuntimeError("observer app did not become ready before timeout")


def _can_bind_port(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as candidate:
        candidate.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            candidate.bind((host, port))
        except OSError:
            return False
    return True


def _url_responds(url: str, timeout_seconds: float = 1.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout_seconds) as response:
            return response.status < 500
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        return False


def _load_json_url(url: str, timeout_seconds: float = 2.0) -> dict[str, object] | None:
    payload, _ = _load_json_url_with_error(url, timeout_seconds=timeout_seconds)
    return payload


def _load_json_url_with_error(url: str, timeout_seconds: float = 2.0) -> tuple[dict[str, object] | None, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout_seconds) as response:
            if response.status >= 500:
                return None, f"http_status:{response.status}"
            body = response.read().decode("utf-8")
    except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
        return None, str(exc)
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return None, "invalid_json"
    if not isinstance(payload, dict):
        return None, "non_object_payload"
    return payload, ""


def _find_available_port(host: str, start_port: int, max_attempts: int = 50) -> int:
    for offset in range(max_attempts):
        candidate = start_port + offset
        if _can_bind_port(host, candidate):
            return candidate
    raise RuntimeError(f"no available observer port found near {start_port}")


def resolve_launch_target(host: str, port: int, open_path: str) -> dict[str, object]:
    url = _compose_dashboard_url(host, port, open_path)
    if _can_bind_port(host, port):
        return {"mode": "start", "port": port, "url": url}
    if _url_responds(url):
        return {"mode": "restart_required", "port": port, "url": url}
    fallback_port = _find_available_port(host, port + 1)
    return {
        "mode": "start",
        "port": fallback_port,
        "url": _compose_dashboard_url(host, fallback_port, open_path),
        "requested_port": port,
    }


def _pid_is_running(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _read_service_state(repo_root: Path | None = None) -> dict[str, object] | None:
    path = _service_state_path(repo_root)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _write_service_state(payload: dict[str, object], repo_root: Path | None = None) -> Path:
    service_dir = _service_dir(repo_root)
    service_dir.mkdir(parents=True, exist_ok=True)
    path = _service_state_path(repo_root)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _clear_service_state(repo_root: Path | None = None) -> None:
    path = _service_state_path(repo_root)
    if path.exists():
        path.unlink()


def _tail_file(path: Path, lines: int = 80) -> str:
    if not path.exists():
        return ""
    content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(content[-max(1, lines) :])


def _build_subprocess_env(repo_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    parts = [str(repo_root), str(repo_root / "src")]
    if env.get("PYTHONPATH"):
        parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(parts)
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _python_bin_supports_observer_service(candidate: Path) -> bool:
    try:
        result = subprocess.run(
            [str(candidate), "-c", "import fastapi, uvicorn"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def _preferred_managed_python_bin(repo_root: Path | None = None) -> Path:
    override = str(os.environ.get("NALR_OBSERVER_PYTHON_BIN") or "").strip()
    if override:
        candidate = Path(override).expanduser()
        if candidate.exists() and os.access(candidate, os.X_OK):
            return candidate.absolute()
    root = Path(repo_root) if repo_root else _repo_root()
    for raw_candidate in PREFERRED_MANAGED_PYTHON_CANDIDATES:
        candidate = Path(raw_candidate)
        if not candidate.is_absolute():
            candidate = root / candidate
        if candidate.exists() and os.access(candidate, os.X_OK) and _python_bin_supports_observer_service(candidate):
            return candidate.absolute()
    current = Path(sys.executable)
    if current.exists() and os.access(current, os.X_OK) and _python_bin_supports_observer_service(current):
        return current.absolute()
    return Path(str(os.environ.get("PYTHON") or "python3"))


def _probe_service(
    *,
    host: str,
    port: int,
    expected_instance_id: str,
    timeout_seconds: float = 1.5,
) -> dict[str, object]:
    accepting_http = False
    http_ready = False
    healthy = False
    last_probe_error = ""
    remote_status: dict[str, object] | None = None
    try:
        with socket.create_connection((host, port), timeout=timeout_seconds):
            accepting_http = True
    except OSError as exc:
        last_probe_error = str(exc)
        return {
            "accepting_http": False,
            "http_ready": False,
            "healthy": False,
            "last_probe_error": last_probe_error,
            "remote_status": None,
            "reachable": False,
            "event_loop_alive": False,
            "last_http_ok_at": None,
        }

    remote_status, http_error = _load_json_url_with_error(_compose_service_status_url(host, port), timeout_seconds=timeout_seconds)
    if remote_status is None:
        last_probe_error = http_error or "service_status_unavailable"
    else:
        http_ready = True
        remote_instance_id = str(remote_status.get("instance_id") or "").strip()
        if expected_instance_id and remote_instance_id == expected_instance_id:
            healthy = bool(remote_status.get("healthy", True))
        else:
            last_probe_error = f"instance_mismatch:{remote_instance_id or 'missing'}"

    return {
        "accepting_http": accepting_http,
        "http_ready": http_ready,
        "healthy": healthy,
        "last_probe_error": last_probe_error,
        "remote_status": remote_status,
        "reachable": http_ready,
        "event_loop_alive": bool((remote_status or {}).get("event_loop_alive", False)),
        "last_http_ok_at": (remote_status or {}).get("last_http_ok_at"),
    }


def _normalize_service_payload(state: dict[str, object] | None, *, repo_root: Path | None = None) -> dict[str, object]:
    root = Path(repo_root) if repo_root else _repo_root()
    payload = dict(state or {})
    host = str(payload.get("host") or DEFAULT_HOST)
    port_value = payload.get("port")
    try:
        port = int(port_value) if port_value is not None else DEFAULT_PORT
    except (TypeError, ValueError):
        port = DEFAULT_PORT
    open_path = str(payload.get("open_path") or DEFAULT_OPEN_PATH)
    url = str(payload.get("url") or _compose_dashboard_url(host, port, open_path))
    service_status_url = _compose_service_status_url(host, port)
    pid = payload.get("pid")
    try:
        pid_int = int(pid) if pid is not None else 0
    except (TypeError, ValueError):
        pid_int = 0
    pid_running = _pid_is_running(pid_int)
    local_instance_id = str(payload.get("instance_id") or "").strip()
    probe = _probe_service(host=host, port=port, expected_instance_id=local_instance_id) if pid_running else {
        "accepting_http": False,
        "http_ready": False,
        "healthy": False,
        "last_probe_error": "pid_not_running" if pid_int else "service_not_started",
        "remote_status": None,
        "reachable": False,
        "event_loop_alive": False,
        "last_http_ok_at": None,
    }
    remote_status = probe["remote_status"]
    remote_instance_id = str((remote_status or {}).get("instance_id") or "").strip()
    managed_match = bool(remote_status and local_instance_id and remote_instance_id == local_instance_id)
    conflict = bool(remote_status and local_instance_id and remote_instance_id and remote_instance_id != local_instance_id)
    return {
        "instance_id": local_instance_id,
        "repo_root": str(root),
        "service_dir": str(_service_dir(root)),
        "service_state_path": str(_service_state_path(root)),
        "log_path": str(payload.get("log_path") or _service_log_path(root)),
        "host": host,
        "port": port,
        "open_path": open_path,
        "url": url,
        "pid": pid_int,
        "pid_running": pid_running,
        "reachable": bool(probe["reachable"]),
        "accepting_http": bool(probe["accepting_http"]),
        "http_ready": bool(probe["http_ready"]),
        "healthy": bool(pid_running and managed_match and probe["healthy"]),
        "managed": managed_match,
        "conflict": conflict,
        "remote_status": remote_status,
        "last_probe_error": str(probe["last_probe_error"] or ""),
        "event_loop_alive": bool(probe["event_loop_alive"]),
        "last_http_ok_at": probe["last_http_ok_at"],
        "started_at": str(payload.get("started_at") or ""),
        "project_root": str(payload.get("project_root") or ""),
        "config_root": str(payload.get("config_root") or ""),
        "python_bin": str(payload.get("python_bin") or ""),
    }


def _stop_pid(pid: int, timeout_seconds: float = 10.0) -> bool:
    if not _pid_is_running(pid):
        return True
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + max(0.5, timeout_seconds * 0.5)
    while time.monotonic() < deadline:
        if not _pid_is_running(pid):
            return True
        time.sleep(0.1)
    if _pid_is_running(pid):
        os.kill(pid, signal.SIGKILL)
    deadline = time.monotonic() + max(0.5, timeout_seconds * 0.5)
    while time.monotonic() < deadline:
        if not _pid_is_running(pid):
            return True
        time.sleep(0.1)
    return not _pid_is_running(pid)


def _open_browser(url: str, no_browser: bool) -> None:
    if not no_browser:
        webbrowser.open(url)


def _launch_service(args: argparse.Namespace, repo_root: Path) -> dict[str, object]:
    service_dir = _service_dir(repo_root)
    service_dir.mkdir(parents=True, exist_ok=True)

    current = _normalize_service_payload(_read_service_state(repo_root), repo_root=repo_root)
    requested_host = str(args.host)
    requested_port = int(args.port)
    requested_open_path = str(args.open_path)
    if current.get("healthy") and current.get("host") == requested_host and int(current.get("port") or 0) == requested_port:
        payload = {**current, "reused": True, "started": False}
        _open_browser(str(payload["url"]), bool(args.no_browser))
        return payload

    stale_pid = int(current.get("pid") or 0)
    if stale_pid:
        _stop_pid(stale_pid, timeout_seconds=2.0)
        _clear_service_state(repo_root)

    target = resolve_launch_target(requested_host, requested_port, requested_open_path)
    if target["mode"] == "restart_required":
        raise RuntimeError(
            f"检测到 {target['url']} 上已有可访问工作台，但它不属于当前受管实例。请先停止旧服务或改用其他端口。"
        )

    effective_port = int(target["port"])
    url = str(target["url"])
    instance_id = uuid.uuid4().hex
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    log_path = _service_log_path(repo_root)
    log_handle = log_path.open("ab")
    child_env = _build_subprocess_env(repo_root)
    python_bin = _preferred_managed_python_bin(repo_root)
    child = subprocess.Popen(
        [
            str(python_bin),
            "-m",
            "nalr.observer_launcher",
            "serve",
            "--host",
            requested_host,
            "--port",
            str(effective_port),
            "--project-root",
            str(args.project_root),
            "--config-root",
            str(args.config_root),
            "--open-path",
            requested_open_path,
            "--instance-id",
            instance_id,
            "--started-at",
            started_at,
        ],
        cwd=repo_root,
        env=child_env,
        stdin=subprocess.DEVNULL,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        close_fds=True,
    )
    log_handle.close()

    payload = {
        "instance_id": instance_id,
        "pid": child.pid,
        "host": requested_host,
        "port": effective_port,
        "open_path": requested_open_path,
        "url": url,
        "started_at": started_at,
        "project_root": str(Path(args.project_root)),
        "config_root": str(Path(args.config_root)),
        "log_path": str(log_path),
        "python_bin": str(python_bin),
    }
    _write_service_state(payload, repo_root)

    try:
        _wait_until_ready(url)
        service_payload, service_error = _load_json_url_with_error(_compose_service_status_url(requested_host, effective_port), timeout_seconds=2.0)
        if not service_payload or str(service_payload.get("instance_id") or "") != instance_id:
            raise RuntimeError(f"observer service became reachable but did not expose the expected managed instance: {service_error or 'instance_mismatch'}")
    except Exception:
        _stop_pid(child.pid, timeout_seconds=2.0)
        tail = _tail_file(log_path, lines=40)
        _clear_service_state(repo_root)
        detail = f"\n{tail}" if tail else ""
        raise RuntimeError(f"observer service failed to start cleanly{detail}") from None

    managed = _normalize_service_payload(_read_service_state(repo_root), repo_root=repo_root)
    payload = {**managed, "reused": False, "started": True}
    _open_browser(url, bool(args.no_browser))
    return payload


def _status_payload(repo_root: Path) -> dict[str, object]:
    state = _read_service_state(repo_root)
    payload = _normalize_service_payload(state, repo_root=repo_root)
    if not state:
        payload.update(
            {
                "healthy": False,
                "managed": False,
                "reachable": False,
                "accepting_http": False,
                "http_ready": False,
                "stopped": True,
                "last_probe_error": "service_not_started",
            }
        )
    return payload


def _stop_service(repo_root: Path) -> dict[str, object]:
    current = _status_payload(repo_root)
    pid = int(current.get("pid") or 0)
    stopped = True if not pid else _stop_pid(pid)
    _clear_service_state(repo_root)
    return {
        **current,
        "stopped": stopped,
        "healthy": False,
        "pid_running": False,
    }


def _start_command(args: argparse.Namespace) -> int:
    repo_root = _ensure_repo_imports()
    autoload_repo_env(repo_root)
    payload = _launch_service(args, repo_root)
    _emit_json(payload)
    return 0


def _status_command(_: argparse.Namespace) -> int:
    repo_root = _ensure_repo_imports()
    autoload_repo_env(repo_root)
    _emit_json(_status_payload(repo_root))
    return 0


def _stop_command(_: argparse.Namespace) -> int:
    repo_root = _ensure_repo_imports()
    autoload_repo_env(repo_root)
    _emit_json(_stop_service(repo_root))
    return 0


def _restart_command(args: argparse.Namespace) -> int:
    repo_root = _ensure_repo_imports()
    autoload_repo_env(repo_root)
    _stop_service(repo_root)
    payload = _launch_service(args, repo_root)
    payload["restarted"] = True
    _emit_json(payload)
    return 0


def _logs_command(args: argparse.Namespace) -> int:
    repo_root = _ensure_repo_imports()
    autoload_repo_env(repo_root)
    log_path = _service_log_path(repo_root)
    payload = {
        "log_path": str(log_path),
        "exists": log_path.exists(),
        "content": _tail_file(log_path, lines=int(args.lines)),
    }
    _emit_json(payload)
    return 0


def _serve_foreground(args: argparse.Namespace) -> int:
    import uvicorn

    repo_root = _ensure_repo_imports()
    autoload_repo_env(repo_root)
    from services.observer.api.app import create_app

    app = create_app(
        project_root=Path(args.project_root),
        config_root=Path(args.config_root),
        service_metadata={
            "instance_id": str(args.instance_id),
            "pid": os.getpid(),
            "host": str(args.host),
            "port": int(args.port),
            "url": _compose_dashboard_url(str(args.host), int(args.port), str(args.open_path)),
            "started_at": str(args.started_at),
            "project_root": str(Path(args.project_root)),
            "config_root": str(Path(args.config_root)),
        },
    )
    uvicorn.run(app, host=str(args.host), port=int(args.port), log_level="warning")
    return 0


def _add_launch_args(parser: argparse.ArgumentParser) -> None:
    repo_root = _repo_root()
    parser.add_argument("--host", default=DEFAULT_HOST, help="Host to bind the observer server.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to bind the observer server.")
    parser.add_argument(
        "--project-root",
        default=str(repo_root),
        help="Project root used by the runtime and observer app.",
    )
    parser.add_argument(
        "--config-root",
        default=str(repo_root / "config"),
        help="Config directory passed to the observer app.",
    )
    parser.add_argument(
        "--open-path",
        default=DEFAULT_OPEN_PATH,
        help="Path opened in the browser and used for readiness checks.",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Start the service without opening a browser tab.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage the NALR observer workbench service.")
    subparsers = parser.add_subparsers(dest="command")

    start_parser = subparsers.add_parser("start", help="Start the observer service in the background.")
    _add_launch_args(start_parser)

    status_parser = subparsers.add_parser("status", help="Show the managed observer service status.")
    status_parser.set_defaults()

    stop_parser = subparsers.add_parser("stop", help="Stop the managed observer service.")
    stop_parser.set_defaults()

    restart_parser = subparsers.add_parser("restart", help="Restart the managed observer service.")
    _add_launch_args(restart_parser)

    logs_parser = subparsers.add_parser("logs", help="Show the managed observer service log tail.")
    logs_parser.add_argument("--lines", type=int, default=80, help="Number of trailing log lines to print.")

    serve_parser = subparsers.add_parser("serve", help=argparse.SUPPRESS)
    _add_launch_args(serve_parser)
    serve_parser.add_argument("--instance-id", required=True, help=argparse.SUPPRESS)
    serve_parser.add_argument("--started-at", required=True, help=argparse.SUPPRESS)
    return parser


def _normalize_argv(argv: list[str]) -> list[str]:
    if not argv:
        return ["start"]
    if argv[0] in {"start", "status", "stop", "restart", "logs", "serve", "-h", "--help"}:
        return argv
    return ["start", *argv]


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(_normalize_argv(list(sys.argv[1:] if argv is None else argv)))
    command = getattr(args, "command", "start")
    try:
        if command == "start":
            raise SystemExit(_start_command(args))
        if command == "status":
            raise SystemExit(_status_command(args))
        if command == "stop":
            raise SystemExit(_stop_command(args))
        if command == "restart":
            raise SystemExit(_restart_command(args))
        if command == "logs":
            raise SystemExit(_logs_command(args))
        if command == "serve":
            raise SystemExit(_serve_foreground(args))
        parser.print_help()
        raise SystemExit(1)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
