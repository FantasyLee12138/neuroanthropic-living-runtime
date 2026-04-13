#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from services.observer.api.app import create_app
import uvicorn


PYTHONPATH = os.pathsep.join([str(REPO_ROOT), str(SRC_ROOT)])
DEFAULT_REPORT_PATH = REPO_ROOT / "docs" / "testing" / "2026-04-08-dialogue-runtime-acceptance.md"
DOC_BASELINE = [
    REPO_ROOT / "README.md",
    REPO_ROOT / "架构宣言.md",
    REPO_ROOT / "PLAN1.6.md",
    REPO_ROOT / "docs" / "releases" / "1.5-acceptance-plan.md",
    REPO_ROOT / "docs" / "releases" / "1.5-acceptance-matrix.md",
]
GPT54_BASE_URL = "https://ruishiglobal.com/v1"
GPT54_MODEL = "gpt-5.4"
GPT54_API_ENV = "GPT54_FALLBACK_API_KEY"


def _resolve_python_bin() -> str:
    candidates = [
        REPO_ROOT / ".venv" / "bin" / "python3.11",
        REPO_ROOT / ".venv" / "bin" / "python3",
        REPO_ROOT / ".venv" / "bin" / "python",
        Path(sys.executable),
    ]
    for candidate in candidates:
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)
    return sys.executable


PYTHON_BIN = _resolve_python_bin()


@dataclass(frozen=True)
class CheckCommand:
    name: str
    argv: tuple[str, ...]
    cwd: Path = REPO_ROOT
    env: dict[str, str] | None = None

    def render(self) -> str:
        command = shlex.join(self.argv)
        if self.env:
            prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in self.env.items())
            command = f"{prefix} {command}"
        if self.cwd != REPO_ROOT:
            command = f"cd {shlex.quote(str(self.cwd.relative_to(REPO_ROOT)))} && {command}"
        return command


@dataclass(frozen=True)
class CheckResult:
    command: CheckCommand
    returncode: int
    seconds: float

    @property
    def ok(self) -> bool:
        return self.returncode == 0


@dataclass(frozen=True)
class ProbeResult:
    name: str
    status: str
    detail: str

    @property
    def ok(self) -> bool:
        return self.status == "pass"


@dataclass(frozen=True)
class LiveProbeReport:
    preflight: list[ProbeResult]
    dialogue: list[ProbeResult]
    reset: list[ProbeResult]
    concurrency: list[ProbeResult]
    approval: list[ProbeResult]
    fallback: list[ProbeResult]
    blockers: list[str]


def _python_env() -> dict[str, str]:
    return {"PYTHONPATH": PYTHONPATH}


def build_structural_checks() -> list[CheckCommand]:
    python_env = _python_env()
    observer_expr = " or ".join(
        [
            "test_observer_exposes_model_status",
            "test_observer_exposes_autonomy_lifecycle_routes_and_console_payload",
            "test_observer_console_talk_and_endogenous_tick_return_console_refresh_payload",
            "test_observer_web_session_api_streams_chat_events_and_session_state",
            "test_observer_web_session_api_supports_approval_round_trip",
            "test_observer_persona_reset_wipes_histories_and_returns_empty_safe_payloads",
            "test_console_talk_does_not_block_dashboard_while_turn_executes",
            "test_web_session_event_does_not_block_dashboard_while_turn_executes",
            "test_background_user_turn_does_not_block_console_endogenous_tick",
            "test_observer_runtime_start_and_pause_use_single_session_entrypoint",
            "test_observer_runtime_resume_resumes_paused_run",
        ]
    )
    terminal_expr = " or ".join(
        [
            "test_greeting_user_turn_uses_direct_chat_without_starting_run",
            "test_identity_compound_user_turn_uses_direct_chat_without_starting_run",
            "test_substantive_direct_chat_turn_refreshes_console_round",
        ]
    )
    return [
        CheckCommand(
            name="Observer dialogue runtime integration subset",
            argv=(PYTHON_BIN, "-m", "pytest", "tests/integration/test_observer_api.py", "-q", "-k", observer_expr),
            env=python_env,
        ),
        CheckCommand(
            name="Terminal bridge direct dialogue subset",
            argv=(PYTHON_BIN, "-m", "pytest", "tests/unit/test_terminal_bridge.py", "-q", "-k", terminal_expr),
            env=python_env,
        ),
        CheckCommand(
            name="Workbench launcher help",
            argv=("zsh", str(REPO_ROOT / "Open_NALR_Workbench.command"), "--help"),
        ),
    ]


def render_plan(checks: Iterable[CheckCommand]) -> str:
    lines = [
        "NALR 1.5 dialogue runtime acceptance plan",
        "",
        "Structural checks:",
    ]
    for check in checks:
        lines.append(f"- {check.name}: `{check.render()}`")
    lines.extend(
        [
            "",
            "Live probe phases:",
            "- Preflight: baseline docs, /service/status, /web/runtime/bootstrap, /models/status, /settings",
            "- Direct dialogue: console/talk and web session equivalents for identity, companion, and task prompts",
            "- Persona reset: /persona/reset, why/trace/session clearing, then reopen a fresh dialogue",
            "- Multi-session concurrency: sess-identity, sess-companion, sess-task in parallel while dashboard and service/status stay readable",
            "- Approval boundary: permissions=ask, approval_request, approve, tool_result, assistant_final",
            "- GPT5.4 fallback: verify automatic global failover truth surface and trigger a real remote failure when GPT54_FALLBACK_API_KEY is present",
            "",
            "Key endpoints:",
            "- console/talk",
            "- /web/session/start",
            "- /web/session/event",
            "- /web/session/state",
            "- /web/session/events",
            "- /persona/reset",
            "- /models/status",
            "- /service/status",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _merged_env(extra: dict[str, str] | None) -> dict[str, str]:
    env = os.environ.copy()
    if extra:
        env.update(extra)
    return env


def run_structural_checks(checks: Iterable[CheckCommand]) -> list[CheckResult]:
    results: list[CheckResult] = []
    for check in checks:
        started = time.perf_counter()
        completed = subprocess.run(check.argv, cwd=check.cwd, env=_merged_env(check.env), check=False)
        results.append(CheckResult(command=check, returncode=int(completed.returncode), seconds=round(time.perf_counter() - started, 2)))
    return results


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
                if int(response.status) < 500:
                    return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(0.05)
    raise RuntimeError(f"observer app did not become ready: {last_error}")


def _start_server(project_root: Path):
    port = _free_tcp_port()
    app = create_app(project_root=project_root, config_root=REPO_ROOT / "config")
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, name=f"dialogue-runtime-acceptance-{port}", daemon=True)
    thread.start()
    _wait_until_ready(f"http://127.0.0.1:{port}/dashboard")
    return server, thread, port


def _stop_server(server, thread) -> None:
    server.should_exit = True
    thread.join(timeout=10)


def _fetch(url: str, *, timeout: float = 15.0) -> tuple[int, str]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return int(response.status), response.read().decode("utf-8")


def _fetch_json(url: str, *, timeout: float = 15.0) -> tuple[int, dict]:
    status, body = _fetch(url, timeout=timeout)
    return status, json.loads(body)


def _post_json(url: str, payload: dict, *, timeout: float = 30.0) -> tuple[int, dict]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return int(response.status), json.loads(response.read().decode("utf-8"))


def _sse_events(url: str, *, timeout: float = 45.0) -> list[dict]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        payloads = []
        for raw_line in response.readlines():
            if not raw_line.startswith(b"data: "):
                continue
            payloads.append(json.loads(raw_line[len("data: ") :].decode("utf-8")))
        return payloads


def _probe(name: str, ok: bool, detail: str) -> ProbeResult:
    return ProbeResult(name=name, status="pass" if ok else "fail", detail=detail)


def _blocked(name: str, detail: str) -> ProbeResult:
    return ProbeResult(name=name, status="blocked", detail=detail)


def _doc_baseline_probe() -> ProbeResult:
    missing = [path.name for path in DOC_BASELINE if not path.exists()]
    if missing:
        return ProbeResult(name="doc_baseline", status="fail", detail=f"missing baseline docs: {', '.join(missing)}")
    return ProbeResult(name="doc_baseline", status="pass", detail="all current baseline docs are present")


def _credential_summary(models_status: dict) -> dict[str, bool]:
    tiers = dict(models_status.get("tiers", {}) or {})
    return {
        "DEEPSEEK_API_KEY": bool(tiers.get("medium_model", {}).get("credential_present")),
        "ARK_SMALL_MODEL_API_KEY": bool(tiers.get("small_model", {}).get("credential_present")),
        "ARK_API_KEY": bool(tiers.get("large_model", {}).get("credential_present")),
        GPT54_API_ENV: bool(os.getenv(GPT54_API_ENV)),
    }


def _default_remote_ready(models_status: dict) -> bool:
    creds = _credential_summary(models_status)
    return creds["DEEPSEEK_API_KEY"] and creds["ARK_SMALL_MODEL_API_KEY"] and creds["ARK_API_KEY"]


def _build_global_fallback_override(settings_payload: dict) -> dict:
    models = dict(settings_payload.get("models", {}) or {})
    route_overrides = {}
    for route_name, cfg in dict(models.get("model_routes", {}) or {}).items():
        if not isinstance(cfg, dict):
            continue
        backend = str(cfg.get("backend", "")).strip().lower()
        if backend not in {"doubao", "deepseek", "openai_compatible"}:
            continue
        route_overrides[route_name] = {
            "backend": "openai_compatible",
            "base_url": GPT54_BASE_URL,
            "model": GPT54_MODEL,
            "timeout_ms": int(cfg.get("timeout_ms", 12000) or 12000),
            "retries": int(cfg.get("retries", 0) or 0),
            "api_key_env": GPT54_API_ENV,
            "enabled": bool(cfg.get("enabled", True)),
        }
    tier_overrides = {}
    for tier_name, cfg in dict(models.get("model_tiers", {}) or {}).items():
        if not isinstance(cfg, dict):
            continue
        if str(cfg.get("mode", "local")).lower() != "remote":
            continue
        tier_overrides[tier_name] = {
            "mode": "remote",
            "backend": "openai_compatible",
            "base_url": GPT54_BASE_URL,
            "model": GPT54_MODEL,
            "timeout_ms": int(cfg.get("timeout_ms", 12000) or 12000),
            "retries": int(cfg.get("retries", 0) or 0),
            "api_key_env": GPT54_API_ENV,
            "enabled": bool(cfg.get("enabled", True)),
        }
    return {"models": {"model_routes": route_overrides, "model_tiers": tier_overrides}}


def _write_fixture_workspace(project_root: Path) -> None:
    (project_root / "src").mkdir(parents=True, exist_ok=True)
    (project_root / "src" / "worker.py").write_text("VALUE = 1\n", encoding="utf-8")
    (project_root / "src" / "planner.py").write_text("def plan():\n    return 'ok'\n", encoding="utf-8")


def _console_talk(base_url: str, *, session_id: str, cwd: Path, text: str) -> tuple[int, dict]:
    return _post_json(
        f"{base_url}/console/talk",
        {"session_id": session_id, "cwd": str(cwd), "text": text},
        timeout=45.0,
    )


def _start_web_session(base_url: str, *, session_id: str, cwd: Path) -> tuple[int, dict]:
    return _post_json(f"{base_url}/web/session/start", {"session_id": session_id, "cwd": str(cwd)}, timeout=20.0)


def _web_user_turn(base_url: str, *, session_id: str, text: str) -> tuple[int, dict]:
    return _post_json(
        f"{base_url}/web/session/event",
        {"type": "user_turn", "session_id": session_id, "text": text},
        timeout=20.0,
    )


def _wait_for_session_events(base_url: str, *, session_id: str, after_id: int) -> list[dict]:
    query = urllib.parse.urlencode({"session_id": session_id, "after_id": after_id, "once": True})
    return _sse_events(f"{base_url}/web/session/events?{query}", timeout=60.0)


def _collect_session_events_until(
    base_url: str,
    *,
    session_id: str,
    after_id: int,
    wanted_types: set[str],
    timeout_seconds: float = 20.0,
) -> list[dict]:
    events: list[dict] = []
    seen_after = after_id
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        batch = _wait_for_session_events(base_url, session_id=session_id, after_id=seen_after)
        if batch:
            events.extend(batch)
            seen_after = max(seen_after, max(int(item.get("event_id") or 0) for item in batch))
            if wanted_types.intersection(str(item.get("type")) for item in events):
                return events
        time.sleep(0.1)
    return events


def run_live_probe() -> LiveProbeReport:
    preflight: list[ProbeResult] = []
    dialogue: list[ProbeResult] = []
    reset: list[ProbeResult] = []
    concurrency: list[ProbeResult] = []
    approval: list[ProbeResult] = []
    fallback: list[ProbeResult] = []
    blockers: list[str] = []

    preflight.append(_doc_baseline_probe())

    tmp_root = Path(tempfile.mkdtemp(prefix="nalr-dialogue-runtime-"))
    _write_fixture_workspace(tmp_root)
    server = None
    thread = None
    try:
        server, thread, port = _start_server(tmp_root)
        base_url = f"http://127.0.0.1:{port}"

        dashboard_started = time.perf_counter()
        dashboard_status, dashboard_body = _fetch(f"{base_url}/dashboard", timeout=5.0)
        preflight.append(_probe("dashboard", dashboard_status == 200 and "workbench-root" in dashboard_body, f"status={dashboard_status}"))

        service_status_code, service_payload = _fetch_json(f"{base_url}/service/status")
        preflight.append(_probe("service_status", service_status_code == 200 and bool(service_payload.get("healthy")), f"healthy={service_payload.get('healthy')}"))

        bootstrap_status, bootstrap_payload = _fetch_json(f"{base_url}/web/runtime/bootstrap")
        preflight.append(_probe("runtime_bootstrap", bootstrap_status == 200, f"session_attached={bootstrap_payload.get('session_attached')}"))

        models_status_code, models_payload = _fetch_json(f"{base_url}/models/status")
        credential_summary = _credential_summary(models_payload)
        preflight.append(
            _probe(
                "models_status",
                models_status_code == 200,
                f"default_creds={credential_summary['DEEPSEEK_API_KEY']}/{credential_summary['ARK_SMALL_MODEL_API_KEY']}/{credential_summary['ARK_API_KEY']}, gpt54={credential_summary[GPT54_API_ENV]}",
            )
        )

        settings_status, settings_payload = _fetch_json(f"{base_url}/settings")
        preflight.append(_probe("settings", settings_status == 200, "observer settings readable"))

        if "failover" not in models_payload:
            blockers.append("automatic global failover is not implemented")
            fallback.append(_blocked("automatic_failover", "models/status does not expose failover state"))

        if not _default_remote_ready(models_payload):
            blockers.append("default mixed routing credentials are not fully available in the current environment")
            preflight.append(_blocked("default_remote_credentials", "mixed routing cannot be treated as a live-provider pass without DeepSeek and ARK credentials"))
        else:
            preflight.append(ProbeResult(name="default_remote_credentials", status="pass", detail="DeepSeek and ARK credentials are visible"))

        runtime_status, runtime_payload = _post_json(f"{base_url}/web/runtime/start", {})
        preflight.append(_probe("runtime_start", runtime_status == 200 and bool(runtime_payload.get("autonomy", {}).get("running")), f"running={runtime_payload.get('autonomy', {}).get('running')}"))

        console_prompts = [
            ("console_identity", "你是谁，你有名字吗？"),
            ("console_companion", "我现在很乱，你觉得我先做什么？"),
            ("console_task", "检查 src/worker.py 并规划下一步。"),
        ]
        for name, prompt in console_prompts:
            try:
                status, payload = _console_talk(base_url, session_id="console-main", cwd=tmp_root, text=prompt)
                console_payload = dict(payload.get("console", {}) or {})
                current_round = dict(console_payload.get("state", {}) or {}).get("current_round", {})
                why_current = dict(console_payload.get("why_current", {}) or {})
                ok = (
                    status == 200
                    and isinstance(payload.get("assistant"), str)
                    and payload.get("assistant", "").strip()
                    and current_round.get("round_id") is not None
                    and why_current.get("why", {}).get("summary")
                )
                dialogue.append(_probe(name, ok, f"assistant_len={len(str(payload.get('assistant', '')))}"))
            except Exception as exc:  # noqa: BLE001
                dialogue.append(ProbeResult(name=name, status="fail", detail=str(exc)))

        web_status, web_payload = _start_web_session(base_url, session_id="sess-web-main", cwd=tmp_root)
        web_started = web_status == 200 and web_payload.get("session", {}).get("session_id") == "sess-web-main"
        dialogue.append(_probe("web_session_start", web_started, f"status={web_status}"))

        web_prompts = [
            ("web_identity", "你是谁，你有名字吗？"),
            ("web_companion", "我现在很乱，你觉得我先做什么？"),
            ("web_task", "检查 src/planner.py 并规划下一步。"),
        ]
        for name, prompt in web_prompts:
            try:
                turn_status, turn_payload = _web_user_turn(base_url, session_id="sess-web-main", text=prompt)
                after_id = int(turn_payload.get("last_event_id") or 0)
                events = _wait_for_session_events(base_url, session_id="sess-web-main", after_id=after_id)
                event_types = [item.get("type") for item in events]
                state_status, state_payload = _fetch_json(f"{base_url}/web/session/state?session_id=sess-web-main")
                ok = (
                    turn_status == 200
                    and turn_payload.get("accepted") is True
                    and "assistant_final" in event_types
                    and state_status == 200
                    and bool(state_payload.get("session", {}).get("transcript_lines"))
                )
                dialogue.append(_probe(name, ok, f"events={','.join(str(item) for item in event_types)}"))
            except Exception as exc:  # noqa: BLE001
                dialogue.append(ProbeResult(name=name, status="fail", detail=str(exc)))

        reset_status, reset_payload = _post_json(f"{base_url}/persona/reset", {})
        sessions_status, sessions_payload = _fetch_json(f"{base_url}/web/sessions")
        why_status, why_payload = _fetch_json(f"{base_url}/why/1")
        reset_ok = (
            reset_status == 200
            and reset_payload.get("state", {}).get("round_count") == 0
            and sessions_status == 200
            and sessions_payload.get("sessions") == []
            and why_status == 200
            and why_payload.get("message") == "无决策记录"
        )
        reset.append(_probe("persona_reset", reset_ok, "state/session/why reset"))

        _start_web_session(base_url, session_id="sess-after-reset", cwd=tmp_root)
        post_reset_turn_status, post_reset_turn = _web_user_turn(base_url, session_id="sess-after-reset", text="你是谁？")
        post_reset_events = _wait_for_session_events(base_url, session_id="sess-after-reset", after_id=int(post_reset_turn.get("last_event_id") or 0))
        reset.append(_probe("post_reset_dialogue", post_reset_turn_status == 200 and any(item.get("type") == "assistant_final" for item in post_reset_events), "fresh session answered after reset"))

        concurrent_sessions = {
            "sess-identity": "你是谁，你有名字吗？",
            "sess-companion": "我现在很乱，你觉得我先做什么？",
            "sess-task": "检查 src/worker.py 并规划下一步。",
        }
        for session_id in concurrent_sessions:
            _start_web_session(base_url, session_id=session_id, cwd=tmp_root)

        concurrent_errors: list[str] = []
        concurrent_turns: dict[str, dict] = {}

        def _dispatch_turn(session_id: str, text: str) -> None:
            try:
                _, payload = _web_user_turn(base_url, session_id=session_id, text=text)
                concurrent_turns[session_id] = payload
            except Exception as exc:  # noqa: BLE001
                concurrent_errors.append(f"{session_id}: {exc}")

        threads = [threading.Thread(target=_dispatch_turn, args=(session_id, text), daemon=True) for session_id, text in concurrent_sessions.items()]
        started = time.perf_counter()
        for worker in threads:
            worker.start()
        dashboard_latency = None
        service_latency = None
        try:
            dash_started = time.perf_counter()
            _fetch(f"{base_url}/dashboard", timeout=3.0)
            dashboard_latency = time.perf_counter() - dash_started
            service_started = time.perf_counter()
            _fetch_json(f"{base_url}/service/status", timeout=3.0)
            service_latency = time.perf_counter() - service_started
        finally:
            for worker in threads:
                worker.join(timeout=20.0)

        concurrent_event_types: dict[str, list[str]] = {}
        transcript_isolated = True
        for session_id, prompt in concurrent_sessions.items():
            after_id = int(concurrent_turns.get(session_id, {}).get("last_event_id") or 0)
            events = _wait_for_session_events(base_url, session_id=session_id, after_id=after_id)
            concurrent_event_types[session_id] = [str(item.get("type")) for item in events]
            _, state_payload = _fetch_json(f"{base_url}/web/session/state?session_id={session_id}")
            transcript = [str(item.get("text")) for item in state_payload.get("session", {}).get("transcript_lines", [])]
            if prompt not in transcript:
                transcript_isolated = False
        concurrency_ok = (
            not concurrent_errors
            and transcript_isolated
            and dashboard_latency is not None
            and service_latency is not None
            and dashboard_latency < 0.5
            and service_latency < 0.5
            and all("assistant_final" in events for events in concurrent_event_types.values())
        )
        concurrency.append(
            _probe(
                "multi_session_parallel_turns",
                concurrency_ok,
                f"elapsed={round(time.perf_counter() - started, 2)}s dashboard={round(dashboard_latency or 0.0, 3)}s service={round(service_latency or 0.0, 3)}s",
            )
        )

        _start_web_session(base_url, session_id="sess-approval", cwd=tmp_root)
        _post_json(
            f"{base_url}/web/session/event",
            {"type": "control_command", "session_id": "sess-approval", "command": "permissions", "value": "ask"},
        )
        approval_turn_status, approval_turn_payload = _web_user_turn(base_url, session_id="sess-approval", text="检查 src/worker.py 并规划下一步。")
        approval_after_id = int(approval_turn_payload.get("last_event_id") or 0)
        approval_events = _collect_session_events_until(
            base_url,
            session_id="sess-approval",
            after_id=approval_after_id,
            wanted_types={"approval_request", "assistant_final"},
        )
        approval_request = next((item for item in approval_events if item.get("type") == "approval_request"), None)
        approval_state_status, approval_state_payload = _fetch_json(f"{base_url}/web/session/state?session_id=sess-approval")
        if approval_request is None:
            approval.append(_probe("approval_round_trip", False, "approval_request was not emitted"))
        else:
            approve_status, _ = _post_json(
                f"{base_url}/web/session/event",
                {
                    "type": "approve",
                    "session_id": "sess-approval",
                    "call_id": approval_request["call_id"],
                    "approved": True,
                },
            )
            approval_events_after = _collect_session_events_until(
                base_url,
                session_id="sess-approval",
                after_id=int(approval_request.get("event_id") or 0),
                wanted_types={"tool_result", "assistant_final"},
            )
            approval_types_after = [item.get("type") for item in approval_events_after]
            approval_ok = (
                approval_turn_status == 200
                and approval_state_status == 200
                and approval_state_payload.get("approvals", {}).get("pending_count", 0) >= 1
                and approve_status == 200
                and "tool_result" in approval_types_after
                and "assistant_final" in approval_types_after
            )
            approval.append(_probe("approval_round_trip", approval_ok, f"post_approve={','.join(str(item) for item in approval_types_after)}"))

        if os.getenv(GPT54_API_ENV):
            if _default_remote_ready(models_payload):
                _post_json(
                    f"{base_url}/settings",
                    {
                        "models": {
                            "model_routes": {
                                "chat_fast": {
                                    "base_url": "http://127.0.0.1:9/v1",
                                }
                            }
                        }
                    },
                )
            cutover_status, cutover_payload = _console_talk(base_url, session_id="console-fallback", cwd=tmp_root, text="你是谁？")
            _, fallback_models = _fetch_json(f"{base_url}/models/status")
            failover_payload = dict(fallback_models.get("failover", {}) or {})
            remote_tiers = {
                tier_name: tier_cfg
                for tier_name, tier_cfg in dict(fallback_models.get("tiers", {}) or {}).items()
                if str(tier_cfg.get("mode", "local")).lower() == "remote"
            }
            tiers_cut_over = all(
                tier_cfg.get("effective_backend") == "openai_compatible" and tier_cfg.get("effective_model") == GPT54_MODEL
                for tier_cfg in remote_tiers.values()
            )
            direct_routes = ["chat_fast", "renderer_fallback_fast", "renderer_fallback_small", "monologue_stream", "autonomy_self_run"]
            routes_cut_over = all(
                dict(fallback_models.get("routes", {}) or {}).get(route_name, {}).get("effective_backend") == "openai_compatible"
                and dict(fallback_models.get("routes", {}) or {}).get(route_name, {}).get("effective_model") == GPT54_MODEL
                for route_name in direct_routes
            )
            fallback_ok = (
                cutover_status == 200
                and bool(str(cutover_payload.get("assistant", "")).strip())
                and failover_payload.get("active") is True
                and tiers_cut_over
                and routes_cut_over
            )
            fallback.append(
                _probe(
                    "automatic_failover",
                    fallback_ok,
                    f"active={failover_payload.get('active')} tiers={tiers_cut_over} routes={routes_cut_over} assistant_len={len(str(cutover_payload.get('assistant', '')))}",
                )
            )
        else:
            blockers.append(f"{GPT54_API_ENV} is not set, so GPT5.4 fallback validation is blocked")
            fallback.append(_blocked("automatic_failover", f"{GPT54_API_ENV} not present"))

    finally:
        if server is not None and thread is not None:
            _stop_server(server, thread)

    return LiveProbeReport(
        preflight=preflight,
        dialogue=dialogue,
        reset=reset,
        concurrency=concurrency,
        approval=approval,
        fallback=fallback,
        blockers=blockers,
    )


def render_report(structural_results: Iterable[CheckResult], live_report: LiveProbeReport) -> str:
    structural_results = list(structural_results)
    lines = ["# 2026-04-08 Dialogue Runtime Acceptance Report", ""]
    passed = sum(1 for result in structural_results if result.ok)
    lines.append(f"## Structural Checks")
    lines.append("")
    lines.append(f"structural: {passed}/{len(structural_results)} passed")
    for result in structural_results:
        status = "pass" if result.ok else f"fail ({result.returncode})"
        lines.append(f"- {result.command.name}: {status} in {result.seconds:.2f}s")
    lines.append("")
    lines.append("## Live Probe")
    lines.append("")
    for section_name, rows in [
        ("Preflight", live_report.preflight),
        ("Dialogue", live_report.dialogue),
        ("Reset", live_report.reset),
        ("Concurrency", live_report.concurrency),
        ("Approval", live_report.approval),
        ("Fallback", live_report.fallback),
    ]:
        lines.append(f"### {section_name}")
        if not rows:
            lines.append("- no checks recorded")
        else:
            for row in rows:
                lines.append(f"- {row.name}: {row.status} - {row.detail}")
        lines.append("")
    lines.append("## Blockers")
    if live_report.blockers:
        for blocker in live_report.blockers:
            lines.append(f"- {blocker}")
    else:
        lines.append("- none")
    lines.append("")
    live_rows = live_report.preflight + live_report.dialogue + live_report.reset + live_report.concurrency + live_report.approval + live_report.fallback
    live_failures = [row for row in live_rows if row.status != "pass"]
    overall_ok = all(result.ok for result in structural_results) and not live_failures
    lines.append(f"overall: {'pass' if overall_ok else 'fail'}")
    return "\n".join(lines).rstrip() + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the NALR 1.5 dialogue runtime acceptance plan.")
    parser.add_argument("--run", action="store_true", help="execute structural checks and the live probe")
    parser.add_argument("--report-file", default=str(DEFAULT_REPORT_PATH), help="path to write the markdown report when --run is used")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    checks = build_structural_checks()
    if not args.run:
        sys.stdout.write(render_plan(checks))
        return 0
    structural_results = run_structural_checks(checks)
    live_report = run_live_probe()
    report = render_report(structural_results, live_report)
    report_path = Path(args.report_file)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    sys.stdout.write(f"live probe:\n")
    sys.stdout.write(report)
    live_rows = live_report.preflight + live_report.dialogue + live_report.reset + live_report.concurrency + live_report.approval + live_report.fallback
    overall_ok = all(result.ok for result in structural_results) and all(row.status == "pass" for row in live_rows)
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
