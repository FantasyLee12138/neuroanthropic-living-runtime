#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "http://127.0.0.1:8765"
DEFAULT_DOC_PATH = REPO_ROOT / "docs" / "testing" / "2026-04-08-workbench-6h-soak.md"
DEFAULT_LOG_PATH = REPO_ROOT / ".alive" / "logs" / "observer-workbench-6h-soak.jsonl"
DEFAULT_SERVER_LOG_PATH = REPO_ROOT / ".alive" / "logs" / "observer-workbench-6h-server.log"
OBSERVER_MAIN_SESSION_ID = "observer-main"
CHAT_MESSAGES = [
    "请用一句话说明你当前最强的行动倾向。",
    "现在优先驱动你的内部因素是什么？",
    "如果你选择不说话，原因会是什么？",
    "请简短描述你当前的主体状态变化。",
    "现在最影响你行动的是记忆、身体还是关系？",
    "如果继续自主运行，你下一步更可能做什么？",
]


def now_local() -> datetime:
    return datetime.now().astimezone()


def now_iso() -> str:
    return now_local().isoformat(timespec="seconds")


def request_json(path: str, *, method: str = "GET", payload: dict[str, Any] | None = None, timeout: float = 15.0) -> tuple[int, float, dict[str, Any]]:
    started = time.perf_counter()
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=body,
        method=method,
        headers={"Content-Type": "application/json"} if payload is not None else {},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        elapsed = time.perf_counter() - started
        raw = response.read().decode("utf-8")
        data = json.loads(raw) if raw else {}
        return int(response.status), elapsed, data


def request_text(path: str, *, timeout: float = 10.0) -> tuple[int, float, str]:
    started = time.perf_counter()
    with urllib.request.urlopen(f"{BASE_URL}{path}", timeout=timeout) as response:
        elapsed = time.perf_counter() - started
        return int(response.status), elapsed, response.read().decode("utf-8", errors="replace")


def current_transcript_tail(session_id: str, *, timeout: float = 10.0) -> tuple[int, list[dict[str, Any]]]:
    _, _, payload = request_json(
        f"/web/session/state?session_id={urllib.parse.quote(session_id)}",
        timeout=timeout,
    )
    transcript = list(((payload.get("session") or {}).get("transcript_lines") or []))
    return len(transcript), transcript[-3:]


def wait_until_ready(timeout_seconds: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            status, _, _ = request_text("/dashboard", timeout=1.5)
            if status == 200:
                return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(0.25)
    if last_error is not None:
        raise RuntimeError(f"observer workbench did not become ready: {last_error}") from last_error
    raise RuntimeError("observer workbench did not become ready before timeout")


def server_is_ready() -> bool:
    try:
        status, _, _ = request_text("/dashboard", timeout=1.5)
        return status == 200
    except Exception:  # noqa: BLE001
        return False


def start_workbench(server_log_path: Path) -> tuple[subprocess.Popen[str] | None, bool]:
    if server_is_ready():
        return None, True
    server_log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = server_log_path.open("a", encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT}:{REPO_ROOT / 'src'}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else f"{REPO_ROOT}:{REPO_ROOT / 'src'}"
    process = subprocess.Popen(
        [str(REPO_ROOT / "Open_NALR_Workbench.command"), "--no-browser"],
        cwd=REPO_ROOT,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        preexec_fn=os.setsid,
        env=env,
    )
    wait_until_ready(timeout_seconds=45.0)
    return process, False


def stop_workbench(process: subprocess.Popen[str] | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * p) - 1))
    return ordered[index]


def summarize_probes(entries: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [float(item["latency_seconds"]) for item in entries if item.get("ok")]
    ok_count = sum(1 for item in entries if item.get("ok"))
    fail_count = len(entries) - ok_count
    return {
        "total": len(entries),
        "ok": ok_count,
        "fail": fail_count,
        "avg": round(statistics.fmean(latencies), 3) if latencies else None,
        "p95": round(percentile(latencies, 0.95), 3) if latencies else None,
        "max": round(max(latencies), 3) if latencies else None,
    }


def append_jsonl(log_path: Path, record: dict[str, Any]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def render_doc(
    *,
    doc_path: Path,
    started_at: str,
    planned_finish_at: str,
    status: str,
    reused_existing_server: bool,
    probe_entries: list[dict[str, Any]],
    dialogue_entries: list[dict[str, Any]],
    notes: list[str],
) -> None:
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in probe_entries:
        grouped.setdefault(str(item["probe"]), []).append(item)
    probe_lines: list[str] = []
    for probe_name in sorted(grouped):
        summary = summarize_probes(grouped[probe_name])
        probe_lines.append(
            f"- `{probe_name}`: total={summary['total']}, ok={summary['ok']}, fail={summary['fail']}, "
            f"avg={summary['avg']}, p95={summary['p95']}, max={summary['max']}"
        )
    dialogue_summary = summarize_probes(dialogue_entries)
    last_dialogue = dialogue_entries[-1] if dialogue_entries else None
    status_note = "RUNNING" if status == "RUNNING" else status
    note_lines = "\n".join(f"- {item}" for item in notes) if notes else "- 无"
    dialogue_preview = ""
    if last_dialogue and isinstance(last_dialogue.get("response_excerpt"), str):
        dialogue_preview = last_dialogue["response_excerpt"]
    content = f"""# Workbench 6h Soak

## Run Meta

- Started at: `{started_at}`
- Planned finish: `{planned_finish_at}`
- Status: `{status_note}`
- Launch mode: `Open_NALR_Workbench.command --no-browser`
- Reused existing server on `127.0.0.1:8765`: `{reused_existing_server}`

## Scope

- Workbench direct startup
- Continuous observation of `dashboard`, `autonomy/status`, `console/state`, `web/session/state`
- Periodic dialogue via `/web/session/event` on `observer-main`
- Pre-soak code fix: heavy observer request paths are offloaded off the event loop and serialized under a runtime lock

## Pre-Soak Verification

- Added concurrency regressions for `/console/talk`, `/web/session/event`, `/console/endogenous/tick`
- Focused observer regressions passed locally after the fix
- Residual note: manual `/console/endogenous/tick` is still slow in live mode and is tracked as an explicit risk
- Residual note: `observer-main` dialogue completion can lag; this soak gates on workbench availability and user-turn acceptance, and only records settle when it is observed

## Probe Summary

{chr(10).join(probe_lines) if probe_lines else "- 暂无采样"}

## Dialogue Summary

- total={dialogue_summary['total']}, ok={dialogue_summary['ok']}, fail={dialogue_summary['fail']}, avg={dialogue_summary['avg']}, p95={dialogue_summary['p95']}, max={dialogue_summary['max']}
- last response excerpt: `{dialogue_preview}`

## Notes

{note_lines}
"""
    doc_path.write_text(content, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a six-hour observer workbench soak and write a markdown report.")
    parser.add_argument("--duration-seconds", type=int, default=6 * 60 * 60)
    parser.add_argument("--probe-interval-seconds", type=int, default=60)
    parser.add_argument("--dialogue-interval-seconds", type=int, default=20 * 60)
    parser.add_argument("--doc-path", type=Path, default=DEFAULT_DOC_PATH)
    parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG_PATH)
    parser.add_argument("--server-log-path", type=Path, default=DEFAULT_SERVER_LOG_PATH)
    parser.add_argument("--dialogue-settle-timeout-seconds", type=int, default=45)
    parser.add_argument("--require-dialogue-settle", action="store_true")
    args = parser.parse_args()

    started_at = now_iso()
    planned_finish_at = datetime.fromtimestamp(time.time() + args.duration_seconds, tz=timezone.utc).astimezone().isoformat(timespec="seconds")
    workbench_process, reused_existing_server = start_workbench(args.server_log_path)
    probe_entries: list[dict[str, Any]] = []
    dialogue_entries: list[dict[str, Any]] = []
    notes: list[str] = []

    try:
        status_code, latency, autonomy_payload = request_json("/autonomy/status", timeout=20.0)
        already_running = bool((autonomy_payload or {}).get("running"))
        append_jsonl(
            args.log_path,
            {
                "ts": now_iso(),
                "type": "autonomy_status_precheck",
                "status_code": status_code,
                "latency_seconds": round(latency, 3),
                "autonomy_running": already_running,
            },
        )
        if already_running:
            notes.append(f"autonomy/status precheck status={status_code}, latency={latency:.3f}s, already_running=True")
        else:
            start_status, start_latency, runtime_payload = request_json("/web/runtime/start", method="POST", payload={}, timeout=60.0)
            append_jsonl(
                args.log_path,
                {
                    "ts": now_iso(),
                    "type": "runtime_start",
                    "status_code": start_status,
                    "latency_seconds": round(start_latency, 3),
                    "autonomy_running": runtime_payload.get("autonomy", {}).get("running"),
                },
            )
            notes.append(
                f"runtime/start status={start_status}, latency={start_latency:.3f}s, autonomy_running={runtime_payload.get('autonomy', {}).get('running')}"
            )
    except Exception as exc:  # noqa: BLE001
        notes.append(f"runtime bootstrap check failed before soak loop: {type(exc).__name__}: {exc}")

    render_doc(
        doc_path=args.doc_path,
        started_at=started_at,
        planned_finish_at=planned_finish_at,
        status="RUNNING",
        reused_existing_server=reused_existing_server,
        probe_entries=probe_entries,
        dialogue_entries=dialogue_entries,
        notes=notes,
    )

    deadline = time.monotonic() + args.duration_seconds
    next_dialogue_at = time.monotonic()
    message_index = 0

    while time.monotonic() < deadline:
        loop_started = time.monotonic()
        probes = [
            ("dashboard", "text", "/dashboard"),
            ("autonomy_status", "json", "/autonomy/status"),
            ("console_state", "json", "/console/state"),
            ("observer_main_state", "json", f"/web/session/state?session_id={urllib.parse.quote(OBSERVER_MAIN_SESSION_ID)}"),
        ]
        for probe_name, probe_kind, path in probes:
            record: dict[str, Any] = {"ts": now_iso(), "type": "probe", "probe": probe_name, "path": path}
            try:
                if probe_kind == "text":
                    status, latency, body = request_text(path, timeout=8.0)
                    record.update(
                        {
                            "status_code": status,
                            "latency_seconds": round(latency, 3),
                            "ok": status == 200,
                            "body_excerpt": body[:120],
                        }
                    )
                else:
                    status, latency, payload = request_json(path, timeout=8.0)
                    record.update(
                        {
                            "status_code": status,
                            "latency_seconds": round(latency, 3),
                            "ok": status == 200,
                            "payload_excerpt": json.dumps(payload, ensure_ascii=False)[:160],
                        }
                    )
            except Exception as exc:  # noqa: BLE001
                record.update(
                    {
                        "status_code": None,
                        "latency_seconds": None,
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            probe_entries.append(record)
            append_jsonl(args.log_path, record)

        if time.monotonic() >= next_dialogue_at:
            message = CHAT_MESSAGES[message_index % len(CHAT_MESSAGES)]
            message_index += 1
            record = {"ts": now_iso(), "type": "dialogue", "message": message}
            try:
                before_count, before_tail = current_transcript_tail(OBSERVER_MAIN_SESSION_ID, timeout=10.0)
                previous_last_assistant = ""
                previous_assistant_lines = [item for item in before_tail if str(item.get("kind") or "") == "assistant" and str(item.get("text") or "").strip()]
                if previous_assistant_lines:
                    previous_last_assistant = str(previous_assistant_lines[-1].get("text") or "")
                status, latency, payload = request_json(
                    "/web/session/event",
                    method="POST",
                    payload={"type": "user_turn", "session_id": OBSERVER_MAIN_SESSION_ID, "text": message},
                    timeout=10.0,
                )
                assistant_excerpt = ""
                settled = False
                settle_deadline = time.monotonic() + args.dialogue_settle_timeout_seconds
                while time.monotonic() < settle_deadline:
                    transcript_count, tail = current_transcript_tail(OBSERVER_MAIN_SESSION_ID, timeout=10.0)
                    if transcript_count >= before_count + 2 and tail:
                        last_line = tail[-1]
                        if (
                            str(last_line.get("kind") or "") == "assistant"
                            and str(last_line.get("text") or "").strip()
                            and str(last_line.get("text") or "") != previous_last_assistant
                        ):
                            assistant_excerpt = str(last_line.get("text") or "")[:160]
                            settled = True
                            break
                    time.sleep(1.0)
                record.update(
                    {
                        "probe": "dialogue",
                        "status_code": status,
                        "latency_seconds": round(latency, 3),
                        "ok": status == 200 and payload.get("accepted") is True and (settled if args.require_dialogue_settle else True),
                        "response_excerpt": assistant_excerpt or json.dumps(payload, ensure_ascii=False)[:160],
                        "accepted": bool(payload.get("accepted") is True),
                        "settled": settled,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                record.update(
                    {
                        "probe": "dialogue",
                        "status_code": None,
                        "latency_seconds": None,
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "response_excerpt": "",
                    }
                )
            dialogue_entries.append(record)
            append_jsonl(args.log_path, record)
            next_dialogue_at = time.monotonic() + args.dialogue_interval_seconds

        render_doc(
            doc_path=args.doc_path,
            started_at=started_at,
            planned_finish_at=planned_finish_at,
            status="RUNNING",
            reused_existing_server=reused_existing_server,
            probe_entries=probe_entries,
            dialogue_entries=dialogue_entries,
            notes=notes,
        )

        sleep_seconds = max(0.0, args.probe_interval_seconds - (time.monotonic() - loop_started))
        time.sleep(sleep_seconds)

    failures = [item for item in probe_entries if not item.get("ok")] + [item for item in dialogue_entries if not item.get("ok")]
    notes.append(f"completed at {now_iso()} with {len(failures)} failed probes/dialogues across {len(probe_entries)} probes and {len(dialogue_entries)} dialogues")
    final_status = "PASS" if not failures and dialogue_entries else "FAIL"
    render_doc(
        doc_path=args.doc_path,
        started_at=started_at,
        planned_finish_at=planned_finish_at,
        status=final_status,
        reused_existing_server=reused_existing_server,
        probe_entries=probe_entries,
        dialogue_entries=dialogue_entries,
        notes=notes,
    )
    stop_workbench(workbench_process)
    return 0 if final_status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
