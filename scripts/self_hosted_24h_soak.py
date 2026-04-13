#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_URL = "http://127.0.0.1:8765"
DEFAULT_DOC_PATH = REPO_ROOT / "docs" / "testing" / "2026-04-08-self-hosted-24h-soak.md"
DEFAULT_LOG_PATH = REPO_ROOT / ".alive" / "logs" / "self-hosted-24h-soak.jsonl"
DEFAULT_SERVER_LOG_PATH = REPO_ROOT / ".alive" / "logs" / "self-hosted-24h-server.log"


def now_local() -> datetime:
    return datetime.now().astimezone()


def now_iso() -> str:
    return now_local().isoformat(timespec="seconds")


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * p) - 1))
    return ordered[index]


def request_json(base_url: str, path: str, *, timeout: float = 15.0) -> tuple[int, float, dict[str, Any]]:
    started = time.perf_counter()
    with urllib.request.urlopen(f"{base_url}{path}", timeout=timeout) as response:
        elapsed = time.perf_counter() - started
        raw = response.read().decode("utf-8")
        payload = json.loads(raw) if raw else {}
        return int(response.status), elapsed, payload


def request_text(base_url: str, path: str, *, timeout: float = 15.0) -> tuple[int, float, str]:
    started = time.perf_counter()
    with urllib.request.urlopen(f"{base_url}{path}", timeout=timeout) as response:
        elapsed = time.perf_counter() - started
        return int(response.status), elapsed, response.read().decode("utf-8", errors="replace")


def server_is_ready(base_url: str) -> bool:
    try:
        status, _, _ = request_text(base_url, "/dashboard", timeout=2.0)
        return status == 200
    except Exception:
        return False


def wait_until_ready(base_url: str, *, timeout_seconds: float = 45.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            if server_is_ready(base_url):
                return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
        time.sleep(0.5)
    if last_error is not None:
        raise RuntimeError(f"observer service did not become ready: {last_error}") from last_error
    raise RuntimeError("observer service did not become ready before timeout")


def _alive_observer_argv(base_url: str, command: str) -> list[str]:
    parsed = urllib.parse.urlparse(base_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 8765
    argv = [str(REPO_ROOT / "alive-observer"), command]
    if command == "start":
        argv.extend(["--no-browser", "--host", host, "--port", str(port)])
    return argv


def start_service(base_url: str, server_log_path: Path) -> bool:
    if server_is_ready(base_url):
        return True
    server_log_path.parent.mkdir(parents=True, exist_ok=True)
    with server_log_path.open("a", encoding="utf-8") as handle:
        result = subprocess.run(
            _alive_observer_argv(base_url, "start"),
            cwd=REPO_ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(f"alive-observer start failed with exit code {result.returncode}")
    wait_until_ready(base_url, timeout_seconds=60.0)
    return False


def stop_service(base_url: str) -> None:
    subprocess.run(
        _alive_observer_argv(base_url, "stop"),
        cwd=REPO_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )


def append_jsonl(log_path: Path, record: dict[str, Any]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def summarize_probes(entries: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in entries:
        grouped.setdefault(str(item.get("probe") or "unknown"), []).append(item)
    summary: dict[str, dict[str, Any]] = {}
    for probe, rows in grouped.items():
        latencies = [float(row["latency_seconds"]) for row in rows if row.get("ok")]
        ok_count = sum(1 for row in rows if row.get("ok"))
        summary[probe] = {
            "total": len(rows),
            "ok": ok_count,
            "fail": len(rows) - ok_count,
            "avg": round(statistics.fmean(latencies), 3) if latencies else None,
            "p95": round(percentile(latencies, 0.95), 3) if latencies else None,
            "max": round(max(latencies), 3) if latencies else None,
        }
    return summary


def render_report(
    *,
    report_path: Path,
    started_at: str,
    planned_finish_at: str,
    finished_at: str,
    reused_existing_server: bool,
    status: str,
    status_reason: str,
    probe_entries: list[dict[str, Any]],
    summary: dict[str, Any],
    notes: list[str],
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    probe_summary = summarize_probes(probe_entries)
    probe_lines = []
    for probe_name in sorted(probe_summary):
        item = probe_summary[probe_name]
        probe_lines.append(
            f"- `{probe_name}`: total={item['total']}, ok={item['ok']}, fail={item['fail']}, "
            f"avg={item['avg']}, p95={item['p95']}, max={item['max']}"
        )
    note_lines = "\n".join(f"- {note}" for note in notes) if notes else "- 无"
    content = f"""# Self-Hosted 24h Soak

## Run Meta

- Started at: `{started_at}`
- Planned finish: `{planned_finish_at}`
- Finished at: `{finished_at}`
- Status: `{status}`
- Status reason: `{status_reason}`
- Reused existing service: `{reused_existing_server}`

## Scope

- Probe `/service/status`, `/autonomy/status`, `/state`, `/monologue/status`
- Record JSONL samples under `.alive/logs/`
- Emit a markdown report under `docs/testing/`
- Treat the report as real 24h evidence only when the full duration completes

## Probe Summary

{chr(10).join(probe_lines) if probe_lines else "- 暂无采样"}

## Runtime Summary

- service_status_reads: `{summary.get("service_status_reads", 0)}`
- autonomy_running_samples: `{summary.get("autonomy_running_samples", 0)}`
- initiative_changes: `{summary.get("initiative_changes", 0)}`
- monologue_changes: `{summary.get("monologue_changes", 0)}`
- current_round_changes: `{summary.get("current_round_changes", 0)}`

## Notes

{note_lines}
"""
    report_path.write_text(content, encoding="utf-8")


def _signature(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _sample_probe(base_url: str, path: str, *, timeout: float = 15.0) -> tuple[dict[str, Any], dict[str, Any]]:
    label = path.strip("/").replace("/", "_") or "root"
    try:
        status_code, latency, payload = request_json(base_url, path, timeout=timeout)
        record = {
            "ts": now_iso(),
            "probe": label,
            "path": path,
            "status_code": status_code,
            "latency_seconds": round(latency, 3),
            "ok": status_code == 200,
        }
        return record, payload
    except Exception as exc:  # noqa: BLE001
        record = {
            "ts": now_iso(),
            "probe": label,
            "path": path,
            "status_code": 0,
            "latency_seconds": None,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
        return record, {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a reproducible self-hosted 24h soak and write JSONL + markdown artifacts.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--duration-seconds", type=int, default=24 * 60 * 60)
    parser.add_argument("--probe-interval-seconds", type=int, default=5 * 60)
    parser.add_argument("--doc-path", type=Path, default=DEFAULT_DOC_PATH)
    parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG_PATH)
    parser.add_argument("--server-log-path", type=Path, default=DEFAULT_SERVER_LOG_PATH)
    parser.add_argument("--keep-service-running", action="store_true")
    args = parser.parse_args(argv)

    started_at = now_iso()
    planned_finish_at = datetime.fromtimestamp(
        time.time() + args.duration_seconds,
        tz=timezone.utc,
    ).astimezone().isoformat(timespec="seconds")

    reused_existing_server = start_service(args.base_url, args.server_log_path)
    probe_entries: list[dict[str, Any]] = []
    notes: list[str] = []
    summary = {
        "service_status_reads": 0,
        "autonomy_running_samples": 0,
        "initiative_changes": 0,
        "monologue_changes": 0,
        "current_round_changes": 0,
    }
    previous_initiative = ""
    previous_monologue = ""
    previous_round = None
    status = "PASS"
    status_reason = "completed_full_duration"

    try:
        deadline = time.monotonic() + max(1, args.duration_seconds)
        while time.monotonic() < deadline:
            for path in ("/service/status", "/autonomy/status", "/state", "/monologue/status"):
                record, payload = _sample_probe(args.base_url, path)
                append_jsonl(args.log_path, record)
                probe_entries.append(record)
                if path == "/service/status":
                    summary["service_status_reads"] += 1
                elif path == "/autonomy/status" and payload.get("running"):
                    summary["autonomy_running_samples"] += 1
                elif path == "/state":
                    initiative_signature = _signature(payload.get("initiative", {}))
                    current_round = dict(payload.get("current_round", {}) or {}).get("round_id")
                    if previous_initiative and initiative_signature != previous_initiative:
                        summary["initiative_changes"] += 1
                    if previous_round is not None and current_round != previous_round:
                        summary["current_round_changes"] += 1
                    previous_initiative = initiative_signature
                    previous_round = current_round
                elif path == "/monologue/status":
                    monologue_signature = _signature(payload)
                    if previous_monologue and monologue_signature != previous_monologue:
                        summary["monologue_changes"] += 1
                    previous_monologue = monologue_signature
                if not record.get("ok"):
                    status = "FAIL"
                    status_reason = f"probe_failed:{record['probe']}"
                    notes.append(f"{record['probe']} probe failed")
                    raise RuntimeError(status_reason)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(max(1, args.probe_interval_seconds), max(1.0, remaining)))
    except KeyboardInterrupt:
        status = "PARTIAL"
        status_reason = "interrupted"
        notes.append("Run interrupted before the requested duration completed.")
    except Exception as exc:  # noqa: BLE001
        if status != "FAIL":
            status = "FAIL"
            status_reason = f"{type(exc).__name__}:{exc}"
        notes.append(f"{type(exc).__name__}: {exc}")
    finally:
        finished_at = now_iso()
        if status == "PASS" and args.duration_seconds < 24 * 60 * 60:
            status = "PARTIAL"
            status_reason = "shortened_duration_not_real_24h"
            notes.append("This run did not cover the full 24h duration, so it cannot promote the matrix row to pass.")
        if status != "PASS":
            notes.append("真实 24h soak 仍未完成。")
        render_report(
            report_path=args.doc_path,
            started_at=started_at,
            planned_finish_at=planned_finish_at,
            finished_at=finished_at,
            reused_existing_server=reused_existing_server,
            status=status,
            status_reason=status_reason,
            probe_entries=probe_entries,
            summary=summary,
            notes=notes,
        )
        if not reused_existing_server and not args.keep_service_running:
            stop_service(args.base_url)

    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
