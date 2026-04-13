#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from uuid import uuid4


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
PYTHONPATH = os.pathsep.join([str(REPO_ROOT), str(SRC_ROOT)])

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from nalr.runtime.controller import RuntimeController
from nalr.schemas.models import RoundEvent


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
    group: str
    name: str
    argv: tuple[str, ...]
    cwd: Path = REPO_ROOT
    env: dict[str, str] | None = None

    def render(self) -> str:
        command = shlex.join(self.argv)
        if self.env:
            env_prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in self.env.items())
            command = f"{env_prefix} {command}"
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


def _python_env() -> dict[str, str]:
    return {"PYTHONPATH": PYTHONPATH}


def _isolated_runtime_root() -> Path:
    root = REPO_ROOT / ".tmp_acceptance_16"
    root.mkdir(parents=True, exist_ok=True)
    return root


def build_subjectivity_probe(runtime_root: Path, *, window: int = 3) -> dict[str, object]:
    runtime_root = Path(runtime_root)
    if runtime_root.name.startswith(".tmp_acceptance_16") or runtime_root.parent == _isolated_runtime_root():
        shutil.rmtree(runtime_root, ignore_errors=True)
    runtime_root.mkdir(parents=True, exist_ok=True)
    controller = RuntimeController(project_root=runtime_root, config_root=REPO_ROOT / "config")
    state = controller.load_runtime_state()
    probe_id = uuid4().hex[:6]
    cue_alpha = f"alpha{probe_id}"
    cue_alphb = f"alphb{probe_id}"

    controller.memory_store.ingest_event(
        RoundEvent(source="user", content=f"remember {cue_alpha} for later", target="user", cue=cue_alpha, valence=0.1),
        round_id=1,
        session_id=state.session_id,
    )
    hot = list(controller.memory_store._read_list(controller.memory_store.episodic_path))
    alpha = next(item for item in hot if item["cue"] == cue_alpha)
    alpha["detail_strength"] = 0.30
    alpha["gist_strength"] = 0.32
    controller.memory_store._write_list(controller.memory_store.episodic_path, [alpha])

    pressured_state = controller.load_runtime_state()
    pressured_state.budget_remaining = 0.0
    pressured_state.resource_state = {"queue_depth": 999}
    controller._save_state(pressured_state, sync=True)
    miss_round = controller.tick(
        RoundEvent(
            source="user",
            content=f"remember {cue_alphb} under pressure",
            target="user",
            cue=cue_alphb,
            valence=0.0,
            cue_quality=0.0,
        ),
        scenario="chat",
        mode="interactive",
    )

    relaxed_state = controller.load_runtime_state()
    relaxed_state.budget_remaining = 1.0
    relaxed_state.resource_state = {}
    controller._save_state(relaxed_state, sync=True)
    rescue_round = controller.tick(
        RoundEvent(
            source="user",
            content=f"remember {cue_alphb} strongly",
            target="user",
            cue=cue_alphb,
            valence=0.0,
            cue_quality=1.0,
        ),
        scenario="chat",
        mode="interactive",
    )

    checkpoint = controller.checkpoint()
    continuity_round = controller.tick(
        RoundEvent(
            source="user",
            content=f"after checkpoint, recall {cue_alphb}",
            target="user",
            cue=cue_alphb,
            valence=0.0,
            cue_quality=1.0,
        ),
        scenario="chat",
        mode="interactive",
    )
    stabilization_round_ids: list[int] = []
    for suffix in ("stabilize-one", "stabilize-two"):
        stable_round = controller.tick(
            RoundEvent(
                source="user",
                content=f"{suffix}: keep continuity around {cue_alphb}",
                target="user",
                cue=cue_alphb,
                valence=0.0,
                cue_quality=1.0,
            ),
            scenario="chat",
            mode="interactive",
        )
        stabilization_round_ids.append(int(stable_round.round_id))
    pre_longrun_round = int(controller.load_runtime_state().round_count or 0)
    longrun_summary = controller.eval_longrun(8)
    post_longrun_round = int(controller.load_runtime_state().round_count or 0)
    longrun_sample_ids = list(range(pre_longrun_round + 1, post_longrun_round + 1))
    window = max(int(window), 16)
    acceptance_report = controller.acceptance_report(window=window)
    return {
        "runtime_root": str(runtime_root),
        "checkpoint_id": checkpoint.checkpoint_id,
        "seeded_round_ids": [
            int(miss_round.round_id),
            int(rescue_round.round_id),
            int(continuity_round.round_id),
            *stabilization_round_ids,
            *[round_id for round_id in longrun_sample_ids if round_id > 0],
        ],
        "longrun_summary": longrun_summary,
        "acceptance_report": acceptance_report,
    }


def build_checks() -> list[CheckCommand]:
    python_env = _python_env()
    probe_root = _isolated_runtime_root() / "probe-runtime"
    return [
        CheckCommand("subjectivity", "Release acceptance report unit tests", (PYTHON_BIN, "-m", "pytest", "tests/unit/test_release_acceptance_report.py", "-q"), env=python_env),
        CheckCommand("subjectivity", "Plan 1.6 runtime unit tests", (PYTHON_BIN, "-m", "pytest", "tests/unit/test_plan16_runtime.py", "-q"), env=python_env),
        CheckCommand("subjectivity", "Long-run acceptance unit tests", (PYTHON_BIN, "-m", "pytest", "tests/unit/test_longrun_acceptance.py", "-q"), env=python_env),
        CheckCommand("subjectivity", "Snapshot retention unit tests", (PYTHON_BIN, "-m", "pytest", "tests/unit/test_run_runtime_snapshot_retention.py", "-q"), env=python_env),
        CheckCommand("subjectivity", "Acceptance 1.6 runner unit tests", (PYTHON_BIN, "-m", "pytest", "tests/unit/test_acceptance_16_runner.py", "-q"), env=python_env),
        CheckCommand("self-hosted", "Observer launcher integration", (PYTHON_BIN, "-m", "pytest", "tests/integration/test_observer_launcher.py", "-q"), env=python_env),
        CheckCommand(
            "self-hosted",
            "Bounded 1.6 subjectivity probe",
            (PYTHON_BIN, str(REPO_ROOT / "scripts" / "acceptance_16.py"), "--emit-report", "--runtime-root", str(probe_root), "--window", "16"),
            env=python_env,
        ),
    ]


def group_names(checks: Iterable[CheckCommand]) -> list[str]:
    names: list[str] = []
    for check in checks:
        if check.group not in names:
            names.append(check.group)
    return names


def selected_checks(checks: Iterable[CheckCommand], groups: set[str] | None) -> list[CheckCommand]:
    if not groups:
        return list(checks)
    return [check for check in checks if check.group in groups]


def render_plan(checks: Iterable[CheckCommand]) -> str:
    checks = list(checks)
    lines = ["NALR 1.6 acceptance command plan", ""]
    for group in group_names(checks):
        lines.append(f"{group}:")
        for check in [item for item in checks if item.group == group]:
            lines.append(f"- {check.name}: `{check.render()}`")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _merged_env(extra: dict[str, str] | None) -> dict[str, str]:
    env = os.environ.copy()
    if extra:
        env.update(extra)
    return env


def run_checks(checks: Iterable[CheckCommand]) -> list[CheckResult]:
    results: list[CheckResult] = []
    for check in checks:
        print(f"[{check.group}] {check.name}")
        started = time.perf_counter()
        completed = subprocess.run(check.argv, cwd=check.cwd, env=_merged_env(check.env), check=False)
        elapsed = round(time.perf_counter() - started, 2)
        result = CheckResult(command=check, returncode=int(completed.returncode), seconds=elapsed)
        status = "ok" if result.ok else f"fail ({result.returncode})"
        print(f"[{check.group}] {check.name}: {status} in {elapsed:.2f}s")
        results.append(result)
    return results


def render_report(results: Iterable[CheckResult]) -> str:
    results = list(results)
    lines = ["NALR 1.6 acceptance report", ""]
    by_group: dict[str, list[CheckResult]] = {}
    for result in results:
        by_group.setdefault(result.command.group, []).append(result)
    for group in group_names(result.command for result in results):
        group_results = by_group[group]
        passed = sum(1 for item in group_results if item.ok)
        lines.append(f"{group}: {passed}/{len(group_results)} passed")
        for item in group_results:
            status = "ok" if item.ok else f"fail ({item.returncode})"
            lines.append(f"- {item.command.name}: {status} in {item.seconds:.2f}s")
        lines.append("")
    failures = [item for item in results if not item.ok]
    lines.append(f"overall: {len(results) - len(failures)}/{len(results)} passed")
    if failures:
        lines.append("failed commands:")
        for item in failures:
            lines.append(f"- {item.command.render()}")
    return "\n".join(lines).rstrip() + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or print the NALR 1.6 bounded acceptance command set.")
    parser.add_argument("--group", action="append", choices=["subjectivity", "self-hosted"], help="limit execution to one or more acceptance groups")
    parser.add_argument("--run", action="store_true", help="execute the checks and print a report instead of only emitting the command plan")
    parser.add_argument("--emit-report", action="store_true", help="seed bounded subjectivity evidence on an isolated runtime and print the resulting acceptance report JSON")
    parser.add_argument("--runtime-root", type=Path, default=_isolated_runtime_root() / "emit-runtime", help="runtime root to use with --emit-report")
    parser.add_argument("--window", type=int, default=16, help="acceptance report window for --emit-report")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.emit_report:
        payload = build_subjectivity_probe(args.runtime_root, window=args.window)
        sys.stdout.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        return 0 if payload["acceptance_report"]["release_16"]["release_subjectivity"]["status"] == "pass" else 1
    checks = selected_checks(build_checks(), set(args.group) if args.group else None)
    if not checks:
        print("No acceptance checks matched the requested groups.", file=sys.stderr)
        return 2
    if not args.run:
        sys.stdout.write(render_plan(checks))
        return 0
    results = run_checks(checks)
    sys.stdout.write(render_report(results))
    return 0 if all(result.ok for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
