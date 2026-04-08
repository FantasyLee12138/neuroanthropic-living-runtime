#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
PYTHONPATH = os.pathsep.join([str(REPO_ROOT), str(SRC_ROOT)])


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


def _isolated_runtime_env() -> dict[str, str]:
    runtime_root = REPO_ROOT / ".tmp_acceptance_15"
    runtime_root.mkdir(parents=True, exist_ok=True)
    return {
        "PYTHONPATH": PYTHONPATH,
        "NALR_HOME": str(runtime_root / ".alive"),
        "NALR_CONFIG_DIR": str(REPO_ROOT / "config"),
    }


def build_checks() -> list[CheckCommand]:
    workbench = REPO_ROOT / "Open_NALR_Workbench.command"
    terminal_root = REPO_ROOT / "apps" / "terminal"
    python_env = _python_env()
    isolated_runtime_env = _isolated_runtime_env()
    acceptance_report_cmd = (PYTHON_BIN, "-m", "nalr.cli.app", "eval", "acceptance-report")
    local_client_dist = REPO_ROOT / ".tmp_acceptance_15" / "local-client-dist"
    return [
        CheckCommand("local-client", "Workbench launcher help", ("zsh", str(workbench), "--help")),
        CheckCommand("local-client", "Local client packager unit tests", (PYTHON_BIN, "-m", "pytest", "tests/unit/test_local_client_packager.py", "-q"), env=python_env),
        CheckCommand("local-client", "Local client bundle materialization", (PYTHON_BIN, str(REPO_ROOT / "scripts" / "local_client_packager.py"), str(local_client_dist))),
        CheckCommand("local-client", "Observer launcher lifecycle", (PYTHON_BIN, "-m", "pytest", "tests/integration/test_observer_launcher.py", "-q"), env=python_env),
        CheckCommand("local-client", "CLI integration", (PYTHON_BIN, "-m", "pytest", "tests/integration/test_cli.py", "-q"), env=python_env),
        CheckCommand("local-client", "CIL integration", (PYTHON_BIN, "-m", "pytest", "tests/integration/test_cil_cli.py", "-q"), env=python_env),
        CheckCommand("local-client", "Terminal npm test", ("npm", "test"), cwd=terminal_root),
        CheckCommand("local-client", "Terminal npm build", ("npm", "run", "build"), cwd=terminal_root),
        CheckCommand("observer", "Observer API integration", (PYTHON_BIN, "-m", "pytest", "tests/integration/test_observer_api.py", "-q"), env=python_env),
        CheckCommand("observer", "Observer dashboard smoke", (PYTHON_BIN, "-m", "pytest", "tests/browser/test_observer_dashboard_smoke.py", "-q"), env=python_env),
        CheckCommand("runtime", "TLH runtime unit tests", (PYTHON_BIN, "-m", "pytest", "tests/unit/test_tlh_runtime.py", "-q"), env=python_env),
        CheckCommand("runtime", "Probability field unit tests", (PYTHON_BIN, "-m", "pytest", "tests/unit/test_probability_field.py", "-q"), env=python_env),
        CheckCommand("runtime", "Runtime controller unit tests", (PYTHON_BIN, "-m", "pytest", "tests/unit/test_runtime_controller.py", "-q"), env=python_env),
        CheckCommand("controlled-learning", "Observer learning settings", (PYTHON_BIN, "-m", "pytest", "tests/integration/test_observer_learning_settings.py", "-q"), env=python_env),
        CheckCommand("controlled-learning", "Terminal bridge unit tests", (PYTHON_BIN, "-m", "pytest", "tests/unit/test_terminal_bridge.py", "-q"), env=python_env),
        CheckCommand("controlled-learning", "Terminal bridge stdio integration", (PYTHON_BIN, "-m", "pytest", "tests/integration/test_terminal_bridge_stdio.py", "-q"), env=python_env),
        CheckCommand("controlled-learning", "Release docs acceptance", (PYTHON_BIN, "-m", "pytest", "tests/unit/test_release_docs.py", "-q"), env=python_env),
        CheckCommand("controlled-learning", "Acceptance runner unit tests", (PYTHON_BIN, "-m", "pytest", "tests/unit/test_acceptance_15_runner.py", "-q"), env=python_env),
        CheckCommand("self-hosted", "Release acceptance report unit tests", (PYTHON_BIN, "-m", "pytest", "tests/unit/test_release_acceptance_report.py", "-q"), env=python_env),
        CheckCommand("self-hosted", "Bounded longrun seed", (PYTHON_BIN, "-m", "nalr.cli.app", "eval", "longrun", "48"), env=isolated_runtime_env),
        CheckCommand("self-hosted", "Acceptance report CLI", acceptance_report_cmd + ("--window", "12"), env=isolated_runtime_env),
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
    lines = ["NALR 1.5 acceptance command plan", ""]
    for group in group_names(checks):
        lines.append(f"{group}:")
        for check in [item for item in checks if item.group == group]:
            lines.append(f"- {check.name}: `{check.render()}`")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def run_checks(checks: Iterable[CheckCommand]) -> list[CheckResult]:
    results: list[CheckResult] = []
    for check in checks:
        print(f"[{check.group}] {check.name}")
        started = time.perf_counter()
        completed = subprocess.run(
            check.argv,
            cwd=check.cwd,
            env=_merged_env(check.env),
            check=False,
        )
        elapsed = round(time.perf_counter() - started, 2)
        result = CheckResult(command=check, returncode=int(completed.returncode), seconds=elapsed)
        status = "ok" if result.ok else f"fail ({result.returncode})"
        print(f"[{check.group}] {check.name}: {status} in {elapsed:.2f}s")
        results.append(result)
    return results


def render_report(results: Iterable[CheckResult]) -> str:
    results = list(results)
    lines = ["NALR 1.5 acceptance report", ""]
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


def _merged_env(extra: dict[str, str] | None) -> dict[str, str]:
    env = os.environ.copy()
    if extra:
        env.update(extra)
    return env


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or print the NALR 1.5 acceptance command set.")
    parser.add_argument(
        "--group",
        action="append",
        choices=["local-client", "observer", "runtime", "controlled-learning", "self-hosted"],
        help="limit execution to one or more acceptance groups",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="execute the checks and print a report instead of only emitting the command plan",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
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
