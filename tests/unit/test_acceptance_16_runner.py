from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "scripts" / "acceptance_16.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("acceptance_16", MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_acceptance_16_runner_plan_mentions_subjectivity_and_self_hosted_checks():
    module = _load_module()

    checks = module.build_checks()
    plan = module.render_plan(checks)

    assert [check.group for check in checks[:5]] == ["subjectivity"] * 5
    assert [check.group for check in checks[5:]] == ["self-hosted"] * 2
    assert "tests/unit/test_release_acceptance_report.py" in plan
    assert "tests/unit/test_plan16_runtime.py" in plan
    assert "tests/unit/test_longrun_acceptance.py" in plan
    assert "tests/unit/test_run_runtime_snapshot_retention.py" in plan
    assert "tests/unit/test_acceptance_16_runner.py" in plan
    assert "tests/integration/test_observer_launcher.py" in plan
    assert "--emit-report" in plan
    assert checks[0].argv[0] == module.PYTHON_BIN


def test_acceptance_16_runner_emit_report_seeds_bounded_subjectivity_pass(tmp_path):
    module = _load_module()

    payload = module.build_subjectivity_probe(tmp_path / "probe-runtime", window=3)

    assert payload["acceptance_report"]["release_16"]["release_subjectivity"]["status"] == "pass"
    assert payload["acceptance_report"]["release_16"]["subjectivity"]["memory"]["status"] == "pass"
    assert payload["acceptance_report"]["release_16"]["subjectivity"]["recovery"]["status"] == "pass"
    assert payload["checkpoint_id"].startswith("ckpt-")
    assert len(payload["seeded_round_ids"]) >= 3


def test_acceptance_16_runner_run_mode_reports_probe_failure_and_uses_pythonpath():
    module = _load_module()
    calls: list[tuple[tuple[str, ...], Path | None, dict[str, str] | None]] = []

    def fake_run(argv, cwd=None, env=None, check=None):
        calls.append((tuple(argv), cwd, env))

        class Result:
            def __init__(self, returncode: int):
                self.returncode = returncode

        if "--emit-report" in argv:
            return Result(1)
        return Result(0)

    module.subprocess = types.SimpleNamespace(run=fake_run)
    buffer = io.StringIO()

    with contextlib.redirect_stdout(buffer):
        exit_code = module.main(["--run", "--group", "self-hosted"])

    output = buffer.getvalue()
    assert exit_code == 1
    assert "self-hosted: 1/2 passed" in output
    assert "failed commands:" in output
    assert len(calls) == 2
    assert all(call[2]["PYTHONPATH"] == module.PYTHONPATH for call in calls)


def test_acceptance_16_runner_emit_report_cli_prints_json(tmp_path):
    module = _load_module()
    buffer = io.StringIO()

    with contextlib.redirect_stdout(buffer):
        exit_code = module.main(["--emit-report", "--runtime-root", str(tmp_path / "emit-runtime"), "--window", "3"])

    payload = json.loads(buffer.getvalue())
    assert exit_code == 0
    assert payload["acceptance_report"]["release_16"]["release_subjectivity"]["status"] == "pass"
    assert payload["runtime_root"] == str(tmp_path / "emit-runtime")
