from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "scripts" / "acceptance_15.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("acceptance_15", MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_acceptance_runner_plan_matches_documented_key_checks():
    module = _load_module()

    checks = module.build_checks()
    plan = module.render_plan(checks)

    assert [check.group for check in checks[:8]] == ["local-client"] * 8
    assert [check.group for check in checks[8:10]] == ["observer"] * 2
    assert [check.group for check in checks[10:13]] == ["runtime"] * 3
    assert [check.group for check in checks[13:18]] == ["controlled-learning"] * 5
    assert [check.group for check in checks[18:]] == ["self-hosted"] * 3
    assert "zsh" in plan
    assert "Open_NALR_Workbench.command" in plan
    assert "--help" in plan
    assert "tests/unit/test_local_client_packager.py" in plan
    assert "local_client_packager.py" in plan
    assert "cd apps/terminal && npm test" in plan
    assert "tests/integration/test_observer_api.py" in plan
    assert "tests/unit/test_tlh_runtime.py" in plan
    assert "tests/integration/test_observer_learning_settings.py" in plan
    assert "tests/unit/test_release_acceptance_report.py" in plan
    assert "tests/unit/test_terminal_bridge.py" in plan
    assert "tests/integration/test_terminal_bridge_stdio.py" in plan
    assert "eval acceptance-report --window 12" in plan
    assert "eval longrun 48" in plan
    assert checks[0].argv[0] == "zsh"
    assert checks[1].argv[0] == module.PYTHON_BIN
    assert checks[2].argv[0] == module.PYTHON_BIN


def test_acceptance_runner_run_mode_reports_failures_and_passes_pythonpath():
    module = _load_module()
    calls: list[tuple[tuple[str, ...], Path | None, dict[str, str] | None]] = []

    def fake_run(argv, cwd=None, env=None, check=None):
        calls.append((tuple(argv), cwd, env))

        class Result:
            def __init__(self, returncode: int):
                self.returncode = returncode

        if any(str(part).endswith("test_release_docs.py") for part in argv):
            return Result(1)
        return Result(0)

    module.subprocess = types.SimpleNamespace(run=fake_run)
    buffer = io.StringIO()

    with contextlib.redirect_stdout(buffer):
        exit_code = module.main(["--run", "--group", "controlled-learning"])

    output = buffer.getvalue()
    assert exit_code == 1
    assert "controlled-learning: 4/5 passed" in output
    assert "failed commands:" in output
    assert len(calls) == 5
    assert all(call[2]["PYTHONPATH"] == module.PYTHONPATH for call in calls)
    assert all(call[0][0] == module.PYTHON_BIN for call in calls)
