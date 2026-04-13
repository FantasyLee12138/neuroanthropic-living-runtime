from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "scripts" / "dialogue_runtime_acceptance.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("dialogue_runtime_acceptance", MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_dialogue_runtime_acceptance_plan_mentions_structural_checks_and_live_probe():
    module = _load_module()

    checks = module.build_structural_checks()
    plan = module.render_plan(checks)

    assert len(checks) >= 3
    assert any("test_observer_api.py" in check.render() for check in checks)
    assert any("test_terminal_bridge.py" in check.render() for check in checks)
    assert "console/talk" in plan
    assert "/web/session/start" in plan
    assert "/persona/reset" in plan
    assert "GPT5.4 fallback" in plan
    assert "models/status" in plan


def test_dialogue_runtime_acceptance_run_mode_reports_structural_and_live_results():
    module = _load_module()
    calls: list[tuple[tuple[str, ...], Path | None, dict[str, str] | None]] = []

    def fake_run(argv, cwd=None, env=None, check=None):
        calls.append((tuple(argv), cwd, env))

        class Result:
            def __init__(self, returncode: int):
                self.returncode = returncode

        if any("test_terminal_bridge.py" in str(part) for part in argv):
            return Result(1)
        return Result(0)

    def fake_live_probe():
        return module.LiveProbeReport(
            preflight=[
                module.ProbeResult(name="service_status", status="pass", detail="service healthy"),
            ],
            dialogue=[
                module.ProbeResult(name="console_identity", status="pass", detail="identity reply available"),
                module.ProbeResult(name="web_identity", status="pass", detail="web stream delivered final event"),
            ],
            reset=[
                module.ProbeResult(name="persona_reset", status="pass", detail="histories cleared"),
            ],
            concurrency=[
                module.ProbeResult(name="multi_session", status="pass", detail="sessions remained isolated"),
            ],
            approval=[
                module.ProbeResult(name="approval_round_trip", status="pass", detail="approval request resolved"),
            ],
            fallback=[
                module.ProbeResult(name="automatic_failover", status="blocked", detail="not implemented"),
            ],
            blockers=["automatic global failover is not implemented"],
        )

    module.subprocess = types.SimpleNamespace(run=fake_run)
    module.run_live_probe = fake_live_probe
    buffer = io.StringIO()

    with contextlib.redirect_stdout(buffer):
        exit_code = module.main(["--run"])

    output = buffer.getvalue()
    assert exit_code == 1
    assert "structural: " in output
    assert "live probe:" in output
    assert "automatic global failover is not implemented" in output
    assert any(call[2]["PYTHONPATH"] == module.PYTHONPATH for call in calls)
