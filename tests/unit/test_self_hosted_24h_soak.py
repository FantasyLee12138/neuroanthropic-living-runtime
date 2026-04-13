from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "scripts" / "self_hosted_24h_soak.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("self_hosted_24h_soak", MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_self_hosted_soak_render_report_marks_pending_real_24h_until_completed(tmp_path):
    module = _load_module()
    report_path = tmp_path / "report.md"

    module.render_report(
        report_path=report_path,
        started_at="2026-04-08T10:00:00+08:00",
        planned_finish_at="2026-04-09T10:00:00+08:00",
        finished_at="2026-04-08T12:00:00+08:00",
        reused_existing_server=False,
        status="PARTIAL",
        status_reason="operator_stopped_before_24h",
        probe_entries=[
            {"probe": "service", "ok": True, "latency_seconds": 0.11},
            {"probe": "service", "ok": True, "latency_seconds": 0.19},
            {"probe": "autonomy", "ok": False, "latency_seconds": 0.21},
        ],
        summary={
            "service_status_reads": 2,
            "autonomy_running_samples": 2,
            "initiative_changes": 1,
            "monologue_changes": 1,
        },
        notes=["真实 24h soak 仍未完成"],
    )

    text = report_path.read_text(encoding="utf-8")

    assert "# Self-Hosted 24h Soak" in text
    assert "Status: `PARTIAL`" in text
    assert "operator_stopped_before_24h" in text
    assert "真实 24h soak 仍未完成" in text
    assert "`service`" in text
