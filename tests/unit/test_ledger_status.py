from pathlib import Path


def test_plan_16_ledger_reports_current_root_and_release_sections():
    ledger = (Path(__file__).resolve().parents[2] / "PLAN1.6.md").read_text(encoding="utf-8")

    assert "# NALR 1.6 Plan" in ledger
    assert "## Status" in ledger
    assert "## Delivered" in ledger
    assert "## In Flight" in ledger
    assert "## Deferred" in ledger
    assert "## References" in ledger
    assert "Chat Kernel V2" in ledger
    assert "chat_micro" in ledger
    assert "module_model_bindings" in ledger
