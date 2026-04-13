from pathlib import Path


def test_plans_ledger_reports_15_release_sections():
    ledger = (Path(__file__).resolve().parents[2] / "PLANS.md").read_text(encoding="utf-8")

    assert "# NALR 1.5 Ledger" in ledger
    assert "## Delivered in 1.5" in ledger
    assert "## In Flight Before 1.5 Freeze" in ledger
    assert "## Deferred After 1.5" in ledger
    assert "## Verification Evidence" in ledger
