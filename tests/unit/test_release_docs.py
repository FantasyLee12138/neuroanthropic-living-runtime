from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_release_docs_promote_15_front_door_and_acceptance_assets():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    plans = (REPO_ROOT / "PLANS.md").read_text(encoding="utf-8")
    tlh_v15 = (REPO_ROOT / "Think_Like_Human(TLH)_v1.5.md")
    tlh_compat = (REPO_ROOT / "Think_Like_Human(TLH)_v1.0.md")
    architecture_manifesto = (REPO_ROOT / "架构宣言.md")
    acceptance_plan = (REPO_ROOT / "docs" / "releases" / "1.5-acceptance-plan.md")
    acceptance_matrix = (REPO_ROOT / "docs" / "releases" / "1.5-acceptance-matrix.md")
    local_client = (REPO_ROOT / "docs" / "deployment" / "1.5-local-client.md")
    self_hosted = (REPO_ROOT / "docs" / "deployment" / "1.5-self-hosted.md")

    assert "# NALR 1.5 Front Door" in readme
    assert "# NALR 1.5 Ledger" in plans
    assert "## Delivered in 1.5" in plans
    assert "## In Flight Before 1.5 Freeze" in plans
    assert "## Deferred After 1.5" in plans

    assert tlh_v15.exists()
    assert "# Think Like Human (TLH) v1.5" in tlh_v15.read_text(encoding="utf-8")

    compat_text = tlh_compat.read_text(encoding="utf-8")
    assert "兼容入口" in compat_text
    assert "Think_Like_Human(TLH)_v1.5.md" in compat_text

    assert architecture_manifesto.exists()
    assert "# NALR 1.5 架构宣言" in architecture_manifesto.read_text(encoding="utf-8")

    assert acceptance_plan.exists()
    assert "# NALR 1.5 Acceptance Plan" in acceptance_plan.read_text(encoding="utf-8")

    assert acceptance_matrix.exists()
    assert "# NALR 1.5 Acceptance Matrix" in acceptance_matrix.read_text(encoding="utf-8")

    assert local_client.exists()
    assert "# NALR 1.5 Local Client Deployment" in local_client.read_text(encoding="utf-8")

    assert self_hosted.exists()
    assert "# NALR 1.5 Self-Hosted Deployment" in self_hosted.read_text(encoding="utf-8")
