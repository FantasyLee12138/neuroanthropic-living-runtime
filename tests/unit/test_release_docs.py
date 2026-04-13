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
    self_hosted_soak = (REPO_ROOT / "scripts" / "self_hosted_24h_soak.py")

    assert "# NALR 1.5 Front Door" in readme
    assert "apps/remote-terminal" in readme
    assert "Remote Terminal" in readme
    assert "# NALR 1.5 Ledger" in plans
    assert "## Delivered in 1.5" in plans
    assert "## In Flight Before 1.5 Freeze" in plans
    assert "## Deferred After 1.5" in plans
    assert "FeedbackLoopRuntime" in tlh_v15.read_text(encoding="utf-8")
    assert "FeedbackLoopRuntime" in architecture_manifesto.read_text(encoding="utf-8")
    assert "Feedback loop runtime" in plans
    assert "Layered controls and runtime baseline" in plans
    assert "Remote Terminal stateless relay client" in plans
    assert "Canonical read-model convergence" in plans
    assert "Frozen route taxonomy" in plans
    assert "24h soak remains pending" in plans

    acceptance_matrix_text = acceptance_matrix.read_text(encoding="utf-8")
    assert "Feedback loop runtime" in acceptance_matrix_text
    assert "Remote Terminal stateless relay client" in acceptance_matrix_text
    assert "scripts/self_hosted_24h_soak.py" in acceptance_matrix_text
    assert "| partial |" in acceptance_matrix_text

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
    self_hosted_text = self_hosted.read_text(encoding="utf-8")
    assert "# NALR 1.5 Self-Hosted Deployment" in self_hosted_text
    assert "scripts/self_hosted_24h_soak.py" in self_hosted_text
    assert "真实 24h soak 仍未完成" in self_hosted_text
    assert "/controls/current" in self_hosted_text

    assert self_hosted_soak.exists()
