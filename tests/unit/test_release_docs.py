from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_ROOT = REPO_ROOT / "archive" / "backup-docs" / "root"


def test_root_markdown_files_are_consolidated_to_three_active_docs():
    root_markdown = sorted(path.name for path in REPO_ROOT.glob("*.md"))

    assert root_markdown == ["PLAN1.6.md", "README.md", "架构宣言.md"]


def test_release_docs_promote_16_front_door_and_archive_legacy_root_docs():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    plan = (REPO_ROOT / "PLAN1.6.md").read_text(encoding="utf-8")
    architecture_manifesto = (REPO_ROOT / "架构宣言.md").read_text(encoding="utf-8")
    archived_plans = ARCHIVE_ROOT / "PLANS.md"
    archived_tlh_v10 = ARCHIVE_ROOT / "Think_Like_Human(TLH)_v1.0.md"
    archived_tlh_v15 = ARCHIVE_ROOT / "Think_Like_Human(TLH)_v1.5.md"
    archived_bereal = ARCHIVE_ROOT / "BEREAL_v0.6_MASTER_SPEC.md"
    archived_agents_ui = ARCHIVE_ROOT / "Agents_UI.md"
    archived_console_pdf = ARCHIVE_ROOT / "NALR_Alive_Console_Product_Design.pdf"

    assert "# NALR" in readme
    assert "Chat Kernel V2" in readme
    assert "PLAN1.6.md" in readme
    assert "# NALR 1.6 Plan" in plan
    assert "chat_micro" in plan
    assert "module_model_bindings" in plan
    assert "# NALR 1.6 架构宣言" in architecture_manifesto
    assert "稀疏默认、异常加深" in architecture_manifesto

    assert archived_plans.exists()
    assert "# NALR 1.5 Ledger" in archived_plans.read_text(encoding="utf-8")
    assert archived_tlh_v10.exists()
    assert archived_tlh_v15.exists()
    assert archived_bereal.exists()
    assert archived_agents_ui.exists()
    assert archived_console_pdf.exists()


def test_release_docs_include_16_acceptance_and_release_artifacts():
    acceptance_plan = REPO_ROOT / "docs" / "releases" / "1.6-acceptance-plan.md"
    acceptance_matrix = REPO_ROOT / "docs" / "releases" / "1.6-acceptance-matrix.md"
    release_notes = REPO_ROOT / "docs" / "releases" / "1.6-release-notes.md"

    assert acceptance_plan.exists()
    assert "# NALR 1.6 Acceptance Plan" in acceptance_plan.read_text(encoding="utf-8")

    assert acceptance_matrix.exists()
    assert "# NALR 1.6 Acceptance Matrix" in acceptance_matrix.read_text(encoding="utf-8")

    assert release_notes.exists()
    assert "# NALR 1.6 Release Notes" in release_notes.read_text(encoding="utf-8")

    acceptance_plan_text = acceptance_plan.read_text(encoding="utf-8")
    acceptance_matrix_text = acceptance_matrix.read_text(encoding="utf-8")

    assert "PLAN1.6.md" in acceptance_plan_text
    assert "release_frontdoor" in acceptance_plan_text
    assert "release_subjectivity" in acceptance_plan_text
    assert "单一真相源" in acceptance_plan_text
    assert "不完美结构性记忆" in acceptance_plan_text
    assert "历史不可逆负担" in acceptance_plan_text
    assert "release_16.subjectivity.single_truth" in acceptance_plan_text
    assert "release_subjectivity" in acceptance_matrix_text
    assert "history_burden" in acceptance_matrix_text
