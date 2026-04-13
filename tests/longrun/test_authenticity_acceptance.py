from pathlib import Path

from nalr.runtime.controller import RuntimeController
from nalr.schemas.models import RoundEvent


CONFIG_ROOT = Path(__file__).resolve().parents[2] / "config"


def test_bounded_longrun_acceptance_surfaces_authenticity_and_vitality_evidence(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    for idx in range(12):
        controller.tick(
            RoundEvent(
                source="user",
                content=f"round {idx} 请记住 tea，并根据关系和状态调整回答",
                target="alex" if idx % 2 else "user",
                cue="tea",
                valence=-0.2 if idx % 3 else 0.18,
            ),
            scenario="companion" if idx % 2 else "task",
            mode="interactive",
        )

    controller.tick(
        RoundEvent(source="system", content="idle shaping tick", target="user", cue="tea"),
        scenario="companion",
        mode="idle",
    )
    controller.tick(
        RoundEvent(source="system", content="sleep shaping tick", target="user", cue="tea"),
        scenario="companion",
        mode="sleep",
    )

    metrics = controller.metrics_summary()
    authenticity = controller.authenticity_timeline()
    vitality = controller.vitality_timeline()

    assert metrics["total_rounds"] >= 14
    assert metrics["self_consistency_score"] >= 0.0
    assert metrics["cue_recall_success_rate"] >= 0.0
    assert len(authenticity["points"]) == metrics["total_rounds"]
    assert len(vitality["points"]) == metrics["total_rounds"]
    assert "rename_reason" in authenticity["points"][-1]
    assert "identity_shaping_sources" in authenticity["points"][-1]
    assert "non_interactive_summary" in vitality["points"][-1]
