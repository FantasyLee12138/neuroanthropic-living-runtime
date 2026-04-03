import json
from pathlib import Path

from nalr.runtime.controller import RuntimeController
from nalr.schemas.models import RoundEvent


CONFIG_ROOT = Path(__file__).resolve().parents[2] / "config"


def test_memory_compaction_writes_raw_evidence_and_tier_artifacts(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    for idx in range(6):
        controller.tick(
            RoundEvent(
                source="user",
                content=f"Remember coffee routine round {idx}",
                target="user",
                cue="coffee" if idx < 4 else "coding",
                valence=0.25,
            ),
            scenario="task",
            mode="interactive",
        )

    summary = controller.compact_memory(hot_max_rounds=2, warm_max_rounds=4)
    sample = controller.sample_memory("warm", limit=2)

    raw_path = tmp_path / ".alive" / "memory" / "raw" / "episodic_events.jsonl"
    raw_rows = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert raw_rows
    assert raw_rows[0]["session_id"] == controller.load_runtime_state().session_id
    assert raw_rows[0]["recorded_date"] == raw_rows[0]["recorded_at"][:10]
    assert summary["tiers"]["hot"]["artifact_count"] > 0
    assert summary["tiers"]["warm"]["artifact_count"] > 0
    assert summary["tiers"]["archive"]["artifact_count"] > 0
    assert sample
    assert all(item["tier"] == "warm" for item in sample)
    assert (tmp_path / ".alive" / "memory" / "episodic_hot").exists()
    assert (tmp_path / ".alive" / "memory" / "episodic_warm").exists()
    assert (tmp_path / ".alive" / "memory" / "episodic_archive").exists()
