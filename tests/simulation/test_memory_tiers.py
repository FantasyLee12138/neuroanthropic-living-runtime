from pathlib import Path

from nalr.memory.store import MemoryStore
from nalr.schemas.models import RoundEvent


def test_memory_store_promotes_hot_entries_into_warm_tier(tmp_path):
    store = MemoryStore(tmp_path / ".alive")

    for _ in range(5):
        store.ingest_event(
            RoundEvent(
                source="user",
                content="Remember that coffee helps me focus every morning.",
                cue="coffee",
                target="user",
                valence=0.2,
            )
        )

    top = store.memory_top(limit=5)

    assert (tmp_path / ".alive" / "memory" / "episodic_warm.json").exists()
    assert (tmp_path / ".alive" / "memory" / "episodic_archive.json").exists()
    assert (tmp_path / ".alive" / "memory" / "stable_priors.json").exists()
    assert any(item["cue"] == "coffee" and item["tier"] == "warm" for item in top)


def test_memory_store_applies_decay_interference_and_habit_cap(tmp_path):
    store = MemoryStore(tmp_path / ".alive")

    store.ingest_event(
        RoundEvent(
            source="user",
            content="Remember that coffee helps me focus every morning.",
            cue="coffee",
            valence=0.2,
        )
    )
    store.ingest_event(
        RoundEvent(
            source="user",
            content="A nearby coffer note is also part of my routine.",
            cue="coffer",
            valence=0.1,
        )
    )
    for _ in range(20):
        store.ingest_event(
            RoundEvent(
                source="user",
                content="Remember that coffee helps me focus every morning.",
                cue="coffee",
                valence=0.3,
            )
        )

    hot_memories = {item["cue"]: item for item in store._read_list(store.episodic_path)}
    habits = {item["pattern"]: item for item in store._read_list(store.habit_path)}

    assert hot_memories["coffee"]["interference"] > 0.0
    assert hot_memories["coffer"]["detail_strength"] < 0.18
    assert store.recall_strength("coffee") <= hot_memories["coffee"]["detail_strength"]
    assert habits["coffee"]["strength"] <= 0.92
