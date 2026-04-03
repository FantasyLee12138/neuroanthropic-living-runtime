from pathlib import Path

from nalr.memory.store import MemoryStore
from nalr.schemas.models import RoundEvent


def test_memory_store_compacts_hot_warm_archive_layers(tmp_path):
    store = MemoryStore(tmp_path, hot_limit=2, warm_limit=2)

    for cue in ["alpha", "beta", "gamma", "delta", "epsilon"]:
        store.ingest_event(RoundEvent(source="user", content=f"Remember {cue}", cue=cue))

    store.compact_layers()
    tiers = store.tier_counts()

    assert tiers["hot"] <= 2
    assert tiers["warm"] <= 2
    assert tiers["archive"] >= 1


def test_memory_store_recall_prefers_detail_then_gist_and_supports_ablation(tmp_path):
    store = MemoryStore(tmp_path, hot_limit=3, warm_limit=2)
    store.ingest_event(RoundEvent(source="user", content="Remember noodles tonight", cue="noodles"))
    store.ingest_event(RoundEvent(source="user", content="Remember noodles tonight again", cue="noodles"))

    recall = store.recall("noodles")
    ablated = store.recall("noodles", allow_detail=False)

    assert recall["mode"] in {"detail", "gist"}
    assert ablated["mode"] == "gist"
    assert ablated["strength"] <= recall["strength"]

