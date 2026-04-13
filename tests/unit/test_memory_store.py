from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading

from nalr.memory.store import MemoryStore
from nalr.runtime.dynamics import bounded_drift_delta, smooth_decay_rate, smooth_resource_pressure
from nalr.schemas.models import RoundEvent


def test_memory_store_repairs_empty_json_list_files(tmp_path):
    store = MemoryStore(tmp_path / ".alive")
    store.stable_priors_path.write_text("", encoding="utf-8")

    priors = store.stable_priors_top(limit=5)

    assert priors == []
    assert json.loads(store.stable_priors_path.read_text(encoding="utf-8")) == []


def test_memory_store_repairs_invalid_json_list_files(tmp_path):
    store = MemoryStore(tmp_path / ".alive")
    store.habit_path.write_text("{broken", encoding="utf-8")

    habits = store.habit_top(limit=5)

    assert habits == []
    assert json.loads(store.habit_path.read_text(encoding="utf-8")) == []


def test_memory_store_repairs_invalid_storage_status_file(tmp_path):
    store = MemoryStore(tmp_path / ".alive")
    store.storage_status_path.write_text("", encoding="utf-8")

    status = store.storage_status()

    assert status["storage_state"] == "healthy"
    assert "parquet_live_ready" in status
    assert json.loads(store.storage_status_path.read_text(encoding="utf-8"))["storage_state"] == "healthy"


def test_memory_store_noninteractive_shaping_updates_memory_and_habit(tmp_path):
    store = MemoryStore(tmp_path / ".alive")
    store.ingest_event(
        RoundEvent(source="user", content="记住 tea 这件事", target="user", cue="tea", valence=0.2),
        round_id=1,
        session_id="sess-1",
    )

    shaping = store.shape_noninteractive(mode="sleep", cue="tea")
    recall = store.recall("tea")

    assert shaping["source"] == "sleep"
    assert shaping["non_interactive"] is True
    assert recall["found"] is True


def test_memory_store_uses_mode_specific_decay_and_feedback_half_life(tmp_path):
    interactive_store = MemoryStore(tmp_path / "interactive" / ".alive")
    sleep_store = MemoryStore(tmp_path / "sleep" / ".alive")

    for store in (interactive_store, sleep_store):
        store.ingest_event(
            RoundEvent(
                source="user",
                content="Remember that coffee helps me focus every morning.",
                target="user",
                cue="coffee",
                valence=0.2,
            ),
            round_id=1,
            session_id="sess-1",
            mode="interactive",
        )

    interactive_store.ingest_event(
        RoundEvent(
            source="system",
            content="Background task reshaping coding context.",
            target="user",
            cue="coding",
            valence=0.0,
        ),
        round_id=2,
        session_id="sess-1",
        mode="task",
    )
    sleep_store.ingest_event(
        RoundEvent(
            source="system",
            content="Sleep reshaping coding context.",
            target="user",
            cue="coding",
            valence=0.0,
        ),
        round_id=2,
        session_id="sess-1",
        mode="sleep",
    )

    task_coffee = next(item for item in interactive_store._read_list(interactive_store.episodic_path) if item["cue"] == "coffee")
    sleep_coffee = next(item for item in sleep_store._read_list(sleep_store.episodic_path) if item["cue"] == "coffee")

    recent_store = MemoryStore(tmp_path / "recent" / ".alive")
    stale_store = MemoryStore(tmp_path / "stale" / ".alive")
    recent_strength = recent_store.update_habit_strength("tea", 1.0, context_recurrence=1.0, round_gap=0)["strength"]
    stale_strength = stale_store.update_habit_strength("tea", 1.0, context_recurrence=1.0, round_gap=6)["strength"]

    assert task_coffee["detail_strength"] > sleep_coffee["detail_strength"]
    assert stale_strength < recent_strength


def test_memory_store_can_build_and_apply_noninteractive_proposal(tmp_path):
    store = MemoryStore(tmp_path / ".alive")
    store.ingest_event(
        RoundEvent(
            source="user",
            content="Remember tea before sleep.",
            target="user",
            cue="tea",
            valence=0.2,
        ),
        round_id=1,
        session_id="sess-1",
    )

    proposal = store.build_noninteractive_proposal(mode="sleep", cue="tea")

    assert proposal["mode"] == "sleep"
    assert proposal["memory_consolidation"]
    assert proposal["habit_adjustments"]
    assert proposal["dream_memory_write"]

    applied = store.apply_noninteractive_proposal(proposal)
    recall = store.recall("tea")

    assert applied["applied"] is True
    assert "memory_consolidation" in applied["applied_types"]
    assert recall["found"] is True


def test_memory_store_uses_v056_interference_threshold_and_high_quality_cue_recovery(tmp_path):
    store = MemoryStore(tmp_path / ".alive")

    store.ingest_event(
        RoundEvent(
            source="user",
            content="remember alpha idea for later",
            target="user",
            cue="alpha",
            valence=0.1,
        ),
        round_id=1,
        session_id="sess-1",
    )
    alpha = next(item for item in store._read_list(store.episodic_path) if item["cue"] == "alpha")
    alpha["detail_strength"] = 0.30
    alpha["gist_strength"] = 0.32
    store._write_list(store.episodic_path, [alpha])

    store.ingest_event(
        RoundEvent(
            source="user",
            content="remember alphb idea for later",
            target="user",
            cue="alphb",
            valence=0.1,
        ),
        round_id=2,
        session_id="sess-1",
    )

    hot = {item["cue"]: item for item in store._read_list(store.episodic_path)}
    assert hot["alphb"]["interference"] > 0.0
    assert hot["alphb"]["interfered"] is True

    recovered = store.ingest_event(
        RoundEvent(
            source="user",
            content="remember alpha with exact quote and time details",
            target="user",
            cue="alpha",
            valence=0.1,
        ),
        round_id=3,
        session_id="sess-1",
        mode="interactive",
        cue_quality=1.0,
    )
    assert recovered == "alpha"

    alpha_after = next(item for item in store._read_list(store.episodic_path) if item["cue"] == "alpha")
    assert alpha_after["last_cue_quality"] == 1.0
    assert alpha_after["detail_strength"] > 0.50


def test_memory_store_ingest_event_unpacks_interference_tuple(tmp_path):
    store = MemoryStore(tmp_path / ".alive")
    store.ingest_event(
        RoundEvent(
            source="user",
            content="remember alpha idea for later",
            target="user",
            cue="alpha",
            valence=0.1,
        ),
        round_id=1,
        session_id="sess-1",
    )

    store.ingest_event(
        RoundEvent(
            source="user",
            content="remember alphb idea for later",
            target="user",
            cue="alphb",
            valence=0.1,
        ),
        round_id=2,
        session_id="sess-1",
    )

    hot = {item["cue"]: item for item in store._read_list(store.episodic_path)}
    assert isinstance(hot["alphb"]["interference"], float)
    assert hot["alphb"]["interfered"] is True


def test_memory_store_suppresses_polluting_write_under_high_interference_and_pressure(tmp_path):
    store = MemoryStore(tmp_path / ".alive")
    store.ingest_event(
        RoundEvent(
            source="user",
            content="remember alpha idea for later",
            target="user",
            cue="alpha",
            valence=0.1,
        ),
        round_id=1,
        session_id="sess-1",
    )
    alpha = next(item for item in store._read_list(store.episodic_path) if item["cue"] == "alpha")
    alpha["detail_strength"] = 0.30
    alpha["gist_strength"] = 0.32
    store._write_list(store.episodic_path, [alpha])

    cue = store.ingest_event(
        RoundEvent(
            source="user",
            content="remember alphb idea for later",
            target="user",
            cue="alphb",
            valence=0.0,
        ),
        round_id=2,
        session_id="sess-1",
        cue_quality=0.0,
        resource_pressure=0.92,
    )

    hot = {item["cue"]: item for item in store._read_list(store.episodic_path)}
    diagnostics = store.last_ingest_diagnostics()

    assert cue == "alphb"
    assert "alphb" not in hot
    assert diagnostics["suppressed"] is True
    assert diagnostics["reason"] == "pollution_guard"
    assert diagnostics["interference"] > 0.0
    assert diagnostics["resource_pressure"] == 0.92


def test_memory_store_assigns_episode_and_separation_ids_to_recallable_memories(tmp_path):
    store = MemoryStore(tmp_path / ".alive")
    store.ingest_event(
        RoundEvent(
            source="user",
            content="remember alpha idea for later",
            target="user",
            cue="alpha",
            valence=0.1,
        ),
        round_id=1,
        session_id="sess-1",
    )
    alpha = next(item for item in store._read_list(store.episodic_path) if item["cue"] == "alpha")
    alpha["detail_strength"] = 0.30
    alpha["gist_strength"] = 0.32
    store._write_list(store.episodic_path, [alpha])

    store.ingest_event(
        RoundEvent(
            source="user",
            content="remember alphb idea for later",
            target="user",
            cue="alphb",
            valence=0.1,
        ),
        round_id=2,
        session_id="sess-1",
        cue_quality=1.0,
    )

    hot = {item["cue"]: item for item in store._read_list(store.episodic_path)}
    recall = store.recall("alphb")

    assert hot["alphb"]["episode_id"].startswith("ep-")
    assert hot["alphb"]["separation_id"].startswith("sep-")
    assert recall["episode_id"] == hot["alphb"]["episode_id"]
    assert recall["separation_id"] == hot["alphb"]["separation_id"]
    assert recall["prior_vector"]["episodic_confidence"] > 0.0
    assert recall["prior_vector"]["interference_penalty"] >= 0.0
    assert recall["prior_vector"]["detail_bias"] >= 0.0


def test_memory_store_recall_surfaces_structural_imperfection_evidence(tmp_path):
    store = MemoryStore(tmp_path / ".alive")
    store.ingest_event(
        RoundEvent(
            source="user",
            content="remember alpha idea for later",
            target="user",
            cue="alpha",
            valence=0.1,
        ),
        round_id=1,
        session_id="sess-1",
    )
    alpha = next(item for item in store._read_list(store.episodic_path) if item["cue"] == "alpha")
    alpha["detail_strength"] = 0.30
    alpha["gist_strength"] = 0.32
    store._write_list(store.episodic_path, [alpha])

    store.ingest_event(
        RoundEvent(
            source="user",
            content="remember alphb idea for later",
            target="user",
            cue="alphb",
            valence=0.1,
        ),
        round_id=2,
        session_id="sess-1",
        cue_quality=1.0,
    )

    recalled = store.recall("alphb")
    missing = store.recall("missing")

    assert "latency_ms" in recalled
    assert recalled["cost_tier"]
    assert recalled["cue_rescue"] in {True, False}
    assert "contamination" in recalled
    assert recalled["recall_failure"] is False
    assert missing["recall_failure"] is True
    assert missing["mode"] == "none"


def test_memory_store_marks_gist_recall_as_summary_contamination(tmp_path):
    store = MemoryStore(tmp_path / ".alive")
    store.ingest_event(
        RoundEvent(source="user", content="记住 tea 这件事", target="user", cue="tea", valence=0.2),
        round_id=1,
        session_id="sess-1",
    )
    store.shape_noninteractive(mode="sleep", cue="tea")

    recall = store.recall("tea", allow_detail=False)

    assert recall["mode"] == "gist"
    assert recall["contamination"]["detected"] is True


def test_memory_store_marks_old_habit_as_suppressed_but_recoverable(tmp_path):
    store = MemoryStore(tmp_path / ".alive")

    for round_id in range(1, 6):
        store.update_habit_strength("coffee", 0.8, context_recurrence=1.0, round_gap=0, context_slot="user::user", round_id=round_id)

    for round_id in range(6, 12):
        store.update_habit_strength("tea", 0.95, context_recurrence=1.0, round_gap=0, context_slot="user::user", round_id=round_id)

    habits = {item["pattern"]: item for item in store._read_list(store.habit_path)}
    assert habits["coffee"]["status"] == "suppressed_recoverable"
    assert habits["coffee"]["suppressed_by"] == "tea"
    assert habits["tea"]["status"] == "active"

    recovered = store.update_habit_strength("coffee", 0.7, context_recurrence=1.0, round_gap=0, context_slot="user::user", round_id=12)
    assert recovered["strength"] > 0.0
    assert recovered["last_context_recurrence"] == 1.0


def test_runtime_dynamics_helpers_are_monotonic_and_bounded():
    strong_support = smooth_decay_rate(base_decay=0.01, recalled_within_window=True, affect_intensity=0.85)
    medium_support = smooth_decay_rate(base_decay=0.01, recalled_within_window=False, affect_intensity=0.45)
    weak_support = smooth_decay_rate(base_decay=0.01, recalled_within_window=False, affect_intensity=0.05)
    low_pressure = smooth_resource_pressure(0.15)
    high_pressure = smooth_resource_pressure(0.85)
    drift = bounded_drift_delta(previous_drift=0.05, push=0.20, recover_rate=0.04, max_abs=0.25)

    assert 0.0 < strong_support < medium_support < weak_support < 0.05
    assert 0.0 <= low_pressure < high_pressure <= 1.0
    assert -0.25 <= drift <= 0.25


def test_memory_store_default_recall_searches_full_tiers(tmp_path):
    store = MemoryStore(tmp_path / ".alive")
    store._write_list(
        store.episodic_cold_path,
        [store._normalize_memory_record({"cue": "tea", "gist_strength": 0.6, "detail_strength": 0.55, "count": 8})],
    )

    recall = store.recall("tea")

    assert recall["found"] is True
    assert recall["tier"] == "cold"
    assert recall["strength"] == 0.6


def test_memory_store_hot_only_budget_skips_warm_and_cold(tmp_path, monkeypatch):
    store = MemoryStore(tmp_path / ".alive")
    store._write_list(
        store.episodic_warm_path,
        [store._normalize_memory_record({"cue": "tea", "gist_strength": 0.6, "detail_strength": 0.55, "count": 4})],
    )

    seen_paths = []
    tier_calls = []
    original_lookup = getattr(store, "_lookup_tiers", None)
    original_read_list = store._read_list

    def tracking_lookup(cue, tier_budget):
        budget = tuple(tier_budget)
        assert budget == ("hot",)
        tier_calls.append((cue, budget))
        if original_lookup is not None:
            return original_lookup(cue, budget)
        return {"cue": cue, "tier": None, "found": False, "strength": 0.0, "detail": False, "evidence": []}

    def tracking_read_list(path):
        seen_paths.append(path)
        return original_read_list(path)

    monkeypatch.setattr(store, "_lookup_tiers", tracking_lookup, raising=False)
    monkeypatch.setattr(store, "_read_list", tracking_read_list)

    recall = store.recall("tea", tier_budget=("hot",))

    assert recall["found"] is False
    assert recall["tier"] is None


def test_memory_store_clusters_identity_questions_instead_of_storing_raw_sentences(tmp_path):
    store = MemoryStore(tmp_path / ".alive")

    store.ingest_event(
        RoundEvent(source="user", content="你是谁？", target="user", valence=0.0),
        round_id=1,
        session_id="sess-1",
    )
    store.ingest_event(
        RoundEvent(source="user", content="你叫什么名字？", target="user", valence=0.0),
        round_id=2,
        session_id="sess-1",
    )

    priors = store.stable_priors_top(limit=10)
    evidence = store.identity_evidence()

    assert priors
    assert priors[0]["cue"].startswith("identity:")
    assert all(item["cue"] not in {"你是谁？", "你叫什么名字？"} for item in priors)
    assert evidence["identity_clusters"]
    assert evidence["identity_score"] > 0.0


def test_memory_store_schema_migration_clusters_identity_cues_and_writes_report(tmp_path):
    store_root = tmp_path / ".alive"
    memory_dir = store_root / "memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    (memory_dir / "stable_priors.json").write_text(
        json.dumps(
            [
                {"cue": "你是谁？", "weight": 0.11},
                {"cue": "你叫什么名字？", "weight": 0.13},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (memory_dir / "habit.json").write_text(
        json.dumps(
            [
                {"pattern": "你叫什么名字？", "strength": 0.28, "count": 2},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    store = MemoryStore(store_root)
    priors = store.stable_priors_top(limit=10)
    report = json.loads(store.migration_status_path.read_text(encoding="utf-8"))

    assert any(item["cue"] == "identity:name_probe" for item in priors)
    assert all(item["cue"] != "你叫什么名字？" for item in priors)
    assert report["schema_version"] >= 2
    assert report["merged_cue_clusters_count"] >= 1
    assert report["identity_evidence_rebuilt"] is True


def test_memory_store_full_budget_can_fallback_to_warm(tmp_path, monkeypatch):
    store = MemoryStore(tmp_path / ".alive")
    store._write_list(
        store.episodic_warm_path,
        [store._normalize_memory_record({"cue": "tea", "gist_strength": 0.6, "detail_strength": 0.55, "count": 4})],
    )

    tier_calls = []
    original_lookup = getattr(store, "_lookup_tiers", None)

    def tracking_lookup(cue, tier_budget):
        budget = tuple(tier_budget)
        tier_calls.append((cue, budget))
        if original_lookup is not None:
            return original_lookup(cue, budget)
        return {"cue": cue, "tier": None, "found": False, "strength": 0.0, "detail": False, "evidence": []}

    monkeypatch.setattr(store, "_lookup_tiers", tracking_lookup, raising=False)

    warm_recall = store.recall("tea", tier_budget=("hot", "warm", "cold"))
    hot_recall = store.recall("tea", tier_budget=("hot",))

    assert warm_recall["found"] is True
    assert warm_recall["tier"] == "warm"
    assert warm_recall["strength"] == 0.6
    assert hot_recall["found"] is False
    assert hot_recall["tier"] is None
    assert tier_calls == [("tea", ("hot", "warm", "cold")), ("tea", ("hot",))]


def test_memory_store_hot_miss_does_not_poison_full_budget_can_fallback(tmp_path, monkeypatch):
    store = MemoryStore(tmp_path / ".alive")
    store._write_list(
        store.episodic_warm_path,
        [store._normalize_memory_record({"cue": "tea", "gist_strength": 0.6, "detail_strength": 0.55, "count": 4})],
    )

    tier_calls = []
    original_lookup = getattr(store, "_lookup_tiers", None)

    def tracking_lookup(cue, tier_budget):
        budget = tuple(tier_budget)
        tier_calls.append((cue, budget))
        assert budget in {("hot",), ("hot", "warm", "cold")}
        if original_lookup is not None:
            return original_lookup(cue, budget)
        if budget == ("hot",):
            return {"cue": cue, "tier": None, "found": False, "strength": 0.0, "detail": False, "evidence": []}
        return {"cue": cue, "tier": "warm", "found": True, "strength": 0.6, "detail": True, "evidence": ["warm"]}

    monkeypatch.setattr(store, "_lookup_tiers", tracking_lookup, raising=False)

    hot_miss = store.recall("tea", tier_budget=("hot",))
    full_hit = store.recall("tea", tier_budget=("hot", "warm", "cold"))

    assert hot_miss["found"] is False
    assert hot_miss["tier"] is None
    assert full_hit["found"] is True
    assert full_hit["tier"] == "warm"
    assert tier_calls == [("tea", ("hot",)), ("tea", ("hot", "warm", "cold"))]


def test_memory_store_hot_budget_uses_lru_for_repeated_queries(tmp_path, monkeypatch):
    store = MemoryStore(tmp_path / ".alive")
    store.ingest_event(
        RoundEvent(source="user", content="Remember tea for later.", target="user", cue="tea", valence=0.2),
        round_id=1,
        session_id="sess-1",
    )

    tier_calls = []
    original_lookup = getattr(store, "_lookup_tiers", None)

    def tracking_lookup(cue, tier_budget):
        budget = tuple(tier_budget)
        assert budget == ("hot",)
        tier_calls.append((cue, budget))
        if original_lookup is not None:
            return original_lookup(cue, budget)
        return {"cue": cue, "tier": "hot", "found": True, "strength": 1.0, "detail": True, "evidence": []}

    monkeypatch.setattr(store, "_lookup_tiers", tracking_lookup, raising=False)

    first = store.recall("tea", tier_budget=("hot",))
    second = store.recall("tea", tier_budget=("hot",))

    assert first["found"] is True
    assert second["found"] is True
    assert first["tier"] == "hot"
    assert second["tier"] == "hot"
    assert tier_calls == [("tea", ("hot",))]


def test_memory_store_write_paths_invalidate_hot_cache(tmp_path, monkeypatch):
    store = MemoryStore(tmp_path / ".alive")
    store.ingest_event(
        RoundEvent(source="user", content="Remember tea for later.", target="user", cue="tea", valence=0.2),
        round_id=1,
        session_id="sess-1",
    )

    tier_calls = []
    original_lookup = getattr(store, "_lookup_tiers", None)

    def tracking_lookup(cue, tier_budget):
        budget = tuple(tier_budget)
        assert budget == ("hot",)
        tier_calls.append((cue, budget))
        if original_lookup is not None:
            return original_lookup(cue, budget)
        return {"cue": cue, "tier": "hot", "found": True, "strength": 0.0, "detail": False, "evidence": []}

    monkeypatch.setattr(store, "_lookup_tiers", tracking_lookup, raising=False)

    initial = store.recall("tea", tier_budget=("hot",))
    store.ingest_event(
        RoundEvent(source="user", content="Remember tea with exact details.", target="user", cue="tea", valence=0.3),
        round_id=2,
        session_id="sess-1",
        cue_quality=1.0,
    )
    refreshed = store.recall("tea", tier_budget=("hot",))

    assert initial["found"] is True
    assert refreshed["found"] is True
    assert initial["tier"] == "hot"
    assert refreshed["tier"] == "hot"
    assert refreshed["strength"] > initial["strength"]
    assert tier_calls == [("tea", ("hot",)), ("tea", ("hot",))]


def test_memory_store_write_paths_invalidate_hot_miss_cache(tmp_path, monkeypatch):
    store = MemoryStore(tmp_path / ".alive")

    tier_calls = []
    original_lookup = getattr(store, "_lookup_tiers", None)

    def tracking_lookup(cue, tier_budget):
        budget = tuple(tier_budget)
        assert budget == ("hot",)
        tier_calls.append((cue, budget))
        if original_lookup is not None:
            return original_lookup(cue, budget)
        return {"cue": cue, "tier": None, "found": False, "strength": 0.0, "detail": False, "evidence": []}

    monkeypatch.setattr(store, "_lookup_tiers", tracking_lookup, raising=False)

    miss = store.recall("tea", tier_budget=("hot",))
    second_miss = store.recall("tea", tier_budget=("hot",))
    assert miss["found"] is False
    assert second_miss["found"] is False
    assert tier_calls == [("tea", ("hot",))]

    store.ingest_event(
        RoundEvent(source="user", content="Remember tea for later.", target="user", cue="tea", valence=0.2),
        round_id=1,
        session_id="sess-1",
    )
    hit = store.recall("tea", tier_budget=("hot",))

    assert hit["found"] is True
    assert hit["tier"] == "hot"
    assert tier_calls == [("tea", ("hot",)), ("tea", ("hot",))]


def test_memory_store_write_paths_invalidate_full_budget_warm_cache(tmp_path, monkeypatch):
    store = MemoryStore(tmp_path / ".alive")
    store._write_list(
        store.episodic_warm_path,
        [store._normalize_memory_record({"cue": "tea", "gist_strength": 0.6, "detail_strength": 0.55, "count": 4})],
    )

    tier_calls = []
    original_lookup = getattr(store, "_lookup_tiers", None)

    def tracking_lookup(cue, tier_budget):
        budget = tuple(tier_budget)
        assert budget == ("hot", "warm", "cold")
        tier_calls.append((cue, budget))
        if original_lookup is not None:
            return original_lookup(cue, budget)
        return {"cue": cue, "tier": "warm", "found": True, "strength": 0.6, "detail": True, "evidence": ["warm"]}

    monkeypatch.setattr(store, "_lookup_tiers", tracking_lookup, raising=False)

    initial = store.recall("tea", tier_budget=("hot", "warm", "cold"))
    store.ingest_event(
        RoundEvent(source="user", content="Remember tea with exact details.", target="user", cue="tea", valence=0.3),
        round_id=2,
        session_id="sess-1",
        cue_quality=1.0,
    )
    refreshed = store.recall("tea", tier_budget=("hot", "warm", "cold"))

    assert initial["found"] is True
    assert initial["tier"] == "warm"
    assert refreshed["found"] is True
    assert refreshed["tier"] == "hot"
    assert refreshed["strength"] > initial["strength"]
    assert tier_calls == [("tea", ("hot", "warm", "cold")), ("tea", ("hot", "warm", "cold"))]


def test_memory_store_compacts_hot_warm_cold_layers(tmp_path):
    store = MemoryStore(tmp_path, hot_limit=2, warm_limit=2)

    for cue in ["alpha", "beta", "gamma", "delta", "epsilon"]:
        store.ingest_event(RoundEvent(source="user", content=f"Remember {cue}", cue=cue), round_id=1, session_id="sess-1")

    store.compact_layers()
    tiers = store.tier_counts()

    assert tiers["hot"] <= 2
    assert tiers["warm"] <= 2
    assert tiers["cold"] >= 1


def test_memory_store_compaction_ignores_suppressed_raw_events(tmp_path):
    store = MemoryStore(tmp_path / ".alive")
    store.ingest_event(
        RoundEvent(source="user", content="remember alpha idea for later", target="user", cue="alpha", valence=0.1),
        round_id=1,
        session_id="sess-1",
    )
    alpha = next(item for item in store._read_list(store.episodic_path) if item["cue"] == "alpha")
    alpha["detail_strength"] = 0.30
    alpha["gist_strength"] = 0.32
    store._write_list(store.episodic_path, [alpha])

    store.ingest_event(
        RoundEvent(source="user", content="remember alphb idea for later", target="user", cue="alphb", valence=0.0),
        round_id=2,
        session_id="sess-1",
        cue_quality=0.0,
        resource_pressure=0.92,
    )

    summary = store.compact_tiers()
    hot_artifacts = store.sample_compacted("hot", limit=10)

    assert summary["raw_event_count"] == 1
    assert [item["cue"] for item in hot_artifacts] == ["alpha"]


def test_memory_store_recall_prefers_detail_then_gist_and_supports_ablation(tmp_path):
    store = MemoryStore(tmp_path, hot_limit=3, warm_limit=2)
    store.ingest_event(RoundEvent(source="user", content="Remember noodles tonight", cue="noodles"), round_id=1, session_id="sess-1")
    store.ingest_event(RoundEvent(source="user", content="Remember noodles tonight again", cue="noodles"), round_id=2, session_id="sess-1")

    recall = store.recall("noodles")
    ablated = store.recall("noodles", allow_detail=False)

    assert recall["mode"] in {"detail", "gist"}
    assert ablated["mode"] == "gist"
    assert ablated["strength"] <= recall["strength"]


def test_memory_store_defers_snapshot_and_raw_parquet_sync_until_flush(tmp_path):
    store = MemoryStore(tmp_path / ".alive")
    snapshot_path = store._snapshot_targets[store.episodic_path]
    initial_snapshot_mtime = snapshot_path.stat().st_mtime_ns if snapshot_path.exists() else 0

    store.ingest_event(
        RoundEvent(source="user", content="remember alpha idea for later", target="user", cue="alpha", valence=0.1),
        round_id=1,
        session_id="sess-1",
    )
    store._io_worker.flush(raise_on_error=True)

    current_snapshot_mtime = snapshot_path.stat().st_mtime_ns if snapshot_path.exists() else 0
    assert current_snapshot_mtime == initial_snapshot_mtime
    assert list(store.raw_parquet_dir.rglob("*.parquet")) == []

    store.flush(raise_on_error=True)

    assert snapshot_path.exists() is True
    assert snapshot_path.stat().st_mtime_ns >= initial_snapshot_mtime
    assert list(store.raw_parquet_dir.rglob("*.parquet"))


def test_memory_store_reload_reads_raw_events_from_jsonl_before_parquet_flush(tmp_path):
    root = tmp_path / ".alive"
    store = MemoryStore(root)

    store.ingest_event(
        RoundEvent(source="user", content="remember alpha idea for later", target="user", cue="alpha", valence=0.1),
        round_id=1,
        session_id="sess-1",
    )
    store._io_worker.flush(raise_on_error=True)

    reloaded = MemoryStore(root)
    rows = reloaded._read_jsonl(reloaded.raw_events_path)

    assert rows
    assert rows[-1]["cue"] == "alpha"


def test_memory_store_atomic_write_does_not_share_tmp_name_under_concurrent_writes(tmp_path, monkeypatch):
    store = MemoryStore(tmp_path / ".alive")
    path = store.episodic_path
    barrier = threading.Barrier(2)
    original_write_text = Path.write_text

    def slow_tmp_write(target: Path, data: str, *args, **kwargs):
        result = original_write_text(target, data, *args, **kwargs)
        if target.parent == path.parent and target.name.endswith(".tmp"):
            barrier.wait(timeout=2.0)
        return result

    monkeypatch.setattr(Path, "write_text", slow_tmp_write)

    def write(content: str) -> None:
        store._write_text_atomic(path, content)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(write, "[]")
        second = executor.submit(write, '[{"cue":"alpha"}]')
        first.result(timeout=2.0)
        second.result(timeout=2.0)

    assert path.exists() is True


def test_memory_store_defers_compacted_artifact_loading_until_requested(tmp_path, monkeypatch):
    calls: list[str] = []
    original = MemoryStore._load_artifacts

    def tracked_load(self, tier_dir):
        calls.append(tier_dir.name)
        return original(self, tier_dir)

    monkeypatch.setattr(MemoryStore, "_load_artifacts", tracked_load)

    store = MemoryStore(tmp_path / ".alive")

    assert calls == []

    rows = store.sample_compacted("hot")

    assert rows == []
    assert calls == ["episodic_hot"]
