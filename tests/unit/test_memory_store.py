import json

from nalr.memory.store import MemoryStore
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
