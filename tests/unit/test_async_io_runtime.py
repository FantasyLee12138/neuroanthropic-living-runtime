import time
from dataclasses import replace
from pathlib import Path

from nalr.runtime.controller import RuntimeController
from nalr.runtime.metadata import iso_date, utc_now_iso
from nalr.schemas.models import RoundEvent


CONFIG_ROOT = Path(__file__).resolve().parents[2] / "config"


def test_save_state_updates_memory_cache_without_waiting_for_disk(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    original_write_text = Path.write_text

    def slow_write_text(path: Path, data: str, *args, **kwargs):
        if path == controller.state_path:
            time.sleep(0.25)
        return original_write_text(path, data, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", slow_write_text)
    state = controller.load_runtime_state()
    state.safe_mode = True

    started_at = time.perf_counter()
    controller._save_state(state)
    elapsed = time.perf_counter() - started_at

    assert elapsed < 0.18
    assert controller.load_runtime_state().safe_mode is True

    controller.flush_pending_io(raise_on_error=True)


def test_write_round_updates_memory_cache_without_waiting_for_disk(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    seeded = controller.tick(
        RoundEvent(
            source="user",
            content="Remember noodles and help me plan dinner.",
            target="user",
            cue="noodles",
            valence=0.2,
        ),
        scenario="task",
        mode="interactive",
    )
    controller.flush_pending_io(raise_on_error=True)

    original_write_text = Path.write_text

    def slow_write_text(path: Path, data: str, *args, **kwargs):
        if path == controller.trace_store.rounds_dir / "round_42.json":
            time.sleep(0.25)
        return original_write_text(path, data, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", slow_write_text)
    monkeypatch.setattr(controller.trace_store, "_sync_parquet_mirror_safely", lambda *args, **kwargs: True)

    recorded_at = utc_now_iso()
    trace = replace(seeded.trace, round_id=42, recorded_at=recorded_at, recorded_date=iso_date(recorded_at))

    started_at = time.perf_counter()
    controller.trace_store.write_round(trace)
    elapsed = time.perf_counter() - started_at

    assert elapsed < 0.18
    assert controller.trace_round(42)["round_id"] == 42

    controller.flush_pending_io(raise_on_error=True)


def test_checkpoint_waits_for_background_state_flush(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    original_write_text = Path.write_text

    def slow_write_text(path: Path, data: str, *args, **kwargs):
        if path == controller.state_path:
            time.sleep(0.25)
        return original_write_text(path, data, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", slow_write_text)

    state = controller.load_runtime_state()
    state.safe_mode = True

    save_started_at = time.perf_counter()
    controller._save_state(state)
    save_elapsed = time.perf_counter() - save_started_at

    checkpoint_started_at = time.perf_counter()
    checkpoint = controller.checkpoint()
    checkpoint_elapsed = time.perf_counter() - checkpoint_started_at

    assert save_elapsed < 0.18
    assert checkpoint.checkpoint_id.startswith("ckpt-")
    assert checkpoint_elapsed >= 0.20


def test_compact_memory_returns_before_artifact_disk_writes_finish(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    for idx in range(6):
        controller.tick(
            RoundEvent(
                source="user",
                content=f"Remember coffee round {idx}",
                target="user",
                cue="coffee" if idx < 4 else "coding",
                valence=0.2,
            ),
            scenario="task",
            mode="interactive",
        )
    controller.flush_pending_io(raise_on_error=True)

    original_write_text = Path.write_text

    def slow_write_text(path: Path, data: str, *args, **kwargs):
        if path.parent in {
            controller.memory_store.episodic_hot_dir,
            controller.memory_store.episodic_warm_dir,
            controller.memory_store.episodic_archive_dir,
        }:
            time.sleep(0.25)
        return original_write_text(path, data, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", slow_write_text)

    started_at = time.perf_counter()
    summary = controller.compact_memory(hot_max_rounds=2, warm_max_rounds=4)
    elapsed = time.perf_counter() - started_at
    sample = controller.sample_memory("warm", limit=2)

    assert elapsed < 0.18
    assert summary["tiers"]["warm"]["artifact_count"] > 0
    assert sample
    assert all(item["tier"] == "warm" for item in sample)

    controller.flush_pending_io(raise_on_error=True)


def test_tick_queues_state_and_trace_writes_without_requesting_sync_flush(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    sync_calls: list[tuple[str, bool]] = []
    original_save_state = controller._save_state
    original_write_round = controller.trace_store.write_round

    def track_save_state(state, *, sync: bool = False):
        sync_calls.append(("_save_state", sync))
        return original_save_state(state, sync=sync)

    def track_write_round(trace, *, sync: bool = False):
        sync_calls.append(("write_round", sync))
        return original_write_round(trace, sync=sync)

    monkeypatch.setattr(controller, "_save_state", track_save_state)
    monkeypatch.setattr(controller.trace_store, "write_round", track_write_round)

    result = controller.tick(
        RoundEvent(
            source="user",
            content="Remember noodles and help me plan dinner.",
            target="user",
            cue="noodles",
            valence=0.2,
        ),
        scenario="task",
        mode="interactive",
    )

    assert result.round_id == 1
    assert ("_save_state", False) in sync_calls
    assert ("write_round", False) in sync_calls


def test_tick_batch_flushes_only_when_cold_flush_threshold_is_reached(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    monkeypatch.setattr(controller, "_cold_flush_round_interval", 2, raising=False)
    monkeypatch.setattr(controller, "_cold_flush_interval_seconds", 999.0, raising=False)

    flush_calls: list[bool] = []
    original_flush = controller.flush_pending_io

    def track_flush(*, raise_on_error: bool = False) -> None:
        flush_calls.append(raise_on_error)
        original_flush(raise_on_error=raise_on_error)

    monkeypatch.setattr(controller, "flush_pending_io", track_flush)
    monkeypatch.setattr(controller.trace_store, "_sync_parquet_mirror_safely", lambda *args, **kwargs: True)

    controller.tick(
        RoundEvent(source="user", content="Remember noodles.", target="user", cue="noodles"),
        scenario="task",
        mode="interactive",
    )
    controller.tick(
        RoundEvent(source="user", content="Remember tea.", target="user", cue="tea"),
        scenario="task",
        mode="interactive",
    )

    assert flush_calls == [True]
