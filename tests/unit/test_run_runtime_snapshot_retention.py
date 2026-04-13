import os
from pathlib import Path

from nalr.runtime.controller import RuntimeController
from nalr.runtime.run_runtime import COMMAND_SNAPSHOT_RETENTION_LIMIT


CONFIG_ROOT = Path(__file__).resolve().parents[2] / "config"


def _make_old_snapshots(snapshot_dir: Path, *, count: int) -> list[Path]:
    snapshots: list[Path] = []
    base_timestamp = 1_700_000_000
    for idx in range(count):
        path = snapshot_dir / f"snap-old-{idx:03d}"
        path.mkdir(parents=True, exist_ok=True)
        ts = base_timestamp + idx
        os.utime(path, (ts, ts))
        snapshots.append(path)
    return snapshots


def test_create_command_snapshot_keeps_latest_snapshot_directories_only(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    state = controller.load_runtime_state()
    envelope = controller._legacy_command_envelope("safe on")
    created = _make_old_snapshots(
        controller.snapshot_dir,
        count=COMMAND_SNAPSHOT_RETENTION_LIMIT + 8,
    )

    snapshot_id = controller._create_command_snapshot(state, envelope)

    remaining = sorted(path.name for path in controller.snapshot_dir.iterdir() if path.is_dir() and path.name.startswith("snap-"))
    assert len(remaining) == COMMAND_SNAPSHOT_RETENTION_LIMIT
    assert created[0].name not in remaining
    assert created[-1].name in remaining
    assert snapshot_id in remaining


def test_create_command_snapshot_does_not_delete_non_matching_or_non_directory_entries(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    state = controller.load_runtime_state()
    envelope = controller._legacy_command_envelope("safe on")

    manual_dir = controller.snapshot_dir / "manual-backup"
    manual_dir.mkdir(parents=True, exist_ok=True)
    non_dir_snap = controller.snapshot_dir / "snap-file-entry"
    non_dir_snap.write_text("keep-me", encoding="utf-8")
    _make_old_snapshots(
        controller.snapshot_dir,
        count=COMMAND_SNAPSHOT_RETENTION_LIMIT + 2,
    )

    controller._create_command_snapshot(state, envelope)

    assert manual_dir.exists()
    assert non_dir_snap.exists()
    assert non_dir_snap.is_file()


def test_create_command_snapshot_never_deletes_the_just_created_snapshot(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    state = controller.load_runtime_state()
    envelope = controller._legacy_command_envelope("safe on")
    _make_old_snapshots(
        controller.snapshot_dir,
        count=COMMAND_SNAPSHOT_RETENTION_LIMIT + 4,
    )
    deleted: list[Path] = []

    def tracked_rmtree(path, *args, **kwargs):
        deleted.append(Path(path))

    monkeypatch.setattr("nalr.runtime.run_runtime.shutil.rmtree", tracked_rmtree)

    snapshot_id = controller._create_command_snapshot(state, envelope)

    snapshot_path = controller.snapshot_dir / snapshot_id
    assert snapshot_path.exists()
    assert snapshot_path not in deleted


def test_create_command_snapshot_retention_errors_do_not_break_creation(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    state = controller.load_runtime_state()
    envelope = controller._legacy_command_envelope("safe on")
    _make_old_snapshots(
        controller.snapshot_dir,
        count=COMMAND_SNAPSHOT_RETENTION_LIMIT + 3,
    )

    def always_fail(*args, **kwargs):
        raise OSError("boom")

    monkeypatch.setattr("nalr.runtime.run_runtime.shutil.rmtree", always_fail)

    snapshot_id = controller._create_command_snapshot(state, envelope)
    snapshot_path = controller.snapshot_dir / snapshot_id

    assert snapshot_path.exists()
