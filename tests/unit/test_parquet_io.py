from pathlib import Path

import duckdb

from nalr.storage.parquet_io import append_dataset, read_dataset_rows, rewrite_snapshot, read_snapshot_rows


def test_append_dataset_writes_partitioned_parquet_parts(tmp_path):
    dataset_dir = tmp_path / "trace" / "round_canonical"
    schema = {
        "session_id": "VARCHAR",
        "recorded_at": "VARCHAR",
        "recorded_date": "VARCHAR",
        "round_id": "BIGINT",
        "payload_json": "VARCHAR",
    }

    append_dataset(
        dataset_dir,
        [
            {
                "session_id": "sess-1",
                "recorded_at": "2026-04-04T10:00:00Z",
                "recorded_date": "2026-04-04",
                "round_id": 1,
                "payload_json": '{"round_id":1}',
            }
        ],
        schema=schema,
        partition_keys=("recorded_date", "round_id"),
    )

    part_files = sorted(dataset_dir.glob("recorded_date=*/round_id=*/*.parquet"))

    assert len(part_files) == 1
    rows = read_dataset_rows(
        dataset_dir,
        "select payload_json from read_parquet(?) where round_id = ?",
        [1],
    )
    assert rows == [{"payload_json": '{"round_id":1}'}]


def test_rewrite_snapshot_replaces_existing_file(tmp_path):
    snapshot_path = tmp_path / "runtime" / "parquet" / "persona_state.parquet"
    schema = {
        "session_id": "VARCHAR",
        "payload_json": "VARCHAR",
    }

    rewrite_snapshot(snapshot_path, [{"session_id": "sess-1", "payload_json": '{"round_count":1}'}], schema=schema)
    rewrite_snapshot(snapshot_path, [{"session_id": "sess-1", "payload_json": '{"round_count":2}'}], schema=schema)

    rows = read_snapshot_rows(snapshot_path, "select payload_json from read_parquet(?)")

    assert rows == [{"payload_json": '{"round_count":2}'}]
    conn = duckdb.connect()
    try:
        count = conn.execute("select count(*) from read_parquet(?)", [str(snapshot_path)]).fetchone()[0]
    finally:
        conn.close()
    assert count == 1
