from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
from uuid import uuid4

import duckdb


def _write_rows(output_path: Path, rows: list[dict], *, schema: dict[str, str]) -> None:
    conn = duckdb.connect()
    table_name = "parquet_rows"
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        schema_sql = ", ".join(f"{name} {dtype}" for name, dtype in schema.items())
        conn.execute(f"create table {table_name} ({schema_sql})")
        if rows:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".jsonl", delete=False) as handle:
                temp_path = Path(handle.name)
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            try:
                columns = ", ".join(schema.keys())
                conn.execute(
                    f"insert into {table_name} ({columns}) select {columns} from read_json_auto(?, format='newline_delimited')",
                    [str(temp_path)],
                )
            finally:
                temp_path.unlink(missing_ok=True)
        conn.execute(f"copy {table_name} to ? (format parquet)", [str(output_path)])
    finally:
        conn.close()


def append_dataset(
    dataset_dir: Path,
    rows: list[dict],
    *,
    schema: dict[str, str],
    partition_keys: tuple[str, ...] = ("recorded_date",),
) -> list[Path]:
    written_paths: list[Path] = []
    grouped: dict[tuple[object, ...], list[dict]] = {}
    for row in rows:
        key = tuple(row.get(partition) for partition in partition_keys)
        grouped.setdefault(key, []).append(row)

    for key, grouped_rows in grouped.items():
        partition_dir = dataset_dir
        for field, value in zip(partition_keys, key, strict=False):
            partition_dir = partition_dir / f"{field}={value}"
        output_path = partition_dir / f"part-{uuid4().hex}.parquet"
        _write_rows(output_path, grouped_rows, schema=schema)
        written_paths.append(output_path)
    return written_paths


def rewrite_snapshot(snapshot_path: Path, rows: list[dict], *, schema: dict[str, str]) -> None:
    tmp_path = snapshot_path.with_name(f"{snapshot_path.name}.tmp-{uuid4().hex}")
    _write_rows(tmp_path, rows, schema=schema)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path.replace(snapshot_path)


def _dataset_glob(dataset_dir: Path) -> str:
    return str(dataset_dir / "**" / "*.parquet")


def read_dataset_rows(dataset_dir: Path, query: str, params: list[object] | None = None) -> list[dict]:
    if not dataset_dir.exists():
        return []
    conn = duckdb.connect()
    try:
        try:
            cursor = conn.execute(query, [_dataset_glob(dataset_dir), *(params or [])])
        except duckdb.IOException as exc:
            if "No files found" in str(exc):
                return []
            raise
        columns = [item[0] for item in cursor.description]
        return [dict(zip(columns, row, strict=False)) for row in cursor.fetchall()]
    finally:
        conn.close()


def read_snapshot_rows(snapshot_path: Path, query: str, params: list[object] | None = None) -> list[dict]:
    if not snapshot_path.exists():
        return []
    conn = duckdb.connect()
    try:
        cursor = conn.execute(query, [str(snapshot_path), *(params or [])])
        columns = [item[0] for item in cursor.description]
        return [dict(zip(columns, row, strict=False)) for row in cursor.fetchall()]
    finally:
        conn.close()


def copy_tree(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    if source.exists():
        shutil.copytree(source, destination)
