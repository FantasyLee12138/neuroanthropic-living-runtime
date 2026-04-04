from pathlib import Path

import duckdb

from nalr.trace.store import TraceStore


class _FailingConnection:
    def execute(self, query: str, params: list[object]):
        raise duckdb.IOException("IO Error: No files found that match '/tmp/missing.parquet'")

    def close(self) -> None:
        return None


def test_query_payload_rows_returns_empty_when_duckdb_reports_missing_parquet(tmp_path, monkeypatch):
    store = TraceStore(tmp_path)
    parquet_path = store.parquet_dir / "round_canonical.parquet"
    parquet_path.write_text("placeholder", encoding="utf-8")

    monkeypatch.setattr("nalr.trace.store.duckdb.connect", lambda: _FailingConnection())

    rows = store._query_payload_rows(
        parquet_path,
        "select payload_json from read_parquet(?)",
        [str(parquet_path)],
    )

    assert rows == []
