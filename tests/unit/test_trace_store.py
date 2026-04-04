from pathlib import Path

import json

import duckdb

from nalr.trace.store import TraceStore


class _FailingConnection:
    def execute(self, query: str, params: list[object]):
        raise duckdb.IOException("IO Error: No files found that match '/tmp/missing.parquet'")

    def close(self) -> None:
        return None


class _SparseCommandResult:
    def __init__(self) -> None:
        self.applied = True
        self.scope = "checkpoint"
        self.delta = {"checkpoint_id": "ckpt-0001", "created": True}


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


def test_append_command_handles_sparse_checkpoint_results(tmp_path):
    store = TraceStore(tmp_path)

    store.append_command(
        "checkpoint create",
        _SparseCommandResult(),
        "before-hash",
        "after-hash",
        session_id="sess-test",
        recorded_at="2026-04-04T00:00:00Z",
        sync=True,
    )

    command_traces = store.list_command_traces()

    assert len(command_traces) == 1
    assert command_traces[0]["command"] == "checkpoint create"
    assert command_traces[0]["command_id"] == ""
    assert command_traces[0]["canonical"] == ""
    assert command_traces[0]["operator_level"] == "read_only"
    assert command_traces[0]["rollback_available"] is False


def test_list_rounds_falls_back_when_parquet_dir_exists_without_files(tmp_path):
    store = TraceStore(tmp_path)
    store.round_canonical_dir.mkdir(parents=True, exist_ok=True)
    round_path = store.rounds_dir / "round_1.json"
    round_path.write_text(
        json.dumps(
            {
                "round_id": 1,
                "session_id": "sess-test",
                "recorded_at": "2026-04-04T00:00:00Z",
                "recorded_date": "2026-04-04",
                "sampled_action": "respond",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    rounds = store.list_rounds()

    assert len(rounds) == 1
    assert rounds[0]["round_id"] == 1


def test_recent_rounds_and_skill_traces_slice_from_hot_cache(tmp_path):
    store = TraceStore(tmp_path)
    store._round_cache = {
        1: {"round_id": 1},
        2: {"round_id": 2},
        3: {"round_id": 3},
    }
    store._skill_cache = [
        {"round_id": 1, "skill_name": "a"},
        {"round_id": 2, "skill_name": "b"},
        {"round_id": 3, "skill_name": "c"},
    ]

    rounds = store.recent_rounds(2)
    skills = store.recent_skill_traces(2)

    assert [row["round_id"] for row in rounds] == [2, 3]
    assert [row["skill_name"] for row in skills] == ["b", "c"]
