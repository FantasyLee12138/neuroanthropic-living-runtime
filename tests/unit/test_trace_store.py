import json
from pathlib import Path

import duckdb

import nalr.trace.store as trace_store_module
from nalr.runtime.controller import RuntimeController
from nalr.schemas.models import CommandResult, RoundEvent
from nalr.trace.exporter import TraceExporter
from nalr.trace.store import ROUND_CANONICAL_SCHEMA, ROUND_SUMMARY_SCHEMA, TraceStore, canonical_probability_field_payload
from nalr.storage.parquet_io import read_dataset_rows


CONFIG_ROOT = Path(__file__).resolve().parents[2] / "config"


class _FailingConnection:
    def execute(self, query: str, params: list[object]):
        raise duckdb.IOException("IO Error: No files found that match '/tmp/missing.parquet'")

    def close(self) -> None:
        return None


class _InvalidParquetConnection:
    def __init__(self, message: str) -> None:
        self._message = message

    def execute(self, query: str, params: list[object]):
        raise duckdb.InvalidInputException(self._message)

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


def test_query_payload_rows_returns_empty_for_transient_invalid_parquet_files(tmp_path, monkeypatch):
    store = TraceStore(tmp_path)
    parquet_path = store.parquet_dir / "round_canonical.parquet"
    parquet_path.write_text("placeholder", encoding="utf-8")

    monkeypatch.setattr(
        "nalr.trace.store.duckdb.connect",
        lambda: _InvalidParquetConnection(
            "Invalid Input Error: File '/tmp/part.parquet' too small to be a Parquet file"
        ),
    )

    rows = store._query_payload_rows(
        parquet_path,
        "select payload_json from read_parquet(?)",
        [str(parquet_path)],
    )

    assert rows == []


def test_list_rounds_skips_transient_invalid_json_round_files(tmp_path):
    store = TraceStore(tmp_path)
    broken_path = store.rounds_dir / "round_0001.json"
    valid_path = store.rounds_dir / "round_0002.json"
    broken_path.parent.mkdir(parents=True, exist_ok=True)
    broken_path.write_text("", encoding="utf-8")
    valid_path.write_text(
        json.dumps(
            {
                "session_id": "observer-main",
                "round_id": 2,
                "sampled_action": "respond",
                "recorded_at": "2026-04-12T00:00:02Z",
                "recorded_date": "2026-04-12",
                "probability_field": {},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    rows = store.list_rounds()

    assert [row["round_id"] for row in rows] == [2]


class _NoFrontInsertBytearray(bytearray):
    def __setitem__(self, key, value):
        if isinstance(key, slice) and key.start in (None, 0) and key.stop == 0:
            raise AssertionError("front insertion should not be used when tail-reading jsonl")
        return super().__setitem__(key, value)


def test_read_jsonl_tail_keeps_order_without_front_insertion(tmp_path, monkeypatch):
    store = TraceStore(tmp_path)
    rows = [
        {"round_id": index, "recorded_at": f"2026-04-12T00:00:{index:02d}Z", "recorded_date": "2026-04-12"}
        for index in range(1, 6)
    ]
    store.rounds_jsonl_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(trace_store_module, "bytearray", _NoFrontInsertBytearray, raising=False)

    tail = store._read_jsonl_tail(store.rounds_jsonl_path, limit=3)

    assert [row["round_id"] for row in tail] == [3, 4, 5]


def test_recent_rounds_prefers_jsonl_tail_in_chronological_order(tmp_path):
    store = TraceStore(tmp_path)
    rows = [
        {
            "session_id": "observer-main",
            "round_id": index,
            "sampled_action": "respond",
            "recorded_at": f"2026-04-12T00:00:{index:02d}Z",
            "recorded_date": "2026-04-12",
            "probability_field": {},
        }
        for index in range(1, 6)
    ]
    store._round_cache.clear()
    store._round_cache_complete = False
    store._trace_sync_status_cache["storage_state"] = "healthy"
    store.rounds_jsonl_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
        encoding="utf-8",
    )

    recent = store.recent_rounds(limit=3)

    assert [row["round_id"] for row in recent] == [3, 4, 5]


def test_recent_skill_traces_prefers_recent_jsonl_tail_without_full_cache_load(tmp_path, monkeypatch):
    store = TraceStore(tmp_path)
    rows = [
        {
            "session_id": "observer-main",
            "recorded_at": f"2026-04-12T00:00:{index:02d}Z",
            "recorded_date": "2026-04-12",
            "round_id": index,
            "skill_name": f"skill-{index}",
            "owner_module": "Renderer",
            "latency_ms": index * 10,
        }
        for index in range(1, 5)
    ]
    store._skill_cache_loaded = False
    store._trace_sync_status_cache["storage_state"] = "healthy"
    store.skill_jsonl_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
        encoding="utf-8",
    )
    store._skill_cache.append(
        {
            "session_id": "observer-main",
            "recorded_at": "2026-04-12T00:00:05Z",
            "recorded_date": "2026-04-12",
            "round_id": 5,
            "skill_name": "skill-5",
            "owner_module": "Renderer",
            "latency_ms": 50,
        }
    )

    monkeypatch.setattr(
        store,
        "_ensure_skill_cache_loaded",
        lambda: (_ for _ in ()).throw(AssertionError("recent_skill_traces should not full-load skill history")),
    )

    recent = store.recent_skill_traces(limit=3)

    assert [row["round_id"] for row in recent] == [3, 4, 5]
    assert store._skill_cache_loaded is False


def test_list_round_summaries_reads_compact_dataset_without_full_round_scan(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    result = controller.tick(
        RoundEvent(
            source="user",
            content="Please summarize this round compactly.",
            target="user",
            cue="compact",
        ),
        scenario="chat",
        mode="interactive",
    )
    controller.flush_pending_io(raise_on_error=True)
    store = controller.trace_store

    monkeypatch.setattr(
        store,
        "list_rounds",
        lambda: (_ for _ in ()).throw(AssertionError("list_round_summaries should not full-scan round history when summary data exists")),
    )

    summaries = store.list_round_summaries()

    assert summaries[-1]["round_id"] == result.round_id
    assert summaries[-1]["rendered_expression"]["text"]
    assert "motivation_pool" in summaries[-1]
    assert summaries[-1]["summary_version"] >= 2
    assert "total_score" in summaries[-1]["conflict_arbitration"]


def test_round_summary_payload_includes_conflict_and_metrics_fields(tmp_path):
    store = TraceStore(tmp_path)

    summary = store._round_summary_payload(
        {
            "session_id": "session-1",
            "recorded_at": "2026-04-13T00:00:00Z",
            "recorded_date": "2026-04-13",
            "round_id": 7,
            "scenario": "chat",
            "mode": "interactive",
            "sampled_action": "plan",
            "cause_type": "endogenous",
            "state_snapshot": {
                "budget_remaining": 0.72,
                "safe_mode": True,
                "focus_lock_count": 2,
                "mood": 0.44,
                "conflict_learning_state": {
                    "adjustment_reasons": {"repair": 2},
                    "last_learning_signal": {"delta": 0.2},
                },
            },
            "conflict_arbitration": {
                "total_score": 0.61,
                "components": {"identity": 0.3},
                "critical_conflict": True,
                "critical_conflict_streak": 3,
                "winning_priority": "identity",
                "circuit_breaker": {"hot_rounds_remaining": 2},
                "compromise": {"template": "soften"},
                "repair_mode": "repair",
                "repair_state_snapshot": {"stage": "stabilizing"},
                "post_error_adjustment": {"sigma": -0.1},
                "repair_ledger_tail": [{"reason": "identity_repair"}],
                "conflict_safe_mode_owned": True,
            },
        }
    )

    assert summary["summary_version"] >= 2
    assert summary["cause_type"] == "endogenous"
    assert summary["state_snapshot"]["budget_remaining"] == 0.72
    assert summary["state_snapshot"]["conflict_learning_state"]["adjustment_reasons"] == {"repair": 2}
    assert summary["conflict_arbitration"]["total_score"] == 0.61
    assert summary["conflict_arbitration"]["components"] == {"identity": 0.3}
    assert summary["conflict_arbitration"]["critical_conflict_streak"] == 3
    assert summary["conflict_arbitration"]["compromise"]["template"] == "soften"
    assert summary["conflict_arbitration"]["repair_state_snapshot"]["stage"] == "stabilizing"
    assert summary["conflict_arbitration"]["repair_ledger_summary"] == {"entries": 1, "latest_reason": "identity_repair"}
    assert summary["conflict_arbitration"]["conflict_safe_mode_owned"] is True


def test_list_round_summaries_rebuilds_stale_summary_versions(tmp_path):
    store = TraceStore(tmp_path)
    payload = {
        "session_id": "session-1",
        "recorded_at": "2026-04-13T00:00:00Z",
        "recorded_date": "2026-04-13",
        "round_id": 1,
        "scenario": "chat",
        "mode": "interactive",
        "sampled_action": "respond",
        "cause_type": "endogenous",
        "state_snapshot": {"budget_remaining": 0.5},
        "conflict_arbitration": {"total_score": 0.25},
    }
    store._append_dataset_rows(
        store.round_canonical_dir,
        [
            {
                "session_id": payload["session_id"],
                "recorded_at": payload["recorded_at"],
                "recorded_date": payload["recorded_date"],
                "round_id": payload["round_id"],
                "payload_json": json.dumps(payload, ensure_ascii=False, sort_keys=True),
            }
        ],
        schema=ROUND_CANONICAL_SCHEMA,
        partition_keys=("recorded_date", "round_id"),
    )
    store._append_dataset_rows(
        store.round_summary_dir,
        [
            {
                "session_id": payload["session_id"],
                "recorded_at": payload["recorded_at"],
                "recorded_date": payload["recorded_date"],
                "round_id": payload["round_id"],
                "payload_json": json.dumps(
                    {
                        "session_id": payload["session_id"],
                        "recorded_at": payload["recorded_at"],
                        "recorded_date": payload["recorded_date"],
                        "round_id": payload["round_id"],
                        "mode": payload["mode"],
                        "sampled_action": payload["sampled_action"],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            }
        ],
        schema=ROUND_SUMMARY_SCHEMA,
        partition_keys=("recorded_date", "round_id"),
    )

    refreshed = TraceStore(tmp_path).list_round_summaries()

    assert refreshed[-1]["summary_version"] >= 2
    assert refreshed[-1]["cause_type"] == "endogenous"
    assert refreshed[-1]["conflict_arbitration"]["total_score"] == 0.25


def test_command_subjectivity_totals_read_compact_file_without_full_command_scan(tmp_path, monkeypatch):
    store = TraceStore(tmp_path)
    store.command_subjectivity_totals_path.write_text(
        json.dumps(
            {
                "summary_version": 1,
                "total_commands": 3,
                "boundary_violation_count": 1,
                "external_count": 2,
                "internal_count": 1,
                "last_recorded_at": "2026-04-13T00:00:00Z",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        store,
        "_ensure_command_cache_loaded",
        lambda: (_ for _ in ()).throw(AssertionError("command_subjectivity_totals should not full-load command payloads when compact totals exist")),
    )

    totals = store.command_subjectivity_totals()

    assert totals["total_commands"] == 3
    assert totals["boundary_violation_count"] == 1
    assert totals["external_count"] == 2
    assert totals["internal_count"] == 1


def test_command_subjectivity_totals_persist_and_update_incrementally(tmp_path):
    store = TraceStore(tmp_path)
    store.append_command(
        "state show",
        CommandResult(applied=True, scope="runtime", delta={}, operator_level="read_only", cause_type="external_stimulus"),
        "before-1",
        "after-1",
        session_id="session-1",
        recorded_at="2026-04-13T00:00:00Z",
        sync=True,
    )
    store.append_command(
        "endogenous tick",
        CommandResult(
            applied=True,
            scope="runtime",
            delta={},
            operator_level="debug_control",
            cause_type="endogenous",
            violation_code="boundary_violation",
        ),
        "before-2",
        "after-2",
        session_id="session-1",
        recorded_at="2026-04-13T00:00:01Z",
        sync=True,
    )

    payload = json.loads(store.command_subjectivity_totals_path.read_text(encoding="utf-8"))
    reloaded = TraceStore(tmp_path).command_subjectivity_totals()

    assert payload["total_commands"] == 2
    assert payload["boundary_violation_count"] == 1
    assert payload["external_count"] == 1
    assert payload["internal_count"] == 1
    assert reloaded == payload


def test_round_subjectivity_totals_read_compact_file_without_loading_round_history(tmp_path, monkeypatch):
    store = TraceStore(tmp_path)
    store.round_subjectivity_totals_path.write_text(
        json.dumps(
            {
                "summary_version": 1,
                "total_rounds": 4,
                "external_round_count": 1,
                "endogenous_round_count": 3,
                "last_recorded_at": "2026-04-13T00:00:00Z",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        store,
        "_ensure_round_summary_loaded",
        lambda: (_ for _ in ()).throw(AssertionError("round_subjectivity_totals should not load round summaries when compact totals exist")),
    )

    totals = store.round_subjectivity_totals()

    assert totals["total_rounds"] == 4
    assert totals["external_round_count"] == 1
    assert totals["endogenous_round_count"] == 3


def test_round_subjectivity_totals_persist_and_update_incrementally(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    controller.tick(
        RoundEvent(source="user", content="第一轮外源输入", target="user", cue="one"),
        scenario="chat",
        mode="interactive",
    )
    controller.run_endogenous_tick(trigger="idle", mode="endogenous_light")
    controller.flush_pending_io(raise_on_error=True)

    store = controller.trace_store
    payload = json.loads(store.round_subjectivity_totals_path.read_text(encoding="utf-8"))
    reloaded = TraceStore(tmp_path / ".alive").round_subjectivity_totals()

    assert payload["total_rounds"] >= 2
    assert payload["external_round_count"] >= 1
    assert payload["endogenous_round_count"] >= 1
    assert reloaded == payload


def test_canonical_probability_field_payload_ignores_top_level_trace_state() -> None:
    payload = {
        "round_id": 7,
        "sampled_action": "plan",
        "token_state": {"step_index": 9},
        "couplings": [{"source_layer": "context", "target_layer": "action", "carrier_signal": "goal_pull"}],
        "probability_field": {
            "action": {"winner_target": "respond", "winner_posterior": {"respond": 0.8, "plan": 0.2}},
            "token_state": {"step_index": 2},
            "couplings": [{"source_layer": "memory", "target_layer": "action", "carrier_signal": "cue_bias"}],
        },
    }

    probability_field = canonical_probability_field_payload(payload)

    assert probability_field == payload["probability_field"]
    assert probability_field["token_state"] == {"step_index": 2}
    assert probability_field["couplings"] == [{"source_layer": "memory", "target_layer": "action", "carrier_signal": "cue_bias"}]


def _write_legacy_round_trace_row(trace_dir: Path, recorded_date: str, row: dict[str, object]) -> None:
    partition_dir = trace_dir / f"recorded_date={recorded_date}"
    partition_dir.mkdir(parents=True, exist_ok=True)
    output_path = partition_dir / "legacy.parquet"
    conn = duckdb.connect()
    try:
        conn.execute(
            """
            create table legacy_round_trace (
                session_id varchar,
                recorded_at varchar,
                recorded_date varchar,
                round_id bigint,
                stage varchar,
                agent_name varchar,
                action varchar,
                distribution_state_json varchar,
                probability_field_json varchar,
                token_state_json varchar,
                conflict_arbitration_json varchar
            )
            """
        )
        conn.execute(
            """
            insert into legacy_round_trace values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                row["session_id"],
                row["recorded_at"],
                row["recorded_date"],
                row["round_id"],
                row["stage"],
                row["agent_name"],
                row["action"],
                row["distribution_state_json"],
                row["probability_field_json"],
                row["token_state_json"],
                row["conflict_arbitration_json"],
            ],
        )
        conn.execute("copy legacy_round_trace to ? (format parquet)", [str(output_path)])
    finally:
        conn.close()


def test_round_trace_rows_use_canonical_probability_field_columns(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    result = controller.tick(
        RoundEvent(
            source="user",
            content="Please help me plan dinner and remember tea.",
            target="user",
            cue="tea",
            valence=0.1,
        ),
        scenario="task",
        mode="interactive",
    )
    controller.flush_pending_io(raise_on_error=True)

    rows = read_dataset_rows(
        controller.trace_store.round_trace_dir,
        (
            "select probability_field_json, token_state_json, conflict_arbitration_json, motivation_pool_json "
            "from read_parquet(?) where round_id = ? order by stage, action limit 1"
        ),
        [result.round_id],
    )

    assert len(rows) == 1
    row = rows[0]
    assert json.loads(row["probability_field_json"]) == result.trace.probability_field
    assert json.loads(row["token_state_json"]) == result.trace.probability_field["token_state"]
    assert json.loads(row["conflict_arbitration_json"])["winner_peak_posterior"]
    assert isinstance(json.loads(row["motivation_pool_json"]), dict)

    conn = duckdb.connect()
    try:
        cursor = conn.execute(
            "select * from read_parquet(?) where round_id = ? limit 0",
            [str(controller.trace_store.round_trace_dir / "**" / "*.parquet"), result.round_id],
        )
        columns = [item[0] for item in cursor.description]
    finally:
        conn.close()

    assert "distribution_state_json" not in columns


def test_read_round_rewrites_legacy_canonical_parquet_to_canonical_payload_and_columns(tmp_path):
    store = TraceStore(tmp_path)
    legacy_payload = {
        "session_id": "legacy-session",
        "recorded_at": "2026-04-06T00:00:00Z",
        "recorded_date": "2026-04-06",
        "round_id": 1,
        "sampled_action": "plan",
        "style_profile": {},
        "state_snapshot": {"mode": "interactive", "safe_mode": False, "focus": "plan", "budget_remaining": 0.8},
        "top_drivers": [],
        "proposal_summaries": [
            {
                "stage": "pfc",
                "agent_name": "PFCAgent",
                "top_action": "plan",
                "selected": True,
                "confidence": 0.8,
                "sigma_scale": 1.0,
                "weight_applied": 1.0,
                "delta_p": {"plan": 0.55},
                "tags": ["legacy"],
            }
        ],
        "distribution_state": {
            "u_base": {"plan": 0.4, "respond": 0.1},
            "u_shifted": {"plan": 1.2, "respond": 0.2},
            "p_base": {"plan": 0.6, "respond": 0.4},
            "p_final": {"plan": 0.75, "respond": 0.25},
            "p_raw": {"plan": 0.75, "respond": 0.25},
            "gate": {"plan": 0.91},
            "ci": {"plan": 0.2},
            "risk_suppressor": {"respond": 0.05},
            "conflict": {"passes": []},
        },
        "gate_decisions": [],
    }
    store._append_dataset_rows(
        store.round_canonical_dir,
        [
            {
                "session_id": legacy_payload["session_id"],
                "recorded_at": legacy_payload["recorded_at"],
                "recorded_date": legacy_payload["recorded_date"],
                "round_id": legacy_payload["round_id"],
                "payload_json": json.dumps(legacy_payload, ensure_ascii=False, sort_keys=True),
            }
        ],
        schema=ROUND_CANONICAL_SCHEMA,
        partition_keys=("recorded_date", "round_id"),
    )
    _write_legacy_round_trace_row(
        store.round_trace_dir,
        legacy_payload["recorded_date"],
        {
            "session_id": legacy_payload["session_id"],
            "recorded_at": legacy_payload["recorded_at"],
            "recorded_date": legacy_payload["recorded_date"],
            "round_id": legacy_payload["round_id"],
            "stage": "legacy",
            "agent_name": "LegacyAgent",
            "action": "plan",
            "distribution_state_json": json.dumps(legacy_payload["distribution_state"], ensure_ascii=False, sort_keys=True),
            "probability_field_json": "{}",
            "token_state_json": "{}",
            "conflict_arbitration_json": "{}",
        },
    )

    rewrite_payload = store.rewrite_legacy_round_parquet_history()
    read_back = store.read_round(1)
    listed = store.list_rounds()

    assert rewrite_payload["rewritten"] is True
    assert rewrite_payload["legacy_payload_rows"] == 1
    assert rewrite_payload["removed_trace_columns"] == ["distribution_state_json"]
    assert read_back["probability_field"]["action"]["winner_posterior"] == {"plan": 0.75, "respond": 0.25}
    assert read_back["probability_field"]["action"]["final_energy"] == {"plan": 1.2, "respond": 0.2}
    assert read_back["probability_field"]["action"]["winner_target"] == "plan"
    assert "distribution_state" not in read_back
    assert listed[0]["probability_field"]["action"]["winner_posterior"] == {"plan": 0.75, "respond": 0.25}
    assert "distribution_state" not in listed[0]

    conn = duckdb.connect()
    try:
        canonical_payload = json.loads(
            conn.execute(
                "select payload_json from read_parquet(?) where round_id = ? limit 1",
                [str(store.round_canonical_dir / "**" / "*.parquet"), legacy_payload["round_id"]],
            ).fetchone()[0]
        )
        cursor = conn.execute(
            "select * from read_parquet(?) where round_id = ? limit 0",
            [str(store.round_trace_dir / "**" / "*.parquet"), legacy_payload["round_id"]],
        )
        trace_columns = [item[0] for item in cursor.description]
    finally:
        conn.close()

    assert "distribution_state" not in canonical_payload
    assert canonical_payload["probability_field"]["action"]["winner_posterior"] == {"plan": 0.75, "respond": 0.25}
    assert "distribution_state_json" not in trace_columns


def test_trace_store_prewarms_recent_signal_views_and_updates_storage_status(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    for index, cue in enumerate(("tea", "coffee", "walk"), start=1):
        controller.tick(
            RoundEvent(
                source="user",
                content=f"remember {cue} and plan step {index}",
                target="user",
                cue=cue,
                valence=0.1,
            ),
            scenario="task",
            mode="interactive",
        )
    controller.flush_pending_io(raise_on_error=True)

    store = TraceStore(tmp_path / ".alive")
    warmed = store.prewarm_recent_views(limit=2)
    views = store.recent_round_signal_views(limit=2)
    status = store.trace_storage_status()

    assert warmed["round_count"] == 2
    assert len(views) == 2
    assert status["signal_view_cache_size"] >= 2
    assert status["signal_view_cache_limit"] >= 2


def test_read_round_record_uses_round_cache_before_parquet_lookup(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    result = controller.tick(
        RoundEvent(
            source="user",
            content="Please keep the latest round cached.",
            target="user",
            cue="cached",
        ),
        scenario="chat",
        mode="interactive",
    )
    controller.flush_pending_io(raise_on_error=True)
    store = controller.trace_store

    monkeypatch.setattr(
        store,
        "_read_round_from_parquet",
        lambda round_id: (_ for _ in ()).throw(AssertionError("parquet lookup should not run on cache hit")),
    )

    payload, read_source = store.read_round_record(result.round_id)

    assert payload["round_id"] == result.round_id
    assert read_source == "memory_cache"


def test_trace_storage_status_reuses_cached_parquet_live_ready(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    controller.tick(
        RoundEvent(
            source="user",
            content="Create a trace row so parquet is live.",
            target="user",
            cue="live",
        ),
        scenario="chat",
        mode="interactive",
    )
    controller.flush_pending_io(raise_on_error=True)
    store = controller.trace_store

    baseline = store.trace_storage_status()
    assert baseline["parquet_live_ready"] is True

    original_rglob = Path.rglob

    def guarded_rglob(self, pattern):
        if self == store.round_canonical_dir and pattern == "*.parquet":
            raise AssertionError("trace_storage_status should not rescan parquet directories")
        return original_rglob(self, pattern)

    monkeypatch.setattr(Path, "rglob", guarded_rglob)

    repeated = store.trace_storage_status()

    assert repeated["parquet_live_ready"] is True


def test_cached_round_reads_skip_repeated_legacy_parquet_scan(tmp_path, monkeypatch):
    payload = {
        "session_id": "session-1",
        "recorded_at": "2026-04-08T00:00:00Z",
        "recorded_date": "2026-04-08",
        "round_id": 1,
        "scenario": "task",
        "mode": "interactive",
        "sampled_action": "plan",
        "probability_field": {
            "action": {
                "winner_target": "plan",
                "winner_posterior": {"plan": 0.8, "respond": 0.2},
            },
            "token_state": {},
            "couplings": [],
        },
        "state_snapshot": {"mode": "interactive", "safe_mode": False, "budget_remaining": 0.8},
        "proposal_summaries": [],
        "gate_decisions": [],
    }
    seed_store = TraceStore(tmp_path)
    seed_store._append_dataset_rows(
        seed_store.round_canonical_dir,
        [
            {
                "session_id": payload["session_id"],
                "recorded_at": payload["recorded_at"],
                "recorded_date": payload["recorded_date"],
                "round_id": payload["round_id"],
                "payload_json": json.dumps(payload, ensure_ascii=False, sort_keys=True),
            }
        ],
        schema=ROUND_CANONICAL_SCHEMA,
        partition_keys=("recorded_date", "round_id"),
    )

    store = TraceStore(tmp_path)
    scan_count = 0
    original_scan = store._raw_round_payloads_from_parquet

    def counted_scan():
        nonlocal scan_count
        scan_count += 1
        return original_scan()

    monkeypatch.setattr(store, "_raw_round_payloads_from_parquet", counted_scan)

    assert store.read_round(1)["round_id"] == 1
    assert store.list_rounds()[0]["round_id"] == 1
    assert scan_count == 0


def test_list_rounds_round_count_cache_avoids_repeated_parquet_scan(tmp_path, monkeypatch):
    store = TraceStore(tmp_path)
    store._round_cache = {
        1: {
            "session_id": "session-1",
            "recorded_at": "2026-04-12T00:00:00Z",
            "recorded_date": "2026-04-12",
            "round_id": 1,
            "probability_field": {},
        },
        2: {
            "session_id": "session-1",
            "recorded_at": "2026-04-12T00:00:01Z",
            "recorded_date": "2026-04-12",
            "round_id": 2,
            "probability_field": {},
        },
    }
    store._round_cache_complete = False
    store._trace_sync_status_cache["storage_state"] = "healthy"
    store._round_count_cache_ttl = 60.0

    call_count = 0

    def counted_round_count():
        nonlocal call_count
        call_count += 1
        return 1

    monkeypatch.setattr(store, "_round_count_from_parquet", counted_round_count)

    first = store.list_rounds()
    second = store.list_rounds()

    assert [row["round_id"] for row in first] == [1, 2]
    assert [row["round_id"] for row in second] == [1, 2]
    assert call_count == 1


def test_export_parquet_rewrites_legacy_distribution_state_to_canonical_columns(tmp_path):
    store = TraceStore(tmp_path)
    legacy_payload = {
        "session_id": "legacy-session",
        "recorded_at": "2026-04-06T00:00:00Z",
        "recorded_date": "2026-04-06",
        "round_id": 1,
        "scenario": "task",
        "mode": "interactive",
        "sampled_action": "plan",
        "style_profile": {},
        "state_snapshot": {"mode": "interactive", "safe_mode": False, "focus": "plan", "budget_remaining": 0.8},
        "top_drivers": [],
        "proposal_summaries": [
            {
                "stage": "pfc",
                "agent_name": "PFCAgent",
                "top_action": "plan",
                "selected": True,
                "confidence": 0.8,
                "sigma_scale": 1.0,
                "weight_applied": 1.0,
                "delta_p": {"plan": 0.55},
                "tags": ["legacy"],
            }
        ],
        "distribution_state": {
            "u_shifted": {"plan": 1.2, "respond": 0.2},
            "p_final": {"plan": 0.75, "respond": 0.25},
            "p_raw": {"plan": 0.75, "respond": 0.25},
            "conflict": {"passes": []},
        },
        "gate_decisions": [],
    }
    store._append_dataset_rows(
        store.round_canonical_dir,
        [
            {
                "session_id": legacy_payload["session_id"],
                "recorded_at": legacy_payload["recorded_at"],
                "recorded_date": legacy_payload["recorded_date"],
                "round_id": legacy_payload["round_id"],
                "payload_json": json.dumps(legacy_payload, ensure_ascii=False, sort_keys=True),
            }
        ],
        schema=ROUND_CANONICAL_SCHEMA,
        partition_keys=("recorded_date", "round_id"),
    )

    exported = TraceExporter(store).export_parquet(overwrite=True)
    round_path = exported["tables"]["round_trace"]["path"]

    conn = duckdb.connect()
    try:
        row = conn.execute("select probability_field_json from read_parquet(?) limit 1", [round_path]).fetchone()
        cursor = conn.execute("select * from read_parquet(?) limit 0", [round_path])
        columns = [item[0] for item in cursor.description]
    finally:
        conn.close()

    assert row is not None
    (probability_field_json,) = row
    assert json.loads(probability_field_json)["action"]["winner_posterior"] == {"plan": 0.75, "respond": 0.25}
    assert "distribution_state_json" not in columns
