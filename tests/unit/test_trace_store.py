import json
from pathlib import Path

import duckdb

from nalr.runtime.controller import RuntimeController
from nalr.schemas.models import RoundEvent
from nalr.trace.exporter import TraceExporter
from nalr.trace.store import ROUND_CANONICAL_SCHEMA, TraceStore, canonical_probability_field_payload
from nalr.storage.parquet_io import read_dataset_rows


CONFIG_ROOT = Path(__file__).resolve().parents[2] / "config"


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


def test_round_trace_rows_surface_cognitive_chain_layer_metrics_and_control_events(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    result = controller.tick(
        RoundEvent(
            source="user",
            content="Please remember tea and explain how you decided to respond.",
            target="user",
            cue="tea",
            valence=0.15,
        ),
        scenario="chat",
        mode="interactive",
    )
    controller.flush_pending_io(raise_on_error=True)

    round_payload = controller.trace_store.read_round(result.round_id)
    cognitive_chain = round_payload["cognitive_chain"]
    layer_metrics = round_payload["layer_metrics"]
    control_events = round_payload["control_events"]

    assert [item["layer"] for item in cognitive_chain] == [
        "perception",
        "memory",
        "cognition",
        "decision",
        "execution",
        "feedback",
    ]
    assert "cognitive" in layer_metrics
    assert "feedback" in layer_metrics
    assert isinstance(control_events, list)

    canonical_rows = read_dataset_rows(
        controller.trace_store.round_canonical_dir,
        "select payload_json from read_parquet(?) where round_id = ?",
        [result.round_id],
    )

    assert len(canonical_rows) == 1
    canonical_payload = json.loads(canonical_rows[0]["payload_json"])
    assert canonical_payload["cognitive_chain"] == cognitive_chain
    assert canonical_payload["layer_metrics"] == layer_metrics
    assert canonical_payload["control_events"] == control_events


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
