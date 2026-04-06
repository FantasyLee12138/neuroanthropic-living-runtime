import json
from pathlib import Path

import duckdb
from typer.testing import CliRunner

from nalr.cli.app import app
from nalr.runtime.controller import RuntimeController
from nalr.schemas.models import RoundEvent


RUNNER = CliRunner()
CONFIG_ROOT = Path(__file__).resolve().parents[2] / "config"


def _write_legacy_round_parquet(home_path: Path) -> None:
    canonical_dir = home_path / "traces" / "parquet" / "round_canonical" / "recorded_date=2026-04-06" / "round_id=1"
    trace_dir = home_path / "traces" / "parquet" / "round_trace" / "recorded_date=2026-04-06"
    canonical_dir.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)
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

    conn = duckdb.connect()
    try:
        conn.execute("create table legacy_round_canonical (session_id varchar, recorded_at varchar, recorded_date varchar, round_id bigint, payload_json varchar)")
        conn.execute(
            "insert into legacy_round_canonical values (?, ?, ?, ?, ?)",
            [
                legacy_payload["session_id"],
                legacy_payload["recorded_at"],
                legacy_payload["recorded_date"],
                legacy_payload["round_id"],
                json.dumps(legacy_payload, ensure_ascii=False, sort_keys=True),
            ],
        )
        conn.execute("copy legacy_round_canonical to ? (format parquet)", [str(canonical_dir / "legacy.parquet")])
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
            "insert into legacy_round_trace values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                legacy_payload["session_id"],
                legacy_payload["recorded_at"],
                legacy_payload["recorded_date"],
                legacy_payload["round_id"],
                "pfc",
                "PFCAgent",
                "plan",
                json.dumps(legacy_payload["distribution_state"], ensure_ascii=False, sort_keys=True),
                "{}",
                "{}",
                "{}",
            ],
        )
        conn.execute("copy legacy_round_trace to ? (format parquet)", [str(trace_dir / "legacy.parquet")])
    finally:
        conn.close()


def test_trace_rows_include_session_and_recorded_fields(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    controller.tick(
        RoundEvent(
            source="user",
            content="Help me remember tea and plan tomorrow.",
            target="user",
            cue="tea",
            valence=0.2,
        ),
        scenario="task",
        mode="interactive",
    )
    controller.apply_command("safe on")
    controller.flush_pending_io(raise_on_error=True)

    state = controller.load_runtime_state()
    round_payload = controller.trace_round(1)
    skill_rows = controller.trace_store.list_skill_traces()
    command_rows = [
        json.loads(line)
        for line in controller.trace_store.commands_jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert round_payload["session_id"] == state.session_id
    assert round_payload["recorded_at"]
    assert round_payload["recorded_date"] == round_payload["recorded_at"][:10]
    assert skill_rows[0]["session_id"] == state.session_id
    assert skill_rows[0]["recorded_date"] == skill_rows[0]["recorded_at"][:10]
    assert command_rows[0]["session_id"] == state.session_id
    assert command_rows[0]["recorded_date"] == command_rows[0]["recorded_at"][:10]
    assert command_rows[0]["command_id"]
    assert command_rows[0]["canonical"] == "safe on"
    assert command_rows[0]["parsed_args"] == {}


def test_cli_trace_export_parquet_writes_three_tables(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    controller.tick(
        RoundEvent(
            source="user",
            content="Help me remember noodles and plan dinner.",
            target="user",
            cue="noodles",
            valence=0.2,
        ),
        scenario="task",
        mode="interactive",
    )
    controller.flush_pending_io(raise_on_error=True)
    controller.apply_command("safe on")

    monkeypatch.setenv("NALR_HOME", str(tmp_path / ".alive"))
    monkeypatch.setenv("NALR_CONFIG_DIR", str(CONFIG_ROOT))

    result = RUNNER.invoke(app, ["trace", "export", "parquet", "--overwrite"])

    assert result.exit_code == 0

    payload = json.loads(result.stdout)
    parquet_dir = tmp_path / ".alive" / "traces" / "parquet"
    round_path = parquet_dir / "round_trace.parquet"
    skill_path = parquet_dir / "skill_trace.parquet"
    command_path = parquet_dir / "command_trace.parquet"

    assert payload["parquet_dir"] == str(parquet_dir)
    assert payload["mode"] == "rebuild"
    assert round_path.exists()
    assert skill_path.exists()
    assert command_path.exists()

    conn = duckdb.connect()
    try:
        round_count = conn.execute("select count(*) from read_parquet(?)", [str(round_path)]).fetchone()[0]
        skill_count = conn.execute("select count(*) from read_parquet(?)", [str(skill_path)]).fetchone()[0]
        command_count = conn.execute("select count(*) from read_parquet(?)", [str(command_path)]).fetchone()[0]
        session_id = conn.execute("select session_id from read_parquet(?) limit 1", [str(round_path)]).fetchone()[0]
    finally:
        conn.close()

    assert round_count > 0
    assert skill_count > 0
    assert command_count == 1
    assert session_id == controller.load_runtime_state().session_id


def test_cli_trace_export_parquet_rewrites_legacy_round_history(tmp_path, monkeypatch):
    home_path = tmp_path / ".alive"
    _write_legacy_round_parquet(home_path)
    monkeypatch.setenv("NALR_HOME", str(home_path))
    monkeypatch.setenv("NALR_CONFIG_DIR", str(CONFIG_ROOT))

    result = RUNNER.invoke(app, ["trace", "export", "parquet", "--rewrite-history", "--overwrite"])

    assert result.exit_code == 0

    payload = json.loads(result.stdout)
    assert payload["history_rewrite"]["rewritten"] is True
    assert payload["history_rewrite"]["round_count"] == 1

    round_trace_dataset = home_path / "traces" / "parquet" / "round_trace"
    exported_round_trace = home_path / "traces" / "parquet" / "round_trace.parquet"

    conn = duckdb.connect()
    try:
        trace_cursor = conn.execute("select * from read_parquet(?) limit 0", [str(round_trace_dataset / "**" / "*.parquet")])
        trace_columns = [item[0] for item in trace_cursor.description]
        exported_cursor = conn.execute("select * from read_parquet(?) limit 0", [str(exported_round_trace)])
        exported_columns = [item[0] for item in exported_cursor.description]
        payload_json = conn.execute(
            "select payload_json from read_parquet(?) where round_id = 1 limit 1",
            [str(home_path / "traces" / "parquet" / "round_canonical" / "**" / "*.parquet")],
        ).fetchone()[0]
    finally:
        conn.close()

    assert "distribution_state_json" not in trace_columns
    assert "distribution_state_json" not in exported_columns
    assert "distribution_state" not in json.loads(payload_json)


def test_cli_trace_export_parquet_includes_repair_ledger_table(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    state = controller.load_runtime_state()
    state.body_energy = 0.08
    state.budget_remaining = 0.04
    controller._save_state(state)
    controller.tick(
        RoundEvent(
            source="user",
            content="I need a quick easy break, but help me plan carefully, reply to Alex, and let me drift.",
            target="alex",
            cue="break",
            valence=-0.35,
            energy_delta=-0.16,
        ),
        scenario="task",
        mode="interactive",
    )
    controller.flush_pending_io(raise_on_error=True)

    monkeypatch.setenv("NALR_HOME", str(tmp_path / ".alive"))
    monkeypatch.setenv("NALR_CONFIG_DIR", str(CONFIG_ROOT))

    result = RUNNER.invoke(app, ["trace", "export", "parquet", "--overwrite"])

    assert result.exit_code == 0

    payload = json.loads(result.stdout)
    parquet_dir = tmp_path / ".alive" / "traces" / "parquet"
    repair_path = parquet_dir / "repair_ledger.parquet"

    assert repair_path.exists()
    assert payload["mode"] == "rebuild"
    assert payload["tables"]["repair_ledger"]["path"] == str(repair_path)

    conn = duckdb.connect()
    try:
        repair_count = conn.execute("select count(*) from read_parquet(?)", [str(repair_path)]).fetchone()[0]
        reason = conn.execute("select reason from read_parquet(?) limit 1", [str(repair_path)]).fetchone()[0]
        conflict_score = conn.execute("select conflict_score from read_parquet(?) limit 1", [str(repair_path)]).fetchone()[0]
    finally:
        conn.close()

    assert repair_count == 1
    assert reason == "forced_compromise"
    assert conflict_score > 0.0


def test_cli_memory_compact_and_sample_use_compacted_artifacts(tmp_path, monkeypatch):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)
    for idx in range(6):
        controller.tick(
            RoundEvent(
                source="user",
                content=f"Remember coffee cue round {idx}",
                target="user",
                cue="coffee" if idx < 4 else "coding",
                valence=0.2,
            ),
            scenario="task",
            mode="interactive",
        )
    controller.flush_pending_io(raise_on_error=True)

    monkeypatch.setenv("NALR_HOME", str(tmp_path / ".alive"))
    monkeypatch.setenv("NALR_CONFIG_DIR", str(CONFIG_ROOT))

    compact_result = RUNNER.invoke(app, ["memory", "compact"])
    sample_result = RUNNER.invoke(app, ["memory", "sample", "--tier", "hot", "--limit", "2"])

    assert compact_result.exit_code == 0
    assert sample_result.exit_code == 0

    compact_payload = json.loads(compact_result.stdout)
    sample_payload = json.loads(sample_result.stdout)

    assert compact_payload["tiers"]["hot"]["artifact_count"] > 0
    assert sample_payload
    assert all(row["tier"] == "hot" for row in sample_payload)
