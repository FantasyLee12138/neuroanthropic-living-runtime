from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


CONFIG_ROOT = Path(__file__).resolve().parents[2] / "config"


def test_dream_bridge_stdio_returns_dream_result(tmp_path):
    env = os.environ.copy()
    env["NALR_HOME"] = str(tmp_path / ".alive")
    env["NALR_CONFIG_DIR"] = str(CONFIG_ROOT)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "src")

    process = subprocess.Popen(
        [sys.executable, "-m", "nalr.dream.bridge"],
        cwd=tmp_path,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        assert process.stdin is not None
        assert process.stdout is not None
        payload = {
            "type": "dream.start",
            "request": {
                "snapshot": {
                    "sleep_session_id": "sleep-1",
                    "mode": "sleep",
                    "trigger": "sleep_full",
                    "memory_refs": {"hot": ["tea"], "warm": [], "archive_sample_pool": []},
                    "resource_state": {"daily_token_surplus_rate": 0.6, "fatigue_level": 0.2, "resource_scarcity": 0.1},
                    "emotional_baseline": {"valence": 0.1, "arousal": 0.2, "dominant_emotion": "平静"},
                    "relationship_state": {"closeness": 0.6, "trust": 0.58, "boundary_tension": 0.12},
                    "identity_evidence_summary": {"anchors": ["cue:tea", "habit:tea"]},
                    "conflict_residue": {"critical_conflict": False},
                },
                "budget": {"max_tokens": 1200, "allow_identity_drift": True},
                "policy": {"allowed_types": ["memory_consolidation", "emotion_adjustments", "habit_adjustments", "dream_memory_write", "relationship_adjustments", "limited_identity_drift"]},
                "trace_context": {"session_id": "sess-1", "round_id": 2},
            },
        }
        process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        process.stdin.flush()
        line = process.stdout.readline()
    finally:
        process.terminate()
        process.wait(timeout=5)

    response = json.loads(line)
    result = response["result"]

    assert response["type"] == "dream.result"
    assert result["trace"]["dream_run_id"].startswith("dream-")
    assert result["trace"]["trigger"] == "sleep_full"
    assert result["proposal_bundle"]["memory_consolidation"]
    assert "limited_identity_drift" in result["trace"]["evaluated_types"]
