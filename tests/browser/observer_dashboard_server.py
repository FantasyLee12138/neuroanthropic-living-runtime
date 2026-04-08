from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import uvicorn


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "config"
DEFAULT_PORT = 8765
FIXTURE_ROOT = Path("/tmp/nalr_observer_browser_fixture")
for path in (REPO_ROOT, REPO_ROOT / "src"):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

from nalr.runtime.controller import RuntimeController
from nalr.schemas.models import RoundEvent
from services.observer.api.app import create_app


def seed_fixture() -> Path:
    if FIXTURE_ROOT.exists():
        shutil.rmtree(FIXTURE_ROOT)
    FIXTURE_ROOT.mkdir(parents=True, exist_ok=True)

    controller = RuntimeController(project_root=FIXTURE_ROOT, config_root=CONFIG_ROOT)
    state = controller.load_runtime_state()
    state.body_energy = 0.11
    state.fatigue = 0.88
    state.memory_fragments = 0.79
    state.self_continuity = 0.31
    state.meaning_strength = 0.22
    state.subjective_state.felt = ["累", "想停下", "先往里收"]
    state.subjective_state.spontaneous = 0.77
    state.subjective_state.reject_all = 0.68
    state.subjective_state.boundary = 0.72
    state.subjective_state.meaning_made = ["先吸收内部线索，再决定如何表达"]
    controller._save_state(state, sync=True)

    rounds = [
        RoundEvent(source="user", content="先别急着答复，先看看内部状态。", target="user", cue="内部状态"),
        RoundEvent(source="user", content="如果不回应，会发生什么？", target="user", cue="不回应"),
        RoundEvent(source="user", content="把当前动作分布和四维空间一起给我看。", target="user", cue="动作分布"),
    ]
    for event in rounds:
        controller.tick(event, scenario="chat", mode="interactive")
    controller.flush_pending_io(raise_on_error=True)
    return FIXTURE_ROOT


def main() -> None:
    fixture_root = seed_fixture()
    app = create_app(project_root=fixture_root, config_root=CONFIG_ROOT)
    port = int(os.environ.get("NALR_OBSERVER_BROWSER_PORT", str(DEFAULT_PORT)) or DEFAULT_PORT)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
