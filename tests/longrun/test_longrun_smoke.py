from pathlib import Path

from nalr.runtime.controller import RuntimeController
from nalr.schemas.models import RoundEvent


CONFIG_ROOT = Path(__file__).resolve().parents[2] / "config"


def test_longrun_smoke_keeps_traces_and_runtime_alive(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    for idx in range(1000):
        controller.tick(
            RoundEvent(
                source="user",
                content=f"round {idx} task update",
                target="user",
                valence=0.05,
            ),
            scenario="task",
            mode="interactive",
        )

    state = controller.load_runtime_state()
    assert state.round_count == 1000
    assert state.safe_mode is False
