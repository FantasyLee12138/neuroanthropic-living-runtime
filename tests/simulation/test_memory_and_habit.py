from pathlib import Path

from nalr.runtime.controller import RuntimeController
from nalr.schemas.models import RoundEvent


CONFIG_ROOT = Path(__file__).resolve().parents[2] / "config"


def test_memory_recall_and_habit_strength_grow_with_repetition(tmp_path):
    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    for _ in range(4):
        controller.tick(
            RoundEvent(
                source="user",
                content="Remember that coffee helps me focus in the morning.",
                target="user",
                cue="coffee",
                valence=0.3,
            ),
            scenario="companion",
            mode="interactive",
        )

    memory_items = controller.memory_top(limit=5)
    habit_items = controller.habit_top(limit=5)

    assert memory_items
    assert memory_items[0]["cue"] == "coffee"
    assert memory_items[0]["detail_strength"] >= memory_items[0]["gist_strength"]
    assert habit_items
    assert habit_items[0]["pattern"] == "coffee"
    assert habit_items[0]["strength"] > 0.1

