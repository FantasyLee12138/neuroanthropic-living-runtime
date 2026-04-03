from __future__ import annotations

import json
from pathlib import Path

from nalr.schemas.models import RoundEvent


def _derive_cue(event: RoundEvent) -> str | None:
    if event.cue:
        return event.cue.lower()
    tokens = [token.strip(".,!?").lower() for token in event.content.split()]
    for token in tokens:
        if len(token) >= 6 and token not in {"please", "remember"}:
            return token
    return None


class MemoryStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.memory_dir = self.root / "memory"
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.episodic_path = self.memory_dir / "episodic_hot.json"
        self.habit_path = self.memory_dir / "habit.json"
        self.relation_path = self.memory_dir / "relation.json"
        for path in (self.episodic_path, self.habit_path, self.relation_path):
            if not path.exists():
                path.write_text("[]", encoding="utf-8")

    def _read_list(self, path: Path) -> list[dict]:
        return json.loads(path.read_text(encoding="utf-8"))

    def _write_list(self, path: Path, payload: list[dict]) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def ingest_event(self, event: RoundEvent) -> str | None:
        cue = _derive_cue(event)
        if cue:
            memories = self._read_list(self.episodic_path)
            matched = next((item for item in memories if item["cue"] == cue), None)
            if matched is None:
                matched = {
                    "cue": cue,
                    "count": 0,
                    "gist_strength": 0.0,
                    "detail_strength": 0.0,
                    "last_content": "",
                }
                memories.append(matched)
            matched["count"] += 1
            matched["gist_strength"] = min(1.0, matched["gist_strength"] + 0.10)
            matched["detail_strength"] = min(1.0, matched["detail_strength"] + 0.18)
            matched["last_content"] = event.content
            self._write_list(self.episodic_path, memories)

            habits = self._read_list(self.habit_path)
            habit = next((item for item in habits if item["pattern"] == cue), None)
            if habit is None:
                habit = {"pattern": cue, "strength": 0.0, "count": 0}
                habits.append(habit)
            habit["count"] += 1
            habit["strength"] = min(1.0, habit["strength"] + 0.06)
            self._write_list(self.habit_path, habits)

        if event.target:
            relations = self._read_list(self.relation_path)
            relation = next((item for item in relations if item["target"] == event.target), None)
            if relation is None:
                relation = {"target": event.target, "closeness": 0.5}
                relations.append(relation)
            relation["closeness"] = min(1.0, max(0.0, relation["closeness"] + (event.valence * 0.08)))
            self._write_list(self.relation_path, relations)

        return cue

    def memory_top(self, limit: int = 5) -> list[dict]:
        memories = sorted(self._read_list(self.episodic_path), key=lambda item: item["detail_strength"], reverse=True)
        return memories[:limit]

    def habit_top(self, limit: int = 5) -> list[dict]:
        habits = sorted(self._read_list(self.habit_path), key=lambda item: item["strength"], reverse=True)
        return habits[:limit]

    def recall_strength(self, cue: str | None) -> float:
        if not cue:
            return 0.0
        for item in self._read_list(self.episodic_path):
            if item["cue"] == cue:
                return max(item["gist_strength"], item["detail_strength"])
        return 0.0

    def habit_strength(self, cue: str | None) -> float:
        if not cue:
            return 0.0
        for item in self._read_list(self.habit_path):
            if item["pattern"] == cue:
                return item["strength"]
        return 0.0

    def closeness(self, target: str | None) -> float:
        if not target:
            return 0.5
        for item in self._read_list(self.relation_path):
            if item["target"] == target:
                return item["closeness"]
        return 0.5

