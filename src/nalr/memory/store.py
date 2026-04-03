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
        self.episodic_hot_dir = self.memory_dir / "episodic_hot"
        self.episodic_warm_dir = self.memory_dir / "episodic_warm"
        self.episodic_archive_dir = self.memory_dir / "episodic_archive"
        self.relation_dir = self.memory_dir / "relation"
        self.habit_dir = self.memory_dir / "habit"
        self.raw_dir = self.memory_dir / "raw"
        self.schema_dir = self.memory_dir / "schema"
        for path in (
            self.episodic_hot_dir,
            self.episodic_warm_dir,
            self.episodic_archive_dir,
            self.relation_dir,
            self.habit_dir,
            self.raw_dir,
            self.schema_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
        self.episodic_path = self.memory_dir / "episodic_hot.json"
        self.episodic_warm_path = self.memory_dir / "episodic_warm.json"
        self.episodic_archive_path = self.memory_dir / "episodic_archive.json"
        self.habit_path = self.memory_dir / "habit.json"
        self.relation_path = self.memory_dir / "relation.json"
        self.relation_trace_path = self.memory_dir / "relation_trace.json"
        self.stable_priors_path = self.memory_dir / "stable_priors.json"
        self.circuit_breaker_path = self.memory_dir / "circuit_breakers.json"
        for path in (
            self.episodic_path,
            self.episodic_warm_path,
            self.episodic_archive_path,
            self.habit_path,
            self.relation_path,
            self.relation_trace_path,
            self.stable_priors_path,
            self.circuit_breaker_path,
        ):
            if not path.exists():
                path.write_text("[]", encoding="utf-8")

    def _read_list(self, path: Path) -> list[dict]:
        return json.loads(path.read_text(encoding="utf-8"))

    def _write_list(self, path: Path, payload: list[dict]) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _decay_memories(self, active_cue: str | None) -> None:
        memories = self._read_list(self.episodic_path)
        for memory in memories:
            cue_match = memory["cue"] == active_cue
            gist_decay = 0.008 if cue_match else 0.035
            detail_decay = 0.015 if cue_match else 0.065
            memory["gist_strength"] = max(0.0, round(memory.get("gist_strength", 0.0) * (1 - gist_decay), 4))
            memory["detail_strength"] = max(0.0, round(memory.get("detail_strength", 0.0) * (1 - detail_decay), 4))
        self._write_list(self.episodic_path, memories)

    def _interference(self, memories: list[dict], cue: str) -> float:
        neighbors = [item for item in memories if item["cue"] != cue and item["cue"][:1] == cue[:1]]
        return min(0.35, round(len(neighbors) * 0.06, 4))

    def _update_stable_priors(self, cue: str, gist_strength: float) -> None:
        priors = self._read_list(self.stable_priors_path)
        prior = next((item for item in priors if item["cue"] == cue), None)
        if prior is None:
            prior = {"cue": cue, "weight": 0.0}
            priors.append(prior)
        prior["weight"] = min(1.0, round(prior["weight"] * 0.92 + gist_strength * 0.18, 4))
        self._write_list(self.stable_priors_path, priors)

    def ingest_event(self, event: RoundEvent) -> str | None:
        cue = _derive_cue(event)
        self._decay_memories(cue)
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
                    "interference": 0.0,
                }
                memories.append(matched)
            matched["count"] += 1
            matched["gist_strength"] = min(1.0, round(matched["gist_strength"] + 0.10, 4))
            matched["detail_strength"] = min(1.0, round(matched["detail_strength"] + 0.18, 4))
            matched["last_content"] = event.content
            matched["interference"] = self._interference(memories, cue)
            self._write_list(self.episodic_path, memories)
            self._sync_memory_tiers(matched)
            self._update_stable_priors(cue, matched["gist_strength"])

            habits = self._read_list(self.habit_path)
            habit = next((item for item in habits if item["pattern"] == cue), None)
            if habit is None:
                habit = {"pattern": cue, "strength": 0.0, "count": 0}
                habits.append(habit)
            habit["count"] += 1
            decay_h = 0.04
            eta_rep = 0.05
            eta_pos = 0.04
            eta_neg = 0.06
            breakthrough_cap = 0.92
            valence_push = max(0.0, event.valence) * eta_pos - max(0.0, -event.valence) * eta_neg
            habit["strength"] = min(
                breakthrough_cap,
                max(0.0, round(habit["strength"] * (1 - decay_h) + eta_rep + valence_push, 4)),
            )
            self._write_list(self.habit_path, habits)

        if event.target:
            relations = self._read_list(self.relation_path)
            relation = next((item for item in relations if item["target"] == event.target), None)
            if relation is None:
                relation = {"target": event.target, "closeness": 0.5}
                relations.append(relation)
            relation["closeness"] = min(1.0, max(0.0, relation["closeness"] + (event.valence * 0.08)))
            self._write_list(self.relation_path, relations)
            traces = self._read_list(self.relation_trace_path)
            traces.append({"target": event.target, "valence": event.valence, "content": event.content})
            self._write_list(self.relation_trace_path, traces)

        return cue

    def memory_top(self, limit: int = 5) -> list[dict]:
        memories = []
        memories.extend({**item, "tier": "hot"} for item in self._read_list(self.episodic_path))
        memories.extend({**item, "tier": "warm"} for item in self._read_list(self.episodic_warm_path))
        memories.extend({**item, "tier": "archive"} for item in self._read_list(self.episodic_archive_path))
        memories = sorted(memories, key=lambda item: (item.get("detail_strength", 0.0), item.get("count", 0)), reverse=True)
        return memories[:limit]

    def habit_top(self, limit: int = 5) -> list[dict]:
        habits = sorted(self._read_list(self.habit_path), key=lambda item: item["strength"], reverse=True)
        return habits[:limit]

    def recall_strength(self, cue: str | None) -> float:
        if not cue:
            return 0.0
        for item in self._read_list(self.episodic_path):
            if item["cue"] == cue:
                detail_strength = item.get("detail_strength", 0.0)
                gist_strength = item.get("gist_strength", 0.0)
                interference = item.get("interference", 0.0)
                if detail_strength >= 0.50:
                    return max(0.0, round(detail_strength * (1 - interference * 0.5), 4))
                return max(0.0, round(gist_strength * (1 - interference * 0.3), 4))
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

    def _sync_memory_tiers(self, memory: dict) -> None:
        if memory["count"] >= 4:
            warm = self._read_list(self.episodic_warm_path)
            warm_item = next((item for item in warm if item["cue"] == memory["cue"]), None)
            if warm_item is None:
                warm_item = {"cue": memory["cue"]}
                warm.append(warm_item)
            warm_item.update(memory)
            self._write_list(self.episodic_warm_path, warm)
        if memory["count"] >= 8:
            archive = self._read_list(self.episodic_archive_path)
            archive_item = next((item for item in archive if item["cue"] == memory["cue"]), None)
            if archive_item is None:
                archive_item = {"cue": memory["cue"]}
                archive.append(archive_item)
            archive_item.update(memory)
            self._write_list(self.episodic_archive_path, archive)
