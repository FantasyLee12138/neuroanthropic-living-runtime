from __future__ import annotations

import json
import hashlib
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
    def __init__(self, root: Path, hot_limit: int = 500, warm_limit: int = 2500) -> None:
        self.root = root
        self.hot_limit = hot_limit
        self.warm_limit = warm_limit
        self.memory_dir = self.root / "memory"
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.episodic_hot_path = self.memory_dir / "episodic_hot.json"
        self.episodic_warm_path = self.memory_dir / "episodic_warm.json"
        self.episodic_archive_path = self.memory_dir / "episodic_archive.json"
        self.habit_path = self.memory_dir / "habit.json"
        self.relation_path = self.memory_dir / "relation.json"
        self.raw_evidence_path = self.memory_dir / "raw_evidence.json"
        self.meta_path = self.memory_dir / "meta.json"
        for path in (
            self.episodic_hot_path,
            self.episodic_warm_path,
            self.episodic_archive_path,
            self.habit_path,
            self.relation_path,
            self.raw_evidence_path,
        ):
            if not path.exists():
                path.write_text("[]", encoding="utf-8")
        if not self.meta_path.exists():
            self.meta_path.write_text(json.dumps({"sequence": 0}, ensure_ascii=False, indent=2), encoding="utf-8")

    def _read_list(self, path: Path) -> list[dict]:
        return json.loads(path.read_text(encoding="utf-8"))

    def _write_list(self, path: Path, payload: list[dict]) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _next_sequence(self) -> int:
        payload = json.loads(self.meta_path.read_text(encoding="utf-8"))
        payload["sequence"] += 1
        self.meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return payload["sequence"]

    def _all_memories(self) -> list[dict]:
        return self._read_list(self.episodic_hot_path) + self._read_list(self.episodic_warm_path) + self._read_list(self.episodic_archive_path)

    def ingest_event(self, event: RoundEvent) -> str | None:
        cue = _derive_cue(event)
        sequence = self._next_sequence()
        if cue:
            memories = self._read_list(self.episodic_hot_path)
            matched = next((item for item in memories if item["cue"] == cue), None)
            if matched is None:
                matched = {
                    "cue": cue,
                    "count": 0,
                    "gist_strength": 0.0,
                    "detail_strength": 0.0,
                    "last_content": "",
                    "last_seen": sequence,
                    "evidence_pointer": "",
                }
                memories.append(matched)
            matched["count"] += 1
            matched["gist_strength"] = min(1.0, matched["gist_strength"] + 0.15)
            matched["detail_strength"] = min(1.0, matched["detail_strength"] + 0.26)
            matched["last_content"] = event.content
            matched["last_seen"] = sequence
            evidence_hash = hashlib.sha1(event.content.encode("utf-8")).hexdigest()
            matched["evidence_pointer"] = f"raw:{evidence_hash}"
            self._write_list(self.episodic_hot_path, memories)
            evidence = self._read_list(self.raw_evidence_path)
            evidence.append({"cue": cue, "hash": evidence_hash, "content": event.content, "sequence": sequence})
            self._write_list(self.raw_evidence_path, evidence[-512:])

            habits = self._read_list(self.habit_path)
            habit = next((item for item in habits if item["pattern"] == cue), None)
            if habit is None:
                habit = {"pattern": cue, "strength": 0.0, "count": 0, "last_seen": sequence, "cached": False}
                habits.append(habit)
            habit["count"] += 1
            habit["strength"] = min(1.0, habit["strength"] + 0.06)
            habit["last_seen"] = sequence
            habit["cached"] = habit["count"] >= 2
            self._write_list(self.habit_path, habits)
            self.compact_layers()

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
        memories = sorted(self._all_memories(), key=lambda item: item.get("detail_strength", 0.0), reverse=True)
        return memories[:limit]

    def habit_top(self, limit: int = 5) -> list[dict]:
        habits = sorted(self._read_list(self.habit_path), key=lambda item: item["strength"], reverse=True)
        return habits[:limit]

    def recall_strength(self, cue: str | None) -> float:
        if not cue:
            return 0.0
        for item in self._all_memories():
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

    def relation_state(self, target: str) -> dict:
        for item in self._read_list(self.relation_path):
            if item["target"] == target:
                return {"target": target, "closeness": item["closeness"], "known": True}
        return {"target": target, "closeness": 0.5, "known": False}

    def compact_layers(self) -> None:
        hot = sorted(self._read_list(self.episodic_hot_path), key=lambda item: item.get("last_seen", 0), reverse=True)
        warm = sorted(self._read_list(self.episodic_warm_path), key=lambda item: item.get("last_seen", 0), reverse=True)
        archive = sorted(self._read_list(self.episodic_archive_path), key=lambda item: item.get("last_seen", 0), reverse=True)

        while len(hot) > self.hot_limit:
            demoted = hot.pop()
            demoted["detail_strength"] = min(demoted.get("detail_strength", 0.0), 0.49)
            demoted["gist_strength"] = max(demoted.get("gist_strength", 0.0), 0.25)
            warm.append(demoted)

        while len(warm) > self.warm_limit:
            demoted = warm.pop()
            archive.append(
                {
                    "cue": demoted["cue"],
                    "count": demoted.get("count", 1),
                    "gist_strength": max(demoted.get("gist_strength", 0.0), 0.25),
                    "detail_strength": min(demoted.get("detail_strength", 0.0), 0.24),
                    "summary": f"Archived memory for {demoted['cue']}",
                    "last_seen": demoted.get("last_seen", 0),
                    "evidence_pointer": demoted.get("evidence_pointer", ""),
                }
            )

        self._write_list(self.episodic_hot_path, hot)
        self._write_list(self.episodic_warm_path, warm)
        self._write_list(self.episodic_archive_path, archive)

    def tier_counts(self) -> dict[str, int]:
        return {
            "hot": len(self._read_list(self.episodic_hot_path)),
            "warm": len(self._read_list(self.episodic_warm_path)),
            "archive": len(self._read_list(self.episodic_archive_path)),
        }

    def recall(self, cue: str | None, allow_detail: bool = True) -> dict:
        if not cue:
            return {"cue": None, "mode": "none", "strength": 0.0, "content": ""}
        for tier, path in (
            ("hot", self.episodic_hot_path),
            ("warm", self.episodic_warm_path),
            ("archive", self.episodic_archive_path),
        ):
            for item in self._read_list(path):
                if item["cue"] != cue:
                    continue
                detail_strength = item.get("detail_strength", 0.0)
                gist_strength = item.get("gist_strength", 0.0)
                if allow_detail and detail_strength >= 0.50:
                    return {
                        "cue": cue,
                        "tier": tier,
                        "mode": "detail",
                        "strength": detail_strength,
                        "content": item.get("last_content", item.get("summary", "")),
                    }
                if gist_strength >= 0.25:
                    return {
                        "cue": cue,
                        "tier": tier,
                        "mode": "gist",
                        "strength": gist_strength,
                        "content": item.get("summary", item.get("last_content", "")),
                    }
        return {"cue": cue, "mode": "none", "strength": 0.0, "content": ""}
