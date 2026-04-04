from __future__ import annotations

import copy
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import shutil

from nalr.runtime.async_io import AsyncIOWorker
from nalr.runtime.dynamics import smooth_decay_rate, smooth_habit_recovery, smooth_interference_penalty
from nalr.runtime.metadata import ensure_recorded_fields, iso_date, utc_now_iso
from nalr.schemas.models import RoundEvent
from nalr.storage.parquet_io import append_dataset, read_dataset_rows, read_snapshot_rows, rewrite_snapshot


def _derive_cue(event: RoundEvent) -> str | None:
    if event.cue:
        return event.cue.lower()
    tokens = [token.strip(".,!?").lower() for token in event.content.split()]
    for token in tokens:
        if len(token) >= 6 and token not in {"please", "remember"}:
            return token
    return None


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


MEMORY_BASE_DECAY = {
    "interactive": 0.010,
    "task": 0.008,
    "idle": 0.016,
    "sleep": 0.020,
}


class MemoryStore:
    def __init__(self, root: Path, *, hot_limit: int = 64, warm_limit: int = 256) -> None:
        self.root = root
        self.hot_limit = max(int(hot_limit), 1)
        self.warm_limit = max(int(warm_limit), 1)
        self.memory_dir = self.root / "memory"
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.parquet_dir = self.memory_dir / "parquet"
        self.current_dir = self.parquet_dir / "current"
        self.raw_parquet_dir = self.parquet_dir / "raw_events"
        self.compacted_dir = self.parquet_dir / "compacted"
        self.storage_status_path = self.memory_dir / "memory_storage_status.json"
        self.migration_status_path = self.memory_dir / "memory_migration_status.json"
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
            self.parquet_dir,
            self.current_dir,
            self.raw_parquet_dir,
            self.compacted_dir,
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
        self.raw_events_path = self.raw_dir / "episodic_events.jsonl"
        self._snapshot_targets = {
            self.episodic_path: self.current_dir / "episodic_hot.parquet",
            self.episodic_warm_path: self.current_dir / "episodic_warm.parquet",
            self.episodic_archive_path: self.current_dir / "episodic_archive.parquet",
            self.habit_path: self.current_dir / "habit.parquet",
            self.relation_path: self.current_dir / "relation.parquet",
            self.relation_trace_path: self.current_dir / "relation_trace.parquet",
            self.stable_priors_path: self.current_dir / "stable_priors.parquet",
            self.circuit_breaker_path: self.current_dir / "circuit_breakers.parquet",
        }
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
                self._write_text_atomic(path, "[]")
        if not self.raw_events_path.exists():
            self._write_text_atomic(self.raw_events_path, "")
        if not self.storage_status_path.exists():
            self._write_storage_status(self._default_storage_status())
        self._migrate_legacy_if_needed()
        self._io_worker = AsyncIOWorker("nalr-memory-io")
        self._list_cache: dict[Path, list[dict]] = {}
        self._jsonl_cache: dict[Path, list[dict]] = {}
        self._artifact_cache = {
            "hot": self._load_artifacts(self.episodic_hot_dir),
            "warm": self._load_artifacts(self.episodic_warm_dir),
            "archive": self._load_artifacts(self.episodic_archive_dir),
        }
        self._recall_cache: dict[tuple[str, tuple[str, ...]], dict] = {}
        self._path_mtimes: dict[Path, int] = {}
        self._dirty_paths: set[Path] = set()
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
            self._list_cache[path] = self._load_list_from_disk(path)
            self._path_mtimes[path] = self._mtime_ns(path)
        self._jsonl_cache[self.raw_events_path] = self._load_jsonl_from_disk(self.raw_events_path)
        self._path_mtimes[self.raw_events_path] = self._mtime_ns(self.raw_events_path)

    def _default_storage_status(self) -> dict:
        parquet_live_ready = all(path.exists() for path in self._snapshot_targets.values())
        return {
            "read_source_default": "parquet",
            "storage_state": "healthy",
            "parquet_live_ready": parquet_live_ready,
            "degraded_reason": None,
            "last_sync_at": None,
        }

    def _write_storage_status(self, payload: dict) -> None:
        self.storage_status_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def storage_status(self) -> dict:
        if not self.storage_status_path.exists():
            payload = self._default_storage_status()
            self._write_storage_status(payload)
            return payload
        payload = json.loads(self.storage_status_path.read_text(encoding="utf-8"))
        payload.setdefault("read_source_default", "parquet")
        payload.setdefault("storage_state", "healthy")
        payload["parquet_live_ready"] = all(path.exists() for path in self._snapshot_targets.values())
        payload.setdefault("degraded_reason", None)
        payload.setdefault("last_sync_at", None)
        return payload

    def _mark_storage(self, *, state: str, reason: str | None = None) -> None:
        payload = self.storage_status()
        payload["storage_state"] = state
        payload["degraded_reason"] = reason
        payload["parquet_live_ready"] = all(path.exists() for path in self._snapshot_targets.values()) if state == "healthy" else False
        payload["last_sync_at"] = utc_now_iso()
        self._write_storage_status(payload)

    def _payload_rows(self, payload: list[dict]) -> list[dict]:
        return [{"payload_json": json.dumps(item, ensure_ascii=False, sort_keys=True)} for item in payload]

    def _read_parquet_payload_rows(self, snapshot_path: Path) -> list[dict]:
        rows = read_snapshot_rows(snapshot_path, "select payload_json from read_parquet(?)")
        return [json.loads(row["payload_json"]) for row in rows]

    def _read_parquet_dataset(self, dataset_dir: Path) -> list[dict]:
        rows = read_dataset_rows(dataset_dir, "select payload_json from read_parquet(?)")
        return [ensure_recorded_fields(json.loads(row["payload_json"])) for row in rows]

    def _rewrite_snapshot(self, snapshot_path: Path, payload: list[dict]) -> None:
        rewrite_snapshot(snapshot_path, self._payload_rows(payload), schema={"payload_json": "VARCHAR"})

    def _append_payload_dataset(self, dataset_dir: Path, payload: dict, *, recorded_date: str) -> None:
        append_dataset(
            dataset_dir,
            [{"recorded_date": recorded_date, "payload_json": json.dumps(payload, ensure_ascii=False, sort_keys=True)}],
            schema={"recorded_date": "VARCHAR", "payload_json": "VARCHAR"},
            partition_keys=("recorded_date",),
        )

    def _compacted_dataset_dir(self, tier: str) -> Path:
        return self.compacted_dir / f"tier={tier}"

    def _migrate_legacy_if_needed(self) -> None:
        if self.migration_status_path.exists():
            return
        has_parquet = any(path.exists() for path in self._snapshot_targets.values()) or any(self.raw_parquet_dir.rglob("*.parquet"))
        has_legacy = any(path.exists() and path.read_text(encoding="utf-8").strip() for path in self._snapshot_targets) or (
            self.raw_events_path.exists() and self.raw_events_path.read_text(encoding="utf-8").strip()
        )
        if not has_legacy or has_parquet:
            return
        for legacy_path, snapshot_path in self._snapshot_targets.items():
            payload = self._load_list_from_json_legacy(legacy_path)
            self._rewrite_snapshot(snapshot_path, payload)
        for row in self._load_jsonl_from_legacy(self.raw_events_path):
            self._append_payload_dataset(self.raw_parquet_dir, row, recorded_date=row.get("recorded_date", "legacy"))
        self.migration_status_path.write_text(
            json.dumps({"migrated_at": utc_now_iso(), "status": "completed"}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _load_list_from_json_legacy(self, path: Path) -> list[dict]:
        if not path.exists():
            return []
        raw_text = path.read_text(encoding="utf-8").strip()
        if not raw_text:
            return []
        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError:
            return []
        if not isinstance(payload, list):
            return []
        if path in {self.episodic_path, self.episodic_warm_path, self.episodic_archive_path}:
            return [self._normalize_memory_record(item) for item in payload]
        if path == self.habit_path:
            return [self._normalize_habit_record(item) for item in payload]
        return payload

    def _load_jsonl_from_legacy(self, path: Path) -> list[dict]:
        if not path.exists():
            return []
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rows.append(ensure_recorded_fields(json.loads(line)))
        return rows

    def _mtime_ns(self, path: Path) -> int:
        if not path.exists():
            return 0
        return path.stat().st_mtime_ns

    def _load_artifacts(self, tier_dir: Path) -> list[dict]:
        tier = {
            self.episodic_hot_dir: "hot",
            self.episodic_warm_dir: "warm",
            self.episodic_archive_dir: "archive",
        }[tier_dir]
        return self._read_parquet_dataset(self._compacted_dataset_dir(tier))

    def _load_list_from_disk(self, path: Path) -> list[dict]:
        snapshot_path = self._snapshot_targets.get(path)
        if snapshot_path is not None and snapshot_path.exists():
            legacy_is_newer = path.exists() and path.stat().st_mtime_ns > snapshot_path.stat().st_mtime_ns
            if legacy_is_newer:
                payload = self._load_list_from_json_legacy(path)
                self._write_text_atomic(path, json.dumps(payload, ensure_ascii=False, indent=2))
                self._rewrite_snapshot(snapshot_path, payload)
                if path in {self.episodic_path, self.episodic_warm_path, self.episodic_archive_path}:
                    return [self._normalize_memory_record(item) for item in payload]
                if path == self.habit_path:
                    return [self._normalize_habit_record(item) for item in payload]
                return payload
            payload = self._read_parquet_payload_rows(snapshot_path)
            if path in {self.episodic_path, self.episodic_warm_path, self.episodic_archive_path}:
                return [self._normalize_memory_record(item) for item in payload]
            if path == self.habit_path:
                return [self._normalize_habit_record(item) for item in payload]
            return payload
        if not path.exists():
            self._write_text_atomic(path, "[]")
            return []

        raw_text = path.read_text(encoding="utf-8").strip()
        if not raw_text:
            self._write_text_atomic(path, "[]")
            return []

        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError:
            backup_path = path.with_name(f"{path.name}.corrupt-{utc_now_iso().replace(':', '').replace('-', '')}")
            path.replace(backup_path)
            self._write_text_atomic(path, "[]")
            return []

        if not isinstance(payload, list):
            backup_path = path.with_name(f"{path.name}.corrupt-{utc_now_iso().replace(':', '').replace('-', '')}")
            path.replace(backup_path)
            self._write_text_atomic(path, "[]")
            return []
        if path in {self.episodic_path, self.episodic_warm_path, self.episodic_archive_path}:
            return [self._normalize_memory_record(item) for item in payload]
        if path == self.habit_path:
            return [self._normalize_habit_record(item) for item in payload]
        return payload

    def _load_jsonl_from_disk(self, path: Path) -> list[dict]:
        if path == self.raw_events_path and any(self.raw_parquet_dir.rglob("*.parquet")):
            return self._read_parquet_dataset(self.raw_parquet_dir)
        if not path.exists():
            return []
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rows.append(ensure_recorded_fields(json.loads(line)))
        return rows

    def _read_list(self, path: Path) -> list[dict]:
        if path not in self._list_cache:
            self._list_cache[path] = self._load_list_from_disk(path)
            self._path_mtimes[path] = self._mtime_ns(path)
        elif path not in self._dirty_paths and self._mtime_ns(path) > self._path_mtimes.get(path, 0):
            self._list_cache[path] = self._load_list_from_disk(path)
            self._path_mtimes[path] = self._mtime_ns(path)
        return self._list_cache[path]

    def _normalize_memory_record(self, item: dict) -> dict:
        normalized = {
            "cue": "",
            "count": 0,
            "gist_strength": 0.0,
            "detail_strength": 0.0,
            "last_content": "",
            "interference": 0.0,
            "interfered": False,
            "interference_source": None,
            "context_slot": "legacy::none",
            "last_recalled_round": 0,
            "last_affect_intensity": 0.0,
            "last_cue_quality": 0.0,
        }
        normalized.update(item)
        normalized["gist_strength"] = round(float(normalized.get("gist_strength", 0.0) or 0.0), 4)
        normalized["detail_strength"] = round(float(normalized.get("detail_strength", 0.0) or 0.0), 4)
        normalized["interference"] = round(float(normalized.get("interference", 0.0) or 0.0), 4)
        normalized["interfered"] = bool(normalized.get("interfered", normalized["interference"] > 0.0))
        normalized["last_affect_intensity"] = round(float(normalized.get("last_affect_intensity", 0.0) or 0.0), 4)
        normalized["last_cue_quality"] = round(float(normalized.get("last_cue_quality", 0.0) or 0.0), 4)
        normalized["last_recalled_round"] = int(normalized.get("last_recalled_round", 0) or 0)
        return normalized

    def _normalize_habit_record(self, item: dict) -> dict:
        normalized = {
            "pattern": "",
            "strength": 0.0,
            "count": 0,
            "recoverable": True,
            "status": "active",
            "suppressed_by": None,
            "suppressed_at_round": None,
            "last_context_recurrence": 0.0,
            "last_updated_round": 0,
            "context_slot": None,
            "breakthrough_rounds": [],
        }
        normalized.update(item)
        normalized["strength"] = round(float(normalized.get("strength", 0.0) or 0.0), 4)
        normalized["count"] = int(normalized.get("count", 0) or 0)
        normalized["recoverable"] = bool(normalized.get("recoverable", True))
        normalized["status"] = str(normalized.get("status") or "active")
        normalized["last_context_recurrence"] = round(float(normalized.get("last_context_recurrence", 0.0) or 0.0), 4)
        normalized["last_updated_round"] = int(normalized.get("last_updated_round", 0) or 0)
        normalized["breakthrough_rounds"] = [int(value) for value in normalized.get("breakthrough_rounds", [])]
        return normalized

    def _write_text_atomic(self, path: Path, content: str) -> None:
        tmp_path = path.with_name(f"{path.name}.tmp")
        tmp_path.write_text(content, encoding="utf-8")
        tmp_path.replace(path)

    def _write_list(self, path: Path, payload: list[dict]) -> None:
        snapshot = copy.deepcopy(payload)
        self._list_cache[path] = snapshot
        self._dirty_paths.add(path)
        self._invalidate_recall_cache()

        def write() -> None:
            snapshot_path = self._snapshot_targets.get(path)
            if snapshot_path is not None:
                self._rewrite_snapshot(snapshot_path, snapshot)
            self._write_text_atomic(path, json.dumps(snapshot, ensure_ascii=False, indent=2))
            self._path_mtimes[path] = self._mtime_ns(path)
            self._dirty_paths.discard(path)
            self._mark_storage(state="healthy")

        self._io_worker.submit(write, on_error=lambda exc: self._mark_storage(state="degraded", reason=str(exc)))

    def _append_jsonl(self, path: Path, payload: dict) -> None:
        entry = copy.deepcopy(payload)
        self._jsonl_cache.setdefault(path, self._load_jsonl_from_disk(path)).append(entry)
        self._dirty_paths.add(path)
        self._invalidate_recall_cache()

        def write() -> None:
            if path == self.raw_events_path:
                self._append_payload_dataset(self.raw_parquet_dir, entry, recorded_date=entry.get("recorded_date", "legacy"))
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
            self._path_mtimes[path] = self._mtime_ns(path)
            self._dirty_paths.discard(path)
            self._mark_storage(state="healthy")

        self._io_worker.submit(write, on_error=lambda exc: self._mark_storage(state="degraded", reason=str(exc)))

    def _safe_filename(self, cue: str) -> str:
        normalized = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in cue.lower())
        return normalized or "unknown"

    def _read_jsonl(self, path: Path) -> list[dict]:
        if path not in self._jsonl_cache:
            self._jsonl_cache[path] = self._load_jsonl_from_disk(path)
            self._path_mtimes[path] = self._mtime_ns(path)
        elif path not in self._dirty_paths and self._mtime_ns(path) > self._path_mtimes.get(path, 0):
            self._jsonl_cache[path] = self._load_jsonl_from_disk(path)
            self._path_mtimes[path] = self._mtime_ns(path)
        return self._jsonl_cache[path]

    def _context_slot(self, event: RoundEvent) -> str:
        target = (event.target or "none").lower()
        return f"{event.source.lower()}::{target}"

    def _invalidate_recall_cache(self) -> None:
        self._recall_cache.clear()

    def _cue_similarity(self, left: str, right: str) -> float:
        left = left.lower().strip()
        right = right.lower().strip()
        if not left or not right:
            return 0.0
        prefix = 0
        for left_char, right_char in zip(left, right):
            if left_char != right_char:
                break
            prefix += 1
        prefix_ratio = prefix / max(len(left), len(right), 1)
        left_bigrams = {left[idx : idx + 2] for idx in range(max(len(left) - 1, 1))}
        right_bigrams = {right[idx : idx + 2] for idx in range(max(len(right) - 1, 1))}
        overlap = len(left_bigrams & right_bigrams)
        union = len(left_bigrams | right_bigrams) or 1
        ratio = SequenceMatcher(None, left, right).ratio()
        if len(left) == len(right) and sum(1 for left_char, right_char in zip(left, right) if left_char != right_char) == 1:
            ratio = max(ratio, 0.84)
        return round(max(prefix_ratio, overlap / union, ratio), 4)

    def _write_artifacts(self, tier_dir: Path, artifacts: list[dict]) -> int:
        tier = {
            self.episodic_hot_dir: "hot",
            self.episodic_warm_dir: "warm",
            self.episodic_archive_dir: "archive",
        }[tier_dir]
        snapshot = copy.deepcopy(artifacts)
        self._artifact_cache[tier] = snapshot
        self._invalidate_recall_cache()

        def write() -> None:
            dataset_dir = self._compacted_dataset_dir(tier)
            if dataset_dir.exists():
                shutil.rmtree(dataset_dir)
            append_dataset(
                dataset_dir,
                [
                    {
                        "payload_json": json.dumps(artifact, ensure_ascii=False, sort_keys=True),
                    }
                    for artifact in snapshot
                ],
                schema={"payload_json": "VARCHAR"},
                partition_keys=(),
            )
            for existing in tier_dir.glob("*.json"):
                existing.unlink()
            for artifact in snapshot:
                path = tier_dir / f"{self._safe_filename(artifact['cue'])}.json"
                path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
            self._mark_storage(state="healthy")

        self._io_worker.submit(write, on_error=lambda exc: self._mark_storage(state="degraded", reason=str(exc)))
        return len(artifacts)

    def _summarize_artifact(self, tier: str, cue: str, events: list[dict], retention_days: int | None) -> dict:
        ordered = sorted(events, key=lambda item: item["round_id"])
        round_ids = [item["round_id"] for item in ordered]
        evidence_refs = [
            {
                "round_id": item["round_id"],
                "recorded_at": item["recorded_at"],
                "recorded_date": item["recorded_date"],
                "content_hash": hashlib.sha1(item["content"].encode("utf-8")).hexdigest(),
            }
            for item in ordered
        ]
        summary = {
            "cue": cue,
            "tier": tier,
            "session_id": ordered[-1]["session_id"],
            "event_count": len(ordered),
            "first_round_id": min(round_ids),
            "last_round_id": max(round_ids),
            "avg_valence": round(sum(item.get("valence", 0.0) for item in ordered) / len(ordered), 4),
            "latest_content": ordered[-1]["content"],
            "recorded_at": ordered[-1]["recorded_at"],
            "recorded_date": ordered[-1]["recorded_date"],
            "retention_days": retention_days,
        }
        if tier == "hot":
            summary["detail_preview"] = [item["content"] for item in ordered[-5:]]
            summary["evidence_refs"] = evidence_refs[-8:]
            return summary
        if tier == "warm":
            summary["gist"] = " | ".join(item["content"] for item in ordered[-3:])
            summary["evidence_refs"] = evidence_refs[-5:]
            return summary
        summary["summary_chunk"] = {
            "gist": " | ".join(item["content"] for item in ordered[-2:]),
            "hash_digest": hashlib.sha1("".join(ref["content_hash"] for ref in evidence_refs).encode("utf-8")).hexdigest(),
        }
        summary["sampled_evidence_refs"] = evidence_refs[-2:]
        return summary

    def _cue_decay_modifier(self, recalled_within_window: bool) -> float:
        return 0.70 if recalled_within_window else 1.00

    def _affect_decay_modifier(self, affect_intensity: float) -> float:
        if affect_intensity >= 0.75:
            return 0.55
        if affect_intensity >= 0.40:
            return 0.85
        if affect_intensity < 0.20:
            return 1.20
        return 1.00

    def _decay_memories(self, active_cue: str | None, mode: str = "interactive", *, round_id: int | None = None) -> None:
        memories = self._read_list(self.episodic_path)
        base_decay = MEMORY_BASE_DECAY.get(mode, MEMORY_BASE_DECAY["interactive"])
        for memory in memories:
            recalled_within_window = memory["cue"] == active_cue or (
                round_id is not None and round_id - int(memory.get("last_recalled_round", 0) or 0) <= 3
            )
            effective_decay = smooth_decay_rate(
                base_decay=base_decay,
                recalled_within_window=recalled_within_window,
                affect_intensity=float(memory.get("last_affect_intensity", 0.0) or 0.0),
            )
            memory["gist_strength"] = max(0.0, round(memory.get("gist_strength", 0.0) * (1 - effective_decay), 4))
            memory["detail_strength"] = max(0.0, round(memory.get("detail_strength", 0.0) * (1 - effective_decay), 4))
        self._write_list(self.episodic_path, memories)

    def _interference(self, memories: list[dict], cue: str, context_slot: str) -> tuple[float, str | None]:
        penalty = 0.0
        source = None
        for memory in memories:
            if memory["cue"] == cue:
                continue
            if memory.get("context_slot") != context_slot:
                continue
            similarity = self._cue_similarity(memory["cue"], cue)
            if similarity < 0.82:
                continue
            old_strength = max(float(memory.get("detail_strength", 0.0)), float(memory.get("gist_strength", 0.0)))
            if old_strength > 0.45:
                continue
            candidate_penalty = smooth_interference_penalty(similarity=similarity, old_strength=old_strength)
            if candidate_penalty >= penalty:
                penalty = candidate_penalty
                source = memory["cue"]
        return round(_clip(penalty), 4), source

    def _update_stable_priors(self, cue: str, gist_strength: float) -> None:
        priors = self._read_list(self.stable_priors_path)
        prior = next((item for item in priors if item["cue"] == cue), None)
        if prior is None:
            prior = {"cue": cue, "weight": 0.0}
            priors.append(prior)
        prior["weight"] = min(1.0, round(prior["weight"] * 0.92 + gist_strength * 0.18, 4))
        self._write_list(self.stable_priors_path, priors)

    def _update_memory_strength(
        self,
        matched: dict,
        *,
        cue_boost: float,
        affect_boost: float,
        interference_penalty: float,
        cue_quality: float = 0.6,
    ) -> None:
        quality_scale = 0.72 + max(0.0, min(1.0, cue_quality)) * 0.48
        matched["gist_strength"] = round(
            _clip(
                matched.get("gist_strength", 0.0)
                + cue_boost * max(0.85, quality_scale * 0.9)
                + affect_boost * 0.75
                - interference_penalty * 0.65
            ),
            4,
        )
        matched["detail_strength"] = round(
            _clip(
                matched.get("detail_strength", 0.0)
                + cue_boost * (1.0 + quality_scale * 0.35)
                + affect_boost
                + cue_quality * 0.14
                - interference_penalty
            ),
            4,
        )

    def _update_habit_record(
        self,
        *,
        pattern: str,
        valence: float,
        context_recurrence: float = 0.0,
        round_gap: int = 0,
        context_slot: str | None = None,
        round_id: int | None = None,
    ) -> dict:
        habits = self._read_list(self.habit_path)
        habit = next((item for item in habits if item["pattern"] == pattern), None)
        if habit is None:
            habit = {
                "pattern": pattern,
                "strength": 0.0,
                "count": 0,
                "recoverable": True,
                "status": "active",
                "suppressed_by": None,
                "suppressed_at_round": None,
                "context_slot": context_slot or "global",
                "last_context_recurrence": 0.0,
                "last_updated_round": 0,
                "breakthrough_rounds": [],
            }
            habits.append(habit)
        habit["count"] += 1
        decay_h = 0.015
        eta_rep = 0.04
        eta_pos = 0.06
        eta_neg = 0.08
        eta_ctx = 0.03
        breakthrough_bonus_cap = 0.08
        gap_penalty = min(0.25, max(0, round_gap) * 0.03)
        success_signal = _clip(max(0.0, valence), 0.0, 1.0)
        window_floor = max(0, int(round_id or 0) - 9)
        habit["breakthrough_rounds"] = [value for value in habit.get("breakthrough_rounds", []) if int(value) >= window_floor]
        breakthrough_bonus = min(breakthrough_bonus_cap, 0.04 + 0.06 * success_signal)
        valence_push = max(0.0, valence) * eta_pos - max(0.0, -valence) * eta_neg
        if success_signal > 0.0 and len(habit["breakthrough_rounds"]) < 2 and round_id is not None:
            habit["breakthrough_rounds"].append(int(round_id))
        if habit.get("status") == "suppressed_recoverable":
            recovery_rate = smooth_habit_recovery(context_recurrence=context_recurrence, valence=valence)
            delta = min(0.10, recovery_rate)
        else:
            delta = min(
                0.10,
                eta_rep
                + valence_push
                + min(eta_ctx, context_recurrence * eta_ctx)
                + breakthrough_bonus * 0.35,
            )
        habit["strength"] = round(_clip(habit["strength"] * (1 - decay_h) + delta, 0.0, 0.92) * (1 - gap_penalty), 4)
        habit["recoverable"] = True
        habit["context_slot"] = context_slot or habit.get("context_slot") or "global"
        habit["last_context_recurrence"] = round(float(context_recurrence), 4)
        if round_id is not None:
            habit["last_updated_round"] = int(round_id)

        if habit.get("status") == "suppressed_recoverable":
            suppressor = next((item for item in habits if item["pattern"] == habit.get("suppressed_by")), None)
            if suppressor is None or habit["strength"] >= float(suppressor.get("strength", 0.0)):
                habit["status"] = "active"
                habit["suppressed_by"] = None
                habit["suppressed_at_round"] = None
        else:
            habit["status"] = "active"
            habit["suppressed_by"] = None
            habit["suppressed_at_round"] = None

        if habit["strength"] >= 0.40 and habit["count"] >= 3:
            for other in habits:
                if other["pattern"] == pattern:
                    continue
                if (other.get("context_slot") or "global") != habit["context_slot"]:
                    continue
                if float(other.get("strength", 0.0)) >= habit["strength"]:
                    continue
                if round_id is not None and int(round_id) - int(other.get("last_updated_round", 0) or 0) > 10:
                    continue
                other["status"] = "suppressed_recoverable"
                other["recoverable"] = True
                other["suppressed_by"] = pattern
                other["suppressed_at_round"] = round_id

        self._write_list(self.habit_path, habits)
        return habit

    def ingest_event(
        self,
        event: RoundEvent,
        *,
        round_id: int | None = None,
        session_id: str | None = None,
        recorded_at: str | None = None,
        update_habit: bool = True,
        mode: str = "interactive",
        cue_quality: float | None = None,
    ) -> str | None:
        effective_recorded_at = recorded_at or utc_now_iso()
        cue = _derive_cue(event)
        context_slot = self._context_slot(event)
        effective_cue_quality = round(_clip(cue_quality if cue_quality is not None else event.cue_quality), 4)
        self._append_jsonl(
            self.raw_events_path,
            {
                "session_id": session_id or "legacy",
                "round_id": round_id or 0,
                "recorded_at": effective_recorded_at,
                "recorded_date": iso_date(effective_recorded_at),
                "cue": cue,
                "context_slot": context_slot,
                "source": event.source,
                "target": event.target,
                "content": event.content,
                "valence": event.valence,
            },
        )
        self._decay_memories(cue, mode=mode, round_id=round_id)
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
                    "interference_source": None,
                    "interfered": False,
                    "last_cue_quality": 0.0,
                    "last_affect_intensity": 0.0,
                    "last_recalled_round": 0,
                    "context_slot": context_slot,
                }
                memories.append(matched)
            matched["count"] += 1
            matched["context_slot"] = context_slot
            interference_penalty, interference_source = self._interference(memories, cue, context_slot)
            matched["interference"] = interference_penalty
            matched["interference_source"] = interference_source
            matched["interfered"] = matched["interference"] > 0.0
            cue_boost = (
                0.12 + 0.20 * effective_cue_quality
                if effective_cue_quality > 0.0
                and (
                    matched.get("interfered")
                    or max(float(matched.get("detail_strength", 0.0)), float(matched.get("gist_strength", 0.0))) <= 0.45
                )
                else (0.08 if matched["count"] > 1 else 0.10)
            )
            affect_boost = max(0.0, abs(event.valence)) * 0.06
            matched["last_cue_quality"] = effective_cue_quality
            matched["last_affect_intensity"] = round(abs(event.valence), 4)
            matched["last_recalled_round"] = int(round_id or matched.get("last_recalled_round", 0) or 0)
            self._update_memory_strength(
                matched,
                cue_boost=cue_boost,
                affect_boost=affect_boost,
                interference_penalty=matched["interference"],
                cue_quality=effective_cue_quality,
            )
            matched["last_content"] = event.content
            self._write_list(self.episodic_path, memories)
            self._sync_memory_tiers(matched)
            self._update_stable_priors(cue, matched["gist_strength"])
            if update_habit:
                self._update_habit_record(
                    pattern=cue,
                    valence=event.valence,
                    context_recurrence=min(1.0, matched["count"] / 10),
                    context_slot=context_slot,
                    round_id=round_id,
                )

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

    def _lookup_tiers(self, cue: str, tier_budget: tuple[str, ...]) -> dict:
        tier_paths = {
            "hot": self.episodic_path,
            "warm": self.episodic_warm_path,
            "archive": self.episodic_archive_path,
        }
        tier = None
        memory = None
        for candidate in tier_budget:
            path = tier_paths.get(candidate)
            if path is None:
                continue
            memory = next((item for item in self._read_list(path) if item["cue"] == cue), None)
            if memory is not None:
                tier = candidate
                break
        if memory is None:
            return {"cue": cue, "tier": None, "strength": 0.0, "detail": False, "found": False, "evidence": []}
        detail_strength = float(memory.get("detail_strength", 0.0))
        gist_strength = float(memory.get("gist_strength", 0.0))
        detail = detail_strength >= 0.50
        evidence = []
        if "detail_preview" in memory:
            evidence = memory.get("detail_preview", [])[:3]
        elif "gist" in memory:
            evidence = [memory.get("gist", "")]
        elif memory.get("summary_chunk", {}).get("gist"):
            evidence = [memory["summary_chunk"]["gist"]]
        return {
            "cue": cue,
            "tier": tier,
            "strength": round(max(detail_strength, gist_strength), 4),
            "detail": detail,
            "interference": round(float(memory.get("interference", 0.0)), 4),
            "interfered": bool(memory.get("interfered", False)),
            "interference_source": memory.get("interference_source"),
            "found": True,
            "evidence": [item for item in evidence if item],
        }

    def recall(
        self,
        cue: str,
        *,
        tier_budget: tuple[str, ...] = ("hot", "warm", "archive"),
        allow_detail: bool = True,
    ) -> dict:
        normalized_cue = cue.lower()
        normalized_budget = tuple(tier_budget)
        cache_key = (normalized_cue, normalized_budget + (f"detail={int(allow_detail)}",))
        cached = self._recall_cache.get(cache_key)
        if cached is not None:
            return copy.deepcopy(cached)
        payload = self._lookup_tiers(normalized_cue, normalized_budget)
        if payload.get("found"):
            evidence = list(payload.get("evidence", []))
            detail_strength = float(payload.get("strength", 0.0)) if payload.get("detail") else 0.0
            gist_strength = max(float(payload.get("strength", 0.0)), 0.0)
            if allow_detail and payload.get("detail"):
                payload["mode"] = "detail"
                payload["content"] = evidence[0] if evidence else ""
                payload["strength"] = round(detail_strength or gist_strength, 4)
            else:
                payload["mode"] = "gist" if gist_strength > 0 else "none"
                payload["content"] = evidence[0] if evidence else ""
                payload["detail"] = False
                if payload["mode"] == "gist":
                    payload["strength"] = round(min(gist_strength, detail_strength or gist_strength), 4)
        else:
            payload["mode"] = "none"
            payload["content"] = ""
        self._recall_cache[cache_key] = copy.deepcopy(payload)
        return payload

    def memory_top(self, limit: int = 5) -> list[dict]:
        memories = []
        memories.extend({**item, "tier": "hot"} for item in self._read_list(self.episodic_path))
        memories.extend({**item, "tier": "warm"} for item in self._read_list(self.episodic_warm_path))
        memories.extend({**item, "tier": "archive"} for item in self._read_list(self.episodic_archive_path))
        memories = sorted(memories, key=lambda item: (item.get("detail_strength", 0.0), item.get("count", 0)), reverse=True)
        return memories[:limit]

    def stable_priors_top(self, limit: int = 5) -> list[dict]:
        priors = sorted(self._read_list(self.stable_priors_path), key=lambda item: item.get("weight", 0.0), reverse=True)
        return priors[:limit]

    def habit_top(self, limit: int = 5) -> list[dict]:
        habits = sorted(self._read_list(self.habit_path), key=lambda item: item["strength"], reverse=True)
        return habits[:limit]

    def shape_noninteractive(self, *, mode: str, cue: str | None = None) -> dict[str, object]:
        proposal = self.build_noninteractive_proposal(mode=mode, cue=cue)
        return self.apply_noninteractive_proposal(proposal)

    def build_noninteractive_proposal(self, *, mode: str, cue: str | None = None) -> dict[str, object]:
        if mode not in {"idle", "sleep"}:
            raise ValueError(f"unsupported noninteractive shaping mode: {mode}")

        effective_cue = (cue or "").lower().strip()
        if not effective_cue:
            stable_priors = self.stable_priors_top(limit=1)
            habits = self.habit_top(limit=1)
            if stable_priors:
                effective_cue = str(stable_priors[0].get("cue", "")).lower().strip()
            elif habits:
                effective_cue = str(habits[0].get("pattern", "")).lower().strip()

        replayed_anchor = effective_cue or None
        gist_delta = 0.045 if mode == "sleep" else 0.02
        detail_delta = -0.02 if mode == "sleep" else -0.005
        stable_prior_delta = 0.03 if mode == "sleep" else 0.012
        habit_delta = 0.022 if mode == "sleep" else 0.008
        interference_scale = 0.88 if mode == "sleep" else 0.95
        allowed_dream_write = mode == "sleep"

        return {
            "source": mode,
            "mode": mode,
            "non_interactive": True,
            "replayed_anchor": replayed_anchor,
            "cue": effective_cue or None,
            "memory_consolidation": (
                [
                    {
                        "cue": effective_cue,
                        "gist_delta": round(gist_delta, 4),
                        "detail_delta": round(detail_delta, 4),
                        "interference_scale": round(interference_scale, 4),
                        "stable_prior_delta": round(stable_prior_delta, 4),
                    }
                ]
                if effective_cue
                else []
            ),
            "emotion_adjustments": [],
            "habit_adjustments": (
                [{"pattern": effective_cue, "delta": round(habit_delta, 4)}]
                if effective_cue
                else []
            ),
            "dream_memory_write": (
                [{"cue": effective_cue, "gist": f"dream::{mode}::{effective_cue}"}]
                if effective_cue and allowed_dream_write
                else []
            ),
            "relationship_adjustments": [],
        }

    def apply_noninteractive_proposal(self, proposal: dict[str, object]) -> dict[str, object]:
        mode = str(proposal.get("mode", proposal.get("source", "")))
        effective_cue = str(proposal.get("cue") or "").lower().strip()
        applied_types: list[str] = []

        memory_ops = list(proposal.get("memory_consolidation", []))
        if effective_cue and memory_ops:
            memories = self._read_list(self.episodic_path)
            matched = next((item for item in memories if item["cue"] == effective_cue), None)
            if matched is not None:
                op = memory_ops[0]
                matched["interference"] = round(
                    max(0.0, float(matched.get("interference", 0.0)) * float(op.get("interference_scale", 1.0))),
                    4,
                )
                matched["gist_strength"] = round(_clip(float(matched.get("gist_strength", 0.0)) + float(op.get("gist_delta", 0.0))), 4)
                matched["detail_strength"] = round(_clip(float(matched.get("detail_strength", 0.0)) + float(op.get("detail_delta", 0.0))), 4)
                self._write_list(self.episodic_path, memories)
                self._sync_memory_tiers(matched)

                priors = self._read_list(self.stable_priors_path)
                prior = next((item for item in priors if item["cue"] == effective_cue), None)
                if prior is None:
                    prior = {"cue": effective_cue, "weight": 0.0}
                    priors.append(prior)
                prior["weight"] = round(_clip(float(prior.get("weight", 0.0)) + float(op.get("stable_prior_delta", 0.0))), 4)
                self._write_list(self.stable_priors_path, priors)
                applied_types.append("memory_consolidation")

        habit_ops = list(proposal.get("habit_adjustments", []))
        if effective_cue and habit_ops:
            habits = self._read_list(self.habit_path)
            habit = next((item for item in habits if item["pattern"] == effective_cue), None)
            if habit is None:
                habit = {"pattern": effective_cue, "strength": 0.0, "count": 0, "recoverable": False, "status": "active", "suppressed_by": None}
                habits.append(habit)
            habit["strength"] = round(_clip(float(habit.get("strength", 0.0)) + float(habit_ops[0].get("delta", 0.0)), 0.0, 0.92), 4)
            habit["status"] = "active"
            habit["suppressed_by"] = None
            self._write_list(self.habit_path, habits)
            applied_types.append("habit_adjustments")

        if proposal.get("dream_memory_write"):
            self._append_jsonl(
                self.raw_events_path,
                {
                    "session_id": "dream",
                    "round_id": 0,
                    "recorded_at": utc_now_iso(),
                    "recorded_date": iso_date(utc_now_iso()),
                    "cue": effective_cue or None,
                    "context_slot": f"dream::{mode or 'sleep'}",
                    "source": "dream",
                    "target": None,
                    "content": f"dream::{mode}::{effective_cue}",
                    "valence": 0.0,
                },
            )
            applied_types.append("dream_memory_write")

        return {
            "source": mode,
            "mode": mode,
            "non_interactive": True,
            "replayed_anchor": proposal.get("replayed_anchor"),
            "cue": effective_cue or None,
            "applied": bool(applied_types),
            "applied_types": applied_types,
            "gist_delta": round(float(memory_ops[0].get("gist_delta", 0.0)) if memory_ops else 0.0, 4),
            "detail_delta": round(float(memory_ops[0].get("detail_delta", 0.0)) if memory_ops else 0.0, 4),
            "stable_prior_delta": round(float(memory_ops[0].get("stable_prior_delta", 0.0)) if memory_ops else 0.0, 4),
            "habit_delta": round(float(habit_ops[0].get("delta", 0.0)) if habit_ops else 0.0, 4),
        }

    def identity_evidence(self) -> dict[str, object]:
        stable_priors = self.stable_priors_top(limit=5)
        habits = self.habit_top(limit=5)
        relations = sorted(self._read_list(self.relation_path), key=lambda item: item.get("closeness", 0.0), reverse=True)[:5]
        memories = self.memory_top(limit=5)
        anchors: list[str] = []
        for item in stable_priors[:2]:
            cue = item.get("cue")
            if cue:
                anchors.append(f"cue:{cue}")
        if memories:
            memory_cue = memories[0].get("cue")
            if memory_cue:
                anchors.append(f"memory:{memory_cue}")
        if habits:
            pattern = habits[0].get("pattern")
            if pattern:
                anchors.append(f"habit:{pattern}")
        if relations:
            target = relations[0].get("target")
            if target:
                anchors.append(f"relation:{target}")
        signature = hashlib.sha1("|".join(sorted(anchors)).encode("utf-8")).hexdigest() if anchors else ""
        return {
            "stable_priors": stable_priors,
            "habits": habits,
            "relations": relations,
            "memories": memories,
            "anchors": anchors,
            "signature": signature,
        }

    def recall_strength(self, cue: str | None, *, tier_budget: tuple[str, ...] = ("hot", "warm", "archive")) -> float:
        if not cue:
            return 0.0
        recall_payload = self.recall(cue, tier_budget=tier_budget)
        if not recall_payload.get("found"):
            return 0.0
        strength = float(recall_payload.get("strength", 0.0))
        interference = float(recall_payload.get("interference", 0.0))
        if recall_payload.get("detail"):
            return max(0.0, round(strength * (1 - interference * 0.5), 4))
        if strength >= 0.25:
            return max(0.0, round(strength * (1 - interference * 0.3), 4))
        return round(strength, 4)

    def habit_strength(self, cue: str | None) -> float:
        if not cue:
            return 0.0
        for item in self._read_list(self.habit_path):
            if item["pattern"] == cue:
                return item["strength"]
        return 0.0

    def update_habit_strength(
        self,
        cue: str | None,
        valence: float,
        *,
        context_recurrence: float = 0.0,
        round_gap: int = 0,
        context_slot: str | None = None,
        round_id: int | None = None,
    ) -> dict | None:
        if not cue:
            return None
        return self._update_habit_record(
            pattern=cue,
            valence=valence,
            context_recurrence=context_recurrence,
            round_gap=round_gap,
            context_slot=context_slot,
            round_id=round_id,
        )

    def reset_habit(self, pattern: str) -> dict:
        habits = self._read_list(self.habit_path)
        habit = next((item for item in habits if item["pattern"] == pattern), None)
        if habit is None:
            habit = self._normalize_habit_record({"pattern": pattern, "strength": 0.0, "count": 0, "recoverable": True})
            habits.append(habit)
        else:
            habit["strength"] = 0.0
            habit["recoverable"] = True
            habit["status"] = "active"
            habit["suppressed_by"] = None
            habit["suppressed_at_round"] = None
        self._write_list(self.habit_path, habits)
        return habit

    def closeness(self, target: str | None) -> float:
        if not target:
            return 0.5
        for item in self._read_list(self.relation_path):
            if item["target"] == target:
                return item["closeness"]
        return 0.5

    def relation_state(self, target: str) -> dict:
        closeness = self.closeness(target)
        return {
            "target": target,
            "closeness": closeness,
            "trust": closeness,
            "boundary_level": _clip(0.8 - closeness, 0.0, 1.0),
            "known": any(item["target"] == target for item in self._read_list(self.relation_path)),
        }

    def nudge_relation(self, target: str, delta: float) -> dict:
        relations = self._read_list(self.relation_path)
        relation = next((item for item in relations if item["target"] == target), None)
        if relation is None:
            relation = {"target": target, "closeness": 0.5}
            relations.append(relation)
        relation["closeness"] = round(_clip(float(relation.get("closeness", 0.5)) + delta), 4)
        self._write_list(self.relation_path, relations)
        traces = self._read_list(self.relation_trace_path)
        traces.append({"target": target, "valence": 0.0, "content": f"trust_nudge:{delta:+.3f}"})
        self._write_list(self.relation_trace_path, traces)
        return relation

    def compact_layers(self) -> None:
        hot = sorted(self._read_list(self.episodic_path), key=lambda item: (item.get("count", 0), item.get("detail_strength", 0.0)), reverse=True)
        warm = sorted(self._read_list(self.episodic_warm_path), key=lambda item: (item.get("count", 0), item.get("detail_strength", 0.0)), reverse=True)
        archive = sorted(self._read_list(self.episodic_archive_path), key=lambda item: (item.get("count", 0), item.get("detail_strength", 0.0)), reverse=True)

        while len(hot) > self.hot_limit:
            demoted = hot.pop()
            demoted["detail_strength"] = min(float(demoted.get("detail_strength", 0.0)), 0.49)
            warm.append(demoted)

        while len(warm) > self.warm_limit:
            demoted = warm.pop()
            demoted["detail_strength"] = min(float(demoted.get("detail_strength", 0.0)), 0.24)
            demoted["summary"] = demoted.get("summary") or f"Archived memory for {demoted.get('cue', 'unknown')}"
            archive.append(demoted)

        self._write_list(self.episodic_path, hot)
        self._write_list(self.episodic_warm_path, warm)
        self._write_list(self.episodic_archive_path, archive)

    def tier_counts(self) -> dict[str, int]:
        return {
            "hot": len(self._read_list(self.episodic_path)),
            "warm": len(self._read_list(self.episodic_warm_path)),
            "archive": len(self._read_list(self.episodic_archive_path)),
        }

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

    def compact_tiers(self, *, hot_max_rounds: int = 500, warm_max_rounds: int = 3000) -> dict:
        events = [row for row in self._read_jsonl(self.raw_events_path) if row.get("cue")]
        latest_round = max((row.get("round_id", 0) for row in events), default=0)
        grouped: dict[str, dict[str, list[dict]]] = {"hot": {}, "warm": {}, "archive": {}}
        for event in events:
            age = latest_round - event.get("round_id", 0)
            if age < hot_max_rounds:
                tier = "hot"
            elif age < warm_max_rounds:
                tier = "warm"
            else:
                tier = "archive"
            grouped[tier].setdefault(event["cue"], []).append(event)

        tier_defs = {
            "hot": (self.episodic_hot_dir, 7),
            "warm": (self.episodic_warm_dir, 30),
            "archive": (self.episodic_archive_dir, None),
        }
        summary = {"latest_round": latest_round, "raw_event_count": len(events), "tiers": {}}
        for tier, (directory, retention_days) in tier_defs.items():
            artifacts = [
                self._summarize_artifact(tier, cue, cue_events, retention_days)
                for cue, cue_events in sorted(grouped[tier].items())
            ]
            artifact_count = self._write_artifacts(directory, artifacts)
            summary["tiers"][tier] = {
                "directory": str(directory),
                "artifact_count": artifact_count,
            }
        return summary

    def sample_compacted(self, tier: str, *, limit: int = 5, cue: str | None = None) -> list[dict]:
        if tier not in self._artifact_cache:
            raise ValueError(f"unsupported tier: {tier}")
        rows = copy.deepcopy(self._artifact_cache[tier])
        if cue:
            rows = [row for row in rows if row.get("cue") == cue]
        rows = sorted(rows, key=lambda item: (item.get("event_count", 0), item.get("last_round_id", 0), item.get("cue", "")), reverse=True)
        return rows[:limit]

    def flush(self, *, raise_on_error: bool = False) -> None:
        self._io_worker.flush(raise_on_error=raise_on_error)
