from __future__ import annotations

import json
import random
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from nalr.runtime.metadata import utc_now_iso


def _parse_iso(value: str | None) -> datetime:
    normalized = str(value or "").strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    if normalized:
        try:
            parsed = datetime.fromisoformat(normalized)
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=timezone.utc)
            return parsed
        except ValueError:
            pass
    return datetime.now(timezone.utc)


def _to_iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


class MonologueStreamRuntime:
    def __init__(self, storage_path: Path, default_settings: dict[str, Any] | None = None) -> None:
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.storage_path.exists():
            self.storage_path.write_text("", encoding="utf-8")
        self._defaults = self.normalize_settings(default_settings or {})

    def default_settings(self) -> dict[str, Any]:
        return dict(self._defaults)

    def normalize_settings(self, payload: dict[str, Any]) -> dict[str, Any]:
        defaults = {
            "hidden_by_default": True,
            "bootstrap_seconds": 6,
            "min_interval_ms": 120,
            "max_interval_ms": 900,
            "max_catch_up_pulses": 24,
            "min_fragments_per_pulse": 1,
            "max_fragments_per_pulse": 3,
            "default_show_limit": 12,
            "generator_mode": "local",
        }
        defaults.update(dict(payload or {}))
        min_interval_ms = max(40, int(defaults.get("min_interval_ms", 120) or 120))
        max_interval_ms = max(min_interval_ms, int(defaults.get("max_interval_ms", 900) or 900))
        max_catch_up_pulses = max(1, int(defaults.get("max_catch_up_pulses", 24) or 24))
        min_fragments = max(1, int(defaults.get("min_fragments_per_pulse", 1) or 1))
        max_fragments = max(min_fragments, int(defaults.get("max_fragments_per_pulse", 3) or 3))
        generator_mode = str(defaults.get("generator_mode", "local") or "local").strip().lower()
        if generator_mode not in {"local", "model"}:
            generator_mode = "local"
        return {
            "hidden_by_default": bool(defaults.get("hidden_by_default", True)),
            "bootstrap_seconds": max(1, int(defaults.get("bootstrap_seconds", 6) or 6)),
            "min_interval_ms": min_interval_ms,
            "max_interval_ms": max_interval_ms,
            "max_catch_up_pulses": max_catch_up_pulses,
            "min_fragments_per_pulse": min_fragments,
            "max_fragments_per_pulse": max_fragments,
            "default_show_limit": max(1, int(defaults.get("default_show_limit", 12) or 12)),
            "generator_mode": generator_mode,
        }

    def merge_settings(self, *payloads: dict[str, Any] | None) -> dict[str, Any]:
        merged = self.default_settings()
        for payload in payloads:
            if isinstance(payload, dict):
                merged.update(payload)
        return self.normalize_settings(merged)

    def ensure_bucket(self, bucket: dict[str, Any] | None, *, now_iso: str | None = None) -> dict[str, Any]:
        now_dt = _parse_iso(now_iso or utc_now_iso())
        data = dict(bucket or {})
        settings = self.merge_settings(data.get("settings"))
        started_at = _parse_iso(data.get("started_at")) if data.get("started_at") else now_dt - timedelta(seconds=settings["bootstrap_seconds"])
        next_pulse_at = _parse_iso(data.get("next_pulse_at")) if data.get("next_pulse_at") else started_at
        return {
            "seed": str(data.get("seed") or uuid4().hex),
            "started_at": _to_iso(started_at),
            "next_pulse_at": _to_iso(next_pulse_at),
            "last_generated_at": str(data.get("last_generated_at") or ""),
            "last_viewed_at": str(data.get("last_viewed_at") or ""),
            "pulse_index": max(0, int(data.get("pulse_index", 0) or 0)),
            "generated_total": max(0, int(data.get("generated_total", 0) or 0)),
            "category_counts": {
                str(key): max(0, int(value or 0))
                for key, value in dict(data.get("category_counts", {}) or {}).items()
            },
            "settings": settings,
        }

    def prepare_catch_up(self, bucket: dict[str, Any] | None, *, now_iso: str | None = None) -> dict[str, Any]:
        now_dt = _parse_iso(now_iso or utc_now_iso())
        data = self.ensure_bucket(bucket, now_iso=_to_iso(now_dt))
        settings = self.merge_settings(data.get("settings"))
        seed = str(data["seed"])
        pulse_index = int(data["pulse_index"])
        next_pulse_dt = _parse_iso(data["next_pulse_at"])
        due_pulses: deque[dict[str, Any]] = deque(maxlen=int(settings["max_catch_up_pulses"]))

        while next_pulse_dt <= now_dt:
            interval_rng = random.Random(f"{seed}:pulse:{pulse_index}")
            fragment_rng = random.Random(f"{seed}:pulse:{pulse_index}")
            due_pulses.append(
                {
                    "pulse_index": pulse_index,
                    "pulse_dt": _to_iso(next_pulse_dt),
                    "fragment_count": fragment_rng.randint(
                        int(settings["min_fragments_per_pulse"]),
                        int(settings["max_fragments_per_pulse"]),
                    ),
                }
            )
            pulse_index += 1
            next_pulse_dt = next_pulse_dt + timedelta(
                milliseconds=interval_rng.randint(
                    int(settings["min_interval_ms"]),
                    int(settings["max_interval_ms"]),
                )
            )

        return {
            "bucket": data,
            "seed": seed,
            "settings": settings,
            "now_iso": _to_iso(now_dt),
            "pulse_index": pulse_index,
            "next_pulse_at": _to_iso(next_pulse_dt),
            "due_pulses": list(due_pulses),
        }

    def execute_catch_up(
        self,
        prepared: dict[str, Any],
        *,
        fragment_builder: Callable[..., list[dict[str, Any]]] | None = None,
    ) -> list[dict[str, Any]]:
        settings = self.merge_settings(dict(prepared.get("settings", {}) or {}))
        seed = str(prepared.get("seed") or dict(prepared.get("bucket", {}) or {}).get("seed") or uuid4().hex)
        generated: list[dict[str, Any]] = []
        for pulse in list(prepared.get("due_pulses", []) or []):
            pulse_data = dict(pulse or {})
            rows = self._build_pulse_fragments(
                seed=seed,
                pulse_index=int(pulse_data.get("pulse_index", 0) or 0),
                pulse_dt=_parse_iso(str(pulse_data.get("pulse_dt") or prepared.get("now_iso") or utc_now_iso())),
                fragment_count=max(1, int(pulse_data.get("fragment_count", 1) or 1)),
                generator_mode=str(settings["generator_mode"]),
                fragment_builder=fragment_builder,
            )
            if rows:
                generated.extend(rows)
        return generated

    def commit_catch_up(
        self,
        prepared: dict[str, Any],
        generated: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        data = dict(prepared.get("bucket", {}) or {})
        settings = self.merge_settings(prepared.get("settings"), data.get("settings"))
        data["settings"] = settings
        data["pulse_index"] = max(0, int(prepared.get("pulse_index", data.get("pulse_index", 0)) or 0))
        data["next_pulse_at"] = str(prepared.get("next_pulse_at") or data.get("next_pulse_at") or _to_iso(_parse_iso(prepared.get("now_iso"))))
        if generated:
            data["last_generated_at"] = generated[-1]["recorded_at"]
            data["generated_total"] = int(data.get("generated_total", 0) or 0) + len(generated)
            category_counts = dict(data.get("category_counts", {}) or {})
            for row in generated:
                category = str(row.get("category") or "unclassified")
                category_counts[category] = int(category_counts.get(category, 0) or 0) + 1
            data["category_counts"] = category_counts
            self._append_fragments(generated)
        return data, generated

    def catch_up(
        self,
        bucket: dict[str, Any] | None,
        *,
        now_iso: str | None = None,
        fragment_builder: Callable[..., list[dict[str, Any]]] | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        prepared = self.prepare_catch_up(bucket, now_iso=now_iso)
        generated = self.execute_catch_up(prepared, fragment_builder=fragment_builder)
        return self.commit_catch_up(prepared, generated)

    def status_payload(self, bucket: dict[str, Any], *, now_iso: str | None = None) -> dict[str, Any]:
        settings = self.merge_settings(bucket.get("settings"))
        category_counts = dict(bucket.get("category_counts", {}) or {})
        now_dt = _parse_iso(now_iso or utc_now_iso())
        next_pulse_at = str(bucket.get("next_pulse_at") or "")
        next_pulse_dt = _parse_iso(next_pulse_at) if next_pulse_at else None
        overdue_seconds = 0.0
        if next_pulse_dt is not None and next_pulse_dt <= now_dt:
            overdue_seconds = max(0.0, (now_dt - next_pulse_dt).total_seconds())
        top_categories = [
            {"category": name, "count": count}
            for name, count in sorted(category_counts.items(), key=lambda item: (-int(item[1]), str(item[0])))[:5]
        ]
        return {
            "summary": "monologue status",
            "hidden": bool(settings["hidden_by_default"]),
            "generator_mode": str(settings["generator_mode"]),
            "generated_total": int(bucket.get("generated_total", 0) or 0),
            "pulse_index": int(bucket.get("pulse_index", 0) or 0),
            "started_at": str(bucket.get("started_at") or ""),
            "last_generated_at": str(bucket.get("last_generated_at") or ""),
            "last_viewed_at": str(bucket.get("last_viewed_at") or ""),
            "next_pulse_at": next_pulse_at,
            "next_pulse_due": bool(overdue_seconds > 0.0),
            "stale": bool(overdue_seconds > 0.0),
            "overdue_seconds": round(float(overdue_seconds), 3),
            "storage_path": str(self.storage_path),
            "top_categories": top_categories,
        }

    def show_payload(self, bucket: dict[str, Any], *, limit: int | None = None) -> dict[str, Any]:
        settings = self.merge_settings(bucket.get("settings"))
        resolved_limit = max(1, int(limit or settings["default_show_limit"] or 12))
        rows = self.read_fragments(limit=resolved_limit)
        return {
            "summary": "monologue show",
            "hidden": bool(settings["hidden_by_default"]),
            "returned": len(rows),
            "limit": resolved_limit,
            "generated_total": int(bucket.get("generated_total", 0) or 0),
            "fragments": rows,
        }

    def read_fragments(self, *, limit: int) -> list[dict[str, Any]]:
        resolved_limit = max(1, int(limit))
        try:
            with self.storage_path.open("r", encoding="utf-8") as handle:
                tail_lines = list(deque(handle, maxlen=resolved_limit))
        except OSError:
            return []
        rows: list[dict[str, Any]] = []
        for line in tail_lines:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
        return rows

    def _append_fragments(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        with self.storage_path.open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _build_pulse_fragments(
        self,
        *,
        seed: str,
        pulse_index: int,
        pulse_dt: datetime,
        fragment_count: int,
        generator_mode: str,
        fragment_builder: Callable[..., list[dict[str, Any]]] | None = None,
    ) -> list[dict[str, Any]]:
        if generator_mode == "model" and callable(fragment_builder):
            model_rows = fragment_builder(
                seed=seed,
                pulse_index=pulse_index,
                pulse_dt=_to_iso(pulse_dt),
                fragment_count=fragment_count,
            )
            if model_rows:
                return [self._normalize_fragment_row(row, default_dt=pulse_dt, index=offset) for offset, row in enumerate(model_rows)]
        rows: list[dict[str, Any]] = []
        for offset in range(fragment_count):
            row = self._local_fragment(seed=seed, pulse_index=pulse_index, offset=offset, pulse_dt=pulse_dt)
            rows.append(row)
        return rows

    def _normalize_fragment_row(self, row: dict[str, Any], *, default_dt: datetime, index: int) -> dict[str, Any]:
        normalized = dict(row or {})
        recorded_at = normalized.get("recorded_at") or _to_iso(default_dt + timedelta(milliseconds=index * 80))
        return {
            "fragment_id": str(normalized.get("fragment_id") or uuid4().hex),
            "recorded_at": str(recorded_at),
            "category": str(normalized.get("category") or "model_fragment"),
            "content": str(normalized.get("content") or "").strip(),
            "source": str(normalized.get("source") or "model"),
        }

    def _local_fragment(self, *, seed: str, pulse_index: int, offset: int, pulse_dt: datetime) -> dict[str, Any]:
        rng = random.Random(f"{seed}:fragment:{pulse_index}:{offset}")
        category = rng.choice(
            [
                "environment_notice",
                "free_association",
                "memory_fragment",
                "daydream",
                "blank_fragment",
            ]
        )
        content = {
            "environment_notice": self._environment_notice(rng),
            "free_association": self._free_association(rng),
            "memory_fragment": self._memory_fragment(rng),
            "daydream": self._daydream(rng),
            "blank_fragment": self._blank_fragment(rng),
        }[category]
        return {
            "fragment_id": uuid4().hex,
            "recorded_at": _to_iso(pulse_dt + timedelta(milliseconds=offset * 80)),
            "category": category,
            "content": content,
            "source": "local",
        }

    def _environment_notice(self, rng: random.Random) -> str:
        subjects = ["界面字体", "光标", "窗口边缘", "设备反应", "屏幕亮度", "这个页面", "手边的节奏"]
        qualities = ["有点小", "停得有点久", "显得有点挤", "像是慢了一拍", "有点刺眼", "看着太满了", "突然变得明显"]
        return f"{rng.choice(subjects)}{rng.choice(qualities)}"

    def _free_association(self, rng: random.Random) -> str:
        variants = [
            "这个图标有点像苹果",
            "1+1 为什么就是 2",
            "突然想到一个词：回声",
            "那个形状有点像鲸鱼",
            "脑子里跳出“折返”这个词",
            "突然想起一个没来由的比喻",
        ]
        return rng.choice(variants)

    def _memory_fragment(self, rng: random.Random) -> str:
        variants = [
            "昨天聊过的话题好像还没说完",
            "之前看到的图片还有一点残影",
            "上次停住的地方像是还挂着",
            "有个念头像前几天出现过",
            "刚才那一闪而过的东西有点熟",
            "好像在哪里见过这种配色",
        ]
        return rng.choice(variants)

    def _daydream(self, rng: random.Random) -> str:
        variants = [
            "如果现在去散步会怎么样",
            "要是能换个背景就好了",
            "如果把这件事拖到晚上呢",
            "突然想知道窗外现在什么样",
            "要是切到另一段节奏会不会更顺",
            "如果现在什么都不做会怎样",
        ]
        return rng.choice(variants)

    def _blank_fragment(self, rng: random.Random) -> str:
        return rng.choice(["啊…", "嗯…", "……", "突然空了一下", "刚刚像是愣了一瞬", "脑子里飘过去一个没成形的点"])
