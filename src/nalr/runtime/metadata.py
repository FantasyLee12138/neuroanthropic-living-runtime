from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def iso_date(recorded_at: str) -> str:
    return recorded_at[:10]


def ensure_recorded_fields(
    payload: dict[str, Any],
    *,
    session_id: str = "legacy",
    recorded_at: str | None = None,
) -> dict[str, Any]:
    normalized = dict(payload)
    effective_recorded_at = normalized.get("recorded_at") or recorded_at or utc_now_iso()
    normalized["session_id"] = normalized.get("session_id") or session_id
    normalized["recorded_at"] = effective_recorded_at
    normalized["recorded_date"] = normalized.get("recorded_date") or iso_date(effective_recorded_at)
    return normalized
