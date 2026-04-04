from __future__ import annotations

from typing import Any


class ProtocolError(ValueError):
    pass


INBOUND_REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "start_session": ("session_id", "cwd"),
    "user_turn": ("session_id", "text"),
    "control_command": ("session_id", "command"),
    "approve": ("session_id", "call_id", "approved"),
    "close_session": ("session_id",),
}

OUTBOUND_EVENT_TYPES = {
    "session_started",
    "sidebar_snapshot",
    "assistant_token",
    "run_status",
    "step_update",
    "tool_call",
    "tool_result",
    "approval_request",
    "assistant_final",
    "error",
    "session_ended",
}


def _require_str(payload: dict[str, Any], field_name: str) -> None:
    value = payload.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ProtocolError(f"{field_name} must be a non-empty string")


def validate_inbound_event(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ProtocolError("event payload must be an object")
    event_type = payload.get("type")
    if not isinstance(event_type, str) or event_type not in INBOUND_REQUIRED_FIELDS:
        raise ProtocolError(f"unsupported event type: {event_type}")

    for field_name in INBOUND_REQUIRED_FIELDS[event_type]:
        if field_name not in payload:
            raise ProtocolError(f"{event_type} requires field: {field_name}")

    _require_str(payload, "session_id")
    if event_type == "start_session":
        _require_str(payload, "cwd")
        if "persist_current" in payload and not isinstance(payload.get("persist_current"), bool):
            raise ProtocolError("persist_current must be a boolean when provided")
    if event_type == "user_turn":
        _require_str(payload, "text")
    if event_type == "control_command":
        _require_str(payload, "command")
        if "value" in payload:
            value = payload.get("value")
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ProtocolError("value must be a non-empty string when provided")
    if event_type == "approve":
        _require_str(payload, "call_id")
        if not isinstance(payload.get("approved"), bool):
            raise ProtocolError("approved must be a boolean")
    if event_type == "close_session":
        if "detach" in payload and not isinstance(payload.get("detach"), bool):
            raise ProtocolError("detach must be a boolean when provided")
        if "transcript_mode" in payload:
            mode = payload.get("transcript_mode")
            if mode not in {"full", "compact"}:
                raise ProtocolError("transcript_mode must be 'full' or 'compact' when provided")
    return payload


def build_outbound_event(event_type: str, **fields: Any) -> dict[str, Any]:
    if event_type not in OUTBOUND_EVENT_TYPES:
        raise ProtocolError(f"unsupported outbound event type: {event_type}")
    return {"type": event_type, **fields}
