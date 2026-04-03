from __future__ import annotations

from typing import Any


def _compact_bool(value: bool) -> str:
    return "yes" if value else "no"


def _compact_delta(delta_p: dict[str, float]) -> str:
    if not delta_p:
        return "-"
    action, score = max(delta_p.items(), key=lambda item: abs(item[1]))
    return f"{action}={score:.2f}"


def format_chat_turn(payload: dict[str, Any]) -> str:
    action_name = payload.get("sampled_action", {}).get("name", "unknown")
    reply = payload.get("rendered_expression", {}).get("text", "").strip() or "(empty)"
    return "\n".join(
        [
            f"Round {payload.get('round_id', '?')}",
            f"Action: {action_name}",
            f"NALR: {reply}",
        ]
    )


def format_why_view(payload: dict[str, Any]) -> str:
    lines = [
        f"Why This Turn (round {payload.get('round_id', '?')})",
        f"Sampled Action: {payload.get('sampled_action', 'unknown')}",
    ]
    for driver in payload.get("top_drivers", []):
        lines.append(
            f"- {driver.get('agent_name')} -> {driver.get('action_name')} "
            f"(score={driver.get('score', 0.0):.4f}) {driver.get('reason', '')}".rstrip()
        )
    return "\n".join(lines)


def format_agents_view(payload: dict[str, Any]) -> str:
    lines = [
        f"Agent Proposals (round {payload.get('round_id', '?')})",
        f"Sampled Action: {payload.get('sampled_action', 'unknown')}",
    ]
    agents = payload.get("agents", [])
    if not agents:
        lines.append("(none)")
        return "\n".join(lines)
    for row in agents:
        lines.append(
            f"- {row.get('stage')}/{row.get('agent_name')} -> {row.get('top_action') or '-'} "
            f"conf={row.get('confidence', 0.0):.2f} "
            f"delta={_compact_delta(row.get('delta_p', {}))} "
            f"selected={_compact_bool(bool(row.get('selected')))} "
            f"veto={_compact_bool(bool(row.get('veto')))}"
        )
    return "\n".join(lines)


def format_skills_view(payload: dict[str, Any]) -> str:
    lines = [
        f"Skill Trace (round {payload.get('round_id', '?')})",
        f"Sampled Action: {payload.get('sampled_action', 'unknown')}",
    ]
    skills = payload.get("skills", [])
    if not skills:
        lines.append("(none)")
        return "\n".join(lines)
    for row in skills:
        lines.append(
            f"- {row.get('skill_name')} [{row.get('owner_module')}] "
            f"latency={row.get('latency_ms', 0)}ms "
            f"degraded={_compact_bool(bool(row.get('degraded')))} "
            f"policy={row.get('failure_policy_applied') or '-'}"
        )
    return "\n".join(lines)


def format_gates_view(payload: dict[str, Any]) -> str:
    lines = [
        f"Gate Decisions (round {payload.get('round_id', '?')})",
        f"Sampled Action: {payload.get('sampled_action', 'unknown')}",
    ]
    gates = payload.get("gates", [])
    if not gates:
        lines.append("(none)")
        return "\n".join(lines)
    for row in gates:
        lines.append(
            f"- {row.get('stage')}/{row.get('owner')} "
            f"allowed={_compact_bool(bool(row.get('allowed')))} "
            f"resample={_compact_bool(bool(row.get('requires_resample')))} "
            f"reason={row.get('reason', '')}"
        )
    return "\n".join(lines)
