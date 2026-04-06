from __future__ import annotations

from typing import Any


def _compact_bool(value: bool) -> str:
    return "yes" if value else "no"


def _compact_delta(delta_p: dict[str, float]) -> str:
    if not delta_p:
        return "-"
    action, score = max(delta_p.items(), key=lambda item: abs(item[1]))
    return f"{action}={score:.2f}"


def _compact_distribution(distribution: dict[str, float], *, limit: int = 3) -> str:
    if not distribution:
        return "-"
    rows = sorted(distribution.items(), key=lambda item: float(item[1]), reverse=True)[:limit]
    return ", ".join(f"{name}={float(score):.3f}" for name, score in rows)


def _compact_peak_rows(rows: list[dict[str, Any]], *, limit: int = 3) -> str:
    if not rows:
        return "-"
    items: list[str] = []
    for row in rows[:limit]:
        items.append(f"{row.get('action', row.get('target', '-'))}={float(row.get('posterior', 0.0) or 0.0):.3f}")
    return ", ".join(items) or "-"


def _audit_strength(row: dict[str, Any]) -> str:
    normalized = row.get("delta_normalized", {})
    if isinstance(normalized, dict) and normalized:
        key, value = max(normalized.items(), key=lambda item: abs(float(item[1])))
        return f"{key}={float(value):.2f}"
    return "-"


def _storage_line(payload: dict[str, Any]) -> str | None:
    storage = payload.get("storage")
    if not storage:
        return None
    return (
        "Storage: "
        f"source={storage.get('read_source', storage.get('read_source_default', '-'))} "
        f"sync={storage.get('trace_sync_state', '-')} "
        f"ready={_compact_bool(bool(storage.get('parquet_live_ready')))}"
    )


def format_chat_turn(payload: dict[str, Any]) -> str:
    action_name = payload.get("sampled_action", {}).get("name", "unknown")
    reply = payload.get("rendered_expression", {}).get("text", "").strip() or "(empty)"
    identity = payload.get("identity", {})
    display_name = identity.get("display_name") or "当前运行体"
    return "\n".join(
        [
            f"Round {payload.get('round_id', '?')}",
            f"Action: {action_name}",
            f"{display_name}: {reply}",
        ]
    )


def format_why_view(payload: dict[str, Any]) -> str:
    lines = [
        f"Why This Turn (round {payload.get('round_id', '?')})",
        f"Sampled Action: {payload.get('sampled_action', 'unknown')}",
    ]
    storage_line = _storage_line(payload)
    if storage_line:
        lines.append(storage_line)
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
    storage_line = _storage_line(payload)
    if storage_line:
        lines.append(storage_line)
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
    storage_line = _storage_line(payload)
    if storage_line:
        lines.append(storage_line)
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
    storage_line = _storage_line(payload)
    if storage_line:
        lines.append(storage_line)
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


def format_probability_view(payload: dict[str, Any]) -> str:
    lines = [
        f"Probability Field (round {payload.get('round_id', '?')})",
        f"Sampled Action: {payload.get('sampled_action', 'unknown')}",
    ]
    storage_line = _storage_line(payload)
    if storage_line:
        lines.append(storage_line)
    probability_field = payload.get("probability_field", {})
    for layer_name in ("context", "memory", "action", "token"):
        layer = probability_field.get(layer_name, {}) if isinstance(probability_field, dict) else {}
        if not isinstance(layer, dict):
            continue
        lines.append(
            f"{layer_name}: "
            f"winner={layer.get('winner_target') or '-'} "
            f"top={_compact_distribution(layer.get('winner_posterior', {}))} "
            f"masked={len(layer.get('hard_masked_targets', []) or [])} "
            f"audit={len(layer.get('contribution_audit', []) or [])}"
        )
    token_state = payload.get("token_state", {})
    if isinstance(token_state, dict) and token_state:
        lines.append(
            "token_state: "
            f"step={token_state.get('step_index', '-')} "
            f"active={len(token_state.get('active_module_sources', []) or [])} "
            f"generated={len(token_state.get('generated_delta_sources', []) or [])}"
        )
    verdict = payload.get("cross_layer_coupling_verdict", {})
    if isinstance(verdict, dict) and verdict:
        lines.append(
            "couplings: "
            f"legal={_compact_bool(bool(verdict.get('legal', False)))} "
            f"observed={len(verdict.get('observed_pairs', []) or [])} "
            f"illegal={len(verdict.get('illegal_pairs', []) or [])}"
        )
    return "\n".join(lines)


def format_probability_layer_view(payload: dict[str, Any]) -> str:
    layer_name = str(payload.get("layer", "unknown"))
    layer = payload.get("layer_state", {})
    lines = [
        f"Probability Layer {layer_name} (round {payload.get('round_id', '?')})",
        f"Sampled Action: {payload.get('sampled_action', 'unknown')}",
    ]
    storage_line = _storage_line(payload)
    if storage_line:
        lines.append(storage_line)
    if not isinstance(layer, dict):
        lines.append("(empty)")
        return "\n".join(lines)
    lines.extend(
        [
            f"winner={layer.get('winner_target') or '-'}",
            f"top={_compact_distribution(layer.get('winner_posterior', {}))}",
            f"peaks={_compact_peak_rows(layer.get('counterfactual_top_peaks', []))}",
            f"masked={len(layer.get('hard_masked_targets', []) or [])}",
            f"base={len(layer.get('base_energy', {}) or {})} final={len(layer.get('final_energy', {}) or {})}",
            f"aggregated={_compact_delta(layer.get('aggregated_normalized_delta', {}))}",
            "audit:",
        ]
    )
    audit_rows = layer.get("contribution_audit", []) if isinstance(layer.get("contribution_audit", []), list) else []
    if not audit_rows:
        lines.append("(none)")
        return "\n".join(lines)
    for row in audit_rows[:8]:
        if not isinstance(row, dict):
            continue
        lines.append(
            f"- {row.get('module_name')} "
            f"conf={float(row.get('confidence_calibrated', row.get('confidence_raw', 0.0)) or 0.0):.2f} "
            f"delta={_audit_strength(row)} "
            f"masked={len(row.get('hard_masked_targets', []) or [])}"
        )
    return "\n".join(lines)


def format_action_probability_view(payload: dict[str, Any]) -> str:
    explanation = payload.get("action_probability_explanation", {})
    target = payload.get("action", explanation.get("target_action", "unknown"))
    lines = [
        f"Action Probability (round {payload.get('round_id', '?')})",
        f"Sampled Action: {payload.get('sampled_action', 'unknown')}",
    ]
    storage_line = _storage_line(payload)
    if storage_line:
        lines.append(storage_line)
    if not isinstance(explanation, dict):
        lines.append("(empty)")
        return "\n".join(lines)
    winner_posterior = explanation.get("winner_posterior", {})
    lines.extend(
        [
            f"target={target} winner={explanation.get('winner_target') or '-'}",
            f"target_p={float((winner_posterior or {}).get(target, 0.0) or 0.0):.3f} "
            f"winner_p={max((float(value) for value in (winner_posterior or {}).values()), default=0.0):.3f} "
            f"hard_masked={_compact_bool(bool(explanation.get('hard_masked')))}",
            f"peaks={_compact_peak_rows(explanation.get('competing_peaks', []))}",
            "stacked:",
        ]
    )
    stacked = explanation.get("stacked_contributions", [])
    if not isinstance(stacked, list) or not stacked:
        lines.append("(none)")
        return "\n".join(lines)
    for row in stacked[:10]:
        if not isinstance(row, dict):
            continue
        lines.append(
            f"- {row.get('module_name')} "
            f"{row.get('direction', 'neutral')} "
            f"norm={float(row.get('delta_normalized', 0.0) or 0.0):.2f} "
            f"proj={float(row.get('delta_projected', 0.0) or 0.0):.2f}"
        )
    return "\n".join(lines)
