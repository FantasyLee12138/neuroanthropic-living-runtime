from __future__ import annotations

import json
import math
import threading
import time
from collections import defaultdict
from typing import Any


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return value


def _projection_truth_contract(event: dict[str, Any]) -> dict[str, Any]:
    event_log_ref = dict(event.get("event_log_ref", {}) or {})
    return {
        "layer": "projection",
        "projection_only": True,
        "authoritative": False,
        "source": "web_session_broker",
        "authoritative_source": "authoritative_runtime_state",
        "event_log_backed": bool(event_log_ref.get("round_trace_ref") or event.get("trace_ref")),
    }


class WebSessionBroker:
    def __init__(self, *, max_events_per_session: int = 2048) -> None:
        self._events: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._next_ids: dict[str, int] = defaultdict(lambda: 1)
        self._lock = threading.Lock()
        self._max_events_per_session = max(1, int(max_events_per_session))

    def append_many(self, session_id: str, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        recorded: list[dict[str, Any]] = []
        with self._lock:
            for event in events:
                event_id = self._next_ids[session_id]
                self._next_ids[session_id] += 1
                payload = _json_safe({"event_id": event_id, **event})
                payload["truth_contract"] = _projection_truth_contract(payload)
                payload["truth_layer"] = "projection"
                payload["projection_only"] = True
                payload["authoritative"] = False
                self._events[session_id].append(payload)
                recorded.append(payload)
            retained = self._events[session_id]
            overflow = len(retained) - self._max_events_per_session
            if overflow > 0:
                del retained[:overflow]
        return recorded

    def read_since(self, session_id: str, after_id: int = 0) -> list[dict[str, Any]]:
        with self._lock:
            events = self._events.get(session_id, [])
            if not events:
                return []
            first_event_id = int(events[0].get("event_id", 0) or 0)
            last_event_id = int(events[-1].get("event_id", 0) or 0)
            if after_id >= last_event_id:
                return []
            start_index = 0 if after_id < first_event_id else max(0, after_id - first_event_id + 1)
            return list(events[start_index:])

    def stream_sse(self, session_id: str, *, after_id: int = 0, once: bool = False, timeout_seconds: float = 15.0):
        last_seen = after_id
        deadline = time.monotonic() + timeout_seconds
        while True:
            pending = self.read_since(session_id, after_id=last_seen)
            if pending:
                for event in pending:
                    last_seen = int(event["event_id"])
                    yield f"id: {event['event_id']}\ndata: {json.dumps(event, ensure_ascii=False, allow_nan=False)}\n\n"
                if once:
                    return
                deadline = time.monotonic() + timeout_seconds
                continue
            if once and time.monotonic() >= deadline:
                return
            if time.monotonic() >= deadline:
                yield ": keep-alive\n\n"
                deadline = time.monotonic() + timeout_seconds
            time.sleep(0.2)

    def clear(self, session_id: str | None = None) -> None:
        with self._lock:
            if session_id is None:
                self._events.clear()
                self._next_ids.clear()
                return
            self._events.pop(session_id, None)
            self._next_ids.pop(session_id, None)


def _scale_text(value: Any) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "0.00"


def _action_label(action: Any) -> str:
    labels = {
        "respond": "回应生成",
        "plan": "规划求解",
        "recall": "记忆检索",
        "rest": "静息回落",
        "connect": "关系联结",
        "clarify": "澄清求证",
        "wander": "游移漫游",
        "absorb": "吸收沉积",
        "nothing": "保持静默",
        "die": "自主结束生命",
        "short_reply": "简短回应",
        "acknowledge_fatigue": "承认疲劳状态",
    }
    name = str(action or "").strip()
    return labels.get(name, name or "暂无")


def _humanize_label(value: Any) -> str:
    labels = {
        "context": "情境层",
        "memory": "记忆层",
        "action": "动作层",
        "token": "Token",
        "PerspectiveModel": "视角模型",
        "EndogenousMotivationPool": "内生动机池",
        "StochasticPolicy": "随机策略",
        "BodyStateAgent": "身体状态代理",
        "PFCAgent": "前额叶控制代理",
    }
    text = str(value or "").strip()
    return labels.get(text, text or "暂无")


def _region_label(value: Any) -> str:
    labels = {
        "express": "外显表达区",
        "withdraw": "退避收缩区",
        "hibernate": "休眠回落区",
        "dissolve": "解体消散区",
        "absorb": "吸收沉积区",
    }
    text = str(value or "").strip()
    return labels.get(text, text or "暂无")


def _compact_text(values: Any, *, fallback: str = "暂无", limit: int = 4) -> str:
    if isinstance(values, dict):
        items = [f"{key}={_scale_text(value)}" for key, value in list(values.items())[:limit]]
    elif isinstance(values, (list, tuple, set)):
        items = [str(item).strip() for item in list(values)[:limit] if str(item).strip()]
    else:
        text = str(values or "").strip()
        items = [text] if text else []
    if not items:
        return fallback
    return "、".join(items)


def _mapping_preview(values: Any, *, keys: tuple[str, ...] | None = None, limit: int = 4) -> str:
    if not isinstance(values, dict) or not values:
        return "暂无"
    items: list[str] = []
    if keys:
        for key in keys:
            if key not in values:
                continue
            items.append(f"{key}={_compact_text(values.get(key), fallback='暂无', limit=limit)}")
    else:
        for key, value in list(values.items())[:limit]:
            items.append(f"{key}={_compact_text(value, fallback='暂无', limit=limit)}")
    return "；".join(items) if items else "暂无"


def _layer_rows_from_model(model_layers: Any) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if not isinstance(model_layers, list):
        return rows
    for index, row in enumerate(model_layers[:6], start=1):
        if not isinstance(row, dict):
            continue
        rows.append(
            {
                "label": str(row.get("label") or row.get("name") or f"layer_{index}"),
                "transform": str(
                    row.get("transform_summary")
                    or row.get("transform")
                    or row.get("summary")
                    or "暂无"
                ),
                "output": str(
                    row.get("output_vector_summary")
                    or row.get("output_vector")
                    or row.get("vector")
                    or row.get("output")
                    or "暂无"
                ),
            }
        )
    return rows


def _fallback_chain_rows(state: dict[str, Any], action_field: dict[str, Any], probability_field: dict[str, Any]) -> list[dict[str, str]]:
    tlh = dict(((state.get("cognitive_snapshot") or {}).get("tlh") or {}))
    instinct_field = dict(tlh.get("instinct_field") or {})
    body_state = dict(tlh.get("body_state") or {})
    subjective_state = dict(tlh.get("subjective_state") or {})
    emotion_state = dict(tlh.get("emotion_state") or {})
    desire_state = dict(tlh.get("desire_state") or {})
    action_layer = dict(probability_field.get("action") or {})
    token_layer = dict(probability_field.get("token") or {})
    output_layer = {
        "context": probability_field.get("context") or {},
        "memory": probability_field.get("memory") or {},
        "action": action_layer,
        "token": token_layer,
    }
    return [
        {
            "label": "1. body_state",
            "transform": "身体基线写入当前运行态",
            "output": _mapping_preview(body_state, keys=("energy", "fatigue", "memory_fragments", "self_continuity", "meaning_strength")),
        },
        {
            "label": "2. subjective_state",
            "transform": "主观感受、边界与拒绝倾向被同步",
            "output": _mapping_preview(subjective_state, keys=("felt", "boundary", "spontaneous", "reject_all", "meaning_made")),
        },
        {
            "label": "3. emotion_state",
            "transform": "情绪场把当前价与激活度折进可读信号",
            "output": _mapping_preview(emotion_state, keys=("valence", "arousal", "dominance", "intensity")),
        },
        {
            "label": "4. desire_state",
            "transform": "欲望与目标张力进入候选偏置",
            "output": _mapping_preview(desire_state, keys=("top_goal", "active_drives", "strongest_drive", "goal_pressure")),
        },
        {
            "label": "5. instinct_field",
            "transform": "E/F/S/M 四轴与区域坍缩形成本轮落点",
            "output": _mapping_preview(instinct_field, keys=("winner_region", "axis_values", "region_scores", "collapse_trace")),
        },
        {
            "label": "6. action_field",
            "transform": "动作后验、竞争峰与解释栈完成收束",
            "output": _mapping_preview(
                action_field,
                keys=("winner", "top_actions", "competing_peaks", "contribution_stack"),
            ),
        },
        {
            "label": "7. probability_field",
            "transform": "四层概率场把上下文/记忆/动作/Token 重新对齐",
            "output": _mapping_preview(output_layer, keys=("context", "memory", "action", "token")),
        },
    ]


def _format_source_link_rows(source_links: Any, *, limit: int = 8) -> list[str]:
    rows: list[str] = []
    if not isinstance(source_links, list):
        return rows
    for entry in source_links[:limit]:
        if not isinstance(entry, dict):
            continue
        panel_id = str(entry.get("panel_id") or "panel").strip()
        api_path = str(entry.get("api_path") or "").strip()
        controller_method = str(entry.get("controller_method") or "").strip()
        parts = [part for part in [panel_id, api_path, controller_method] if part]
        if parts:
            rows.append(" · ".join(parts))
    return rows


def build_workbench_cards(snapshot: dict[str, Any]) -> dict[str, Any]:
    console = dict(snapshot.get("console") or {})
    state = dict(console.get("state") or {})
    action_field = dict(console.get("action_field") or {})
    why_current = dict((console.get("why_current") or {}).get("why") or {})
    why_not = dict((console.get("why_not") or {}).get("why_not") or {})
    cognitive_chain = dict(console.get("cognitive_chain") or {})
    controls_model = dict(console.get("controls") or {})
    alerts_model = dict(console.get("alerts") or {})
    dashboards_model = dict(console.get("dashboards") or {})
    tlh = dict(((state.get("cognitive_snapshot") or {}).get("tlh") or {}))
    instinct_field = dict(tlh.get("instinct_field") or {})
    probability_field = dict(console.get("probability_field") or {})
    winner = dict(action_field.get("winner") or {})
    subjective = dict(tlh.get("subjective_state") or {})
    axis_values = dict(instinct_field.get("axis_values") or {})
    axis_summary = ", ".join(
        f"{axis}={_scale_text(axis_values.get(axis))}"
        for axis in ("E", "F", "S", "M")
    )
    available_layers = ", ".join(
        _humanize_label(layer) for layer in ("context", "memory", "action", "token") if probability_field.get(layer)
    )
    blocked_by = ", ".join(_humanize_label(item) for item in (why_not.get("blocked_by") or [])[:3])
    source_link_rows = _format_source_link_rows(console.get("source_links") or snapshot.get("source_links") or [])
    recent_rounds = list(console.get("recent_rounds") or snapshot.get("recent_rounds") or [])
    autonomy = dict(console.get("autonomy") or state.get("autonomy") or {})
    service = dict(console.get("service") or snapshot.get("service") or {})
    behavior_policies = dict(controls_model.get("behavior_policies") or {})
    control_proposals = controls_model.get("proposal_count")
    dashboard_specs = dashboards_model.get("dashboard_specs")
    layer_fuses = alerts_model.get("layer_fuses")
    alert_rules = alerts_model.get("alert_rules")
    alert_history = alerts_model.get("alert_history")
    model_chain_rows = _layer_rows_from_model(cognitive_chain.get("layers") or cognitive_chain.get("chain"))
    if not model_chain_rows:
        model_chain_rows = _layer_rows_from_model(cognitive_chain.get("read_model_layers"))
    if not model_chain_rows:
        model_chain_rows = _fallback_chain_rows(state, action_field, probability_field)

    control_policy_rows = [
        f"运行配置：{_compact_text(controls_model.get('mode') or autonomy.get('profile') or state.get('mode'))}",
        f"学习边界：{_compact_text(controls_model.get('learning_mode') or autonomy.get('learning_mode'))}",
        f"允许命令：{_compact_text(behavior_policies.get('allowed_commands') or autonomy.get('allowed_commands'))}",
        f"阻断命令：{_compact_text(behavior_policies.get('blocked_commands') or autonomy.get('blocked_commands'))}",
        f"网络白名单：{_compact_text(behavior_policies.get('allowed_network_domains') or autonomy.get('allowed_network_domains'))}",
        f"可写根目录：{_compact_text(behavior_policies.get('writable_roots') or autonomy.get('writable_roots'))}",
    ]
    proposal_count = control_proposals
    if proposal_count is None:
        proposal_count = len(action_field.get("top_actions") or [])
    dashboard_spec_rows = [
        *source_link_rows[:6],
        *(
            [f"当前提案数：{_compact_text(proposal_count, fallback='0')}"]
            if proposal_count is not None
            else []
        ),
    ]
    fuse_rows = [
        f"服务连通：{_compact_text(service.get('healthy'))} / { _compact_text(service.get('http_ready')) }",
        f"运行串行：{_compact_text(service.get('runtime_serial_active'))}",
        f"观察者轮次：{_compact_text(service.get('observer_turn_active'))}",
        f"自治运行：{_compact_text(autonomy.get('running'))}",
        f"自治卡顿：{_compact_text(autonomy.get('stalled'))}",
        f"安全模式：{_compact_text(state.get('safe_mode'))}",
    ]
    if isinstance(layer_fuses, list) and layer_fuses:
        fuse_rows = [f"layer_fuses：{_compact_text(layer_fuses)}", *fuse_rows]
    rules_rows = [
        f"alerts rules：{_compact_text(alert_rules)}",
        f"告警摘要：{_compact_text(why_not.get('summary') or why_current.get('summary'))}",
        f"阻断项：{blocked_by or '暂无'}",
    ]
    recent_round_labels = [f"#{item.get('round_id')}: {_action_label(item.get('sampled_action'))}" for item in recent_rounds[-4:]]
    history_rows = [f"alerts history：{_compact_text(alert_history)}"]
    if recent_round_labels:
        history_rows.append(f"recent_rounds：{_compact_text(recent_round_labels, fallback='暂无', limit=4)}")

    cards = [
        {
            "panel_id": "cognitive_chain",
            "title": "认知链路",
            "summary": "六层链路从身体基线一路收束到动作与概率场，全部来自当前只读 snapshot。",
            "explanation": "这里优先展示 console.cognitive_chain；如果还没下发该读模型，就按当前 state / trace / probability_field 组合出等价的只读层级视图。",
            "rows": [
                *model_chain_rows[:6],
                {
                    "label": "输出向量",
                    "transform": "当前动作与概率场输出",
                    "output": _compact_text(
                        [
                            f"winner={_action_label(winner.get('action'))}",
                            f"posterior={_mapping_preview(action_field.get('winner_posterior'), limit=3)}",
                            f"region={_region_label(instinct_field.get('winner_region'))}",
                            f"axes={axis_summary}",
                        ]
                    ),
                },
            ],
        },
        {
            "panel_id": "controls",
            "title": "层级配置",
            "summary": "behavior_policies、proposal 数与 dashboard specs 都来自当前只读控制面。",
            "explanation": "这里优先展示 console.controls；如果控制面还没显式下发，就从 autonomy、action_field 和 source_links 读出当前边界。",
            "rows": [
                {
                    "label": "behavior_policies",
                    "transform": "运行、学习与权限边界的只读摘要",
                    "output": "；".join(control_policy_rows),
                },
                {
                    "label": "proposal 数",
                    "transform": "当前轮次动作候选被折叠成可读计数",
                    "output": f"当前动作提案数：{_compact_text(proposal_count, fallback='0')}；竞争峰数：{_compact_text(len(action_field.get('competing_peaks') or []), fallback='0')}",
                },
                {
                    "label": "dashboard specs",
                    "transform": "控制台面板的来源映射与读接口",
                    "output": _compact_text(dashboard_spec_rows, fallback="暂无"),
                },
            ],
        },
        {
            "panel_id": "alerts",
            "title": "应急熔断",
            "summary": "layer_fuses、alerts rules 与 alerts history 都是只读熔断视图，不接收前端回写。",
            "explanation": "这里优先展示 console.alerts；如果还没有显式读模型，就把服务健康、自治卡顿和 why_not 阻断项折叠成熔断摘要。",
            "rows": [
                {
                    "label": "layer_fuses",
                    "transform": "把高风险层级收束到熔断状态",
                    "output": "；".join(fuse_rows),
                },
                {
                    "label": "alerts rules",
                    "transform": "当前只读告警规则的摘要",
                    "output": "；".join(rules_rows),
                },
                {
                    "label": "alerts history",
                    "transform": "最近告警历史与阻断轮次",
                    "output": "；".join(history_rows),
                },
            ],
        },
        {
            "panel_id": "dashboards",
            "title": "仪表总览",
            "summary": "console.dashboards 会把当前可读面板、来源路径和最近轮次聚合成总览。",
            "explanation": "这块保持纯展示，不存本地控制态，也不单独生成第二真相面；只把当前 snapshot 里已有的面板来源和历史折叠出来。",
            "rows": [
                {
                    "label": "dashboard_specs",
                    "transform": "面板来源与控制台读接口",
                    "output": _compact_text(source_link_rows or dashboard_spec_rows, fallback="暂无"),
                },
                {
                    "label": "recent_rounds",
                    "transform": "最近轮次的只读活动摘要",
                    "output": _compact_text(
                        [f"#{item.get('round_id')}: {_action_label(item.get('sampled_action'))}" for item in recent_rounds[-6:]],
                        fallback="暂无",
                    ),
                },
                {
                    "label": "overview",
                    "transform": "当前控制台可见模型",
                    "output": _compact_text(
                        [
                            f"chain={len(model_chain_rows[:6])}",
                            f"controls={_compact_text(proposal_count, fallback='0')}",
                            f"alerts={_compact_text((alerts_model.get('alert_history') or alert_history or []), fallback='0')}",
                            f"layers={available_layers or '暂无'}",
                        ]
                    ),
                },
            ],
        },
    ]
    return {"cards": cards}
