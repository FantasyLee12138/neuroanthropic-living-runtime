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


class WebSessionBroker:
    def __init__(self) -> None:
        self._events: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._next_ids: dict[str, int] = defaultdict(lambda: 1)
        self._lock = threading.Lock()

    def append_many(self, session_id: str, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        recorded: list[dict[str, Any]] = []
        with self._lock:
            for event in events:
                event_id = self._next_ids[session_id]
                self._next_ids[session_id] += 1
                payload = _json_safe({"event_id": event_id, **event})
                self._events[session_id].append(payload)
                recorded.append(payload)
        return recorded

    def read_since(self, session_id: str, after_id: int = 0) -> list[dict[str, Any]]:
        with self._lock:
            events = list(self._events.get(session_id, []))
        return [event for event in events if int(event.get("event_id", 0) or 0) > after_id]

    def clear(self) -> None:
        with self._lock:
            self._events.clear()
            self._next_ids.clear()

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


def build_workbench_cards(snapshot: dict[str, Any]) -> dict[str, Any]:
    console = dict(snapshot.get("console") or {})
    state = dict(console.get("state") or {})
    action_field = dict(console.get("action_field") or {})
    why_current = dict((console.get("why_current") or {}).get("why") or {})
    why_not = dict((console.get("why_not") or {}).get("why_not") or {})
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

    cards = [
        {
            "panel_id": "current_state",
            "title": "当前主体状态",
            "summary": f"当前模式为 {state.get('brain_state', {}).get('mode') or 'interactive'}。",
            "explanation": "这一栏概括当前身体活力、自我连续性和主观感受容器中的主要信号。",
            "drivers": [f"当下感受：{'、'.join(subjective.get('felt') or []) or '暂无'}"],
            "blockers": [f"全盘拒斥倾向：{_scale_text(subjective.get('reject_all'))}"],
            "next_paths": [f"已生成意义：{'、'.join(subjective.get('meaning_made') or []) or '暂无'}"],
        },
        {
            "panel_id": "action_tendency",
            "title": "当前动作倾向",
            "summary": f"当前后验最高的是“{_action_label(winner.get('action'))}”。",
            "explanation": "这是动作场在当前轮次的优胜峰，表示系统最可能选择的外显或内收动作。",
            "drivers": [f"后验分布峰值：{_scale_text((action_field.get('winner_posterior') or {}).get(winner.get('action')))}"],
            "blockers": [f"主要竞争峰：{_action_label((action_field.get('competing_peaks') or [{}])[0].get('action')) if action_field.get('competing_peaks') else '暂无'}"],
            "next_paths": [f"why：{why_current.get('summary') or '暂无'}"],
        },
        {
            "panel_id": "instinct_space",
            "title": "四维空间",
            "summary": f"本轮优胜区为 {_region_label(instinct_field.get('winner_region'))}。",
            "explanation": "E/F/S/M 四轴显示当前本能场落点，能直观看到坍缩正在往哪一类区域聚集。",
            "drivers": [f"E/F/S/M：{axis_summary}"],
            "blockers": [f"坍缩轨迹节点：{len(instinct_field.get('collapse_trace') or [])}"],
            "next_paths": [f"区域分数条目：{len(instinct_field.get('region_scores') or {})}"],
        },
        {
            "panel_id": "probability_layers",
            "title": "四层概率场",
            "summary": "情境层 / 记忆层 / 动作层 / Token 四层独立呈现。",
            "explanation": "这四层显示从语境、记忆、动作到输出门控的概率演化，不和四维空间混用。",
            "drivers": [f"可用层：{available_layers or '暂无'}"],
            "blockers": [f"why-not 阻滞项：{blocked_by or '暂无'}"],
            "next_paths": [f"反事实预览：{'已生成' if console.get('counterfactual_preview') else '暂无'}"],
        },
    ]
    return {"cards": cards}
