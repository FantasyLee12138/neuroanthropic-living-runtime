from __future__ import annotations

from typing import TYPE_CHECKING, Any

from nalr.schemas.models import to_dict

if TYPE_CHECKING:
    from nalr.runtime.controller import RuntimeController


def _clip_unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _top_probability_peaks(layer_payload: dict[str, Any], *, limit: int = 4) -> list[dict[str, Any]]:
    if not isinstance(layer_payload, dict):
        return []
    posterior = layer_payload.get("winner_posterior")
    if isinstance(posterior, dict) and posterior:
        ranked = sorted(
            (
                (str(name), float(score))
                for name, score in posterior.items()
                if str(name).strip()
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        return [{"name": name, "score": round(score, 6)} for name, score in ranked[:limit]]
    counterfactual = layer_payload.get("counterfactual_top_peaks")
    if isinstance(counterfactual, list) and counterfactual:
        peaks: list[dict[str, Any]] = []
        for row in counterfactual[:limit]:
            if not isinstance(row, dict):
                continue
            name = str(row.get("target") or row.get("action") or row.get("name") or "").strip()
            if not name:
                continue
            peaks.append({"name": name, "score": round(float(row.get("score", 0.0) or 0.0), 6)})
        return peaks
    return []


def _mode_label(mode: str, cause_type: str) -> str:
    normalized_mode = str(mode or "").strip()
    normalized_cause = str(cause_type or "").strip()
    if normalized_cause == "endogenous":
        return {
            "endogenous_light": "内生整理",
            "endogenous_regulation": "内在调节",
            "endogenous_replay": "内部回放",
        }.get(normalized_mode, "内部处理")
    return {
        "interactive": "对外互动",
        "idle": "待机",
        "sleep": "睡眠",
        "safe": "安全模式",
    }.get(normalized_mode, normalized_mode or "未知模式")


def _cause_label(cause_type: str, mode: str) -> str:
    normalized_cause = str(cause_type or "").strip()
    if normalized_cause == "endogenous":
        return _mode_label(mode, normalized_cause)
    return {
        "external_stimulus": "外界刺激",
        "dream": "梦境加工",
    }.get(normalized_cause, normalized_cause or "外部触发")


def _latency_breakdown(runtime_metrics: dict[str, Any]) -> dict[str, Any]:
    total_turn_ms = max(1, int(runtime_metrics.get("total_turn_ms", 0) or 0))
    model_wait_ms = max(0, int(runtime_metrics.get("total_model_wait_ms", 0) or 0))
    local_compute_ms = max(0, total_turn_ms - model_wait_ms)
    if model_wait_ms > local_compute_ms * 1.15:
        dominant = "model_wait"
        summary = "模型等待占主导"
    elif local_compute_ms > model_wait_ms * 1.15:
        dominant = "local_compute"
        summary = "本地计算占主导"
    else:
        dominant = "mixed"
        summary = "模型等待与本地计算接近"
    return {
        "total_turn_ms": total_turn_ms,
        "model_wait_ms": model_wait_ms,
        "local_compute_ms": local_compute_ms,
        "latency_dominant": dominant,
        "latency_summary": summary,
    }


class DiagnosticsRuntimeService:
    def __init__(self, controller: "RuntimeController") -> None:
        self.controller = controller

    def latest_runtime_metrics(self) -> dict[str, Any]:
        controller = self.controller
        round_id = controller.console_round_or_none()
        if round_id is None:
            return {
                "route_type": "",
                "model_call_count": 0,
                "parallel_task_count": 0,
                "parallel_groups": [],
                "optional_timeout_count": 0,
                "speculative_drop_count": 0,
                "total_model_wait_ms": 0,
                "total_turn_ms": 0,
                "local_compute_ms": 0,
                "latency_dominant": "mixed",
                "latency_summary": "暂无运行时性能数据",
            }
        latest_trace = controller.trace_round(round_id)
        runtime_metrics = dict(latest_trace.get("runtime_metrics", {}) or {})
        timing = _latency_breakdown(runtime_metrics)
        return {
            **runtime_metrics,
            "model_call_count": int(runtime_metrics.get("model_call_count", len(list(latest_trace.get("model_call_traces", []) or [])))),
            "parallel_task_count": int(runtime_metrics.get("parallel_task_count", 0) or 0),
            "parallel_groups": list(runtime_metrics.get("parallel_groups", []) or []),
            **timing,
        }

    def performance_payload(self) -> dict[str, Any]:
        runtime_metrics = self.latest_runtime_metrics()
        return {
            "runtime_metrics": runtime_metrics,
            "latency": {
                "total_turn_ms": int(runtime_metrics.get("total_turn_ms", 0) or 0),
                "model_wait_ms": int(runtime_metrics.get("total_model_wait_ms", runtime_metrics.get("model_wait_ms", 0)) or 0),
                "local_compute_ms": int(runtime_metrics.get("local_compute_ms", 0) or 0),
                "latency_dominant": str(runtime_metrics.get("latency_dominant") or "mixed"),
                "latency_summary": str(runtime_metrics.get("latency_summary") or ""),
            },
        }

    def console_state(self) -> dict[str, Any]:
        controller = self.controller
        payload = controller.state_payload()
        cognitive = dict(payload.get("cognitive_snapshot", {}) or {})
        vital_signs = dict(cognitive.get("vital_signs", {}) or {})
        identity = dict(cognitive.get("identity", {}) or {})
        authenticity = dict(cognitive.get("authenticity", {}) or {})
        round_id = controller.console_round_or_none()
        current_round = None
        latest_trace = None
        if round_id is not None:
            latest_trace = controller.trace_round(round_id)
            rendered_expression = dict(latest_trace.get("rendered_expression", {}) or {})
            runtime_metrics = dict(latest_trace.get("runtime_metrics", {}) or {})
            timing = _latency_breakdown(runtime_metrics)
            cause_type = str(latest_trace.get("cause_type") or "")
            mode = str(latest_trace.get("mode") or "")
            current_round = {
                "round_id": latest_trace["round_id"],
                "sampled_action": latest_trace.get("sampled_action"),
                "trace_ref": latest_trace.get("trace_ref"),
                "route_type": runtime_metrics.get("route_type"),
                "cause_type": cause_type,
                "cause_label": _cause_label(cause_type, mode),
                "mode": mode,
                "mode_label": _mode_label(mode, cause_type),
                "render_route": rendered_expression.get("route"),
                "render_model": rendered_expression.get("model"),
                "render_degraded": bool(rendered_expression.get("degraded", False)),
                "failure_policy_applied": rendered_expression.get("failure_policy_applied"),
                "model_call_count": len(list(latest_trace.get("model_call_traces", []) or [])),
                "parallel_task_count": int(runtime_metrics.get("parallel_task_count", 0) or 0),
                **timing,
            }
        try:
            run = controller.run_status()
        except FileNotFoundError:
            if current_round is not None:
                run = {
                    "status": "responded",
                    "round_id": current_round["round_id"],
                    "trace_ref": current_round.get("trace_ref"),
                }
            else:
                run = {"status": "idle", "round_id": None, "trace_ref": None}
        return {
            "brain_state": {
                "mode": vital_signs.get("mode") or payload.get("mode"),
                "vitality": vital_signs.get("body_energy"),
                "self_continuity": identity.get("continuity"),
                "authenticity_pressure": authenticity.get("summary"),
                "long_run_drift_risk": None,
            },
            "neuromodulators": {
                "dopamine": None,
                "noradrenaline": None,
                "serotonin": None,
                "acetylcholine": None,
                "gaba": None,
            },
            "motivation_pool": dict(latest_trace.get("motivation_pool", {}) or {}) if isinstance(latest_trace, dict) else {},
            "long_run": {
                "dream": payload.get("dream", {}),
                "trace_storage": payload.get("trace_storage", {}),
            },
            "current_round": current_round,
            "performance": self.performance_payload(),
            "session": {
                "session_id": payload.get("session_id"),
                "mode": payload.get("mode"),
                "safe_mode": payload.get("safe_mode"),
            },
            "run": run,
            "cognitive_snapshot": cognitive,
        }

    def console_refresh_payload(self, round_ref: int | str | None = None) -> dict[str, Any]:
        controller = self.controller
        effective_round = round_ref if round_ref is not None else controller.console_round_or_none()
        payload: dict[str, Any] = {
            "state": controller.console_state(),
            "autonomy": controller.autonomy_runtime_status(),
        }
        if effective_round is None:
            payload["action_field"] = {
                "round_id": None,
                "trace_ref": None,
                "top_actions": [],
                "winner": {"action": None, "score": 0.0},
                "winner_posterior": {},
                "conflict": {},
                "token_field": {},
                "contribution_stack": [],
                "competing_peaks": [],
            }
            payload["timeline"] = {"round_id": None, "trace_ref": None, "events": []}
            payload["why_current"] = {
                "round_id": None,
                "trace_ref": None,
                "why": {"summary": "暂无", "sampled_action": None, "top_drivers": [], "vitality_snapshot": {}, "authenticity": {}},
            }
            payload["why_not"] = None
            payload["probability_field"] = {}
            payload["counterfactual_preview"] = None
            payload["recent_rounds"] = controller.console_recent_rounds()
            payload["source_links"] = controller.console_source_links(round_id=None, trace_ref=None, why_not_action=None)
            return payload
        payload["action_field"] = controller.console_action_field(effective_round)
        payload["probability_field"] = controller.trace_probability_field(effective_round).get("probability_field", {})
        payload["timeline"] = controller.console_timeline(effective_round)
        payload["why_current"] = controller.console_why_current(effective_round)
        default_why_not_action = controller.console_default_why_not_action(effective_round, action_field=payload["action_field"])
        payload["why_not"] = controller.console_why_not(default_why_not_action, effective_round) if default_why_not_action else None
        payload["counterfactual_preview"] = controller.replay(int(controller.resolve_round_ref(effective_round))).get("counterfactual_preview", {})
        payload["recent_rounds"] = controller.console_recent_rounds()
        payload["source_links"] = controller.console_source_links(
            round_id=int(controller.resolve_round_ref(effective_round)),
            trace_ref=payload["action_field"].get("trace_ref"),
            why_not_action=default_why_not_action,
        )
        return payload

    def console_recent_actions(self, *, limit: int = 8) -> dict[str, Any]:
        controller = self.controller
        controller.trace_store.flush(raise_on_error=False)
        rows = controller.trace_store.recent_rounds(limit=limit)
        actions = [
            {
                "round_id": int(row.get("round_id", 0) or 0),
                "trace_ref": f"round://{int(row.get('round_id', 0) or 0)}",
                "action": str(row.get("sampled_action") or "nothing"),
                "summary": _cause_label(str(row.get("cause_type") or "external_stimulus"), str(row.get("mode") or "interactive")),
                "recorded_at": row.get("recorded_at"),
                "cause_type": str(row.get("cause_type") or "external_stimulus"),
                "mode": str(row.get("mode") or "interactive"),
            }
            for row in rows
        ]
        return {
            "actions": actions,
            "message": "暂无最近动作" if not actions else "",
        }

    def console_probability_space(self, round_ref: int | str | None = None) -> dict[str, Any]:
        controller = self.controller
        latest_round = controller.resolve_round_ref(round_ref) if round_ref is not None else controller.console_round_or_none()
        if latest_round is not None:
            latest_trace = controller.trace_round(latest_round)
            instinct_field = dict(latest_trace.get("state_snapshot", {}).get("instinct_field", {}) or {})
            anchor = dict(latest_trace.get("state_snapshot", {}).get("personality_anchor", {}) or {})
        else:
            state = controller.load_runtime_state()
            controller.sync_tlh_state(state)
            instinct_field = to_dict(state.instinct_field)
            anchor = to_dict(state.personality_anchor)
        probability_field = (
            controller.trace_probability_field(latest_round).get("probability_field", {})
            if latest_round is not None
            else {}
        )
        region_scores = dict(instinct_field.get("region_scores", {}) or {})
        winner_region = str(instinct_field.get("winner_region") or "")
        axis_values = {
            axis: round(float((instinct_field.get("axis_values") or {}).get(axis, 0.0) or 0.0), 4)
            for axis in ("E", "F", "S", "M")
        }
        anchor_axes = {
            axis: round(float((anchor.get("axis_baseline") or {}).get(axis, 0.5) or 0.5), 4)
            for axis in ("E", "F", "S", "M")
        }
        region_vectors = {
            "express": {"E": 0.82, "F": 0.24, "S": 0.48, "M": 0.74},
            "withdraw": {"E": 0.18, "F": 0.78, "S": 0.32, "M": 0.36},
            "hibernate": {"E": 0.12, "F": 0.86, "S": 0.12, "M": 0.22},
            "dissolve": {"E": 0.08, "F": 0.62, "S": 0.56, "M": 0.12},
            "absorb": {"E": 0.36, "F": 0.34, "S": 0.82, "M": 0.58},
        }
        winner_vector = region_vectors.get(
            winner_region,
            {"E": axis_values["E"], "F": axis_values["F"], "S": axis_values["S"], "M": axis_values["M"]},
        )
        winner_strength = max(region_scores.values(), default=0.0)
        predicted_axes = {
            axis: round(
                _clip_unit(
                    axis_values[axis] * 0.7
                    + float(winner_vector.get(axis, axis_values[axis])) * min(0.3, winner_strength * 0.18)
                ),
                4,
            )
            for axis in ("E", "F", "S", "M")
        }
        candidate_center = {
            axis: round((axis_values[axis] + predicted_axes[axis]) / 2.0, 4)
            for axis in ("E", "F", "S", "M")
        }
        plots = []
        for x_axis, y_axis in (("E", "F"), ("E", "S"), ("E", "M"), ("F", "S")):
            plots.append(
                {
                    "label": f"{x_axis}-{y_axis}",
                    "x_axis": x_axis,
                    "y_axis": y_axis,
                    "current_point": {"x": axis_values[x_axis], "y": axis_values[y_axis]},
                    "predicted_point": {"x": predicted_axes[x_axis], "y": predicted_axes[y_axis]},
                    "anchor_point": {"x": anchor_axes[x_axis], "y": anchor_axes[y_axis]},
                    "candidate_point": {"x": candidate_center[x_axis], "y": candidate_center[y_axis]},
                    "winner_region": winner_region,
                    "has_data": any(abs(value) > 1e-9 for value in axis_values.values()) or bool(winner_region),
                }
            )
        layer_labels = {
            "context": "情境层",
            "memory": "记忆层",
            "action": "动作层",
            "token": "Token",
        }
        layers = []
        for key in ("context", "memory", "action", "token"):
            layer_payload = dict(probability_field.get(key, {}) or {})
            peaks = _top_probability_peaks(layer_payload)
            peak = peaks[0] if peaks else None
            layers.append(
                {
                    "key": key,
                    "label": layer_labels[key],
                    "winner": layer_payload.get("winner") or layer_payload.get("winner_target") or "",
                    "peak": peak.get("name") if peak else "",
                    "peak_score": peak.get("score") if peak else 0.0,
                    "peaks": peaks,
                    "keys": len(layer_payload),
                    "entry_count": len(dict(layer_payload.get("winner_posterior", {}) or {})),
                    "empty": not bool(layer_payload),
                }
            )
        space_3d = {
            "axes": {"x": "E", "y": "F", "z": "S", "intensity": "M"},
            "current_point": {
                "x": axis_values["E"],
                "y": axis_values["F"],
                "z": axis_values["S"],
                "intensity": axis_values["M"],
            },
            "predicted_point": {
                "x": predicted_axes["E"],
                "y": predicted_axes["F"],
                "z": predicted_axes["S"],
                "intensity": predicted_axes["M"],
            },
            "anchor_point": {
                "x": anchor_axes["E"],
                "y": anchor_axes["F"],
                "z": anchor_axes["S"],
                "intensity": anchor_axes["M"],
            },
            "candidate_point": {
                "x": candidate_center["E"],
                "y": candidate_center["F"],
                "z": candidate_center["S"],
                "intensity": candidate_center["M"],
            },
            "winner_point": {
                "x": round(float(winner_vector.get("E", axis_values["E"]) or axis_values["E"]), 4),
                "y": round(float(winner_vector.get("F", axis_values["F"]) or axis_values["F"]), 4),
                "z": round(float(winner_vector.get("S", axis_values["S"]) or axis_values["S"]), 4),
                "intensity": round(float(winner_vector.get("M", axis_values["M"]) or axis_values["M"]), 4),
            },
            "winner_region": winner_region,
            "winner_strength": round(float(winner_strength or 0.0), 6),
            "has_data": any(abs(value) > 1e-9 for value in axis_values.values()) or bool(winner_region),
        }
        return {
            "source_round_id": latest_round,
            "plots": plots,
            "layers": layers,
            "space_3d": space_3d,
            "instinct_field": instinct_field,
            "message": "尚无概率分布" if latest_round is None else "",
        }
