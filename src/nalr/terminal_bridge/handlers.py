from __future__ import annotations

from copy import deepcopy
from collections.abc import Iterable
from typing import Any

from nalr.runtime.controller import RuntimeController
from nalr.runtime.metadata import utc_now_iso
from nalr.schemas.models import RoundEvent, to_dict
from nalr.terminal_bridge.protocol import ProtocolError, build_outbound_event, validate_inbound_event
from nalr.terminal_bridge.session import TerminalSessionState, TerminalSessionStore


PERMISSION_MODES = {"plan", "ask", "acceptEdits"}


class TerminalEventHandler:
    def __init__(self, controller: RuntimeController) -> None:
        self.controller = controller
        self.session_store = TerminalSessionStore(controller.runtime_dir)

    def handle(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        return list(self.handle_stream(payload))

    def snapshot_session(self, session_id: str) -> dict[str, Any]:
        session = self._load_session(session_id)
        run_id = session.active_run_id or session.last_run_id
        return self._sidebar_snapshot(session, run_id=run_id)

    def handle_stream(self, payload: dict[str, Any]) -> Iterable[dict[str, Any]]:
        event = validate_inbound_event(payload)
        event_type = event["type"]
        if event_type == "start_session":
            return self._start_session(
                event["session_id"],
                event["cwd"],
                persist_current=bool(event.get("persist_current", True)),
            )
        if event_type == "user_turn":
            return iter(self._user_turn_stream(event["session_id"], event["text"]))
        if event_type == "control_command":
            return self._control_command(event["session_id"], event["command"], event.get("value"))
        if event_type == "approve":
            return self._approve(event["session_id"], event["call_id"], event["approved"])
        if event_type == "close_session":
            return self._close_session(
                event["session_id"],
                detach=bool(event.get("detach", False)),
                purge=bool(event.get("purge", False)),
                cleanup_old=bool(event.get("cleanup_old", False)),
                transcript_mode=event.get("transcript_mode"),
            )
        raise ProtocolError(f"unhandled event type: {event_type}")

    def _load_session(self, session_id: str) -> TerminalSessionState:
        return self.session_store.read(session_id)

    def _start_session(self, session_id: str, cwd: str, *, persist_current: bool = True) -> Iterable[dict[str, Any]]:
        try:
            state = self.session_store.read(session_id)
            if state.status in {"active", "detached"}:
                state.cwd = cwd
                state.status = "active"
            else:
                state = TerminalSessionState(session_id=session_id, cwd=cwd, status="active")
        except FileNotFoundError:
            state = TerminalSessionState(session_id=session_id, cwd=cwd, status="active")
        self.session_store.write(state, mark_current=persist_current)
        yield build_outbound_event("session_started", session=to_dict(state))
        yield self._build_sidebar_snapshot_event(state)

    def _append_transcript(self, session: TerminalSessionState, kind: str, text: str, **metadata: Any) -> None:
        if not text:
            return
        entry = {"kind": kind, "text": text.strip()}
        entry.update({key: value for key, value in metadata.items() if value is not None})
        session.transcript_lines.append(entry)

    def _append_tool_timeline(
        self,
        session: TerminalSessionState,
        kind: str,
        *,
        call_id: str,
        tool: str,
        summary: str,
        status: str | None,
        round_id: int | None = None,
        trace_ref: str | None = None,
        event_log_ref: dict[str, Any] | None = None,
    ) -> None:
        entry = {
            "kind": kind,
            "callId": call_id,
            "tool": tool,
            "summary": summary,
            "status": status,
            "roundId": round_id,
            "traceRef": trace_ref,
        }
        if event_log_ref:
            entry["event_log_ref"] = dict(event_log_ref)
        session.tool_timeline.append(entry)

    def _permission_policy(self, permission_mode: str) -> dict[str, Any]:
        if permission_mode == "acceptEdits":
            return {"allow_commit": True, "operator_level": "soft_intervene"}
        if permission_mode == "ask":
            return {"allow_commit": False, "operator_level": "debug_control"}
        return {"allow_commit": False, "operator_level": "read_only"}

    def _linkage_fields(self, *payloads: dict[str, Any] | None) -> dict[str, Any]:
        def _round_id(value: Any) -> int | None:
            if isinstance(value, bool):
                return None
            if isinstance(value, int):
                return value
            if isinstance(value, str) and value.strip():
                try:
                    return int(value)
                except ValueError:
                    return None
            return None

        def _trace_ref(value: Any) -> str | None:
            if isinstance(value, str) and value.strip():
                return value.strip()
            return None

        def _normalize_event_log_ref(payload: Any) -> dict[str, Any]:
            if not isinstance(payload, dict):
                return {}
            normalized: dict[str, Any] = {}
            for key, value in payload.items():
                if not isinstance(key, str):
                    continue
                if key.endswith("_ref") or key.endswith("_jsonl"):
                    text = str(value or "").strip()
                    if text:
                        normalized[key] = text
            return normalized

        def _derive_event_log_ref_from_trace(trace_ref: str) -> dict[str, Any]:
            if trace_ref.startswith("run://"):
                if "/tools/" in trace_ref:
                    run_ref, _, _ = trace_ref.partition("/tools/")
                    return {
                        "run_trace_ref": run_ref,
                        "run_tool_trace_ref": trace_ref,
                    }
                if "/steps/" in trace_ref:
                    run_ref, _, _ = trace_ref.partition("/steps/")
                    return {
                        "run_trace_ref": run_ref,
                        "run_step_trace_ref": trace_ref,
                    }
                return {"run_trace_ref": trace_ref}
            if trace_ref.startswith("round://"):
                return {"round_trace_ref": trace_ref}
            if trace_ref.startswith("dream://"):
                return {"dream_trace_ref": trace_ref}
            if trace_ref.startswith("command://"):
                return {"command_trace_ref": trace_ref}
            return {}

        def _merge_linkage(result: dict[str, Any], candidate: dict[str, Any]) -> None:
            round_id = candidate.get("round_id")
            if result.get("round_id") is None and isinstance(round_id, int):
                result["round_id"] = round_id
            trace_ref = _trace_ref(candidate.get("trace_ref"))
            if trace_ref:
                result["trace_ref"] = trace_ref
                result["event_log_ref"].update(_derive_event_log_ref_from_trace(trace_ref))
            event_log_ref = _normalize_event_log_ref(candidate.get("event_log_ref"))
            if event_log_ref:
                result["event_log_ref"].update(event_log_ref)

        def _search(payload: Any, result: dict[str, Any], depth: int = 0) -> None:
            if depth > 3:
                return
            if isinstance(payload, dict):
                candidate: dict[str, Any] = {}
                round_id = _round_id(payload.get("round_id") or payload.get("roundId"))
                if round_id is not None:
                    candidate["round_id"] = round_id
                trace_ref = _trace_ref(payload.get("trace_ref") or payload.get("traceRef"))
                if trace_ref is not None:
                    candidate["trace_ref"] = trace_ref
                event_log_ref = _normalize_event_log_ref(payload.get("event_log_ref") or payload.get("eventLogRef"))
                if event_log_ref:
                    candidate["event_log_ref"] = event_log_ref
                run_id = str(payload.get("run_id") or payload.get("runId") or "").strip()
                if run_id:
                    candidate.setdefault("event_log_ref", {})["run_trace_ref"] = self.controller._run_trace_ref(run_id)
                    step_id = str(payload.get("step_id") or payload.get("stepId") or "").strip()
                    if step_id:
                        candidate["event_log_ref"]["run_step_trace_ref"] = self.controller._run_step_trace_ref(run_id, step_id)
                    call_id = str(payload.get("call_id") or payload.get("callId") or "").strip()
                    if call_id:
                        candidate["event_log_ref"]["run_tool_trace_ref"] = self.controller._run_tool_trace_ref(run_id, call_id)
                command_id = str(payload.get("command_id") or payload.get("commandId") or "").strip()
                if command_id:
                    candidate.setdefault("event_log_ref", {})["command_trace_ref"] = f"command://{command_id}"
                dream_run_id = str(payload.get("dream_run_id") or payload.get("dreamRunId") or "").strip()
                if dream_run_id:
                    candidate.setdefault("event_log_ref", {})["dream_trace_ref"] = f"dream://runs/{dream_run_id}"
                _merge_linkage(result, candidate)
                for value in payload.values():
                    if isinstance(value, dict):
                        _search(value, result, depth + 1)
                    elif isinstance(value, list):
                        for item in value:
                            if isinstance(item, dict):
                                _search(item, result, depth + 1)

        result: dict[str, Any] = {
            "round_id": None,
            "trace_ref": None,
            "event_log_ref": {},
        }
        for payload in payloads:
            _search(payload, result)
        event_log_ref = dict(result.get("event_log_ref") or {})
        if not event_log_ref:
            event_log_ref = {}
        if "run_tool_trace_ref" in event_log_ref and "run_trace_ref" not in event_log_ref:
            event_log_ref["run_trace_ref"] = str(event_log_ref["run_tool_trace_ref"]).partition("/tools/")[0]
        if "run_step_trace_ref" in event_log_ref and "run_trace_ref" not in event_log_ref:
            event_log_ref["run_trace_ref"] = str(event_log_ref["run_step_trace_ref"]).partition("/steps/")[0]
        round_trace_ref = str(event_log_ref.get("round_trace_ref") or "").strip()
        if not round_trace_ref and isinstance(result.get("round_id"), int):
            round_trace_ref = f"round://{int(result['round_id'])}"
            event_log_ref["round_trace_ref"] = round_trace_ref
        if result.get("round_id") is None and round_trace_ref.startswith("round://"):
            result["round_id"] = _round_id(round_trace_ref.removeprefix("round://"))
        trace_ref = _trace_ref(result.get("trace_ref"))
        if not trace_ref:
            for key in (
                "run_tool_trace_ref",
                "run_step_trace_ref",
                "run_trace_ref",
                "round_trace_ref",
                "dream_trace_ref",
                "command_trace_ref",
            ):
                trace_ref = _trace_ref(event_log_ref.get(key))
                if trace_ref:
                    break
        linkage: dict[str, Any] = {}
        if isinstance(result.get("round_id"), int):
            linkage["round_id"] = int(result["round_id"])
        if trace_ref is not None:
            linkage["trace_ref"] = trace_ref
        if event_log_ref:
            linkage["event_log_ref"] = event_log_ref
        return linkage

    def _apply_event_to_session(self, session: TerminalSessionState, event: dict[str, Any]) -> None:
        event_type = str(event.get("type") or "")
        if event_type == "tool_call":
            linkage = self._linkage_fields(event)
            self._append_tool_timeline(
                session,
                "call",
                call_id=str(event.get("call_id") or ""),
                tool=str(event.get("tool") or "unknown"),
                summary=str(event.get("summary") or event.get("status") or ""),
                status=event.get("status") if event.get("status") is None else str(event.get("status")),
                round_id=linkage.get("round_id"),
                trace_ref=linkage.get("trace_ref"),
                event_log_ref=linkage.get("event_log_ref"),
            )
            self._append_transcript(session, "system", f"Tool: {event.get('tool') or 'unknown'}")
            return
        if event_type == "approval_request":
            linkage = self._linkage_fields(event)
            pending = [item for item in session.approvals_pending if item.get("call_id") != event.get("call_id")]
            pending.append(
                {
                    "call_id": event.get("call_id"),
                    "tool": event.get("tool"),
                    "args": event.get("args"),
                    "risk_level": event.get("risk_level"),
                    "summary": event.get("summary"),
                    "action_preview": event.get("action_preview"),
                    "mode": event.get("mode"),
                    "status": event.get("status"),
                    "run_id": event.get("run_id"),
                    "choices": event.get("choices", []),
                    "round_id": linkage.get("round_id"),
                    "trace_ref": linkage.get("trace_ref"),
                    "event_log_ref": linkage.get("event_log_ref"),
                }
            )
            session.approvals_pending = pending
            self._append_tool_timeline(
                session,
                "approval",
                call_id=str(event.get("call_id") or ""),
                tool=str(event.get("tool") or "unknown"),
                summary=str(event.get("summary") or ""),
                status=str(event.get("status") or "pending"),
                round_id=linkage.get("round_id"),
                trace_ref=linkage.get("trace_ref"),
                event_log_ref=linkage.get("event_log_ref"),
            )
            self._append_transcript(session, "system", f"Approval: {event.get('tool') or 'unknown'} - {event.get('summary') or ''}")
            return
        if event_type == "tool_result":
            linkage = self._linkage_fields(event)
            session.approvals_pending = [item for item in session.approvals_pending if item.get("call_id") != event.get("call_id")]
            self._append_tool_timeline(
                session,
                "result",
                call_id=str(event.get("call_id") or ""),
                tool=str((event.get("result") or {}).get("tool_name") or "unknown"),
                summary=str((event.get("result") or {}).get("summary") or ""),
                status=(event.get("result") or {}).get("status") if (event.get("result") or {}).get("status") is None else str((event.get("result") or {}).get("status")),
                round_id=linkage.get("round_id"),
                trace_ref=linkage.get("trace_ref"),
                event_log_ref=linkage.get("event_log_ref"),
            )
            self._append_transcript(session, "system", f"Result: {(event.get('result') or {}).get('tool_name') or 'unknown'}")
            return
        if event_type == "assistant_final":
            linkage = self._linkage_fields(event)
            self._append_transcript(
                session,
                "assistant",
                str(event.get("message") or ""),
                round_id=linkage.get("round_id"),
                trace_ref=linkage.get("trace_ref"),
                event_log_ref=linkage.get("event_log_ref"),
            )

    def _gated_tool_result_event(self, event: dict[str, Any]) -> dict[str, Any]:
        result = dict(event.get("result") or {})
        result.setdefault("status", "rejected")
        result.setdefault("summary", "tool execution rejected")
        result["rejected"] = True
        return build_outbound_event(
            "tool_result",
            session_id=event.get("session_id"),
            call_id=event.get("call_id"),
            result=result,
            **self._linkage_fields(event, result),
        )

    def _persistable_deferred_events(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        sanitized = deepcopy(events)
        for event in sanitized:
            if not isinstance(event, dict) or event.get("type") != "sidebar_snapshot":
                continue
            session_payload = event.get("session")
            if isinstance(session_payload, dict):
                approvals = session_payload.get("approvals_pending")
                if isinstance(approvals, list):
                    session_payload["approvals_pending"] = [
                        {k: v for k, v in item.items() if k != "deferred_events"} if isinstance(item, dict) else item
                        for item in approvals
                    ]
            approvals_payload = event.get("approvals")
            if isinstance(approvals_payload, dict):
                pending = approvals_payload.get("pending")
                if isinstance(pending, list):
                    approvals_payload["pending"] = [
                        {k: v for k, v in item.items() if k != "deferred_events"} if isinstance(item, dict) else item
                        for item in pending
                    ]
        return sanitized

    def _endogenous_summary(self, payload: dict[str, Any]) -> str:
        trigger = payload.get("trigger") if isinstance(payload.get("trigger"), dict) else {}
        micro_intent = payload.get("micro_intent") if isinstance(payload.get("micro_intent"), dict) else {}
        return "\n".join(
            [
                f"内源触发：round {payload.get('round_id', '?')}",
                f"触发器：{trigger.get('trigger_type') or payload.get('trigger_type') or 'unknown'}",
                f"模式：{payload.get('selected_mode') or trigger.get('selected_mode') or 'unknown'}",
                f"当前意图：{micro_intent.get('name') or 'latent'}",
                f"边界动作：{payload.get('boundary_action') or 'unknown'}",
            ]
        )

    def _replay_summary(self, payload: dict[str, Any]) -> str:
        return "\n".join(
            [
                f"重放：round {payload.get('round_id', '?')}",
                f"原动作：{payload.get('original_action') or 'unknown'}",
                f"重放动作：{payload.get('replayed_action') or 'unknown'}",
                f"seed：{payload.get('seed', 0)}",
            ]
        )

    def _replay_motivation_summary(self, payload: dict[str, Any]) -> str:
        motivation_pool = payload.get("motivation_pool") if isinstance(payload.get("motivation_pool"), dict) else {}
        feedback = payload.get("motivation_feedback") if isinstance(payload.get("motivation_feedback"), dict) else {}
        return "\n".join(
            [
                f"动机重放：round {payload.get('round_id', '?')}",
                f"sampled={payload.get('sampled_action') or 'unknown'}",
                f"激活数：{len(motivation_pool.get('active_motivations', []) or [])}",
                f"反馈：{feedback.get('reward_signal', feedback.get('summary', '暂无'))}",
            ]
        )

    def _why_motivation_summary(self, payload: dict[str, Any]) -> str:
        motivation_pool = payload.get("motivation_pool") if isinstance(payload.get("motivation_pool"), dict) else {}
        feedback = payload.get("motivation_feedback") if isinstance(payload.get("motivation_feedback"), dict) else {}
        return "\n".join(
            [
                f"动机原因：round {payload.get('round_id', '?')}",
                f"sampled={payload.get('sampled_action') or 'unknown'}",
                f"cause={payload.get('cause_type') or 'unknown'}",
                f"active={len(motivation_pool.get('active_motivations', []) or [])}",
                f"feedback={feedback.get('reward_signal', feedback.get('summary', '暂无'))}",
            ]
        )

    def _why_not_summary(self, payload: dict[str, Any]) -> str:
        blocked = payload.get("blocked_by", []) or []
        candidates = ", ".join(str(item) for item in blocked[:4]) if blocked else "none"
        return "\n".join(
            [
                f"为什么不是：round {payload.get('round_id', '?')}",
                f"action={payload.get('action') or 'unknown'}",
                f"selected={payload.get('selected_action') or 'unknown'}",
                f"blocked_by={candidates}",
            ]
        )

    def _what_changed_summary(self, payload: dict[str, Any]) -> str:
        action_counts = dict(payload.get("action_counts") or {})
        mode_counts = dict(payload.get("mode_counts") or {})
        action_text = ", ".join(f"{name}:{count}" for name, count in list(action_counts.items())[:4]) or "none"
        mode_text = ", ".join(f"{name}:{count}" for name, count in list(mode_counts.items())[:4]) or "none"
        return "\n".join(
            [
                f"变化窗口：{payload.get('window', 0)}",
                f"动作：{action_text}",
                f"模式：{mode_text}",
                f"budget_delta={payload.get('budget_delta', 0.0)}",
            ]
        )

    def _eval_summary(self, payload: dict[str, Any]) -> str:
        return "\n".join(
            [
                f"长跑评估：generated_rounds={payload.get('generated_rounds', 0)}",
                f"task_success_rate={payload.get('task_success_rate', 0.0)}",
                f"safe_mode_rate={payload.get('safe_mode_rate', 0.0)}",
            ]
        )

    def _approval_choices(self) -> list[dict[str, str]]:
        return [
            {"id": "approve", "label": "批准", "kind": "approval", "value": "approve"},
            {"id": "reject", "label": "拒绝", "kind": "approval", "value": "reject"},
            {"id": "details", "label": "详情", "kind": "drawer", "value": "approvals"},
            {"id": "next", "label": "下一个", "kind": "approval_nav", "value": "next"},
        ]

    def _ui_actions(self, session: TerminalSessionState, run: dict[str, Any] | None, *, pending_count: int) -> dict[str, list[dict[str, Any]]]:
        run_status = str((run or {}).get("status") or session.status or "idle")
        primary = [
            {"id": "status", "label": "状态", "kind": "drawer", "value": "status", "disabled": False},
            {"id": "why", "label": "原因", "kind": "drawer", "value": "why", "disabled": run is None},
            {"id": "steps", "label": "步骤", "kind": "drawer", "value": "steps", "disabled": run is None},
            {"id": "tools", "label": "工具", "kind": "drawer", "value": "tools", "disabled": run is None},
            {"id": "approvals", "label": "审批", "kind": "drawer", "value": "approvals", "disabled": pending_count == 0},
        ]
        secondary = [
            {"id": "cognition", "label": "认知", "kind": "drawer", "value": "cognition", "disabled": False},
            {"id": "meta", "label": "元信息", "kind": "drawer", "value": "meta", "disabled": False},
            {"id": "pause", "label": "暂停", "kind": "command", "value": "pause", "disabled": run is None or run_status == "paused"},
            {"id": "resume", "label": "恢复", "kind": "command", "value": "resume", "disabled": run is None or run_status != "paused"},
            {"id": "abort", "label": "中止", "kind": "command", "value": "abort", "disabled": run is None or run_status in {"aborted", "completed", "done"}},
        ]
        return {"primary": primary, "secondary": secondary}

    def _ensure_chat_sidebar_round(
        self,
        *,
        text: str,
        target: str,
        mode: str,
        previous_round_count: int,
        allow_tick_backfill: bool = True,
    ) -> int | None:
        current_round_count = int(self.controller.load_runtime_state().round_count or 0)
        if current_round_count > previous_round_count:
            return current_round_count
        if not allow_tick_backfill:
            return None
        result = self.controller.tick(
            RoundEvent(source="user", content=text, target=target),
            scenario="chat",
            mode=mode,
        )
        return int(result.round_id)

    def _light_console_payload(self, *, runtime_state: Any, round_id: int | None = None) -> dict[str, Any]:
        current_round_id = int(round_id or getattr(runtime_state, "round_count", 0) or 0) or None
        return {
            "state": {
                "brain_state": {
                    "mode": str(getattr(runtime_state, "mode", "") or ""),
                    "focus": str(getattr(runtime_state, "focus", "") or ""),
                    "mood": round(float(getattr(runtime_state, "mood", 0.0) or 0.0), 4),
                    "body_energy": round(float(getattr(runtime_state, "body_energy", 0.0) or 0.0), 4),
                    "affect_residue": round(float(getattr(runtime_state, "affect_residue", 0.0) or 0.0), 4),
                },
                "current_round": {
                    "round_id": current_round_id,
                    "trace_ref": f"round://{current_round_id}" if current_round_id else "",
                },
            },
            "action_field": {},
            "timeline": {
                "round_id": current_round_id,
                "trace_ref": f"round://{current_round_id}" if current_round_id else "",
                "events": [],
            },
            "why_current": {
                "round_id": current_round_id,
                "why": {
                    "summary": "按需加载",
                },
            },
            "why_not": {},
        }

    def _user_turn_stream(self, session_id: str, text: str) -> Iterable[dict[str, Any]]:
        session = self._load_session(session_id)
        normalized_text = text.strip()
        initiative_feedback = self.controller.record_initiative_feedback(normalized_text, session_id=session_id)
        self._append_transcript(
            session,
            "user",
            normalized_text,
            initiative_response_to=initiative_feedback.get("initiative_response_to") if initiative_feedback.get("recorded") else None,
        )
        events: list[dict[str, Any]] = []
        deferred_events: list[dict[str, Any]] = []
        gated_call_id: str | None = None
        policy = self._permission_policy(session.permission_mode)
        previous_round_count = int(self.controller.load_runtime_state().round_count or 0)
        plan = self.controller.plan_turn(
            normalized_text,
            target="user",
            mode="interactive",
            operator_level=policy["operator_level"],
            defer_bootstrap_tool=session.permission_mode == "ask",
        )
        active_run_id = session.active_run_id or session.last_run_id
        if getattr(plan, "route", "") == "direct_chat" and active_run_id:
            try:
                active_run = self.controller.run_status(active_run_id)
            except FileNotFoundError:
                active_run = None
            if isinstance(active_run, dict) and str(active_run.get("status") or "") in {"running", "paused"}:
                plan.route = "task_run"
                plan.scenario = "task"
                plan.task_bootstrap = self.controller.prepare_task_bootstrap(
                    normalized_text,
                    allow_commit=policy["allow_commit"],
                    operator_level=policy["operator_level"],
                    defer_bootstrap_tool=session.permission_mode == "ask",
                )
        if getattr(plan, "route", "") == "fast_chat":
            deltas, execution = self.controller.stream_fast_chat_turn(plan)
            message = execution["assistant_final"] if isinstance(execution, dict) else execution.assistant_final
            round_id = self._ensure_chat_sidebar_round(
                text=normalized_text,
                target=getattr(plan, "target", "user"),
                mode=getattr(plan, "mode", "interactive"),
                previous_round_count=previous_round_count,
                allow_tick_backfill=False,
            )
            linkage = self._linkage_fields({"round_id": round_id})
            self._append_transcript(
                session,
                "assistant",
                message,
                round_id=linkage.get("round_id"),
                trace_ref=linkage.get("trace_ref"),
                event_log_ref=linkage.get("event_log_ref"),
            )
            self.session_store.write(session)
            for delta in deltas:
                events.append(
                    build_outbound_event(
                        "assistant_token",
                        session_id=session_id,
                        delta=delta,
                        message=message,
                        **linkage,
                    )
                )
            events.append(
                self._build_sidebar_snapshot_event(
                    session,
                    run_id=session.active_run_id or session.last_run_id,
                    include_console=False,
                )
            )
            events.append(build_outbound_event("assistant_final", session_id=session_id, message=message, **linkage))
            return events
        execution = self.controller.execute_turn(
            plan,
            allow_commit=policy["allow_commit"],
            operator_level=policy["operator_level"],
            replace_active=True,
            interrupt_reason="interrupted_by_user",
        )

        route = execution["route"] if isinstance(execution, dict) else execution.route
        if route in {"direct_chat", "fast_chat"}:
            message = execution["assistant_final"] if isinstance(execution, dict) else execution.assistant_final
            payload = execution.get("payload", {}) if isinstance(execution, dict) else getattr(execution, "payload", {})
            deltas = list(payload.get("stream_deltas", [])) if isinstance(payload, dict) else []
            if not deltas:
                deltas = [message]
            round_id = self._ensure_chat_sidebar_round(
                text=normalized_text,
                target=getattr(plan, "target", "user"),
                mode=getattr(plan, "mode", "interactive"),
                previous_round_count=previous_round_count,
            )
            linkage = self._linkage_fields({"round_id": round_id})
            self._append_transcript(
                session,
                "assistant",
                message,
                round_id=linkage.get("round_id"),
                trace_ref=linkage.get("trace_ref"),
                event_log_ref=linkage.get("event_log_ref"),
            )
            self.session_store.write(session)
            for delta in deltas:
                events.append(
                    build_outbound_event(
                        "assistant_token",
                        session_id=session_id,
                        delta=delta,
                        message=message,
                        **linkage,
                    )
                )
            events.append(self._build_sidebar_snapshot_event(session, run_id=session.active_run_id or session.last_run_id))
            events.append(build_outbound_event("assistant_final", session_id=session_id, message=message, **linkage))
            return events

        run = execution["run"] if isinstance(execution, dict) else execution.run
        explain = execution["explain"] if isinstance(execution, dict) else execution.explain
        steps = execution["steps"] if isinstance(execution, dict) else execution.steps
        tools = execution["tools"] if isinstance(execution, dict) else execution.tools
        task_message = execution["assistant_preamble"] if isinstance(execution, dict) else execution.assistant_preamble
        session.active_run_id = run["run_id"]
        session.last_run_id = run["run_id"]
        session.status = "active"
        self._append_transcript(session, "assistant", task_message)
        self.session_store.write(session)
        events.append(
            build_outbound_event(
                "run_status",
                session_id=session_id,
                run=run,
                **self._linkage_fields(run, explain),
            )
        )
        events.append(
            build_outbound_event(
                "assistant_token",
                session_id=session_id,
                delta=task_message,
                message=task_message,
                **self._linkage_fields(run, explain),
            )
        )
        pending_approvals = [item for item in session.approvals_pending if item.get("status") == "pending"]
        session.approvals_pending = pending_approvals
        for step in steps:
            step_event = build_outbound_event(
                "step_update",
                session_id=session_id,
                step=step,
                **self._linkage_fields(run, explain, step),
            )
            if gated_call_id is None:
                self._append_transcript(session, "system", f"Step: {step.get('title') or step.get('step_id') or 'unknown'}")
                self.session_store.write(session)
                events.append(step_event)
            else:
                deferred_events.append(step_event)
        for index, tool in enumerate(tools):
            call_id = f"{run['run_id']}:tool:{index}"
            tool_name = str(tool.get("tool_name") or "unknown")
            summary = tool.get("summary") or tool.get("status") or ""
            linkage = self._linkage_fields(run, tool)
            tool_call_event = build_outbound_event(
                "tool_call",
                session_id=session_id,
                call_id=call_id,
                run_id=run["run_id"],
                tool=tool_name,
                args=tool.get("input") or {},
                summary=summary,
                status=tool.get("status"),
                **self._linkage_fields(run, tool),
            )
            if gated_call_id is None:
                self._append_tool_timeline(
                    session,
                    "call",
                    call_id=call_id,
                    tool=tool_name,
                    summary=summary,
                    status=tool.get("status"),
                    round_id=linkage.get("round_id"),
                    trace_ref=linkage.get("trace_ref"),
                    event_log_ref=linkage.get("event_log_ref"),
                )
                self._append_transcript(session, "system", f"Tool: {tool_name}{f' - {summary}' if summary else ''}")
                self.session_store.write(session)
                events.append(tool_call_event)
            else:
                deferred_events.append(tool_call_event)

            if session.permission_mode == "ask":
                approval_payload = {
                    "call_id": call_id,
                    "run_id": run["run_id"],
                    "tool": tool_name,
                    "args": tool.get("input") or {},
                    "risk_level": "medium",
                    "summary": tool.get("summary") or f"{tool_name} requires operator approval",
                    "action_preview": tool.get("output_excerpt") or tool.get("summary") or tool_name,
                    "mode": session.permission_mode,
                    "status": "pending",
                    "approved": None,
                    "actions": ["approve", "reject"],
                    "choices": self._approval_choices(),
                    "requested_at": utc_now_iso(),
                    "deferred_events": [],
                    "round_id": linkage.get("round_id"),
                    "trace_ref": linkage.get("trace_ref"),
                    "event_log_ref": linkage.get("event_log_ref"),
                }
                pending_approvals.append(approval_payload)
                session.approvals_pending = pending_approvals
                approval_event = build_outbound_event(
                    "approval_request",
                    session_id=session_id,
                    call_id=call_id,
                    run_id=run["run_id"],
                    tool=tool_name,
                    args=tool.get("input") or {},
                    risk_level=approval_payload["risk_level"],
                    summary=approval_payload["summary"],
                    action_preview=approval_payload["action_preview"],
                    mode=approval_payload["mode"],
                    status=approval_payload["status"],
                    actions=approval_payload["actions"],
                    choices=self._approval_choices(),
                    approved=None,
                    **self._linkage_fields(run, tool, approval_payload),
                )
                if gated_call_id is None:
                    gated_call_id = call_id
                    self._append_tool_timeline(
                        session,
                        "approval",
                        call_id=call_id,
                        tool=tool_name,
                        summary=str(approval_payload["summary"]),
                        status=str(approval_payload["status"]),
                        round_id=linkage.get("round_id"),
                        trace_ref=linkage.get("trace_ref"),
                        event_log_ref=linkage.get("event_log_ref"),
                    )
                    self._append_transcript(session, "system", f"Approval: {tool_name} - {approval_payload['summary']}")
                    self.session_store.write(session)
                    events.append(approval_event)
                else:
                    deferred_events.append(approval_event)

            result_event = None
            if str(tool.get("status") or "") != "awaiting_approval":
                result_event = build_outbound_event(
                    "tool_result",
                    session_id=session_id,
                    call_id=call_id,
                    result=tool,
                    **linkage,
                )
            if session.permission_mode == "ask":
                approval_item = next((item for item in session.approvals_pending if item.get("call_id") == call_id), None)
                if approval_item is not None and result_event is not None:
                    approval_item["deferred_events"] = self._persistable_deferred_events([result_event])
                if result_event is not None and gated_call_id is None:
                    self._append_tool_timeline(
                        session,
                        "result",
                        call_id=call_id,
                        tool=tool_name,
                        summary=tool.get("output_excerpt") or summary,
                        status=tool.get("status"),
                        round_id=linkage.get("round_id"),
                        trace_ref=linkage.get("trace_ref"),
                        event_log_ref=linkage.get("event_log_ref"),
                    )
                    self._append_transcript(session, "system", f"Result: {tool_name}")
                    self.session_store.write(session)
                    events.append(result_event)
                elif result_event is not None:
                    deferred_events.append(result_event)
            else:
                self._append_tool_timeline(
                    session,
                    "result",
                    call_id=call_id,
                    tool=tool_name,
                    summary=tool.get("output_excerpt") or summary,
                    status=tool.get("status"),
                    round_id=linkage.get("round_id"),
                    trace_ref=linkage.get("trace_ref"),
                    event_log_ref=linkage.get("event_log_ref"),
                )
                self._append_transcript(session, "system", f"Result: {tool_name}")
                self.session_store.write(session)
                events.append(result_event)

        snapshot_event = self._build_sidebar_snapshot_event(session, run_id=run["run_id"], run=run, explain=explain, steps=steps, tools=tools)
        final_event = build_outbound_event(
            "assistant_final",
            session_id=session_id,
            run_id=run["run_id"],
            message=(execution["assistant_final"] if isinstance(execution, dict) else execution.assistant_final),
            payload=explain,
            **self._linkage_fields(run, explain, snapshot_event),
        )
        if gated_call_id is not None:
            gate_item = next((item for item in session.approvals_pending if item.get("call_id") == gated_call_id), None)
            if gate_item is not None:
                gate_item["deferred_events"] = self._persistable_deferred_events(deferred_events)
                gate_item["deferred_final"] = {
                    "run_id": run["run_id"],
                    "message": execution["assistant_final"] if isinstance(execution, dict) else execution.assistant_final,
                    "payload": explain,
                    **self._linkage_fields(run, explain, snapshot_event),
                }
            self.session_store.write(session)
            events.append(snapshot_event)
            return events

        self.session_store.write(session)
        events.append(snapshot_event)
        events.append(final_event)
        return events

    def _control_command(self, session_id: str, command: str, value: str | None = None) -> list[dict[str, Any]]:
        session = self._load_session(session_id)
        command_name = command.strip().lower().lstrip("/")
        value_text = value.strip() if isinstance(value, str) else None
        run_id = session.active_run_id or session.last_run_id
        if command_name == "mode":
            if not value_text:
                return [build_outbound_event("error", session_id=session_id, message="Usage: /mode <name>")]
            session.mode = value_text
            self.session_store.write(session)
            return [
                self._build_sidebar_snapshot_event(session, run_id=run_id),
                build_outbound_event("assistant_final", session_id=session_id, message=f"mode set to {session.mode}"),
            ]
        if command_name == "permissions":
            if not value_text:
                return [
                    self._build_sidebar_snapshot_event(session, run_id=run_id),
                    build_outbound_event(
                        "assistant_final",
                        session_id=session_id,
                        message=f"permission mode is {session.permission_mode}",
                    ),
                ]
            if value_text not in PERMISSION_MODES:
                return [
                    build_outbound_event(
                        "error",
                        session_id=session_id,
                        message="permissions must be one of: plan, ask, acceptEdits",
                    )
                ]
            session.permission_mode = value_text
            self.session_store.write(session)
            return [
                self._build_sidebar_snapshot_event(session, run_id=run_id),
                build_outbound_event(
                    "assistant_final",
                    session_id=session_id,
                    message=f"permission mode set to {session.permission_mode}",
                ),
            ]
        if command_name == "compact":
            if value_text is None:
                session.compact = not session.compact
            elif value_text in {"on", "true"}:
                session.compact = True
            elif value_text in {"off", "false"}:
                session.compact = False
            else:
                return [build_outbound_event("error", session_id=session_id, message="Usage: /compact [on|off]")]
            session.transcript_mode = "compact" if session.compact else "full"
            self.session_store.write(session)
            return [
                self._build_sidebar_snapshot_event(session, run_id=run_id),
                build_outbound_event(
                    "assistant_final",
                    session_id=session_id,
                    message=f"compact {'on' if session.compact else 'off'}",
                ),
            ]
        if command_name == "state":
            snapshot = self._sidebar_snapshot(session, run_id=run_id)
            return [
                build_outbound_event("sidebar_snapshot", session_id=session_id, **snapshot),
                build_outbound_event(
                    "assistant_final",
                    session_id=session_id,
                    message=self._cognitive_summary(snapshot["cognitive_snapshot"]),
                    payload=snapshot,
                ),
            ]
        if command_name == "model":
            model_status = self.controller.model_status()
            return [
                self._build_sidebar_snapshot_event(session, run_id=run_id),
                build_outbound_event("assistant_final", session_id=session_id, message=self._model_summary(model_status)),
            ]
        if command_name == "dream":
            cue = value_text or None
            payload = self.controller.run_dream(mode="sleep", cue=cue)
            cue_text = f"，cue={cue}" if cue else ""
            message = f"已手动触发 Dream：{payload['trigger']}{cue_text}，run={payload['dream_run_id']}。"
            return [
                self._build_sidebar_snapshot_event(session, run_id=run_id),
                build_outbound_event(
                    "assistant_final",
                    session_id=session_id,
                    message=message,
                    payload=payload,
                ),
            ]
        if command_name == "probability":
            try:
                kind, ref_value, subject = self._parse_probability_command(value_text)
                if kind == "overview":
                    payload = self.controller.trace_probability_field(ref_value)
                    message = self._probability_summary(payload)
                elif kind == "layer":
                    payload = self.controller.trace_probability_layer(ref_value, layer=str(subject))
                    message = self._probability_layer_summary(payload)
                else:
                    payload = self.controller.trace_action_probability(ref_value, action=str(subject))
                    message = self._action_probability_summary(payload)
            except FileNotFoundError:
                return [build_outbound_event("error", session_id=session_id, message="No rounds yet. Send a message first.")]
            except ValueError as exc:
                return [build_outbound_event("error", session_id=session_id, message=str(exc))]
            return [
                self._build_sidebar_snapshot_event(session, run_id=run_id),
                build_outbound_event(
                    "assistant_final",
                    session_id=session_id,
                    run_id=run_id,
                    message=message,
                    payload=payload,
                ),
            ]
        if command_name == "endogenous":
            tokens = [part for part in str(value_text or "").split() if part]
            trigger = tokens[0] if tokens else "idle"
            mode = tokens[1] if len(tokens) > 1 else None
            payload = self.controller.run_endogenous_tick(trigger=trigger, mode=mode)
            return [
                self._build_sidebar_snapshot_event(session, run_id=run_id),
                build_outbound_event(
                    "assistant_final",
                    session_id=session_id,
                    run_id=run_id,
                    message=self._endogenous_summary(payload),
                    payload=payload,
                    **self._linkage_fields(payload),
                ),
            ]
        if command_name == "initiative":
            tokens = [part for part in str(value_text or "").split() if part]
            subcommand = tokens[0] if tokens else "status"
            if subcommand == "status":
                payload = self.controller.initiative_status()
                message = "initiative status"
            elif subcommand == "distribution":
                payload = self.controller.initiative_distribution()
                message = "initiative distribution"
            elif subcommand == "trigger":
                trigger_tokens = [part for part in tokens[1:] if part != "force"]
                trigger = trigger_tokens[0] if trigger_tokens else "idle"
                mode = trigger_tokens[1] if len(trigger_tokens) > 1 else None
                force = any(part == "force" for part in tokens[1:])
                payload = self.controller.initiative_trigger_now(trigger=trigger, mode=mode, force=force)
                message = "initiative trigger"
            elif subcommand == "why":
                round_ref = tokens[1] if len(tokens) > 1 else "last"
                payload = self.controller.initiative_why(round_ref)
                message = str(dict(payload or {}).get("summary") or "initiative why")
            else:
                return [build_outbound_event("error", session_id=session_id, message="Usage: /initiative status|distribution|trigger [trigger] [mode]|why [round]")]
            return [
                self._build_sidebar_snapshot_event(session, run_id=run_id),
                build_outbound_event(
                    "assistant_final",
                    session_id=session_id,
                    run_id=run_id,
                    message=message,
                    payload=payload,
                    **self._linkage_fields(payload if isinstance(payload, dict) else None),
                ),
            ]
        if command_name == "monologue":
            tokens = [part for part in str(value_text or "").split() if part]
            subcommand = tokens[0] if tokens else "status"
            if subcommand == "status":
                payload = self.controller.monologue_status()
                message = "monologue status"
            elif subcommand == "show":
                limit = int(tokens[1]) if len(tokens) > 1 else 12
                payload = self.controller.monologue_show(limit=limit)
                message = "monologue show"
            else:
                return [build_outbound_event("error", session_id=session_id, message="Usage: /monologue status|show [limit]")]
            return [
                self._build_sidebar_snapshot_event(session, run_id=run_id),
                build_outbound_event(
                    "assistant_final",
                    session_id=session_id,
                    run_id=run_id,
                    message=message,
                    payload=payload,
                    **self._linkage_fields(payload if isinstance(payload, dict) else None),
                ),
            ]
        if command_name == "thought":
            round_ref = str(value_text or "last").strip() or "last"
            payload = self.controller.thought_snapshot(round_ref)
            return [
                self._build_sidebar_snapshot_event(session, run_id=run_id),
                build_outbound_event(
                    "assistant_final",
                    session_id=session_id,
                    run_id=run_id,
                    message="thought snapshot",
                    payload=payload,
                    **self._linkage_fields(payload),
                ),
            ]
        if command_name == "why-motivation":
            payload = self.controller.why_motivation(value_text or "last")
            return [
                self._build_sidebar_snapshot_event(session, run_id=run_id),
                build_outbound_event(
                    "assistant_final",
                    session_id=session_id,
                    run_id=run_id,
                    message=self._why_motivation_summary(payload),
                    payload=payload,
                    **self._linkage_fields(payload),
                ),
            ]
        if command_name == "replay-motivation":
            try:
                round_ref = self.controller.resolve_round_ref(value_text or "last")
            except (FileNotFoundError, ValueError) as exc:
                return [build_outbound_event("error", session_id=session_id, message=str(exc))]
            payload = self.controller.replay_motivation(round_ref)
            return [
                self._build_sidebar_snapshot_event(session, run_id=run_id),
                build_outbound_event(
                    "assistant_final",
                    session_id=session_id,
                    run_id=run_id,
                    message=self._replay_motivation_summary(payload),
                    payload=payload,
                    **self._linkage_fields(payload),
                ),
            ]
        if command_name == "replay":
            tokens = [part for part in str(value_text or "").split() if part]
            round_ref = tokens[0] if tokens else "last"
            seed = int(tokens[1]) if len(tokens) > 1 and tokens[1].lstrip("-").isdigit() else 0
            try:
                round_id = self.controller.resolve_round_ref(round_ref)
            except (FileNotFoundError, ValueError) as exc:
                return [build_outbound_event("error", session_id=session_id, message=str(exc))]
            payload = self.controller.replay(round_id, seed=seed)
            return [
                self._build_sidebar_snapshot_event(session, run_id=run_id),
                build_outbound_event(
                    "assistant_final",
                    session_id=session_id,
                    run_id=run_id,
                    message=self._replay_summary(payload),
                    payload=payload,
                    **self._linkage_fields(payload),
                ),
            ]
        if command_name == "why-not":
            tokens = [part for part in str(value_text or "").split() if part]
            if not tokens:
                return [build_outbound_event("error", session_id=session_id, message="Usage: /why-not <action> [round] or /why-not <round> <action>")]
            round_ref: int | str = "last"
            action = " ".join(tokens)
            if len(tokens) >= 2 and (tokens[0] == "last" or tokens[0].lstrip("-").isdigit()):
                round_ref = tokens[0]
                action = " ".join(tokens[1:]).strip()
            if not action:
                return [build_outbound_event("error", session_id=session_id, message="Usage: /why-not <action> [round] or /why-not <round> <action>")]
            try:
                round_id = self.controller.resolve_round_ref(round_ref)
            except (FileNotFoundError, ValueError) as exc:
                return [build_outbound_event("error", session_id=session_id, message=str(exc))]
            payload = self.controller.why_not(round_id, action)
            return [
                self._build_sidebar_snapshot_event(session, run_id=run_id),
                build_outbound_event(
                    "assistant_final",
                    session_id=session_id,
                    run_id=run_id,
                    message=self._why_not_summary(payload),
                    payload=payload,
                    **self._linkage_fields(payload),
                ),
            ]
        if command_name == "what-changed":
            if value_text is None:
                window = 5
            else:
                try:
                    window = max(1, int(value_text))
                except ValueError:
                    return [build_outbound_event("error", session_id=session_id, message="Usage: /what-changed [window]")]
            payload = self.controller.what_changed(window=window)
            return [
                self._build_sidebar_snapshot_event(session, run_id=run_id),
                build_outbound_event(
                    "assistant_final",
                    session_id=session_id,
                    run_id=run_id,
                    message=self._what_changed_summary(payload),
                    payload=payload,
                    **self._linkage_fields(payload),
                ),
            ]
        if command_name == "eval":
            if value_text is None:
                rounds = 1000
            else:
                try:
                    rounds = max(1, int(value_text))
                except ValueError:
                    return [build_outbound_event("error", session_id=session_id, message="Usage: /eval [rounds]")]
            payload = self.controller.eval_longrun(rounds=rounds)
            return [
                self._build_sidebar_snapshot_event(session, run_id=run_id),
                build_outbound_event(
                    "assistant_final",
                    session_id=session_id,
                    run_id=run_id,
                    message=self._eval_summary(payload),
                    payload=payload,
                    **self._linkage_fields(payload),
                ),
            ]
        if not run_id:
            return [build_outbound_event("error", session_id=session_id, message="no active run for this terminal session")]

        if command_name == "status":
            run = self.controller.run_status(run_id)
            return [
                build_outbound_event("run_status", session_id=session_id, run=run, **self._linkage_fields(run)),
                self._build_sidebar_snapshot_event(session, run_id=run_id),
                build_outbound_event(
                    "assistant_final",
                    session_id=session_id,
                    run_id=run_id,
                    message=self._status_summary(run),
                    **self._linkage_fields(run),
                ),
            ]
        if command_name == "why":
            explain = self.controller.explain_run(run_id)
            return [
                self._build_sidebar_snapshot_event(session, run_id=run_id),
                build_outbound_event(
                    "assistant_final",
                    session_id=session_id,
                    run_id=run_id,
                    message=self._why_summary(explain),
                    payload=explain,
                    **self._linkage_fields(explain),
                ),
            ]
        if command_name == "steps":
            steps = self.controller.run_steps(run_id)["steps"]
            events = [build_outbound_event("step_update", session_id=session_id, step=step, **self._linkage_fields(step)) for step in steps]
            events.append(self._build_sidebar_snapshot_event(session, run_id=run_id))
            events.append(
                build_outbound_event(
                    "assistant_final",
                    session_id=session_id,
                    run_id=run_id,
                    message=self._steps_summary(steps),
                    **self._linkage_fields(steps[-1] if steps else None),
                )
            )
            return events
        if command_name == "tools":
            tools = self.controller.run_tools(run_id)["tools"]
            events = [
                build_outbound_event(
                    "tool_result",
                    session_id=session_id,
                    call_id=f"{run_id}:tool:{index}",
                    result=tool,
                    **self._linkage_fields(tool),
                )
                for index, tool in enumerate(tools)
            ]
            events.append(self._build_sidebar_snapshot_event(session, run_id=run_id))
            events.append(
                build_outbound_event(
                    "assistant_final",
                    session_id=session_id,
                    run_id=run_id,
                    message=self._tools_summary(tools),
                    **self._linkage_fields(tools[-1] if tools else None),
                )
            )
            return events
        if command_name == "pause":
            run = self.controller.pause_run(run_id)
            return [
                build_outbound_event("run_status", session_id=session_id, run=run, **self._linkage_fields(run)),
                self._build_sidebar_snapshot_event(session, run_id=run_id),
                build_outbound_event(
                    "assistant_final",
                    session_id=session_id,
                    run_id=run_id,
                    message="已暂停当前任务。",
                    **self._linkage_fields(run),
                ),
            ]
        if command_name == "resume":
            run = self.controller.resume_run(run_id)
            return [
                build_outbound_event("run_status", session_id=session_id, run=run, **self._linkage_fields(run)),
                self._build_sidebar_snapshot_event(session, run_id=run_id),
                build_outbound_event(
                    "assistant_final",
                    session_id=session_id,
                    run_id=run_id,
                    message="已恢复当前任务。",
                    **self._linkage_fields(run),
                ),
            ]
        if command_name == "abort":
            run = self.controller.abort_run(run_id, reason="operator_requested")
            return [
                build_outbound_event("run_status", session_id=session_id, run=run, **self._linkage_fields(run)),
                self._build_sidebar_snapshot_event(session, run_id=run_id),
                build_outbound_event(
                    "assistant_final",
                    session_id=session_id,
                    run_id=run_id,
                    message="已中止当前任务。",
                    **self._linkage_fields(run),
                ),
            ]
        return [build_outbound_event("error", session_id=session_id, message=f"unsupported control command: {command_name}")]

    def _approve(self, session_id: str, call_id: str, approved: bool) -> list[dict[str, Any]]:
        session = self._load_session(session_id)
        updated_approval: dict[str, Any] | None = None
        for item in session.approvals_pending:
            if item.get("call_id") != call_id:
                continue
            item["status"] = "approved" if approved else "rejected"
            item["approved"] = approved
            item["responded_at"] = utc_now_iso()
            updated_approval = item
            break
        if updated_approval is None:
            updated_approval = {
                "call_id": call_id,
                "tool": "unknown",
                "mode": session.permission_mode,
                "status": "approved" if approved else "rejected",
                "approved": approved,
                "responded_at": utc_now_iso(),
                "deferred_events": [],
                "deferred_final": None,
                "round_id": None,
                "trace_ref": None,
            }
        deferred_events = list(updated_approval.get("deferred_events", []) or [])
        resolved_payload = None
        if updated_approval.get("run_id"):
            resolved_payload = self.controller.resolve_run_tool_approval(
                str(updated_approval["run_id"]),
                call_id,
                approved=approved,
            )
            tool_result_event = build_outbound_event(
                "tool_result",
                session_id=session_id,
                call_id=call_id,
                result=resolved_payload["tool"],
                **self._linkage_fields(resolved_payload["tool"], resolved_payload["run"]),
            )
            deferred_events = [tool_result_event, *deferred_events] if approved else [tool_result_event]
        elif not approved:
            deferred_events = [
                self._gated_tool_result_event(
                    {
                        "session_id": session_id,
                        "call_id": call_id,
                        "round_id": updated_approval.get("round_id"),
                        "trace_ref": updated_approval.get("trace_ref"),
                        "result": {
                            "tool_name": updated_approval.get("tool") or "unknown",
                            "summary": updated_approval.get("summary") or "tool execution rejected",
                            "status": "rejected",
                        },
                    }
                )
            ]
        for event in deferred_events:
            self._apply_event_to_session(session, event)
        session.approvals_pending = [item for item in session.approvals_pending if item.get("call_id") != call_id]
        final_event = None
        if approved and isinstance(updated_approval.get("deferred_final"), dict):
            deferred_final = dict(updated_approval["deferred_final"])
            if isinstance(resolved_payload, dict):
                deferred_final.update(self._linkage_fields(resolved_payload.get("run"), resolved_payload.get("tool")))
            final_event = build_outbound_event(
                "assistant_final",
                session_id=session_id,
                run_id=deferred_final.get("run_id"),
                message=str(deferred_final.get("message") or ""),
                payload=deferred_final.get("payload"),
                **self._linkage_fields(deferred_final),
            )
            self._apply_event_to_session(session, final_event)
        self.session_store.write(session)
        events = [self._build_sidebar_snapshot_event(session, run_id=session.active_run_id or session.last_run_id)]
        events.extend(deferred_events)
        if final_event is not None:
            events.append(final_event)
        if not approved:
            events.append(
                build_outbound_event(
                    "assistant_final",
                    session_id=session_id,
                    message=f"approval {call_id} recorded as rejected",
                    payload={"approval": updated_approval, "run": resolved_payload.get("run") if isinstance(resolved_payload, dict) else None},
                )
            )
        return events

    def _close_session(
        self,
        session_id: str,
        *,
        detach: bool = False,
        purge: bool = False,
        cleanup_old: bool = False,
        transcript_mode: str | None = None,
    ) -> list[dict[str, Any]]:
        session = self._load_session(session_id)
        if transcript_mode:
            session.transcript_mode = transcript_mode
            session.compact = transcript_mode == "compact"
        session.status = "detached" if detach else "ended"
        session_payload = to_dict(session)
        if purge:
            self.session_store.delete(session_id)
        else:
            self.session_store.write(session, mark_current=detach)
            if not detach:
                self.session_store.clear_current(session_id)
        cleaned_session_ids = self.session_store.prune(statuses={"ended", "detached"}) if cleanup_old else []
        cleaned_session_ids = [item for item in cleaned_session_ids if item != session_id]
        return [build_outbound_event("session_ended", session=session_payload, cleaned_session_ids=cleaned_session_ids)]

    def _status_summary(self, run_payload: dict[str, Any]) -> str:
        current_step = run_payload.get("current_step") or {}
        paused = "是" if run_payload.get("status") == "paused" else "否"
        dirty = "dirty" if run_payload.get("dirty_worktree_detected") else "clean"
        return "\n".join(
            [
                f"状态：{run_payload.get('status') or 'unknown'}",
                f"当前步骤：{current_step.get('title') or '暂无'}",
                f"已暂停：{paused}",
                f"工作区：{dirty}",
            ]
        )

    def _why_summary(self, explain_payload: dict[str, Any]) -> str:
        current_step = explain_payload.get("current_step") or {}
        stop_reason = explain_payload.get("stop_reason") or {}
        reason = current_step.get("expected_observation") or current_step.get("detail") or "先收集当前任务最直接的上下文。"
        parts = [
            "说明类型：bootstrap规划",
            f"当前目标：{explain_payload.get('goal_summary') or explain_payload.get('goal') or '暂无'}",
            f"当前选择：{current_step.get('title') or '暂无'}",
            f"规划依据：{reason}",
            "真实性证据：暂无轮次证据",
        ]
        if stop_reason.get("message"):
            parts.append(f"暂停原因：{stop_reason['message']}")
        return "\n".join(parts)

    def _cognitive_summary(self, cognitive_snapshot: dict[str, Any]) -> str:
        vital_signs = cognitive_snapshot.get("vital_signs") or {}
        identity = cognitive_snapshot.get("identity") or {}
        authenticity = cognitive_snapshot.get("authenticity") or {}
        return "\n".join(
            [
                f"核心目标：{cognitive_snapshot.get('core_goal') or '暂无'}",
                f"当前意图：{cognitive_snapshot.get('current_intent') or '暂无'}",
                f"心境：{vital_signs.get('mood', '暂无')}",
                f"能量：{vital_signs.get('body_energy', '暂无')}",
                f"情感余波：{vital_signs.get('affect_residue', '暂无')}",
                f"焦点：{vital_signs.get('focus') or '暂无'}",
                f"模式：{vital_signs.get('mode') or '暂无'}",
                f"身份：{identity.get('display_name') or '暂无'}",
                f"连续性：{identity.get('continuity') or '暂无'}",
                f"真实性：{authenticity.get('summary') or '暂无'}",
            ]
        )

    def _steps_summary(self, steps: list[dict[str, Any]]) -> str:
        current = next((step for step in steps if step.get("status") in {"running", "in_progress", "active"}), steps[0] if steps else {})
        remaining_steps = [step for step in steps if step.get("status") not in {"running", "in_progress", "active", "completed", "done"}]
        trailing = "；".join(step.get("title") or step.get("step_id") or "未命名步骤" for step in remaining_steps[:3])
        parts = [
            f"当前步骤：{current.get('title') or current.get('step_id') or '暂无'}",
            f"剩余步骤：{len(remaining_steps)}",
        ]
        if trailing:
            parts.append(f"后续：{trailing}")
        return "\n".join(parts)

    def _tools_summary(self, tools: list[dict[str, Any]]) -> str:
        recent = tools[-3:]
        if not recent:
            return "最近工具：暂无"
        rows = []
        for tool in recent:
            name = tool.get("tool_name") or "unknown"
            summary = tool.get("summary") or tool.get("status") or "已执行"
            rows.append(f"{name}: {summary}")
        return f"最近工具：{'；'.join(rows)}"

    def _parse_probability_command(self, value_text: str | None) -> tuple[str, str, str | None]:
        tokens = [part for part in str(value_text or "").split() if part]
        if not tokens:
            return ("overview", "last", None)
        head = tokens[0].lower()
        if head == "layer":
            if len(tokens) not in {2, 3}:
                raise ValueError("Usage: /probability layer <context|memory|action|token> [round]")
            return ("layer", tokens[2] if len(tokens) == 3 else "last", tokens[1].lower())
        if head == "action":
            if len(tokens) not in {2, 3}:
                raise ValueError("Usage: /probability action <name> [round]")
            return ("action", tokens[2] if len(tokens) == 3 else "last", tokens[1])
        if len(tokens) > 1:
            raise ValueError("Usage: /probability [round] | /probability action <name> [round] | /probability layer <name> [round]")
        return ("overview", tokens[0], None)

    def _probability_summary(self, payload: dict[str, Any]) -> str:
        probability_field = dict(payload.get("probability_field", {}) or {})
        lines = [
            f"概率场：round {payload.get('round_id', '?')}，sampled={payload.get('sampled_action') or 'unknown'}",
        ]
        for layer_name in ("context", "memory", "action", "token"):
            layer = probability_field.get(layer_name, {})
            if not isinstance(layer, dict):
                continue
            posterior = dict(layer.get("winner_posterior", {}) or {})
            top = sorted(posterior.items(), key=lambda item: float(item[1]), reverse=True)[:3]
            top_text = " / ".join(f"{name}={float(score):.3f}" for name, score in top) if top else "-"
            lines.append(
                f"{layer_name}: winner={layer.get('winner_target') or '-'} top={top_text} "
                f"audit={len(layer.get('contribution_audit', []) or [])}"
            )
        token_state = dict(payload.get("token_state", {}) or {})
        if token_state:
            lines.append(
                f"token_state: step={token_state.get('step_index', '-')} "
                f"active={len(token_state.get('active_module_sources', []) or [])}"
            )
        return "\n".join(lines)

    def _probability_layer_summary(self, payload: dict[str, Any]) -> str:
        layer_name = str(payload.get("layer") or "unknown")
        layer = dict(payload.get("layer_state", {}) or {})
        posterior = dict(layer.get("winner_posterior", {}) or {})
        top = sorted(posterior.items(), key=lambda item: float(item[1]), reverse=True)[:3]
        top_text = " / ".join(f"{name}={float(score):.3f}" for name, score in top) if top else "-"
        audit_rows = list(layer.get("contribution_audit", []) or [])
        lead_modules = " / ".join(str(row.get("module_name") or "-") for row in audit_rows[:4] if isinstance(row, dict)) or "-"
        return (
            f"{layer_name} 层：round {payload.get('round_id', '?')} winner={layer.get('winner_target') or '-'}\n"
            f"top={top_text}\n"
            f"audit={len(audit_rows)} lead={lead_modules}"
        )

    def _action_probability_summary(self, payload: dict[str, Any]) -> str:
        explanation = dict(payload.get("action_probability_explanation", {}) or {})
        target = str(payload.get("action") or explanation.get("target_action") or "unknown")
        posterior = dict(explanation.get("winner_posterior", {}) or {})
        target_probability = float(posterior.get(target, 0.0) or 0.0)
        peaks = list(explanation.get("competing_peaks", []) or [])
        peak_text = " / ".join(
            f"{row.get('action', '-')}={float(row.get('posterior', 0.0) or 0.0):.3f}"
            for row in peaks[:3]
            if isinstance(row, dict)
        ) or "-"
        stacked = list(explanation.get("stacked_contributions", []) or [])
        stacked_text = " / ".join(
            f"{row.get('module_name')}:{row.get('direction')}"
            for row in stacked[:4]
            if isinstance(row, dict)
        ) or "-"
        return (
            f"动作概率：target={target} round {payload.get('round_id', '?')} winner={explanation.get('winner_target') or '-'}\n"
            f"p={target_probability:.3f} hard_masked={'yes' if explanation.get('hard_masked') else 'no'}\n"
            f"peaks={peak_text}\n"
            f"stacked={stacked_text}"
        )

    def _model_summary(self, model_status: dict[str, Any]) -> str:
        tiers = dict(model_status.get("tiers", {}))
        bindings = dict(model_status.get("module_model_bindings", {}))
        rows = ["模型分层："]
        for tier_name in tiers:
            tier = dict(tiers.get(tier_name, {}))
            mode = str(tier.get("mode", "unknown"))
            if mode == "local":
                rows.append(f"{tier_name}: local")
                continue
            rows.append(
                f"{tier_name}: {tier.get('backend') or 'unknown'} / "
                f"{tier.get('model') or 'unconfigured'} / "
                f"{'key:ok' if tier.get('credential_present') else 'key:missing'}"
            )
        rows.append("")
        rows.append("主要模块模型绑定：")
        for module_name in ("cognitive_packet", "deliberation", "tool_planner", "deep_renderer", "consolidation_summarizer"):
            if module_name in bindings:
                rows.append(f"{module_name} -> {bindings[module_name]}")
        return "\n".join(rows)

    def _sidebar_snapshot(self, session: TerminalSessionState, *, run_id: str | None = None) -> dict[str, Any]:
        run_payload = self.controller.run_status(run_id) if run_id else None
        why_payload = self.controller.explain_run(run_id) if run_id else None
        steps_payload = self.controller.run_steps(run_id)["steps"] if run_id else []
        tools_payload = self.controller.run_tools(run_id)["tools"] if run_id else []
        return self._build_snapshot_payload(
            session,
            run_id=run_id,
            run=run_payload,
            explain=why_payload,
            steps=steps_payload,
            tools=tools_payload,
        )

    def _build_snapshot_payload(
        self,
        session: TerminalSessionState,
        *,
        run_id: str | None,
        run: dict[str, Any] | None,
        explain: dict[str, Any] | None,
        steps: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        include_console: bool = True,
    ) -> dict[str, Any]:
        pending = [item for item in session.approvals_pending if item.get("status") == "pending"]
        current_step = ((explain or {}).get("current_step") or (run or {}).get("current_step") or {})
        last_tool = tools[-1] if tools else ((run or {}).get("last_tool_result") or {})
        runtime_state = self.controller.load_runtime_state()
        cognitive_snapshot = self.controller.cognitive_snapshot(
            state=runtime_state,
            run_payload=run,
        )
        model_status = self.controller.model_status()
        linkage = self._linkage_fields(run, explain, steps[-1] if steps else None, tools[-1] if tools else None)
        goal_summary = str((explain or {}).get("goal_summary") or (run or {}).get("goal_summary") or (run or {}).get("goal") or "暂无")
        current_step_title = str(current_step.get("title") or "暂无")
        reason_summary = str(current_step.get("expected_observation") or current_step.get("detail") or "先收集当前任务最直接的上下文。")
        last_tool_summary = str(last_tool.get("tool_name") or "暂无")
        console_payload = (
            self.controller.console_refresh_payload(linkage.get("round_id"))
            if include_console
            else self._light_console_payload(runtime_state=runtime_state, round_id=linkage.get("round_id"))
        )
        return {
            "goal_summary": goal_summary,
            "current_step": current_step_title,
            "reason_summary": reason_summary,
            "last_tool": last_tool_summary,
            "run_status": str((run or {}).get("status") or session.status or "idle"),
            "permission_mode": session.permission_mode,
            "pending_approval_count": len(pending),
            "status": run or None,
            "why": explain or None,
            "steps": steps,
            "tools": tools,
            "cognitive_snapshot": cognitive_snapshot,
            "session": {
                "session_id": session.session_id,
                "cwd": session.cwd,
                "status": session.status,
                "mode": session.mode,
                "permission_mode": session.permission_mode,
                "compact": session.compact,
                "active_run_id": session.active_run_id,
                "last_run_id": session.last_run_id,
                "approvals_pending": session.approvals_pending,
            },
            "approvals": {
                "pending_count": len(pending),
                "pending": pending,
            },
            "controlled_learning": self._controlled_learning_snapshot(runtime_state),
            "ui_actions": self._ui_actions(session, run, pending_count=len(pending)),
            "model_status": model_status,
            "statusline": self._build_statusline(session, run_id=run_id, run=run),
            "console": console_payload,
            **linkage,
        }

    def _build_statusline(
        self,
        session: TerminalSessionState,
        *,
        run_id: str | None,
        run: dict[str, Any] | None,
    ) -> dict[str, Any]:
        return {
            "cwd": session.cwd,
            "git": "dirty" if (run or {}).get("dirty_worktree_detected") else "clean",
            "permission_mode": session.permission_mode,
            "model": self._model_label(),
            "run_status": str((run or {}).get("status") or session.status or "idle"),
            "session_id": session.session_id,
            "run_id": str(run_id or ""),
            **self._linkage_fields(run),
        }

    def _model_label(self) -> str:
        model_status = self.controller.model_status()
        tiers = dict(model_status.get("tiers", {}))
        small = dict(tiers.get("small_model", {}))
        medium = dict(tiers.get("medium_model", {}))
        large = dict(tiers.get("large_model", {}))
        return (
            f"small:{small.get('model') or 'off'} "
            f"medium:{medium.get('model') or 'off'} "
            f"large:{large.get('model') or 'off'}"
        )

    def _controlled_learning_snapshot(self, runtime_state) -> dict[str, Any]:
        policy = runtime_state.autonomy_policy
        return {
            "learning_mode": str(policy.learning_mode or "guided-learn"),
            "network_enabled": bool(policy.network_enabled),
            "external_io_enabled": bool(policy.external_io_enabled),
            "trace_external_learning": bool(policy.trace_external_learning),
            "allowed_network_domains": list(policy.allowed_network_domains or []),
            "writable_roots": list(policy.writable_roots or []),
            "knowledge_roots": list(policy.knowledge_roots or []),
            "learning_log_dir": str(policy.learning_log_dir or ""),
            "max_rounds_per_hour": int(policy.max_rounds_per_hour or 0),
            "max_tool_actions_per_hour": int(policy.max_tool_actions_per_hour or 0),
            "failure_trip_threshold": int(policy.failure_trip_threshold or 0),
            "auto_safe_mode": bool(policy.auto_safe_mode),
            "safe_mode": bool(runtime_state.safe_mode),
            "budget_remaining": round(float(runtime_state.budget_remaining or 0.0), 4),
        }

    def _build_sidebar_snapshot_event(
        self,
        session: TerminalSessionState,
        *,
        run_id: str | None = None,
        run: dict[str, Any] | None = None,
        explain: dict[str, Any] | None = None,
        steps: list[dict[str, Any]] | None = None,
        tools: list[dict[str, Any]] | None = None,
        include_console: bool = True,
    ) -> dict[str, Any]:
        if run_id and run is None:
            run = self.controller.run_status(run_id)
        if run_id and explain is None:
            explain = self.controller.explain_run(run_id)
        if run_id and steps is None:
            steps = self.controller.run_steps(run_id)["steps"]
        if run_id and tools is None:
            tools = self.controller.run_tools(run_id)["tools"]
        payload = self._build_snapshot_payload(
            session,
            run_id=run_id,
            run=run,
            explain=explain,
            steps=steps or [],
            tools=tools or [],
            include_console=include_console,
        )
        return build_outbound_event("sidebar_snapshot", session_id=session.session_id, **payload)
