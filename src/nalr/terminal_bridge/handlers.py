from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from nalr.runtime.controller import RuntimeController
from nalr.runtime.metadata import utc_now_iso
from nalr.schemas.models import to_dict
from nalr.terminal_bridge.protocol import ProtocolError, build_outbound_event, validate_inbound_event
from nalr.terminal_bridge.session import TerminalSessionState, TerminalSessionStore


PERMISSION_MODES = {"plan", "ask", "acceptEdits"}


class TerminalEventHandler:
    def __init__(self, controller: RuntimeController) -> None:
        self.controller = controller
        self.session_store = TerminalSessionStore(controller.runtime_dir)

    def handle(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        return list(self.handle_stream(payload))

    def handle_stream(self, payload: dict[str, Any]) -> Iterable[dict[str, Any]]:
        event = validate_inbound_event(payload)
        event_type = event["type"]
        if event_type == "start_session":
            return self._start_session(event["session_id"], event["cwd"])
        if event_type == "user_turn":
            return self._user_turn_stream(event["session_id"], event["text"])
        if event_type == "control_command":
            return self._control_command(event["session_id"], event["command"], event.get("value"))
        if event_type == "approve":
            return self._approve(event["session_id"], event["call_id"], event["approved"])
        if event_type == "close_session":
            return self._close_session(event["session_id"])
        raise ProtocolError(f"unhandled event type: {event_type}")

    def _load_session(self, session_id: str) -> TerminalSessionState:
        return self.session_store.read(session_id)

    def _start_session(self, session_id: str, cwd: str) -> list[dict[str, Any]]:
        try:
            state = self.session_store.read(session_id)
            state.cwd = cwd
            state.status = "active"
        except FileNotFoundError:
            state = TerminalSessionState(session_id=session_id, cwd=cwd, status="active")
        self.session_store.write(state)
        return [
            build_outbound_event("session_started", session=to_dict(state)),
            self._build_sidebar_snapshot_event(state),
        ]

    def _user_turn_stream(self, session_id: str, text: str) -> Iterable[dict[str, Any]]:
        session = self._load_session(session_id)
        plan = self.controller.plan_turn(
            text.strip(),
            target="user",
            mode="interactive",
            operator_level="read_only",
        )
        execution = self.controller.execute_turn(
            plan,
            operator_level="read_only",
            replace_active=True,
            interrupt_reason="interrupted_by_user",
        )

        route = execution["route"] if isinstance(execution, dict) else execution.route
        if route == "direct_chat":
            yield build_outbound_event(
                "assistant_final",
                session_id=session_id,
                message=(execution["assistant_final"] if isinstance(execution, dict) else execution.assistant_final),
            )
            return

        run = execution["run"] if isinstance(execution, dict) else execution.run
        explain = execution["explain"] if isinstance(execution, dict) else execution.explain
        steps = execution["steps"] if isinstance(execution, dict) else execution.steps
        tools = execution["tools"] if isinstance(execution, dict) else execution.tools
        task_message = execution["assistant_preamble"] if isinstance(execution, dict) else execution.assistant_preamble
        session.active_run_id = run["run_id"]
        session.last_run_id = run["run_id"]
        session.status = "active"
        self.session_store.write(session)
        yield build_outbound_event("run_status", session_id=session_id, run=run)
        yield build_outbound_event("assistant_token", session_id=session_id, delta=task_message)
        pending_approvals = [item for item in session.approvals_pending if item.get("status") == "pending"]
        for step in steps:
            yield build_outbound_event("step_update", session_id=session_id, step=step)
        for index, tool in enumerate(tools):
            call_id = f"{run['run_id']}:tool:{index}"
            tool_name = str(tool.get("tool_name") or "unknown")
            yield build_outbound_event(
                "tool_call",
                session_id=session_id,
                call_id=call_id,
                run_id=run["run_id"],
                tool=tool_name,
                args=tool.get("input") or {},
                summary=tool.get("summary") or tool.get("status") or "",
                status=tool.get("status"),
            )
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
                    "requested_at": utc_now_iso(),
                }
                pending_approvals.append(approval_payload)
                session.approvals_pending = pending_approvals
                self.session_store.write(session)
                yield build_outbound_event(
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
                    approved=None,
                )
            yield build_outbound_event("tool_result", session_id=session_id, call_id=call_id, result=tool)

        yield self._build_sidebar_snapshot_event(session, run_id=run["run_id"], run=run, explain=explain, steps=steps, tools=tools)
        yield build_outbound_event(
            "assistant_final",
            session_id=session_id,
            run_id=run["run_id"],
            message=(execution["assistant_final"] if isinstance(execution, dict) else execution.assistant_final),
            payload=explain,
        )

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
            routes = self.controller.config["models"]["model_routes"]
            return [
                self._build_sidebar_snapshot_event(session, run_id=run_id),
                build_outbound_event("assistant_final", session_id=session_id, message=self._model_summary(routes)),
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
        if not run_id:
            return [build_outbound_event("error", session_id=session_id, message="no active run for this terminal session")]

        if command_name == "status":
            run = self.controller.run_status(run_id)
            return [
                build_outbound_event("run_status", session_id=session_id, run=run),
                self._build_sidebar_snapshot_event(session, run_id=run_id),
                build_outbound_event("assistant_final", session_id=session_id, run_id=run_id, message=self._status_summary(run)),
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
                ),
            ]
        if command_name == "steps":
            steps = self.controller.run_steps(run_id)["steps"]
            events = [build_outbound_event("step_update", session_id=session_id, step=step) for step in steps]
            events.append(self._build_sidebar_snapshot_event(session, run_id=run_id))
            events.append(
                build_outbound_event(
                    "assistant_final",
                    session_id=session_id,
                    run_id=run_id,
                    message=self._steps_summary(steps),
                )
            )
            return events
        if command_name == "tools":
            tools = self.controller.run_tools(run_id)["tools"]
            events = [build_outbound_event("tool_result", session_id=session_id, call_id=f"{run_id}:tool:{index}", result=tool) for index, tool in enumerate(tools)]
            events.append(self._build_sidebar_snapshot_event(session, run_id=run_id))
            events.append(
                build_outbound_event(
                    "assistant_final",
                    session_id=session_id,
                    run_id=run_id,
                    message=self._tools_summary(tools),
                )
            )
            return events
        if command_name == "pause":
            run = self.controller.pause_run(run_id)
            return [
                build_outbound_event("run_status", session_id=session_id, run=run),
                self._build_sidebar_snapshot_event(session, run_id=run_id),
                build_outbound_event("assistant_final", session_id=session_id, run_id=run_id, message="已暂停当前任务。"),
            ]
        if command_name == "resume":
            run = self.controller.resume_run(run_id)
            return [
                build_outbound_event("run_status", session_id=session_id, run=run),
                self._build_sidebar_snapshot_event(session, run_id=run_id),
                build_outbound_event("assistant_final", session_id=session_id, run_id=run_id, message="已恢复当前任务。"),
            ]
        if command_name == "abort":
            run = self.controller.abort_run(run_id, reason="operator_requested")
            return [
                build_outbound_event("run_status", session_id=session_id, run=run),
                self._build_sidebar_snapshot_event(session, run_id=run_id),
                build_outbound_event("assistant_final", session_id=session_id, run_id=run_id, message="已中止当前任务。"),
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
            }
            session.approvals_pending.append(updated_approval)
        self.session_store.write(session)
        return [
            self._build_sidebar_snapshot_event(session, run_id=session.active_run_id or session.last_run_id),
            build_outbound_event(
                "assistant_final",
                session_id=session_id,
                message=f"approval {call_id} recorded as {'approved' if approved else 'rejected'}",
                payload={"approval": updated_approval},
            )
        ]

    def _close_session(self, session_id: str) -> list[dict[str, Any]]:
        session = self._load_session(session_id)
        session.status = "ended"
        self.session_store.write(session)
        return [build_outbound_event("session_ended", session=to_dict(session))]

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

    def _model_summary(self, routes: dict[str, Any]) -> str:
        enabled = [
            f"{name}: {route.get('model') or '未配置'}"
            for name, route in routes.items()
            if route.get("enabled", False)
        ]
        if not enabled:
            return "当前没有启用的模型路由。"
        return "已启用模型路由：\n" + "\n".join(enabled)

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
    ) -> dict[str, Any]:
        pending = [item for item in session.approvals_pending if item.get("status") == "pending"]
        current_step = ((explain or {}).get("current_step") or (run or {}).get("current_step") or {})
        last_tool = tools[-1] if tools else ((run or {}).get("last_tool_result") or {})
        runtime_state = self.controller.load_runtime_state()
        cognitive_snapshot = self.controller.cognitive_snapshot(
            state=runtime_state,
            run_payload=run,
        )
        goal_summary = str((explain or {}).get("goal_summary") or (run or {}).get("goal_summary") or (run or {}).get("goal") or "暂无")
        current_step_title = str(current_step.get("title") or "暂无")
        reason_summary = str(current_step.get("expected_observation") or current_step.get("detail") or "先收集当前任务最直接的上下文。")
        last_tool_summary = str(last_tool.get("tool_name") or "暂无")
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
            "statusline": self._build_statusline(session, run_id=run_id, run=run),
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
        }

    def _model_label(self) -> str:
        routes = self.controller.config["models"]["model_routes"]
        enabled = [
            f"{name}:{route.get('model') or 'unconfigured'}"
            for name, route in routes.items()
            if route.get("enabled", False)
        ]
        return enabled[0] if enabled else "default"

    def _build_sidebar_snapshot_event(
        self,
        session: TerminalSessionState,
        *,
        run_id: str | None = None,
        run: dict[str, Any] | None = None,
        explain: dict[str, Any] | None = None,
        steps: list[dict[str, Any]] | None = None,
        tools: list[dict[str, Any]] | None = None,
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
        )
        return build_outbound_event("sidebar_snapshot", session_id=session.session_id, **payload)
