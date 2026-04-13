from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from nalr.memory.store import _derive_cue
from nalr.providers import ModelRequest
from nalr.runtime.metadata import iso_date, utc_now_iso
from nalr.schemas.models import (
    ActionCandidate,
    ActivationPlan,
    AgentContribution,
    CognitivePacket,
    ConsolidationJob,
    MemoryReadPlan,
    ProactiveActionState,
    RenderedExpression,
    RoundDiagnosticsV2,
    RoundEvent,
    RoundTrace,
    RuntimeState,
    StatePatch,
    TurnExecution,
    TurnPlan,
    to_dict,
)

if TYPE_CHECKING:
    from nalr.runtime.controller import RuntimeController


class _ControllerBackedRuntime:
    def __init__(self, controller: "RuntimeController") -> None:
        self.controller = controller

    def __getattr__(self, name: str):
        return getattr(self.controller, name)


class ChatKernelV2(_ControllerBackedRuntime):
    _BASE_MODULES = [
        "Router",
        "HotStateLoader",
        "BudgetAllocator",
        "PacketAssembler",
        "SafetyGate",
        "StateWriter",
    ]

    def execute(self, plan: TurnPlan) -> TurnExecution:
        started = time.perf_counter()
        state = RuntimeState(**to_dict(self.load_runtime_state()))
        event = RoundEvent(source="user", content=plan.text, target=plan.target)
        cue = _derive_cue(event)

        state.mode = plan.mode
        state.mode_history = (list(state.mode_history or []) + [plan.mode])[-20:]
        state.round_count = int(state.round_count or 0) + 1
        round_id = state.round_count

        hot_state = self._hot_state_payload(state)
        packet, model_call_traces, packet_response = self._generate_packet(
            plan=plan,
            state=state,
            hot_state=hot_state,
            cue=cue,
        )
        activation_plan = self._activation_plan(plan=plan, packet=packet, cue=cue)
        memory_context = self._memory_context(
            cue=cue,
            read_plan=activation_plan.memory_read_plan,
        )

        final_message = packet.draft_reply or "你好，我在。你想让我帮你做什么？"
        deepen_reason = activation_plan.deepen_reason

        if plan.route_type == "chat_deep":
            final_message, deepen_reason, deep_trace = self._run_deep_deliberation(
                plan=plan,
                packet=packet,
                hot_state=hot_state,
                memory_context=memory_context,
                cue=cue,
            )
            model_call_traces.extend(deep_trace)
            if deepen_reason:
                activation_plan.deepen_reason = deepen_reason
            if "DeepRenderer" not in activation_plan.activation_set:
                activation_plan.activation_set.append("DeepRenderer")
            if deepen_reason and deepen_reason not in activation_plan.activation_reason:
                activation_plan.activation_reason.append(deepen_reason)

        validated_patch = self._validate_state_patch(packet.proposed_state_patch, cue=cue)
        self._apply_state_patch(
            state,
            validated_patch,
            event=event,
            cue=cue,
            route_type=plan.route_type,
            salience=packet.salience,
        )

        diagnostics = RoundDiagnosticsV2(
            route_type=plan.route_type,
            route_budget_ms=self._route_budget_ms(plan.route_type),
            activation_set=list(activation_plan.activation_set),
            activation_reason=list(activation_plan.activation_reason),
            memory_tiers_read=list(activation_plan.memory_read_plan.tiers),
            packet_summary=self._packet_summary(packet),
            background_jobs=[to_dict(job) for job in activation_plan.background_jobs],
            deepen_reason=activation_plan.deepen_reason,
            model_call_count=len(model_call_traces),
        )
        rendered_model = ""
        if model_call_traces:
            rendered_model = str(model_call_traces[-1].get("model") or "")
        if not rendered_model:
            rendered_model = str((packet_response or {}).get("model") or "fallback")
        total_turn_ms = max(1, int((time.perf_counter() - started) * 1000))

        rendered_expression = RenderedExpression(
            text=final_message,
            route="deep_renderer" if plan.route_type == "chat_deep" else "cognitive_packet",
            model=rendered_model,
            degraded=not bool(model_call_traces),
        )
        trace = self._write_round_trace(
            state=state,
            plan=plan,
            event=event,
            packet=packet,
            activation_plan=activation_plan,
            diagnostics=diagnostics,
            model_call_traces=model_call_traces,
            rendered_expression=rendered_expression,
            total_turn_ms=total_turn_ms,
        )
        self._save_state(state)
        self.memory_store.ingest_event(
            event,
            round_id=round_id,
            session_id=state.session_id,
            recorded_at=trace.recorded_at,
            mode=plan.mode,
            cue_quality=min(1.0, max(0.0, 0.25 + packet.salience * 0.5)),
            resource_pressure=max(0.0, 1.0 - float(state.budget_remaining or 0.0)),
        )

        return TurnExecution(
            route="direct_chat",
            route_type=plan.route_type,
            assistant_final=final_message,
            payload={
                "route_type": plan.route_type,
                "packet": to_dict(packet),
                "activation_set": list(activation_plan.activation_set),
                "memory_tiers_read": list(activation_plan.memory_read_plan.tiers),
                "background_jobs": [to_dict(job) for job in activation_plan.background_jobs],
                "trace_ref": f"round://{round_id}",
            },
        )

    def _hot_state_payload(self, state: RuntimeState) -> dict[str, Any]:
        commitments = [
            str(item.summary or "").strip()
            for item in list(state.proactive_backlog or [])
            if str(item.status or "") not in {"done", "archived"}
        ]
        if state.current_goal:
            commitments.insert(0, f"继续围绕“{state.current_goal}”推进。")
        hot_meta = dict(state.session_metadata.get("chat_kernel_v2", {}) or {})
        hot_state = {
            "current_goal": str(state.current_goal or ""),
            "current_focus": str(state.focus or ""),
            "emotion_bias": str(hot_meta.get("emotion_bias") or ""),
            "todo": commitments[:5],
            "recent_commitments": list(hot_meta.get("recent_commitments", []) or [])[:5],
            "repair_status": str(hot_meta.get("repair_status") or state.repair_mode or ""),
            "mood": round(float(state.mood or 0.0), 4),
            "body_energy": round(float(state.body_energy or 0.0), 4),
            "affect_residue": round(float(state.affect_residue or 0.0), 4),
        }
        return hot_state

    def _packet_request(
        self,
        *,
        plan: TurnPlan,
        hot_state: dict[str, Any],
        cue: str | None,
        memory_context: dict[str, Any] | None = None,
    ) -> ModelRequest:
        payload = {
            "route_type": plan.route_type,
            "user_text": plan.text,
            "hot_state": hot_state,
            "cue": cue,
            "memory_context": memory_context or {},
        }
        return ModelRequest(
            system_prompt=(
                "You are ChatKernelV2. Produce one sparse cognitive packet for the current round. "
                "Keep semantics free, keep bookkeeping in code, and do not roleplay a fixed persona."
            ),
            user_prompt=self._json_prompt(payload),
            response_schema={
                "salience": "float",
                "uncertainty": "float",
                "memory_need": "bool",
                "tool_need": "bool",
                "conflict_need": "bool",
                "candidate_action_prior": "str",
                "draft_reply": "str",
                "proposed_state_patch": "dict",
                "deepen_reason": "str",
            },
            metadata={"route_type": plan.route_type, "cue": cue or "", "mode": plan.mode},
        )

    def _deep_deliberation_request(
        self,
        *,
        plan: TurnPlan,
        packet: CognitivePacket,
        hot_state: dict[str, Any],
        memory_context: dict[str, Any],
        cue: str | None,
    ) -> ModelRequest:
        return ModelRequest(
            system_prompt=(
                "You are the deep deliberation module for ChatKernelV2. "
                "Resolve conflict, planning depth, and re-alignment needs only when necessary."
            ),
            user_prompt=self._json_prompt(
                {
                    "route_type": plan.route_type,
                    "user_text": plan.text,
                    "packet": to_dict(packet),
                    "hot_state": hot_state,
                    "memory_context": memory_context,
                    "cue": cue,
                }
            ),
            response_schema={
                "draft_reply": "str",
                "deepen_reason": "str",
            },
            metadata={"route_type": plan.route_type, "cue": cue or "", "mode": plan.mode},
        )

    def _generate_packet(
        self,
        *,
        plan: TurnPlan,
        state: RuntimeState,
        hot_state: dict[str, Any],
        cue: str | None,
    ) -> tuple[CognitivePacket, list[dict[str, Any]], dict[str, Any]]:
        traces: list[dict[str, Any]] = []
        request = self._packet_request(plan=plan, hot_state=hot_state, cue=cue)
        try:
            response = self.model_router.generate("cognitive_packet", request)
            traces.append(self._model_call_trace("cognitive_packet", request, response))
            payload = dict(response.payload or {})
            raw_patch = dict(payload.get("proposed_state_patch", {}) or {})
            if "focus" not in raw_patch and raw_patch.get("current_focus"):
                raw_patch["focus"] = raw_patch.get("current_focus")
            if "obligations" not in raw_patch and raw_patch.get("recent_commitments"):
                raw_patch["obligations"] = list(raw_patch.get("recent_commitments", []) or [])
            raw_patch.pop("current_focus", None)
            raw_patch.pop("recent_commitments", None)
            payload["proposed_state_patch"] = raw_patch
            return CognitivePacket(**payload), traces, {"model": response.model}
        except Exception:
            fallback = CognitivePacket(
                salience=0.32 if plan.route_type == "chat_fast" else 0.58,
                uncertainty=0.18 if plan.route_type == "chat_fast" else 0.42,
                memory_need=plan.route_type in {"chat_standard", "chat_deep"} and bool(cue),
                tool_need=False,
                conflict_need=plan.route_type == "chat_deep",
                candidate_action_prior="respond",
                draft_reply="我先基于当前状态给你一个直接回应。",
                proposed_state_patch=StatePatch(focus="maintain_sparse_chat_response"),
                deepen_reason="",
            )
            return fallback, traces, {}

    def _run_deep_deliberation(
        self,
        *,
        plan: TurnPlan,
        packet: CognitivePacket,
        hot_state: dict[str, Any],
        memory_context: dict[str, Any],
        cue: str | None,
    ) -> tuple[str, str, list[dict[str, Any]]]:
        traces: list[dict[str, Any]] = []
        request = self._deep_deliberation_request(
            plan=plan,
            packet=packet,
            hot_state=hot_state,
            memory_context=memory_context,
            cue=cue,
        )
        try:
            response = self.model_router.generate("deliberation", request)
            traces.append(self._model_call_trace("deliberation", request, response))
            payload = dict(response.payload or {})
            message = str(payload.get("draft_reply") or packet.draft_reply or "").strip()
            reason = str(payload.get("deepen_reason") or packet.deepen_reason or "long_horizon_alignment").strip()
            return message or packet.draft_reply or "我会先把这些冲突拆开再回应你。", reason, traces
        except Exception:
            reason = packet.deepen_reason or "long_horizon_alignment"
            return packet.draft_reply or "我会先把这些冲突拆开再回应你。", reason, traces

    def _activation_plan(self, *, plan: TurnPlan, packet: CognitivePacket, cue: str | None) -> ActivationPlan:
        tiers = ["hot"]
        reasons: list[str] = []
        activation_set = list(self._BASE_MODULES)
        deepen_reason = ""

        if plan.route_type == "chat_standard":
            if cue and packet.memory_need:
                tiers.append("warm")
                activation_set.append("MemoryRecall")
                reasons.append("memory_cue_detected")
            if packet.tool_need:
                activation_set.append("ToolPlanner")
                reasons.append("tool_need_detected")
            if packet.conflict_need:
                activation_set.append("ConflictArbiter")
                reasons.append("conflict_need_detected")
        elif plan.route_type == "chat_deep":
            tiers = ["hot", "warm"]
            if cue:
                tiers.append("cold")
            activation_set.extend(["MemoryRecall", "ConflictArbiter", "LongPlanner", "Reflection/DMN", "DeepRenderer"])
            reasons.extend(["deep_route_requested"])
            if packet.tool_need:
                activation_set.append("ToolPlanner")
                reasons.append("tool_need_detected")
            deepen_reason = packet.deepen_reason or "long_horizon_alignment"

        if plan.route_type == "chat_fast":
            deepen_reason = ""
            reasons.append("default_sparse_round")
        elif plan.route_type == "chat_standard" and not reasons:
            reasons.append("conditional_upgrade")

        jobs = self._build_background_jobs(route_type=plan.route_type, cue=cue, packet=packet)
        return ActivationPlan(
            route_type=plan.route_type,
            activation_set=activation_set,
            activation_reason=reasons,
            memory_read_plan=MemoryReadPlan(tiers=tiers, cue=cue, reason="route_policy"),
            background_jobs=jobs,
            deepen_reason=deepen_reason,
        )

    def _memory_context(self, *, cue: str | None, read_plan: MemoryReadPlan) -> dict[str, Any]:
        if not cue:
            return {}
        budget = tuple(read_plan.tiers)
        recall = self.memory_store.recall(cue, tier_budget=budget)
        return {
            "cue": cue,
            "mode": recall.get("mode"),
            "strength": recall.get("strength"),
            "content": recall.get("content"),
            "tier": str(recall.get("tier") or ""),
            "search_trace": dict(recall.get("search_trace", {}) or {}),
        }

    def _build_background_jobs(self, *, route_type: str, cue: str | None, packet: CognitivePacket) -> list[ConsolidationJob]:
        jobs = [
            ConsolidationJob(
                kind="event_index",
                cue=cue,
                route_type=route_type,
                priority=max(0.2, packet.salience),
                payload={"memory_need": packet.memory_need},
            )
        ]
        if cue and route_type in {"chat_standard", "chat_deep"}:
            jobs.append(
                ConsolidationJob(
                    kind="memory_consolidation",
                    cue=cue,
                    route_type=route_type,
                    priority=max(packet.salience, 0.45),
                    payload={"topic": cue, "uncertainty": packet.uncertainty},
                )
            )
        if route_type == "chat_deep":
            jobs.append(
                ConsolidationJob(
                    kind="topic_clustering",
                    cue=cue,
                    route_type=route_type,
                    priority=max(packet.salience, 0.55),
                    payload={"deepen_reason": packet.deepen_reason},
                )
            )
        return jobs

    def _validate_state_patch(self, raw_patch: StatePatch | dict[str, Any] | None, *, cue: str | None) -> StatePatch:
        payload = dict(to_dict(raw_patch) if raw_patch is not None else {})
        obligations = list(payload.get("obligations", []) or payload.get("recent_commitments", []) or [])
        patch = StatePatch(
            focus=payload.get("focus") or payload.get("current_focus"),
            current_goal=payload.get("current_goal"),
            obligations=obligations,
            emotion_bias=payload.get("emotion_bias"),
            repair_status=payload.get("repair_status"),
            relation_delta=payload.get("relation_delta", 0.0),
            habit_delta=payload.get("habit_delta", 0.0),
            extras={
                key: value
                for key, value in payload.items()
                if key not in {"focus", "current_focus", "current_goal", "obligations", "recent_commitments", "emotion_bias", "repair_status", "relation_delta", "habit_delta"}
            },
        )
        if cue and patch.habit_delta == 0.0:
            patch.habit_delta = 0.03
        if patch.relation_delta == 0.0:
            patch.relation_delta = 0.01
        return patch

    def _apply_state_patch(
        self,
        state: RuntimeState,
        patch: StatePatch,
        *,
        event: RoundEvent,
        cue: str | None,
        route_type: str,
        salience: float,
    ) -> None:
        if patch.focus:
            state.focus = patch.focus
        if patch.current_goal:
            state.current_goal = patch.current_goal
        state.last_action = route_type
        state.budget_remaining = max(0.0, min(1.0, float(state.budget_remaining or 1.0) - 0.01))

        existing = [
            item if isinstance(item, ProactiveActionState) else ProactiveActionState(**item)
            for item in list(state.proactive_backlog or [])
        ]
        known = {item.summary for item in existing}
        for obligation in patch.obligations:
            if obligation in known:
                continue
            existing.insert(
                0,
                ProactiveActionState(
                    kind="chat_obligation",
                    summary=obligation,
                    priority=max(0.25, salience),
                    channel="planning",
                    status="pending",
                ),
            )
        state.proactive_backlog = existing[:10]
        state.session_metadata["chat_kernel_v2"] = {
            "emotion_bias": patch.emotion_bias or "",
            "repair_status": patch.repair_status or "",
            "recent_commitments": list(patch.obligations),
            "last_route_type": route_type,
        }
        if cue:
            self.memory_store.update_habit_strength(
                cue,
                patch.habit_delta,
                context_recurrence=min(1.0, 0.4 + salience * 0.4),
                round_gap=0,
                context_slot="chat_kernel_v2",
                round_id=state.round_count,
            )
        if event.target:
            self.memory_store.nudge_relation(event.target, patch.relation_delta)

    def _packet_summary(self, packet: CognitivePacket) -> dict[str, Any]:
        return {
            "salience": packet.salience,
            "uncertainty": packet.uncertainty,
            "memory_need": packet.memory_need,
            "tool_need": packet.tool_need,
            "conflict_need": packet.conflict_need,
            "candidate_action_prior": packet.candidate_action_prior,
        }

    def _write_round_trace(
        self,
        *,
        state: RuntimeState,
        plan: TurnPlan,
        event: RoundEvent,
        packet: CognitivePacket,
        activation_plan: ActivationPlan,
        diagnostics: RoundDiagnosticsV2,
        model_call_traces: list[dict[str, Any]],
        rendered_expression: RenderedExpression,
        total_turn_ms: int,
    ) -> RoundTrace:
        recorded_at = utc_now_iso()
        round_id = int(state.round_count or 0)
        total_model_wait_ms = sum(int(item.get("latency_ms", 0) or 0) for item in model_call_traces)
        trace = RoundTrace(
            session_id=state.session_id,
            recorded_at=recorded_at,
            recorded_date=iso_date(recorded_at),
            round_id=round_id,
            subject_id=state.identity_state.internal_handle,
            continuity_nonce=state.identity_state.internal_handle,
            scenario=plan.scenario,
            mode=plan.mode,
            sampled_action=packet.candidate_action_prior,
            contributions=[
                AgentContribution(
                    agent_name="PacketAssembler",
                    action_name=packet.candidate_action_prior,
                    score=max(packet.salience, 0.01),
                    reason=f"route={plan.route_type}",
                )
            ],
            top_drivers=[
                AgentContribution(
                    agent_name="PacketAssembler",
                    action_name=packet.candidate_action_prior,
                    score=max(packet.salience, 0.01),
                    reason=f"uncertainty={packet.uncertainty:.2f}",
                )
            ],
            style_profile={},
            state_snapshot=self._state_snapshot_payload(state),
            render_plan={"route_type": plan.route_type},
            rendered_expression=to_dict(rendered_expression),
            event_log_ref={"round_id": round_id, "cue": _derive_cue(event)},
            route_budget_ms=diagnostics.route_budget_ms,
            activation_set=list(diagnostics.activation_set),
            activation_reason=list(diagnostics.activation_reason),
            memory_tiers_read=list(diagnostics.memory_tiers_read),
            packet_summary=dict(diagnostics.packet_summary),
            background_jobs=list(diagnostics.background_jobs),
            deepen_reason=diagnostics.deepen_reason,
            model_call_traces=list(model_call_traces),
            runtime_metrics={
                "route_type": diagnostics.route_type,
                "route_budget_ms": diagnostics.route_budget_ms,
                "activation_set": list(diagnostics.activation_set),
                "activation_reason": list(diagnostics.activation_reason),
                "memory_tiers_read": list(diagnostics.memory_tiers_read),
                "packet_summary": dict(diagnostics.packet_summary),
                "background_jobs": list(diagnostics.background_jobs),
                "deepen_reason": diagnostics.deepen_reason,
                "model_call_count": diagnostics.model_call_count,
                "parallel_task_count": 0,
                "parallel_groups": [],
                "optional_timeout_count": 0,
                "speculative_drop_count": 0,
                "total_model_wait_ms": total_model_wait_ms,
                "total_turn_ms": total_turn_ms,
                "local_compute_ms": max(1, total_turn_ms - total_model_wait_ms),
            },
        )
        self.trace_store.write_round(trace)
        return trace

    def _route_budget_ms(self, route_type: str) -> int:
        policies = self._route_policy_contract()
        policy = dict(policies.get(route_type, {}) or {})
        return max(0, int(policy.get("latency_budget_ms", 0) or 0))

    def _model_call_trace(self, route_name: str, request: ModelRequest, response: Any) -> dict[str, Any]:
        return {
            "route": route_name,
            "backend": getattr(response, "backend", ""),
            "model": getattr(response, "model", ""),
            "latency_ms": int(getattr(response, "latency_ms", 0) or 0),
            "prompt_chars": len(request.system_prompt) + len(request.user_prompt),
            "response_keys": sorted(dict(getattr(response, "payload", {}) or {}).keys()),
        }

    def _state_snapshot_payload(self, state: RuntimeState) -> dict[str, Any]:
        payload = {
            "round_count": int(state.round_count or 0),
            "mode": str(state.mode or ""),
            "focus": str(state.focus or ""),
            "current_goal": str(state.current_goal or ""),
            "mood": round(float(state.mood or 0.0), 4),
            "body_energy": round(float(state.body_energy or 0.0), 4),
            "budget_remaining": round(float(state.budget_remaining or 0.0), 4),
            "chat_kernel_v2": dict(state.session_metadata.get("chat_kernel_v2", {}) or {}),
        }
        return payload
