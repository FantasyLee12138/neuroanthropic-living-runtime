from __future__ import annotations

from typing import TYPE_CHECKING, Any

from nalr.providers import MissingModelCredentialError, ModelRequest, ModelRouteConfig

if TYPE_CHECKING:
    from nalr.runtime.controller import RuntimeController


class _ControllerBackedRuntime:
    def __init__(self, controller: "RuntimeController") -> None:
        self.controller = controller

    def __getattr__(self, name: str):
        return getattr(self.controller, name)


class ModelRoutingRuntime(_ControllerBackedRuntime):
    _DEFAULT_AGENT_MODEL_BINDINGS = {
        "planner": "medium_model",
        "PFCAgent": "medium_model",
        "InitiativeInteractionAgent": "state_machine",
        "PerspectiveModel": "medium_model",
        "Renderer": "medium_model",
        "MonologueStream": "small_model",
        "AutonomySelfRun": "small_model",
        "SalienceAgent": "small_model",
        "ValueAgent": "small_model",
        "BodyStateAgent": "state_machine",
        "RelationshipAgent": "state_machine",
        "DesireAgent": "state_machine",
        "EmotionAgent": "state_machine",
        "DMNAgent": "state_machine",
        "UnconsciousAgent": "state_machine",
        "HippocampusAgent": "state_machine",
        "ThalamusAttentionAgent": "state_machine",
        "HabitAgent": "state_machine",
        "ConflictMonitorAgent": "state_machine",
        "ResourceAgent": "state_machine",
        "CerebellarPredictor": "state_machine",
        "BehaviorPlausibilityGuard": "state_machine",
        "ForcedModeSwitch": "state_machine",
        "OutputGate": "state_machine",
    }

    def _provider_descriptor_for_route(self, route_name: str = "renderer") -> tuple[str, str]:
        route_cfg = self.config["models"]["model_routes"].get(route_name, {})
        binding_key = {
            "planner": "planner",
            "pfc": "PFCAgent",
            "perspective": "PerspectiveModel",
            "renderer": "Renderer",
            "autonomy_self_run": "AutonomySelfRun",
        }.get(route_name)
        if binding_key:
            resolved_route = self._route_config_for_binding(binding_key, route_name=route_name)
            if resolved_route is not None:
                route_cfg = {
                    "backend": resolved_route.backend,
                    "model": resolved_route.model,
                }
        backend = str(route_cfg.get("backend", "model")).lower()
        if backend == "doubao":
            provider_label = "Doubao/Ark route"
        elif backend == "deepseek":
            provider_label = "DeepSeek route"
        else:
            provider_label = f"{backend.title()} route"
        return provider_label, str(route_cfg.get("model", "")).strip()

    def _provider_descriptor(self) -> tuple[str, str]:
        return self._provider_descriptor_for_route("renderer")

    def _model_tiers(self) -> dict[str, Any]:
        return dict(self.config["models"].get("model_tiers", {}))

    def _agent_model_bindings(self) -> dict[str, str]:
        bindings = self.config["models"].get("agent_model_bindings", {})
        normalized = dict(self._DEFAULT_AGENT_MODEL_BINDINGS)
        normalized.update({str(key): str(value) for key, value in dict(bindings).items()})
        return normalized

    def _module_model_bindings(self) -> dict[str, str]:
        bindings = self.config["models"].get("module_model_bindings", {})
        defaults = {
            "cognitive_packet": "medium_model",
            "deliberation": "large_model",
            "tool_planner": "medium_model",
            "deep_renderer": "large_model",
            "consolidation_summarizer": "small_model",
        }
        normalized = {str(key): str(value) for key, value in dict(defaults).items()}
        normalized.update({str(key): str(value) for key, value in dict(bindings).items()})
        return normalized

    def _module_tier(self, module_name: str) -> str:
        return self._module_model_bindings().get(module_name, "state_machine")

    def _agent_tier(self, binding_key: str) -> str:
        bindings = self._agent_model_bindings()
        return bindings.get(binding_key, "state_machine")

    def _effective_agent_tier(self, binding_key: str, *, metadata: dict[str, Any] | None = None) -> str:
        base_tier = self._agent_tier(binding_key)
        info = dict(metadata or {})
        if base_tier == "state_machine":
            return base_tier
        if binding_key == "PFCAgent":
            escalate = (
                float(info.get("relation_risk", 0.0) or 0.0) >= 0.62
                or float(info.get("disclosure_sensitivity", 0.0) or 0.0) >= 0.55
                or float(info.get("authenticity_risk", 0.0) or 0.0) >= 0.32
                or float(info.get("conflict_score", 0.0) or 0.0) >= self.config["thresholds"]["thresholds"]["conflict_high"]
                or int(info.get("resample_count", 0) or 0) >= 2
            )
            return "large_model" if escalate else base_tier
        if binding_key == "PerspectiveModel":
            escalate = (
                float(info.get("relation_risk", 0.0) or 0.0) >= 0.72
                or float(info.get("disclosure_sensitivity", 0.0) or 0.0) >= 0.72
            )
            return "large_model" if escalate else base_tier
        if binding_key == "Renderer":
            route_type = str(info.get("route_type", "") or "").strip()
            query_kind = str(info.get("query_kind", "") or "").strip()
            escalate = (
                route_type in {"chat_deep", "task_run", "endogenous_deep"}
                or query_kind in {"self_identity", "provider_identity", "answer_explanation"}
                or float(info.get("relation_risk", 0.0) or 0.0) >= 0.72
                or bool(info.get("conflict_hot"))
            )
            return "large_model" if escalate else base_tier
        return base_tier

    def _tier_config(self, tier_name: str) -> dict[str, Any]:
        return dict(self._model_tiers().get(tier_name, {}))

    def _infer_tier_name_for_route(self, route_cfg: ModelRouteConfig | None) -> str | None:
        if route_cfg is None:
            return None
        for tier_name, tier_cfg in self._model_tiers().items():
            if str(tier_cfg.get("mode", "local")).lower() == "local":
                continue
            if (
                str(tier_cfg.get("backend", "")).strip() == str(route_cfg.backend or "").strip()
                and str(tier_cfg.get("model", "")).strip() == str(route_cfg.model or "").strip()
                and str(tier_cfg.get("api_key_env", "")).strip() == str(route_cfg.api_key_env or "").strip()
            ):
                return tier_name
        return None

    def _route_policy_contract(self) -> dict[str, Any]:
        route_cfg = self.config["models"].get("route_policies", {})
        contract = {
            "chat_micro": {
                "latency_budget_ms": 250,
                "entry_mode": "interactive",
                "decision_mode": "micro_shortcut",
                "default_binding": "cognitive_packet",
                "default_tier": self._module_tier("cognitive_packet"),
                "always_on_modules": ["Router", "HotStateLoader", "Responder"],
                "conditional_modules": [],
                "upgrade_routes": ["chat_fast"],
                "upgrade_conditions": [],
            },
            "chat_fast": {
                "latency_budget_ms": 700,
                "entry_mode": "interactive",
                "decision_mode": "single_packet",
                "default_binding": "cognitive_packet",
                "default_tier": self._module_tier("cognitive_packet"),
                "always_on_modules": ["Router", "HotStateLoader", "BudgetAllocator", "PacketAssembler", "SafetyGate", "StateWriter"],
                "conditional_modules": [],
                "upgrade_routes": ["chat_standard"],
                "upgrade_conditions": [],
            },
            "chat_standard": {
                "latency_budget_ms": 1200,
                "entry_mode": "interactive",
                "decision_mode": "packet_plus_conditional_modules",
                "default_binding": "cognitive_packet",
                "default_tier": self._module_tier("cognitive_packet"),
                "always_on_modules": ["Router", "HotStateLoader", "BudgetAllocator", "PacketAssembler", "SafetyGate", "StateWriter"],
                "conditional_modules": ["MemoryRecall", "ConflictArbiter", "ToolPlanner", "HabitController"],
                "upgrade_routes": ["chat_deep"],
                "upgrade_conditions": [
                    "memory_cue_detected",
                    "conflict_detected",
                    "tool_need_detected",
                    "salience_high",
                    "uncertainty_high",
                ],
            },
            "chat_deep": {
                "latency_budget_ms": 2200,
                "entry_mode": "interactive",
                "decision_mode": "deliberation_pipeline",
                "default_binding": "deliberation",
                "default_tier": self._module_tier("deliberation"),
                "always_on_modules": ["Router", "HotStateLoader", "BudgetAllocator", "PacketAssembler", "SafetyGate", "StateWriter"],
                "conditional_modules": ["MemoryRecall", "ConflictArbiter", "LongPlanner", "RecoveryAligner", "Reflection/DMN", "DeepRenderer", "ToolPlanner"],
                "upgrade_routes": [],
                "upgrade_conditions": [
                    "multi_objective_conflict",
                    "high_risk_action",
                    "self_correction",
                    "long_horizon_plan",
                    "recovery_alignment",
                ],
            },
            "task_run": {
                "latency_budget_ms": 1200,
                "entry_mode": "interactive",
                "decision_mode": "supervisor_run",
                "default_binding": "planner",
                "default_tier": self._agent_tier("planner"),
                "always_on_modules": ["RunSupervisor"],
                "conditional_modules": ["ToolPlanner", "RecoveryAligner"],
                "upgrade_routes": [],
                "upgrade_conditions": ["planner_complexity_high", "relation_risk_high"],
            },
            "endogenous_light": {
                "latency_budget_ms": 300,
                "entry_mode": "idle",
                "decision_mode": "endogenous_tick",
                "default_binding": "monologue_stream",
                "default_tier": self._agent_tier("MonologueStream"),
                "always_on_modules": ["Router", "HotStateLoader", "Reflection/DMN"],
                "conditional_modules": ["HabitController"],
                "upgrade_routes": [],
                "upgrade_conditions": [],
            },
            "endogenous_deep": {
                "latency_budget_ms": 300,
                "entry_mode": "idle",
                "decision_mode": "endogenous_tick",
                "default_binding": "deliberation",
                "default_tier": "medium_model",
                "always_on_modules": ["Router", "HotStateLoader", "Reflection/DMN"],
                "conditional_modules": ["MemoryRecall", "ConflictArbiter", "LongPlanner"],
                "upgrade_routes": [],
                "upgrade_conditions": ["endogenous_replay", "endogenous_regulation", "relation_risk_high"],
            },
            "dream_sleep": {
                "latency_budget_ms": 5000,
                "entry_mode": "sleep",
                "decision_mode": "dream_sidecar",
                "default_binding": "consolidation_summarizer",
                "default_tier": "medium_model",
                "always_on_modules": ["ConsolidationSubstrate"],
                "conditional_modules": ["MemoryRecall", "Reflection/DMN"],
                "upgrade_routes": [],
                "upgrade_conditions": ["sleep_mode_only"],
            },
        }
        for route_name, overrides in dict(route_cfg).items():
            if route_name in contract and isinstance(overrides, dict):
                contract[route_name] = {**contract[route_name], **dict(overrides)}
        return contract

    def _activation_thresholds_contract(self) -> dict[str, Any]:
        payload = dict(self.config["models"].get("activation_thresholds", {}) or {})
        return {
            "salience_high": float(payload.get("salience_high", 0.72) or 0.72),
            "uncertainty_high": float(payload.get("uncertainty_high", 0.58) or 0.58),
            "conflict_high": float(payload.get("conflict_high", 0.62) or 0.62),
            "tool_need_high": float(payload.get("tool_need_high", 0.5) or 0.5),
        }

    def _memory_tier_policy_contract(self) -> dict[str, Any]:
        payload = dict(self.config["models"].get("memory_tier_policy", {}) or {})
        return {
            "chat_fast": list(payload.get("chat_fast", ["hot"]) or ["hot"]),
            "chat_standard_default": list(payload.get("chat_standard_default", ["hot"]) or ["hot"]),
            "chat_standard_on_cue": list(payload.get("chat_standard_on_cue", ["hot", "warm"]) or ["hot", "warm"]),
            "chat_deep": list(payload.get("chat_deep", ["hot", "warm", "cold"]) or ["hot", "warm", "cold"]),
        }

    def _consolidation_policy_contract(self) -> dict[str, Any]:
        payload = dict(self.config["models"].get("consolidation_policy", {}) or {})
        return {
            "enabled": bool(payload.get("enabled", True)),
            "queue": str(payload.get("queue", "dream_sidecar") or "dream_sidecar"),
            "write_inline": list(payload.get("write_inline", ["state_patch", "obligations", "habit_delta", "relation_delta", "event_index"]) or []),
            "write_deferred": list(payload.get("write_deferred", ["long_term_summarization", "memory_consolidation", "topic_clustering", "compaction", "replay"]) or []),
        }

    def _route_config_for_binding(
        self,
        binding_key: str,
        *,
        route_name: str,
        metadata: dict[str, Any] | None = None,
    ) -> ModelRouteConfig | None:
        tier_name = self._effective_agent_tier(binding_key, metadata=metadata)
        tier_cfg = self._tier_config(tier_name)
        if not tier_cfg:
            route_cfg = self.model_router.route_configs.get(route_name)
            if route_cfg is None:
                return None
            fallback_route = ModelRouteConfig(
                name=route_cfg.name,
                backend=route_cfg.backend,
                model=route_cfg.model,
                timeout_ms=route_cfg.timeout_ms,
                retries=route_cfg.retries,
                enabled=route_cfg.enabled,
                base_url=route_cfg.base_url,
                api_key_env=route_cfg.api_key_env,
            )
            fallback_route = self.model_router.effective_route_config(fallback_route)
            setattr(fallback_route, "effective_tier", tier_name)
            setattr(fallback_route, "effective_mode", "route_fallback")
            return fallback_route
        mode = str(tier_cfg.get("mode", "local")).lower()
        if mode == "local":
            return None
        route_config = ModelRouteConfig(
            name=route_name,
            backend=str(tier_cfg.get("backend", "")),
            model=str(tier_cfg.get("model", "")),
            timeout_ms=int(tier_cfg.get("timeout_ms", 12000)),
            retries=int(tier_cfg.get("retries", 0)),
            enabled=bool(tier_cfg.get("enabled", True)),
            base_url=str(tier_cfg.get("base_url", self.config["models"]["models"].get("base_url", ""))),
            api_key_env=str(tier_cfg.get("api_key_env", "")).strip() or None,
        )
        setattr(route_config, "effective_tier", tier_name)
        setattr(route_config, "effective_mode", mode)
        return route_config

    def _record_model_call(
        self,
        bucket: list[dict[str, Any]],
        *,
        skill_name: str,
        binding_key: str,
        route_config: ModelRouteConfig,
        response,
        prompt_chars: int,
        parallel_group: str | None = None,
    ) -> None:
        usage = dict(getattr(response, "usage", {}) or {})
        bucket.append(
            {
                "skill_name": skill_name,
                "binding_key": binding_key,
                "agent_tier": getattr(route_config, "effective_tier", self._agent_tier(binding_key)),
                "route": response.route,
                "backend": getattr(response, "backend", route_config.backend),
                "model": response.model,
                "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
                "prompt_chars": prompt_chars,
                "latency_ms": int(getattr(response, "latency_ms", 0) or 0),
                "parallel_group": parallel_group,
            }
        )

    def _call_bound_model_route(
        self,
        binding_key: str,
        *,
        route_name: str,
        request: ModelRequest,
        binding_metadata: dict[str, Any] | None = None,
        model_call_traces: list[dict[str, Any]] | None = None,
        skill_name: str | None = None,
        parallel_group: str | None = None,
    ):
        route_config = self._route_config_for_binding(binding_key, route_name=route_name, metadata=binding_metadata)
        generate_override = getattr(getattr(self, "model_router", None), "__dict__", {}).get("generate")
        generate_config_override = getattr(getattr(self, "model_router", None), "__dict__", {}).get("generate_config")
        if route_config is not None and callable(generate_config_override):
            response = generate_config_override(route_config, request)
        elif callable(generate_override):
            response = generate_override(route_name, request)
        elif route_config is None:
            response = self.model_router.generate(route_name, request)
        else:
            try:
                response = self.model_router.generate_config(route_config, request)
            except MissingModelCredentialError:
                response = self.model_router.generate(route_name, request)
        if model_call_traces is not None:
            self._record_model_call(
                model_call_traces,
                skill_name=skill_name or route_name,
                binding_key=binding_key,
                route_config=route_config,
                response=response,
                prompt_chars=len(request.system_prompt) + len(request.user_prompt),
                parallel_group=parallel_group,
            )
        return response
