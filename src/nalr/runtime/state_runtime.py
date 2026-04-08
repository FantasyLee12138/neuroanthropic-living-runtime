from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from nalr.schemas.models import to_dict

if TYPE_CHECKING:
    from nalr.runtime.controller import RuntimeController


class StateRuntimeService:
    def __init__(self, controller: "RuntimeController") -> None:
        self.controller = controller

    def update_observer_settings(self, payload: dict[str, Any]) -> dict[str, Any]:
        controller = self.controller
        normalized = controller.normalize_observer_settings_payload(payload, base=controller.observer_settings_current())
        controller.observer_settings_path.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")
        controller.refresh_runtime_components_from_config()
        state = controller.load_runtime_state()
        controller.apply_startup_unlock_preferences(state)
        if getattr(state.autonomy_loop, "profile", ""):
            previous_enabled = bool(getattr(state.autonomy_policy, "enabled", False))
            state.autonomy_policy = controller._autonomy_policy_for_profile(state.autonomy_loop.profile or state.autonomy_policy.profile)
            state.autonomy_policy.enabled = previous_enabled
        controller.save_runtime_state(state, sync=True)
        return controller.observer_settings_payload()

    def state_payload(self) -> dict[str, Any]:
        controller = self.controller
        controller.flush_pending_io(raise_on_error=False)
        state = controller.load_runtime_state()
        controller.sync_tlh_state(state)
        controller.sync_autonomy_state(state)
        payload = to_dict(state)
        payload["initiative"] = controller.initiative_status_from_state(state)
        payload["subjectivity"] = controller.subjectivity_metrics()
        payload["trace_storage"] = controller.trace_storage_status()
        payload["memory_storage"] = controller.memory_store.storage_status()
        payload["runtime_storage"] = controller.runtime_storage_status()
        payload["migration"] = controller.runtime_migration_report()
        payload["entropy"] = controller.entropy_pool.health_snapshot()
        payload["dream"] = controller.dream_status()
        payload["cognitive_snapshot"] = controller.cognitive_snapshot(state=state)
        payload["runtime_metrics"] = controller.latest_runtime_metrics()
        payload["performance"] = controller.runtime_performance_payload()
        return payload
