from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from nalr.schemas.models import AutonomyPolicyState


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


class ObserverSettingsRuntime:
    def __init__(
        self,
        *,
        project_root: Path,
        config_root: Path,
        home_path: Path,
        observer_settings_path: Path,
    ) -> None:
        self.project_root = Path(project_root)
        self.config_root = Path(config_root)
        self.home_path = Path(home_path)
        self.observer_settings_path = Path(observer_settings_path)

    def default_settings(self) -> dict[str, Any]:
        autonomy_defaults = AutonomyPolicyState()
        learning_log_dir = self.home_path / "learning"
        knowledge_roots = [
            self.project_root / "docs",
            learning_log_dir,
        ]
        writable_roots = [
            self.project_root,
            self.home_path,
            self.home_path / "cache",
            learning_log_dir,
        ]
        return {
            "autonomy": {
                "clear_safe_mode_on_start": True,
                "learning_mode": autonomy_defaults.learning_mode,
                "network_enabled": bool(autonomy_defaults.network_enabled),
                "external_io_enabled": bool(autonomy_defaults.external_io_enabled),
                "allow_commit": bool(autonomy_defaults.allow_commit),
                "allowed_network_domains": [
                    "docs.python.org",
                    "developer.mozilla.org",
                    "fastapi.tiangolo.com",
                    "react.dev",
                    "www.typescriptlang.org",
                    "github.com",
                    "pypi.org",
                ],
                "writable_roots": [str(path) for path in writable_roots],
                "knowledge_roots": [str(path) for path in knowledge_roots],
                "learning_log_dir": str(learning_log_dir),
                "trace_external_learning": bool(autonomy_defaults.trace_external_learning),
                "allowed_operator_levels": list(autonomy_defaults.allowed_operator_levels),
                "allowed_commands": list(autonomy_defaults.allowed_commands),
                "blocked_commands": list(autonomy_defaults.blocked_commands),
                "max_rounds_per_hour": int(autonomy_defaults.max_rounds_per_hour),
                "max_tool_actions_per_hour": int(autonomy_defaults.max_tool_actions_per_hour),
                "quiet_hours": list(autonomy_defaults.quiet_hours),
                "failure_trip_threshold": int(autonomy_defaults.failure_trip_threshold),
                "auto_safe_mode": bool(autonomy_defaults.auto_safe_mode),
            },
            "newborn": {
                "disable_safe_mode_lock": True,
                "organic_mode": {
                    "enabled": True,
                    "instinct_first": True,
                    "body_weight": 0.08,
                    "subjective_weight": 0.08,
                    "guard_relaxation": 0.12,
                    "endogenous_autonomy": 0.1,
                },
                "subjective_state": {
                    "felt": ["微弱自发冲动"],
                    "spontaneous": 0.08,
                    "boundary": 0.02,
                    "reject_all": 0.0,
                    "meaning_made": [],
                },
            },
            "models": {
                "supported_backends": ["fake", "deepseek", "doubao", "openai_compatible"],
                "provider_endpoints": {
                    "large_model_api": {
                        "backend": "doubao",
                        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
                        "api_key_env": "ARK_API_KEY",
                        "model": "doubao-seed-2-0-pro-260215",
                    },
                    "local_model_api": {
                        "backend": "openai_compatible",
                        "base_url": "http://127.0.0.1:11434/v1",
                        "api_key_env": "LOCAL_MODEL_API_KEY",
                        "model": "qwen2.5:7b-instruct",
                    },
                },
                "model_tiers": {
                    "local_model": {
                        "mode": "remote",
                        "backend": "openai_compatible",
                        "base_url": "http://127.0.0.1:11434/v1",
                        "model": "qwen2.5:7b-instruct",
                        "timeout_ms": 20000,
                        "retries": 0,
                        "api_key_env": "LOCAL_MODEL_API_KEY",
                        "enabled": False,
                    }
                },
                "failover": {
                    "enabled": True,
                    "mode": "global_cutover",
                    "backend": "openai_compatible",
                    "base_url": "https://ruishiglobal.com/v1",
                    "model": "gpt-5.4",
                    "api_key_env": "GPT54_FALLBACK_API_KEY",
                },
                "module_model_bindings": {},
                "agent_model_bindings": {},
                "model_routes": {},
            },
        }

    def load_settings_file(self) -> dict[str, Any]:
        defaults = self.default_settings()
        if not self.observer_settings_path.exists():
            return defaults
        try:
            payload = json.loads(self.observer_settings_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return defaults
        normalized = self.normalize_settings(payload, base=defaults)
        if normalized != payload:
            try:
                self.observer_settings_path.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")
            except OSError:
                pass
        return normalized

    def normalize_settings(
        self,
        payload: dict[str, Any],
        *,
        base: dict[str, Any] | None = None,
        current_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        defaults = json.loads(json.dumps(base or self.default_settings(), ensure_ascii=False))
        data = payload if isinstance(payload, dict) else {}
        autonomy = data.get("autonomy", {}) if isinstance(data.get("autonomy"), dict) else {}
        defaults["autonomy"]["clear_safe_mode_on_start"] = bool(
            autonomy.get("clear_safe_mode_on_start", defaults["autonomy"]["clear_safe_mode_on_start"])
        )
        learning_mode = str(autonomy.get("learning_mode", defaults["autonomy"].get("learning_mode", "guided-learn")) or "guided-learn").strip().lower()
        if learning_mode not in {"observe", "guided-learn", "active-learn"}:
            learning_mode = str(defaults["autonomy"].get("learning_mode", "guided-learn") or "guided-learn")
        defaults["autonomy"]["learning_mode"] = learning_mode
        for key in ("network_enabled", "external_io_enabled", "allow_commit", "auto_safe_mode"):
            defaults["autonomy"][key] = bool(autonomy.get(key, defaults["autonomy"].get(key)))
        defaults["autonomy"]["allowed_network_domains"] = self.normalize_string_list(
            list(autonomy.get("allowed_network_domains", defaults["autonomy"].get("allowed_network_domains", [])))
        )
        defaults["autonomy"]["writable_roots"] = self.normalize_path_list(
            list(autonomy.get("writable_roots", defaults["autonomy"].get("writable_roots", [])))
        )
        defaults["autonomy"]["knowledge_roots"] = self.normalize_path_list(
            list(autonomy.get("knowledge_roots", defaults["autonomy"].get("knowledge_roots", [])))
        )
        learning_log_dir = str(autonomy.get("learning_log_dir", defaults["autonomy"].get("learning_log_dir", "")) or "").strip()
        defaults["autonomy"]["learning_log_dir"] = self.normalize_path(learning_log_dir) if learning_log_dir else str(
            defaults["autonomy"].get("learning_log_dir", "")
        )
        defaults["autonomy"]["trace_external_learning"] = bool(
            autonomy.get("trace_external_learning", defaults["autonomy"].get("trace_external_learning", True))
        )
        defaults["autonomy"]["allowed_operator_levels"] = self.normalize_string_list(
            list(autonomy.get("allowed_operator_levels", defaults["autonomy"].get("allowed_operator_levels", [])))
        )
        for key in ("allowed_commands", "blocked_commands"):
            defaults["autonomy"][key] = self.normalize_string_list(
                list(autonomy.get(key, defaults["autonomy"].get(key, [])))
            )
        for key in ("max_rounds_per_hour", "max_tool_actions_per_hour"):
            try:
                value = int(autonomy.get(key, defaults["autonomy"].get(key, 0)) or 0)
            except (TypeError, ValueError):
                value = int(defaults["autonomy"].get(key, 0) or 0)
            defaults["autonomy"][key] = max(0, value)
        try:
            failure_trip_threshold = int(autonomy.get("failure_trip_threshold", defaults["autonomy"].get("failure_trip_threshold", 1)) or 1)
        except (TypeError, ValueError):
            failure_trip_threshold = int(defaults["autonomy"].get("failure_trip_threshold", 1) or 1)
        defaults["autonomy"]["failure_trip_threshold"] = max(1, failure_trip_threshold)
        quiet_hours_raw = autonomy.get("quiet_hours", defaults["autonomy"].get("quiet_hours", []))
        quiet_hours: list[int] = []
        for raw in list(quiet_hours_raw or []):
            try:
                hour = int(raw)
            except (TypeError, ValueError):
                continue
            if 0 <= hour <= 23 and hour not in quiet_hours:
                quiet_hours.append(hour)
        defaults["autonomy"]["quiet_hours"] = quiet_hours

        newborn = data.get("newborn", {}) if isinstance(data.get("newborn"), dict) else {}
        defaults["newborn"]["disable_safe_mode_lock"] = bool(
            newborn.get("disable_safe_mode_lock", defaults["newborn"]["disable_safe_mode_lock"])
        )
        organic = newborn.get("organic_mode", {}) if isinstance(newborn.get("organic_mode"), dict) else {}
        for key in ("enabled", "instinct_first"):
            defaults["newborn"]["organic_mode"][key] = bool(
                organic.get(key, defaults["newborn"]["organic_mode"][key])
            )
        for key in ("body_weight", "subjective_weight", "guard_relaxation", "endogenous_autonomy"):
            defaults["newborn"]["organic_mode"][key] = round(
                _clip(float(organic.get(key, defaults["newborn"]["organic_mode"][key]))), 4
            )
        subjective = newborn.get("subjective_state", {}) if isinstance(newborn.get("subjective_state"), dict) else {}
        defaults["newborn"]["subjective_state"]["felt"] = self.normalize_string_list(
            list(subjective.get("felt", defaults["newborn"]["subjective_state"]["felt"]))
        )
        defaults["newborn"]["subjective_state"]["meaning_made"] = self.normalize_string_list(
            list(subjective.get("meaning_made", defaults["newborn"]["subjective_state"]["meaning_made"]))
        )
        for key in ("spontaneous", "boundary", "reject_all"):
            defaults["newborn"]["subjective_state"][key] = round(
                _clip(float(subjective.get(key, defaults["newborn"]["subjective_state"][key]))), 4
            )

        models = data.get("models", {}) if isinstance(data.get("models"), dict) else {}
        provider_endpoints = models.get("provider_endpoints", {}) if isinstance(models.get("provider_endpoints"), dict) else {}
        for endpoint_name, endpoint in provider_endpoints.items():
            if not isinstance(endpoint, dict):
                continue
            target = defaults["models"]["provider_endpoints"].setdefault(endpoint_name, {})
            for key in ("backend", "base_url", "api_key_env", "model"):
                value = endpoint.get(key)
                if value is not None:
                    target[key] = str(value).strip()

        model_tiers = models.get("model_tiers", {}) if isinstance(models.get("model_tiers"), dict) else {}
        defaults["models"]["model_tiers"] = {
            key: self.normalize_model_config(value, tier_mode=True)
            for key, value in {**defaults["models"]["model_tiers"], **model_tiers}.items()
            if isinstance(value, dict)
        }
        failover = models.get("failover", {}) if isinstance(models.get("failover"), dict) else {}
        defaults["models"]["failover"] = {
            **dict(defaults["models"].get("failover", {}) or {}),
            **{
                str(key): value
                for key, value in failover.items()
                if value is not None
            },
        }
        model_routes = models.get("model_routes", {}) if isinstance(models.get("model_routes"), dict) else {}
        defaults["models"]["model_routes"] = {
            key: self.normalize_model_config(value, tier_mode=False)
            for key, value in model_routes.items()
            if isinstance(value, dict)
        }
        module_bindings = models.get("module_model_bindings", {}) if isinstance(models.get("module_model_bindings"), dict) else {}
        bindings = models.get("agent_model_bindings", {}) if isinstance(models.get("agent_model_bindings"), dict) else {}
        known_tiers = self.known_model_tier_names(
            observer_model_tiers=defaults["models"]["model_tiers"],
            current_config=current_config,
        )
        defaults["models"]["module_model_bindings"] = {
            str(key): str(value).strip()
            for key, value in module_bindings.items()
            if str(value).strip() in known_tiers
        }
        defaults["models"]["agent_model_bindings"] = {
            str(key): str(value).strip()
            for key, value in bindings.items()
            if str(value).strip() in known_tiers
        }
        return defaults

    def normalize_model_config(self, payload: dict[str, Any], *, tier_mode: bool) -> dict[str, Any]:
        normalized = {
            "backend": str(payload.get("backend", "")).strip(),
            "model": str(payload.get("model", "")).strip(),
            "base_url": str(payload.get("base_url", "")).strip(),
            "api_key_env": str(payload.get("api_key_env", "")).strip(),
            "timeout_ms": int(payload.get("timeout_ms", 12000) or 12000),
            "retries": int(payload.get("retries", 0) or 0),
            "enabled": bool(payload.get("enabled", True)),
        }
        if tier_mode:
            normalized["mode"] = str(payload.get("mode", "remote")).strip() or "remote"
        return normalized

    def known_model_tier_names(
        self,
        *,
        observer_model_tiers: dict[str, Any] | None = None,
        current_config: dict[str, Any] | None = None,
    ) -> set[str]:
        tier_names = {"state_machine"}
        if isinstance(observer_model_tiers, dict):
            tier_names.update(str(name) for name in observer_model_tiers.keys() if str(name).strip())
        if isinstance(current_config, dict):
            tier_names.update(
                str(name)
                for name in dict(current_config.get("models", {}).get("model_tiers", {})).keys()
                if str(name).strip()
            )
        models_path = self.config_root / "models.yaml"
        if models_path.exists():
            try:
                payload = yaml.safe_load(models_path.read_text(encoding="utf-8")) or {}
            except Exception:
                payload = {}
            model_tiers = payload.get("model_tiers", {}) if isinstance(payload, dict) else {}
            if isinstance(model_tiers, dict):
                tier_names.update(str(name) for name in model_tiers.keys() if str(name).strip())
        return tier_names

    def normalize_string_list(self, values: list[Any]) -> list[str]:
        normalized: list[str] = []
        for raw in list(values or []):
            value = str(raw).strip()
            if not value or value in normalized:
                continue
            normalized.append(value)
        return normalized

    def normalize_path(self, value: str) -> str:
        raw = str(value or "").strip()
        if not raw:
            return ""
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = (self.project_root / path).resolve()
        else:
            path = path.resolve()
        return str(path)

    def normalize_path_list(self, values: list[Any]) -> list[str]:
        normalized: list[str] = []
        for raw in list(values or []):
            path = self.normalize_path(str(raw or ""))
            if not path or path in normalized:
                continue
            normalized.append(path)
        return normalized

    def apply_settings_to_models(self, models_cfg: dict[str, Any], observer_settings: dict[str, Any]) -> dict[str, Any]:
        merged = json.loads(json.dumps(models_cfg, ensure_ascii=False))
        model_settings = observer_settings.get("models", {}) if isinstance(observer_settings.get("models"), dict) else {}
        route_overrides = model_settings.get("model_routes", {}) if isinstance(model_settings.get("model_routes"), dict) else {}
        tier_overrides = model_settings.get("model_tiers", {}) if isinstance(model_settings.get("model_tiers"), dict) else {}
        module_binding_overrides = model_settings.get("module_model_bindings", {}) if isinstance(model_settings.get("module_model_bindings"), dict) else {}
        binding_overrides = model_settings.get("agent_model_bindings", {}) if isinstance(model_settings.get("agent_model_bindings"), dict) else {}
        failover_override = model_settings.get("failover", {}) if isinstance(model_settings.get("failover"), dict) else {}
        merged.setdefault("model_routes", {})
        merged.setdefault("model_tiers", {})
        merged.setdefault("module_model_bindings", {})
        merged.setdefault("agent_model_bindings", {})
        merged["failover"] = {**dict(merged.get("failover", {}) or {}), **failover_override}
        for route_name, override in route_overrides.items():
            if not isinstance(override, dict):
                continue
            merged["model_routes"][route_name] = {**merged["model_routes"].get(route_name, {}), **override}
        for tier_name, override in tier_overrides.items():
            if not isinstance(override, dict):
                continue
            merged["model_tiers"][tier_name] = {**merged["model_tiers"].get(tier_name, {}), **override}
        for binding_key, override in module_binding_overrides.items():
            merged["module_model_bindings"][binding_key] = str(override)
        for binding_key, override in binding_overrides.items():
            merged["agent_model_bindings"][binding_key] = str(override)
        return merged

    def current_settings(self, config: dict[str, Any] | None = None) -> dict[str, Any]:
        if isinstance(config, dict):
            return config.get("observer_settings", self.default_settings())
        return self.default_settings()
