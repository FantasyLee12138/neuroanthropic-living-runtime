from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable

import httpx


ARK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"


class ModelProviderError(RuntimeError):
    pass


class MissingModelCredentialError(ModelProviderError):
    pass


@dataclass
class ModelRouteConfig:
    name: str
    backend: str
    model: str
    timeout_ms: int
    retries: int = 0
    enabled: bool = True
    base_url: str = ARK_BASE_URL


@dataclass
class ModelRequest:
    system_prompt: str
    user_prompt: str
    response_schema: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelResponse:
    route: str
    model: str
    payload: dict[str, Any]
    raw_text: str = ""


class FakeBackend:
    def generate(
        self,
        *,
        route: ModelRouteConfig,
        request: ModelRequest,
        api_key: str | None = None,
    ) -> ModelResponse:
        del api_key
        content = request.user_prompt.lower()
        payload: dict[str, Any]

        if "action_preferences" in request.response_schema:
            prefs = {"respond": 0.08}
            if any(token in content for token in {"plan", "help"}):
                prefs["plan"] = 0.34
            if any(token in content for token in {"remember", "recall"}) or request.metadata.get("cue"):
                prefs["recall"] = max(prefs.get("recall", 0.0), 0.27)
            payload = {
                "action_preferences": prefs,
                "confidence": 0.84,
                "sigma_scale": 0.92,
                "reason": f"fake-model route={route.name}",
            }
        elif "state_hypothesis" in request.response_schema:
            closeness = float(request.metadata.get("closeness", 0.5))
            payload = {
                "state_hypothesis": {
                    "confused": request.metadata.get("sampled_action") == "clarify" or "clarify" in content,
                    "closeness": closeness,
                    "guarded": request.metadata.get("relationship_risk", 0.0) >= 0.4,
                }
            }
        elif "reaction_hypothesis" in request.response_schema:
            closeness = float(request.metadata.get("closeness", 0.5))
            sampled_action = request.metadata.get("sampled_action", "respond")
            payload = {
                "reaction_hypothesis": {
                    "positive": closeness > 0.6 and sampled_action != "clarify",
                    "risk": round(max(0.0, 1.0 - closeness), 4),
                    "action": sampled_action,
                }
            }
        else:
            action = request.metadata.get("action", "respond")
            action_text = {
                "plan": "我先给你一个简短计划。",
                "recall": "我先把相关记忆线索提出来。",
                "clarify": "我先确认一下你的真实意图。",
                "connect": "我会先顺着你的情绪接住你。",
                "rest": "我需要先收一点强度，再继续回应。",
                "short_reply": "我先给你一个更短更直接的回应。",
                "wander": "我先停一下，理一理散开的念头。",
                "respond": "我先直接回应你的核心问题。",
            }.get(action, "我先直接回应你的核心问题。")
            payload = {"text": action_text}

        return ModelResponse(
            route=route.name,
            model=route.model,
            payload=payload,
            raw_text=json.dumps(payload, ensure_ascii=False),
        )


class DoubaoBackend:
    def __init__(self, client_factory: Callable[..., Any] | None = None) -> None:
        self.client_factory = client_factory

    def _build_client(self, *, api_key: str, base_url: str, timeout_ms: int):
        timeout_s = max(timeout_ms / 1000.0, 1.0)
        if self.client_factory is not None:
            return self.client_factory(base_url=base_url, api_key=api_key, timeout_s=timeout_s)
        return httpx.Client(
            base_url=base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=timeout_s,
        )

    def _extract_response_text(self, raw_payload: dict[str, Any]) -> str:
        if not isinstance(raw_payload, dict):
            raise ModelProviderError("Doubao returned a non-dict response payload")
        if isinstance(raw_payload.get("text"), str) and raw_payload["text"].strip():
            return raw_payload["text"].strip()
        if isinstance(raw_payload.get("output_text"), str) and raw_payload["output_text"].strip():
            return raw_payload["output_text"].strip()

        texts: list[str] = []
        for item in raw_payload.get("output", []):
            if not isinstance(item, dict):
                continue
            for content in item.get("content", []):
                if not isinstance(content, dict):
                    continue
                if isinstance(content.get("text"), str) and content["text"].strip():
                    texts.append(content["text"].strip())
        if texts:
            return "\n".join(texts)
        raise ModelProviderError("Doubao response did not contain text output")

    def _strip_code_fences(self, text: str) -> str:
        stripped = text.strip()
        if not stripped.startswith("```"):
            return stripped
        lines = stripped.splitlines()
        if len(lines) >= 3 and lines[0].startswith("```") and lines[-1].startswith("```"):
            return "\n".join(lines[1:-1]).strip()
        return stripped

    def _extract_payload(self, raw_payload: dict[str, Any], request: ModelRequest) -> dict[str, Any]:
        if not isinstance(raw_payload, dict):
            raise ModelProviderError("Doubao returned a non-dict response payload")

        expected_keys = tuple(request.response_schema.keys())
        if expected_keys and all(key in raw_payload for key in expected_keys):
            return {key: raw_payload[key] for key in expected_keys}

        response_text = self._extract_response_text(raw_payload)
        cleaned_text = self._strip_code_fences(response_text)
        try:
            parsed = json.loads(cleaned_text)
        except json.JSONDecodeError:
            parsed = None

        if isinstance(parsed, dict) and expected_keys and all(key in parsed for key in expected_keys):
            return parsed
        if expected_keys == ("text",):
            if isinstance(parsed, dict) and isinstance(parsed.get("text"), str):
                return {"text": parsed["text"].strip()}
            return {"text": cleaned_text}
        raise ModelProviderError("Doubao response did not contain the required structured fields")

    def generate(
        self,
        *,
        route: ModelRouteConfig,
        request: ModelRequest,
        api_key: str | None = None,
    ) -> ModelResponse:
        effective_key = api_key or os.getenv("ARK_API_KEY")
        if not effective_key:
            raise MissingModelCredentialError("ARK_API_KEY is required for Doubao backend")

        client = self._build_client(
            api_key=effective_key,
            base_url=route.base_url,
            timeout_ms=route.timeout_ms,
        )
        request_body = {
            "model": route.model,
            "input": [
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": request.system_prompt}],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": request.user_prompt}],
                },
            ],
        }
        try:
            response = client.post("/responses", json=request_body)
            response.raise_for_status()
            raw_payload = response.json()
            raw_text = response.text
            payload = self._extract_payload(raw_payload, request)
            return ModelResponse(route=route.name, model=route.model, payload=payload, raw_text=raw_text)
        finally:
            close = getattr(client, "close", None)
            if callable(close):
                close()


class ModelRouter:
    def __init__(
        self,
        route_configs: dict[str, ModelRouteConfig],
        *,
        backends: dict[str, Any] | None = None,
    ) -> None:
        self.route_configs = route_configs
        self.backends = {
            "fake": FakeBackend(),
            "doubao": DoubaoBackend(),
        }
        if backends:
            self.backends.update(backends)

    @classmethod
    def from_config(cls, payload: dict[str, Any]) -> "ModelRouter":
        routes = {
            route_name: ModelRouteConfig(name=route_name, **route_cfg)
            for route_name, route_cfg in payload.get("model_routes", {}).items()
        }
        return cls(routes)

    def generate(self, route_name: str, request: ModelRequest) -> ModelResponse:
        route = self.route_configs[route_name]
        if not route.enabled:
            raise ModelProviderError(f"model route {route_name} is disabled")
        backend = self.backends[route.backend]
        return backend.generate(route=route, request=request)
