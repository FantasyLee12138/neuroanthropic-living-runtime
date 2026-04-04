from __future__ import annotations

import json
import os
import time
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
    api_key_env: str | None = None


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
    usage: dict[str, Any] = field(default_factory=dict)
    latency_ms: int = 0
    backend: str = ""


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
            usage={},
            latency_ms=0,
            backend=route.backend,
        )


class DoubaoBackend:
    def __init__(self, client_factory: Callable[..., Any] | None = None) -> None:
        self.client_factory = client_factory
        self._client_cache: dict[tuple[str, str, int], Any] = {}

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

    def _client_for(self, *, api_key: str, base_url: str, timeout_ms: int):
        cache_key = (api_key, base_url, timeout_ms)
        client = self._client_cache.get(cache_key)
        if client is None:
            client = self._build_client(api_key=api_key, base_url=base_url, timeout_ms=timeout_ms)
            self._client_cache[cache_key] = client
        return client

    def _extract_response_text(self, raw_payload: dict[str, Any]) -> str:
        if not isinstance(raw_payload, dict):
            raise ModelProviderError("Doubao returned a non-dict response payload")
        if isinstance(raw_payload.get("text"), str) and raw_payload["text"].strip():
            return raw_payload["text"].strip()
        if isinstance(raw_payload.get("output_text"), str) and raw_payload["output_text"].strip():
            return raw_payload["output_text"].strip()
        choices = raw_payload.get("choices", [])
        if isinstance(choices, list):
            texts: list[str] = []
            for choice in choices:
                if not isinstance(choice, dict):
                    continue
                message = choice.get("message", {})
                if isinstance(message, dict) and isinstance(message.get("content"), str) and message["content"].strip():
                    texts.append(message["content"].strip())
            if texts:
                return "\n".join(texts)

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

    def _uses_chat_completions(self, route: ModelRouteConfig) -> bool:
        return route.model.startswith("ep-")

    def generate(
        self,
        *,
        route: ModelRouteConfig,
        request: ModelRequest,
        api_key: str | None = None,
    ) -> ModelResponse:
        effective_key = api_key or os.getenv(route.api_key_env or "ARK_API_KEY")
        if not effective_key:
            raise MissingModelCredentialError(f"{route.api_key_env or 'ARK_API_KEY'} is required for Doubao backend")

        client = self._client_for(
            api_key=effective_key,
            base_url=route.base_url,
            timeout_ms=route.timeout_ms,
        )
        started = time.perf_counter()
        if self._uses_chat_completions(route):
            path = "/chat/completions"
            request_body = {
                "model": route.model,
                "messages": [
                    {"role": "system", "content": request.system_prompt},
                    {"role": "user", "content": request.user_prompt},
                ],
            }
        else:
            path = "/responses"
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
            response = client.post(path, json=request_body)
            response.raise_for_status()
            raw_payload = response.json()
            raw_text = response.text
            payload = self._extract_payload(raw_payload, request)
            usage = raw_payload.get("usage", {}) if isinstance(raw_payload, dict) else {}
            return ModelResponse(
                route=route.name,
                model=route.model,
                payload=payload,
                raw_text=raw_text,
                usage=usage if isinstance(usage, dict) else {},
                latency_ms=max(1, int((time.perf_counter() - started) * 1000)),
                backend=route.backend,
            )
        finally:
            pass


class DeepSeekBackend:
    def __init__(self, client_factory: Callable[..., Any] | None = None) -> None:
        self.client_factory = client_factory
        self._client_cache: dict[tuple[str, str, int], Any] = {}

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

    def _client_for(self, *, api_key: str, base_url: str, timeout_ms: int):
        cache_key = (api_key, base_url, timeout_ms)
        client = self._client_cache.get(cache_key)
        if client is None:
            client = self._build_client(api_key=api_key, base_url=base_url, timeout_ms=timeout_ms)
            self._client_cache[cache_key] = client
        return client

    def _request_body(self, *, route: ModelRouteConfig, request: ModelRequest, stream: bool) -> dict[str, Any]:
        return {
            "model": route.model,
            "messages": [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_prompt},
            ],
            "stream": stream,
        }

    def _extract_response_text(self, raw_payload: dict[str, Any]) -> str:
        if not isinstance(raw_payload, dict):
            raise ModelProviderError("DeepSeek returned a non-dict response payload")
        choices = raw_payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ModelProviderError("DeepSeek response did not contain choices")
        message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
        raise ModelProviderError("DeepSeek response did not contain message content")

    def _extract_payload(self, raw_payload: dict[str, Any], request: ModelRequest) -> dict[str, Any]:
        expected_keys = tuple(request.response_schema.keys())
        if expected_keys and all(key in raw_payload for key in expected_keys):
            return {key: raw_payload[key] for key in expected_keys}
        response_text = self._extract_response_text(raw_payload).strip()
        try:
            parsed = json.loads(response_text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict) and expected_keys and all(key in parsed for key in expected_keys):
            return parsed
        if expected_keys == ("text",):
            if isinstance(parsed, dict) and isinstance(parsed.get("text"), str):
                return {"text": parsed["text"].strip()}
            return {"text": response_text}
        raise ModelProviderError("DeepSeek response did not contain the required structured fields")

    def generate(
        self,
        *,
        route: ModelRouteConfig,
        request: ModelRequest,
        api_key: str | None = None,
    ) -> ModelResponse:
        effective_key = api_key or os.getenv(route.api_key_env or "DEEPSEEK_API_KEY")
        if not effective_key:
            raise MissingModelCredentialError(f"{route.api_key_env or 'DEEPSEEK_API_KEY'} is required for DeepSeek backend")
        client = self._client_for(api_key=effective_key, base_url=route.base_url, timeout_ms=route.timeout_ms)
        request_body = self._request_body(route=route, request=request, stream=False)
        started = time.perf_counter()
        response = client.post("/chat/completions", json=request_body)
        response.raise_for_status()
        raw_payload = response.json()
        raw_text = response.text
        payload = self._extract_payload(raw_payload, request)
        usage = raw_payload.get("usage", {}) if isinstance(raw_payload, dict) else {}
        return ModelResponse(
            route=route.name,
            model=route.model,
            payload=payload,
            raw_text=raw_text,
            usage=usage if isinstance(usage, dict) else {},
            latency_ms=max(1, int((time.perf_counter() - started) * 1000)),
            backend=route.backend,
        )

    def stream_generate(
        self,
        *,
        route: ModelRouteConfig,
        request: ModelRequest,
        api_key: str | None = None,
    ):
        effective_key = api_key or os.getenv(route.api_key_env or "DEEPSEEK_API_KEY")
        if not effective_key:
            raise MissingModelCredentialError(f"{route.api_key_env or 'DEEPSEEK_API_KEY'} is required for DeepSeek backend")
        client = self._client_for(api_key=effective_key, base_url=route.base_url, timeout_ms=route.timeout_ms)
        request_body = self._request_body(route=route, request=request, stream=True)
        with client.stream("POST", "/chat/completions", json=request_body) as response:
            response.raise_for_status()
            for raw_line in response.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    payload = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = payload.get("choices")
                if not isinstance(choices, list) or not choices:
                    continue
                delta = choices[0].get("delta", {}) if isinstance(choices[0], dict) else {}
                content = delta.get("content")
                if isinstance(content, str) and content:
                    yield content


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
            "deepseek": DeepSeekBackend(),
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

    def generate_config(self, route: ModelRouteConfig, request: ModelRequest) -> ModelResponse:
        if not route.enabled:
            raise ModelProviderError(f"model route {route.name} is disabled")
        backend = self.backends[route.backend]
        return backend.generate(route=route, request=request)

    def stream_generate(self, route_name: str, request: ModelRequest):
        route = self.route_configs[route_name]
        if not route.enabled:
            raise ModelProviderError(f"model route {route_name} is disabled")
        backend = self.backends[route.backend]
        stream_generate = getattr(backend, "stream_generate", None)
        if callable(stream_generate):
            yield from stream_generate(route=route, request=request)
            return
        response = backend.generate(route=route, request=request)
        text = str(response.payload.get("text", "")).strip()
        if text:
            yield text

    def stream_generate_config(self, route: ModelRouteConfig, request: ModelRequest):
        if not route.enabled:
            raise ModelProviderError(f"model route {route.name} is disabled")
        backend = self.backends[route.backend]
        stream_generate = getattr(backend, "stream_generate", None)
        if callable(stream_generate):
            yield from stream_generate(route=route, request=request)
            return
        response = backend.generate(route=route, request=request)
        text = str(response.payload.get("text", "")).strip()
        if text:
            yield text
