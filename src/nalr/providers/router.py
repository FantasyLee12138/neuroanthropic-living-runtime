from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
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


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


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
        elif "fragments" in request.response_schema:
            fragment_count = max(1, int(request.metadata.get("fragment_count", 1) or 1))
            variants = [
                ("free_association", "我脑子里突然闪过一个词：回声。"),
                ("environment_notice", "我注意到这个界面好像有点挤。"),
                ("memory_fragment", "我好像还惦记着之前没续上的那段话。"),
                ("daydream", "我在想如果现在出去走一圈会怎样。"),
                ("blank_fragment", "我刚刚空了一瞬。"),
            ]
            payload = {
                "fragments": [
                    {
                        "category": variants[index % len(variants)][0],
                        "content": variants[index % len(variants)][1],
                        "source": "model",
                    }
                    for index in range(fragment_count)
                ]
            }
        elif "goal" in request.response_schema:
            payload = {
                "goal": "Inspect recent runtime traces and identify the next read-only self-check step.",
                "reason": "fake-model route suggests a lightweight self-study pass",
            }
        elif "salience" in request.response_schema and "draft_reply" in request.response_schema:
            route_type = str(request.metadata.get("route_type") or "chat_fast")
            memory_need = any(token in content for token in {"记得", "上次", "之前", "remember", "memory"})
            tool_need = any(token in content for token in {"tool", "命令", "shell", "查一下", "搜索"})
            conflict_need = any(token in content for token in {"冲突", "矛盾", "纠结", "conflict"})
            deepen_reason = ""
            if route_type == "chat_deep":
                deepen_reason = "long_horizon_alignment"
            salience = 0.35
            if memory_need or tool_need or conflict_need:
                salience = 0.58
            if route_type == "chat_deep":
                salience = 0.82
            uncertainty = 0.22 if route_type == "chat_fast" else 0.44
            payload = {
                "salience": salience,
                "uncertainty": uncertainty,
                "memory_need": memory_need or route_type in {"chat_standard", "chat_deep"},
                "tool_need": tool_need,
                "conflict_need": conflict_need or route_type == "chat_deep",
                "candidate_action_prior": "respond",
                "draft_reply": "我会先基于当前状态给你一个直接回应。",
                "proposed_state_patch": {
                    "focus": "maintain_sparse_chat_response",
                    "obligations": ["延续当前回合的焦点并保持状态连续"],
                },
                "deepen_reason": deepen_reason,
            }
        elif "draft_reply" in request.response_schema and "deepen_reason" in request.response_schema:
            payload = {
                "draft_reply": "我先把这些目标、冲突和接下来的路径收束成一个更稳定的回应。",
                "deepen_reason": "long_horizon_alignment",
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
        timeout = httpx.Timeout(timeout_s, connect=min(timeout_s, 10.0), read=timeout_s, write=timeout_s, pool=min(timeout_s, 10.0))
        if self.client_factory is not None:
            return self.client_factory(base_url=base_url, api_key=api_key, timeout_s=timeout_s)
        return httpx.Client(
            base_url=base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=timeout,
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
        timeout = httpx.Timeout(timeout_s, connect=min(timeout_s, 10.0), read=timeout_s, write=timeout_s, pool=min(timeout_s, 10.0))
        if self.client_factory is not None:
            return self.client_factory(base_url=base_url, api_key=api_key, timeout_s=timeout_s)
        return httpx.Client(
            base_url=base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=timeout,
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


class OpenAICompatibleBackend:
    def __init__(self, client_factory: Callable[..., Any] | None = None) -> None:
        self.client_factory = client_factory
        self._client_cache: dict[tuple[str, str, int], Any] = {}

    def _build_client(self, *, api_key: str, base_url: str, timeout_ms: int):
        timeout_s = max(timeout_ms / 1000.0, 1.0)
        timeout = httpx.Timeout(timeout_s, connect=min(timeout_s, 10.0), read=timeout_s, write=timeout_s, pool=min(timeout_s, 10.0))
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        if self.client_factory is not None:
            return self.client_factory(base_url=base_url, api_key=api_key, timeout_s=timeout_s)
        return httpx.Client(base_url=base_url, headers=headers, timeout=timeout)

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

    def _extract_payload(self, raw_payload: dict[str, Any], request: ModelRequest) -> dict[str, Any]:
        if not isinstance(raw_payload, dict):
            raise ModelProviderError("OpenAI-compatible backend returned a non-dict response payload")
        choices = raw_payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ModelProviderError("OpenAI-compatible backend response did not contain choices")
        message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ModelProviderError("OpenAI-compatible backend response did not contain message content")
        text = content.strip()
        expected_keys = tuple(request.response_schema.keys())
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict) and expected_keys and all(key in parsed for key in expected_keys):
            return parsed
        if expected_keys == ("text",):
            if isinstance(parsed, dict) and isinstance(parsed.get("text"), str):
                return {"text": parsed["text"].strip()}
            return {"text": text}
        raise ModelProviderError("OpenAI-compatible backend did not contain the required structured fields")

    def generate(
        self,
        *,
        route: ModelRouteConfig,
        request: ModelRequest,
        api_key: str | None = None,
    ) -> ModelResponse:
        effective_key = api_key or os.getenv(route.api_key_env or "") or ""
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
        effective_key = api_key or os.getenv(route.api_key_env or "") or ""
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
        failover: dict[str, Any] | None = None,
    ) -> None:
        self.route_configs = route_configs
        self.backends = {
            "fake": FakeBackend(),
            "doubao": DoubaoBackend(),
            "deepseek": DeepSeekBackend(),
            "openai_compatible": OpenAICompatibleBackend(),
        }
        if backends:
            self.backends.update(backends)
        self.failover = self._normalize_failover_config(failover)

    @classmethod
    def from_config(cls, payload: dict[str, Any]) -> "ModelRouter":
        routes = {
            route_name: ModelRouteConfig(name=route_name, **route_cfg)
            for route_name, route_cfg in payload.get("model_routes", {}).items()
        }
        return cls(routes, failover=payload.get("failover"))

    def _normalize_failover_config(self, payload: dict[str, Any] | None) -> dict[str, Any]:
        config = payload if isinstance(payload, dict) else {}
        return {
            "enabled": bool(config.get("enabled", False)),
            "mode": str(config.get("mode", "global_cutover") or "global_cutover"),
            "backend": str(config.get("backend", "openai_compatible") or "openai_compatible"),
            "base_url": str(config.get("base_url", "") or ""),
            "model": str(config.get("model", "") or ""),
            "api_key_env": str(config.get("api_key_env", "") or "").strip() or None,
            "active": bool(config.get("active", False)),
            "reason": str(config.get("reason", "") or ""),
            "activated_at": str(config.get("activated_at", "") or ""),
        }

    def _is_remote_route(self, route: ModelRouteConfig) -> bool:
        mode = str(getattr(route, "effective_mode", "remote") or "remote").lower()
        return mode != "local" and route.backend != "fake"

    def activate_failover(self, reason: str) -> None:
        if not self.failover.get("enabled"):
            return
        if self.failover.get("active"):
            if not self.failover.get("reason"):
                self.failover["reason"] = str(reason or "")
            return
        self.failover["active"] = True
        self.failover["reason"] = str(reason or "")
        self.failover["activated_at"] = _utc_now_iso()

    def failover_status(self) -> dict[str, Any]:
        return dict(self.failover)

    def effective_route_config(self, route: ModelRouteConfig) -> ModelRouteConfig:
        if not self.failover.get("active") or not self._is_remote_route(route):
            return route
        return replace(
            route,
            backend=str(self.failover.get("backend") or route.backend),
            base_url=str(self.failover.get("base_url") or route.base_url),
            model=str(self.failover.get("model") or route.model),
            api_key_env=str(self.failover.get("api_key_env") or route.api_key_env or "").strip() or None,
        )

    def _route_failure_triggers_failover(self, route: ModelRouteConfig, exc: Exception) -> bool:
        if not self.failover.get("enabled") or not self._is_remote_route(route):
            return False
        if isinstance(exc, MissingModelCredentialError):
            return False
        if isinstance(exc, (ModelProviderError, TimeoutError, OSError)):
            return True
        return isinstance(exc, httpx.HTTPError)

    def _execute_route(self, route: ModelRouteConfig, request: ModelRequest) -> ModelResponse:
        backend = self.backends[route.backend]
        return backend.generate(route=route, request=request)

    def _stream_route(self, route: ModelRouteConfig, request: ModelRequest):
        backend = self.backends[route.backend]
        stream_generate = getattr(backend, "stream_generate", None)
        if callable(stream_generate):
            yield from stream_generate(route=route, request=request)
            return
        response = backend.generate(route=route, request=request)
        text = str(response.payload.get("text", "")).strip()
        if text:
            yield text

    def _with_failover(self, route: ModelRouteConfig, request: ModelRequest) -> ModelResponse:
        effective_route = self.effective_route_config(route)
        if effective_route is not route:
            return self._execute_route(effective_route, request)
        try:
            return self._execute_route(route, request)
        except Exception as exc:  # noqa: BLE001
            if not self._route_failure_triggers_failover(route, exc):
                raise
            self.activate_failover(f"{route.name}: {exc}")
            effective_route = self.effective_route_config(route)
            if effective_route is route:
                raise
            return self._execute_route(effective_route, request)

    def _stream_with_failover(self, route: ModelRouteConfig, request: ModelRequest):
        effective_route = self.effective_route_config(route)
        if effective_route is not route:
            yield from self._stream_route(effective_route, request)
            return
        try:
            yield from self._stream_route(route, request)
        except Exception as exc:  # noqa: BLE001
            if not self._route_failure_triggers_failover(route, exc):
                raise
            self.activate_failover(f"{route.name}: {exc}")
            effective_route = self.effective_route_config(route)
            if effective_route is route:
                raise
            yield from self._stream_route(effective_route, request)

    def generate(self, route_name: str, request: ModelRequest) -> ModelResponse:
        route = self.route_configs[route_name]
        if not route.enabled:
            raise ModelProviderError(f"model route {route_name} is disabled")
        return self._with_failover(route, request)

    def generate_config(self, route: ModelRouteConfig, request: ModelRequest) -> ModelResponse:
        if not route.enabled:
            raise ModelProviderError(f"model route {route.name} is disabled")
        return self._with_failover(route, request)

    def stream_generate(self, route_name: str, request: ModelRequest):
        route = self.route_configs[route_name]
        if not route.enabled:
            raise ModelProviderError(f"model route {route_name} is disabled")
        yield from self._stream_with_failover(route, request)

    def stream_generate_config(self, route: ModelRouteConfig, request: ModelRequest):
        if not route.enabled:
            raise ModelProviderError(f"model route {route.name} is disabled")
        yield from self._stream_with_failover(route, request)
