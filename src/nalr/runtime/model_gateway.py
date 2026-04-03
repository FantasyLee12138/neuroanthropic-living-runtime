from __future__ import annotations

import os
from typing import Any, Callable


class ModelGateway:
    def __init__(
        self,
        provider: str,
        base_url: str,
        primary_model: str,
        fast_model: str,
        timeout_seconds: int = 20,
        retries: int = 1,
        max_output_tokens: int = 512,
        client_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.provider = provider
        self.base_url = base_url
        self.primary_model = primary_model
        self.fast_model = fast_model
        self.timeout_seconds = timeout_seconds
        self.retries = retries
        self.max_output_tokens = max_output_tokens
        self.client_factory = client_factory

    @classmethod
    def from_config(cls, config: dict, client_factory: Callable[[], Any] | None = None) -> "ModelGateway":
        return cls(
            provider=config["provider"],
            base_url=config["base_url"],
            primary_model=config["primary_model"],
            fast_model=config["fast_model"],
            timeout_seconds=config.get("timeout_seconds", 20),
            retries=config.get("retries", 1),
            max_output_tokens=config.get("max_output_tokens", 512),
            client_factory=client_factory,
        )

    def _get_client(self):
        if self.client_factory is not None:
            return self.client_factory()
        from openai import OpenAI

        return OpenAI(base_url=self.base_url, api_key=os.getenv("ARK_API_KEY"), timeout=self.timeout_seconds)

    def _extract_text(self, response: Any) -> str:
        if isinstance(response, dict):
            return response.get("output_text", "")
        if hasattr(response, "output_text"):
            return getattr(response, "output_text")
        if hasattr(response, "output") and response.output:
            chunks = []
            for item in response.output:
                for content in getattr(item, "content", []):
                    text = getattr(content, "text", None)
                    if text:
                        chunks.append(text)
            return "\n".join(chunks)
        return ""

    def _invoke(self, prompt: str, model: str) -> dict[str, Any]:
        if not os.getenv("ARK_API_KEY"):
            return {
                "text": prompt,
                "provider": "rule_fallback",
                "model": "fallback",
                "latency_ms": 0,
                "usage": {},
            }

        client = self._get_client()
        try:
            response = client.responses.create(
                model=model,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": prompt,
                            }
                        ],
                    }
                ],
            )
            usage = response.get("usage", {}) if isinstance(response, dict) else getattr(response, "usage", {})
            text = self._extract_text(response)
        except AttributeError:
            chat = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            usage = getattr(chat, "usage", {})
            text = chat.choices[0].message.content
        return {
            "text": text,
            "provider": self.provider,
            "model": model,
            "latency_ms": 0,
            "usage": usage,
        }

    def plan(self, prompt: str) -> dict[str, Any]:
        fallback_text = f"Fallback plan: {prompt}"
        try:
            result = self._invoke(prompt, self.primary_model)
        except Exception:
            return {"text": fallback_text, "provider": "rule_fallback", "model": "fallback", "latency_ms": 0}
        if result["provider"] == "rule_fallback":
            result["text"] = fallback_text
        return result

    def render(self, action_name: str, style_profile: dict, user_text: str) -> dict[str, Any]:
        prompt = (
            f"Action: {action_name}\n"
            f"Style: {style_profile.get('style_name', 'neutral')}\n"
            f"User input: {user_text}\n"
            "Return one natural reply."
        )
        fallback_text = f"[{style_profile.get('style_name', 'neutral')}] {action_name}: {user_text}"
        try:
            result = self._invoke(prompt, self.fast_model)
        except Exception:
            return {"text": fallback_text, "provider": "rule_fallback", "model": "fallback", "latency_ms": 0}
        if result["provider"] == "rule_fallback":
            result["text"] = fallback_text
        return result
