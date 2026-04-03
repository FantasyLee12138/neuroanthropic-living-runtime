import os

import pytest

from nalr.providers.router import DoubaoBackend, ModelRequest, ModelRouteConfig


@pytest.mark.skipif(not os.getenv("ARK_API_KEY"), reason="requires ARK_API_KEY")
def test_doubao_backend_smoke():
    backend = DoubaoBackend()

    response = backend.generate(
        route=ModelRouteConfig(
            name="renderer",
            backend="doubao",
            model="doubao-seed-2-0-pro-260215",
            timeout_ms=60000,
            retries=0,
            enabled=True,
        ),
        request=ModelRequest(
            system_prompt="你是最终表达 renderer。只返回 JSON，不要额外解释。",
            user_prompt='{"task":"返回一个 JSON 对象，包含 text 字段，值是一句简短中文。"}',
            response_schema={"text": "str"},
        ),
    )

    assert response.payload["text"]


@pytest.mark.skipif(not os.getenv("ARK_API_KEY"), reason="requires ARK_API_KEY")
def test_doubao_backend_structured_smoke():
    backend = DoubaoBackend()

    response = backend.generate(
        route=ModelRouteConfig(
            name="pfc",
            backend="doubao",
            model="doubao-seed-2-0-pro-260215",
            timeout_ms=60000,
            retries=0,
            enabled=True,
        ),
        request=ModelRequest(
            system_prompt="你是 PFCAgent。只返回 JSON，不要额外解释。输出 action_preferences、confidence、sigma_scale、reason。",
            user_prompt='{"task":"用户说要规划今晚并记住晚饭想吃面。返回 action_preferences、confidence、sigma_scale、reason。"}',
            response_schema={
                "action_preferences": "dict[str, float]",
                "confidence": "float",
                "sigma_scale": "float",
                "reason": "str",
            },
        ),
    )

    assert response.payload["action_preferences"]
    assert response.payload["confidence"] >= 0.0
