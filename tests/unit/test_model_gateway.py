from nalr.runtime.model_gateway import ModelGateway


class _FakeResponses:
    def create(self, **kwargs):
        return {
            "output_text": f"model:{kwargs['model']}",
            "usage": {"input_tokens": 12, "output_tokens": 24},
        }


class _FakeClient:
    def __init__(self):
        self.responses = _FakeResponses()


def test_model_gateway_falls_back_without_api_key(monkeypatch):
    monkeypatch.delenv("ARK_API_KEY", raising=False)
    gateway = ModelGateway(
        provider="doubao_ark",
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        primary_model="doubao-seed-2-0-pro-260215",
        fast_model="doubao-seed-2-0-pro-260215",
    )

    result = gateway.plan("Plan my evening")

    assert result["provider"] == "rule_fallback"
    assert result["model"] == "fallback"
    assert result["text"]


def test_model_gateway_uses_doubao_client_when_key_is_present(monkeypatch):
    monkeypatch.setenv("ARK_API_KEY", "test-key")
    gateway = ModelGateway(
        provider="doubao_ark",
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        primary_model="doubao-seed-2-0-pro-260215",
        fast_model="doubao-seed-2-0-pro-260215",
        client_factory=_FakeClient,
    )

    result = gateway.render("respond", {"style_name": "task_focused"}, "Please answer directly.")

    assert result["provider"] == "doubao_ark"
    assert result["model"] == "doubao-seed-2-0-pro-260215"
    assert "model:doubao-seed-2-0-pro-260215" in result["text"]

