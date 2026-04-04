from nalr.runtime.model_gateway import ModelGateway


class _FakeResponses:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "output_text": f"model:{kwargs['model']}",
            "usage": {"input_tokens": 12, "output_tokens": 24},
        }


class _FakeClient:
    def __init__(self):
        self.responses = _FakeResponses()


class _FakeChatCompletions:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)

        class _Choice:
            def __init__(self, content: str) -> None:
                self.message = type("Message", (), {"content": content})()

        return type(
            "ChatResponse",
            (),
            {
                "choices": [_Choice(f"chat:{kwargs['model']}")],
                "usage": {"prompt_tokens": 8, "completion_tokens": 16},
            },
        )()


class _FakeChatClient:
    def __init__(self):
        self.chat = type("ChatNamespace", (), {"completions": _FakeChatCompletions()})()


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
        client_factory=lambda **_: _FakeClient(),
    )

    result = gateway.render("respond", {"style_name": "task_focused"}, "Please answer directly.")

    assert result["provider"] == "doubao_ark"
    assert result["model"] == "doubao-seed-2-0-pro-260215"
    assert "model:doubao-seed-2-0-pro-260215" in result["text"]


def test_model_gateway_forwards_openai_style_limits_and_client_kwargs(monkeypatch):
    monkeypatch.setenv("ARK_API_KEY", "test-key")
    factory_kwargs = {}
    client = _FakeClient()
    gateway = ModelGateway(
        provider="doubao_ark",
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        primary_model="doubao-seed-2-0-pro-260215",
        fast_model="doubao-seed-2-0-pro-260215",
        client_factory=lambda **kwargs: factory_kwargs.update(kwargs) or client,
        max_output_tokens=128,
    )

    gateway.plan("Plan my evening")

    assert factory_kwargs["base_url"] == "https://ark.cn-beijing.volces.com/api/v3"
    assert factory_kwargs["api_key"] == "test-key"
    assert factory_kwargs["timeout_s"] == 20
    assert client.responses.calls[0]["max_output_tokens"] == 128


def test_model_gateway_falls_back_to_chat_completions_when_responses_api_is_missing(monkeypatch):
    monkeypatch.setenv("ARK_API_KEY", "test-key")
    factory_kwargs = {}
    client = _FakeChatClient()
    gateway = ModelGateway(
        provider="doubao_ark",
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        primary_model="doubao-seed-2-0-pro-260215",
        fast_model="doubao-seed-2-0-pro-260215",
        client_factory=lambda **kwargs: factory_kwargs.update(kwargs) or client,
        max_output_tokens=96,
    )

    result = gateway.render("respond", {"style_name": "task_focused"}, "Please answer directly.")

    assert result["provider"] == "doubao_ark"
    assert result["model"] == "doubao-seed-2-0-pro-260215"
    assert result["text"] == "chat:doubao-seed-2-0-pro-260215"
    assert factory_kwargs["base_url"] == "https://ark.cn-beijing.volces.com/api/v3"
    assert client.chat.completions.calls[0]["max_tokens"] == 96
