import json

from nalr.providers.router import (
    FakeBackend,
    DeepSeekBackend,
    DoubaoBackend,
    OpenAICompatibleBackend,
    ModelRequest,
    ModelRouteConfig,
    ModelRouter,
    MissingModelCredentialError,
)


class _StubResponses:
    def __init__(self) -> None:
        self.calls: list[dict] = []


class _StubHTTPResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload
        self.text = json.dumps(payload, ensure_ascii=False)

    def json(self):
        return self._payload

    def raise_for_status(self):
        return None


class _StubClient:
    def __init__(self, payload: dict | None = None) -> None:
        self.responses = _StubResponses()
        self.post_calls: list[dict] = []
        self.payload = payload or {
            "output": [
                {
                    "content": [
                        {
                            "type": "output_text",
                            "text": '{"text":"ok"}',
                        }
                    ]
                }
            ]
        }

    def post(self, path, *, json):
        self.post_calls.append({"path": path, "json": json})
        return _StubHTTPResponse(self.payload)


class _StubStreamContext:
    def __init__(self, lines: list[str]) -> None:
        self.lines = lines

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def raise_for_status(self):
        return None

    def iter_lines(self):
        for line in self.lines:
            yield line


class _StubStreamingClient(_StubClient):
    def __init__(self, lines: list[str]) -> None:
        super().__init__()
        self.stream_calls: list[dict] = []
        self.lines = lines

    def stream(self, method, path, *, json):
        self.stream_calls.append({"method": method, "path": path, "json": json})
        return _StubStreamContext(self.lines)


def test_model_router_uses_fake_backend_for_configured_route():
    router = ModelRouter(
        {
            "renderer": ModelRouteConfig(
                name="renderer",
                backend="fake",
                model="fake-renderer",
                timeout_ms=100,
                retries=0,
                enabled=True,
            )
        }
    )

    response = router.generate(
        "renderer",
        ModelRequest(
            system_prompt="You are a renderer.",
            user_prompt="Render a short answer.",
            response_schema={"text": "str"},
            metadata={"action": "respond"},
        ),
    )

    assert response.route == "renderer"
    assert response.model == "fake-renderer"
    assert response.payload["text"]


def test_model_router_supports_dynamic_route_config_generation():
    router = ModelRouter({}, backends={"fake": FakeBackend()})

    response = router.generate_config(
        ModelRouteConfig(
            name="small_model",
            backend="fake",
            model="fake-small",
            timeout_ms=100,
            retries=0,
            enabled=True,
        ),
        ModelRequest(
            system_prompt="You are a scoring model.",
            user_prompt="Return JSON only.",
            response_schema={"text": "str"},
            metadata={"action": "respond"},
        ),
    )

    assert response.route == "small_model"
    assert response.model == "fake-small"
    assert response.payload["text"]


def test_doubao_backend_requires_ark_api_key():
    backend = DoubaoBackend(client_factory=lambda **_: _StubClient())

    try:
        backend.generate(
            route=ModelRouteConfig(
                name="pfc",
                backend="doubao",
                model="doubao-seed-2-0-pro-260215",
                timeout_ms=300,
                retries=0,
                enabled=True,
            ),
            request=ModelRequest(
                system_prompt="You are a planner.",
                user_prompt="Return JSON only.",
                response_schema={"action_preferences": "dict[str, float]"},
            ),
        )
    except MissingModelCredentialError as exc:
        assert "ARK_API_KEY" in str(exc)
    else:
        raise AssertionError("expected MissingModelCredentialError")


def test_deepseek_backend_uses_route_specific_api_key_env(monkeypatch):
    stub_client = _StubClient(
        {
            "choices": [
                {
                    "message": {
                        "content": '{"text":"ok"}',
                    }
                }
            ],
            "usage": {"prompt_tokens": 12, "completion_tokens": 4},
        }
    )
    backend = DeepSeekBackend(client_factory=lambda **_: stub_client)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("CUSTOM_SMALL_MODEL_KEY", "small-key")

    response = backend.generate(
        route=ModelRouteConfig(
            name="small_model",
            backend="deepseek",
            model="deepseek-chat",
            timeout_ms=300,
            retries=0,
            enabled=True,
            base_url="https://api.deepseek.com",
            api_key_env="CUSTOM_SMALL_MODEL_KEY",
        ),
        request=ModelRequest(
            system_prompt="You are a scorer.",
            user_prompt="Return JSON only.",
            response_schema={"text": "str"},
        ),
    )

    assert response.payload["text"] == "ok"
    assert response.usage["prompt_tokens"] == 12


def test_openai_compatible_backend_allows_missing_api_key_for_local_endpoint():
    stub_client = _StubClient(
        {
            "choices": [
                {
                    "message": {
                        "content": '{"text":"local ok"}',
                    }
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
    )
    backend = OpenAICompatibleBackend(client_factory=lambda **_: stub_client)

    response = backend.generate(
        route=ModelRouteConfig(
            name="local_model",
            backend="openai_compatible",
            model="qwen-local",
            timeout_ms=500,
            retries=0,
            enabled=True,
            base_url="http://127.0.0.1:11434/v1",
            api_key_env="LOCAL_MODEL_API_KEY",
        ),
        request=ModelRequest(
            system_prompt="You are a local model.",
            user_prompt="Return JSON only.",
            response_schema={"text": "str"},
        ),
    )

    assert response.payload["text"] == "local ok"
    assert stub_client.post_calls[0]["path"] == "/chat/completions"


def test_doubao_backend_shapes_openai_responses_request():
    stub_client = _StubClient()
    factory_kwargs = {}
    backend = DoubaoBackend(client_factory=lambda **kwargs: factory_kwargs.update(kwargs) or stub_client)

    response = backend.generate(
        route=ModelRouteConfig(
            name="renderer",
            backend="doubao",
            model="doubao-seed-2-0-pro-260215",
            timeout_ms=300,
            retries=1,
            enabled=True,
        ),
        request=ModelRequest(
            system_prompt="You are a renderer.",
            user_prompt="Return JSON only.",
            response_schema={"text": "str"},
            metadata={"stage": "renderer"},
        ),
        api_key="test-key",
    )

    assert response.payload["text"] == "ok"
    assert factory_kwargs["timeout_s"] == 1.0
    assert stub_client.post_calls[0]["path"] == "/responses"
    assert stub_client.post_calls[0]["json"]["model"] == "doubao-seed-2-0-pro-260215"
    assert stub_client.post_calls[0]["json"]["input"][0]["role"] == "system"
    assert stub_client.post_calls[0]["json"]["input"][1]["role"] == "user"


def test_doubao_endpoint_id_uses_chat_completions_request():
    stub_client = _StubClient(
        {
            "choices": [
                {
                    "message": {
                        "content": '{"text":"ok"}',
                    }
                }
            ],
            "usage": {"prompt_tokens": 9, "completion_tokens": 3},
        }
    )
    backend = DoubaoBackend(client_factory=lambda **_: stub_client)

    response = backend.generate(
        route=ModelRouteConfig(
            name="salience_small_model",
            backend="doubao",
            model="ep-20260404191810-qfn7s",
            timeout_ms=300,
            retries=0,
            enabled=True,
        ),
        request=ModelRequest(
            system_prompt="You are a small model scorer.",
            user_prompt='Return {"text":"ok"} only.',
            response_schema={"text": "str"},
        ),
        api_key="test-key",
    )

    assert response.payload["text"] == "ok"
    assert response.usage["prompt_tokens"] == 9
    assert stub_client.post_calls[0]["path"] == "/chat/completions"
    assert stub_client.post_calls[0]["json"]["model"] == "ep-20260404191810-qfn7s"
    assert stub_client.post_calls[0]["json"]["messages"][0]["role"] == "system"
    assert stub_client.post_calls[0]["json"]["messages"][1]["role"] == "user"


def test_doubao_backend_parses_structured_json_from_output_text():
    stub_client = _StubClient(
        {
            "output": [
                {
                    "content": [
                        {
                            "type": "output_text",
                            "text": json.dumps(
                                {
                                    "action_preferences": {"plan": 0.85, "recall": 0.82},
                                    "confidence": 0.88,
                                    "sigma_scale": 1.0,
                                    "reason": "structured",
                                },
                                ensure_ascii=False,
                            ),
                        }
                    ]
                }
            ]
        }
    )
    backend = DoubaoBackend(client_factory=lambda **_: stub_client)

    response = backend.generate(
        route=ModelRouteConfig(
            name="pfc",
            backend="doubao",
            model="doubao-seed-2-0-pro-260215",
            timeout_ms=15000,
            retries=0,
            enabled=True,
        ),
        request=ModelRequest(
            system_prompt="You are a planner.",
            user_prompt="Return JSON only.",
            response_schema={
                "action_preferences": "dict[str, float]",
                "confidence": "float",
                "sigma_scale": "float",
                "reason": "str",
            },
        ),
        api_key="test-key",
    )

    assert response.payload["action_preferences"]["plan"] == 0.85
    assert response.payload["confidence"] == 0.88
    assert response.payload["reason"] == "structured"


def test_doubao_backend_reuses_persistent_client_for_same_route_settings():
    stub_client = _StubClient()
    factory_calls: list[dict] = []
    backend = DoubaoBackend(client_factory=lambda **kwargs: factory_calls.append(kwargs) or stub_client)
    route = ModelRouteConfig(
        name="renderer",
        backend="doubao",
        model="doubao-seed-2-0-pro-260215",
        timeout_ms=300,
        retries=0,
        enabled=True,
    )
    request = ModelRequest(
        system_prompt="You are a renderer.",
        user_prompt="Return JSON only.",
        response_schema={"text": "str"},
    )

    backend.generate(route=route, request=request, api_key="test-key")
    backend.generate(route=route, request=request, api_key="test-key")

    assert len(factory_calls) == 1
    assert len(stub_client.post_calls) == 2


def test_deepseek_backend_requires_api_key():
    backend = DeepSeekBackend(client_factory=lambda **_: _StubClient())

    try:
        backend.generate(
            route=ModelRouteConfig(
                name="chat_fast",
                backend="deepseek",
                model="deepseek-chat",
                timeout_ms=300,
                retries=0,
                enabled=True,
                base_url="https://api.deepseek.com",
            ),
            request=ModelRequest(
                system_prompt="You are a fast chat model.",
                user_prompt="你好",
                response_schema={"text": "str"},
            ),
        )
    except MissingModelCredentialError as exc:
        assert "DEEPSEEK_API_KEY" in str(exc)
    else:
        raise AssertionError("expected MissingModelCredentialError")


def test_deepseek_backend_shapes_chat_completions_request():
    stub_client = _StubClient(
        {
            "choices": [
                {
                    "message": {
                        "content": '{"text":"你好，我在。"}',
                    }
                }
            ]
        }
    )
    factory_kwargs = {}
    backend = DeepSeekBackend(client_factory=lambda **kwargs: factory_kwargs.update(kwargs) or stub_client)

    response = backend.generate(
        route=ModelRouteConfig(
            name="chat_fast",
            backend="deepseek",
            model="deepseek-chat",
            timeout_ms=300,
            retries=0,
            enabled=True,
            base_url="https://api.deepseek.com",
        ),
        request=ModelRequest(
            system_prompt="You are a fast chat model.",
            user_prompt="你好",
            response_schema={"text": "str"},
        ),
        api_key="test-key",
    )

    assert response.payload["text"] == "你好，我在。"
    assert factory_kwargs["timeout_s"] == 1.0
    assert stub_client.post_calls[0]["path"] == "/chat/completions"
    assert stub_client.post_calls[0]["json"]["model"] == "deepseek-chat"
    assert stub_client.post_calls[0]["json"]["messages"][0]["role"] == "system"
    assert stub_client.post_calls[0]["json"]["messages"][1]["role"] == "user"


def test_deepseek_backend_streams_text_deltas():
    stream_client = _StubStreamingClient(
        [
            'data: {"choices":[{"delta":{"content":"你好"}}]}',
            'data: {"choices":[{"delta":{"content":"，我在。"}}]}',
            "data: [DONE]",
        ]
    )
    backend = DeepSeekBackend(client_factory=lambda **_: stream_client)

    chunks = list(
        backend.stream_generate(
            route=ModelRouteConfig(
                name="chat_fast",
                backend="deepseek",
                model="deepseek-chat",
                timeout_ms=300,
                retries=0,
                enabled=True,
                base_url="https://api.deepseek.com",
            ),
            request=ModelRequest(
                system_prompt="You are a fast chat model.",
                user_prompt="你好",
                response_schema={"text": "str"},
            ),
            api_key="test-key",
        )
    )

    assert chunks == ["你好", "，我在。"]
    assert stream_client.stream_calls[0]["path"] == "/chat/completions"
    assert stream_client.stream_calls[0]["json"]["stream"] is True
