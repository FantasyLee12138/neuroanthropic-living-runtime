from __future__ import annotations

import json

from services.observer.api.web_sessions import WebSessionBroker


def test_web_session_broker_stream_sse_sanitizes_non_finite_numbers():
    broker = WebSessionBroker()
    broker.append_many(
        "sess-browser",
        [
            {
                "type": "sidebar_snapshot",
                "score": float("nan"),
                "nested": {"high": float("inf"), "low": float("-inf")},
            }
        ],
    )

    chunks = list(broker.stream_sse("sess-browser", once=True))
    payload_line = next(chunk for chunk in chunks if chunk.startswith("id: "))
    json_line = next(line for line in payload_line.splitlines() if line.startswith("data: "))
    payload = json.loads(json_line[len("data: ") :])

    assert payload["score"] is None
    assert payload["nested"]["high"] is None
    assert payload["nested"]["low"] is None
