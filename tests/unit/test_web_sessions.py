from __future__ import annotations

import json

from services.observer.api.web_sessions import WebSessionBroker, build_workbench_cards


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


def test_web_session_broker_keeps_only_recent_events_per_session():
    broker = WebSessionBroker(max_events_per_session=3)

    broker.append_many(
        "sess-browser",
        [{"type": "assistant_final", "message": f"event-{index}"} for index in range(5)],
    )

    recent = broker.read_since("sess-browser", after_id=0)
    latest_only = broker.read_since("sess-browser", after_id=4)

    assert [item["event_id"] for item in recent] == [3, 4, 5]
    assert [item["message"] for item in recent] == ["event-2", "event-3", "event-4"]
    assert [item["event_id"] for item in latest_only] == [5]


def test_web_session_broker_marks_events_as_projection_only():
    broker = WebSessionBroker()

    recorded = broker.append_many(
        "sess-browser",
        [{"type": "assistant_final", "message": "hello", "trace_ref": "round://1"}],
    )

    event = recorded[0]

    assert event["truth_contract"]["projection_only"] is True
    assert event["truth_contract"]["authoritative"] is False
    assert event["truth_contract"]["source"] == "web_session_broker"
    assert event["truth_contract"]["event_log_backed"] is True


def test_build_workbench_cards_surfaces_the_four_read_models():
    snapshot = {
        "console": {
            "state": {
                "mode": "interactive",
                "cognitive_snapshot": {
                    "tlh": {
                        "body_state": {"energy": 0.7, "fatigue": 0.2, "memory_fragments": 0.3, "self_continuity": 0.6, "meaning_strength": 0.4},
                        "subjective_state": {"felt": ["累", "想停"], "boundary": 0.8, "spontaneous": 0.7, "reject_all": 0.1, "meaning_made": ["先收回去"]},
                        "emotion_state": {"valence": 0.2, "arousal": 0.4},
                        "desire_state": {"top_goal": "rest"},
                        "instinct_field": {"winner_region": "absorb", "axis_values": {"E": 0.3, "F": 0.4, "S": 0.5, "M": 0.6}},
                    }
                },
            },
            "action_field": {"winner": {"action": "rest"}, "top_actions": [{"action": "rest"}], "competing_peaks": [{"action": "plan"}], "winner_posterior": {"rest": 0.8, "plan": 0.2}},
            "probability_field": {"context": {"x": 1}, "memory": {"y": 2}, "action": {"winner_posterior": {"rest": 0.8}}, "token": {"z": 3}},
            "autonomy": {"profile": "tool_level", "learning_mode": "observe", "allowed_commands": ["mode set sleep"], "blocked_commands": ["danger"], "allowed_network_domains": ["example.com"], "writable_roots": ["/tmp"], "running": True},
            "why_current": {"why": {"summary": "rest"}},
            "why_not": {"why_not": {"summary": "plan was blocked", "blocked_by": ["body"]}},
            "recent_rounds": [{"round_id": 1, "sampled_action": "rest"}],
            "source_links": [{"panel_id": "brain_state", "api_path": "/console/state", "controller_method": "RuntimeController.console_state"}],
        }
    }

    cards = build_workbench_cards(snapshot)["cards"]

    assert [card["panel_id"] for card in cards] == ["cognitive_chain", "controls", "alerts", "dashboards"]
    assert len(cards[0]["rows"]) >= 6
    assert cards[1]["rows"][0]["label"] == "behavior_policies"
    assert "layer_fuses" in cards[2]["rows"][0]["label"]
    assert "dashboard_specs" in cards[3]["rows"][0]["label"]
