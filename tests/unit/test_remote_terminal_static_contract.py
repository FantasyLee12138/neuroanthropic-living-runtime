from __future__ import annotations

from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[2] / "apps" / "remote-terminal"
DASHBOARD_HTML = Path(__file__).resolve().parents[2] / "services" / "observer" / "dashboard" / "index.html"


def test_remote_terminal_static_contract_mentions_observer_web_session_api():
    index_html = (APP_ROOT / "index.html").read_text(encoding="utf-8")
    readme = (APP_ROOT / "README.md").read_text(encoding="utf-8")

    assert "stateless relay client" in index_html
    assert "/web/session/start" in index_html
    assert "/web/session/event" in index_html
    assert "/web/session/events" in index_html
    assert "/web/session/state" in index_html
    assert "user_turn" in index_html
    assert "control_command" in index_html
    assert "transcript" in index_html
    assert "observer base URL" in index_html
    assert "localStorage" not in index_html
    assert "sessionStorage" not in index_html
    assert "stateless relay client" in readme
    assert "GET /web/session/state" in readme
    assert "POST /web/session/event" in readme


def test_dashboard_static_contract_uses_light_session_state_endpoint():
    index_html = DASHBOARD_HTML.read_text(encoding="utf-8")

    assert "/web/session/state?session_id=" in index_html
    assert "full=true" not in index_html
