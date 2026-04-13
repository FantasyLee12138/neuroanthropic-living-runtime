from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_remote_terminal_static_client_references_web_session_api():
    html_path = REPO_ROOT / "apps" / "remote-terminal" / "index.html"
    readme_path = REPO_ROOT / "apps" / "remote-terminal" / "README.md"

    assert html_path.exists()
    assert readme_path.exists()

    html = html_path.read_text(encoding="utf-8")
    readme = readme_path.read_text(encoding="utf-8")

    assert "/web/session/start" in html
    assert "/web/session/event" in html
    assert "/web/session/events" in html
    assert "/web/session/state" in html
    assert "stateless relay client" in readme.lower()
    assert "observer" in readme.lower()
