from pathlib import Path

from fastapi.testclient import TestClient

from services.observer.api.app import create_app


CONFIG_ROOT = Path(__file__).resolve().parents[2] / "config"


def test_observer_defaults_surface_guided_learning_policy(tmp_path):
    client = TestClient(create_app(project_root=tmp_path, config_root=CONFIG_ROOT))

    payload = client.get("/settings").json()
    autonomy = payload["autonomy"]

    assert autonomy["learning_mode"] == "guided-learn"
    assert autonomy["trace_external_learning"] is True
    assert str(tmp_path) in autonomy["writable_roots"]
    assert str(tmp_path / ".alive" / "learning") == autonomy["learning_log_dir"]
    assert str(tmp_path / "docs") in autonomy["knowledge_roots"]


def test_observer_settings_can_override_controlled_learning_boundaries(tmp_path):
    client = TestClient(create_app(project_root=tmp_path, config_root=CONFIG_ROOT))
    workspace_root = tmp_path / "workspace"
    cache_root = tmp_path / ".alive" / "learning-cache"

    update_response = client.post(
        "/settings",
        json={
            "autonomy": {
                "learning_mode": "active-learn",
                "network_enabled": True,
                "external_io_enabled": True,
                "allowed_network_domains": ["docs.python.org", "developer.mozilla.org"],
                "writable_roots": [str(workspace_root), str(cache_root)],
                "knowledge_roots": [str(tmp_path / "docs"), str(cache_root)],
                "learning_log_dir": str(tmp_path / ".alive" / "learning"),
                "trace_external_learning": True,
            }
        },
    )

    assert update_response.status_code == 200
    settings_payload = update_response.json()["autonomy"]
    assert settings_payload["learning_mode"] == "active-learn"
    assert settings_payload["allowed_network_domains"] == ["docs.python.org", "developer.mozilla.org"]
    assert settings_payload["writable_roots"] == [str(workspace_root), str(cache_root)]
    assert settings_payload["knowledge_roots"] == [str(tmp_path / "docs"), str(cache_root)]
    assert settings_payload["learning_log_dir"] == str(tmp_path / ".alive" / "learning")
    assert settings_payload["trace_external_learning"] is True

    start_response = client.post("/web/runtime/start", json={})
    assert start_response.status_code == 200

    state_payload = client.get("/state").json()
    autonomy_state = state_payload["autonomy_policy"]
    assert autonomy_state["learning_mode"] == "active-learn"
    assert autonomy_state["allowed_network_domains"] == ["docs.python.org", "developer.mozilla.org"]
    assert autonomy_state["writable_roots"] == [str(workspace_root), str(cache_root)]
    assert autonomy_state["knowledge_roots"] == [str(tmp_path / "docs"), str(cache_root)]
    assert autonomy_state["learning_log_dir"] == str(tmp_path / ".alive" / "learning")
    assert autonomy_state["trace_external_learning"] is True


def test_observer_dashboard_shell_includes_controlled_learning_labels(tmp_path):
    client = TestClient(create_app(project_root=tmp_path, config_root=CONFIG_ROOT))

    dashboard_response = client.get("/dashboard")

    assert dashboard_response.status_code == 200
    assert "受控学习" in dashboard_response.text
    assert "学习模式" in dashboard_response.text
    assert "允许域名" in dashboard_response.text
    assert "可写根目录" in dashboard_response.text
    assert "追踪状态" in dashboard_response.text
