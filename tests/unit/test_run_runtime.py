from pathlib import Path

from nalr.runtime.controller import RuntimeController


CONFIG_ROOT = Path(__file__).resolve().parents[2] / "config"


def test_start_run_creates_read_only_supervisor_state_and_traces(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("def hello():\n    return 'world'\n", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_app.py").write_text("def test_placeholder():\n    assert True\n", encoding="utf-8")

    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    run_payload = controller.start_run("修复 app.py 并检查相关测试")
    status_payload = controller.run_status()
    steps_payload = controller.run_steps(run_payload["run_id"])
    tools_payload = controller.run_tools(run_payload["run_id"])
    explain_payload = controller.explain_run(run_payload["run_id"])

    assert run_payload["run_id"]
    assert run_payload["status"] == "running"
    assert run_payload["goal"] == "修复 app.py 并检查相关测试"
    assert status_payload["run_id"] == run_payload["run_id"]
    assert status_payload["current_step_id"]
    assert steps_payload["steps"]
    assert tools_payload["tools"]
    assert tools_payload["tools"][0]["tool_name"] == "repo_scan"
    assert explain_payload["current_step"]["tool_choice"] == "repo_scan"
    state = controller.load_runtime_state()
    assert state.active_run_id == run_payload["run_id"]
    assert state.run_status == "running"


def test_run_lifecycle_supports_pause_resume_and_abort(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "worker.py").write_text("VALUE = 1\n", encoding="utf-8")

    controller = RuntimeController(project_root=tmp_path, config_root=CONFIG_ROOT)

    started = controller.start_run("检查 worker.py")
    paused = controller.pause_run(started["run_id"])
    resumed = controller.resume_run(started["run_id"])
    aborted = controller.abort_run(started["run_id"], reason="operator_requested")

    assert paused["status"] == "paused"
    assert paused["stop_reason"]["code"] == "operator_paused"
    assert resumed["status"] == "running"
    assert aborted["status"] == "aborted"
    assert aborted["stop_reason"]["code"] == "operator_requested"
