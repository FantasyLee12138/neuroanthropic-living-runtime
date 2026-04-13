# Foreground Port Conflict Handling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make public foreground observer startup replace an existing same-project observer process on the requested port instead of crashing with a bind error.

**Architecture:** Add a small preflight branch in `src/nalr/observer_launcher.py` before `_serve_foreground()` is called from the public `foreground` command. Reuse the existing port target resolution and service-status JSON to decide whether the conflicting process belongs to the current project and can be stopped safely.

**Tech Stack:** Python, argparse, pytest

---

### Task 1: Preflight Foreground Startup

**Files:**
- Modify: `src/nalr/observer_launcher.py`
- Modify: `tests/integration/test_observer_launcher.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_observer_launcher_main_foreground_replaces_existing_same_project_instance(monkeypatch, tmp_path):
    observed: dict[str, object] = {}

    monkeypatch.setattr(
        observer_launcher,
        "resolve_launch_target",
        lambda host, port, open_path: {"mode": "restart_required", "port": port, "url": f"http://{host}:{port}{open_path}"},
    )
    monkeypatch.setattr(
        observer_launcher,
        "_load_json_url",
        lambda url, timeout_seconds=2.0: {"project_root": str(tmp_path), "pid": 45678},
    )
    monkeypatch.setattr(observer_launcher, "_stop_pid", lambda pid, timeout_seconds=10.0: observed.setdefault("stopped_pid", pid) or True)
    monkeypatch.setattr(observer_launcher, "_serve_foreground", lambda args: observed.setdefault("served_port", args.port) or 0)

    with pytest.raises(SystemExit) as exc_info:
        observer_launcher.main(["foreground", "--project-root", str(tmp_path), "--config-root", str(CONFIG_ROOT)])

    assert exc_info.value.code == 0
    assert observed["stopped_pid"] == 45678
    assert observed["served_port"] == 8765
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/integration/test_observer_launcher.py::test_observer_launcher_main_foreground_replaces_existing_same_project_instance -v`
Expected: FAIL because public foreground startup currently skips launch-target conflict handling.

- [ ] **Step 3: Write the minimal implementation**

```python
target = resolve_launch_target(str(args.host), int(args.port), str(args.open_path))
if target["mode"] == "restart_required":
    remote_status = _load_json_url(_compose_service_status_url(str(args.host), int(args.port)), timeout_seconds=2.0) or {}
    if str(remote_status.get("project_root") or "") == str(Path(args.project_root)):
        remote_pid = int(remote_status.get("pid") or 0)
        if remote_pid and _stop_pid(remote_pid, timeout_seconds=2.0):
            target = {"mode": "start", "port": int(args.port), "url": _compose_dashboard_url(str(args.host), int(args.port), str(args.open_path))}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/integration/test_observer_launcher.py::test_observer_launcher_main_foreground_replaces_existing_same_project_instance tests/integration/test_observer_launcher.py::test_observer_launcher_main_dispatches_public_foreground_command -v`
Expected: PASS

- [ ] **Step 5: Run direct wrapper verification**

Run: `pytest tests/integration/test_observer_launcher.py::test_workbench_command_invokes_foreground_by_default -v`
Expected: PASS
