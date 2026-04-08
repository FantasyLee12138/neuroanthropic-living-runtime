# Observer Service Health Workbench Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make observer service health reflect live truth, and align workbench connectivity and probability-space reads to the same runtime snapshot.

**Architecture:** Keep `service.json` as static launch metadata, continue computing health at read time, and extend `/service/status` with runtime occupancy diagnostics that help distinguish "service reachable" from "turn or runtime currently occupied". On the dashboard side, treat `/service/status` as the source of truth for connectivity and fetch probability-space against the same `current_round` snapshot used by the rest of the console panels.

**Tech Stack:** Python, FastAPI, static dashboard HTML/JS, pytest

---

### Task 1: Lock In Service-Status Diagnostics

**Files:**
- Modify: `tests/integration/test_observer_api.py`
- Modify: `services/observer/api/app.py`

- [ ] **Step 1: Write the failing test**

```python
def test_observer_service_status_reports_active_turn_diagnostics(tmp_path, monkeypatch):
    ...
    assert payload["observer_turn_active"] is True
    assert "observer-main" in payload["active_turn_sessions"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH='/Users/fantasylee/类脑架构:/Users/fantasylee/类脑架构/src' pytest tests/integration/test_observer_api.py -k service_status_reports_active_turn_diagnostics -q`
Expected: FAIL because `/service/status` does not yet expose active turn diagnostics.

- [ ] **Step 3: Write minimal implementation**

```python
def _active_turn_sessions() -> list[str]:
    ...

def _service_status_payload() -> dict:
    return {
        ...
        "observer_turn_active": _observer_turn_active(),
        "active_turn_sessions": _active_turn_sessions(),
        "runtime_serial_active": ...,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH='/Users/fantasylee/类脑架构:/Users/fantasylee/类脑架构/src' pytest tests/integration/test_observer_api.py -k service_status_reports_active_turn_diagnostics -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/integration/test_observer_api.py services/observer/api/app.py
git commit -m "fix: expose live observer service diagnostics"
```

### Task 2: Align Probability-Space Reads To A Chosen Round

**Files:**
- Modify: `tests/integration/test_observer_api.py`
- Modify: `src/nalr/runtime/diagnostics_runtime.py`
- Modify: `services/observer/api/app.py`

- [ ] **Step 1: Write the failing test**

```python
def test_console_probability_space_accepts_round_ref_for_consistent_snapshot(tmp_path):
    ...
    response = client.get("/console/probability-space", params={"round_ref": 1})
    assert response.json()["source_round_id"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH='/Users/fantasylee/类脑架构:/Users/fantasylee/类脑架构/src' pytest tests/integration/test_observer_api.py -k consistent_snapshot -q`
Expected: FAIL because the endpoint always uses the latest round.

- [ ] **Step 3: Write minimal implementation**

```python
def console_probability_space(self, round_ref: int | str | None = None) -> dict[str, Any]:
    target_round = controller.resolve_round_ref(round_ref) if round_ref is not None else controller.console_round_or_none()
    ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH='/Users/fantasylee/类脑架构:/Users/fantasylee/类脑架构/src' pytest tests/integration/test_observer_api.py -k consistent_snapshot -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/integration/test_observer_api.py src/nalr/runtime/diagnostics_runtime.py services/observer/api/app.py
git commit -m "fix: allow observer probability space to pin round snapshots"
```

### Task 3: Move Dashboard Connection Truth To `/service/status`

**Files:**
- Modify: `tests/integration/test_observer_api.py`
- Modify: `services/observer/dashboard/index.html`

- [ ] **Step 1: Write the failing test**

```python
def test_dashboard_connection_badge_uses_service_status_not_backend_online_flag(tmp_path):
    ...
    assert "&& backendOnline" not in dashboard_response.text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH='/Users/fantasylee/类脑架构:/Users/fantasylee/类脑架构/src' pytest tests/integration/test_observer_api.py -k backend_online_flag -q`
Expected: FAIL because the dashboard still mixes `backendOnline` into the connection badge.

- [ ] **Step 3: Write minimal implementation**

```html
const connected = Boolean((serviceStatusState || {}).http_ready);
const probabilityPath = currentRoundId
  ? `/console/probability-space?round_ref=${encodeURIComponent(currentRoundId)}`
  : "/console/probability-space";
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH='/Users/fantasylee/类脑架构:/Users/fantasylee/类脑架构/src' pytest tests/integration/test_observer_api.py -k backend_online_flag -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/integration/test_observer_api.py services/observer/dashboard/index.html
git commit -m "fix: derive dashboard connection truth from service status"
```
