# Workbench Command Foreground Default Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `Open_NALR_Workbench.command` launch the observer in foreground mode by default so the terminal owns the web process lifecycle.

**Architecture:** Keep the existing shell wrapper and only replace its delegated subcommand from `start` to `foreground`. Cover the change with one direct wrapper behavior test and one packaging fixture update so the repo script and bundled copy stay aligned.

**Tech Stack:** zsh, pytest

---

### Task 1: Switch the Wrapper to Foreground Mode

**Files:**
- Modify: `Open_NALR_Workbench.command`
- Modify: `tests/integration/test_observer_launcher.py`
- Modify: `tests/unit/test_local_client_packager.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_workbench_command_invokes_foreground_by_default(tmp_path):
    command_path = tmp_path / "Open_NALR_Workbench.command"
    alive_path = tmp_path / "alive-observer"
    command_path.write_text((REPO_ROOT / "Open_NALR_Workbench.command").read_text(encoding="utf-8"), encoding="utf-8")
    command_path.chmod(0o755)
    alive_path.write_text("#!/bin/zsh\nprintf '%s\\n' \"$@\"\n", encoding="utf-8")
    alive_path.chmod(0o755)

    result = subprocess.run(
        ["zsh", str(command_path), "--port", "9900"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0
    assert result.stdout.splitlines()[:3] == ["foreground", "--port", "9900"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/integration/test_observer_launcher.py::test_workbench_command_invokes_foreground_by_default tests/unit/test_local_client_packager.py -k Open_NALR_Workbench -v`
Expected: FAIL because the wrapper and sample fixture still point at `start`.

- [ ] **Step 3: Write the minimal implementation**

```zsh
if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
  exec "$ROOT_DIR/alive-observer" foreground --help
fi

exec "$ROOT_DIR/alive-observer" foreground "$@"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/integration/test_observer_launcher.py::test_workbench_command_invokes_foreground_by_default tests/unit/test_local_client_packager.py -k Open_NALR_Workbench -v`
Expected: PASS

- [ ] **Step 5: Run the direct help regression**

Run: `pytest tests/integration/test_observer_launcher.py::test_workbench_command_exposes_observer_launcher_help -v`
Expected: PASS
