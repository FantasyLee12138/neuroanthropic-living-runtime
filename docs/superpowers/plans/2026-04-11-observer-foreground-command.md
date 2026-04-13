# Observer Foreground Command Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose a supported `foreground` observer launcher command so users can run the web in the current terminal instead of the detached background manager.

**Architecture:** Reuse the existing foreground server path in `src/nalr/observer_launcher.py` and only change CLI parsing and dispatch. Keep the hidden `serve` command in place for the background launcher internals, so the diff stays small and behavior stays stable.

**Tech Stack:** Python, argparse, pytest

---

### Task 1: Expose the Public CLI Command

**Files:**
- Modify: `src/nalr/observer_launcher.py`
- Test: `tests/integration/test_observer_launcher.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_observer_launcher_help_lists_public_foreground_command():
    parser = observer_launcher.build_parser()

    help_text = parser.format_help()

    assert "foreground" in help_text


def test_observer_launcher_normalize_argv_preserves_foreground_command():
    assert observer_launcher._normalize_argv(["foreground"]) == ["foreground"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/integration/test_observer_launcher.py -k foreground -v`
Expected: FAIL because `foreground` is not yet a public command and `_normalize_argv()` still rewrites unknown commands to `start`.

- [ ] **Step 3: Write the minimal implementation**

```python
foreground_parser = subparsers.add_parser(
    "foreground",
    help="Start the observer service in the foreground.",
)
_add_launch_args(foreground_parser)

if argv[0] in {"start", "status", "stop", "restart", "logs", "foreground", "serve", "-h", "--help"}:
    return argv

if command in {"foreground", "serve"}:
    raise SystemExit(_serve_foreground(args))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/integration/test_observer_launcher.py -k foreground -v`
Expected: PASS

- [ ] **Step 5: Run the broader launcher test file**

Run: `pytest tests/integration/test_observer_launcher.py -v`
Expected: PASS
