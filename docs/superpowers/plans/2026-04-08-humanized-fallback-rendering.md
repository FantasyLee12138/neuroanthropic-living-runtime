# Humanized Fallback Rendering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make user-visible fallback expression model-backed and humanized before deterministic bottom-line fallback is used.

**Architecture:** Keep typed skill fallbacks deterministic and schema-safe. Add a dedicated renderer fallback model chain in the runtime controller for user-visible text, shared by primary renderer fallback and authenticity fallback paths.

**Tech Stack:** Python, pytest, existing `ModelRouter` routes, runtime controller render pipeline

---

### Task 1: Lock fallback behavior with regression tests

**Files:**
- Modify: `tests/unit/test_runtime_controller.py`
- Test: `tests/unit/test_runtime_controller.py`

- [ ] **Step 1: Write the failing tests**

- [ ] **Step 2: Run the targeted tests and confirm they fail**

- [ ] **Step 3: Implement minimal runtime changes**

- [ ] **Step 4: Re-run the targeted tests and confirm they pass**

### Task 2: Add shared model-backed fallback renderer

**Files:**
- Modify: `src/nalr/runtime/controller.py`
- Modify: `config/models.yaml`
- Test: `tests/unit/test_runtime_controller.py`

- [ ] **Step 1: Add fallback route selection and prompting helpers**

- [ ] **Step 2: Wire the helper into renderer fallback and authenticity fallback**

- [ ] **Step 3: Preserve deterministic `fallback_render_text()` as last-resort output**

- [ ] **Step 4: Re-run targeted tests**

### Task 3: Verify end-to-end behavior

**Files:**
- Modify: `tests/unit/test_output_expression_layer.py` (only if deterministic fallback expectations need adjustment)
- Test: `tests/unit/test_runtime_controller.py`
- Test: `tests/unit/test_output_expression_layer.py`

- [ ] **Step 1: Run runtime controller fallback tests**

- [ ] **Step 2: Run output fallback rendering tests**

- [ ] **Step 3: Confirm final behavior still preserves deterministic bottom-line fallback**
