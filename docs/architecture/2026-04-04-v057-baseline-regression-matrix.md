# NALR v0.57 Baseline Regression Matrix

Date: 2026-04-04
Worktree: `/Users/fantasylee/.config/superpowers/worktrees/类脑架构/codex-nalr-v057`
Branch: `codex/nalr-v057-probability-field`

## Summary

This matrix captures Gate 1 baseline status before schema convergence cleanup. It reflects the current in-flight v0.57 worktree rather than a clean branch baseline.

## Suites

| Suite | Command Group | Status | Notes |
| --- | --- | --- | --- |
| Core probability/runtime/observer | `test_probability_field*`, `test_runtime_*`, `test_trace_store`, `test_observer_*`, `test_skill_runtime` | 97 passed | Green after schema convergence, observer canonicalization, and checkpoint trace fixes. |
| Runtime entrypoints | `test_terminal_bridge`, `test_nalr_terminal`, `test_terminal_bridge_stdio`, `test_dream_bridge_stdio`, `test_async_io_runtime`, `test_run_runtime`, `test_dream_runtime` | 33 passed, 6 skipped | Green after checkpoint trace fix. |
| Compatibility/CLI/gateway | `test_cli`, `test_cil_cli`, `test_trace_maintenance_cli`, `test_model_providers`, `test_model_gateway` | 44 passed | Green after launcher interpreter fallback, identity CLI seeding, gateway contract tightening, and trace fallback fix. |
| Longrun/authenticity | `test_longrun_smoke`, `test_authenticity_acceptance` | 2 passed | Green after runtime hot-path reductions in breaker persistence, dataclass serialization, and recent-trace window access. |

## Failure Classification

### Fixed in This Wave

1. Runtime/observer
   - `tests/unit/test_runtime_controller.py::test_command_safe_mode_and_checkpoint_emit_command_trace`
   - `tests/unit/test_runtime_controller.py::test_rewind_restores_checkpointed_runtime_state`
   - `tests/unit/test_async_io_runtime.py::test_checkpoint_waits_for_background_state_flush`
   - Resolution: `TraceStore.append_command()` now tolerates sparse checkpoint-like `CommandResult` payloads and preserves command trace writes.

2. Compatibility
   - `tests/integration/test_cli.py::test_repo_launcher_exposes_alive_help`
   - Resolution: `alive` now selects an interpreter that can import the repo CLI dependencies before launching the CLI module.

3. Compatibility + runtime identity surface
   - `tests/integration/test_cli.py::test_cli_identity_show_and_set_name`
   - Resolution: CLI compatibility path seeds the runtime identity directly while leaving CIL boundary rejection semantics intact.

4. Compatibility + trace fallback
   - `tests/integration/test_cli.py::test_cli_supports_trace_compact_and_counterfactual_commands`
   - Resolution: `TraceStore.list_rounds()` now falls back to JSON/memory when parquet directories exist without live parquet files.

## Triage Labels

- `blocker`
  - none remaining in the validated runtime/observer/compatibility suites
- `known gap`
  - none remaining in the validated baseline matrix
- `unrelated`
  - none identified yet
