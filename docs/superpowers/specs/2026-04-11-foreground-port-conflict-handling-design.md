# Foreground Port Conflict Handling Design

**Date:** 2026-04-11

## Goal

Keep `Open_NALR_Workbench.command` in foreground mode while avoiding raw bind failures when the requested observer port is already occupied by the same project.

## Current State

The public `foreground` path now starts the observer in the current terminal, but it calls `uvicorn.run()` directly on the requested port. Unlike the background `start` path, it does not perform any launch-target conflict handling before binding.

## Proposed Change

Add a small preflight step for the public `foreground` command:

- Reuse `resolve_launch_target()` to inspect the requested host/port.
- If the port is already serving an observer dashboard for the same `project_root`, read `/service/status`, stop that existing process by PID, then continue foreground startup.
- Keep the current error behavior when the conflicting dashboard belongs to another project or does not expose a stoppable PID.

The hidden internal `serve` command remains unchanged.

## User-Facing Behavior

- Double-clicking `Open_NALR_Workbench.command` still runs the observer in the current terminal.
- If the previous observer instance from the same project is still holding `127.0.0.1:8765`, the new foreground launch replaces it instead of failing with `[Errno 48] address already in use`.
- Conflicts with other projects still fail clearly instead of killing unrelated processes.

## Scope

This change is limited to the public `foreground` path and its tests. It does not alter the background `start` flow or generic service probing logic.

## Testing

- Verify public `foreground` startup injects metadata and dispatches correctly.
- Verify public `foreground` stops an existing same-project observer PID before serving.
