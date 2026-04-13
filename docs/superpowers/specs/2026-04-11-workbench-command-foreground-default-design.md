# Workbench Command Foreground Default Design

**Date:** 2026-04-11

## Goal

Make `Open_NALR_Workbench.command` start the observer web in the current terminal session by default, so closing that terminal also stops the web process.

## Current State

`Open_NALR_Workbench.command` currently delegates to `alive-observer start`, which backgrounds the observer service and detaches it from the terminal.

## Proposed Change

Switch the script's default delegation from `alive-observer start` to `alive-observer foreground`.

Keep the script shape otherwise unchanged: same root resolution, same argument forwarding, and the same `--help` passthrough pattern, but pointed at `foreground --help`.

## User-Facing Behavior

- Double-clicking `Open_NALR_Workbench.command` starts the observer web in the terminal window that opened it.
- Pressing `Ctrl-C` or closing that terminal stops the observer web.
- Extra CLI arguments still pass through to `alive-observer foreground`.

## Scope

This change only affects the command wrapper's default startup mode and the tests that assert its contents. It does not change `alive-observer`, the web server implementation, or the macOS app wrapper structure.

## Testing

- Verify the command wrapper delegates to `foreground` by default.
- Verify the packaged sample repo fixture expects the updated wrapper contents.
