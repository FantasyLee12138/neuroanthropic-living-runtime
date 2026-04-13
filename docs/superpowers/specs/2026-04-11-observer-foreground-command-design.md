# Observer Foreground Command Design

**Date:** 2026-04-11

## Goal

Expose a public foreground startup command for the observer web so users can run it in the current terminal and have the web process stop when that terminal exits.

## Current State

`alive-observer` currently exposes public management commands for background service control, while the actual foreground runtime is hidden behind the internal `serve` subcommand in `src/nalr/observer_launcher.py`.

## Proposed Change

Add a public `foreground` subcommand that reuses the existing `_serve_foreground()` implementation and accepts the same launch arguments as `start` and `restart`.

Keep the existing hidden `serve` subcommand for internal compatibility, especially because background `start` still launches `nalr.observer_launcher serve` as a child process.

## User-Facing Behavior

- `./alive-observer foreground` starts the observer web in the current terminal session.
- `Ctrl-C` or closing that terminal stops the foreground web process.
- `./alive-observer --help` shows `foreground` as an officially supported command.

## Scope

This change is intentionally limited to CLI exposure. It does not change background `start`, `stop`, or the web server implementation itself.

## Testing

- Verify CLI help exposes `foreground`.
- Verify argv normalization preserves `foreground` instead of rewriting it to `start`.
- Verify `main()` dispatches `foreground` to `_serve_foreground()`.
