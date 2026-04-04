# Terminal Streaming Output Design

## Goal

Force terminal responses onto a streaming path so the first visible assistant text is pushed to the user immediately after it is available, rather than waiting for the full run, tool trace, or final summary. This applies to both:

- interactive terminal mode
- one-shot `NALR "prompt"` mode

The primary product goal is lower perceived latency. The implementation must expose an end-to-end streaming path now, while keeping the code structure compatible with future true model token streaming.

## Current Problem

The current terminal path does not deliver visible output incrementally:

1. Python `terminal_bridge` handles `user_turn` synchronously and returns a fully materialized event list.
2. `serve_stdio()` writes that list only after the handler has finished building it.
3. the terminal reducer ignores `assistant_token`
4. one-shot CLI only prints `assistant_final`

This means the user sees no assistant text until the entire backend run, agent scheduling, and reasoning phase has already completed.

## Design Summary

We will convert the terminal bridge path from batch event delivery to incremental event delivery.

For task-oriented `user_turn` requests, the bridge will emit a short assistant preamble as an `assistant_token` event immediately after run creation. The rest of the run events will follow in order, and `assistant_final` will close the stream.

This is not a fake frontend-only loading effect. It is a real streaming transport path through the bridge, the terminal UI reducer, and the one-shot CLI printer.

## Scope

### In scope

- stream `user_turn` events incrementally over stdio
- emit visible assistant text before the full run event sequence completes
- support streamed text rendering in the interactive Ink UI
- support streamed text printing in one-shot CLI mode
- preserve existing run, step, tool, approval, and sidebar events
- keep a backward-compatible final message fallback for non-streamed paths

### Out of scope

- full background worker architecture for terminal runs
- true token streaming from the underlying model provider
- broad redesign of slash command behavior
- changing the semantic meaning of existing run inspection commands

## Event Model

### Streaming sequence for task runs

For task-oriented `user_turn` requests, the bridge should emit events in this order:

1. `run_status`
2. first `assistant_token`
3. zero or more `step_update`
4. zero or more `tool_call`
5. zero or more `approval_request`
6. zero or more `tool_result`
7. `sidebar_snapshot`
8. `assistant_final`

The first `assistant_token` is the critical latency-hiding event. It should be emitted immediately after run creation and should be a short, stable natural-language preamble such as:

`已接收请求，正在调度 Agent 并读取上下文…`

The exact text can be adjusted during implementation, but it must remain short, natural, and semantically compatible with the final assistant message.

### Direct chat sequence

Direct-chat turns such as greetings can remain non-streamed for now and continue returning a single `assistant_final`, because their latency profile is already short and they do not rely on the run orchestration path.

## Backend Design

### `TerminalEventHandler`

The handler currently returns `list[dict[str, Any]]` for all inbound events. We will introduce a streaming path for `user_turn`:

- keep the public `handle()` entrypoint
- allow `handle()` to return an iterable of outbound events instead of requiring a fully built list first
- implement `user_turn` as a generator-like sequence for task runs

The direct-chat path can still emit a single final event.

### `serve_stdio()`

`serve_stdio()` must consume outbound events as they are produced:

- if the handler returns an iterable stream, write each event immediately
- flush stdout after each event
- preserve current error behavior

This is the transport change that makes the early token visible to the user instead of buffering it behind handler completion.

### Session persistence

Session state updates should still happen before any streamed task events that depend on the stored `active_run_id`. The session must be persisted as soon as the run is created so control commands continue to work during and after the streamed sequence.

## Frontend Design

### Interactive terminal reducer

`apps/terminal/src/state/sessionStore.ts` currently ignores `assistant_token`. We will add streamed-assistant line assembly:

- first `assistant_token` creates a new assistant transcript line
- subsequent `assistant_token` events append to the same in-progress assistant line
- `assistant_final` finalizes that in-progress line if one exists
- if no token was streamed, `assistant_final` falls back to the current behavior and appends a normal assistant line

This keeps transcript rendering simple and avoids duplicating the same assistant response twice.

### Interactive Ink UI

No large layout redesign is required. The current transcript view will automatically reflect reducer updates once `assistant_token` is applied to state.

## One-shot CLI Design

`apps/terminal/src/cli.ts` must stop treating `assistant_final` as the only printable assistant output.

The one-shot printer should:

- print streamed `assistant_token` deltas immediately using `stdout.write()`
- avoid duplicating the already streamed content when `assistant_final` arrives
- still print `assistant_final` when no prior token stream exists
- keep `error` behavior unchanged

This allows `NALR "prompt"` to show assistant text as soon as the first streamed token arrives.

## Compatibility Rules

- existing event types remain valid
- `assistant_final` remains the authoritative closing event
- consumers that do not yet interpret `assistant_token` still receive a usable final message
- slash commands other than the main task `user_turn` path can remain batch-oriented in this iteration

## Testing Strategy

Implementation will follow test-first development.

### Python tests

- add a unit test proving task `user_turn` emits `assistant_token` before `assistant_final`
- add an integration test proving stdio bridge output can be read incrementally in event order

### TypeScript tests

- add reducer tests proving multiple `assistant_token` events merge into one assistant transcript line
- add reducer tests proving `assistant_final` finalizes a streamed line without duplicating it
- add one-shot CLI coverage proving token output is visible before the final event path

## Risks And Mitigations

### Duplicate assistant text

Risk: streamed token content and final message both appear in transcript or one-shot output.

Mitigation: reducer and CLI printer both track whether a streamed assistant line is already active and treat `assistant_final` as a close/flush event instead of unconditional append.

### Event ordering drift

Risk: downstream UI logic may assume previous batch ordering.

Mitigation: keep a stable event order and place the early `assistant_token` directly after `run_status`, before observational events.

### Over-expansion into async run infrastructure

Risk: implementation grows into a full background-run rewrite.

Mitigation: limit this change to incremental emission over the existing task execution path and explicitly defer broader worker architecture changes.

## Implementation Boundary

The minimal acceptable implementation is:

- task `user_turn` emits an early visible `assistant_token`
- stdio bridge flushes that token immediately
- interactive terminal renders the token incrementally
- one-shot CLI prints the token incrementally
- final output is not duplicated

If those five conditions are satisfied, the user-visible latency problem is materially improved and the codebase is ready for later true model-token streaming.
