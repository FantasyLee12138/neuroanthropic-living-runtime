# Workbench 6h Soak

## Run Meta

- Started at: `2026-04-08T04:05:46+08:00`
- Planned finish: `2026-04-08T10:05:46+08:00`
- Status: `PASS`
- Launch mode: `Open_NALR_Workbench.command --no-browser`
- Reused existing server on `127.0.0.1:8765`: `True`

## Scope

- Workbench direct startup
- Continuous observation of `dashboard`, `autonomy/status`, `console/state`, `web/session/state`
- Periodic dialogue via `/web/session/event` on `observer-main`
- Pre-soak code fix: heavy observer request paths are offloaded off the event loop and serialized under a runtime lock

## Pre-Soak Verification

- Added concurrency regressions for `/console/talk`, `/web/session/event`, `/console/endogenous/tick`
- Focused observer regressions passed locally after the fix
- Residual note: manual `/console/endogenous/tick` is still slow in live mode and is tracked as an explicit risk
- Residual note: `observer-main` dialogue completion can lag; this soak gates on workbench availability and user-turn acceptance, and only records settle when it is observed

## Probe Summary

- `autonomy_status`: total=360, ok=360, fail=0, avg=0.103, p95=0.154, max=0.181
- `console_state`: total=360, ok=360, fail=0, avg=0.673, p95=0.776, max=0.945
- `dashboard`: total=360, ok=360, fail=0, avg=0.006, p95=0.008, max=0.013
- `observer_main_state`: total=360, ok=360, fail=0, avg=1.278, p95=1.34, max=1.387

## Dialogue Summary

- total=12, ok=12, fail=0, avg=0.001, p95=0.001, max=0.001
- last response excerpt: `{"accepted": true, "session_id": "observer-main", "event_count": 0, "last_event_id": 5, "queued": true}`

## Notes

- autonomy/status precheck status=200, latency=0.072s, already_running=True
- completed at 2026-04-08T10:05:50+08:00 with 0 failed probes/dialogues across 1440 probes and 12 dialogues
