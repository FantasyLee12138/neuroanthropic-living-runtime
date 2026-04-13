# Self-Hosted 24h Soak

## Run Meta

- Started at: `2026-04-08T22:33:43+08:00`
- Planned finish: `2026-04-08T22:33:46+08:00`
- Finished at: `2026-04-08T22:33:47+08:00`
- Status: `PARTIAL`
- Status reason: `shortened_duration_not_real_24h`
- Reused existing service: `True`

## Scope

- Probe `/service/status`, `/autonomy/status`, `/state`, `/monologue/status`
- Record JSONL samples under `.alive/logs/`
- Emit a markdown report under `docs/testing/`
- Treat the report as real 24h evidence only when the full duration completes

## Probe Summary

- `autonomy_status`: total=2, ok=2, fail=0, avg=0.051, p95=0.052, max=0.052
- `monologue_status`: total=2, ok=2, fail=0, avg=0.009, p95=0.009, max=0.009
- `service_status`: total=2, ok=2, fail=0, avg=0.025, p95=0.025, max=0.025
- `state`: total=2, ok=2, fail=0, avg=0.75, p95=0.928, max=0.928

## Runtime Summary

- service_status_reads: `2`
- autonomy_running_samples: `0`
- initiative_changes: `1`
- monologue_changes: `0`
- current_round_changes: `0`

## Notes

- This run did not cover the full 24h duration, so it cannot promote the matrix row to pass.
- 真实 24h soak 仍未完成。
