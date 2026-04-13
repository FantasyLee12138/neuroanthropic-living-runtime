# NALR 1.6 Plan

## Status

- Current line: `1.6` / `Chat Kernel V2`
- Root docs: `README.md`, `架构宣言.md`, `PLAN1.6.md`
- Public read model: route-first, module-first, hot/warm/cold

## Delivered

- Chat routes now distinguish `chat_micro`, `chat_fast`, `chat_standard`, `chat_deep`.
- Default chat path is sparse: one `CognitivePacket` call plus deterministic scheduling and validated state patching.
- Public memory semantics have converged on `hot / warm / cold`; `archive` remains only as migration compatibility.
- Observer `/models/status`, terminal summary, dashboard, and workbench now read `module_model_bindings`, `activation_set`, `memory_tiers_read`, `packet_summary`, `background_jobs`, and `deepen_reason`.
- Consolidation work is treated as background jobs instead of first-token blockers.
- Snapshot restore and monologue stream regressions found during migration have been fixed.
- Legacy root docs, including the old console PDF, have been moved into `archive/backup-docs/root`.

## In Flight

- Full-file soak across the broader observer / launcher / browser smoke matrix.
- Real 24h soak and longer long-run evidence.
- Cleanup of remaining internal compatibility seams such as legacy agent-tier helpers retained for migration.

## Deferred

- Multi-user serviceization.
- Wider distribution / installer work.
- Any further mechanism growth that does not materially improve the `1.6` V2 chat path or its observability.

## References

- [`README.md`](/Users/fantasylee/类脑架构/README.md)
- [`架构宣言.md`](/Users/fantasylee/类脑架构/架构宣言.md)
- [`docs/releases/1.6-acceptance-plan.md`](/Users/fantasylee/类脑架构/docs/releases/1.6-acceptance-plan.md)
- [`docs/releases/1.6-acceptance-matrix.md`](/Users/fantasylee/类脑架构/docs/releases/1.6-acceptance-matrix.md)
- [`docs/releases/1.6-release-notes.md`](/Users/fantasylee/类脑架构/docs/releases/1.6-release-notes.md)
- [`archive/backup-docs/INDEX.md`](/Users/fantasylee/类脑架构/archive/backup-docs/INDEX.md)
