# NALR 1.5 Self-Hosted Bounded Long-Run Acceptance Report

## Run Meta

- Date: `2026-04-08`
- Scope: bounded acceptance evidence for the self-hosted / controlled-learning release path
- Status: `PARTIAL` for real 24h soak, `PASS` for the bounded report surface
- Non-claim: this report does **not** assert a real 24h run happened

## What This Covers

- Terminal-facing acceptance report output via `nalr eval acceptance-report`
- Release-level controlled-learning summary in `release_15.controlled_learning`
- Canonical bounded report surface in `release_15.bounded_long_run`
- Bounded long-run acceptance evidence via `tests/longrun/test_authenticity_acceptance.py`

## Why This Exists

The 1.5 self-hosted line needed a bounded long-running acceptance report path instead of only a short launcher/API smoke. This artifact records the bounded replacement:

- the CLI can now print the acceptance report directly
- the runner can include a self-hosted group that exercises the bounded long-run acceptance path
- the true 24h soak remains pending and is still tracked as in-flight

## Acceptance Surface

The report output is expected to include:

- `release_15.bounded_long_run.window`
- `release_15.bounded_long_run.rounds_considered`
- `release_15.bounded_long_run.long_run_prior`
- `long_run_prior`
- `release_15.controlled_learning.learning_mode`
- `release_15.controlled_learning.allowed_network_domains`
- `release_15.controlled_learning.writable_roots`
- `release_15.controlled_learning.knowledge_roots`
- `release_15.controlled_learning.learning_log_dir`
- `release_15.controlled_learning.trace_external_learning`

## Commands

- `python3 /Users/fantasylee/类脑架构/scripts/acceptance_15.py --run --group self-hosted`
- `python3 -m nalr.cli.app eval acceptance-report --window 20`

## Status Note

This is a bounded long-run acceptance artifact, not proof of a 24h soak. The 24h line in the matrix and ledger remains partial until a real long-duration run is captured.
