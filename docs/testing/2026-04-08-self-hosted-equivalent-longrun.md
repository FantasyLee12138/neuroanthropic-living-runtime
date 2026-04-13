# NALR 1.5 Self-Hosted Equivalent Long-Run Report

## Run Meta

- Date: `2026-04-08`
- Scope: bounded equivalent acceptance evidence for the self-hosted / controlled-learning release path
- Status: `PARTIAL` for real 24h soak, `PASS` for the bounded report surface
- Non-claim: this report does **not** assert a real 24h run happened

## What This Covers

- Terminal-facing acceptance report output via `nalr eval acceptance-report`
- Release-level controlled-learning summary in `release_15.controlled_learning`
- Bounded long-run acceptance evidence via `tests/longrun/test_authenticity_acceptance.py`

## Why This Exists

The 1.5 self-hosted line needed an equivalent long-running acceptance report path instead of only a short launcher/API smoke. This artifact records the bounded replacement:

- the CLI can now print the acceptance report directly
- the runner can include a self-hosted group that exercises the long-run acceptance path
- the true 24h soak remains pending and is still tracked as in-flight

## Acceptance Surface

The report output is expected to include:

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

This is an equivalent long-run acceptance artifact, not proof of a 24h soak. The 24h line in the matrix and ledger remains partial until a real long-duration run is captured.
