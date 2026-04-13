# 2026-04-08 Dialogue Runtime Acceptance Report

## Structural Checks

structural: 3/3 passed
- Observer dialogue runtime integration subset: pass in 40.45s
- Terminal bridge direct dialogue subset: pass in 20.79s
- Workbench launcher help: pass in 0.51s

## Live Probe

### Preflight
- doc_baseline: pass - all 1.5 baseline docs are present
- dashboard: pass - status=200
- service_status: pass - healthy=True
- runtime_bootstrap: pass - session_attached=False
- models_status: pass - default_creds=False/False/False, gpt54=False
- settings: pass - observer settings readable
- default_remote_credentials: blocked - mixed routing cannot be treated as a live-provider pass without DeepSeek and ARK credentials
- runtime_start: pass - running=True

### Dialogue
- console_identity: pass - assistant_len=8
- console_companion: pass - assistant_len=38
- console_task: pass - assistant_len=45
- web_session_start: pass - status=200
- web_identity: pass - events=assistant_token,sidebar_snapshot,assistant_final
- web_companion: pass - events=assistant_token,sidebar_snapshot,assistant_final
- web_task: pass - events=assistant_token,sidebar_snapshot,assistant_final

### Reset
- persona_reset: pass - state/session/why reset
- post_reset_dialogue: pass - fresh session answered after reset

### Concurrency
- multi_session_parallel_turns: pass - elapsed=6.71s dashboard=0.021s service=0.002s

### Approval
- approval_round_trip: pass - post_approve=sidebar_snapshot,sidebar_snapshot,tool_result,assistant_final

### Fallback
- automatic_failover: blocked - GPT54_FALLBACK_API_KEY not present

## Blockers
- default mixed routing credentials are not fully available in the current environment
- GPT54_FALLBACK_API_KEY is not set, so GPT5.4 fallback validation is blocked

overall: fail
