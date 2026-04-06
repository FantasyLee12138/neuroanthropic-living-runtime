# Trace Canonical Storage

## Status

- 状态：`active`
- 角色：当前 trace canonical 写入、Parquet live read 和 observer 读取模型的生效规格
- 读取优先级：高于历史设计稿，应与 `docs/architecture/overview.md` 和当前测试一起阅读

## Overview

NALR 将 trace 收口为双层语义：

- JSON / JSONL：append-only replay / audit / debug artifacts
- Parquet：round / skill / command / repair 的默认 live read model

关键路径：

- round trace：`.alive/traces/rounds/round_<id>.json`
- round stream：`.alive/traces/round_traces.jsonl`
- skill stream：`.alive/traces/skills/skill_traces.jsonl`
- command stream：`.alive/traces/command_traces.jsonl`

runtime 在 `tick()` / `apply_command()` 结束时会同步刷新 canonical Parquet live read tables；`alive trace export parquet` 继续保留，但角色改为全量 rebuild / 历史 backfill。

JSON 写成功但 Parquet 同步失败时，runtime 不阻断本轮，而是把 trace storage 标记为 `degraded`；公共读取端仅在该状态下回退 JSON，并暴露 `read_source` / `trace_sync_state` / `degraded_reason`。observer 的 `/diagnostics/*` 端点也遵循同一读取模型，并把 `storage` 信息一并返回。

## Canonical Trace Metadata

三类 trace row 统一带以下字段：

- `session_id`
- `recorded_at`
- `recorded_date`

规则：

- 新写入数据直接落这三个字段
- 历史 trace 导出时允许 backfill
- 缺 `session_id` 的旧数据使用 `legacy`
- 缺时间字段时优先使用文件 `mtime`，再退回导出时间

## Parquet Outputs

导出目录：

- `.alive/traces/parquet/round_trace.parquet`
- `.alive/traces/parquet/skill_trace.parquet`
- `.alive/traces/parquet/command_trace.parquet`

rebuild / backfill 命令：

```bash
alive trace export parquet
alive trace export parquet --since-round 100
alive trace export parquet --overwrite
```

### `round_trace.parquet`

这是 analytics-oriented flatten 表，不承担原样回放；完整 payload live read 默认走 `round_canonical.parquet`。

一行对应一个：

- `(session_id, round_id, stage, agent_name, action)`

由 `proposal_summaries[].delta_p` 展开得到，主要列包括：

- `session_id`
- `recorded_at`
- `recorded_date`
- `round_id`
- `scenario`
- `mode`
- `sampled_action`
- `stage`
- `agent_name`
- `action`
- `top_action`
- `selected`
- `confidence`
- `sigma_scale`
- `weight_applied`
- `delta_p`
- `resample_count`
- `resample_idx`
- `conflict_score`
- `plausibility_fail_score`

嵌套字段先保留为 JSON string 列：

- `tags_json`
- `distribution_state_json`
- `state_snapshot_json`
- `render_plan_json`
- `gate_decisions_json`
- `rendered_expression_json`

round canonical payload 中当前还会保留以下诊断字段，供 `/diagnostics/state-delta`、`/diagnostics/run-contamination`、`/diagnostics/why-no-change/{round_ref}` 等接口直接读取：

- `appraisal_snapshot`
- `state_delta_before_clip`
- `state_delta_after_clip`
- `delta_suppression_reason`
- `run_context`
- `run_contamination_detected`
- `identity_evidence_score`
- `identity_trigger_blockers`
- `temperament_window_summary`

### `skill_trace.parquet`

一行一个 skill invocation，主要列包括：

- `session_id`
- `recorded_at`
- `recorded_date`
- `round_id`
- `skill_name`
- `owner_module`
- `latency_ms`
- `cost_class`
- `input_hash`
- `output_hash`
- `failure_policy_applied`
- `degraded`
- `seed_ref`

### `command_trace.parquet`

一行一个 command mutation，主要列包括：

- `session_id`
- `recorded_at`
- `recorded_date`
- `command`
- `applied`
- `scope`
- `delta_json`
- `ttl`
- `risk_note`
- `rollback_hint`
- `operator_level`
- `rollback_available`
- `before_state_hash`
- `after_state_hash`

## Observer Read Model

当前 observer 直接建立在 canonical trace 之上，相关读取面包括：

- `/trace/{round_id}`
- `/why/{round_id}`
- `/contributions/{round_id}`
- `/metrics/conflicts`
- `/metrics/mode-switches`
- `/metrics/timeline`
- `/metrics/heatmap`
- `/replay/{round_id}`
- `/why-not/{round_id}/{action}`
- `/diagnostics/state-delta`
- `/diagnostics/run-contamination`
- `/diagnostics/why-no-change/{round_ref}`

设计约束：

- observer 保持只读，不参与写路径
- 当 Parquet live read 健康时，公共读取默认优先走 canonical tables
- 当存储退化时，返回结果必须带上 `storage` 说明，而不是静默切换读取源

## Non-Goals

本期不包含：

- trace hot / warm / archive lifecycle
- 后台异步 exporter
- scheduler 驱动的自动 flush
- dashboard 视图本身的交互规格
