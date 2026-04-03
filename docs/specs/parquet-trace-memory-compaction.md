# Parquet Trace Export And Memory Compaction

## Overview

NALR 将 trace 收口为双层语义：

- JSON / JSONL：append-only replay / audit / debug artifacts
- Parquet：round / skill / command / repair 的默认 live read model

- round trace：`.alive/traces/rounds/round_<id>.json`
- round stream：`.alive/traces/round_traces.jsonl`
- skill stream：`.alive/traces/skills/skill_traces.jsonl`
- command stream：`.alive/traces/command_traces.jsonl`
- memory raw evidence：`.alive/memory/raw/episodic_events.jsonl`

runtime 在 `tick()` / `apply_command()` 结束时会同步刷新 canonical Parquet live read tables；`alive trace export parquet` 继续保留，但角色改为全量 rebuild / 历史 backfill。

JSON 写成功但 Parquet 同步失败时，runtime 不阻断本轮，而是把 trace storage 标记为 `degraded`；公共读取端仅在该状态下回退 JSON，并暴露 `read_source` / `trace_sync_state` / `degraded_reason`。

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

## Memory Raw Evidence

每次 `MemoryStore.ingest_event(...)` 都会追加一条 raw evidence：

- `session_id`
- `round_id`
- `recorded_at`
- `recorded_date`
- `cue`
- `source`
- `target`
- `content`
- `valence`

位置：

- `.alive/memory/raw/episodic_events.jsonl`

## Memory Compaction

兼容聚合文件仍保留：

- `.alive/memory/episodic_hot.json`
- `.alive/memory/episodic_warm.json`
- `.alive/memory/episodic_archive.json`

richer compaction artifact 写入目录：

- `.alive/memory/episodic_hot/`
- `.alive/memory/episodic_warm/`
- `.alive/memory/episodic_archive/`

维护命令：

```bash
alive memory compact
alive memory sample --tier hot --limit 5
alive memory sample --tier warm --limit 3 --cue coffee
alive memory sample --tier archive --limit 2
```

### Tier Policy

- `hot`：最近 `<=500` rounds，保留 detail preview 与 evidence refs
- `warm`：`501~3000` rounds，保留 gist、聚合统计和 evidence refs
- `archive`：`>3000` rounds，保留 summary chunk、sampled evidence refs、hash digest

### Retention Policy

- hot evidence：7 天
- warm evidence：30 天
- archive：只保留被引用 evidence 与 hash 摘要

## Sampling Read Path

`alive memory sample` 只读取 compaction artifact，不读取 runtime 热态聚合文件。

这保证：

- sampling 不影响 runtime 读路径
- `memory_top()` 继续服务现有 CLI / observer 兼容面
- compacted artifact 可以独立做离线分析和人工 spot check

## Non-Goals

本期不包含：

- trace hot / warm / archive lifecycle
- 后台异步 exporter / compactor
- scheduler 驱动的自动 flush
