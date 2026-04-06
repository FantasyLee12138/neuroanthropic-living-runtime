# Memory Cue Migration And Compaction

## Status

- 状态：`active`
- 角色：当前 memory raw evidence、cue 归一化、identity 聚类、compaction 与迁移摘要的生效规格
- 读取优先级：高于历史设计稿，应与 `src/nalr/memory/store.py` 和相关测试一起阅读

## Overview

当前 memory 规格关注四件事：

- raw evidence 如何写入 `.alive`
- cue 在写入前如何清洗与归一化
- hot / warm / archive compaction 如何组织
- 旧 `.alive` 如何迁移到当前 schema

关键路径：

- raw evidence：`.alive/memory/raw/episodic_events.jsonl`
- 兼容聚合文件：`.alive/memory/episodic_hot.json`、`.alive/memory/episodic_warm.json`、`.alive/memory/episodic_archive.json`
- richer compaction artifact：`.alive/memory/episodic_hot/`、`.alive/memory/episodic_warm/`、`.alive/memory/episodic_archive/`
- migration status：`.alive/memory/memory_migration_status.json`

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

## Cue Hygiene And Normalization

cue 不再把任意整句输入直接当作长期记忆键。写入前会先做清洗与归一化：

- 控制字符、终端 escape、明显的命令残片会被过滤
- 无效 freeform cue 会在 ingest 前被丢弃，避免污染 stable prior / alias / habit
- 身份询问会聚类到固定 family，而不是分裂成大量离散整句

当前内建的 identity family 包括：

- `identity:self_probe`
- `identity:name_probe`
- `identity:name_origin`

设计约束：

- alias 不能再直接吸收整句用户问话
- cue 清洗必须同时适用于新写入事件和 legacy 迁移
- `identity_evidence()` 读取的 cluster / naming / continuity 信号应来自归一化后的 cue，而不是原始脏输入

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

## Memory Migration And Identity Evidence

为兼容旧 `.alive`，memory runtime 现在维护：

- schema version：`memory_migration_status.json`
- 基础迁移报告：`schema_version`、`removed_aliases_count`、`merged_cue_clusters_count`、`preserved_memory_count`、`identity_evidence_rebuilt`、`migrated_at`

首次启动新逻辑时，迁移会：

- 重新清洗 stable priors / habits / episodic hot-warm-archive 中的 legacy cue
- 合并 identity family 的碎片 cue
- 重建 `identity_evidence()` 所需的 cluster / naming / continuity 信号

observer `/diagnostics/migration` 与 runtime `migration_report()` 直接返回这份摘要；它的目标是保守说明“清理发生过什么”，不是替代更细粒度的逐条迁移审计。

## Relation To Retrieval Budget

memory retrieval budget、tier-aware recall 和 LRU 热路径缓存已经落地，但它们不再单独定义在本文件中。

如果要确认检索预算与缓存的真实行为，应优先看：

- `src/nalr/memory/store.py`
- `src/nalr/runtime/controller.py`
- `tests/unit/test_memory_store.py`
- `tests/unit/test_runtime_controller.py`
- `docs/superpowers/specs/2026-04-04-memory-retrieval-tiering-design.md`

## Non-Goals

本期不包含：

- 证据指针-only archive compaction 的完整落地
- ANN / vector retrieval 引擎
- 更细粒度的 relation / habit / legacy memory 迁移审计报表
