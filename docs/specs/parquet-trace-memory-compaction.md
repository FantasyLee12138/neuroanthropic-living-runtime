# Parquet Trace Export And Memory Compaction

## Status

- 状态：`split_index`
- 角色：兼容旧链接的拆分入口，不再承载完整 active spec 正文

这份文档已拆成两份当前生效的规格：

- [trace-canonical-storage.md](/Users/fantasylee/类脑架构/docs/specs/trace-canonical-storage.md)
  - 覆盖 canonical trace、Parquet live read、observer 读取模型、diagnostics 依赖字段
- [memory-cue-migration-and-compaction.md](/Users/fantasylee/类脑架构/docs/specs/memory-cue-migration-and-compaction.md)
  - 覆盖 raw evidence、cue 归一化、identity family、memory compaction、schema migration 摘要

## Why Split

旧文档同时承载了两类已经明显分叉的语义：

- trace canonical storage / observer read model
- memory cue hygiene / compaction / migration

继续放在同一份 spec 里会导致：

- trace 变更和 memory 变更互相打断阅读
- observer diagnostics 依赖字段和 memory migration 摘要混在一起，不利于维护
- 历史设计稿和当前实现对账时，边界越来越难讲清

## Reading Order

如果你是第一次进入这组文档，建议按下面顺序读：

1. [trace-canonical-storage.md](/Users/fantasylee/类脑架构/docs/specs/trace-canonical-storage.md)
2. [memory-cue-migration-and-compaction.md](/Users/fantasylee/类脑架构/docs/specs/memory-cue-migration-and-compaction.md)
3. [docs/architecture/overview.md](/Users/fantasylee/类脑架构/docs/architecture/overview.md)

如果你是从旧链接跳过来的，把这份文件当作跳转页即可。
