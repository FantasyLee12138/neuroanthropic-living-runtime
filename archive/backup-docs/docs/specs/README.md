# Specs Index

本目录现在按“当前约束 / 历史设计 / 未来拆分”三类来理解，避免把已经落地的设计稿和仍然生效的规格混在一起读。

## 当前仍生效的规格

- [conflict-controller-v056.md](/Users/fantasylee/类脑架构/docs/specs/conflict-controller-v056.md)
  - 状态：`active`
  - 作用：当前冲突控制器的细粒度行为规格，包括 scoring、优先级裁决、妥协模板、熔断与 repair FSM。
  - 读取方式：把它当作冲突闭环的当前约束，而不是历史提案。
- [trace-canonical-storage.md](/Users/fantasylee/类脑架构/docs/specs/trace-canonical-storage.md)
  - 状态：`active`
  - 作用：当前 canonical trace、Parquet live read、observer 读取模型与 diagnostics 依赖字段的规格。
  - 读取方式：把它当作 trace 存储层与 observer 读取面的当前约束。
- [memory-cue-migration-and-compaction.md](/Users/fantasylee/类脑架构/docs/specs/memory-cue-migration-and-compaction.md)
  - 状态：`active`
  - 作用：当前 memory raw evidence、cue 归一化、identity family、compaction 与 migration 摘要的规格。
  - 读取方式：把它当作 memory 慢变量存储层的当前约束。
- [runtime-appraisal-and-state-evolution.md](/Users/fantasylee/类脑架构/docs/specs/runtime-appraisal-and-state-evolution.md)
  - 状态：`active`
  - 作用：当前 appraisal、状态更新、慢变量累积、identity 证据与 observer diagnostics 断点的规格。
  - 读取方式：把它当作“为什么会变 / 为什么没变”的当前约束。
- [observer-diagnostics-and-blockers.md](/Users/fantasylee/类脑架构/docs/specs/observer-diagnostics-and-blockers.md)
  - 状态：`active`
  - 作用：当前 diagnostics API、blocker 分类和 future dashboard 目标视图的规格。
  - 读取方式：把它当作 observer 诊断层的当前约束。

## 兼容入口

- [parquet-trace-memory-compaction.md](/Users/fantasylee/类脑架构/docs/specs/parquet-trace-memory-compaction.md)
  - 状态：`split_index`
  - 作用：兼容旧链接的拆分入口，指向新的 trace / memory 两份 active spec。

## 历史设计稿

以下文档保留，是为了说明“为什么这么做”和“实现时原本的切入点”，但它们不再是当前 source of truth：

- [../superpowers/specs/2026-04-04-terminal-streaming-design.md](/Users/fantasylee/类脑架构/docs/superpowers/specs/2026-04-04-terminal-streaming-design.md)
  - 状态：`historical_design`
  - 原因：终端流式输出路径、`assistant_token`、session persistence、one-shot printer 已经落地；其中部分假设已经被当前实现超出或替代。
- [../superpowers/specs/2026-04-04-memory-retrieval-tiering-design.md](/Users/fantasylee/类脑架构/docs/superpowers/specs/2026-04-04-memory-retrieval-tiering-design.md)
  - 状态：`historical_design`
  - 原因：memory retrieval budget、tier-aware recall、LRU 与 controller 传 budget 的主思路已落地；当前准确行为应以代码、测试和 active spec 为准。

## 当前清理结论

- `conflict-controller-v056.md`
  - 结论：未过时，继续保留在 `docs/specs/`。
  - 原因：它描述的是当前运行时仍在执行的细粒度规则，不是一次性设计草稿。
- `trace-canonical-storage.md`
  - 结论：未过时，作为新拆出的 active spec 保留。
  - 原因：它只承载 trace canonical storage 与 observer read model，边界已经足够清楚。
- `memory-cue-migration-and-compaction.md`
  - 结论：未过时，作为新拆出的 active spec 保留。
  - 原因：它只承载 memory cue hygiene、compaction 和 migration，边界已经独立。
- `parquet-trace-memory-compaction.md`
  - 结论：不再作为完整规格维护，保留为 split index。
  - 原因：它的原始职责已经拆分到两份 active spec。
- `runtime-appraisal-and-state-evolution.md`
  - 结论：未过时，作为新补齐的 active spec 保留。
  - 原因：它把此前散落在 README、overview 和 trace 字段里的闭环逻辑正式收口成规格。
- `observer-diagnostics-and-blockers.md`
  - 结论：未过时，作为新拆出的 active spec 保留。
  - 原因：它把 diagnostics API 输出、blocker taxonomy 和 dashboard 目标视图单独收口，避免继续挤在闭环总 spec 里。
- `2026-04-04-terminal-streaming-design.md`
  - 结论：作为历史设计保留，不应继续被当成当前规格。
- `2026-04-04-memory-retrieval-tiering-design.md`
  - 结论：作为历史设计保留，不应继续被当成当前规格。

## 下一轮建议拆分

以下不是本轮强制拆分，而是已经明确出现“继续长下去会失控”的地方：

- dashboard
  - 建议单独成 spec，而不是继续附着在 observer API 文档里。
  - 目标主题：
  - `observer-dashboard-views.md`

## Source Of Truth 顺序

建议读文档时遵循这个顺序：

1. `docs/specs/*.md` 中标记为 `active` 的当前规格
2. `docs/architecture/overview.md`
3. `docs/decisions/*.md`
4. `README.md`
5. `docs/specs/*.md` 中标记为 `split_index` 的兼容入口
6. `docs/superpowers/specs/*.md` 中标记为 `historical_design` 的历史设计稿

如果 active spec、代码和历史设计稿发生冲突，以 active spec 和代码为准。
