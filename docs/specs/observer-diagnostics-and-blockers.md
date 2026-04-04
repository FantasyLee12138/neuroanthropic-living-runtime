# Observer Diagnostics And Blockers

## Status

- 状态：`active`
- 角色：当前 observer diagnostics API、blocker 分类和 dashboard 目标视图的生效规格
- 读取优先级：高于 README 和占位说明，应与 `services/observer/api/app.py`、`src/nalr/runtime/controller.py` 一起阅读

## Scope

本规格只覆盖 observer 中专门回答“哪里没链接、为什么没变化、当前被什么卡住”的诊断层，不覆盖全部 observer API。

当前聚焦三类问题：

- 当前 trace / state 已经暴露了哪些诊断输出
- blocker 在 API 层如何分类与返回
- dashboard 未来应该把哪些 API 视图可视化

## Current Diagnostics Endpoints

当前 observer 已暴露以下 diagnostics 入口：

- `GET /diagnostics/state-delta`
- `GET /diagnostics/identity-blockers`
- `GET /diagnostics/cue-fragmentation`
- `GET /diagnostics/run-contamination`
- `GET /diagnostics/why-no-change/{round_ref}`
- `GET /diagnostics/migration`

这些接口当前都直接由 `RuntimeController` 提供只读 payload，不参与写路径。

## Shared Response Contract

### Storage Payload

diagnostics 响应当前都应带上 `storage` 或等价读源信息，以说明读取来自哪里。

最低预期包括：

- `read_source_default`
- `storage_state`
- `parquet_live_ready`
- `degraded_reason`
- `last_sync_at`

设计约束：

- observer 不能静默隐藏读取退化
- 当结果来自 degraded 路径时，调用方必须能看见读源说明

### Round-Coupled Diagnostics

依赖 round trace 的 diagnostics，当前主要建立在这些字段上：

- `appraisal_snapshot`
- `state_delta_before_clip`
- `state_delta_after_clip`
- `delta_suppression_reason`
- `run_context`
- `run_contamination_detected`
- `identity_evidence_score`
- `identity_trigger_blockers`
- `temperament_window_summary`

## Endpoint Shapes

### `/diagnostics/state-delta`

用途：

- 查看最近窗口内每轮 appraisal 输入与状态 delta
- 对比 clip 前后变化

当前输出结构：

- `points`
- `storage`

每个 `point` 当前包括：

- `round_id`
- `sampled_action`
- `appraisal_snapshot`
- `state_delta_before_clip`
- `state_delta_after_clip`
- `delta_suppression_reason`

主要用途：

- 判断是输入太弱、clip 过强，还是状态变了但表达不明显

### `/diagnostics/identity-blockers`

用途：

- 查看当前 identity evidence 是否足够形成稳定 `display_name`

当前输出结构：

- `display_name`
- `score`
- `naming_signal`
- `continuity_signal`
- `blockers`
- `clusters`
- `storage`

主要用途：

- 解释 identity 还没稳定的原因
- 让 naming / continuity / relation 稳定度不足不再停留在猜测层

### `/diagnostics/cue-fragmentation`

用途：

- 查看 cue family 是否被写碎，尤其是 identity family 是否仍然离散

当前输出结构：

- `families`
- `storage`

每个 family 当前包括：

- `family`
- `members`
- `sources`
- `count`
- `fragmented`

主要用途：

- 排查 stable prior / habit / memory 是否仍在被脏 cue 污染
- 确认 cue normalization 是否真正把 identity 家族合并了

### `/diagnostics/run-contamination`

用途：

- 查看最近窗口内聊天轮次是否被旧 task run 污染

当前输出结构：

- `points`
- `storage`

每个 `point` 当前包括：

- `round_id`
- `scenario`
- `run_contamination_detected`
- `drive_source`
- `current_goal`

主要用途：

- 排查为什么当前聊天还在被历史 run 的目标拉偏

### `/diagnostics/why-no-change/{round_ref}`

用途：

- 给某一轮返回最保守的“为什么没变化”解释

当前输出结构：

- `round_id`
- `failure_mode`
- `blockers`
- `appraisal_snapshot`
- `state_delta_before_clip`
- `state_delta_after_clip`
- `storage`

这是当前最直接的断点 API，不要求调用方自己理解整条 round trace。

### `/diagnostics/migration`

用途：

- 返回当前 `.alive` schema migration 的摘要

当前输出结构：

- `schema_version`
- `removed_aliases_count`
- `merged_cue_clusters_count`
- `preserved_memory_count`
- `identity_evidence_rebuilt`
- `migrated_at`
- `storage`

主要用途：

- 区分“当前没变化”到底是新逻辑没生效，还是旧污染还没清干净

## Failure Modes

当前 `why_no_change()` 的标准 failure mode 为：

- `not_called`
- `called_neutral`
- `updated_but_clipped`
- `updated_but_not_expressed`

解释约束：

- `not_called`
  - 含义：当前轮没有形成有效 appraisal 输入
- `called_neutral`
  - 含义：执行了 appraisal，但输入强度不足以推动状态变化
- `updated_but_clipped`
  - 含义：变化产生了，但被 clip / suppression 压回
- `updated_but_not_expressed`
  - 含义：内部已有变化，但还没有跨过表达阈值，或受运行态污染导致文本层未显化

## Blocker Taxonomy

### Core Blockers

当前已使用或应优先保留的 blocker 包括：

- `no_appraisal_input`
- `run_contamination`
- `delta_clipped`
- `expression_threshold_not_met`
- `identity_evidence_insufficient`
- `naming_signal_weak`
- `continuity_signal_weak`
- `relation_stability_low`

### Source Mapping

这些 blocker 当前主要来自两条链：

- `why_no_change()` 根据 appraisal 和 delta 推导出的 blocker
- `identity_evidence()` / `IdentityRuntime` 返回的 rejection reasons

设计约束：

- blocker 应尽量使用稳定的 machine-readable key，而不是自由文本句子
- 同一响应内的 blocker 应去重
- observer 展示层可以补解释文字，但不应改写 key

### Blocker Priorities

当前展示优先级建议保持：

1. 运行态污染类
2. appraisal 缺失类
3. delta suppression / expression threshold 类
4. identity evidence 不足类
5. migration / cue hygiene 类

理由：

- 先解释“根本没走对链路”
- 再解释“走了但输入太弱”
- 最后解释“identity / memory 这类慢变量为什么还没形成”

## Dashboard Targets

### Current State

当前 `/dashboard` 仍是静态占位页，不构成完整诊断面。

因此本规格中的 dashboard 部分描述的是未来目标视图，而不是已落地实现。

### Minimum Future Views

未来 dashboard 至少应覆盖以下视图：

- `state delta timeline`
  - 展示最近窗口内 `appraisal_snapshot`、before/after delta、suppression reason
- `identity blockers`
  - 展示 `display_name`、`identity_score`、`naming_signal`、`continuity_signal`、clusters、blockers
- `cue fragmentation`
  - 展示 family、member 数量、source 分布、是否 fragmented
- `run contamination`
  - 展示 `scenario`、`drive_source`、`current_goal`、污染标记
- `why-no-change inspector`
  - 输入 round ref 后展示 failure mode 与 blockers
- `migration summary`
  - 展示 schema version、merged clusters、preserved memory 和 rebuild 状态

### Suggested View Composition

建议 dashboard 分成三块：

- `State Diagnostics`
  - state delta timeline
  - why-no-change inspector
- `Identity Diagnostics`
  - identity blockers
  - cue fragmentation
- `Runtime Hygiene`
  - run contamination
  - migration summary

### Dashboard Interaction Rules

未来 dashboard 应遵守：

- 所有图表都能回跳到对应 round 或 trace
- 所有 blocker 都能显示来源字段，而不是只显示标签
- 视图默认按最近窗口排序，但允许调整 `window`
- 不应把 API 的原始 key 丢掉，只显示润色后的文案

## Non-Goals

本规格当前不覆盖：

- 完整 dashboard 前端 UI 视觉设计
- metrics / conflicts / authenticity 等非 diagnostics 图表
- 自动 remediation 或自动修复动作
- diagnostics API 的权限分层
