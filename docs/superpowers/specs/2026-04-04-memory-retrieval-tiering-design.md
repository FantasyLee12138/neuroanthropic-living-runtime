# 记忆检索分层优化设计

## Status

- 状态：`historical_design`
- 当前结论：主设计目标大体已落地，但这份文档不再是当前 memory 行为的唯一准绳

哪些已经落地：

- controller 会按显著性传入 memory retrieval budget
- `MemoryStore.recall()` / `recall_strength()` 共享 tier-aware 检索语义
- tier budget 已进入测试覆盖
- 检索结果已带本地缓存，重复查询不会每次都重扫底层 tier

哪些地方已经与原设计稿不完全一致：

- 当前实现的缓存键和适用范围以实际代码为准，不应再机械理解为只覆盖文中举例的最小 hot-only 形态
- 当前 memory 规格已经同时纳入 cue 清洗、identity cue 聚类和 migration 摘要，这些内容不在本文最初范围内

现在如果要确认真实行为，应优先看：

- `src/nalr/memory/store.py`
- `src/nalr/runtime/controller.py`
- `tests/unit/test_memory_store.py`
- `tests/unit/test_runtime_controller.py`
- `docs/specs/memory-cue-migration-and-compaction.md`

## 背景

当前运行时在构建 memory context 时会直接调用 `MemoryStore.recall()` 和 `MemoryStore.recall_strength()`。这两条路径都会按 `hot -> warm -> archive` 顺序读取记忆层，且没有进程内缓存。对低显著性输入而言，这会让本可在 `hot` 层完成的查询仍然支付跨层检索成本，也会让同一 `cue` 的重复查询反复执行底层扫描。

本次优化目标是在不改变现有记忆语义的前提下，缩短低显著性和重复查询的热路径耗时，将记忆检索主路径控制在 100ms 以内。

## 目标

- 低显著性场景仅检索 `hot` 层，不访问 `warm` / `archive`
- 复用现有显著性信号，不新增独立的 memory salience 阈值
- 给 `hot` 层检索增加本地 LRU 缓存，减少重复底层检索
- 高显著性场景保持当前回落行为，仍可命中 `warm` / `archive`
- 在记忆写入和分层同步后正确失效缓存，避免读到旧结果

## 非目标

- 不引入新的外部缓存或持久化缓存
- 不重写 `HippocampusAgent` 的技能协议
- 不更改现有 memory tier compact artifact 的格式
- 不在本次实现真正引入新的 ANN 引擎；优化对象是当前统一检索入口及其未来 ANN 挂接点

## 现状总结

- `RuntimeController._probe_context()` 与主运行时上下文构建都会读取 `MemoryStore.recall_strength()` 和 `MemoryStore.recall()`
- `MemoryStore.recall()` 当前固定按 `hot -> warm -> archive` 顺序查找
- `MemoryStore.recall_strength()` 仅查看 `hot` 层，和 `recall()` 的层级语义不一致
- 当前没有统一的“检索预算”概念，controller 也没有把显著性结果传递给 memory store

## 方案

### 1. 统一检索入口

在 `MemoryStore` 内新增统一的检索入口，由它同时负责：

- 根据显著性决定允许访问的 tier 集合
- 优先查询 `hot`
- 在允许时回落 `warm` / `archive`
- 统一返回 `recall` payload 和 `strength`

`recall()` 与 `recall_strength()` 改为复用这个统一入口，避免层级语义分叉。

### 2. 复用现有显著性信号

不新增新的 memory-only 阈值。controller 在构建 context 时，复用现有显著性链路与 `config/thresholds.yaml` 中的显著性阈值，得到一个 memory retrieval budget：

- 低显著性：`["hot"]`
- 其他场景：`["hot", "warm", "archive"]`

controller 只负责把预算传给 memory store，不承载 tier 检索细节。

### 3. Hot 层本地 LRU

在 `MemoryStore` 内加入进程内 LRU 缓存，键至少包含：

- `cue`
- tier budget（例如 `hot_only` / `full_tiers`）

缓存内容为统一检索入口的结果对象。命中时直接返回，跳过底层 tier 扫描。

缓存范围仅覆盖包含 `hot` 的热路径，不为 `warm` / `archive` 建独立长期缓存，避免扩大失效面。

### 4. 缓存失效

以下操作后，失效对应 `cue` 的缓存项：

- `ingest_event()` 写入或更新 cue
- `apply_noninteractive_proposal()` 中发生 memory consolidation
- `_sync_memory_tiers()` 将 cue 同步到 warm / archive

若是全量 tier compact 或文件修复路径，可直接清空 LRU。

## 数据流

1. controller 构建 context
2. 复用现有显著性信号，生成 memory retrieval budget
3. controller 调用 `MemoryStore` 统一检索入口
4. memory store 先查 hot-LRU
5. 未命中时按 budget 允许的 tier 顺序查询
6. 返回 recall payload，并在适用时写回 LRU

## 接口调整

计划保持外部调用点尽量少：

- `RuntimeController`：只新增 memory retrieval budget 的传递
- `MemoryStore`：
  - 新增统一检索辅助方法
  - `recall()` 接收可选 tier budget
  - `recall_strength()` 复用统一检索结果

CLI、dream、现有 agent 技能继续走现有 `recall()` / `recall_strength()` 对外接口，避免扩散改动。

## 性能约束

100ms 目标针对主路径，即：

- 低显著性查询命中 hot-only 路径
- 相同 `cue` 的重复查询命中 hot LRU

本次通过单元测试验证“跳过跨层访问”和“重复查询不重复执行底层检索”，并补一个轻量耗时测试，确认热路径明显低于 100ms。该测试只作为回归护栏，不做严格硬实时承诺。

## 测试策略

### 单元测试

- 低显著性 budget 下仅访问 `hot`，不读取 `warm` / `archive`
- 高显著性 budget 下可回落到 `warm` / `archive`
- 重复 hot 查询命中 LRU，底层 tier 扫描次数不增加
- `ingest_event()` / consolidation / tier sync 后对应 cache key 被失效
- `recall()` 与 `recall_strength()` 对同一 budget 返回一致语义

### 集成回归

- controller 构建 context 时会把低显著性输入收敛到 `hot`
- 现有高显著性或强提示输入不丢失 warm/archive 召回能力

## 风险与缓解

### 风险 1：显著性预算判断过于激进

如果低显著性覆盖面过大，可能让本该命中 warm/archive 的查询退化为 miss。

缓解：

- 仅复用现有阈值，不新增更激进的裁剪条件
- 保留高显著性全层回落行为
- 用回归测试覆盖 warm/archive 命中场景

### 风险 2：缓存失效不完整

如果写路径后未失效，可能返回旧 strength 或旧 tier。

缓解：

- 将失效逻辑集中在 `MemoryStore`
- 对每条会影响 cue 内容的写路径补测试

### 风险 3：接口改动扩散

如果 controller、agent、CLI 各自理解 tier budget，后续维护会分叉。

缓解：

- 预算解释权放在 `MemoryStore`
- controller 只传预算，不实现 tier 查找策略

## 预期改动文件

- 修改 `src/nalr/memory/store.py`
- 修改 `src/nalr/runtime/controller.py`
- 修改 `tests/unit/test_memory_store.py`
- 视情况补充 `tests/unit/test_runtime_controller.py` 或相邻运行时测试
