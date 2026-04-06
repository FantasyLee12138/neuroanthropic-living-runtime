# 冲突控制器剩余 Phase 计划

## 摘要

当前 `ConflictMonitorAgent` 的主体已经到位：5 个分项分数、优先级裁决链、最多 3 个 pass 的重采样/妥协、`critical conflict` 熔断、`conflict_hot` 和基础 trace/observer 都已落地。离你这次的预设目标，剩下的不是再扩一轮 conflict score，而是把“冲突后的修复语义”补成一等公民。

按你刚才锁定的方向，剩余工作只收 conflict 尾差，不扩到 slow-state 公式和 typed CIL；repair 需要进入 `state + trace + observer`，并且做成跨轮 ledger + 状态机，`deadlock_fuse` 触发的 `safe_mode` 在恢复完成后自动退出。

## Remaining Phases

### Phase 1：补齐 Repair / Post-error 契约

- 在 [models.py](/Users/fantasylee/类脑架构/src/nalr/schemas/models.py) 新增一组一等公民类型：
  `ConflictPostErrorAdjustment`、`ConflictRepairLedgerEntry`、`ConflictRepairState`。
- 扩展 `RuntimeState`，不再只靠 `repair_mode` 和几个计数器表达 repair，而是显式保存：
  `repair_state`、`repair_ledger`、`conflict_safe_mode_owner`、`last_post_error_adjustment`。
- 扩展 `ConflictResolution` 和 `ActionDistributionState.conflict`，新增：
  `post_error_adjustment`、`repair_transition`、`repair_state_snapshot`、`repair_ledger_tail`。
- 保持现有 `score_conflict` / `trigger_control_escalation` / `request_resample` 三个 skill 名称不变；不新增新的顶层 stage，也不把 repair 变成第四个公开 conflict skill。

### Phase 2：把 Repair 做成跨轮状态机，而不是单轮打标

- 在 [controller.py](/Users/fantasylee/类脑架构/src/nalr/runtime/controller.py) 把当前 “compromise + deadlock_fuse + safe_mode” 的散点逻辑收束成 repair 状态机：
  `idle -> adjusting -> repairing -> cooling -> recovered`。
- `mark_post_error_adjustment` 的进入条件固定为：
  当前轮 conflict 处理改变了原本顶部动作，或强制套用了妥协模板，或触发了 deadlock fuse。
- 每条 ledger entry 必须记录：
  `round_id`、`reason`、`winning_priority`、`template`、`blocked_actions`、`top_action_before`、`top_action_after`、`safe_mode_delta`、`repair_stage_after`。
- `deadlock_fuse` 继续在连续 critical conflict 达标时触发，但进入 repair 后要显式标记 `conflict_safe_mode_owner="conflict"`，避免后续恢复逻辑和其他 safe-mode 来源混淆。
- 自动恢复规则固定为：
  `conflict_hot_rounds` 和 `repair_cooldown_rounds` 走完、repair 状态进入 `recovered` 后，自动清除 conflict 自己触发的 `safe_mode`、`repair_mode`、repair ownership 和活动中的 repair markers。
- 现有行为继续保留：
  `BehaviorPlausibilityGuard` 不并入 conflict；优先级链、妥协模板映射、3-pass 上限和 critical fuse 阈值不改。

### Phase 3：把 Repair 变成可观测、可验收的行为

- observer 的 conflict 视图需要从“看到结果”升级为“看到 repair 过程”：
  返回当前 `repair_stage`、最近一次 `post_error_adjustment`、ledger 摘要、是否处于 conflict-owned safe mode。
- round trace / why / contributions 需要能回放 repair 迁移，而不只是看到 `compromise` 和 `deadlock_fuse_triggered`。
- README 的 conflict gap 要更新成：
  conflict controller 已完整覆盖 repair/post-error 语义；剩余 gap 不再包含 conflict repair。
- 规格文档需要补一节 repair FSM 和 ledger 契约，明确进入、持续、冷却、恢复、清除的状态迁移。

## 测试与验收

- 单元测试：forced compromise 会产出 `post_error_adjustment`，并记录前后 top action。
- 单元测试：连续 critical conflict 会累积 repair ledger，触发 deadlock fuse，并进入 repairing/cooling/recovered 的完整状态迁移。
- 单元测试：冷却完成后，conflict 触发的 `safe_mode` 会自动退出；非 conflict 来源的 safe-mode 不能被误清。
- 集成测试：`/metrics/conflicts` 返回 repair stage、last adjustment、ledger summary、conflict-owned safe-mode 标记。
- 回归测试：现有 5 分项分数、优先级链、重采样阈值、妥协模板、`conflict_hot` 恢复逻辑全部保持通过。

## 已锁定的默认项

- 这轮只收 conflict controller 尾差，不把 slow-state 和 typed CIL 一起规划。
- repair 必须进入 `RuntimeState`、round trace、observer；不是内部私有标记。
- repair 采用跨轮 ledger + 状态机，不接受只做单轮 marker。
- 不新增顶层 pipeline stage；现有 conflict skill 名称保持稳定，只扩输出结构。
- `deadlock_fuse` 导致的 `safe_mode` 在恢复完成后自动退出。
