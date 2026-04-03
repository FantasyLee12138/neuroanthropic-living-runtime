# Conflict Controller v0.56

## Scope

本规范定义当前运行时内的完整 `ConflictMonitorAgent` 多 pass 冲突控制器。它覆盖：

- proposal 元数据输入契约
- 五个分项冲突分数
- 固定优先级裁决链
- 多轮重采样与妥协模板
- `critical conflict` 熔断与恢复
- trace / observer 暴露字段

本控制器位于 `conflict -> thalamus -> plausibility_guard` 中的 `conflict` 阶段内部，不新增新的顶层 pipeline stage。

## Proposal Signal Contract

`ProposalBundle` 在进入冲突控制器前必须带上以下字段：

- `priority_bucket`: `body_safety | budget_overload | relation_boundary | task_goal | immediate_desire | roaming`
- `control_domain`: `body | resource | relation | task | desire | dmn`
- `gated_actions`: 当前 proposal 认为应被压制的动作集合
- `risk_hints`: 结构化风险提示，如 `body_load`、`overload`、`boundary_level`、`relationship_risk`

运行时在 proposal 汇总后会补齐这些元数据。冲突分项不再只靠 `delta_p` 近似推断。

## Conflict Components

### 1. Proposal Divergence

衡量 proposal 顶部动作是否分叉，以及当前分布顶部是否足够接近到需要额外裁决。

工程定义：

```text
unique_ratio = (unique_top_actions - 1) / max(total_top_actions - 1, 1)
closeness = clip(1 - (top1 - top2) / 0.30)
proposal_divergence = clip(0.32 + 0.40 * unique_ratio + 0.35 * closeness)
```

### 2. Veto Tension

衡量 veto / gated action 与当前高概率动作之间的拉扯强度。

工程定义：

```text
gated_ratio = proposals_with_gated_actions / total_proposals
max_gated_mass = max(sum(p_raw[action] for action in gated_actions))
veto_ratio = proposals_with_veto / total_proposals
veto_tension = clip(0.12 + 1.20 * max_gated_mass + 0.35 * gated_ratio + 0.45 * veto_ratio)
```

### 3. Value Gap

衡量任务目标、即时欲望和漫游三路价值向量是否互相顶撞。

工程定义：

```text
value_gap = clip(
  max(
    vector_conflict(task_goal, immediate_desire),
    vector_conflict(task_goal, roaming),
    0.18 + 0.55 * priority_signal(task_goal)
         + 0.45 * max(priority_signal(immediate_desire), priority_signal(roaming))
  )
)
```

其中 `vector_conflict(left, right)` 采用归一化向量距离并在顶部动作不一致时追加 `0.18` mismatch bonus。

### 4. Relation Risk Gap

衡量关系边界 / 社交风险与 `connect`、`clarify` 等社交动作的冲突。

工程定义：

```text
relation_mass = p_raw[connect] + 0.35 * p_raw[clarify]
relation_risk_gap = clip(0.08 + 0.78 * priority_signal(relation_boundary) + 0.55 * relation_mass)
```

### 5. Body Gap

衡量身体/安全和预算/过载对高负荷动作的抑制需求。

工程定义：

```text
risky_body_mass = p_raw[plan] + p_raw[connect] + p_raw[wander]
body_gap = clip(
  0.12
  + 0.80 * max(priority_signal(body_safety), 0.92 * priority_signal(budget_overload))
  + 0.25 * min(1.0, risky_body_mass)
)
```

## Total Score

总分采用固定加权和：

```text
conflict_score =
  0.35 * proposal_divergence
+ 0.20 * veto_tension
+ 0.20 * value_gap
+ 0.15 * relation_risk_gap
+ 0.10 * body_gap
```

输出同时保留：

- `score`
- `total_score`
- `components`
- `priority_signals`
- `dominant_conflicts`
- `critical_conflict`

## Priority Adjudication Chain

固定优先级如下：

1. `body_safety`
2. `budget_overload`
3. `relation_boundary`
4. `task_goal`
5. `immediate_desire`
6. `roaming`

控制器按顺序扫描 `priority_signals`，高优先级一旦越过阈值就直接获胜，不再允许低优先级覆盖。

阈值：

- `body_safety` / `budget_overload` / `relation_boundary`: `>= 0.48`
- `task_goal` / `immediate_desire` / `roaming`: `>= 0.42`

默认动作压制：

- `body_safety`: 压制 `plan` / `connect` / `wander`
- `budget_overload`: 压制 `plan` / `connect` / `wander`
- `relation_boundary`: 压制 `connect`
- `task_goal`: 压制 `wander`
- `immediate_desire`: 压制 `plan`
- `roaming`: 压制 `plan`

若 `conflict_hot` 已激活，额外压制 `connect`。

## Multi-pass Control Flow

单轮 `tick()` 内最多执行 3 个 conflict pass。每个 pass 顺序固定：

1. 重新计算 `ConflictAssessment`
2. 按优先级链执行 `trigger_control_escalation`
3. 根据赢家优先级把 `action_scales` 写回 `gate` / `risk_suppressor` / `p_raw`
4. 按总分决定是否请求下一次重采样

重采样预算：

- `score < 0.65`: `0` 次
- `0.65 <= score < 0.82`: `1` 次
- `score >= 0.82`: `2` 次，并视为 `critical_conflict`

当 `conflict_hot` 已激活时，本 pass 会额外：

- 在总分上追加 `+0.04`
- 把 `body_safety` 信号至少抬到 `0.55`

## Compromise Templates

当达到允许的最大重采样次数后仍存在 unresolved conflict，会强制落入妥协模板。当前只保留四类模板：

- `body_first`
  - 映射来源：`body_safety`、`immediate_desire`
  - 典型动作调整：进一步压低 `plan` / `connect` / `wander`
  - 表达调整：更慢回复、更高 hedging、更强 repair tendency
- `budget_first`
  - 映射来源：`budget_overload`、`roaming`
  - 典型动作调整：压低高成本动作并抬升 `rest`
  - 表达调整：更短、更少自我暴露、轻度延迟
- `relation_first`
  - 映射来源：`relation_boundary`
  - 典型动作调整：压低 `connect`，保留更谨慎的 `clarify`
  - 表达调整：更高 hedging、更强 repair tendency、更低 self-disclosure
- `task_first`
  - 映射来源：`task_goal`
  - 典型动作调整：压低 `wander`，维持 `plan` / `respond`
  - 表达调整：更直接、更少 self-disclosure、更低 hedging

## Critical Conflict Circuit Breaker

### Trigger

满足以下条件时触发熔断：

- 当前轮 `critical_conflict == true`
- `critical_conflict_streak >= 3`

触发时写入：

- `critical_conflict_streak += 1`
- `conflict_hot_rounds = 5`
- `conflict_recovery_rounds = 5`

### Hot Behavior

`conflict_hot` 激活期间：

- 强制延续 5 轮保护窗口
- 高风险社交动作额外收缩，尤其是 `connect`
- `PFCAgent` 权重临时提升
- 输出风格改为 hesitation / uncertainty，更高 hedging、更高 repair tendency、更低 sharpness

### Recovery

若后续轮次没有新的 `critical_conflict`：

- 每轮把 `conflict_hot_rounds` 和 `conflict_recovery_rounds` 各减 `1`
- 计数减到 `0` 后清除 `last_compromise_template` 和 `last_conflict_priority`

## Trace And Observer Contract

`ActionDistributionState.conflict` 必须包含以下字段：

- `score`
- `total_score`
- `components`
- `priority_signals`
- `passes`
- `compromise`
- `critical_conflict`
- `critical_conflict_streak`
- `winning_priority`
- `circuit_breaker`

其中：

- `passes` 是每个 pass 的 `assessment + resolution + resample_requested`
- `compromise` 包含 `triggered`、`template`、`winning_priority`、`reason`
- `circuit_breaker` 包含 `active`、`triggered`、`hot_rounds_remaining`、`recovery_rounds_remaining`

`RuntimeController.conflict_timeline()` 和 observer `GET /metrics/conflicts` 必须返回：

- `round_id`
- `conflict_score`
- `components`
- `winning_priority`
- `template`
- `critical_conflict`
- `critical_conflict_streak`
- `conflict_hot_rounds`

## Non-goals

- 不把 `BehaviorPlausibilityGuard` 并入 `ConflictMonitorAgent`
- 不新增 CLI 命令或新的服务
- 不改变 `thalamus` 和 `plausibility_guard` 的顶层 stage 顺序
