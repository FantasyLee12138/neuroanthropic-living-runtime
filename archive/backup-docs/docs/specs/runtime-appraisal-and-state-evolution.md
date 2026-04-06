# Runtime Appraisal And State Evolution

## Status

- 状态：`active`
- 角色：当前对话输入如何驱动 appraisal、状态更新、慢变量累积、identity 证据和 diagnostics 的生效规格
- 读取优先级：高于 README 和历史设计稿，应与 `docs/architecture/overview.md`、`src/nalr/runtime/controller.py`、`src/nalr/agents/modules.py` 一起阅读

## Scope

本规格定义当前闭环中“输入到变化”这部分的主链：

```text
用户输入 / 事件
-> appraisal
-> 当前轮状态更新（mood / energy / affect）
-> cue / memory / relation 写入
-> chronic signal 窗口化累积
-> identity evidence / naming signal
-> expression plan / renderer
-> trace / observer diagnostics
```

它不试图覆盖完整 runtime 全部阶段，而是聚焦以下问题：

- 普通聊天如何避免被旧 task run 污染
- 自然语言如何被补成可消费的 appraisal 信号
- `mood`、`body_energy`、`affect_residue` 如何变化
- chronic drift 如何从多轮轨迹中提取慢变量
- identity evidence 如何与 cue / relation / memory 联动
- 为什么“没变化”时 observer 能给出断点

## Execution-State Isolation

### Goal

普通聊天不应被残留的 `active_run_id`、`current_goal`、`pending_steps` 持续驱动。

### Current Behavior

`RuntimeController._state_for_reasoning(...)` 会先判断本轮 route 和 drive source：

- `user_input`
- `active_task`
- `noninteractive_shaping`
- `mixed`

当前规则：

- `requested_mode in {"idle", "sleep"}` 时，`drive_source = "noninteractive_shaping"`
- `scenario == "task"` 时，`drive_source = "active_task"` 或 `user_input`
- 非 task 且存在 active run，同时文本像任务续写时，`drive_source = "mixed"`
- 其他聊天场景默认为 `user_input`

当满足以下条件时，判定为 run contamination：

- 当前存在 `active_run_id`
- `run_status in {"running", "paused"}`
- 当前 `scenario != "task"`
- 当前 `drive_source == "user_input"`

一旦检测到污染，对 reasoning 可见的运行态会被剥离：

- `active_run_id`
- `run_status`
- `run_mode`
- `current_goal`
- `current_step_id`
- `pending_steps`
- `completed_steps`
- `last_tool_result`
- `stop_reason`
- `dirty_worktree_detected`

同时把真实执行态作为 `run_context` 写入 trace，用于事后诊断，而不是继续驱动 PFCAgent。

## Appraisal Contract

### Goal

让自然语言在没有外部显式 `valence` / `energy_delta` 的情况下，也能驱动状态变化。

### Current Appraisal Fields

当前 appraisal layer 会产出：

- `semantic_valence`
- `semantic_arousal`
- `social_approach_pull`
- `cognitive_load`
- `fatigue_push`
- `inferred_energy_delta`
- `identity_salience`
- `relation_charge`
- `mood_band`
- `appraisal_band`

### Current Inputs

当前 appraisal 由以下因素共同决定：

- 文本语义 token
- 当前关系亲近度 `closeness`
- 负荷 / 疲惫线索
- 身份相关询问
- 关系确认或疏离线索

当前实现里，典型 token family 包括：

- negative tokens
- positive tokens
- fatigue tokens
- load tokens
- identity tokens
- relation tokens

### Event Override Rule

`RoundEvent.valence` 和 `RoundEvent.energy_delta` 仍可作为外部 override，但在默认情况下会由 appraisal 自动补全：

- 当 `event.valence` 近似 0 时，用 `semantic_valence`
- 当 `event.energy_delta` 近似 0 时，用 `inferred_energy_delta`

因此当前系统不再把“调用者没传 valence”视为状态冻结的充分理由。

## State Evolution

### Mood

`EmotionAgent.update_affect_state(...)` 当前会结合三部分更新 `mood`：

- 当前轮输入的 `event.valence`
- 既有 mood 的惯性项
- 已有 `affect_residue` 的回拉项

同时还会写入：

- `session_metadata.last_appraisal_band`

### Body Energy

`BodyStateAgent.update_body_state(...)` 当前会消耗：

- `fatigue_push`
- `cognitive_load`
- `repair_mode` 带来的 repair drag

它不再只依赖外部传入的体力变化，因此“我累了 / 我准备睡了 / 我有点撑不住”这类语句能直接影响 `body_energy`。

### Affect Residue

`RuntimeController._update_affect_residue(...)` 维护即时冲击后的残留：

- 负向输入对 residue 的 shock 更强
- `idle` 和 `sleep` 会改变 decay / input scale
- residue 会逐轮衰减，而不是一轮归零

这部分是“本轮已经过去，但余波还在”的主要状态载体。

### Before-Clip / After-Clip Diagnostics

每轮 trace 会同时记录：

- `state_delta_before_clip`
- `state_delta_after_clip`

当前固定观测的键包括：

- `mood`
- `body_energy`
- `affect_residue`

这让 observer 可以区分：

- 本来就没输入
- 有输入但变化被 clip 掉
- 有变化但还没过表达阈值

## Chronic Signal And Temperament Drift

### Goal

人格慢变量不再只看单轮瞬时值，而是看窗口化趋势。

### Current Window Summary

`RuntimeController._build_temperament_window_summary(...)` 当前会综合最近窗口中的：

- `avg_positive_valence`
- `avg_negative_valence`
- `relation_trend`
- `repeated_rejection_count`
- `repeated_confirmation_count`
- `resource_stress_span`
- `identity_cue_repeat_count`
- `noninteractive_residue`
- `current_relation_charge`
- `scenario`

### Current Drift Semantics

`UnconsciousAgent.apply_chronic_shift(...)` 现在消费上面的 chronic signal，而不是只消费单轮 `event.valence`。

当前 drift 诊断至少应能解释：

- `delta_reason`
- `window_support`
- `suppressed_by_clip`
- `freeze_reason`

因此“drift == 0”不再应被视为静默结果，而应具备可解释原因。

## Cue, Identity, And Naming

### Cue Normalization

当前 memory 写入前会先做 cue 清洗，避免以下内容直接进入长期证据：

- 控制字符
- 终端 escape
- 明显命令残片
- 大段一次性自由文本

身份相关问句当前聚类到：

- `identity:self_probe`
- `identity:name_probe`
- `identity:name_origin`

### Identity Evidence

`MemoryStore.identity_evidence()` 当前输出的核心字段包括：

- `identity_clusters`
- `identity_score`
- `naming_signal`
- `continuity_signal`
- `rejection_reasons`

### Identity Runtime Coupling

`IdentityRuntime` 不再只盯着 top stable prior，而是结合：

- `identity_score`
- `naming_signal`
- `continuity_signal`
- relation 稳定度

来决定：

- 是否仍处于 unnamed / provisional 阶段
- 是否进入稳定 `display_name`

当前 observer 侧暴露的 identity blockers 就建立在这套证据上。

## Expression Coupling

当前表达层已经与状态产生真实耦合，但强度仍是“部分完成”。

当前至少已经存在这些耦合点：

- `affect_residue` 进入 render plan 的 `slow_variables`
- renderer 会根据 `affect_residue`、`resource_scarcity`、`identity_context` 等信号调整 opening / disclosure / self-explanation
- `expression_threshold_not_met` 会作为 suppression reason 写入 trace / why-no-change

仍需注意：

- 当前系统已经能解释“变了但没明显表达出来”
- 但还不能宣称所有状态变化都能被用户强感知

## Diagnostics Surface

更细的 observer API 输出、blocker taxonomy 和 dashboard 目标视图，已单独收口到：

- [observer-diagnostics-and-blockers.md](/Users/fantasylee/类脑架构/docs/specs/observer-diagnostics-and-blockers.md)

### Trace Fields

当前 `RoundTrace` 中与本规格直接相关的字段包括：

- `appraisal_snapshot`
- `state_delta_before_clip`
- `state_delta_after_clip`
- `delta_suppression_reason`
- `run_context`
- `run_contamination_detected`
- `identity_evidence_score`
- `identity_trigger_blockers`
- `temperament_window_summary`

### Observer Endpoints

observer 当前直接暴露以下诊断面：

- `/diagnostics/state-delta`
- `/diagnostics/identity-blockers`
- `/diagnostics/cue-fragmentation`
- `/diagnostics/run-contamination`
- `/diagnostics/why-no-change/{round_ref}`
- `/diagnostics/migration`

### Failure Modes

当前 `why_no_change()` 会把“没有变化”保守分成以下几类：

- `not_called`
- `called_neutral`
- `updated_but_clipped`
- `updated_but_not_expressed`

常见 blocker 包括：

- `no_appraisal_input`
- `run_contamination`
- `delta_clipped`
- `expression_threshold_not_met`
- `identity_evidence_insufficient`

## Non-Goals

本规格当前不覆盖：

- dashboard 视图本身的交互设计
- renderer prompt 的最终文案策略细节
- 每个 temperament key 的精确 drift 公式
- dream sidecar 的完整非交互塑形规格

## Current Gaps

以下仍属于已知未完项，而不是本规格已宣布完成的行为：

- expression 到 opening / pacing / disclosure / directness 的强耦合映射还可以继续增强
- identity explanation path 还可以进一步从 renderer 层独立出来
- diagnostics 目前 API 完整，但 dashboard 仍未可视化到位
- chronic drift 的超长窗口稳定性仍未恢复重型 soak 结论
