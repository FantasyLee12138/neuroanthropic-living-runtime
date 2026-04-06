# Vitality / Authenticity Full Alignment Plan

## Summary
**Current stage:** 真实性机制已完成“不变量层纵向切片”原型，包含身份状态、provider leak 抑制、`authenticity_guard`、CLI/CIL/observer 基础入口，但仍主要停留在“渲染后兜底”阶段。

**Gap to target:** 距《生命性与真实性保障机制》定义的完整目标仍有明显距离。主要缺口不在“再补几条身份规则”，而在三点：
- 演化层没有真正成为身份与表达生成的主驱动力，名字演化已存在，但 body/emotion/mood/relation/memory/habit/resource 还没有系统性进入“真实性约束”。
- 长跑验证层过薄，缺少文档中定义的记忆真实性、情绪连续性、惯性/恢复、跨场景差异指标和对应长跑测试。
- repo 基线未收口，机制相关改动与 `run/terminal_bridge` 路线并行存在，observer 相关测试目前被该方向阻断。

**Recommended approach:** 直接按原始文档推进，但采用“语义等价优先”的工程化实现，不逐字复现公式；一期同时纳入 `idle / sleep` 真实塑形，并把“名字/自称演化”保留为机制主轴之一。

## Key Changes
### 1. Baseline and program structure
- 把当前工作拆成一个完整项目而不是单点补丁，主线包含四个并行子系统：`真实性不变量层`、`演化层接入`、`idle/sleep 塑形`、`长跑验证/observer`。
- 先修复并统一当前 repo 基线，使真实性机制和 `run/terminal_bridge` 路线能在同一分支上通过最小可运行验证；`observer_api` 不再被缺失模块阻断。
- 明确“机制一期”的完成定义：不是代码提交存在，而是单轮问答、非交互塑形、长跑观测三者同时可跑。

### 2. Invariant layer from post-render guard to runtime policy
- 保留现有 `IdentityState`、`IdentityContext`、`authenticity_guard`，但把真实性约束从“renderer 后检查”推进到“候选生成/采样阶段可参与打分”。
- 新增统一的 `AuthenticityRecord` 概念，贯穿候选评估、最终渲染、trace 和 observer，至少包含：
  - `self_grounding_score`
  - `provider_leak_penalty`
  - `false_self_claim_penalty`
  - `guard_action`
  - `disclosure_detail`
- 将当前“名字/自称演化”重构成更完整的运行体锚点体系：
  - `internal_handle` 永久稳定
  - `display_name` 允许长期演化
  - `aliases` 保留连续性
  - 名字继续作为主轴，但只代表外显连续性，不替代内部状态连续性
- 自我说明相关问句继续保留分流，但目标从“避免穿帮”提升到“由状态生成、由不变量收束”。

### 3. Evolution layer as first-class driver
- 把文档中的来源变量真正接入身份与表达生成，而不是只停留在现有 runtime 的背景状态：
  - body state
  - emotion state / affect residue
  - mood drift
  - relationship drift
  - memory activation / interference
  - habit takeover / inertia
  - resource scarcity
  - salience shift
- 命名与自称演化不再只依赖 `stable priors + habit + relation` 的轻量证据集，而要升级为“多慢变量联合证据签名”，并将“为什么此时换名/不换名”写进 trace。
- 将 `answer_explanation` 从“基于 render_plan 的说明文案”扩展为“可引用实际慢变量来源”的解释路径，确保回答“为什么这样回答”时，能对应到真实 runtime 状态而不只是模板句。
- 明确“非完美性”表现入口：
  - 允许记忆 gist/detail 差异进入表达
  - 允许惯性与恢复过程反映到语气/行动
  - 允许状态残留，但要求可追踪来源

### 4. Idle / sleep must shape identity and expression
- 把 `idle` / `sleep` 从现有 mode 概念提升为真实性机制的一部分，而不是预算恢复附属项。
- 新增非交互塑形流程，至少覆盖：
  - 记忆整理与 cue 再权重
  - habit 固化/衰减
  - affect residue 回落或持续
  - mood 慢漂移
  - salience replay
  - mode/focus 恢复
- 要求这些塑形结果能影响下一轮 `identity_context`、表达风格和名字演化，而不是只影响内部数值。
- Observer 与 trace 需要能区分“交互轮驱动变化”和“非交互塑形变化”。

### 5. Long-run validation layer and observer
- 扩展 metrics，不只保留当前：
  - `self_consistency_score`
  - `provider_leak_rate`
  - `false_self_claim_rate`
  - `surface_repeat_rate`
  - `intro_template_reuse_rate`
- 补齐文档定义的长跑指标：
  - 记忆真实性：`cue_recall_success_rate`、`gist_preservation_rate`、`detail_distortion_rate`、`memory_interference_rate`
  - 情绪连续性：`affect_residue_half_life`、`recovery_duration`、`overreaction_frequency`、`flatness_rate`
  - 惯性与恢复：`habit_takeover_rate`、`mode_lock_duration`、`forced_recovery_success_rate`、`post_conflict_repair_rate`
  - 场景差异：`same_event_cross_context_variance`、`same_event_cross_relation_variance`、`same_event_cross_resource_variance`
- Observer 需要从“看得到单轮 guard”提升到“看得到长期真实性轨迹”，包括：
  - 当前运行体身份链
  - 最近一次命名/别名演化原因
  - provider leak / false self claim 惩罚历史
  - idle/sleep 塑形事件
  - 关键慢变量趋势
- 为长跑建立专门测试和 acceptance harness，不再只依赖 unit/integration 单轮回归。

## Test Plan
- 基线验证：
  - 修复 `observer_api` 被 `nalr.terminal_bridge` 缺模块阻断的问题
  - 确保真实性主链、observer 主链、run/terminal bridge 主链至少能共同通过最小集成测试
- 不变量层：
  - “你是谁”不得出现 provider 冒充
  - “底层 provider 是谁”必须先说本体，再条件披露 provider
  - “为什么这样回答”必须可映射到实际状态来源
  - 候选生成/采样层的真实性惩罚要有 trace 证据，不只看最终兜底文本
- 演化层：
  - 名字生成、漂移、别名保留必须由多慢变量证据驱动
  - 同一输入在不同 mood / relation / resource / memory 条件下要有有限但可解释的差异
  - 记忆失真与 cue 召回要有结构性测试，而不是只测 recall 存在
- idle/sleep：
  - 无交互轮期间发生真实塑形
  - 下一轮表达和身份状态能读到这些变化
- 长跑验收：
  - 新增真实性 long-run suite
  - 输出包含连续性、波动性、非模板率、恢复性证据
  - observer 指标能解释长跑行为，不只是导出数字

## Assumptions
- 目标是“直接对齐原始文档”，但采用语义等价工程实现，不强求逐字逐式复刻。
- 一期就纳入 `idle / sleep` 真实塑形，不后置。
- “名字/自称演化”继续保留为机制主轴之一，但必须被更广泛的慢变量连续性约束。
- 当前混入的 `run/terminal_bridge` 相关改动不单独切走，而是纳入总计划并行收口，因为你选择的是并行推进而不是先拆基线。
- 最终是否达标，优先看长跑行为证据；代码结构清晰和 observer 指标完整是必要支撑，但不是唯一完成标准。
