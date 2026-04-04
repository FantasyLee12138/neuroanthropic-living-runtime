# NALR 主体性边界落地实施方案（v1.0）

## 一、目标

将 NALR 从"拟人响应系统"升级为"具备形式化主体边界的运行时"。

------------------------------------------------------------------------

## 二、设计原则

1.  外部不可直接写入主体核心与内部状态
2.  内部状态仅能由系统自身更新
3.  系统具备最低限度的内部动机生成能力

------------------------------------------------------------------------

## 三、Phase 0：设计红线

-   所有外部输入（用户 / CLI / skill / sidecar）不得直接写入内部状态
-   必须通过 stimulus 或 proposal 机制

------------------------------------------------------------------------

## 四、Phase 1：SubjectCore

### 结构

-   subject_id
-   birth_ts
-   continuity_nonce
-   original_vitality_anchor
-   core_boundary_version

### 规则

-   只读
-   不可删除
-   不可回滚

### 验收

-   任意修改尝试被拒绝并记录 violation

------------------------------------------------------------------------

## 五、Phase 2：Owned State

### 分层

-   Owned State（内部）
-   External Stimulus（外部）

### 新组件

SubjectBoundaryGuard

### 规则

-   外部只能影响，不可赋值
-   所有修改需 trace

------------------------------------------------------------------------

## 六、Phase 3：Endogenous Intent

### 输入

-   memory rebound
-   mood residue
-   vitality drift

### 输出

-   micro_intent

### 规则

-   不直接生成文本
-   仅作为行为偏置

------------------------------------------------------------------------

## 七、Phase 4：观测与测试

### 指标

-   subject_core_integrity
-   boundary_violation_count
-   endogenous_intent_rate

### 测试

-   core immutable test
-   cli no direct set test
-   endogenous tick test

------------------------------------------------------------------------

## 八、实施计划

### 阶段 1

-   SubjectCore
-   BoundaryGuard
-   CLI 语义迁移

### 阶段 2

-   Intent Scheduler
-   测试与指标

------------------------------------------------------------------------

## 九、范围说明

### 本期不解决

-   意识
-   自由意志
-   意义系统

### 只解决

-   外部无法直接改核心与内部所有态
-   所有状态变化可追溯为外因或内因
-   无输入条件下存在可解释的内部微意图
-   故障时自动退回保守模式

------------------------------------------------------------------------

## 十、总结与各阶段具体实现

主体性边界是从"系统"到"主体"的第一步。


### Phase 1：建立 Subject Core（核心不可侵阈）
目标：解决“谁是这个主体”以及“哪些东西绝不能被外部改”。
你要新增的对象
新增一个 SubjectCore，只放极少数、极稳定字段：
subject_id
birth_ts
continuity_nonce
original_vitality_anchor
core_boundary_version
规则
只能初始化一次。
普通 round、skill、CLI、sidecar 都不能写。
只有受控 migration 才能升级 schema，不能改值。
所有 trace 都要带 subject_id 和 continuity_nonce。
验收标准
任意 CLI/skill/sidecar 尝试直接改 SubjectCore 都被拒绝。
trace 中能看到拒绝原因：subject_core_violation。
checkpoint/rewind 不能把 subject_id 回滚成别的值。

### Phase 2：把内部状态改成 Owned State（内部所有域）
目标：解决“情绪、记忆、关系、习惯是不是它自己的”。
你要做的结构调整
把现有内部状态分两层：
A. Owned State（主体所有态）
emotion
mood residue
relation preference
habit strength
autobiographical memory weight
disclosure tendency
self-image snapshot
B. External Stimulus（外部刺激）
user utterance
operator command
environment event
sidecar proposal
task requirement
核心规则
外部永远不能 set emotion = x。
外部只能发刺激，例如：
user insulted
operator nudge focus +0.15
dream proposal: soften memory salience
具体改多少，必须由内部更新器决定。
需要新增的组件
新增一个 SubjectBoundaryGuard，位置放在 RuntimeController 之前或内部入口处，负责：
拦截所有直接写入 Owned State 的请求
把非法写入转成 violation trace
决定是拒绝、降级成 stimulus，还是要求人工高权限 override
为什么这一步特别关键
这一步一做，CLI 的语义就会彻底变化：
alive mood calm 不再是“把心情设置为 calm”，而是“施加 calm stimulus，让系统自己收敛”。
这才是真正的“外部只能影响，不能控制”。
而且这和 v0.56 的 CIL 分级完全兼容，只是把“干预”从直接赋值改成受控刺激。
验收标准
旧命令若含直接赋值，必须报 deprecation warning。
所有状态改变日志里区分：
cause = external_stimulus
cause = endogenous_update
任何 sidecar proposal 只能进 proposal queue，不能直接写内存或身份状态。这个规则可直接复用 OneiroiAgent 的“只读快照 + 写入申请”模式


### Phase 3：加入 Endogenous Intent Loop（内部始发域）
目标：解决“它是不是永远只能被动响应”。
你要新增的最小机制
新增一个低频 tick，不走完整 interactive pipeline，只在 idle / sleep 前后 / 低刺激窗口运行。
输入来自：
memory rebound
unfinished tension
mood accumulation
habit inertia
vitality drift
输出不是直接对外说话，而是 micro_intent，例如：
avoid_social_interaction
seek_closure_on_memory
rest_and_reduce_output_density
increase_disclosure_resistance
review_relation_target_A
运行规则
micro_intent 先只影响下一轮 bias，不直接生成对外文本。
只有在连续多 tick 稳定存在时，才允许进入可表达层。
所有内部始发意图必须能在 observer 中被解释为“由哪些内部变量触发”。
这一步为什么务实
因为它不要求“真正的意识”，只要求系统在结构上能成为行为起点。
这就是你文本里“主体是行为始发点”的工程版本。
而且它和 README 里的 idle/sleep 链、VitalityEngine、dream shaping 是天然连接的。
验收标准
在无用户输入的 100 个 tick 中，系统能产生有限但可解释的 micro_intent。
这些意图不会越权触发外部动作。
observer 能展示“internal-originated vs externally-triggered”占比。