# NALR v0.57 内生动机池 / 自驱闭环 / 内生 Tick 调度设计方案

（基于当前 `PLANS.md`、`README.md` 与现有神经调制规范的落地扩展版）

---

# 一、设计结论

在当前规范下，这三个能力**可以加，而且适合现在这个阶段加**，因为仓库已经具备：

- `ProbabilisticContribution`、`ProbabilityFieldSnapshot`、`TokenFieldState`、`ProbabilityFieldIntegrator` 等 v0.57 主骨架。fileciteturn1file0turn1file1
- controller 正在从旧 proposal 管线收口到 `context / memory / action / token` 四层概率场。fileciteturn1file0turn1file1
- trace / why / why-not / acceptance / parquet export 已大体切到 `probability_field` 解释面。fileciteturn1file0turn1file1
- 现有神经调制规范已经明确：系统应通过“调制 + 竞争 + 抑制 + 恢复”形成行为，而不是回到硬规则触发。fileciteturn0file0

但前提是：

> 这三个新增能力都必须作为 **v0.57 field-first 主链的扩展** 接入，  
> 不能重新长出一套“绕开四层场的主动性 sidecar”。

---

# 二、原需求与规范对齐后的总体设计

你提出的三个能力：

1. **内生动机池模块**
2. **自驱行为闭环反馈**
3. **内生 tick 调度器**

在当前规范里，最合理的落法是：

```text
四层概率场
+ 内生动机 bias heads
+ trace 驱动的慢反馈更新
+ controller 外围的 endogenous trigger
```

而不是：

```text
新 planner / 新 agent loop / 新决策内核
```

---

# 三、这份设计里需要新增/更新的地方

---

## 1. 需要新增一个正式模块：`EndogenousMotivationPool`

### 作用
根据四层概率场状态特征，动态生成“内生动机”的 `ProbabilisticContribution`，直接写入 **action 层**。

### 它不是
- 规则触发器
- 直接动作选择器
- 第二套 planner

### 它是
- action 层的内生 bias source
- 主动行为的“起因场”

### 推荐新增文件
- `src/nalr/runtime/motivation_pool.py`

### 推荐新增 schema
放到 `src/nalr/schemas/models.py`：

```python
class EndogenousMotivationSignal(BaseModel):
    motivation_id: str
    motivation_type: str
    raw_drive: float
    source_features: dict[str, float]
    target_actions: dict[str, float]
    state_tags: list[str]
    audit_reason: str

class MotivationPoolState(BaseModel):
    active_motivations: list[EndogenousMotivationSignal]
    pool_weight_snapshot: dict[str, float]
    endogenous_activation_score: float
    last_feedback_update_at: str | None = None
```

---

## 2. 需要新增一个正式闭环模块：`MotivationFeedbackUpdater`

### 作用
复用现有 trace / memory / acceptance / longrun，对本轮“内生行为”的结果做轻量反馈更新。

闭环结构：

```text
动机
-> action contribution
-> sampled action
-> 执行结果
-> trace 记录
-> motivation weight update
-> 下轮动机分布改变
```

### 推荐新增文件
- `src/nalr/runtime/motivation_feedback.py`

### 推荐新增 schema
```python
class MotivationFeedbackRecord(BaseModel):
    round_id: str
    motivation_id: str
    sampled_action: str
    user_response: str | None
    response_latency_ms: int | None
    affect_delta: dict[str, float]
    memory_activation_delta: float
    relation_delta: float
    reward_signal: float
    update_reason: str

class MotivationLearningState(BaseModel):
    motivation_weights: dict[str, float]
    recent_feedback: list[MotivationFeedbackRecord]
    endogenous_policy_shift: dict[str, float]
```

---

## 3. 需要新增一个外围调度器：`EndogenousTickScheduler`

### 作用
在 **不改 `_tick_impl` 核心决策真相** 的前提下，基于当前概率场状态决定是否发起一轮轻量 endogenous tick。

### 推荐新增文件
- `src/nalr/runtime/endogenous_scheduler.py`

### 推荐新增 schema
```python
class EndogenousTickTrigger(BaseModel):
    trigger_type: str
    trigger_score: float
    source_metrics: dict[str, float]
    selected_mode: str
    audit_reason: str

class EndogenousSchedulerState(BaseModel):
    last_endogenous_tick_at: str | None
    recent_triggers: list[EndogenousTickTrigger]
    suppression_reason: str | None = None
```

---

# 四、三个能力分别怎么接入当前主链

---

## A. 内生动机池：接 action 层，不接 token 层

推荐接入位置：

```text
state_encoding
-> context_routing
-> memory_reactivation
-> endogenous_motivation_pool    ← 新增
-> action_field_build
-> executive_reweight
-> peak_arbitration
-> guard_penalty
-> token_field_build
-> trace_commit
```

### 原因
- 它需要读取 context / memory / vitality / emotion / relation 的状态
- 但必须在 action 层建场之前进入，才能作为 first-class contribution 参与竞争

### 不允许
- motivation pool 直接产出 token bias
- motivation pool 直接产出 winner
- motivation pool 跳过 integrator

---

## B. 反馈闭环：接 writeback / longrun 之间

推荐位置：

```text
trace_commit
-> writeback
-> motivation_feedback_update   ← 新增
-> longrun / dream / memory gate
```

### 原因
- 要先吃到本轮真实结果
- 再更新动机权重
- 再进入长期塑形总结

### 不允许
- 直接在 action sampling 后立刻改下一步 action
- 绕过 trace 做 hidden weight update

---

## C. 内生 tick 调度器：接 controller 外围，不改 `_tick_impl`

推荐方式：

```python
if endogenous_scheduler.should_trigger(snapshot, motivation_pool_state):
    controller.run_endogenous_tick(mode="endogenous_light")
```

### 关键原则
- `run_endogenous_tick()` 仍然复用正式主 tick 入口
- 不创建 shadow runtime
- 不另起平行的 sampled action 真相

---

# 五、这三个能力如何具体设计

---

## 1. 内生动机池（EndogenousMotivationPool）

### 首批建议动机类型

#### a. `memory_exploration`
触发条件示例：
- memory 层未激活占比 > 70%
- recall coverage 持续低
- context->memory coupling 存在但 memory prior 弱

偏置动作：
- `recall`
- `rest`
- `reflect`
- `replay_memory`
- `inspect_context`

---

#### b. `behavior_exploration`
触发条件示例：
- action 层能量分布连续 X 轮单调
- competing peaks 过窄
- `mode locking` 风险增加

偏置动作：
- `probe`
- `explore`
- `branch`
- `small_shift`
- `reframe`

---

#### c. `affect_regulation`
触发条件示例：
- 情绪层长期低活力
- `global_brain_state` 处于 `flat / withdrawn / stressed`
- vitality / emotion head 长期低振幅

偏置动作：
- `rest`
- `self_adjust`
- `quiet_repair`
- `internal_rebalance`

---

#### d. `relation_calibration`
触发条件示例：
- 连续多轮用户无回应
- relation drift 增大
- affiliation_modulator 失衡

偏置动作：
- `wait`
- `soften`
- `reduce_push`
- `relation_probe`

---

#### e. `internal_replay`
触发条件示例：
- unresolved conflict 持续存在
- memory contamination 风险高
- dream/longrun 提示内部残余未清

偏置动作：
- `internal_replay`
- `consolidate`
- `quiet_repair`
- `reindex`

---

## 2. 自驱反馈闭环（MotivationFeedbackUpdater）

### 反馈信号来源
优先复用现有 trace / acceptance / memory / relation / vitality 路径：

- 用户回应 / 无回应
- 回应延迟
- affect 变化
- memory activation 变化
- relation drift 变化
- conflict 是否下降
- guard / authenticity penalty 是否升高
- token/source integrity 是否恶化

这些现有规范和 README 已经说明大体具备基础。fileciteturn1file0turn1file1

### 更新规则建议

#### 正反馈
如果：
- memory activation 提升
- relation tension 降低
- affect 恢复
- conflict 降低
- 用户回应更积极

则：
- 对应动机权重小幅上调

#### 负反馈
如果：
- 用户持续无回应
- authenticity penalty 升高
- conflict / stress 变差
- memory contamination 风险上升

则：
- 对应动机权重下调
- 连续失败进入抑制态

#### 中性反馈
- 轻微衰减或保持

### 必须禁止
- 把 “无回应” 简单当失败
- 把 reward_signal 直接写进 action sampler
- 不留 trace 直接更新 motivation weights

---

## 3. 内生 tick 调度器（EndogenousTickScheduler）

### 建议触发条件

#### a. `field_imbalance`
- action entropy collapse
- conflict residual 偏高
- vitality / affect 长期异常
- mode locking 增强

#### b. `motivation_sum_high`
- 所有活跃内生动机的 posterior 总和 > 阈值

#### c. `silent_but_active`
- 无外部输入持续一段时间
- 但 internal replay / affect regulation / memory exploration 动机持续活跃

#### d. `endogenous_wake`
- memory burst
- salience burst
- conflict spike
- resource pressure shift

### 推荐 tick 模式
- `interactive`
- `endogenous_light`
- `endogenous_replay`
- `endogenous_regulation`

### 内生 tick 允许的行为
- 内部记忆重放
- 轻量自我状态调整
- relation 低强度校准
- quiet repair
- 低成本、低强度探索

### 不允许的行为
- 高成本 skill
- 未审批的外部动作
- 强干预用户
- 高强度长文本输出
- 绕过 safe mode / breaker / policy check

---

# 六、必须更新哪些现有文件

---

## 1. `PLANS.md`
需要追加正式范围：

### 新增 Agent 6：Endogenous Motivation / Scheduler / Feedback
负责：
- `EndogenousMotivationPool`
- `MotivationFeedbackUpdater`
- `EndogenousTickScheduler`
- motivation trace / observer / acceptance / README delta

### 新增阶段 C2：内生主动性接线
- 动机池接入 action 层
- 闭环反馈接入 trace/writeback
- endogenous scheduler 接入 RuntimeController 外围
- 不允许绕开四层场 / trace / policy / safe mode

---

## 2. `README.md`
需要新增 delta：

### 当前实现主线
加入：
- endogenous motivation heads
- endogenous lightweight tick modes
- motivation feedback loop

### 当前运行时逻辑
补一条：

```text
idle / no-external-input
-> endogenous scheduler
-> motivation pool activation
-> endogenous light tick
-> trace / feedback / longrun
```

### 当前可直接观测的能力
建议补：
- `alive trace motivation last`
- `alive endogenous status`
- `/metrics/motivation`
- `/metrics/endogenous`

---

## 3. `src/nalr/schemas/models.py`
新增：
- `EndogenousMotivationSignal`
- `MotivationPoolState`
- `MotivationFeedbackRecord`
- `MotivationLearningState`
- `EndogenousTickTrigger`
- `EndogenousSchedulerState`

---

## 4. `src/nalr/runtime/controller.py`
新增：
- motivation pool 接线
- `run_endogenous_tick(...)`
- endogenous trigger metadata

但必须保持：
- `_tick_impl` 仍是正式主真相面
- 不新增 shadow sampler

---

## 5. 新增 runtime 文件
- `src/nalr/runtime/motivation_pool.py`
- `src/nalr/runtime/motivation_feedback.py`
- `src/nalr/runtime/endogenous_scheduler.py`

---

## 6. `src/nalr/trace/exporter.py`
新增导出：
- `motivation_pool_json`
- `motivation_feedback_json`
- `endogenous_scheduler_json`
- `endogenous_tick_reason_json`

---

## 7. `src/nalr/runtime/longrun.py`
把：
- motivation weight drift
- endogenous policy shift
纳入 longrun summary / online prior diagnostics

---

## 8. `services/observer/api/app.py`
新增接口建议：
- `/metrics/motivation`
- `/metrics/endogenous`
- `/why-motivation/{round_id}`
- `/replay/motivation/{round_id}`

---

# 七、必须新增的 acceptance / trace / observer 约束

---

## trace 必须新增
- `motivation_pool`
- `motivation_feedback`
- `endogenous_tick_reason`
- `endogenous_policy_shift`

---

## acceptance_report 必须新增
- `motivation_pool_coverage`
- `endogenous_motivation_activation_rate`
- `motivation_feedback_coverage`
- `endogenous_learning_update_rate`
- `endogenous_tick_count`
- `endogenous_tick_suppression_rate`
- `unsafe_endogenous_attempt_count`
- `motivation_pool_bypass_count`

---

## 阻塞断言必须新增
- `motivation_pool_bypass_count = 0`
- `endogenous_tick_without_trace = 0`
- `unsafe_endogenous_attempt_count = 0`
- `renderer_decision_integrity.decision_mutated = false`
- `token_source_integrity_rate = 1.0`

---

# 八、必须避免的设计错误

---

## 错误 1：动机池变成硬规则触发器
不允许：

```text
if inactive_memory > 0.7:
    force action = recall
```

正确：
```text
inactive_memory -> memory_exploration signal -> contribution -> action field
```

---

## 错误 2：闭环反馈变成旁路奖励模型
不允许直接根据结果去改 action sampler。  
只允许更新：
- motivation weights
- endogenous policy shift
- neuromodulation bias 的慢变量

---

## 错误 3：scheduler 变成第二 controller
内生 tick 调度器只能决定“是否触发”，不能决定“做什么动作”。

---

## 错误 4：在没有用户输入时偷偷做高成本行为
内生 tick 必须默认轻量、低成本、可追踪、可抑制。

---

## 错误 5：主动性不进 trace
这是最危险的伪主动性形式。  
如果系统主动了，但 trace / acceptance / why 不知道，那就是新黑箱。

---

# 九、推荐实施顺序

### Step 1
先做 `EndogenousMotivationPool`  
只接 action field，不做 scheduler

### Step 2
再做 `MotivationFeedbackUpdater`  
只更新 motivation weights，不改 controller 核心

### Step 3
最后做 `EndogenousTickScheduler`  
只支持 `endogenous_light / endogenous_replay / endogenous_regulation`

### Step 4
补 observer / acceptance / README delta

这样最稳，因为：
- 不会一次性炸主链
- 每一步都能落到 trace / acceptance
- “主动性”不会先变黑箱

---

# 十、最终落地结论

> 这三个能力非常适合当前 v0.57 阶段加入。  
> 但正确落法不是“新增一个自主 agent 内核”，而是：

1. **把主动性做成 action field 的内生 bias heads**
2. **把学习做成 trace 驱动的慢反馈更新**
3. **把自主思考做成 RuntimeController 外围的轻量 endogenous tick 触发器**

只有这样，它们才会继续服从：
- 四层概率场
- contribution-first
- trace-first
- acceptance-first
- safe mode / breaker / policy check

而不会把系统重新拖回一套不可审计的平行主链。
