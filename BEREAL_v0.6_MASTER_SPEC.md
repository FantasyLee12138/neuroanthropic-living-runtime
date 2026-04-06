# BEREAL v0.6 MASTER SPEC

## 1. 文档定位

本文件是 NALR 当前主线 `bereal / v0.6` 的唯一规范。
所有旧规格、设计草案、机制说明、迁移计划与历史开发文档已被整合并移入备份目录。

当前前台文档只保留：

- [`README.md`](/Users/fantasylee/类脑架构/README.md)
- [`BEREAL_v0.6_MASTER_SPEC.md`](/Users/fantasylee/类脑架构/BEREAL_v0.6_MASTER_SPEC.md)
- [`PLANS.md`](/Users/fantasylee/类脑架构/PLANS.md)

## 2. 核心目标

NALR `bereal / v0.6` 的目标不是继续堆叠 sidecar 机制，而是在统一主链上稳定以下能力：

- 四层概率场作为唯一决策真相面
- 多模块贡献写入 action field，而不是串行抢夺 winner
- 内生动机与内生调度作为 field-first 扩展
- 真实性、生命性、边界与长期一致性作为一等约束
- 小中大模型在准确率优先前提下按风险和预算高效协同
- trace、why、why-not、replay、observer 始终能解释真实决策来源

## 3. 唯一真相面

系统当前唯一真相面是：

```text
context -> memory -> action -> token
```

这 4 层共同构成运行时的正式概率场。任何模块都不能绕过这条主链直接决定最终动作。

以下函数是这条真相面的关键入口：

- `_tick_impl()`：主回合执行入口，负责从输入事件到 sampled action、render、trace 的完整主链。
- `_collect_probability_field_contributions()`：贡献收集入口，负责汇总各模块对概率场的贡献。
- `_integrate_probability_field_snapshot()`：概率场积分入口，负责将 base energy、贡献与 coupling 统一融合。
- `_reintegrate_probability_snapshot()`：重积分入口，负责在 conflict、guard、vitality 等阶段继续在同一真相面上更新 action/token 层。

## 4. Runtime 主链

当前主链按以下顺序工作：

```text
state update
-> context routing
-> memory prior
-> action contributions
-> endogenous motivation contribution
-> conflict arbitration
-> stochastic shaping
-> authenticity / plausibility / output gate
-> vitality modulation
-> renderer / token unfold
-> trace / writeback / long-run evidence
```

设计原则：

- 模块只写 contribution，不直接写 winner
- winner 只能从统一 action field 中采样
- token 层只能承接 action -> token 合法 coupling
- 所有高风险调整都必须保留 trace 证据

## 5. 概率场约束

### 5.1 Contribution contract

当前模块统一输出 `ProbabilisticContribution`。
其含义是：

- `raw_signal`：原始驱动信号
- `modulated_delta`：经过调制后真正写入概率场的变化量
- `inhibitory_drive`：抑制项
- `hard_mask`：硬性禁止项

系统不再使用旧的 proposal truth 作为主链事实源。

### 5.2 Integrator contract

以下条件必须一直成立：

- model output 不能绕过 integrator 直接成为 winner
- winner 必须能回溯到 contribution audit
- `why_this()`：单轮解释入口 必须解释 winner 为什么成立
- `why_not()`：反事实解释入口 必须解释某个动作为什么被压制
- `replay()`：回放入口 必须能回放候选分布和模块影响

### 5.3 有效性标准

概率场输出“真实有效”的定义不是主观印象，而是以下门槛同时成立：

- winner 与 contribution audit 一致
- token source integrity 不破坏
- observer 和 replay 仍可解释
- 关键回归指标不下降：
  - `probability_field_coverage`
  - `action_audit_coverage`
  - `token_source_integrity_rate`

## 6. Endogenous 主线

### 6.1 动机池

`EndogenousMotivationPool`：内生动机池，负责根据当前状态生成 first-class action bias。
它只能写 action layer，不能直接写 token，也不能直接选 winner。

首批动机类型：

- `memory_exploration`
- `behavior_exploration`
- `affect_regulation`
- `relation_calibration`
- `internal_replay`

### 6.2 反馈闭环

`MotivationFeedbackUpdater`：动机反馈更新器，负责在 trace/writeback 后根据本轮结果更新动机学习状态。
它不能在 sampled action 之前偷偷回写下一步动作。

反馈来源包括：

- `vitality_snapshot`
- `authenticity`
- sampled action
- memory activation
- relation drift
- affect residue

### 6.3 内生调度

`EndogenousTickScheduler`：内生调度器，负责判定何时需要发起一轮 endogenous tick。
它分成两层：

- `build_trigger()`：触发判定入口，负责判断当前是否该触发以及选择哪种内生模式
- `update_state()`：调度状态更新入口，负责记录最近触发、抑制和上次内生轮次时间

### 6.4 内生入口

`run_endogenous_tick()`：内生 tick 触发入口，负责构造内生触发并最终复用 `_tick_impl()`：主回合执行入口。
它不是单独的 sidecar runtime，也不是另一套 sampled action 真相面。

## 7. Boundary / Subjectivity / Authenticity / Vitality

当前主线坚持以下边界：

- SubjectCore 不允许被外部 actor 直接改写
- 自我不是 prompt 外挂角色，而是持续状态先验
- 边界不是单独后置阀门，而是 identity、relationship、authenticity、long-run value 的场内约束
- 生命性与真实性不是装饰项，而是长期稳定性的运行时条件

关键函数：

- `state_payload()`：运行时状态读取入口，负责输出当前 runtime、subjectivity、storage 和 cognitive snapshot
- `cognitive_snapshot()`：认知快照入口，负责给终端和 observer 提供人类可读的当前认知状态摘要
- `authenticity_timeline()`：真实性时间线入口，负责输出真实性相关长期证据
- `vitality_timeline()`：生命性时间线入口，负责输出 vitality 相关长期证据

## 8. Model Routing Strategy

### 8.1 唯一路由真相面

当前主线只承认以下组合为正式模型路由真相面：

```text
ModelRouter + model_tiers + agent_model_bindings
```

`model_gateway.py` 只保留兼容角色，不再作为当前主线的第一决策入口。

### 8.2 默认分工

- `SalienceAgent -> small_model`
- `ValueAgent -> small_model`
- `PFCAgent -> medium_model`
- `PerspectiveModel -> medium_model`
- `Renderer -> large_model`

### 8.3 升级条件

以下条件命中时，路由允许升级到更大模型：

- conflict 高
- authenticity risk 高
- disclosure sensitivity 高
- relation risk 高
- 多轮 resample
- renderer violation / fallback

关键函数：

- `_route_config_for_binding()`：模型档位解析入口，负责根据模块绑定和风险上下文决定当前 effective tier
- `_call_bound_model_route()`：模型调用封装入口，负责执行当前选中的模型路由并记录调用证据
- `model_status()`：模型状态观测入口，负责暴露当前 tier、binding、route 和最近降级情况

## 9. Observer / Trace / Replay

当前正式观测面包括：

- `/trace/{round_id}`
- `/why/{round_id}`
- `/why-not/{round_id}/{action}`
- `/replay/{round_id}`
- `/metrics/summary`
- `/metrics/motivation`
- `/metrics/endogenous`
- `/why-motivation/{round_id}`
- `/replay/motivation/{round_id}`

关键函数：

- `trace_round()`：trace 读取入口，负责将 round payload 组装为统一观测视图
- `why_motivation()`：动机解释入口，负责解释该轮 active motivations、trigger 和反馈摘要
- `replay_motivation()`：动机回放入口，负责重放该轮 motivation pool 与 endogenous policy shift
- `motivation_metrics()`：动机指标入口，负责聚合 active rate、activation score 和最新 pool 状态
- `endogenous_metrics()`：内生指标入口，负责聚合 endogenous round rate 和 trigger 类型

## 10. 当前状态结构

运行时当前保留的 endogenous typed state 包括：

- `MotivationPoolState`
- `MotivationLearningState`
- `EndogenousSchedulerState`
- `EndogenousTickTrigger`

trace 当前新增字段包括：

- `motivation_pool`
- `motivation_feedback`
- `endogenous_tick_reason`
- `endogenous_policy_shift`

这些字段已经进入 trace store、exporter 和 observer，而不是只停留在内存态。

## 11. 测试与验收

### 11.1 模型接入测试

所有接入模型的模块都要有 tier 路由与 fallback 测试：

- `SalienceAgent`
- `ValueAgent`
- `PFCAgent`
- `PerspectiveModel`
- `Renderer`

### 11.2 概率场有效性测试

必须覆盖：

- winner/audit consistency
- model output 不绕过 integrator
- why/why-not/replay 解释仍成立
- key metrics 不退化

### 11.3 Endogenous 测试

必须覆盖：

- motivation pool 只投 action layer
- feedback updater 只在 writeback 后更新
- scheduler 能覆盖核心 trigger 类型
- `run_endogenous_tick()`：内生 tick 触发入口 走完整主链而不是 sidecar

## 12. 文档与归档规则

当前只保留三份前台文档。
其余旧文件全部进入备份目录，并在索引中保留：

- 原路径
- 新路径
- 是否已被吸收到主规范
- 保留原因

历史版本号保持原样，不在当前主线内重写。
