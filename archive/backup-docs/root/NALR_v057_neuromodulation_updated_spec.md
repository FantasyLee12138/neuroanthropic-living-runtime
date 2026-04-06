# NALR v0.57 神经调制映射（对账增强版）+ Structured Instability 可执行规范

（基于现有文件 + v0.57 对账计划融合后的最终落地版）

---

# 一、为什么需要本次更新

原文件：fileciteturn0file0

已经完成：
- 模块 → 神经调制映射 ✔
- wiring ✔
- structured instability 概念 ✔

但结合 v0.57 对账计划，存在以下关键缺口：

### ❗缺口总结

1. **没有绑定 v0.57 四层概率场语义**
2. **没有 Integrator / Contribution 约束**
3. **没有 TokenFieldState 硬边界**
4. **没有语义冻结（Phase A contract）**
5. **没有 failure taxonomy**
6. **没有 legacy path kill 策略**
7. **没有 cross-layer coupling 明确规则**

👉 当前文档是“类脑解释层”，还不是“系统执行规范”。

---

# 二、核心升级目标（这版文档要解决什么）

本版本目标：

> 把“类脑神经调制系统”升级为  
> **可直接进入 v0.57 runtime 的执行级规范**

---

# 三、必须新增的系统级约束（核心补丁）

---

## 1️⃣ 统一概率接口（新增）

所有模块输出必须变为：

```python
ProbabilisticContribution:
    module_name
    layer
    target_space
    raw_signal
    modulated_delta
    inhibitory_drive
    confidence
    dependency_trace
```

### ❗新增原则

- 模块不再直接决定力度
- 只输出 raw signal
- 调制系统负责放大/抑制

---

## 2️⃣ 神经调制层成为“唯一放大器”

替代旧逻辑：

```text
Δ -> τ -> merge ❌
```

改为：

```text
raw_signal -> neuromodulation -> delta ✔
```

---

## 3️⃣ TokenFieldState 硬约束（关键新增）

必须加入：

> ❗TokenFieldState 不能生成新 Δ

只允许：

- 展开已有 Δ
- 传播 modulation
- 应用 inhibition

禁止：

- decoder 自己加 bias
- renderer 改分布

---

## 4️⃣ Cross-layer coupling（新增）

必须定义：

| from | to | allowed |
|------|----|--------|
| context | memory | ✔ |
| memory | action | ✔ |
| action | token | ✔ |
| token | memory | ⚠ gated |
| token | context | ❌ |

---

## 5️⃣ Memory 写入门槛（强化）

新增：

```text
memory_write_allowed if:
    novelty > T1
    AND relevance > T2
    AND stress < T3
```

否则：

❌ 禁止写入

---

## 6️⃣ Legacy Path Kill（新增）

必须删除：

- Thalamus 末端采样逻辑
- PFC proposal ownership
- Renderer 后改 decision
- Guard 覆盖 winner

---

## 7️⃣ Failure Taxonomy（新增）

系统必须能识别：

- entropy collapse
- mode locking
- oscillation
- token drift
- memory contamination
- identity drift
- guard overreach

---

## 8️⃣ 全局不变量（新增）

必须满足：

- 总能量不发散
- 冲突峰不单向增加
- identity 不无证据漂移
- token 不脱离 action

---

# 四、神经调制系统（升级版）

---

## 调制变量（保持原设计）

```python
dopamine
norepinephrine
serotonin
acetylcholine
gaba_tone
stress_modulator
affiliation_modulator
```

---

## 新增：调制状态机

```python
state_tags = [
    "focused",
    "stressed",
    "exploratory",
    "compulsive",
    "withdrawn"
]
```

---

## 新增规则

> ❗失控发生在 modulation，不发生在 Δ

---

# 五、模块映射（只保留关键修改点）

（原映射保留，此处只列需要更新的）

---

## Thalamus（必须修改）

旧：
- sampling

新：
- context routing owner
- entropy routing
- attention bias 输出

---

## Hippocampus（必须修改）

旧：
- recall

新：
- memory prior generator
- pattern separation
- write gate

---

## PFC（必须修改）

旧：
- candidate generator

新：
- inhibition head
- goal prior

---

## ConflictMonitor（必须升级）

旧：
- conflict detect

新：
- multi-peak clustering
- winner arbitration

---

## Renderer（必须降级）

只能：
- decode

不能：
- 改决策

---

# 六、Structured Instability（增强版）

---

## 正确结构

允许：

- dopamine spike
- stress burst
- emotion dominance

但必须有：

- conflict monitor
- inhibitory loop
- long-run correction
- dream reset

---

## 禁止结构

- 全模块同步放大
- token 层逃逸
- memory 无限污染

---

# 七、执行顺序（对齐 v0.57 阶段）

---

## Phase A（必须新增）

冻结：

- contribution schema
- token rule
- coupling rule

---

## Phase B

切换：

- 四层概率场

---

## Phase C

迁移模块

---

## Phase D

观测 + failure

---

## Phase E

系统级验证

---

# 八、最终总结（最关键一句）

> 旧文档是“类脑解释”  
> 本文档是“类脑系统执行规范”

---

# 九、一句话核心原则

> ❗系统不再由模块控制，而是由“调制 + 竞争 + 抑制 + 约束”共同形成行为

# 十 推荐的新命名表（信号层）

建议把原来较工程化的信号名，逐步过渡到更类脑但仍然可执行的名称：

| 现有工程概念 | 建议新名 | 作用说明 |
|---|---|---|
| module temperature | `neuromod_gain` | 模块对神经调制的响应增益 |
| global temperature | `arousal_gain` | 全局唤醒水平 |
| penalty | `inhibitory_drive` | 抑制驱动 |
| hard mask | `gaba_block` | 强抑制阻断 |
| soft mask | `inhibitory_bias` | 软抑制偏置 |
| context weight | `thalamic_gate` | 丘脑门控强度 |
| memory prior | `hippocampal_reactivation_bias` | 海马重激活偏置 |
| identity prior | `self_model_prior` | 自我模型先验 |
| authenticity penalty | `self_distortion_penalty` | 自我失真惩罚 |
| vitality | `global_brain_state` | 全局脑态 |
| conflict spike | `acc_conflict_signal` | 冲突监测信号 |
| salience jump | `salience_burst` | 显著性突发 |
| resource pressure | `metabolic_pressure` | 代谢/资源压力 |

# 十一 功能更新 但保持0.57中概率场的设计规范
### A. 调度与总线层

#### 1. `RuntimeController`
- **主对应**：丘脑-皮层-基底节之间的路由编排，不是单一脑区
- **调制归属**：不直接等价于 dopamine / norepi / serotonin 中任何一个
- **真实效用对齐**：
  - 更像“多回路编排壳”
  - 负责时序、门控、回写、恢复
- **工程原则**：
  - 不要把它叫“前额叶”
  - 它应该是**回路编排器**，不是认知本体

#### 2. `SkillExecutor`
- **主对应**：前额叶-纹状体的“外部动作 affordance 评估” + 额顶控制网络的工具调用控制
- **主调制**：
  - dopamine：工具值得不值得试
  - serotonin：是否克制高成本/高风险调用
  - norepinephrine：在高不确定时是否切换到外部工具
- **真实效用对齐**：
  - 不像“技能模块”，更像“行动可能性评估器”
  - 真实人脑不会把工具视为外挂，而是视为延伸动作系统

---

### B. 输入前处理 / 状态信号层

#### 3. `Salience`
- **主对应**：显著性网络（anterior insula / ACC 功能组合）
- **主调制**：norepinephrine
- **次调制**：acetylcholine
- **真实效用对齐**：
  - 不是“重要性分数”
  - 而是“哪些东西值得抢占当前处理资源”
- **Structured Instability 表现**：
  - 显著性过高：系统被小刺激劫持
  - 显著性过低：系统迟钝、抓不住关键信号

#### 4. `Body`
- **主对应**：内感受与自主状态映射（insula / hypothalamic body-state integration）
- **主调制**：
  - serotonin：稳定性、抑制性
  - norepinephrine：警觉/应激
- **真实效用对齐**：
  - 不是“生理字段”
  - 而是整体脑态的底噪和阈值条件
- **Structured Instability 表现**：
  - 身体负荷高时，系统应更短视、更保守或更易烦躁

#### 5. `Emotion`
- **主对应**：边缘系统的情绪偏置层（amygdala-centered bias，不等于单一杏仁核）
- **主调制**：
  - norepinephrine：警觉性和急迫度
  - dopamine：趋近、冲动、期待
  - serotonin：情绪抑制与稳定
- **真实效用对齐**：
  - 情绪不是标签，而是对候选分布的增益/削弱机制
- **Structured Instability 表现**：
  - 可允许短时支配行为
  - 但必须被长期层与 guard 追踪，而不是完全无痕

#### 6. `Relationship`
- **主对应**：社会认知与依附边界系统（mPFC / TPJ / attachment-related valuation）
- **主调制**：
  - oxytocin-like social bonding analogue（工程上可单独命名为 `affiliation_modulator`）
  - serotonin：边界保持
- **真实效用对齐**：
  - 决定披露、亲近、安抚、疏离、界限
- **Structured Instability 表现**：
  - 高关系张力时，可能放大迎合、退缩、防御或过度解释

#### 7. `Resource`
- **主对应**：机会成本 / 能量预算 / 执行代价估计
- **主调制**：
  - dopamine：是否值得投入更多资源
  - serotonin：是否抑制高代价冲动
- **真实效用对齐**：
  - 更像“脑是否愿意为这个方向花代价”
  - 不是简单 quota

---

### C. 执行与候选生成层

#### 8. `PFCAgent`
- **主对应**：前额叶执行控制网络
- **主调制**：
  - dopamine：目标维持、奖励驱动
  - norepinephrine：在高冲突/高负荷时提高控制增益
  - serotonin：抑制短视冲动、维持长期一致性
- **真实效用对齐**：
  - 真正作用不是 if-else
  - 而是“维持目标 + 抑制底层冲动 + 维持任务集”
- **Structured Instability 表现**：
  - dopamine 过高：过于激进、追求高回报动作
  - norepinephrine 过高：僵硬、过紧、反复确认
  - serotonin 过高：过度克制、迟疑

#### 9. `Value`
- **主对应**：轨道额叶/腹内侧前额叶的价值评估功能
- **主调制**：dopamine
- **真实效用对齐**：
  - 给候选打长期/短期价值，不直接输出 winner
- **Structured Instability 表现**：
  - 价值信号飘高时会“过度解释机会”
  - 飘低时会“无动力、无意义感”

#### 10. `Perspective`
- **主对应**：视角切换与心理理论网络
- **主调制**：
  - acetylcholine：上下文切换敏感性
  - norepinephrine：在冲突和新奇下切换视角
- **真实效用对齐**：
  - 不是只是“多角度想想”
  - 而是是否真的切换到他者/未来/外部观察者框架

---

### D. 记忆、自发流与习惯层

#### 11. `Hippocampus`
- **主对应**：海马体及其情境记忆索引功能
- **主调制**：
  - acetylcholine：编码/检索切换
  - dopamine：新奇促进记忆标记
  - norepinephrine：高唤醒事件更易被激活
- **真实效用对齐**：
  - 不应只是“检索文本”
  - 而应是“情境模式重激活 + 记忆索引”
- **Structured Instability 表现**：
  - stress 下更容易被高情绪记忆绑架
  - 需要 memory write gate 和污染负反馈

#### 12. `Habit`
- **主对应**：背侧纹状体-习惯回路
- **主调制**：
  - dopamine：强化习惯路径
- **真实效用对齐**：
  - 把重复成功路径做成默认转移
- **Structured Instability 表现**：
  - 习惯可压过新目标
  - 但不应永久不可逆

#### 13. `Desire`
- **主对应**：趋近驱动 / 奖励寻求 / incentive salience
- **主调制**：dopamine
- **真实效用对齐**：
  - 不是“想要”字段
  - 而是候选趋近增益器
- **Structured Instability 表现**：
  - 高 dopamine 时显著放大
  - 很像人类的“想去做、想得到、想靠近”

#### 14. `DMN`
- **主对应**：默认模式网络
- **主调制**：
  - acetylcholine 低、外部任务控制弱时更活跃
  - serotonin 与 self-related stability 有关
- **真实效用对齐**：
  - 负责自发联想、自我叙事、内在连续性
- **Structured Instability 表现**：
  - 可导致沉浸自我叙事、过度联想、跑题
  - 但这恰恰是“活人感”的来源之一

---

### E. 仲裁、注意与守卫层

#### 15. `ConflictMonitorAgent`
- **主对应**：ACC / 冲突监测与控制招募功能
- **主调制**：
  - norepinephrine：高冲突时提高系统警觉
  - serotonin：抑制多峰同时爆发
- **真实效用对齐**：
  - 不是规则判断器
  - 而是“检测互斥峰值并招募更强控制”
- **Structured Instability 表现**：
  - 冲突监测不足：系统随便选
  - 冲突监测过强：系统反复犹豫、停转

#### 16. `ThalamusAttentionAgent`
- **主对应**：丘脑门控 + 注意力路由
- **主调制**：
  - acetylcholine：选择性增强输入通道
  - norepinephrine：在高显著/高不确定条件下重新分配注意
- **真实效用对齐**：
  - 不是“过滤器”
  - 而是“输入流量控制器”
- **Structured Instability 表现**：
  - 可允许短时把某类意外信号放得过大
  - 但必须保留回落能力

#### 17. `BehaviorPlausibilityGuard`
- **主对应**：前额叶抑制回路 + 现实可行性检查
- **主调制**：serotonin / GABA-like inhibition
- **真实效用对齐**：
  - 更像“别这么做”的抑制层
- **Structured Instability 表现**：
  - guard 太弱：冲动泛滥
  - guard 太强：像病理性自抑

#### 18. `OutputGate / Renderer`
- **主对应**：运动输出整形 / 语言表达门控
- **主调制**：
  - 不应承载主要神经调制语义
  - 只承接最后表达成形
- **真实效用对齐**：
  - 不是决策器
  - 而是“说出来之前怎么整形”

---

### F. 慢变量、身份与长期层

#### 19. `IdentityRuntime`
- **主对应**：自我模型与稳定先验（mPFC/self-model analogue）
- **主调制**：
  - serotonin：稳定、自我边界
  - DMN-related self continuity
- **真实效用对齐**：
  - 不是“人设 prompt”
  - 而是“连续自我先验”
- **Structured Instability 表现**：
  - 在压力、冲突、关系张力下可收缩或摇摆
  - 但长期不应无锚漂移

#### 20. `AuthenticityPolicy`
- **主对应**：自我一致性与不失真抑制
- **主调制**：
  - serotonin：抑制迎合
  - ACC-like conflict with self-prior
- **真实效用对齐**：
  - 当输出偏离自我先验时施加惩罚
- **Structured Instability 表现**：
  - 不是永远正确
  - 而是会在某些关系/压力条件下失守，并被 trace 看到

#### 21. `VitalityEngine`
- **主对应**：全局脑态/唤醒/精力调制
- **主调制**：
  - norepinephrine：唤醒
  - dopamine：动机
  - serotonin：稳定与恢复
- **真实效用对齐**：
  - 决定系统今天是兴奋、虚弱、迟钝、敏感还是收缩
- **Structured Instability 表现**：
  - 这是最适合“允许波动”的层
  - 但不能没有底线恢复机制

#### 22. `LongRunAnalyzer`
- **主对应**：长期监控与误差累积审计，不是单一脑区
- **调制归属**：更像元控制分析器
- **真实效用对齐**：
  - 相当于“系统有没有在长期变坏”
- **Structured Instability 表现**：
  - 负责发现 drift，而不是阻止一切 drift

#### 23. `DreamOrchestrator`
- **主对应**：离线重放、重组、巩固、去噪
- **主调制**：
  - acetylcholine-style encoding/replay switch
  - serotonin / slow stability recovery
- **真实效用对齐**：
  - 不是瞎总结
  - 而是“离线状态下的自我-记忆-关系重整”
- **Structured Instability 表现**：
  - 梦境重放可放大某些模式，也可错误固化
  - 因此 dream 需要对 memory 污染、身份崩塌、关系偏置做二次监控