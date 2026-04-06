# NALR 概率化类脑架构升级指南（高质量工程版）

## 一、目标

本方案的目标不是重写系统，而是在**不违背原有框架宣言**的前提下：

- 降低单位 tick 成本
- 提升运行频率
- 统一概率驱动机制
- 保持可观测性与一致性

核心原则：

> 所有模块只通过影响概率分布参与决策

---

## 二、核心改造方法

### 1. 概率统一接口（最关键）

所有模块输出统一结构：

```python
class ProbabilisticContribution:
    module_name: str
    delta_logits: Optional[List[float]]
    attention_bias: Optional[List[float]]
    soft_mask: Optional[List[float]]
    hard_mask: Optional[List[bool]]
    confidence: float
    trace_reason: str
```

统一融合：

```python
logits_final = logits_base + Σ(delta_logits)
```

---

### 2. 并行替代串行

旧：

PFC → Conflict → Guard → Sampling

新：

所有模块并行 → 概率融合 → 一次采样

---

### 3. micro / macro tick 分层

| 类型 | 频率 | 内容 |
|------|------|------|
| micro tick | 高频（10~100ms） | 状态更新、轻量 Δ |
| macro tick | 低频（>300ms） | LLM、复杂推理 |

---

### 4. 模块频率分级

- 每 tick：Emotion / Body / Vitality
- 每 N tick：Habit / Desire
- 条件触发：ConflictMonitor
- macro tick：PFC / LLM

---

### 5. Guard 软硬分离

- soft：概率惩罚（常态）
- hard：违规强制阻断

---

### 6. Memory & Thalamus 前置

- Thalamus：上下文 attention bias
- Hippocampus：memory → logits bias

---

## 三、必须防范的风险

### 1. 模块重新变成“决策者”

❌ 错误：

if PFC decides: output A

✔ 正确：

PFC 输出 Δ_pfc

---

### 2. 隐式逻辑污染

❌ renderer 修改内容  
❌ guard 直接覆盖结果  

---

### 3. LLM 滥用

每个模块调用 LLM → 性能崩溃

---

### 4. 假 micro tick

空循环，没有状态演化 → 架构失效

---

### 5. 串行伪并行

模块仍按顺序执行 → 无性能提升

---

## 四、验证方法（强烈建议实施）

### 1. 分布一致性验证

检查：

- ΣΔ 是否等价于最终变化
- 无模块绕过融合器

---

### 2. 性能基准

指标：

- tick 时间
- 吞吐稳定性
- LLM 调用次数

---

### 3. 可解释性验证

必须能回答：

- 哪个模块改变了概率？
- 改变了多少？

---

### 4. 冲突验证

构造：

- 多峰输入
- 检查是否合理收敛

---

### 5. Counterfactual 测试

移除某模块：

- 输出变化是否合理？

---

### 6. 长程稳定性（longrun smoke）

- 无漂移
- 无爆炸
- 无卡死

---

## 五、验收标准

✔ 无模块直接决策  
✔ 单次采样完成决策  
✔ tick 时间下降  
✔ trace 可解释  
✔ longrun 稳定  

---

## 六、总结

这次改造的本质：

> 从“模块流程系统”升级为“概率动力系统”

关键不是更多模块，而是：

- 模块统一表达
- 概率统一融合
- 时间分层执行

---

## 最重要一句话

> 如果一个模块不能表达为对概率分布的影响，它就不应该存在于系统中。
