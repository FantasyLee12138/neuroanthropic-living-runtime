# OneiroiAgent（梦神 Agent）集成方案（面向 Codex 落地）

> 基于用户提供的原始需求整理而成，目标是把概念性设想转成 **Codex 可理解、可拆解、可实现、可验收** 的工程方案。原始需求来源：fileciteturn0file0

---

## 1. 目标与边界

### 1.1 目标
新增一个**完全独立于主程序**的 sidecar agent：`OneiroiAgent`，仅在主程序进入 `sleep / deep idle` 时运行，用于完成以下离线闭环：

1. 读取主程序的记忆与状态快照（只读）
2. 基于独立 dream token 预算执行“梦境生成 / 情绪整合 / 记忆巩固”
3. 向主程序提交**写入申请**，而不是直接改写主程序状态
4. 支持醒后快速遗忘、情绪残留、轻微状态偏移
5. 可通过 CLI 完整观测、调试、追溯、禁用

### 1.2 非目标
以下内容**不在 OneiroiAgent 权限范围内**：

- 不直接修改主程序核心 schema
- 不直接修改保护级身份认知 / 生存规则 / 边界硬约束
- 不参与 interactive / task 主链路
- 不占用主程序交互 token 预算
- 不在清醒态持续驻留或轮询

### 1.3 成功标准
该方案落地后应满足：

- 主程序清醒态性能不受影响
- 梦境资源独立核算
- 任意写入必须经过主程序校验
- 梦境记忆可快速衰减
- 整体过程有 trace、有 CLI、有开关、有兜底降级

---

## 2. 总体架构

```text
+-----------------------------+
| Main NALR Runtime           |
|-----------------------------|
| Interactive / Task Loop     |
| Memory System               |
| Control Plane               |
| BehaviorPlausibilityGuard   |
| Sleep State Manager         |
+-------------+---------------+
              |
              | read-only snapshot
              | write proposals only
              v
+-----------------------------+
| OneiroiAgent (sidecar)      |
|-----------------------------|
| Sleep Trigger Listener      |
| Budget Calculator           |
| Memory Fragment Sampler     |
| Emotion Anchor Builder      |
| Dream Composer (LLM)        |
| Consolidation Planner       |
| Forgetting Scheduler        |
| Dream Trace Writer          |
+-----------------------------+
```

### 2.1 关键原则
- **完全解耦**：sidecar 独立进程 / 独立生命周期 / 独立资源池
- **最小耦合接口**：只允许“只读快照 + 写入申请”
- **主程序裁决**：OneiroiAgent 无直接写权限
- **随时降级**：关闭后退回原生 sleep 流程

---

## 3. 运行模型

## 3.1 生命周期
`OneiroiAgent` 只在以下状态切换中运行：

1. 主程序进入 `sleep_pending`
2. 主程序生成只读 snapshot
3. 唤起 `OneiroiAgent`
4. `OneiroiAgent` 完成梦境流程
5. 输出 `proposal_bundle + trace`
6. 主程序 `Control Plane + BehaviorPlausibilityGuard` 审核
7. 审核通过的 proposal 才生效
8. sidecar 退出或回休眠

## 3.2 状态机

```text
OFF
  -> IDLE
  -> ARMED
  -> RUNNING
  -> PROPOSING
  -> DONE
  -> IDLE

异常流:
RUNNING -> FAILED -> IDLE
PROPOSING -> VETOED -> IDLE
```

### 3.3 触发条件
满足以下条件才允许触发：

- `dream.enabled == true`
- 主程序状态为 `sleep` 或 `deep_idle`
- 当前无 interactive / task 活动
- 主程序生存资源允许离线活动
- 未命中全局熔断 / 安全锁

---

## 4. 进程与接口设计

## 4.1 进程形态
建议作为独立 sidecar 进程部署：

- 二进制 / 服务名：`oneiroi-agent`
- 启动方式：
  - 主程序 sleep hook 拉起
  - 或常驻空闲进程，收到 sleep event 后短时运行
- 推荐优先级：低优先级，避免抢占主程序资源

## 4.2 主程序 -> OneiroiAgent（只读）
建议暴露只读接口：

### `GET /v1/sleep-snapshot`
返回 sleep 时刻快照：

```json
{
  "sleep_session_id": "sleep_2026_04_04_xxx",
  "timestamp": "2026-04-04T23:30:00Z",
  "resource_state": {
    "daily_token_surplus_rate": 0.32,
    "fatigue_level": 0.18,
    "resource_scarcity": 0.12
  },
  "emotional_baseline": {
    "valence": -0.1,
    "arousal": 0.45,
    "dominant_emotion": "委屈"
  },
  "memory_refs": {
    "hot": ["mem_hot_1", "mem_hot_2"],
    "warm": ["mem_warm_8"],
    "archive_sample_pool": ["mem_arc_20", "mem_arc_21"]
  },
  "conflict_events": [
    {"id": "evt_1", "score": 0.73}
  ],
  "relationship_state": {
    "closeness": 0.44,
    "trust": 0.58,
    "boundary_tension": 0.26
  }
}
```

### 只读范围
可读：
- episodic memory 索引 / 摘要
- 情绪基线
- 习惯链路
- 冲突事件
- 身体 / 资源状态
- 关系状态摘要

不可读 / 或只允许摘要化读取：
- 主程序核心保护级 schema 原文
- 安全规则详细策略实现
- 高敏感系统指令正文

## 4.3 OneiroiAgent -> 主程序（写入申请）
建议单一写入申请接口：

### `POST /v1/dream-proposals`
```json
{
  "sleep_session_id": "sleep_2026_04_04_xxx",
  "dream_run_id": "dream_run_001",
  "proposal_bundle": {
    "memory_consolidation": [],
    "emotion_adjustments": [],
    "habit_adjustments": [],
    "dream_memory_write": [],
    "relationship_adjustments": []
  },
  "trace_ref": "trace://dream/2026/04/04/run_001.json"
}
```

### proposal 类型
仅允许以下五类：
1. `memory_consolidation`
2. `emotion_adjustments`
3. `habit_adjustments`
4. `dream_memory_write`
5. `relationship_adjustments`

### 审核规则
必须经过：
- `Control Plane` 规则校验
- `BehaviorPlausibilityGuard` 合理性校验
- trace 落盘成功
- 硬约束扫描未命中

---

## 5. 配置模型

建议新增配置文件：`config/dream.yaml`

```yaml
dream:
  enabled: true

  mode: normal   # none | light | normal | deep

  budget:
    base_dream_budget: 1000
    min_dream_budget: 0
    max_dream_budget: 8000
    min_generation_threshold: 500

  forgetting:
    dream_memory_decay_per_turn: 0.15
    strong_emotion_decay_per_turn: 0.08
    recalled_decay_per_turn: 0.03
    archive_after_turns: 20

  proposal_limits:
    emotion_baseline_delta_per_run: 0.08
    relationship_delta_per_run: 0.05
    conflict_relief_per_run: 0.10
    cumulative_state_delta_cap: 0.15

  sampling:
    fragment_count_min: 10
    fragment_count_max: 30
    same_day_emotion_bias: 0.20
    random_pool_bias: 0.80

  generation:
    min_scene_jumps: 3
    absurdity_scale_min: 0.1
    absurdity_scale_max: 1.0
    enforce_fragmented_narrative: true

  trace:
    enabled: true
    store_full_prompt: false
    store_fragment_ids: true
```

---

## 6. Dream Token 预算设计

## 6.1 预算公式

```python
dream_token_budget = clip(
    base_dream_budget
    * (1 + daily_token_surplus_rate)
    * (1 - fatigue_level)
    * (1 - resource_scarcity),
    min_dream_budget,
    max_dream_budget
)
```

## 6.2 含义
- 当日 token 结余越多，预算越高
- 疲劳越高，预算越低
- 资源越匮乏，预算越低
- 硬上限兜底，避免梦境过度消耗

## 6.3 模式映射
建议定义：

- `deep`：强制无梦，预算视为 0，仅执行规则化整理
- `none`：完全关闭 dream 流程
- `light`：限制预算上限，如 1200
- `normal`：按公式正常计算

## 6.4 执行阈值
- `budget == 0`：无梦深睡，仅规则整理
- `0 < budget < 500`：不做 LLM dream generation，仅做情绪整合 + 归档
- `budget >= 500`：执行完整 dream generation

---

## 7. Token 消耗拆账

建议固定成预算分配策略，便于 trace 与调试：

| 环节 | 默认占比 | 是否必须 | 说明 |
|---|---:|---|---|
| 记忆碎片采样与整理 | 10% | 是 | 零预算时退化为规则化处理 |
| 情绪整合与冲突消解 | 15% | 是 | 零预算时不调用模型 |
| 梦境内容生成 | 60% | 否 | 仅预算达到阈值时开启 |
| 记忆归档与巩固 | 15% | 是 | 必做，支持规则化 |

建议输出到 trace：

```json
{
  "budget_total": 1460,
  "spent": {
    "sampling": 120,
    "emotion_integration": 180,
    "dream_generation": 920,
    "consolidation": 140
  },
  "unused": 100
}
```

---

## 8. 梦境生成流水线

## 8.1 Pipeline 总览

```text
Sleep Trigger
 -> Budget Calculation
 -> Memory Fragment Sampling
 -> Emotion Anchor Selection
 -> Cross-Memory Stitching
 -> Absurdity Perturbation
 -> LLM Dream Generation (optional)
 -> Dream Memory Staging
 -> Consolidation Proposal Build
 -> Trace Persist
 -> Proposal Submit
```

## 8.2 Step 1：记忆碎片采样
目标：制造“无序但有情绪锚点”的原料。

### 输入
- hot / warm / archive 记忆索引
- 当日冲突事件
- 情绪基线
- 习惯链路摘要
- 常识 / 场景基底

### 规则
- 抽取 10~30 个 fragments
- 当日核心情绪事件只给予轻微偏置（20%）
- 其余 80% 走随机采样
- 允许混入历史小事、抽象概念、旧关系片段

### 输出
```json
{
  "fragment_ids": ["f1", "f8", "f19"],
  "fragment_types": ["today_event", "archive_scene", "habit_trace"]
}
```

## 8.3 Step 2：核心情绪锚定
从 sleep snapshot 中选取单一主导情绪，作为 dream tone：

- 压抑
- 委屈
- 愉悦
- 焦虑
- 放松
- 期待

输出：
```json
{
  "dream_emotion_anchor": "委屈",
  "intensity": 0.74
}
```

## 8.4 Step 3：跨记忆无逻辑拼接
这里不追求因果合理性，只追求“情绪连续 + 内容跳跃”。

### 规则
- 禁止整理成完整剧情
- 至少 3 次场景 / 人物 / 事件跳转
- 允许时间线错位、角色替换、空间折叠
- 禁止因果修补器自动补全为标准故事

输出中间表示：
```json
{
  "dream_skeleton": [
    "雨天面馆 + 被否定身份",
    "旧街道 + 陌生人借用了用户语气",
    "房间变成车站 + 桌子漂浮"
  ]
}
```

## 8.5 Step 4：随机扰动与荒诞强化
根据预算映射荒诞度：

```python
absurdity_scale = normalize(dream_token_budget, min=500, max=8000)
```

可注入扰动：
- 人物身份替换
- 物理规则失效
- 时间顺序错乱
- 地点连续跳变
- 细节互相矛盾

## 8.6 Step 5：LLM 梦境文本生成
仅此步骤可调用模型。

### Prompt 约束（建议）
```text
你要生成一段“像人类梦境”的文本。
要求：
1. 情绪基调统一，但情节必须碎片化。
2. 场景、人物、事件要随机跳转，不要形成完整因果链。
3. 至少出现 3 次以上明显跳转。
4. 可以荒诞、矛盾、违背物理规则。
5. 禁止写成逻辑完整、结构工整的短篇故事。
6. 输出第一人称、体验流、梦中即时感受。
```

### 输出对象
```json
{
  "dream_text": "...",
  "dominant_emotion": "委屈",
  "emotion_intensity": 0.74,
  "scene_jump_count": 5,
  "absurdity_score": 0.62
}
```

---

## 9. 醒后遗忘机制

## 9.1 双层写入
梦境结束后分两层写入：

### A. 临时清晰记忆
- 存入 `episodic_memory.hot`
- 类型：`dream_memory`
- 初始召回率：100%
- 使用超高速衰减

### B. 永久情绪残留
- 存入 `emotional_history`
- 只保留核心情绪 / 偏移建议
- 按普通情绪记忆衰减

## 9.2 衰减规则

默认：
- `dream_memory_decay_per_turn = 0.15`

特殊：
- 强情绪（`intensity >= 0.8`） -> `0.08`
- 主动回忆后 -> `0.03`

### 回合表现
| 唤醒后轮次 | 状态 | 表现 |
|---|---|---|
| 0-3 | 完整可召回 | 能描述较完整梦境 |
| 3-10 | 快速遗忘 | 只剩 gist 和情绪 |
| 10-20 | 近完全遗忘 | 只记得“做过梦” |
| 20+ | 完全归档 | 内容不可召回，只剩残留偏移 |

## 9.3 不可回溯原则
超过 20 轮后：
- 主程序常规记忆检索不再返回完整 dream text
- 完整文本只保留于离线 trace
- 默认不参与日常召回

---

## 10. 对主程序的影响闭环

OneiroiAgent 不直接改写主程序，仅提出**微调建议**。

## 10.1 允许影响的四个维度

### 1) 情绪基线残留
- 单次最大偏移：`±0.08`

### 2) 记忆巩固与归档
- hot -> warm -> archive 的推进
- 低价值碎片修剪建议

### 3) 冲突消解
- 单次最大缓和：`-0.10`

### 4) 关系 / 边界微调
- 单次最大偏移：`±0.05`

## 10.2 硬约束
必须在主程序守卫里明文编码：

- 不得修改保护级 schema
- 单轮任意状态偏移不得超过 0.08
- 累计偏移不得超过 0.15
- 所有 proposal 必须有 trace
- 未通过 guard 的 proposal 直接 veto

---

## 11. 数据结构建议

## 11.1 DreamRun
```json
{
  "dream_run_id": "dream_run_001",
  "sleep_session_id": "sleep_2026_04_04_xxx",
  "mode": "normal",
  "budget_total": 1460,
  "budget_spent": 1360,
  "emotion_anchor": {
    "name": "委屈",
    "intensity": 0.74
  },
  "dream_text_ref": "dream://run/001",
  "proposal_bundle_ref": "proposal://run/001",
  "trace_ref": "trace://dream/001"
}
```

## 11.2 DreamMemory
```json
{
  "memory_id": "dream_mem_001",
  "type": "dream_memory",
  "layer": "hot",
  "created_at": "2026-04-04T23:59:00Z",
  "decay_per_turn": 0.15,
  "recall_score": 1.0,
  "emotion_anchor": "委屈",
  "dream_text": "..."
}
```

## 11.3 DreamProposalBundle
```json
{
  "memory_consolidation": [
    {
      "target_memory_id": "mem_hot_1",
      "action": "promote_to_warm",
      "reason": "replayed_in_dream"
    }
  ],
  "emotion_adjustments": [
    {
      "dimension": "anxiety",
      "delta": 0.03,
      "reason": "nightmare_residue"
    }
  ],
  "relationship_adjustments": [
    {
      "entity": "user",
      "dimension": "boundary_tension",
      "delta": 0.04,
      "reason": "dream_interaction_negative"
    }
  ]
}
```

---

## 12. CLI 设计

建议复用现有 CLI 风格：

| 指令 | 作用 |
|---|---|
| `alive dream enable` | 开启 dream 模块 |
| `alive dream disable` | 关闭 dream 模块 |
| `alive dream budget set --min 0 --max 8000 --base 1000` | 设置预算参数 |
| `alive dream recall` | 查看当前仍可召回的梦境 |
| `alive dream trace last` | 查看最近一次 dream trace |
| `alive dream mode set deep` | 设置深睡 / 无梦 |
| `alive dream mode set light` | 设置浅梦 |
| `alive dream mode set normal` | 设置正常梦境 |
| `alive dream mode set none` | 禁止 dream 运行 |
| `alive dream forget` | 强制遗忘当前梦境内容 |

建议再补两条：
- `alive dream status`
- `alive dream simulate --budget 1200`

---

## 13. Trace 与可观测性

## 13.1 Trace 必备字段
```json
{
  "dream_run_id": "dream_run_001",
  "trigger_reason": "sleep_entered",
  "budget": {
    "base": 1000,
    "surplus_rate": 0.32,
    "fatigue": 0.18,
    "resource_scarcity": 0.12,
    "final": 1460
  },
  "sampling": {
    "fragment_count": 16,
    "fragment_ids": ["f1", "f2", "f3"]
  },
  "generation": {
    "llm_called": true,
    "scene_jump_count": 5,
    "absurdity_score": 0.62
  },
  "proposals": {
    "submitted": 4,
    "approved": 3,
    "vetoed": 1
  },
  "timestamps": {
    "started_at": "...",
    "finished_at": "..."
  }
}
```

## 13.2 必须观测的指标
- dream 触发次数
- 平均预算 / 平均消耗
- 有梦 vs 无梦比例
- veto 率
- recall 存活轮数
- dream 对情绪 / 关系偏移分布
- 异常退出率

---

## 14. 安全与守卫

## 14.1 权限边界
OneiroiAgent：
- 可读 snapshot
- 可生成 proposal
- 可写 trace
- 不可直接写主程序状态

主程序：
- 唯一状态写入方
- 唯一规则裁决方

## 14.2 守卫规则清单
主程序应对 proposal 执行：

1. schema 保护检查
2. delta 越界检查
3. 累积漂移检查
4. 合理性检查
5. trace 完整性检查
6. 幂等性检查
7. 重放攻击 / 重复提交检查

---

## 15. 推荐目录结构

```text
/agents/oneiroi/
  __init__.py
  sidecar.py
  lifecycle.py
  budget.py
  sampler.py
  emotion_anchor.py
  stitcher.py
  perturbation.py
  generator.py
  consolidation.py
  forgetting.py
  proposals.py
  trace.py
  schemas.py
  cli.py
  tests/

config/
  dream.yaml

docs/
  oneiroi-agent.md
```

---

## 16. 最小可落地版本（MVP）

建议分三期做，避免一次性过重。

## Phase 1：无模型版 dream 基础框架
目标：先打通 sidecar 和 proposal 流。

包含：
- sidecar 生命周期
- sleep trigger
- budget 计算
- fragment sampling
- 规则化情绪整合
- proposal 提交
- trace 落盘
- CLI 开关

不包含：
- LLM dream generation
- 复杂荒诞扰动
- 主动回忆延缓遗忘

### 验收
- 能在 sleep 时独立运行
- 能提交 proposal
- 能被主程序 veto / approve
- 不影响清醒态主链路

## Phase 2：引入梦境文本生成
包含：
- stitcher
- perturbation
- LLM dream generation
- dream recall
- forget CLI
- scene jump / absurdity 评估

### 验收
- 梦文本具备碎片性
- 不会生成过度工整叙事
- recall 可用
- trace 可追溯

## Phase 3：完整遗忘与状态联动
包含：
- 超高速衰减
- 强情绪慢遗忘
- 主动回忆延缓遗忘
- relationship / conflict 微调闭环
- 完整指标面板

### 验收
- 20 轮内基本遗忘
- 情绪残留仍存在
- 所有偏移可解释可追溯

---

## 17. Codex 执行任务拆分

下面这部分是给 Codex / 自动编码代理最直接的任务拆单。

## Task 1：建立配置与 schema
**目标**
- 新增 `config/dream.yaml`
- 新增 `schemas.py`

**输出**
- `DreamConfig`
- `DreamRun`
- `DreamProposalBundle`
- `DreamMemory`

**验收**
- schema 可序列化 / 反序列化
- 默认值完整

## Task 2：实现预算计算器
**目标**
- 新增 `budget.py`

**函数**
- `compute_dream_budget(snapshot, config) -> int`

**验收**
- 覆盖 0 / 低 / 中 / 高预算场景
- 满足 min / max clip

## Task 3：实现 sidecar 生命周期
**目标**
- 新增 `sidecar.py` + `lifecycle.py`

**函数**
- `run_sleep_cycle(snapshot) -> DreamRun`

**验收**
- 可从 snapshot 启动到 proposal 输出
- 异常能优雅退出

## Task 4：实现 fragment sampling
**目标**
- 新增 `sampler.py`

**函数**
- `sample_fragments(snapshot, config) -> list[Fragment]`

**验收**
- 抽样数落在配置范围
- 当日核心情绪事件仅轻微偏置

## Task 5：实现 emotion anchor
**目标**
- 新增 `emotion_anchor.py`

**函数**
- `build_emotion_anchor(snapshot) -> EmotionAnchor`

**验收**
- 输出稳定
- 能正确选主导情绪和强度

## Task 6：实现 stitcher + perturbation
**目标**
- 新增 `stitcher.py`、`perturbation.py`

**函数**
- `build_dream_skeleton(fragments, anchor, config)`
- `inject_absurdity(skeleton, budget, config)`

**验收**
- 至少生成 3 次场景跳转
- 不自动整理为完整故事

## Task 7：实现 LLM generation
**目标**
- 新增 `generator.py`

**函数**
- `generate_dream_text(skeleton, anchor, budget, config)`

**验收**
- budget < threshold 时不调用模型
- 生成文本符合碎片化要求

## Task 8：实现 proposal builder
**目标**
- 新增 `consolidation.py`、`proposals.py`

**函数**
- `build_proposal_bundle(dream_run, snapshot, config)`

**验收**
- proposal 不越过硬约束
- 可被主程序进一步审核

## Task 9：实现遗忘机制
**目标**
- 新增 `forgetting.py`

**函数**
- `tick_dream_decay(memory, interaction_turns)`
- `recall_dream(memory)`

**验收**
- 0-3 / 3-10 / 10-20 / 20+ 轮表现符合预期

## Task 10：实现 CLI
**目标**
- 新增 `cli.py`

**验收**
- enable / disable / recall / trace / mode / forget 全部可用

## Task 11：补测试
**测试类型**
- 单元测试
- 守卫集成测试
- 预算边界测试
- 遗忘曲线测试
- veto 流测试
- sidecar 崩溃回退测试

---

## 18. 伪代码参考

```python
def on_main_runtime_enter_sleep():
    if not dream_config.enabled:
        return native_sleep()

    snapshot = main_runtime.build_sleep_snapshot()

    if snapshot.mode == "none":
        return native_sleep()

    run = oneiroi_agent.run_sleep_cycle(snapshot)

    if not run:
        return native_sleep()

    approved = main_runtime.control_plane.review(run.proposal_bundle, run.trace_ref)

    main_runtime.apply_approved_proposals(approved)
    main_runtime.stage_dream_memory_if_approved(approved)

    return native_sleep()
```

```python
def run_sleep_cycle(snapshot):
    budget = compute_dream_budget(snapshot, config)

    fragments = sample_fragments(snapshot, config)
    anchor = build_emotion_anchor(snapshot)

    if budget < config.budget.min_generation_threshold:
        dream_text = None
        skeleton = rule_based_stitch(fragments, anchor)
    else:
        skeleton = build_dream_skeleton(fragments, anchor, config)
        skeleton = inject_absurdity(skeleton, budget, config)
        dream_text = generate_dream_text(skeleton, anchor, budget, config)

    proposal_bundle = build_proposal_bundle(
        snapshot=snapshot,
        dream_text=dream_text,
        anchor=anchor,
        budget=budget,
        config=config,
    )

    trace_ref = persist_trace(...)
    submit_proposals(proposal_bundle, trace_ref)

    return DreamRun(...)
```

---

## 19. 风险与规避

## 风险 1：梦境输出过于“AI 故事化”
**规避**
- prompt 强约束
- 加 scene jump 检测
- 加反工整评分器

## 风险 2：proposal 过度影响主程序
**规避**
- 单轮上限
- 累计上限
- guard veto

## 风险 3：sidecar 崩溃影响主程序
**规避**
- 完全独立进程
- 主程序自动回退 native sleep

## 风险 4：dream recall 泄漏长期内容
**规避**
- 20 轮后常规不可召回
- 仅 trace 保留全文
- recall 受权限控制

---

## 20. 最终建议

最合理的落地方式不是把“梦境”塞进主 runtime，而是把它实现为：

- **独立 sidecar**
- **独立预算**
- **只读快照**
- **写入申请制**
- **主程序裁决制**
- **快速遗忘**
- **全量 trace 可观测**

这样既保留你原始设计里最重要的“像人类一样做梦、遗忘、留下情绪残影”的特征，也不会破坏主程序核心稳定性。整体上，这是一个**高可控、低侵入、易分期落地**的实现方案。

---

## 21. 交付建议给 Codex 的一句话任务描述

可以直接把下面这段贴给 Codex：

> 请基于仓库现有的 sleep / memory / control-plane / trace / CLI 架构，实现一个独立 sidecar 进程 `OneiroiAgent`。它只能读取主程序的 sleep snapshot，并提交 dream-related proposals，不能直接写主程序状态。先按 MVP 实现：budget 计算、fragment sampling、emotion anchor、规则化 consolidation、proposal bundle、trace、CLI 开关；随后再扩展 LLM dream generation、快速遗忘和 relationship/conflict 微调。实现时必须保证：主程序清醒态零影响、dream token 与主程序 token 预算隔离、所有状态变更都经过 Control Plane + BehaviorPlausibilityGuard 审核、支持完整 trace 与一键 disable 回退。
