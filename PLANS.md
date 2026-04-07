# TLH v1.2 Ledger

此文件是 TLH v1.2 的唯一状态真相面。

- `done`：仓库内已存在并已被代码或测试证实的能力
- `doing`：当前唯一活跃切片
- `not-done`：TLH v1.2 仍未完成的目标
- `verification evidence`：每个关键里程碑对应的验证命令或观测证据

## Done

### TLH-001 Frontline Docs Rebase

- 已将 [README.md](/Users/fantasylee/类脑架构/README.md) 改为 TLH v1.2 导航入口
- 已新增 [Think_Like_Human(TLH)_v1.0.md](/Users/fantasylee/类脑架构/Think_Like_Human(TLH)_v1.0.md) 作为前台 manifesto
- 已将 [BEREAL_v0.6_MASTER_SPEC.md](/Users/fantasylee/类脑架构/BEREAL_v0.6_MASTER_SPEC.md) 标记为 frozen substrate，并加入 TLH 映射关系

### TLH-002 Runtime Subject State Foundations

- 已在 [src/nalr/schemas/models.py](/Users/fantasylee/类脑架构/src/nalr/schemas/models.py) 增加 TLH 运行时一等状态：
  - `fatigue`
  - `memory_fragments`
  - `self_continuity`
  - `meaning_strength`
  - `base_metabolism`
- 已新增结构化状态：
  - `BodyState`
  - `SubjectiveState`
  - `InstinctFieldState`
  - `OrganicModeState`
  - `EmergentActionSketch`
- 已将这些状态接入 `RuntimeState` 并保持序列化/恢复兼容

### TLH-003 Instinct Field Initial Integration

- 已在 [src/nalr/runtime/controller.py](/Users/fantasylee/类脑架构/src/nalr/runtime/controller.py) 接入 `InstinctField` action contribution
- instinct field 初始支持 TLH 动作压力：
  - `absorb`
  - `nothing`
  - `die`
- 已将 instinct field snapshot 写入 trace state snapshot 与 cognitive snapshot

### TLH-004 TLH Action Semantics Initial Rendering

- 已在 [src/nalr/output/renderer.py](/Users/fantasylee/类脑架构/src/nalr/output/renderer.py) 增加 `absorb / nothing / die` 的 fallback 渲染语义
- `nothing` 当前默认表现为无外显文本输出

### TLH-005 Endogenous Suppression Grading

- 已将 `_endogenous_suppression_decision()` 从“task 场景硬压制”改为可被 `organic_mode.instinct_first` 放宽的分级抑制

### TLH-007 Subjectivity Mainline Feedback

- `EndogenousTickScheduler` 已将 `subjective_pressure` 接入主链触发判定，来源包括 `spontaneous`、`reject_all`、`meaning_made`、`fatigue`、`memory_fragments`、`self_continuity`
- `_endogenous_suppression_decision()` 已使用 `subjective_pressure` 放宽 task / non-companion 场景下的 endogenous 抑制阈值
- subjectivity 现在不只是 observer 指标，也会真实改写 scheduler / gate 的放行结果

### TLH-008 Replay / Observer Initial Upgrade

- `why_this()` 已暴露：
  - `subjective_state`
  - `instinct_field`
  - `organic_mode`
  - `counterfactual_replays`
- `replay()` 已不再只是原动作复述，当前会返回基于 competing peaks 和 instinct candidates 的 `counterfactual_replays`
- `/state` 与 `/why/{round}` 已可看到 TLH 初始 runtime 字段

### TLH-009 Emergent Actions Growth

- `_update_emergent_action_sketches()` 已将 instinct winner region、subjective pressure、action posterior 写入 `EmergentActionSketch`
- `EmergentActionSketch` 已进入 runtime state、trace snapshot、`why_this()`、`replay()` 与 observer 输出
- sketches 当前已经能在重复内部压力下自然增长，并保留 `signal_sources`、`support_actions`、`growth_score`、`status`

### TLH-006 Body / Emotion / Desire Deep Upgrade

- `BodyState` 已成为显式 interoception 结构体，并通过 `fatigue`、`memory_fragments`、`self_continuity`、`meaning_strength`、`base_metabolism` 回流主链
- `TLHMemoryBridge` 与 `organic_memory` coupling 已让 body / subjectivity 跨层影响 action field，而不是只停留在报表层
- emotion 的持续性状态继续由 `affect_residue`、vitality slow variables 与 `subjective_pressure` 共同承担
- desire 已通过 endogenous motivation pool、instinct field 与 emergent sketches 进入更接近 latent drive 的统一动作竞争

### Inherited Baseline

以下能力视为 TLH 继承完成项，不再重复作为 v1.0 新增目标：

- 四层概率真相面 `context -> memory -> action -> token`
- `ProbabilisticContribution`
- `ProbabilityFieldIntegrator`
- field-first controller 主链
- token source integrity
- memory write gate
- body / emotion / desire / vitality / identity / endogenous / dream 主链
- observer / trace / why / why-not / replay 基础能力
- `wander` 已在现有主链中是 first-class action

### TLH-010 Acceptance Closeout

- `plan / recall / connect / clarify` 已在 organic mode 下被明确降为 derived-action baseline，而不再天然压过 innate pool
- `organic_memory` coupling 已进入合法白名单并通过 trace / why 暴露
- `EmergentActionSketch` 已升级为后续轮次可回流 action field 的正式 contributor
- replay 已加入 seeded counterfactual action、preview 与 instinct / sketch 证据
- TLH v1.2 的当前 ledger 已清空 `doing / not-done`

### TLH-011 Workbench Delivery Closeout

- `Open_NALR_Workbench.command` 已成为正式 web-first 工作台入口，并支持向 `nalr.observer_launcher` 透传参数
- observer dashboard 已有正式 browser smoke pytest 入口，不再只有手工脚本
- `NALR_Alive_Console_Product_Design.md/.pdf` 与 `alive trace why latest.json` 已按正式交付物纳入 root 入口说明
- web session SSE 已对非有限数值做 JSON-safe 清洗，避免 dashboard 事件流解析间歇失效

### TLH-012 Vector-First Subject Core

- 已新增 [src/nalr/runtime/tlh_vectors.py](/Users/fantasylee/类脑架构/src/nalr/runtime/tlh_vectors.py)，统一承载 TLH 的主体向量、调制方向、sketch 向量反推与行为向量映射
- [src/nalr/runtime/probability_field.py](/Users/fantasylee/类脑架构/src/nalr/runtime/probability_field.py) 已新增 `compute_tlh_vector_collapse()`，用主体向量与行为向量的余弦相似度替换旧的随机点欧式坍缩
- [src/nalr/runtime/controller.py](/Users/fantasylee/类脑架构/src/nalr/runtime/controller.py) 中的 instinct field 已改为组装 `V_main / V_mod / V_anchor` 并调用正式向量坍缩算子；`collapse_trace` 保留兼容键，同时新增 `v_main`、`v_mod`、`v_anchor`、`subject_vector`、`action_vectors`
- `personality_anchor.axis_baseline` 已改为基于历史主体向量的在线 EMA 人格锚点，不新增持久化 schema
- `EmergentActionSketch` 仍不新增存储字段，但其回流 action field 的方式已改为 sketch 向量反推后再映射回动作势能

## Doing

- none

## Not-Done

- none

## Verification Evidence

### TLH v1.2 Runtime Slice

已执行：

```bash
cd /Users/fantasylee/类脑架构/apps/terminal && npm test
cd /Users/fantasylee/类脑架构/apps/terminal && npm run build
pytest /Users/fantasylee/类脑架构/tests/unit/test_tlh_runtime.py -q
pytest /Users/fantasylee/类脑架构/tests/unit/test_output_expression_layer.py -q
pytest /Users/fantasylee/类脑架构/tests/unit/test_probability_field.py -q
pytest /Users/fantasylee/类脑架构/tests/unit/test_runtime_controller.py -q
pytest /Users/fantasylee/类脑架构/tests/unit/test_longrun_acceptance.py -q
pytest /Users/fantasylee/类脑架构/tests/longrun/test_authenticity_acceptance.py -q
PYTHONPATH='/Users/fantasylee/类脑架构' pytest /Users/fantasylee/类脑架构/tests/unit/test_web_sessions.py -q
PYTHONPATH='/Users/fantasylee/类脑架构' pytest /Users/fantasylee/类脑架构/tests/integration/test_observer_api.py /Users/fantasylee/类脑架构/tests/integration/test_observer_diagnostics.py -q
pytest /Users/fantasylee/类脑架构/tests/integration/test_cli.py -q
PYTHONPATH='/Users/fantasylee/类脑架构' pytest /Users/fantasylee/类脑架构/tests/integration/test_observer_launcher.py -q
PYTHONPATH='/Users/fantasylee/类脑架构' pytest /Users/fantasylee/类脑架构/tests/browser/test_observer_dashboard_smoke.py -q
zsh /Users/fantasylee/类脑架构/Open_NALR_Workbench.command --help
```

当前结果：

- `apps/terminal npm test`: `23` files / `85` tests passed
- `apps/terminal npm run build`: `exit 0`
- `tests/unit/test_tlh_runtime.py`: `11 passed`
- `tests/unit/test_output_expression_layer.py`: `7 passed`
- `tests/unit/test_probability_field.py`: `32 passed`
- `tests/unit/test_runtime_controller.py`: `88 passed`
- `tests/unit/test_longrun_acceptance.py`: `7 passed`
- `tests/longrun/test_authenticity_acceptance.py`: `1 passed`
- `tests/unit/test_web_sessions.py`: `1 passed`
- `tests/integration/test_observer_api.py + tests/integration/test_observer_diagnostics.py`: `18 passed`
- `tests/integration/test_cli.py`: `21 passed`
- `tests/integration/test_observer_launcher.py`: `1 passed`
- `tests/browser/test_observer_dashboard_smoke.py`: `1 passed`
- `Open_NALR_Workbench.command --help`: `exit 0`，并显示 `--no-browser` / `--port`

### TLH v1.2 Vector Upgrade

本轮已执行：

```bash
PYTHONPATH='/Users/fantasylee/类脑架构' pytest \
  /Users/fantasylee/类脑架构/tests/unit/test_probability_field.py \
  /Users/fantasylee/类脑架构/tests/unit/test_tlh_runtime.py \
  /Users/fantasylee/类脑架构/tests/unit/test_longrun_acceptance.py \
  /Users/fantasylee/类脑架构/tests/unit/test_runtime_controller.py \
  /Users/fantasylee/类脑架构/tests/unit/test_conflict_controller.py \
  /Users/fantasylee/类脑架构/tests/integration/test_cli.py \
  /Users/fantasylee/类脑架构/tests/integration/test_cil_cli.py \
  /Users/fantasylee/类脑架构/tests/integration/test_observer_api.py \
  /Users/fantasylee/类脑架构/tests/integration/test_observer_diagnostics.py -q
```

当前结果：

- 串行总回归：`233 passed in 134.32s`
- 其中覆盖 TLH v1.2 的向量主体坍缩、人格 EMA、conflict circuit、observer reset 兼容、observer diagnostics 与 CLI 版本同步

### Inherited Field-First Baseline

以下是 TLH 继承的最近一轮 field-first 基线证据，继续作为 substrate 证明保留：

- `tests/unit/test_conflict_controller.py`: `18 passed`
- `tests/unit/test_probability_field.py`: `32 passed`
- `tests/unit/test_runtime_controller.py`: `88 passed`
- `tests/unit/test_longrun_acceptance.py`: `7 passed`
- `tests/longrun/test_authenticity_acceptance.py`: `1 passed`
- `tests/integration/test_cli.py`: `21 passed`
- `tests/integration/test_observer_api.py + tests/integration/test_observer_diagnostics.py`: `18 passed`

## Archive

- 历史文档与旧版本资料见 [`archive/backup-docs/INDEX.md`](/Users/fantasylee/类脑架构/archive/backup-docs/INDEX.md)
