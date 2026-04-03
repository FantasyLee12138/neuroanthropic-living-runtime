# NeuroAnthropic Living Runtime (NALR)

NeuroAnthropic Living Runtime（NALR）是一个基于 [开发文档v0.56](./开发文档v0.56.md) 落地的类脑活人运行时原型。首版目标不是一次性还原全部认知复杂性，而是先把多 Agent、可控预算、可追踪 proposal、记忆/习惯层、输出表达层和观测侧车稳定串成一条可执行链路。

## 本次更新记录（2026-04-03）

- 新增 `./alive` 启动脚本，支持自动加载仓库根目录 `.env.local` / `.env`，并补充 `config/models.yaml` 作为 PFC、Perspective、renderer 的模型路由配置入口。
- 将 skill runtime 收口为 typed validator：补上 contract 解析/序列化/校验、运行时元数据约束、权限边界、熔断策略与 fallback route，并新增 provider router、Doubao backend 与 fake backend。
- trace / memory 维护面扩展为显式路径：增加 `session_id` / `recorded_at` / `recorded_date` 元数据、`alive trace export parquet`、`alive memory compact`、`alive memory sample`，同时让 `checkpoint create` 也写 command trace。
- CLI、observer 与表达层同步增强：新增 `trace agents|skills|gates` 视图、`memory recall` / `habit reset` / `nudge relation` / `budget set` 命令、`GET /memory/recall/{cue}` 观测接口，以及更贴近表达参数的 fallback renderer。
- 文档与验收材料补齐：新增 `v056_acceptance_matrix.md`、ADR `0002-typed-skill-runtime-validator`、冲突控制器规格与 Parquet/记忆压缩说明；当前整仓验证快照为 `59 passed, 2 skipped`。

## Implementation Status

| Phase | Status | Observable Result | Verification |
|---|---|---|---|
| P0. truth-first baseline | done | README, CLI surface, and tests now reflect verified behavior instead of future intent; `checkpoint create` also leaves command trace. | `.venv/bin/python -m pytest tests/unit/test_runtime_controller.py tests/integration/test_cil_cli.py -q` |
| P1. formula kernel + guard closure | in progress | `ActionDistributionState` now records `u_base`, `u_shifted`, `p_base_stochastic`, `q_noise`, `p_mix`, `p_final`, `risk_suppressor`, and `conflict_mode`; `sigma_scale` and `utility_shift` now change the runtime distribution; `risk_suppressor` now changes `p_final`; task-time high-probability `wander` is explicitly blocked by `BehaviorPlausibilityGuard` with trace evidence. | `.venv/bin/python -m pytest tests/unit/test_formula_runtime_alignment.py tests/unit/test_agent_expansion.py tests/unit/test_conflict_controller.py -q` |
| P2. slow state: memory / habit / resource / temperament | in progress | memory now exposes structured `recall`, uses context-slot + cue-similarity interference, and habit strength is updated through the runtime path while still remaining bounded and resettable; runtime state now has stable `temperament_state` and `resource_state` slots. | `.venv/bin/python -m pytest tests/simulation/test_memory_tiers.py tests/simulation/test_memory_and_habit.py tests/unit/test_runtime_controller.py -q` |
| P3. stochastic + output expression | in progress | stochastic mixing now starts from `Softmax(u_shifted)` and lowers `lambda_noise` under stronger task control; fallback renderer now consumes `ExpressionProfile`, relation risk, and safety constraints instead of only action templates. | `.venv/bin/python -m pytest tests/unit/test_formula_runtime_alignment.py tests/unit/test_output_expression_layer.py tests/unit/test_runtime_controller.py -q` |
| P4. skill runtime + trace + Parquet | in progress | typed skill runtime, JSON/JSONL trace, Parquet export, and approximate ablation are usable; round trace now preserves richer formula-state fields and command trace covers control-plane mutations including checkpoint creation. | `.venv/bin/python -m pytest tests/unit/test_skill_runtime.py tests/integration/test_trace_maintenance_cli.py tests/integration/test_observer_diagnostics.py -q` |
| P5. CIL / CLI / observer core plane | in progress | CIL-backed commands now include `memory recall`, `habit reset`, `nudge relation <target> trust <delta>`, `budget set --cap <int>`, and observer adds `/memory/recall/{cue}`. | `.venv/bin/python -m pytest tests/integration/test_cil_cli.py tests/integration/test_cli.py tests/integration/test_observer_api.py -q` |
| Regression baseline | done | Current branch passes the repo test suite after the above changes. | `.venv/bin/python -m pytest tests -q` |

## Current Observable Capabilities

- `./alive trace round 1` and `./alive trace why 1` now show the richer distribution state: `u_base`, `u_shifted`, `p_base_stochastic`, `q_noise`, `p_mix`, `p_final`, `risk_suppressor`, and `conflict_mode`.
- `./alive memory recall <cue>` returns tier, strength, detail/gist mode, interference, and evidence snippets.
- `./alive habit reset <pattern>` resets a habit to `strength = 0` while keeping it recoverable.
- `./alive nudge relation <target> trust <delta>` writes a relation nudge through CIL and leaves command trace.
- `./alive budget set --cap 50000` updates runtime budget in operator space while keeping internal runtime state normalized to `0..1`.
- `./alive checkpoint create` now writes command trace in addition to checkpoint state.
- `./alive` auto-loads `.env.local` or `.env`, so local model routing can be switched without exporting every variable by hand.
- Fallback renderer now changes wording based on `directness_level`, `hedging_level`, `warmth_level`, `repair_tendency`, relation risk, and `conflict_hot`.
- Observer now supports `GET /memory/recall/{cue}`, `GET /skills/stats`, `GET /metrics/conflicts`, `GET /metrics/mode-switches`, and `GET /analysis/ablation`.
- Offline analytics export remains available through `alive trace export parquet`, producing round / skill / command Parquet tables under `.alive/traces/parquet/`.
- Current repo verification result: `59 passed, 2 skipped`.

## Next Milestone

- Finish the remaining formula-accuracy gaps: make `ConflictMonitorAgent` carry explicit post-error adjustment / repair markers, deepen slow-state formulas beyond the current bounded implementation, and replace string-split CIL routing with stricter typed command parsing.

## Known Gaps Against v0.56

- Conflict control is stronger than before, but it still does not expose a first-class `mark_post_error_adjustment` state or a richer multi-round repair ledger from the doc.
- Slow-state formulas are now closer to the doc, but memory decay/interference, habit recovery, resource scarcity, and temperament drift are still bounded approximations rather than a full parameter-complete v0.56 implementation.
- CIL is still string-routed. It now covers the documented core commands, but it does not yet use a stricter typed command schema with execution snapshots and rollback objects for every mutable operator action.
- Observer still reads canonical JSON trace/state rather than using Parquet as the primary live read path.
- Model-backed routes remain optional enhancements; the no-model path is the primary validated runtime path in this branch.

## v0.56 公式层实现

- 决策核：各 agent 输出 `ProposalBundle(delta_p, confidence, sigma_scale, utility_shift, veto)`，controller 记录 `u_base -> u_shifted -> p_base_stochastic -> q_noise -> p_mix -> p_final`，并把 CI 写回状态。
- 冲突与采样层：`ConflictMonitorAgent` 负责 5 个分项冲突分数、固定优先级裁决链、最多 3 个 pass 的重采样/妥协控制，以及 `critical_conflict` 熔断与恢复；`ThalamusAttentionAgent` 负责聚合、归一化、采样。
- 分布级守卫层：`BehaviorPlausibilityGuard` 会扫描最终分布中的高概率违规动作，先把它们打到 `gate=0` 再重采样；`ForcedModeSwitch` 处理 mode/focus lock；`OutputGate` 做最终约束下压。
- 慢变量层：memory 使用 gist/detail + cue-weighted decay + interference，habit 使用 bounded strengthening，relation trace 和 stable priors 都会落盘。
- 情境随机层：在 deterministic distribution 上叠加 `xi_emo`、`xi_mood`、`lambda_noise`、`r_intensity`，并记录 `KL` guard 结果。
- 输出表达层：`build_expression_profile()` 按 `reply_delay`、`self_disclosure`、`tone_sharpness`、`repair_tendency` 公式和 jitter 范围生成表达参数。
- model-backed 规划层：`PFCAgent.generate_candidates` 通过 provider route 生成结构化候选动作；失败或无 `ARK_API_KEY` 时退回显式启发式 fallback planner。
- late Perspective 层：`infer_other_state` / `simulate_other_reaction` 在 `output_gate` 之后按风险门控触发，消费最终 gated action。
- renderer 接口层：`build_render_plan()` 现在生成自包含计划，随后由 renderer/provider 消费并产出 `RenderedExpression`。
- typed skill runtime 层：`SkillExecutor` 统一做 typed input/output contract 校验、运行时元数据约束、权限边界检查、`policy_check`、熔断持久化和低成本 fallback 接管。
- 控制与观测层：CIL 命令统一写 command trace；observer 与 trace/why payload 现在都能读到 `rendered_expression`、late Perspective 结果、renderer stage 证据，以及 conflict components / compromise template / `conflict_hot`。

## v0.56 对齐范围

- Phase 1：运行时骨架、CLI MVP、trace、safe mode
- Phase 2：可控记忆层与 gist/detail 召回
- Phase 3：习惯强度与 hot cache 思路
- Phase 4：关系层、Perspective、输出风格层
- Phase 5：DMN、长跑 smoke、observer 可视化入口

## 架构概览

- `src/nalr/runtime`：单轮调度、模式切换、checkpoint、safe mode
- `docs/specs/conflict-controller-v056.md`：完整冲突控制器的分项分数、优先级链、妥协模板与熔断恢复规范
- `src/nalr/agents`：Body、Resource、PFC、Hippocampus、Habit、Relationship、DMN、Perspective proposals
- `src/nalr/providers`：模型 route、Doubao/Ark backend、测试 fake backend
- `src/nalr/memory`：事件写入、记忆强度、习惯强度、关系状态
- `src/nalr/trace`：canonical trace 存储与 Parquet 导出
- `src/nalr/output`：表达参数计算、`RenderPlan` 组装、renderer fallback
- `src/nalr/cli`：`alive` CLI
- `services/observer`：只读 observer API 与 dashboard 占位

## 关键文档

- `v056_acceptance_matrix.md`：按 spec item 对照代码、测试和 trace 证据的保守验收矩阵。
- `docs/decisions/0002-typed-skill-runtime-validator.md`：typed skill runtime validator 的设计决策、约束与后果。
- `docs/specs/conflict-controller-v056.md`：冲突控制器的完整规格对照文档。
- `docs/specs/parquet-trace-memory-compaction.md`：Parquet 导出与记忆压缩维护路径说明。
- `config/models.yaml`：PFC / Perspective / renderer 的模型路由默认配置。

## 目录说明

```text
config/                  规格参数与 scenario preset
docs/                    架构说明与设计决策
src/nalr/                Python 运行时代码
services/observer/       只读 sidecar
tests/                   unit/integration/simulation/longrun
.alive/                  默认本地运行态目录
开发文档v0.56.md          上游规格文档
```

运行态 trace / memory 产物补充：

```text
.alive/
  traces/
    rounds/
    skills/
    round_traces.jsonl
    command_traces.json
    command_traces.jsonl
    parquet/
      round_trace.parquet
      skill_trace.parquet
      command_trace.parquet
  memory/
    raw/
      episodic_events.jsonl
    episodic_hot/
    episodic_warm/
    episodic_archive/
```

## Quickstart

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m pytest tests -q
```

针对 typed runtime validator 的最小验证集：

```bash
.venv/bin/python -m pytest tests/unit/test_registry_contracts.py tests/unit/test_skill_runtime.py tests/unit/test_runtime_controller.py tests/unit/test_runtime_pipeline.py tests/integration/test_cil_cli.py tests/simulation/test_memory_tiers.py -q
```

启用 live Doubao / Ark 路由时额外设置：

```bash
export ARK_API_KEY=your-ark-key
```

模型路由默认写在 `config/models.yaml`；如果使用仓库根目录的 `.env.local` 或 `.env`，`./alive` 会在启动时自动加载。

运行 CLI：

```bash
./alive state show
./alive chat "帮我规划今晚，并记住我晚饭想吃面"
./alive repl
./alive focus show
./alive trace round 1
./alive trace agents last
./alive trace skills last
./alive trace gates last
./alive trace export parquet --overwrite
./alive memory compact
./alive memory recall 面
./alive memory sample --tier hot --limit 3
./alive habit reset coffee
./alive nudge relation user trust +0.05
./alive budget set --cap 50000
./alive safe on
```

如果你更想直接敲 `alive ...`，先激活虚拟环境即可：

```bash
source .venv/bin/activate
alive chat "帮我规划今晚"
alive repl
```

运行 observer：

```bash
.venv/bin/python -m uvicorn services.observer.api.app:app --reload
```

如果使用自定义路径：

```bash
export NALR_HOME=.alive
export NALR_CONFIG_DIR=config
export NALR_SCENARIO=chat
export NALR_MODE=interactive
export ARK_API_KEY=your-ark-key
```

## CLI 示例

终端对话：

```bash
alive chat "我今天有点乱，帮我理一下待办"
alive chat "记住我晚饭想吃面" --cue 面 --show agents
alive chat "刚才你记住了什么？" --json
alive memory recall 面
alive habit reset coffee
alive nudge relation user trust +0.05
alive budget set --cap 50000
alive repl
```

在 `alive repl` 中可用：

```text
/help
/why
/agents
/skills
/gates
/state
/mode task
/safe on
/budget
/exit
```

思考过程查看：

```bash
alive trace round 1
alive trace why last
alive trace contribution last
alive trace agents last
alive trace skills last
alive trace gates last
```

说明：这里的“思考过程”指 runtime 已落盘的结构化 trace 证据，包括 `top_drivers`、`proposal_summaries`、`gate_decisions`、`skill_traces`，不是自由文本 chain-of-thought。

```bash
alive state show
alive body show
alive mood show
alive focus show
alive memory top
alive habit top
alive mode set task
alive trace round 1
alive trace why 1
alive trace contribution 1
alive trace export parquet --overwrite
alive agent list
alive agent disable DMNAgent
alive checkpoint create
alive checkpoint rewind ckpt-0000
alive memory compact
alive memory sample --tier warm --limit 3
alive safe on
alive budget show
```

observer 诊断入口：

```text
GET /state
GET /trace/{round_id}
GET /why/{round_id}
GET /contributions/{round_id}
GET /metrics/summary
GET /dashboard
```

## 阶段路线图

1. 把当前基于规则的 runtime 扩展成更细的 proposal/veto 分层。
2. 将记忆热层扩展为 hot/warm/archive 压缩与 replay。
3. 将 observer 从只读 JSON API 扩展到 why-this、贡献度和长跑指标面板。
4. 为现有 Doubao route 补充更强的 typed schema、更多 backend 和更稳的长跑校准。

## 运行截图占位

- CLI screenshot: `docs/architecture/cli-screenshot-placeholder.md`
- Observer screenshot: `docs/architecture/observer-screenshot-placeholder.md`

## 贡献说明

- 先阅读 `开发文档v0.56.md`
- 修改规则或阈值时优先更新 `config/`
- 新增行为前先补测试，再补实现
- 所有状态变更都应保留 trace 证据

## GitHub

- Repository slug: `FantasyLee12138/neuroanthropic-living-runtime`
- Project title: `NeuroAnthropic Living Runtime (NALR)`
