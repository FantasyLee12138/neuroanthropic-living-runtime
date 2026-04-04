# NeuroAnthropic Living Runtime（NALR）

NeuroAnthropic Living Runtime（NALR）是一个对齐[开发文档 v0.56](./开发文档v0.56.md)的类脑活人运行时原型。这个仓库当前追求的不是一次性“补齐所有认知理论”，而是把多 Agent、概率驱动、冲突闭环、慢变量、命令控制层、终端交互层和可观测证据链先稳定串成一条能跑、能看、能调、能降级的工程主链。

当前入口分为两层：

- `./NALR`：普通用户入口，默认启动认知控制台式终端
- `./alive`：开发、运维、观测、回放、维护、干预入口

## 项目定位

这个仓库遵循开发文档 v0.56 的核心思路：

- 先把心理机制离散成可执行的模块、参数、阈值和边界
- 先保证系统可运行、可回放、可解释，再继续追求“活人感”
- 先限制权限和影响范围，再逐步开放更高成本、更强能力的模块
- 先按对话、陪伴、任务等场景区分运行时，再考虑进一步泛化

因此，README 不再按“零散更新日志 + 命令堆叠”来写，而是改成和开发文档一致的阅读顺序：版本定位、模块主线、控制链路、观测能力、使用入口、验证边界。

## 当前实现主线

| 开发文档主线 | 当前仓库中的落地情况 |
|---|---|
| Agent 注册表 | `RuntimeController` 负责编排显性 Agent、隐性模块、守卫层和控制面；`PFCAgent` 负责候选动作，`ConflictMonitorAgent` 负责冲突闭环，`ThalamusAttentionAgent` 负责聚合与采样。 |
| Skill 注册表 | `SkillExecutor` 已统一承接类型契约校验、`timeout_ms` / `cost_class` / `failure_policy` 约束、权限边界、`policy_check`、降级路由和熔断器。 |
| CLI / CIL / 命令控制层 | `alive` 已切到类型化命令封套；命令执行会留下 command trace、规范化命令形态、解析参数、快照和回滚契约。`NALR` 则作为认知控制台消费终端桥接事件；`CommandInterfaceLayer`、`runtime/adapters.py` 与 `runtime/model_gateway.py` 共同承担兼容接口层。 |
| 概率行为与冲突闭环 | runtime 会记录 `u_base -> u_shifted -> p_base_stochastic -> q_noise -> p_mix -> p_final`；冲突层支持 5 个冲突分项、优先级裁决链、重采样、妥协模板、修复状态机和事后纠偏。 |
| 慢变量与长期运行 | memory / habit / relation / resource / temperament 已接入主链；真实性、身份演化、生命性塑形、长跑投影拆分为 `IdentityRuntime`、`AuthenticityPolicy`、`VitalityEngine`、`LongRunAnalyzer`；`idle/sleep` 还能触发 dream 侧车塑形。 |
| trace / observer / 可观测性 | round / skill / command / repair trace 已形成规范化存储；observer 默认读取 Parquet 实时读取模型，并暴露 `why`、贡献拆解、冲突指标、entropy 健康、counterfactual replay、真实性、生命性和 dream 指标。 |

## 当前运行时逻辑

交互轮主链：

```text
state update
-> Salience / Body / Emotion / Relationship / Resource
-> PFC candidate generation
-> Habit / Desire / DMN / Hippocampus / Perspective / Value
-> ConflictMonitor
-> Thalamus sampling
-> BehaviorPlausibilityGuard
-> OutputGate / Renderer
-> trace / writeback / health check
```

非交互塑形链：

```text
idle / sleep
-> VitalityEngine
-> DreamOrchestrator
-> memory / habit / relation / identity shaping
-> authenticity / vitality / long-run evidence
-> why / observer / dream trace
```

这个结构和开发文档中的“显性提案者 -> 冲突闭环 -> 守卫层 -> 输出层 -> 观测层”保持一致；新增的 dream、真实性、生命性链路也被放在慢变量和长期运行逻辑中，而不是散落在终端功能之后。

## 近期已落地改动

### 2026-04-04

- 身份与披露链路从静态 `query_kind / disclosure_detail` 升级为概率化意图层：runtime 会先形成 `query_intent posterior`，再条件化生成 `disclosure_intent posterior`，并把两者写入分布、`render_plan`、`why` 和 observer 指标。
- 情境随机层开始统一消费 `QuantumEntropyPool`；随机扰动和动作采样会记录 `entropy_ref`，外部 QRNG 失败时显式降级为确定性兜底，而不是静默退回伪随机。
- 真实性 / 生命性主链从 `RuntimeController` 内部逻辑拆分为 `IdentityRuntime`、`AuthenticityPolicy`、`VitalityEngine`、`LongRunAnalyzer` 四个运行时边界；controller 继续保留外部 API，但内部职责收敛为编排层。
- `RoundTrace`、`why` 和 observer 时间线现在可以直接暴露 `authenticity`、`identity_evolution`、`vitality_snapshot`、`long_run_projection` 等证据面；`AuthenticityRecord` 也新增 `candidate_penalties` 与 `sampling_penalty_applied`。
- 新增 dream 侧车链路：`idle` 与 `sleep` 模式可触发 `DreamOrchestrator`，生成 `dream_run_id`、`dream_trace_ref`、`dream_guard_summary`、`dream_effect_summary`；`alive dream ...` 与 observer `/dream/*` 已可直接查看。
- `NALR` 已升级为 v2 认知控制台：主区显示 transcript、工具时间线与审批提示，侧栏显示 `core_goal`、`current_intent`、`vital_signs`、`identity`、`authenticity`，底部状态线显示权限、工作区与审批计数。
- 终端桥接事件面扩展为 `run_status`、`step_update`、`tool_call`、`tool_result`、`approval_request`、`sidebar_snapshot`、`assistant_final`，终端与 observer 现在可以共享同一条任务证据链。
- 终端 session 现在会显式持久化 `transcript_lines`、`tool_timeline`、`transcript_mode`、`approvals_pending`，`detach` / 恢复同一 session 时不会丢失上下文。
- `NALR` slash 面补齐为 `/help /status /why /steps /tools /state /dream [cue] /pause /resume /abort /clear /compact /mode [value] /permissions [value] /model /exit`，并支持按 cwd 维度记录输入历史、`Ctrl+C` 中止、`Ctrl+D` 退出、`Ctrl+L` 清屏、`\ + Enter` 多行草稿。
- `alive` / observer 补齐了 counterfactual 与维护接口：`trace compact`、`why this`、`why not`、`what changed`、`eval longrun`、`/metrics/entropy`、`/metrics/timeline`、`/metrics/heatmap`、`/replay/{round_id}`、`/why-not/{round_id}/{action}`。
- 本轮补验已重新跑通：`tests/unit/test_dream_runtime.py`、`tests/integration/test_dream_bridge_stdio.py`、`tests/longrun/test_authenticity_acceptance.py`、`tests/unit/test_terminal_bridge.py`、`tests/integration/test_nalr_terminal.py`，共 `24 passed`；同时 `npm --prefix apps/terminal test` 与 `npm --prefix apps/terminal run build` 是当前终端侧的标准验证集。

### 2026-04-03

- 新增 `./alive` 启动脚本，支持自动加载仓库根目录 `.env.local` / `.env`，`config/models.yaml` 成为 PFC、Perspective、renderer 的模型路由默认入口。
- `config/models.yaml` 现在同时保留 `model_routes` 与 `models` 兼容块：前者供 `ModelRouter` 消费，后者供 `ModelGateway` / 兼容适配层消费。
- skill runtime 已收口为类型化校验器：补齐 contract 解析/序列化/校验、运行时元数据约束、权限边界、熔断策略、降级路由与 provider 路由。
- trace / memory 维护路径扩展为显式命令：`alive trace export parquet`、`alive memory compact`、`alive memory sample`，并让 `checkpoint create` 也写入 command trace。
- CLI、observer 和表达层同步增强：新增 `trace agents|skills|gates|compact` 视图、`memory recall` / `habit reset` / `nudge relation` / `budget set` / `why this|not` / `what changed` / `eval longrun` 命令、`GET /memory/recall/{cue}` / `GET /replay/{round_id}` / `GET /why-not/{round_id}/{action}` 观测接口，以及更贴近表达参数的兜底渲染器。

## 当前可直接观测的能力

- `alive trace round 1`、`alive trace why 1` 可以直接看到 `u_base`、`u_shifted`、`p_base_stochastic`、`q_noise`、`p_mix`、`p_final`、`risk_suppressor`、`conflict_mode`。
- `render_plan.identity_context` 会暴露 `query_intent`、`query_intent_posterior`、`disclosure_intent`、`disclosure_intent_posterior`、`disclosure_clipped`。
- `stochastic_state` 会写出 `entropy_ref`，用于标识本轮使用的是哪一段量子熵或哪次降级。
- `alive trace why <round>`、observer `/why/{round}` 和 `/metrics/conflicts` 会显式给出 `repair_mode`、`post_error_adjustment`、`repair_state_snapshot`、`repair_ledger_tail`、`conflict_safe_mode_owned`。
- `alive memory recall <cue>` 会返回记忆层级、强度、gist/detail 模式、干扰信息和证据片段。
- `alive habit reset <pattern>` 可以把习惯重置到 `strength = 0`，同时保留可恢复语义。
- `alive budget set --cap 50000` 会通过 CIL 更新预算，并保持 runtime 内部状态归一到 `0..1`。
- `alive trace compact` 会把 round / skill / command / repair trace 重新导出成 Parquet，并返回生成路径。
- `alive why this <round>`、`alive why not <action> <round>`、`alive what changed <window>`、`alive eval longrun <rounds>` 可以直接做解释、反事实和轻量长跑评估。
- `alive dream status`、`alive dream trace last`、`alive dream proposals last`、`alive dream metrics` 可以查看非交互塑形与 dream proposal 证据。
- `alive trace round|why|contribution` 与 observer `/trace|/why|/contributions` 默认优先从 Parquet 规范化实时读模型读取，并显式返回 `storage.read_source` / `trace_sync_state`。
- observer 已支持 `/metrics/authenticity`、`/metrics/vitality`、`/metrics/entropy`、`/metrics/timeline`、`/metrics/heatmap`、`/dream/status`、`/dream/runs`、`/dream/metrics`、`/skills/stats`、`/metrics/conflicts`、`/metrics/mode-switches`、`/analysis/ablation`、`/replay/{round_id}`、`/why-not/{round_id}/{action}`。

## v0.56 对账与归档来源

这次整理工作树时，README 不再只记录“改了什么”，而是把可验证能力按开发文档主线重新对账，并把改动来源固定到可回溯分支。

| 对账项 | 当前仓库中的结果 | 归档来源 |
|---|---|---|
| Agent / Skill / CLI-CIL 主线 | `RuntimeController`、`SkillExecutor`、`CommandInterfaceLayer`、`runtime/adapters.py`、`runtime/model_gateway.py` 都已在主链中可见；`alive` 命令与 observer 接口对齐到同一批运行时能力。 | `codex/nalr-integration@ae81a40` + `codex/archive-main-wip-2026-04-04@b2bab0e` |
| entropy 配置与降级 | `config/entropy.yaml` 定义 `endpoint`、`timeout_s`、`prefetch_bytes`、`min_batch_bytes`、`max_batch_bytes`、`hard_block_on_unavailable`；`QuantumEntropyPool` 会记录 `entropy_ref`、健康状态与降级原因；observer 暴露 `/metrics/entropy`。 | 主要来自 `codex/archive-main-wip-2026-04-04@b2bab0e` |
| terminal route 规划/执行分离 | `TurnPlan` / `TurnExecution` 已进入 runtime schema；terminal bridge 通过 `plan_turn()` + `execute_turn()` 统一 direct chat、fast chat 与只读 task run，并持久化 transcript 与 tool timeline。 | 主要来自 `codex/archive-main-wip-2026-04-04@b2bab0e` |
| trace / repair / observer 增强 | repair ledger 已带 `conflict_score`、learning signal 与 repair tail；observer 补齐 `/metrics/timeline`、`/metrics/heatmap`、`/replay/{round_id}`、`/why-not/{round_id}/{action}`。 | `codex/nalr-integration@ae81a40` + `codex/archive-main-wip-2026-04-04@b2bab0e` |
| conflict-controller deadlock fuse 对标 | deadlock fuse / circuit breaker 的 v0.56 断言被重新对齐到 `tests/unit/test_conflict_controller.py`，并保留 repair FSM 语义。 | `codex/archive-elegant-jemison-wip-2026-04-04@b6476fe` |

归档说明：

- `codex/nalr-integration` 已经吸收 `codex/runtime-parity`，因此本次只保留前者作为 codex 集成来源，不再重复整理 `codex/runtime-parity`。
- 主工作树的未提交内容先归档到 `codex/archive-main-wip-2026-04-04`，再并入 `codex/archive-cleanup`，避免在清理 worktree 时丢失未提交状态。
- `elegant-jemison` 的快照分支只吸收与 `docs/specs/conflict-controller-v056.md` 一致的 deadlock fuse / repair 断言，不盲目回滚到旧 runtime 实现。

## 目录说明

```text
config/                  参数、阈值、场景预设、dream 配置
docs/                    架构说明、规格文档、设计决策
apps/terminal/           NALR 终端控制台前端
services/observer/       observer 只读侧车
src/nalr/                Python 运行时、CLI、trace、dream、终端桥接
tests/                   单测、集成、仿真、长跑验收
.alive/                  默认本地运行态目录
开发文档v0.56.md          当前对齐的上游规格文档
```

运行态产物的关键路径：

```text
.alive/
  traces/
    rounds/
    skills/
    parquet/
      round_trace.parquet
      skill_trace.parquet
      command_trace.parquet
      repair_trace.parquet
  memory/
    raw/
    episodic_hot/
    episodic_warm/
    episodic_archive/
  dream/
    dream_runs.jsonl
    runs/
```

## 快速开始

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
npm --prefix apps/terminal install
.venv/bin/python -m pytest tests -q
```

如果你使用 live Doubao / Ark 路由，还需要：

```bash
export ARK_API_KEY=your-ark-key
```

`./alive` 会自动加载仓库根目录下的 `.env.local` 或 `.env`；模型路由默认配置在 `config/models.yaml`。

## 使用入口

### 普通用户入口：`NALR`

启动交互终端：

```bash
./NALR
./NALR "总结这个仓库结构"
./NALR Dream tea
```

当前可用 slash 命令：

```text
/help
/status
/why
/steps
/tools
/state
/dream [cue]
/pause
/resume
/abort
/clear
/compact
/mode [value]
/permissions [value]
/model
/exit
```

交互补充：

```text
- 宽终端默认双栏，窄终端自动退化为上下布局
- Ctrl+C：中止当前任务
- Ctrl+D：退出终端
- Ctrl+L：清空当前可见 transcript
- Up / Down：按当前 cwd 回溯输入历史
- \ + Enter：继续输入多行草稿
- y / n：审批当前待确认动作
```

### 开发 / 运维入口：`alive`

常用命令示例：

```bash
./alive chat "我今天有点乱，帮我理一下待办"
./alive chat "记住我晚饭想吃面" --cue 面 --show agents
./alive state show
./alive identity show
./alive focus show
./alive trace why last
./alive trace agents last
./alive trace skills last
./alive trace gates last
./alive memory recall 面
./alive memory sample --tier hot --limit 3
./alive habit reset coffee
./alive nudge relation user trust +0.05
./alive budget set --cap 50000
./alive checkpoint create
./alive trace export parquet --overwrite
./alive dream status
./alive dream run --mode sleep --cue tea
./alive dream trace last
./alive dream proposals last
./alive dream metrics
./alive safe on
```

如果你更想直接敲 `alive ...`，先激活虚拟环境即可：

```bash
source .venv/bin/activate
alive chat "帮我规划今晚"
alive repl
```

### observer 入口

启动方式：

```bash
.venv/bin/python -m uvicorn services.observer.api.app:app --reload
```

常用接口：

```text
GET /state
GET /identity
GET /runs/current
GET /runs/{run_id}/steps
GET /runs/{run_id}/tools
GET /trace/{round_id}
GET /why/{round_id}
GET /contributions/{round_id}
GET /memory/recall/{cue}
GET /metrics/summary
GET /metrics/authenticity
GET /metrics/vitality
GET /metrics/entropy
GET /metrics/timeline
GET /metrics/heatmap
GET /dream/status
GET /dream/runs
GET /dream/runs/{round_ref}
GET /dream/metrics
GET /skills/stats
GET /metrics/conflicts
GET /metrics/mode-switches
GET /analysis/ablation
GET /replay/{round_id}
GET /why-not/{round_id}/{action}
GET /dashboard
```

## 验证口径

当前 README 采用“聚焦验证 + 有界长跑验收”的表述，不把重型 soak 冒充成已重新验完。

推荐验证集：

```bash
.venv/bin/python -m pytest -q tests/unit/test_runtime_controller.py tests/unit/test_terminal_bridge.py tests/unit/test_dream_runtime.py tests/integration/test_observer_api.py tests/integration/test_terminal_bridge_stdio.py tests/integration/test_nalr_terminal.py tests/integration/test_dream_bridge_stdio.py tests/longrun/test_authenticity_acceptance.py
npm --prefix apps/terminal test
npm --prefix apps/terminal run build
```

类型化运行时 / trace / CIL 的最小验证集：

```bash
.venv/bin/python -m pytest -q tests/unit/test_registry_contracts.py tests/unit/test_skill_runtime.py tests/unit/test_runtime_controller.py tests/integration/test_cil_cli.py tests/integration/test_cli.py tests/integration/test_trace_maintenance_cli.py
```

重型 soak 仍单列为后续验收项：

```bash
.venv/bin/python -m pytest -q tests/longrun/test_longrun_smoke.py
```

## 当前边界

- 类型化 CIL 当前重点覆盖已经接入的可变操作类命令，而不是开发文档中全部潜在命令面。
- Parquet 实时读取当前主覆盖 round / skill / command / repair；run / step / tool 级 trace 仍保留现有实现。
- `NALR` 的终端壳、审批状态和 bridge 协议已经到位，但真正的可变更 coding tools 还没有全部接入。
- 多行输入当前稳定支持 `\ + Enter`；`Shift+Enter` / `Option+Enter` 仍受当前 Ink 输入栈限制。
- 模型增强路径仍属于可选能力；缺失 `ARK_API_KEY` 时，当前验证口径默认以无模型路径为主。
- dream 侧车已经接入 `idle/sleep` 塑形、trace 和 observer，但仍属于慢变量与长期运行链路的一部分，不是独立的对话主调度器。
- `tests/longrun/test_longrun_smoke.py` 仍被视为单独 soak gate；当前分支的声明是“聚焦验证 + 有界长跑验收已收口”，不是“全仓长跑已重新跑完”。

## 关键文档

- [开发文档 v0.56](./开发文档v0.56.md)：当前对齐的总规格文档。
- [架构总览](./docs/architecture/overview.md)：运行时边界、skill 调用链和 observer 读取模型。
- [v0.56 验收矩阵](./v056_acceptance_matrix.md)：按规格项对照代码、测试和 trace 证据的保守验收表。
- [冲突控制器规格](./docs/specs/conflict-controller-v056.md)：冲突分项、优先级链、重采样、妥协模板和 repair FSM。
- [Parquet / 记忆压缩说明](./docs/specs/parquet-trace-memory-compaction.md)：trace 导出和记忆压缩维护路径。
- [typed skill runtime 设计决策](./docs/decisions/0002-typed-skill-runtime-validator.md)：类型化校验器、权限边界和熔断器设计。
- [梦境侧车集成方案](./oneiroi_agent_codex_plan.md)：dream 侧车的目标、接口、预算和 proposal 设计。
