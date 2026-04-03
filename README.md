# NeuroAnthropic Living Runtime (NALR)

NeuroAnthropic Living Runtime（NALR）是一个基于 [开发文档v0.56](./开发文档v0.56.md) 落地的类脑活人运行时原型。首版目标不是一次性还原全部认知复杂性，而是先把多 Agent、可控预算、可追踪 proposal、记忆/习惯层、输出表达层和观测侧车稳定串成一条可执行链路。

## 当前开发边界

- `codex/runtime-parity`：负责编排、CLI/CIL、trace/analytics、observer/dashboard、Doubao gateway、longrun/eval。
- `codex/v056-runtime`：负责 agents、skills、layers 与核心公式实现。
- 两条线程会在后续合并。当前分支把上游线程视为接口面，README 只描述已验证能力与已预留接入点，不把未合并实现写成“已完成”。

## 已完成能力

### 编排层已验证

- `trace compact`：输出 JSON round replay + Parquet contribution analytics。
- `replay round`、`why this`、`why not`、`what changed`：可回放、可解释、可给出 counterfactual。
- `eval longrun`：输出 crash/safe mode/conflict/habit/recall/relation/task/top-driver 等汇总指标。
- observer：已提供 `/metrics/timeline`、`/metrics/heatmap`、`/replay/{round_id}`、`/why-not/{round_id}/{action}`、`/skills/stats`、`/skills/profile/{skill_name}`、`/dashboard`。
- Doubao gateway：PFC 与 renderer 通过火山 ARK OpenAI 兼容接口接入，缺少 `ARK_API_KEY` 时自动退回规则路径。

### 上游接口已预留接入点

- `build_agents() -> list[BaseAgent]`
- `build_skill_registry() -> dict[str, SkillSpec]`
- 共享 schema：`Proposal`、`RoundEvent`、`RuntimeState`、`SkillSpec`
- compatibility adapter：对较瘦的上游 `Proposal/SkillSpec` 自动补齐 `provider/model/latency_ms` 等 richer trace 字段

## 当前已知未完成项

- 上游线程的完整 agents/skills/layers 还未合并到当前分支。
- Parquet analytics 目前已落 `contribution` 级字段，但仍未覆盖文档要求的完整分区/index 策略。
- `10k` longrun、关系一致性、安全 policy 和多线程合流后的真实指标仍待验证。

## v0.56 对齐范围

- Phase 1：运行时骨架、CLI MVP、trace、safe mode
- Phase 2：可控记忆层与 gist/detail 召回
- Phase 3：习惯强度与 hot cache 思路
- Phase 4：关系层、Perspective、输出风格层
- Phase 5：DMN、长跑 smoke、observer 可视化入口

## 架构概览

- `src/nalr/runtime`：单轮调度、模式切换、checkpoint、safe mode
- `src/nalr/agents`：上游 agent 接口面与当前分支的兼容占位
- `src/nalr/memory`：事件写入、记忆强度、习惯强度、关系状态
- `src/nalr/trace`：round trace 与命令 trace 存储
- `src/nalr/output`：输出风格映射
- `src/nalr/cil`：命令解析、路由与标准回显
- `src/nalr/cli`：`alive` CLI
- `services/observer`：只读 observer API 与 dashboard 占位
- `config/models.yaml`：Doubao ARK 模型网关配置

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

## Quickstart

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m pytest tests -q
```

运行 CLI：

```bash
.venv/bin/alive state show
.venv/bin/alive focus show
.venv/bin/alive trace round 1
.venv/bin/alive safe on
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
export NALR_MODEL_PROVIDER=doubao_ark
export NALR_PRIMARY_MODEL=doubao-seed-2-0-pro-260215
export NALR_FAST_MODEL=doubao-seed-2-0-pro-260215
export ARK_API_KEY=...
```

## CLI 示例

```bash
alive state show
alive body show
alive body rest
alive mood show
alive mood calm
alive relation show user
alive focus show
alive memory top
alive habit top
alive mode set task
alive trace round 1
alive trace why 1
alive trace contribution 1
alive trace compact
alive agent list
alive agent disable DMNAgent
alive skill stats
alive skill profile generate_candidates
alive checkpoint create
alive checkpoint rewind ckpt-0000
alive safe on
alive budget show
alive debug weight PFCAgent 0.3
alive replay round 1 --seed 7
alive why this 1
alive why not rest 1
alive what changed --window 5
alive eval longrun --rounds 1000
alive rest
alive calm
```

observer 诊断入口：

```text
GET /state
GET /trace/{round_id}
GET /why/{round_id}
GET /contributions/{round_id}
GET /metrics/summary
GET /metrics/timeline
GET /metrics/heatmap
GET /replay/{round_id}
GET /why-not/{round_id}/{action}
GET /skills/stats
GET /skills/profile/{skill_name}
GET /dashboard
```

## 阶段路线图

1. 合并 `v056-runtime` 的 agents/skills/layers/公式实现，替换当前兼容占位。
2. 把 skill runtime 和 command trace 收敛到统一 CIL/registry 观测面。
3. 将 Parquet analytics 补齐到文档要求的索引/分区与离线分析能力。
4. 以合并后的统一 runtime 跑完 `10k` longrun，并回填关系一致性与安全 policy 指标。

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
