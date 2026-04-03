# NeuroAnthropic Living Runtime (NALR)

NeuroAnthropic Living Runtime（NALR）是一个基于 [开发文档v0.56](./开发文档v0.56.md) 落地的类脑活人运行时原型。首版目标不是一次性还原全部认知复杂性，而是先把多 Agent、可控预算、可追踪 proposal、记忆/习惯层、输出表达层和观测侧车稳定串成一条可执行链路。

## v0.56 对齐范围

- Phase 1：运行时骨架、CLI MVP、trace、safe mode
- Phase 2：可控记忆层与 gist/detail 召回
- Phase 3：习惯强度与 hot cache 思路
- Phase 4：关系层、Perspective、输出风格层
- Phase 5：DMN、长跑 smoke、observer 可视化入口

## 架构概览

- `src/nalr/runtime`：单轮调度、模式切换、checkpoint、safe mode
- `src/nalr/agents`：Body、Resource、PFC、Hippocampus、Habit、Relationship、DMN、Perspective proposals
- `src/nalr/memory`：事件写入、记忆强度、习惯强度、关系状态
- `src/nalr/trace`：round trace 与命令 trace 存储
- `src/nalr/output`：输出风格映射
- `src/nalr/cli`：`alive` CLI
- `services/observer`：只读 observer API 与 dashboard 占位

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
.venv/bin/python -m uvicorn services.observer.api.app:create_app --factory --reload
```

如果使用自定义路径：

```bash
export NALR_HOME=.alive
export NALR_CONFIG_DIR=config
export NALR_SCENARIO=chat
export NALR_MODE=interactive
```

## CLI 示例

```bash
alive state show
alive body show
alive mood show
alive focus show
alive memory top
alive habit top
alive mode set task
alive trace round 1
alive agent list
alive agent disable DMNAgent
alive checkpoint create
alive safe on
alive budget show
```

## 阶段路线图

1. 把当前基于规则的 runtime 扩展成更细的 proposal/veto 分层。
2. 将记忆热层扩展为 hot/warm/archive 压缩与 replay。
3. 将 observer 从只读 JSON API 扩展到 why-this、贡献度和长跑指标面板。
4. 为 PFC/renderer 接入真正的主模型与小模型路由。

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
