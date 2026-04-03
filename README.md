# NeuroAnthropic Living Runtime (NALR)

NeuroAnthropic Living Runtime（NALR）是一个基于 [开发文档v0.56](./开发文档v0.56.md) 落地的类脑活人运行时原型。首版目标不是一次性还原全部认知复杂性，而是先把多 Agent、可控预算、可追踪 proposal、记忆/习惯层、输出表达层和观测侧车稳定串成一条可执行链路。

## Implementation Status

| Phase | Status | Observable Result | Verification |
|---|---|---|---|
| 1. contracts + registry | done | `AgentSpec`, expanded `SkillSpec`, `ProposalBundle`, `ActionDistributionState`, `StochasticState`, `ExpressionProfile`, `RenderPlan`, `GateDecision`, `RoundContext`, `CommandEnvelope` are live and serialized through trace. | `.venv/bin/python -m pytest tests/unit/test_registry_contracts.py tests/unit/test_formula_runtime_alignment.py -q` |
| 2. runtime pipeline | done | `tick()` now runs the v0.56-style stage chain, records `P_base -> P_raw -> P_mix -> P_final`, CI, gates, resample count, stochastic state, and render plan. | `.venv/bin/python -m pytest tests/unit/test_runtime_pipeline.py tests/unit/test_formula_runtime_alignment.py -q` |
| 3. agents / guards | done | `ConflictMonitorAgent`, `ThalamusAttentionAgent`, `BehaviorPlausibilityGuard`, `ForcedModeSwitch`, and `OutputGate` now own real runtime skills; plausibility is distribution-level and can preemptively gate disallowed actions. | `.venv/bin/python -m pytest tests/unit/test_agent_expansion.py -q` |
| 4. skill runtime + trace + memory | done | `SkillExecutor` now does output-schema validation, retry/degrade, circuit-breaking, and shaped fallback payloads; memory uses hot/warm/archive tiers plus cue-weighted decay, interference, and bounded habit strengthening. | `.venv/bin/python -m pytest tests/unit/test_skill_runtime.py tests/simulation/test_memory_tiers.py -q` |
| 5. CIL + CLI | done | CLI routes through `CommandInterfaceLayer`; command trace now records `operator_level` and `rollback_available`; user aliases and operator domains share one command schema. | `.venv/bin/python -m pytest tests/integration/test_cil_cli.py -q` |
| 6. observer + ablation + longrun | done | Observer exposes skill stats, conflict/mode timelines, and ablation as approximate `P_with - P_without` gain over implicit modules; full suite and longrun smoke pass. | `.venv/bin/python -m pytest tests/integration/test_observer_api.py tests/integration/test_observer_diagnostics.py tests/longrun/test_longrun_smoke.py -q` |

## Current Observable Capabilities

- Current stable baseline still supports `.venv/bin/alive state show`, `.venv/bin/alive trace round 1`, `.venv/bin/alive agent list`.
- Development branch for this rollout: `codex/v056-runtime`.
- Registry skeleton is live and enumerable through `RuntimeController.skill_list()` and the new unit contract tests.
- Runtime trace now exposes `distribution_state`, `stochastic_state`, `render_plan`, stage order, and stage-level proposal summaries.
- Formula-driven decision core is active: `P_base`, per-agent `delta_p`, CI update, clip/normalize, stochastic mixing, distribution-level plausibility gating, and output gating all leave trace evidence.
- Output expression layer now emits `reply_delay`, `latency_style`, `sentence_fragmentation`, `hedging_level`, `warmth_level`, `directness_level`, `self_disclosure`, `tone_sharpness`, `repair_tendency`, and jitter terms from the v0.56-style parameterization.
- Skill-level trace is written to `.alive/traces/skills/skill_traces.jsonl`.
- Command trace now includes `operator_level` and `rollback_available`.
- CIL-backed commands now include `.venv/bin/alive skill stats`, `.venv/bin/alive relation show user`, `.venv/bin/alive rest`, `.venv/bin/alive calm`.
- Observer diagnostics now include `GET /skills/stats`, `GET /skills/profile/{skill_name}`, `GET /metrics/conflicts`, `GET /metrics/mode-switches`, `GET /analysis/ablation`.
- Longrun smoke currently holds for 1000 interactive rounds without falling into `safe_mode`.

## Next Milestone

- Keep the runtime stable while replacing remaining heuristic providers with model-backed `PFC`, `Perspective`, and renderer implementations.

## Known Gaps Against v0.56

- Parquet analytics output is not enabled yet; current trace dual-path is JSON round files + JSONL streams.
- `ConflictMonitorAgent` still uses a lightweight heuristic conflict score and capped resample loop, not the full multi-pass controller from the spec.
- `SkillExecutor` currently applies lightweight input validation and strict output validation; it is not yet a typed runtime validator.
- High-cost model providers for `PFCAgent`, `PerspectiveModel`, and renderer are still fallback-first interfaces.
- Trace storage layout already reserves the v0.56 layers, but Parquet export and richer archive compaction are still missing.

## v0.56 公式层实现

- 决策核：各 agent 输出 `ProposalBundle(delta_p, confidence, sigma_scale, utility_shift, veto)`，controller 计算 `P_base -> P_raw -> Clip -> Normalize`，并把 CI 写回状态。
- 冲突与采样层：`ConflictMonitorAgent` 负责 `score_conflict -> trigger_control_escalation -> request_resample`；`ThalamusAttentionAgent` 负责聚合、归一化、采样。
- 分布级守卫层：`BehaviorPlausibilityGuard` 会扫描最终分布中的高概率违规动作，先把它们打到 `gate=0` 再重采样；`ForcedModeSwitch` 处理 mode/focus lock；`OutputGate` 做最终约束下压。
- 慢变量层：memory 使用 gist/detail + cue-weighted decay + interference，habit 使用 bounded strengthening，relation trace 和 stable priors 都会落盘。
- 情境随机层：在 deterministic distribution 上叠加 `xi_emo`、`xi_mood`、`lambda_noise`、`r_intensity`，并记录 `KL` guard 结果。
- 输出表达层：`build_expression_profile()` 按 `reply_delay`、`self_disclosure`、`tone_sharpness`、`repair_tendency` 公式和 jitter 范围生成表达参数。
- renderer 接口层：`build_render_plan()` 将 `sampled_action + ExpressionProfile + safety_constraints` 组合成可被后续 renderer/provider 消费的统一计划。
- 控制与观测层：CIL 命令统一写 command trace；observer 提供 conflicts、mode switches、skill stats、ablation 近似贡献。

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
.venv/bin/python -m uvicorn services.observer.api.app:app --reload
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
alive trace why 1
alive trace contribution 1
alive agent list
alive agent disable DMNAgent
alive checkpoint create
alive checkpoint rewind ckpt-0000
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
