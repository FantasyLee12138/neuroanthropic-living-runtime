# v0.56 Acceptance Matrix

Snapshot date: 2026-04-03

## Preface / 前言

- Source of truth order / 事实来源顺序: `开发文档v0.56.md` -> repo code/tests/traces -> `README.md` as navigation only
- Status legend / 状态图例: `implemented = 已实现`, `partial = 部分实现`, `missing = 缺失`
- Bilingual rule / 双语规则: every `Spec Item / 规格项` cell keeps Chinese and English together
- Evidence rule / 证据规则: every evidence cell uses `spec anchor -> code path -> proof`
- Verification snapshot / 验证快照:
  - Positive baseline evidence pack passed on 2026-04-03: `26 passed`
  - Supplemental skill-runtime regression check passed on 2026-04-03: `tests/unit/test_skill_runtime.py -q -> 5 passed`
  - Earlier exploratory failure around `test_tick_uses_heuristic_pfc_fallback_when_model_provider_fails` was not reproducible in final verification, so it is not used as blocking evidence in this matrix

## Summary Matrix / 总览矩阵

| Spec Item / 规格项 | Status / 实现状态 | Test / Trace Evidence / 测试或追踪证据 |
|---|---|---|
| Agent registry / Agent 注册表 | implemented | `§3.2-§3.7 Agent Registry -> src/nalr/agents/registry.py -> build_agent_registry()` defines the v0.56 roster, class split, wakeup rules, budgets, fallback policies; `tests/unit/test_registry_contracts.py` asserts the expected agent set. |
| Skill registry / Skill 注册表 | partial | `§4.1-§4.8 Skill Registry -> src/nalr/skills/registry.py; src/nalr/skills/executor.py -> typed skill specs, sync/async metadata, trace tags, policy flags, timeout/cost/failure metadata exist; tests/unit/test_registry_contracts.py and tests/unit/test_skill_runtime.py` pass, but async skills are declared in the registry more than they are scheduled as async runtime work. |
| runtime pipeline / 运行时管线 | implemented | `§3.6 Agent 唤醒顺序; §5.1-§5.3 行为采样总公式 -> src/nalr/runtime/controller.py -> PIPELINE_ORDER, distribution build, render plan, trace writeback are live; tests/unit/test_runtime_pipeline.py and tests/unit/test_formula_runtime_alignment.py` pass. |
| stochastic layer / 情境随机层 | partial | `§5.4-§5.12 情境随机层 -> src/nalr/runtime/controller.py; src/nalr/output/style.py -> xi_emo, xi_mood, lambda_noise, r_intensity, KL/noise guard, and expression mapping are implemented; spec-level social/public calibration rules are only partially reflected. |
| conflict loop / 冲突闭环 | partial | `§8.1-§8.5 冲突闭环 -> src/nalr/runtime/controller.py -> score_conflict -> request_resample -> plausibility second sampling exists; tests/unit/test_formula_runtime_alignment.py and tests/unit/test_agent_expansion.py` cover the live path, but explicit priority ordering and compromise templates are absent. |
| memory / habit / relation / 记忆-习惯-关系 | partial | `§9-§10 Habit/Hippocampus; relation touchpoints in §3/§5 -> src/nalr/memory/store.py -> hot/warm/archive tiers, decay/interference, habit strengthening, relation trace/closeness persist; tests/simulation/test_memory_and_habit.py and tests/simulation/test_memory_tiers.py` pass, but archive compaction and some recovery semantics are incomplete. |
| CLI / CIL / 命令行与命令接口层 | partial | `§5.1-§5.10 CLI/CIL -> src/nalr/cli/app.py; src/nalr/cil/runtime.py -> aliases, operator levels, domain routing, rollback-aware command traces exist; tests/integration/test_cil_cli.py and tests/integration/test_cli.py` pass, but only a subset of the registry/domain surface is implemented. |
| observer / analytics / 观测与分析 | partial | `§15-§16 proposal_trace/调试分析 -> services/observer/api/app.py; services/observer/dashboard/index.html -> read-only API, skill/conflict/mode/ablation views exist; tests/integration/test_observer_api.py and tests/integration/test_observer_diagnostics.py` pass, but dashboard depth and analytics storage are incomplete. |
| safe mode / fault guard / 安全模式与故障守卫 | partial | `§13.1-§13.4 故障兜底与 Safe Mode -> src/nalr/runtime/controller.py -> manual safe mode, checkpoint/rewind, starvation-triggered safe mode exist; tests/unit/test_runtime_controller.py and tests/longrun/test_longrun_smoke.py` pass, but heartbeat/fault replacement is only declared in the registry, not wired into runtime control. |

## Confirmed Implemented Groups / 已确认实现组

### Agent Registry / Agent 注册表

| Spec Item / 规格项 | Status / 实现状态 | Test / Trace Evidence / 测试或追踪证据 |
|---|---|---|
| 模块总表与显隐性分类 / Module roster and explicit-implicit-guard split | implemented | `§3.1-§3.5 Agent Registry -> src/nalr/agents/registry.py -> BodyStateAgent, RelationshipAgent, PFCAgent, HippocampusAgent, ThalamusAttentionAgent, guards, and implicit modules are all registered with class_kind; tests/unit/test_registry_contracts.py` checks the v0.56 roster. |
| 唤醒规则、预算级别、降级策略 / Wakeup rules, budget classes, fallback policies | implemented | `§3.2-§3.4 Agent Registry -> src/nalr/agents/registry.py -> each AgentSpec carries wakeup_rule, budget_class, fallback_policy; tests/unit/test_registry_contracts.py` validates PFCAgent as a representative contract. |
| 显隐性边界不让 Agent 越权 / Boundary preservation between agents and controller | implemented | `§3.6-§3.7 Agent 唤醒顺序与边界 -> src/nalr/runtime/controller.py -> controller owns orchestration and sampling order; agents contribute proposals only; tests/unit/test_runtime_pipeline.py` confirms the fixed stage chain ending in thalamus -> guards -> output gate. |

### Runtime Pipeline / 运行时管线

| Spec Item / 规格项 | Status / 实现状态 | Test / Trace Evidence / 测试或追踪证据 |
|---|---|---|
| 单轮阶段顺序 / Per-round stage order | implemented | `§3.6 Agent 唤醒顺序 -> src/nalr/runtime/controller.py -> PIPELINE_ORDER lists state_update -> salience -> ... -> output_gate -> writeback; tests/unit/test_runtime_pipeline.py` asserts the full ordered stage list. |
| `P_base -> P_raw -> P_mix -> P_final` 分布链 / Distribution chain | implemented | `§5.1-§5.9 行为采样总公式 -> src/nalr/runtime/controller.py::_build_distribution_state(); _apply_stochastic_layer() -> distribution_state records p_base, p_raw, p_mix, p_final, ci, gate; tests/unit/test_formula_runtime_alignment.py` asserts these fields and normalization. |
| render plan 与 trace 写回 / Render plan and trace writeback | implemented | `§15.1-§15.2 proposal_trace; §18 输出表达层 -> src/nalr/runtime/controller.py; src/nalr/trace/store.py -> build_render_plan() output is saved into RoundTrace and TraceStore.write_round(); .alive/traces/rounds/round_1.json` shows persisted round evidence. |

## Mixed Or Partial Groups / 混合或部分实现组

### Skill Registry / Skill 注册表

| Spec Item / 规格项 | Status / 实现状态 | Test / Trace Evidence / 测试或追踪证据 |
|---|---|---|
| typed callable 定义 / Typed callable spec fields | implemented | `§4.1 skill 定义 -> src/nalr/schemas/models.py::SkillSpec; src/nalr/skills/registry.py -> name, owner_module, input_schema, output_schema, timeout_ms, cost_class, failure_policy, trace_tags are all materialized; tests/unit/test_registry_contracts.py` passes. |
| sync/async、trace tags、权限标记 / Sync mode, trace tags, policy flags | implemented | `§4.3-§4.8 Skill Registry -> src/nalr/skills/registry.py -> flush_trace_batch is async, guarding skills carry policy_check, all specs carry trace_tags; tests/unit/test_registry_contracts.py` checks async + typed metadata. |
| failure policy、fallback、circuit breaker / Failure policy, fallback, circuit breaker | implemented | `§4.5 skill 失败策略 -> src/nalr/skills/executor.py -> fallback, output validation, circuit-breaker paths exist; tests/unit/test_skill_runtime.py` passes for fallback policy, output validation rejection, circuit-breaker trips, and trace-file emission. |
| async 调用协议真正落地 / Async invocation protocol actually executed | partial | `§4.4 调用协议 -> src/nalr/skills/registry.py -> checkpoint_create, record_round_trace, record_agent_contribution, flush_trace_batch are declared with sync_mode=\"async\"; rg search across src/tests shows sync_mode is asserted in registry contracts, but there is no separate async dispatcher in runtime/controller code.` |

### Stochastic Layer / 情境随机层

| Spec Item / 规格项 | Status / 实现状态 | Test / Trace Evidence / 测试或追踪证据 |
|---|---|---|
| `xi_emo`、`xi_mood`、`lambda_noise`、`r_intensity` 落地 / Core stochastic variables | implemented | `§5.4-§5.10 情境随机层 -> src/nalr/runtime/controller.py::_apply_stochastic_layer() -> StochasticState stores emo_channel, xi_emo, xi_mood, lambda_noise, r_intensity, round_seed; tests/unit/test_formula_runtime_alignment.py` confirms stochastic_state is persisted. |
| KL / noise guard 与 trace 字段 / KL-noise guard and trace coverage | implemented | `§5.12 调试与安全边界 -> src/nalr/runtime/controller.py::_apply_stochastic_layer() -> KL divergence is computed and noise_guard_triggered is set when over threshold; .alive/traces/rounds/round_1.json and .alive/traces/round_traces.jsonl` reserve stochastic_state fields in persisted traces. |
| 强度到表达层映射 / Intensity-to-expression mapping | implemented | `§5.10-§5.11 强度采样与输出映射 -> src/nalr/output/style.py::build_expression_profile() -> r_intensity changes delay, self_disclosure, tone_sharpness, repair_tendency; tests/unit/test_output_expression_layer.py` passes. |
| 陌生关系/公共场景默认降幅 / Stranger-public default downshift calibration | partial | `§5.12 调试与安全边界 -> src/nalr/runtime/controller.py; src/nalr/output/style.py -> relation_risk and privacy_level influence noise/intensity, but there is no explicit public-scene or stranger-mode calibration table matching the spec percentages.` |

### Conflict Loop / 冲突闭环

| Spec Item / 规格项 | Status / 实现状态 | Test / Trace Evidence / 测试或追踪证据 |
|---|---|---|
| conflict score 与 resample 请求 / Conflict score and resample request | implemented | `§8.1-§8.3 冲突分数与重采样 -> src/nalr/runtime/controller.py -> score_conflict, trigger_control_escalation, request_resample are executed inside tick(); tests/unit/test_formula_runtime_alignment.py` shows the conflict and thalamus skills run independently. |
| plausibility 二次采样联动 / Coupling to plausibility second sampling | implemented | `§8.3 重采样规则; §12 Behavior Plausibility Guard -> src/nalr/runtime/controller.py -> dominant blocked action can trigger request_second_sampling and escalate_value_reestimate; tests/unit/test_agent_expansion.py` proves wander can be blocked and resampled in task runtime. |
| 显式冲突优先级裁决 / Explicit conflict priority ordering | missing | `§8.2 冲突优先级 -> src/nalr/runtime/controller.py -> there is no dedicated priority table or ordered resolver for body/safety/relationship/task layers; current flow uses aggregate scoring plus guards only.` |
| 多轮妥协模板 / Multi-round compromise templates | missing | `§8.4 多轮冲突后的妥协模板 -> src/nalr/runtime/controller.py -> no template selector for “最低风险输出 / 关系优先妥协 / 预算优先妥协”; current path resamples and then continues into guards/output gate.` |
| 死锁熔断 / Deadlock fuse after repeated conflicts | partial | `§8.5 冲突死锁熔断 -> src/nalr/runtime/controller.py -> resample_count is tracked and safe-mode edges exist elsewhere, but there is no standalone deadlock fuse controller matching the spec’s explicit fuse policy.` |

### Memory / Habit / Relation / 记忆-习惯-关系

| Spec Item / 规格项 | Status / 实现状态 | Test / Trace Evidence / 测试或追踪证据 |
|---|---|---|
| hot/warm/archive 分层与 stable priors / Tiering and stable priors | implemented | `§10.7 存储清理/压缩; §24 建议目录结构 -> src/nalr/memory/store.py -> episodic_hot, episodic_warm, episodic_archive, stable_priors files are created; tests/simulation/test_memory_tiers.py` checks warm/archive/stable_priors files exist. |
| 线索衰减、干扰、习惯上限 / Cue decay, interference, habit cap | implemented | `§9.1-§9.5 Habit; §10.1-§10.4 Hippocampus -> src/nalr/memory/store.py -> gist/detail decay, interference penalty, bounded habit strengthening are applied; tests/simulation/test_memory_tiers.py` checks decay/interference and habit cap, and `tests/simulation/test_memory_and_habit.py` checks growth under repetition. |
| relation closeness 与 relation trace / Relation closeness and trace | implemented | `§3 RelationshipAgent; §5 relation domain -> src/nalr/memory/store.py -> relation.json and relation_trace.json are updated on ingest_event(); src/nalr/runtime/controller.py::relation_show() exposes closeness; tests/integration/test_cil_cli.py` verifies `relation show user`. |
| 旧习惯 suppressed-but-recoverable 状态 / Old-habit suppressed-but-recoverable state | missing | `§9.4-§9.5 旧习惯恢复 -> src/nalr/memory/store.py -> strength/count are updated, but there is no explicit suppressed state or recovery-mode state machine.` |
| 证据指针/哈希摘要压缩 / Evidence-pointer and hash-summary compaction | missing | `§10.7 存储清理/压缩 -> src/nalr/memory/store.py -> storage stays JSON lists with copied entries; there is no evidence-pointer-only or hash-summary archive compaction path.` |

### CLI / CIL / 命令行与命令接口层

| Spec Item / 规格项 | Status / 实现状态 | Test / Trace Evidence / 测试或追踪证据 |
|---|---|---|
| 命令解析、别名、权限级别 / Parsing, aliases, operator levels | implemented | `§5.1-§5.4 CLI/CIL 分层与权限 -> src/nalr/cil/runtime.py -> alias_map resolves rest/calm/explain current, operator_level is assigned by domain; tests/integration/test_cil_cli.py` passes for aliases and operator-level traces. |
| CLI -> CIL -> Runtime 路由 / CLI to CIL to Runtime routing | implemented | `§5.9 CIL 实现职责 -> src/nalr/cli/app.py; src/nalr/cil/runtime.py -> relation/memory/habit/trace/agent/skill/safe/checkpoint routes are wired; tests/integration/test_cli.py and tests/integration/test_cil_cli.py` pass. |
| rollback-aware command trace / 可回滚命令追踪 | implemented | `§5.6-§5.7 标准反馈与 trace 联动 -> src/nalr/trace/store.py::append_command(); src/nalr/runtime/controller.py::apply_command() -> command traces include operator_level and rollback_available; .alive/traces/command_traces.json` persists the fields. |
| 最低可用命令清单完整覆盖 / Full minimum command-set coverage | partial | `§5.10 最低可用 CLI 命令清单 -> src/nalr/cil/runtime.py; src/nalr/cli/app.py -> core state/trace/agent/relation/memory/habit/safe/checkpoint commands exist, but the spec’s wider domain surface is only partially present.` |
| 双层交互完整分层 / Full dual-layer user/operator interface separation | partial | `§17.1-§17.3 双层交互 -> src/nalr/cil/runtime.py -> user aliases and operator levels coexist, but they still share one Typer surface and one routing layer rather than a more strongly separated UX.` |

### Observer / Analytics / 观测与分析

| Spec Item / 规格项 | Status / 实现状态 | Test / Trace Evidence / 测试或追踪证据 |
|---|---|---|
| 只读 observer API / Read-only observer API | implemented | `§15 proposal_trace; §16 隐性模块调试 -> services/observer/api/app.py -> /state, /trace/{round_id}, /why/{round_id}, /contributions/{round_id}, /skills/stats, /skills/profile/{skill_name}, /metrics/conflicts, /metrics/mode-switches, /analysis/ablation are exposed; tests/integration/test_observer_api.py and tests/integration/test_observer_diagnostics.py` pass. |
| conflict / mode / skill / ablation 诊断视图 / Diagnostic views | implemented | `§15.5 必备图表; §16.2 贡献度量化 -> src/nalr/runtime/controller.py::conflict_timeline(), mode_switch_timeline(), skill_profile(), ablation_summary(); tests/integration/test_observer_diagnostics.py` verifies populated responses. |
| dashboard 落地深度 / Dashboard implementation depth | partial | `§15.4 可视化落地路径 -> services/observer/dashboard/index.html -> dashboard exists, but it is a static placeholder that tells readers to start from /state and /trace/{round_id}; tests/integration/test_observer_api.py` only checks the page is served. |
| Parquet 批量分析输出 / Parquet analytics output | missing | `§15.1 存储格式 -> src/nalr/trace/store.py -> write_round() and append_command() write JSON/JSONL only; .alive/traces/round_traces.jsonl and .alive/traces/skills/skill_traces.jsonl` exist, but there is no Parquet writer or export path. |

### Safe Mode / Fault Guard / 安全模式与故障守卫

| Spec Item / 规格项 | Status / 实现状态 | Test / Trace Evidence / 测试或追踪证据 |
|---|---|---|
| 手动 safe on/off 与 checkpoint/rewind / Manual safe mode and checkpoint-rewind | implemented | `§13.4 Safe Mode; §5 checkpoint commands -> src/nalr/runtime/controller.py::apply_command(); checkpoint(); rewind() -> safe mode and checkpoint flows are live; tests/unit/test_runtime_controller.py and tests/integration/test_cli.py` pass. |
| starvation 触发 safe mode / Starvation-triggered safe mode | partial | `§6.5 starvation 模式; §13.3 概率异常保护 -> src/nalr/runtime/controller.py -> budget_remaining below the starvation threshold flips state.safe_mode and mode=safe, but there is no dedicated regression test covering this exact edge.` |
| safe mode 命令 trace 与可回滚信息 / Safe-mode command trace and rollback metadata | implemented | `§5.6-§5.7 CLI 与 trace 联动 -> src/nalr/runtime/controller.py; src/nalr/trace/store.py -> safe on/off commands emit rollback_hint, operator_level, rollback_available and persist them into command traces; tests/integration/test_cil_cli.py` verifies the stored fields. |
| heartbeat / replace / rollback 梯度故障守卫 / Heartbeat-replace-rollback fault guard | missing | `§13.1 Agent 响应梯度; §13.4 Safe Mode -> src/nalr/skills/registry.py -> heartbeat_check and switch_to_safe_mode are only registered as skills; rg search shows no runtime invocation or process-replace path in controller/tests.` |

## Notes / 备注

- This matrix is intentionally conservative. If a spec behavior exists only as a registry entry, config placeholder, or README claim, it is not upgraded to `implemented` without code-path or test-path evidence.
- Local trace evidence currently exists in `.alive/traces/rounds/round_1.json`, `.alive/traces/round_traces.jsonl`, `.alive/traces/skills/skill_traces.jsonl`, and `.alive/traces/command_traces.json`.
