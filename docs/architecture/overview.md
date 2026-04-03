# NALR Architecture Overview

NALR 采用 Python-first 的模块化运行时，并将 observer 保持为只读 sidecar。核心设计意图是让 proposal、状态存储、trace 和 CLI 共享同一套本地运行态目录，而不是过早拆成多服务写路径。

当前代码边界：

- `RuntimeController` 负责编排、状态持久化、命令应用、checkpoint，以及 model-backed late stage 的接线；它负责组装 canonical skill inputs 和 `SkillRuntimeContext`，在 `conflict` 阶段内部运行完整 multi-pass conflict controller，并把 `ConflictMonitorAgent.mark_post_error_adjustment` 返回的 repair contract 持久化为 canonical state/trace，但不直接替 provider 做 schema 校验。
- `IdentityRuntime` 负责身份锚点、query 分类、disclosure 策略、名字/别名演化，以及把 identity shaping sources 写入 trace/why 证据面。
- `AuthenticityPolicy` 负责候选级真实性惩罚、采样级降权，以及最终 `AuthenticityRecord` 的构造；真实性不再只是 render 后兜底。
- `VitalityEngine` 负责慢变量快照和 `idle/sleep` 非交互塑形，把这些变化写成结构化 vitality 事件并影响下一轮表达。
- `LongRunAnalyzer` 负责从 canonical round trace 聚合 `metrics_summary()`、`authenticity_timeline()`、`vitality_timeline()` 和每轮 `long_run_projection`。
- `SkillExecutor` 是统一 skill runtime：它拥有 typed contract coercion、`timeout_ms` / `cost_class` / `failure_policy` / `trace_tags` 约束、权限边界、`policy_check`、fallback routing 和 circuit breaker 持久化。
- Agent 模块仍不直接执行动作；`PFCAgent.generate_candidates` 改为 model-first，其他 proposal agent 继续走本地启发式或数值逻辑。
- `ModelRouter` 统一承接 `pfc` / `perspective` / `renderer` 三条 route，当前内置 Doubao / Ark backend 和 fake backend；provider route 只负责模型调用，不绕过 runtime policy。
- `TraceStore` 负责 JSON/JSONL replay-audit artifacts 与 canonical Parquet live read model 的双写；trace row 统一带 `session_id`、`recorded_at`、`recorded_date`，公共 trace/observer 读取默认走 Parquet canonical tables，并在降级时显式回退 JSON。
- `MemoryStore` 统一承接记忆、习惯、关系三个慢变量；它除了兼容聚合文件外，还维护 raw evidence stream 和 hot / warm / archive compaction artifacts。
- observer 仍是只读 sidecar，但其 trace 读取现在默认依赖 canonical Parquet live read model，不参与写路径；`/metrics/conflicts` 直接消费 round trace 中的 conflict components、winning priority、compromise template、repair stage / ledger summary 和 `conflict_hot` 状态。
- `terminal_bridge` 保持当前事件协议兼容，但 task-turn 现在固定输出 `run_status`、`step_update`、`tool_result`、`assistant_final` 序列，让 terminal panel 与 observer 共享同一条任务诊断链。

当前 skill 调用链路：

1. `RuntimeController` 为每个 skill 组装声明式输入，如 `event` / `state` / `scenario` / `context`，并附带 `SkillRuntimeContext(round_id, scenario, mode, safe_mode, operator_level, policy_flags)`。
2. `SkillExecutor` 先按 registry 中的 typed contract 校验并规范化输入，再把 provider 作为严格对齐的 kwargs 调用。
3. provider 返回后，`SkillExecutor` 会再次按 typed output contract 校验；若失败或被 policy 拒绝，则按 skill 的 failure policy 路由到 typed fallback。
4. 对带 breaker 的 H-skill，连续失败达到阈值后会开路；冷却期间 primary provider 会被跳过，直接由声明的低成本 fallback route 接管。
5. `ConflictMonitorAgent` 在同一轮内最多执行 3 个 pass：重算 5 个冲突分项、按优先级链裁决、必要时申请重采样或妥协模板，并通过公开 skill `mark_post_error_adjustment` 产出 `repair_mode`、post-error adjustment、repair FSM 迁移和 conflict-owned safe-mode recovery 建议。
6. 通过 `output_gate` 后，late `PerspectiveModel` 消费最终 gated action，随后 `build_render_plan()` 生成自包含的 `RenderPlan`，renderer 输出结构化 `RenderedExpression`。
7. `AuthenticityPolicy` 会先对候选分布施加 query-aware sampling penalty，再由 renderer 执行最终真实性 guard；round trace 同时保留 `authenticity`、`identity_evolution`、`vitality_snapshot`、`long_run_projection`。

配置与运行时约束：

- 模型路由配置在 `config/models.yaml`。
- live Doubao 路由依赖 `ARK_API_KEY`；缺失时运行时允许 degrade，不会阻断整轮 tick。
- 当前只有 Doubao / Ark 一个真实 backend，route 抽象已预留多 backend 扩展点。
- `alive trace export parquet`、`alive memory compact`、`alive memory sample` 都是显式 maintenance 命令，不写入 command trace；其中 `trace export parquet` 负责 rebuild/backfill，而不是常规实时生成入口。
- skill registry 在初始化阶段就 fail fast：非法 `timeout_ms`、`cost_class`、`failure_policy`、脏 `trace_tags`、缺少 fallback route 的 H-skill breaker 配置都会报错。
- skill 不能直接拿到 controller/store 句柄，也不能直接改写 `proposal_trace` 历史；它们只能通过 typed return contract 回传 `state_patch`、`gate`、`trace append request` 等受限载荷。
- 涉及 `external_io` 或 `social_risk` 的 skill 必须显式通过 `policy_check`，并受 `mode` / `safe_mode` / `operator_level` / `policy_flags` 约束。
- 当前封板口径是 focused runtime/observer/terminal verification + bounded long-run acceptance；重型 `test_longrun_smoke.py` 仍单列为后续 soak 验收，不在本页暗示为已 fresh 完成。
