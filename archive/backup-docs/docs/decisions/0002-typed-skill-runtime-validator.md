# ADR 0002: Typed Skill Runtime Validator

## Context

v0.56 要求 skill registry 中的每个 skill 都是可注册、可计时、可降级的 typed callable。此前实现只有轻量 input 检查和较严格的 output shape 校验，仍然允许以下偏差同时存在：

- registry 里的 `input_schema` / `output_schema` 更像文档标签，不是真正的 runtime contract
- `timeout_ms` / `cost_class` / `failure_policy` / `trace_tags` 由各处零散解释
- 高成本 skill 熔断后主要依赖 shaped payload，而不是真正的低成本 provider takeover
- external IO / social-risk skill 缺少统一权限边界与 `policy_check`

## Decision

将 `SkillExecutor` 升级为统一的 typed runtime validator，并把 contract、fallback 和 breaker 都收口到 `SkillSpec`。

- contract 采用 Python stdlib `typing` + dataclass introspection 表达，不引入新的 schema/validation 依赖。
- `SkillExecutor` 在 provider 调用前后统一执行 typed input/output validation，并对 `timeout_ms`、`cost_class`、`failure_policy`、`trace_tags` 做统一运行时约束。
- `SkillSpec` 显式承载 `permission_profile`、`fallback_route`、`breaker_policy`；涉及 `external_io` 或 `social_risk` 的 skill 必须开启 `policy_check`。
- breaker 默认策略为连续 3 次失败后开路，冷却 5 轮；开路期间 primary provider 跳过，由声明的低成本 fallback route 自动接管。
- fallback routing 挂在 skill/provider contract 上，而不是保留 executor 内部的静态 shaped fallback payload。

## Consequences

- registry 构建阶段可以 fail fast，避免非法 contract 或无效 fallback 配置进入运行态。
- controller、executor、provider 签名和 trace payload 可以共享同一套 contract，减少“声明输入”和“真实调用”分叉。
- skill 不再能隐式持有 controller/store 写权限，必须通过 typed return payload 表达 `state_patch`、`gate`、`trace append request` 等有限结果。
- fallback 行为变成可观测 runtime 事件，trace 和 CLI/observer 可以直接统计 `fallback_count`、`breaker_trip_count`、`policy_rejection_reason`。
- 代价是 provider adapter 需要更严格遵守 contract，模型 prompt/response 适配器也必须承担更明确的结构化输出责任。
