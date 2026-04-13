# 模块模型映射现状说明

## 1. 术语定义

- `module / agent`
  指运行时中的行为模块。本文默认以 [`src/nalr/agents/modules.py`](src/nalr/agents/modules.py) 的 `build_agents()` 产物为主，因为它更接近实际运行时装配；[`src/nalr/agents/registry.py`](src/nalr/agents/registry.py) 是契约注册表，和运行时 agent 列表并不完全一一对应。
- `binding`
  指 `agent_model_bindings` 中的绑定结果，例如 `PFCAgent -> medium_model`。
- `tier`
  指 `model_tiers` 中定义的模型层，如 `state_machine`、`small_model`、`medium_model`、`large_model`。
- `route`
  指 `model_routes` 中声明的命名调用入口，例如 `planner`、`pfc`、`renderer_fallback_fast`。
- `effective route`
  指运行时真正送到 provider 的配置。对于走 `_call_bound_model_route()` 的链路，effective route 主要由 `binding -> tier` 解析得到；对于 fallback renderer 这种直接 `model_router.generate(route_name, ...)` 的链路，则按命名 route 的静态声明执行。

当前 tier 基线如下，来源是 [`config/models.yaml`](config/models.yaml)：

| tier | mode | backend | model | 说明 |
| --- | --- | --- | --- | --- |
| `state_machine` | `local` | 本地规则 | 无远程模型 | 只走状态机 / 规则，不发远程请求 |
| `small_model` | `remote` | `doubao` | `ep-20260404191810-qfn7s` | 小模型层 |
| `medium_model` | `remote` | `deepseek` | `deepseek-chat` | 中模型层 |
| `large_model` | `remote` | `doubao` | `doubao-seed-2-0-pro-260215` | 大模型层 |

## 2. 配置源与优先级

模型映射不是单层配置，当前生效顺序是：

1. [`config/models.yaml`](config/models.yaml)
   提供 `model_routes`、`model_tiers`、`agent_model_bindings` 的基础声明。
2. observer settings 覆盖
   [`src/nalr/runtime/controller.py`](src/nalr/runtime/controller.py) 中 `_apply_observer_settings_to_models()` 会覆盖 `model_routes`、`model_tiers`、`agent_model_bindings`。
3. 运行时 tier 解析
   `_agent_model_bindings()` 读取 agent 到 tier 的绑定，`_effective_agent_tier()` 在运行时按风险决定是否升级 tier，`_route_config_for_binding()` 再把 tier 展开为最终 `backend/model`。
4. 命名 route 直连例外
   fallback renderer 链路在 `_fallback_render_route_names()` 和对应 fallback 调用里直接用 `model_router.generate(route_name, ...)`，这里优先遵循命名 route 声明，而不是 `Renderer -> large_model` 的 tier 解析。

这意味着文档里必须区分三件事：

- `binding` 是 agent 对 tier 的绑定。
- `route` 是命名入口。
- `backend/model` 才是最终发给 provider 的具体模型。

## 3. 完整映射表

### 3.1 运行时 agent 模块

下表按 [`src/nalr/agents/modules.py`](src/nalr/agents/modules.py) 的 `build_agents()` 列出当前运行时 agent。`默认 backend / 默认 model` 表示默认条件下的实际有效模型；若为 `state_machine`，则明确标成“本地规则，无远程模型”。

| 模块名 | 模块类型 | binding/tier | 默认 backend | 默认 model | 是否动态升级 | 升级条件 | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `SalienceAgent` | agent | `small_model` | `doubao` | `ep-20260404191810-qfn7s` | 否 | 无 | 通过内部 route 名 `salience_small_model` 解析，不在 `model_routes` 静态表中单独声明 |
| `BodyStateAgent` | agent | `state_machine` | 本地规则 | 无远程模型 | 否 | 无 | 只走本地状态机 |
| `EmotionAgent` | agent | `state_machine` | 本地规则 | 无远程模型 | 否 | 无 | 只走本地状态机 |
| `RelationshipAgent` | agent | `state_machine` | 本地规则 | 无远程模型 | 否 | 无 | 只走本地状态机 |
| `ResourceAgent` | agent | `state_machine` | 本地规则 | 无远程模型 | 否 | 无 | 只走本地状态机 |
| `PFCAgent` | agent | `medium_model` | `deepseek` | `deepseek-chat` | 是 | `relation_risk >= 0.62`，或 `disclosure_sensitivity >= 0.55`，或 `authenticity_risk >= 0.32`，或 `conflict_score >= conflict_high`，或 `resample_count >= 2` | 升级后切到 `large_model -> doubao-seed-2-0-pro-260215` |
| `InitiativeInteractionAgent` | agent | `state_machine` | 本地规则 | 无远程模型 | 否 | 无 | 当前配置绑定为 `state_machine` |
| `HabitAgent` | agent | `state_machine` | 本地规则 | 无远程模型 | 否 | 无 | 只走本地状态机 |
| `DesireAgent` | agent | `state_machine` | 本地规则 | 无远程模型 | 否 | 无 | 只走本地状态机 |
| `DMNAgent` | agent | `state_machine` | 本地规则 | 无远程模型 | 否 | 无 | 只走本地状态机 |
| `HippocampusAgent` | agent | `state_machine` | 本地规则 | 无远程模型 | 否 | 无 | 只走本地状态机 |
| `PerspectiveModel` | agent | `medium_model` | `deepseek` | `deepseek-chat` | 是 | `relation_risk >= 0.72`，或 `disclosure_sensitivity >= 0.72` | 升级后切到 `large_model -> doubao-seed-2-0-pro-260215` |
| `ValueAgent` | agent | `small_model` | `doubao` | `ep-20260404191810-qfn7s` | 否 | 无 | 通过内部 route 名 `value_small_model` 解析，不在 `model_routes` 静态表中单独声明 |
| `ConflictMonitorAgent` | agent | `state_machine` | 本地规则 | 无远程模型 | 否 | 无 | 只走本地状态机 |
| `ThalamusAttentionAgent` | agent | `state_machine` | 本地规则 | 无远程模型 | 否 | 无 | 只走本地状态机 |
| `UnconsciousAgent` | agent | `state_machine` | 本地规则 | 无远程模型 | 否 | 无 | 只走本地状态机 |
| `CerebellarPredictor` | agent | `state_machine` | 本地规则 | 无远程模型 | 否 | 无 | 只走本地状态机 |
| `BehaviorPlausibilityGuard` | agent | `state_machine` | 本地规则 | 无远程模型 | 否 | 无 | 只走本地状态机 |
| `ForcedModeSwitch` | agent | `state_machine` | 本地规则 | 无远程模型 | 否 | 无 | 只走本地状态机 |
| `OutputGate` | agent | `state_machine` | 本地规则 | 无远程模型 | 否 | 无 | 只走本地状态机 |

### 3.2 命名 routes

下表以 [`config/models.yaml`](config/models.yaml) 的 `model_routes` 为主，回答“这个 route 名字声明成了什么”。其中“默认 backend / 默认 model”按 route 自身声明填写；备注里补充它在运行时是否还会被 binding tier 覆盖。

| 模块名 | 模块类型 | binding/tier | 默认 backend | 默认 model | 是否动态升级 | 升级条件 | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `planner` | route | `planner -> medium_model` | `doubao` | `doubao-seed-2-0-pro-260215` | 间接 | 取决于 `planner` binding；默认 effective route 实际走 `medium_model -> deepseek-chat` | route 声明是 Doubao，但 `_plan_run_via_model()` 默认经 `_call_bound_model_route()` 解析为 `deepseek-chat`；若 `planner` route 被禁用，则退到 `pfc` |
| `pfc` | route | `PFCAgent -> medium_model` | `doubao` | `doubao-seed-2-0-pro-260215` | 间接 | 取决于 `PFCAgent` 动态升级规则 | route 声明是 Doubao，默认 effective route 实际走 `deepseek-chat` |
| `perspective` | route | `PerspectiveModel -> medium_model` | `doubao` | `doubao-seed-2-0-pro-260215` | 间接 | 取决于 `PerspectiveModel` 动态升级规则 | route 声明是 Doubao，默认 effective route 实际走 `deepseek-chat` |
| `renderer` | route | `Renderer -> large_model` | `doubao` | `doubao-seed-2-0-pro-260215` | 否 | 无 | route 声明与默认 effective route 一致 |
| `renderer_fallback_fast` | route | 直连 route | `deepseek` | `deepseek-chat` | 否 | 无 | fallback renderer 链直接按 route 名调用，优先尝试这一条 |
| `renderer_fallback_small` | route | 直连 route | `doubao` | `ep-20260404191810-qfn7s` | 否 | 无 | fallback renderer 第二跳；只有 `renderer_fallback_fast` 不可用时才会试 |
| `monologue_stream` | route | `MonologueStream -> small_model` | `doubao` | `ep-20260404191810-qfn7s` | 否 | 无 | route 声明与默认 effective route 一致 |
| `autonomy_self_run` | route | `AutonomySelfRun -> small_model` | `doubao` | `ep-20260404191810-qfn7s` | 否 | 无 | 主要供自治只读自查 goal 生成使用 |
| `chat_fast` | route | 直连 route | `deepseek` | `deepseek-chat` | 否 | 无 | 用于快路径聊天和 fallback route 组选择 |
| `tool_policy` | route | 直连 route | `doubao` | `doubao-seed-2-0-pro-260215` | 否 | 无 | 当前 `enabled: false` |
| `vision_observer` | route | 直连 route | `doubao` | `doubao-seed-2-0-pro-260215` | 否 | 无 | 当前 `enabled: false` |
| `browser_observer` | route | 直连 route | `doubao` | `doubao-seed-2-0-pro-260215` | 否 | 无 | 当前 `enabled: false` |
| `audio_observer` | route | 直连 route | `doubao` | `doubao-seed-2-0-pro-260215` | 否 | 无 | 当前 `enabled: false` |
| `external_llm_adapter` | route | 直连 route | `doubao` | `doubao-seed-2-0-pro-260215` | 否 | 无 | 当前 `enabled: false` |

### 3.3 特殊运行时模块

这些模块不在 `build_agents()` 主表里，但运行时明确以独立 binding 使用，值得单独列出。

| 模块名 | 模块类型 | binding/tier | 默认 backend | 默认 model | 是否动态升级 | 升级条件 | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `planner` | 特殊运行时模块 | `medium_model` | `deepseek` | `deepseek-chat` | 否 | 无 | `_plan_run_via_model()` 中优先用 `planner` route 名，但 effective route 默认按 binding 展开到 `medium_model` |
| `Renderer` | 特殊运行时模块 | `large_model` | `doubao` | `doubao-seed-2-0-pro-260215` | 否 | 无 | 主渲染链走 `_call_bound_model_route("Renderer", route_name="renderer")` |
| `MonologueStream` | 特殊运行时模块 | `small_model` | `doubao` | `ep-20260404191810-qfn7s` | 否 | 无 | 内生轻量路径的显式 route 是 `monologue_stream` |
| `AutonomySelfRun` | 特殊运行时模块 | `small_model` | `doubao` | `ep-20260404191810-qfn7s` | 否 | 无 | 用于自治模式下只读 goal 生成，测试已覆盖其优先走 `small_model` |

## 4. 动态规则说明

### 4.1 PFCAgent 的 tier 升级

`PFCAgent` 的基础绑定是 `medium_model`，默认有效模型是 `deepseek-chat`。但 [`src/nalr/runtime/controller.py`](src/nalr/runtime/controller.py) 中 `_effective_agent_tier()` 会在下列任一条件满足时把它升级到 `large_model`：

- `relation_risk >= 0.62`
- `disclosure_sensitivity >= 0.55`
- `authenticity_risk >= 0.32`
- `conflict_score >= conflict_high`
- `resample_count >= 2`

其中 `conflict_high` 不写死在代码里，而是来自 [`config/thresholds.yaml`](config/thresholds.yaml)，当前值是 `0.75`。因此 `0.71` 这类输入不会升级，`0.75` 及以上才会触发这一路条件。

### 4.2 PerspectiveModel 的 tier 升级

`PerspectiveModel` 的基础绑定也是 `medium_model`，默认有效模型是 `deepseek-chat`。当前只在两个条件下升级到 `large_model`：

- `relation_risk >= 0.72`
- `disclosure_sensitivity >= 0.72`

它当前不会因为 `authenticity_risk`、`conflict_score` 或 `resample_count` 单独升级。

### 4.3 planner 的 route 选择

`planner` 的逻辑分两层：

- route 选择层：`_plan_run_via_model()` 会先看 `config.models.model_routes.planner.enabled`
- effective model 层：只要走 `_call_bound_model_route("planner", route_name=...)`，默认仍按 `planner -> medium_model -> deepseek-chat` 展开

当前行为是：

- `planner` route 启用时，优先使用 route 名 `planner`
- `planner` route 关闭时，退回 route 名 `pfc`
- 但默认有效模型依然取决于 `planner` binding，而不是取决于 `planner` route 静态声明里写的 `doubao-seed-2-0-pro-260215`

### 4.4 observer settings 覆盖

observer settings 可在运行时覆盖三类模型配置：

- `model_routes`
- `model_tiers`
- `agent_model_bindings`

因此“某模块当前对应什么模型”理论上分为两层答案：

- 仓库默认答案：以 `config/models.yaml` 为准
- 当前实例生效答案：以加载 observer settings 后的 `self.config["models"]` 为准

本文表格描述的是仓库默认答案，并结合当前控制器解析逻辑补上默认 effective route。

### 4.5 renderer fallback 链是例外

`renderer` 主链通过 `Renderer -> large_model` 解析；但 fallback 链不是这样。

当前 fallback 顺序由 `_fallback_render_route_names()` 固定为：

1. `renderer_fallback_fast`
2. `renderer_fallback_small`
3. 如果两者都失败，再回 deterministic `fallback_render_text()`

这一段在代码里直接调用 `model_router.generate(route_name, request)`，所以它遵循命名 route 的静态配置：

- `renderer_fallback_fast -> deepseek-chat`
- `renderer_fallback_small -> ep-20260404191810-qfn7s`

不要把这一段和 `Renderer -> large_model` 的主渲染链混为一谈。

## 5. 现有查询入口与局限

当前仓库已经有一个现成入口：terminal 控制命令 `model`。相关输出整理逻辑在 [`src/nalr/terminal_bridge/handlers.py`](src/nalr/terminal_bridge/handlers.py) 的 `_model_summary()`。

它目前能回答的主要是：

- tier 层的配置概览
- 主要 agent 的 binding，例如 `SalienceAgent -> small_model`
- credential 是否存在

它目前不能完整回答的内容包括：

- 所有运行时 agent 的全量映射
- 内部 route 名和特殊运行时模块，如 `AutonomySelfRun`、`MonologueStream`
- route 静态声明与 effective route 之间的差异
- `PFCAgent` / `PerspectiveModel` 的动态升级条件
- fallback renderer 链路是 route 直连而不是 tier 展开的事实

因此，如果问题是“这个系统现在每个模块到底对应哪个模型”，不能只看 `model` 命令输出；还需要同时看：

- [`config/models.yaml`](config/models.yaml)
- [`src/nalr/runtime/controller.py`](src/nalr/runtime/controller.py)
- 相关测试：
  - [`tests/unit/test_runtime_controller.py`](tests/unit/test_runtime_controller.py)
  - [`tests/unit/test_terminal_bridge.py`](tests/unit/test_terminal_bridge.py)
