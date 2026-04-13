# 2026-04-08 Observer / Runtime Truth Audit

## Scope

本审计只以当前主线文档和当前代码为准，不把 `archive/` 内历史文档当成硬验收标准。

实际判定基线如下：

- [`README.md`](/Users/fantasylee/类脑架构/README.md)
- [`PLAN1.6.md`](/Users/fantasylee/类脑架构/PLAN1.6.md)
- [`架构宣言.md`](/Users/fantasylee/类脑架构/架构宣言.md)
- [`archive/backup-docs/root/BEREAL_v0.6_MASTER_SPEC.md`](/Users/fantasylee/类脑架构/archive/backup-docs/root/BEREAL_v0.6_MASTER_SPEC.md)
- [`archive/backup-docs/root/PLANS.md`](/Users/fantasylee/类脑架构/archive/backup-docs/root/PLANS.md)
- [`archive/backup-docs/root/Think_Like_Human(TLH)_v1.0.md`](/Users/fantasylee/类脑架构/archive/backup-docs/root/Think_Like_Human(TLH)_v1.0.md)
- [`archive/backup-docs/root/Think_Like_Human(TLH)_v1.5.md`](/Users/fantasylee/类脑架构/archive/backup-docs/root/Think_Like_Human(TLH)_v1.5.md)

说明：

- `archive/backup-docs/root/Think_Like_Human(TLH)_v1.0.md` 当前已经退化为兼容入口，正文明确把有效 TLH 内容指向 `v1.5`。
- 因此本审计把 `v1.0 -> v1.5` 视作当前 TLH 主线的合法跳转，而不是两个并列规范。

## Acceptance Matrix

| Requirement | Status | Code evidence | Fresh verification | Live evidence | Notes |
| --- | --- | --- | --- | --- | --- |
| 单一概率真相面不能被第二状态面破坏；observer / dashboard / launcher 不能各说各话 | `partial` | [`services/observer/api/app.py`](/Users/fantasylee/类脑架构/services/observer/api/app.py) 中 `/service/status`、`/autonomy/status`、`/web/runtime/bootstrap` 统一从同一 runtime/controller 派生；[`services/observer/dashboard/index.html`](/Users/fantasylee/类脑架构/services/observer/dashboard/index.html) 首屏 badge 改为消费 bootstrap + service/autonomy 轻量真相 | `pytest tests/integration/test_observer_api.py -q` `60 passed`；`pytest tests/browser/test_observer_dashboard_smoke.py -q` `1 passed` | 隔离实例 `8875` 上，轻量状态接口把 runner 附着后，`service/autonomy/bootstrap` 对“服务在线、session 未附着、接纳未开启”和“显式开始运行后已附着”给出一致结论 | 仍不是 `met`，因为 [`PLAN1.6.md`](/Users/fantasylee/类脑架构/PLAN1.6.md) 仍把 canonical read-model 收敛列为 in-flight，terminal / dashboard / observer API 尚未完全共用一套 schema |
| body / subjective / spontaneous 仍是一等行为来源，而不是 observer 报表装饰 | `met` | [`src/nalr/runtime/controller.py`](/Users/fantasylee/类脑架构/src/nalr/runtime/controller.py) 的 `_trace_state_snapshot_payload()` 继续把 `body_state`、`subjective_state`、`emotion_state`、`desire_state`、`instinct_field`、`organic_mode` 写入 runtime/trace 快照 | `pytest tests/unit/test_tlh_runtime.py -q` `18 passed` | workbench/browser smoke 仍可渲染分析面和当前状态，没有因 observer truth 收紧而丢掉主体态读取 | 这一项本轮没有新增逻辑，只验证当前主线未被回归破坏 |
| autonomy 真相必须拆成“已启用 / runner 已附着 / heartbeat 真在推进”，且 stalled 要可见 | `met` | [`src/nalr/runtime/controller.py`](/Users/fantasylee/类脑架构/src/nalr/runtime/controller.py) 新增 `autonomy_runtime_status()`；[`services/observer/api/app.py`](/Users/fantasylee/类脑架构/services/observer/api/app.py) 新增 stall 判定与 `heartbeat_state`，并让 heartbeat loop 走轻量状态路径 | `pytest tests/unit/test_runtime_controller.py -k 'autonomy_status or autonomy_runtime_status_is_lightweight' -q` `4 passed`；`pytest tests/integration/test_observer_api.py -q` `60 passed`；`pytest tests/integration/test_observer_diagnostics.py -q` `4 passed` | 隔离实例 `8875` 上，请求轻量 autonomy/bootstrap 接口后 runner 会被附着；随后 `autonomy_after.last_step_at` 推进，payload 显式暴露 `stalled=false` / `heartbeat_state=running`，不再靠模糊文案兜底 | 这是本轮修复的核心闭环 |
| fault guard 必须能在 runtime / observer 读到，但 `replace` 只能是 baseline fallback contract，不伪装 process replace | `met` | [`src/nalr/runtime/controller.py`](/Users/fantasylee/类脑架构/src/nalr/runtime/controller.py) 新增 `fault_guard_status()`，并把 `fault_guard` 挂到 `runtime_status_truth_payload()` 与 `/models/status`；[`src/nalr/skills/registry.py`](/Users/fantasylee/类脑架构/src/nalr/skills/registry.py) 补齐 `replace_failed_agent_with_baseline` / `rollback_invalid_sigma` / `switch_to_light_cache_mode` 的 contract skill | `PYTHONPATH=src:. pytest tests/unit/test_runtime_controller.py -k 'fault_guard or model_status_surfaces_fault_guard_contract_and_checkpoint_relationship or fault_guard_skill_registry'` `2 passed`；`PYTHONPATH=src:. pytest tests/integration/test_observer_api.py -k 'exposes_model_status'` `1 passed` | runtime truth 和 observer model status 都能直接看到 `heartbeat_check`、`replace_failed_agent_with_baseline`、`rollback_invalid_sigma`、`checkpoint_contract`，且 `process_replace=false` 明确暴露 | `rollback_invalid_sigma` 明确绑定 checkpoint/create + rewind，`replace` 只给出 baseline / neutral delta contract，不声称真实进程替换 |
| dashboard 首屏不能再用“状态同步中”掩盖真实运行态；必须明确未附着 / 已附着 / stalled | `met` | [`services/observer/dashboard/index.html`](/Users/fantasylee/类脑架构/services/observer/dashboard/index.html) 中 `permissionLabel()` 默认回到 `未附着`；`permissionBadgeText()` / `sessionBadgeText()` / `applyRuntimeBootstrap()` / `refreshConsole()` / `bootstrap()` 全部改为先吃 `/web/runtime/bootstrap` 再补重型 console | `pytest tests/integration/test_observer_api.py -q` `60 passed`；`pytest tests/browser/test_observer_dashboard_smoke.py -q` `1 passed` | 隔离实例启动后，未手动附着前 `bootstrap_before.session_attached=false`；显式调用 `web/runtime/start` 后 `bootstrap_after.session_attached=true` | “打开 dashboard 不偷偷改写 runtime” 仍成立，启动动作仍由显式入口触发 |
| observer / workbench / launcher 必须使用和仓库一致的 Python 环境，优先 `.venv/bin/python` | `met` | [`src/nalr/observer_launcher.py`](/Users/fantasylee/类脑架构/src/nalr/observer_launcher.py) 调整为优先 `.venv/bin/python`，并且保留 entrypoint 路径；[`alive-observer`](/Users/fantasylee/类脑架构/alive-observer) shell 包装同样先探测 repo `.venv`；[`Open_NALR_Workbench.command`](/Users/fantasylee/类脑架构/Open_NALR_Workbench.command) 统一走 `alive-observer` | `pytest tests/integration/test_observer_launcher.py -q` `12 passed` | 隔离实例 `8875` 的 `status` 记录 `python_bin=/Users/fantasylee/类脑架构/.venv/bin/python` | 当前已经在跑的 `8766` 实例仍显示全局 `python3.11`，那是旧进程未重启，不是当前代码的选择结果 |
| 1.5 要有长期在线证据，不能只做短时真相对齐 | `partial` | 当前只有 heartbeat / stalled contract 和 launcher 收敛，没有新增 24h soak 产物 | 本轮未执行 24h soak | 只做了短时隔离实例 live-ish 验收 | 与 [`PLAN1.6.md`](/Users/fantasylee/类脑架构/PLAN1.6.md) 一致，`24h 运行证据` 仍然是 in-flight 项 |

## Live-ish Verification Snapshot

本轮使用隔离 service dir 和隔离 project root 启动一份 observer，避免影响当前用户已经打开的 `8766` 实例。

执行方式：

- `NALR_OBSERVER_SERVICE_DIR=/tmp/... ./alive-observer start --port 8875 --project-root /tmp/.../project --config-root /Users/fantasylee/类脑架构/config --no-browser`
- 依次请求：
  - `/service/status`
  - `/autonomy/status`
  - `/web/runtime/bootstrap`
  - `POST /web/runtime/start`
  - 再次请求上述状态接口

关键结果：

- 启动记录的 `python_bin` 为仓库 `.venv/bin/python`
- 首次 `service/status` 读取时：
  - `service_before.healthy=true`
  - `service_before.autonomy_runner_alive=false`
- 轻量 `autonomy/status` / `web/runtime/bootstrap` 读取后：
  - runtime 若已启用，会附着 heartbeat runner
  - `bootstrap_before.session_attached=false`
- 显式开始运行后：
  - `service_after.autonomy_runner_alive=true`
  - `autonomy_after.heartbeat_state=running`
  - `autonomy_after.last_step_at` 已推进
  - `bootstrap_after.session_attached=true`
- 说明当前 contract 已经能区分：
  - 服务在线但未附着
  - runner 已附着并在推进
  - stalled 字段可直接暴露，而不是继续伪装成“同步中”

## Current Caveats

- 当前正在运行的 `http://127.0.0.1:8766/dashboard` 实例是旧进程，`./alive-observer status` 仍显示全局 `python3.11`，且它的 `/service/status` 还没有新加的 stall 字段。要观察本轮修复效果，需要重启该实例。
- browser smoke 原先把 `8765` 写死，容易被本机已有进程占端口干扰。本轮已把 smoke server/test 改为动态端口，避免把环境冲突误判成 observer 回归。
- `archive/backup-docs/root/PLANS.md` 当前没有旧版 `Doing` 清单，而是 1.5 ledger 结构；本审计按现行 ledger 解释“已交付 / in-flight”，没有再回写旧格式状态。
