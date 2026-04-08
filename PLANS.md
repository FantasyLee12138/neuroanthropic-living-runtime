# NALR 1.5 Ledger

此文件是 NALR `1.5` 的唯一 ledger。只有满足“文档声明存在 + 用户可达 + 有验证证据”的能力，才计入 1.5 完成率。

## Delivered in 1.5

- **TLH runtime baseline**
  四层概率真相面、TLH 主体状态、instinct field、向量优先主体坍缩、why / why-not / replay / observer 基线已存在并有测试证据。
- **Workbench runtime**
  `Open_NALR_Workbench.command`、`alive-observer`、observer dashboard、browser smoke、web session SSE 与 service status 已打通。
- **Autonomy heartbeat foundations**
  autonomy / initiative / monologue 已进入后台持续运行主线，并已有 observer API、launcher 和 diagnostics 回归。
- **Controlled learning policy surface**
  1.5 已将 `learning_mode`、网络 allowlist、可写目录、知识目录、学习日志目录、外部学习 trace 开关并入 observer settings 与 runtime autonomy policy。
- **Terminal-facing controlled learning**
  terminal `sidebar_snapshot` / `/state` 现在能直接看到 controlled-learning 边界，和 observer settings、acceptance-report 对齐。
- **Acceptance artifacts**
  1.5 acceptance plan、matrix、local client / self-hosted deployment 文档已建立。
- **Bounded long-run acceptance report**
  `nalr eval acceptance-report` 已提供 terminal-facing 的 release acceptance report 入口，并补齐了 [`docs/testing/2026-04-08-self-hosted-equivalent-longrun.md`](/Users/fantasylee/类脑架构/docs/testing/2026-04-08-self-hosted-equivalent-longrun.md) 这份等价长跑报告。

## In Flight Before 1.5 Freeze

- **Canonical read-model 收敛**
  terminal / dashboard / observer API 仍需进一步收敛到同一读模型，继续清理第二真相面。
- **Route taxonomy closeout**
  `chat_fast / chat_standard / chat_deep / endogenous_* / dream_sleep` 仍需全部固化到配置、UI 和文档。
- **24h 运行证据**
  已补齐 bounded 等价长跑 acceptance report，但正式 24h soak 仍未执行；仍需补跑一次真实长时运行验收，覆盖 service status、autonomy、initiative、monologue。

## Deferred After 1.5

- 多实例 / 多用户服务化
- 计划性学习任务和定时学习编排
- 更深入的桌面安装器与跨平台分发
- 超出 1.5 可用性目标的新机制扩张

## Verification Evidence

### Runtime / TLH

- `pytest tests/unit/test_tlh_runtime.py -q`
- `pytest tests/unit/test_probability_field.py -q`
- `pytest tests/unit/test_runtime_controller.py -q`
- `pytest tests/unit/test_trace_store.py -q`
- `pytest tests/longrun/test_authenticity_acceptance.py -q`

### Workbench / Observer / Launcher

- `PYTHONPATH='/Users/fantasylee/类脑架构:/Users/fantasylee/类脑架构/src' pytest tests/integration/test_observer_api.py -q`
- `PYTHONPATH='/Users/fantasylee/类脑架构:/Users/fantasylee/类脑架构/src' pytest tests/integration/test_observer_launcher.py -q`
- `PYTHONPATH='/Users/fantasylee/类脑架构:/Users/fantasylee/类脑架构/src' pytest tests/browser/test_observer_dashboard_smoke.py -q`
- `zsh /Users/fantasylee/类脑架构/Open_NALR_Workbench.command --help`

### Terminal / CLI

- `cd /Users/fantasylee/类脑架构/apps/terminal && npm test`
- `cd /Users/fantasylee/类脑架构/apps/terminal && npm run build`
- `pytest tests/integration/test_cli.py -q`
- `pytest tests/integration/test_cil_cli.py -q`

### 1.5 Policy / Docs

- `pytest tests/unit/test_release_docs.py -q`
- `pytest tests/unit/test_release_acceptance_report.py -q`
- `PYTHONPATH='/Users/fantasylee/类脑架构:/Users/fantasylee/类脑架构/src' pytest tests/integration/test_observer_learning_settings.py -q`
- `PYTHONPATH='/Users/fantasylee/类脑架构:/Users/fantasylee/类脑架构/src' pytest tests/unit/test_local_client_packager.py tests/unit/test_acceptance_15_runner.py tests/unit/test_terminal_bridge.py tests/integration/test_terminal_bridge_stdio.py -q`
- `python3 /Users/fantasylee/类脑架构/scripts/acceptance_15.py --run --group controlled-learning`
- `python3 /Users/fantasylee/类脑架构/scripts/acceptance_15.py --run --group self-hosted`

## Feature Matrix Summary

- **P0**
  本地客户端入口、服务可达、state / why / why-not / initiative / monologue、单实例服务状态观测。
- **P1**
  TLH 一致性、trace / replay / diagnostics、一致的权限模型与 controlled learning policy surface。
- **P2**
  长时 soak、分发增强、持续学习策略增强。

## Archive

- 历史文档与旧版本资料见 [`archive/backup-docs/INDEX.md`](/Users/fantasylee/类脑架构/archive/backup-docs/INDEX.md)。
