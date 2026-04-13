# NALR

NALR 当前主线是 `1.6` / `Chat Kernel V2`。默认聊天不再走旧的 `full_tick` 全量多 agent 仪式，而是改成稀疏默认、异常加深的路由内核：`chat_micro`、`chat_fast`、`chat_standard`、`chat_deep`。

根目录只保留三份活文档：[`README.md`](/Users/fantasylee/类脑架构/README.md)、[`架构宣言.md`](/Users/fantasylee/类脑架构/架构宣言.md)、[`PLAN1.6.md`](/Users/fantasylee/类脑架构/PLAN1.6.md)。其余历史根文档已经归档到 [`archive/backup-docs/root`](</Users/fantasylee/类脑架构/archive/backup-docs/root>)。

## Overview

`1.6` 的目标是把聊天内核、memory tier、observer、terminal、workbench 和 trace 合同收口到同一套 V2 语义，而不是继续维护 agent-first 的旧壳。

当前公开语义已经切到：

- route-first: `chat_micro | chat_fast | chat_standard | chat_deep`
- module-first: `module_model_bindings`、`activation_set`、`memory_tiers_read`
- packet-first: 普通回合以 `CognitivePacket` 为单次主模型产物
- memory-first: `hot / warm / cold`
- background consolidation: 长摘要、compaction、topic clustering 不阻塞首 token

## What Shipped

- 默认普通聊天已经切到 `chat_fast` + 单次 packet 主路径。
- `chat_standard` / `chat_deep` 已按显著性、记忆、工具、冲突和 deepen reason 升级。
- observer `/models/status`、dashboard、terminal summary、workbench 概览已切到 module / route / activation 视图。
- memory 公共语义已从 `archive` 收束到 `cold`，并保留迁移兼容。
- 根目录历史文档已归档，README / 计划 / 架构宣言已同步到当前进度。

## Entry Points

- [`Open_NALR_Workbench.command`](/Users/fantasylee/类脑架构/Open_NALR_Workbench.command): 本地 Workbench 入口
- [`alive-observer`](/Users/fantasylee/类脑架构/alive-observer): observer / runtime 服务入口
- [`NALR`](/Users/fantasylee/类脑架构/NALR): 终端控制台入口
- [`alive`](/Users/fantasylee/类脑架构/alive): 开发、观测、维护、回放入口

## Verification

这轮迁移里已直接回放过的关键验证包括：

- `PYTHONPATH=. pytest tests/integration/test_cil_cli.py -q -k 'monologue or snapshot_restore'`
- `PYTHONPATH=. pytest tests/unit/test_memory_store.py -q`
- `PYTHONPATH=. pytest tests/unit/test_diagnostics_runtime.py tests/integration/test_cil_cli.py tests/integration/test_dream_bridge_stdio.py tests/unit/test_runtime_controller_facades.py tests/unit/test_async_io_runtime.py -q -k 'not dashboard'`
- `PYTHONPATH=. pytest tests/integration/test_observer_api.py -q -k 'observer_exposes_model_status or workbench_overview_snapshot_surfaces_persona_mood_from_runtime_state or observer_settings_can_override_newborn_unlocks_and_local_model'`
- `node --test apps/workbench/src/view-models.test.js`
- `npm --prefix apps/terminal test -- --run src/panelSummary.test.ts src/state/sessionStore.test.ts`

## Docs

- [`PLAN1.6.md`](/Users/fantasylee/类脑架构/PLAN1.6.md): 当前交付、剩余工作、引用
- [`架构宣言.md`](/Users/fantasylee/类脑架构/架构宣言.md): 当前架构立场
- [`docs/releases/1.6-acceptance-plan.md`](/Users/fantasylee/类脑架构/docs/releases/1.6-acceptance-plan.md): 验收计划
- [`docs/releases/1.6-acceptance-matrix.md`](/Users/fantasylee/类脑架构/docs/releases/1.6-acceptance-matrix.md): 验收矩阵
- [`docs/releases/1.6-release-notes.md`](/Users/fantasylee/类脑架构/docs/releases/1.6-release-notes.md): release notes
- [`archive/backup-docs/INDEX.md`](/Users/fantasylee/类脑架构/archive/backup-docs/INDEX.md): 历史文档索引

## Current Gaps

- 真实 24h soak 和更长程 long-run 证据还没补齐。
- 单实例服务硬化、更多治理策略和更深的恢复对齐仍在后续批次。
