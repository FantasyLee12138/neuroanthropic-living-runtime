# NALR `bereal` v0.6 Implementation Plan

此文件是当前唯一执行计划文档。

## Goal

一次性完成以下收束：

- 根目录前台文档只保留 3 份
- 背景文档统一归档
- `bereal / v0.6` 成为当前主线版本
- 内生动机体系接入 field-first 主链
- 小中大模型分档逻辑重排
- 概率场有效性测试补齐

## Current Status

- [x] 当前主线版本字符串已更新为 `0.6.0`
- [x] root 前台文档已收束为：
  - [`README.md`](/Users/fantasylee/类脑架构/README.md)
  - [`BEREAL_v0.6_MASTER_SPEC.md`](/Users/fantasylee/类脑架构/BEREAL_v0.6_MASTER_SPEC.md)
  - [`PLANS.md`](/Users/fantasylee/类脑架构/PLANS.md)
- [x] 历史背景文件、开发文档、机制文档已移入 [`archive/backup-docs`](/Users/fantasylee/类脑架构/archive/backup-docs)
- [x] `docs/` 下旧规格、架构说明、决策记录和历史设计稿已移入 [`archive/backup-docs/docs`](/Users/fantasylee/类脑架构/archive/backup-docs/docs)
- [x] 已新增：
  - `EndogenousMotivationPool`
  - `MotivationFeedbackUpdater`
  - `EndogenousTickScheduler`
- [x] `run_endogenous_tick()`：内生 tick 触发入口，已改为复用主链执行
- [x] `motivation_metrics()`：动机指标汇总接口，已加入 observer
- [x] `endogenous_metrics()`：内生调度指标汇总接口，已加入 observer
- [x] `why_motivation()`：解释内生动机来源的接口，已加入 observer
- [x] `replay_motivation()`：回放内生动机证据的接口，已加入 observer
- [x] `PFCAgent` 默认档位已调整为 `medium`
- [x] `Renderer` 默认档位保留为 `large`
- [x] motivation/endogenous trace/export 字段已补齐
- [x] 首轮关键回归已通过
- [x] 更大范围回归已通过
- [x] 前台三文档函数注释规则与链接完整性已校验
- [x] 非前台 `docs/` 文档已收束为备份，不再作为当前入口
- [x] observer 熵指标重复路由已去重
- [x] 最终综合回归已通过

## Key Decisions

- `_tick_impl()`：主回合执行入口，继续保持唯一决策真相面
- `run_endogenous_tick()`：只负责内生触发，不再保留 sidecar sampler
- `ModelRouter`：继续作为正式模型路由真相面
- `ModelGateway`：只保留兼容角色，不继续承担当前主线调度职责
- 英文函数中文说明规则只覆盖前台文档，不覆盖备份文档

## Verification

已通过的关键回归：

```bash
PYTHONPATH='/Users/fantasylee/类脑架构' .venv/bin/pytest \
  tests/unit/test_runtime_controller.py \
  tests/unit/test_probability_field.py \
  tests/unit/test_trace_store.py \
  tests/integration/test_observer_api.py -q
```

结果：

- `126 passed`

继续补跑的扩大回归：

```bash
PYTHONPATH='/Users/fantasylee/类脑架构' .venv/bin/pytest \
  /Users/fantasylee/类脑架构/tests/unit/test_longrun_acceptance.py \
  /Users/fantasylee/类脑架构/tests/unit/test_registry_contracts.py \
  /Users/fantasylee/类脑架构/tests/unit/test_trace_store.py \
  /Users/fantasylee/类脑架构/tests/integration/test_cli.py -q
```

结果：

- `35 passed in 12.72s`

最终综合回归：

```bash
PYTHONPATH='/Users/fantasylee/类脑架构' .venv/bin/pytest \
  /Users/fantasylee/类脑架构/tests/unit/test_runtime_controller.py \
  /Users/fantasylee/类脑架构/tests/unit/test_probability_field.py \
  /Users/fantasylee/类脑架构/tests/unit/test_trace_store.py \
  /Users/fantasylee/类脑架构/tests/unit/test_longrun_acceptance.py \
  /Users/fantasylee/类脑架构/tests/unit/test_registry_contracts.py \
  /Users/fantasylee/类脑架构/tests/integration/test_observer_api.py \
  /Users/fantasylee/类脑架构/tests/integration/test_cli.py -q
```

结果：

- `156 passed in 59.55s`

文档静态校验：

- 前台三文档里的英文函数名均带中文说明
- `README.md`、`BEREAL_v0.6_MASTER_SPEC.md`、`PLANS.md` 与备份索引中的本地链接均可解析
- root 当前前台主文档保持为 3 份，旧背景文档已迁入 [`archive/backup-docs/INDEX.md`](/Users/fantasylee/类脑架构/archive/backup-docs/INDEX.md)
- `docs/` 当前不再保留 Markdown 规格入口，旧内容已迁入 [`archive/backup-docs/docs`](/Users/fantasylee/类脑架构/archive/backup-docs/docs)

## Remaining Checks

- [x] 更大范围回归已完成：
  - `tests/unit/test_longrun_acceptance.py`
  - `tests/unit/test_registry_contracts.py`
  - `tests/unit/test_trace_store.py`
  - `tests/integration/test_cli.py`
- [x] 本轮扩大回归未暴露新的阻塞问题

## Archive

备份索引见 [`archive/backup-docs/INDEX.md`](/Users/fantasylee/类脑架构/archive/backup-docs/INDEX.md)。
