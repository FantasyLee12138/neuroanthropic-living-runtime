# NALR `bereal` v0.6

NeuroAnthropic Living Runtime（NALR）当前主线版本代号为 `bereal`，主线版本号统一为 `v0.6`。
当前前台文档只保留 3 个文件：

- [`README.md`](/Users/fantasylee/类脑架构/README.md)：入口说明
- [`BEREAL_v0.6_MASTER_SPEC.md`](/Users/fantasylee/类脑架构/BEREAL_v0.6_MASTER_SPEC.md)：唯一主规范
- [`PLANS.md`](/Users/fantasylee/类脑架构/PLANS.md)：当前执行计划与验证记录

其余开发文档、机制设计、历史规格和中间计划已移入备份目录 [`archive/backup-docs/INDEX.md`](/Users/fantasylee/类脑架构/archive/backup-docs/INDEX.md)。

## 项目入口

NALR 当前不再把旧版 `v0.56 / v0.57` 文档当作并列事实源。
当前唯一事实源是 [`BEREAL_v0.6_MASTER_SPEC.md`](/Users/fantasylee/类脑架构/BEREAL_v0.6_MASTER_SPEC.md)，当前执行进度与验证记录以 [`PLANS.md`](/Users/fantasylee/类脑架构/PLANS.md) 为准。

## 当前状态

`bereal / v0.6` 当前主线已经收口到：

- 保持 `context / memory / action / token` 四层概率场为唯一决策真相面
- 将内生动机、反馈闭环和内生调度接入 field-first 主链
- 重设计小中大模型使用逻辑，在保证准确率的前提下提升效率
- 用结构化测试保证概率场输出真实有效、可解释、可回放

## 运行入口

- `./NALR`：普通用户入口，启动认知控制台
- `./alive`：开发、运维、观测、回放、维护入口

## 三文件说明

- [`README.md`](/Users/fantasylee/类脑架构/README.md)：项目入口、当前状态、运行与备份说明
- [`BEREAL_v0.6_MASTER_SPEC.md`](/Users/fantasylee/类脑架构/BEREAL_v0.6_MASTER_SPEC.md)：唯一主规范
- [`PLANS.md`](/Users/fantasylee/类脑架构/PLANS.md)：执行计划、决策日志与验证结果

## 验证入口

当前验证记录统一写在 [`PLANS.md`](/Users/fantasylee/类脑架构/PLANS.md)。
快速查看时优先看：

- `tests/unit/test_runtime_controller.py`
- `tests/unit/test_probability_field.py`
- `tests/unit/test_trace_store.py`
- `tests/integration/test_observer_api.py`
- `tests/integration/test_cli.py`

如需确认当前观测状态，可直接查看：

- `/models/status`
- `/metrics/summary`
- `/metrics/motivation`
- `/metrics/endogenous`
- `/why/{round_id}`
- `/why-motivation/{round_id}`

## 历史资料

历史开发文档、机制文档、旧版本规格、验收矩阵和迁移材料统一存放在：

- [`archive/backup-docs/INDEX.md`](/Users/fantasylee/类脑架构/archive/backup-docs/INDEX.md)

`docs/` 下的旧规格、架构说明、决策记录和历史设计稿也已一并归档。
这些文件仍保留原版本号和原始文件名，用作溯源与备份，不再承担当前主线事实源角色。
