# Think Like Human (TLH) v1.2 — NALR Front Door

NALR 的当前主线是 TLH v1.2。这个版本将关键事实源和状态治理集中在四份文档：`README.md`、`Think_Like_Human(TLH)_v1.0.md`、`BEREAL_v0.6_MASTER_SPEC.md`（legacy/frozen substrate）、`PLANS.md`。

## 当前航标
- **Active line:** TLH v1.2（具体状态见 `PLANS.md`）。
- **核心宣言:** 不再让系统像“被规则修饰的概率系统”，而是尽可能变成“由身体状态、主观感受、自发冲动、自造意义与高维概率坍缩共同形成行为的连续主体”。TLH v1.2 进一步把主体状态升级为向量优先的连续表征。详见 `Think_Like_Human(TLH)_v1.0.md` 开头。

## 运行入口
- `./NALR`：普通用户入口，启动认知控制台。
- `./alive`：开发、运维、观测、回放、维护入口。
- `./Open_NALR_Workbench.command`：web-first 工作台入口，启动 observer dashboard，并支持透传 `nalr.observer_launcher` 参数。

## 工作台交付物
- [`Open_NALR_Workbench.command`](/Users/fantasylee/类脑架构/Open_NALR_Workbench.command)：面向本地使用的 workbench 启动脚本，可直接打开 `/dashboard`，也可透传 `--no-browser`、`--port` 等参数。
- [`NALR_Alive_Console_Product_Design.md`](/Users/fantasylee/类脑架构/NALR_Alive_Console_Product_Design.md)：Alive Console / Web Console 的正式产品设计稿。
- [`NALR_Alive_Console_Product_Design.pdf`](/Users/fantasylee/类脑架构/NALR_Alive_Console_Product_Design.pdf)：产品设计稿导出版。
- [`alive trace why latest.json`](/Users/fantasylee/类脑架构/alive%20trace%20why%20latest.json)：最近一轮 explainability / replay 示例产物，可用于 observer 与回放验证。

## 前台文档地图
- [`README.md`](/Users/fantasylee/类脑架构/README.md)：当前页面，负责导航与状态概览。
- [`Think_Like_Human(TLH)_v1.0.md`](/Users/fantasylee/类脑架构/Think_Like_Human(TLH)_v1.0.md)：当前 TLH manifesto 文件；内容已升级到 TLH v1.2，文件名暂保留以兼容现有链接。
- [`BEREAL_v0.6_MASTER_SPEC.md`](/Users/fantasylee/类脑架构/BEREAL_v0.6_MASTER_SPEC.md)：冻结的 `bereal v0.6` 技术基线，对本地底层契约做参考。
- [`PLANS.md`](/Users/fantasylee/类脑架构/PLANS.md)：唯一的 `done / doing / not-done` 状态真相面，附验证证据。

## 观测与验证
- 所有验证数据、测试结果与 acceptance 记录都写在 `PLANS.md`。
- TLH v1.2 的 acceptance gate 由 `PLANS.md` 中的 `TLH-` 记录控制，README 不再重复细节。

## 历史与归档
- 历史版本、旧规格和背景文档统一归档在 [`archive/backup-docs/INDEX.md`](/Users/fantasylee/类脑架构/archive/backup-docs/INDEX.md)。
