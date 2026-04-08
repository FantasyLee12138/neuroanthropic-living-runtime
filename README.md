# NALR 1.5 Front Door

NALR 的当前公开主线是 `1.5`。这个版本的目标不是继续堆概念，而是把 TLH runtime、observer/workbench、terminal、启动器、服务端常驻和验收体系收敛成一个可运行、可打开、可长期在线的整体。

## 1.5 航标

- **主线版本**：`1.5`
- **运行理念**：继续坚持 TLH 的单一概率真相面、trace-first、safe mode、approval、memory write gate，不因为交付形态而退化成普通 agent 壳。
- **交付定义**：
  - 本地可双击打开的客户端入口
  - 可 24h 运行的单实例服务端
  - 一套有文档、有矩阵、有证据的 acceptance program

## 使用入口

- `./Open_NALR_Workbench.command`
  当前 1.5 的本地客户端入口。双击后拉起本地 observer/runtime 服务并打开 workbench。
- `./alive-observer start`
  当前 1.5 的本地或远端服务启动入口。
- `./NALR`
  终端控制台入口。
- `./alive`
  开发、观测、维护和回放入口。

## 1.5 文档地图

- [`README.md`](/Users/fantasylee/类脑架构/README.md)
  当前页面，只负责导航和交付入口。
- [`Think_Like_Human(TLH)_v1.5.md`](/Users/fantasylee/类脑架构/Think_Like_Human(TLH)_v1.5.md)
  当前 TLH manifesto。
- [`架构宣言.md`](/Users/fantasylee/类脑架构/架构宣言.md)
  当前 1.5 架构 doctrine。
- [`PLANS.md`](/Users/fantasylee/类脑架构/PLANS.md)
  当前 1.5 ledger 与验证入口。
- [`BEREAL_v0.6_MASTER_SPEC.md`](/Users/fantasylee/类脑架构/BEREAL_v0.6_MASTER_SPEC.md)
  冻结的 substrate 文档。

## 验收与发布

- [`docs/releases/1.5-acceptance-plan.md`](/Users/fantasylee/类脑架构/docs/releases/1.5-acceptance-plan.md)
  1.5 的正式验收总计划。
- [`docs/releases/1.5-acceptance-matrix.md`](/Users/fantasylee/类脑架构/docs/releases/1.5-acceptance-matrix.md)
  功能矩阵、状态和验证入口。
- [`docs/deployment/1.5-local-client.md`](/Users/fantasylee/类脑架构/docs/deployment/1.5-local-client.md)
  本地客户端交付说明。
- [`docs/deployment/1.5-self-hosted.md`](/Users/fantasylee/类脑架构/docs/deployment/1.5-self-hosted.md)
  单实例自托管部署说明。
- [`scripts/acceptance_15.py`](/Users/fantasylee/类脑架构/scripts/acceptance_15.py)
  可执行的 1.5 acceptance runner，可打印命令计划或直接执行分组验收。
- [`scripts/local_client_packager.py`](/Users/fantasylee/类脑架构/scripts/local_client_packager.py)
  当前 bundle-contained 本地客户端的 `.app` materializer，会把 launcher、observer 资源、config 与可复用 runtime 一起打进 bundle。

## 当前交付物

- [`Open_NALR_Workbench.command`](/Users/fantasylee/类脑架构/Open_NALR_Workbench.command)
- [`alive-observer`](/Users/fantasylee/类脑架构/alive-observer)
- [`NALR_Alive_Console_Product_Design.md`](/Users/fantasylee/类脑架构/NALR_Alive_Console_Product_Design.md)
- [`NALR_Alive_Console_Product_Design.pdf`](/Users/fantasylee/类脑架构/NALR_Alive_Console_Product_Design.pdf)
- [`alive trace why latest.json`](/Users/fantasylee/类脑架构/alive%20trace%20why%20latest.json)

## 历史与归档

- 历史版本和旧规格见 [`archive/backup-docs/INDEX.md`](/Users/fantasylee/类脑架构/archive/backup-docs/INDEX.md)。
