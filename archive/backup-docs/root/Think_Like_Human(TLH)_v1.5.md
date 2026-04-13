# Think Like Human (TLH) v1.5

## 定义

TLH v1.5 是 NALR 当前前台主线。它延续 TLH v1.2 的主体化方向，但把重点从“机制扩张”切到“可运行、可打开、可长期在线、可验收”的真实交付。

TLH v1.5 不是对 `BEREAL v0.6` 的推翻，而是在保留以下底线的前提下，把系统推进到真正可用的 1.5 状态：

- 单一概率真相面 `context -> memory -> action -> token`
- field-first controller
- token source integrity
- trace / why / why-not / replay / observer 可解释性
- safe mode、approval、memory write gate、rollback 能力

## 1.5 主张

TLH v1.5 的核心不是再加一个“更像人”的 sidecar，而是让以下东西一起成立：

- 主体状态继续是一等行为来源
- 终端、workbench、observer、launcher 说的是同一套状态真相
- 系统能以本地客户端形态被直接打开
- 系统能以单实例服务端形态长期运行
- 外部学习能力必须走 trace、policy、budget、safe mode、approval，而不是旁路脚本

## 1.5 相比 v1.2 的新增交付重点

- **交付形态前移**
  workbench 与 launcher 不再只是开发辅助，而是正式用户入口。
- **服务连续性前移**
  autonomy / initiative / monologue 要以持续运行能力成立，而不是只在读取接口时补算。
- **验收成为主线**
  所有能力都必须进入 acceptance program 和 feature matrix。
- **受控自主学习**
  网络访问和文件夹操作可以开放，但只能在 allowlist、可回滚、可观测的边界内开放。

## 受控自主学习原则

- 默认学习模式是 `guided-learn`
- 允许联网读取，但必须使用 allowlist 域名和可观测来源
- 允许在指定工作目录、`.alive`、学习/缓存目录内读写
- 学习产物必须与 canonical runtime state 隔离
- 所有外部学习行为必须留下 trace：来源、用途、写入、影响、回滚信息

## 1.5 达成条件

TLH v1.5 只有在以下条件同时成立时才算完成：

- `README`、`TLH`、`架构宣言`、`PLANS` 四文档体系统一
- local client 能双击打开并拉起 workbench
- self-hosted 单实例服务可以稳定运行并提供状态探测
- autonomy / initiative / monologue 在 observer 与 runtime 中保持一致
- acceptance matrix 中 P0 / P1 条目达到发布门槛
- 受控自主学习 policy surface 进入 observer settings、runtime state 和文档体系
