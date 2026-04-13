# Workbench UI 与分析工作台重构设计

## 目标

把当前 `/dashboard` 从“单页堆叠观察面板”重构成更适合人类阅读的 Workbench，先回答两类问题：

- 系统现在是不是在线、有没有 stalled、session / autonomy / model / round 是否健康
- 这一轮为什么这么做、为什么没那样做、证据来自哪些 probability / contribution / model route

## 结构

新的 Workbench 采用双层信息架构：

- `运行总览`
  聚焦 service、autonomy、session、current round、model tiers/bindings、controlled learning 与最近轮次。
- `分析工作台`
  用三轨布局组织 round 阅读：
  - 左轨：最近轮次与过滤器
  - 中轨：why / why-not / timeline / 最终表达
  - 右轨：probability / contribution / model route / source links / raw JSON
- `对话`
  保留轻量聊天入口，但不再承担主分析任务。

## 数据来源

优先复用现有 observer API，不重造命令协议：

- 顶部健康条与 overview 首屏：`/web/runtime/bootstrap`、`/service/status`、`/autonomy/status`、`/models/status`
- overview 补充状态：`/state`
- analysis 主读模型：`/console/refresh`
- analysis 深读：`/trace/{round_id}`、`/why/{round_id}`、`/contributions/{round_id}`、`/why-not/{round_id}/{action}`、`/console/probability-space?round_ref=...`
- initiative 补充：`/initiative/status`、`/initiative/distribution`、`/initiative/why`
- chat：`/console/talk`

## 实现策略

- 新增独立前端工程：`apps/workbench`
- Workbench 采用模块化原生 ES module 前端，而不是继续向 `services/observer/dashboard/index.html` 堆脚本
- observer 服务通过 `/dashboard-static/*` 托管新前端产物，`/dashboard` 直接返回新的 Workbench 入口
- 旧的 `services/observer/dashboard/index.html` 保留为 legacy fallback，不主动删除

## 阅读策略

- 默认展示解释后的中文摘要，不直接把 raw JSON 暴露给用户
- raw JSON 只进入右侧 `专家入口 / 原始 JSON`
- 页面上的跳转与筛选替代原先“命令面板优先”的阅读方式
- terminal 继续保留专家命令入口，但 Workbench 成为默认阅读界面

## 内在空间正文约束

- `思考 / 回忆 / 梦境` 这类正文卡片属于 read model，不是新的生成层
- 标题、cue、token 可以做人类可读化，但正文优先保留运行时已经给出的 `reason / last_content / summary / identity_summary`
- 当正文里出现技术性自由文本、英文摘要、key/value 片段或 schema 风格短句时，Workbench 可以做轻量排版，但不能把它们整体改写成统一抒情模板
- 只有在确实没有自由文本正文时，才允许退回到空态说明；空态说明应明确表示“暂无正文”，而不是伪装成已有内容的解释
- 这套约束同样适用于后续的 dream / monologue / initiative 衍生阅读面：解释层可以覆盖在证据之上，但不能替换证据语义
