# Analysis Trend Chinese Labels And Animated Timeline Design

**Date:** 2026-04-11

## Goal

将分析面板中的 `u_base`、`p_final`、`strength` 改成用户可理解的中文主名，并把三条指标曲线、冲突点、修复点统一到同一时间轴上，同时增加默认轻动画和可切换的回放模式。

## Current State

当前 Workbench 已具备三部分基础能力：

1. `apps/workbench/src/view-models.js`
   已经从 round / trace 中抽取 `u_base`、`p_final`、`strength`，并生成冲突点、修复点 marker。

2. `apps/workbench/src/analysis-brainflow.js`
   已经能渲染三条 SVG 趋势线、round 轴上的红蓝标记、脑流阶段卡片。

3. `apps/workbench/src/main.js` 与 `apps/workbench/src/index.html`
   已经把趋势统计、图例、趋势图、独立时间轴文本块串起来。

当前问题不在数据缺失，而在表达层：

- 指标名仍暴露内部英文符号，用户难以直接理解。
- 来源说明仍偏技术路径，而不是中文解释。
- 冲突 / 修复虽然有标记，但与时间轴叙事分离，理解成本高。
- 动画层很弱，无法形成“变化中”与“回放”两种观看方式。

## Naming Strategy

采用“中文主名 + 英文副标”的命名方式，满足可读性与技术可追溯性。

### Three Metrics

1. `基础驱动力 u_base`
   含义：动作在进入后续裁决前的原始推动力。

2. `最终胜出概率 p_final`
   含义：动作经过概率场裁决后，最终成为赢家的概率。

3. `习惯牵引强度 strength`
   含义：记忆 / 习惯系统对当前动作的牵引力度。

### Source Copy

来源文本不再直接显示英文路径，而改为中文解释：

- `基础驱动力`
  来源：动作原始驱动力

- `最终胜出概率`
  来源：概率场最终裁决

- `习惯牵引强度`
  来源：习惯 / 记忆牵引

保留英文副标只出现在图例、统计卡主标题、hover 细节中，不再在“来源”位置出现原始链路路径。

## UX Design

### 1. Trend Panel

保留现有“参数时序曲线”面板结构，不重做组件骨架。

改动如下：

- 标题副文案改成中文解释，不再直接列出裸露的 `u_base、p_final、strength`。
- 右上角 pill 改为中文主名汇总，例如：`基础驱动力 / 最终胜出概率 / 习惯牵引强度`
- 图例中的每个系列改成：
  - 主行：中文主名
  - 副行：英文副标

### 2. Unified Timeline

趋势图下方或同卡片内补一层“统一时间轴”表达，而不是把冲突点与修复点仅作为旁边的静态说明。

统一时间轴包含：

- 三条指标曲线
- round 轴
- 冲突点 marker
- 修复点 marker
- 当前选中 round 的游标

交互规则：

- hover / 点击某个 round 时：
  - 三条曲线在该 round 的点一起高亮
  - 同 round 的冲突点 / 修复点同步高亮
  - 下方时间轴说明块滚动到对应事件

- 如果某一轮同时存在冲突与修复：
  - 冲突点显示在时间轴的上层轨道
  - 修复点显示在时间轴的下层轨道
  - 避免两者重叠成一个点

### 3. Marker Semantics

保留现有红 / 蓝语义，但用中文直觉化文案。

- 红色：冲突点
  例如：
  - `冲突 · 身份校验`
  - `冲突升高`
  - `冲突 · 输出守卫`

- 蓝色：修复点
  例如：
  - `修复 · 重新采样`
  - `修复 · 回退输出`
  - `修复 · 进入恢复阶段`

时间轴详情面板中，marker 文案优先显示中文摘要，原始 reason 作为次级说明。

## Animation Design

用户已确认需要“两者都要”：默认轻动画 + 回放模式。

### A. Default Motion

默认进入面板时使用轻动画：

- 趋势线更新时做平滑 path 过渡
- 点位做轻微 scale / opacity 变化
- 冲突点红色脉冲
- 修复点蓝色回弹
- 当前选中 round 的游标缓动移动

目标是让用户感知“参数在持续变化”，但不造成监控面板眩晕。

### B. Replay Mode

增加一个回放控制区：

- `回放`
- `暂停`
- `继续`
- `重置`

回放规则：

- 以 round 为步进单位推进
- 每步先移动时间游标，再依次高亮三条曲线在该 round 的点
- 如果该 round 有冲突 / 修复事件，则让 marker 出现并触发一次强调动画
- 时间轴详情自动切换到当前 round 的事件摘要

### C. Reduced Motion

必须兼容现有 `prefers-reduced-motion` 逻辑：

- reduced 模式下取消 pulse / scale / path tween
- 保留静态切换、高亮和回放的离散步进
- 不让新动画破坏当前的 `data-motion="reduced"` 分支

## Architecture

### 1. View Model Layer

文件：`apps/workbench/src/view-models.js`

负责：

- 将 `u_base`、`p_final`、`strength` 的 label / subtitle / sourceText 统一封装
- 将冲突点 / 修复点整理成统一时间轴事件模型
- 为回放模式提供按 round 顺序可消费的数据结构

新增或调整的数据字段建议：

- `displayLabel`
- `displaySubLabel`
- `displaySource`
- `timelineTrack`
- `timelineMarkers`
- `replayFrames`

原则：

- 不改 trace 原始字段名
- 只在 view model 层做“翻译”和“展示结构重排”

### 2. Render Layer

文件：`apps/workbench/src/analysis-brainflow.js`

负责：

- 趋势图图例中文化
- 统计卡中文化
- SVG trend 点位、marker、游标动画
- 回放游标和逐步高亮渲染

原则：

- 复用现有 SVG 结构，不重写成 canvas
- 保持当前三条曲线配色语义
- 新动画尽量通过 class / data-state 驱动，而不是把时序硬写进 innerHTML 模板

### 3. Interaction Layer

文件：`apps/workbench/src/main.js`

负责：

- 维护回放状态
- 处理播放 / 暂停 / 重置
- 处理 hover / click 的 round 联动
- 将“当前选中 round”与“当前回放 round”统一协调

状态建议：

- `analysisReplayEnabled`
- `analysisReplayPlaying`
- `analysisReplayRoundIndex`
- `analysisHoveredRoundId`

### 4. Markup Layer

文件：`apps/workbench/src/index.html`

负责：

- 增加回放控制区容器
- 调整趋势面板的中文副文案
- 为统一时间轴和播放控件预留挂载点

### 5. Style Layer

文件：`apps/workbench/src/styles.css`

负责：

- 中文图例和副标样式
- marker 双轨道样式
- 轻动画
- 回放游标样式
- reduced motion 回退

## Data Flow

数据流保持现有结构，只增加展示层转换：

1. `trace` / `recent_rounds` / `roundCatalog`
2. `buildAnalysisTrendSeries()` 生成三条系列基础点位
3. `buildConflictRepairMarkers()` 生成冲突 / 修复 marker
4. 新增统一 view model，把指标标签、来源解释、marker 事件、回放 frame 串起来
5. `main.js` 根据当前交互状态决定：
   - 正常静态查看
   - hover 联动
   - 回放推进
6. `analysis-brainflow.js` 渲染最终图形和动画 class

## Error Handling

### Missing Metric Values

当某轮缺少某项指标值时：

- 该点不参与折线绘制
- hover 卡片显示 `--`
- 时间轴仍保留该 round 的冲突 / 修复事件

### Missing Conflict / Repair Data

当没有冲突 / 修复数据时：

- 不显示对应 marker
- 时间轴轨道仍保留布局，不压缩整体图形高度

### Replay Edge Cases

- 没有 round 数据：禁用回放按钮
- 只有 1 个 round：允许“播放”，但立即停在唯一帧
- 用户手动点击某一 round 时：暂停自动回放，并把当前帧切到该 round

## Testing Strategy

### Unit / View Model Tests

- 验证三项中文命名映射正确
- 验证 source 文案为中文解释而非英文路径
- 验证冲突 / 修复 marker 会挂到统一时间轴事件模型
- 验证回放 frame 顺序与 round 顺序一致

### Render Tests

- 验证图例显示“中文主名 + 英文副标”
- 验证趋势统计卡显示中文来源
- 验证存在冲突 / 修复时会生成对应 class / data-state

### Interaction Tests

- 点击回放后 round 游标前进
- 点击暂停后停止推进
- 点击某个 round 时，游标与详情联动
- reduced motion 模式下不注入持续脉冲类

## Out Of Scope

本次不做：

- 重写底层 trace API
- 将趋势图从 SVG 改成 canvas / WebGL
- 引入第三方图表库
- 改动“完整脑流”卡片的核心结构

## Recommendation

采用“中等改动，复用现有趋势图骨架”的方案：

- 成本可控
- 风险低于重做播放器
- 能完整覆盖中文命名、同轴标记、默认轻动画、回放模式四个目标

这是当前需求下最稳妥的实现路径。
