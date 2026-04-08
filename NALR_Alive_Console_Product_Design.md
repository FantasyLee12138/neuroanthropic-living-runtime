# NALR Alive Console 产品设计文档（v0.57 执行级版）

（完整版，已补充细节，可直接用于实现与PRD）

---

## 一、产品定位

Alive Console 是一个“可观察的概率认知系统控制台”。

它不是聊天UI，而是一个：

> 持续存在、可被观察、可被解释的认知系统界面

---

## 二、核心目标

用户必须在 30 秒内理解：

1. 系统在持续运行（无输入也变化）
2. 行为来自内部状态
3. 可以解释为什么

---

## 三、整体结构

三栏布局：

- 左：脑态系统
- 中：认知流（时间轴）
- 右：解释系统
- 下：输入控制

---

## 四、左栏：脑态系统

### 1. 全局脑态

- 模式（interactive / endogenous）
- 活力
- 自我连续性
- 真实性压力
- 长期漂移风险

### 2. 神经调制

- 多巴胺（探索）
- 去甲肾上腺素（注意）
- 血清素（抑制）
- 乙酰胆碱（选择）
- GABA（局部抑制）

### 3. 动机池

- 记忆探索
- 行为探索
- 情绪调节
- 关系校准
- 内部重整

---

## 五、中栏：认知流

### Action Field

展示当前行动分布：

- top actions
- winner
- conflict

### Token Field

- 表达倾向
- 输出门控
- token完整性

### Timeline

示例：

- 外部刺激进入
- 记忆激活
- 动机上升
- 行动仲裁

---

## 六、右栏：解释系统

### Why

展示当前决策原因

### Why-Not

展示未选择路径

### Contribution

模块贡献叠加

---

## 七、底部控制

支持：

- 输入
- /why
- /state
- /trace
- 触发内生tick

---

## 八、API设计

### 状态

GET /console/state  
GET /console/action-field  

### 时间轴

GET /console/timeline  

### 解释

GET /console/why/current  
GET /console/why-not/{action}  

### 控制

POST /console/talk  
POST /console/endogenous/tick  

---

## 九、组件结构

AliveConsole  
├── BrainStatePanel  
├── CognitiveStream  
├── ExplanationPanel  
└── InputDock  

---

## 十、实现路径

### Phase1

- 状态 + action field + why

### Phase2

- timeline + why-not

### Phase3

- 内生tick + 动机池

### Phase4

- 长期状态 + dream

---

## 十一、核心原则

> 不是聊天系统，而是“活系统”

