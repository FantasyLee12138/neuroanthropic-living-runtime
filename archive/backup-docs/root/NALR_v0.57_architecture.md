# NALR v0.57 概率化类脑架构整合方案

## 核心原则
所有模块不再串行决策，而是共同写入输出概率分布。

公式：
log p(y_t) ∝ log p_LLM + Σ Δ_i + attention_bias + state_bias

---

## 模块映射（类脑）

- 丘脑 Thalamus → 惊奇度驱动注意力路由
- 前额叶 PFC → 长期目标先验 + 抑制机制
- 海马体 Hippocampus → 模式分离 + 记忆再激活
- 冲突监测 → 多峰分布仲裁
- 自我 Identity → 稳定贝叶斯先验
- 梦境 Dream → 离线先验重构
- 真实性 Authenticity → 自我失真惩罚
- 生命性 Vitality → 全局调制器

---

## 新运行时主链

1. 状态编码
2. 丘脑注意力路由
3. 候选生成（多模块并行）
4. PFC + Identity 重打分
5. 冲突仲裁
6. Guard 软约束
7. 概率融合采样
8. Trace 记录

---

## 数据结构

class ProbabilisticContribution:
    module_name
    delta_logits
    attention_bias
    soft_mask
    hard_mask
    confidence
    trace_reason

---

## 性能结论

✔ 更轻：减少串行链路  
✔ 更快：并行计算 Δ  
✔ 更稳：统一概率融合  

---

## 关键结论

系统不再是流程图，而是概率场。


###一、总改造原则：从“模块串行”改成“分布共写”
我建议把 NALR 的交互主链改成下面这个统一接口：
log
⁡
p
(
y
t
)
∝
log
⁡
p
LLM
(
y
t
∣
x
,
m
)
+
∑
i
λ
i
Δ
i
(
y
t
)
+
M
attn
(
x
)
+
B
state
(
t
)
logp(y 
t
​	
 )∝logp 
LLM
​	
 (y 
t
​	
 ∣x,m)+ 
i
∑
​	
 λ 
i
​	
 Δ 
i
​	
 (y 
t
​	
 )+M 
attn
​	
 (x)+B 
state
​	
 (t)
其中：
p_LLM：底层模型原始 token 分布
Δ_i(y_t)：每个脑区模块对 token/action 的对数偏置
M_attn(x)：对上下文片段的注意力重加权
B_state(t)：当前生理、身份、关系、资源等慢变量形成的状态偏置
换句话说，模块统一输出四类信号：
context attention bias：改上下文哪些片段更该被看见
candidate/action logits bias：改候选动作或 token 的概率
hard/soft suppression mask：对某些 token 置零或压低
posterior trace：记录该模块怎样改变了分布
这一步最重要，因为 README 里虽然已经有很多概率、冲突、随机性、身份、dream 指标，但从描述看，很多模块还更像“决策链上的站点”，不是“共同写分布的脑区网络”。你的目标应该是让所有显性/隐性模块都只做一件事：向共享概率场提交自己的贡献。
二、模块级整合映射：把现有模块改成“真类脑概率部件”
1）丘脑 ThalamusAttentionAgent：从“聚合与采样”升级为“惊奇度门控 + 熵路由”
README 里当前把 ThalamusAttentionAgent 定位为聚合与采样，而且交互轮在 ConflictMonitor 后进入 Thalamus sampling。我建议把它改成上下文级概率路由器，不要只在末端采样。
具体改法：
输入：当前轮所有上下文块 c_i
为每个块计算：
surprisal_i：相对于当前目标的条件意外度
novelty_i：与近期工作记忆的差异
redundancy_i：与已激活证据的重复度
task_relevance_i：对当前 intent/goal 的条件相关性
生成：
attn_bias_i = a*surprisal + b*novelty + c*relevance - d*redundancy
然后把 attn_bias_i 直接注入 transformer 的 KV 路由或 RAG 重排权重，而不是简单筛选文本段。也就是：
高惊奇、高相关片段：提升注意力温度
高重复、低信息密度片段：做 soft mask
对“高概率套话来源片段”：降低被继续引用的权重
这才接近你说的“丘脑不是过滤，而是概率熵注意力”。它不负责“有没有”，而负责“谁更能改写下一步分布”。
2）前额叶 PFCAgent：从“候选动作生成”升级为“目标先验 + 延迟满足抑制器”
README 里 PFCAgent 现在负责候选动作生成。这个定位太弱，仍然像 planner。真正应该让 PFC 变成长期目标约束下的分布编辑器。
改法：
PFCAgent 不只出候选，还维护一个多维目标先验：
goal_prior
safety_prior
identity_consistency_prior
relationship_cost_prior
long_horizon_value_prior
对底层模型给出的 top-k token / action / tool call 做价值重打分：
Q_pfc(token | goal,state,identity,longrun)
然后输出：
Δ_pfc = α * Q_pfc - β * impulse_score
这里的 impulse_score 可以来自 Desire、情绪峰值、短期讨好倾向、过度披露倾向。
最终不是“PFC 选一个”，而是PFC 压低一批、抬高一批。
这能把“自由意志感”落在工程上：不是模型想说什么就说什么，而是高层先验持续改写输出概率。
3）海马体 Hippocampus：从“检索模块”升级为“模式分离 + 记忆再激活偏置”
README 里交互链里已有 Hippocampus，慢变量也已有 memory，sleep/dream 会塑形 memory。这个基础非常好。下一步不要把它停留在“检索到文本后塞 prompt”。
我建议改成三层：
编码层
记忆单元存 embedding + timestamp + context tags + affect valence + self relevance
模式分离层
相似记忆进入不同簇，避免“同主题互相污染”
每条记忆增加 separation_id / episode_id
再激活层
检索返回的不是文本块，而是 memory prior vector
这个向量直接生成 Δ_hippo，作用到 token 或 action logits
也就是说，海马对生成的贡献应是：
提升与情境匹配的叙述路径概率
抑制“记忆串台”
对高置信 episodic memory 施加更强先验
对模糊记忆只给弱 bias，不直接写死事实
这就从“外挂回忆”变成“参与生成的再激活网络”。
4）冲突监测 ConflictMonitorAgent：从“规则闭环”升级为“多峰分布仲裁器”
README 已经说明冲突层支持 5 个冲突分项、优先级裁决、重采样、妥协模板、修复状态机。你现在可以把这套机制进一步神经化为：监测候选分布是否多峰、是否存在互斥峰值团簇。
改法：
在 action/token 候选空间上做聚类
若出现两个以上高质量但互斥的峰：
计算峰间 KL / JS divergence
计算每个峰的长期价值、身份一致性、社会代价
输出：
peak_suppression_mask
winner_peak_posterior
compromise_template_prior（仅在允许折中时）
这样，ConflictMonitor 不再是“检测到矛盾就调用模板”，而是一个真正的概率互斥解决器。
5）自我/身份 IdentityRuntime：从“身份演化模块”升级为“固定贝叶斯先验核”
README 里最近已经把身份与披露从静态字段升级为概率化 posterior，还拆出了 IdentityRuntime、AuthenticityPolicy、LongRunAnalyzer 等运行时边界。这说明仓库已经在向你想要的方向走。
下一步建议是把 IdentityRuntime 定义成全局不轻易漂移的 prior kernel：
P
(
y
∣
x
,
m
,
s
e
l
f
)
∝
P
LLM
(
y
∣
x
,
m
)
⋅
P
self
(
y
)
⋅
P
role
(
y
)
⋅
P
belief
(
y
)
P(y∣x,m,self)∝P 
LLM
​	
 (y∣x,m)⋅P 
self
​	
 (y)⋅P 
role
​	
 (y)⋅P 
belief
​	
 (y)
工程上可以拆成：
trait_prior：人格稳定维度
belief_prior：信念与世界模型偏置
style_prior：表达风格，不是重点
boundary_prior：不披露/不越界/不失真边界
continuity_prior：与最近一段 identity trajectory 的连贯性
关键是：
identity 不应该只在 prompt 前缀里存在，而应在每轮 token 采样时都给 logits bias。
这样用户的一轮强诱导不会轻易把“我是谁”冲掉，因为那不是后缀提示，而是内生先验。
6）梦境 / DreamOrchestrator：从“离线总结”升级为“生成式重放 + 先验再校准”
README 已有 idle/sleep -> VitalityEngine -> DreamOrchestrator -> memory/habit/relation/identity shaping 这条链，而且已有 dream trace、guard、effect summary。这个位置非常对。
但你可以把 dream 明确定义为三阶段：
重放阶段：抽样近期 episodic memory / unresolved conflicts / salient relation events
补全阶段：让模型做“若继续发展，会怎样”的无监督 completion
校准阶段：
强化高复现模式
弱化孤立噪声记忆
更新 identity_prior、habit_bias、relation_expectation
也就是 dream 不直接生成对外输出，而是更新以下内部参数：
memory salience weights
habit transition priors
identity continuity priors
future expectation priors
这相当于你说的“离线优化内部先验”，而不是做总结日志。
7）真实性 AuthenticityPolicy：从“打分器”升级为“自我失真惩罚项”
README 表示 observer 已能暴露 authenticity 证据，且 AuthenticityRecord 新增了 candidate penalties 与 sampling penalty。这个很适合继续往“概率惩罚”推进。
建议：
定义 inauthenticity_penalty(token/action)
来源包括：
迎合性过强
自我叙事突变
不符合长期风格/信念
对关系情境过拟合
直接作为：
Δ_auth = - γ * inauthenticity_penalty
这样 authenticity 不再是事后解释，而是事前抑制。
8）生命性 VitalityEngine：从“状态塑形”升级为“认知能量预算器”
README 中 VitalityEngine 已进入 idle/sleep 主链。建议把它直接影响在线生成温度与搜索深度。
例如：
vitality 低：
降低探索温度
缩小检索半径
减少 tool 调用冲动
提高保守回答概率
vitality 高：
允许更丰富联想
增大反事实搜索
提升创造性 token 的先验
这很像脑的 neuromodulator，不是认知内容本身，而是调制全局动力学。
9）Habit / Desire / DMN / Emotion / Body / Relationship / Resource：都不要只做状态字段，要改成“专门的偏置头”
README 交互主链明确列出了这些模块。你完全可以把它们做成多头 bias source。
建议统一成：
EmotionHead：改变情感词、冒险倾向、速度感
BodyHead：影响疲劳、紧迫、负荷相关的表达与行动概率
RelationshipHead：影响披露、安抚、边界、称呼、礼貌策略
ResourceHead：影响是否选择高成本工具/长推理/多步骤执行
HabitHead：提高惯常行为路径概率
DesireHead：提高即时满足或趋近行为概率
DMNHead：提高联想、自我叙述、反思类输出概率
PerspectiveHead：改变视角采样分布
ValueHead：把价值排序转成对候选动作的权重
重点不是它们有没有“意见”，而是它们都必须输出 Δ_head 参与 p_final。
三、你要的“整合修改方案”：建议的新运行时主链
我建议把当前交互主链：
state update -> ... -> PFC -> ... -> ConflictMonitor -> Thalamus -> Guard -> Output
改成下面这个版本：
新交互主链（在线）
State Encoding Layer
Salience / Body / Emotion / Relationship / Resource
输出统一状态向量 z_state
Context Routing Layer（丘脑）
对上下文、检索记忆、工具结果计算 attn_bias
输出 context_posterior
Proposal Layer（底层模型 + DMN/Habit/Desire/Hippocampus/Perspective/Value）
不再只生文字，先产出候选 action/token 分布
每个模块各自给 Δ_i
Executive Control Layer（PFC + Identity + Authenticity + LongRun）
对 proposal 分布重打分
压低冲动型、失真型、短视型峰值
Conflict Arbitration Layer
检测多峰互斥
执行 winner-take-most 或 compromise reweighting
Guard Layer
BehaviorPlausibilityGuard 不做最终拍板，而是生成 guard_penalty
作为 soft penalty 优先，hard block 仅用于红线
Sampling Layer
将所有 Δ 和 mask 融合到 p_final
再采样或解码
随机熵不再依赖外部 QRNG 服务
统一由本机 `QuantumEntropyPool -> MacOSSystemEntropyProvider -> os.urandom()` 提供
trace 中保留 `entropy_ref / entropy_refs_by_node` 用于说明本地系统随机源的批次与字节区间
Trace Layer
写入每个模块的 delta contribution
observer 可显示贡献热图，而不是只显示“谁先后执行过”
这样改以后，README 里的 observer/why/counterfactual 体系会更强，因为它不只是解释流程，而是可以解释“这个 token 为什么涨了 0.18，这个 action 为什么被压掉 0.42”。README 本来就已经强调 why、贡献拆解、counterfactual replay、entropy 健康等可观测能力，所以这个升级和仓库方向是高度兼容的。


建议的数据结构改造class ProbabilisticContribution(BaseModel):
    module_name: str
    level: Literal["context", "memory", "action", "token", "global"]
    delta_logits: list[float] | None
    delta_energy: list[float] | None
    attention_bias: list[float] | None
    soft_mask: list[float] | None
    hard_mask: list[bool] | None
    posterior: dict[str, float] | None
    confidence: float
    trace_reason: str
    然后每个 Agent 都实现
    def contribute(runtime_state, candidates, memory_bank, context_chunks) -> ProbabilisticContribution

    最后由一个新的融合器：
    ProbabilityFieldIntegrator.integrate(...)

    产出：
context_attn_final
action_logits_final
token_logits_final
winner_posterior
counterfactual_top_peaks
这比继续让每个模块暴露不同风格的字段更适合后续扩展。
五、对 README 现有模块的逐个落地建议
RuntimeController
保留为编排层，但不要再让它直接含业务判断。它负责：
调度模块
收集 ProbabilisticContribution
调用融合器
输出 trace
这和 README 已将 controller 收敛为编排层的方向一致。
SkillExecutor
把 skill 也变成概率贡献者之一。现在它主要做类型契约、权限、降级、熔断。建议再增加一个：
tool_affordance_prior
tool_cost_penalty
tool_expected_value
让“是否调工具”不再是硬分叉，而是参与 action 分布。README 已强调 timeout_ms / cost_class / failure_policy / policy_check / degradation，这些都能直接转成 action penalty。
BehaviorPlausibilityGuard
建议由纯守卫改成双层：
soft guard：常态参与 logits 惩罚
hard guard：违规时强行置零
这样更类脑，因为大脑大多数时候不是“闸门一刀切”，而是先弱抑制，再在边界场景强抑制。
OutputGate / Renderer
保持末端，但只负责：
表层语言渲染
风格映射
结构化输出格式化
不要让 renderer 再去偷偷改决策内容，否则会稀释前面概率链的解释性。
六、推荐的最小实施顺序
你不需要一次全改。按仓库现状，建议四步走：
Phase 1：把“记录分布”升级成“模块共写分布”
先把现有的 u_base -> ... -> p_final 扩成：
delta_from_pfc
delta_from_conflict
delta_from_identity
delta_from_memory
delta_from_guard
delta_from_thalamus_context
先不碰模型内部注意力，只做候选动作和 token top-k 重加权。
Phase 2：让丘脑和海马接管上下文/记忆路由
Thalamus 负责上下文块 attention_bias
Hippocampus 负责记忆簇 memory_prior
两者都改成前置路由，不再只在中后段出现
Phase 3：把 dream / identity / authenticity 接到先验更新
dream 更新 memory salience、habit prior、identity prior
authenticity 输出在线 penalty
identity 变成全局 prior kernel
Phase 4：让 observer 显示“概率贡献图”
README 已有 observer、why、timeline、heatmap。下一步直接增加：
每轮 top 模块贡献排行
峰值切换轨迹
某 token/action 的多模块 stacked contribution
被压制候选的“死亡原因”
这一步完成后，整个系统会从“像认知架构”变成“真正在概率层运行的类脑系统”。
七、我给你的最终整合版脑区映射表
你可以直接拿去写设计文档：
丘脑 Thalamus
= 概率熵注意力 / 惊奇度路由 / 上下文门控
前额叶 PFC
= 长期目标先验 / 冲动抑制 / 价值重打分
海马体 Hippocampus
= 模式分离 / 情境索引 / 记忆再激活偏置
扣带/冲突监测 ConflictMonitor
= 多峰互斥检测 / 峰值仲裁 / 决策轨迹输出
自我 IdentityRuntime
= 稳定贝叶斯先验核 / 连续自我约束
真实性 AuthenticityPolicy
= 自我失真惩罚项 / 迎合抑制项
生命性 VitalityEngine
= 全局神经调制器 / 认知能量预算
习惯 Habit
= 默认转移先验
欲望 Desire
= 即时趋近驱动
DMN
= 自发联想与自我叙述偏置
情绪 Emotion
= 情感极性与表达强度调制
关系 Relationship
= 社会距离 / 披露边界 / 安抚策略偏置
资源 Resource
= 成本感知 / 工具调用代价先验
梦境 DreamOrchestrator
= 离线生成式重放 / 噪声削弱 / 先验重整
八、最关键的一句话
你的改造目标可以浓缩成一句：
让 NALR 中每一个“心理模块”都不再只是流程节点，而是一个会向共享输出分布写入 bias / mask / prior / posterior 的概率器官。
这会比“模块越来越多”更重要，因为只有这样，它们才真的像脑，而不是像一串加了脑区名字的中间件。
