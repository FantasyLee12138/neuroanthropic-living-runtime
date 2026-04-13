import {
  WORKBENCH_READ_MODEL_PATH,
  buildWorkbenchRoundPath,
  enqueueWorkbenchUserTurn,
  loadWorkbenchReadModelEnvelope,
  loadWorkbenchRoundCatalogData,
  loadWorkbenchRoundData,
  pollWorkbenchSessionEventsOnce,
  postJson,
} from "./api.js";
import {
  renderBrainflowOutput,
  renderBrainflowStages,
  renderTrendLegend,
  renderTrendStats,
  renderTrendSvg,
} from "./analysis-brainflow.js";
import * as analysisBrainflowModule from "./analysis-brainflow.js";
import {
  deriveRuntimeChatName,
  formatChatTimestamp,
  resolveChatHydration,
  sanitizeChatMessages,
} from "./chat-state.js";
import {
  compactList,
  escapeHtml,
  formatNumber,
  humanizeRuntimeText,
  humanizeRuntimeToken,
  maybeText,
  routeHealthLabel,
  translateDisplayJson,
} from "./format.js";
import {
  buildAnalysisFlowView,
  buildAnalysisTrendSeries,
  buildBrainflowOutput,
  buildBrainflowStages,
  buildConflictRepairMarkers,
  buildChatTurnView,
  buildRoundInitiativeView,
  deriveRoundInitiativeState,
  buildInnerSpaceView,
  buildOverviewSnapshot,
  buildSettingsConsoleView,
} from "./view-models.js";
import * as viewModelsModule from "./view-models.js";

const FALLBACK_LOCALE = "zh-CN";
const SUPPORTED_LOCALES = new Set(["zh-CN", "en"]);
const CHAT_HISTORY_STORAGE_KEY = "workbench.chat.history";
const TREND_REPLAY_STEP_MS = 760;

const MESSAGES = {
  "zh-CN": {
    "document.title": "NALR 工作台",
    "brand.eyebrow": "运行前台",
    "brand.title": "NALR 运行工作台",
    "brand.subtitle": "把主体、意义、主动性和轮次证据收束到同一个可读前台，先判断状态，再决定往哪里深入。",
    "language.label": "语言",
    "status.connecting": "正在连接运行体…",
    "nav.overview": "运行总览",
    "nav.analysis": "分析阅读流",
    "nav.chat": "对话",
    "health.service": "服务健康",
    "health.autonomy": "自治脉搏",
    "health.session": "会话",
    "health.round": "当前轮次",
    "health.model": "模块摘要",
    "overview.hero.eyebrow": "运行前台",
    "overview.hero.title": "先看运行体是否稳定",
    "runtime.start": "开始运行",
    "runtime.pause": "暂停自治",
    "runtime.resume": "恢复任务",
    "runtime.wake": "回到交互",
    "overview.autonomy.title": "自治脉搏",
    "overview.autonomy.copy": "优先阅读自治状态、动作节奏和稳定性信号。",
    "overview.model.title": "模型与路由",
    "overview.model.copy": "把模块绑定、路由策略和关键配置放在一个视野里。",
    "overview.learning.title": "受控学习",
    "overview.learning.copy": "保持只读可见性，减少操作焦虑。",
    "common.readonly": "只读",
    "overview.learning.mode": "学习模式",
    "overview.learning.trace": "追踪状态",
    "overview.learning.domains": "允许域名",
    "overview.learning.roots": "可写根目录",
    "overview.rounds.title": "人格心情",
    "overview.rounds.copy": "把当前心境、主观感受、边界、自发性与连续性放在一张状态卡里。",
    "overview.rounds.jump": "进入分析工作台",
    "analysis.rounds.title": "最近轮次",
    "analysis.rounds.copy": "按轮次阅读，先筛选，再展开原因链。",
    "analysis.rounds.badge": "按轮次阅读",
    "analysis.filters.round": "轮次编号",
    "analysis.filters.round.placeholder": "426",
    "analysis.filters.action": "动作",
    "analysis.filters.action.all": "全部动作",
    "analysis.filters.cause": "来源",
    "analysis.filters.cause.all": "全部来源",
    "analysis.filters.cause.user": "用户触发",
    "analysis.filters.cause.endogenous": "内源触发",
    "analysis.filters.route": "响应路径",
    "analysis.filters.route.all": "全部路径",
    "analysis.filters.initiative": "主动性",
    "analysis.filters.initiative.all": "全部",
    "analysis.filters.activity": "工具/审批",
    "analysis.filters.activity.all": "全部",
    "analysis.why.title": "为什么这么做",
    "analysis.why.copy": "把最终选择、约束和上下文压缩成适合人读的解释。",
    "analysis.whynot.title": "为什么不是",
    "analysis.whynot.copy": "保留未采纳视角，方便确认被压制的备选路径。",
    "analysis.whynot.badge": "未采纳原因",
    "analysis.timeline.title": "时间线与外显表达",
    "analysis.timeline.copy": "把关键时序、对外表达和状态变化放在同一块阅读面。",
    "analysis.timeline.badge": "时间线",
    "analysis.probability.title": "概率层",
    "analysis.probability.copy": "只在侧栏核对最终胜出概率、信心与能量分布。",
    "analysis.probability.badge": "概率层",
    "analysis.contribution.title": "贡献叠加",
    "analysis.contribution.copy": "把各模块的作用拆开看，避免只见结论。",
    "analysis.contribution.badge": "贡献叠加",
    "analysis.route.title": "V2 路由",
    "analysis.route.copy": "按模块路由和激活路径阅读，减少跳转成本。",
    "analysis.route.badge": "V2 路由",
    "analysis.links.title": "证据面板",
    "analysis.links.copy": "集中展示主动性、门控与轮次详情入口。",
    "analysis.links.badge": "证据与跳转",
    "analysis.raw.title": "专家入口 / 原始数据",
    "chat.title": "对话",
    "chat.copy": "保留轻量输入面，主分析阅读迁移到分析工作台。",
    "chat.intro": "这里保留简洁对话入口，但主分析任务已经转移到分析工作台，不再把命令面板当作主要阅读方式。",
    "chat.input.placeholder": "输入消息，读取当前运行体状态。",
    "chat.send": "发送",
    "chat.pendingReply": "正在整理当前状态…",
    "chat.pendingLongReply": "仍在进行完整一轮计算…",
    "chat.sendBusy": "上一轮对话还在返回，先等这轮落稳。",
    "common.none": "暂无",
    "common.noneRound": "尚无当前轮次",
    "common.notConnected": "未连接",
    "common.notLoaded": "未加载",
    "common.unknown": "未知",
    "common.unknownAction": "未知动作",
    "common.notSelectedRound": "未选择轮次",
    "common.unavailable": "当前不可用",
    "common.clear": "清空",
    "common.open": "开启",
    "common.closed": "关闭",
    "common.pending": "待处理",
    "common.attached": "已附着",
    "common.detached": "未附着",
    "common.running": "运行中",
    "common.stopped": "未运行",
    "common.alive": "存活",
    "common.offline": "离线",
    "common.healthy": "健康",
    "common.degraded": "降级",
    "common.ready": "就绪",
    "common.hold": "保留",
    "common.yes": "是",
    "common.no": "否",
    "common.round": "轮次",
    "common.event": "事件",
    "summary.service": "服务",
    "summary.autonomy": "自治",
    "summary.session": "会话",
    "summary.approvals": "审批",
    "summary.attached": "已附着",
    "summary.detached": "未附着",
    "summary.pendingApprovals": "{count} 个待审批",
    "summary.noApprovals": "无待审批",
    "summary.httpReady": "HTTP 已就绪",
    "summary.httpWarming": "HTTP 预热中",
    "summary.currentRound": "当前轮次 #{roundId}，输出动作 {action}。当前会话 {sessionState}，工具活动 {toolCount}，待审批 {approvalCount}。{autonomyReason}",
    "summary.noCurrentRound": "当前还没有活跃轮次。先判断服务 / 自治 / 会话是否在线；{autonomyReason}",
    "summary.workbenchLoadFailed": "运行前台启动失败：{message}",
    "summary.sessionEstablished": "会话已建立",
    "summary.sessionMissing": "会话未建立",
    "summary.currentlyReading": "主读模型来源",
    "summary.gateToolApproval": "门控 / 工具 / 审批",
    "summary.traceReplayLinks": "轮次详情跳转",
    "summary.sourceLinks": "来源链接",
    "summary.readModels": "当前读模型来源",
    "summary.roundInitiative": "轮次主动性",
    "summary.distribution": "分布判断",
    "summary.why": "原因摘要",
    "summary.mainSource": "主读模型",
    "summary.trace": "过程",
    "summary.initiative": "主动性",
    "summary.replay": "回放",
    "summary.modelHint": "渲染线索",
    "summary.renderHint": "渲染线索",
    "summary.user": "你",
    "summary.assistant": "运行体",
    "summary.resultReading": "结论阅读",
    "summary.finalExpression": "最终表达",
    "summary.replayPreview": "回放预览",
    "summary.routeNarrative": "当前路径 {route}，输出 {action}，表达方式 {delivery}。",
    "summary.pendingApproval": "待审批",
    "summary.sessionTool": "会话工具",
    "summary.traceTool": "过程工具",
    "summary.blockedBy": "被拦截于",
    "summary.currentRoundTag": "轮次 #{roundId} · {action}",
    "summary.deliveryUnknown": "未知",
    "summary.latestAction": "最近动作",
    "summary.heartbeat": "心跳",
    "summary.runner": "自治线程",
    "summary.lastAction": "最近动作",
    "summary.budget": "预算",
    "summary.serviceHealth": "服务",
    "summary.approvalsLabel": "审批",
    "summary.stallReason": "停滞原因",
    "summary.candidatePeak": "候选峰值",
    "summary.learningTraceOn": "开启",
    "summary.learningTraceOff": "关闭",
    "summary.keyPresent": "密钥已接入",
    "summary.keyMissing": "部分缺失",
    "summary.localRules": "本地规则",
    "summary.noWhy": "当前没有原因摘要。",
    "summary.noWhyNot": "当前没有未采纳摘要。",
    "summary.noExpressionSummary": "当前没有外显表达摘要。",
    "summary.noReplayPreview": "暂无回放预览",
    "summary.noTimeline": "当前没有时间线数据。",
    "summary.noProbability": "当前没有概率层数据。",
    "summary.noContributions": "当前没有贡献叠加数据。",
    "summary.noModelCalls": "当前没有模型调用记录。",
    "summary.noSourceLinks": "当前没有来源链接。",
    "summary.noGates": "当前没有显式门控决策。",
    "summary.noToolDetails": "当前接口没有暴露轮次级工具或审批明细。",
    "summary.noDeepLinks": "没有可跳转的轮次。",
    "summary.noRoundSelected": "请选择左侧轮次。",
    "summary.noRoundData": "当前还没有轮次数据。",
    "summary.noSessionAnalysis": "当前会话未附着，暂时没有分析对象。",
    "summary.noRoundsAttached": "当前会话已附着，但还没有可读轮次。",
    "summary.noRoundsDetached": "当前还没有附着会话，也没有可读轮次。",
    "summary.noMatchingRounds": "没有匹配的轮次。请调整左侧过滤条件。",
    "summary.waitingRoundData": "等待轮次数据。",
    "summary.sendMessagePrompt": "发送一条消息，界面会同步刷新当前读模型。",
    "summary.readFailed": "读取轮次 #{roundId} 失败：{message}",
    "summary.talkFailed": "发送失败：{message}",
    "summary.controlNotice": "{notice} / {base}",
    "summary.autonomyReason": "{reason}：{detail}{peak}",
    "summary.currentPeak": "；当前最强倾向仍是 {action}，但它还没有真正执行出来",
    "summary.healthStalled": "停滞 · {seconds}s · {reason}",
    "summary.healthRunning": "{heartbeat} · {running} · {reason}",
    "summary.sessionHealth": "{attachState} · {approvalState}",
    "summary.modelHealth": "认知包 {packet} / 深度表达 {renderer}",
    "runtime.skipped.interactive_turn_active": "前台对话正在进行，当前控制操作已让路。",
    "runtime.skipped.autonomy_runner_busy": "自治步骤仍在推进，控制操作已快速返回。",
    "runtime.skipped.runtime_busy": "运行体正在写入状态，控制操作已快速返回。",
    "routeType.interactive": "交互",
    "routeType.endogenous": "内源",
    "routeType.endogenous_light": "内源轻触发",
    "routeType.endogenous_deep": "内源深读",
    "routeType.endogenous_replay": "内源回放",
    "routeType.external_stimulus": "外部触发",
    "routeType.chat": "聊天",
    "routeType.chat_fast": "快速对话",
    "routeType.chat_standard": "标准对话",
    "routeType.chat_deep": "深度对话",
    "routeType.task": "任务",
    "routeType.task_run": "任务执行",
    "routeType.companion": "陪伴",
    "initiative.initiative": "主动表达",
    "initiative.reactive": "外部响应",
    "initiative.mixed": "混合驱动",
    "initiative.unknown": "未知",
    "activity.approval": "有审批",
    "activity.tool": "有工具线索",
    "activity.gated": "有门控痕迹",
    "activity.quiet": "平稳",
    "stall.running": "自治可推进",
    "stall.observer_turn_active": "会话占用中",
    "stall.self_run_blocked": "只读自运行被阻断",
    "stall.safe_mode_active": "安全模式阻断",
    "stall.budget_exhausted": "预算耗尽",
    "stall.quiet_hours": "静默时段",
    "stall.no_candidate_action": "暂无可执行候选",
    "stall.runner_blocked": "自治线程无进度",
    "stall.idle": "尚未启动",
    "healthStatus.stalled": "停滞",
    "healthStatus.running": "运行中",
    "healthStatus.ready": "就绪",
    "healthStatus.degraded": "降级",
    "heartbeat.running": "运行中",
    "heartbeat.idle": "空闲",
    "heartbeat.stalled": "停滞",
    "actions.respond": "回应",
    "actions.recall": "回忆",
    "actions.plan": "规划",
    "actions.clarify": "澄清",
    "actions.connect": "连接",
    "actions.rest": "休整",
    "actions.wander": "游走",
    "actions.monologue": "独白",
    "actions.nothing": "静止",
    "actions.absorb": "吸收",
    "actions.die": "终止",
    "actions.short_reply": "短答",
    "layer.action": "动作层",
    "layer.token": "词元层",
    "layer.instinct": "本能层",
    "layer.context": "上下文层",
    "gate.block": "阻断",
    "gate.pass": "放行",
    "gate.resample": "重采样",
    "deepLink.trace": "过程",
    "deepLink.why": "原因",
    "deepLink.contribution": "贡献叠加",
    "deepLink.replay": "回放",
    "deepLink.whyNot": "未采纳原因",
    "deepLink.round": "轮次详情",
  },
  en: {
    "document.title": "NALR Workbench",
    "brand.eyebrow": "NALR Workbench",
    "brand.title": "Runtime Overview and Analysis Workbench",
    "brand.subtitle": "Use the live observer APIs to gather service health, decision rationale, and evidence reading into one readable surface.",
    "language.label": "Language",
    "status.connecting": "Connecting to the runtime…",
    "nav.overview": "Overview",
    "nav.analysis": "Analysis Workbench",
    "nav.chat": "Chat",
    "health.service": "Service Health",
    "health.autonomy": "Autonomy Pulse",
    "health.session": "Session",
    "health.round": "Current Round",
    "health.model": "Module Summary",
    "overview.hero.eyebrow": "Overview",
    "overview.hero.title": "System Health",
    "runtime.start": "Start Runtime",
    "runtime.pause": "Pause Autonomy",
    "runtime.resume": "Resume Run",
    "runtime.wake": "Wake to Interactive",
    "overview.autonomy.title": "Autonomy Pulse",
    "overview.autonomy.copy": "Read autonomy state, action tempo, and stability signals first.",
    "overview.model.title": "Models and Routes",
    "overview.model.copy": "Keep module bindings, route policy, and key settings in one view.",
    "overview.learning.title": "Controlled Learning",
    "overview.learning.copy": "Keep it visible and read-only to reduce operational anxiety.",
    "common.readonly": "Read-only",
    "overview.learning.mode": "Learning Mode",
    "overview.learning.trace": "Trace State",
    "overview.learning.domains": "Allowed Domains",
    "overview.learning.roots": "Writable Roots",
    "overview.rounds.title": "Persona Mood",
    "overview.rounds.copy": "Read the current mood, subjective felt markers, boundary, spontaneity, and continuity in one state card.",
    "overview.rounds.jump": "Open Analysis Workbench",
    "analysis.rounds.title": "Recent Rounds",
    "analysis.rounds.copy": "Read by round: filter first, then expand the causal chain.",
    "analysis.rounds.badge": "Read by round",
    "analysis.filters.round": "round_id",
    "analysis.filters.round.placeholder": "426",
    "analysis.filters.action": "Action",
    "analysis.filters.action.all": "All actions",
    "analysis.filters.cause": "Source",
    "analysis.filters.cause.all": "All sources",
    "analysis.filters.cause.user": "User-triggered",
    "analysis.filters.cause.endogenous": "Endogenous",
    "analysis.filters.route": "Route Type",
    "analysis.filters.route.all": "All routes",
    "analysis.filters.initiative": "Initiative",
    "analysis.filters.initiative.all": "All",
    "analysis.filters.activity": "Tools / Approval",
    "analysis.filters.activity.all": "All",
    "analysis.why.title": "Why This",
    "analysis.why.copy": "Compress the final choice, constraints, and context into a narrative meant for human reading.",
    "analysis.whynot.title": "Why Not",
    "analysis.whynot.copy": "Keep the suppressed alternatives visible so the rejected path is still inspectable.",
    "analysis.whynot.badge": "why-not",
    "analysis.timeline.title": "Timeline and Expression",
    "analysis.timeline.copy": "Read timing, outward expression, and state changes in one surface.",
    "analysis.timeline.badge": "timeline",
    "analysis.probability.title": "Probability Layers",
    "analysis.probability.copy": "Show posterior, confidence, and the final energy layout.",
    "analysis.probability.badge": "probability",
    "analysis.contribution.title": "Contribution Stack",
    "analysis.contribution.copy": "Split each module's effect apart so the conclusion is not the only thing you see.",
    "analysis.contribution.badge": "contribution",
    "analysis.route.title": "V2 Routes",
    "analysis.route.copy": "Bring module routing and activation paths into one reading flow.",
    "analysis.route.badge": "V2 route",
    "analysis.links.title": "Evidence Panel",
    "analysis.links.copy": "Collect initiative, gate, and deep links in one place.",
    "analysis.links.badge": "evidence and links",
    "analysis.raw.title": "Expert Entry / Raw JSON",
    "chat.title": "Chat",
    "chat.copy": "Keep a light chat surface here and move the heavy reading work into the analysis page.",
    "chat.intro": "This page keeps a simple chat entry point, while the primary analysis work has moved into the analysis workbench instead of the old command-first view.",
    "chat.input.placeholder": "Type a message and read the current runtime state.",
    "chat.send": "Send",
    "chat.pendingReply": "Working through the current state…",
    "chat.pendingLongReply": "Still finishing a full turn...",
    "chat.sendBusy": "The previous turn is still running.",
    "common.none": "None",
    "common.noneRound": "No current round",
    "common.notConnected": "Not connected",
    "common.notLoaded": "Not loaded",
    "common.unknown": "Unknown",
    "common.unknownAction": "Unknown action",
    "common.notSelectedRound": "No round selected",
    "common.unavailable": "Unavailable",
    "common.clear": "Clear",
    "common.open": "On",
    "common.closed": "Off",
    "common.pending": "Pending",
    "common.attached": "Attached",
    "common.detached": "Detached",
    "common.running": "Running",
    "common.stopped": "Stopped",
    "common.alive": "Alive",
    "common.offline": "Offline",
    "common.healthy": "Healthy",
    "common.degraded": "Degraded",
    "common.ready": "Ready",
    "common.hold": "Hold",
    "common.yes": "Yes",
    "common.no": "No",
    "common.round": "round",
    "common.event": "event",
    "summary.service": "service",
    "summary.autonomy": "autonomy",
    "summary.session": "session",
    "summary.approvals": "approvals",
    "summary.attached": "attached",
    "summary.detached": "detached",
    "summary.pendingApprovals": "{count} pending",
    "summary.noApprovals": "clear",
    "summary.httpReady": "HTTP ready",
    "summary.httpWarming": "HTTP warming",
    "summary.currentRound": "Current round #{roundId}, sampled action {action}. Session is {sessionState}, tool activity {toolCount}, approvals {approvalCount}. {autonomyReason}",
    "summary.noCurrentRound": "There is no active round yet. Check service, autonomy, and session first. {autonomyReason}",
    "summary.workbenchLoadFailed": "Workbench startup failed: {message}",
    "summary.sessionEstablished": "Session ready",
    "summary.sessionMissing": "Session missing",
    "summary.currentlyReading": "Primary read models",
    "summary.gateToolApproval": "Gate / Tool / Approval",
    "summary.traceReplayLinks": "Trace / Replay Links",
    "summary.sourceLinks": "Source Links",
    "summary.readModels": "Primary read models",
    "summary.roundInitiative": "Round initiative",
    "summary.distribution": "Distribution",
    "summary.why": "Why",
    "summary.mainSource": "Primary surface",
    "summary.trace": "Trace",
    "summary.initiative": "Initiative",
    "summary.replay": "Replay",
    "summary.modelHint": "Model hint",
    "summary.renderHint": "Render hint",
    "summary.user": "You",
    "summary.assistant": "Runtime",
    "summary.resultReading": "Read the result",
    "summary.finalExpression": "Final expression",
    "summary.replayPreview": "Replay preview",
    "summary.routeNarrative": "Route {route}, sampled action {action}, delivery {delivery}.",
    "summary.pendingApproval": "Pending approval",
    "summary.sessionTool": "Session tool",
    "summary.traceTool": "Trace tool",
    "summary.blockedBy": "Blocked by",
    "summary.currentRoundTag": "round #{roundId} · {action}",
    "summary.deliveryUnknown": "unknown",
    "summary.latestAction": "Latest action",
    "summary.heartbeat": "heartbeat",
    "summary.runner": "runner",
    "summary.lastAction": "last action",
    "summary.budget": "budget",
    "summary.serviceHealth": "service",
    "summary.approvalsLabel": "approvals",
    "summary.stallReason": "stall reason",
    "summary.candidatePeak": "candidate peak",
    "summary.learningTraceOn": "On",
    "summary.learningTraceOff": "Off",
    "summary.keyPresent": "Credentials present",
    "summary.keyMissing": "Credentials partially missing",
    "summary.localRules": "Local rules",
    "summary.noWhy": "No why summary is available.",
    "summary.noWhyNot": "No why-not summary is available.",
    "summary.noExpressionSummary": "No expression summary is available.",
    "summary.noReplayPreview": "No replay preview yet.",
    "summary.noTimeline": "No timeline data is available.",
    "summary.noProbability": "No probability layers are available.",
    "summary.noContributions": "No contribution stack is available.",
    "summary.noModelCalls": "No model call trace is available.",
    "summary.noSourceLinks": "No source links are available.",
    "summary.noGates": "No explicit gate decisions are available.",
    "summary.noToolDetails": "The current APIs do not expose round-level tool or approval details.",
    "summary.noDeepLinks": "There is no round to link out to.",
    "summary.noRoundSelected": "Select a round from the left rail.",
    "summary.noRoundData": "No round data is available yet.",
    "summary.noSessionAnalysis": "No session is attached yet, so there is nothing to analyze.",
    "summary.noRoundsAttached": "A session is attached, but there are no readable rounds yet.",
    "summary.noRoundsDetached": "No session is attached, and there are no readable rounds yet.",
    "summary.noMatchingRounds": "No rounds match the current filters.",
    "summary.waitingRoundData": "Waiting for round data.",
    "summary.sendMessagePrompt": "Send a message and the page will refresh the console read model.",
    "summary.readFailed": "Failed to read round #{roundId}: {message}",
    "summary.talkFailed": "Send failed: {message}",
    "summary.controlNotice": "{notice} / {base}",
    "summary.autonomyReason": "{reason}: {detail}{peak}",
    "summary.currentPeak": " Current strongest tendency {action}, but it has not executed yet",
    "summary.healthStalled": "stalled · {seconds}s · {reason}",
    "summary.healthRunning": "{heartbeat} · {running} · {reason}",
    "summary.sessionHealth": "{attachState} · {approvalState}",
    "summary.modelHealth": "cognitive packet {packet} / deep renderer {renderer}",
    "runtime.skipped.interactive_turn_active": "A foreground conversation is active, so the control action yielded immediately.",
    "runtime.skipped.autonomy_runner_busy": "An autonomy step is still running, so the control action returned immediately.",
    "runtime.skipped.runtime_busy": "The runtime is committing state, so the control action returned immediately.",
    "routeType.interactive": "Interactive",
    "routeType.endogenous": "Endogenous",
    "routeType.endogenous_light": "Endogenous Light",
    "routeType.endogenous_deep": "Endogenous Deep",
    "routeType.endogenous_replay": "Endogenous Replay",
    "routeType.external_stimulus": "External Stimulus",
    "routeType.chat": "Chat",
    "routeType.chat_fast": "Fast Chat",
    "routeType.chat_standard": "Standard Chat",
    "routeType.chat_deep": "Deep Chat",
    "routeType.task": "Task",
    "routeType.task_run": "Task Run",
    "routeType.companion": "Companion",
    "initiative.initiative": "Proactive",
    "initiative.reactive": "Reactive",
    "initiative.mixed": "Mixed",
    "initiative.unknown": "Unknown",
    "activity.approval": "Approval",
    "activity.tool": "Tool signal",
    "activity.gated": "Gate trace",
    "activity.quiet": "Quiet",
    "stall.running": "Autonomy can still advance",
    "stall.observer_turn_active": "Session is occupied",
    "stall.self_run_blocked": "Read-only self_run is blocked",
    "stall.safe_mode_active": "Blocked by safe mode",
    "stall.budget_exhausted": "Budget exhausted",
    "stall.quiet_hours": "Quiet hours",
    "stall.no_candidate_action": "No actionable candidate",
    "stall.runner_blocked": "Autonomy runner has no progress",
    "stall.idle": "Not started",
    "healthStatus.stalled": "stalled",
    "healthStatus.running": "running",
    "healthStatus.ready": "ready",
    "healthStatus.degraded": "degraded",
    "heartbeat.running": "running",
    "heartbeat.idle": "idle",
    "heartbeat.stalled": "stalled",
    "actions.respond": "Respond",
    "actions.recall": "Recall",
    "actions.plan": "Plan",
    "actions.clarify": "Clarify",
    "actions.connect": "Connect",
    "actions.rest": "Rest",
    "actions.wander": "Wander",
    "actions.monologue": "Monologue",
    "actions.nothing": "Stillness",
    "actions.absorb": "Absorb",
    "actions.die": "Terminate",
    "actions.short_reply": "Short reply",
    "layer.action": "Action layer",
    "layer.token": "Token layer",
    "layer.instinct": "Instinct layer",
    "layer.context": "Context layer",
    "gate.block": "Block",
    "gate.pass": "Pass",
    "gate.resample": "Resample",
    "deepLink.trace": "Trace",
    "deepLink.why": "Why",
    "deepLink.contribution": "Contribution",
    "deepLink.replay": "Replay",
    "deepLink.whyNot": "Why-Not",
    "deepLink.round": "Round Detail",
  },
};

Object.assign(MESSAGES["zh-CN"], {
  "nav.overview": "运行总览",
  "nav.analysis": "分析阅读流",
  "nav.innerSpace": "内在空间",
  "nav.chat": "对话",
  "nav.settings": "设置",
  "chat.reasonSummary": "我为什么这样说",
  "chat.clear": "清空聊天记录",
  "chat.clearConfirm": "这会结束当前聊天会话并清空这页记录。继续吗？",
  "chat.sessionReady": "已载入 {count} 条聊天记录",
  "chat.sessionEmpty": "聊天会话已就绪",
  "chat.sessionCleared": "聊天记录已清空",
  "chat.sessionResetFailed": "清空聊天记录失败：{message}",
  "summary.runtimeRecovered": "检测到运行体停在上次暂停状态，已自动恢复自治",
  "settings.danger.unlock": "确认后解锁危险操作",
  "settings.danger.unlocked": "危险操作已解锁",
  "settings.danger.confirm": "确认短语",
  "settings.danger.confirm.placeholder": "UNLOCK DANGER",
  "settings.danger.notice.locked": "危险区默认锁定。先确认，再允许修改高风险开关与阈值。",
  "settings.danger.notice.unlocked": "本次页面会话内已解锁危险区。请确认修改后再保存。",
  "settings.personaReset": "删除当前人格",
  "settings.personaResetConfirm": "这会重置当前人格、清空会话与相关状态。继续吗？",
  "settings.personaResetDone": "人格已重置",
  "settings.personaResetFailed": "人格重置失败：{message}",
  "scheduledTask.summary.total": "任务总数",
  "scheduledTask.summary.running": "运行中",
  "scheduledTask.summary.nextRun": "最近下一次",
  "scheduledTask.status.idle": "待执行",
  "scheduledTask.status.running": "运行中",
  "scheduledTask.status.disabled": "已停用",
  "scheduledTask.excerpt.prompt": "提示摘录",
  "scheduledTask.excerpt.goal": "目标摘录",
  "scheduledTask.empty": "当前还没有已登记的计划任务。",
  "scheduledTask.skill": "技能",
  "scheduledTask.nextRun": "下一次执行",
  "overview.rounds.selector": "快速选择",
  "overview.rounds.expand": "查看全部",
  "analysis.rounds.selector": "选择轮次",
  "analysis.rounds.expand": "展开全部",
  "analysis.rounds.collapse": "仅看最近三条",
  "settings.title": "设置",
  "settings.copy": "把主体基线、权限边界、受控学习和模型绑定分层管理，不再和阅读视图混在一起。",
  "settings.save": "保存设置",
  "settings.status.idle": "尚未读取",
  "settings.status.saving": "正在保存",
  "settings.status.saved": "已保存",
  "settings.status.failed": "保存失败",
  "settings.status.loaded": "已同步设置",
  "settings.baseline.title": "主体启动与新生基线",
  "settings.baseline.copy": "决定启动时默认的有机模式与主观底色。",
  "settings.boundaries.title": "权限边界与预算",
  "settings.boundaries.copy": "决定自治可走到哪一层，以及什么时候自动回到保护态。",
  "settings.learning.title": "受控学习",
  "settings.learning.copy": "统一管理学习模式、允许域名、知识根目录和学习日志。",
  "settings.models.title": "模型绑定与路由",
    "settings.models.copy": "优先暴露最常改的模块绑定，保持路由调整的可读性。",
  "settings.clear_safe_mode.title": "启动时解除保护锁",
  "settings.clear_safe_mode.copy": "开始运行时自动解除顶层保护态。",
  "settings.instinct_first.title": "新生主体启用本能优先",
  "settings.instinct_first.copy": "把“本能优先”作为默认有机模式。",
  "settings.allow_commit.title": "允许提交",
  "settings.allow_commit.copy": "放开 git commit 级别的自治写入。",
  "settings.network_enabled.title": "允许网络",
  "settings.network_enabled.copy": "记录网络访问边界偏好，供自治策略读取。",
  "settings.external_io_enabled.title": "允许外部 IO",
  "settings.external_io_enabled.copy": "记录外部读写边界偏好。",
  "settings.auto_safe_mode.title": "失败后自动回到安全模式",
  "settings.auto_safe_mode.copy": "触发失败阈值时自动切回保护态。",
  "settings.spontaneous": "初始自发冲动",
  "settings.boundary": "初始边界感",
  "settings.felt": "初始感受提示词",
  "settings.felt.placeholder": "微弱自发冲动、安静但清醒",
  "settings.max_rounds_per_hour": "每小时内生轮次上限",
  "settings.max_tool_actions_per_hour": "每小时工具动作上限",
  "settings.failure_trip_threshold": "失败熔断阈值",
  "settings.quiet_hours": "静默时段",
  "settings.quiet_hours.placeholder": "1,2,3",
  "settings.learning_mode": "学习模式",
  "settings.trace_external_learning": "学习追踪",
  "settings.allowed_domains": "允许域名",
  "settings.allowed_domains.placeholder": "docs.python.org, react.dev",
  "settings.writable_roots": "可写根目录",
  "settings.knowledge_roots": "知识根目录",
  "settings.learning_log_dir": "学习日志目录",
  "settings.binding.renderer": "认知包绑定",
  "settings.binding.planner": "深度推理绑定",
  "settings.binding.pfc": "工具规划绑定",
  "settings.binding.perspective": "深度表达绑定",
  "settings.binding.consolidation": "巩固摘要绑定",
  "settings.models.supported": "支持后端",
  "settings.models.local": "本地层级",
  "summary.quickSelectEmpty": "暂无轮次可选",
  "summary.identityIntro": "我是谁",
  "summary.currentFocus": "我现在最在意的事",
  "summary.details": "展开细节",
  "summary.functionBlocked": "被拦截于",
  "summary.functionLabel": "函数位点",
  "summary.translationLabel": "中文释义",
  "summary.evidenceCards": "证据卡",
  "summary.weightLabel": "强度",
  "summary.timelineDetails": "阶段函数收纳",
  "summary.whyDrivers": "作用主因",
  "summary.whyNotDrivers": "拦截链路",
  "summary.timelineStage": "阶段",
  "summary.sourcePanel": "来源面板",
  "summary.apiPath": "接口路径",
  "summary.controllerMethod": "控制器方法",
  "summary.gateStage": "门控阶段",
  "summary.gateResult": "门控结果",
  "summary.toolAction": "工具动作",
  "summary.approvalAction": "审批动作",
  "summary.timelineEmpty": "当前没有可读时间线。",
  "summary.settingsSaved": "设置已写入运行观察服务",
  "summary.settingsFailed": "设置保存失败：{message}",
  "summary.settingsLoaded": "设置已同步",
  "summary.rangeUnlimited": "0 表示不限",
  "summary.currentSelection": "当前选择",
  "summary.recentThree": "最近三条",
  "summary.expandAllRounds": "查看全部轮次",
  "summary.settingsModelBinding": "当前绑定",
  "function.generate_candidates": "候选生成",
  "function.render_expression": "表达生成",
  "function.infer_other_state": "推测对方状态",
  "function.simulate_other_reaction": "模拟对方反应",
  "function.compute_boundary_gate": "关系边界门控",
  "function.apply_body_veto": "身体否决",
  "function.map_budget_to_bias": "预算映射偏置",
  "function.score_salience": "显著性打分",
  "function.estimate_subjective_value": "主观价值估计",
  "function.bind_working_memory": "绑定工作记忆",
  "function.estimate_plan_depth": "估计规划深度",
  "function.EmergentActionSketch": "涌现行动草图",
  "function.MemoryRecallBias": "记忆召回牵引",
  "function.BodyBudgetGuard": "身体预算守门",
  "function.RelationBoundaryGate": "关系边界门控",
  "function.AffectResidueDrive": "情绪残留驱动",
  "function.GoalContinuationDrive": "目标延续驱动",
  "function.memory": "记忆调取",
  "function.pfc": "前额叶候选生成",
  "function.thalamus": "注意采样",
  "function.identity_guard": "身份校验",
  "function.conflict_repair": "冲突修复",
  "function.output": "输出收口",
  "function.guard": "守卫层",
  "phrase.silent_but_active": "静默但仍活跃",
  "phrase.field_imbalance": "场域还不平衡",
  "phrase.sleep": "睡眠整理",
  "phrase.idle": "空转余波",
});

Object.assign(MESSAGES.en, {
  "nav.overview": "Overview",
  "nav.analysis": "Analysis Workbench",
  "nav.innerSpace": "Inner Space",
  "nav.chat": "Chat",
  "nav.settings": "Settings",
  "chat.reasonSummary": "Why I answered this way",
  "chat.clear": "Clear Chat",
  "chat.clearConfirm": "This ends the current chat session and clears the history shown here. Continue?",
  "chat.sessionReady": "{count} messages loaded",
  "chat.sessionEmpty": "Chat session is ready",
  "chat.sessionCleared": "Chat history cleared",
  "chat.sessionResetFailed": "Failed to clear chat history: {message}",
  "summary.runtimeRecovered": "Recovered the runtime from its previous paused state automatically",
  "settings.danger.unlock": "Unlock Danger Zone",
  "settings.danger.unlocked": "Danger Zone Unlocked",
  "settings.danger.confirm": "Confirmation phrase",
  "settings.danger.confirm.placeholder": "UNLOCK DANGER",
  "settings.danger.notice.locked": "Danger settings stay locked by default. Confirm first before changing high-risk toggles or limits.",
  "settings.danger.notice.unlocked": "Danger settings are unlocked for this page session. Review changes carefully before saving.",
  "settings.personaReset": "Delete Current Persona",
  "settings.personaResetConfirm": "This resets the current persona and clears related session state. Continue?",
  "settings.personaResetDone": "Persona reset completed",
  "settings.personaResetFailed": "Persona reset failed: {message}",
  "scheduledTask.summary.total": "Tasks",
  "scheduledTask.summary.running": "Running",
  "scheduledTask.summary.nextRun": "Next Run",
  "scheduledTask.status.idle": "Queued",
  "scheduledTask.status.running": "Running",
  "scheduledTask.status.disabled": "Disabled",
  "scheduledTask.excerpt.prompt": "Prompt Excerpt",
  "scheduledTask.excerpt.goal": "Goal Excerpt",
  "scheduledTask.empty": "No scheduled tasks are currently registered.",
  "scheduledTask.skill": "Skill",
  "scheduledTask.nextRun": "Next Run",
  "overview.rounds.selector": "Quick Select",
  "overview.rounds.expand": "Show All",
  "analysis.rounds.selector": "Pick a Round",
  "analysis.rounds.expand": "Expand All",
  "analysis.rounds.collapse": "Only Recent Three",
  "settings.title": "Settings",
  "settings.copy": "Manage baseline state, control boundaries, guided learning, and model bindings in layered panels instead of mixing them into reading surfaces.",
  "settings.save": "Save Settings",
  "settings.status.idle": "Not loaded",
  "settings.status.saving": "Saving",
  "settings.status.saved": "Saved",
  "settings.status.failed": "Save failed",
  "settings.status.loaded": "Settings synced",
  "settings.baseline.title": "Boot and Newborn Baseline",
  "settings.baseline.copy": "Choose the default organic mode and subjective baseline applied at startup.",
  "settings.boundaries.title": "Permissions and Budgets",
  "settings.boundaries.copy": "Define how far autonomy can go and when it must fall back to protection mode.",
  "settings.learning.title": "Controlled Learning",
  "settings.learning.copy": "Manage learning mode, allowed domains, knowledge roots, and learning logs in one place.",
  "settings.models.title": "Model Bindings and Routes",
    "settings.models.copy": "Surface the module bindings most likely to change without flooding the page with every route field.",
  "settings.clear_safe_mode.title": "Clear Safe Mode on Start",
  "settings.clear_safe_mode.copy": "Release the top-level lock when runtime start is triggered.",
  "settings.instinct_first.title": "Enable Instinct-First for Newborn State",
  "settings.instinct_first.copy": "Use instinct-first as the default organic mode.",
  "settings.allow_commit.title": "Allow Commit",
  "settings.allow_commit.copy": "Allow autonomy to reach git commit level writes.",
  "settings.network_enabled.title": "Allow Network",
  "settings.network_enabled.copy": "Expose network boundary preference to autonomy policy.",
  "settings.external_io_enabled.title": "Allow External IO",
  "settings.external_io_enabled.copy": "Expose external read/write boundary preference.",
  "settings.auto_safe_mode.title": "Auto-return to Safe Mode on Failures",
  "settings.auto_safe_mode.copy": "Switch back to protection mode after the failure threshold trips.",
  "settings.spontaneous": "Initial Spontaneous Drive",
  "settings.boundary": "Initial Boundary Sense",
  "settings.felt": "Initial Felt Prompts",
  "settings.felt.placeholder": "quiet urge, alert but calm",
  "settings.max_rounds_per_hour": "Max Endogenous Rounds per Hour",
  "settings.max_tool_actions_per_hour": "Max Tool Actions per Hour",
  "settings.failure_trip_threshold": "Failure Trip Threshold",
  "settings.quiet_hours": "Quiet Hours",
  "settings.quiet_hours.placeholder": "1,2,3",
  "settings.learning_mode": "Learning Mode",
  "settings.trace_external_learning": "Trace Learning",
  "settings.allowed_domains": "Allowed Domains",
  "settings.allowed_domains.placeholder": "docs.python.org, react.dev",
  "settings.writable_roots": "Writable Roots",
  "settings.knowledge_roots": "Knowledge Roots",
  "settings.learning_log_dir": "Learning Log Directory",
  "settings.binding.renderer": "Cognitive Packet Binding",
  "settings.binding.planner": "Deliberation Binding",
  "settings.binding.pfc": "Tool Planner Binding",
  "settings.binding.perspective": "Deep Renderer Binding",
  "settings.binding.consolidation": "Consolidation Binding",
  "settings.models.supported": "Supported Backends",
  "settings.models.local": "Local Tier",
  "summary.quickSelectEmpty": "No rounds yet",
  "summary.identityIntro": "Who I am",
  "summary.currentFocus": "Current focus",
  "summary.details": "Details",
  "summary.functionBlocked": "Blocked At",
  "summary.functionLabel": "Function",
  "summary.translationLabel": "Meaning",
  "summary.evidenceCards": "Evidence Cards",
  "summary.weightLabel": "Weight",
  "summary.timelineDetails": "Collapsed Stage Details",
  "summary.whyDrivers": "Main Drivers",
  "summary.whyNotDrivers": "Blocking Chain",
  "summary.timelineStage": "Stage",
  "summary.sourcePanel": "Source Panel",
  "summary.apiPath": "API Path",
  "summary.controllerMethod": "Controller Method",
  "summary.gateStage": "Gate Stage",
  "summary.gateResult": "Gate Result",
  "summary.toolAction": "Tool Action",
  "summary.approvalAction": "Approval Action",
  "summary.timelineEmpty": "No readable timeline is available.",
  "summary.settingsSaved": "Settings written to observer",
  "summary.settingsFailed": "Saving settings failed: {message}",
  "summary.settingsLoaded": "Settings synced",
  "summary.rangeUnlimited": "0 means unlimited",
  "summary.currentSelection": "Current Selection",
  "summary.recentThree": "Recent Three",
  "summary.expandAllRounds": "Show all rounds",
  "summary.settingsModelBinding": "Current Binding",
  "function.generate_candidates": "Candidate generation",
  "function.render_expression": "Expression rendering",
  "function.infer_other_state": "Infer other state",
  "function.simulate_other_reaction": "Simulate other reaction",
  "function.compute_boundary_gate": "Boundary gate",
  "function.apply_body_veto": "Body veto",
  "function.map_budget_to_bias": "Budget-to-bias mapping",
  "function.score_salience": "Salience scoring",
  "function.estimate_subjective_value": "Subjective value estimate",
  "function.bind_working_memory": "Bind working memory",
  "function.estimate_plan_depth": "Estimate plan depth",
  "function.EmergentActionSketch": "Emergent action sketch",
  "function.MemoryRecallBias": "Memory recall bias",
  "function.BodyBudgetGuard": "Body budget guard",
  "function.RelationBoundaryGate": "Relation boundary gate",
  "function.AffectResidueDrive": "Affect residue drive",
  "function.GoalContinuationDrive": "Goal continuation drive",
  "phrase.silent_but_active": "silent but still active",
  "phrase.field_imbalance": "field imbalance",
  "phrase.sleep": "sleep consolidation",
  "phrase.idle": "idle residue",
});

const state = {
  locale: FALLBACK_LOCALE,
  page: "overview-page",
  bootstrap: null,
  sessionState: null,
  chatSessionState: null,
  service: null,
  autonomy: null,
  runtimeState: null,
  models: null,
  console: null,
  subject: null,
  meaning: null,
  agency: null,
  performance: null,
  initiativeStatus: null,
  initiativeDistribution: null,
  selectedRoundId: null,
  selectedRoundAction: "",
  selectedRoundPayload: null,
  pendingRoundHydrationId: null,
  hoveredTrendRoundId: null,
  replayPlaying: false,
  replayRoundIndex: 0,
  replayStepTimer: null,
  replayHasStarted: false,
  trendReplayFrames: [],
  roundCatalog: {},
  roundCatalogLoading: {},
  roundListExpanded: false,
  settings: null,
  innerSpace: {
    thought: null,
    memoryTop: [],
    dreamOverview: null,
    monologueShow: null,
  },
  controlNotice: "",
  filters: {
    roundQuery: "",
    action: "",
    cause: "",
    route: "",
    initiative: "",
    activity: "",
  },
  chatSessionId: "workbench-chat",
  chatMessages: [],
  chatLocalMutationAt: "",
  chatSendInFlight: false,
  dangerZoneUnlocked: false,
  userPausedRuntime: false,
};

const elements = {
  root: document.getElementById("workbench-root"),
  topStatusMini: document.getElementById("top-status-mini"),
  clock: document.getElementById("system-clock"),
  navOverview: document.getElementById("nav-overview"),
  navAnalysis: document.getElementById("nav-analysis"),
  navInnerSpace: document.getElementById("nav-inner-space"),
  navChat: document.getElementById("nav-chat"),
  navSettings: document.getElementById("nav-settings"),
  overviewPage: document.getElementById("overview-page"),
  analysisPage: document.getElementById("analysis-page"),
  innerSpacePage: document.getElementById("inner-space-page"),
  chatPage: document.getElementById("chat-page"),
  settingsPage: document.getElementById("settings-page"),
  healthService: document.getElementById("health-service"),
  healthAutonomy: document.getElementById("health-autonomy"),
  healthSession: document.getElementById("health-session"),
  healthRound: document.getElementById("health-round"),
  healthModel: document.getElementById("health-model"),
  overviewSummary: document.getElementById("overview-summary"),
  overviewFocusCues: document.getElementById("overview-focus-cues"),
  autonomyBadge: document.getElementById("autonomy-badge"),
  autonomySummary: document.getElementById("autonomy-summary"),
  modelBadge: document.getElementById("model-badge"),
  modelSummary: document.getElementById("model-summary"),
  learningMode: document.getElementById("learning-mode"),
  learningTrace: document.getElementById("learning-trace"),
  learningDomains: document.getElementById("learning-domains"),
  learningRoots: document.getElementById("learning-roots"),
  overviewPersonaSummary: document.getElementById("overview-persona-summary"),
  overviewRoundSelect: document.getElementById("overview-round-select"),
  overviewRoundExpand: document.getElementById("overview-round-expand"),
  overviewRounds: document.getElementById("overview-rounds"),
  analysisRoundSelect: document.getElementById("analysis-round-select"),
  analysisRoundToggle: document.getElementById("analysis-round-toggle"),
  analysisRoundList: document.getElementById("analysis-round-list"),
  analysisCortex: document.getElementById("analysis-cortex"),
  analysisLiveCopy: document.getElementById("analysis-live-copy"),
  analysisTrendPanel: document.getElementById("analysis-trend-panel"),
  analysisTrendLegend: document.getElementById("analysis-trend-legend"),
  analysisTrendStats: document.getElementById("analysis-trend-stats"),
  analysisTrendSvg: document.getElementById("analysis-trend-svg"),
  analysisTrendReplayControls: document.getElementById("analysis-trend-replay-controls"),
  analysisTrendFrameStrip: document.getElementById("analysis-trend-frame-strip"),
  analysisBrainflow: document.getElementById("analysis-brainflow"),
  analysisBrainflowOutput: document.getElementById("analysis-brainflow-output"),
  analysisEvidenceDrawers: document.getElementById("analysis-evidence-drawers"),
  analysisWhy: document.getElementById("analysis-why"),
  analysisWhyNot: document.getElementById("analysis-why-not"),
  analysisTimeline: document.getElementById("analysis-timeline"),
  analysisProbability: document.getElementById("analysis-probability"),
  analysisContributions: document.getElementById("analysis-contributions"),
  analysisModelRoute: document.getElementById("analysis-model-route"),
  analysisLinks: document.getElementById("analysis-links"),
  analysisRawJson: document.getElementById("analysis-raw-json"),
  selectedRoundBadge: document.getElementById("selected-round-badge"),
  innerSpaceStatus: document.getElementById("inner-space-status"),
  innerSpaceSelfNarrative: document.getElementById("inner-space-self-narrative"),
  innerSpaceThinking: document.getElementById("inner-space-thinking"),
  innerSpaceMemory: document.getElementById("inner-space-memory"),
  innerSpaceDream: document.getElementById("inner-space-dream"),
  filterRound: document.getElementById("filter-round"),
  filterAction: document.getElementById("filter-action"),
  filterCause: document.getElementById("filter-cause"),
  filterRoute: document.getElementById("filter-route"),
  filterInitiative: document.getElementById("filter-initiative"),
  filterActivity: document.getElementById("filter-activity"),
  runtimeStart: document.getElementById("runtime-start"),
  runtimePause: document.getElementById("runtime-pause"),
  runtimeResume: document.getElementById("runtime-resume"),
  runtimeWake: document.getElementById("runtime-wake"),
  jumpAnalysis: document.getElementById("overview-jump-analysis"),
  languageButtons: [...document.querySelectorAll("[data-lang]")],
  chatInput: document.getElementById("chat-input"),
  chatSend: document.getElementById("chat-send"),
  chatClear: document.getElementById("chat-clear"),
  chatMessages: document.getElementById("chat-messages"),
  chatSessionState: document.getElementById("chat-session-state"),
  settingsStatus: document.getElementById("settings-status"),
  saveSettings: document.getElementById("save-settings"),
  settingsConfigurationSummary: document.getElementById("settings-configuration-summary"),
  settingsDiagnosticsSummary: document.getElementById("settings-diagnostics-summary"),
  settingsScheduledSummary: document.getElementById("settings-scheduled-summary"),
  settingsScheduledList: document.getElementById("settings-scheduled-list"),
  settingsDangerSummary: document.getElementById("settings-danger-summary"),
  settingsDangerUnlock: document.getElementById("settings-danger-unlock"),
  settingsDangerConfirm: document.getElementById("settings-danger-confirm"),
  settingsDangerControls: document.getElementById("settings-danger-controls"),
  settingsDangerNotice: document.getElementById("settings-danger-notice"),
  settingsPersonaReset: document.getElementById("settings-persona-reset"),
  settingsModelSummary: document.getElementById("settings-model-summary"),
  analysisViewRefresh: document.getElementById("analysis-view-refresh"),
  analysisViewCurrent: document.getElementById("analysis-view-current"),
  analysisViewLatest: document.getElementById("analysis-view-latest"),
  settingClearSafeMode: document.getElementById("setting-clear-safe-mode"),
  settingInstinctFirst: document.getElementById("setting-instinct-first"),
  settingSpontaneous: document.getElementById("setting-spontaneous"),
  settingBoundary: document.getElementById("setting-boundary"),
  settingFelt: document.getElementById("setting-felt"),
  settingAllowCommit: document.getElementById("setting-allow-commit"),
  settingNetworkEnabled: document.getElementById("setting-network-enabled"),
  settingExternalIoEnabled: document.getElementById("setting-external-io-enabled"),
  settingAutoSafeMode: document.getElementById("setting-auto-safe-mode"),
  settingMaxRoundsPerHour: document.getElementById("setting-max-rounds-per-hour"),
  settingMaxToolActionsPerHour: document.getElementById("setting-max-tool-actions-per-hour"),
  settingFailureTripThreshold: document.getElementById("setting-failure-trip-threshold"),
  settingQuietHours: document.getElementById("setting-quiet-hours"),
  settingLearningMode: document.getElementById("setting-learning-mode"),
  settingTraceExternalLearning: document.getElementById("setting-trace-external-learning"),
  settingAllowedDomains: document.getElementById("setting-allowed-domains"),
  settingWritableRoots: document.getElementById("setting-writable-roots"),
  settingKnowledgeRoots: document.getElementById("setting-knowledge-roots"),
  settingLearningLogDir: document.getElementById("setting-learning-log-dir"),
  settingBindingRenderer: document.getElementById("setting-binding-renderer"),
  settingBindingPlanner: document.getElementById("setting-binding-planner"),
  settingBindingPfc: document.getElementById("setting-binding-pfc"),
  settingBindingPerspective: document.getElementById("setting-binding-perspective"),
  settingBindingConsolidation: document.getElementById("setting-binding-consolidation"),
};

const DANGER_ZONE_FIELDS = [
  "settingAllowCommit",
  "settingNetworkEnabled",
  "settingExternalIoEnabled",
  "settingAutoSafeMode",
  "settingMaxRoundsPerHour",
  "settingMaxToolActionsPerHour",
  "settingFailureTripThreshold",
  "settingQuietHours",
];

let autoRefreshTimer = null;
let loadWorkbenchPromise = null;
let runtimeAutoRecoveryAttempted = false;
let motionPreferenceMedia = null;

const NAV_ITEMS = [
  ["overview-page", "navOverview"],
  ["analysis-page", "navAnalysis"],
  ["inner-space-page", "navInnerSpace"],
  ["chat-page", "navChat"],
  ["settings-page", "navSettings"],
];

function resolveLocale(value) {
  if (!value) {
    return FALLBACK_LOCALE;
  }
  if (SUPPORTED_LOCALES.has(value)) {
    return value;
  }
  return String(value).toLowerCase().startsWith("zh") ? "zh-CN" : "en";
}

function detectInitialLocale() {
  try {
    const stored = localStorage.getItem("workbench.locale");
    if (stored) {
      return resolveLocale(stored);
    }
  } catch {
    // Ignore localStorage failures in restricted browser contexts.
  }
  return FALLBACK_LOCALE;
}

function lookupMessage(locale, key) {
  return MESSAGES[locale]?.[key];
}

function t(key, vars = {}) {
  const template = lookupMessage(state.locale, key) ?? lookupMessage(FALLBACK_LOCALE, key) ?? key;
  return String(template).replace(/\{(\w+)\}/g, (_, name) => String(vars[name] ?? ""));
}

function hasItems(value) {
  if (Array.isArray(value)) {
    return value.length > 0;
  }
  if (!value || typeof value !== "object") {
    return false;
  }
  return Object.keys(value).length > 0;
}

function firstText(...values) {
  for (const value of values) {
    if (typeof value === "string" && value.trim()) {
      return value.trim();
    }
  }
  return "";
}

function labelFor(prefix, value, emptyKey = "common.unknown") {
  if (value === undefined || value === null || value === "") {
    return t(emptyKey);
  }
  const mapped = lookupMessage(state.locale, `${prefix}.${value}`) ?? lookupMessage(FALLBACK_LOCALE, `${prefix}.${value}`);
  return mapped || String(value);
}

function actionLabel(value) {
  const mapped = labelFor("actions", value, "common.unknownAction");
  return mapped === String(value ?? "") ? humanizeRuntimeToken(value, mapped) : mapped;
}

function routeTypeLabel(value) {
  return labelFor("routeType", value);
}

function initiativeLabel(value) {
  return labelFor("initiative", value);
}

function activityLabel(value) {
  return labelFor("activity", value);
}

function stallReasonLabel(value) {
  return labelFor("stall", value, "common.unknown");
}

function healthStatusLabel(value) {
  return labelFor("healthStatus", value, "common.unknown");
}

function moodLabel(value) {
  const mood = Number(value);
  if (!Number.isFinite(mood)) {
    return t("common.unknown");
  }
  const zh = state.locale === "zh-CN";
  if (mood >= 0.72) {
    return zh ? "情绪明亮" : "bright mood";
  }
  if (mood >= 0.58) {
    return zh ? "情绪平稳" : "steady mood";
  }
  if (mood >= 0.42) {
    return zh ? "情绪微沉" : "slightly lowered mood";
  }
  return zh ? "情绪低落" : "low mood";
}

function heartbeatLabel(value) {
  return labelFor("heartbeat", value, "common.unknown");
}

function boolWord(value, truthyKey = "common.yes", falsyKey = "common.no") {
  return value ? t(truthyKey) : t(falsyKey);
}

function moduleLabel(value) {
  return humanizeFunctionName(value);
}

function bindingLabel(value) {
  const key = String(value || "").trim();
  return {
    planner: "规划层",
    Renderer: "表达层",
    PFCAgent: "前额叶控制",
    PerspectiveModel: "视角模型",
    cognitive_packet: "认知包",
    deliberation: "深度推理",
    tool_planner: "工具规划",
    deep_renderer: "深度表达",
    consolidation_summarizer: "巩固摘要",
  }[key] || key || t("common.unknown");
}

function normalizeNarrativeToken(value) {
  return String(value || "")
    .trim()
    .replace(/^endogenous[:\s_-]*/i, "")
    .replace(/^endogenous\s+trigger[:\s_-]*/i, "")
    .replace(/^trigger[:\s_-]*/i, "")
    .replace(/^mode[:\s_-]*/i, "")
    .replace(/^state[:\s_-]*/i, "")
    .trim();
}

function tokenLabel(value) {
  const key = normalizeNarrativeToken(value);
  if (!key) {
    return t("common.unknown");
  }
  const mapped = lookupMessage(state.locale, `phrase.${key}`) ?? lookupMessage(FALLBACK_LOCALE, `phrase.${key}`);
  return mapped || humanizeRuntimeToken(key, humanizeFunctionName(key));
}

function viewModelHelpers() {
  return {
    t,
    actionLabel,
    moduleLabel,
    tokenLabel,
    moodLabel,
    routeTypeLabel,
    initiativeLabel,
    stallReasonLabel,
    heartbeatLabel,
    healthStatusLabel,
    boolWord,
    compactList,
    formatNumber,
    maybeText,
  };
}

function renderRefLinks(refs, emptyMessage = "") {
  const items = Array.isArray(refs) ? refs.filter((item) => item?.href && item?.label) : [];
  if (!items.length) {
    return emptyMessage ? `<div class="empty-state">${escapeHtml(emptyMessage)}</div>` : "";
  }
  return `<div class="jump-links">${items
    .map((item) => `<a class="jump-link" href="${escapeHtml(item.href)}" target="_blank" rel="noreferrer">${escapeHtml(item.label)}</a>`)
    .join("")}</div>`;
}

function renderNarrativeFeed(items, emptyMessage) {
  if (!Array.isArray(items) || items.length === 0) {
    return `<div class="empty-state">${escapeHtml(emptyMessage)}</div>`;
  }
  return items
    .map((item) => {
      const meta = item.emotional_weight ? `<span class="narrative-meta">${escapeHtml(`${t("summary.weightLabel")} ${formatNumber(item.emotional_weight, 2)}`)}</span>` : "";
      const tags = Array.isArray(item.symbolic_tags) && item.symbolic_tags.length
        ? `<div class="round-tags narrative-tags">${item.symbolic_tags.map((tag) => `<span class="mini-pill">${escapeHtml(tag)}</span>`).join("")}</div>`
        : "";
      return `<article class="stack-item narrative-card">
        <div class="narrative-header${meta ? " has-meta" : ""}">
          <h4>${escapeHtml(item.title || t("common.none"))}</h4>
          ${meta}
        </div>
        <p class="narrative-body">${escapeHtml(item.body || t("common.none"))}</p>
        ${tags ? `<div class="narrative-footer">${tags}</div>` : ""}
        ${renderRefLinks(item.refs)}
      </article>`;
    })
    .join("");
}

function applyStaticTranslations() {
  document.documentElement.lang = state.locale;
  document.title = t("document.title");
  document.querySelectorAll("[data-i18n]").forEach((node) => {
    node.textContent = t(node.dataset.i18n);
  });
  document.querySelectorAll("[data-i18n-placeholder]").forEach((node) => {
    node.setAttribute("placeholder", t(node.dataset.i18nPlaceholder));
  });
  elements.languageButtons.forEach((button) => {
    button.classList.toggle("is-active", resolveLocale(button.dataset.lang) === state.locale);
  });
}

function setLocale(locale, { persist = true } = {}) {
  state.locale = resolveLocale(locale);
  if (persist) {
    try {
      localStorage.setItem("workbench.locale", state.locale);
    } catch {
      // Ignore storage failures.
    }
  }
  applyStaticTranslations();
  updateClock();
  renderAll();
}

function nowIso() {
  return new Date().toISOString();
}

function markChatMutation(timestamp = nowIso()) {
  state.chatLocalMutationAt = timestamp;
  return timestamp;
}

function latestChatTimestamp(messages = state.chatMessages) {
  const sanitized = sanitizeChatMessages(messages, { preservePending: true });
  return sanitized[sanitized.length - 1]?.timestamp || "";
}

function withChatTimestamp(message, fallbackTimestamp = nowIso()) {
  return {
    ...message,
    timestamp: message?.timestamp || fallbackTimestamp,
  };
}

function roundRecord(roundId) {
  return (state.console?.recent_rounds || []).find((item) => Number(item.round_id) === Number(roundId)) || null;
}

function parseNumericRoundId(value) {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === "string" && value.trim()) {
    const normalized = value.trim().startsWith("round://") ? value.trim().slice("round://".length) : value.trim();
    const parsed = Number(normalized);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

function resolveChatEventRoundId(event) {
  return (
    parseNumericRoundId(event?.round_id) ||
    parseNumericRoundId(event?.roundId) ||
    parseNumericRoundId(event?.trace_ref) ||
    parseNumericRoundId(event?.traceRef) ||
    parseNumericRoundId(event?.event_log_ref?.round_trace_ref) ||
    null
  );
}

function deriveRoundMeta(round, detail = {}) {
  const trace = detail.trace || {};
  const initiativeWhy = detail.initiativeWhy || {};
  const gateDecisions = Array.isArray(trace.gate_decisions) ? trace.gate_decisions : [];
  const toolTraces = Array.isArray(trace.tool_traces) ? trace.tool_traces : [];
  const skillTraces = Array.isArray(trace.skill_traces) ? trace.skill_traces : [];
  const renderPlan = trace.render_plan || {};
  const messagePlan = renderPlan.message_plan || {};
  const routeType = String(round?.mode || trace.mode || round?.cause_type || trace.scenario || "unknown");
  const initiativeState = deriveRoundInitiativeState(round, { trace, initiativeWhy });
  const hasToolSignal =
    toolTraces.length > 0 ||
    skillTraces.length > 0 ||
    Boolean(renderPlan.tool_choice || messagePlan.tool_choice || messagePlan.tool_name);
  const hasApprovalSignal =
    toolTraces.some((item) => String(item?.status || "").includes("approval")) ||
    gateDecisions.some((item) => item && (!item.allowed || item.requires_resample));
  const activity = hasApprovalSignal ? "approval" : hasToolSignal ? "tool" : gateDecisions.length ? "gated" : "quiet";
  return {
    routeType,
    initiativeState,
    activity,
    gateCount: gateDecisions.length,
    approvalCount: hasApprovalSignal ? 1 : 0,
    toolCount: toolTraces.length || skillTraces.length || (hasToolSignal ? 1 : 0),
    modelHint:
      renderPlan?.identity_context?.model_label ||
      renderPlan?.identity_context?.provider_label ||
      renderPlan?.message_plan?.focus ||
      "",
  };
}

function roundMeta(round) {
  const detail = state.roundCatalog[String(round.round_id)] || {};
  return detail.meta || deriveRoundMeta(round, detail);
}

function toDataToken(value, fallback = "none") {
  const normalized = String(value ?? "")
    .trim()
    .toLowerCase()
    .replaceAll(/[^a-z0-9]+/g, "-")
    .replaceAll(/^-+|-+$/g, "");
  return normalized || fallback;
}

function setDataAttr(node, key, value) {
  if (!node) {
    return;
  }
  node.dataset[key] = String(value);
}

function normalizeRoundId(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : null;
}

function fallbackBuildTrendReplayFrames(trendSeries = []) {
  const basePoints = trendSeries[0]?.points || [];
  return basePoints.map((point, index) => {
    const markers = [];
    if (point?.hasConflict) {
      markers.push({ track: "conflict", label: "冲突" });
    }
    if (point?.hasRepair) {
      markers.push({ track: "repair", label: "修复" });
    }
    return {
      id: `${point.roundId}-${index}`,
      roundId: point.roundId,
      roundLabel: `第 ${point.roundId} 轮`,
      summary: point?.summary || "",
      metrics: trendSeries.map((series) => ({
        key: series?.key || "",
        displayLabel: series?.displayLabel || series?.label || series?.key || "",
        displaySubLabel: series?.displaySubLabel || series?.label || series?.key || "",
        value: series?.points?.[index]?.value ?? null,
      })),
      markers,
    };
  });
}

function buildTrendReplayFrames(trendSeries = []) {
  if (typeof viewModelsModule.buildAnalysisReplayFrames === "function") {
    const frames = viewModelsModule.buildAnalysisReplayFrames(trendSeries);
    if (Array.isArray(frames)) {
      return frames;
    }
  }
  return fallbackBuildTrendReplayFrames(trendSeries);
}

function renderTrendReplayControlsMarkup({
  hasFrames = false,
  isPlaying = false,
  hasStarted = false,
  activeRoundLabel = "",
  activeIndex = 0,
  totalFrames = 0,
} = {}) {
  if (typeof analysisBrainflowModule.renderTrendReplayControls === "function") {
    return analysisBrainflowModule.renderTrendReplayControls({
      hasFrames,
      isPlaying,
      hasStarted,
      activeRoundLabel,
      activeIndex,
      totalFrames,
    });
  }
  if (!hasFrames) {
    return `<div class="empty-state">等待回放帧。</div>`;
  }
  const safeTotal = Math.max(0, Number(totalFrames) || 0);
  const safeIndex = Math.max(0, Math.min(Number(activeIndex) || 0, Math.max(safeTotal - 1, 0)));
  const stateLabel = isPlaying ? "正在回放" : hasStarted ? "已暂停" : "等待开始";
  return `<div class="trend-replay-toolbar-inner">
    <div class="trend-replay-status" data-state="${isPlaying ? "playing" : hasStarted ? "paused" : "idle"}">
      <strong>${escapeHtml(stateLabel)}</strong>
      <span>${escapeHtml(`${safeIndex + 1} / ${safeTotal} · ${activeRoundLabel || "尚未定位轮次"}`)}</span>
    </div>
    <div class="trend-replay-actions">
      <button type="button" class="analysis-toolbar-button trend-replay-button" data-action="start" data-trend-replay-action="start"${isPlaying ? " disabled" : ""}>回放</button>
      <button type="button" class="analysis-toolbar-button trend-replay-button" data-action="pause" data-trend-replay-action="pause"${isPlaying ? "" : " disabled"}>暂停</button>
      <button type="button" class="analysis-toolbar-button trend-replay-button" data-action="resume" data-trend-replay-action="resume"${!hasStarted || isPlaying ? " disabled" : ""}>继续</button>
      <button type="button" class="analysis-toolbar-button trend-replay-button" data-action="reset" data-trend-replay-action="reset">重置</button>
    </div>
  </div>`;
}

function renderTrendFrameStripMarkup(frames = [], { activeRoundId = null } = {}) {
  if (typeof analysisBrainflowModule.renderTrendFrameStrip === "function") {
    return analysisBrainflowModule.renderTrendFrameStrip(frames, { activeRoundId });
  }
  if (!frames.length) {
    return `<div class="empty-state">等待回放帧。</div>`;
  }
  return frames
    .map((frame, index) => {
      const roundId = normalizeRoundId(frame?.roundId);
      const isActive = roundId !== null && Number(roundId) === Number(activeRoundId);
      const label = frame?.roundLabel || frame?.label || `第 ${roundId || "--"} 轮`;
      const summary = frame?.summary || "";
      const metrics = Array.isArray(frame?.metrics) ? frame.metrics : [];
      const markers = Array.isArray(frame?.markers) ? frame.markers : [];
      return `<button type="button" class="trend-frame${isActive ? " is-active" : ""}" data-replay-frame data-round-id="${escapeHtml(roundId ?? "")}" data-trend-round-id="${escapeHtml(roundId ?? "")}" data-replay-frame-index="${escapeHtml(index)}">
        <strong>${escapeHtml(label)}</strong>
        ${summary ? `<span class="trend-frame-summary">${escapeHtml(summary)}</span>` : ""}
        ${
          metrics.length
            ? `<div class="trend-frame-metrics">${metrics
                .map((metric) => {
                  const metricLabel = metric?.displayLabel || metric?.displaySubLabel || metric?.key || "--";
                  const metricValue = Number.isFinite(metric?.value) ? formatNumber(metric.value, 2) : "--";
                  return `<span class="trend-frame-metric" data-key="${escapeHtml(metric?.key || "metric")}"><span>${escapeHtml(metricLabel)}</span><strong>${escapeHtml(metricValue)}</strong></span>`;
                })
                .join("")}</div>`
            : ""
        }
        <div class="trend-frame-markers">${markers
          .map((marker) => `<span class="trend-frame-marker track-${escapeHtml(marker?.track || "default")}" data-track="${escapeHtml(marker?.track || "default")}">${escapeHtml(marker?.label || "")}</span>`)
          .join("")}</div>
      </button>`;
    })
    .join("");
}

function replayRoundIdAtIndex(index = state.replayRoundIndex) {
  const frames = Array.isArray(state.trendReplayFrames) ? state.trendReplayFrames : [];
  if (!frames.length) {
    return null;
  }
  const safeIndex = Math.max(0, Math.min(Number(index) || 0, frames.length - 1));
  return normalizeRoundId(frames[safeIndex]?.roundId);
}

function resolveTrendReplayState() {
  if (state.replayPlaying) {
    return "playing";
  }
  if (state.replayHasStarted) {
    return "paused";
  }
  return "idle";
}

function syncTrendReplayDataset(activeRoundId = null) {
  setDataAttr(elements.analysisTrendPanel, "replayState", resolveTrendReplayState());
  setDataAttr(elements.analysisTrendPanel, "replayPlaying", state.replayPlaying ? "true" : "false");
  setDataAttr(elements.analysisTrendPanel, "replayStarted", state.replayHasStarted ? "true" : "false");
  setDataAttr(elements.analysisTrendPanel, "replayRoundIndex", state.replayRoundIndex);
  setDataAttr(elements.analysisTrendPanel, "activeRoundId", activeRoundId ?? "");
}

function stopTrendReplayStepTimer() {
  if (state.replayStepTimer === null) {
    return;
  }
  window.clearInterval(state.replayStepTimer);
  state.replayStepTimer = null;
}

function resetTrendReplayState({ clearHoveredRound = false, clearFrames = false } = {}) {
  state.replayPlaying = false;
  state.replayRoundIndex = 0;
  state.replayHasStarted = false;
  stopTrendReplayStepTimer();
  if (clearHoveredRound) {
    state.hoveredTrendRoundId = null;
  }
  if (clearFrames) {
    state.trendReplayFrames = [];
  }
  syncTrendReplayDataset(state.selectedRoundId);
}

function resolveActiveTrendRoundId(trendSeries = []) {
  const available = new Set(
    (trendSeries[0]?.points || [])
      .map((point) => normalizeRoundId(point?.roundId))
      .filter((roundId) => roundId !== null),
  );
  const hoveredRoundId = normalizeRoundId(state.hoveredTrendRoundId);
  if (hoveredRoundId !== null && available.has(hoveredRoundId)) {
    return hoveredRoundId;
  }
  if (state.replayPlaying || state.replayRoundIndex > 0) {
    const replayRoundId = replayRoundIdAtIndex(state.replayRoundIndex);
    if (replayRoundId !== null && available.has(replayRoundId)) {
      return replayRoundId;
    }
  }
  const selectedRoundId = normalizeRoundId(state.selectedRoundId);
  if (selectedRoundId !== null && (available.size === 0 || available.has(selectedRoundId))) {
    return selectedRoundId;
  }
  const ordered = [...available];
  return ordered.length ? ordered[ordered.length - 1] : null;
}

function syncTrendFrameStripScroll() {
  const frameStrip = elements.analysisTrendFrameStrip;
  if (!frameStrip) {
    return;
  }
  const activeFrame = frameStrip.querySelector(
    ".trend-frame.is-active, .trend-frame[data-active='true'], [data-replay-frame].is-active, [data-replay-frame][data-active='true']",
  );
  if (!activeFrame) {
    return;
  }
  const reducedMotion = document.body?.dataset?.motion === "reduced" || elements.root?.dataset?.motion === "reduced";
  activeFrame.scrollIntoView({
    block: "nearest",
    inline: "center",
    behavior: reducedMotion ? "auto" : "smooth",
  });
}

function startTrendReplayStepTimer() {
  stopTrendReplayStepTimer();
  state.replayStepTimer = window.setInterval(() => {
    if (!state.replayPlaying) {
      return;
    }
    if (!state.trendReplayFrames.length) {
      resetTrendReplayState({ clearHoveredRound: true, clearFrames: true });
      renderAnalysis();
      return;
    }
    const nextIndex = state.replayRoundIndex + 1;
    if (nextIndex >= state.trendReplayFrames.length) {
      state.replayPlaying = false;
      stopTrendReplayStepTimer();
      renderAnalysis();
      return;
    }
    state.replayRoundIndex = nextIndex;
    const nextRoundId = replayRoundIdAtIndex(nextIndex);
    if (nextRoundId !== null) {
      state.hoveredTrendRoundId = nextRoundId;
    }
    renderAnalysis();
  }, TREND_REPLAY_STEP_MS);
}

function handleTrendReplayAction(action) {
  if (!state.trendReplayFrames.length) {
    return;
  }
  if (action === "start") {
    state.replayRoundIndex = 0;
    state.replayHasStarted = true;
    const firstRoundId = replayRoundIdAtIndex(0);
    if (firstRoundId !== null) {
      state.hoveredTrendRoundId = firstRoundId;
    }
    state.replayPlaying = true;
    startTrendReplayStepTimer();
    renderAnalysis();
    return;
  }
  if (action === "pause") {
    state.replayPlaying = false;
    stopTrendReplayStepTimer();
    renderAnalysis();
    return;
  }
  if (action === "resume") {
    state.replayHasStarted = true;
    if (state.replayRoundIndex >= state.trendReplayFrames.length - 1) {
      state.replayRoundIndex = 0;
    }
    const replayRoundId = replayRoundIdAtIndex(state.replayRoundIndex);
    if (replayRoundId !== null) {
      state.hoveredTrendRoundId = replayRoundId;
    }
    state.replayPlaying = true;
    startTrendReplayStepTimer();
    renderAnalysis();
    return;
  }
  if (action === "reset") {
    resetTrendReplayState({ clearHoveredRound: true });
    renderAnalysis();
  }
}

function roundIdFromTimelineTarget(target) {
  const linked = target?.closest?.(
    "[data-trend-round-id], [data-round-id], [data-round-ref], [data-round-jump], [data-replay-frame-round-id], [data-replay-round-id], [data-replay-frame], [data-round-row]",
  );
  if (!linked) {
    return null;
  }
  const dataset = linked.dataset || {};
  const explicitRound =
    dataset.trendRoundId || dataset.roundId || dataset.roundRef || dataset.roundJump || dataset.replayFrameRoundId || dataset.replayRoundId;
  const resolvedRoundId = normalizeRoundId(explicitRound || linked.getAttribute("data-round-id"));
  if (resolvedRoundId !== null) {
    return resolvedRoundId;
  }
  const frameIndex = Number(dataset.replayFrameIndex || dataset.frameIndex);
  if (Number.isFinite(frameIndex)) {
    return replayRoundIdAtIndex(frameIndex);
  }
  return null;
}

function setHoveredTrendRound(roundId, { stopReplay = false } = {}) {
  const normalizedRoundId = normalizeRoundId(roundId);
  if (normalizedRoundId === null || normalizedRoundId === state.hoveredTrendRoundId) {
    return;
  }
  if (stopReplay) {
    state.replayPlaying = false;
    stopTrendReplayStepTimer();
  }
  state.hoveredTrendRoundId = normalizedRoundId;
  renderAnalysis();
}

function clearHoveredTrendRound() {
  const fallbackRoundId = state.replayPlaying || state.replayRoundIndex > 0 ? replayRoundIdAtIndex(state.replayRoundIndex) : null;
  const nextRoundId = fallbackRoundId ?? null;
  if (nextRoundId === state.hoveredTrendRoundId) {
    return;
  }
  state.hoveredTrendRoundId = nextRoundId;
  renderAnalysis();
}

function syncMotionPreference() {
  const prefersReduced = Boolean(motionPreferenceMedia?.matches);
  const motionMode = prefersReduced ? "reduced" : "full";
  setDataAttr(document.body, "motion", motionMode);
  setDataAttr(elements.root, "motion", motionMode);
}

function updateNavPresentationState() {
  const activeIndex = NAV_ITEMS.findIndex(([pageId]) => pageId === state.page);
  NAV_ITEMS.forEach(([pageId, elementKey], index) => {
    const button = elements[elementKey];
    if (!button) {
      return;
    }
    const isActive = pageId === state.page;
    button.classList.toggle("active", isActive);
    button.classList.toggle("is-active", isActive);
    button.classList.toggle("is-inactive", !isActive);
    button.setAttribute("aria-current", isActive ? "page" : "false");
    setDataAttr(button, "navIndex", index);
    setDataAttr(button, "active", isActive ? "true" : "false");
  });
  setDataAttr(elements.root, "navActiveIndex", activeIndex >= 0 ? activeIndex : 0);
  setDataAttr(document.body, "navActiveIndex", activeIndex >= 0 ? activeIndex : 0);
}

function applyPresentationHooks() {
  const service = state.service || {};
  const autonomy = state.autonomy || {};
  const runtimeMode = state.runtimeState?.mode || "";
  const healthState = toDataToken(routeHealthLabel(service, autonomy), "unknown");
  const sessionAttached = Boolean(state.bootstrap?.session_attached);
  const hasRound = Boolean(state.selectedRoundId || state.console?.state?.current_round?.round_id);
  setDataAttr(document.body, "workbenchPage", state.page);
  setDataAttr(document.body, "workbenchLocale", state.locale);
  setDataAttr(document.body, "health", healthState);
  setDataAttr(document.body, "runtimeMode", toDataToken(runtimeMode, "unknown"));
  setDataAttr(document.body, "sessionAttached", sessionAttached ? "true" : "false");
  setDataAttr(document.body, "hasRound", hasRound ? "true" : "false");
  setDataAttr(elements.root, "page", state.page);
  setDataAttr(elements.root, "locale", state.locale);
  setDataAttr(elements.root, "health", healthState);
  setDataAttr(elements.root, "runtimeMode", toDataToken(runtimeMode, "unknown"));
  setDataAttr(elements.root, "sessionAttached", sessionAttached ? "true" : "false");
  setDataAttr(elements.root, "hasRound", hasRound ? "true" : "false");
  setDataAttr(elements.root, "autonomyRunning", autonomy.running ? "true" : "false");
  setDataAttr(elements.root, "autonomyStalled", autonomy.stalled ? "true" : "false");
  updateNavPresentationState();
}

function parseDelimitedText(value) {
  return String(value || "")
    .split(/[,\n，]+/)
    .map((item) => item.trim())
    .filter(Boolean);
}

function formatDelimitedLines(values) {
  return Array.isArray(values) && values.length ? values.join("\n") : "";
}

function humanizeFunctionName(name) {
  const key = String(name || "").trim();
  if (!key) {
    return t("common.unknown");
  }
  const mapped = lookupMessage(state.locale, `function.${key}`) ?? lookupMessage(FALLBACK_LOCALE, `function.${key}`);
  if (mapped) {
    return mapped;
  }
  return humanizeRuntimeToken(key, key);
}

function summarizeRounds(rounds, count = 3) {
  return (rounds || []).slice(0, count);
}

function optionMarkup(value, label, selected = false) {
  return `<option value="${escapeHtml(value)}"${selected ? " selected" : ""}>${escapeHtml(label)}</option>`;
}

function updateRoundSelectors(rounds) {
  const options = rounds.length
    ? rounds.map((round) => optionMarkup(round.round_id, `#${round.round_id} · ${actionLabel(round.sampled_action)}`, Number(round.round_id) === Number(state.selectedRoundId)))
    : [optionMarkup("", t("summary.quickSelectEmpty"), true)];
  const markup = options.join("");
  if (elements.overviewRoundSelect) {
    elements.overviewRoundSelect.innerHTML = markup;
    elements.overviewRoundSelect.disabled = rounds.length === 0;
  }
  if (elements.analysisRoundSelect) {
    elements.analysisRoundSelect.innerHTML = markup;
    elements.analysisRoundSelect.disabled = rounds.length === 0;
  }
}

function buildRoundCard(round, options = {}) {
  const { compact = false } = options;
  const meta = roundMeta(round);
  const causeLabel = round.cause_type
    ? lookupMessage(state.locale, `analysis.filters.cause.${round.cause_type}`) ??
      lookupMessage(FALLBACK_LOCALE, `analysis.filters.cause.${round.cause_type}`) ??
      humanizeFunctionName(round.cause_type)
    : routeTypeLabel(meta.routeType);
  return `<button class="${compact ? "list-row" : "round-row"} ${Number(round.round_id) === Number(state.selectedRoundId) ? "selected" : ""}" data-round-row data-round-id="${escapeHtml(round.round_id)}" data-round-action="${escapeHtml(round.sampled_action || "")}">
    <div class="round-title">
      <strong>#${escapeHtml(round.round_id)} · ${escapeHtml(actionLabel(round.sampled_action))}</strong>
      <span>${escapeHtml(routeTypeLabel(meta.routeType))}</span>
      <div class="round-tags">
        <span class="mini-pill">${escapeHtml(initiativeLabel(meta.initiativeState))}</span>
        <span class="mini-pill">${escapeHtml(activityLabel(meta.activity))}</span>
      </div>
    </div>
    <div class="round-meta">
      <span>${escapeHtml(causeLabel)}</span>
      <span>${escapeHtml(`冲突 ${formatNumber(round.conflict_score, 2)}`)}</span>
    </div>
  </button>`;
}

function renderSettingsStatus(statusKey, vars = {}) {
  elements.settingsStatus.textContent = t(statusKey, vars);
}

async function loadRoundCatalogEntry(round) {
  const roundId = String(round.round_id);
  if (state.roundCatalog[roundId]) {
    return state.roundCatalog[roundId];
  }
  if (state.roundCatalogLoading[roundId]) {
    return state.roundCatalogLoading[roundId];
  }
  state.roundCatalogLoading[roundId] = loadWorkbenchRoundCatalogData(round.round_id)
    .then((roundDetail) => {
      const initiativeWhy = roundDetail.initiativeWhy || null;
      const mergedDetail = {
        trace: roundDetail.trace,
        initiativeWhy,
        thought: roundDetail.thought || null,
        why: roundDetail.why || null,
        contributions: roundDetail.contributions || null,
        probability: roundDetail.probability || null,
        whyNot: roundDetail.whyNot || null,
        replay: roundDetail.replay || null,
        meta: deriveRoundMeta(round, { trace: roundDetail.trace, initiativeWhy }),
      };
      state.roundCatalog[roundId] = mergedDetail;
      delete state.roundCatalogLoading[roundId];
      return mergedDetail;
    })
    .catch(() => {
      const detail = { trace: null, initiativeWhy: null, meta: deriveRoundMeta(round, {}) };
      state.roundCatalog[roundId] = detail;
      delete state.roundCatalogLoading[roundId];
      return detail;
    });
  return state.roundCatalogLoading[roundId];
}

async function warmRoundCatalog(rounds) {
  await Promise.all((rounds || []).map((round) => loadRoundCatalogEntry(round)));
}

function scheduleRoundCatalogWarm(rounds) {
  void warmRoundCatalog(rounds)
    .then(() => {
      if (state.page === "analysis-page") {
        renderRoundList();
      }
    })
    .catch(() => {});
}

function setPage(pageId) {
  state.page = pageId;
  [elements.overviewPage, elements.analysisPage, elements.innerSpacePage, elements.chatPage, elements.settingsPage].filter(Boolean).forEach((page) => {
    page.classList.toggle("active", page.id === pageId);
    setDataAttr(page, "active", page.id === pageId ? "true" : "false");
  });
  updateNavPresentationState();
  applyPresentationHooks();
  if (pageId === "chat-page") {
    restorePersistedChatHistory();
  }
  if (pageId === "analysis-page") {
    renderAnalysis();
    const currentRound = state.console?.state?.current_round;
    const latestRound = state.console?.recent_rounds?.[0];
    const preferredRound = currentRound?.round_id ? currentRound : latestRound;
    if (preferredRound?.round_id && !state.selectedRoundPayload) {
      queueSelectedRoundHydration({
        round_id: preferredRound.round_id,
        sampled_action: preferredRound.sampled_action || state.selectedRoundAction || "respond",
      });
    }
  }
  if (
    (pageId === "analysis-page" && !state.console?.action_field) ||
    (pageId === "inner-space-page" && !state.initiativeStatus) ||
    (pageId === "settings-page" && !state.settings)
  ) {
    void loadWorkbench({ preserveControlNotice: true });
  }
}

function updateClock() {
  elements.clock.textContent = new Date().toLocaleTimeString(state.locale, {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function stallReasonSummary(autonomy) {
  const reason = stallReasonLabel(autonomy?.stall_reason);
  const detail = autonomy?.stall_reason_detail || t("common.none");
  const peak = autonomy?.candidate_peak
    ? t("summary.currentPeak", { action: actionLabel(autonomy.candidate_peak) })
    : "";
  return t("summary.autonomyReason", { reason, detail, peak });
}

function renderHealthStrip() {
  const service = state.service || {};
  const autonomy = state.autonomy || {};
  const runtimeMode = state.runtimeState?.mode || "";
  const currentRound = state.console?.state?.current_round || {};
  const health = healthStatusLabel(routeHealthLabel(service, autonomy));
  const pendingApprovals = state.sessionState?.approvals?.pending_count || 0;
  const baseSummary = [
    `${t("summary.service")} ${health}`,
    `${t("summary.autonomy")} ${heartbeatLabel(autonomy.heartbeat_state || "idle")}`,
    `${t("summary.session")} ${state.bootstrap?.session_attached ? t("summary.attached") : t("summary.detached")}`,
    `${t("summary.approvals")} ${pendingApprovals}`,
    stallReasonLabel(autonomy?.stall_reason),
  ].join(" / ");
  elements.topStatusMini.textContent = state.controlNotice
    ? t("summary.controlNotice", { notice: state.controlNotice, base: baseSummary })
    : baseSummary;

  elements.healthService.textContent = service.healthy
    ? `${t("common.ready")} · ${service.http_ready ? t("summary.httpReady") : t("summary.httpWarming")}`
    : t("common.degraded");
  elements.healthAutonomy.textContent = autonomy.stalled
    ? t("summary.healthStalled", {
        seconds: formatNumber(autonomy.stall_seconds, 1),
        reason: stallReasonLabel(autonomy.stall_reason),
      })
    : t("summary.healthRunning", {
        heartbeat: heartbeatLabel(autonomy.heartbeat_state || "idle"),
        running: autonomy.running ? t("common.running") : t("common.stopped"),
        reason: stallReasonLabel(autonomy.stall_reason),
      });
  elements.healthSession.textContent = t("summary.sessionHealth", {
    attachState: state.bootstrap?.session_attached ? t("common.attached") : t("common.detached"),
    approvalState: pendingApprovals
      ? t("summary.pendingApprovals", { count: pendingApprovals })
      : t("summary.noApprovals"),
  });
  elements.healthRound.textContent = currentRound.round_id
    ? `#${currentRound.round_id} · ${actionLabel(currentRound.sampled_action)}`
    : t("common.noneRound");
  const bindings = state.models?.module_model_bindings || {};
  elements.healthModel.textContent = t("summary.modelHealth", {
    packet: humanizeRuntimeToken(bindings.cognitive_packet || "--"),
    renderer: humanizeRuntimeToken(bindings.deep_renderer || "--"),
  });
  elements.runtimeStart.disabled = Boolean(autonomy.running && autonomy.runner_attached);
  elements.runtimePause.disabled = !autonomy.running;
  elements.runtimeResume.disabled = Boolean(autonomy.running);
  elements.runtimeWake.disabled = runtimeMode === "interactive";
  setDataAttr(elements.healthService, "health", service.healthy ? "healthy" : "degraded");
  setDataAttr(elements.healthAutonomy, "health", autonomy.stalled ? "stalled" : autonomy.running ? "running" : "idle");
  setDataAttr(elements.healthSession, "health", state.bootstrap?.session_attached ? "attached" : "detached");
  setDataAttr(elements.healthRound, "health", currentRound.round_id ? "active" : "empty");
  setDataAttr(elements.healthModel, "health", bindings.cognitive_packet || bindings.deep_renderer ? "bound" : "missing");
  setDataAttr(elements.topStatusMini, "health", toDataToken(routeHealthLabel(service, autonomy), "unknown"));
  applyPresentationHooks();
}

function renderOverview() {
  const service = state.service || {};
  const autonomy = state.autonomy || {};
  const runtime = state.runtimeState || {};
  const consoleState = state.console || {};
  const learning = runtime.autonomy_policy || {};
  const subjectKernel = state.subject?.subject_kernel || {};
  const meaningSystem = state.meaning?.meaning_system || {};
  const purposeMemory = Array.isArray(state.meaning?.purpose_memory) ? state.meaning.purpose_memory : [];
  const relationshipCommitments = Array.isArray(state.meaning?.relationship_commitments) ? state.meaning.relationship_commitments : [];
  const agencyLoop = state.agency?.agency_loop || {};
  const proactiveBacklog = Array.isArray(state.agency?.proactive_backlog) ? state.agency.proactive_backlog : [];
  const suppressedActions = Array.isArray(agencyLoop.suppressed_actions) ? agencyLoop.suppressed_actions : [];
  const budget = agencyLoop.budget || {};
  const bindings = state.models?.module_model_bindings || {};
  const snapshot = buildOverviewSnapshot(
    {
      bootstrap: state.bootstrap,
      service,
      autonomy,
      runtimeState: runtime,
      consoleData: consoleState,
      models: state.models,
      sessionState: state.sessionState,
      settings: state.settings,
      initiativeStatus: state.initiativeStatus,
      subject: state.subject,
      meaning: state.meaning,
      agency: state.agency,
    },
    viewModelHelpers(),
  );
  const meaningSources = (meaningSystem.sources || []).map((item) => item?.label).filter(Boolean);
  const focusBacklog = humanizeRuntimeText(
    firstText(proactiveBacklog[0]?.summary, proactiveBacklog[0]?.title),
    firstText(proactiveBacklog[0]?.summary, proactiveBacklog[0]?.title),
  );
  const readableAgencySummary = humanizeRuntimeText(firstText(agencyLoop.summary), firstText(agencyLoop.summary));
  const readableSubjectNarrative = humanizeRuntimeText(firstText(subjectKernel.current_narrative), firstText(subjectKernel.current_narrative));
  const readableMeaningNarrative = humanizeRuntimeText(firstText(meaningSystem.survival_narrative), firstText(meaningSystem.survival_narrative));
  const continuityScore = Number(subjectKernel.continuity_score);
  const overviewSummary = [
    readableSubjectNarrative || maybeText(subjectKernel.current_narrative, snapshot.identity_intro),
    readableMeaningNarrative || maybeText(meaningSystem.survival_narrative),
    focusBacklog ? `当前最靠前的主动事项是：${focusBacklog}。` : "",
    snapshot.current_focus,
  ]
    .filter(Boolean)
    .join(" ")
    .trim();

  elements.overviewSummary.textContent = overviewSummary || `${snapshot.identity_intro} ${snapshot.current_state_summary} ${snapshot.current_focus}`.trim();
  if (elements.overviewFocusCues) {
    const focusCues = [
      {
        label: "主体",
        value: maybeText(subjectKernel.continuity_summary, snapshot.identity_intro),
      },
      {
        label: "意义",
        value: readableMeaningNarrative || maybeText(meaningSystem.survival_narrative, meaningSources[0] ? `当前意义来源：${meaningSources[0]}` : "当前意义线索仍在收口。"),
      },
      {
        label: "下一步",
        value: maybeText(focusBacklog, snapshot.current_focus, "当前还没有显性下一步。"),
      },
    ];
    elements.overviewFocusCues.innerHTML = focusCues
      .map(
        (item) => `<article class="hero-cue-card">
          <span>${escapeHtml(item.label)}</span>
          <strong>${escapeHtml(item.value)}</strong>
        </article>`,
      )
      .join("");
  }

  elements.autonomyBadge.textContent = Number.isFinite(continuityScore)
    ? `连续性 ${formatNumber(continuityScore, 2)}`
    : maybeText(subjectKernel.continuity_summary, t("common.pending"));
  elements.autonomySummary.innerHTML = [
    ["当前自我叙述", readableSubjectNarrative || maybeText(subjectKernel.current_narrative, snapshot.identity_intro)],
    ["核心承诺", compactList(subjectKernel.core_commitments, "当前还没有显性核心承诺")],
    ["生存叙事", readableMeaningNarrative || maybeText(meaningSystem.survival_narrative, "当前意义叙事仍在收口。")],
    ["意义来源", compactList(meaningSources.slice(0, 2), "当前没有显性意义来源")],
    ["目的记忆", compactList(purposeMemory.slice(0, 2).map((item) => item?.summary).filter(Boolean), "当前没有稳定目的记忆")],
    ["关系承诺", compactList(relationshipCommitments.slice(0, 2).map((item) => item?.commitment).filter(Boolean), "当前没有显性关系承诺")],
  ]
    .map(([label, value]) => `<div class="stack-item"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`)
    .join("");

  elements.modelBadge.textContent = proactiveBacklog.length
    ? `待执行 ${proactiveBacklog.length} 项`
    : autonomy.running
      ? "闭环运行中"
      : "闭环待机";
  elements.modelSummary.innerHTML = [
    ["当前主动说明", readableAgencySummary || maybeText(agencyLoop.summary, "当前没有显性主动闭环。")],
    ["最靠前事项", maybeText(focusBacklog, "当前没有待执行主动项。")],
    ["已到期任务", Number(budget.scheduled_due || 0) > 0 ? `${Number(budget.scheduled_due || 0)} 项` : "无到期任务"],
    [
      "被抑制动作",
      compactList(
        suppressedActions.slice(0, 2).map((item) => item?.summary || item?.suppression_reason).filter(Boolean),
        "当前没有被抑制动作",
      ),
    ],
    [
      "剩余对外预算",
      `${Number(budget.outward_dispatch_remaining || 0)} 项 / 冷却 ${Number(budget.cooldown_remaining_seconds || 0)}s`,
    ],
    [
      "自治执行计数",
      `工具 ${Number(budget.autonomy_tool_actions || 0)} · 内生轮次 ${Number(budget.autonomy_endogenous_rounds || 0)}`,
    ],
  ]
    .map(([label, value]) => `<div class="stack-item"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`)
    .join("");

  elements.learningMode.textContent = humanizeRuntimeToken(learning.learning_mode || t("common.none"), maybeText(learning.learning_mode, t("common.none")));
  elements.learningTrace.textContent = [
    `${bindingLabel("cognitive_packet")} ${humanizeRuntimeToken(bindings.cognitive_packet || "--")}`,
    `${bindingLabel("deep_renderer")} ${humanizeRuntimeToken(bindings.deep_renderer || "--")}`,
  ].join(" / ");
  elements.learningDomains.textContent = compactList(
    learning.allowed_network_domains,
    learning.network_enabled ? "网络已开启，但未限制具体域名" : "当前未放开网络域名",
  );
  elements.learningRoots.textContent = compactList(
    [...(learning.writable_roots || []), ...(learning.knowledge_roots || [])],
    learning.allow_commit ? "允许提交，但尚未登记可写根目录" : t("common.none"),
  );
  if (elements.overviewPersonaSummary) {
    elements.overviewPersonaSummary.textContent = snapshot.persona_mood_summary;
  }

  const rounds = consoleState.recent_rounds || [];
  updateRoundSelectors(rounds);
  setDataAttr(elements.overviewPage, "roundCount", rounds.length);
  setDataAttr(elements.overviewPage, "roundListExpanded", state.roundListExpanded ? "true" : "false");
  const overviewRounds = rounds.length
    ? rounds
        .slice(0, state.roundListExpanded ? 6 : 3)
        .map((round) => buildRoundCard(round))
        .join("")
    : `<div class="empty-state">${escapeHtml(
        state.bootstrap?.session_attached ? t("summary.noRoundsAttached") : t("summary.noRoundsDetached"),
      )}</div>`;
  elements.overviewRounds.innerHTML = `
    <section class="subsection">
      <h4 class="subsection-title">当前人格状态</h4>
      <div class="subsection-stack">
        ${snapshot.persona_mood_signals
          .map((item) => `<div class="list-row compact-row">
            <span>${escapeHtml(item.label)}</span>
            <strong>${escapeHtml(item.value)}</strong>
          </div>`)
          .join("")}
      </div>
    </section>
    <section class="subsection">
      <h4 class="subsection-title">最近轮次</h4>
      <div class="subsection-stack">${overviewRounds}</div>
    </section>
  `;
}

function ensureFilterOptions() {
  const rounds = state.console?.recent_rounds || [];
  const actions = new Set(rounds.map((item) => item.sampled_action).filter(Boolean));
  const routes = new Set(rounds.map((item) => roundMeta(item).routeType).filter(Boolean));
  const initiatives = new Set(rounds.map((item) => roundMeta(item).initiativeState).filter(Boolean));
  const activities = new Set(rounds.map((item) => roundMeta(item).activity).filter(Boolean));
  elements.filterAction.innerHTML = `<option value="">${escapeHtml(t("analysis.filters.action.all"))}</option>${[...actions]
    .map((item) => `<option value="${escapeHtml(item)}">${escapeHtml(actionLabel(item))}</option>`)
    .join("")}`;
  elements.filterAction.value = state.filters.action;
  elements.filterRoute.innerHTML = `<option value="">${escapeHtml(t("analysis.filters.route.all"))}</option>${[...routes]
    .map((item) => `<option value="${escapeHtml(item)}">${escapeHtml(routeTypeLabel(item))}</option>`)
    .join("")}`;
  elements.filterRoute.value = state.filters.route;
  elements.filterInitiative.innerHTML = `<option value="">${escapeHtml(t("analysis.filters.initiative.all"))}</option>${[...initiatives]
    .map((item) => `<option value="${escapeHtml(item)}">${escapeHtml(initiativeLabel(item))}</option>`)
    .join("")}`;
  elements.filterInitiative.value = state.filters.initiative;
  elements.filterActivity.innerHTML = `<option value="">${escapeHtml(t("analysis.filters.activity.all"))}</option>${[...activities]
    .map((item) => `<option value="${escapeHtml(item)}">${escapeHtml(activityLabel(item))}</option>`)
    .join("")}`;
  elements.filterActivity.value = state.filters.activity;
}

function filteredRounds() {
  return (state.console?.recent_rounds || []).filter((round) => {
    const meta = roundMeta(round);
    if (state.filters.roundQuery && !String(round.round_id).includes(state.filters.roundQuery)) {
      return false;
    }
    if (state.filters.action && round.sampled_action !== state.filters.action) {
      return false;
    }
    if (state.filters.cause && round.cause_type !== state.filters.cause) {
      return false;
    }
    if (state.filters.route && meta.routeType !== state.filters.route) {
      return false;
    }
    if (state.filters.initiative && meta.initiativeState !== state.filters.initiative) {
      return false;
    }
    if (state.filters.activity && meta.activity !== state.filters.activity) {
      return false;
    }
    return true;
  });
}

function renderRoundList() {
  ensureFilterOptions();
  const rounds = filteredRounds();
  updateRoundSelectors(rounds);
  elements.analysisRoundList.classList.toggle("is-collapsed", !state.roundListExpanded);
  setDataAttr(elements.analysisPage, "roundCount", rounds.length);
  setDataAttr(elements.analysisPage, "roundListExpanded", state.roundListExpanded ? "true" : "false");
  setDataAttr(elements.analysisRoundList, "empty", rounds.length ? "false" : "true");
  setDataAttr(elements.analysisRoundList, "hasFilters", Object.values(state.filters).some(Boolean) ? "true" : "false");
  elements.analysisRoundToggle.textContent = t(state.roundListExpanded ? "analysis.rounds.collapse" : "analysis.rounds.expand");
  if (!rounds.length) {
    const hasFilters = Object.values(state.filters).some(Boolean);
    const message = hasFilters
      ? t("summary.noMatchingRounds")
      : state.bootstrap?.session_attached
        ? t("summary.noRoundsAttached")
        : t("summary.noRoundsDetached");
    elements.analysisRoundList.innerHTML = `<div class="empty-state">${escapeHtml(message)}</div>`;
    return;
  }
  if (state.page === "analysis-page" && !state.selectedRoundPayload) {
    const preferredRound =
      rounds.find((round) => Number(round.round_id) === Number(state.selectedRoundId)) ||
      rounds[0];
    queueSelectedRoundHydration(preferredRound);
  }
  elements.analysisRoundList.innerHTML = rounds.map((round) => buildRoundCard(round)).join("");
}

function formatContributionRows(rows) {
  if (!Array.isArray(rows) || rows.length === 0) {
    return `<div class="empty-state">${escapeHtml(t("summary.noContributions"))}</div>`;
  }
  return rows
    .slice(0, 12)
    .map((row) => {
      const source = row.module_name || row.agent_name || row.source || t("common.unknown");
      const weight =
        row.delta_normalized?.respond ??
        row.delta_projected?.respond ??
        row.weight ??
        row.score ??
        row.confidence_calibrated ??
        row.confidence ??
        0;
      return `<div class="stack-item"><span>${escapeHtml(humanizeFunctionName(source))}</span><strong>${escapeHtml(formatNumber(weight, 3))}</strong></div>`;
    })
    .join("");
}

function formatProbabilityLayers(probabilityField) {
  const layers = [
    ["action", probabilityField?.action],
    ["token", probabilityField?.token],
    ["instinct", probabilityField?.instinct],
    ["context", probabilityField?.context],
  ].filter(([, layer]) => layer);
  if (!layers.length) {
    return `<div class="empty-state">${escapeHtml(t("summary.noProbability"))}</div>`;
  }
  return layers
    .map(([name, layer]) => {
      const winner = layer.winner?.action || layer.winner_target || "";
      const probability =
        layer.winner_posterior && winner ? formatNumber(layer.winner_posterior[winner], 3) : formatNumber(layer.posterior_max, 3, "--");
      const auditSize = Array.isArray(layer.contribution_audit) ? layer.contribution_audit.length : 0;
      return `<div class="stack-item"><span>${escapeHtml(labelFor("layer", name))}</span><strong>${escapeHtml(`胜出 ${winner ? actionLabel(winner) : t("common.none")} · 概率 ${probability} · 审计 ${auditSize}`)}</strong></div>`;
    })
    .join("");
}

function formatSourceLinks(links) {
  if (!Array.isArray(links) || links.length === 0) {
    return `<div class="empty-state">${escapeHtml(t("summary.noSourceLinks"))}</div>`;
  }
  return links
    .slice(0, 10)
    .map(
      (link) => `<div class="stack-item">
        <span>${escapeHtml(`${t("summary.sourcePanel")} · ${link.panel_id || "面板"}`)}</span>
        <strong>${escapeHtml(link.api_path || link.controller_method || "--")}</strong>
      </div>`,
    )
    .join("");
}

function formatGateRows(gates) {
  if (!Array.isArray(gates) || gates.length === 0) {
    return `<div class="empty-state">${escapeHtml(t("summary.noGates"))}</div>`;
  }
  return gates
    .slice(0, 6)
    .map((gate) => {
      const status = gate.allowed ? (gate.requires_resample ? t("gate.resample") : t("gate.pass")) : t("gate.block");
      const owner = humanizeFunctionName(gate.owner || gate.stage || "gate");
      return `<div class="stack-item"><span>${escapeHtml(`${t("summary.gateStage")} · ${owner}`)}</span><strong>${escapeHtml(`${t("summary.gateResult")} · ${status} · ${humanizeRuntimeText(gate.reason || t("common.none"))}`)}</strong></div>`;
    })
    .join("");
}

function formatToolApprovalRows(sessionState, trace) {
  const tools = Array.isArray(sessionState?.tools) ? sessionState.tools : [];
  const approvals = Array.isArray(sessionState?.approvals?.pending) ? sessionState.approvals.pending : [];
  const traceTools = Array.isArray(trace?.tool_traces) ? trace.tool_traces : [];
  const rows = [];
  approvals.slice(0, 3).forEach((item) => {
    rows.push(
      `<div class="stack-item"><span>${escapeHtml(t("summary.approvalAction"))}</span><strong>${escapeHtml(
        `${humanizeRuntimeToken(item.tool_name || item.call_id || "工具")} · ${humanizeRuntimeText(item.reason || t("common.pending"))}`,
      )}</strong></div>`,
    );
  });
  tools.slice(0, 3).forEach((item) => {
    rows.push(
      `<div class="stack-item"><span>${escapeHtml(t("summary.toolAction"))}</span><strong>${escapeHtml(
        `${humanizeRuntimeToken(item.tool_name || item.call_id || "工具")} · ${humanizeRuntimeToken(item.status || t("common.unknown"))}`,
      )}</strong></div>`,
    );
  });
  traceTools.slice(0, 3).forEach((item) => {
    rows.push(
      `<div class="stack-item"><span>${escapeHtml(t("summary.toolAction"))}</span><strong>${escapeHtml(
        `${humanizeRuntimeToken(item.tool_name || item.call_id || "工具")} · ${humanizeRuntimeToken(item.status || t("common.unknown"))}`,
      )}</strong></div>`,
    );
  });
  return rows.length ? rows.join("") : `<div class="empty-state">${escapeHtml(t("summary.noToolDetails"))}</div>`;
}

function formatWhyDriverRows(rows) {
  if (!rows.length) {
    return `<div class="empty-state">${escapeHtml(t("summary.noContributions"))}</div>`;
  }
  return rows
    .slice(0, 4)
    .map((item) => `<div class="stack-item insight-card"><span>${escapeHtml(humanizeFunctionName(item.agent_name || item.module_name || item.name || "driver"))}</span><strong>${escapeHtml(humanizeRuntimeText(item.reason || item.summary || "active"))}</strong></div>`)
    .join("");
}

function formatBlockedFunctions(whyNot, trace) {
  const blockedBy = [...(whyNot?.blocked_by || whyNot?.why_not?.blocked_by || [])];
  if (!blockedBy.length) {
    const gateOwners = Array.isArray(trace?.gate_decisions)
      ? trace.gate_decisions.filter((item) => item && item.allowed === false).map((item) => item.owner || item.stage).filter(Boolean)
      : [];
    blockedBy.push(...gateOwners);
  }
  if (!blockedBy.length) {
    return `<div class="empty-state">${escapeHtml(t("summary.noWhyNot"))}</div>`;
  }
  return blockedBy
    .slice(0, 6)
    .map((name) => `<div class="detail-row">
      <span class="detail-label">${escapeHtml(t("summary.functionBlocked"))}</span>
      <strong>${escapeHtml(humanizeFunctionName(name))}</strong>
      <div class="detail-copy">${escapeHtml(`${t("summary.translationLabel")} · ${humanizeFunctionName(name)}`)}</div>
    </div>`)
    .join("");
}

function formatMarkerLedgerRows(title, tone, markers) {
  if (!markers.length) {
    return "";
  }
  return `<section class="ledger-section tone-${escapeHtml(tone)}">
    <h4 class="subsection-title">${escapeHtml(title)}</h4>
    <div class="detail-list">
      ${markers
        .map(
          (item) => `<div class="detail-row">
            <span class="detail-label">${escapeHtml(humanizeFunctionName(item.stageLabel || item.stageKey))}</span>
            <strong>${escapeHtml(item.label)}</strong>
            <div class="detail-copy">${escapeHtml(item.detail)}</div>
          </div>`,
        )
        .join("")}
    </div>
  </section>`;
}

function formatTimelineTrack(trace, replay, meta, selectedAction) {
  const timeline = Array.isArray(trace?.pipeline_stages) ? trace.pipeline_stages : Array.isArray(trace?.proposal_summaries) ? trace.proposal_summaries : [];
  const renderPlan = trace?.render_plan || {};
  const expressionSummary = renderPlan.event_summary || trace?.style_profile?.summary || t("summary.noExpressionSummary");
  const counterfactualSummary = replay?.counterfactual_preview?.text || replay?.replayed_action || t("summary.noReplayPreview");
  const baseNodes = [
    {
      title: t("summary.resultReading"),
      body: t("summary.routeNarrative", {
        route: routeTypeLabel(meta.routeType),
        action: actionLabel(selectedAction),
        delivery: humanizeRuntimeToken(renderPlan.delivery_mode || t("summary.deliveryUnknown")),
      }),
    },
    { title: t("summary.finalExpression"), body: humanizeRuntimeText(expressionSummary) },
    { title: t("summary.replayPreview"), body: humanizeRuntimeText(counterfactualSummary) },
  ];
  const stageNodes = timeline.slice(0, 8).map((item) => ({
    title: humanizeRuntimeText(item.label || item.stage || item.agent_name || item.module_name || t("summary.timelineStage")),
    body: humanizeRuntimeText(item.summary || item.top_action || item.reason || item.trace_reason || item.mode || t("common.running")),
  }));
  const nodes = [...baseNodes, ...stageNodes];
  if (!nodes.length) {
    return `<div class="empty-state">${escapeHtml(t("summary.timelineEmpty"))}</div>`;
  }
  return `<div class="timeline-track">${nodes
    .map((node) => `<article class="timeline-node"><span>${escapeHtml(node.title)}</span><strong>${escapeHtml(node.body)}</strong></article>`)
    .join("")}</div>
    <details class="details-card">
      <summary>${escapeHtml(t("summary.timelineDetails"))}</summary>
      <div class="detail-list">
        ${stageNodes.length
          ? stageNodes
              .map((node) => `<div class="detail-row"><span class="detail-label">${escapeHtml(t("summary.timelineStage"))}</span><strong>${escapeHtml(node.title)}</strong><div class="detail-copy">${escapeHtml(node.body)}</div></div>`)
              .join("")
          : `<div class="empty-state">${escapeHtml(t("summary.timelineEmpty"))}</div>`}
      </div>
    </details>`;
}

function formatDeepLinks(roundId) {
  if (!roundId) {
    return `<div class="empty-state">${escapeHtml(t("summary.noDeepLinks"))}</div>`;
  }
  const links = [
    [buildWorkbenchRoundPath(roundId), t("deepLink.round")],
  ];
  return `<div class="jump-links">${links
    .map(([href, label]) => `<a class="jump-link" href="${escapeHtml(href)}" target="_blank" rel="noreferrer">${escapeHtml(label)}</a>`)
    .join("")}</div>`;
}

function renderAnalysis() {
  renderRoundList();
  if (elements.analysisViewCurrent) {
    elements.analysisViewCurrent.disabled = !state.console?.state?.current_round?.round_id;
  }
  if (elements.analysisViewLatest) {
    elements.analysisViewLatest.disabled = !(state.console?.recent_rounds || []).length;
  }
  const selected = state.selectedRoundPayload;
  setDataAttr(elements.analysisPage, "roundSelected", selected ? "true" : "false");
  setDataAttr(elements.analysisPage, "selectedRoundId", selected ? state.selectedRoundId : "");
  if (!selected) {
    elements.selectedRoundBadge.textContent = t("common.notSelectedRound");
    const baseMessage = (state.console?.recent_rounds || []).length
      ? t("summary.noRoundSelected")
      : state.bootstrap?.session_attached
        ? t("summary.noRoundData")
        : t("summary.noSessionAnalysis");
    resetTrendReplayState({ clearHoveredRound: true, clearFrames: true });
    elements.analysisLiveCopy.innerHTML = `<div class="empty-state">${escapeHtml(baseMessage)}</div>`;
    elements.analysisTrendStats.innerHTML = `<div class="empty-state">${escapeHtml(baseMessage)}</div>`;
    elements.analysisTrendLegend.innerHTML = `<div class="empty-state">${escapeHtml(baseMessage)}</div>`;
    elements.analysisTrendSvg.innerHTML = `<div class="empty-state">${escapeHtml(baseMessage)}</div>`;
    if (elements.analysisTrendReplayControls) {
      elements.analysisTrendReplayControls.innerHTML = `<div class="empty-state">${escapeHtml(baseMessage)}</div>`;
    }
    if (elements.analysisTrendFrameStrip) {
      elements.analysisTrendFrameStrip.innerHTML = `<div class="empty-state">${escapeHtml(baseMessage)}</div>`;
    }
    syncTrendReplayDataset(null);
    elements.analysisBrainflow.innerHTML = `<div class="empty-state">${escapeHtml(baseMessage)}</div>`;
    elements.analysisBrainflowOutput.innerHTML = `<div class="empty-state">${escapeHtml(baseMessage)}</div>`;
    elements.analysisWhy.innerHTML = `<div class="empty-state">${escapeHtml(baseMessage)}</div>`;
    elements.analysisWhyNot.innerHTML = `<div class="empty-state">${escapeHtml(baseMessage)}</div>`;
    elements.analysisTimeline.innerHTML = `<div class="empty-state">${escapeHtml(baseMessage)}</div>`;
    elements.analysisProbability.innerHTML = `<div class="empty-state">${escapeHtml(baseMessage)}</div>`;
    elements.analysisContributions.innerHTML = `<div class="empty-state">${escapeHtml(baseMessage)}</div>`;
    elements.analysisModelRoute.innerHTML = `<div class="empty-state">${escapeHtml(baseMessage)}</div>`;
    elements.analysisLinks.innerHTML = `<div class="empty-state">${escapeHtml(baseMessage)}</div>`;
    elements.analysisRawJson.textContent = t("summary.waitingRoundData");
    return;
  }
  const { trace, why, whyNot, contributions, probability, initiativeWhy, replay, thought } = selected;
  const selectedRound = roundRecord(state.selectedRoundId);
  const meta = selectedRound ? roundMeta(selectedRound) : deriveRoundMeta({}, selected);
  const renderPlan = trace?.render_plan || {};
  const selectedAction = trace?.sampled_action || state.selectedRoundAction;
  const whyNotSummary = humanizeRuntimeText(whyNot?.summary || whyNot?.why_not?.summary, t("summary.noWhyNot"));
  const flowView = buildAnalysisFlowView(
    {
      roundId: state.selectedRoundId,
      selectedAction,
      selectedPayload: selected,
      selectedRound,
    },
    viewModelHelpers(),
  );
  const selectedInitiativeView = buildRoundInitiativeView(selected, state.initiativeDistribution);
  const trendSeries = buildAnalysisTrendSeries({
    recentRounds: state.console?.recent_rounds || [],
    roundCatalog: state.roundCatalog,
  });
  const trendReplayFrames = buildTrendReplayFrames(trendSeries);
  state.trendReplayFrames = trendReplayFrames;
  if (!trendReplayFrames.length) {
    state.replayPlaying = false;
    state.replayRoundIndex = 0;
    state.replayHasStarted = false;
    stopTrendReplayStepTimer();
  } else if (state.replayRoundIndex >= trendReplayFrames.length) {
    state.replayRoundIndex = trendReplayFrames.length - 1;
  }
  const activeTrendRoundId = resolveActiveTrendRoundId(trendSeries);
  const activeTrendFrameIndex = trendReplayFrames.findIndex((frame) => Number(frame?.roundId) === Number(activeTrendRoundId));
  const activeTrendFrame = activeTrendFrameIndex >= 0 ? trendReplayFrames[activeTrendFrameIndex] : null;
  const brainflowStages = buildBrainflowStages({
    roundId: state.selectedRoundId,
    selectedAction,
    selectedPayload: selected,
  });
  const markerView = buildConflictRepairMarkers({
    roundId: state.selectedRoundId,
    selectedPayload: selected,
  });
  const brainflowOutput = buildBrainflowOutput({
    roundId: state.selectedRoundId,
    selectedAction,
    selectedPayload: selected,
  });

  elements.selectedRoundBadge.textContent = t("summary.currentRoundTag", {
    roundId: state.selectedRoundId,
    action: actionLabel(selectedAction),
  });
  elements.analysisLiveCopy.innerHTML = `
    <p class="reading-lead">${escapeHtml(flowView.decision_summary)}</p>
    <p class="lede">${escapeHtml(flowView.route_story)}</p>
    <div class="cortex-live-chip-grid">
      <span class="info-chip">${escapeHtml(`动作 ${actionLabel(selectedAction)}`)}</span>
      <span class="info-chip">${escapeHtml(`主动性 ${initiativeLabel(meta.initiativeState)}`)}</span>
      <span class="info-chip">${escapeHtml(`冲突强度 ${formatNumber(selectedRound?.conflict_score, 2)}`)}</span>
      <span class="info-chip">${escapeHtml(`冲突点 ${markerView.conflictMarkers.length}`)}</span>
      <span class="info-chip">${escapeHtml(`阻断 ${markerView.blockedMarkers.length}`)}</span>
      <span class="info-chip">${escapeHtml(`跳过 ${markerView.skippedMarkers.length}`)}</span>
      <span class="info-chip">${escapeHtml(`修复动作 ${markerView.repairMarkers.length}`)}</span>
    </div>
  `;
  elements.analysisTrendStats.innerHTML = renderTrendStats(trendSeries);
  elements.analysisTrendLegend.innerHTML = renderTrendLegend(trendSeries);
  elements.analysisTrendSvg.innerHTML = renderTrendSvg(trendSeries, {
    selectedRoundId: state.selectedRoundId,
    activeRoundId: activeTrendRoundId,
  });
  if (elements.analysisTrendReplayControls) {
    elements.analysisTrendReplayControls.innerHTML = renderTrendReplayControlsMarkup({
      hasFrames: trendReplayFrames.length > 0,
      isPlaying: state.replayPlaying,
      hasStarted: state.replayHasStarted,
      activeRoundLabel: activeTrendFrame?.roundLabel || "",
      activeIndex: activeTrendFrameIndex >= 0 ? activeTrendFrameIndex : 0,
      totalFrames: trendReplayFrames.length,
    });
  }
  if (elements.analysisTrendFrameStrip) {
    elements.analysisTrendFrameStrip.innerHTML = renderTrendFrameStripMarkup(trendReplayFrames, {
      activeRoundId: activeTrendRoundId,
    });
    requestAnimationFrame(() => syncTrendFrameStripScroll());
  }
  syncTrendReplayDataset(activeTrendRoundId);
  setDataAttr(elements.analysisTrendPanel, "hoveredRoundId", state.hoveredTrendRoundId ?? "");
  elements.analysisBrainflow.innerHTML = renderBrainflowStages(brainflowStages, markerView);
  elements.analysisBrainflowOutput.innerHTML = renderBrainflowOutput(brainflowOutput);
  const markerLedger = [
    formatMarkerLedgerRows("红色标注 · 冲突", "conflict", markerView.conflictMarkers),
    formatMarkerLedgerRows("橙色标注 · 阻断", "blocked", markerView.blockedMarkers),
    formatMarkerLedgerRows("灰色标注 · 跳过", "skipped", markerView.skippedMarkers),
    formatMarkerLedgerRows("蓝色标注 · 修复动作", "repair", markerView.repairMarkers),
  ].join("");

  elements.analysisWhy.innerHTML = `
    <p class="reading-lead">${escapeHtml(flowView.decision_summary)}</p>
    <p class="lede">${escapeHtml(flowView.route_story)}</p>
    <div class="insight-grid">${flowView.evidence_cards
      .slice(0, 3)
      .map((item) => `<div class="stack-item insight-card"><span>${escapeHtml(item.title)}</span><strong>${escapeHtml(item.body)}</strong></div>`)
      .join("")}</div>
    <details class="details-card">
      <summary>${escapeHtml(t("summary.evidenceCards"))}</summary>
      <div class="detail-list">
        ${flowView.evidence_cards.length
          ? flowView.evidence_cards
              .map((item) => `<div class="detail-row"><span class="detail-label">${escapeHtml(item.title)}</span><strong>${escapeHtml(item.body)}</strong>${renderRefLinks(item.refs)}</div>`)
              .join("")
          : `<div class="empty-state">${escapeHtml(t("summary.noContributions"))}</div>`}
      </div>
    </details>
  `;

  elements.analysisWhyNot.innerHTML = `
    <p class="reading-lead">${escapeHtml(whyNotSummary)}</p>
    <details class="details-card" open>
      <summary>${escapeHtml(t("summary.whyNotDrivers"))}</summary>
      <div class="detail-list">${formatBlockedFunctions(whyNot, trace)}</div>
    </details>
  `;

  elements.analysisTimeline.innerHTML = flowView.timeline_events.length
    ? `${markerLedger}
      <div class="timeline-track">${flowView.timeline_events
        .map((item) => `<article class="timeline-node"><span>${escapeHtml(item.title)}</span><strong>${escapeHtml(item.body)}</strong></article>`)
        .join("")}</div>
      <details class="details-card"><summary>${escapeHtml(t("summary.timelineDetails"))}</summary><div class="detail-list">${flowView.timeline_events
        .map((item) => `<div class="detail-row"><span class="detail-label">${escapeHtml(item.ts)}</span><strong>${escapeHtml(item.title)}</strong><div class="detail-copy">${escapeHtml(item.body)}</div>${renderRefLinks(item.refs)}</div>`)
        .join("")}</div></details>`
    : `${markerLedger}
      ${formatTimelineTrack(trace, replay, meta, selectedAction)}`;

  elements.analysisProbability.innerHTML = flowView.probability_summary.length
    ? flowView.probability_summary
        .map((item) => `<div class="stack-item"><span>${escapeHtml(item.label)}</span><strong>${escapeHtml(`${formatNumber(item.value, 3)}${item.delta ? ` · ${item.delta}` : ""}`)}</strong></div>`)
        .join("")
    : formatProbabilityLayers(probability?.probability_field || trace?.probability_field || thought?.action_field || {});
  elements.analysisContributions.innerHTML = formatContributionRows(contributions?.stacked_contributions || trace?.proposal_summaries || []);
  const modelTraces = Array.isArray(trace?.model_call_traces) ? trace.model_call_traces : [];
  const fallbackModelRoute = renderPlan?.identity_context?.model_label || renderPlan?.identity_context?.provider_label;
  const readableProviderLabel = humanizeRuntimeText(
    String(renderPlan?.identity_context?.provider_label || "--").replaceAll("/", " / "),
    renderPlan?.identity_context?.provider_label || "--",
  );
  elements.analysisModelRoute.innerHTML = modelTraces.length
    ? modelTraces
        .slice(0, 10)
        .map(
          (row) => `<div class="stack-item"><span>${escapeHtml(humanizeRuntimeToken(row.binding_key || row.skill_name || "model"))}</span><strong>${escapeHtml(`${humanizeRuntimeToken(row.binding_tier || "--")} / ${humanizeRuntimeToken(row.backend || "--")} / ${row.model || "--"}`)}</strong></div>`,
        )
        .join("")
    : fallbackModelRoute
      ? `<div class="stack-item"><span>${escapeHtml(t("summary.renderHint"))}</span><strong>${escapeHtml(
          `${bindingLabel("deep_renderer")} / ${readableProviderLabel} / ${renderPlan?.identity_context?.model_label || "--"}`,
        )}</strong></div>`
      : `<div class="empty-state">${escapeHtml(t("summary.noModelCalls"))}</div>`;
  elements.analysisLinks.innerHTML = `
    <section class="subsection">
      <h4 class="subsection-title">${escapeHtml(t("summary.readModels"))}</h4>
      <div class="subsection-stack">
        <div class="stack-item"><span>${escapeHtml(t("summary.apiPath"))}</span><strong>${escapeHtml(WORKBENCH_READ_MODEL_PATH)}</strong></div>
        <div class="stack-item"><span>${escapeHtml(t("summary.apiPath"))}</span><strong>${escapeHtml(buildWorkbenchRoundPath(state.selectedRoundId))}</strong></div>
        <div class="stack-item"><span>${escapeHtml(t("summary.apiPath"))}</span><strong>/subject/status</strong></div>
        <div class="stack-item"><span>${escapeHtml(t("summary.apiPath"))}</span><strong>/meaning/status</strong></div>
        <div class="stack-item"><span>${escapeHtml(t("summary.apiPath"))}</span><strong>/agency/status</strong></div>
      </div>
    </section>
    <section class="subsection">
      <h4 class="subsection-title">${escapeHtml(t("summary.initiative"))}</h4>
      <div class="subsection-stack">
        <div class="stack-item"><span>${escapeHtml(t("summary.roundInitiative"))}</span><strong>${escapeHtml(initiativeLabel(meta.initiativeState))}</strong></div>
        <div class="stack-item"><span>${escapeHtml(t("summary.distribution"))}</span><strong>${escapeHtml(
          `${humanizeRuntimeToken(selectedInitiativeView.proposal_type || "--")} / ${boolWord(selectedInitiativeView.should_send, "common.yes", "common.hold")}`,
        )}</strong></div>
        <div class="stack-item"><span>${escapeHtml(t("summary.why"))}</span><strong>${escapeHtml(humanizeRuntimeText(selectedInitiativeView.summary || t("summary.noWhy")))}</strong></div>
      </div>
    </section>
    <section class="subsection">
      <h4 class="subsection-title">${escapeHtml(t("summary.gateToolApproval"))}</h4>
      <div class="subsection-stack">
        ${formatGateRows(trace?.gate_decisions)}
        ${formatToolApprovalRows(state.sessionState, trace)}
      </div>
    </section>
    <section class="subsection">
      <h4 class="subsection-title">${escapeHtml(t("summary.traceReplayLinks"))}</h4>
      ${renderRefLinks(flowView.raw_links, t("summary.noDeepLinks"))}
    </section>
    <section class="subsection">
      <h4 class="subsection-title">${escapeHtml(t("summary.sourceLinks"))}</h4>
      <div class="subsection-stack">${formatSourceLinks(state.console?.source_links)}</div>
    </section>
  `;
  elements.analysisRawJson.textContent = JSON.stringify(
    translateDisplayJson({
      roundId: state.selectedRoundId,
      action: state.selectedRoundAction,
      thought,
      why,
      whyNot,
      trace,
      contributions,
      probability,
      replay,
    }),
    null,
    2,
  );
  setDataAttr(elements.analysisPage, "selectedAction", toDataToken(selectedAction, "unknown"));
  setDataAttr(elements.analysisPage, "selectedRoute", toDataToken(meta.routeType, "unknown"));
  setDataAttr(elements.analysisPage, "selectedInitiative", toDataToken(meta.initiativeState, "unknown"));
  setDataAttr(elements.analysisPage, "selectedActivity", toDataToken(meta.activity, "unknown"));
}

function populateBindingSelect(select, bindings, selectedValue) {
  const options = Object.keys(bindings || {});
  select.innerHTML = options
    .map((item) => optionMarkup(item, item, item === selectedValue))
    .join("");
}

function renderInnerSpace() {
  if (!elements.innerSpaceSelfNarrative || !elements.innerSpaceThinking || !elements.innerSpaceMemory || !elements.innerSpaceDream) {
    return;
  }
  const view = buildInnerSpaceView(
    {
      roundId: state.selectedRoundId,
      thought: state.innerSpace?.thought || state.selectedRoundPayload?.thought,
      memoryTop: state.innerSpace?.memoryTop || [],
      dreamOverview: state.innerSpace?.dreamOverview,
      monologueShow: state.innerSpace?.monologueShow,
      consoleData: state.console,
    },
    viewModelHelpers(),
  );
  if (elements.innerSpaceStatus) {
    elements.innerSpaceStatus.textContent = state.selectedRoundId ? `轮次 #${state.selectedRoundId}` : t("common.pending");
  }
  setDataAttr(elements.innerSpacePage, "roundSelected", state.selectedRoundId ? "true" : "false");
  setDataAttr(elements.innerSpacePage, "hasThinking", Array.isArray(view.thinking_stream) && view.thinking_stream.length ? "true" : "false");
  setDataAttr(elements.innerSpacePage, "hasMemory", Array.isArray(view.memory_recall) && view.memory_recall.length ? "true" : "false");
  setDataAttr(elements.innerSpacePage, "hasDream", Array.isArray(view.dream_fragments) && view.dream_fragments.length ? "true" : "false");
  elements.innerSpaceSelfNarrative.innerHTML = `
    <p class="reading-lead">${escapeHtml(view.self_narrative)}</p>
    ${renderRefLinks(view.cross_links_to_rounds.map((item) => ({ label: item.label, href: buildWorkbenchRoundPath(item.round_id) })), t("summary.noDeepLinks"))}
  `;
  elements.innerSpaceThinking.innerHTML = renderNarrativeFeed(view.thinking_stream, "当前还没有稳定成形的思考片段。");
  elements.innerSpaceMemory.innerHTML = renderNarrativeFeed(view.memory_recall, "当前没有特别突出的回忆片段。");
  elements.innerSpaceDream.innerHTML = renderNarrativeFeed(view.dream_fragments, "当前梦境整理还没有产生新的片段。");
}

function persistChatHistory(messages) {
  try {
    const sanitized = sanitizeChatMessages((Array.isArray(messages) ? messages : []).filter((item) => !item?.pending));
    if (sanitized.length) {
      localStorage.setItem(CHAT_HISTORY_STORAGE_KEY, JSON.stringify(sanitized));
      return;
    }
    localStorage.removeItem(CHAT_HISTORY_STORAGE_KEY);
  } catch {
    // Ignore localStorage failures in restricted browser contexts.
  }
}

function clearPersistedChatHistory() {
  try {
    localStorage.removeItem(CHAT_HISTORY_STORAGE_KEY);
  } catch {
    // Ignore localStorage failures in restricted browser contexts.
  }
}

function readPersistedChatHistory() {
  try {
    const raw = localStorage.getItem(CHAT_HISTORY_STORAGE_KEY);
    if (!raw) {
      return [];
    }
    return sanitizeChatMessages(JSON.parse(raw));
  } catch {
    return [];
  }
}

function applyDangerZoneLockState() {
  const locked = !state.dangerZoneUnlocked;
  if (elements.settingsDangerControls) {
    elements.settingsDangerControls.setAttribute("aria-disabled", String(locked));
  }
  DANGER_ZONE_FIELDS.forEach((key) => {
    const field = elements[key];
    if (!field) {
      return;
    }
    field.disabled = locked;
    field.closest(".toggle-row, .field-stack")?.classList.toggle("is-locked", locked);
  });
  if (elements.settingsDangerUnlock) {
    elements.settingsDangerUnlock.textContent = t(locked ? "settings.danger.unlock" : "settings.danger.unlocked");
    elements.settingsDangerUnlock.classList.toggle("is-unlocked", !locked);
  }
  if (elements.settingsDangerConfirm) {
    elements.settingsDangerConfirm.disabled = !locked;
  }
  if (elements.settingsDangerNotice) {
    elements.settingsDangerNotice.textContent = t(
      locked ? "settings.danger.notice.locked" : "settings.danger.notice.unlocked",
    );
  }
  if (elements.settingsPersonaReset) {
    elements.settingsPersonaReset.disabled = locked;
  }
  setDataAttr(elements.settingsPage, "dangerLocked", locked ? "true" : "false");
  setDataAttr(elements.settingsPage, "dangerUnlocked", locked ? "false" : "true");
}

function renderStackItems(items) {
  return items
    .map((item) => `<div class="stack-item"><span>${escapeHtml(item.label)}</span><strong>${escapeHtml(item.value)}</strong></div>`)
    .join("");
}

function renderScheduledTaskList(tasks) {
  if (!Array.isArray(tasks) || tasks.length === 0) {
    return `<div class="empty-state">${escapeHtml(t("scheduledTask.empty"))}</div>`;
  }
  return tasks
    .map(
      (task) => `<article class="scheduled-task-item" data-status="${escapeHtml(toDataToken(task.status, "idle"))}">
        <div class="scheduled-task-head">
          <div class="scheduled-task-copy">
            <strong>${escapeHtml(task.task_id)}</strong>
            <div class="scheduled-task-meta">
              <span>${escapeHtml(t("scheduledTask.skill"))} · ${escapeHtml(task.skill_name)}</span>
              <span>${escapeHtml(t("scheduledTask.nextRun"))} · ${escapeHtml(task.next_run_at)}</span>
            </div>
          </div>
          <span class="pill subtle scheduled-task-status">${escapeHtml(task.status_label)}</span>
        </div>
        ${task.excerpt ? `<div class="scheduled-task-excerpt"><span>${escapeHtml(task.excerpt_label)}</span><p>${escapeHtml(task.excerpt)}</p></div>` : ""}
      </article>`,
    )
    .join("");
}

function renderSettings() {
  const payload = state.settings;
  if (!payload) {
    renderSettingsStatus("settings.status.idle");
    setDataAttr(elements.settingsPage, "loaded", "false");
    return;
  }
  const autonomy = payload.autonomy || {};
  const newborn = payload.newborn || {};
  const organicMode = newborn.organic_mode || {};
  const subjectiveState = newborn.subjective_state || {};
  const models = payload.models || {};
  const bindings = models.module_model_bindings || {};
  const settingsView = buildSettingsConsoleView(
    {
      settings: payload,
      service: state.service,
      autonomy: state.autonomy,
      runtimeState: state.runtimeState,
      models: state.models,
      consoleData: state.console,
      selectedRoundId: state.selectedRoundId,
      agency: state.agency,
      performance: state.performance,
    },
    viewModelHelpers(),
  );
  elements.settingClearSafeMode.checked = Boolean(autonomy.clear_safe_mode_on_start);
  elements.settingInstinctFirst.checked = Boolean(organicMode.instinct_first);
  elements.settingSpontaneous.value = subjectiveState.spontaneous ?? 0;
  elements.settingBoundary.value = subjectiveState.boundary ?? 0;
  elements.settingFelt.value = Array.isArray(subjectiveState.felt) ? subjectiveState.felt.join("、") : "";
  elements.settingAllowCommit.checked = Boolean(autonomy.allow_commit);
  elements.settingNetworkEnabled.checked = Boolean(autonomy.network_enabled);
  elements.settingExternalIoEnabled.checked = Boolean(autonomy.external_io_enabled);
  elements.settingAutoSafeMode.checked = Boolean(autonomy.auto_safe_mode);
  elements.settingMaxRoundsPerHour.value = autonomy.max_rounds_per_hour ?? 0;
  elements.settingMaxToolActionsPerHour.value = autonomy.max_tool_actions_per_hour ?? 0;
  elements.settingFailureTripThreshold.value = autonomy.failure_trip_threshold ?? 3;
  elements.settingQuietHours.value = Array.isArray(autonomy.quiet_hours) ? autonomy.quiet_hours.join(",") : "";
  elements.settingLearningMode.value = autonomy.learning_mode || "";
  elements.settingTraceExternalLearning.value = String(Boolean(autonomy.trace_external_learning));
  elements.settingAllowedDomains.value = formatDelimitedLines(autonomy.allowed_network_domains);
  elements.settingWritableRoots.value = formatDelimitedLines(autonomy.writable_roots);
  elements.settingKnowledgeRoots.value = formatDelimitedLines(autonomy.knowledge_roots);
  elements.settingLearningLogDir.value = autonomy.learning_log_dir || "";
  populateBindingSelect(elements.settingBindingRenderer, models.model_tiers, bindings.cognitive_packet || "");
  populateBindingSelect(elements.settingBindingPlanner, models.model_tiers, bindings.deliberation || "");
  populateBindingSelect(elements.settingBindingPfc, models.model_tiers, bindings.tool_planner || "");
  populateBindingSelect(elements.settingBindingPerspective, models.model_tiers, bindings.deep_renderer || "");
  populateBindingSelect(elements.settingBindingConsolidation, models.model_tiers, bindings.consolidation_summarizer || "");
  if (elements.settingsConfigurationSummary) {
    elements.settingsConfigurationSummary.innerHTML = renderStackItems(settingsView.runtime_bindings);
  }
  if (elements.settingsDiagnosticsSummary) {
    elements.settingsDiagnosticsSummary.innerHTML = renderStackItems([
      ...settingsView.memory_dream_controls,
      ...settingsView.diagnostic_panels
        .filter((panel) => panel.title !== "计划任务")
        .flatMap((panel) => panel.rows.map((row) => ({ label: `${panel.title} · ${row.label}`, value: row.value }))),
    ]);
  }
  if (elements.settingsScheduledSummary) {
    elements.settingsScheduledSummary.innerHTML = renderStackItems(settingsView.scheduled_task_surface.summary);
  }
  if (elements.settingsScheduledList) {
    elements.settingsScheduledList.innerHTML = renderScheduledTaskList(settingsView.scheduled_task_surface.tasks);
  }
  if (elements.settingsDangerSummary) {
    elements.settingsDangerSummary.innerHTML = renderStackItems([
      ...settingsView.autonomy_controls,
      ...settingsView.danger_zone_actions.map((item) => ({ label: item.title, value: item.body })),
    ]);
  }
  elements.settingsModelSummary.innerHTML = renderStackItems([
    [t("settings.models.supported"), Array.isArray(models.supported_backends) ? models.supported_backends.join(" / ") : t("common.none")],
    [t("settings.models.local"), models.model_tiers?.local_model?.enabled === false ? t("common.closed") : t("common.open")],
    [t("settings.binding.renderer"), bindings.cognitive_packet || "--"],
    [t("settings.binding.planner"), bindings.deliberation || "--"],
    [t("settings.binding.pfc"), bindings.tool_planner || "--"],
    [t("settings.binding.perspective"), bindings.deep_renderer || "--"],
    [t("settings.binding.consolidation"), bindings.consolidation_summarizer || "--"],
  ].map(([label, value]) => ({ label, value })));
  applyDangerZoneLockState();
  renderSettingsStatus("settings.status.loaded");
  setDataAttr(elements.settingsPage, "loaded", "true");
  setDataAttr(elements.settingsPage, "learningMode", toDataToken(autonomy.learning_mode || "none"));
  setDataAttr(elements.settingsPage, "allowCommit", autonomy.allow_commit ? "true" : "false");
  setDataAttr(elements.settingsPage, "networkEnabled", autonomy.network_enabled ? "true" : "false");
  setDataAttr(elements.settingsPage, "externalIoEnabled", autonomy.external_io_enabled ? "true" : "false");
}

function buildSettingsPayload() {
  return {
    autonomy: {
      clear_safe_mode_on_start: elements.settingClearSafeMode.checked,
      learning_mode: elements.settingLearningMode.value.trim(),
      network_enabled: elements.settingNetworkEnabled.checked,
      external_io_enabled: elements.settingExternalIoEnabled.checked,
      allow_commit: elements.settingAllowCommit.checked,
      allowed_network_domains: parseDelimitedText(elements.settingAllowedDomains.value),
      writable_roots: parseDelimitedText(elements.settingWritableRoots.value),
      knowledge_roots: parseDelimitedText(elements.settingKnowledgeRoots.value),
      learning_log_dir: elements.settingLearningLogDir.value.trim(),
      trace_external_learning: elements.settingTraceExternalLearning.value === "true",
      max_rounds_per_hour: Number(elements.settingMaxRoundsPerHour.value || 0),
      max_tool_actions_per_hour: Number(elements.settingMaxToolActionsPerHour.value || 0),
      quiet_hours: parseDelimitedText(elements.settingQuietHours.value)
        .map((item) => Number(item))
        .filter((item) => Number.isInteger(item) && item >= 0 && item <= 23),
      failure_trip_threshold: Number(elements.settingFailureTripThreshold.value || 3),
      auto_safe_mode: elements.settingAutoSafeMode.checked,
    },
    newborn: {
      disable_safe_mode_lock: elements.settingClearSafeMode.checked,
      organic_mode: {
        enabled: true,
        instinct_first: elements.settingInstinctFirst.checked,
      },
      subjective_state: {
        felt: parseDelimitedText(elements.settingFelt.value),
        spontaneous: Number(elements.settingSpontaneous.value || 0),
        boundary: Number(elements.settingBoundary.value || 0),
        reject_all: 0,
        meaning_made: [],
      },
    },
    models: {
      module_model_bindings: {
        ...(state.settings?.models?.module_model_bindings || {}),
        cognitive_packet: elements.settingBindingRenderer.value,
        deliberation: elements.settingBindingPlanner.value,
        tool_planner: elements.settingBindingPfc.value,
        deep_renderer: elements.settingBindingPerspective.value,
        consolidation_summarizer: elements.settingBindingConsolidation.value,
      },
    },
  };
}

async function saveSettings() {
  renderSettingsStatus("settings.status.saving");
  try {
    state.settings = await postJson("/settings", buildSettingsPayload());
    renderSettings();
    state.controlNotice = t("summary.settingsSaved");
    await loadWorkbench({ preserveControlNotice: true });
  } catch (error) {
    renderSettingsStatus("settings.status.failed");
    state.controlNotice = t("summary.settingsFailed", { message: error.message });
    renderHealthStrip();
  }
}

function renderChat() {
  if (!state.chatMessages.length) {
    const persisted = readPersistedChatHistory();
    if (persisted.length) {
      state.chatMessages = persisted;
      state.chatLocalMutationAt = state.chatLocalMutationAt || latestChatTimestamp(persisted);
      if (!state.chatSessionState?.session?.session_id) {
        state.chatSessionState = {
          session: {
            session_id: state.chatSessionId,
            transcript_lines: [],
          },
        };
      }
    }
  }
  persistChatHistory(state.chatMessages);
  const assistantLabel = deriveRuntimeChatName({
    subject: state.subject,
    runtimeState: state.runtimeState,
    settings: state.settings,
    fallbackLabel: t("summary.assistant"),
  });
  if (elements.chatClear) {
    elements.chatClear.disabled = state.chatSendInFlight || !state.chatMessages.length;
  }
  if (elements.chatSend) {
    elements.chatSend.disabled = state.chatSendInFlight;
  }
  setDataAttr(elements.chatPage, "hasMessages", state.chatMessages.length ? "true" : "false");
  setDataAttr(elements.chatPage, "messageCount", state.chatMessages.length);
  setDataAttr(elements.chatMessages, "empty", state.chatMessages.length ? "false" : "true");
  elements.chatSessionState.textContent = state.chatSessionState?.session?.session_id
    ? t(state.chatMessages.length ? "chat.sessionReady" : "chat.sessionEmpty", { count: state.chatMessages.length })
    : t("summary.sessionMissing");
  elements.chatMessages.innerHTML = state.chatMessages.length
    ? state.chatMessages
        .map(
          (item) => `<article class="message" data-message-role="${escapeHtml(item.role)}">
            <div class="message-head">
              <div class="message-role">${escapeHtml(item.role === "assistant" ? assistantLabel : t("summary.user"))}</div>
              ${item.timestamp ? `<time class="message-timestamp" datetime="${escapeHtml(item.timestamp)}">${escapeHtml(formatChatTimestamp(item.timestamp, state.locale))}</time>` : ""}
            </div>
            <div class="message-bubble">
              <div class="message-body">${escapeHtml(item.text)}</div>
            </div>
            ${item.reasonSummary ? `<details class="details-card message-details"><summary>${escapeHtml(t("chat.reasonSummary"))}</summary><div class="detail-list"><div class="detail-row"><strong>${escapeHtml(item.reasonSummary)}</strong>${item.memoryHint ? `<div class="detail-copy">${escapeHtml(item.memoryHint)}</div>` : ""}${renderRefLinks(item.refs)}</div></div></details>` : ""}
          </article>`,
        )
        .join("")
    : `<div class="empty-state">${escapeHtml(t("summary.sendMessagePrompt"))}</div>`;
  requestAnimationFrame(() => {
    elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
  });
}

function restorePersistedChatHistory({ force = false } = {}) {
  const persisted = readPersistedChatHistory();
  if (!persisted.length) {
    return;
  }
  if (!force && state.chatMessages.length) {
    return;
  }
  state.chatMessages = persisted;
  state.chatLocalMutationAt = state.chatLocalMutationAt || latestChatTimestamp(persisted);
  if (!state.chatSessionState?.session?.session_id) {
    state.chatSessionState = {
      session: {
        session_id: state.chatSessionId,
        transcript_lines: [],
      },
    };
  }
  renderChat();
}

function renderAll() {
  renderHealthStrip();
  renderOverview();
  renderAnalysis();
  renderInnerSpace();
  renderChat();
  renderSettings();
  applyPresentationHooks();
}

async function hydrateSelectedRound(roundId, action) {
  const nextRoundId = normalizeRoundId(roundId);
  const previousRoundId = normalizeRoundId(state.selectedRoundId);
  if (nextRoundId !== null && previousRoundId !== nextRoundId) {
    resetTrendReplayState({ clearHoveredRound: true });
  }
  state.pendingRoundHydrationId = nextRoundId;
  state.selectedRoundId = roundId;
  state.selectedRoundAction = action || state.selectedRoundAction || "respond";
  const baseRound = roundRecord(roundId) || { round_id: roundId, sampled_action: state.selectedRoundAction };
  try {
    const detail = await loadWorkbenchRoundData(roundId, state.selectedRoundAction);
    state.roundCatalog[String(roundId)] = {
      ...(state.roundCatalog[String(roundId)] || {}),
      trace: detail?.trace || null,
      initiativeWhy: detail?.initiativeWhy || null,
      meta: deriveRoundMeta(baseRound, {
        trace: detail?.trace || null,
        initiativeWhy: detail?.initiativeWhy || null,
      }),
    };
    state.selectedRoundPayload = {
      trace: detail?.trace || null,
      initiativeWhy: detail?.initiativeWhy || null,
      thought: detail?.thought || null,
      why: detail?.why || null,
      contributions: detail?.contributions || null,
      probability: detail?.probability || null,
      whyNot: detail?.whyNot || null,
      replay: detail?.replay || null,
    };
    state.innerSpace = {
      ...state.innerSpace,
      thought: detail?.thought || null,
    };
  } catch (error) {
    state.selectedRoundPayload = {
      trace: null,
      initiativeWhy: null,
      thought: null,
      why: { why: { summary: t("summary.readFailed", { roundId, message: error.message }) } },
      contributions: null,
      probability: null,
      whyNot: null,
      replay: null,
    };
    state.innerSpace = {
      ...state.innerSpace,
      thought: null,
    };
  }
  state.pendingRoundHydrationId = null;
  renderAnalysis();
  renderInnerSpace();
}

function queueSelectedRoundHydration(round) {
  const roundId = normalizeRoundId(round?.round_id);
  if (!roundId || state.pendingRoundHydrationId === roundId || state.selectedRoundPayload) {
    return;
  }
  state.pendingRoundHydrationId = roundId;
  window.setTimeout(() => {
    void hydrateSelectedRound(roundId, round?.sampled_action || "respond");
  }, 0);
}

function buildInitiativeViewModel(runtimeState, initiativeStatus) {
  const latestTrigger =
    initiativeStatus?.latest_trigger ||
    runtimeState?.endogenous_scheduler_state?.recent_triggers?.slice?.(-1)?.[0] ||
    null;
  const proposal = initiativeStatus?.proposal || {};
  const memoryBacking = initiativeStatus?.memory_backing || initiativeStatus?.last_evaluation?.memory_backing || null;
  return {
    top_intent: initiativeStatus?.top_intent || proposal.top_intent || "",
    suppression_reason: initiativeStatus?.suppression_reason || proposal.suppression_reason || "",
    should_send: typeof initiativeStatus?.should_send === "boolean" ? initiativeStatus.should_send : Boolean(proposal.should_send),
    speech_cost: initiativeStatus?.speech_cost ?? proposal.speech_cost ?? null,
    proposal_type: proposal.proposal_type || initiativeStatus?.proposal_type || "",
    memory_backing: memoryBacking,
    latest_trigger: latestTrigger,
  };
}

function shouldLoadAnalysisData() {
  return state.page === "analysis-page";
}

function shouldLoadInnerSpaceData() {
  return state.page === "inner-space-page";
}

function shouldLoadSettingsData() {
  return state.page === "settings-page";
}

function shouldAutoRecoverRuntime(bootstrap) {
  const autonomy = bootstrap?.autonomy || {};
  if (runtimeAutoRecoveryAttempted || state.userPausedRuntime) {
    return false;
  }
  return autonomy?.stall_reason === "dashboard_pause" || (autonomy?.running && !autonomy?.runner_attached);
}

async function loadWorkbench({ preserveControlNotice = false, background = false } = {}) {
  if (loadWorkbenchPromise) {
    return loadWorkbenchPromise;
  }
  loadWorkbenchPromise = (async () => {
    const needAnalysis = shouldLoadAnalysisData();
    const needInnerSpace = shouldLoadInnerSpaceData();
    const needSettings = shouldLoadSettingsData();
    const needChatSession = state.page === "chat-page";
    const loadPrimaryReadModel = () =>
      loadWorkbenchReadModelEnvelope({
        analysis: needAnalysis,
        innerSpace: needInnerSpace,
        settings: needSettings,
        chatSessionId: needChatSession ? state.chatSessionId : "",
      });
    let readModel = await loadPrimaryReadModel().catch(() => null);
    let bootstrap = readModel?.bootstrap || null;
    let recoveredRuntime = false;
    if (!readModel || shouldAutoRecoverRuntime(bootstrap)) {
      runtimeAutoRecoveryAttempted = true;
      await postJson("/web/runtime/start", {});
      readModel = await loadPrimaryReadModel();
      bootstrap = readModel?.bootstrap || null;
      recoveredRuntime = true;
    }

    const previousSelectedRound = state.selectedRoundId;
    const summary = readModel.summary || {};
    const consoleData = readModel.console || {};
    const initiativeStatus = readModel.initiativeStatus || readModel.agency?.initiative || state.initiativeStatus;
    const settings = readModel.settings || state.settings;
    const memoryTop = readModel.memoryTop || readModel.inner_space?.memory_top || state.innerSpace?.memoryTop || [];
    const dreamOverview = readModel.dreamOverview || readModel.inner_space?.dream_overview || state.innerSpace?.dreamOverview || null;
    const monologueShow = readModel.monologueShow || readModel.inner_space?.monologue_show || state.innerSpace?.monologueShow || null;
    const sessionState = readModel.sessionState || null;
    const chatSessionState = readModel.chatSessionState || null;
    if (!preserveControlNotice && !recoveredRuntime) {
      state.controlNotice = "";
    }
    if (recoveredRuntime && (!preserveControlNotice || !state.controlNotice)) {
      state.controlNotice = t("summary.runtimeRecovered");
    }
    state.bootstrap = bootstrap;
    state.sessionState = sessionState;
    const runtimeState = summary?.runtime_state || state.runtimeState || {};
    const summaryConsole = summary?.console || {};
    const mergedConsole = needAnalysis && consoleData ? consoleData : summaryConsole;
    const settingsModels = settings?.models || state.settings?.models || {};
    const summaryModels = summary?.models || {};
    const moduleModelBindings = {
      ...(summaryModels.module_model_bindings || {}),
      ...(settingsModels.module_model_bindings || {}),
    };
    const models = {
      ...summaryModels,
      module_model_bindings: moduleModelBindings,
      tiers: settingsModels.model_tiers || summaryModels.tiers || state.models?.tiers || {},
      credential_present:
        summaryModels.credential_present ??
        state.models?.credential_present ??
        Boolean(Object.keys(moduleModelBindings).length),
    };
    const persistedChatMessages = readPersistedChatHistory();
    const chatHydration = resolveChatHydration({
      chatSessionId: state.chatSessionId,
      chatSessionState,
      observerSessionState: sessionState,
      inMemoryMessages: state.chatMessages,
      persistedMessages: persistedChatMessages,
      localMutationAt: state.chatLocalMutationAt,
    });
    state.chatSessionState = chatHydration.sessionState;
    state.chatMessages = chatHydration.messages;
    if (!state.chatLocalMutationAt && state.chatMessages.length) {
      state.chatLocalMutationAt = latestChatTimestamp(state.chatMessages);
    }
    state.service = bootstrap?.service || null;
    state.autonomy = bootstrap?.autonomy || null;
    state.runtimeState = runtimeState;
    state.models = models;
    state.console = mergedConsole;
    state.subject = readModel.subject || state.subject;
    state.meaning = readModel.meaning || state.meaning;
    state.agency = readModel.agency || state.agency;
    state.performance = readModel.performance || state.performance;
    state.settings = settings || state.settings;
    state.initiativeStatus = initiativeStatus;
    state.initiativeDistribution = buildInitiativeViewModel(runtimeState, initiativeStatus);
    state.innerSpace = {
      ...state.innerSpace,
      memoryTop,
      dreamOverview,
      monologueShow,
    };
    const currentRound = mergedConsole?.state?.current_round?.round_id;
    const currentAction = mergedConsole?.action_field?.winner?.action || mergedConsole?.why_not?.action || mergedConsole?.state?.current_round?.sampled_action || "respond";
    const fallbackRound = mergedConsole?.recent_rounds?.[0]?.round_id;
    const fallbackAction = mergedConsole?.recent_rounds?.[0]?.sampled_action || currentAction;
    const nextRound = state.selectedRoundId || currentRound || fallbackRound;
    const nextAction = state.selectedRoundAction || currentAction || fallbackAction;
    renderAll();
    if (needAnalysis) {
      scheduleRoundCatalogWarm(mergedConsole?.recent_rounds || []);
    }
    if (needAnalysis && nextRound && (!background || nextRound !== previousSelectedRound || !state.selectedRoundPayload)) {
      await hydrateSelectedRound(nextRound, nextAction);
    }
  })();
  try {
    await loadWorkbenchPromise;
  } finally {
    loadWorkbenchPromise = null;
  }
}

async function sendChat() {
  const text = elements.chatInput.value.trim();
  if (!text) {
    return;
  }
  if (state.chatSendInFlight) {
    state.controlNotice = t("chat.sendBusy");
    renderHealthStrip();
    return;
  }
  state.chatSendInFlight = true;
  state.controlNotice = "";
  const sentAt = markChatMutation();
  if (!state.chatSessionState?.session?.session_id) {
    state.chatSessionState = {
      session: {
        session_id: state.chatSessionId,
        transcript_lines: [],
      },
    };
  }
  state.chatMessages.push(withChatTimestamp({ role: "user", text }, sentAt));
  state.chatMessages.push(withChatTimestamp({ role: "assistant", text: t("chat.pendingReply"), pending: true }, sentAt));
  renderChat();
  elements.chatInput.value = "";
  try {
    if (state.autonomy?.stall_reason === "dashboard_pause" && !state.userPausedRuntime) {
      state.userPausedRuntime = false;
      await runRuntimeAction("/web/runtime/start", {});
    }
    const queued = await enqueueWorkbenchUserTurn(state.chatSessionId, text);
    let lastEventId = Number(queued?.last_event_id || 0);
    let nextRound = null;
    let assistantText = "";
    let surfacedLongReply = false;
    let assistantFinalReceived = false;

    while (true) {
      const events = await pollWorkbenchSessionEventsOnce(state.chatSessionId, lastEventId);
      if (!events.length) {
        if (!surfacedLongReply) {
          surfacedLongReply = true;
          replacePendingAssistantChatMessage({
            role: "assistant",
            text: t("chat.pendingLongReply"),
            timestamp: nowIso(),
            pending: true,
          });
          renderChat();
        }
        continue;
      }
      for (const event of events) {
        lastEventId = Math.max(lastEventId, Number(event?.event_id || 0));
        const eventType = String(event?.type || "");
        const eventRoundId = resolveChatEventRoundId(event);
        if (eventRoundId !== null) {
          nextRound = eventRoundId;
        }
        if (eventType === "error") {
          throw new Error(String(event?.message || "session event failed"));
        }
        if (eventType === "assistant_token") {
          assistantText = String(event?.message || event?.delta || assistantText).trim();
          if (assistantText) {
            replacePendingAssistantChatMessage({
              role: "assistant",
              text: assistantText,
              timestamp: nowIso(),
              pending: true,
            });
            renderChat();
          }
          continue;
        }
        if (eventType === "assistant_final") {
          assistantText = String(event?.message || assistantText).trim();
          assistantFinalReceived = true;
          break;
        }
      }
      if (assistantFinalReceived) {
        break;
      }
    }

    if (nextRound) {
      state.selectedRoundId = nextRound;
    }
    const chatView = buildChatTurnView(
      {
        assistantText: assistantText || t("common.none"),
        selectedPayload: null,
        selectedAction: state.selectedRoundAction || "respond",
        roundId: nextRound,
        innerSpace: state.innerSpace,
      },
      viewModelHelpers(),
    );
    replacePendingAssistantChatMessage({
      role: "assistant",
      text: chatView.assistant_message,
      timestamp: nowIso(),
    });
    renderChat();

    void loadWorkbench({ preserveControlNotice: true, background: true })
      .then(() => {
        const refreshedRound = nextRound || state.console?.state?.current_round?.round_id || state.console?.recent_rounds?.[0]?.round_id;
        const refreshedAction = roundRecord(refreshedRound)?.sampled_action || state.console?.state?.current_round?.sampled_action || state.console?.action_field?.winner?.action || "respond";
        if (!refreshedRound) {
          return;
        }
        state.selectedRoundId = refreshedRound;
        state.selectedRoundAction = refreshedAction;
        return hydrateSelectedRound(refreshedRound, refreshedAction)
          .then(() => {
          const detailedChatView = buildChatTurnView(
            {
              assistantText: assistantText || t("common.none"),
              selectedPayload: state.selectedRoundPayload,
              selectedAction: refreshedAction,
              roundId: refreshedRound,
              innerSpace: state.innerSpace,
            },
            viewModelHelpers(),
          );
          upsertAssistantChatMessage({
            role: "assistant",
            text: detailedChatView.assistant_message,
            timestamp: nowIso(),
            reasonSummary: detailedChatView.implicit_reason_summary,
            memoryHint: detailedChatView.contextual_memory_hint,
            refs: detailedChatView.expandable_evidence_refs,
          });
          renderChat();
        })
          .catch(() => {});
      })
      .catch(() => {});
  } catch (error) {
    replacePendingAssistantChatMessage({
      role: "assistant",
      text: t("summary.talkFailed", { message: error.message }),
      timestamp: nowIso(),
    });
    renderChat();
  } finally {
    state.chatSendInFlight = false;
    renderChat();
  }
}

async function clearChatHistory() {
  if (!window.confirm(t("chat.clearConfirm"))) {
    return;
  }
  try {
    await postJson("/web/session/event", {
      type: "close_session",
      session_id: state.chatSessionId,
      purge: true,
    });
    await postJson("/web/session/start", {
      session_id: state.chatSessionId,
      persist_current: false,
    });
    state.chatSessionState = {
      session: {
        session_id: state.chatSessionId,
        transcript_lines: [],
      },
    };
    markChatMutation();
    state.chatMessages = [];
    clearPersistedChatHistory();
    state.controlNotice = t("chat.sessionCleared");
    renderChat();
    renderHealthStrip();
  } catch (error) {
    state.controlNotice = t("chat.sessionResetFailed", { message: error.message });
    renderHealthStrip();
  }
}

async function resetPersona() {
  if (!window.confirm(t("settings.personaResetConfirm"))) {
    return;
  }
  try {
    const payload = await postJson("/persona/reset", {});
    await postJson("/web/session/event", {
      type: "close_session",
      session_id: state.chatSessionId,
      purge: true,
    }).catch(() => {});
    await postJson("/web/session/start", {
      session_id: state.chatSessionId,
      persist_current: false,
    }).catch(() => {});
    state.dangerZoneUnlocked = false;
    state.chatSessionState = {
      session: {
        session_id: state.chatSessionId,
        transcript_lines: [],
      },
    };
    state.chatMessages = [];
    markChatMutation();
    clearPersistedChatHistory();
    state.selectedRoundId = null;
    state.selectedRoundAction = "";
    state.selectedRoundPayload = null;
    resetTrendReplayState({ clearHoveredRound: true, clearFrames: true });
    state.controlNotice = t("settings.personaResetDone");
    renderHealthStrip();
    if (payload?.state) {
      await loadWorkbench({ preserveControlNotice: true });
    } else {
      renderAll();
    }
  } catch (error) {
    state.controlNotice = t("settings.personaResetFailed", { message: error.message });
    renderHealthStrip();
  }
}

async function runRuntimeAction(path, payload = {}) {
  const response = await postJson(path, payload);
  if (response?.skipped) {
    state.controlNotice = t(`runtime.skipped.${response.reason}`);
    await loadWorkbench({ preserveControlNotice: true });
    return response;
  }
  await loadWorkbench();
  return response;
}

function ensureAutoRefreshLoop() {
  if (autoRefreshTimer) {
    return;
  }
  autoRefreshTimer = window.setInterval(() => {
    if (document.hidden) {
      return;
    }
    loadWorkbench({ preserveControlNotice: true, background: true }).catch(() => {});
  }, 8000);
}

function upsertAssistantChatMessage(message) {
  const lastMessage = state.chatMessages[state.chatMessages.length - 1];
  if (lastMessage?.role === "assistant" && lastMessage.text === message.text) {
    Object.assign(lastMessage, withChatTimestamp(message, lastMessage.timestamp || nowIso()));
    markChatMutation(lastMessage.timestamp || latestChatTimestamp([lastMessage]) || nowIso());
    return;
  }
  const stamped = withChatTimestamp(message);
  state.chatMessages.push(stamped);
  markChatMutation(stamped.timestamp || nowIso());
}

function replacePendingAssistantChatMessage(message) {
  for (let index = state.chatMessages.length - 1; index >= 0; index -= 1) {
    const candidate = state.chatMessages[index];
    if (candidate?.role === "assistant" && candidate.pending) {
      state.chatMessages[index] = {
        ...candidate,
        ...withChatTimestamp(message, candidate.timestamp || nowIso()),
        pending: Boolean(message?.pending),
      };
      markChatMutation(state.chatMessages[index].timestamp || nowIso());
      return;
    }
  }
  upsertAssistantChatMessage(message);
}

function wireEvents() {
  [elements.navOverview, elements.navAnalysis, elements.navInnerSpace, elements.navChat, elements.navSettings].filter(Boolean).forEach((button) => {
    button.addEventListener("click", () => setPage(button.dataset.pageTarget));
  });
  elements.jumpAnalysis.addEventListener("click", () => setPage("analysis-page"));
  elements.languageButtons.forEach((button) => {
    button.addEventListener("click", () => setLocale(button.dataset.lang));
  });
  elements.runtimeStart.addEventListener("click", async () => {
    state.userPausedRuntime = false;
    state.controlNotice = "";
    await runRuntimeAction("/web/runtime/start", {});
  });
  elements.runtimePause.addEventListener("click", async () => {
    state.userPausedRuntime = true;
    await runRuntimeAction("/web/runtime/pause", {});
  });
  elements.runtimeResume.addEventListener("click", async () => {
    state.userPausedRuntime = false;
    state.controlNotice = "";
    await runRuntimeAction("/web/runtime/resume", {});
  });
  elements.runtimeWake.addEventListener("click", async () => {
    state.userPausedRuntime = false;
    state.controlNotice = "";
    await runRuntimeAction("/web/runtime/wake", {});
  });
  elements.filterRound.addEventListener("input", (event) => {
    state.filters.roundQuery = event.target.value.trim();
    renderRoundList();
  });
  elements.filterAction.addEventListener("change", (event) => {
    state.filters.action = event.target.value;
    renderRoundList();
  });
  elements.filterCause.addEventListener("change", (event) => {
    state.filters.cause = event.target.value;
    renderRoundList();
  });
  elements.filterRoute.addEventListener("change", (event) => {
    state.filters.route = event.target.value;
    renderRoundList();
  });
  elements.filterInitiative.addEventListener("change", (event) => {
    state.filters.initiative = event.target.value;
    renderRoundList();
  });
  elements.filterActivity.addEventListener("change", (event) => {
    state.filters.activity = event.target.value;
    renderRoundList();
  });
  elements.overviewRoundExpand?.addEventListener("click", () => {
    state.roundListExpanded = !state.roundListExpanded;
    renderOverview();
    renderRoundList();
  });
  elements.analysisRoundToggle?.addEventListener("click", () => {
    state.roundListExpanded = !state.roundListExpanded;
    renderOverview();
    renderRoundList();
  });
  elements.overviewRoundSelect?.addEventListener("change", async (event) => {
    const roundId = Number(event.target.value || 0);
    if (roundId) {
      setPage("analysis-page");
      await hydrateSelectedRound(roundId, roundRecord(roundId)?.sampled_action || "respond");
    }
  });
  elements.analysisRoundSelect?.addEventListener("change", async (event) => {
    const roundId = Number(event.target.value || 0);
    if (roundId) {
      await hydrateSelectedRound(roundId, roundRecord(roundId)?.sampled_action || "respond");
    }
  });
  elements.analysisRoundList.addEventListener("click", async (event) => {
    const button = event.target.closest("[data-round-row]");
    if (!button) {
      return;
    }
    await hydrateSelectedRound(Number(button.dataset.roundId), button.dataset.roundAction);
  });
  elements.analysisRoundList.addEventListener("mouseover", (event) => {
    const roundId = roundIdFromTimelineTarget(event.target);
    if (roundId !== null) {
      setHoveredTrendRound(roundId);
    }
  });
  elements.analysisRoundList.addEventListener("mouseleave", () => {
    clearHoveredTrendRound();
  });
  elements.analysisTimeline?.addEventListener("mouseover", (event) => {
    const roundId = roundIdFromTimelineTarget(event.target);
    if (roundId !== null) {
      setHoveredTrendRound(roundId);
    }
  });
  elements.analysisTimeline?.addEventListener("click", (event) => {
    const roundId = roundIdFromTimelineTarget(event.target);
    if (roundId !== null) {
      setHoveredTrendRound(roundId, { stopReplay: true });
    }
  });
  elements.analysisTimeline?.addEventListener("mouseleave", () => {
    clearHoveredTrendRound();
  });
  elements.analysisTrendPanel?.addEventListener("mouseover", (event) => {
    const roundId = roundIdFromTimelineTarget(event.target);
    if (roundId !== null) {
      setHoveredTrendRound(roundId);
    }
  });
  elements.analysisTrendPanel?.addEventListener("click", (event) => {
    const roundId = roundIdFromTimelineTarget(event.target);
    if (roundId !== null) {
      setHoveredTrendRound(roundId, { stopReplay: true });
    }
  });
  elements.analysisTrendPanel?.addEventListener("mouseleave", () => {
    clearHoveredTrendRound();
  });
  elements.analysisTrendReplayControls?.addEventListener("click", (event) => {
    const trigger = event.target.closest("[data-action]");
    if (!trigger) {
      return;
    }
    handleTrendReplayAction(trigger.dataset.action);
  });
  elements.analysisTrendFrameStrip?.addEventListener("mouseover", (event) => {
    const roundId = roundIdFromTimelineTarget(event.target);
    if (roundId !== null) {
      setHoveredTrendRound(roundId);
    }
  });
  elements.analysisTrendFrameStrip?.addEventListener("click", (event) => {
    const roundId = roundIdFromTimelineTarget(event.target);
    if (roundId === null) {
      return;
    }
    const nextIndex = state.trendReplayFrames.findIndex((frame) => Number(frame?.roundId) === Number(roundId));
    if (nextIndex >= 0) {
      state.replayRoundIndex = nextIndex;
    }
    setHoveredTrendRound(roundId, { stopReplay: true });
  });
  elements.analysisTrendFrameStrip?.addEventListener("mouseleave", () => {
    clearHoveredTrendRound();
  });
  elements.overviewRounds.addEventListener("click", async (event) => {
    const roundButton = event.target.closest("[data-round-jump]");
    if (roundButton) {
      setPage("analysis-page");
      await hydrateSelectedRound(Number(roundButton.dataset.roundJump), roundButton.dataset.roundAction || "respond");
      return;
    }
    const pageButton = event.target.closest("[data-page-jump]");
    if (pageButton) {
      setPage(pageButton.dataset.pageJump);
    }
  });
  elements.chatSend.addEventListener("click", sendChat);
  elements.chatClear?.addEventListener("click", clearChatHistory);
  elements.analysisViewRefresh?.addEventListener("click", () => {
    void loadWorkbench({ preserveControlNotice: true });
  });
  elements.analysisViewCurrent?.addEventListener("click", async () => {
    const currentRound = state.console?.state?.current_round?.round_id;
    const currentAction = state.console?.state?.current_round?.sampled_action || "respond";
    if (!currentRound) {
      return;
    }
    setPage("analysis-page");
    await hydrateSelectedRound(currentRound, currentAction);
  });
  elements.analysisViewLatest?.addEventListener("click", async () => {
    const latestRound = state.console?.recent_rounds?.[0];
    if (!latestRound?.round_id) {
      return;
    }
    setPage("analysis-page");
    await hydrateSelectedRound(latestRound.round_id, latestRound.sampled_action || "respond");
  });
  elements.settingsDangerUnlock?.addEventListener("click", () => {
    const confirmPhrase = t("settings.danger.confirm.placeholder");
    if ((elements.settingsDangerConfirm?.value || "").trim() !== confirmPhrase) {
      elements.settingsDangerNotice.textContent = `${t("settings.danger.notice.locked")} ${t("settings.danger.confirm")}：${confirmPhrase}`;
      return;
    }
    state.dangerZoneUnlocked = true;
    elements.settingsDangerConfirm.value = "";
    applyDangerZoneLockState();
  });
  elements.settingsPersonaReset?.addEventListener("click", () => {
    void resetPersona();
  });
  elements.saveSettings?.addEventListener("click", () => {
    void saveSettings();
  });
  elements.chatInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void sendChat();
    }
  });
}

function startClock() {
  updateClock();
  window.setInterval(updateClock, 1000);
}

function wireMotionPreference() {
  if (!window.matchMedia) {
    return;
  }
  motionPreferenceMedia = window.matchMedia("(prefers-reduced-motion: reduce)");
  syncMotionPreference();
  const handleChange = () => {
    syncMotionPreference();
  };
  if (typeof motionPreferenceMedia.addEventListener === "function") {
    motionPreferenceMedia.addEventListener("change", handleChange);
  } else if (typeof motionPreferenceMedia.addListener === "function") {
    motionPreferenceMedia.addListener(handleChange);
  }
}

setLocale(detectInitialLocale(), { persist: false });
state.chatMessages = readPersistedChatHistory();
if (state.chatMessages.length) {
  state.chatLocalMutationAt = latestChatTimestamp(state.chatMessages);
  state.chatSessionState = {
    session: {
      session_id: state.chatSessionId,
      transcript_lines: [],
    },
  };
}
wireEvents();
wireMotionPreference();
startClock();
loadWorkbench().catch((error) => {
  elements.topStatusMini.textContent = t("summary.workbenchLoadFailed", { message: error.message });
  renderAll();
});
ensureAutoRefreshLoop();
