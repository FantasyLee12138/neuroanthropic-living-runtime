const ACTION_LABELS = {
  respond: "回应",
  recall: "回忆",
  plan: "规划",
  clarify: "澄清",
  connect: "连接",
  wait: "等待",
  rest: "休整",
  wander: "游走",
  monologue: "独白",
  nothing: "静止",
  absorb: "吸收",
  die: "终止",
  short_reply: "短答",
};

const DISPLAY_TOKEN_LABELS = {
  roundId: "轮次",
  round_id: "轮次",
  run_id: "运行 ID",
  run_status: "运行状态",
  run_visible: "运行可见",
  run_blocking: "运行阻断",
  run_block_reason: "运行阻断原因",
  run_block_run_id: "阻断运行 ID",
  run_pending_approval: "待审批",
  run_dirty_worktree: "运行工作区脏状态",
  run_stop_reason: "停止原因",
  run_stale_dirty_worktree: "陈旧脏工作区",
  runtime_revision: "运行时修订",
  last_mutation_at: "最近变更时间",
  last_http_ok_at: "最近 HTTP 正常时间",
  last_runtime_activity_at: "最近运行活动时间",
  last_probe_error: "最近探测错误",
  action: "动作",
  sampled_action: "采样动作",
  selectedAction: "当前动作",
  thought: "思考摘要",
  why: "原因",
  whyNot: "未采纳路径",
  why_not: "未采纳路径",
  agent_name: "代理名称",
  module_name: "模块名称",
  ref: "引用",
  conflict: "冲突",
  name: "名称",
  trace: "过程追踪",
  contributions: "贡献叠加",
  probability: "概率层",
  replay: "回放",
  initiativeWhy: "主动性判断",
  initiative_why: "主动性判断",
  rendered_expression: "外显表达",
  render_plan: "表达规划",
  top_drivers: "主要驱动",
  blocked_by: "被阻断模块",
  timeline: "时间线",
  events: "事件",
  summary: "摘要",
  text: "文本",
  route: "路径",
  trigger: "触发",
  delivery_mode: "输出方式",
  model: "模型",
  query_kind: "查询语境",
  disclosure_intent: "暴露意图",
  planner: "规划层",
  mode: "模式",
  tool_level: "工具层",
  large_model: "大模型",
  small_model: "小模型",
  medium_model: "中模型",
  status: "状态",
  detail: "详情",
  phase: "阶段",
  phase_detail: "阶段细节",
  phase_started_at: "阶段开始时间",
  phase_age_seconds: "阶段持续秒数",
  phase_waiting: "阶段等待中",
  phase_stalled: "阶段停滞",
  last_progress_at: "最近推进时间",
  last_progress_age_seconds: "最近推进间隔秒数",
  updated_at: "更新时间",
  output: "输出",
  memory: "记忆调取",
  pfc: "前额叶候选",
  thalamus: "注意采样",
  identity_guard: "身份校验",
  conflict_repair: "冲突修复",
  output_gate: "输出门控",
  renderer: "表达层",
  chat: "聊天",
  internal: "内部",
  token: "令牌",
  speak: "表达",
  speech: "文字",
  text_mode: "文字",
  text_channel: "文字通道",
  interactive: "交互通道",
  observer: "观察器",
  heartbeat: "心跳",
  tick: "脉冲",
  endogenous: "内源",
  endogenous_tick: "内源脉冲",
  endogenous_replay: "内源回放",
  endogenous_light: "内源轻触发",
  endogenous_deep: "内源深读",
  silent_but_active: "静默但活跃",
  field_imbalance: "场域失衡",
  task_focused: "任务聚焦",
  no_fresh_session: "没有新鲜会话",
  completion: "完成",
  comfort: "舒缓",
  exploration: "探索",
  meaning: "意义",
  relation: "关系",
  general: "常规语境",
  general_exchange: "常规交流",
  repo_scan: "仓库巡检",
  check_relation: "维持关系脉冲",
  express_state: "表达当前状态",
  memory_pull: "记忆调取",
  desire_state: "欲望状态",
  fallback: "回退模型",
  applied: "已应用",
  active: "活跃",
  idle: "空闲",
  running: "运行中",
  ready: "就绪",
  healthy: "健康",
  degraded: "降级",
  stalled: "停滞",
  paused: "已暂停",
  stopped: "已停止",
  blocked: "已阻断",
  blocking: "阻断中",
  background: "后台",
  foreground: "前台",
  present: "已接入",
  missing: "缺失",
  visible: "可见",
  hidden: "隐藏",
  current: "当前",
  recent: "最近",
  latest: "最新",
  user: "用户",
  pointer: "指针",
  turn: "轮次",
  active_turn_sessions: "活跃轮次会话",
  observer_turn_active: "观察页轮次占用",
  autonomy_runner_alive: "自治线程存活",
  autonomy_runner_stalled: "自治线程停滞",
  autonomy_runner_stall_seconds: "自治停滞秒数",
  autonomy_runner_stall_threshold_seconds: "自治停滞阈值秒数",
  runtime_serial_active: "串行写入活跃",
  runtime_busy: "运行体繁忙",
  fresh_session_available: "有新鲜会话",
  selected_session_id: "已选会话",
  selected_session_age_seconds: "已选会话年龄秒数",
  selected_session_reason: "已选会话原因",
  session_fresh: "会话新鲜",
  initiative_runtime_busy: "主动线程繁忙",
  monologue_runtime_busy: "独白线程繁忙",
  initiative_busy_skip_count: "主动跳过次数",
  monologue_busy_skip_count: "独白跳过次数",
  stall_reason: "停滞原因",
  stall_reason_detail: "停滞原因细节",
  candidate_peak: "候选峰值",
  posterior_below_threshold: "后验低于阈值",
  dirty_worktree: "工作区脏状态",
  worktree: "工作区",
  checkpoint: "检查点",
  checkpoint_contract: "检查点合同",
  checkpoint_backed: "检查点支撑",
  checkpoint_backed_contract: "检查点回滚合同",
  create: "创建",
  rewind: "回退",
  rollback_binding: "回滚绑定",
  rollback_invalid_sigma: "回滚无效 sigma",
  switch_to_light_cache_mode: "切换到轻缓存模式",
  light_cache: "轻缓存",
  cache_mode: "缓存模式",
  contract_status: "合同状态",
  contract_only: "仅合同层",
  implemented: "已实现",
  registered: "已注册",
  reads_runtime_state: "读取运行时状态",
  writes_state: "写入状态",
  process_replace: "进程替换",
  replace_failed_agent_with_baseline: "失败代理回退到基线",
  fallback_contract: "回退合同",
  baseline_source: "基线来源",
  baseline_or_neutral_delta: "基线或中性差值",
  neutral: "中性",
  delta: "差值",
  uses_checkpoint_create: "使用检查点创建",
  uses_checkpoint_rewind: "使用检查点回退",
  state: "状态",
  safe_mode: "保护模式",
  last_checkpoint_id: "最近检查点",
  observer_heartbeat: "观察器心跳",
  observer_main: "观察主会话",
  observermain: "观察主会话",
  current_pointer_recent_user_turn: "当前指向最近用户轮次",
  current_pointer_recent_user_session: "当前指向最近用户会话",
  pfcagent: "前额叶控制",
  perspectivemodel: "视角模型",
  emergentactionsketch: "涌现行动草图",
  affectresiduedrive: "情绪余波驱力",
  goalcontinuationdrive: "目标延续驱力",
  relationboundarygate: "关系边界门",
  forced_mode_switch: "强制模式切换",
  initiativeruntime: "主动运行时",
  expressiveimpulse: "表达冲动",
  monologuestream: "独白流",
  salienceagent: "显著性代理",
  bodystateagent: "身体状态代理",
  emotionagent: "情绪代理",
  cue: "线索",
  lock_score: "锁定分数",
  motivation_pool: "动机池",
  task_goal: "任务目标",
  fail_score: "失败分数",
  peak: "主峰",
  gate: "门控",
  intent: "意图",
  withhold: "保守",
  penalized: "受抑制",
  stay_silent: "保持静默",
  cost: "成本",
  p: "概率",
  memory_backing: "记忆牵引",
  top_intent: "主意图",
  instinctfield: "本能场",
  relationshipagent: "关系代理",
  resourceagent: "资源代理",
  habitagent: "习惯代理",
  desireagent: "欲望代理",
  dmnagent: "默认模式网络代理",
  hippocampusagent: "海马体代理",
  valueagent: "价值代理",
  authenticitypolicy: "身份真实性策略",
  behaviorplausibilityguard: "行为合理性守卫",
  memorywritegate: "记忆写入门",
};

const SENTENCE_REPLACEMENTS = [
  [/authenticity guard requested resample/gi, "身份校验要求重新采样"],
  [/authenticity guard resample/gi, "身份校验已触发重新采样"],
  [/authenticity guard fallback/gi, "身份校验要求回退输出"],
  [/repair transition/gi, "修复阶段切换"],
  [/critical conflict/gi, "显性冲突"],
  [/endogenous trigger/gi, "内源触发"],
  [/mode=silent/gi, "当前处于静默模式"],
  [/no fresh session/gi, "没有新鲜会话"],
  [/task focused/gi, "任务聚焦"],
  [/silent but active/gi, "静默但活跃"],
  [/observer heartbeat/gi, "观察器心跳"],
  [/endogenous tick/gi, "内源脉冲"],
  [/contract only/gi, "仅合同层"],
  [/checkpoint backed contract/gi, "检查点回滚合同"],
  [/no fresh external cue but endogenous activation stays live/gi, "没有新的外部线索，但内源激活仍在持续"],
  [/focus lock or affect residue remains elevated/gi, "注意锁定或情绪余波仍然偏高"],
  [/autonomy started in tool_level after leaving safe mode/gi, "离开保护模式后，自治已在工具层启动"],
  [/inspect the repository and summarize notable changes in a read-only way\.?/gi, "只读巡检仓库并总结值得注意的变化"],
  [/inspect the repository and current runtime state, then identify the next read-only self-directed step\.?/gi, "检查仓库与当前运行状态，再确定下一步只读自驱动作"],
  [/monologue status/gi, "独白状态"],
  [/hard block triggered/gi, "已触发硬阻断"],
];

function lookupRuntimeTokenLabel(raw) {
  const direct = DISPLAY_TOKEN_LABELS[raw];
  if (direct) {
    return direct;
  }
  const canonical = canonicalDisplayToken(raw);
  return Object.entries(DISPLAY_TOKEN_LABELS).find(([key]) => canonicalDisplayToken(key) === canonical)?.[1] || "";
}

function humanizeCompositeToken(raw) {
  const segments = String(raw || "")
    .split(/[:/_-]+/)
    .map((segment) => segment.trim())
    .filter(Boolean);
  if (segments.length < 2 || segments.some((segment) => /\d/.test(segment))) {
    return "";
  }
  const translated = segments.map((segment) => lookupRuntimeTokenLabel(segment) || ACTION_LABELS[segment] || segment);
  const changedCount = translated.filter((segment, index) => segment !== segments[index]).length;
  if (changedCount < Math.ceil(segments.length / 2)) {
    return "";
  }
  return translated.join(" · ");
}

function canonicalDisplayToken(value) {
  return String(value || "")
    .trim()
    .toLowerCase()
    .replaceAll(/([a-z0-9])([A-Z])/g, "$1_$2")
    .replaceAll(/[^a-z0-9]+/g, "");
}

function looksDisplayIdentifier(value) {
  const text = String(value || "").trim();
  if (!text) {
    return false;
  }
  const compact = !/\s/.test(text);
  return (
    text.startsWith("/") ||
    text.startsWith("http://") ||
    text.startsWith("https://") ||
    text.startsWith("round://") ||
    (compact && /^[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)+$/.test(text)) ||
    text.includes("{") ||
    text.includes("[") ||
    /[A-Za-z0-9]+\/[A-Za-z0-9]/.test(text)
  );
}

function mostlyEnglish(value) {
  const text = String(value || "");
  const asciiLetters = (text.match(/[A-Za-z]/g) || []).length;
  const han = (text.match(/[\u4e00-\u9fff]/g) || []).length;
  return asciiLetters >= 10 && asciiLetters > han * 4;
}

export function actionLabel(value) {
  return ACTION_LABELS[value] || value || "未知动作";
}

export function humanizeRuntimeToken(value, fallback = "") {
  const raw = String(value || "").trim();
  if (!raw) {
    return fallback || "暂无";
  }
  const direct = lookupRuntimeTokenLabel(raw);
  if (direct) {
    return direct;
  }
  if (ACTION_LABELS[raw]) {
    return ACTION_LABELS[raw];
  }
  const composite = humanizeCompositeToken(raw);
  if (composite) {
    return composite;
  }
  if (raw.includes(":")) {
    const segments = raw.split(":").map((segment) => humanizeRuntimeToken(segment, segment));
    if (segments.some((segment, index) => segment !== raw.split(":")[index])) {
      return segments.join(" · ");
    }
  }
  return raw;
}

export function humanizeRuntimeText(value, fallback = "") {
  const raw = String(value || "").trim();
  if (!raw) {
    return fallback || "";
  }
  if (looksDisplayIdentifier(raw)) {
    return raw;
  }
  if (/^[A-Za-z][A-Za-z0-9_:-]*$/.test(raw)) {
    const direct = humanizeRuntimeToken(raw);
    if (direct !== raw) {
      return direct;
    }
  }
  let next = raw;
  for (const [pattern, replacement] of SENTENCE_REPLACEMENTS) {
    next = next.replace(pattern, replacement);
  }
  next = next.replace(/endogenous trigger ([a-z_:-]+)/gi, (_, token) => `内源触发 ${humanizeRuntimeToken(token, token)}`);
  next = next.replace(/endogenous:([a-z_:-]+)/gi, (_, token) => `内源 · ${humanizeRuntimeToken(token, token)}`);
  next = next.replace(/[A-Za-z][A-Za-z0-9_:-]*/g, (token) => {
    if (looksDisplayIdentifier(token)) {
      return token;
    }
    const mapped = humanizeRuntimeToken(token, token);
    return mapped;
  });
  if (fallback && mostlyEnglish(next)) {
    return fallback;
  }
  return next;
}

export function translateDisplayJson(value) {
  if (Array.isArray(value)) {
    return value.map((item) => translateDisplayJson(item));
  }
  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value).map(([key, item]) => [
        humanizeRuntimeToken(key, key),
        translateDisplayJson(item),
      ]),
    );
  }
  if (typeof value === "string") {
    return humanizeRuntimeText(value);
  }
  return value;
}

export function boolLabel(value, truthy = "是", falsy = "否") {
  return value ? truthy : falsy;
}

export function compactList(values, empty = "暂无") {
  if (!Array.isArray(values) || values.length === 0) {
    return empty;
  }
  return values.map((item) => String(item)).join(" / ");
}

export function formatNumber(value, digits = 2, empty = "--") {
  return Number.isFinite(value) ? Number(value).toFixed(digits) : empty;
}

export function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

export function maybeText(value, empty = "暂无") {
  if (typeof value === "string" && value.trim()) {
    return value.trim();
  }
  if (typeof value === "number" && Number.isFinite(value)) {
    return String(value);
  }
  return empty;
}

export function routeHealthLabel(service, autonomy) {
  if (autonomy?.stalled) {
    return "stalled";
  }
  if (service?.healthy && autonomy?.running) {
    return "running";
  }
  if (service?.healthy) {
    return "ready";
  }
  return "degraded";
}
