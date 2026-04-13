const ACTION_LABELS: Record<string, string> = {
  respond: "回应生成",
  rest: "静息回落",
  plan: "规划求解",
  recall: "记忆检索",
  absorb: "吸收沉积",
  monologue: "内心独白",
  wander: "游移漫游",
  nothing: "保持静默",
  die: "自主结束生命",
  inspect: "上下文检查",
  clarify: "澄清求证",
  connect: "关系联结",
  abort: "中止",
  dream: "离线整理",
  endogenous: "内源驱动",
};

const TOOL_LABELS: Record<string, string> = {
  read_file: "读取文件",
  write_file: "写入文件",
  run_shell: "执行命令",
  repo_scan: "仓库扫描",
  apply_patch: "补丁写入",
  skill_executor: "技能执行器",
  SkillExecutor: "技能执行器",
  skillexecutor: "技能执行器",
  exec_command: "命令执行器",
};

const RUN_STATUS_LABELS: Record<string, string> = {
  running: "运行中",
  active: "活跃",
  paused: "已暂停",
  idle: "空闲",
  completed: "已完成",
  done: "已完成",
  aborted: "已中止",
  failed: "异常",
  error: "异常",
  responded: "已回应",
  detached: "已分离",
  pending: "待处理",
  blocked: "阻塞",
};

const PERMISSION_MODE_LABELS: Record<string, string> = {
  ask: "需确认",
  plan: "规划模式",
  acceptEdits: "自动接纳修改",
  "workspace-write": "工作区写入",
  exec: "命令执行",
};

const TRANSCRIPT_MODE_LABELS: Record<string, string> = {
  compact: "紧凑",
  full: "完整",
};

const RISK_LEVEL_LABELS: Record<string, string> = {
  high: "高",
  medium: "中",
  low: "低",
};

const FOCUS_LABELS: Record<string, string> = {
  task: "任务定向",
  respond: "回应生成",
  wander: "注意游移",
  rest: "恢复回落",
  monologue: "内心独白态",
  boot: "启动校准",
  deep: "深度聚焦",
};

const COGNITIVE_MODE_LABELS: Record<string, string> = {
  interactive: "交互态",
  idle: "静息待命",
  sleep: "离线整理",
  safe: "审慎约束",
  plan: "规划态",
  reflective: "反思态",
  respond: "回应生成态",
  rest: "静息回落态",
  ForcedModeSwitch: "强制模式切换",
  forced_mode_switch: "强制模式切换",
};

function translateFromMap(value: unknown, labels: Record<string, string>, fallback: string): string {
  if (typeof value !== "string") {
    return fallback;
  }
  const trimmed = value.trim();
  if (!trimmed) {
    return fallback;
  }
  return labels[trimmed] ?? trimmed;
}

export function translateActionName(value: unknown, fallback = "尚未接入"): string {
  return translateFromMap(value, ACTION_LABELS, fallback);
}

export function translateToolName(value: unknown, fallback = "尚未接入"): string {
  return translateFromMap(value, TOOL_LABELS, fallback);
}

export function translateRunStatus(value: unknown, fallback = "尚未接入"): string {
  return translateFromMap(value, RUN_STATUS_LABELS, fallback);
}

export function translatePermissionMode(value: unknown, fallback = "尚未接入"): string {
  return translateFromMap(value, PERMISSION_MODE_LABELS, fallback);
}

export function translateTranscriptMode(value: unknown, fallback = "尚未接入"): string {
  return translateFromMap(value, TRANSCRIPT_MODE_LABELS, fallback);
}

export function translateRiskLevel(value: unknown, fallback = "尚未接入"): string {
  return translateFromMap(value, RISK_LEVEL_LABELS, fallback);
}

export function translateFocusState(value: unknown, fallback = "尚未接入"): string {
  return translateFromMap(value, FOCUS_LABELS, fallback);
}

export function translateCognitiveMode(value: unknown, fallback = "尚未接入"): string {
  return translateFromMap(value, COGNITIVE_MODE_LABELS, fallback);
}
