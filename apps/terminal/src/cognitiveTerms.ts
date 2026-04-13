import { translateBrainIdentifier, translateBrainIdentifierList } from "./brainLabels.js";

const TIMELINE_TYPE_LABELS: Record<string, string> = {
  stimulus: "外界刺激",
  memory_activation: "记忆激活",
  motivation_rise: "动机唤起",
  action_arbitration: "行动选择解算",
  token_gate: "言语生成门控",
  endogenous_tick: "内源驱动更新",
  dream_effect: "长期漂移",
};

const COGNITIVE_PHRASE_LABELS: Record<string, string> = {
  respond: "回应生成",
  rest: "静息回落",
  plan: "规划求解",
  recall: "记忆检索",
  inspect: "上下文检查",
  clarify: "澄清求证",
  connect: "关系联结",
  abort: "中止",
  clarity_first: "清晰度优先",
  winner_selected: "已完成胜出选择",
  coupling_checked: "表达门控已校验",
  token_coupling_checked: "表达门控已校验",
  internal_trigger: "内源触发",
  goal_stack: "目标堆栈",
  top_drivers: "主导通路",
  token_state: "言语预激活场",
  motivation_pool: "动机池",
  runtime_input: "当前输入",
  endogenous: "内源驱动",
  SkillExecutor: "技能执行器",
  skill_executor: "技能执行器",
  skillexecutor: "技能执行器",
  ForcedModeSwitch: "强制模式切换",
  forced_mode_switch: "强制模式切换",
};

const AUTHENTICITY_SOURCES: Record<string, string> = {
  trace: "当前轮次轨迹",
  snapshot: "认知快照",
};

const AUTHENTICITY_GUARD_ACTIONS: Record<string, string> = {
  pass: "通过",
  resample: "重采样回拉",
  fallback: "回退收束",
};

function translateDelimitedIdentifiers(text: string): string | null {
  const parts = text
    .split(/\s*,\s*/)
    .map((part) => part.trim())
    .filter((part) => part.length > 0);
  if (parts.length < 2) {
    return null;
  }
  return translateBrainIdentifierList(parts).join("、");
}

export function translateTimelineType(value: string): string {
  return TIMELINE_TYPE_LABELS[value] ?? value;
}

export function translateCognitivePhrase(value: unknown, fallback = "尚未接入"): string {
  if (typeof value !== "string") {
    return fallback;
  }
  const trimmed = value.trim();
  if (!trimmed) {
    return fallback;
  }
  const motivationCount = /^active=(\d+)$/.exec(trimmed);
  if (motivationCount) {
    return `活跃动机 ${motivationCount[1]} 项`;
  }
  const translatedIdentifiers = translateDelimitedIdentifiers(trimmed);
  if (translatedIdentifiers) {
    return translatedIdentifiers;
  }
  return COGNITIVE_PHRASE_LABELS[trimmed] ?? translateBrainIdentifier(trimmed, trimmed);
}

export function translateAuthenticitySource(value: unknown): string | null {
  if (typeof value !== "string") {
    return null;
  }
  const trimmed = value.trim();
  if (!trimmed || trimmed === "none") {
    return null;
  }
  return AUTHENTICITY_SOURCES[trimmed] ?? trimmed;
}

export function translateAuthenticityGuardAction(value: unknown): string | null {
  if (typeof value !== "string") {
    return null;
  }
  const trimmed = value.trim();
  if (!trimmed || trimmed === "none") {
    return null;
  }
  return AUTHENTICITY_GUARD_ACTIONS[trimmed] ?? trimmed;
}
