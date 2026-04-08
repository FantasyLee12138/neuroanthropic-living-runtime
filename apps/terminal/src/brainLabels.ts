const BRAIN_LABELS: Record<string, string> = {
  PFCAgent: "前额叶执行控制",
  HippocampusAgent: "海马记忆检索",
  ConflictMonitorAgent: "前扣带冲突监测",
  DMNAgent: "默认模式网络",
  BodyStateAgent: "躯体内感监测",
  AuthenticityPolicy: "真实性监测",
  LongRunAnalyzer: "长程连续性评估",
  InstinctField: "本能场",
  PerspectiveModel: "社会认知建模",
  IdentityRuntime: "自我模型维持",
  ValueAgent: "价值评估",
  SalienceAgent: "显著性筛选",
  HabitAgent: "习惯预测",
  VitalityEngine: "活力调节系统",
  ResourceAgent: "资源分配",
  StochasticPolicy: "随机探索",
  EmotionAgent: "情绪评估",
  DesireAgent: "欲求驱动",
  RelationshipAgent: "关系校准",
  CerebellarPredictor: "预测校正",
  ThalamusAttentionAgent: "丘脑注意门控",
  UnconsciousAgent: "无意识预整合",
  EndogenousMotivationPool: "内源动机池",
  Renderer: "语言表述生成",
  planner: "前额叶规划",
  not_proposed: "本轮未进入候选场",
  gate: "门控抑制",
  conflict: "冲突仲裁",
};

export function translateBrainIdentifier(value: unknown, fallback = "尚未接入"): string {
  if (typeof value !== "string") {
    return fallback;
  }
  const trimmed = value.trim();
  if (!trimmed) {
    return fallback;
  }
  return BRAIN_LABELS[trimmed] ?? trimmed;
}

export function translateBrainIdentifierList(values: unknown[]): string[] {
  return values
    .map((value) => translateBrainIdentifier(value, ""))
    .filter((value): value is string => value.length > 0);
}

export const describeBrainLabel = translateBrainIdentifier;
