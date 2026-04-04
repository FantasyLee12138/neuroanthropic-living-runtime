import type { CognitiveSnapshotState, PanelKey, UiState } from "./types.js";

function line(label: string, value: string): string {
  return `${label}：${value}`;
}

function statusSummary(state: UiState): string {
  const run = state.run ?? {};
  const currentStep = (run.current_step as Record<string, unknown> | undefined) ?? {};
  return [
    line("状态", String(run.status ?? "unknown")),
    line("当前步骤", String(currentStep.title ?? "暂无")),
    line("已暂停", run.status === "paused" ? "是" : "否"),
    line("工作区", run.dirty_worktree_detected ? "dirty" : "clean")
  ].join("\n");
}

function whySummary(state: UiState): string {
  const why = state.lastWhy ?? {};
  const currentStep = (why.current_step as Record<string, unknown> | undefined) ?? {};
  const stopReason = (why.stop_reason as Record<string, unknown> | undefined) ?? {};
  const reason =
    String(currentStep.expected_observation ?? currentStep.detail ?? "先收集当前任务最直接的上下文。");
  const rows = [
    line("当前目标", String(why.goal_summary ?? why.goal ?? "暂无")),
    line("当前选择", String(currentStep.title ?? "暂无")),
    line("原因", reason)
  ];
  if (stopReason.message) {
    rows.push(line("暂停原因", String(stopReason.message)));
  }
  return rows.join("\n");
}

function stepsSummary(state: UiState): string {
  const current =
    state.steps.find((step) => ["running", "in_progress", "active"].includes(String(step.status ?? ""))) ?? state.steps[0] ?? {};
  const remaining = state.steps.filter(
    (step) => !["running", "in_progress", "active", "completed", "done"].includes(String(step.status ?? ""))
  );
  const rows = [
    line("当前步骤", String(current.title ?? current.step_id ?? "暂无")),
    line("剩余步骤", String(remaining.length))
  ];
  if (remaining.length > 0) {
    rows.push(line("后续", remaining.slice(0, 3).map((step) => String(step.title ?? step.step_id ?? "未命名步骤")).join("；")));
  }
  return rows.join("\n");
}

function toolsSummary(state: UiState): string {
  const recent = state.tools.slice(-3);
  if (recent.length === 0) {
    return line("最近工具", "暂无");
  }
  return line(
    "最近工具",
    recent.map((tool) => `${String(tool.tool_name ?? "unknown")}: ${String(tool.summary ?? tool.status ?? "已执行")}`).join("；")
  );
}

function metaSummary(state: UiState): string {
  const session = state.sessionMeta ?? {};
  return [
    line("权限模式", String(session.permission_mode ?? state.permissionMode ?? "plan")),
    line("工作区", String(session.cwd ?? "暂无")),
    line("待审批", String(state.pendingApprovals.length)),
  ].join("\n");
}

function labeledProbability(label: string, value: number): string {
  return `${label} (${value.toFixed(2)})`;
}

function describeMood(value: number): string {
  if (value < 0.25) {
    return "很不开心";
  }
  if (value < 0.45) {
    return "不太开心";
  }
  if (value < 0.62) {
    return "较开心";
  }
  if (value < 0.8) {
    return "开心";
  }
  return "很开心";
}

function describeEnergy(value: number): string {
  if (value < 0.2) {
    return "很疲惫";
  }
  if (value < 0.4) {
    return "有点累";
  }
  if (value < 0.65) {
    return "还算稳定";
  }
  if (value < 0.82) {
    return "状态不错";
  }
  return "精力充沛";
}

function describeAffectResidue(value: number): string {
  if (value < 0.08) {
    return "基本平稳";
  }
  if (value < 0.22) {
    return "轻微波动";
  }
  if (value < 0.45) {
    return "波动明显";
  }
  return "波动很强";
}

function describeFocus(value: string): string {
  return {
    task: "正在专心处理眼前的事",
    respond: "注意力放在如何回应上",
    wander: "思绪有些发散",
    rest: "正在慢慢回落和恢复",
    boot: "还在慢慢进入状态",
  }[value] ?? value;
}

function describeMode(value: string): string {
  return {
    interactive: "正常交流中",
    idle: "安静待机中",
    sleep: "在休眠整理里",
    safe: "处于谨慎收束中",
    plan: "正在规划整理中",
  }[value] ?? value;
}

function formatModelSection(modelStatus: Record<string, unknown> | undefined): string {
  if (!modelStatus) {
    return ["模型分层", "  暂无"].join("\n");
  }
  const tiers = (modelStatus.tiers as Record<string, Record<string, unknown>> | undefined) ?? {};
  const bindings = (modelStatus.agent_bindings as Record<string, string> | undefined) ?? {};
  const tierNames = Object.keys(tiers);
  const tierRows = ["模型分层"];
  for (const tierName of tierNames) {
    const tier = tiers[tierName] ?? {};
    const mode = String(tier.mode ?? "unknown");
    if (mode === "local") {
      tierRows.push(`  ${tierName}：local`);
      continue;
    }
    const backend = String(tier.backend ?? "unknown");
    const model = String(tier.model ?? "unconfigured");
    const credential = Boolean(tier.credential_present) ? "key:ok" : "key:missing";
    tierRows.push(`  ${tierName}：${backend} / ${model} / ${credential}`);
  }
  tierRows.push("");
  tierRows.push("主要绑定");
  for (const agentName of ["SalienceAgent", "ValueAgent", "PFCAgent", "PerspectiveModel", "Renderer", "planner"]) {
    const tier = bindings[agentName];
    if (tier) {
      tierRows.push(`  ${agentName} -> ${tier}`);
    }
  }
  return tierRows.join("\n");
}

export function formatCognitiveSummary(snapshot: CognitiveSnapshotState | null | undefined): string {
  if (!snapshot) {
    return "核心目标\n  暂无";
  }
  return [
    "核心目标",
    `  ${snapshot.coreGoal}`,
    "",
    "当前意图",
    `  ${snapshot.currentIntent}`,
    "",
    "生命体征",
    line("心境", labeledProbability(describeMood(snapshot.vitalSigns.mood), snapshot.vitalSigns.mood)),
    line("能量", labeledProbability(describeEnergy(snapshot.vitalSigns.bodyEnergy), snapshot.vitalSigns.bodyEnergy)),
    line("情感余波", labeledProbability(describeAffectResidue(snapshot.vitalSigns.affectResidue), snapshot.vitalSigns.affectResidue)),
    line("焦点", describeFocus(snapshot.vitalSigns.focus)),
    line("模式", describeMode(snapshot.vitalSigns.mode)),
    "",
    "身份与连续性",
    line("身份", snapshot.identity.displayName),
    line("连续性", snapshot.identity.continuity),
    "",
    "真实性",
    `  ${snapshot.authenticity.summary}`,
  ].join("\n");
}

export function formatPanelBody(state: UiState, panel: PanelKey): string {
  if (panel === "status") {
    return statusSummary(state);
  }
  if (panel === "why") {
    return whySummary(state);
  }
  if (panel === "steps") {
    return stepsSummary(state);
  }
  if (panel === "tools") {
    return toolsSummary(state);
  }
  if (panel === "state") {
    const blocks = [
      formatCognitiveSummary(state.sidebarSnapshot?.cognitiveSnapshot),
      formatModelSection(state.sidebarSnapshot?.modelStatus),
    ].filter(Boolean);
    return blocks.join("\n\n");
  }
  return "";
}
