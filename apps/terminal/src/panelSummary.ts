import type { CognitiveSnapshotState, PanelKey, UiState } from "./types.js";
import { translateBrainIdentifier } from "./brainLabels.js";
import { translateAuthenticityGuardAction, translateAuthenticitySource, translateCognitivePhrase, translateTimelineType } from "./cognitiveTerms.js";
import {
  translateActionName,
  translateCognitiveMode,
  translateFocusState,
  translatePermissionMode,
  translateRiskLevel,
  translateRunStatus,
  translateToolName,
} from "./displayLabels.js";
import { summarizeMonologueStream } from "./expressiveTrace.js";
import { contributionSectionTitle, whyCurrentLabel, whyNotSectionTitle } from "./terminalCopy.js";

export interface SidebarSummaryItem {
  label: string;
  value: string;
  tone: "normal" | "muted" | "warning" | "accent";
}

function line(label: string, value: string): string {
  return `${label}：${value}`;
}

function formatScore(value: number | null | undefined): string {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(2) : "0.00";
}

function presentText(value: unknown): string {
  if (value === null || value === undefined) {
    return "尚未接入";
  }
  if (typeof value === "string") {
    const trimmed = value.trim();
    return trimmed.length > 0 ? value : "尚未接入";
  }
  return String(value);
}

function statusSummary(state: UiState): string {
  const consoleState = state.console.state;
  const run = state.run ?? {};
  const currentStep = (run.current_step as Record<string, unknown> | undefined) ?? {};
  const rawStatus = presentText(consoleState?.run.status ?? run.status);
  const status = translateRunStatus(rawStatus, rawStatus);
  const currentRound = consoleState?.currentRound;
  const rawCurrentStepTitle =
    currentRound?.causeType === "endogenous"
      ? presentText(currentRound.modeLabel ?? currentRound.causeLabel ?? currentRound.sampledAction ?? currentStep.title)
      : presentText(currentRound?.sampledAction ?? currentStep.title);
  const currentStepTitle = translateActionName(rawCurrentStepTitle, rawCurrentStepTitle);
  const worktree = run.dirty_worktree_detected;
  const rows = [
    line("脑态", status),
    line("当前驱动", currentStepTitle),
    line("外显活动", worktree === undefined ? "尚未接入" : worktree ? "工作区有变更" : "工作区已收束"),
  ];
  const causeLabel = presentText(currentRound?.causeLabel);
  if (causeLabel !== "尚未接入") {
    rows.push(line("处理来源", causeLabel));
  }
  const latencySummary = presentText(currentRound?.latencySummary);
  if (latencySummary !== "尚未接入") {
    const modelWait = currentRound?.modelWaitMs == null ? "?" : `${currentRound.modelWaitMs}ms`;
    const localCompute = currentRound?.localComputeMs == null ? "?" : `${currentRound.localComputeMs}ms`;
    rows.push(line("本轮时延", `${latencySummary}（模型 ${modelWait} / 本地 ${localCompute}）`));
  }
  if (currentRound?.activationSet?.length) {
    rows.push(line("激活模块", currentRound.activationSet.join(" · ")));
  }
  if (currentRound?.memoryTiersRead?.length) {
    rows.push(line("记忆层", currentRound.memoryTiersRead.join(" → ")));
  }
  if (currentRound?.backgroundJobs?.length) {
    rows.push(line("后台任务", `${currentRound.backgroundJobs.length} 项`));
  }
  const deepenReason = presentText(currentRound?.deepenReason);
  if (deepenReason !== "尚未接入") {
    rows.push(line("加深原因", deepenReason));
  }
  return rows.join("\n");
}

function whySummary(state: UiState): string {
  const consoleWhy = state.console.whyCurrent?.why;
  const actionField = state.console.actionField;
  const why = state.lastWhy ?? {};
  const currentStep = (why.current_step as Record<string, unknown> | undefined) ?? {};
  const stopReason = (why.stop_reason as Record<string, unknown> | undefined) ?? {};
  const initiative = (consoleWhy?.initiative ?? {}) as Record<string, unknown>;
  const reason = presentText(consoleWhy?.summary ?? currentStep.expected_observation ?? currentStep.detail);
  const rows = [
    line(whyCurrentLabel(), presentText(consoleWhy?.summary ?? why.goal_summary ?? why.goal)),
    line(
      "最终输出动作",
      translateActionName(
        presentText(consoleWhy?.sampledAction ?? currentStep.title),
        presentText(consoleWhy?.sampledAction ?? currentStep.title),
      ),
    ),
  ];
  if (reason !== "尚未接入") {
    rows.push(line(whyCurrentLabel(), reason));
  } else {
    const stopMessage = presentText(stopReason.message);
    if (stopMessage !== "尚未接入") {
      rows.push(line(whyCurrentLabel(), stopMessage));
    }
  }
  const stopMessage = presentText(stopReason.message);
  if (reason !== "尚未接入" && stopMessage !== "尚未接入") {
    rows.push(line("暂停说明", stopMessage));
  }
  if (actionField?.competingPeaks.length) {
    rows.push(
      line(
        whyNotSectionTitle(),
        actionField.competingPeaks
          .map((item) => `${translateActionName(presentText(item.action), presentText(item.action))}(${formatScore(item.score)})`)
          .join("；"),
      ),
    );
  }
  if (actionField?.contributionStack.length) {
    rows.push(line(contributionSectionTitle(), actionField.contributionStack.map((item) => `${translateBrainIdentifier(item.source)}(${formatScore(item.weight)})`).join("；")));
  }
  const initiativeMode = presentText(initiative.expression_mode);
  const initiativeIntent = presentText(initiative.top_intent);
  if (initiativeMode !== "尚未接入" || initiativeIntent !== "尚未接入") {
    const shouldSend = typeof initiative.should_send === "boolean" ? String(initiative.should_send) : "尚未接入";
    rows.push(line("主动性判定", `${initiativeMode} · ${initiativeIntent} · should_send=${shouldSend}`));
  }
  const suppressionReason = presentText(initiative.suppression_reason);
  if (suppressionReason !== "尚未接入") {
    rows.push(line("抑制原因", suppressionReason));
  }
  const memoryBacking = ((initiative.memory_backing ?? {}) as Record<string, unknown>);
  const cue = presentText(memoryBacking.cue ?? memoryBacking.summary);
  if (cue !== "尚未接入") {
    const topicSource = presentText(memoryBacking.topic_source ?? memoryBacking.topicSource);
    const topicRelevance = typeof (memoryBacking.topic_relevance ?? memoryBacking.topicRelevance) === "number"
      ? formatScore(Number(memoryBacking.topic_relevance ?? memoryBacking.topicRelevance))
      : "尚未接入";
    rows.push(line("记忆牵引", `${cue} · source=${topicSource} · relevance=${topicRelevance}`));
  }
  rows.push(...summarizeMonologueStream(consoleWhy?.expressiveTrace));
  return rows.join("\n");
}

function timelineEventLabel(entry: { type: string; label: string; summary: string }): string {
  return `${translateTimelineType(entry.type)}：${translateCognitivePhrase(entry.summary)}`;
}

function stepsSummary(state: UiState): string {
  const timeline = state.console.timeline;
  const winner = state.console.actionField?.winner;
  if (timeline?.events.length) {
    return [
      line("当前驱动", translateActionName(presentText(winner?.action), presentText(winner?.action))),
      line("外显活动", `${timeline.events.length} 段流转`),
      line("接续", timeline.events.slice(0, 3).map((entry) => translateCognitivePhrase(entry.label, translateBrainIdentifier(entry.label, presentText(entry.label)))).join("；")),
    ].join("\n");
  }
  const current =
    state.steps.find((step) => ["running", "in_progress", "active"].includes(String(step.status ?? ""))) ?? state.steps[0] ?? {};
  const remaining = state.steps.filter(
    (step) => !["running", "in_progress", "active", "completed", "done"].includes(String(step.status ?? "")),
  );
  const rows = [
    line("当前驱动", presentText(current.title ?? current.step_id)),
    line("外显活动", remaining.length === 0 ? "已收束" : `${remaining.length} 步待续`),
  ];
  if (remaining.length > 0) {
    rows.push(line("接续", remaining.slice(0, 3).map((step) => presentText(step.title ?? step.step_id)).join("；")));
  }
  return rows.join("\n");
}

function toolsSummary(state: UiState): string {
  const timeline = state.console.timeline;
  if (timeline?.events.length) {
    return line("外显活动", timeline.events.slice(0, 3).map(timelineEventLabel).join("；"));
  }
  const recent = state.tools.slice(-3);
  if (recent.length === 0) {
    return line("外显活动", "尚未接入");
  }
  const entries = recent
    .map((tool) => `${translateToolName(presentText(tool.tool_name), presentText(tool.tool_name))}：${presentText(tool.summary ?? tool.status)}`)
    .join("；");
  return line(
    "外显活动",
    entries,
  );
}

function approvalsSummary(state: UiState): string {
  if (state.pendingApprovals.length === 0) {
    return "等待本轮解释：尚未接入";
  }
  return [
    `等待本轮解释：${state.pendingApprovals.length}`,
    ...state.pendingApprovals
      .map((approval, index) => {
        const prefix = index === state.approvalCursor ? ">" : " ";
        const rows = [
          `${prefix} [${index + 1}/${state.pendingApprovals.length}] ${translateToolName(presentText(approval.tool), presentText(approval.tool))}`,
          `  ${presentText(approval.summary ?? approval.actionPreview)}`,
        ];
        const riskLevel = presentText(approval.riskLevel);
        if (riskLevel !== "尚未接入") {
          rows.push(`  风险：${translateRiskLevel(riskLevel, riskLevel)}`);
        }
        const mode = presentText(approval.mode);
        if (mode !== "尚未接入") {
          rows.push(`  模式：${translatePermissionMode(mode, mode)}`);
        }
        const status = presentText(approval.status);
        if (status !== "尚未接入") {
          rows.push(`  状态：${translateRunStatus(status, status)}`);
        }
        return rows.join("\n");
      }),
  ].join("\n");
}

function metaSummary(state: UiState): string {
  const session = state.sessionMeta ?? {};
  const statusline = state.statusline ?? undefined;
  return [
    line("当前环境", presentText(statusline?.cwd ?? session.cwd)),
    line(
      "当前权限",
      translatePermissionMode(
        presentText(state.sidebarSnapshot?.permissionMode ?? statusline?.permission_mode ?? state.permissionMode),
        presentText(state.sidebarSnapshot?.permissionMode ?? statusline?.permission_mode ?? state.permissionMode),
      ),
    ),
    line("会话", presentText(statusline?.session_id ?? state.activeSessionId)),
    line("运行", presentText(statusline?.run_id ?? state.activeRunId)),
    "",
    formatModelSection(state.sidebarSnapshot?.modelStatus),
  ].join("\n");
}

function labeledProbability(label: string, value: number): string {
  return `${label} (${value.toFixed(2)})`;
}

function describeMood(value: number): string {
  if (value < 0.25) {
    return "低正价";
  }
  if (value < 0.45) {
    return "中性偏低";
  }
  if (value < 0.62) {
    return "中性";
  }
  if (value < 0.8) {
    return "中性偏高";
  }
  return "高正价";
}

function describeEnergy(value: number): string {
  if (value < 0.2) {
    return "低唤醒";
  }
  if (value < 0.4) {
    return "轻度低唤醒";
  }
  if (value < 0.65) {
    return "中等唤醒";
  }
  if (value < 0.82) {
    return "中高唤醒";
  }
  return "高唤醒";
}

function describeAffectResidue(value: number): string {
  if (value < 0.08) {
    return "低残留";
  }
  if (value < 0.22) {
    return "轻度残留";
  }
  if (value < 0.45) {
    return "中度残留";
  }
  return "高残留";
}

function describeFocus(value: string): string {
  return translateFocusState(value, presentText(value));
}

function describeMode(value: string): string {
  return translateCognitiveMode(value, presentText(value));
}

function formatModelSection(modelStatus: Record<string, unknown> | undefined): string {
  if (!modelStatus) {
    return ["模型分层", "  尚未接入"].join("\n");
  }
  const tiers = (modelStatus.tiers as Record<string, Record<string, unknown>> | undefined) ?? {};
  const bindings = (modelStatus.module_model_bindings as Record<string, string> | undefined) ?? {};
  const tierRows = ["模型分层"];
  for (const [tierName, tier] of Object.entries(tiers)) {
    const mode = presentText(tier.mode);
    if (mode === "local") {
      tierRows.push(`  ${tierName}：本地`);
      continue;
    }
    tierRows.push(
      `  ${tierName}：${presentText(tier.backend)} / ${presentText(tier.model)} / ${Boolean(tier.credential_present) ? "密钥已接入" : "密钥尚未接入"}`,
    );
  }
  tierRows.push("");
  tierRows.push("主要模块模型绑定");
  for (const moduleName of ["cognitive_packet", "deliberation", "tool_planner", "deep_renderer", "consolidation_summarizer"]) {
    const tier = bindings[moduleName];
    if (tier) {
      tierRows.push(`  ${moduleName}：${tier}`);
    }
  }
  return tierRows.join("\n");
}

export function formatCognitiveSummary(snapshot: CognitiveSnapshotState | null | undefined): string {
  if (!snapshot) {
    return "脑态\n  尚未接入";
  }
  const rows = [
    "脑态",
    line("当前驱动", presentText(snapshot.currentIntent)),
    "",
    "内在状态",
    line("心境", labeledProbability(describeMood(snapshot.vitalSigns.mood), snapshot.vitalSigns.mood)),
    line("能量", labeledProbability(describeEnergy(snapshot.vitalSigns.bodyEnergy), snapshot.vitalSigns.bodyEnergy)),
    line("情感余波", labeledProbability(describeAffectResidue(snapshot.vitalSigns.affectResidue), snapshot.vitalSigns.affectResidue)),
    line("注意焦点", describeFocus(snapshot.vitalSigns.focus)),
    line("运行方式", describeMode(snapshot.vitalSigns.mode)),
    "",
    "身份与连续性",
    line("身份", presentText(snapshot.identity.displayName)),
    line("连续性", presentText(snapshot.identity.continuity)),
    "",
    "真实性校验",
    `  ${presentText(snapshot.authenticity.summary)}`,
  ];
  const source = translateAuthenticitySource(snapshot.authenticity.source);
  if (source) {
    rows.push(line("校验来源", source));
  }
  const guardAction = translateAuthenticityGuardAction(snapshot.authenticity.guardAction);
  if (guardAction) {
    rows.push(line("校验动作", guardAction));
  }
  return rows.join("\n");
}

export function formatDetailSummary(state: UiState, panel: Exclude<PanelKey, null>): string {
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
  if (panel === "approvals") {
    return approvalsSummary(state);
  }
  if (panel === "meta") {
    return metaSummary(state);
  }
  return formatCognitiveSummary(state.console.state?.cognitiveSnapshot ?? state.sidebarSnapshot?.cognitiveSnapshot);
}

export function formatSidebarSummary(state: UiState): SidebarSummaryItem[] {
  const snapshot = state.sidebarSnapshot;
  const consoleState = state.console.state;
  const actionField = state.console.actionField;
  const whyCurrent = state.console.whyCurrent;
  const timeline = state.console.timeline;
  const lastTool = state.tools.at(-1);
  const latestTimeline = timeline?.events.at(-1);
  return [
    {
      label: "当前驱动",
      value: translateActionName(presentText(actionField?.winner.action ?? snapshot?.goalSummary), presentText(actionField?.winner.action ?? snapshot?.goalSummary)),
      tone: "muted",
    },
    { label: "外显活动", value: presentText(whyCurrent?.why.summary ?? snapshot?.currentStep ?? state.run?.current_step?.title), tone: "normal" },
    {
      label: "工具痕迹",
      value: lastTool
        ? `${translateToolName(presentText(lastTool.tool_name), presentText(lastTool.tool_name))}：${presentText(lastTool.summary ?? lastTool.status)}`
        : latestTimeline
          ? `${translateTimelineType(latestTimeline.type)}：${translateCognitivePhrase(latestTimeline.summary)}`
          : "尚未接入",
      tone: "muted",
    },
    {
      label: "等待本轮解释",
      value: state.pendingApprovals.length > 0 ? `${state.pendingApprovals.length} 项待处理` : "尚未接入",
      tone: state.pendingApprovals.length > 0 ? "warning" : "muted",
    },
    {
      label: "脑态",
      value: presentText(consoleState?.cognitiveSnapshot.currentIntent ?? snapshot?.cognitiveSnapshot.currentIntent),
      tone: "accent",
    },
  ];
}
