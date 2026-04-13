import type {
  ActivityEntry,
  AliveConsoleColumnState,
  AliveConsoleState,
  ConsoleState,
  ConsoleStateState,
  ConsoleRefreshPayload,
  CognitiveTimelineEntry,
  CognitiveSnapshotState,
  OutboundBridgeEvent,
  UiAction,
  UiLine,
  UiState,
} from "../types.js";
import { translateBrainIdentifier } from "../brainLabels.js";
import { translateAuthenticityGuardAction, translateAuthenticitySource } from "../cognitiveTerms.js";
import { translateActionName, translateCognitiveMode, translateFocusState, translatePermissionMode, translateRunStatus, translateToolName } from "../displayLabels.js";
import { normalizeExpressiveTrace, summarizeMonologueStream } from "../expressiveTrace.js";
import { contributionSectionTitle, whyCurrentLabel } from "../terminalCopy.js";

export function createInitialUiState(): UiState {
  return refreshAliveConsoleState({
    activeSessionId: null,
    activeRunId: null,
    sessionMeta: null,
    run: null,
    steps: [],
    tools: [],
    toolTimeline: [],
    lastWhy: null,
    sidebarSnapshot: null,
    statusline: null,
    lines: [],
    activityRail: [],
    transcriptMode: "full",
    promptHistoryByCwd: {},
    pendingApprovals: [],
    actionBar: {
      primary: [],
      secondary: [],
      selectedIndex: 0,
    },
    permissionMode: "plan",
    assistantStreamActive: false,
    detailDrawer: null,
    focusZone: "input",
    approvalCursor: 0,
    console: emptyConsoleState(),
    aliveConsole: emptyAliveConsoleState(),
  });
}

function appendLine(state: UiState, line: UiLine): UiState {
  return { ...state, lines: [...state.lines, line] };
}

function replaceByKey(rows: Array<Record<string, any>>, key: string, nextRow: Record<string, any>): Array<Record<string, any>> {
  const rowKey = String(nextRow[key] ?? "");
  if (!rowKey) {
    return [...rows, nextRow];
  }
  const index = rows.findIndex((item) => String(item[key] ?? "") === rowKey);
  if (index < 0) {
    return [...rows, nextRow];
  }
  return rows.map((item, itemIndex) => (itemIndex === index ? nextRow : item));
}

function normalizeLinkage(entry: Record<string, unknown>): { roundId?: number; traceRef?: string } {
  const roundValue = entry.roundId ?? entry.round_id;
  const traceValue = entry.traceRef ?? entry.trace_ref;
  const roundId =
    typeof roundValue === "number"
      ? roundValue
      : typeof roundValue === "string" && roundValue.trim() && !Number.isNaN(Number(roundValue))
        ? Number(roundValue)
        : undefined;
  const traceRef = typeof traceValue === "string" && traceValue.trim() ? traceValue : undefined;
  return { roundId, traceRef };
}

function emptyConsoleColumn(title: string): AliveConsoleColumnState {
  return {
    title,
    summary: "暂无",
    details: [],
  };
}

function emptyAliveConsoleState(): AliveConsoleState {
  return {
    state: emptyConsoleColumn("脑态"),
    actionField: emptyConsoleColumn("思绪流"),
    why: emptyConsoleColumn("解释层"),
  };
}

function emptyConsoleState(): ConsoleState {
  return {
    state: null,
    actionField: null,
    timeline: null,
    whyCurrent: null,
    whyNot: null,
  };
}

function formatProbability(value: number): string {
  return Number.isFinite(value) ? value.toFixed(2) : "0.00";
}

function formatActionLabel(action: UiAction, selected: boolean): string {
  return `${selected ? ">" : " "} ${translateActionName(action.label, action.label)}${action.disabled ? "（禁用）" : ""}`;
}

function normalizeRecord(value: unknown): Record<string, unknown> | undefined {
  return typeof value === "object" && value !== null && !Array.isArray(value) ? (value as Record<string, unknown>) : undefined;
}

function normalizeRecordList(value: unknown): Array<Record<string, unknown>> {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.filter((entry): entry is Record<string, unknown> => typeof entry === "object" && entry !== null);
}

function readConsoleNumber(value: unknown): number | undefined {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === "string" && value.trim() && !Number.isNaN(Number(value))) {
    return Number(value);
  }
  return undefined;
}

function readConsoleText(value: unknown, fallback: string): string {
  if (typeof value === "string" && value.trim()) {
    return value.trim();
  }
  const numeric = readConsoleNumber(value);
  if (numeric !== undefined) {
    return formatProbability(numeric);
  }
  return fallback;
}

function formatConsoleInlineValue(value: unknown): string | null {
  if (typeof value === "string" && value.trim()) {
    return value.trim();
  }
  const numeric = readConsoleNumber(value);
  if (numeric !== undefined) {
    return formatProbability(numeric);
  }
  if (typeof value === "boolean") {
    return String(value);
  }
  if (Array.isArray(value)) {
    const parts = value
      .map((entry) => formatConsoleInlineValue(entry))
      .filter((entry): entry is string => Boolean(entry));
    return parts.length > 0 ? parts.join("、") : null;
  }
  return null;
}

function formatTokenSequence(value: unknown): string | null {
  if (!Array.isArray(value)) {
    return null;
  }
  const parts = value
    .map((entry) => (typeof entry === "string" ? entry.trim() : ""))
    .filter((entry) => entry.length > 0);
  if (parts.length === 0) {
    return null;
  }
  const sentence = parts.join(" ")
    .replace(/\s+([,.;:!?])/g, "$1")
    .replace(/\(\s+/g, "(")
    .replace(/\s+\)/g, ")");
  return sentence.trim();
}

function describeDeltaGenerationPolicy(value: unknown): string | null {
  if (typeof value !== "string" || !value.trim()) {
    return null;
  }
  return {
    propagate_only: "仅传播当前已形成的状态，不额外生成新增量",
    focused: "围绕当前焦点生成局部增量",
    regenerate: "重新生成整段表达增量",
  }[value.trim()] ?? value.trim();
}

function summarizeTokenFieldState(value: Record<string, unknown> | undefined, emptyText: string): string {
  if (!value) {
    return emptyText;
  }
  const parts: string[] = [];
  const policy = describeDeltaGenerationPolicy(value.delta_generation_policy);
  if (policy) {
    parts.push(`生成策略为${policy}`);
  }
  const prefix = formatTokenSequence(value.prefix_tokens);
  if (prefix) {
    parts.push(`前缀语句为${prefix}`);
  }
  return parts.length > 0 ? parts.join("；") : emptyText;
}

function summarizeConsolePairs(value: Record<string, unknown> | undefined, emptyText: string): string {
  if (!value) {
    return emptyText;
  }
  const parts = Object.entries(value)
    .map(([key, entry]) => {
      const rendered = formatConsoleInlineValue(entry);
      return rendered ? `${key}=${rendered}` : null;
    })
    .filter((entry): entry is string => Boolean(entry))
    .slice(0, 3);
  return parts.length > 0 ? parts.join("；") : emptyText;
}

function summarizeInitiativeMemoryBacking(value: Record<string, unknown> | undefined): string | null {
  if (!value) {
    return null;
  }
  const cue = formatConsoleInlineValue(value.cue ?? value.summary);
  if (!cue) {
    return null;
  }
  const parts = [cue];
  const topicSource = formatConsoleInlineValue(value.topic_source ?? value.topicSource);
  if (topicSource) {
    parts.push(`source=${topicSource}`);
  }
  const topicRelevance = readConsoleNumber(value.topic_relevance ?? value.topicRelevance);
  if (topicRelevance !== undefined) {
    parts.push(`relevance=${formatProbability(topicRelevance)}`);
  }
  return parts.join(" · ");
}

function formatConsoleAction(entry: Record<string, unknown>): string {
  const rawAction = readConsoleText(entry.action ?? entry.label ?? entry.name, "unknown");
  const action = translateActionName(rawAction, rawAction);
  const score = readConsoleNumber(entry.score);
  return score === undefined ? action : `${action}(${formatProbability(score)})`;
}

function formatConsoleContribution(entry: Record<string, unknown>): string {
  const source = translateBrainIdentifier(entry.source ?? entry.agent_name ?? entry.module_name ?? entry.name, "unknown");
  const weight = readConsoleNumber(entry.weight ?? entry.score ?? entry.value ?? entry.delta_normalized ?? entry.delta_projected);
  return weight === undefined ? source : `${source}(${formatProbability(Math.abs(weight))})`;
}

function formatNormalizedActionScore(entry: { action: string | null; score: number | null }): string {
  const rawAction = readConsoleText(entry.action, "unknown");
  const action = translateActionName(rawAction, rawAction);
  return entry.score == null ? action : `${action}(${formatProbability(entry.score)})`;
}

function readConsoleNullableText(value: unknown): string | null {
  if (typeof value === "string" && value.trim()) {
    return value.trim();
  }
  return null;
}

function readConsoleNullableNumber(value: unknown): number | null {
  return readConsoleNumber(value) ?? null;
}

function normalizeConsoleActionScore(entry: Record<string, unknown>): { action: string | null; score: number | null } {
  return {
    action: readConsoleNullableText(entry.action ?? entry.label ?? entry.name),
    score: readConsoleNullableNumber(entry.score),
  };
}

function normalizeConsoleContribution(entry: Record<string, unknown>): { source: string | null; weight: number | null; raw: Record<string, unknown> } {
  const weight = readConsoleNullableNumber(entry.weight ?? entry.score ?? entry.value ?? entry.delta_normalized ?? entry.delta_projected);
  return {
    source: readConsoleNullableText(entry.source ?? entry.agent_name ?? entry.module_name ?? entry.name),
    weight: weight == null ? null : Math.abs(weight),
    raw: entry,
  };
}

function formatActivityLabel(entry: ActivityEntry): string {
  const prefix = {
    step: "步骤",
    tool: "外部动作",
    result: "结果",
    approval: "人工确认",
  }[entry.kind];
  const label =
    entry.kind === "tool" || entry.kind === "result" || entry.kind === "approval"
      ? translateToolName(entry.label, entry.label)
      : entry.label;
  return entry.summary ? `${prefix}：${label} · ${entry.summary}` : `${prefix}：${label}`;
}

function buildStateColumn(state: UiState): AliveConsoleColumnState {
  const snapshot = state.sidebarSnapshot;
  const cognitive = snapshot?.cognitiveSnapshot;
  const vitalSigns = cognitive?.vitalSigns;
  const linkage = snapshot ?? state.activityRail.at(-1);
  const summary =
    snapshot?.runStatus ??
    state.run?.status ??
    state.statusline?.run_status ??
    (state.activeRunId ? "running" : "idle");

  return {
    title: "脑态",
    summary: `${translateRunStatus(summary, summary)} · ${translatePermissionMode(snapshot?.permissionMode ?? state.permissionMode ?? "plan", String(snapshot?.permissionMode ?? state.permissionMode ?? "plan"))}`,
    details: [
      `模式：${translateCognitiveMode(String(vitalSigns?.mode ?? snapshot?.permissionMode ?? state.permissionMode ?? "plan"), String(vitalSigns?.mode ?? snapshot?.permissionMode ?? state.permissionMode ?? "plan"))}`,
      `活力：心境 ${formatProbability(Number(vitalSigns?.mood ?? 0))} / 能量 ${formatProbability(Number(vitalSigns?.bodyEnergy ?? 0))} / 余波 ${formatProbability(Number(vitalSigns?.affectResidue ?? 0))}`,
      `注意焦点：${translateFocusState(String(vitalSigns?.focus ?? "unknown"), String(vitalSigns?.focus ?? "unknown"))}`,
      `身份：${String(cognitive?.identity.displayName ?? "暂无")} · ${String(cognitive?.identity.continuity ?? "暂无")}`,
      `真实性校验：${String(cognitive?.authenticity.summary ?? "暂无")}`,
    ],
    roundId: linkage?.roundId,
    traceRef: linkage?.traceRef,
  };
}

function buildActionFieldColumn(state: UiState): AliveConsoleColumnState {
  const selectedAction =
    state.actionBar.primary[state.actionBar.selectedIndex] ??
    state.actionBar.secondary[state.actionBar.selectedIndex - state.actionBar.primary.length] ??
    state.actionBar.primary[0] ??
    null;
  const linkage = state.activityRail.at(-1) ?? state.sidebarSnapshot;
  const details: string[] = [];

  if (state.actionBar.primary.length > 0) {
    details.push(`主动作：${state.actionBar.primary.map((action, index) => formatActionLabel(action, index === state.actionBar.selectedIndex)).join("；")}`);
  } else {
    details.push("主动作：暂无");
  }

  if (state.actionBar.secondary.length > 0) {
    details.push(`辅助动作：${state.actionBar.secondary.map((action) => formatActionLabel(action, false)).join("；")}`);
  } else {
    details.push("辅助动作：暂无");
  }

  if (state.pendingApprovals.length > 0) {
    details.push(
      `待审批：${state.pendingApprovals
        .map((approval, index) => `${index === state.approvalCursor ? ">" : " "} ${translateToolName(approval.tool, approval.tool)}${approval.summary ? ` · ${approval.summary}` : ""}`)
        .join("；")}`,
    );
  } else {
    details.push("待审批：无");
  }

  const recentActivity = state.activityRail.slice(-3);
  if (recentActivity.length > 0) {
    details.push(`最近活动：${recentActivity.map(formatActivityLabel).join("；")}`);
  }

  return {
    title: "思绪流",
    summary: selectedAction ? `${selectedAction.label} · 待审批 ${state.pendingApprovals.length}` : `待审批 ${state.pendingApprovals.length}`,
    details,
    roundId: linkage?.roundId,
    traceRef: linkage?.traceRef,
  };
}

function buildWhyColumn(state: UiState): AliveConsoleColumnState {
  const snapshot = state.sidebarSnapshot;
  const why = state.lastWhy ?? {};
  const currentStep = (why.current_step as Record<string, unknown> | undefined) ?? {};
  const goalSummary = String(snapshot?.goalSummary ?? why.goal_summary ?? why.goal ?? "暂无");
  const stepTitle = String(snapshot?.currentStep ?? currentStep.title ?? "暂无");
  const reasonSummary = String(
    snapshot?.reasonSummary ?? currentStep.expected_observation ?? currentStep.detail ?? "先收集当前任务最直接的上下文。",
  );
  const details = [`当前目标：${goalSummary}`, `当前选择：${stepTitle}`, `${whyCurrentLabel()}：${reasonSummary}`];
  const lastTool = snapshot?.lastTool ?? String((why.last_tool as string | undefined) ?? "");
  if (lastTool) {
    details.push(`最近工具：${translateToolName(lastTool, lastTool)}`);
  }
  if (snapshot?.roundId !== undefined || snapshot?.traceRef) {
    details.push(
      [snapshot?.roundId !== undefined ? `round ${snapshot.roundId}` : null, snapshot?.traceRef ? `trace ${snapshot.traceRef}` : null]
        .filter(Boolean)
        .join(" · "),
    );
  }
  return {
    title: "解释层",
    summary: goalSummary,
    details,
    roundId: snapshot?.roundId,
    traceRef: snapshot?.traceRef,
  };
}

function buildAliveConsoleState(state: UiState): AliveConsoleState {
  const derived = {
    state: buildStateColumn(state),
    actionField: buildActionFieldColumn(state),
    why: buildWhyColumn(state),
  };
  const hydrated = buildAliveConsoleFromConsoleState(state.console, state);
  return {
    state: hydrated?.state ?? derived.state,
    actionField: hydrated?.actionField ?? derived.actionField,
    why: hydrated?.why ?? derived.why,
  };
}

function buildAliveConsoleFromConsoleState(consoleState: ConsoleState, fallbackState: UiState): Partial<AliveConsoleState> {
  const hydrated: Partial<AliveConsoleState> = {};
  const statePayload = consoleState.state;
  if (statePayload) {
    const brainState = statePayload.brainState;
    const currentRound = statePayload.currentRound;
    const run = statePayload.run;
    const cognitive = statePayload.cognitiveSnapshot;
    hydrated.state = {
      title: "脑态",
      summary: `${translateRunStatus(readConsoleText(run.status ?? fallbackState.run?.status, "idle"), readConsoleText(run.status ?? fallbackState.run?.status, "idle"))} · ${translateCognitiveMode(readConsoleText(brainState.mode ?? cognitive.vitalSigns.mode, "尚未接入"), readConsoleText(brainState.mode ?? cognitive.vitalSigns.mode, "尚未接入"))}`,
      details: [
        `模式：${translateCognitiveMode(readConsoleText(brainState.mode ?? cognitive.vitalSigns.mode, "尚未接入"), readConsoleText(brainState.mode ?? cognitive.vitalSigns.mode, "尚未接入"))}`,
        `活力：${readConsoleText(brainState.vitality, "尚未接入")}`,
        `自我连续性：${readConsoleText(brainState.selfContinuity, "尚未接入")}`,
        `真实性校验：${readConsoleText(brainState.authenticityPressure, "尚未接入")}`,
        `注意焦点：${translateFocusState(readConsoleText(cognitive.vitalSigns.focus, "尚未接入"), readConsoleText(cognitive.vitalSigns.focus, "尚未接入"))}`,
      ],
      roundId: currentRound.roundId ?? undefined,
      traceRef: currentRound.traceRef ?? undefined,
    };
  }

  const actionFieldPayload = consoleState.actionField;
  if (actionFieldPayload) {
    const winnerAction = translateActionName(readConsoleText(actionFieldPayload.winner.action, "等待接入"), readConsoleText(actionFieldPayload.winner.action, "等待接入"));
    const winnerScore = actionFieldPayload.winner.score ?? undefined;
    const topActions = actionFieldPayload.topActions.map(formatNormalizedActionScore);
    const competingPeaks = actionFieldPayload.competingPeaks.map(formatNormalizedActionScore);
    const contributionStack = actionFieldPayload.contributionStack.map((entry) => formatConsoleContribution(entry.raw));
    const expressive = normalizeRecord(actionFieldPayload.expressive) ?? {};
    const expressionMode = readConsoleText(expressive.expression_mode, "");
    const proposalType = readConsoleText(expressive.proposal_type, "");
    const topIntent = readConsoleText(expressive.top_intent, "");
    const speechCost = readConsoleText(expressive.speech_cost, "");
    const intrinsicValue = readConsoleText(expressive.intrinsic_value, "");
    const shouldSend = typeof expressive.should_send === "boolean" ? String(expressive.should_send) : "";
    const suppressionReason = readConsoleText(expressive.suppression_reason, "");
    const memoryBacking = summarizeInitiativeMemoryBacking(normalizeRecord(expressive.memory_backing));
    hydrated.actionField = {
      title: "思绪流",
      summary: winnerScore === undefined ? winnerAction : `${winnerAction} · 倾向 ${formatProbability(winnerScore)}`,
      details: [
        `当前主导动作：${winnerAction}`,
        `竞争候选路径：${topActions.length > 0 ? topActions.join("；") : "等待接入"}`,
        `未采纳路径：${competingPeaks.length > 0 ? competingPeaks.join("；") : "本轮未采纳路径尚不清晰"}`,
        `${contributionSectionTitle()}：${contributionStack.length > 0 ? contributionStack.join("；") : "贡献叠层尚未完全暴露"}`,
        `言语预激活：${summarizeTokenFieldState(actionFieldPayload.tokenField.raw, "尚未接入")}`,
        `表达模式：${expressionMode || "尚未接入"}${proposalType ? ` · proposal=${proposalType}` : ""}${topIntent ? ` · intent=${topIntent}` : ""}${shouldSend ? ` · should_send=${shouldSend}` : ""}`,
        `表达权衡：intrinsic=${intrinsicValue || "尚未接入"} · cost=${speechCost || "尚未接入"}`,
        `抑制原因：${suppressionReason || "尚未接入"}`,
        `记忆牵引：${memoryBacking || "尚未接入"}`,
      ],
      roundId: actionFieldPayload.roundId ?? undefined,
      traceRef: actionFieldPayload.traceRef ?? undefined,
    };
  }

  const whyCurrentPayload = consoleState.whyCurrent;
  if (whyCurrentPayload) {
    const why = whyCurrentPayload.why;
    const authenticity = why.authenticity;
    const winnerAction =
      hydrated.actionField?.summary && hydrated.actionField.summary.includes(" · ")
        ? hydrated.actionField.summary.split(" · ")[0] ?? "等待接入"
        : "等待接入";
    const topDrivers = why.topDrivers
      .map((entry) => translateBrainIdentifier(entry.agent_name ?? entry.module_name ?? entry.name, ""))
      .filter((entry) => entry.length > 0);
    const initiative = normalizeRecord(why.initiative) ?? {};
    const authenticityDetails = [`真实性校验：${readConsoleText(authenticity.summary, "尚未接入")}`];
    const authenticitySource = translateAuthenticitySource(authenticity.source);
    if (authenticitySource) {
      authenticityDetails.push(`校验来源：${authenticitySource}`);
    }
    const guardAction = translateAuthenticityGuardAction(authenticity.guard_action);
    if (guardAction) {
      authenticityDetails.push(`校验动作：${guardAction}`);
    }
    const monologueDetails = summarizeMonologueStream(why.expressiveTrace);
    const initiativeMemoryBacking = summarizeInitiativeMemoryBacking(normalizeRecord(initiative.memory_backing));
    hydrated.why = {
      title: "解释层",
      summary: readConsoleText(why.summary, "等待本轮解释"),
      details: [
        `${whyCurrentLabel()}：${readConsoleText(why.summary, "等待本轮解释")}`,
        `最终输出动作：${translateActionName(readConsoleText(why.sampledAction, winnerAction), readConsoleText(why.sampledAction, winnerAction))}`,
        `主导模块：${topDrivers.length > 0 ? topDrivers.join("；") : "等待接入"}`,
        `表达模式：${readConsoleText(initiative.expression_mode, "尚未接入")} · intent=${readConsoleText(initiative.top_intent, "尚未接入")} · should_send=${typeof initiative.should_send === "boolean" ? String(initiative.should_send) : "尚未接入"}`,
        `独白/发言权衡：intrinsic=${readConsoleText(initiative.intrinsic_value, "尚未接入")} · cost=${readConsoleText(initiative.speech_cost, "尚未接入")}`,
        `抑制原因：${readConsoleText(initiative.suppression_reason, "尚未接入")}`,
        `记忆牵引：${initiativeMemoryBacking || "尚未接入"}`,
        ...monologueDetails,
        ...authenticityDetails,
      ],
      roundId: whyCurrentPayload.roundId ?? undefined,
      traceRef: whyCurrentPayload.traceRef ?? undefined,
    };
  }
  return hydrated;
}

function refreshAliveConsoleState(state: UiState): UiState {
  return {
    ...state,
    aliveConsole: buildAliveConsoleState(state),
  };
}

export function addUserLine(state: UiState, text: string): UiState {
  return refreshAliveConsoleState(appendLine(state, { kind: "user", text }));
}

export function addLocalLine(state: UiState, text: string, kind: UiLine["kind"] = "system"): UiState {
  return refreshAliveConsoleState(appendLine(state, { kind, text }));
}

export function clearLines(state: UiState): UiState {
  return refreshAliveConsoleState({ ...state, lines: [], assistantStreamActive: false });
}

function appendActivity(state: UiState, entry: ActivityEntry): UiState {
  return {
    ...state,
    activityRail: [...state.activityRail.slice(-11), entry],
  };
}

function appendAssistantDelta(state: UiState, delta: string): UiState {
  const last = state.lines.at(-1);
  if (last?.kind === "assistant" && state.assistantStreamActive) {
    return refreshAliveConsoleState({
      ...state,
      lines: [...state.lines.slice(0, -1), { kind: "assistant", text: `${last.text}${delta}` }],
      assistantStreamActive: true,
    });
  }
  return refreshAliveConsoleState(appendLine({ ...state, assistantStreamActive: true }, { kind: "assistant", text: delta }));
}

function appendTimeline(
  state: UiState,
  entry: UiState["toolTimeline"][number],
): UiState {
  return {
    ...state,
    toolTimeline: [...state.toolTimeline.slice(-23), entry],
  };
}

function replacePendingApproval(state: UiState, next: UiState["pendingApprovals"][number]): UiState["pendingApprovals"] {
  const index = state.pendingApprovals.findIndex((item) => item.callId === next.callId);
  if (index < 0) {
    return [...state.pendingApprovals, next];
  }
  return state.pendingApprovals.map((item, itemIndex) => (itemIndex === index ? next : item));
}

function normalizeLines(lines: unknown): UiLine[] {
  if (!Array.isArray(lines)) {
    return [];
  }
  return lines
    .filter((line): line is Record<string, unknown> => typeof line === "object" && line !== null)
    .map((line) => ({
      kind: String(line.kind ?? "system") as UiLine["kind"],
      text: String(line.text ?? ""),
    }));
}

function normalizeTimeline(entries: unknown): UiState["toolTimeline"] {
  if (!Array.isArray(entries)) {
    return [];
  }
  return entries
    .filter((entry): entry is Record<string, unknown> => typeof entry === "object" && entry !== null)
    .map((entry) => ({
      kind: String(entry.kind ?? "result") as UiState["toolTimeline"][number]["kind"],
      callId: String(entry.callId ?? entry.call_id ?? ""),
      tool: String(entry.tool ?? "unknown"),
      summary: entry.summary == null ? undefined : String(entry.summary),
      status: entry.status == null ? undefined : String(entry.status),
      ...normalizeLinkage(entry),
    }));
}

function normalizeActivityRail(
  timeline: UiState["toolTimeline"],
  approvals: UiState["pendingApprovals"],
): ActivityEntry[] {
  const fromTimeline = timeline.map((entry) => ({
    kind: entry.kind === "call" ? "tool" : entry.kind,
    label: entry.tool,
    summary: entry.summary,
    status: entry.status,
    callId: entry.callId,
    roundId: entry.roundId,
    traceRef: entry.traceRef,
  })) satisfies ActivityEntry[];
  const missingApprovals = approvals
    .filter((approval) => !fromTimeline.some((entry) => entry.kind === "approval" && entry.callId === approval.callId))
    .map((approval) => ({
      kind: "approval",
      label: approval.tool,
      summary: approval.summary,
      status: approval.status,
      callId: approval.callId,
      roundId: approval.roundId,
      traceRef: approval.traceRef,
    })) satisfies ActivityEntry[];
  return [...fromTimeline, ...missingApprovals].slice(-12);
}

function normalizePendingApprovals(entries: unknown): UiState["pendingApprovals"] {
  if (!Array.isArray(entries)) {
    return [];
  }
  return entries
    .filter((entry): entry is Record<string, unknown> => typeof entry === "object" && entry !== null)
    .map((entry) => ({
      callId: String(entry.callId ?? entry.call_id ?? ""),
      tool: String(entry.tool ?? "unknown"),
      args: (entry.args as Record<string, unknown> | undefined) ?? undefined,
      riskLevel: entry.riskLevel == null ? (entry.risk_level == null ? undefined : String(entry.risk_level)) : String(entry.riskLevel),
      summary: entry.summary == null ? undefined : String(entry.summary),
      actionPreview: entry.actionPreview == null ? (entry.action_preview == null ? undefined : String(entry.action_preview)) : String(entry.actionPreview),
      mode: entry.mode == null ? undefined : String(entry.mode),
      status: entry.status == null ? undefined : String(entry.status),
      runId: entry.runId == null ? (entry.run_id == null ? undefined : String(entry.run_id)) : String(entry.runId),
      choices: normalizeUiActions(entry.choices),
      ...normalizeLinkage(entry),
    }));
}

function normalizeUiActions(entries: unknown): UiAction[] {
  if (!Array.isArray(entries)) {
    return [];
  }
  return entries
    .filter((entry): entry is Record<string, unknown> => typeof entry === "object" && entry !== null)
    .map((entry) => ({
      id: String(entry.id ?? ""),
      label: String(entry.label ?? ""),
      kind: String(entry.kind ?? "command") as UiAction["kind"],
      value: String(entry.value ?? ""),
      disabled: Boolean(entry.disabled),
    }))
    .filter((entry) => entry.id.length > 0 && entry.label.length > 0 && entry.value.length > 0);
}

function normalizeApprovalCursor(entries: UiState["pendingApprovals"]): number {
  return entries.length === 0 ? 0 : Math.min(entries.length - 1, 0);
}

function appendLifecycleLine(state: UiState, status: string | undefined, previousStatus?: string): UiState {
  const normalized = String(status ?? "");
  if (normalized === "running" && previousStatus !== "running") {
    const text = previousStatus === "paused" ? "任务已恢复。" : "任务已启动。";
    return appendLine(state, { kind: "system", text });
  }
  if (normalized === "paused" && previousStatus !== "paused") {
    return appendLine(state, { kind: "system", text: "任务已暂停。" });
  }
  if (normalized === "aborted" && previousStatus !== "aborted") {
    return appendLine(state, { kind: "system", text: "任务已中止。" });
  }
  if ((normalized === "completed" || normalized === "done") && previousStatus !== normalized) {
    return appendLine(state, { kind: "system", text: "任务已完成。" });
  }
  return state;
}

function normalizeCognitiveSnapshot(snapshot: Record<string, unknown> | undefined): CognitiveSnapshotState {
  const vitalSigns = (snapshot?.vital_signs as Record<string, unknown> | undefined) ?? {};
  const identity = (snapshot?.identity as Record<string, unknown> | undefined) ?? {};
  const authenticity = (snapshot?.authenticity as Record<string, unknown> | undefined) ?? {};
  return {
    coreGoal: String(snapshot?.core_goal ?? "暂无"),
    currentIntent: String(snapshot?.current_intent ?? "暂无"),
    vitalSigns: {
      mood: Number(vitalSigns.mood ?? 0),
      bodyEnergy: Number(vitalSigns.body_energy ?? 0),
      affectResidue: Number(vitalSigns.affect_residue ?? 0),
      focus: String(vitalSigns.focus ?? "unknown"),
      mode: String(vitalSigns.mode ?? "interactive"),
    },
    identity: {
      displayName: String(identity.display_name ?? "暂无"),
      continuity: String(identity.continuity ?? "暂无"),
    },
    authenticity: {
      summary: String(authenticity.summary ?? "暂无"),
      source: String(authenticity.source ?? "none"),
      guardAction: String(authenticity.guard_action ?? "none"),
    },
  };
}

function normalizeConsoleStatePayload(payload: ConsoleRefreshPayload | undefined): ConsoleState {
  if (!payload) {
    return emptyConsoleState();
  }

  const statePayload = normalizeRecord(payload.state);
  const actionFieldPayload = normalizeRecord(payload.action_field);
  const timelinePayload = normalizeRecord(payload.timeline);
  const whyCurrentPayload = normalizeRecord(payload.why_current);
  const stateBrain = normalizeRecord(statePayload?.brain_state);
  const stateSession = normalizeRecord(statePayload?.session);
  const stateRun = normalizeRecord(statePayload?.run);
  const stateLongRun = normalizeRecord(statePayload?.long_run);
  const stateCurrentRound = normalizeRecord(statePayload?.current_round);
  const stateNeuromodulators = normalizeRecord(statePayload?.neuromodulators);
  const stateMotivationPool = normalizeRecord(statePayload?.motivation_pool);
  const whyPayload = normalizeRecord(whyCurrentPayload?.why);

  const normalizedState: ConsoleStateState | null = statePayload
    ? {
        brainState: {
          mode: readConsoleNullableText(stateBrain?.mode),
          vitality: readConsoleNullableNumber(stateBrain?.vitality),
          selfContinuity: readConsoleNullableText(stateBrain?.self_continuity),
          authenticityPressure: readConsoleNullableText(stateBrain?.authenticity_pressure),
          longRunDriftRisk: readConsoleNullableText(stateBrain?.long_run_drift_risk),
        },
        neuromodulators: {
          dopamine: readConsoleNullableNumber(stateNeuromodulators?.dopamine),
          noradrenaline: readConsoleNullableNumber(stateNeuromodulators?.noradrenaline),
          serotonin: readConsoleNullableNumber(stateNeuromodulators?.serotonin),
          acetylcholine: readConsoleNullableNumber(stateNeuromodulators?.acetylcholine),
          gaba: readConsoleNullableNumber(stateNeuromodulators?.gaba),
        },
        motivationPool: {
          activeMotivations: normalizeRecordList(stateMotivationPool?.active_motivations),
          raw: stateMotivationPool ?? {},
        },
        longRun: {
          dream: normalizeRecord(stateLongRun?.dream) ?? {},
          traceStorage: normalizeRecord(stateLongRun?.trace_storage) ?? {},
        },
        currentRound: {
          roundId: readConsoleNullableNumber(stateCurrentRound?.round_id),
          sampledAction: readConsoleNullableText(stateCurrentRound?.sampled_action),
          traceRef: readConsoleNullableText(stateCurrentRound?.trace_ref),
          routeType: readConsoleNullableText(stateCurrentRound?.route_type),
          routeBudgetMs: readConsoleNullableNumber(stateCurrentRound?.route_budget_ms),
          causeType: readConsoleNullableText(stateCurrentRound?.cause_type),
          causeLabel: readConsoleNullableText(stateCurrentRound?.cause_label),
          mode: readConsoleNullableText(stateCurrentRound?.mode),
          modeLabel: readConsoleNullableText(stateCurrentRound?.mode_label),
          activationSet: Array.isArray(stateCurrentRound?.activation_set)
            ? stateCurrentRound.activation_set.filter((value): value is string => typeof value === "string" && value.trim().length > 0)
            : [],
          activationReason: Array.isArray(stateCurrentRound?.activation_reason)
            ? stateCurrentRound.activation_reason.filter((value): value is string => typeof value === "string" && value.trim().length > 0)
            : [],
          memoryTiersRead: Array.isArray(stateCurrentRound?.memory_tiers_read)
            ? stateCurrentRound.memory_tiers_read.filter((value): value is string => typeof value === "string" && value.trim().length > 0)
            : [],
          packetSummary: normalizeRecord(stateCurrentRound?.packet_summary) ?? {},
          backgroundJobs: normalizeRecordList(stateCurrentRound?.background_jobs),
          deepenReason: readConsoleNullableText(stateCurrentRound?.deepen_reason),
          modelCallCount: readConsoleNullableNumber(stateCurrentRound?.model_call_count),
          totalTurnMs: readConsoleNullableNumber(stateCurrentRound?.total_turn_ms),
          modelWaitMs: readConsoleNullableNumber(stateCurrentRound?.model_wait_ms),
          localComputeMs: readConsoleNullableNumber(stateCurrentRound?.local_compute_ms),
          latencyDominant: readConsoleNullableText(stateCurrentRound?.latency_dominant),
          latencySummary: readConsoleNullableText(stateCurrentRound?.latency_summary),
          parallelTaskCount: readConsoleNullableNumber(stateCurrentRound?.parallel_task_count),
        },
        session: {
          sessionId: readConsoleNullableText(stateSession?.session_id),
          mode: readConsoleNullableText(stateSession?.mode),
          safeMode: typeof stateSession?.safe_mode === "boolean" ? stateSession.safe_mode : null,
        },
        run: {
          runId: readConsoleNullableText(stateRun?.run_id),
          status: readConsoleNullableText(stateRun?.status),
          raw: stateRun ?? {},
        },
        cognitiveSnapshot: normalizeCognitiveSnapshot(normalizeRecord(statePayload.cognitive_snapshot)),
      }
    : null;

  return {
    state: normalizedState,
    actionField: actionFieldPayload
      ? {
          roundId: readConsoleNullableNumber(actionFieldPayload.round_id),
          traceRef: readConsoleNullableText(actionFieldPayload.trace_ref),
          topActions: normalizeRecordList(actionFieldPayload.top_actions).map(normalizeConsoleActionScore),
          winner: normalizeConsoleActionScore(normalizeRecord(actionFieldPayload.winner) ?? {}),
          conflict: normalizeRecord(actionFieldPayload.conflict) ?? {},
          tokenField: { raw: normalizeRecord(actionFieldPayload.token_field) ?? {} },
          contributionStack: normalizeRecordList(actionFieldPayload.contribution_stack).map(normalizeConsoleContribution),
          competingPeaks: normalizeRecordList(actionFieldPayload.competing_peaks).map(normalizeConsoleActionScore),
          expressive: normalizeRecord(actionFieldPayload.expressive) ?? {},
        }
      : null,
    timeline: timelinePayload
      ? {
          roundId: readConsoleNullableNumber(timelinePayload.round_id),
          traceRef: readConsoleNullableText(timelinePayload.trace_ref),
          events: normalizeRecordList(timelinePayload.events).map(
            (entry): CognitiveTimelineEntry => ({
              type: readConsoleText(entry.type, "unknown"),
              label: readConsoleText(entry.label, "unknown"),
              summary: readConsoleText(entry.summary, "尚未接入"),
            }),
          ),
        }
      : null,
    whyCurrent: whyCurrentPayload
      ? {
          roundId: readConsoleNullableNumber(whyCurrentPayload.round_id),
          traceRef: readConsoleNullableText(whyCurrentPayload.trace_ref),
          why: {
            summary: readConsoleText(whyPayload?.summary, "等待本轮解释"),
            sampledAction: readConsoleNullableText(whyPayload?.sampled_action),
            topDrivers: normalizeRecordList(whyPayload?.top_drivers),
            vitalitySnapshot: normalizeRecord(whyPayload?.vitality_snapshot) ?? {},
            authenticity: normalizeRecord(whyPayload?.authenticity) ?? {},
            initiative: normalizeRecord(whyPayload?.initiative) ?? {},
            expressiveTrace: normalizeExpressiveTrace(whyPayload?.expressive_trace ?? whyPayload?.expressiveTrace),
          },
        }
      : null,
    whyNot: normalizeWhyNotPayload(payload),
  };
}

function normalizeWhyNotPayload(payload: unknown): ConsoleState["whyNot"] {
  if (!payload) {
    return null;
  }
  const normalizedPayload = normalizeRecord(payload);
  if (!normalizedPayload) {
    return null;
  }
  const envelopePayload =
    "action" in normalizedPayload || "round_id" in normalizedPayload || "trace_ref" in normalizedPayload
      ? normalizedPayload
      : (normalizeRecord(normalizedPayload.why_not) ?? normalizedPayload);
  const detailPayload = normalizeRecord(envelopePayload.why_not) ?? envelopePayload;
  if (!("action" in envelopePayload) && !("blocked_by" in detailPayload) && !("selected_action" in detailPayload)) {
    return null;
  }
  return {
    roundId: readConsoleNullableNumber(envelopePayload.round_id),
    traceRef: readConsoleNullableText(envelopePayload.trace_ref),
    action: readConsoleNullableText(envelopePayload.action),
    whyNot: {
      selected_action: detailPayload.selected_action,
      candidate_score: detailPayload.candidate_score,
      blocked_by: Array.isArray(detailPayload.blocked_by) ? detailPayload.blocked_by : [],
      competing_peaks: Array.isArray(detailPayload.competing_peaks) ? detailPayload.competing_peaks : [],
      stacked_contributions: Array.isArray(detailPayload.stacked_contributions) ? detailPayload.stacked_contributions : [],
      expressive_trace: normalizeExpressiveTrace(detailPayload.expressive_trace ?? detailPayload.expressiveTrace),
      summary: readConsoleNullableText(detailPayload.summary),
    },
  };
}

function isWhyPayload(payload: Record<string, unknown> | null | undefined): boolean {
  if (!payload) {
    return false;
  }
  return "goal_summary" in payload || "goal" in payload || "current_step" in payload || "stop_reason" in payload || "last_tool" in payload;
}

export function applyBridgeEvent(state: UiState, event: OutboundBridgeEvent): UiState {
  if (event.type === "session_started") {
    const restoredLines = normalizeLines(event.session.transcript_lines);
    const restoredTimeline = normalizeTimeline(event.session.tool_timeline);
    const restoredApprovals = normalizePendingApprovals(event.session.approvals_pending);
    const transcriptMode =
      String(event.session.transcript_mode ?? "") === "compact" || Boolean(event.session.compact)
        ? "compact"
        : "full";
    return refreshAliveConsoleState({
      ...state,
      activeSessionId: String(event.session.session_id ?? "") || state.activeSessionId,
      activeRunId: String(event.session.active_run_id ?? event.session.last_run_id ?? "") || state.activeRunId,
      sessionMeta: event.session,
      console: emptyConsoleState(),
      lines: restoredLines,
      toolTimeline: restoredTimeline,
      pendingApprovals: restoredApprovals,
      actionBar: {
        primary: [],
        secondary: [],
        selectedIndex: 0,
      },
      activityRail: normalizeActivityRail(restoredTimeline, restoredApprovals),
      transcriptMode,
      permissionMode: String(event.session.permission_mode ?? state.permissionMode),
      approvalCursor: normalizeApprovalCursor(restoredApprovals),
    });
  }
  if (event.type === "assistant_token") {
    return appendAssistantDelta(state, event.delta);
  }
  if (event.type === "run_status") {
    const nextState = {
      ...state,
      activeSessionId: event.session_id,
      activeRunId: String(event.run.run_id ?? "") || state.activeRunId,
      run: { ...event.run, round_id: event.round_id, trace_ref: event.trace_ref },
    };
    return refreshAliveConsoleState(appendLifecycleLine(nextState, String(event.run.status ?? ""), String(state.run?.status ?? "")));
  }
  if (event.type === "step_update") {
    return refreshAliveConsoleState(appendActivity(
      {
        ...state,
        activeSessionId: event.session_id,
        steps: replaceByKey(state.steps, "step_id", { ...event.step, round_id: event.round_id, trace_ref: event.trace_ref }),
      },
      {
        kind: "step",
        label: String(event.step.title ?? event.step.step_id ?? "unknown"),
        status: event.step.status == null ? undefined : String(event.step.status),
        roundId: event.round_id,
        traceRef: event.trace_ref,
      },
    ));
  }
  if (event.type === "tool_call") {
    return refreshAliveConsoleState(appendActivity(
      appendTimeline(
        { ...state, activeSessionId: event.session_id },
        {
          kind: "call",
          callId: event.call_id,
          tool: event.tool,
          summary: event.summary,
          status: event.status,
          roundId: event.round_id,
          traceRef: event.trace_ref,
        }
      ),
      {
        kind: "tool",
        label: event.tool,
        summary: event.summary,
        status: event.status,
        callId: event.call_id,
        roundId: event.round_id,
        traceRef: event.trace_ref,
      }
    ));
  }
  if (event.type === "tool_result") {
    const nextApprovals = state.pendingApprovals.filter((item) => item.callId !== event.call_id);
    return refreshAliveConsoleState(appendActivity(
      appendTimeline(
        {
          ...state,
          activeSessionId: event.session_id,
          tools: replaceByKey(state.tools, "call_id", { ...event.result, call_id: event.call_id, round_id: event.round_id, trace_ref: event.trace_ref }),
          pendingApprovals: nextApprovals,
          approvalCursor: nextApprovals.length === 0 ? 0 : Math.min(state.approvalCursor, nextApprovals.length - 1),
        },
        {
          kind: "result",
          callId: event.call_id,
          tool: String(event.result.tool_name ?? "unknown"),
          summary: String(event.result.summary ?? ""),
          status: event.result.status == null ? undefined : String(event.result.status),
          roundId: event.round_id,
          traceRef: event.trace_ref,
        }
      ),
      {
        kind: "result",
        label: String(event.result.tool_name ?? "unknown"),
        summary: String(event.result.summary ?? ""),
        status: event.result.status == null ? undefined : String(event.result.status),
        callId: event.call_id,
        roundId: event.round_id,
        traceRef: event.trace_ref,
      }
    ));
  }
  if (event.type === "assistant_final") {
    const nextWhyNot = normalizeWhyNotPayload(event.payload);
    const nextLastWhy = isWhyPayload(event.payload) ? (event.payload ?? state.lastWhy) : state.lastWhy;
    const nextState = {
      ...state,
      assistantStreamActive: false,
      lastWhy: nextLastWhy,
      console: {
        ...state.console,
        whyNot: nextWhyNot ?? state.console.whyNot,
      },
    };
    if (state.assistantStreamActive) {
      return refreshAliveConsoleState(nextState);
    }
    return refreshAliveConsoleState(appendLine(nextState, { kind: "assistant", text: event.message }));
  }
  if (event.type === "error") {
    return refreshAliveConsoleState(appendLine(state, { kind: "error", text: event.message }));
  }
  if (event.type === "session_ended") {
    return state;
  }
  if (event.type === "approval_request") {
    const nextApprovals = replacePendingApproval(state, {
      callId: event.call_id,
      tool: event.tool,
      args: event.args,
      riskLevel: event.risk_level,
      summary: event.summary,
      actionPreview: event.action_preview,
      mode: event.mode,
      status: event.status,
      runId: event.run_id,
      choices: normalizeUiActions(event.choices),
      roundId: event.round_id,
      traceRef: event.trace_ref,
    });
    const nextState = appendActivity(
      appendTimeline(
        {
          ...state,
          activeSessionId: event.session_id,
          pendingApprovals: nextApprovals,
          approvalCursor: nextApprovals.findIndex((item) => item.callId === event.call_id),
        },
        {
          kind: "approval",
          callId: event.call_id,
          tool: event.tool,
          summary: event.summary,
          status: event.status ?? "pending",
          roundId: event.round_id,
          traceRef: event.trace_ref,
        }
      ),
      {
        kind: "approval",
        label: event.tool,
        summary: event.summary,
        status: event.status ?? "pending",
        callId: event.call_id,
        roundId: event.round_id,
        traceRef: event.trace_ref,
      }
    );
    return refreshAliveConsoleState(appendLine(nextState, {
      kind: "system",
      text: `等待审批：${event.tool}`,
    }));
  }
  if (event.type === "sidebar_snapshot") {
    const normalizedConsole = normalizeConsoleStatePayload(event.console);
    const incomingRoundId =
      normalizedConsole.state?.currentRound.roundId ??
      normalizedConsole.actionField?.roundId ??
      normalizedConsole.timeline?.roundId ??
      normalizedConsole.whyCurrent?.roundId ??
      (typeof event.round_id === "number" ? event.round_id : null);
    const nextState = {
      ...state,
      activeSessionId: event.session_id,
      activeRunId: String(event.status?.run_id ?? event.statusline?.run_id ?? "") || state.activeRunId,
      sessionMeta: event.session ?? state.sessionMeta,
      run: event.status ?? state.run,
      lastWhy: event.why ?? state.lastWhy,
      steps: event.steps ?? state.steps,
      tools: event.tools ?? state.tools,
      sidebarSnapshot: {
        goalSummary: event.goal_summary,
        currentStep: event.current_step,
        reasonSummary: event.reason_summary,
        lastTool: event.last_tool,
        runStatus: event.run_status,
        permissionMode: event.permission_mode,
        pendingApprovalCount: event.pending_approval_count,
        cognitiveSnapshot: normalizeCognitiveSnapshot(event.cognitive_snapshot),
        modelStatus: event.model_status,
        console: event.console,
        roundId: event.round_id,
        traceRef: event.trace_ref,
        uiActions: {
          primary: normalizeUiActions(event.ui_actions?.primary),
          secondary: normalizeUiActions(event.ui_actions?.secondary),
        },
      },
      console: {
        ...normalizedConsole,
        whyNot:
          normalizedConsole.whyNot ??
          (state.console.whyNot && state.console.whyNot.roundId === incomingRoundId ? state.console.whyNot : null),
      },
      statusline: ((event.statusline ?? undefined) as unknown as UiState["statusline"]) ?? state.statusline,
      actionBar: {
        primary: normalizeUiActions(event.ui_actions?.primary),
        secondary: normalizeUiActions(event.ui_actions?.secondary),
        selectedIndex: 0,
      },
      permissionMode: event.permission_mode || state.permissionMode,
      pendingApprovals:
        Array.isArray(event.session?.approvals_pending)
          ? normalizePendingApprovals(event.session.approvals_pending)
          : state.pendingApprovals,
    };
    return refreshAliveConsoleState(nextState);
  }
  return state;
}
