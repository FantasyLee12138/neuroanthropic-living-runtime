import type { CognitiveSnapshotState, OutboundBridgeEvent, UiLine, UiState } from "../types.js";

export function createInitialUiState(): UiState {
  return {
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
    transcriptMode: "full",
    promptHistoryByCwd: {},
    pendingApprovals: [],
    permissionMode: "plan",
    assistantStreamActive: false,
  };
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

export function addUserLine(state: UiState, text: string): UiState {
  return appendLine(state, { kind: "user", text });
}

export function addLocalLine(state: UiState, text: string, kind: UiLine["kind"] = "system"): UiState {
  return appendLine(state, { kind, text });
}

export function clearLines(state: UiState): UiState {
  return { ...state, lines: [], assistantStreamActive: false };
}

function appendAssistantDelta(state: UiState, delta: string): UiState {
  const last = state.lines.at(-1);
  if (last?.kind === "assistant" && state.assistantStreamActive) {
    return {
      ...state,
      lines: [...state.lines.slice(0, -1), { kind: "assistant", text: `${last.text}${delta}` }],
      assistantStreamActive: true,
    };
  }
  return appendLine({ ...state, assistantStreamActive: true }, { kind: "assistant", text: delta });
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
    }));
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
    }));
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

export function applyBridgeEvent(state: UiState, event: OutboundBridgeEvent): UiState {
  if (event.type === "session_started") {
    const restoredLines = normalizeLines(event.session.transcript_lines);
    const restoredTimeline = normalizeTimeline(event.session.tool_timeline);
    const restoredApprovals = normalizePendingApprovals(event.session.approvals_pending);
    const transcriptMode =
      String(event.session.transcript_mode ?? "") === "compact" || Boolean(event.session.compact)
        ? "compact"
        : "full";
    return {
      ...state,
      activeSessionId: String(event.session.session_id ?? "") || state.activeSessionId,
      activeRunId: String(event.session.active_run_id ?? event.session.last_run_id ?? "") || state.activeRunId,
      sessionMeta: event.session,
      lines: restoredLines,
      toolTimeline: restoredTimeline,
      pendingApprovals: restoredApprovals,
      transcriptMode,
      permissionMode: String(event.session.permission_mode ?? state.permissionMode),
    };
  }
  if (event.type === "assistant_token") {
    return appendAssistantDelta(state, event.delta);
  }
  if (event.type === "run_status") {
    return {
      ...state,
      activeSessionId: event.session_id,
      activeRunId: String(event.run.run_id ?? "") || state.activeRunId,
      run: event.run,
    };
  }
  if (event.type === "step_update") {
    return appendLine({
      ...state,
      activeSessionId: event.session_id,
      steps: replaceByKey(state.steps, "step_id", event.step)
    }, { kind: "system", text: `Step: ${String(event.step.title ?? event.step.step_id ?? "unknown")}` });
  }
  if (event.type === "tool_call") {
    return appendLine(
      appendTimeline(
        { ...state, activeSessionId: event.session_id },
        { kind: "call", callId: event.call_id, tool: event.tool, summary: event.summary, status: event.status }
      ),
      { kind: "system", text: `Tool: ${event.tool}${event.summary ? ` - ${event.summary}` : ""}` }
    );
  }
  if (event.type === "tool_result") {
    return appendLine(
      appendTimeline(
        {
          ...state,
          activeSessionId: event.session_id,
          tools: replaceByKey(state.tools, "call_id", { ...event.result, call_id: event.call_id }),
          pendingApprovals: state.pendingApprovals.filter((item) => item.callId !== event.call_id),
        },
        {
          kind: "result",
          callId: event.call_id,
          tool: String(event.result.tool_name ?? "unknown"),
          summary: String(event.result.summary ?? ""),
          status: event.result.status == null ? undefined : String(event.result.status),
        }
      ),
      { kind: "system", text: `Result: ${String(event.result.tool_name ?? event.call_id)}` }
    );
  }
  if (event.type === "assistant_final") {
    const nextState = {
      ...state,
      assistantStreamActive: false,
      lastWhy: event.payload && !("cognitive_snapshot" in event.payload) ? event.payload : state.lastWhy
    };
    if (state.assistantStreamActive) {
      return nextState;
    }
    return appendLine(nextState, { kind: "assistant", text: event.message });
  }
  if (event.type === "error") {
    return appendLine(state, { kind: "error", text: event.message });
  }
  if (event.type === "session_ended") {
    return state;
  }
  if (event.type === "approval_request") {
    const nextState = appendTimeline(
      {
        ...state,
        activeSessionId: event.session_id,
        pendingApprovals: replacePendingApproval(state, {
          callId: event.call_id,
          tool: event.tool,
          args: event.args,
          riskLevel: event.risk_level,
          summary: event.summary,
          actionPreview: event.action_preview,
          mode: event.mode,
          status: event.status,
          runId: event.run_id,
        }),
      },
      {
        kind: "approval",
        callId: event.call_id,
        tool: event.tool,
        summary: event.summary,
        status: event.status ?? "pending",
      }
    );
    return appendLine(nextState, {
      kind: "system",
      text: `Approval: ${event.tool}${event.summary ? ` - ${event.summary}` : ""}`,
    });
  }
  if (event.type === "sidebar_snapshot") {
    return {
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
      },
      statusline: ((event.statusline ?? undefined) as unknown as UiState["statusline"]) ?? state.statusline,
      permissionMode: event.permission_mode || state.permissionMode,
      pendingApprovals:
        Array.isArray(event.session?.approvals_pending)
          ? (event.session.approvals_pending as UiState["pendingApprovals"])
          : state.pendingApprovals,
    };
  }
  return state;
}
