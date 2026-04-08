export type BridgeControlCommand =
  | "status"
  | "why"
  | "steps"
  | "tools"
  | "state"
  | "pause"
  | "resume"
  | "abort"
  | "model"
  | "mode"
  | "permissions"
  | "dream"
  | "probability"
  | "endogenous"
  | "initiative"
  | "monologue"
  | "why-motivation"
  | "replay-motivation"
  | "replay"
  | "why-not"
  | "what-changed"
  | "eval";

export type PanelKey = "status" | "why" | "steps" | "tools" | "cognition" | "approvals" | "meta" | null;

export type InboundBridgeEvent =
  | { type: "start_session"; session_id: string; cwd: string; persist_current?: boolean }
  | { type: "user_turn"; session_id: string; text: string }
  | { type: "control_command"; session_id: string; command: BridgeControlCommand; value?: string }
  | { type: "approve"; session_id: string; call_id: string; approved: boolean }
  | { type: "close_session"; session_id: string; detach?: boolean; transcript_mode?: "full" | "compact" };

export type OutboundBridgeEvent =
  | { type: "session_started"; session: Record<string, unknown> }
  | { type: "assistant_token"; session_id: string; delta: string }
  | { type: "run_status"; session_id: string; run: Record<string, any>; round_id?: number; trace_ref?: string }
  | { type: "step_update"; session_id: string; step: Record<string, any>; round_id?: number; trace_ref?: string }
  | {
      type: "tool_call";
      session_id: string;
      call_id: string;
      tool: string;
      args: Record<string, unknown>;
      summary?: string;
      run_id?: string;
      status?: string;
      round_id?: number;
      trace_ref?: string;
    }
  | { type: "tool_result"; session_id: string; call_id: string; result: Record<string, any>; round_id?: number; trace_ref?: string }
  | {
      type: "approval_request";
      session_id: string;
      call_id: string;
      tool: string;
      args?: Record<string, unknown>;
      risk_level?: string;
      summary?: string;
      action_preview?: string;
      mode?: string;
      status?: string;
      actions?: string[];
      choices?: UiAction[];
      run_id?: string;
      approved?: boolean | null;
      round_id?: number;
      trace_ref?: string;
    }
  | {
      type: "sidebar_snapshot";
      session_id: string;
      goal_summary: string;
      current_step: string;
      reason_summary: string;
      last_tool: string;
      run_status: string;
      permission_mode: string;
      pending_approval_count: number;
      cognitive_snapshot?: Record<string, unknown>;
      status?: Record<string, any>;
      why?: Record<string, unknown>;
      steps?: Array<Record<string, any>>;
      tools?: Array<Record<string, any>>;
      session?: Record<string, unknown>;
      statusline?: Record<string, unknown>;
      model_status?: Record<string, unknown>;
      console?: ConsoleRefreshPayload;
      round_id?: number;
      trace_ref?: string;
      ui_actions?: {
        primary?: UiAction[];
        secondary?: UiAction[];
      };
    }
  | { type: "assistant_final"; session_id?: string; run_id?: string; message: string; payload?: Record<string, unknown>; round_id?: number; trace_ref?: string }
  | { type: "error"; session_id?: string; message: string }
  | { type: "session_ended"; session: Record<string, unknown> };

export type UiLineKind = "user" | "assistant" | "system" | "error";

export interface UiLine {
  kind: UiLineKind;
  text: string;
}

export interface PendingApproval {
  callId: string;
  tool: string;
  args?: Record<string, unknown>;
  riskLevel?: string;
  summary?: string;
  actionPreview?: string;
  mode?: string;
  status?: string;
  runId?: string;
  choices?: UiAction[];
  roundId?: number;
  traceRef?: string;
}

export interface ToolTimelineEntry {
  kind: "call" | "approval" | "result";
  callId: string;
  tool: string;
  summary?: string;
  status?: string;
  roundId?: number;
  traceRef?: string;
}

export interface ActivityEntry {
  kind: "step" | "tool" | "result" | "approval";
  label: string;
  summary?: string;
  status?: string;
  callId?: string;
  roundId?: number;
  traceRef?: string;
}

export interface CognitiveSnapshotState {
  coreGoal: string;
  currentIntent: string;
  vitalSigns: {
    mood: number;
    bodyEnergy: number;
    affectResidue: number;
    focus: string;
    mode: string;
  };
  identity: {
    displayName: string;
    continuity: string;
  };
  authenticity: {
    summary: string;
    source: string;
    guardAction: string;
  };
}

export interface UiAction {
  id: string;
  label: string;
  kind: "command" | "drawer" | "approval" | "approval_nav";
  value: string;
  disabled: boolean;
}

export interface SidebarSnapshotState {
  goalSummary: string;
  currentStep: string;
  reasonSummary: string;
  lastTool: string;
  runStatus: string;
  permissionMode: string;
  pendingApprovalCount: number;
  cognitiveSnapshot: CognitiveSnapshotState;
  modelStatus?: Record<string, unknown>;
  console?: ConsoleRefreshPayload;
  roundId?: number;
  traceRef?: string;
  uiActions?: {
    primary: UiAction[];
    secondary: UiAction[];
  };
}

export interface AliveConsoleColumnState {
  title: string;
  summary: string;
  details: string[];
  roundId?: number;
  traceRef?: string;
}

export interface AliveConsoleState {
  state: AliveConsoleColumnState;
  actionField: AliveConsoleColumnState;
  why: AliveConsoleColumnState;
}

export interface BrainStateState {
  mode: string | null;
  vitality: number | null;
  selfContinuity: string | null;
  authenticityPressure: string | null;
  longRunDriftRisk: string | null;
}

export interface NeuromodulatorState {
  dopamine: number | null;
  noradrenaline: number | null;
  serotonin: number | null;
  acetylcholine: number | null;
  gaba: number | null;
}

export interface MotivationPoolState {
  activeMotivations: Array<Record<string, unknown>>;
  raw: Record<string, unknown>;
}

export interface LongRunState {
  dream: Record<string, unknown>;
  traceStorage: Record<string, unknown>;
}

export interface CurrentRoundState {
  roundId: number | null;
  sampledAction: string | null;
  traceRef: string | null;
  routeType?: string | null;
  causeType: string | null;
  causeLabel?: string | null;
  mode?: string | null;
  modeLabel?: string | null;
  totalTurnMs?: number | null;
  modelWaitMs?: number | null;
  localComputeMs?: number | null;
  latencyDominant?: string | null;
  latencySummary?: string | null;
  parallelTaskCount?: number | null;
}

export interface ConsoleSessionState {
  sessionId: string | null;
  mode: string | null;
  safeMode: boolean | null;
}

export interface ConsoleRunState {
  runId: string | null;
  status: string | null;
  raw: Record<string, unknown>;
}

export interface ConsoleStatePayload {
  brain_state?: Record<string, unknown>;
  neuromodulators?: Record<string, unknown>;
  motivation_pool?: Record<string, unknown>;
  long_run?: Record<string, unknown>;
  current_round?: Record<string, unknown> | null;
  session?: Record<string, unknown>;
  run?: Record<string, unknown>;
  cognitive_snapshot?: Record<string, unknown>;
}

export interface ConsoleStateState {
  brainState: BrainStateState;
  neuromodulators: NeuromodulatorState;
  motivationPool: MotivationPoolState;
  longRun: LongRunState;
  currentRound: CurrentRoundState;
  session: ConsoleSessionState;
  run: ConsoleRunState;
  cognitiveSnapshot: CognitiveSnapshotState;
}

export interface ActionScoreState {
  action: string | null;
  score: number | null;
}

export interface TokenFieldState {
  raw: Record<string, unknown>;
}

export interface ContributionStackState {
  source: string | null;
  weight: number | null;
  raw: Record<string, unknown>;
}

export interface ConsoleActionFieldPayload {
  round_id?: number | null;
  trace_ref?: string | null;
  top_actions?: Array<Record<string, unknown>>;
  winner?: Record<string, unknown>;
  conflict?: Record<string, unknown>;
  token_field?: Record<string, unknown>;
  contribution_stack?: Array<Record<string, unknown>>;
  competing_peaks?: Array<Record<string, unknown>>;
  expressive?: Record<string, unknown>;
}

export interface ActionFieldState {
  roundId: number | null;
  traceRef: string | null;
  topActions: ActionScoreState[];
  winner: ActionScoreState;
  conflict: Record<string, unknown>;
  tokenField: TokenFieldState;
  contributionStack: ContributionStackState[];
  competingPeaks: ActionScoreState[];
  expressive?: Record<string, unknown>;
}

export interface ConsoleTimelinePayload {
  round_id?: number | null;
  trace_ref?: string | null;
  events?: Array<Record<string, unknown>>;
}

export interface CognitiveTimelineEntry {
  type: string;
  label: string;
  summary: string;
}

export interface CognitiveTimelineState {
  roundId: number | null;
  traceRef: string | null;
  events: CognitiveTimelineEntry[];
}

export interface ConsoleWhyCurrentPayload {
  round_id?: number | null;
  trace_ref?: string | null;
  why?: Record<string, unknown>;
}

export interface ConsoleWhyNotPayload {
  round_id?: number | null;
  trace_ref?: string | null;
  action?: string | null;
  why_not?: Record<string, unknown>;
}

export interface WhyCurrentBodyState {
  summary: string;
  sampledAction: string | null;
  topDrivers: Array<Record<string, unknown>>;
  vitalitySnapshot: Record<string, unknown>;
  authenticity: Record<string, unknown>;
  initiative?: Record<string, unknown>;
  expressiveTrace?: Record<string, unknown>;
}

export interface WhyCurrentState {
  roundId: number | null;
  traceRef: string | null;
  why: WhyCurrentBodyState;
}

export interface WhyNotState {
  roundId: number | null;
  traceRef: string | null;
  action: string | null;
  whyNot: Record<string, unknown>;
}

export interface ConsoleRefreshPayload {
  state?: ConsoleStatePayload;
  action_field?: ConsoleActionFieldPayload;
  timeline?: ConsoleTimelinePayload;
  why_current?: ConsoleWhyCurrentPayload;
  why_not?: ConsoleWhyNotPayload;
}

export interface ConsoleState {
  state: ConsoleStateState | null;
  actionField: ActionFieldState | null;
  timeline: CognitiveTimelineState | null;
  whyCurrent: WhyCurrentState | null;
  whyNot: WhyNotState | null;
}

export interface StatusLineState {
  cwd: string;
  git: string;
  permission_mode: string;
  model: string;
  run_status: string;
  session_id: string;
  run_id: string;
}

export interface UiState {
  activeSessionId: string | null;
  activeRunId: string | null;
  sessionMeta: Record<string, unknown> | null;
  run: Record<string, any> | null;
  steps: Array<Record<string, any>>;
  tools: Array<Record<string, any>>;
  toolTimeline: ToolTimelineEntry[];
  lastWhy: Record<string, unknown> | null;
  sidebarSnapshot: SidebarSnapshotState | null;
  statusline: StatusLineState | null;
  lines: UiLine[];
  activityRail: ActivityEntry[];
  transcriptMode: "full" | "compact";
  promptHistoryByCwd: Record<string, string[]>;
  pendingApprovals: PendingApproval[];
  actionBar: {
    primary: UiAction[];
    secondary: UiAction[];
    selectedIndex: number;
  };
  permissionMode: string;
  assistantStreamActive: boolean;
  detailDrawer: PanelKey;
  focusZone: "input" | "transcript" | "drawer" | "actions" | "approval";
  approvalCursor: number;
  console: ConsoleState;
  aliveConsole: AliveConsoleState;
}
