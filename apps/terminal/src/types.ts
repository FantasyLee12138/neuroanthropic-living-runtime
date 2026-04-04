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
  | "dream";

export type PanelKey = "status" | "why" | "steps" | "tools" | "state" | null;

export type InboundBridgeEvent =
  | { type: "start_session"; session_id: string; cwd: string }
  | { type: "user_turn"; session_id: string; text: string }
  | { type: "control_command"; session_id: string; command: BridgeControlCommand; value?: string }
  | { type: "approve"; session_id: string; call_id: string; approved: boolean }
  | { type: "close_session"; session_id: string };

export type OutboundBridgeEvent =
  | { type: "session_started"; session: Record<string, unknown> }
  | { type: "assistant_token"; session_id: string; delta: string }
  | { type: "run_status"; session_id: string; run: Record<string, any> }
  | { type: "step_update"; session_id: string; step: Record<string, any> }
  | {
      type: "tool_call";
      session_id: string;
      call_id: string;
      tool: string;
      args: Record<string, unknown>;
      summary?: string;
      run_id?: string;
      status?: string;
    }
  | { type: "tool_result"; session_id: string; call_id: string; result: Record<string, any> }
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
      run_id?: string;
      approved?: boolean | null;
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
    }
  | { type: "assistant_final"; session_id?: string; run_id?: string; message: string; payload?: Record<string, unknown> }
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
}

export interface ToolTimelineEntry {
  kind: "call" | "approval" | "result";
  callId: string;
  tool: string;
  summary?: string;
  status?: string;
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

export interface SidebarSnapshotState {
  goalSummary: string;
  currentStep: string;
  reasonSummary: string;
  lastTool: string;
  runStatus: string;
  permissionMode: string;
  pendingApprovalCount: number;
  cognitiveSnapshot: CognitiveSnapshotState;
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
  transcriptMode: "full" | "compact";
  promptHistoryByCwd: Record<string, string[]>;
  pendingApprovals: PendingApproval[];
  permissionMode: string;
  assistantStreamActive: boolean;
}
