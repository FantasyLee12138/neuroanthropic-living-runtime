import type { UiState } from "./types.js";

export function formatStatusLine(state: UiState): string {
  const statusline = state.statusline;
  const runStatus = String(statusline?.run_status ?? state.run?.status ?? "idle");
  const cwd = String(statusline?.cwd ?? state.sessionMeta?.cwd ?? "-");
  const git = String(statusline?.git ?? (state.run?.dirty_worktree_detected ? "dirty" : "clean"));
  const permissionsValue = String(statusline?.permission_mode ?? state.permissionMode ?? "plan");
  const model = String(statusline?.model ?? "default");
  const sessionId = String(statusline?.session_id ?? state.activeSessionId ?? "-");
  const runId = String(statusline?.run_id ?? state.activeRunId ?? "-");
  return [
    `cwd:${cwd}`,
    `git:${git}`,
    `run:${runStatus}`,
    `permissions:${permissionsValue}`,
    `model:${model}`,
    `session:${sessionId}`,
    `runId:${runId}`,
    `view:${state.transcriptMode}`,
    `approval:${state.pendingApprovals.length}`
  ].join(" | ");
}
