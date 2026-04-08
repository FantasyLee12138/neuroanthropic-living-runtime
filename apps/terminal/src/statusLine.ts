import type { UiState } from "./types.js";
import { translatePermissionMode, translateRunStatus, translateTranscriptMode } from "./displayLabels.js";

export interface StatusLineToken {
  label: string;
  value: string;
  tone: "muted" | "accent" | "warning";
}

function truncateValue(value: string, maxLength: number): string {
  if (value.length <= maxLength) {
    return value;
  }
  return `${value.slice(0, Math.max(1, maxLength - 1))}…`;
}

function compactModelValue(value: string): string {
  const parts = value.trim().split(/\s+/).filter(Boolean);
  if (parts.length > 1) {
    return `${parts[0] ?? "default"} +${parts.length - 1}`;
  }
  return truncateValue(value, 25);
}

export function tokenizeStatusLine(state: UiState): StatusLineToken[] {
  const statusline = state.statusline;
  const rawRunStatus = String(state.console.state?.run.status ?? statusline?.run_status ?? state.run?.status ?? "idle");
  const runStatus = translateRunStatus(rawRunStatus, rawRunStatus);
  const permissionsValue = translatePermissionMode(String(statusline?.permission_mode ?? state.permissionMode ?? "plan"));
  const model = compactModelValue(String(statusline?.model ?? "default"));
  return [
    { label: "脑态", value: runStatus, tone: ["running", "active"].includes(rawRunStatus) ? "accent" : "muted" },
    { label: "权限", value: permissionsValue, tone: "muted" },
    { label: "模型", value: model, tone: "muted" },
    { label: "审批", value: String(state.pendingApprovals.length), tone: state.pendingApprovals.length > 0 ? "warning" : "muted" },
    { label: "视图", value: translateTranscriptMode(state.transcriptMode), tone: "muted" },
  ];
}

export function formatStatusLine(state: UiState): string {
  return tokenizeStatusLine(state).map((token) => `${token.label}:${token.value}`).join(" | ");
}
