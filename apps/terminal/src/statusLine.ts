import type { UiState } from "./types.js";

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
  const runStatus = String(statusline?.run_status ?? state.run?.status ?? "idle");
  const permissionsValue = String(statusline?.permission_mode ?? state.permissionMode ?? "plan");
  const model = compactModelValue(String(statusline?.model ?? "default"));
  return [
    { label: "run", value: runStatus, tone: ["running", "active"].includes(runStatus) ? "accent" : "muted" },
    { label: "perms", value: permissionsValue, tone: "muted" },
    { label: "model", value: model, tone: "muted" },
    { label: "approvals", value: String(state.pendingApprovals.length), tone: state.pendingApprovals.length > 0 ? "warning" : "muted" },
    { label: "view", value: state.transcriptMode, tone: "muted" },
  ];
}

export function formatStatusLine(state: UiState): string {
  return tokenizeStatusLine(state).map((token) => `${token.label}:${token.value}`).join(" | ");
}
