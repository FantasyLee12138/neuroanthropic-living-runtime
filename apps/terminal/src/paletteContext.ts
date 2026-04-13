import type { PaletteContext } from "./commandPalette.js";
import type { UiState } from "./types.js";

export function derivePaletteContext(state: UiState): PaletteContext {
  if (state.pendingApprovals.length > 0) {
    return "approval";
  }

  const status = String(state.console.state?.run.status ?? state.sidebarSnapshot?.runStatus ?? state.run?.status ?? "").toLowerCase();
  if (state.assistantStreamActive) {
    return "running";
  }
  if (!status) {
    return state.activeRunId ? "running" : "idle";
  }
  if (["completed", "done", "aborted", "failed", "error", "idle"].includes(status)) {
    return "idle";
  }
  return "running";
}
