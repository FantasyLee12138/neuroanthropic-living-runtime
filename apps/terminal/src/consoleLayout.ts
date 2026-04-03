export type ConsoleLayoutMode = "split" | "stack";

export interface ConsoleViewportPlan {
  layout: ConsoleLayoutMode;
  transcriptLines: number;
  compactSidebar: boolean;
  sidebarWidth: number;
}

export function resolveConsoleLayout(width: number): ConsoleLayoutMode {
  return width >= 110 ? "split" : "stack";
}

export function buildConsoleViewportPlan(input: {
  width: number;
  height: number;
  transcriptMode: "full" | "compact";
  hasPanel: boolean;
  hasPendingApproval: boolean;
}): ConsoleViewportPlan {
  const layout = resolveConsoleLayout(input.width);
  const sidebarWidth = layout === "split" ? Math.max(38, Math.min(42, Math.floor(input.width * 0.32))) : input.width;
  const reservedRows =
    (layout === "split" ? 12 : 14) +
    (input.hasPanel ? 4 : 0) +
    (input.hasPendingApproval ? 4 : 0);
  const rawTranscriptLines = Math.max(5, input.height - reservedRows);
  const transcriptLines = Math.min(rawTranscriptLines, input.transcriptMode === "compact" ? 12 : 10);
  return {
    layout,
    transcriptLines,
    compactSidebar: input.height <= 24,
    sidebarWidth
  };
}
