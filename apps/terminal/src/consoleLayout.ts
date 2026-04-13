export type ConsoleLayoutMode = "split" | "stack";
export type DetailPlacement = "side" | "bottom";

export interface ConsoleViewportPlan {
  layout: ConsoleLayoutMode;
  detailPlacement: DetailPlacement;
  transcriptLines: number;
  sidebarLines: number;
  detailLines: number;
  compactSidebar: boolean;
  sidebarWidth: number;
  detailWidth: number;
}

export function resolveConsoleLayout(width: number): ConsoleLayoutMode {
  return width >= 110 ? "split" : "stack";
}

export function buildConsoleViewportPlan(input: {
  width: number;
  height: number;
  hasDetailDrawer: boolean;
  hasPendingApproval: boolean;
}): ConsoleViewportPlan {
  const layout = resolveConsoleLayout(input.width);
  const compactSidebar = input.height <= 22;
  if (layout === "split") {
    const sharedLines = Math.max(8, input.height - 9);
    return {
      layout,
      detailPlacement: "side",
      transcriptLines: sharedLines,
      sidebarLines: sharedLines,
      detailLines: input.hasDetailDrawer ? sharedLines : 0,
      compactSidebar,
      sidebarWidth: 28,
      detailWidth: 34,
    };
  }

  const baseRows = Math.max(14, input.height - 7);
  const sidebarLines = compactSidebar ? 4 : 5;
  const detailLines = input.hasDetailDrawer ? 6 : 0;
  const transcriptLines = Math.max(8, baseRows - sidebarLines - detailLines);

  return {
    layout,
    detailPlacement: "bottom",
    transcriptLines,
    sidebarLines,
    detailLines,
    compactSidebar,
    sidebarWidth: input.width,
    detailWidth: input.width,
  };
}
