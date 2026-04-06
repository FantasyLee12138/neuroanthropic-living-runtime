import { describe, expect, it } from "vitest";

import { buildConsoleViewportPlan, resolveConsoleLayout } from "./consoleLayout.js";

describe("resolveConsoleLayout", () => {
  it("uses split layout for wide terminals", () => {
    expect(resolveConsoleLayout(140)).toBe("split");
  });

  it("falls back to stacked layout for narrow terminals", () => {
    expect(resolveConsoleLayout(90)).toBe("stack");
  });

  it("keeps the transcript dominant on short wide terminals", () => {
    expect(buildConsoleViewportPlan({ width: 140, height: 20, hasDetailDrawer: true, hasPendingApproval: true })).toEqual({
      layout: "split",
      detailPlacement: "side",
      transcriptLines: 12,
      sidebarLines: 12,
      detailLines: 12,
      compactSidebar: true,
      sidebarWidth: 36
    });
  });

  it("gives tall terminals a much roomier transcript without a hard cap", () => {
    expect(buildConsoleViewportPlan({ width: 140, height: 36, hasDetailDrawer: false, hasPendingApproval: false })).toEqual({
      layout: "split",
      detailPlacement: "side",
      transcriptLines: 28,
      sidebarLines: 28,
      detailLines: 0,
      compactSidebar: false,
      sidebarWidth: 36
    });
  });

  it("moves detail content below the transcript on narrow terminals", () => {
    expect(buildConsoleViewportPlan({ width: 90, height: 28, hasDetailDrawer: true, hasPendingApproval: false })).toEqual({
      layout: "stack",
      detailPlacement: "bottom",
      transcriptLines: 9,
      sidebarLines: 5,
      detailLines: 6,
      compactSidebar: false,
      sidebarWidth: 90
    });
  });
});
