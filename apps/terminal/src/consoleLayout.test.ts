import { describe, expect, it } from "vitest";

import { buildConsoleViewportPlan, resolveConsoleLayout } from "./consoleLayout.js";

describe("resolveConsoleLayout", () => {
  it("uses split layout for wide terminals", () => {
    expect(resolveConsoleLayout(140)).toBe("split");
  });

  it("falls back to stacked layout for narrow terminals", () => {
    expect(resolveConsoleLayout(90)).toBe("stack");
  });

  it("caps transcript rows and uses compact sidebar on short terminals", () => {
    expect(buildConsoleViewportPlan({ width: 140, height: 20, transcriptMode: "full", hasPanel: true, hasPendingApproval: true })).toEqual({
      layout: "split",
      transcriptLines: 5,
      compactSidebar: true,
      sidebarWidth: 42
    });
  });

  it("allows a roomier transcript on taller terminals", () => {
    expect(buildConsoleViewportPlan({ width: 140, height: 36, transcriptMode: "compact", hasPanel: false, hasPendingApproval: false })).toEqual({
      layout: "split",
      transcriptLines: 12,
      compactSidebar: false,
      sidebarWidth: 42
    });
  });
});
