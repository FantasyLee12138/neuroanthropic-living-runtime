import { describe, expect, it } from "vitest";

import { buildPaletteEntries, buildPaletteSections, filterPaletteEntries, filterPaletteSections } from "./commandPalette.js";
import type { UiAction } from "./types.js";

describe("commandPalette", () => {
  const primary: UiAction[] = [
    { id: "status", label: "状态", kind: "drawer", value: "status", disabled: false },
    { id: "cognition", label: "认知", kind: "drawer", value: "cognition", disabled: false },
    { id: "meta", label: "元信息", kind: "drawer", value: "meta", disabled: false },
    { id: "pause", label: "暂停", kind: "command", value: "pause", disabled: false },
  ];
  const secondary: UiAction[] = [
    { id: "approvals", label: "审批", kind: "drawer", value: "approvals", disabled: false },
    { id: "abort", label: "中止", kind: "command", value: "abort", disabled: false },
  ];

  it("builds grouped palette entries from ui actions and local commands", () => {
    const entries = buildPaletteEntries(primary, secondary);

    expect(entries.map((item) => item.id)).toEqual([
      "status",
      "cognition",
      "meta",
      "pause",
      "approvals",
      "abort",
      "help",
      "clear",
      "compact",
      "exit",
    ]);
    expect(entries[0]?.group).toBe("查看");
    expect(entries[5]?.group).toBe("控制");
    expect(entries[6]?.group).toBe("本地");
  });

  it("builds recommended, recent, and all sections from context and recency", () => {
    const sections = buildPaletteSections(primary, secondary, {
      context: "idle",
      recentIds: ["abort", "meta", "help", "abort", "missing"],
    });

    expect(sections.map((section) => section.id)).toEqual(["recommended", "recent", "all"]);
    expect(sections[0]?.entries.map((entry) => entry.id).slice(0, 4)).toEqual(["cognition", "meta", "help", "compact"]);
    expect(sections[1]?.entries.map((entry) => entry.id)).toEqual(["abort", "meta", "help"]);
    expect(sections[2]?.entries.map((entry) => entry.id)).toEqual([
      "status",
      "cognition",
      "meta",
      "pause",
      "approvals",
      "abort",
      "help",
      "clear",
      "compact",
      "exit",
    ]);
  });

  it("raises approval and running priorities into the recommended section", () => {
    const approvalSections = buildPaletteSections(primary, secondary, { context: "approval" });
    const runningSections = buildPaletteSections(primary, secondary, { context: "running" });

    expect(approvalSections[0]?.entries[0]?.id).toBe("approvals");
    expect(runningSections[0]?.entries.slice(0, 4).map((entry) => entry.id)).toEqual(["status", "cognition", "meta", "approvals"]);
  });

  it("filters palette entries by label, command value, and aliases", () => {
    const entries = buildPaletteEntries(primary, secondary);

    expect(filterPaletteEntries(entries, "meta").map((item) => item.id)).toEqual(["meta"]);
    expect(filterPaletteEntries(entries, "中止").map((item) => item.id)).toEqual(["abort"]);
    expect(filterPaletteEntries(entries, "help").map((item) => item.id)).toEqual(["help"]);
    expect(filterPaletteEntries(entries, "why").map((item) => item.id)).toEqual([]);
  });

  it("filters sectioned palette data without changing item order", () => {
    const sections = buildPaletteSections(primary, secondary, { context: "idle", recentIds: ["meta", "abort"] });
    const filtered = filterPaletteSections(sections, "a");

    expect(filtered.map((section) => section.id)).toEqual(["recommended", "recent", "all"]);
    expect(filtered[0]?.entries.map((entry) => entry.id)).toEqual(["meta", "compact", "status", "approvals", "pause", "abort", "clear"]);
    expect(filtered[1]?.entries.map((entry) => entry.id)).toEqual(["meta", "abort"]);
    expect(filtered[2]?.entries.map((entry) => entry.id)).toEqual(["status", "meta", "pause", "approvals", "abort", "clear", "compact"]);
  });
});
