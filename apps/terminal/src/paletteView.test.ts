import { describe, expect, it } from "vitest";

import { compactPaletteSections } from "./paletteView.js";
import type { PaletteSection } from "./commandPalette.js";

function entry(id: string, label = id, disabled = false) {
  return {
    id,
    label,
    group: id.startsWith("local-") ? "本地" : id.startsWith("control-") ? "控制" : "查看",
    disabled,
    run: () => undefined,
    haystack: `${id} ${label}`.toLowerCase(),
  } as const;
}

describe("paletteView", () => {
  const sections: PaletteSection[] = [
    {
      id: "recommended",
      label: "Recommended",
      entries: [
        entry("status"),
        entry("why", "原因", true),
        entry("steps", "步骤", true),
        entry("tools", "工具", true),
        entry("control-cognition", "认知"),
        entry("control-meta", "元信息"),
        entry("control-abort", "中止", true),
        entry("local-help", "帮助"),
        entry("local-clear", "清空"),
      ],
    },
    {
      id: "recent",
      label: "Recent",
      entries: [entry("control-meta", "元信息"), entry("local-help", "帮助"), entry("status")],
    },
    {
      id: "all",
      label: "All",
      entries: [
        entry("status"),
        entry("why", "原因", true),
        entry("steps", "步骤", true),
        entry("tools", "工具", true),
        entry("control-cognition", "认知"),
        entry("control-meta", "元信息"),
        entry("control-abort", "中止", true),
        entry("local-help", "帮助"),
        entry("local-clear", "清空"),
        entry("local-compact", "紧凑视图"),
        entry("local-exit", "退出"),
      ],
    },
  ];

  it("prioritizes enabled recommended entries and removes duplicates from later sections when query is empty", () => {
    const compacted = compactPaletteSections(sections, "");

    expect(compacted[0]?.id).toBe("recommended");
    expect(compacted[0]?.entries.map((item) => item.id)).toEqual([
      "status",
      "control-cognition",
      "control-meta",
      "local-help",
      "local-clear",
      "why",
    ]);
    expect(compacted[1]?.entries.map((item) => item.id)).toEqual([]);
    expect(compacted[2]?.entries.map((item) => item.id)).toEqual([
      "steps",
      "tools",
      "control-abort",
      "local-compact",
      "local-exit",
    ]);
  });

  it("leaves filtered sections unchanged when searching", () => {
    const compacted = compactPaletteSections(sections, "meta");

    expect(compacted).toEqual(sections);
  });
});
