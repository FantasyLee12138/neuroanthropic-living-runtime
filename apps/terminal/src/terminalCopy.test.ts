import { describe, expect, it } from "vitest";

import { actionGroupLabel, actionTitle, approvalHeadline, paletteSectionLabel, paletteTitle } from "./terminalCopy.js";

describe("terminalCopy", () => {
  it("uses compact Chinese titles for interactive surfaces", () => {
    expect(actionTitle()).toBe("操作");
    expect(paletteTitle()).toBe("命令");
    expect(actionGroupLabel("inspect")).toBe("查看");
    expect(actionGroupLabel("control")).toBe("执行");
    expect(paletteSectionLabel("recommended")).toBe("推荐");
    expect(paletteSectionLabel("recent")).toBe("最近");
    expect(paletteSectionLabel("all")).toBe("全部");
  });

  it("builds a compact approval headline with queue position", () => {
    expect(approvalHeadline({ cursor: 0, total: 1, tool: "write_file" })).toBe("审批 1/1 · write_file");
    expect(approvalHeadline({ cursor: 1, total: 3, tool: "run_shell", summary: "npm test" })).toBe("审批 2/3 · run_shell · npm test");
  });
});
