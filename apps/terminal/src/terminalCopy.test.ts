import { describe, expect, it } from "vitest";

import {
  actionGroupLabel,
  actionTitle,
  approvalHeadline,
  consoleSectionTitle,
  consoleShellSubtitle,
  consoleShellTitle,
  contributionSectionTitle,
  emptyDrawerText,
  emptyMindsetText,
  flowSectionTitle,
  paletteSectionLabel,
  paletteTitle,
  whyCurrentLabel,
  whyNotSectionTitle,
} from "./terminalCopy.js";

describe("terminalCopy", () => {
  it("uses Chinese-first titles for the Alive Console shell", () => {
    expect(consoleShellTitle()).toBe("NALR 活体控制台");
    expect(consoleShellSubtitle()).toBe("脑态持续波动，思绪不断流动，解释随时可见。");
    expect(flowSectionTitle()).toBe("认知时序");
    expect(consoleSectionTitle("mindset")).toBe("脑态");
    expect(consoleSectionTitle("transcript")).toBe("思绪流");
    expect(consoleSectionTitle("explanation")).toBe("解释层");
    expect(consoleSectionTitle("input")).toBe("输入坞");
    expect(actionTitle()).toBe("控制动作");
    expect(paletteTitle()).toBe("命令面板");
    expect(actionGroupLabel("inspect")).toBe("查看");
    expect(actionGroupLabel("control")).toBe("执行");
    expect(paletteSectionLabel("recommended")).toBe("推荐");
    expect(paletteSectionLabel("recent")).toBe("最近");
    expect(paletteSectionLabel("all")).toBe("全部");
    expect(emptyMindsetText()).toBe("尚未接入");
    expect(emptyDrawerText()).toBe("等待本轮解释");
    expect(whyNotSectionTitle()).toBe("未采纳路径");
    expect(whyCurrentLabel()).toBe("当前选择依据");
    expect(contributionSectionTitle()).toBe("贡献叠层");
  });

  it("builds a compact approval headline with queue position", () => {
    expect(approvalHeadline({ cursor: 0, total: 1, tool: "write_file" })).toBe("审批 1/1 · 写入文件");
    expect(approvalHeadline({ cursor: 1, total: 3, tool: "run_shell", summary: "npm test" })).toBe("审批 2/3 · 执行命令 · npm test");
  });
});
