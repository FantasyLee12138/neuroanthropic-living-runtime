import type { PaletteSectionId } from "./commandPalette.js";
import { translateToolName } from "./displayLabels.js";

export type ConsoleSectionKey = "mindset" | "transcript" | "explanation" | "input";

export function consoleShellTitle(): string {
  return "NALR 活体控制台";
}

export function consoleShellSubtitle(): string {
  return "脑态持续波动，思绪不断流动，解释随时可见。";
}

export function flowSectionTitle(): string {
  return "认知时序";
}

export function consoleSectionTitle(section: ConsoleSectionKey): string {
  switch (section) {
    case "mindset":
      return "脑态";
    case "transcript":
      return "思绪流";
    case "explanation":
      return "解释层";
    case "input":
      return "输入坞";
  }
}

export function emptyMindsetText(): string {
  return "尚未接入";
}

export function emptyDrawerText(): string {
  return "等待本轮解释";
}

export function whyNotSectionTitle(): string {
  return "未采纳路径";
}

export function whyCurrentLabel(): string {
  return "当前选择依据";
}

export function contributionSectionTitle(): string {
  return "贡献叠层";
}

export function actionTitle(): string {
  return "控制动作";
}

export function actionGroupLabel(group: "inspect" | "control"): string {
  return group === "inspect" ? "查看" : "执行";
}

export function paletteTitle(): string {
  return "命令面板";
}

export function paletteSectionLabel(sectionId: PaletteSectionId): string {
  switch (sectionId) {
    case "recommended":
      return "推荐";
    case "recent":
      return "最近";
    case "all":
      return "全部";
  }
}

export function approvalHeadline(input: {
  cursor: number;
  total: number;
  tool: string;
  summary?: string;
}): string {
  const base = `审批 ${input.cursor + 1}/${input.total} · ${translateToolName(input.tool, input.tool)}`;
  return input.summary ? `${base} · ${input.summary}` : base;
}
