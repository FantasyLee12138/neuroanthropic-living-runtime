import type { PaletteSectionId } from "./commandPalette.js";

export function actionTitle(): string {
  return "操作";
}

export function actionGroupLabel(group: "inspect" | "control"): string {
  return group === "inspect" ? "查看" : "执行";
}

export function paletteTitle(): string {
  return "命令";
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
  const base = `审批 ${input.cursor + 1}/${input.total} · ${input.tool}`;
  return input.summary ? `${base} · ${input.summary}` : base;
}
