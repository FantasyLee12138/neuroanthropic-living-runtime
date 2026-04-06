import type { UiState } from "./types.js";

export function buildHelpHint(state: UiState, options: { paletteOpen: boolean }): string {
  if (options.paletteOpen) {
    return "命令：输入筛选，↑↓移动，Enter确认，Esc关闭，空查询可用 1-9";
  }

  if (state.focusZone === "actions") {
    return "操作：←→移动，Enter执行，1-9直选，Tab切换焦点";
  }

  if (state.focusZone === "approval" && state.pendingApprovals.length > 0) {
    return "审批：←→切换选项，Enter确认，1-9直选，[ ] 切队列，y/n 快捷批准";
  }

  if (state.focusZone === "drawer") {
    return "详情：Ctrl+J/K切换抽屉，Tab切焦点，s/w/t/a/i/m 快速打开";
  }

  if (state.focusZone === "transcript") {
    return "转录：PageUp/PageDown滚动，Tab切焦点，Ctrl+P打开命令";
  }

  return "输入：直接提问，Ctrl+P命令，Tab切焦点，PageUp/PageDown滚动，/help查看更多";
}
