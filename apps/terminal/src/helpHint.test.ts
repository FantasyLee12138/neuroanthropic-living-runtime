import { describe, expect, it } from "vitest";

import { buildHelpHint } from "./helpHint.js";
import { createInitialUiState } from "./state/sessionStore.js";
import type { UiState } from "./types.js";

function makeState(overrides: Partial<UiState> = {}): UiState {
  return {
    ...createInitialUiState(),
    activeSessionId: "sess-1",
    activeRunId: "run-1",
    ...overrides,
  };
}

describe("buildHelpHint", () => {
  it("shows palette-focused help when palette is open", () => {
    expect(buildHelpHint(makeState(), { paletteOpen: true })).toBe("命令：输入筛选，↑↓移动，Enter确认，Esc关闭，空查询可用 1-9");
  });

  it("shows action selection help when actions are focused", () => {
    expect(buildHelpHint(makeState({ focusZone: "actions" }), { paletteOpen: false })).toBe("操作：←→移动，Enter执行，1-9直选，Tab切换焦点");
  });

  it("shows approval help when approvals are focused", () => {
    expect(
      buildHelpHint(
        makeState({
          focusZone: "approval",
          pendingApprovals: [{ callId: "a1", tool: "write_file", summary: "改 app.tsx" }],
        }),
        { paletteOpen: false },
      ),
    ).toBe("审批：←→切换选项，Enter确认，1-9直选，[ ] 切队列，y/n 快捷批准");
  });

  it("falls back to compact input help for the default state", () => {
    expect(buildHelpHint(makeState(), { paletteOpen: false })).toBe("输入：直接提问，Ctrl+P命令，Tab切焦点，PageUp/PageDown滚动，/help查看更多");
  });
});
