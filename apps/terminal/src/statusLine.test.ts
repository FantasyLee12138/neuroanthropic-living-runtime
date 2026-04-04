import { describe, expect, it } from "vitest";

import { formatStatusLine } from "./statusLine.js";
import type { UiState } from "./types.js";

function makeState(overrides: Partial<UiState> = {}): UiState {
  return {
    activeSessionId: "sess-1",
    activeRunId: "run-1",
    sessionMeta: { session_id: "sess-1", cwd: "/tmp/demo", mode: "plan", permission_mode: "ask" },
    run: { status: "running" },
    steps: [],
    tools: [],
    toolTimeline: [],
    lastWhy: null,
    sidebarSnapshot: null,
    statusline: null,
    lines: [],
    transcriptMode: "full",
    promptHistoryByCwd: {},
    pendingApprovals: [],
    permissionMode: "ask",
    assistantStreamActive: false,
    ...overrides
  };
}

describe("formatStatusLine", () => {
  it("renders compact state, mode, permissions, and pending approval", () => {
    const line = formatStatusLine(
      makeState({
        transcriptMode: "compact",
        pendingApprovals: [{ callId: "run-1:tool:0", tool: "write_file", riskLevel: "high", actionPreview: "write src/app.tsx" }],
        statusline: {
          cwd: "/tmp/demo",
          git: "dirty",
          permission_mode: "ask",
          model: "planner:gpt-5.4",
          run_status: "running",
          session_id: "sess-1",
          run_id: "run-1"
        }
      })
    );

    expect(line).toContain("cwd:/tmp/demo");
    expect(line).toContain("git:dirty");
    expect(line).toContain("run:running");
    expect(line).toContain("permissions:ask");
    expect(line).toContain("model:planner:gpt-5.4");
    expect(line).toContain("session:sess-1");
    expect(line).toContain("runId:run-1");
    expect(line).toContain("view:compact");
    expect(line).toContain("approval:1");
  });
});
