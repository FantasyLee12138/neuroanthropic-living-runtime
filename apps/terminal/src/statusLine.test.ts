import { describe, expect, it } from "vitest";

import { formatStatusLine, tokenizeStatusLine } from "./statusLine.js";
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
    actionBar: {
      primary: [],
      secondary: [],
      selectedIndex: 0,
    },
    permissionMode: "ask",
    assistantStreamActive: false,
    activityRail: [],
    detailDrawer: null,
    focusZone: "input",
    approvalCursor: 0,
    ...overrides
  };
}

describe("formatStatusLine", () => {
  it("returns low-noise semantic tokens for chip rendering", () => {
    const tokens = tokenizeStatusLine(
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

    expect(tokens).toEqual([
      { label: "run", value: "running", tone: "accent" },
      { label: "perms", value: "ask", tone: "muted" },
      { label: "model", value: "planner:gpt-5.4", tone: "muted" },
      { label: "approvals", value: "1", tone: "warning" },
      { label: "view", value: "compact", tone: "muted" }
    ]);
  });

  it("folds multi-tier model strings into a compact status token", () => {
    const tokens = tokenizeStatusLine(
      makeState({
        statusline: {
          cwd: "/tmp/demo",
          git: "dirty",
          permission_mode: "plan",
          model: "small:ep-20260404191810-qfn7s medium:deepseek-chat large:doubao-seed-2-0-pro-260215",
          run_status: "active",
          session_id: "sess-1",
          run_id: "run-1"
        }
      })
    );

    expect(tokens[2]).toEqual({ label: "model", value: "small:ep-20260404191810-qfn7s +2", tone: "muted" });
  });

  it("truncates a single oversized model token", () => {
    const tokens = tokenizeStatusLine(
      makeState({
        statusline: {
          cwd: "/tmp/demo",
          git: "dirty",
          permission_mode: "plan",
          model: "this-model-name-is-ridiculously-long-for-a-chip",
          run_status: "active",
          session_id: "sess-1",
          run_id: "run-1"
        }
      })
    );

    expect(tokens[2]).toEqual({ label: "model", value: "this-model-name-is-ridic…", tone: "muted" });
  });

  it("renders the compact primary tokens only", () => {
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

    expect(line).toBe("run:running | perms:ask | model:planner:gpt-5.4 | approvals:1 | view:compact");
  });
});
