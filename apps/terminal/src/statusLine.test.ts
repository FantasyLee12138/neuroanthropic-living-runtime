import { describe, expect, it } from "vitest";

import { formatStatusLine, tokenizeStatusLine } from "./statusLine.js";
import { createInitialUiState } from "./state/sessionStore.js";
import type { UiState } from "./types.js";

function makeState(overrides: Partial<UiState> = {}): UiState {
  return {
    ...createInitialUiState(),
    activeSessionId: "sess-1",
    activeRunId: "run-1",
    sessionMeta: { session_id: "sess-1", cwd: "/tmp/demo", mode: "plan", permission_mode: "ask" },
    run: { status: "running" },
    permissionMode: "ask",
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
      { label: "脑态", value: "运行中", tone: "accent" },
      { label: "权限", value: "需确认", tone: "muted" },
      { label: "模型", value: "planner:gpt-5.4", tone: "muted" },
      { label: "审批", value: "1", tone: "warning" },
      { label: "视图", value: "紧凑", tone: "muted" }
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

    expect(tokens[2]).toEqual({ label: "模型", value: "small:ep-20260404191810-qfn7s +2", tone: "muted" });
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

    expect(tokens[2]).toEqual({ label: "模型", value: "this-model-name-is-ridic…", tone: "muted" });
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

    expect(line).toBe("脑态:运行中 | 权限:需确认 | 模型:planner:gpt-5.4 | 审批:1 | 视图:紧凑");
  });

  it("prefers official console run status when available", () => {
    const tokens = tokenizeStatusLine(
      makeState({
        run: { status: "paused" },
        console: {
          ...createInitialUiState().console,
          state: {
            brainState: {
              mode: "reflective",
              vitality: null,
              selfContinuity: null,
              authenticityPressure: null,
              longRunDriftRisk: null,
            },
            neuromodulators: {
              dopamine: null,
              noradrenaline: null,
              serotonin: null,
              acetylcholine: null,
              gaba: null,
            },
            motivationPool: {
              activeMotivations: [],
              raw: {},
            },
            longRun: {
              dream: {},
              traceStorage: {},
            },
            currentRound: {
              roundId: 12,
              sampledAction: "inspect",
              traceRef: "round://12",
              causeType: null,
            },
            session: {
              sessionId: "sess-1",
              mode: "plan",
              safeMode: false,
            },
            run: {
              runId: "run-1",
              status: "running",
              raw: {},
            },
            cognitiveSnapshot: {
              coreGoal: "维持生命性、真实性与连续性",
              currentIntent: "正式 console state 已接管状态条",
              vitalSigns: {
                mood: 0,
                bodyEnergy: 0,
                affectResidue: 0,
                focus: "task",
                mode: "interactive",
              },
              identity: {
                displayName: "阿澜",
                continuity: "稳定",
              },
              authenticity: {
                summary: "一致",
                source: "official_console",
                guardAction: "none",
              },
            },
          },
        },
      })
    );

    expect(tokens[0]).toEqual({ label: "脑态", value: "运行中", tone: "accent" });
  });
});
