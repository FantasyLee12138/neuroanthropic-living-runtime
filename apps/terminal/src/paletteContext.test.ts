import { describe, expect, it } from "vitest";

import { derivePaletteContext } from "./paletteContext.js";
import { createInitialUiState } from "./state/sessionStore.js";
import type { UiState } from "./types.js";

function makeState(overrides: Partial<UiState> = {}): UiState {
  return {
    ...createInitialUiState(),
    ...overrides,
  };
}

describe("derivePaletteContext", () => {
  it("prefers approval mode when pending approvals exist", () => {
    expect(
      derivePaletteContext(
        makeState({
          pendingApprovals: [{ callId: "a1", tool: "write_file" }],
        }),
      ),
    ).toBe("approval");
  });

  it("uses official console run status when available", () => {
    expect(
      derivePaletteContext(
        makeState({
          run: { status: "completed" },
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
                currentIntent: "正式 console state 已接管命令面板上下文",
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
        }),
      ),
    ).toBe("running");
  });

  it("falls back to idle when there is no active run", () => {
    expect(derivePaletteContext(makeState())).toBe("idle");
  });
});
