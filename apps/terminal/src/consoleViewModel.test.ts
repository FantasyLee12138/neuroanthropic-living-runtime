import { describe, expect, it } from "vitest";

import { buildConsoleFlowRows, buildContributionText, buildWhyNotText } from "./consoleViewModel.js";
import { createInitialUiState } from "./state/sessionStore.js";
import type { UiState } from "./types.js";

function makeState(overrides: Partial<UiState> = {}): UiState {
  return {
    ...createInitialUiState(),
    ...overrides,
  };
}

describe("consoleViewModel", () => {
  it("prefers official console timeline rows over activity rail", () => {
    const rows = buildConsoleFlowRows(
      makeState({
        activityRail: [{ kind: "tool", label: "read_file", summary: "读取 app.tsx" }],
        console: {
          ...createInitialUiState().console,
          timeline: {
            roundId: 12,
            traceRef: "round://12",
            events: [
              { type: "action_arbitration", label: "inspect", summary: "clarity_first" },
              { type: "endogenous_tick", label: "endogenous", summary: "internal_trigger" },
            ],
          },
        },
      }),
    );

    expect(rows).toEqual(["行动选择解算：上下文检查 · 清晰度优先", "内源驱动更新：内源驱动 · 内源触发"]);
  });

  it("falls back to activity rail when console timeline is absent", () => {
    const rows = buildConsoleFlowRows(
      makeState({
        activityRail: [{ kind: "tool", label: "read_file", summary: "读取 app.tsx" }],
      }),
    );

    expect(rows).toEqual(["外部动作：读取文件 · 读取 app.tsx"]);
  });

  it("prefers official why-not and contribution text", () => {
    const state = makeState({
      console: {
        ...createInitialUiState().console,
        actionField: {
          roundId: 12,
          traceRef: "round://12",
          topActions: [],
          winner: { action: "inspect", score: 0.91 },
          conflict: {},
          tokenField: { raw: {} },
          contributionStack: [{ source: "planner", weight: 0.7, raw: {} }],
          competingPeaks: [{ action: "abort", score: 0.22 }],
        },
        whyNot: {
          roundId: 12,
          traceRef: "trace://round/12",
          action: "abort",
          whyNot: {
            selected_action: "inspect",
            candidate_score: 0.22,
            blocked_by: ["ConflictMonitorAgent", "planner"],
            stacked_contributions: [
              { module_name: "ConflictMonitorAgent", direction: "block", delta_normalized: -0.64 },
              { module_name: "planner", direction: "support", delta_normalized: 0.42 }
            ]
          },
        },
      },
    });

    expect(buildWhyNotText(state)).toBe("中止 这条候选路径曾进入竞争，后验概率 0.22，但受到 前扣带冲突监测、前额叶规划 的抑制，最终输出 上下文检查");
    expect(buildContributionText(state)).toBe("前扣带冲突监测 · 压制 0.64；前额叶规划 · 促进 0.42");
  });

  it("derives why-not and contribution text from the current action field during normal chat", () => {
    const state = makeState({
      console: {
        ...createInitialUiState().console,
        actionField: {
          roundId: 21,
          traceRef: "round://21",
          topActions: [
            { action: "respond", score: 0.61 },
            { action: "plan", score: 0.33 },
          ],
          winner: { action: "respond", score: 0.61 },
          conflict: {},
          tokenField: { raw: {} },
          contributionStack: [
            { source: "PFCAgent", weight: 0.64, raw: {} },
            { source: "ConflictMonitorAgent", weight: 0.52, raw: {} },
          ],
          competingPeaks: [{ action: "plan", score: 0.33 }],
        },
        whyCurrent: {
          roundId: 21,
          traceRef: "round://21",
          why: {
            summary: "先稳住回应，再决定是否转入规划。",
            sampledAction: "respond",
            topDrivers: [{ agent_name: "PFCAgent" }, { agent_name: "ConflictMonitorAgent" }],
            vitalitySnapshot: {},
            authenticity: {},
          },
        },
        whyNot: null,
      },
    });

    expect(buildWhyNotText(state)).toBe("规划求解 这条候选路径也进入过竞争，后验概率 0.33，但受到 前额叶执行控制、前扣带冲突监测 的抑制，最终输出 回应生成");
    expect(buildContributionText(state)).toBe("前额叶执行控制 · 概率贡献 0.64；前扣带冲突监测 · 概率贡献 0.52");
  });

  it("prefers official why-not summary when it carries monologue-stream evidence", () => {
    const state = makeState({
      console: {
        ...createInitialUiState().console,
        whyNot: {
          roundId: 34,
          traceRef: "round://34",
          action: "monologue",
          whyNot: {
            selected_action: "respond",
            candidate_score: 0.28,
            blocked_by: ["PFCAgent"],
            stacked_contributions: [],
            summary: "这轮独白没有胜出；虽然隐藏独白流仍在提供 2 条碎片，但最终被 respond 压过。",
            expressive_trace: {
              monologue_stream: {
                hidden_by_default: true,
                recent_fragment_count: 2,
              },
            },
          },
        },
      },
    });

    expect(buildWhyNotText(state)).toBe("这轮独白没有胜出；虽然隐藏独白流仍在提供 2 条碎片，但最终被 respond 压过。");
  });
});
