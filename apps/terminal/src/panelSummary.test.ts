import { describe, expect, it } from "vitest";

import { formatPanelBody } from "./panelSummary.js";
import type { UiState } from "./types.js";

function makeState(overrides: Partial<UiState> = {}): UiState {
  return {
    activeSessionId: "sess-1",
    activeRunId: "run-1",
    sessionMeta: { session_id: "sess-1", cwd: "/tmp/demo", mode: "plan", permission_mode: "plan" },
    run: null,
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
    permissionMode: "plan",
    assistantStreamActive: false,
    ...overrides
  };
}

describe("formatPanelBody", () => {
  it("renders a readable status summary instead of JSON", () => {
    const summary = formatPanelBody(
      makeState({
        run: {
          status: "paused",
          current_step: { title: "检查 planner.py" },
          dirty_worktree_detected: true
        }
      }),
      "status"
    );

    expect(summary).toContain("状态：paused");
    expect(summary).toContain("当前步骤：检查 planner.py");
    expect(summary).toContain("工作区：dirty");
    expect(summary).not.toContain("{");
  });

  it("renders why, steps, and tools as compact summaries", () => {
    const state = makeState({
      run: {
        current_step: { title: "检查 planner.py" },
        pending_steps: 4
      },
      lastWhy: {
        goal_summary: "理解仓库结构",
        current_step: { title: "检查 planner.py" },
        stop_reason: { message: "paused by operator" }
      },
      steps: [
        { step_id: "step-1", title: "检查 planner.py", status: "running" },
        { step_id: "step-2", title: "总结模块边界", status: "pending" },
        { step_id: "step-3", title: "整理结论", status: "pending" }
      ],
      tools: [
        { call_id: "run-1:tool:0", tool_name: "repo_scan", summary: "扫描了 3 个文件" },
        { call_id: "run-1:tool:1", tool_name: "read_file", summary: "读取了 handlers.py" }
      ]
    });

    expect(formatPanelBody(state, "why")).toContain("当前目标：理解仓库结构");
    expect(formatPanelBody(state, "steps")).toContain("剩余步骤：2");
    expect(formatPanelBody(state, "steps")).toContain("后续：总结模块边界；整理结论");
    expect(formatPanelBody(state, "tools")).toContain("最近工具：repo_scan: 扫描了 3 个文件；read_file: 读取了 handlers.py");
  });

  it("renders unified state snapshot for sidebar", () => {
    const state = makeState({
      run: { status: "running", current_step: { title: "写 slash 测试" } },
      lastWhy: { goal_summary: "实现终端切片" },
      steps: [{ step_id: "s1", title: "写 slash 测试", status: "running" }],
      sidebarSnapshot: {
        goalSummary: "维持生命性、真实性与连续性",
        currentStep: "写 slash 测试",
        reasonSummary: "当前进入测试优先的收口阶段",
        lastTool: "read_file",
        runStatus: "running",
        permissionMode: "ask",
        pendingApprovalCount: 1,
        cognitiveSnapshot: {
          coreGoal: "维持生命性、真实性与连续性",
          currentIntent: "收口终端架构态改造",
          vitalSigns: {
            mood: 0.58,
            bodyEnergy: 0.71,
            affectResidue: 0.12,
            focus: "task",
            mode: "interactive"
          },
          identity: {
            displayName: "阿澜",
            continuity: "名称与身份连续性稳定"
          },
          authenticity: {
            summary: "还没有足够证据判断这轮真实感",
            source: "none",
            guardAction: "none"
          }
        }
      },
      sessionMeta: { session_id: "sess-1", cwd: "/tmp/demo", mode: "plan", permission_mode: "ask" },
      pendingApprovals: [{ callId: "run-1:tool:0", tool: "write_file", riskLevel: "high" }]
    });

    const summary = formatPanelBody(state, "state");
    expect(summary).toContain("核心目标");
    expect(summary).toContain("维持生命性、真实性与连续性");
    expect(summary).toContain("当前意图");
    expect(summary).toContain("收口终端架构态改造");
    expect(summary).toContain("心境：较开心 (0.58)");
    expect(summary).toContain("能量：状态不错 (0.71)");
    expect(summary).toContain("情感余波：轻微波动 (0.12)");
    expect(summary).toContain("焦点：正在专心处理眼前的事");
    expect(summary).toContain("模式：正常交流中");
    expect(summary).toContain("身份：阿澜");
    expect(summary).toContain("连续性：名称与身份连续性稳定");
    expect(summary).toContain("真实性");
    expect(summary).toContain("还没有足够证据判断这轮真实感");
    expect(summary).not.toContain("权限模式：ask");
  });
});
