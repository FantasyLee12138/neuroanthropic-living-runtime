import { describe, expect, it } from "vitest";

import { formatDetailSummary, formatSidebarSummary } from "./panelSummary.js";
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
    actionBar: {
      primary: [],
      secondary: [],
      selectedIndex: 0,
    },
    permissionMode: "plan",
    assistantStreamActive: false,
    activityRail: [],
    detailDrawer: null,
    focusZone: "input",
    approvalCursor: 0,
    ...overrides
  };
}

describe("panelSummary", () => {
  it("renders a readable status summary instead of JSON", () => {
    const summary = formatDetailSummary(
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

    expect(formatDetailSummary(state, "why")).toContain("当前目标：理解仓库结构");
    expect(formatDetailSummary(state, "steps")).toContain("剩余步骤：2");
    expect(formatDetailSummary(state, "steps")).toContain("后续：总结模块边界；整理结论");
    expect(formatDetailSummary(state, "tools")).toContain("最近工具：repo_scan: 扫描了 3 个文件；read_file: 读取了 handlers.py");
  });

  it("renders approval details with queue position, risk, and mode", () => {
    const summary = formatDetailSummary(
      makeState({
        approvalCursor: 1,
        pendingApprovals: [
          { callId: "a1", tool: "write_file", summary: "改 app.tsx", riskLevel: "high", mode: "workspace-write", status: "pending" },
          { callId: "a2", tool: "run_shell", actionPreview: "npm test", riskLevel: "medium", mode: "exec", status: "blocked" }
        ]
      }),
      "approvals"
    );

    expect(summary).toContain("待审批：2");
    expect(summary).toContain("> [2/2] run_shell");
    expect(summary).toContain("风险：medium");
    expect(summary).toContain("模式：exec");
    expect(summary).toContain("状态：blocked");
  });

  it("renders cognition and meta drawers with decision-useful details", () => {
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
        modelStatus: {
          tiers: {
            state_machine: { mode: "local", credential_present: true },
            small_model: {
              mode: "remote",
              backend: "doubao",
              model: "ep-20260404191810-qfn7s",
              api_key_env: "ARK_SMALL_MODEL_API_KEY",
              credential_present: true
            },
            medium_model: {
              mode: "remote",
              backend: "deepseek",
              model: "deepseek-chat",
              api_key_env: "DEEPSEEK_API_KEY",
              credential_present: true
            },
            large_model: {
              mode: "remote",
              backend: "doubao",
              model: "doubao-seed-2-0-pro-260215",
              api_key_env: "ARK_API_KEY",
              credential_present: true
            }
          },
          agent_bindings: {
            SalienceAgent: "small_model",
            ValueAgent: "small_model",
            PerspectiveModel: "medium_model",
            PFCAgent: "large_model",
            Renderer: "large_model",
            planner: "medium_model"
          }
        },
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

    const cognition = formatDetailSummary(state, "cognition");
    expect(cognition).toContain("核心目标");
    expect(cognition).toContain("维持生命性、真实性与连续性");
    expect(cognition).toContain("当前意图");
    expect(cognition).toContain("收口终端架构态改造");
    expect(cognition).toContain("心境：较开心 (0.58)");
    expect(cognition).toContain("焦点：正在专心处理眼前的事");
    expect(cognition).toContain("真实性");
    expect(cognition).toContain("还没有足够证据判断这轮真实感");

    const meta = formatDetailSummary(state, "meta");
    expect(meta).toContain("工作区：/tmp/demo");
    expect(meta).toContain("权限模式：ask");
    expect(meta).toContain("模型分层");
    expect(meta).toContain("small_model：doubao / ep-20260404191810-qfn7s");
    expect(meta).toContain("PerspectiveModel -> medium_model");
  });

  it("renders a compact sidebar summary for the default shell", () => {
    const summary = formatSidebarSummary(
      makeState({
        sidebarSnapshot: {
          goalSummary: "完成终端重构",
          currentStep: "改 app.tsx 布局",
          reasonSummary: "主屏要先稳定",
          lastTool: "read_file",
          runStatus: "running",
          permissionMode: "ask",
          pendingApprovalCount: 2,
          modelStatus: undefined,
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
        steps: [{ step_id: "s1", title: "改 app.tsx 布局", status: "running" }],
        tools: [{ call_id: "t1", tool_name: "read_file", summary: "读取 app.tsx" }],
        pendingApprovals: [
          { callId: "a1", tool: "write_file", summary: "准备写 app.tsx", status: "pending" },
          { callId: "a2", tool: "run_shell", summary: "跑测试", status: "pending" }
        ]
      })
    );

    expect(summary).toEqual([
      { label: "目标", value: "完成终端重构", tone: "muted" },
      { label: "步骤", value: "改 app.tsx 布局", tone: "normal" },
      { label: "工具", value: "read_file: 读取 app.tsx", tone: "muted" },
      { label: "审批", value: "2 项待处理", tone: "warning" },
      { label: "认知", value: "收口终端架构态改造", tone: "accent" }
    ]);
  });
});
