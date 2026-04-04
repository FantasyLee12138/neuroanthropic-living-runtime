import { describe, expect, it } from "vitest";

import { applyBridgeEvent, createInitialUiState } from "./sessionStore.js";

describe("sessionStore", () => {
  it("tracks active session and active run from bridge events", () => {
    const initial = createInitialUiState();
    const withSession = applyBridgeEvent(initial, {
      type: "session_started",
      session: { session_id: "sess-1", cwd: "/tmp/demo", status: "active" }
    });
    const withRun = applyBridgeEvent(withSession, {
      type: "run_status",
      session_id: "sess-1",
      run: { run_id: "run-1", status: "running", goal: "检查 src" }
    });

    expect(withRun.activeSessionId).toBe("sess-1");
    expect(withRun.activeRunId).toBe("run-1");
    expect(withRun.run?.status).toBe("running");
    expect(withRun.lines).toEqual([]);
    expect(withRun.pendingApprovals).toEqual([]);
    expect(withRun.toolTimeline).toEqual([]);
  });

  it("accumulates steps and tool results for side panels", () => {
    const initial = createInitialUiState();
    const withStep = applyBridgeEvent(initial, {
      type: "step_update",
      session_id: "sess-1",
      step: { step_id: "step-1", title: "检查 planner.py" }
    });
    const withTool = applyBridgeEvent(withStep, {
      type: "tool_result",
      session_id: "sess-1",
      call_id: "run-1:tool:0",
      result: { tool_name: "repo_scan", summary: "scanned 3 files" }
    });

    expect(withTool.steps).toHaveLength(1);
    expect(withTool.tools).toHaveLength(1);
    expect(withTool.tools[0]?.tool_name).toBe("repo_scan");
    expect(withTool.lines).toEqual([
      { kind: "system", text: "Step: 检查 planner.py" },
      { kind: "system", text: "Result: repo_scan" }
    ]);
    expect(withTool.toolTimeline).toEqual([
      { kind: "result", callId: "run-1:tool:0", tool: "repo_scan", summary: "scanned 3 files", status: undefined }
    ]);
  });

  it("hydrates sidebar state from sidebar_snapshot events", () => {
    const initial = createInitialUiState();
    const next = applyBridgeEvent(initial, {
      type: "sidebar_snapshot",
      session_id: "sess-1",
      goal_summary: "完成终端界面切片",
      current_step: "写测试",
      reason_summary: "先钉住 reducer 行为",
      last_tool: "read_file",
      run_status: "running",
      permission_mode: "ask",
      pending_approval_count: 1,
      status: { run_id: "run-42", status: "running", current_step: { title: "写测试" } },
      why: { goal_summary: "完成终端界面切片" },
      steps: [{ step_id: "s1", title: "写测试", status: "running" }],
      tools: [{ call_id: "c1", tool_name: "read_file", summary: "读取 app.tsx" }],
      session: { session_id: "sess-1", cwd: "/tmp/demo", mode: "plan", permission_mode: "ask" },
      cognitive_snapshot: {
        core_goal: "维持生命性、真实性与连续性",
        current_intent: "收口终端架构态改造",
        vital_signs: {
          mood: 0.61,
          body_energy: 0.72,
          affect_residue: 0.08,
          focus: "task",
          mode: "interactive"
        },
        identity: {
          display_name: "阿澜",
          continuity: "名称与身份连续性稳定"
        },
        authenticity: {
          summary: "暂无轮次证据",
          source: "none",
          guard_action: "none"
        }
      },
      statusline: {
        cwd: "/tmp/demo",
        git: "clean",
        permission_mode: "ask",
        model: "planner:gpt",
        run_status: "running",
        session_id: "sess-1",
        run_id: "run-42"
      }
    });

    expect(next.activeSessionId).toBe("sess-1");
    expect(next.activeRunId).toBe("run-42");
    expect(next.run?.status).toBe("running");
    expect(next.lastWhy?.goal_summary).toBe("完成终端界面切片");
    expect(next.steps).toHaveLength(1);
    expect(next.tools).toHaveLength(1);
    expect(next.sidebarSnapshot?.currentStep).toBe("写测试");
    expect(next.sidebarSnapshot?.pendingApprovalCount).toBe(1);
    expect(next.sidebarSnapshot?.cognitiveSnapshot.coreGoal).toBe("维持生命性、真实性与连续性");
    expect(next.sidebarSnapshot?.cognitiveSnapshot.vitalSigns.mood).toBe(0.61);
    expect(next.sidebarSnapshot?.cognitiveSnapshot.identity.displayName).toBe("阿澜");
    expect(next.statusline?.cwd).toBe("/tmp/demo");
    expect(next.permissionMode).toBe("ask");
  });

  it("tracks pending approvals, timeline, and clears approval on tool result", () => {
    const initial = createInitialUiState();
    const withCall = applyBridgeEvent(initial, {
      type: "tool_call",
      session_id: "sess-1",
      call_id: "run-1:tool:0",
      tool: "write_file",
      args: { path: "src/app.tsx" },
      summary: "准备写入 app.tsx",
      run_id: "run-1"
    });
    const withApproval = applyBridgeEvent(withCall, {
      type: "approval_request",
      session_id: "sess-1",
      call_id: "run-1:tool:0",
      tool: "write_file",
      args: { path: "src/app.tsx" },
      risk_level: "high",
      summary: "将修改工作区文件",
      action_preview: "write src/app.tsx"
    });
    const resolved = applyBridgeEvent(withApproval, {
      type: "tool_result",
      session_id: "sess-1",
      call_id: "run-1:tool:0",
      result: { tool_name: "write_file", status: "ok", summary: "updated app.tsx" }
    });

    expect(withApproval.pendingApprovals).toHaveLength(1);
    expect(withApproval.pendingApprovals[0]?.callId).toBe("run-1:tool:0");
    expect(withApproval.toolTimeline).toEqual([
      { kind: "call", callId: "run-1:tool:0", tool: "write_file", summary: "准备写入 app.tsx", status: undefined },
      { kind: "approval", callId: "run-1:tool:0", tool: "write_file", summary: "将修改工作区文件", status: "pending" }
    ]);
    expect(resolved.pendingApprovals).toEqual([]);
    expect(resolved.toolTimeline.at(-1)).toEqual({
      kind: "result",
      callId: "run-1:tool:0",
      tool: "write_file",
      summary: "updated app.tsx",
      status: "ok"
    });
  });

  it("accumulates assistant tokens into one transcript line", () => {
    const initial = createInitialUiState();
    const withFirst = applyBridgeEvent(initial, {
      type: "assistant_token",
      session_id: "sess-1",
      delta: "已进入只读任务处理。"
    });
    const withSecond = applyBridgeEvent(withFirst, {
      type: "assistant_token",
      session_id: "sess-1",
      delta: "可用 /status /why /steps /tools 查看进度。"
    });

    expect(withSecond.lines).toEqual([
      { kind: "assistant", text: "已进入只读任务处理。可用 /status /why /steps /tools 查看进度。" }
    ]);
  });

  it("finalizes a streamed assistant line without duplicating the final message", () => {
    const initial = createInitialUiState();
    const streaming = applyBridgeEvent(initial, {
      type: "assistant_token",
      session_id: "sess-1",
      delta: "已进入只读任务处理。可用 /status /why /steps /tools 查看进度。"
    });
    const completed = applyBridgeEvent(streaming, {
      type: "assistant_final",
      session_id: "sess-1",
      message: "已进入只读任务处理。可用 /status /why /steps /tools 查看进度。"
    });

    expect(completed.lines).toEqual([
      { kind: "assistant", text: "已进入只读任务处理。可用 /status /why /steps /tools 查看进度。" }
    ]);
  });
});
