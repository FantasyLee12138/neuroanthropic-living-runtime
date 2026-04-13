import { describe, expect, it } from "vitest";

import { applyBridgeEvent, createInitialUiState } from "./sessionStore.js";

describe("sessionStore", () => {
  it("tracks active session and active run from bridge events", () => {
    const initial = createInitialUiState();
    expect(initial.aliveConsole.state.title).toBe("脑态");
    expect(initial.aliveConsole.actionField.title).toBe("思绪流");
    expect(initial.aliveConsole.why.title).toBe("解释层");
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
    expect(withRun.lines).toEqual([{ kind: "system", text: "任务已启动。" }]);
    expect(withRun.pendingApprovals).toEqual([]);
    expect(withRun.activityRail).toEqual([]);
    expect(withRun.toolTimeline).toEqual([]);
    expect(withRun.aliveConsole.state.summary).toBe("运行中 · 规划模式");
    expect(withRun.aliveConsole.actionField.summary).toBe("待审批 0");
    expect(withRun.aliveConsole.why.summary).toBe("暂无");
    expect(withRun.detailDrawer).toBeNull();
    expect(withRun.focusZone).toBe("input");
    expect(withRun.approvalCursor).toBe(0);
  });

  it("hydrates transcript, timeline, approvals, and transcript mode from session_started", () => {
    const initial = createInitialUiState();
    const next = applyBridgeEvent(initial, {
      type: "session_started",
      session: {
        session_id: "sess-restore",
        cwd: "/tmp/demo",
        status: "detached",
        permission_mode: "ask",
        transcript_mode: "compact",
        transcript_lines: [
          { kind: "user", text: "你好" },
          { kind: "assistant", text: "我在。" }
        ],
        tool_timeline: [
          { kind: "call", callId: "run-1:tool:0", tool: "repo_scan", summary: "scan", status: "completed" }
        ],
        approvals_pending: [
          {
            call_id: "run-1:tool:0",
            tool: "repo_scan",
            summary: "scan",
            action_preview: "scan src",
            status: "pending",
            run_id: "run-1"
          }
        ]
      }
    });

    expect(next.activeSessionId).toBe("sess-restore");
    expect(next.permissionMode).toBe("ask");
    expect(next.transcriptMode).toBe("compact");
    expect(next.lines).toEqual([
      { kind: "user", text: "你好" },
      { kind: "assistant", text: "我在。" }
    ]);
    expect(next.toolTimeline).toEqual([
      { kind: "call", callId: "run-1:tool:0", tool: "repo_scan", summary: "scan", status: "completed" }
    ]);
    expect(next.activityRail).toEqual([
      { kind: "tool", label: "repo_scan", summary: "scan", status: "completed", callId: "run-1:tool:0" },
      { kind: "approval", label: "repo_scan", summary: "scan", status: "pending", callId: "run-1:tool:0" }
    ]);
    expect(next.pendingApprovals).toEqual([
      {
        callId: "run-1:tool:0",
        tool: "repo_scan",
        args: undefined,
        riskLevel: undefined,
        summary: "scan",
        actionPreview: "scan src",
        mode: undefined,
        status: "pending",
        runId: "run-1",
        choices: [],
      }
    ]);
    expect(next.approvalCursor).toBe(0);
  });

  it("routes steps and tool results into the activity rail instead of the main transcript", () => {
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
      round_id: 12,
      trace_ref: "trace://round/12",
      result: { tool_name: "repo_scan", summary: "scanned 3 files" }
    });

    expect(withTool.steps).toHaveLength(1);
    expect(withTool.tools).toHaveLength(1);
    expect(withTool.tools[0]?.tool_name).toBe("repo_scan");
    expect(withTool.lines).toEqual([]);
    expect(withTool.activityRail).toEqual([
      { kind: "step", label: "检查 planner.py", status: undefined },
      {
        kind: "result",
        label: "repo_scan",
        summary: "scanned 3 files",
        status: undefined,
        callId: "run-1:tool:0",
        roundId: 12,
        traceRef: "trace://round/12"
      }
    ]);
    expect(withTool.toolTimeline).toEqual([
      {
        kind: "result",
        callId: "run-1:tool:0",
        tool: "repo_scan",
        summary: "scanned 3 files",
        status: undefined,
        roundId: 12,
        traceRef: "trace://round/12"
      }
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
      round_id: 12,
      trace_ref: "trace://round/12",
      ui_actions: {
        primary: [
          { id: "status", label: "状态", kind: "drawer", value: "status", disabled: false },
          { id: "approvals", label: "审批", kind: "drawer", value: "approvals", disabled: false }
        ],
        secondary: [
          { id: "abort", label: "中止", kind: "command", value: "abort", disabled: false }
        ]
      },
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
      console: {
        state: {
          brain_state: {
            mode: "reflective",
            vitality: 0.93,
            self_continuity: "长期身份连续",
            authenticity_pressure: "解释一致性较高",
            long_run_drift_risk: null
          },
          neuromodulators: {},
          motivation_pool: {},
          long_run: {},
          current_round: {
            round_id: 12,
            sampled_action: "inspect",
            trace_ref: "round://12",
            route_type: "endogenous_light",
            route_budget_ms: 300,
            cause_type: "endogenous",
            cause_label: "内生整理",
            mode: "endogenous_light",
            mode_label: "内生整理",
            activation_set: ["Router", "HotStateLoader", "Reflection/DMN"],
            activation_reason: ["deep_route_requested"],
            memory_tiers_read: ["hot", "warm"],
            packet_summary: { candidate_action_prior: "inspect" },
            background_jobs: [{ kind: "memory_consolidation" }],
            deepen_reason: "long_horizon_alignment",
            model_call_count: 2,
            total_turn_ms: 420,
            model_wait_ms: 310,
            local_compute_ms: 110,
            latency_dominant: "model_wait",
            latency_summary: "模型等待占主导",
            parallel_task_count: 5
          },
          session: {
            session_id: "sess-1",
            mode: "plan",
            safe_mode: false
          },
          run: {
            run_id: "run-42",
            status: "running"
          },
          cognitive_snapshot: {
            current_intent: "以正式 console payload 收口前端状态",
            vital_signs: {
              focus: "deep"
            }
          }
        },
        action_field: {
          round_id: 12,
          trace_ref: "round://12",
          top_actions: [
            { action: "inspect", score: 0.91 },
            { action: "abort", score: 0.22 }
          ],
          winner: {
            action: "inspect",
            score: 0.91
          },
          conflict: {
            winning_priority: "clarity_first"
          },
          token_field: {
            attention_anchor: "sessionStore"
          },
          contribution_stack: [
            { source: "planner", weight: 0.7 }
          ],
          competing_peaks: [
            { action: "abort", score: 0.22 }
          ]
        },
        timeline: {
          round_id: 12,
          trace_ref: "round://12",
          events: [
            { type: "action_arbitration", label: "inspect", summary: "clarity_first" }
          ]
        },
        why_current: {
          round_id: 12,
          trace_ref: "round://12",
          why: {
            summary: "因为先钉住正式 payload 的单一事实源",
            sampled_action: "inspect",
            top_drivers: [
              { agent_name: "planner" }
            ],
            vitality_snapshot: {},
            authenticity: {
              summary: "与当前设计口径一致"
            },
            initiative: {
              expression_mode: "external",
              proposal_type: "speak",
              top_intent: "follow_up_task",
              speech_cost: 0.21,
              intrinsic_value: 0.66,
              should_send: false,
              suppression_reason: "cooldown_active",
              memory_backing: {
                cue: "写作业",
                summary: "写作业",
                topic_source: "current_goal",
                topic_relevance: 1
              }
            },
            expressive_trace: {
              monologue_stream: {
                hidden_by_default: true,
                generated_total: 42,
                recent_fragment_count: 2,
                sample_fragments: [
                  { content: "先喝口水" },
                  { content: "啊…" }
                ]
              }
            }
          }
        },
        why_not: {
          round_id: 12,
          trace_ref: "round://12",
          action: "abort",
          why_not: {
            selected_action: "inspect",
            candidate_score: 0.22,
            blocked_by: ["ConflictMonitorAgent", "planner"],
            stacked_contributions: [{ module_name: "ConflictMonitorAgent" }, { module_name: "planner" }],
            competing_peaks: [{ action: "abort", score: 0.22 }],
            summary: "这轮中止没有胜出，主要被正式检查链压住。"
          }
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
    expect(next.actionBar.primary.map((item) => item.id)).toEqual(["status", "approvals"]);
    expect(next.actionBar.secondary.map((item) => item.id)).toEqual(["abort"]);
    expect(next.sidebarSnapshot?.cognitiveSnapshot.coreGoal).toBe("维持生命性、真实性与连续性");
    expect(next.sidebarSnapshot?.cognitiveSnapshot.vitalSigns.mood).toBe(0.61);
    expect(next.sidebarSnapshot?.cognitiveSnapshot.identity.displayName).toBe("阿澜");
    expect(next.statusline?.cwd).toBe("/tmp/demo");
    expect(next.permissionMode).toBe("ask");
    expect(next.sidebarSnapshot?.roundId).toBe(12);
    expect(next.console.state?.currentRound.routeType).toBe("endogenous_light");
    expect(next.sidebarSnapshot?.traceRef).toBe("trace://round/12");
    expect(next.console.state?.brainState.mode).toBe("reflective");
    expect(next.console.state?.currentRound.causeType).toBe("endogenous");
    expect(next.console.state?.currentRound.causeLabel).toBe("内生整理");
    expect(next.console.state?.currentRound.modeLabel).toBe("内生整理");
    expect(next.console.state?.currentRound.routeBudgetMs).toBe(300);
    expect(next.console.state?.currentRound.activationSet).toEqual(["Router", "HotStateLoader", "Reflection/DMN"]);
    expect(next.console.state?.currentRound.memoryTiersRead).toEqual(["hot", "warm"]);
    expect(next.console.state?.currentRound.packetSummary).toEqual({ candidate_action_prior: "inspect" });
    expect(next.console.state?.currentRound.backgroundJobs).toEqual([{ kind: "memory_consolidation" }]);
    expect(next.console.state?.currentRound.deepenReason).toBe("long_horizon_alignment");
    expect(next.console.state?.currentRound.modelCallCount).toBe(2);
    expect(next.console.state?.currentRound.modelWaitMs).toBe(310);
    expect(next.console.state?.currentRound.localComputeMs).toBe(110);
    expect(next.console.state?.currentRound.latencySummary).toBe("模型等待占主导");
    expect(next.console.actionField?.winner.action).toBe("inspect");
    expect(next.console.timeline?.events[0]?.type).toBe("action_arbitration");
    expect(next.console.whyCurrent?.why.summary).toBe("因为先钉住正式 payload 的单一事实源");
    expect((next.console.whyCurrent?.why.expressiveTrace as Record<string, any>)?.monologueStream?.recentFragmentCount).toBe(2);
    expect(next.console.whyNot?.action).toBe("abort");
    expect(next.console.whyNot?.whyNot.blocked_by).toEqual(["ConflictMonitorAgent", "planner"]);
    expect(next.console.whyNot?.whyNot.summary).toBe("这轮中止没有胜出，主要被正式检查链压住。");
    expect(next.aliveConsole.state.details[0]).toBe("模式：反思态");
    expect(next.aliveConsole.state.details.some((line) => line.includes("活力：0.93"))).toBe(true);
    expect(next.aliveConsole.actionField.summary).toBe("上下文检查 · 倾向 0.91");
    expect(next.aliveConsole.actionField.details.some((line) => line.includes("未采纳路径：中止(0.22)"))).toBe(true);
    expect(next.aliveConsole.why.summary).toBe("因为先钉住正式 payload 的单一事实源");
    expect(next.aliveConsole.why.details.some((line) => line.includes("主导模块：前额叶规划"))).toBe(true);
    expect(next.aliveConsole.why.details.some((line) => line.includes("表达模式：external · intent=follow_up_task · should_send=false"))).toBe(true);
    expect(next.aliveConsole.why.details.some((line) => line.includes("抑制原因：cooldown_active"))).toBe(true);
    expect(next.aliveConsole.why.details.some((line) => line.includes("记忆牵引：写作业 · source=current_goal · relevance=1.00"))).toBe(true);
    expect(next.aliveConsole.why.details.some((line) => line.includes("隐藏独白流：最近 2 条碎片，累计 42 条，默认不显示"))).toBe(true);
  });

  it("renders token field arrays in alive console details", () => {
    const initial = createInitialUiState();
    const next = applyBridgeEvent(initial, {
      type: "sidebar_snapshot",
      session_id: "sess-1",
      goal_summary: "",
      current_step: "",
      reason_summary: "",
      last_tool: "",
      run_status: "running",
      permission_mode: "plan",
      pending_approval_count: 0,
      status: { run_id: "run-7", status: "running" },
      why: {},
      steps: [],
      tools: [],
      session: { session_id: "sess-1", cwd: "/tmp/demo", mode: "plan", permission_mode: "plan" },
      cognitive_snapshot: {
        core_goal: "",
        current_intent: "检查 token state 展示",
        vital_signs: {
          mood: 0,
          body_energy: 0,
          affect_residue: 0,
          focus: "deep",
          mode: "plan"
        },
        identity: {
          display_name: "",
          continuity: ""
        },
        authenticity: {
          summary: "",
          source: "none",
          guard_action: "none"
        }
      },
      console: {
        state: {
          brain_state: {},
          neuromodulators: {},
          motivation_pool: {},
          long_run: {},
          current_round: {
            round_id: 7,
            sampled_action: "inspect",
            trace_ref: "round://7"
          },
          session: {
            session_id: "sess-1",
            mode: "plan",
            safe_mode: false
          },
          run: {
            run_id: "run-7",
            status: "running"
          },
          cognitive_snapshot: {
            current_intent: "检查 token state 展示",
            vital_signs: {
              focus: "deep"
            }
          }
        },
        action_field: {
          round_id: 7,
          trace_ref: "round://7",
          top_actions: [{ action: "inspect", score: 0.8 }],
          winner: {
            action: "inspect",
            score: 0.8
          },
          conflict: {},
          token_field: {
            active_module_sources: ["PFCAgent", "Renderer"],
            delta_generation_policy: "propagate_only",
            prefix_tokens: ["hello", "world"]
          },
          contribution_stack: [],
          competing_peaks: []
        }
      }
    });

    expect(next.aliveConsole.actionField.details).toContain(
      "言语预激活：生成策略为仅传播当前已形成的状态，不额外生成新增量；前缀语句为hello world"
    );
  });

  it("falls back to derived alive console columns when sidebar_snapshot has no console payload", () => {
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
      round_id: 12,
      trace_ref: "trace://round/12",
      ui_actions: {
        primary: [
          { id: "status", label: "状态", kind: "drawer", value: "status", disabled: false },
          { id: "approvals", label: "审批", kind: "drawer", value: "approvals", disabled: false }
        ],
        secondary: [
          { id: "abort", label: "中止", kind: "command", value: "abort", disabled: false }
        ]
      },
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

    expect(next.aliveConsole.state.details[0]).toBe("模式：交互态");
    expect(next.aliveConsole.actionField.summary).toBe("状态 · 待审批 0");
    expect(next.aliveConsole.why.summary).toBe("完成终端界面切片");
    expect(next.console.state).toBeNull();
    expect(next.console.actionField).toBeNull();
    expect(next.console.timeline).toBeNull();
    expect(next.console.whyCurrent).toBeNull();
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
      action_preview: "write src/app.tsx",
      round_id: 3,
      trace_ref: "trace://round/3",
      choices: [
        { id: "approve", label: "批准", kind: "approval", value: "approve", disabled: false },
        { id: "reject", label: "拒绝", kind: "approval", value: "reject", disabled: false }
      ]
    });
    const resolved = applyBridgeEvent(withApproval, {
      type: "tool_result",
      session_id: "sess-1",
      call_id: "run-1:tool:0",
      result: { tool_name: "write_file", status: "ok", summary: "updated app.tsx" }
    });

    expect(withApproval.pendingApprovals).toHaveLength(1);
    expect(withApproval.pendingApprovals[0]?.callId).toBe("run-1:tool:0");
    expect(withApproval.pendingApprovals[0]?.choices).toEqual([
      { id: "approve", label: "批准", kind: "approval", value: "approve", disabled: false },
      { id: "reject", label: "拒绝", kind: "approval", value: "reject", disabled: false }
    ]);
    expect(withApproval.lines).toEqual([{ kind: "system", text: "等待审批：write_file" }]);
    expect(withApproval.toolTimeline).toEqual([
      { kind: "call", callId: "run-1:tool:0", tool: "write_file", summary: "准备写入 app.tsx", status: undefined },
      {
        kind: "approval",
        callId: "run-1:tool:0",
        tool: "write_file",
        summary: "将修改工作区文件",
        status: "pending",
        roundId: 3,
        traceRef: "trace://round/3"
      }
    ]);
    expect(withApproval.activityRail).toEqual([
      { kind: "tool", label: "write_file", summary: "准备写入 app.tsx", status: undefined, callId: "run-1:tool:0" },
      {
        kind: "approval",
        label: "write_file",
        summary: "将修改工作区文件",
        status: "pending",
        callId: "run-1:tool:0",
        roundId: 3,
        traceRef: "trace://round/3"
      }
    ]);
    expect(withApproval.approvalCursor).toBe(0);
    expect(withApproval.aliveConsole.actionField.summary).toBe("待审批 1");
    expect(withApproval.aliveConsole.actionField.details.some((line) => line.includes("写入文件"))).toBe(true);
    expect(resolved.pendingApprovals).toEqual([]);
    expect(resolved.toolTimeline.at(-1)).toEqual({
      kind: "result",
      callId: "run-1:tool:0",
      tool: "write_file",
      summary: "updated app.tsx",
      status: "ok"
    });
    expect(resolved.activityRail.at(-1)).toEqual({
      kind: "result",
      label: "write_file",
      summary: "updated app.tsx",
      status: "ok",
      callId: "run-1:tool:0"
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

  it("hydrates console why-not state from assistant_final payloads", () => {
    const initial = createInitialUiState();
    const seeded = {
      ...initial,
      lastWhy: {
        goal_summary: "保留原有 why 解释",
        current_step: { title: "检查 planner.py" },
      },
    };
    const next = applyBridgeEvent(seeded, {
      type: "assistant_final",
      session_id: "sess-1",
      message: "为什么不是 plan：被冲突仲裁压制。",
      payload: {
        round_id: 12,
        action: "plan",
        selected_action: "inspect",
        candidate_score: 0.22,
        blocked_by: ["ConflictMonitorAgent", "planner"],
        stacked_contributions: [{ source: "planner", weight: 0.7 }],
        competing_peaks: [{ action: "inspect", score: 0.91 }],
        trace_ref: "trace://round/12"
      }
    });

    expect(next.console.whyNot?.roundId).toBe(12);
    expect(next.console.whyNot?.action).toBe("plan");
    expect(next.console.whyNot?.whyNot.blocked_by).toEqual(["ConflictMonitorAgent", "planner"]);
    expect(next.console.whyNot?.traceRef).toBe("trace://round/12");
    expect(next.lastWhy?.goal_summary).toBe("保留原有 why 解释");
  });

  it("keeps why-not state across same-round sidebar snapshots and clears it on a new round", () => {
    const initial = createInitialUiState();
    const withWhyNot = applyBridgeEvent(initial, {
      type: "assistant_final",
      session_id: "sess-1",
      message: "为什么不是 plan：被冲突仲裁压制。",
      payload: {
        round_id: 12,
        action: "plan",
        selected_action: "inspect",
        blocked_by: ["ConflictMonitorAgent"],
        trace_ref: "trace://round/12"
      }
    });
    const sameRound = applyBridgeEvent(withWhyNot, {
      type: "sidebar_snapshot",
      session_id: "sess-1",
      goal_summary: "完成终端界面切片",
      current_step: "写测试",
      reason_summary: "先钉住 reducer 行为",
      last_tool: "read_file",
      run_status: "running",
      permission_mode: "ask",
      pending_approval_count: 0,
      round_id: 12,
      trace_ref: "trace://round/12",
      console: {
        state: {
          current_round: { round_id: 12, sampled_action: "inspect", trace_ref: "round://12" },
          run: { run_id: "run-42", status: "running" },
          cognitive_snapshot: { current_intent: "继续当前轮次" }
        },
        action_field: {
          round_id: 12,
          trace_ref: "round://12",
          winner: { action: "inspect", score: 0.91 }
        },
        timeline: { round_id: 12, trace_ref: "round://12", events: [] },
        why_current: { round_id: 12, trace_ref: "round://12", why: { summary: "继续当前轮次" } }
      }
    });
    const nextRound = applyBridgeEvent(sameRound, {
      type: "sidebar_snapshot",
      session_id: "sess-1",
      goal_summary: "进入新轮次",
      current_step: "继续测试",
      reason_summary: "新轮次应清掉旧 why-not",
      last_tool: "read_file",
      run_status: "running",
      permission_mode: "ask",
      pending_approval_count: 0,
      round_id: 13,
      trace_ref: "trace://round/13",
      console: {
        state: {
          current_round: { round_id: 13, sampled_action: "respond", trace_ref: "round://13" },
          run: { run_id: "run-43", status: "running" },
          cognitive_snapshot: { current_intent: "进入新轮次" }
        },
        action_field: {
          round_id: 13,
          trace_ref: "round://13",
          winner: { action: "respond", score: 0.88 }
        },
        timeline: { round_id: 13, trace_ref: "round://13", events: [] },
        why_current: { round_id: 13, trace_ref: "round://13", why: { summary: "进入新轮次" } }
      }
    });

    expect(sameRound.console.whyNot?.action).toBe("plan");
    expect(sameRound.console.whyNot?.roundId).toBe(12);
    expect(nextRound.console.whyNot).toBeNull();
  });
});
