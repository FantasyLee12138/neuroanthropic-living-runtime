import { describe, expect, it } from "vitest";

import { formatDetailSummary, formatSidebarSummary } from "./panelSummary.js";
import { createInitialUiState } from "./state/sessionStore.js";
import type { UiState } from "./types.js";

function makeState(overrides: Partial<UiState> = {}): UiState {
  return {
    ...createInitialUiState(),
    activeSessionId: "sess-1",
    activeRunId: "run-1",
    sessionMeta: { session_id: "sess-1", cwd: "/tmp/demo", mode: "plan", permission_mode: "plan" },
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

    expect(summary).toContain("脑态：已暂停");
    expect(summary).toContain("当前驱动：检查 planner.py");
    expect(summary).toContain("外显活动：工作区有变更");
    expect(summary).not.toContain("{");
  });

  it("renders endogenous source and latency split in status summary", () => {
    const summary = formatDetailSummary(
      makeState({
        console: {
          state: {
            brainState: {
              mode: "interactive",
              vitality: 0.52,
              selfContinuity: "稳定",
              authenticityPressure: "自然",
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
              roundId: 18,
              sampledAction: "respond",
              traceRef: "round://18",
              routeBudgetMs: 700,
              causeType: "endogenous",
              causeLabel: "内生整理",
              mode: "endogenous_light",
              modeLabel: "内生整理",
              activationSet: ["Router", "HotStateLoader", "Reflection/DMN"],
              memoryTiersRead: ["hot", "warm"],
              backgroundJobs: [{ kind: "memory_consolidation" }],
              deepenReason: "long_horizon_alignment",
              totalTurnMs: 420,
              modelWaitMs: 310,
              localComputeMs: 110,
              latencyDominant: "model_wait",
              latencySummary: "模型等待占主导",
              parallelTaskCount: 5,
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
              currentIntent: "正在进行内部整理，先在心里消化线索",
              vitalSigns: {
                mood: 0.0,
                bodyEnergy: 0.0,
                affectResidue: 0.0,
                focus: "deep",
                mode: "reflective",
              },
              identity: {
                displayName: "阿澜",
                continuity: "长期连续",
              },
              authenticity: {
                summary: "与当前设计一致",
                source: "official_console",
                guardAction: "none",
              },
            },
          },
          actionField: null,
          timeline: null,
          whyCurrent: null,
          whyNot: null,
        },
      }),
      "status",
    );

    expect(summary).toContain("当前驱动：内生整理");
    expect(summary).toContain("处理来源：内生整理");
    expect(summary).toContain("本轮时延：模型等待占主导（模型 310ms / 本地 110ms）");
    expect(summary).toContain("激活模块：Router · HotStateLoader · Reflection/DMN");
    expect(summary).toContain("记忆层：hot → warm");
    expect(summary).toContain("后台任务：1 项");
    expect(summary).toContain("加深原因：long_horizon_alignment");
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

    expect(formatDetailSummary(state, "why")).toContain("当前选择依据：理解仓库结构");
    expect(formatDetailSummary(state, "why")).toContain("当前选择依据：paused by operator");
    expect(formatDetailSummary(state, "steps")).toContain("当前驱动：检查 planner.py");
    expect(formatDetailSummary(state, "steps")).toContain("外显活动：2 步待续");
    expect(formatDetailSummary(state, "steps")).toContain("接续：总结模块边界；整理结论");
    expect(formatDetailSummary(state, "tools")).toContain("外显活动：仓库扫描：扫描了 3 个文件；读取文件：读取了 handlers.py");
  });

  it("renders hidden monologue-stream evidence in why summary", () => {
    const summary = formatDetailSummary(
      makeState({
        console: {
          ...createInitialUiState().console,
          actionField: {
            roundId: 52,
            traceRef: "round://52",
            topActions: [
              { action: "respond", score: 0.58 },
              { action: "monologue", score: 0.31 },
            ],
            winner: { action: "respond", score: 0.58 },
            conflict: {},
            tokenField: { raw: {} },
            contributionStack: [{ source: "MonologueStream", weight: 0.27, raw: {} }],
            competingPeaks: [{ action: "monologue", score: 0.31 }],
          },
          whyCurrent: {
            roundId: 52,
            traceRef: "round://52",
            why: {
              summary: "先回应当前对话，再把零散念头留在隐藏流里。",
              sampledAction: "respond",
              topDrivers: [{ agent_name: "PFCAgent" }, { agent_name: "MonologueStream" }],
              vitalitySnapshot: {},
              authenticity: {},
              initiative: {
                expression_mode: "external",
                top_intent: "respond",
                intrinsic_value: 0.46,
                speech_cost: 0.19,
                should_send: false,
                suppression_reason: "cooldown_active",
                memory_backing: {
                  cue: "写作业",
                  topic_source: "current_goal",
                  topic_relevance: 1,
                },
              },
              expressiveTrace: {
                monologueStream: {
                  hiddenByDefault: true,
                  generatedTotal: 42,
                  recentFragmentCount: 2,
                  sampleFragments: [
                    { content: "先喝口水" },
                    { content: "啊…" },
                  ],
                },
              },
            },
          },
          whyNot: null,
        },
      }),
      "why",
    );

    expect(summary).toContain("当前选择依据：先回应当前对话，再把零散念头留在隐藏流里。");
    expect(summary).toContain("主动性判定：external · respond · should_send=false");
    expect(summary).toContain("抑制原因：cooldown_active");
    expect(summary).toContain("记忆牵引：写作业 · source=current_goal · relevance=1.00");
    expect(summary).toContain("隐藏独白流：最近 2 条碎片，累计 42 条，默认不显示");
    expect(summary).toContain("样本：先喝口水；啊…");
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

    expect(summary).toContain("等待本轮解释：2");
    expect(summary).toContain("> [2/2] 执行命令");
    expect(summary).toContain("风险：中");
    expect(summary).toContain("模式：命令执行");
    expect(summary).toContain("状态：阻塞");
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
          module_model_bindings: {
            cognitive_packet: "small_model",
            deliberation: "large_model",
            tool_planner: "medium_model",
            deep_renderer: "large_model",
            consolidation_summarizer: "small_model"
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
    expect(cognition).toContain("脑态");
    expect(cognition).toContain("当前驱动：收口终端架构态改造");
    expect(cognition).toContain("内在状态");
    expect(cognition).toContain("收口终端架构态改造");
    expect(cognition).toContain("心境：中性 (0.58)");
    expect(cognition).toContain("注意焦点：任务定向");
    expect(cognition).toContain("真实性校验");
    expect(cognition).toContain("还没有足够证据判断这轮真实感");

    const meta = formatDetailSummary(state, "meta");
    expect(meta).toContain("当前环境：/tmp/demo");
    expect(meta).toContain("当前权限：需确认");
    expect(meta).toContain("模型分层");
    expect(meta).toContain("small_model：doubao / ep-20260404191810-qfn7s / 密钥已接入");
    expect(meta).toContain("主要模块模型绑定");
    expect(meta).toContain("cognitive_packet：small_model");
    expect(meta).toContain("deep_renderer：large_model");
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
      { label: "当前驱动", value: "完成终端重构", tone: "muted" },
      { label: "外显活动", value: "改 app.tsx 布局", tone: "normal" },
      { label: "工具痕迹", value: "读取文件：读取 app.tsx", tone: "muted" },
      { label: "等待本轮解释", value: "2 项待处理", tone: "warning" },
      { label: "脑态", value: "收口终端架构态改造", tone: "accent" }
    ]);
  });

  it("prefers official console state in summaries when available", () => {
    const state = makeState({
      sidebarSnapshot: {
        goalSummary: "完成终端重构",
        currentStep: "改 app.tsx 布局",
        reasonSummary: "主屏要先稳定",
        lastTool: "read_file",
        runStatus: "running",
        permissionMode: "ask",
        pendingApprovalCount: 0,
        modelStatus: undefined,
        console: undefined,
        cognitiveSnapshot: {
          coreGoal: "维持生命性、真实性与连续性",
          currentIntent: "旧派生态",
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
      console: {
        state: {
          brainState: {
            mode: "reflective",
            vitality: 0.84,
            selfContinuity: "长期连续",
            authenticityPressure: "解释收束中",
            longRunDriftRisk: null
          },
          neuromodulators: {
            dopamine: null,
            noradrenaline: null,
            serotonin: null,
            acetylcholine: null,
            gaba: null
          },
          motivationPool: {
            activeMotivations: [],
            raw: {}
          },
          longRun: {
            dream: {},
            traceStorage: {}
          },
          currentRound: {
            roundId: 12,
            sampledAction: "inspect",
            traceRef: "round://12",
            causeType: null
          },
          session: {
            sessionId: "sess-1",
            mode: "plan",
            safeMode: false
          },
          run: {
            runId: "run-1",
            status: "running",
            raw: {}
          },
          cognitiveSnapshot: {
            coreGoal: "维持生命性、真实性与连续性",
            currentIntent: "正式 console state 已接管摘要层",
            vitalSigns: {
              mood: 0,
              bodyEnergy: 0,
              affectResidue: 0,
              focus: "deep",
              mode: "reflective"
            },
            identity: {
              displayName: "阿澜",
              continuity: "长期连续"
            },
            authenticity: {
              summary: "与当前设计一致",
              source: "official_console",
              guardAction: "none"
            }
          }
        },
        actionField: {
          roundId: 12,
          traceRef: "round://12",
          topActions: [
            { action: "inspect", score: 0.91 },
            { action: "abort", score: 0.22 }
          ],
          winner: { action: "inspect", score: 0.91 },
          conflict: { winningPriority: "clarity_first" },
          tokenField: { raw: {} },
          contributionStack: [{ source: "planner", weight: 0.7, raw: {} }],
          competingPeaks: [{ action: "abort", score: 0.22 }]
        },
        timeline: {
          roundId: 12,
          traceRef: "round://12",
          events: [
            { type: "memory_activation", label: "planner", summary: "goal_stack" },
            { type: "action_arbitration", label: "inspect", summary: "clarity_first" },
            { type: "token_gate", label: "token_state", summary: "coupling_checked" }
          ]
        },
        whyCurrent: {
          roundId: 12,
          traceRef: "round://12",
          why: {
            summary: "因为先收口官方 console state",
            sampledAction: "inspect",
            topDrivers: [{ agentName: "planner" }],
            vitalitySnapshot: {},
            authenticity: { summary: "与当前设计一致" }
          }
        },
        whyNot: null
      },
      tools: []
    });

    expect(formatDetailSummary(state, "status")).toContain("脑态：运行中");
    expect(formatDetailSummary(state, "status")).toContain("当前驱动：上下文检查");
    expect(formatDetailSummary(state, "why")).toContain("当前选择依据：因为先收口官方 console state");
    expect(formatDetailSummary(state, "why")).toContain("未采纳路径：中止(0.22)");
    expect(formatDetailSummary(state, "why")).toContain("贡献叠层：前额叶规划(0.70)");
    expect(formatDetailSummary(state, "cognition")).toContain("当前驱动：正式 console state 已接管摘要层");
    expect(formatDetailSummary(state, "cognition")).toContain("注意焦点：深度聚焦");
    expect(formatDetailSummary(state, "steps")).toContain("当前驱动：上下文检查");
    expect(formatDetailSummary(state, "steps")).toContain("外显活动：3 段流转");
    expect(formatDetailSummary(state, "steps")).toContain("接续：前额叶规划；上下文检查；言语预激活场");
    expect(formatDetailSummary(state, "tools")).toContain("外显活动：记忆激活：目标堆栈；行动选择解算：清晰度优先；言语生成门控：表达门控已校验");
    expect(formatSidebarSummary(state)).toEqual([
      { label: "当前驱动", value: "上下文检查", tone: "muted" },
      { label: "外显活动", value: "因为先收口官方 console state", tone: "normal" },
      { label: "工具痕迹", value: "言语生成门控：表达门控已校验", tone: "muted" },
      { label: "等待本轮解释", value: "尚未接入", tone: "muted" },
      { label: "脑态", value: "正式 console state 已接管摘要层", tone: "accent" }
    ]);
  });

  it("preserves raw tokens and shows missing values as 尚未接入", () => {
    const summary = formatDetailSummary(
      makeState({
        run: {
          status: "mystery_mode",
          current_step: { title: "" },
          dirty_worktree_detected: undefined
        },
        lastWhy: {
          goal_summary: "",
          current_step: { title: "trace step", expected_observation: "" },
          stop_reason: { message: "" }
        },
        sidebarSnapshot: {
          goalSummary: "",
          currentStep: "",
          reasonSummary: "",
          lastTool: "",
          runStatus: "",
          permissionMode: "",
          pendingApprovalCount: 0,
          modelStatus: undefined,
          cognitiveSnapshot: {
            coreGoal: "",
            currentIntent: "",
            vitalSigns: {
              mood: 0.1,
              bodyEnergy: 0.1,
              affectResidue: 0.1,
              focus: "unmapped_focus",
              mode: "unmapped_mode"
            },
            identity: {
              displayName: "",
              continuity: ""
            },
            authenticity: {
              summary: "",
              source: "none",
              guardAction: "none"
            }
          }
        }
      }),
      "cognition"
    );

    expect(summary).toContain("当前驱动：尚未接入");
    expect(summary).toContain("注意焦点：unmapped_focus");
    expect(summary).toContain("运行方式：unmapped_mode");
    expect(summary).toContain("真实性校验");
    expect(summary).toContain("尚未接入");
  });
});
