import test from "node:test";
import assert from "node:assert/strict";

import {
  buildAnalysisFlowView,
  buildAnalysisReplayFrames,
  buildAnalysisTrendSeries,
  buildBrainflowOutput,
  buildBrainflowStages,
  buildChatTurnView,
  buildConflictRepairMarkers,
  buildInnerSpaceView,
  buildOverviewSnapshot,
  buildSettingsConsoleView,
} from "./view-models.js";

const helpers = {
  actionLabel(value) {
    return value === "respond" ? "回应" : value;
  },
  routeTypeLabel(value) {
    return value || "交互";
  },
  t(key) {
    return (
      {
        "summary.noWhy": "当前没有 why 摘要。",
        "summary.deliveryUnknown": "未知表达方式",
        "deepLink.round": "轮次详情",
      }[key] || key
    );
  },
};

const recentRounds = [
  { round_id: 7, sampled_action: "respond" },
  { round_id: 8, sampled_action: "wait" },
];

const roundCatalog = {
  "7": {
    trace: {
      sampled_action: "respond",
      action_bookkeeping: {
        u_base: { respond: -0.131 },
        ci: { respond: 0.2 },
      },
      probability_field: {
        action: {
          winner_posterior: { respond: 0.791 },
        },
      },
      gate_decisions: [
        {
          stage: "identity_guard",
          owner: "身份校验",
          allowed: false,
          requires_resample: true,
          reason: "表达越界，需要重新采样。",
        },
      ],
      conflict_arbitration: {
        critical_conflict: true,
        dominant_conflicts: ["身份校验"],
        repair_transition: {
          to_stage: "恢复阶段",
          reason: "完成修复后切回恢复流程。",
        },
        repair_ledger_tail: [{ reason: "回退输出" }],
      },
      initiative: {
        memory_backing: {
          strength: 0.633,
          cue: "关系边界",
        },
      },
      rendered_expression: {
        text: "先收住，再重新组织表达。",
      },
    },
  },
  "8": {
    trace: {
      sampled_action: "wait",
      action_bookkeeping: {
        u_base: { wait: 0.112 },
      },
      probability_field: {
        action: {
          winner_posterior: { wait: 0.33 },
        },
      },
      initiative: {
        memory_backing: {
          strength: 0.41,
          cue: "暂缓推进",
        },
      },
      rendered_expression: {
        text: "先观察，再决定是否继续。",
      },
    },
  },
};

test("buildAnalysisTrendSeries exposes Chinese-first labels and display sources", () => {
  const [uBase, pFinal, strength] = buildAnalysisTrendSeries({ recentRounds, roundCatalog });

  assert.equal(uBase.displayLabel, "基础驱动力");
  assert.equal(uBase.displaySubLabel, "动作原始驱动力");
  assert.equal(uBase.points[0].displaySource, "动作原始驱动力");

  assert.equal(pFinal.displayLabel, "最终胜出概率");
  assert.equal(pFinal.displaySubLabel, "概率场最终裁决");
  assert.equal(pFinal.points[0].displaySource, "概率场最终裁决");

  assert.equal(strength.displayLabel, "习惯牵引强度");
  assert.equal(strength.displaySubLabel, "习惯 / 记忆牵引");
  assert.equal(strength.points[0].displaySource, "习惯 / 记忆牵引");
  assert.equal(strength.points[0].source, "action_bookkeeping.ci");
  assert.equal(strength.points[1].source, "initiative.memory_backing.strength");
});

test("buildAnalysisReplayFrames aligns conflict and repair markers to the same round timeline", () => {
  const trendSeries = buildAnalysisTrendSeries({ recentRounds, roundCatalog });
  const frames = buildAnalysisReplayFrames(trendSeries);

  assert.equal(frames.length, 2);
  assert.equal(frames[0].roundLabel, "第 7 轮");
  assert.equal(frames[0].summary, "先收住，再重新组织表达。");
  assert.ok(frames[0].markers.some((item) => item.track === "conflict" && item.label === "冲突 · 身份校验"));
  assert.ok(
    frames[0].markers.some(
      (item) => item.track === "repair" && item.label === "修复 · 重新采样",
    ),
  );
  assert.equal(frames[0].metrics[0].displayLabel, "基础驱动力");
  assert.equal(frames[0].metrics[1].displayLabel, "最终胜出概率");
  assert.equal(frames[0].metrics[2].displayLabel, "习惯牵引强度");
});

test("buildAnalysisReplayFrames humanizes English repair fallback into Chinese summary", () => {
  const trendSeries = buildAnalysisTrendSeries({
    recentRounds: [{ round_id: 9, sampled_action: "respond" }],
    roundCatalog: {
      "9": {
        trace: {
          sampled_action: "respond",
          authenticity: { guard_action: "fallback" },
          action_bookkeeping: {
            u_base: { respond: 0.19 },
          },
          probability_field: {
            action: {
              winner_posterior: { respond: 0.58 },
            },
          },
          initiative: {
            memory_backing: {
              strength: 0.44,
            },
          },
        },
      },
    },
  });
  const frames = buildAnalysisReplayFrames(trendSeries);

  assert.equal(frames[0].summary, "身份校验要求回退输出。");
  assert.ok(
    frames[0].markers.some((item) => item.track === "repair" && item.detail === "身份校验要求回退输出。"),
  );
});

test("brainflow view humanizes action labels, blocked owners, and output copy", () => {
  const selectedPayload = {
    trace: {
      sampled_action: "plan",
      action_bookkeeping: {
        u_base: { plan: -1.55 },
        ci: { plan: 0.2 },
      },
      probability_field: {
        action: {
          winner_posterior: { plan: 0.07 },
        },
      },
      initiative: {
        memory_backing: {
          strength: 0.63,
          cue: "下午好",
          topic_source: "recent_user_turn",
        },
      },
      proposal_summaries: [
        {
          stage: "pfc",
          top_action: "plan",
        },
      ],
      gate_decisions: [
        {
          stage: "conflict_repair",
          owner: "ForcedModeSwitch",
          allowed: false,
          reason: "hard block triggered",
        },
        {
          stage: "conflict_repair",
          owner: "PerspectiveModel",
          allowed: false,
          reason: "perspective_disabled",
        },
      ],
      render_plan: {
        identity_context: {
          display_label: "朝屿",
          query_kind: "general",
        },
      },
      rendered_expression: {
        text: "你刚才提到“endogenous trigger silent_but_active”。 我还挂着“endogenous:silent_but_active”这条线。",
        route: "renderer",
        delivery_mode: "speech",
        model: "fallback",
      },
      conflict_arbitration: {
        critical_conflict: false,
      },
    },
  };

  const stages = buildBrainflowStages({ selectedPayload, selectedAction: "plan" });
  assert.match(stages[2].body, /规划/);
  assert.match(stages[3].meta, /常规语境/);

  const markers = buildConflictRepairMarkers({ roundId: 3441, selectedPayload });
  assert.ok(markers.blockedMarkers.some((item) => item.label === "阻断 · 强制模式切换"));
  assert.ok(markers.skippedMarkers.some((item) => item.label === "跳过 · 视角模型"));

  const output = buildBrainflowOutput({ selectedPayload });
  assert.match(output.text, /内源触发 静默但活跃/);
  assert.match(output.text, /内源 · 静默但活跃/);
  assert.equal(output.model, "回退模型");
});

test("buildAnalysisFlowView routes raw links through the new round detail entry", () => {
  const view = buildAnalysisFlowView(
    {
      roundId: 17,
      selectedAction: "respond",
      selectedRound: { mode: "interactive" },
      selectedPayload: {
        thought: {
          sampled_action: "respond",
          rendered_expression: { text: "先回应再展开。", delivery_mode: "文本" },
        },
        why: { why: { summary: "先完成回应。" } },
      },
    },
    helpers,
  );

  assert.equal(view.raw_links[0].label, "轮次详情");
  assert.equal(view.raw_links[0].href, "/workbench/round/17");
  assert.ok(view.evidence_cards.every((item) => item.refs.every((ref) => !ref.href.startsWith("/trace/"))));
});

test("buildAnalysisFlowView suppresses mixed English driver copy and humanizes why-not summaries", () => {
  const view = buildAnalysisFlowView(
    {
      roundId: 19,
      selectedAction: "plan",
      selectedRound: { mode: "chat" },
      selectedPayload: {
        thought: {
          sampled_action: "plan",
          action_field: {
            winner_posterior: { plan: 0.71, respond: 0.24 },
          },
          top_drivers: [
            {
              agent_name: "EmergentActionSketch",
              action_name: "plan",
              reason: "repeated internal pressure grows learned sketches that re-enter competition",
            },
          ],
          rendered_expression: { text: "先给一个短计划。" },
        },
        whyNot: {
          summary: "respond 没有胜出，最终由 plan 占优。主要阻力是 ForcedModeSwitch。",
        },
      },
    },
    helpers,
  );

  assert.match(view.evidence_cards[0].body, /动作倾向：plan|动作倾向：规划/);
  assert.match(view.evidence_cards[0].body, /中文说明/);
  assert.doesNotMatch(view.evidence_cards[0].body, /repeated|competition/);
  assert.match(view.evidence_cards.at(-1).body, /回应/);
  assert.match(view.evidence_cards.at(-1).body, /规划/);
  assert.match(view.evidence_cards.at(-1).body, /强制模式切换/);
});

test("buildInnerSpaceView routes memory, dream, and monologue evidence through the aggregated read model", () => {
  const view = buildInnerSpaceView(
    {
      roundId: 17,
      thought: { sampled_action: "respond", top_drivers: [] },
      memoryTop: [],
      dreamOverview: { status: { enabled: false }, recent_runs: [] },
      monologueShow: {
        fragments: [{ text: "我还在整理边界。" }],
      },
      consoleData: { recent_rounds: [] },
    },
    helpers,
  );

  assert.equal(view.thinking_stream.at(-1).refs[0].href, "/workbench/read-model?inner_space=true");
  assert.equal(view.memory_recall[0].refs[0].href, "/workbench/read-model?inner_space=true");
  assert.equal(view.dream_fragments[0].refs[0].href, "/workbench/read-model?inner_space=true");
});

test("buildChatTurnView routes memory evidence through the aggregated read model", () => {
  const view = buildChatTurnView(
    {
      assistantText: "先回应。",
      selectedPayload: { why: { why: { summary: "当前先回应。" } } },
      selectedAction: "respond",
      roundId: 17,
      innerSpace: { memoryTop: [{ cue: "边界" }] },
    },
    helpers,
  );

  assert.equal(view.expandable_evidence_refs[0].href, "/workbench/round/17");
  assert.equal(view.expandable_evidence_refs[1].href, "/workbench/read-model?inner_space=true");
});

test("buildOverviewSnapshot folds subject, meaning, and agency narratives into the overview summary", () => {
  const snapshot = buildOverviewSnapshot(
    {
      bootstrap: { session_attached: true },
      service: { healthy: true },
      autonomy: { running: true },
      runtimeState: {
        cognitive_snapshot: {
          identity: { display_name: "NALR" },
          vital_signs: { mood: 0.62 },
          tlh: {
            body_state: { fatigue: 0.18, self_continuity: 0.71, meaning_strength: 0.66 },
            subjective_state: { felt: ["稳定", "专注"], boundary: 0.74, spontaneous: 0.58 },
          },
        },
      },
      consoleData: {
        state: { current_round: { round_id: 12, sampled_action: "respond", mode: "interactive" } },
      },
      models: { module_model_bindings: { cognitive_packet: "planner-tier", deep_renderer: "renderer-tier" } },
      settings: { identity: { display_name: "fallback" } },
      subject: {
        subject_kernel: {
          display_name: "NALR",
          current_self_narrative: "我在维护边界与连续性。",
          core_commitments: ["维护连续性", "保持关系边界"],
        },
      },
      meaning: {
        meaning_system: {
          survival_narrative: "继续维护关系并完成长期学习。",
        },
      },
      agency: {
        agency_loop: {
          summary: "先整理当前目标，再主动联络重要关系。",
        },
        proactive_backlog: [{ title: "主动联系重要关系" }],
      },
    },
    helpers,
  );

  assert.match(snapshot.identity_intro, /NALR/);
  assert.match(snapshot.current_focus, /继续维护关系并完成长期学习/);
  assert.match(snapshot.persona_mood_summary, /先整理当前目标，再主动联络重要关系/);
  assert.ok(snapshot.persona_mood_signals.some((item) => item.label === "主体承诺" && /维护连续性/.test(item.value)));
  assert.ok(snapshot.persona_mood_signals.some((item) => item.label === "主动节奏" && /主动联系重要关系/.test(item.value)));
});

test("buildSettingsConsoleView surfaces the new workbench read-model and round detail endpoints", () => {
  const view = buildSettingsConsoleView(
    {
      selectedRoundId: 25,
      consoleData: { state: { current_round: { round_id: 25 } } },
      settings: { models: { module_model_bindings: {} }, autonomy: {} },
      runtimeState: {},
      service: { healthy: true },
      autonomy: { running: true },
    },
    {
      t(key) {
        return (
          {
            "common.healthy": "健康",
            "common.degraded": "降级",
            "common.running": "运行中",
            "common.stopped": "未运行",
            "common.noneRound": "暂无当前轮次",
            "common.none": "暂无",
            "common.open": "开启",
            "common.closed": "关闭",
          }[key] || key
        );
      },
    },
  );

  const evidenceApi = view.diagnostic_panels[1].rows.map((item) => item.value);
  assert.ok(evidenceApi.includes("/workbench/read-model"));
  assert.ok(evidenceApi.includes("/workbench/round/25"));
  assert.ok(evidenceApi.includes("/workbench/read-model?inner_space=true"));
  assert.ok(evidenceApi.every((value) => value !== "/memory/top?limit=6"));
  assert.ok(evidenceApi.every((value) => value !== "/dream/overview?limit=6"));
});

test("buildSettingsConsoleView adds scheduled task and hot-path diagnostics from the plan1.6 read model", () => {
  const view = buildSettingsConsoleView(
    {
      selectedRoundId: 25,
      consoleData: { state: { current_round: { round_id: 25 } } },
      settings: { models: { module_model_bindings: {} }, autonomy: {} },
      runtimeState: {},
      service: { healthy: true },
      autonomy: { running: true },
      agency: {
        scheduled_tasks: [
          { task_id: "weekly-repo-scan", status: "idle", next_run_at: "2026-04-12T12:00:00Z" },
          { task_id: "relationship-ping", status: "running", next_run_at: "2026-04-12T18:00:00Z" },
        ],
      },
      performance: {
        latency: {
          total_turn_ms: 182,
          model_wait_ms: 121,
          local_compute_ms: 61,
          latency_summary: "模型等待占主导",
        },
      },
    },
    {
      t(key) {
        return (
          {
            "common.healthy": "健康",
            "common.degraded": "降级",
            "common.running": "运行中",
            "common.stopped": "未运行",
            "common.noneRound": "暂无当前轮次",
            "common.none": "暂无",
            "common.open": "开启",
            "common.closed": "关闭",
          }[key] || key
        );
      },
    },
  );

  const scheduledPanel = view.diagnostic_panels.find((panel) => panel.title === "计划任务");
  assert.ok(scheduledPanel);
  assert.ok(scheduledPanel.rows.some((item) => item.label === "任务状态" && /2 个任务/.test(item.value)));
  assert.ok(scheduledPanel.rows.some((item) => item.value === "/scheduled-tasks"));

  const hotPathPanel = view.diagnostic_panels.find((panel) => panel.title === "热路径");
  assert.ok(hotPathPanel);
  assert.ok(hotPathPanel.rows.some((item) => item.label === "耗时摘要" && item.value === "模型等待占主导"));
  assert.ok(hotPathPanel.rows.some((item) => item.value === "/performance/hot-path"));
});

test("buildSettingsConsoleView shapes a dedicated scheduled-task surface for settings", () => {
  const view = buildSettingsConsoleView(
    {
      settings: { models: { module_model_bindings: {} }, autonomy: {} },
      runtimeState: {},
      service: { healthy: true },
      autonomy: { running: true },
      agency: {
        scheduled_tasks: [
          {
            task_id: "relationship-ping",
            skill_name: "scheduled_self_run",
            status: "running",
            next_run_at: "2026-04-12T18:00:00Z",
            prompt: "继续跟进关系维护节奏，整理一条只读提醒。",
          },
          {
            task_id: "weekly-repo-scan",
            skill_name: "repo_scan",
            status: "idle",
            next_run_at: "2026-04-12T12:00:00Z",
            task_payload: { goal: "扫描工作区内最近变更，并给出只读检查建议。" },
          },
        ],
      },
    },
    {
      t(key) {
        return (
          {
            "common.healthy": "健康",
            "common.degraded": "降级",
            "common.running": "运行中",
            "common.stopped": "未运行",
            "common.noneRound": "暂无当前轮次",
            "common.none": "暂无",
            "common.open": "开启",
            "common.closed": "关闭",
            "scheduledTask.summary.total": "任务总数",
            "scheduledTask.summary.running": "运行中",
            "scheduledTask.summary.nextRun": "最近下一次",
            "scheduledTask.status.idle": "待执行",
            "scheduledTask.status.running": "运行中",
            "scheduledTask.status.disabled": "已停用",
            "scheduledTask.excerpt.prompt": "提示摘录",
            "scheduledTask.excerpt.goal": "目标摘录",
          }[key] || key
        );
      },
    },
  );

  assert.deepEqual(
    view.scheduled_task_surface.summary.map((item) => item.label),
    ["任务总数", "运行中", "最近下一次"],
  );
  assert.equal(view.scheduled_task_surface.summary[0].value, "2");
  assert.equal(view.scheduled_task_surface.summary[1].value, "1");
  assert.equal(view.scheduled_task_surface.summary[2].value, "2026-04-12T12:00:00Z");
  assert.equal(view.scheduled_task_surface.tasks[0].task_id, "relationship-ping");
  assert.equal(view.scheduled_task_surface.tasks[0].status_label, "运行中");
  assert.equal(view.scheduled_task_surface.tasks[0].excerpt_label, "提示摘录");
  assert.match(view.scheduled_task_surface.tasks[0].excerpt, /关系维护节奏/);
  assert.equal(view.scheduled_task_surface.tasks[1].task_id, "weekly-repo-scan");
  assert.equal(view.scheduled_task_surface.tasks[1].skill_name, "repo_scan");
  assert.equal(view.scheduled_task_surface.tasks[1].status_label, "待执行");
  assert.equal(view.scheduled_task_surface.tasks[1].excerpt_label, "目标摘录");
  assert.match(view.scheduled_task_surface.tasks[1].excerpt, /扫描工作区内最近变更/);
});
