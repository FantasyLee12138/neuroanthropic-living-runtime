import { WORKBENCH_READ_MODEL_PATH, buildWorkbenchRoundPath } from "./api.js";
import { humanizeRuntimeText, humanizeRuntimeToken } from "./format.js";

const INNER_SPACE_READ_MODEL_PATH = `${WORKBENCH_READ_MODEL_PATH}?inner_space=true`;

function firstText(...values) {
  for (const value of values) {
    if (typeof value === "string" && value.trim()) {
      return value.trim();
    }
  }
  return "";
}

function asArray(value) {
  return Array.isArray(value) ? value : [];
}

function hasItems(value) {
  if (Array.isArray(value)) {
    return value.length > 0;
  }
  if (value && typeof value === "object") {
    return Object.keys(value).length > 0;
  }
  return Boolean(value);
}

function topPosteriorRows(actionField = {}) {
  const winnerPosterior = actionField?.winner_posterior || {};
  return Object.entries(winnerPosterior)
    .sort((a, b) => Number(b[1] || 0) - Number(a[1] || 0))
    .slice(0, 5);
}

function deepLink(label, href) {
  return { label, href };
}

function humanizeRouteLabel(value) {
  const key = String(value || "").trim().toLowerCase();
  return {
    renderer: "表达层",
    interactive: "交互通道",
    chat: "对话通道",
    speech: "文字",
    text: "文字",
    voice: "语音",
  }[key] || humanizeRuntimeToken(firstText(value, "未知路径"));
}

function innerSpaceDeepLink(label = "内在空间") {
  return deepLink(label, INNER_SPACE_READ_MODEL_PATH);
}

function naturalList(values, empty) {
  const rows = asArray(values).filter(Boolean);
  return rows.length ? rows.join("、") : empty;
}

function moduleLabel(h, value) {
  if (typeof h?.moduleLabel === "function") {
    return h.moduleLabel(value);
  }
  return firstText(value, "驱动因素");
}

function normalizeNarrativeToken(value) {
  return String(value || "")
    .trim()
    .replace(/^endogenous[:\s_-]*/i, "")
    .replace(/^endogenous\s+trigger[:\s_-]*/i, "")
    .replace(/^trigger[:\s_-]*/i, "")
    .replace(/^mode[:\s_-]*/i, "")
    .replace(/^state[:\s_-]*/i, "")
    .trim();
}

function tokenLabel(h, value) {
  const normalized = normalizeNarrativeToken(value);
  if (typeof h?.tokenLabel === "function") {
    return h.tokenLabel(normalized);
  }
  return humanizeRuntimeToken(normalized, normalized.replaceAll(/[_-]+/g, " ").trim());
}

function cueLabel(h, value) {
  const normalized = normalizeNarrativeToken(value);
  return firstText(tokenLabel(h, normalized), normalized, "这条线索");
}

function driverKey(item) {
  return String(item?.agent_name || item?.module_name || "").trim();
}

function driverNounPhrase(item) {
  switch (driverKey(item)) {
    case "EmergentActionSketch":
      return "一个还在成形的行动念头";
    case "GoalContinuationDrive":
      return "一股还没放下的目标牵引";
    case "AffectResidueDrive":
      return "一阵尚未退去的情绪余波";
    case "RelationBoundaryGate":
      return "一层正在收紧的边界感";
    default:
      return "一股仍在起作用的内部牵引";
  }
}

function hasResidualEnglishLeak(value) {
  const text = String(value || "");
  const englishWords = text.match(/[A-Za-z]{4,}/g) || [];
  const hanCount = (text.match(/[\u4e00-\u9fff]/g) || []).length;
  return englishWords.length >= 2 && englishWords.length * 3 >= Math.max(hanCount, 1);
}

function driverActionLead(h, item, selectedAction) {
  const action = h.actionLabel(item.action_name || selectedAction);
  const source = driverNounPhrase(item);
  const reason = readableNarrative(item.reason, "当前这股内部牵引仍在起作用，但后端还没有给出更适合前台阅读的中文说明。");
  if (reason) {
    return `动作倾向：${action}。来源：${source}。${reason}`;
  }
  return `动作倾向：${action}。来源：${source}。当前还没有额外的自由文本说明。`;
}

function readableNarrative(raw, fallback = "") {
  const next = humanizeRuntimeText(firstText(raw), fallback);
  if (!fallback && /[A-Za-z]{8,}/.test(next) && !/[\u4e00-\u9fff]/.test(next)) {
    return "";
  }
  if (fallback && hasResidualEnglishLeak(next)) {
    return fallback;
  }
  return next;
}

function memoryRecallBody(h, item) {
  const content = firstText(item.last_content, item.summary, item.gist);
  if (content) {
    return `最近记录：${humanizeRuntimeText(content)}`;
  }
  return `当前只保留“${cueLabel(h, item.cue)}”这条线索，还没有展开成更多正文。`;
}

function dreamFragmentBody(item) {
  const summary = firstText(item.summary, item.identity_summary);
  if (summary) {
    return humanizeRuntimeText(summary, "梦境片段已经记录，但当前还没有更适合前台阅读的中文摘要。");
  }
  return "当前只有梦境记录元数据，还没有自由文本摘要。";
}

function numericSignal(h, value) {
  const number = Number(value);
  if (!Number.isFinite(number)) {
    return h.t("common.none");
  }
  if (typeof h?.formatNumber === "function") {
    return h.formatNumber(number, 2);
  }
  return number.toFixed(2);
}

function scheduledTaskStatusLabel(h, value) {
  switch (String(value || "").trim()) {
    case "running":
      return h.t("scheduledTask.status.running");
    case "disabled":
      return h.t("scheduledTask.status.disabled");
    case "idle":
    default:
      return h.t("scheduledTask.status.idle");
  }
}

function scheduledTaskExcerpt(task) {
  const prompt = firstText(task.prompt);
  if (prompt) {
    return { labelKey: "scheduledTask.excerpt.prompt", text: humanizeRuntimeText(prompt, "计划任务会读取仓库变化并生成只读摘要。") };
  }
  const goal = firstText(task.task_payload?.goal_summary, task.task_payload?.goal, task.goal_summary, task.goal);
  if (goal) {
    return { labelKey: "scheduledTask.excerpt.goal", text: humanizeRuntimeText(goal, "当前任务目标已记录。") };
  }
  return { labelKey: "", text: "" };
}

export function deriveRoundInitiativeState(round = {}, detail = {}) {
  const trace = detail.trace || {};
  const initiativeWhy = detail.initiativeWhy || {};
  const initiative = initiativeWhy.initiative || trace.initiative || {};
  const topIntent = firstText(initiative.top_intent, initiative.topIntent);
  const suppressionReason = firstText(initiative.suppression_reason, initiative.suppressionReason);
  const shouldSend =
    typeof initiative.should_send === "boolean"
      ? initiative.should_send
      : typeof initiative.shouldSend === "boolean"
        ? initiative.shouldSend
        : null;
  const memoryBacking = initiative.memory_backing || initiative.memoryBacking || {};
  const topicRelevance = Number(memoryBacking.topic_relevance ?? memoryBacking.topicRelevance ?? 0);
  const hasEndogenousSignal =
    String(round?.cause_type || "").includes("endogenous") ||
    hasItems(trace.endogenous_tick_reason) ||
    hasItems(trace.endogenous_trigger_context) ||
    hasItems(trace.endogenous_policy_shift);

  if (
    topIntent &&
    topIntent !== "stay_silent" &&
    topIntent !== "nothing" &&
    (shouldSend === true || (!suppressionReason && topicRelevance >= 0.8))
  ) {
    return "initiative";
  }
  if (hasEndogenousSignal) {
    return "initiative";
  }
  if (String(round?.cause_type || "").includes("user") || String(round?.cause_type || "").includes("external")) {
    return "reactive";
  }
  if (initiativeWhy?.summary) {
    return "mixed";
  }
  return "unknown";
}

export function buildRoundInitiativeView(detail = {}, fallback = null) {
  const trace = detail.trace || {};
  const initiativeWhy = detail.initiativeWhy || {};
  const roundInitiative = initiativeWhy.initiative || trace.initiative || {};
  const effective = Object.keys(roundInitiative).length > 0 ? roundInitiative : (fallback || {});
  const memoryBacking = effective.memory_backing || effective.memoryBacking || {};
  return {
    proposal_type: firstText(effective.proposal_type, effective.proposalType, fallback?.proposal_type, fallback?.proposalType),
    top_intent: firstText(effective.top_intent, effective.topIntent, fallback?.top_intent, fallback?.topIntent),
    should_send:
      typeof effective.should_send === "boolean"
        ? effective.should_send
        : typeof effective.shouldSend === "boolean"
          ? effective.shouldSend
          : typeof fallback?.should_send === "boolean"
            ? fallback.should_send
            : typeof fallback?.shouldSend === "boolean"
              ? fallback.shouldSend
              : false,
    suppression_reason: firstText(
      effective.suppression_reason,
      effective.suppressionReason,
      fallback?.suppression_reason,
      fallback?.suppressionReason,
    ),
    memory_backing: memoryBacking,
    summary: firstText(initiativeWhy.summary, fallback?.summary),
  };
}

export function buildOverviewSnapshot(input, h) {
  const snapshot = input.runtimeState?.cognitive_snapshot || {};
  const runtimeState = input.runtimeState || {};
  const identity = input.runtimeState?.cognitive_snapshot?.identity || {};
  const subjectKernel = input.subject?.subject_kernel || {};
  const meaningSystem = input.meaning?.meaning_system || {};
  const agencyLoop = input.agency?.agency_loop || {};
  const proactiveBacklog = asArray(input.agency?.proactive_backlog);
  const currentRound = input.consoleData?.state?.current_round || {};
  const vitalSigns = snapshot.vital_signs || {};
  const tlh = snapshot.tlh || {};
  const bodyState = {
    ...(runtimeState.body_state || {}),
    ...(tlh.body_state || {}),
    fatigue: tlh.body_state?.fatigue ?? runtimeState.fatigue,
    self_continuity: tlh.body_state?.self_continuity ?? runtimeState.self_continuity,
    meaning_strength: tlh.body_state?.meaning_strength ?? runtimeState.meaning_strength,
  };
  const subjectiveState = {
    ...(runtimeState.subjective_state || {}),
    ...(tlh.subjective_state || {}),
  };
  const authenticity = snapshot.authenticity || {};
  const moodValue = Number(vitalSigns.mood ?? 0.55);
  const feltSummary = naturalList(subjectiveState.felt, "暂时没有特别挂在前台的主观感受");
  const moodSummary = typeof h?.moodLabel === "function" ? h.moodLabel(moodValue) : numericSignal(h, moodValue);
  const survivalNarrative = humanizeRuntimeText(firstText(meaningSystem.survival_narrative));
  const subjectNarrative = humanizeRuntimeText(firstText(subjectKernel.current_narrative, subjectKernel.current_self_narrative));
  const commitmentSummary = naturalList(asArray(subjectKernel.core_commitments).slice(0, 2), "当前还没有显性核心承诺浮到前台");
  const proactiveLead = humanizeRuntimeText(
    firstText(proactiveBacklog[0]?.title, proactiveBacklog[0]?.summary),
    firstText(proactiveBacklog[0]?.title, proactiveBacklog[0]?.summary),
  );
  const agencyLoopSummary = humanizeRuntimeText(firstText(agencyLoop.summary), firstText(agencyLoop.summary));
  const agencySummary = firstText(
    agencyLoopSummary && proactiveLead ? `${agencyLoopSummary} 当前最靠前的主动事项是：${proactiveLead}` : "",
    agencyLoopSummary,
    proactiveLead ? `当前最靠前的主动事项是：${proactiveLead}` : "",
  );
  const hasPlan16Subject = hasItems(subjectKernel);
  const hasPlan16Meaning = hasItems(meaningSystem);
  const hasPlan16Agency = hasItems(agencyLoop) || proactiveBacklog.length > 0;
  const currentIntent = firstText(
    input.runtimeState?.cognitive_snapshot?.current_intent,
    input.runtimeState?.run?.current_step?.title,
    input.runtimeState?.run?.goal_summary,
    survivalNarrative,
    agencySummary,
  );
  const identityName = firstText(subjectKernel.display_name, identity.display_name, input.settings?.identity?.display_name, "当前运行体");
  const continuity = firstText(subjectNarrative, identity.continuity, input.runtimeState?.state?.brain_state?.self_continuity, "仍在形成稳定的自我称呼");
  const mode = firstText(currentRound.mode_label, currentRound.mode, input.runtimeState?.state?.brain_state?.mode, h.t("common.unknown"));
  const action = currentRound.sampled_action ? h.actionLabel(currentRound.sampled_action) : h.t("common.noneRound");
  const serviceReady = input.service?.healthy ? h.t("common.healthy") : h.t("common.degraded");
  const autonomyState = input.autonomy?.running ? h.t("common.running") : h.t("common.stopped");
  const healthSignals = [
    { label: "服务", value: serviceReady, tone: input.service?.healthy ? "healthy" : "degraded" },
    { label: "自治", value: autonomyState, tone: input.autonomy?.running ? "healthy" : "soft" },
    { label: "会话", value: input.bootstrap?.session_attached ? h.t("common.attached") : h.t("common.detached"), tone: input.bootstrap?.session_attached ? "healthy" : "soft" },
    { label: "当前动作", value: action, tone: "focus" },
    {
      label: "模型",
      value: `${humanizeRuntimeToken(input.models?.module_model_bindings?.cognitive_packet || "--")} / ${humanizeRuntimeToken(input.models?.module_model_bindings?.deep_renderer || "--")}`,
      tone: "soft",
    },
  ];
  const personaMoodSignals = [
    { label: "主观感受", value: feltSummary },
    { label: "当前心境", value: moodSummary },
    ...(hasPlan16Subject ? [{ label: "主体承诺", value: commitmentSummary }] : []),
    ...(hasPlan16Meaning ? [{ label: "意义叙事", value: firstText(survivalNarrative, "当前还没有显性生存叙事") }] : []),
    ...(hasPlan16Agency ? [{ label: "主动节奏", value: firstText(agencySummary, "当前没有显性主动动作排队") }] : []),
    { label: "边界感", value: numericSignal(h, subjectiveState.boundary) },
    { label: "自发性", value: numericSignal(h, subjectiveState.spontaneous) },
    { label: "疲劳", value: numericSignal(h, bodyState.fatigue) },
    { label: "连续性", value: numericSignal(h, bodyState.self_continuity) },
    { label: "意义感", value: numericSignal(h, bodyState.meaning_strength) },
    { label: "真实感", value: firstText(authenticity.summary, "还没有足够证据判断这轮真实感") },
  ];
  return {
    identity_intro: `我是${identityName}。我现在保持在 ${mode} 状态，${continuity}。`,
    current_state_summary: currentRound.round_id
      ? `我这会儿正在处理第 ${currentRound.round_id} 轮，更明显的外显动作是${action}。服务目前${serviceReady}，自治${autonomyState}。`
      : `我已经在线，但暂时还没有形成可供细读的一轮行动记录。当前最先要确认的是服务、自治和会话是否都已经进入稳定状态。`,
    current_focus: currentIntent
      ? `我眼下最集中的事是：${humanizeRuntimeText(currentIntent)}。`
      : "我现在还没有固定住一个足够明确的当前焦点。",
    persona_mood_title: "人格心情",
    persona_mood_summary: `我现在更接近${moodSummary}。主观感受里还挂着${feltSummary}。${survivalNarrative ? `当前意义线索是：${survivalNarrative}。` : ""}${agencySummary ? `主动节奏上，${agencySummary}。` : ""}边界感${numericSignal(
      h,
      subjectiveState.boundary,
    )}，自发性${numericSignal(h, subjectiveState.spontaneous)}，疲劳${numericSignal(h, bodyState.fatigue)}。`,
    persona_mood_signals: personaMoodSignals,
    health_signals: healthSignals,
    quick_actions: [
      { id: "runtime-start", label: h.t("runtime.start"), enabled: true },
      { id: "runtime-pause", label: h.t("runtime.pause"), enabled: true },
      { id: "runtime-resume", label: h.t("runtime.resume"), enabled: true },
      { id: "runtime-wake", label: h.t("runtime.wake"), enabled: true },
    ],
  };
}

function tracePayload(input) {
  if (input?.trace && typeof input.trace === "object") {
    return input.trace;
  }
  return input && typeof input === "object" ? input : {};
}

function finiteNumber(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function clampUnit(value) {
  const number = finiteNumber(value);
  if (number === null) {
    return null;
  }
  return Math.max(0, Math.min(1, Number(number.toFixed(4))));
}

function formatMetricSource(key) {
  switch (key) {
    case "u_base":
      return "action_bookkeeping.u_base";
    case "p_final":
      return "probability_field.action.winner_posterior";
    case "strength":
      return "action_bookkeeping.ci";
    default:
      return "";
  }
}

const ANALYSIS_METRIC_META = {
  u_base: {
    displayLabel: "基础驱动力",
    displaySubLabel: "动作原始驱动力",
    displaySource: "动作原始驱动力",
  },
  p_final: {
    displayLabel: "最终胜出概率",
    displaySubLabel: "概率场最终裁决",
    displaySource: "概率场最终裁决",
  },
  strength: {
    displayLabel: "习惯牵引强度",
    displaySubLabel: "习惯 / 记忆牵引",
    displaySource: "习惯 / 记忆牵引",
  },
};

function metricMeta(key) {
  return (
    ANALYSIS_METRIC_META[key] || {
      displayLabel: key,
      displaySubLabel: key,
      displaySource: "",
    }
  );
}

function humanizeTrendNarrative(text) {
  const source = firstText(text);
  if (!source) {
    return "";
  }
  const normalized = source.trim().toLowerCase();
  const exactMap = {
    "authenticity guard requested resample": "身份校验要求重新采样。",
    "authenticity guard resample": "身份校验已触发重新采样。",
    "authenticity guard fallback": "身份校验要求回退输出。",
    "repair transition": "修复流程已切换阶段。",
    "critical conflict": "检测到显性冲突。",
    "gate triggered": "门控已触发。",
  };
  if (exactMap[normalized]) {
    return exactMap[normalized];
  }
  const containsMap = [
    ["risk_gate_closed", "风险门控已关闭。"],
    ["route_gate_closed", "路径门控已关闭。"],
    ["no_target", "当前没有可用目标。"],
    ["perspective_disabled", "视角推演当前未启用。"],
    ["resource_starvation", "当前资源不足。"],
    ["mode=silent", "当前处于静默模式。"],
  ];
  const matched = containsMap.find(([token]) => normalized.includes(token));
  if (matched) {
    return matched[1];
  }
  return humanizeRuntimeText(source);
}

function mapTrendTimelineMarkers(roundId, markerView = {}) {
  const markers = [];
  asArray(markerView.conflictMarkers).forEach((marker, index) => {
    markers.push({
      id: `round-${roundId}-conflict-${index + 1}`,
      roundId,
      track: "conflict",
      stageKey: marker.stageKey,
      stageLabel: stageKeyFromTelemetry(marker.stageKey, marker.label),
      label: marker.label,
      detail: marker.detail,
      tone: marker.tone,
    });
  });
  asArray(markerView.repairMarkers).forEach((marker, index) => {
    markers.push({
      id: `round-${roundId}-repair-${index + 1}`,
      roundId,
      track: "repair",
      stageKey: marker.stageKey,
      stageLabel: stageKeyFromTelemetry(marker.stageKey, marker.label),
      label: marker.label,
      detail: marker.detail,
      tone: marker.tone,
    });
  });
  return markers;
}
function stageKeyFromTelemetry(stage, owner = "") {
  const token = `${String(stage || "")}:${String(owner || "")}`.toLowerCase();
  if (token.includes("memory") || token.includes("cue") || token.includes("hippocampus")) {
    return "memory";
  }
  if (token.includes("pfc")) {
    return "pfc";
  }
  if (token.includes("thalamus")) {
    return "thalamus";
  }
  if (token.includes("auth") || token.includes("identity") || token.includes("plausibility") || token.includes("output_gate")) {
    return "identity_guard";
  }
  if (token.includes("renderer") || token.includes("render") || token.includes("output")) {
    return "output";
  }
  if (token.includes("repair") || token.includes("resample") || token.includes("conflict") || token.includes("forced_mode_switch")) {
    return "conflict_repair";
  }
  return "conflict_repair";
}

function latestRepairReasons(conflict = {}) {
  return asArray(conflict.repair_ledger_tail || conflict.repairLedgerTail)
    .map((item) => firstText(item.reason, item.label, item.stage))
    .filter(Boolean)
    .slice(0, 3);
}

function gateMarkerTone(gate = {}) {
  if (gate?.requires_resample) {
    return "conflict";
  }
  if (gate?.allowed !== false) {
    return "";
  }
  const owner = String(gate?.owner || "").trim().toLowerCase();
  const stage = String(gate?.stage || "").trim().toLowerCase();
  const reason = firstText(gate?.reason, gate?.summary).toLowerCase();
  if (
    owner === "perspectivemodel" ||
    owner === "initiativeruntime" ||
    stage === "late_perspective" ||
    stage === "expressive_field" ||
    reason.includes("risk_gate_closed") ||
    reason.includes("route_gate_closed") ||
    reason.includes("no_target") ||
    reason.includes("perspective_disabled") ||
    reason.includes("resource_starvation") ||
    reason.includes("mode=silent")
  ) {
    return "skipped";
  }
  return "blocked";
}

function repairStageLabel(stage) {
  const key = String(stage || "").trim().toLowerCase();
  return {
    adjusting: "调整中",
    repairing: "修复中",
    cooling: "冷却中",
    recovered: "已恢复",
    resampling: "重新采样",
  }[key] || firstText(stage, "处理中");
}

function shouldDisplayRepairTransition(stage) {
  return new Set(["adjusting", "repairing", "cooling", "recovered", "resampling"]).has(
    String(stage || "").trim().toLowerCase(),
  );
}

function uniqueMarkers(markers) {
  const seen = new Set();
  return markers.filter((marker) => {
    const signature = [marker.stageKey, marker.label, marker.detail, marker.tone].join("|");
    if (seen.has(signature)) {
      return false;
    }
    seen.add(signature);
    return true;
  });
}

export function deriveHabitStrength(input = {}) {
  const trace = tracePayload(input);
  const ci = trace.action_bookkeeping?.ci;
  if (ci && typeof ci === "object") {
    const values = Object.values(ci).map(finiteNumber).filter((item) => item !== null);
    if (values.length) {
      const average = values.reduce((sum, item) => sum + item, 0) / values.length;
      return clampUnit(1.0 - average / 0.55);
    }
  }
  const memoryBacking = trace.initiative?.memory_backing || trace.initiative?.memoryBacking || {};
  return clampUnit(memoryBacking.strength);
}

export function buildAnalysisTrendSeries({ recentRounds = [], roundCatalog = {} } = {}) {
  const orderedRounds = asArray(recentRounds)
    .filter((round) => Number.isFinite(Number(round?.round_id)))
    .slice()
    .sort((a, b) => Number(a.round_id) - Number(b.round_id));
  const series = ["u_base", "p_final", "strength"].map((key) => ({
    key,
    ...metricMeta(key),
    points: [],
  }));

  orderedRounds.forEach((round) => {
    const roundId = Number(round.round_id);
    const detail = roundCatalog[String(roundId)] || {};
    const trace = tracePayload(detail);
    const sampledAction = firstText(trace.sampled_action, round.sampled_action);
    const uBase = finiteNumber(trace.action_bookkeeping?.u_base?.[sampledAction]);
    const pFinal =
      finiteNumber(trace.probability_field?.action?.winner_posterior?.[sampledAction]) ??
      finiteNumber(trace.candidate_distribution?.[sampledAction]);
    const ci = trace.action_bookkeeping?.ci;
    const hasCiSource =
      ci && typeof ci === "object" && Object.keys(ci).length > 0;
    const strength = deriveHabitStrength(trace);
    const markerView = buildConflictRepairMarkers({ roundId, selectedPayload: detail });
    const timelineMarkers = mapTrendTimelineMarkers(roundId, markerView);
    const summary = firstText(
      trace.rendered_expression?.text,
      markerView.conflictMarkers[0]?.detail,
      markerView.repairMarkers[0]?.detail,
      "这一轮没有额外的自由文本说明。",
    );
    const basePoint = {
      roundId,
      action: sampledAction,
      hasConflict: markerView.conflictMarkers.length > 0,
      hasBlocked: markerView.blockedMarkers.length > 0,
      hasSkipped: markerView.skippedMarkers.length > 0,
      hasRepair: markerView.repairMarkers.length > 0,
      timelineMarkers,
      summary: humanizeTrendNarrative(summary),
    };

    series[0].points.push({
      ...basePoint,
      value: uBase,
      source: formatMetricSource("u_base"),
      displaySource: metricMeta("u_base").displaySource,
    });
    series[1].points.push({
      ...basePoint,
      value: pFinal,
      source: formatMetricSource("p_final"),
      displaySource: metricMeta("p_final").displaySource,
    });
    series[2].points.push({
      ...basePoint,
      value: strength,
      source: hasCiSource ? formatMetricSource("strength") : "initiative.memory_backing.strength",
      displaySource: metricMeta("strength").displaySource,
    });
  });

  return series;
}

export function buildAnalysisReplayFrames(trendSeries = []) {
  const baseSeries = trendSeries[0]?.points || [];
  return baseSeries.map((point, index) => ({
    frameId: `round-${point.roundId}`,
    roundId: point.roundId,
    roundLabel: `第 ${point.roundId} 轮`,
    action: point.action,
    summary: point.summary,
    markers: point.timelineMarkers || [],
    metrics: trendSeries.map((series) => ({
      key: series.key,
      displayLabel: series.displayLabel,
      displaySubLabel: series.displaySubLabel,
      value: series.points[index]?.value ?? null,
    })),
  }));
}

export function buildConflictRepairMarkers({ roundId = null, selectedPayload = {} } = {}) {
  const trace = tracePayload(selectedPayload);
  const conflict = trace.conflict_arbitration || {};
  const conflictMarkers = [];
  const blockedMarkers = [];
  const skippedMarkers = [];
  const repairMarkers = [];

  const pushMarker = (bucket, tone, stageKey, label, detail) => {
    const safeLabel = firstText(label);
    const safeDetail = humanizeTrendNarrative(firstText(detail));
    if (!safeLabel && !safeDetail) {
      return;
    }
    bucket.push({
      id: `${tone}-${stageKey}-${bucket.length + 1}`,
      roundId,
      stageKey,
      label: safeLabel || safeDetail,
      detail: safeDetail || safeLabel,
      tone,
    });
  };

  asArray(trace.gate_decisions).forEach((gate) => {
    const stageKey = stageKeyFromTelemetry(gate?.stage, gate?.owner);
    const ownerRaw = firstText(gate?.owner, gate?.stage, "guard");
    const owner = humanizeRuntimeToken(ownerRaw, ownerRaw);
    const reason = firstText(gate?.reason, gate?.summary, "gate triggered");
    const tone = gateMarkerTone(gate);
    if (tone === "conflict") {
      pushMarker(conflictMarkers, "conflict", stageKey, `冲突 · ${owner}`, reason);
    } else if (tone === "blocked") {
      pushMarker(blockedMarkers, "blocked", stageKey, `阻断 · ${owner}`, reason);
    } else if (tone === "skipped") {
      pushMarker(skippedMarkers, "skipped", stageKey, `跳过 · ${owner}`, reason);
    }
    if (gate?.requires_resample) {
      pushMarker(repairMarkers, "repair", "conflict_repair", "修复 · 重新采样", reason);
    }
  });

  if (conflict.critical_conflict) {
    pushMarker(
      conflictMarkers,
      "conflict",
      "conflict_repair",
      "冲突升高",
      firstText(
        naturalList(conflict.dominant_conflicts, ""),
        `critical conflict`,
      ),
    );
  }

  const guardAction = firstText(trace.authenticity?.guard_action, trace.renderer_decision_integrity?.auth_guard_action);
  if (guardAction === "resample") {
    pushMarker(conflictMarkers, "conflict", "identity_guard", "冲突 · 身份校验重采样", "authenticity guard requested resample");
    pushMarker(repairMarkers, "repair", "conflict_repair", "修复 · 重采样输出", "authenticity guard resample");
  } else if (guardAction === "fallback") {
    pushMarker(repairMarkers, "repair", "conflict_repair", "修复 · 回退输出", "authenticity guard fallback");
  }

  const repairTransition = conflict.repair_transition || conflict.repairTransition || {};
  if (repairTransition && typeof repairTransition === "object" && Object.keys(repairTransition).length > 0) {
    const repairStage = firstText(repairTransition.to_stage, repairTransition.toStage);
    if (shouldDisplayRepairTransition(repairStage)) {
      pushMarker(
        repairMarkers,
        "repair",
        "conflict_repair",
        `修复 · ${repairStageLabel(repairStage)}`,
        firstText(repairTransition.reason, "repair transition"),
      );
    }
  }

  latestRepairReasons(conflict).forEach((reason) => {
    pushMarker(repairMarkers, "repair", "conflict_repair", "修复 · 处理动作", reason);
  });

  return {
    conflictMarkers: uniqueMarkers(conflictMarkers),
    blockedMarkers: uniqueMarkers(blockedMarkers),
    skippedMarkers: uniqueMarkers(skippedMarkers),
    repairMarkers: uniqueMarkers(repairMarkers),
  };
}

export function buildBrainflowStages(input = {}) {
  const detail = input.selectedPayload || {};
  const trace = tracePayload(detail);
  const selectedAction = firstText(input.selectedAction, trace.sampled_action, "respond");
  const selectedActionLabel = humanizeRuntimeToken(selectedAction, selectedAction);
  const initiativeView = buildRoundInitiativeView(detail, null);
  const memoryBacking = initiativeView.memory_backing || {};
  const pfcProposal = asArray(trace.proposal_summaries).find(
    (item) => stageKeyFromTelemetry(item?.stage, item?.agent_name) === "pfc",
  );
  const thalamusProposal = asArray(trace.proposal_summaries).find(
    (item) => stageKeyFromTelemetry(item?.stage, item?.agent_name) === "thalamus",
  );
  const conflict = trace.conflict_arbitration || {};
  const repairTransition = conflict.repair_transition || conflict.repairTransition || {};
  const renderPlan = trace.render_plan || {};
  const renderedExpression = trace.rendered_expression || {};
  const posterior = finiteNumber(trace.probability_field?.action?.winner_posterior?.[selectedAction]);
  const derivedStrength = deriveHabitStrength(trace);
  const gateDecisions = asArray(trace.gate_decisions);
  const relevantGuards = gateDecisions.filter((item) => stageKeyFromTelemetry(item?.stage, item?.owner) === "identity_guard");
  const allowedGuards = relevantGuards.filter((item) => item?.allowed !== false).length;
  const identityContext = renderPlan.identity_context || {};
  const repairReasons = latestRepairReasons(conflict);

  return [
    {
      key: "memory",
      title: firstText(memoryBacking.cue) ? `调取记忆 · ${humanizeRuntimeText(memoryBacking.cue)}` : "调取记忆",
      body: readableNarrative(
        memoryBacking.summary,
        memoryBacking.cue ? `先把“${humanizeRuntimeText(memoryBacking.cue)}”这条记忆线索拉到前台。` : "当前还没有显性记忆线索被拉到前台。",
      ),
      meta: `牵引强度 ${derivedStrength !== null ? derivedStrength.toFixed(2) : "--"} · ${humanizeRuntimeToken(firstText(memoryBacking.topic_source, "记忆来源"))}`,
    },
    {
      key: "pfc",
      title: "前额叶候选生成",
      body: readableNarrative(
        firstText(pfcProposal?.reason, pfcProposal?.summary),
        `前额叶控制层先把 ${humanizeRuntimeToken(selectedAction)} 保留为当前候选。`,
      ),
      meta: `前额叶控制层 · ${humanizeRuntimeToken(firstText(pfcProposal?.top_action, selectedAction))}`,
    },
    {
      key: "thalamus",
      title: "丘脑注意采样",
      body: readableNarrative(
        firstText(thalamusProposal?.reason, thalamusProposal?.summary),
        posterior !== null ? `注意采样后，${selectedActionLabel} 的最终后验稳定在 ${posterior.toFixed(2)}。` : "当前没有独立的丘脑采样说明。",
      ),
      meta: `注意采样层 · 最终胜出概率 ${posterior !== null ? posterior.toFixed(2) : "--"}`,
    },
    {
      key: "identity_guard",
      title: "守卫层身份校验",
      body: readableNarrative(
        identityContext.display_label ? `以 ${identityContext.display_label} 的身份语境继续校验当前表达。` : "",
        "守卫层正在检查身份一致性、表达边界和可输出性。",
      ),
      meta: `${humanizeRuntimeText(firstText(identityContext.display_label, "当前身份语境"))} · ${humanizeRuntimeToken(firstText(identityContext.query_kind, "守卫校验"))} · 放行 ${allowedGuards}/${Math.max(relevantGuards.length, 1)}`,
    },
    {
      key: "conflict_repair",
      title: "冲突修复",
      body: readableNarrative(
        firstText(repairReasons[0], repairTransition.reason),
        conflict.critical_conflict ? "这一轮存在显性冲突，需要进入修复阶段。" : "没有触发显性冲突，修复层维持监视。",
      ),
      meta: `阶段 ${humanizeRuntimeToken(firstText(conflict.repair_state_snapshot?.stage, repairTransition.to_stage, "空闲"))} · 显性冲突 ${conflict.critical_conflict ? "是" : "否"}`,
    },
    {
      key: "output",
      title: "输出话术",
      body: readableNarrative(renderedExpression.text, "当前还没有形成可输出的话术。"),
      meta: `输出路径 ${humanizeRouteLabel(firstText(renderedExpression.route, "表达层"))} · 输出方式 ${humanizeRouteLabel(firstText(renderedExpression.delivery_mode, "文字"))}`,
    },
  ];
}

export function buildBrainflowOutput(input = {}) {
  const detail = input.selectedPayload || {};
  const trace = tracePayload(detail);
  const renderedExpression = trace.rendered_expression || {};
  const renderPlan = trace.render_plan || {};
  const identityContext = renderPlan.identity_context || {};
  const route = humanizeRouteLabel(firstText(renderedExpression.route, "renderer"));
  const deliveryMode = humanizeRouteLabel(firstText(renderedExpression.delivery_mode, "speech"));
  const modelValue = firstText(
    renderedExpression.model,
    identityContext.model_label,
    identityContext.provider_label,
    "",
  );
  return {
    title: firstText(identityContext.display_label, "最终外显表达"),
    text: humanizeRuntimeText(firstText(renderedExpression.text), "当前没有可展示的输出话术。"),
    route,
    delivery_mode: deliveryMode,
    model: modelValue ? humanizeRuntimeToken(modelValue, modelValue) : "--",
    summary: `通过 ${route} 路径，以 ${deliveryMode} 方式输出。`,
  };
}

export function buildAnalysisFlowView(input, h) {
  const thought = input.selectedPayload?.thought || {};
  const why = input.selectedPayload?.why || {};
  const whyNot = input.selectedPayload?.whyNot || {};
  const trace = input.selectedPayload?.trace || {};
  const replay = input.selectedPayload?.replay || {};
  const currentAction = firstText(
    thought.sampled_action,
    trace.sampled_action,
    input.selectedAction,
  );
  const decisionBody = firstText(
    readableNarrative(why?.why?.summary),
    readableNarrative(thought.rendered_expression?.text),
    readableNarrative(trace.rendered_expression?.text),
    h.t("summary.noWhy"),
  );
  const route = firstText(
    thought.rendered_expression?.route,
    trace.rendered_expression?.route,
    input.selectedRound?.mode,
    "renderer",
  );
  const delivery = firstText(
    thought.rendered_expression?.delivery_mode,
    trace.rendered_expression?.delivery_mode,
    h.t("summary.deliveryUnknown"),
  );
  const decisionSummary = `我这轮先选择了${h.actionLabel(currentAction)}。${decisionBody}`;
  const routeStory = `这条决策是沿着 ${h.routeTypeLabel(input.selectedRound?.mode || input.selectedRound?.cause_type || trace.mode || "interactive")} 进入的，最后通过 ${humanizeRouteLabel(route)} 输出，并以 ${humanizeRouteLabel(delivery)} 的方式对外表达。`;
  const probabilitySummary = topPosteriorRows(thought.action_field || input.selectedPayload?.probability?.probability_field?.action || {})
    .map(([action, value], index) => ({
      label: h.actionLabel(action),
      value: Number(value || 0),
      delta: index === 0 ? "主峰" : "",
    }));
  const topDrivers = asArray(thought.top_drivers || why?.why?.top_drivers).slice(0, 4);
  const evidenceCards = [
    ...topDrivers.map((item) => ({
      title: `${moduleLabel(h, item.agent_name || item.module_name || "驱动因素")} 推着我靠近 ${h.actionLabel(item.action_name || currentAction)}`,
      body: driverActionLead(h, item, currentAction),
      type: "driver",
      refs: [deepLink(h.t("deepLink.round"), buildWorkbenchRoundPath(input.roundId))],
    })),
  ];
  if (firstText(whyNot.summary, whyNot.why_not?.summary)) {
    evidenceCards.push({
      title: "被压下去的路径",
      body: readableNarrative(firstText(whyNot.summary, whyNot.why_not?.summary), "这条路径在本轮被压到了后面。"),
      type: "why-not",
      refs: [deepLink(h.t("deepLink.round"), buildWorkbenchRoundPath(input.roundId))],
    });
  }
  const timelineSource = asArray(thought.timeline?.events || trace.timeline || replay.timeline || []);
  const timelineEvents = timelineSource.slice(0, 8).map((item, index) => ({
    ts: String(index + 1),
    title: humanizeRuntimeText(firstText(item.label, item.type, `阶段 ${index + 1}`)),
    body: readableNarrative(firstText(item.summary, item.reason, item.stage), "这一段没有额外描述。"),
    refs: [deepLink(h.t("deepLink.round"), buildWorkbenchRoundPath(input.roundId))],
  }));
  const rawLinks = [
    deepLink(h.t("deepLink.round"), buildWorkbenchRoundPath(input.roundId)),
  ];
  return {
    decision_summary: decisionSummary,
    route_story: routeStory,
    timeline_events: timelineEvents,
    probability_summary: probabilitySummary,
    evidence_cards: evidenceCards,
    raw_links: rawLinks,
  };
}

export function buildInnerSpaceView(input, h) {
  const thought = input.thought || {};
  const topDrivers = asArray(thought.top_drivers).slice(0, 4);
  const memoryTop = asArray(input.memoryTop);
  const dreamRuns = asArray(input.dreamOverview?.recent_runs);
  const monologueFragments = asArray(input.monologueShow?.fragments);
  const selectedAction = firstText(thought.sampled_action, input.consoleData?.state?.current_round?.sampled_action, "respond");
  const leadingMemory = memoryTop[0];
  const dreamEnabled = Boolean(input.dreamOverview?.status?.enabled);
  const leadingMemoryLabel = leadingMemory?.cue ? cueLabel(h, leadingMemory.cue) : "";
  const selfNarrative = [
    `我现在更明显地想朝${h.actionLabel(selectedAction)}这个方向靠近。`,
    topDrivers[0] ? `最先把我往前推的，是${driverNounPhrase(topDrivers[0])}。` : "这一刻还没有形成特别强的显性驱动说明。",
    leadingMemoryLabel ? `记忆里最亮的一条线索还是“${leadingMemoryLabel}”。` : "眼下没有特别突出的回忆片段浮上来。",
    dreamEnabled
      ? dreamRuns.length
        ? "梦境整理里已经出现了新的片段，我可以把它们和当前行动放在一起读。"
        : "梦境整理是开着的，但这一轮之后还没有出现新的梦境片段。"
      : "当前梦境整理未开启，因此这里没有新的梦境片段。",
  ].join("");
  const thinkingStream = topDrivers.length
    ? topDrivers.map((item, index) => ({
        title: index === 0 ? "我现在最靠前的想法" : `继续往后的想法 ${index + 1}`,
        body: driverActionLead(h, item, selectedAction),
        refs: input.roundId ? [deepLink(h.t("deepLink.round"), buildWorkbenchRoundPath(input.roundId))] : [],
      }))
    : [
        {
          title: "我现在的思考还很松散",
          body: "这一刻我还没有把内部推演稳定成一串足够清楚的思考链，因此这里只显示一个温和的空态。",
          refs: input.roundId ? [deepLink(h.t("deepLink.round"), buildWorkbenchRoundPath(input.roundId))] : [],
        },
      ];
  if (monologueFragments.length) {
    thinkingStream.push(
      ...monologueFragments.slice(0, 2).map((item, index) => ({
        title: `未说出口的独白 ${index + 1}`,
        body: firstText(item.text, item.summary, "这里有一段未完全外显的独白。"),
        refs: [innerSpaceDeepLink("内在空间")],
      })),
    );
  }
  const memoryRecall = memoryTop.length
    ? memoryTop.slice(0, 6).map((item) => ({
        title: `我想起了“${cueLabel(h, item.cue)}”`,
        body: memoryRecallBody(h, item),
        emotional_weight: Number(item.detail_strength || item.gist_strength || 0),
        refs: [
          innerSpaceDeepLink("记忆细节"),
          deepLink(h.t("deepLink.round"), buildWorkbenchRoundPath(item.last_recalled_round || input.roundId || 0)),
        ],
      }))
    : [
        {
          title: "我暂时没有新的回忆上浮",
          body: "当前没有特别强的热点记忆被拉到前台，所以这里保持安静。",
          emotional_weight: 0,
          refs: [innerSpaceDeepLink("内在空间")],
        },
      ];
  const dreamFragments = dreamRuns.length
    ? dreamRuns.slice(0, 6).map((item) => ({
        title: tokenLabel(h, firstText(item.title, item.mode, item.trigger, "梦境片段")),
        body: dreamFragmentBody(item),
        symbolic_tags: asArray(item.symbolic_tags || item.tags || []),
        refs: [innerSpaceDeepLink("梦境片段")],
      }))
    : [
        {
          title: dreamEnabled ? "还没有新的梦境片段" : "梦境整理当前未开启",
          body: dreamEnabled
            ? "梦境整理虽然已开启，但目前还没有可展示的新片段。"
            : "当前梦境整理未开启，因此不会出现新的梦境片段。",
          symbolic_tags: [],
          refs: [innerSpaceDeepLink("内在空间")],
        },
      ];
  const recentRounds = asArray(input.consoleData?.recent_rounds).slice(0, 4);
  const crossLinksToRounds = recentRounds.map((item) => ({
    round_id: item.round_id,
    label: `回到第 ${item.round_id} 轮，重看 ${h.actionLabel(item.sampled_action)}`,
  }));
  return {
    self_narrative: selfNarrative,
    thinking_stream: thinkingStream,
    memory_recall: memoryRecall,
    dream_fragments: dreamFragments,
    cross_links_to_rounds: crossLinksToRounds,
  };
}

export function buildChatTurnView(input, h) {
  const memoryCue = input.innerSpace?.memoryTop?.[0]?.cue || "";
  const reasonSummary = firstText(
    input.selectedPayload?.why?.why?.summary,
    input.selectedPayload?.thought?.thought_summary?.why_summary,
  );
  return {
    assistant_message: input.assistantText || "",
    implicit_reason_summary: reasonSummary ? `我之所以这样回答，是因为当前更倾向于先 ${h.actionLabel(input.selectedAction || input.selectedPayload?.thought?.sampled_action || "respond")}，再把其他路径压到后面。${reasonSummary}` : null,
    contextual_memory_hint: memoryCue ? `这段回应受“${memoryCue}”这条回忆影响。` : null,
    expandable_evidence_refs: input.roundId
      ? [deepLink(h.t("deepLink.round"), buildWorkbenchRoundPath(input.roundId)), innerSpaceDeepLink("内在空间")]
      : [],
  };
}

export function buildSettingsConsoleView(input, h) {
  const settings = input.settings || {};
  const bindings = settings.models?.module_model_bindings || {};
  const autonomy = settings.autonomy || {};
  const scheduledTasks = asArray(input.agency?.scheduled_tasks);
  const runningTasks = scheduledTasks.filter((item) => item.status === "running");
  const nextScheduledTask = scheduledTasks
    .filter((item) => firstText(item.next_run_at))
    .sort((left, right) => String(left.next_run_at).localeCompare(String(right.next_run_at)))[0];
  const scheduledTaskSurface = {
    summary: [
      { label: h.t("scheduledTask.summary.total"), value: String(scheduledTasks.length) },
      { label: h.t("scheduledTask.summary.running"), value: String(runningTasks.length) },
      { label: h.t("scheduledTask.summary.nextRun"), value: firstText(nextScheduledTask?.next_run_at, h.t("common.none")) },
    ],
    tasks: scheduledTasks.map((item) => {
      const excerpt = scheduledTaskExcerpt(item);
      return {
        task_id: firstText(item.task_id, h.t("common.none")),
        skill_name: firstText(item.skill_name, h.t("common.none")),
        status: firstText(item.status, "idle"),
        status_label: scheduledTaskStatusLabel(h, item.status),
        next_run_at: firstText(item.next_run_at, h.t("common.none")),
        excerpt: excerpt.text,
        excerpt_label: excerpt.labelKey ? h.t(excerpt.labelKey) : "",
      };
    }),
  };
  const latency = input.performance?.latency || {};
  const diagnostics = [
    {
      title: "运行诊断",
      rows: [
        { label: "服务状态", value: input.service?.healthy ? h.t("common.healthy") : h.t("common.degraded") },
        { label: "自治状态", value: input.autonomy?.running ? h.t("common.running") : h.t("common.stopped") },
        { label: "当前轮次", value: input.consoleData?.state?.current_round?.round_id ? `#${input.consoleData.state.current_round.round_id}` : h.t("common.noneRound") },
        { label: "过程存储", value: firstText(input.runtimeState?.state?.long_run?.trace_storage?.storage_state, "unknown") },
      ],
    },
    {
      title: "证据接口",
      rows: [
        { label: "主读接口", value: WORKBENCH_READ_MODEL_PATH },
        { label: "轮次详情", value: input.selectedRoundId ? buildWorkbenchRoundPath(input.selectedRoundId) : "/workbench/round/{id}" },
        { label: "内在空间", value: INNER_SPACE_READ_MODEL_PATH },
        { label: "主体接口", value: "/subject/status" },
      ],
    },
    {
      title: "热路径",
      rows: [
        { label: "耗时摘要", value: firstText(latency.latency_summary, h.t("common.none")) },
        { label: "总耗时", value: `${Number(latency.total_turn_ms || 0)} ms` },
        { label: "模型等待", value: `${Number(latency.model_wait_ms || 0)} ms` },
        { label: "接口", value: "/performance/hot-path" },
      ],
    },
    {
      title: "计划任务",
      rows: [
        { label: "任务状态", value: `${scheduledTasks.length} 个任务 / ${runningTasks.length} 个运行中` },
        { label: "最近下一次", value: firstText(nextScheduledTask?.next_run_at, h.t("common.none")) },
        { label: "接口", value: "/scheduled-tasks" },
      ],
    },
  ];
  return {
    runtime_bindings: [
      { label: "认知包", value: bindings.cognitive_packet || "--" },
      { label: "深度推理", value: bindings.deliberation || "--" },
      { label: "工具规划", value: bindings.tool_planner || "--" },
      { label: "深度表达", value: bindings.deep_renderer || "--" },
      { label: "巩固摘要", value: bindings.consolidation_summarizer || "--" },
    ],
    memory_dream_controls: [
      { label: "学习模式", value: firstText(autonomy.learning_mode, h.t("common.none")) },
      { label: "学习追踪", value: autonomy.trace_external_learning ? h.t("common.open") : h.t("common.closed") },
      { label: "梦境整理", value: input.runtimeState?.state?.long_run?.dream?.enabled ? h.t("common.open") : h.t("common.closed") },
      { label: "知识根目录", value: naturalList(autonomy.knowledge_roots, h.t("common.none")) },
    ],
    autonomy_controls: [
      { label: "允许网络", value: autonomy.network_enabled ? h.t("common.open") : h.t("common.closed") },
      { label: "允许外部 IO", value: autonomy.external_io_enabled ? h.t("common.open") : h.t("common.closed") },
      { label: "允许提交", value: autonomy.allow_commit ? h.t("common.open") : h.t("common.closed") },
      { label: "失败熔断阈值", value: String(autonomy.failure_trip_threshold ?? 0) },
    ],
    scheduled_task_surface: scheduledTaskSurface,
    diagnostic_panels: diagnostics,
    danger_zone_actions: [
      {
        title: "高风险命令仍然受限",
        body: naturalList(autonomy.blocked_commands, "当前没有额外的高风险命令被列出。"),
      },
      {
        title: "这里展示的是边界，不是主阅读层",
        body: "诊断与危险操作应保持后置，避免它们和主体叙述混成一块噪音。",
      },
    ],
  };
}
