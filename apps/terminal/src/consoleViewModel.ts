import type { ActivityEntry, UiState } from "./types.js";
import { translateBrainIdentifier, translateBrainIdentifierList } from "./brainLabels.js";
import { translateCognitivePhrase, translateTimelineType } from "./cognitiveTerms.js";
import { translateActionName, translateToolName } from "./displayLabels.js";

function presentText(value: unknown, fallback = "尚未接入"): string {
  if (typeof value === "string" && value.trim()) {
    return value.trim();
  }
  if (typeof value === "number" && Number.isFinite(value)) {
    return value.toFixed(2);
  }
  return fallback;
}

function formatScore(value: number | null | undefined): string {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(2) : "0.00";
}

function readContributionScore(record: Record<string, unknown>): number | null {
  const value = record.delta_normalized ?? record.delta_projected ?? record.weight ?? record.score ?? record.value;
  return typeof value === "number" && Number.isFinite(value)
    ? Math.abs(value)
    : typeof value === "string" && value.trim() && !Number.isNaN(Number(value))
      ? Math.abs(Number(value))
      : null;
}

function describeContributionDirection(record: Record<string, unknown>): string {
  if (record.hard_masked === true) {
    return "抑制";
  }
  const direction = typeof record.direction === "string" ? record.direction.trim() : "";
  if (direction === "block" || direction === "suppress") {
    return "压制";
  }
  if (direction === "support") {
    return "促进";
  }
  if (direction === "background") {
    return "背景偏置";
  }
  return "概率贡献";
}

function formatContributionRow(record: Record<string, unknown>): string | null {
  const source = translateBrainIdentifier(record.module_name ?? record.agent_name ?? record.source ?? record.name, "");
  if (!source) {
    return null;
  }
  const score = readContributionScore(record);
  const direction = describeContributionDirection(record);
  return score == null ? `${source} · ${direction}` : `${source} · ${direction} ${formatScore(score)}`;
}

function uniqueValues(values: string[]): string[] {
  return [...new Set(values)];
}

function activityText(entry: ActivityEntry): string {
  const prefix = {
    step: "步骤",
    tool: "外部动作",
    result: "结果",
    approval: "人工确认",
  }[entry.kind];
  const label =
    entry.kind === "tool" || entry.kind === "result" || entry.kind === "approval"
      ? translateToolName(entry.label, entry.label)
      : entry.label;
  const detail = entry.summary ? ` · ${entry.summary}` : "";
  return `${prefix}：${label}${detail}`;
}

export function buildConsoleFlowRows(state: UiState): string[] {
  if (state.console.timeline?.events.length) {
    return state.console.timeline.events.map((entry) => {
      const label = translateCognitivePhrase(entry.label, translateBrainIdentifier(entry.label, entry.label));
      const summary = translateCognitivePhrase(entry.summary, entry.summary);
      return `${translateTimelineType(entry.type)}：${label} · ${summary}`;
    });
  }
  return state.activityRail.map(activityText);
}

export function buildWhyNotText(state: UiState): string {
  const officialSummary =
    typeof state.console.whyNot?.whyNot.summary === "string" && state.console.whyNot.whyNot.summary.trim().length > 0
      ? state.console.whyNot.whyNot.summary.trim()
      : null;
  if (officialSummary) {
    return officialSummary;
  }
  if (state.console.whyNot?.action) {
    const blockedBy = Array.isArray(state.console.whyNot.whyNot.blocked_by)
      ? state.console.whyNot.whyNot.blocked_by.filter((item): item is string => typeof item === "string" && item.trim().length > 0)
      : [];
    const candidateScore =
      typeof state.console.whyNot.whyNot.candidate_score === "number" && Number.isFinite(state.console.whyNot.whyNot.candidate_score)
        ? state.console.whyNot.whyNot.candidate_score.toFixed(2)
        : null;
    const selectedAction =
      typeof state.console.whyNot.whyNot.selected_action === "string" && state.console.whyNot.whyNot.selected_action.trim().length > 0
        ? state.console.whyNot.whyNot.selected_action.trim()
        : null;
    if (blockedBy.length > 0) {
      const scoreText = candidateScore ? `，后验概率 ${candidateScore}` : "";
      const suffix = selectedAction ? `，最终输出 ${translateActionName(selectedAction, selectedAction)}` : "";
      return `${translateActionName(state.console.whyNot.action, state.console.whyNot.action)} 这条候选路径曾进入竞争${scoreText}，但受到 ${translateBrainIdentifierList(blockedBy).join("、")} 的抑制${suffix}`;
    }
  }
  const winnerAction = state.console.actionField?.winner.action;
  const candidate = state.console.actionField?.competingPeaks.find((item) => item.action && item.action !== winnerAction) ?? state.console.actionField?.competingPeaks[0];
  if (candidate?.action) {
    const candidateScore = typeof candidate.score === "number" && Number.isFinite(candidate.score) ? candidate.score.toFixed(2) : null;
    const topDrivers = state.console.whyCurrent?.why.topDrivers ?? [];
    const contributionStack = state.console.actionField?.contributionStack ?? [];
    const blockers = uniqueValues(
      translateBrainIdentifierList([
        ...topDrivers.map((entry) => entry.agent_name ?? entry.module_name ?? entry.name),
        ...contributionStack.map((entry) => entry.source),
      ]),
    ).slice(0, 3);
    const scoreText = candidateScore ? `，后验概率 ${candidateScore}` : "";
    const blockerText = blockers.length > 0 ? `，但受到 ${blockers.join("、")} 的抑制` : "";
    const winnerText = winnerAction ? `，最终输出 ${translateActionName(winnerAction, winnerAction)}` : "";
    return `${translateActionName(candidate.action, candidate.action)} 这条候选路径也进入过竞争${scoreText}${blockerText}${winnerText}`;
  }
  if (state.console.actionField?.competingPeaks.length) {
    return state.console.actionField.competingPeaks
      .map((item) => `${translateActionName(presentText(item.action), presentText(item.action))}(${formatScore(item.score)})`)
      .join("；");
  }
  return "本轮未采纳路径尚不清晰";
}

export function buildContributionText(state: UiState): string | null {
  const whyNotContributions = Array.isArray(state.console.whyNot?.whyNot.stacked_contributions)
    ? state.console.whyNot?.whyNot.stacked_contributions
    : [];
  if (whyNotContributions && whyNotContributions.length > 0) {
    const rows = whyNotContributions
      .map((item) => {
        if (typeof item !== "object" || item === null) {
          return null;
        }
        const record = item as Record<string, unknown>;
        return formatContributionRow(record);
      })
      .filter((item): item is string => typeof item === "string" && item.length > 0);
    if (rows.length > 0) {
      return rows.join("；");
    }
  }
  if (state.console.actionField?.contributionStack.length) {
    return state.console.actionField.contributionStack
      .map((item) => formatContributionRow(item.raw) ?? `${translateBrainIdentifier(item.source)} · 概率贡献 ${formatScore(item.weight)}`)
      .join("；");
  }
  return null;
}
