import React, { useEffect, useMemo, useRef, useState } from "react";
import { Box, Text, useApp, useInput } from "ink";
import TextInput from "ink-text-input";

import { cycleMenuIndex, flattenVisibleActions, isDigitSelection, splitActionGroups } from "./actionBar.js";
import { PythonBridgeClient, resolveInteractiveSessionId } from "./bridge/client.js";
import { parseSlashCommand } from "./commands/slash.js";
import { buildPaletteSections, filterPaletteSections, type PaletteContext, type PaletteEntry } from "./commandPalette.js";
import { buildConsoleViewportPlan } from "./consoleLayout.js";
import { cycleApprovalCursor, cycleDetailDrawer, toggleDetailDrawer } from "./detailDrawer.js";
import { buildHelpHint } from "./helpHint.js";
import { compactPaletteSections } from "./paletteView.js";
import { derivePaletteContext } from "./paletteContext.js";
import { formatDetailSummary, formatSidebarSummary } from "./panelSummary.js";
import { buildConsoleFlowRows, buildContributionText, buildWhyNotText } from "./consoleViewModel.js";
import { commitPrompt, createPromptHistoryState, movePromptCursor } from "./promptHistory.js";
import { joinReadableLabel, toReadableLines } from "./readableText.js";
import { canUseDigitShortcut, shortcutDraftTarget } from "./shortcutDraft.js";
import { addLocalLine, addUserLine, applyBridgeEvent, clearLines, createInitialUiState } from "./state/sessionStore.js";
import { formatStatusLine, tokenizeStatusLine } from "./statusLine.js";
import { actionChipStyle, borderColorForZone, paletteRowStyle, sidebarToneColor, terminalTheme, transcriptLineStyle } from "./terminalTheme.js";
import {
  actionGroupLabel,
  actionTitle,
  approvalHeadline,
  consoleSectionTitle,
  contributionSectionTitle,
  consoleShellSubtitle,
  consoleShellTitle,
  emptyDrawerText,
  emptyMindsetText,
  flowSectionTitle,
  paletteSectionLabel,
  paletteTitle,
  whyNotSectionTitle,
} from "./terminalCopy.js";
import { translatePermissionMode, translateRiskLevel, translateRunStatus } from "./displayLabels.js";
import { buildTranscriptWindow, moveTranscriptOffset } from "./transcriptViewport.js";
import type { ActivityEntry, PanelKey, UiAction, UiLine, UiState } from "./types.js";

const HELP_TEXT = [
  "快捷键",
  "  Tab 切换焦点区 | Ctrl+P 打开命令面板 | PageUp/PageDown 滚动 | Ctrl+J / Ctrl+K 切换抽屉",
  "  a 审批 | s 状态 | w 解释 | t 步骤/工具 | i 认知 | m 元信息",
  "  [ / ] 切审批队列 | 1-9 直选 | ← → 移动 | Enter 确认",
  "",
  "斜杠命令",
  "  /status /why /steps /tools /state /probability [round|action <name>|layer <name>]",
  "  /dream [cue] /pause /resume /abort /mode [value] /permissions [value] /model",
  "  /endogenous [trigger] [mode] /why-motivation [round] /replay-motivation [round]",
  "  /initiative [status|distribution|trigger [trigger] [mode]|why [round]]",
  "  /replay [round] [seed] /why-not [action] [round] /what-changed [window] /eval [rounds]",
  "",
  "本地命令",
  "  /help /clear /compact /exit",
].join("\n");

const SURFACE_GAP = 1;
const SECTION_GAP = 1;
const CHIP_GAP = 1;

function renderLine(line: UiLine, index: number): React.ReactNode {
  const { color, label } = transcriptLineStyle(line.kind);
  return (
    <Text key={`${line.kind}-${index}`} color={color}>
      {label}: {line.text}
    </Text>
  );
}

function truncateBlock(text: string, maxLines: number): string {
  return text
    .split("\n")
    .slice(0, Math.max(1, maxLines))
    .join("\n");
}

function renderReadableRows(rows: string[], color: string, keyPrefix: string): React.ReactNode {
  return rows.map((row, index) => (
    <Text key={`${keyPrefix}-${index}`} color={color} wrap="wrap">
      {row}
    </Text>
  ));
}

function drawerTitle(panel: Exclude<PanelKey, null>): string {
  return {
    status: "状态",
    why: "原因",
    steps: "步骤",
    tools: "工具",
    cognition: "认知",
    approvals: "审批",
    meta: "元信息",
  }[panel];
}

function activityText(entry: ActivityEntry): string {
  const prefix = {
    step: "步骤",
    tool: "工具",
    result: "结果",
    approval: "审批",
  }[entry.kind];
  const detail = entry.summary ? ` · ${entry.summary}` : "";
  return `${prefix}: ${entry.label}${detail}`;
}

function nextFocusZone(current: UiState["focusZone"], hasDrawer: boolean): UiState["focusZone"] {
  const zones: UiState["focusZone"][] = hasDrawer
    ? ["transcript", "drawer", "actions", "approval", "input"]
    : ["transcript", "actions", "approval", "input"];
  const index = zones.indexOf(current);
  return zones[(index + 1) % zones.length] ?? "input";
}

function panelForCommand(command: string, current: PanelKey): PanelKey {
  if (command === "status" || command === "why" || command === "steps" || command === "tools") {
    return command;
  }
  if (command === "state") {
    return toggleDetailDrawer(current, "cognition");
  }
  if (command === "model" || command === "permissions" || command === "mode") {
    return "meta";
  }
  if (
    command === "endogenous" ||
    command === "initiative" ||
    command === "why-motivation" ||
    command === "replay-motivation" ||
    command === "replay" ||
    command === "why-not" ||
    command === "what-changed" ||
    command === "eval"
  ) {
    return "meta";
  }
  return current;
}

function paletteEmptyText(sectionId: "recommended" | "recent" | "all"): string {
  switch (sectionId) {
    case "recommended":
      return "暂无推荐命令。";
    case "recent":
      return "执行命令后会出现在这里。";
    case "all":
      return "尚未接入命令。";
  }
}

export function App({ bridge, cwd, repoRoot }: { bridge: PythonBridgeClient; cwd: string; repoRoot: string }) {
  const { exit } = useApp();
  const stdoutWidth = process.stdout.columns ?? 120;
  const stdoutHeight = process.stdout.rows ?? 32;
  const [input, setInput] = useState("");
  const [uiState, setUiState] = useState<UiState>(createInitialUiState());
  const [promptHistoryState, setPromptHistoryState] = useState(createPromptHistoryState());
  const [transcriptOffset, setTranscriptOffset] = useState(0);
  const [approvalChoiceIndex, setApprovalChoiceIndex] = useState(0);
  const [paletteOpen, setPaletteOpen] = useState(false);
  const [paletteQuery, setPaletteQuery] = useState("");
  const [paletteIndex, setPaletteIndex] = useState(0);
  const [paletteRecentIds, setPaletteRecentIds] = useState<string[]>([]);
  const sessionId = useMemo(() => resolveInteractiveSessionId(repoRoot), [repoRoot]);
  const effectiveSessionId = uiState.activeSessionId ?? sessionId;
  const pendingApproval = uiState.pendingApprovals[uiState.approvalCursor] ?? null;
  const viewportPlan = buildConsoleViewportPlan({
    width: stdoutWidth,
    height: stdoutHeight,
    hasDetailDrawer: uiState.detailDrawer !== null,
    hasPendingApproval: pendingApproval !== null,
  });
  const transcriptSourceLines =
    uiState.transcriptMode === "compact"
      ? uiState.lines.filter((line) => line.kind === "user" || line.kind === "assistant" || line.kind === "error")
      : uiState.lines;
  const previousTranscriptCountRef = useRef(transcriptSourceLines.length);
  const previousPendingCountRef = useRef(uiState.pendingApprovals.length);
  const sidebarSummary = formatSidebarSummary(uiState);
  const detailBody = uiState.detailDrawer ? formatDetailSummary(uiState, uiState.detailDrawer) : "";
  const visibleActions = flattenVisibleActions(uiState.actionBar.primary, uiState.actionBar.secondary).slice(0, 9);
  const actionGroups = splitActionGroups(uiState.actionBar.primary, uiState.actionBar.secondary);
  const approvalChoices = pendingApproval?.choices?.length ? pendingApproval.choices.slice(0, 9) : [];

  useEffect(() => {
    const disposeListener = bridge.onEvent((event) => {
      setUiState((current) => applyBridgeEvent(current, event));
    });
    void bridge.startSession(sessionId, cwd);
    return () => {
      disposeListener();
      bridge.dispose();
    };
  }, [bridge, cwd, sessionId]);

  useEffect(() => {
    const previousCount = previousTranscriptCountRef.current;
    if (transcriptOffset > 0 && transcriptSourceLines.length > previousCount) {
      setTranscriptOffset((current) => current + (transcriptSourceLines.length - previousCount));
    }
    previousTranscriptCountRef.current = transcriptSourceLines.length;
  }, [transcriptOffset, transcriptSourceLines.length]);

  useEffect(() => {
    if (uiState.pendingApprovals.length > previousPendingCountRef.current) {
      setUiState((current) => ({ ...current, focusZone: "approval", detailDrawer: "approvals" }));
      setApprovalChoiceIndex(0);
    }
    previousPendingCountRef.current = uiState.pendingApprovals.length;
  }, [uiState.pendingApprovals.length]);

  useEffect(() => {
    setApprovalChoiceIndex(0);
  }, [pendingApproval?.callId]);

  useEffect(() => {
    setPaletteIndex(0);
  }, [paletteQuery, paletteOpen]);

  function rememberPaletteEntry(entry: PaletteEntry): void {
    setPaletteRecentIds((current) => [entry.id, ...current.filter((id) => id !== entry.id)].slice(0, 5));
  }

  async function executeLocalCommand(command: "help" | "clear" | "compact" | "exit"): Promise<void> {
    if (command === "help") {
      setUiState((current) => addLocalLine(current, HELP_TEXT));
      return;
    }
    if (command === "clear") {
      setUiState((current) => clearLines(current));
      setTranscriptOffset(0);
      return;
    }
    if (command === "compact") {
      setUiState((current) => ({
        ...current,
        transcriptMode: current.transcriptMode === "compact" ? "full" : "compact",
      }));
      return;
    }
    await bridge.detachSession(effectiveSessionId, { transcriptMode: uiState.transcriptMode });
    exit();
  }

  function handleApprovalChoice(choice: UiAction): void {
    if (!pendingApproval) {
      return;
    }
    if (choice.kind === "approval") {
      bridge.send({
        type: "approve",
        session_id: effectiveSessionId,
        call_id: pendingApproval.callId,
        approved: choice.value === "approve",
      });
      return;
    }
    if (choice.kind === "drawer") {
      setUiState((current) => ({ ...current, detailDrawer: "approvals", focusZone: "drawer" }));
      return;
    }
    if (choice.kind === "approval_nav") {
      setUiState((current) => ({
        ...current,
        approvalCursor: cycleApprovalCursor(current.approvalCursor, current.pendingApprovals.length, "next"),
        detailDrawer: "approvals",
        focusZone: "approval",
      }));
      setApprovalChoiceIndex(0);
    }
  }

  function handleUiAction(action: UiAction): void {
    if (action.disabled) {
      return;
    }
    if (action.kind === "drawer") {
      setUiState((current) => ({
        ...current,
        detailDrawer: action.value as Exclude<PanelKey, null>,
        focusZone: "drawer",
      }));
      return;
    }
    bridge.send({ type: "control_command", session_id: effectiveSessionId, command: action.value as any });
    setUiState((current) => ({
      ...current,
      detailDrawer: panelForCommand(action.value, current.detailDrawer),
      focusZone: "drawer",
    }));
  }

  const paletteContext = derivePaletteContext(uiState);
  const paletteSections = compactPaletteSections(
    filterPaletteSections(
      buildPaletteSections(uiState.actionBar.primary, uiState.actionBar.secondary, {
        context: paletteContext,
        recentIds: paletteRecentIds,
        onUiAction: (action) => {
          setPaletteOpen(false);
          setPaletteQuery("");
          handleUiAction(action);
        },
        onLocalAction: async (command) => {
          setPaletteOpen(false);
          setPaletteQuery("");
          await executeLocalCommand(command);
        },
      }),
      paletteQuery,
    ),
    paletteQuery,
  );
  const paletteVisibleEntries = paletteSections.flatMap((section) => section.entries);

  useEffect(() => {
    setPaletteIndex((current) => {
      if (paletteVisibleEntries.length === 0) {
        return 0;
      }
      return Math.min(current, paletteVisibleEntries.length - 1);
    });
  }, [paletteVisibleEntries.length]);

  useInput((rawInput, key) => {
    const normalizedInput = rawInput.toLowerCase();
    const inputEmpty = input.length === 0;
    const actionSelection = isDigitSelection(normalizedInput);
    const digitShortcutEnabled = canUseDigitShortcut({
      paletteOpen,
      paletteQueryEmpty: paletteQuery.length === 0,
      focusZone: uiState.focusZone,
      inputEmpty,
    });
    const shortcutTarget = shortcutDraftTarget({
      ctrl: Boolean(key.ctrl),
      key: normalizedInput,
      paletteOpen,
    });
    const restoreShortcutDraft = () => {
      if (shortcutTarget !== "input") {
        return;
      }
      const preservedInput = input;
      setTimeout(() => {
        setInput(preservedInput);
      }, 0);
    };

    if (key.ctrl && normalizedInput === "p") {
      restoreShortcutDraft();
      setPaletteOpen((current) => !current);
      setPaletteQuery("");
      setPaletteIndex(0);
      return;
    }
    if (paletteOpen) {
      if (key.escape) {
        setPaletteOpen(false);
        setPaletteQuery("");
        return;
      }
      if (key.upArrow) {
        setPaletteIndex((current) => cycleMenuIndex(current, paletteVisibleEntries.length, "prev"));
        return;
      }
      if (key.downArrow) {
        setPaletteIndex((current) => cycleMenuIndex(current, paletteVisibleEntries.length, "next"));
        return;
      }
      if (digitShortcutEnabled && actionSelection !== null && actionSelection < paletteVisibleEntries.length) {
        const selected = paletteVisibleEntries[actionSelection];
        if (selected && !selected.disabled) {
          setPaletteIndex(actionSelection);
          rememberPaletteEntry(selected);
          void selected.run();
        }
        return;
      }
      return;
    }
    if (key.ctrl && normalizedInput === "c") {
      restoreShortcutDraft();
      bridge.send({ type: "control_command", session_id: effectiveSessionId, command: "abort" });
      return;
    }
    if (key.ctrl && normalizedInput === "d") {
      restoreShortcutDraft();
      void bridge.detachSession(effectiveSessionId, { transcriptMode: uiState.transcriptMode }).finally(() => exit());
      return;
    }
    if (key.ctrl && normalizedInput === "l") {
      restoreShortcutDraft();
      setUiState((current) => clearLines(current));
      return;
    }
    if (key.ctrl && (normalizedInput === "j" || normalizedInput === "k")) {
      restoreShortcutDraft();
      setUiState((current) => ({
        ...current,
        detailDrawer: cycleDetailDrawer(current.detailDrawer, normalizedInput === "j" ? "next" : "prev"),
        focusZone: "drawer",
      }));
      return;
    }
    if (key.tab) {
      setUiState((current) => ({
        ...current,
        focusZone: nextFocusZone(current.focusZone, current.detailDrawer !== null),
      }));
      return;
    }
    if (digitShortcutEnabled && actionSelection !== null) {
      if (pendingApproval && uiState.focusZone === "approval" && actionSelection < approvalChoices.length) {
        setInput("");
        handleApprovalChoice(approvalChoices[actionSelection]!);
        return;
      }
      if (actionSelection < visibleActions.length) {
        setInput("");
        setUiState((current) => ({ ...current, actionBar: { ...current.actionBar, selectedIndex: actionSelection } }));
        handleUiAction(visibleActions[actionSelection]!);
        return;
      }
    }
    if (pendingApproval && (normalizedInput === "y" || normalizedInput === "n")) {
      bridge.send({
        type: "approve",
        session_id: effectiveSessionId,
        call_id: pendingApproval.callId,
        approved: normalizedInput === "y",
      });
      return;
    }
    if (inputEmpty && key.leftArrow && uiState.focusZone === "actions" && visibleActions.length > 0) {
      setUiState((current) => ({
        ...current,
        actionBar: {
          ...current.actionBar,
          selectedIndex: cycleMenuIndex(current.actionBar.selectedIndex, visibleActions.length, "prev"),
        },
      }));
      return;
    }
    if (inputEmpty && key.rightArrow && uiState.focusZone === "actions" && visibleActions.length > 0) {
      setUiState((current) => ({
        ...current,
        actionBar: {
          ...current.actionBar,
          selectedIndex: cycleMenuIndex(current.actionBar.selectedIndex, visibleActions.length, "next"),
        },
      }));
      return;
    }
    if (inputEmpty && key.leftArrow && uiState.focusZone === "approval" && approvalChoices.length > 0) {
      setApprovalChoiceIndex((current) => cycleMenuIndex(current, approvalChoices.length, "prev"));
      return;
    }
    if (inputEmpty && key.rightArrow && uiState.focusZone === "approval" && approvalChoices.length > 0) {
      setApprovalChoiceIndex((current) => cycleMenuIndex(current, approvalChoices.length, "next"));
      return;
    }
    if (inputEmpty && key.return && uiState.focusZone === "actions" && visibleActions.length > 0) {
      handleUiAction(visibleActions[uiState.actionBar.selectedIndex] ?? visibleActions[0]!);
      return;
    }
    if (inputEmpty && key.return && uiState.focusZone === "approval" && approvalChoices.length > 0) {
      handleApprovalChoice(approvalChoices[approvalChoiceIndex] ?? approvalChoices[0]!);
      return;
    }
    if (inputEmpty && (normalizedInput === "[" || normalizedInput === "]")) {
      setUiState((current) => ({
        ...current,
        approvalCursor: cycleApprovalCursor(
          current.approvalCursor,
          current.pendingApprovals.length,
          normalizedInput === "]" ? "next" : "prev",
        ),
        detailDrawer: current.pendingApprovals.length > 0 ? "approvals" : current.detailDrawer,
      }));
      return;
    }
    if (inputEmpty && ["a", "s", "w", "t", "i", "m"].includes(normalizedInput)) {
      setUiState((current) => {
        if (normalizedInput === "t") {
          const nextPanel = current.detailDrawer === "steps" ? "tools" : "steps";
          return { ...current, detailDrawer: toggleDetailDrawer(current.detailDrawer, nextPanel), focusZone: "drawer" };
        }
        const mapping: Record<string, Exclude<PanelKey, null>> = {
          a: "approvals",
          s: "status",
          w: "why",
          i: "cognition",
          m: "meta",
          t: "steps",
        };
        return {
          ...current,
          detailDrawer: toggleDetailDrawer(current.detailDrawer, mapping[normalizedInput] ?? "status"),
          focusZone: "drawer",
        };
      });
      return;
    }
    if (key.pageUp) {
      setTranscriptOffset((current) =>
        moveTranscriptOffset({
          currentOffset: current,
          direction: "older",
          pageSize: viewportPlan.transcriptLines,
          totalLines: transcriptSourceLines.length,
          visibleCount: viewportPlan.transcriptLines,
        }),
      );
      return;
    }
    if (key.pageDown) {
      setTranscriptOffset((current) =>
        moveTranscriptOffset({
          currentOffset: current,
          direction: "newer",
          pageSize: viewportPlan.transcriptLines,
          totalLines: transcriptSourceLines.length,
          visibleCount: viewportPlan.transcriptLines,
        }),
      );
      return;
    }
    if (key.upArrow) {
      const moved = movePromptCursor(promptHistoryState, "up", input);
      setPromptHistoryState(moved.state);
      setInput(moved.input);
      return;
    }
    if (key.downArrow) {
      const moved = movePromptCursor(promptHistoryState, "down", input);
      setPromptHistoryState(moved.state);
      setInput(moved.input);
    }
  });

  async function handleSubmit(value: string) {
    if (paletteOpen) {
      const selected = paletteVisibleEntries[paletteIndex] ?? paletteVisibleEntries[0];
      if (selected && !selected.disabled) {
        rememberPaletteEntry(selected);
        await selected.run();
      }
      return;
    }
    if (value.endsWith("\\")) {
      setInput(`${value.slice(0, -1)}\n`);
      return;
    }
    const trimmed = value.trim();
    if (!trimmed) {
      setInput("");
      return;
    }
    const nextPromptHistoryState = commitPrompt(promptHistoryState, trimmed);
    setPromptHistoryState(nextPromptHistoryState);
    setUiState((current) => ({
      ...current,
      promptHistoryByCwd: {
        ...current.promptHistoryByCwd,
        [cwd]: nextPromptHistoryState.entries,
      },
      focusZone: "input",
    }));
    setInput("");
    const parsed = parseSlashCommand(trimmed);
    if (!parsed) {
      setUiState((current) => addUserLine(current, trimmed));
      bridge.send({ type: "user_turn", session_id: effectiveSessionId, text: trimmed });
      return;
    }
    if (parsed.kind === "local") {
      await executeLocalCommand(parsed.command);
      return;
    }
    if (parsed.kind === "unknown") {
      setUiState((current) => addLocalLine(current, `未知斜杠命令：/${parsed.command}`, "error"));
      return;
    }
    setUiState((current) => ({
      ...current,
      detailDrawer: panelForCommand(parsed.command, current.detailDrawer),
      focusZone:
        [
          "status",
          "why",
          "steps",
          "tools",
          "state",
          "model",
          "permissions",
          "mode",
          "endogenous",
          "initiative",
          "why-motivation",
          "replay-motivation",
          "replay",
          "why-not",
          "what-changed",
          "eval",
        ].includes(parsed.command)
          ? "drawer"
          : current.focusZone,
    }));
    bridge.send({ type: "control_command", session_id: effectiveSessionId, command: parsed.command, value: parsed.value });
  }

  const transcriptWindow = buildTranscriptWindow(transcriptSourceLines, viewportPlan.transcriptLines, transcriptOffset);
  const activityRows = buildConsoleFlowRows(uiState).slice(-viewportPlan.sidebarLines);
  const visibleLines = transcriptWindow.lines;
  const helpHint = buildHelpHint(uiState, { paletteOpen });
  const statusTokens = tokenizeStatusLine(uiState);
  const transcriptBorder = borderColorForZone("transcript", uiState.focusZone === "transcript");
  const drawerBorder = borderColorForZone("drawer", uiState.focusZone === "drawer");
  const inputBorder = borderColorForZone("input", uiState.focusZone === "input");
  const approvalBorder = borderColorForZone("approval", uiState.focusZone === "approval");
  const actionBorder = borderColorForZone("actions", uiState.focusZone === "actions");
  const mindState = uiState.aliveConsole.state;
  const actionFieldState = uiState.aliveConsole.actionField;
  const whyState = uiState.aliveConsole.why;
  const whyNotText = buildWhyNotText(uiState);
  const contributionText = buildContributionText(uiState);
  const whyNotRows = toReadableLines(whyNotText, { fallback: "本轮未采纳路径尚不清晰", maxLines: Math.max(2, viewportPlan.sidebarLines - 1) });
  const contributionRows = contributionText
    ? joinReadableLabel(contributionSectionTitle(), contributionText, { fallback: "贡献叠层尚未完全暴露", maxLines: Math.max(2, viewportPlan.sidebarLines - 1) })
    : uiState.detailDrawer
      ? toReadableLines(truncateBlock(detailBody || emptyDrawerText(), Math.max(2, viewportPlan.detailLines - 8)), { fallback: emptyDrawerText() })
      : ["贡献叠层尚未完全暴露"];
  const approvalMetaRows =
    pendingApproval && (pendingApproval.riskLevel || pendingApproval.mode || pendingApproval.status)
      ? [
          pendingApproval.riskLevel ? `风险：${translateRiskLevel(pendingApproval.riskLevel, pendingApproval.riskLevel)}` : null,
          pendingApproval.mode ? `模式：${translatePermissionMode(pendingApproval.mode, pendingApproval.mode)}` : null,
          pendingApproval.status ? `状态：${translateRunStatus(pendingApproval.status, pendingApproval.status)}` : null,
        ].filter((item): item is string => Boolean(item))
      : [];

  const summaryBox = (
    <Box borderStyle="round" borderColor={terminalTheme.ink.panelBorder} paddingX={1} flexDirection="column">
      <Text color={terminalTheme.ink.subtle} wrap="wrap">{mindState.title}</Text>
      <Text color={terminalTheme.ink.text} wrap="wrap">{mindState.summary || emptyMindsetText()}</Text>
      {mindState.details.length > 0 ? (
        mindState.details.flatMap((item, index) => toReadableLines(item, { maxLines: 3 }).map((row, rowIndex) => (
          <Text key={`${mindState.title}-${index}-${rowIndex}`} color={terminalTheme.ink.muted} wrap="wrap">
            {row}
          </Text>
        )))
      ) : null}
      {sidebarSummary.length > 0 ? (
        <Box marginTop={SECTION_GAP} flexDirection="column">
          <Text color={terminalTheme.ink.subtle} wrap="wrap">稳定观测</Text>
          {sidebarSummary.flatMap((item) =>
            joinReadableLabel(item.label, item.value, { maxLines: 3 }).map((row, index) => (
              <Text key={`${item.label}-${index}`} color={sidebarToneColor(item.tone)} wrap="wrap">
                {row}
              </Text>
            )),
          )}
        </Box>
      ) : mindState.details.length === 0 ? (
        <Text color={terminalTheme.ink.muted} wrap="wrap">{emptyMindsetText()}</Text>
      ) : null}
    </Box>
  );

  const actionFieldBox = (
    <Box borderStyle="round" borderColor={terminalTheme.ink.panelBorder} paddingX={1} flexDirection="column">
      <Text color={terminalTheme.ink.subtle} wrap="wrap">{actionFieldState.title}</Text>
      <Text color={terminalTheme.ink.text} wrap="wrap">{actionFieldState.summary || emptyMindsetText()}</Text>
      {actionFieldState.details.length > 0 ? (
        actionFieldState.details
          .slice(0, Math.max(2, viewportPlan.sidebarLines - 2))
          .flatMap((item, index) => toReadableLines(item, { maxLines: 3 }).map((row, rowIndex) => (
            <Text key={`${actionFieldState.title}-${index}-${rowIndex}`} color={terminalTheme.ink.muted} wrap="wrap">
              {row}
            </Text>
          )))
      ) : (
        <Text color={terminalTheme.ink.muted} wrap="wrap">{emptyMindsetText()}</Text>
      )}
    </Box>
  );

  const activityBox = (
    <Box marginTop={1} borderStyle="round" borderColor={terminalTheme.ink.panelBorder} paddingX={1} flexDirection="column">
      <Text color={terminalTheme.ink.subtle} wrap="wrap">{flowSectionTitle()}</Text>
      {activityRows.length > 0 ? (
        activityRows.flatMap((entry, index) => toReadableLines(entry, { maxLines: 3 }).map((row, rowIndex) => (
          <Text key={`flow-${index}-${rowIndex}`} color={terminalTheme.ink.muted} wrap="wrap">
            {row}
          </Text>
        )))
      ) : (
        <Text color={terminalTheme.ink.muted} wrap="wrap">{emptyMindsetText()}</Text>
      )}
    </Box>
  );

  const drawerBox = (
    <Box borderStyle="round" borderColor={drawerBorder} paddingX={1} flexDirection="column">
      <Text color={terminalTheme.ink.subtle} wrap="wrap">{consoleSectionTitle("explanation")}</Text>
      <Text color={terminalTheme.ink.text} wrap="wrap">{whyState.summary || emptyDrawerText()}</Text>
      {whyState.details.length > 0 ? (
        whyState.details.slice(0, 4).flatMap((item, index) => toReadableLines(item, { maxLines: 3 }).map((row, rowIndex) => (
          <Text key={`${whyState.title}-${index}-${rowIndex}`} color={terminalTheme.ink.muted} wrap="wrap">
            {row}
          </Text>
        )))
      ) : (
        <Text color={terminalTheme.ink.muted} wrap="wrap">{emptyDrawerText()}</Text>
      )}
      <Box marginTop={SECTION_GAP} flexDirection="column">
        <Text color={terminalTheme.ink.subtle} wrap="wrap">{whyNotSectionTitle()}</Text>
        {renderReadableRows(whyNotRows, terminalTheme.ink.muted, "why-not")}
      </Box>
      <Box marginTop={SECTION_GAP} flexDirection="column">
        <Text color={terminalTheme.ink.subtle} wrap="wrap">{contributionSectionTitle()}</Text>
        {renderReadableRows(contributionRows, terminalTheme.ink.muted, "contribution")}
        {uiState.detailDrawer ? <Text color={terminalTheme.ink.muted} wrap="wrap">{`展开视角：${drawerTitle(uiState.detailDrawer)}`}</Text> : null}
      </Box>
    </Box>
  );

  const actionBarBox = visibleActions.length > 0 ? (
    <Box marginTop={SURFACE_GAP} flexDirection="column">
      <Text color={uiState.focusZone === "actions" ? terminalTheme.ink.accent : terminalTheme.ink.muted}>{actionTitle()}</Text>
      <Box flexWrap="wrap">
        <Text color={terminalTheme.ink.muted}>{`${actionGroupLabel("inspect")} `}</Text>
        {actionGroups.inspect.slice(0, 5).map((action) => {
          const index = visibleActions.findIndex((item) => item.id === action.id);
          const selected = index === uiState.actionBar.selectedIndex;
          const chip = actionChipStyle({ selected, disabled: action.disabled, tone: "accent" });
          return (
            <Box key={action.id} marginRight={CHIP_GAP}>
              <Text color={chip.color} backgroundColor={chip.backgroundColor}>
                {`${index + 1}.${action.label}${action.disabled ? "（禁用）" : ""}`}
              </Text>
            </Box>
          );
        })}
      </Box>
      {actionGroups.control.length > 0 ? (
        <Box flexWrap="wrap" marginTop={SECTION_GAP}>
          <Text color={terminalTheme.ink.muted}>{`${actionGroupLabel("control")} `}</Text>
          {actionGroups.control.slice(0, Math.max(0, 9 - actionGroups.inspect.length)).map((action) => {
            const index = visibleActions.findIndex((item) => item.id === action.id);
            const selected = index === uiState.actionBar.selectedIndex;
            const chip = actionChipStyle({ selected, disabled: action.disabled, tone: "warning" });
            return (
              <Box key={action.id} marginRight={CHIP_GAP}>
                <Text color={chip.color} backgroundColor={chip.backgroundColor}>
                  {`${index + 1}.${action.label}${action.disabled ? "（禁用）" : ""}`}
                </Text>
              </Box>
            );
          })}
        </Box>
      ) : null}
    </Box>
  ) : null;

  const approvalBox = pendingApproval ? (
    <Box marginTop={SURFACE_GAP} borderStyle="round" borderColor={approvalBorder} paddingX={1} flexDirection="column">
      <Text color={terminalTheme.ink.warning}>
        {approvalHeadline({
          cursor: uiState.approvalCursor,
          total: uiState.pendingApprovals.length,
          tool: pendingApproval.tool,
          summary: pendingApproval.summary,
        })}
      </Text>
      {pendingApproval.actionPreview ? renderReadableRows(toReadableLines(pendingApproval.actionPreview, { maxLines: 3 }), terminalTheme.ink.muted, "approval-preview") : null}
      {approvalMetaRows.length > 0 ? renderReadableRows(approvalMetaRows, terminalTheme.ink.muted, "approval-meta") : null}
      {approvalChoices.length > 0 ? (
        <Box flexWrap="wrap" marginTop={SECTION_GAP}>
          {approvalChoices.map((choice, index) => {
            const selected = index === approvalChoiceIndex;
            const chip = actionChipStyle({ selected, disabled: false, tone: "warning" });
            return (
              <Box key={choice.id} marginRight={CHIP_GAP}>
                <Text color={chip.color} backgroundColor={chip.backgroundColor}>
                  {`${index + 1}.${choice.label}`}
                </Text>
              </Box>
            );
          })}
        </Box>
      ) : (
        <Text color={terminalTheme.ink.muted}>按 `y` 通过，按 `n` 拒绝。</Text>
      )}
    </Box>
  ) : null;

  const paletteBox = paletteOpen ? (
    <Box marginTop={SURFACE_GAP} borderStyle="round" borderColor={terminalTheme.ink.accentBorder} paddingX={1} flexDirection="column">
      <Text color={terminalTheme.ink.accentSoft}>{paletteTitle()}</Text>
      {paletteVisibleEntries.length === 0 ? (
        <Text color={terminalTheme.ink.muted}>没有匹配项。</Text>
      ) : (
        paletteSections.map((section, sectionIndex) => {
          let sectionOffset = 0;
          for (const priorSection of paletteSections) {
            if (priorSection === section) {
              break;
            }
            sectionOffset += priorSection.entries.length;
          }
          return (
            <Box key={section.id} flexDirection="column" marginTop={sectionIndex === 0 ? 0 : SECTION_GAP}>
              <Text color={terminalTheme.ink.muted}>{paletteSectionLabel(section.id)}</Text>
              {section.entries.length === 0 ? <Text color={terminalTheme.ink.muted}>{paletteEmptyText(section.id)}</Text> : null}
              {section.entries.map((entry, index) => {
                const absoluteIndex = sectionOffset + index;
                const selected = absoluteIndex === paletteIndex;
                const shortcut = absoluteIndex < 9 ? `${absoluteIndex + 1}. ` : "   ";
                const row = paletteRowStyle(selected, entry.disabled);
                return (
                  <Text
                    key={`${section.id}-${entry.group}-${entry.id}-${absoluteIndex}`}
                    color={row.color}
                    backgroundColor={row.backgroundColor}
                  >
                    {`${shortcut}${entry.group} · ${entry.label}${entry.disabled ? "（禁用）" : ""}`}
                  </Text>
                );
              })}
            </Box>
          );
        })
      )}
    </Box>
  ) : null;

  return (
    <Box flexDirection="column">
      <Text color={terminalTheme.ink.text}>{consoleShellTitle()}</Text>
      <Text color={terminalTheme.ink.muted}>{consoleShellSubtitle()}</Text>
      <Box marginTop={SURFACE_GAP} flexDirection={viewportPlan.layout === "split" ? "row" : "column"}>
        <Box
          width={viewportPlan.layout === "split" ? viewportPlan.sidebarWidth : undefined}
          marginRight={viewportPlan.layout === "split" ? 1 : 0}
          flexDirection="column"
        >
          <Box borderStyle="round" borderColor={terminalTheme.ink.panelBorder} paddingX={1} flexDirection="column">
            <Text color={terminalTheme.ink.subtle}>{consoleSectionTitle("mindset")}</Text>
            {summaryBox}
            {activityBox}
          </Box>
        </Box>
        <Box
          flexDirection="column"
          flexGrow={1}
          marginRight={viewportPlan.layout === "split" ? 1 : 0}
          marginTop={viewportPlan.layout === "split" ? 0 : SURFACE_GAP}
        >
          {actionFieldBox}
          <Box borderStyle="round" borderColor={transcriptBorder} paddingX={1} flexDirection="column">
            <Text color={terminalTheme.ink.subtle}>
              {`${consoleSectionTitle("transcript")}${transcriptWindow.offset > 0 ? ` · 回看 +${transcriptWindow.offset}` : " · 实时"}`}
            </Text>
            {pendingApproval ? <Text color={terminalTheme.ink.warning}>等待审批后继续</Text> : null}
            {visibleLines.length > 0 ? visibleLines.map((line, index) => renderLine(line, index)) : <Text color={terminalTheme.ink.muted}>{emptyMindsetText()}</Text>}
          </Box>
        </Box>
        {viewportPlan.detailPlacement === "side" ? (
          <Box
            width={viewportPlan.layout === "split" ? viewportPlan.detailWidth : undefined}
            marginTop={viewportPlan.layout === "split" ? 0 : SURFACE_GAP}
            flexDirection="column"
          >
            {drawerBox}
          </Box>
        ) : null}
      </Box>
      {viewportPlan.detailPlacement === "bottom" ? (
        <Box marginTop={SURFACE_GAP} flexDirection="column">
          {drawerBox}
        </Box>
      ) : null}
      <Box marginTop={SURFACE_GAP} flexDirection="column">
        <Box borderStyle="round" borderColor={inputBorder} paddingX={1} flexDirection="column">
          <Text color={uiState.focusZone === "input" || paletteOpen ? terminalTheme.ink.accent : terminalTheme.ink.muted}>
            {consoleSectionTitle("input")}
          </Text>
          {approvalBox}
          {actionBarBox ? (
            <Box marginTop={SECTION_GAP} borderStyle="round" borderColor={actionBorder} paddingX={1} flexDirection="column">
              {actionBarBox}
            </Box>
          ) : null}
          {paletteBox}
          <Box marginTop={SURFACE_GAP}>
            <Text color={uiState.focusZone === "input" || paletteOpen ? terminalTheme.ink.accent : terminalTheme.ink.muted}>
              {paletteOpen ? "palette> " : "> "}
            </Text>
            <TextInput
              value={paletteOpen ? paletteQuery : input}
              focus={paletteOpen || uiState.focusZone === "input"}
              onChange={paletteOpen ? setPaletteQuery : setInput}
              onSubmit={handleSubmit}
            />
          </Box>
          <Text color={terminalTheme.ink.muted}>{helpHint}</Text>
        </Box>
      </Box>
      <Box marginTop={SURFACE_GAP} flexWrap="wrap">
        {statusTokens.map((token) => {
          const chip = actionChipStyle({
            selected: true,
            disabled: false,
            tone: token.tone === "warning" ? "warning" : "accent",
          });
          const mutedChip =
            token.tone === "muted"
              ? { color: terminalTheme.ink.text, backgroundColor: terminalTheme.ink.panelBorder }
              : chip;
          return (
            <Box key={`${token.label}-${token.value}`} marginRight={CHIP_GAP}>
              <Text color={mutedChip.color} backgroundColor={mutedChip.backgroundColor}>
                {`${token.label}:${token.value}`}
              </Text>
            </Box>
          );
        })}
      </Box>
    </Box>
  );
}
