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
import { formatDetailSummary, formatSidebarSummary } from "./panelSummary.js";
import { commitPrompt, createPromptHistoryState, movePromptCursor } from "./promptHistory.js";
import { canUseDigitShortcut, shortcutDraftTarget } from "./shortcutDraft.js";
import { addLocalLine, addUserLine, applyBridgeEvent, clearLines, createInitialUiState } from "./state/sessionStore.js";
import { formatStatusLine, tokenizeStatusLine } from "./statusLine.js";
import { actionChipStyle, borderColorForZone, paletteRowStyle, sidebarToneColor, terminalTheme, transcriptLineStyle } from "./terminalTheme.js";
import { actionGroupLabel, actionTitle, approvalHeadline, paletteSectionLabel, paletteTitle } from "./terminalCopy.js";
import { buildTranscriptWindow, moveTranscriptOffset } from "./transcriptViewport.js";
import type { ActivityEntry, PanelKey, UiAction, UiLine, UiState } from "./types.js";

const HELP_TEXT = [
  "Controls",
  "  Tab focus zones | Ctrl+P command palette | PageUp/PageDown scroll | Ctrl+J / Ctrl+K cycle drawer",
  "  a approvals | s status | w why | t steps/tools | i cognition | m meta",
  "  [ / ] approval queue | 1-9 direct select | ← → move | Enter confirm",
  "",
  "Slash",
  "  /status /why /steps /tools /state /probability [round|action <name>|layer <name>]",
  "  /dream [cue] /pause /resume /abort /mode [value] /permissions [value] /model",
  "",
  "Local",
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
    step: "Step",
    tool: "Tool",
    result: "Result",
    approval: "Approval",
  }[entry.kind];
  const detail = entry.summary ? ` - ${entry.summary}` : "";
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
  return current;
}

function getPaletteContext(state: UiState): PaletteContext {
  if (state.pendingApprovals.length > 0) {
    return "approval";
  }

  const status = String(state.sidebarSnapshot?.runStatus ?? state.run?.status ?? "").toLowerCase();
  if (state.assistantStreamActive) {
    return "running";
  }
  if (!status) {
    return state.activeRunId ? "running" : "idle";
  }
  if (["completed", "done", "aborted", "failed", "error", "idle"].includes(status)) {
    return "idle";
  }
  return "running";
}

function paletteEmptyText(sectionId: "recommended" | "recent" | "all"): string {
  switch (sectionId) {
    case "recommended":
      return "No recommended commands.";
    case "recent":
      return "Run a command to populate recent items.";
    case "all":
      return "No commands available.";
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

  const paletteContext = getPaletteContext(uiState);
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
      setUiState((current) => addLocalLine(current, `Unknown slash command: /${parsed.command}`, "error"));
      return;
    }
    setUiState((current) => ({
      ...current,
      detailDrawer: panelForCommand(parsed.command, current.detailDrawer),
      focusZone: ["status", "why", "steps", "tools", "state", "model", "permissions", "mode"].includes(parsed.command) ? "drawer" : current.focusZone,
    }));
    bridge.send({ type: "control_command", session_id: effectiveSessionId, command: parsed.command, value: parsed.value });
  }

  const transcriptWindow = buildTranscriptWindow(transcriptSourceLines, viewportPlan.transcriptLines, transcriptOffset);
  const activityRows = uiState.activityRail.slice(-viewportPlan.sidebarLines);
  const visibleLines = transcriptWindow.lines;
  const helpHint = buildHelpHint(uiState, { paletteOpen });
  const statusTokens = tokenizeStatusLine(uiState);
  const transcriptBorder = borderColorForZone("transcript", uiState.focusZone === "transcript");
  const drawerBorder = borderColorForZone("drawer", uiState.focusZone === "drawer");
  const inputBorder = borderColorForZone("input", uiState.focusZone === "input");
  const actionBorder = borderColorForZone("actions", uiState.focusZone === "actions");
  const approvalBorder = borderColorForZone("approval", uiState.focusZone === "approval");

  const summaryBox = (
    <Box borderStyle="round" borderColor={terminalTheme.ink.panelBorder} paddingX={1} flexDirection="column">
      <Text color={terminalTheme.ink.subtle}>态势</Text>
      {sidebarSummary.map((item) => (
        <Text key={item.label} color={sidebarToneColor(item.tone)}>
          {item.label}: {item.value}
        </Text>
      ))}
    </Box>
  );

  const activityBox = (
    activityRows.length > 0 ? (
      <Box marginTop={1} borderStyle="round" borderColor={terminalTheme.ink.panelBorder} paddingX={1} flexDirection="column">
        <Text color={terminalTheme.ink.subtle}>Activity</Text>
        {activityRows.map((entry, index) => (
          <Text key={`${entry.kind}-${index}`} color={entry.kind === "approval" ? terminalTheme.ink.warning : terminalTheme.ink.muted}>
            {activityText(entry)}
          </Text>
        ))}
      </Box>
    ) : null
  );

  const drawerBox =
    uiState.detailDrawer ? (
      <Box marginTop={1} borderStyle="round" borderColor={drawerBorder} paddingX={1} flexDirection="column">
        <Text color={terminalTheme.ink.subtle}>{drawerTitle(uiState.detailDrawer)}</Text>
        <Text color={terminalTheme.ink.text}>{truncateBlock(detailBody || "(empty)", viewportPlan.detailLines || viewportPlan.sidebarLines)}</Text>
      </Box>
    ) : null;

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
                {`${index + 1}.${action.label}${action.disabled ? " (disabled)" : ""}`}
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
                  {`${index + 1}.${action.label}${action.disabled ? " (disabled)" : ""}`}
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
      {pendingApproval.actionPreview ? <Text color={terminalTheme.ink.muted}>{pendingApproval.actionPreview}</Text> : null}
      {(pendingApproval.riskLevel || pendingApproval.mode || pendingApproval.status) ? (
        <Text color={terminalTheme.ink.muted}>
          {[
            pendingApproval.riskLevel ? `risk:${pendingApproval.riskLevel}` : null,
            pendingApproval.mode ? `mode:${pendingApproval.mode}` : null,
            pendingApproval.status ? `status:${pendingApproval.status}` : null,
          ].filter(Boolean).join(" | ")}
        </Text>
      ) : null}
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
        <Text color={terminalTheme.ink.muted}>Press `y` to approve or `n` to reject.</Text>
      )}
    </Box>
  ) : null;

  const paletteBox = paletteOpen ? (
    <Box marginTop={SURFACE_GAP} borderStyle="round" borderColor={terminalTheme.ink.accentBorder} paddingX={1} flexDirection="column">
      <Text color={terminalTheme.ink.accentSoft}>{paletteTitle()}</Text>
      {paletteVisibleEntries.length === 0 ? (
        <Text color={terminalTheme.ink.muted}>No matches.</Text>
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
                    {`${shortcut}${entry.group} · ${entry.label}${entry.disabled ? " (disabled)" : ""}`}
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
      <Text color={terminalTheme.ink.text}>NALR</Text>
      <Text color={terminalTheme.ink.muted}>Runtime console first. Transcript stays primary; state and cognition stay inspectable.</Text>
      <Box marginTop={SURFACE_GAP} flexDirection={viewportPlan.layout === "split" ? "row" : "column"}>
        <Box flexDirection="column" flexGrow={1} marginRight={viewportPlan.layout === "split" ? 1 : 0}>
          <Box borderStyle="round" borderColor={transcriptBorder} paddingX={1} flexDirection="column">
            <Text color={terminalTheme.ink.subtle}>
              {`任务转录${transcriptWindow.offset > 0 ? ` (Scrolled +${transcriptWindow.offset})` : " (Latest)"}`}
            </Text>
            {pendingApproval ? <Text color={terminalTheme.ink.warning}>blocked on approval</Text> : null}
            {visibleLines.map((line, index) => renderLine(line, index))}
          </Box>
          {approvalBox}
        </Box>
        <Box
          marginTop={viewportPlan.layout === "split" ? 0 : SURFACE_GAP}
          width={viewportPlan.layout === "split" ? viewportPlan.sidebarWidth : undefined}
          flexDirection="column"
        >
          {summaryBox}
          {activityBox}
          {viewportPlan.detailPlacement === "side" ? drawerBox : null}
        </Box>
      </Box>
      {viewportPlan.detailPlacement === "bottom" ? drawerBox : null}
      {actionBarBox}
      {paletteBox}
      <Box marginTop={SURFACE_GAP} flexDirection="column">
        <Box borderStyle="round" borderColor={inputBorder} paddingX={1}>
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
