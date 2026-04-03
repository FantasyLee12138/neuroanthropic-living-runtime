import React, { useEffect, useMemo, useRef, useState } from "react";
import { Box, Text, useApp, useInput } from "ink";
import TextInput from "ink-text-input";

import { parseSlashCommand } from "./commands/slash.js";
import { PythonBridgeClient, createSessionId } from "./bridge/client.js";
import { buildConsoleViewportPlan } from "./consoleLayout.js";
import { formatCognitiveSummary, formatPanelBody } from "./panelSummary.js";
import { commitPrompt, createPromptHistoryState, movePromptCursor } from "./promptHistory.js";
import { addLocalLine, addUserLine, applyBridgeEvent, clearLines, createInitialUiState } from "./state/sessionStore.js";
import { formatStatusLine } from "./statusLine.js";
import { buildTranscriptWindow, moveTranscriptOffset } from "./transcriptViewport.js";
import type { PanelKey, UiLine, UiState } from "./types.js";

const HELP_TEXT =
  "Slash commands: /help /status /why /steps /tools /state /dream [cue] /pause /resume /abort /clear /compact /mode [value] /permissions [value] /model /exit";

function renderLine(line: UiLine, index: number): React.ReactNode {
  const color = line.kind === "user" ? "cyan" : line.kind === "assistant" ? "green" : line.kind === "error" ? "red" : "gray";
  const label = line.kind === "user" ? "You" : line.kind === "assistant" ? "NALR" : line.kind === "error" ? "Error" : "System";
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

export function App({ bridge, cwd }: { bridge: PythonBridgeClient; cwd: string }) {
  const { exit } = useApp();
  const stdoutWidth = process.stdout.columns ?? 120;
  const stdoutHeight = process.stdout.rows ?? 32;
  const [input, setInput] = useState("");
  const [uiState, setUiState] = useState<UiState>(createInitialUiState());
  const [promptHistoryState, setPromptHistoryState] = useState(createPromptHistoryState());
  const [panel, setPanel] = useState<PanelKey>(null);
  const [transcriptOffset, setTranscriptOffset] = useState(0);
  const sessionId = useMemo(() => createSessionId(), []);
  const pendingApproval = uiState.pendingApprovals[0] ?? null;
  const viewportPlan = buildConsoleViewportPlan({
    width: stdoutWidth,
    height: stdoutHeight,
    transcriptMode: uiState.transcriptMode,
    hasPanel: panel !== null,
    hasPendingApproval: pendingApproval !== null,
  });
  const transcriptSourceLines = uiState.transcriptMode === "compact"
    ? uiState.lines.filter((line) => line.kind === "user" || line.kind === "assistant" || line.kind === "error")
    : uiState.lines;
  const previousTranscriptCountRef = useRef(transcriptSourceLines.length);

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

  useInput((rawInput, key) => {
    const normalizedInput = rawInput.toLowerCase();
    if (key.ctrl && normalizedInput === "c") {
      bridge.send({ type: "control_command", session_id: sessionId, command: "abort" });
      return;
    }
    if (key.ctrl && normalizedInput === "d") {
      void bridge.closeSession(sessionId).finally(() => exit());
      return;
    }
    if (key.ctrl && normalizedInput === "l") {
      setUiState((current) => clearLines(current));
      return;
    }
    if (pendingApproval && (normalizedInput === "y" || normalizedInput === "n")) {
      bridge.send({
        type: "approve",
        session_id: sessionId,
        call_id: pendingApproval.callId,
        approved: normalizedInput === "y",
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
    }));
    setInput("");
    const parsed = parseSlashCommand(trimmed);
    if (!parsed) {
      setUiState((current) => addUserLine(current, trimmed));
      bridge.send({ type: "user_turn", session_id: sessionId, text: trimmed });
      return;
    }
    if (parsed.kind === "local") {
      if (parsed.command === "help") {
        setUiState((current) => addLocalLine(current, HELP_TEXT));
        return;
      }
      if (parsed.command === "clear") {
        setUiState((current) => clearLines(current));
        setTranscriptOffset(0);
        return;
      }
      if (parsed.command === "compact") {
        setUiState((current) => ({
          ...current,
          transcriptMode: current.transcriptMode === "compact" ? "full" : "compact"
        }));
        return;
      }
      if (parsed.command === "exit") {
        await bridge.closeSession(sessionId);
        exit();
      }
      return;
    }
    if (parsed.kind === "unknown") {
      setUiState((current) => addLocalLine(current, `Unknown slash command: /${parsed.command}`, "error"));
      return;
    }
    if (
      parsed.command === "status" ||
      parsed.command === "why" ||
      parsed.command === "steps" ||
      parsed.command === "tools" ||
      parsed.command === "state"
    ) {
      setPanel(parsed.command);
    }
    bridge.send({ type: "control_command", session_id: sessionId, command: parsed.command, value: parsed.value });
  }

  const transcriptWindow = buildTranscriptWindow(
    transcriptSourceLines,
    uiState.transcriptMode === "compact"
      ? Math.min(8, viewportPlan.transcriptLines)
      : viewportPlan.transcriptLines,
    transcriptOffset,
  );
  const visibleLines = transcriptWindow.lines;
  const cognitiveSnapshot = uiState.sidebarSnapshot?.cognitiveSnapshot;
  const sidebarSummary = formatCognitiveSummary(cognitiveSnapshot);

  return (
    <Box flexDirection="column">
      <Text color="green">NALR Terminal</Text>
      <Text color="gray">Repo-local autonomous planning shell. Type /help for controls.</Text>
      <Box marginTop={1} flexDirection={viewportPlan.layout === "split" ? "row" : "column"}>
        <Box flexDirection="column" flexGrow={1} marginRight={viewportPlan.layout === "split" ? 1 : 0}>
          <Box borderStyle="round" borderColor="cyan" paddingX={1} flexDirection="column">
            <Text color="cyan">
              {`Transcript${transcriptWindow.offset > 0 ? ` (Scrolled +${transcriptWindow.offset})` : " (Latest)"} `}
            </Text>
            {visibleLines.map((line, index) => renderLine(line, index))}
          </Box>
          {pendingApproval ? (
            <Box marginTop={1} borderStyle="round" borderColor="yellow" paddingX={1} flexDirection="column">
              <Text color="yellow">Pending Approval</Text>
              <Text>{`${pendingApproval.tool}${pendingApproval.summary ? ` - ${pendingApproval.summary}` : ""}`}</Text>
              {pendingApproval.actionPreview ? <Text color="gray">{pendingApproval.actionPreview}</Text> : null}
              <Text color="gray">Press `y` to approve or `n` to reject.</Text>
            </Box>
          ) : null}
          {panel ? (
            <Box marginTop={1} borderStyle="round" borderColor="gray" paddingX={1} flexDirection="column">
              <Text color="yellow">/{panel}</Text>
              <Text>{truncateBlock(formatPanelBody(uiState, panel) || "(empty)", viewportPlan.compactSidebar ? 6 : 10)}</Text>
            </Box>
          ) : null}
        </Box>
        <Box
          marginTop={viewportPlan.layout === "split" ? 0 : 1}
          width={viewportPlan.layout === "split" ? viewportPlan.sidebarWidth : undefined}
          flexDirection="column"
        >
          <Box borderStyle="round" borderColor="green" paddingX={1} flexDirection="column">
            <Text color="green">认知摘要</Text>
            <Text>{sidebarSummary}</Text>
          </Box>
        </Box>
      </Box>
      <Box marginTop={1} flexDirection="column">
        <Box>
          <Text color="cyan">&gt; </Text>
          <TextInput value={input} onChange={setInput} onSubmit={handleSubmit} />
        </Box>
        <Text color="gray">Use `\\` + Enter for multi-line input. PageUp/PageDown scroll transcript. Ctrl+C aborts, Ctrl+D exits, Ctrl+L clears.</Text>
      </Box>
      <Text color="gray">{formatStatusLine(uiState)}</Text>
    </Box>
  );
}
