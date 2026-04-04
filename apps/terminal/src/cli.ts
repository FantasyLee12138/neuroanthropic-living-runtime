#!/usr/bin/env node

import React from "react";
import { render } from "ink";

import { App } from "./app.js";
import { PythonBridgeClient, createSessionId, resolveRepoRoot } from "./bridge/client.js";
import { createOneShotPrinter } from "./oneShotPrinter.js";

const DEFAULT_ONE_SHOT_TIMEOUT_MS = 70_000;

function oneShotTimeoutMs(): number {
  const raw = Number(process.env.NALR_ONE_SHOT_TIMEOUT_MS ?? DEFAULT_ONE_SHOT_TIMEOUT_MS);
  if (!Number.isFinite(raw)) {
    return DEFAULT_ONE_SHOT_TIMEOUT_MS;
  }
  return Math.max(1_000, Math.trunc(raw));
}

function printHelp(): void {
  console.log("NALR");
  console.log("");
  console.log("Usage:");
  console.log("  NALR                  Start interactive terminal shell");
  console.log('  NALR "总结这个仓库结构"  Run a one-shot planning session');
  console.log("  NALR Dream [cue]      Trigger a manual dream run");
  console.log("");
  console.log("Slash commands:");
  console.log("  /help /status /why /steps /tools /state /dream [cue] /pause /resume /abort /clear /compact /mode [value] /permissions [value] /model /exit");
}

async function runOneShot(prompt: string): Promise<number> {
  const launchCwd = process.env.NALR_LAUNCH_CWD ?? process.cwd();
  const repoRoot = resolveRepoRoot(launchCwd);
  const bridge = new PythonBridgeClient({ repoRoot, cwd: launchCwd });
  const sessionId = createSessionId();
  const printEvent = createOneShotPrinter({ write: (text) => process.stdout.write(text) });
  const dispose = bridge.onEvent((event) => {
    printEvent(event);
  });
  try {
    await bridge.startSession(sessionId, launchCwd, { persistCurrent: false });
    const finalEvent = bridge.waitFor(
      (event) => event.type === "assistant_final" || event.type === "error",
      oneShotTimeoutMs()
    );
    bridge.send({ type: "user_turn", session_id: sessionId, text: prompt });
    await finalEvent;
    await bridge.closeSession(sessionId);
    return 0;
  } finally {
    dispose();
    bridge.dispose();
  }
}

async function runOneShotControlCommand(command: "dream", value?: string): Promise<number> {
  const launchCwd = process.env.NALR_LAUNCH_CWD ?? process.cwd();
  const repoRoot = resolveRepoRoot(launchCwd);
  const bridge = new PythonBridgeClient({ repoRoot, cwd: launchCwd });
  const sessionId = createSessionId();
  const printEvent = createOneShotPrinter({ write: (text) => process.stdout.write(text) });
  const dispose = bridge.onEvent((event) => {
    printEvent(event);
  });
  try {
    await bridge.startSession(sessionId, launchCwd, { persistCurrent: false });
    const finalEvent = bridge.waitFor(
      (event) => event.type === "assistant_final" || event.type === "error",
      oneShotTimeoutMs()
    );
    bridge.send({ type: "control_command", session_id: sessionId, command, value });
    await finalEvent;
    await bridge.closeSession(sessionId);
    return 0;
  } finally {
    dispose();
    bridge.dispose();
  }
}

async function runInteractive(): Promise<void> {
  const launchCwd = process.env.NALR_LAUNCH_CWD ?? process.cwd();
  const repoRoot = resolveRepoRoot(launchCwd);
  const bridge = new PythonBridgeClient({ repoRoot, cwd: launchCwd });
  const instance = render(React.createElement(App, { bridge, cwd: launchCwd, repoRoot }));
  await instance.waitUntilExit();
}

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  if (args.includes("--help") || args.includes("-h")) {
    printHelp();
    return;
  }
  if (args.length > 0) {
    if (args[0]?.toLowerCase() === "dream") {
      const cue = args.slice(1).join(" ").trim() || undefined;
      process.exitCode = await runOneShotControlCommand("dream", cue);
      return;
    }
    process.exitCode = await runOneShot(args.join(" "));
    return;
  }
  await runInteractive();
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exitCode = 1;
});
