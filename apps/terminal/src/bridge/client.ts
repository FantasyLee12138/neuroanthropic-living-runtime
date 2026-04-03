import { randomUUID } from "node:crypto";
import { existsSync } from "node:fs";
import path from "node:path";
import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";

import { createJsonLineParser } from "./jsonLines.js";
import type { InboundBridgeEvent, OutboundBridgeEvent } from "../types.js";

type EventListener = (event: OutboundBridgeEvent) => void;
type Waiter = {
  predicate: (event: OutboundBridgeEvent) => boolean;
  resolve: (event: OutboundBridgeEvent) => void;
  reject: (error: Error) => void;
  timeout: NodeJS.Timeout;
};

export function resolveRepoRoot(startCwd: string): string {
  if (process.env.NALR_REPO_ROOT) {
    return process.env.NALR_REPO_ROOT;
  }
  let current = path.resolve(startCwd);
  while (true) {
    if (existsSync(path.join(current, "pyproject.toml")) && existsSync(path.join(current, "src", "nalr"))) {
      return current;
    }
    const parent = path.dirname(current);
    if (parent === current) {
      return startCwd;
    }
    current = parent;
  }
}

export function createSessionId(prefix = "nalr"): string {
  return `${prefix}-${randomUUID().slice(0, 8)}`;
}

export class PythonBridgeClient {
  private child: ChildProcessWithoutNullStreams | null = null;
  private listeners = new Set<EventListener>();
  private waiters: Waiter[] = [];

  constructor(
    private readonly options: {
      repoRoot: string;
      cwd: string;
    }
  ) {}

  async startSession(sessionId: string, cwd: string): Promise<OutboundBridgeEvent> {
    this.ensureProcess();
    const started = this.waitFor((event) => event.type === "session_started" && String(event.session.session_id ?? "") === sessionId);
    this.send({ type: "start_session", session_id: sessionId, cwd });
    return started;
  }

  onEvent(listener: EventListener): () => void {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  }

  send(event: InboundBridgeEvent): void {
    this.ensureProcess();
    this.child?.stdin.write(JSON.stringify(event) + "\n");
  }

  waitFor(predicate: (event: OutboundBridgeEvent) => boolean, timeoutMs = 8000): Promise<OutboundBridgeEvent> {
    return new Promise<OutboundBridgeEvent>((resolve, reject) => {
      const timeout = setTimeout(() => {
        this.waiters = this.waiters.filter((item) => item.timeout !== timeout);
        reject(new Error("bridge event timeout"));
      }, timeoutMs);
      this.waiters.push({ predicate, resolve, reject, timeout });
    });
  }

  async closeSession(sessionId: string): Promise<void> {
    if (!this.child) {
      return;
    }
    const closed = this.waitFor((event) => event.type === "session_ended" && String(event.session.session_id ?? "") === sessionId, 4000);
    this.send({ type: "close_session", session_id: sessionId });
    try {
      await closed;
    } finally {
      this.dispose();
    }
  }

  dispose(): void {
    for (const waiter of this.waiters) {
      clearTimeout(waiter.timeout);
      waiter.reject(new Error("bridge disposed"));
    }
    this.waiters = [];
    if (this.child) {
      this.child.kill();
      this.child = null;
    }
  }

  private ensureProcess(): void {
    if (this.child) {
      return;
    }
    const pythonBin = this.resolvePythonBin();
    const env = {
      ...process.env,
      NALR_HOME: process.env.NALR_HOME ?? path.join(this.options.repoRoot, ".alive"),
      NALR_CONFIG_DIR: process.env.NALR_CONFIG_DIR ?? path.join(this.options.repoRoot, "config"),
      PYTHONPATH: process.env.PYTHONPATH
        ? `${path.join(this.options.repoRoot, "src")}:${process.env.PYTHONPATH}`
        : path.join(this.options.repoRoot, "src")
    };
    this.child = spawn(pythonBin, ["-m", "nalr.terminal_bridge.bridge"], {
      cwd: this.options.cwd,
      env,
      stdio: "pipe"
    });
    const feed = createJsonLineParser((payload) => {
      this.dispatchEvent(payload as OutboundBridgeEvent);
    });
    this.child.stdout.setEncoding("utf8");
    this.child.stdout.on("data", (chunk: string) => feed(chunk));
    this.child.stderr.setEncoding("utf8");
    this.child.stderr.on("data", (chunk: string) => {
      this.dispatchEvent({ type: "error", message: chunk.trim() });
    });
  }

  private resolvePythonBin(): string {
    if (process.env.NALR_PYTHON_BIN) {
      return process.env.NALR_PYTHON_BIN;
    }
    const repoPython = path.join(this.options.repoRoot, ".venv", "bin", "python");
    if (existsSync(repoPython)) {
      return repoPython;
    }
    return process.env.PYTHON ?? "python3";
  }

  private dispatchEvent(event: OutboundBridgeEvent): void {
    for (const listener of this.listeners) {
      listener(event);
    }
    const remaining: Waiter[] = [];
    for (const waiter of this.waiters) {
      if (waiter.predicate(event)) {
        clearTimeout(waiter.timeout);
        waiter.resolve(event);
      } else {
        remaining.push(waiter);
      }
    }
    this.waiters = remaining;
  }
}
