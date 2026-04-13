import { randomUUID } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
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

export const DEFAULT_SESSION_START_TIMEOUT_MS = 45_000;
export const DEFAULT_SESSION_CLOSE_TIMEOUT_MS = 15_000;

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

function timeoutFromEnv(rawValue: string | undefined, fallbackMs: number): number {
  const raw = Number(rawValue ?? fallbackMs);
  if (!Number.isFinite(raw)) {
    return fallbackMs;
  }
  return Math.max(1_000, Math.trunc(raw));
}

export function sessionStartTimeoutMs(): number {
  return timeoutFromEnv(process.env.NALR_SESSION_START_TIMEOUT_MS, DEFAULT_SESSION_START_TIMEOUT_MS);
}

export function sessionCloseTimeoutMs(): number {
  return timeoutFromEnv(process.env.NALR_SESSION_CLOSE_TIMEOUT_MS, DEFAULT_SESSION_CLOSE_TIMEOUT_MS);
}

function resolveNalrHome(repoRoot: string): string {
  return process.env.NALR_HOME ?? path.join(repoRoot, ".alive");
}

export function resolveInteractiveSessionId(repoRoot: string): string {
  const currentPath = path.join(resolveNalrHome(repoRoot), "runtime", "current_terminal_session.json");
  if (!existsSync(currentPath)) {
    return createSessionId();
  }
  try {
    const payload = JSON.parse(readFileSync(currentPath, "utf8")) as { session_id?: string; status?: string };
    if (payload.session_id && (payload.status === "active" || payload.status === "detached")) {
      return payload.session_id;
    }
  } catch {
    return createSessionId();
  }
  return createSessionId();
}

type CleanupProcessLike = {
  pid: number;
  once(event: string, listener: (...args: unknown[]) => void): unknown;
  off(event: string, listener: (...args: unknown[]) => void): unknown;
  kill(pid: number, signal?: NodeJS.Signals): boolean;
};

export function registerBridgeProcessCleanup(
  bridge: { dispose(): void },
  processLike: CleanupProcessLike = process,
): () => void {
  const listeners = new Map<string, (...args: unknown[]) => void>();

  const unregister = () => {
    for (const [event, listener] of listeners) {
      processLike.off(event, listener);
    }
    listeners.clear();
  };

  const register = (event: string, listener: (...args: unknown[]) => void) => {
    listeners.set(event, listener);
    processLike.once(event, listener);
  };

  register("exit", () => {
    bridge.dispose();
    unregister();
  });

  for (const signal of ["SIGINT", "SIGTERM", "SIGHUP"] as const) {
    register(signal, () => {
      bridge.dispose();
      unregister();
      processLike.kill(processLike.pid, signal);
    });
  }

  return unregister;
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

  async startSession(sessionId: string, cwd: string, options?: { persistCurrent?: boolean }): Promise<OutboundBridgeEvent> {
    this.ensureProcess();
    const started = this.waitFor(
      (event) => event.type === "session_started" && String(event.session.session_id ?? "") === sessionId,
      sessionStartTimeoutMs(),
    );
    this.send({ type: "start_session", session_id: sessionId, cwd, persist_current: options?.persistCurrent ?? true });
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

  async closeSession(sessionId: string, options?: { purge?: boolean; cleanupOld?: boolean }): Promise<void> {
    if (!this.child) {
      return;
    }
    const closed = this.waitFor(
      (event) => event.type === "session_ended" && String(event.session.session_id ?? "") === sessionId,
      sessionCloseTimeoutMs(),
    );
    this.send({
      type: "close_session",
      session_id: sessionId,
      purge: options?.purge ?? false,
      cleanup_old: options?.cleanupOld ?? false,
    });
    try {
      await closed;
    } finally {
      this.dispose();
    }
  }

  async detachSession(sessionId: string, options?: { transcriptMode?: "full" | "compact" }): Promise<void> {
    if (!this.child) {
      return;
    }
    const detached = this.waitFor(
      (event) =>
        event.type === "session_ended" &&
        String(event.session.session_id ?? "") === sessionId &&
        String(event.session.status ?? "") === "detached",
      sessionCloseTimeoutMs(),
    );
    this.send({ type: "close_session", session_id: sessionId, detach: true, transcript_mode: options?.transcriptMode });
    try {
      await detached;
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
      NALR_HOME: resolveNalrHome(this.options.repoRoot),
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
