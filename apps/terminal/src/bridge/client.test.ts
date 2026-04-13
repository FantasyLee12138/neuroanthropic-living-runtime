import { afterEach, describe, expect, it, vi } from "vitest";
import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";

import {
  createSessionId,
  DEFAULT_SESSION_CLOSE_TIMEOUT_MS,
  DEFAULT_SESSION_START_TIMEOUT_MS,
  registerBridgeProcessCleanup,
  resolveInteractiveSessionId,
  sessionCloseTimeoutMs,
  sessionStartTimeoutMs,
} from "./client.js";

function makeRepoRoot(): string {
  const root = mkdtempSync(path.join(tmpdir(), "nalr-client-"));
  mkdirSync(path.join(root, "src", "nalr"), { recursive: true });
  writeFileSync(path.join(root, "pyproject.toml"), "[project]\nname='nalr'\n", "utf8");
  return root;
}

describe("client session restore", () => {
  const created: string[] = [];

  afterEach(() => {
    for (const entry of created) {
      rmSync(entry, { recursive: true, force: true });
    }
    created.length = 0;
    delete process.env.NALR_HOME;
    delete process.env.NALR_SESSION_START_TIMEOUT_MS;
    delete process.env.NALR_SESSION_CLOSE_TIMEOUT_MS;
  });

  it("reuses current session when status is active or detached", () => {
    const repoRoot = makeRepoRoot();
    created.push(repoRoot);
    const home = path.join(repoRoot, ".alive");
    mkdirSync(path.join(home, "runtime"), { recursive: true });
    writeFileSync(
      path.join(home, "runtime", "current_terminal_session.json"),
      JSON.stringify({ session_id: "sess-detached", status: "detached" }),
      "utf8",
    );
    process.env.NALR_HOME = home;

    expect(resolveInteractiveSessionId(repoRoot)).toBe("sess-detached");
  });

  it("creates a new session when current session is ended or missing", () => {
    const repoRoot = makeRepoRoot();
    created.push(repoRoot);
    const home = path.join(repoRoot, ".alive");
    mkdirSync(path.join(home, "runtime"), { recursive: true });
    writeFileSync(
      path.join(home, "runtime", "current_terminal_session.json"),
      JSON.stringify({ session_id: "sess-ended", status: "ended" }),
      "utf8",
    );
    process.env.NALR_HOME = home;

    const resolved = resolveInteractiveSessionId(repoRoot);

    expect(resolved).not.toBe("sess-ended");
    expect(resolved).toMatch(/^nalr-/);
    expect(createSessionId("nalr")).toMatch(/^nalr-/);
  });

  it("uses longer default bridge timeouts and allows env overrides", () => {
    expect(sessionStartTimeoutMs()).toBe(DEFAULT_SESSION_START_TIMEOUT_MS);
    expect(sessionCloseTimeoutMs()).toBe(DEFAULT_SESSION_CLOSE_TIMEOUT_MS);

    process.env.NALR_SESSION_START_TIMEOUT_MS = "60000";
    process.env.NALR_SESSION_CLOSE_TIMEOUT_MS = "22000";

    expect(sessionStartTimeoutMs()).toBe(60000);
    expect(sessionCloseTimeoutMs()).toBe(22000);
  });

  it("disposes the bridge when the host process exits", () => {
    const dispose = vi.fn();
    const listeners = new Map<string, (...args: unknown[]) => void>();
    const processLike = {
      pid: 4321,
      once(event: string, listener: (...args: unknown[]) => void) {
        listeners.set(event, listener);
        return this;
      },
      off(event: string) {
        listeners.delete(event);
        return this;
      },
      kill: vi.fn(),
    };

    const unregister = registerBridgeProcessCleanup({ dispose }, processLike as any);
    listeners.get("exit")?.();

    expect(dispose).toHaveBeenCalledTimes(1);

    unregister();
  });

  it("disposes the bridge and re-raises termination signals", () => {
    const dispose = vi.fn();
    const listeners = new Map<string, (...args: unknown[]) => void>();
    const processLike = {
      pid: 9876,
      once(event: string, listener: (...args: unknown[]) => void) {
        listeners.set(event, listener);
        return this;
      },
      off(event: string) {
        listeners.delete(event);
        return this;
      },
      kill: vi.fn(),
    };

    registerBridgeProcessCleanup({ dispose }, processLike as any);
    listeners.get("SIGTERM")?.();

    expect(dispose).toHaveBeenCalledTimes(1);
    expect(processLike.kill).toHaveBeenCalledWith(9876, "SIGTERM");
    expect(listeners.has("SIGTERM")).toBe(false);
  });
});
