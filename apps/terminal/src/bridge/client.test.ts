import { afterEach, describe, expect, it } from "vitest";
import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";

import { createSessionId, resolveInteractiveSessionId } from "./client.js";

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
});
