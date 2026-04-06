import { describe, expect, it } from "vitest";

import { parseSlashCommand } from "./slash.js";

describe("parseSlashCommand", () => {
  it("routes runtime control commands through the bridge", () => {
    expect(parseSlashCommand("/status")).toEqual({ kind: "bridge", command: "status" });
    expect(parseSlashCommand("/why")).toEqual({ kind: "bridge", command: "why" });
    expect(parseSlashCommand("/abort")).toEqual({ kind: "bridge", command: "abort" });
    expect(parseSlashCommand("/model")).toEqual({ kind: "bridge", command: "model" });
    expect(parseSlashCommand("/state")).toEqual({ kind: "bridge", command: "state" });
    expect(parseSlashCommand("/dream")).toEqual({ kind: "bridge", command: "dream" });
    expect(parseSlashCommand("/probability")).toEqual({ kind: "bridge", command: "probability" });
  });

  it("keeps local shell commands local", () => {
    expect(parseSlashCommand("/help")).toEqual({ kind: "local", command: "help" });
    expect(parseSlashCommand("/clear")).toEqual({ kind: "local", command: "clear" });
    expect(parseSlashCommand("/exit")).toEqual({ kind: "local", command: "exit" });
    expect(parseSlashCommand("/compact")).toEqual({ kind: "local", command: "compact" });
  });

  it("supports optional command value payload for bridge control commands", () => {
    expect(parseSlashCommand("/mode plan")).toEqual({ kind: "bridge", command: "mode", value: "plan" });
    expect(parseSlashCommand("/permissions ask")).toEqual({ kind: "bridge", command: "permissions", value: "ask" });
    expect(parseSlashCommand("/mode")).toEqual({ kind: "bridge", command: "mode" });
    expect(parseSlashCommand("/dream tea")).toEqual({ kind: "bridge", command: "dream", value: "tea" });
    expect(parseSlashCommand("/probability action wander")).toEqual({
      kind: "bridge",
      command: "probability",
      value: "action wander",
    });
    expect(parseSlashCommand("/probability layer token")).toEqual({
      kind: "bridge",
      command: "probability",
      value: "layer token",
    });
  });

  it("returns null for plain chat input", () => {
    expect(parseSlashCommand("总结这个仓库结构")).toBeNull();
  });
});
