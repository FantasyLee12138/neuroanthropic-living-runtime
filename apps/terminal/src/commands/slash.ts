import type { BridgeControlCommand } from "../types.js";

export type ParsedSlashCommand =
  | { kind: "local"; command: "help" | "clear" | "exit" | "compact" }
  | { kind: "bridge"; command: BridgeControlCommand; value?: string }
  | { kind: "unknown"; command: string };

const BRIDGE_COMMANDS = new Set<BridgeControlCommand>([
  "status",
  "why",
  "steps",
  "tools",
  "state",
  "pause",
  "resume",
  "abort",
  "model",
  "mode",
  "permissions",
  "dream",
  "probability"
]);

const LOCAL_COMMANDS = new Set<"help" | "clear" | "exit" | "compact">(["help", "clear", "exit", "compact"]);

export function parseSlashCommand(input: string): ParsedSlashCommand | null {
  const trimmed = input.trim();
  if (!trimmed.startsWith("/")) {
    return null;
  }
  const raw = trimmed.slice(1).trim();
  const [head = "", ...rest] = raw.split(/\s+/);
  const command = head.toLowerCase();
  const value = rest.join(" ").trim() || undefined;
  if (!command) {
    return { kind: "local", command: "help" };
  }
  if (LOCAL_COMMANDS.has(command as "help" | "clear" | "exit" | "compact")) {
    return { kind: "local", command: command as "help" | "clear" | "exit" | "compact" };
  }
  if (BRIDGE_COMMANDS.has(command as BridgeControlCommand)) {
    return value
      ? { kind: "bridge", command: command as BridgeControlCommand, value }
      : { kind: "bridge", command: command as BridgeControlCommand };
  }
  return { kind: "unknown", command: value ? `${command} ${value}` : command };
}
