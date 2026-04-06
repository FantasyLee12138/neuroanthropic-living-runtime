import type { UiLine, UiState } from "./types.js";

export const terminalTheme = {
  ink: {
    bg: "#0f1115",
    text: "#d7dde5",
    muted: "#7d8793",
    subtle: "#9aa4af",
    accent: "#78bced",
    accentSoft: "#9ac2e6",
    accentBorder: "#4f6f8f",
    panelBorder: "#2f3842",
    success: "#8fd39a",
    warning: "#e2be74",
    warningBorder: "#c59d62",
    warningMuted: "#4a3c2a",
    danger: "#e07c75",
    onTint: "#0f1115",
  },
} as const;

export function transcriptLineStyle(kind: UiLine["kind"]): { color: string; label: string } {
  switch (kind) {
    case "user":
      return { color: terminalTheme.ink.accent, label: "You" };
    case "assistant":
      return { color: terminalTheme.ink.success, label: "NALR" };
    case "error":
      return { color: terminalTheme.ink.danger, label: "Error" };
    case "system":
    default:
      return { color: terminalTheme.ink.muted, label: "System" };
  }
}

export function sidebarToneColor(tone: "normal" | "muted" | "warning" | "accent"): string {
  switch (tone) {
    case "muted":
      return terminalTheme.ink.muted;
    case "warning":
      return terminalTheme.ink.warning;
    case "accent":
      return terminalTheme.ink.accentSoft;
    case "normal":
    default:
      return terminalTheme.ink.text;
  }
}

export function borderColorForZone(zone: UiState["focusZone"], focused: boolean): string {
  if (zone === "approval") {
    return focused ? terminalTheme.ink.warningBorder : terminalTheme.ink.warningMuted;
  }
  return focused ? terminalTheme.ink.accentBorder : terminalTheme.ink.panelBorder;
}

export function actionChipStyle(options: {
  selected: boolean;
  disabled: boolean;
  tone: "accent" | "warning";
}): { color: string; backgroundColor?: string } {
  if (options.disabled) {
    return { color: terminalTheme.ink.muted, backgroundColor: undefined };
  }
  if (!options.selected) {
    return {
      color: options.tone === "warning" ? terminalTheme.ink.warning : terminalTheme.ink.accent,
      backgroundColor: undefined,
    };
  }
  return {
    color: terminalTheme.ink.onTint,
    backgroundColor: options.tone === "warning" ? terminalTheme.ink.warning : terminalTheme.ink.accentSoft,
  };
}

export function paletteRowStyle(selected: boolean, disabled: boolean): { color: string; backgroundColor?: string } {
  if (disabled) {
    return {
      color: selected ? terminalTheme.ink.onTint : terminalTheme.ink.muted,
      backgroundColor: selected ? terminalTheme.ink.accentSoft : undefined,
    };
  }
  return {
    color: selected ? terminalTheme.ink.onTint : terminalTheme.ink.text,
    backgroundColor: selected ? terminalTheme.ink.accentSoft : undefined,
  };
}
