import type { UiState } from "./types.js";

const DRAFT_PRESERVING_CTRL_KEYS = new Set(["p", "c", "d", "l", "j", "k"]);

export function shortcutDraftTarget(options: { ctrl: boolean; key: string; paletteOpen: boolean }): "input" | null {
  if (!options.ctrl) {
    return null;
  }
  if (!DRAFT_PRESERVING_CTRL_KEYS.has(options.key.toLowerCase())) {
    return null;
  }
  if (options.paletteOpen) {
    return null;
  }
  return "input";
}

export function canUseDigitShortcut(options: {
  paletteOpen: boolean;
  paletteQueryEmpty: boolean;
  focusZone: UiState["focusZone"];
  inputEmpty: boolean;
}): boolean {
  if (options.paletteOpen) {
    return options.paletteQueryEmpty;
  }
  if (!options.inputEmpty) {
    return false;
  }
  return options.focusZone === "actions" || options.focusZone === "approval";
}
