import type { UiAction } from "./types.js";

export function flattenVisibleActions(primary: UiAction[], secondary: UiAction[]): UiAction[] {
  return [...primary, ...secondary];
}

export function splitActionGroups(primary: UiAction[], secondary: UiAction[]): {
  inspect: UiAction[];
  control: UiAction[];
} {
  return {
    inspect: primary,
    control: secondary,
  };
}

export function cycleMenuIndex(current: number, total: number, direction: "next" | "prev"): number {
  if (total <= 0) {
    return 0;
  }
  return direction === "next" ? (current + 1) % total : (current - 1 + total) % total;
}

export function isDigitSelection(input: string): number | null {
  if (!/^[1-9]$/.test(input)) {
    return null;
  }
  return Number(input) - 1;
}
