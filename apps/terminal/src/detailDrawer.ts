import type { PanelKey } from "./types.js";

const DRAWER_ORDER: Exclude<PanelKey, null>[] = ["status", "why", "steps", "tools", "cognition", "approvals", "meta"];

export function toggleDetailDrawer(current: PanelKey, next: Exclude<PanelKey, null>): PanelKey {
  return current === next ? null : next;
}

export function cycleDetailDrawer(current: PanelKey, direction: "next" | "prev"): Exclude<PanelKey, null> {
  const index = current == null ? -1 : DRAWER_ORDER.indexOf(current);
  if (direction === "next") {
    return DRAWER_ORDER[(index + 1 + DRAWER_ORDER.length) % DRAWER_ORDER.length] ?? DRAWER_ORDER[0];
  }
  return DRAWER_ORDER[(index - 1 + DRAWER_ORDER.length) % DRAWER_ORDER.length] ?? DRAWER_ORDER[DRAWER_ORDER.length - 1];
}

export function cycleApprovalCursor(current: number, total: number, direction: "next" | "prev"): number {
  if (total <= 0) {
    return 0;
  }
  if (direction === "next") {
    return (current + 1) % total;
  }
  return (current - 1 + total) % total;
}
