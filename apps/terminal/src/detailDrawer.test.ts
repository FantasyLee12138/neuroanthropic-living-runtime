import { describe, expect, it } from "vitest";

import { cycleApprovalCursor, cycleDetailDrawer, toggleDetailDrawer } from "./detailDrawer.js";

describe("detailDrawer controls", () => {
  it("toggles the active drawer panel", () => {
    expect(toggleDetailDrawer(null, "status")).toBe("status");
    expect(toggleDetailDrawer("status", "status")).toBeNull();
    expect(toggleDetailDrawer("status", "tools")).toBe("tools");
  });

  it("cycles through the drawer panels in a stable order", () => {
    expect(cycleDetailDrawer(null, "next")).toBe("status");
    expect(cycleDetailDrawer("status", "next")).toBe("why");
    expect(cycleDetailDrawer("why", "next")).toBe("steps");
    expect(cycleDetailDrawer("status", "prev")).toBe("meta");
  });

  it("cycles approval focus within the queue bounds", () => {
    expect(cycleApprovalCursor(0, 0, "next")).toBe(0);
    expect(cycleApprovalCursor(0, 3, "next")).toBe(1);
    expect(cycleApprovalCursor(2, 3, "next")).toBe(0);
    expect(cycleApprovalCursor(0, 3, "prev")).toBe(2);
  });
});
