import { describe, expect, it } from "vitest";

import { cycleMenuIndex, flattenVisibleActions, isDigitSelection, splitActionGroups } from "./actionBar.js";
import type { UiAction } from "./types.js";

describe("actionBar helpers", () => {
  const primary: UiAction[] = [
    { id: "status", label: "状态", kind: "drawer", value: "status", disabled: false },
    { id: "why", label: "原因", kind: "drawer", value: "why", disabled: false },
  ];
  const secondary: UiAction[] = [
    { id: "abort", label: "中止", kind: "command", value: "abort", disabled: false },
    { id: "approvals", label: "审批", kind: "drawer", value: "approvals", disabled: true },
  ];
  const actions = flattenVisibleActions(primary, secondary);

  it("flattens action groups in display order", () => {
    expect(actions.map((item) => item.id)).toEqual(["status", "why", "abort", "approvals"]);
  });

  it("keeps primary and secondary actions grouped for rendering", () => {
    expect(splitActionGroups(primary, secondary)).toEqual({
      inspect: [...primary],
      control: [...secondary],
    });
  });

  it("cycles selection within bounds", () => {
    expect(cycleMenuIndex(0, actions.length, "next")).toBe(1);
    expect(cycleMenuIndex(3, actions.length, "next")).toBe(0);
    expect(cycleMenuIndex(0, actions.length, "prev")).toBe(3);
  });

  it("recognizes numeric shortcuts from 1 to 9", () => {
    expect(isDigitSelection("1")).toBe(0);
    expect(isDigitSelection("4")).toBe(3);
    expect(isDigitSelection("0")).toBeNull();
    expect(isDigitSelection("x")).toBeNull();
  });
});
