import { describe, expect, it } from "vitest";

import { joinReadableLabel, toReadableLines } from "./readableText.js";

describe("readableText", () => {
  it("splits dense inline text into readable rows", () => {
    expect(toReadableLines("候选动作：respond(0.61)；plan(0.33)；rest(0.08)")).toEqual([
      "候选动作：respond(0.61)",
      "plan(0.33)",
      "rest(0.08)",
    ]);
  });

  it("preserves newline groups and trims blanks", () => {
    expect(toReadableLines("当前驱动：respond\n\n风险：high | 模式：ask")).toEqual([
      "当前驱动：respond",
      "风险：high",
      "模式：ask",
    ]);
  });

  it("formats labels as multi-line blocks", () => {
    expect(joinReadableLabel("贡献叠层", "前额叶执行控制(0.64)；前扣带冲突监控(0.52)")).toEqual([
      "贡献叠层：前额叶执行控制(0.64)",
      "  前扣带冲突监控(0.52)",
    ]);
  });
});
