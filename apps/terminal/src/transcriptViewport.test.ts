import { describe, expect, it } from "vitest";

import { buildTranscriptWindow, moveTranscriptOffset } from "./transcriptViewport.js";

describe("transcriptViewport", () => {
  const lines = Array.from({ length: 8 }, (_, index) => `line-${index + 1}`);

  it("shows the latest lines when offset is zero", () => {
    expect(buildTranscriptWindow(lines, 3, 0)).toEqual({
      lines: ["line-6", "line-7", "line-8"],
      offset: 0,
      maxOffset: 5
    });
  });

  it("shows older lines when scrolled up", () => {
    expect(buildTranscriptWindow(lines, 3, 2)).toEqual({
      lines: ["line-4", "line-5", "line-6"],
      offset: 2,
      maxOffset: 5
    });
  });

  it("moves the offset by page and clamps at both ends", () => {
    expect(moveTranscriptOffset({ currentOffset: 0, direction: "older", pageSize: 4, totalLines: 8, visibleCount: 3 })).toBe(4);
    expect(moveTranscriptOffset({ currentOffset: 4, direction: "older", pageSize: 4, totalLines: 8, visibleCount: 3 })).toBe(5);
    expect(moveTranscriptOffset({ currentOffset: 5, direction: "newer", pageSize: 4, totalLines: 8, visibleCount: 3 })).toBe(1);
    expect(moveTranscriptOffset({ currentOffset: 1, direction: "newer", pageSize: 4, totalLines: 8, visibleCount: 3 })).toBe(0);
  });
});
