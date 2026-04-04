import { describe, expect, it, vi } from "vitest";

import { createOneShotPrinter } from "./oneShotPrinter.js";

describe("createOneShotPrinter", () => {
  it("streams assistant tokens without duplicating the final message", () => {
    const write = vi.fn();
    const printer = createOneShotPrinter({ write });

    printer({
      type: "assistant_token",
      session_id: "sess-1",
      delta: "已进入只读任务处理。"
    });
    printer({
      type: "assistant_token",
      session_id: "sess-1",
      delta: "可用 /status /why /steps /tools 查看进度。"
    });
    printer({
      type: "assistant_final",
      session_id: "sess-1",
      message: "已进入只读任务处理。可用 /status /why /steps /tools 查看进度。"
    });

    expect(write.mock.calls).toEqual([
      ["NALR: 已进入只读任务处理。"],
      ["可用 /status /why /steps /tools 查看进度。"],
      ["\n"]
    ]);
  });

  it("falls back to the final assistant message when no token was streamed", () => {
    const write = vi.fn();
    const printer = createOneShotPrinter({ write });

    printer({
      type: "assistant_final",
      session_id: "sess-1",
      message: "你好，我在。你想让我帮你做什么？"
    });

    expect(write.mock.calls).toEqual([["NALR: 你好，我在。你想让我帮你做什么？\n"]]);
  });
});
