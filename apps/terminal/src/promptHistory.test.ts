import { describe, expect, it } from "vitest";

import { commitPrompt, createPromptHistoryState, movePromptCursor } from "./promptHistory.js";

describe("promptHistory", () => {
  it("stores submitted prompts and traverses history with up/down", () => {
    let state = createPromptHistoryState();
    state = commitPrompt(state, "first question");
    state = commitPrompt(state, "second question");

    const up1 = movePromptCursor(state, "up", "");
    expect(up1.input).toBe("second question");
    const up2 = movePromptCursor(up1.state, "up", up1.input);
    expect(up2.input).toBe("first question");

    const down1 = movePromptCursor(up2.state, "down", up2.input);
    expect(down1.input).toBe("second question");
    const down2 = movePromptCursor(down1.state, "down", down1.input);
    expect(down2.input).toBe("");
  });

  it("ignores empty prompt submissions", () => {
    const state = commitPrompt(createPromptHistoryState(), "   ");
    expect(state.entries).toEqual([]);
  });
});
