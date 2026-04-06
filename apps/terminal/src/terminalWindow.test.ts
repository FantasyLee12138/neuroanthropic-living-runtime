import { describe, expect, it, vi } from "vitest";

import { buildTerminalTitleSequence, setTerminalTitle } from "./terminalWindow.js";

describe("terminalWindow", () => {
  it("builds a standard OSC window-title sequence", () => {
    expect(buildTerminalTitleSequence("NALR")).toBe("\u001b]0;NALR\u0007");
  });

  it("writes the title sequence only for interactive terminals", () => {
    const write = vi.fn();

    setTerminalTitle("NALR", { isTTY: true, write, setProcessTitle: vi.fn() });
    expect(write).toHaveBeenCalledWith("\u001b]0;NALR\u0007");

    write.mockClear();
    setTerminalTitle("NALR", { isTTY: false, write, setProcessTitle: vi.fn() });
    expect(write).not.toHaveBeenCalled();
  });
});
