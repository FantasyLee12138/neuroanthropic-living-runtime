import { describe, expect, it, vi } from "vitest";

import { createJsonLineParser } from "./jsonLines.js";

describe("createJsonLineParser", () => {
  it("reassembles chunked stdio frames into parsed events", () => {
    const seen = vi.fn();
    const feed = createJsonLineParser(seen);

    feed('{"type":"session_started","session":{"session_id":"sess-1"}}\n{"type":"run_');
    feed('status","run":{"run_id":"run-1"}}\n');

    expect(seen).toHaveBeenCalledTimes(2);
    expect(seen.mock.calls[0]?.[0]).toEqual({
      type: "session_started",
      session: { session_id: "sess-1" }
    });
    expect(seen.mock.calls[1]?.[0]).toEqual({
      type: "run_status",
      run: { run_id: "run-1" }
    });
  });
});
