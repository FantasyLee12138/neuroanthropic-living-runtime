import { describe, expect, it } from "vitest";

import {
  actionChipStyle,
  borderColorForZone,
  paletteRowStyle,
  sidebarToneColor,
  transcriptLineStyle,
} from "./terminalTheme.js";

describe("terminalTheme", () => {
  it("maps transcript line kinds to restrained semantic colors", () => {
    expect(transcriptLineStyle("user")).toEqual({ color: "#78bced", label: "You" });
    expect(transcriptLineStyle("assistant")).toEqual({ color: "#8fd39a", label: "NALR" });
    expect(transcriptLineStyle("system")).toEqual({ color: "#7d8793", label: "System" });
    expect(transcriptLineStyle("error")).toEqual({ color: "#e07c75", label: "Error" });
  });

  it("uses cool focus borders for general focus and warm borders for approvals", () => {
    expect(borderColorForZone("transcript", false)).toBe("#2f3842");
    expect(borderColorForZone("transcript", true)).toBe("#4f6f8f");
    expect(borderColorForZone("approval", false)).toBe("#4a3c2a");
    expect(borderColorForZone("approval", true)).toBe("#c59d62");
  });

  it("keeps sidebar tones on the same muted palette family", () => {
    expect(sidebarToneColor("normal")).toBe("#d7dde5");
    expect(sidebarToneColor("muted")).toBe("#7d8793");
    expect(sidebarToneColor("warning")).toBe("#e2be74");
    expect(sidebarToneColor("accent")).toBe("#9ac2e6");
  });

  it("renders selected chips with tint backgrounds and disabled chips as muted text", () => {
    expect(actionChipStyle({ selected: true, disabled: false, tone: "accent" })).toEqual({
      color: "#0f1115",
      backgroundColor: "#9ac2e6",
    });
    expect(actionChipStyle({ selected: true, disabled: false, tone: "warning" })).toEqual({
      color: "#0f1115",
      backgroundColor: "#e2be74",
    });
    expect(actionChipStyle({ selected: false, disabled: true, tone: "accent" })).toEqual({
      color: "#7d8793",
      backgroundColor: undefined,
    });
  });

  it("gives palette rows low-noise defaults and a brighter selected state", () => {
    expect(paletteRowStyle(false, false)).toEqual({ color: "#d7dde5", backgroundColor: undefined });
    expect(paletteRowStyle(false, true)).toEqual({ color: "#7d8793", backgroundColor: undefined });
    expect(paletteRowStyle(true, false)).toEqual({ color: "#0f1115", backgroundColor: "#9ac2e6" });
  });
});
