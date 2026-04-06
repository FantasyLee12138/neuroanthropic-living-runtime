import { describe, expect, it } from "vitest";

import { canUseDigitShortcut, shortcutDraftTarget } from "./shortcutDraft.js";

describe("shortcutDraftTarget", () => {
  it("preserves the main input draft for ctrl shortcuts outside the palette", () => {
    expect(shortcutDraftTarget({ ctrl: true, key: "p", paletteOpen: false })).toBe("input");
    expect(shortcutDraftTarget({ ctrl: true, key: "l", paletteOpen: false })).toBe("input");
    expect(shortcutDraftTarget({ ctrl: true, key: "j", paletteOpen: false })).toBe("input");
  });

  it("does not preserve a draft when the palette toggle is closing the palette", () => {
    expect(shortcutDraftTarget({ ctrl: true, key: "p", paletteOpen: true })).toBeNull();
  });

  it("ignores normal typing and unrelated keys", () => {
    expect(shortcutDraftTarget({ ctrl: false, key: "p", paletteOpen: false })).toBeNull();
    expect(shortcutDraftTarget({ ctrl: true, key: "x", paletteOpen: false })).toBeNull();
  });

  it("limits global digit shortcuts to focused menus instead of any empty input", () => {
    expect(canUseDigitShortcut({ paletteOpen: true, paletteQueryEmpty: true, focusZone: "input", inputEmpty: false })).toBe(true);
    expect(canUseDigitShortcut({ paletteOpen: true, paletteQueryEmpty: false, focusZone: "input", inputEmpty: false })).toBe(false);
    expect(canUseDigitShortcut({ paletteOpen: false, paletteQueryEmpty: true, focusZone: "actions", inputEmpty: true })).toBe(true);
    expect(canUseDigitShortcut({ paletteOpen: false, paletteQueryEmpty: true, focusZone: "approval", inputEmpty: true })).toBe(true);
    expect(canUseDigitShortcut({ paletteOpen: false, paletteQueryEmpty: true, focusZone: "input", inputEmpty: true })).toBe(false);
    expect(canUseDigitShortcut({ paletteOpen: false, paletteQueryEmpty: true, focusZone: "drawer", inputEmpty: true })).toBe(false);
  });
});
