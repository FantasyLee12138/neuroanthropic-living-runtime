import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";

test("settings page includes a dedicated scheduled tasks section", () => {
  const html = fs.readFileSync(new URL("./index.html", import.meta.url), "utf8");

  assert.match(html, /id="settings-scheduled-title"/);
  assert.match(html, /id="settings-scheduled-summary"/);
  assert.match(html, /id="settings-scheduled-list"/);
});
