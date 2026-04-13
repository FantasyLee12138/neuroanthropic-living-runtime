# Analysis Trend Chinese Labels And Replay Timeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the analysis trend panel speak human-readable Chinese, place `u_base` / `p_final` / `strength` together with conflict and repair markers on one round timeline, and add lightweight motion plus replay controls.

**Architecture:** Keep the existing trend panel and SVG renderer, but enrich the view-model with Chinese display metadata and replay frames. Extend the render layer to draw same-axis conflict/repair tracks, a moving round cursor, replay controls, and a horizontally scrollable explanation strip. Wire replay state in `main.js` and keep all new motion gated by the existing `data-motion="reduced"` behavior.

**Tech Stack:** Vanilla ESM JavaScript, HTML, CSS, Node test runner (`node --test`), pytest + Playwright

---

### Task 1: Enrich the Trend View-Model With Chinese Labels and Replay Frames

**Files:**
- Modify: `apps/workbench/src/view-models.js`
- Create: `apps/workbench/src/view-models.test.js`

- [ ] **Step 1: Write the failing unit tests**

```js
import test from "node:test";
import assert from "node:assert/strict";

import { buildAnalysisReplayFrames, buildAnalysisTrendSeries } from "./view-models.js";

const recentRounds = [
  { round_id: 7, sampled_action: "respond" },
  { round_id: 8, sampled_action: "wait" },
];

const roundCatalog = {
  "7": {
    trace: {
      sampled_action: "respond",
      action_bookkeeping: {
        u_base: { respond: -0.131 },
        ci: { respond: 0.2 },
      },
      probability_field: {
        action: {
          winner_posterior: { respond: 0.791 },
        },
      },
      gate_decisions: [
        {
          stage: "identity_guard",
          owner: "身份校验",
          allowed: false,
          requires_resample: true,
          reason: "表达越界，需要重新采样。",
        },
      ],
      conflict_arbitration: {
        critical_conflict: true,
        dominant_conflicts: ["身份校验"],
        repair_transition: {
          to_stage: "恢复阶段",
          reason: "完成修复后切回恢复流程。",
        },
        repair_ledger_tail: [{ reason: "回退输出" }],
      },
      initiative: {
        memory_backing: {
          strength: 0.633,
          cue: "关系边界",
        },
      },
      rendered_expression: {
        text: "先收住，再重新组织表达。",
      },
    },
  },
  "8": {
    trace: {
      sampled_action: "wait",
      action_bookkeeping: {
        u_base: { wait: 0.112 },
      },
      probability_field: {
        action: {
          winner_posterior: { wait: 0.33 },
        },
      },
      initiative: {
        memory_backing: {
          strength: 0.41,
          cue: "暂缓推进",
        },
      },
      rendered_expression: {
        text: "先观察，再决定是否继续。",
      },
    },
  },
};

test("buildAnalysisTrendSeries exposes Chinese-first labels and display sources", () => {
  const [uBase, pFinal, strength] = buildAnalysisTrendSeries({ recentRounds, roundCatalog });

  assert.equal(uBase.displayLabel, "基础驱动力");
  assert.equal(uBase.displaySubLabel, "u_base");
  assert.equal(uBase.points[0].displaySource, "动作原始驱动力");

  assert.equal(pFinal.displayLabel, "最终胜出概率");
  assert.equal(pFinal.points[0].displaySource, "概率场最终裁决");

  assert.equal(strength.displayLabel, "习惯牵引强度");
  assert.equal(strength.points[0].displaySource, "习惯 / 记忆牵引");
});

test("buildAnalysisReplayFrames aligns conflict and repair markers to the same round timeline", () => {
  const trendSeries = buildAnalysisTrendSeries({ recentRounds, roundCatalog });
  const frames = buildAnalysisReplayFrames(trendSeries);

  assert.equal(frames.length, 2);
  assert.equal(frames[0].roundLabel, "第 7 轮");
  assert.equal(frames[0].summary, "先收住，再重新组织表达。");
  assert.ok(frames[0].markers.some((item) => item.track === "conflict" && item.label === "冲突 · 身份校验"));
  assert.ok(frames[0].markers.some((item) => item.track === "repair" && item.label === "修复 · 重新采样"));
  assert.equal(frames[0].metrics[0].displayLabel, "基础驱动力");
  assert.equal(frames[0].metrics[1].displayLabel, "最终胜出概率");
  assert.equal(frames[0].metrics[2].displayLabel, "习惯牵引强度");
});
```

- [ ] **Step 2: Run the unit test to verify it fails**

Run: `node --test apps/workbench/src/view-models.test.js`
Expected: FAIL because `buildAnalysisReplayFrames` does not exist yet and the current trend series still returns raw English labels / raw path sources.

- [ ] **Step 3: Implement the smallest view-model change that makes the tests pass**

```js
const ANALYSIS_METRIC_META = {
  u_base: {
    displayLabel: "基础驱动力",
    displaySubLabel: "u_base",
    displaySource: "动作原始驱动力",
  },
  p_final: {
    displayLabel: "最终胜出概率",
    displaySubLabel: "p_final",
    displaySource: "概率场最终裁决",
  },
  strength: {
    displayLabel: "习惯牵引强度",
    displaySubLabel: "strength",
    displaySource: "习惯 / 记忆牵引",
  },
};

function metricMeta(key) {
  return ANALYSIS_METRIC_META[key] || {
    displayLabel: key,
    displaySubLabel: key,
    displaySource: "",
  };
}

function mapTrendTimelineMarkers(roundId, markerView = {}) {
  return [
    ...(markerView.conflictMarkers || []).map((marker, index) => ({
      id: `round-${roundId}-conflict-${index + 1}`,
      roundId,
      track: "conflict",
      label: marker.label,
      detail: marker.detail,
      tone: marker.tone,
    })),
    ...(markerView.repairMarkers || []).map((marker, index) => ({
      id: `round-${roundId}-repair-${index + 1}`,
      roundId,
      track: "repair",
      label: marker.label,
      detail: marker.detail,
      tone: marker.tone,
    })),
  ];
}

export function buildAnalysisTrendSeries({ recentRounds = [], roundCatalog = {} } = {}) {
  const orderedRounds = asArray(recentRounds)
    .filter((round) => Number.isFinite(Number(round?.round_id)))
    .slice()
    .sort((a, b) => Number(a.round_id) - Number(b.round_id));
  const series = ["u_base", "p_final", "strength"].map((key) => ({
    key,
    ...metricMeta(key),
    points: [],
  }));

  orderedRounds.forEach((round) => {
    const roundId = Number(round.round_id);
    const detail = roundCatalog[String(roundId)] || {};
    const trace = tracePayload(detail);
    const markerView = buildConflictRepairMarkers({ roundId, selectedPayload: detail });
    const timelineMarkers = mapTrendTimelineMarkers(roundId, markerView);
    const summary = firstText(
      trace.rendered_expression?.text,
      markerView.conflictMarkers[0]?.detail,
      markerView.repairMarkers[0]?.detail,
      "这一轮没有额外的自由文本说明。",
    );
    const basePoint = {
      roundId,
      action: firstText(trace.sampled_action, round.sampled_action),
      hasConflict: markerView.conflictMarkers.length > 0,
      hasRepair: markerView.repairMarkers.length > 0,
      timelineMarkers,
      summary,
    };

    series[0].points.push({
      ...basePoint,
      value: finiteNumber(trace.action_bookkeeping?.u_base?.[basePoint.action]),
      source: formatMetricSource("u_base"),
      displaySource: metricMeta("u_base").displaySource,
    });
    series[1].points.push({
      ...basePoint,
      value:
        finiteNumber(trace.probability_field?.action?.winner_posterior?.[basePoint.action]) ??
        finiteNumber(trace.candidate_distribution?.[basePoint.action]),
      source: formatMetricSource("p_final"),
      displaySource: metricMeta("p_final").displaySource,
    });
    series[2].points.push({
      ...basePoint,
      value: deriveHabitStrength(trace),
      source: formatMetricSource("strength"),
      displaySource: metricMeta("strength").displaySource,
    });
  });

  return series;
}

export function buildAnalysisReplayFrames(trendSeries = []) {
  const baseSeries = trendSeries[0]?.points || [];
  return baseSeries.map((point, index) => ({
    frameId: `round-${point.roundId}`,
    roundId: point.roundId,
    roundLabel: `第 ${point.roundId} 轮`,
    action: point.action,
    summary: point.summary,
    markers: point.timelineMarkers || [],
    metrics: trendSeries.map((series) => ({
      key: series.key,
      displayLabel: series.displayLabel,
      displaySubLabel: series.displaySubLabel,
      value: series.points[index]?.value ?? null,
    })),
  }));
}
```

- [ ] **Step 4: Run the unit test to verify it passes**

Run: `node --test apps/workbench/src/view-models.test.js`
Expected: PASS

- [ ] **Step 5: Commit the view-model slice**

```bash
git add apps/workbench/src/view-models.js apps/workbench/src/view-models.test.js
git commit -m "feat: add chinese trend labels and replay frames"
```

### Task 2: Extend the Render Layer for Chinese Labels, Unified Marker Tracks, and Replay UI

**Files:**
- Modify: `apps/workbench/src/analysis-brainflow.js`
- Create: `apps/workbench/src/analysis-brainflow.test.js`

- [ ] **Step 1: Write the failing render tests**

```js
import test from "node:test";
import assert from "node:assert/strict";

import {
  renderTrendFrameStrip,
  renderTrendLegend,
  renderTrendReplayControls,
  renderTrendStats,
  renderTrendSvg,
} from "./analysis-brainflow.js";

const trendSeries = [
  {
    key: "u_base",
    displayLabel: "基础驱动力",
    displaySubLabel: "u_base",
    points: [
      {
        roundId: 7,
        value: 0.41,
        displaySource: "动作原始驱动力",
        hasConflict: true,
        hasRepair: true,
        timelineMarkers: [
          { id: "c1", track: "conflict", label: "冲突 · 身份校验", detail: "表达越界" },
          { id: "r1", track: "repair", label: "修复 · 重新采样", detail: "重新采样" },
        ],
      },
      {
        roundId: 8,
        value: 0.34,
        displaySource: "动作原始驱动力",
        hasConflict: false,
        hasRepair: false,
        timelineMarkers: [],
      },
    ],
  },
  {
    key: "p_final",
    displayLabel: "最终胜出概率",
    displaySubLabel: "p_final",
    points: [
      { roundId: 7, value: 0.79, displaySource: "概率场最终裁决", hasConflict: true, hasRepair: true, timelineMarkers: [] },
      { roundId: 8, value: 0.33, displaySource: "概率场最终裁决", hasConflict: false, hasRepair: false, timelineMarkers: [] },
    ],
  },
  {
    key: "strength",
    displayLabel: "习惯牵引强度",
    displaySubLabel: "strength",
    points: [
      { roundId: 7, value: 0.63, displaySource: "习惯 / 记忆牵引", hasConflict: true, hasRepair: true, timelineMarkers: [] },
      { roundId: 8, value: 0.41, displaySource: "习惯 / 记忆牵引", hasConflict: false, hasRepair: false, timelineMarkers: [] },
    ],
  },
];

const frames = [
  {
    frameId: "round-7",
    roundId: 7,
    roundLabel: "第 7 轮",
    summary: "先收住，再重新组织表达。",
    markers: [
      { id: "c1", track: "conflict", label: "冲突 · 身份校验", detail: "表达越界" },
      { id: "r1", track: "repair", label: "修复 · 重新采样", detail: "重新采样" },
    ],
    metrics: [
      { key: "u_base", displayLabel: "基础驱动力", displaySubLabel: "u_base", value: 0.41 },
      { key: "p_final", displayLabel: "最终胜出概率", displaySubLabel: "p_final", value: 0.79 },
      { key: "strength", displayLabel: "习惯牵引强度", displaySubLabel: "strength", value: 0.63 },
    ],
  },
];

test("renderTrendLegend and renderTrendStats use Chinese-first copy", () => {
  const legend = renderTrendLegend(trendSeries);
  const stats = renderTrendStats(trendSeries);

  assert.match(legend, /基础驱动力/);
  assert.match(legend, /trend-legend-sub">u_base</);
  assert.match(stats, /习惯牵引强度/);
  assert.match(stats, /习惯 \/ 记忆牵引/);
  assert.doesNotMatch(stats, /action_bookkeeping\.u_base/);
});

test("renderTrendSvg draws the unified marker tracks and active cursor", () => {
  const svg = renderTrendSvg(trendSeries, {
    selectedRoundId: 7,
    activeRoundId: 7,
    replayRoundId: 7,
  });

  assert.match(svg, /trend-timeline-cursor/);
  assert.match(svg, /data-trend-round-id="7"/);
  assert.match(svg, /trend-point is-active/);
  assert.match(svg, /trend-marker-track is-conflict/);
  assert.match(svg, /trend-marker-track is-repair/);
});

test("renderTrendReplayControls and renderTrendFrameStrip expose replay affordances", () => {
  const controls = renderTrendReplayControls({ hasFrames: true, isPlaying: false, hasStarted: true });
  const frameStrip = renderTrendFrameStrip(frames, { activeRoundId: 7 });

  assert.match(controls, /data-trend-replay-action="start"/);
  assert.match(controls, /回放/);
  assert.match(controls, /继续/);
  assert.match(frameStrip, /trend-frame-card is-active/);
  assert.match(frameStrip, /冲突 · 身份校验/);
});
```

- [ ] **Step 2: Run the render test to verify it fails**

Run: `node --test apps/workbench/src/analysis-brainflow.test.js`
Expected: FAIL because the render layer still prints raw labels, has no replay controls, and only draws simple top-of-axis conflict / repair dots.

- [ ] **Step 3: Implement the render changes**

```js
export function renderTrendLegend(trendSeries = []) {
  if (!trendSeries.length) {
    return `<div class="empty-state">等待曲线数据。</div>`;
  }
  return trendSeries
    .map((series) => {
      const style = SERIES_STYLE[series.key] || SERIES_STYLE.p_final;
      return `<div class="trend-legend-item">
        <span class="trend-legend-swatch" style="--swatch:${escapeHtml(style.color)}"></span>
        <span class="trend-legend-copy">
          <strong>${escapeHtml(series.displayLabel || series.label)}</strong>
          <span class="trend-legend-sub">${escapeHtml(series.displaySubLabel || series.key)}</span>
        </span>
      </div>`;
    })
    .join("");
}

export function renderTrendStats(trendSeries = []) {
  if (!trendSeries.length) {
    return `<div class="empty-state">等待参数汇总。</div>`;
  }
  return trendSeries
    .map((series) => {
      const numeric = series.points.filter((point) => clampUnit(point.value) !== null);
      const latest = numeric[numeric.length - 1] || null;
      const previous = numeric[numeric.length - 2] || null;
      const delta = latest && previous ? latest.value - previous.value : null;
      return `<article class="trend-stat-card">
        <span>${escapeHtml(series.displayLabel || series.label)}</span>
        <strong>${escapeHtml(latest ? formatNumber(latest.value, 3) : "--")}</strong>
        <div class="trend-stat-meta">
          <span>${escapeHtml(latest?.displaySource || "--")}</span>
          <span>${escapeHtml(delta === null ? "Δ --" : `Δ ${delta >= 0 ? "+" : ""}${formatNumber(delta, 3)}`)}</span>
        </div>
      </article>`;
    })
    .join("");
}

export function renderTrendReplayControls({ hasFrames = false, isPlaying = false, hasStarted = false } = {}) {
  return `<div class="trend-replay-controls">
    <button type="button" class="analysis-toolbar-button" data-trend-replay-action="start" ${hasFrames ? "" : "disabled"}>回放</button>
    <button type="button" class="analysis-toolbar-button" data-trend-replay-action="pause" ${isPlaying ? "" : "disabled"}>暂停</button>
    <button type="button" class="analysis-toolbar-button" data-trend-replay-action="resume" ${hasStarted && !isPlaying ? "" : "disabled"}>继续</button>
    <button type="button" class="analysis-toolbar-button" data-trend-replay-action="reset" ${hasStarted ? "" : "disabled"}>重置</button>
  </div>`;
}

export function renderTrendFrameStrip(frames = [], { activeRoundId = null } = {}) {
  if (!frames.length) {
    return `<div class="empty-state">选中或回放某一轮后，这里会同步显示该轮说明。</div>`;
  }
  return `<div class="trend-frame-strip">${frames
    .map((frame) => `<article class="trend-frame-card${Number(frame.roundId) === Number(activeRoundId) ? " is-active" : ""}" data-trend-round-id="${escapeHtml(frame.roundId)}">
      <div class="trend-frame-head">
        <strong>${escapeHtml(frame.roundLabel)}</strong>
        <span>${escapeHtml(`marker ${frame.markers.length}`)}</span>
      </div>
      <p>${escapeHtml(frame.summary)}</p>
      <div class="trend-frame-marker-row">${frame.markers
        .map((marker) => `<span class="trend-frame-marker tone-${escapeHtml(marker.track)}">${escapeHtml(marker.label)}</span>`)
        .join("")}</div>
    </article>`)
    .join("")}</div>`;
}

export function renderTrendSvg(trendSeries = [], { selectedRoundId = null, activeRoundId = null, replayRoundId = null } = {}) {
  const basePoints = trendSeries[0]?.points || [];
  if (!basePoints.length) {
    return `<div class="empty-state">等待时序数据。</div>`;
  }

  const width = 1040;
  const height = 352;
  const padding = { top: 58, right: 34, bottom: 48, left: 44 };
  const denominator = Math.max(basePoints.length - 1, 1);
  const scaledSeries = trendSeries.map((series) => ({
    ...series,
    points: series.points.map((point, index) => ({
      ...point,
      x: denominator === 0 ? 0 : index / denominator,
    })),
  }));
  const currentRoundId = Number(replayRoundId ?? activeRoundId ?? selectedRoundId);
  const currentPoint =
    scaledSeries[0]?.points.find((point) => Number(point.roundId) === currentRoundId) || scaledSeries[0]?.points[0] || null;

  return `<svg class="trend-chart-svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="analysis trends">
    ${currentPoint
      ? `<line class="trend-timeline-cursor" x1="${(padding.left + currentPoint.x * (width - padding.left - padding.right)).toFixed(2)}" y1="${padding.top - 30}" x2="${(padding.left + currentPoint.x * (width - padding.left - padding.right)).toFixed(2)}" y2="${height - padding.bottom + 6}"></line>`
      : ""}
    ${scaledSeries
      .map((series) => `<g class="trend-series trend-series-${escapeHtml(series.key)}">
        <path pathLength="1" d="${escapeHtml(pathForPoints(series.points, width, height, padding))}" style="--series-color:${escapeHtml((SERIES_STYLE[series.key] || SERIES_STYLE.p_final).color)}"></path>
        ${series.points
          .filter((point) => clampUnit(point.value) !== null)
          .map((point) => {
            const x = padding.left + point.x * (width - padding.left - padding.right);
            const y = padding.top + (1 - clampUnit(point.value)) * (height - padding.top - padding.bottom);
            const isActive = Number(point.roundId) === currentRoundId;
            return `<g class="trend-point${isActive ? " is-active" : ""}" data-trend-round-id="${escapeHtml(point.roundId)}" style="--series-color:${escapeHtml((SERIES_STYLE[series.key] || SERIES_STYLE.p_final).color)}">
              <circle cx="${x.toFixed(2)}" cy="${y.toFixed(2)}" r="${isActive ? 5.6 : 3.8}">
                <title>round ${escapeHtml(point.roundId)} · ${escapeHtml(series.displayLabel || series.label)} ${escapeHtml(formatNumber(point.value, 3))}</title>
              </circle>
            </g>`;
          })
          .join("")}
      </g>`)
      .join("")}
    ${scaledSeries[0].points
      .map((point) => (point.timelineMarkers || [])
        .map((marker, trackIndex) => {
          const y = marker.track === "conflict" ? padding.top - 18 - trackIndex * 12 : padding.top - 2 - trackIndex * 12;
          return `<circle class="trend-marker-track is-${escapeHtml(marker.track)}${Number(point.roundId) === currentRoundId ? " is-active" : ""}" data-trend-round-id="${escapeHtml(point.roundId)}" cx="${(padding.left + point.x * (width - padding.left - padding.right)).toFixed(2)}" cy="${y.toFixed(2)}" r="5">
            <title>${escapeHtml(marker.label)}</title>
          </circle>`;
        })
        .join(""))
      .join("")}
  </svg>`;
}
```

- [ ] **Step 4: Run the render test to verify it passes**

Run: `node --test apps/workbench/src/analysis-brainflow.test.js`
Expected: PASS

- [ ] **Step 5: Commit the render slice**

```bash
git add apps/workbench/src/analysis-brainflow.js apps/workbench/src/analysis-brainflow.test.js
git commit -m "feat: add replay timeline trend renderers"
```

### Task 3: Wire Replay State, Round Interaction, Scroll Sync, and Motion-Safe Styling

**Files:**
- Modify: `apps/workbench/src/index.html`
- Modify: `apps/workbench/src/main.js`
- Modify: `apps/workbench/src/styles.css`
- Modify: `tests/browser/test_observer_dashboard_smoke.py`

- [ ] **Step 1: Extend the browser smoke test before changing the UI**

```python
if page.locator("#analysis-round-list [data-round-row]").count() > 0:
    page.locator("#analysis-round-list [data-round-row]").first.click()
    _wait_for_non_empty_text(page, "#analysis-live-copy")
    page.wait_for_function(
        """
        () => {
          const panel = document.getElementById("analysis-trend-panel");
          return Boolean(panel)
            && panel.textContent.includes("基础驱动力")
            && Boolean(document.querySelector("[data-trend-replay-action='start']"))
            && Boolean(document.querySelector("#analysis-trend-frames .trend-frame-card"));
        }
        """
    )
    page.click("[data-trend-replay-action='start']")
    page.wait_for_function("() => document.getElementById('analysis-trend-panel').dataset.replayState === 'playing'")
    page.click("[data-trend-replay-action='pause']")
    page.wait_for_function("() => document.getElementById('analysis-trend-panel').dataset.replayState === 'paused'")
    page.click("[data-trend-replay-action='resume']")
    page.wait_for_function("() => document.getElementById('analysis-trend-panel').dataset.replayState === 'playing'")
    page.click("[data-trend-replay-action='reset']")
    page.wait_for_function("() => document.getElementById('analysis-trend-panel').dataset.replayState === 'idle'")
```

- [ ] **Step 2: Run the browser smoke test and watch it fail**

Run: `pytest tests/browser/test_observer_dashboard_smoke.py::test_observer_dashboard_smoke_script_runs_under_pytest -v`
Expected: FAIL because the trend panel has no replay controls, no frame strip, and no replay state data attributes yet.

- [ ] **Step 3: Add the markup, state machine, and styles**

```html
<article class="card cortex-trend-card" id="analysis-trend-panel">
  <div class="section-head">
    <div>
      <h3>参数时序曲线</h3>
      <p class="section-copy">实时跟踪基础驱动力、最终胜出概率、习惯牵引强度的滑动变化，并把冲突点 / 修复动作挂到同一条轮次时间轴。</p>
    </div>
    <span class="pill subtle">基础驱动力 / 最终胜出概率 / 习惯牵引强度</span>
  </div>
  <div class="trend-panel-toolbar">
    <div id="analysis-trend-legend" class="trend-legend"></div>
    <div id="analysis-trend-controls" class="trend-panel-controls"></div>
  </div>
  <div id="analysis-trend-svg" class="trend-chart-shell">
    <div class="empty-state">等待时序数据。</div>
  </div>
  <div id="analysis-trend-frames" class="trend-frame-shell">
    <div class="empty-state">选中 round 后，这里会同步显示该轮的指标、冲突点和修复动作。</div>
  </div>
</article>
```

Add these named imports to the existing import lists in `apps/workbench/src/main.js`:

```js
import {
  renderTrendFrameStrip,
  renderTrendReplayControls,
} from "./analysis-brainflow.js";

import {
  buildAnalysisReplayFrames,
} from "./view-models.js";
```

Add these replay constants near the other top-level constants:

```js
const ANALYSIS_REPLAY_STEP_MS = 760;
let analysisReplayTimer = 0;
```

Add these fields to the existing `state` object:

```js
analysisHoveredRoundId: null,
analysisReplayPlaying: false,
analysisReplayRoundIndex: -1,
```

Add these fields to the existing `elements` object:

```js
analysisTrendControls: document.getElementById("analysis-trend-controls"),
analysisTrendFrames: document.getElementById("analysis-trend-frames"),
```

Add these helper functions near the other analysis helpers:

```js
function stopAnalysisReplay() {
  if (analysisReplayTimer) {
    window.clearTimeout(analysisReplayTimer);
    analysisReplayTimer = 0;
  }
  state.analysisReplayPlaying = false;
}

function scheduleAnalysisReplay(frames = []) {
  stopAnalysisReplay();
  if (!frames.length) {
    state.analysisReplayRoundIndex = -1;
    renderAnalysis();
    return;
  }
  state.analysisReplayPlaying = true;
  const tick = () => {
    if (state.analysisReplayRoundIndex >= frames.length - 1) {
      stopAnalysisReplay();
      renderAnalysis();
      return;
    }
    state.analysisReplayRoundIndex += 1;
    renderAnalysis();
    analysisReplayTimer = window.setTimeout(tick, ANALYSIS_REPLAY_STEP_MS);
  };
  if (state.analysisReplayRoundIndex < 0) {
    state.analysisReplayRoundIndex = 0;
    renderAnalysis();
  }
  analysisReplayTimer = window.setTimeout(tick, ANALYSIS_REPLAY_STEP_MS);
}

function activeTrendRoundId(replayFrames = []) {
  const replayRoundId =
    state.analysisReplayRoundIndex >= 0 ? replayFrames[state.analysisReplayRoundIndex]?.roundId ?? null : null;
  return replayRoundId ?? state.analysisHoveredRoundId ?? state.selectedRoundId;
}

function scrollActiveTrendFrameIntoView() {
  const active = elements.analysisTrendFrames?.querySelector(".trend-frame-card.is-active");
  if (!active) {
    return;
  }
  active.scrollIntoView({
    block: "nearest",
    inline: "center",
    behavior: motionPreferenceMedia?.matches ? "auto" : "smooth",
  });
}
```

Insert this replay reset block at the top of `hydrateSelectedRound`, before `const baseRound = ...`:

```js
stopAnalysisReplay();
state.analysisReplayRoundIndex = -1;
state.analysisHoveredRoundId = null;
```

Insert this replay-render block inside `renderAnalysis`, immediately after the existing `const trendSeries = buildAnalysisTrendSeries(...)` call:

```js
const replayFrames = buildAnalysisReplayFrames(trendSeries);
const activeRoundId = activeTrendRoundId(replayFrames);
const hasReplayStarted = state.analysisReplayRoundIndex >= 0;

setDataAttr(elements.analysisTrendPanel, "replayState", state.analysisReplayPlaying ? "playing" : hasReplayStarted ? "paused" : "idle");
setDataAttr(elements.analysisTrendPanel, "activeRoundId", activeRoundId || "");

elements.analysisTrendControls.innerHTML = renderTrendReplayControls({
  hasFrames: replayFrames.length > 1,
  isPlaying: state.analysisReplayPlaying,
  hasStarted: hasReplayStarted,
});
elements.analysisTrendSvg.innerHTML = renderTrendSvg(trendSeries, {
  selectedRoundId: state.selectedRoundId,
  activeRoundId,
  replayRoundId: hasReplayStarted ? replayFrames[state.analysisReplayRoundIndex]?.roundId : null,
});
elements.analysisTrendFrames.innerHTML = renderTrendFrameStrip(replayFrames, { activeRoundId });
requestAnimationFrame(scrollActiveTrendFrameIntoView);
```

Add this event delegation block inside `wireEvents`, after the round-list listeners and before the chat listeners:

```js
elements.analysisTrendPanel.addEventListener("click", (event) => {
  const replayButton = event.target.closest("[data-trend-replay-action]");
  if (replayButton) {
    const trendSeries = buildAnalysisTrendSeries({
      recentRounds: state.console?.recent_rounds || [],
      roundCatalog: state.roundCatalog,
    });
    const replayFrames = buildAnalysisReplayFrames(trendSeries);
    switch (replayButton.dataset.trendReplayAction) {
      case "start":
        state.analysisReplayRoundIndex = -1;
        scheduleAnalysisReplay(replayFrames);
        return;
      case "pause":
        stopAnalysisReplay();
        renderAnalysis();
        return;
      case "resume":
        scheduleAnalysisReplay(replayFrames);
        return;
      case "reset":
        stopAnalysisReplay();
        state.analysisReplayRoundIndex = -1;
        renderAnalysis();
        return;
    }
  }

  const roundTarget = event.target.closest("[data-trend-round-id]");
  if (roundTarget) {
    state.analysisHoveredRoundId = Number(roundTarget.dataset.trendRoundId);
    renderAnalysis();
  }
});

elements.analysisTrendPanel.addEventListener("mouseover", (event) => {
  const roundTarget = event.target.closest("[data-trend-round-id]");
  if (!roundTarget) {
    return;
  }
  state.analysisHoveredRoundId = Number(roundTarget.dataset.trendRoundId);
  renderAnalysis();
});

elements.analysisTrendPanel.addEventListener("mouseleave", () => {
  state.analysisHoveredRoundId = null;
  renderAnalysis();
});
```

```css
.trend-panel-toolbar {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
  flex-wrap: wrap;
}

.trend-panel-controls,
.trend-replay-controls {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}

.trend-legend-copy {
  display: grid;
  gap: 2px;
}

.trend-legend-sub {
  color: var(--muted);
  font-size: 0.74rem;
  letter-spacing: 0.04em;
}

.trend-chart-shell {
  min-height: 352px;
}

.trend-timeline-cursor {
  stroke: rgba(12, 124, 120, 0.55);
  stroke-width: 2;
  stroke-dasharray: 6 6;
}

.trend-marker-track {
  stroke: rgba(255, 255, 255, 0.92);
  stroke-width: 1.5;
}

.trend-marker-track.is-conflict {
  fill: #c64545;
}

.trend-marker-track.is-repair {
  fill: #1c6fd3;
}

.trend-frame-shell {
  margin-top: 14px;
}

.trend-frame-strip {
  display: flex;
  gap: 12px;
  overflow-x: auto;
  padding-bottom: 8px;
  scroll-snap-type: x proximity;
}

.trend-frame-card {
  min-width: 260px;
  max-width: 320px;
  display: grid;
  gap: 10px;
  padding: 16px 18px;
  border-radius: 20px;
  border: 1px solid rgba(49, 44, 37, 0.08);
  background: rgba(255, 255, 255, 0.78);
  scroll-snap-align: center;
}

.trend-frame-card.is-active {
  border-color: rgba(12, 124, 120, 0.32);
  box-shadow: 0 12px 28px rgba(12, 124, 120, 0.12);
}

.trend-frame-marker-row {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}

.trend-frame-marker {
  display: inline-flex;
  align-items: center;
  min-height: 28px;
  padding: 0 10px;
  border-radius: 999px;
  font-size: 0.78rem;
}

.trend-frame-marker.tone-conflict {
  background: rgba(198, 69, 69, 0.12);
  color: #9d2b2b;
}

.trend-frame-marker.tone-repair {
  background: rgba(28, 111, 211, 0.12);
  color: #16539d;
}

#workbench-root[data-motion="full"] .trend-series path {
  stroke-dasharray: 1;
  animation: trendSeriesReveal 420ms ease both;
}

#workbench-root[data-motion="full"] .trend-point.is-selected circle,
#workbench-root[data-motion="full"] .trend-marker-track.is-active {
  animation: trendMarkerPulse 520ms ease both;
}

body[data-motion="reduced"] .trend-series path,
body[data-motion="reduced"] .trend-point circle,
body[data-motion="reduced"] .trend-marker-track,
body[data-motion="reduced"] .trend-timeline-cursor,
#workbench-root[data-motion="reduced"] .trend-series path,
#workbench-root[data-motion="reduced"] .trend-point circle,
#workbench-root[data-motion="reduced"] .trend-marker-track,
#workbench-root[data-motion="reduced"] .trend-timeline-cursor {
  animation: none;
  transition: none;
}

@keyframes trendSeriesReveal {
  from {
    stroke-dashoffset: 1;
    opacity: 0.36;
  }

  to {
    stroke-dashoffset: 0;
    opacity: 1;
  }
}

@keyframes trendMarkerPulse {
  0% {
    transform: scale(0.92);
    opacity: 0.7;
  }

  100% {
    transform: scale(1);
    opacity: 1;
  }
}
```

- [ ] **Step 4: Re-run the browser smoke test**

Run: `pytest tests/browser/test_observer_dashboard_smoke.py::test_observer_dashboard_smoke_script_runs_under_pytest -v`
Expected: PASS

- [ ] **Step 5: Commit the interaction slice**

```bash
git add apps/workbench/src/index.html apps/workbench/src/main.js apps/workbench/src/styles.css tests/browser/test_observer_dashboard_smoke.py
git commit -m "feat: add animated analysis replay timeline"
```

### Task 4: Run the Final Verification Sweep and Build the Workbench Bundle

**Files:**
- Test: `apps/workbench/src/view-models.test.js`
- Test: `apps/workbench/src/analysis-brainflow.test.js`
- Test: `tests/browser/test_observer_dashboard_smoke.py`
- Verify: `apps/workbench/dist`

- [ ] **Step 1: Run all focused JavaScript tests together**

Run: `node --test apps/workbench/src/view-models.test.js apps/workbench/src/analysis-brainflow.test.js`
Expected: PASS

- [ ] **Step 2: Build the static workbench output**

Run: `cd apps/workbench && npm run build`
Expected: PASS and `apps/workbench/dist/index.html` plus the updated JS / CSS assets are refreshed.

- [ ] **Step 3: Run the browser smoke test one more time after the build**

Run: `pytest tests/browser/test_observer_dashboard_smoke.py::test_observer_dashboard_smoke_script_runs_under_pytest -v`
Expected: PASS

- [ ] **Step 4: Manual reduced-motion sanity check in a browser session**

Run: `python tests/browser/observer_dashboard_smoke.py`
Expected: The workbench opens, the trend card shows Chinese-first labels, replay buttons work, and the reduced-motion environment does not pulse or reveal-animate the SVG.

- [ ] **Step 5: Create the final verification commit**

```bash
git add apps/workbench/dist
git commit -m "test: verify animated analysis trend timeline"
```
