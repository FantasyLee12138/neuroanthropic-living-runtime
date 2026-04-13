import test from "node:test";
import assert from "node:assert/strict";
import {
  renderTrendLegend,
  renderTrendStats,
  renderTrendSvg,
  renderTrendReplayControls,
  renderTrendFrameStrip,
  renderBrainflowOutput,
} from "./analysis-brainflow.js";

const trendSeries = [
  {
    key: "u_base",
    label: "u_base",
    displayLabel: "基础驱动力",
    displaySubLabel: "动作原始驱动力",
    points: [
      {
        roundId: 101,
        value: 0.41,
        source: "action_bookkeeping.u_base",
        displaySource: "动作原始驱动力",
        hasConflict: true,
        hasRepair: false,
      },
      {
        roundId: 102,
        value: 0.44,
        source: "action_bookkeeping.u_base",
        displaySource: "动作原始驱动力",
        hasConflict: false,
        hasRepair: true,
      },
    ],
  },
  {
    key: "p_final",
    label: "p_final",
    displayLabel: "最终胜出概率",
    displaySubLabel: "概率场最终裁决",
    points: [
      {
        roundId: 101,
        value: 0.62,
        source: "probability_field.action.winner_posterior",
        displaySource: "概率场最终裁决",
        hasConflict: true,
        hasRepair: false,
      },
      {
        roundId: 102,
        value: 0.58,
        source: "probability_field.action.winner_posterior",
        displaySource: "概率场最终裁决",
        hasConflict: false,
        hasRepair: true,
      },
    ],
  },
  {
    key: "strength",
    label: "strength",
    displayLabel: "习惯牵引强度",
    displaySubLabel: "习惯 / 记忆牵引",
    points: [
      {
        roundId: 101,
        value: 0.39,
        source: "initiative.memory_backing.strength",
        displaySource: "习惯 / 记忆牵引",
        hasConflict: true,
        hasRepair: false,
      },
      {
        roundId: 102,
        value: 0.54,
        source: "action_bookkeeping.ci",
        displaySource: "习惯 / 记忆牵引",
        hasConflict: false,
        hasRepair: true,
      },
    ],
  },
];

test("legend and stats render Chinese-first copy", () => {
  const legend = renderTrendLegend(trendSeries);
  assert.match(legend, /基础驱动力/);
  assert.match(legend, /动作原始驱动力/);
  assert.doesNotMatch(legend, /u_base|p_final|strength/);

  const stats = renderTrendStats(trendSeries);
  assert.match(stats, /动作原始驱动力/);
  assert.doesNotMatch(stats, /action_bookkeeping\.u_base/);
  assert.doesNotMatch(stats, /u_base|p_final|strength/);
});

test("trend svg includes active cursor, named marker tracks, and active round points", () => {
  const svg = renderTrendSvg(trendSeries, { selectedRoundId: 102 });
  assert.match(svg, /trend-active-band/);
  assert.match(svg, /trend-round-cursor/);
  assert.match(svg, /trend-marker-track-conflict/);
  assert.match(svg, /trend-marker-track-repair/);
  assert.match(svg, /trend-track-label/);
  assert.match(svg, /冲突/);
  assert.match(svg, /修复/);
  assert.match(svg, /trend-round-marker is-conflict/);
  assert.match(svg, /trend-round-marker is-repair/);
  assert.equal((svg.match(/trend-point is-active/g) || []).length, 3);
});

test("replay controls and frame strip expose playback actions and active frame", () => {
  const controls = renderTrendReplayControls({
    hasFrames: true,
    isPlaying: true,
    hasStarted: true,
    activeRoundLabel: "第102轮",
    activeIndex: 1,
    totalFrames: 2,
  });
  assert.match(controls, /data-action="start"/);
  assert.match(controls, /data-trend-replay-action="start"/);
  assert.match(controls, /data-action="pause"/);
  assert.match(controls, /data-action="resume"/);
  assert.match(controls, /data-action="reset"/);
  assert.match(controls, />\s*回放\s*</);
  assert.match(controls, /正在回放/);
  assert.match(controls, /第102轮/);
  assert.match(controls, /2 \/ 2/);

  const strip = renderTrendFrameStrip(
    [
      {
        roundId: 101,
        roundLabel: "第101轮",
        summary: "冲突上升，进入修复预备。",
        metrics: [
          { key: "u_base", displayLabel: "基础驱动力", value: 0.41 },
          { key: "p_final", displayLabel: "最终胜出概率", value: 0.62 },
          { key: "strength", displayLabel: "习惯牵引强度", value: 0.39 },
        ],
        markers: [
          { track: "conflict", label: "冲突" },
          { track: "repair", label: "修复" },
        ],
      },
      {
        roundId: 102,
        roundLabel: "第102轮",
        summary: "修复完成，回到稳定轨道。",
        metrics: [
          { key: "u_base", displayLabel: "基础驱动力", value: 0.44 },
          { key: "p_final", displayLabel: "最终胜出概率", value: 0.58 },
          { key: "strength", displayLabel: "习惯牵引强度", value: 0.54 },
        ],
        markers: [{ track: "repair", label: "修复完成" }],
      },
    ],
    { activeRoundId: 102 },
  );
  assert.match(strip, /trend-frame is-active/);
  assert.match(strip, /第102轮/);
  assert.match(strip, /修复完成，回到稳定轨道。/);
  assert.match(strip, /trend-frame-metrics/);
  assert.match(strip, /基础驱动力/);
  assert.match(strip, />0\.58</);
  assert.match(strip, /冲突/);
  assert.match(strip, /修复完成/);
});

test("brainflow output meta uses Chinese visible labels", () => {
  const output = renderBrainflowOutput({
    title: "表达结果",
    text: "先停一下，再继续。",
    summary: "已完成输出整理。",
    route: "深度对话",
    delivery_mode: "文字",
    model: "planner-main",
  });

  assert.match(output, /已完成输出整理/);
  assert.match(output, /输出路径：深度对话/);
  assert.match(output, /输出方式：文字/);
  assert.match(output, /模型：planner-main/);
  assert.doesNotMatch(output, /route /);
  assert.doesNotMatch(output, /delivery /);
  assert.doesNotMatch(output, /model /);
});
