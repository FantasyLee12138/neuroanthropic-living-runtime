import { escapeHtml, formatNumber } from "./format.js";

const SERIES_STYLE = {
  u_base: { color: "#bb742d", glow: "rgba(187, 116, 45, 0.18)" },
  p_final: { color: "#0c7c78", glow: "rgba(12, 124, 120, 0.18)" },
  strength: { color: "#1b5fa7", glow: "rgba(27, 95, 167, 0.18)" },
};

function finiteNumber(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) {
    return null;
  }
  return number;
}

function computeValueDomain(trendSeries = []) {
  const values = trendSeries
    .flatMap((series) => series?.points || [])
    .map((point) => finiteNumber(point?.value))
    .filter((value) => value !== null);
  if (!values.length) {
    return { min: 0, max: 1 };
  }
  let min = Math.min(...values);
  let max = Math.max(...values);
  if (min === max) {
    const baseSpan = Math.max(Math.abs(min) * 0.18, 0.2);
    min -= baseSpan;
    max += baseSpan;
  }
  const span = Math.max(max - min, 0.36);
  const center = (min + max) / 2;
  const paddedSpan = span * 1.14;
  return {
    min: center - paddedSpan / 2,
    max: center + paddedSpan / 2,
  };
}

function normalizeValue(value, domain) {
  const number = finiteNumber(value);
  if (number === null) {
    return null;
  }
  const span = Math.max(domain.max - domain.min, 0.0001);
  return (number - domain.min) / span;
}

function scaleY(value, domain, height, padding) {
  const normalized = normalizeValue(value, domain);
  if (normalized === null) {
    return null;
  }
  return padding.top + (1 - normalized) * (height - padding.top - padding.bottom);
}

function formatAxisValue(value) {
  const safeValue = Math.abs(value) < 0.0005 ? 0 : value;
  const abs = Math.abs(safeValue);
  if (abs >= 100) {
    return safeValue.toFixed(0);
  }
  if (abs >= 10) {
    return safeValue.toFixed(1);
  }
  return safeValue.toFixed(2);
}

function buildAxisTicks(domain, count = 5) {
  return Array.from({ length: count }, (_, index) => {
    const ratio = count === 1 ? 0 : index / (count - 1);
    return {
      ratio,
      value: domain.min + (domain.max - domain.min) * ratio,
    };
  });
}

function stageMap(markers = []) {
  return markers.reduce((map, marker) => {
    const key = String(marker?.stageKey || "");
    if (!key) {
      return map;
    }
    map[key] = map[key] || [];
    map[key].push(marker);
    return map;
  }, {});
}

function pathForPoints(points, width, height, padding, domain) {
  const usable = points.filter((point) => normalizeValue(point.value, domain) !== null);
  if (!usable.length) {
    return "";
  }
  return usable
    .map((point, index) => {
      const x = padding.left + point.x * (width - padding.left - padding.right);
      const y = scaleY(point.value, domain, height, padding);
      return `${index === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`;
    })
    .join(" ");
}

function renderFrameMetrics(metrics = []) {
  const usable = metrics.filter((metric) => metric && (metric.displayLabel || metric.key));
  if (!usable.length) {
    return "";
  }
  return `<div class="trend-frame-metrics" aria-label="轮次指标快照">
    ${usable
      .map((metric) => {
        const key = metric?.key || "metric";
        const label = metric?.displayLabel || metric?.displaySubLabel || key;
        const value = Number.isFinite(metric?.value) ? formatNumber(metric.value, 2) : "--";
        return `<span class="trend-frame-metric" data-key="${escapeHtml(key)}">
          <span>${escapeHtml(label)}</span>
          <strong>${escapeHtml(value)}</strong>
        </span>`;
      })
      .join("")}
  </div>`;
}

function renderTrendTrackLabels({ left, conflictY, repairY, blockedY, skippedY }) {
  const rows = [
    { key: "conflict", label: "冲突", y: conflictY },
    { key: "repair", label: "修复", y: repairY },
    { key: "blocked", label: "阻断", y: blockedY },
    { key: "skipped", label: "跳过", y: skippedY },
  ];
  return `<g class="trend-track-labels" aria-hidden="true">
    ${rows
      .map(
        (row) =>
          `<text class="trend-track-label track-${escapeHtml(row.key)}" x="${left}" y="${row.y + 4}" text-anchor="end">${escapeHtml(row.label)}</text>`,
      )
      .join("")}
  </g>`;
}

function renderTrendReplayStatus({ hasFrames = false, isPlaying = false, hasStarted = false, activeRoundLabel = "", activeIndex = 0, totalFrames = 0 } = {}) {
  if (!hasFrames) {
    return `<div class="trend-replay-status" data-state="idle">
      <strong>等待回放</strong>
      <span>当前还没有可回放的轮次。</span>
    </div>`;
  }
  const safeTotal = Math.max(0, Number(totalFrames) || 0);
  const safeIndex = Math.max(0, Math.min(Number(activeIndex) || 0, Math.max(safeTotal - 1, 0)));
  const progressLabel = safeTotal ? `${safeIndex + 1} / ${safeTotal}` : "--";
  const stateLabel = isPlaying ? "正在回放" : hasStarted ? "已暂停" : "等待开始";
  const roundLabel = activeRoundLabel || "尚未定位轮次";
  return `<div class="trend-replay-status" data-state="${isPlaying ? "playing" : hasStarted ? "paused" : "idle"}">
    <strong>${escapeHtml(stateLabel)}</strong>
    <span>${escapeHtml(`${progressLabel} · ${roundLabel}`)}</span>
  </div>`;
}

export function renderTrendLegend(trendSeries = []) {
  if (!trendSeries.length) {
    return `<div class="empty-state">等待曲线数据。</div>`;
  }
  return trendSeries
    .map((series) => {
      const style = SERIES_STYLE[series.key] || SERIES_STYLE.p_final;
      const label = series.displayLabel || series.label || series.key || "--";
      const subLabel = series.displaySubLabel || series.displaySource || series.label || series.key || "";
      const hasSubLabel = subLabel && subLabel !== label;
      return `<div class="trend-legend-item">
        <span class="trend-legend-swatch" style="--swatch:${escapeHtml(style.color)}"></span>
        <span class="trend-legend-copy">
          <strong>${escapeHtml(label)}</strong>
          ${hasSubLabel ? `<span class="trend-legend-sub">${escapeHtml(subLabel)}</span>` : ""}
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
      const numeric = series.points.filter((point) => finiteNumber(point.value) !== null);
      const latest = numeric[numeric.length - 1] || null;
      const previous = numeric[numeric.length - 2] || null;
      const delta = latest && previous ? latest.value - previous.value : null;
      const label = series.displayLabel || series.label || series.key || "--";
      const subLabel = series.displaySubLabel || series.displaySource || series.label || series.key || "";
      const hasSubLabel = subLabel && subLabel !== label;
      return `<article class="trend-stat-card">
        <span>${escapeHtml(label)}</span>
        ${hasSubLabel ? `<small>${escapeHtml(subLabel)}</small>` : ""}
        <strong>${escapeHtml(latest ? formatNumber(latest.value, 3) : "--")}</strong>
        <div class="trend-stat-meta">
          <span>${escapeHtml(latest?.displaySource || "--")}</span>
          <span>${escapeHtml(delta === null ? "Δ --" : `Δ ${delta >= 0 ? "+" : ""}${formatNumber(delta, 3)}`)}</span>
        </div>
      </article>`;
    })
    .join("");
}

export function renderTrendSvg(trendSeries = [], { selectedRoundId = null, activeRoundId = null } = {}) {
  const basePoints = trendSeries[0]?.points || [];
  if (!basePoints.length) {
    return `<div class="empty-state">等待时序数据。</div>`;
  }

  const width = 1320;
  const height = 420;
  const padding = { top: 88, right: 56, bottom: 76, left: 72 };
  const domain = computeValueDomain(trendSeries);
  const denominator = Math.max(basePoints.length - 1, 1);
  const scaledSeries = trendSeries.map((series) => ({
    ...series,
    points: series.points.map((point, index) => ({
      ...point,
      x: denominator === 0 ? 0 : index / denominator,
    })),
  }));
  const labels = buildAxisTicks(domain);
  const roundPoints = basePoints.map((point, index) => ({
    roundId: point.roundId,
    x: padding.left + (index / denominator) * (width - padding.left - padding.right),
    hasConflict: Boolean(point.hasConflict),
    hasBlocked: Boolean(point.hasBlocked),
    hasSkipped: Boolean(point.hasSkipped),
    hasRepair: Boolean(point.hasRepair),
  }));
  const currentRoundId = activeRoundId ?? selectedRoundId ?? basePoints[basePoints.length - 1]?.roundId ?? null;
  const activeRoundPoint = roundPoints.find((point) => Number(point.roundId) === Number(currentRoundId)) || null;
  const axisY = height - 42;
  const conflictTrackY = axisY - 18;
  const repairTrackY = axisY - 42;
  const blockedTrackY = axisY - 66;
  const skippedTrackY = axisY - 90;
  const stepWidth = denominator === 0 ? width - padding.left - padding.right : (width - padding.left - padding.right) / denominator;
  const activeBandWidth = Math.max(58, Math.min(stepWidth * 0.78, 92));
  const activeBandX = activeRoundPoint ? Math.max(padding.left, activeRoundPoint.x - activeBandWidth / 2) : null;

  return `<svg class="trend-chart-svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="分析趋势">
    <defs>
      <filter id="trendGlow" x="-20%" y="-20%" width="140%" height="140%">
        <feGaussianBlur stdDeviation="4" result="blur" />
        <feMerge>
          <feMergeNode in="blur" />
          <feMergeNode in="SourceGraphic" />
        </feMerge>
      </filter>
    </defs>
    ${labels
      .map((label) => {
        const y = padding.top + (1 - label.ratio) * (height - padding.top - padding.bottom);
        return `<g class="trend-grid-row">
          <line x1="${padding.left}" y1="${y}" x2="${width - padding.right}" y2="${y}"></line>
          <text x="${padding.left - 16}" y="${y + 5}" text-anchor="end">${formatAxisValue(label.value)}</text>
        </g>`;
      })
      .join("")}
    ${activeRoundPoint ? `<rect class="trend-active-band" x="${activeBandX?.toFixed(2)}" y="${padding.top}" width="${activeBandWidth.toFixed(2)}" height="${(axisY - padding.top).toFixed(2)}" rx="18"></rect>` : ""}
    ${scaledSeries
      .map((series) => {
        const style = SERIES_STYLE[series.key] || SERIES_STYLE.p_final;
        return `<g class="trend-series trend-series-${escapeHtml(series.key)}">
          <path d="${escapeHtml(pathForPoints(series.points, width, height, padding, domain))}" style="--series-color:${escapeHtml(style.color)}"></path>
          ${series.points
            .filter((point) => normalizeValue(point.value, domain) !== null)
            .map((point) => {
              const x = padding.left + point.x * (width - padding.left - padding.right);
              const y = scaleY(point.value, domain, height, padding);
              const isActive = Number(point.roundId) === Number(currentRoundId);
              const label = series.displayLabel || series.label || series.key || "--";
              return `<g class="trend-point${isActive ? " is-active" : ""}" style="--series-color:${escapeHtml(style.color)}" data-state="${isActive ? "active" : "idle"}" data-round-id="${escapeHtml(point.roundId)}" data-trend-round-id="${escapeHtml(point.roundId)}">
                <circle cx="${x.toFixed(2)}" cy="${y.toFixed(2)}" r="${isActive ? 7.2 : 4.6}" filter="url(#trendGlow)">
                  <title>第 ${escapeHtml(point.roundId)} 轮 · ${escapeHtml(label)} ${escapeHtml(formatNumber(point.value, 3))} · ${escapeHtml(point.displaySource || point.source || "--")}</title>
                </circle>
              </g>`;
            })
            .join("")}
        </g>`;
      })
      .join("")}
    <line class="trend-round-axis" x1="${padding.left}" y1="${axisY}" x2="${width - padding.right}" y2="${axisY}"></line>
    ${activeRoundPoint ? `<line class="trend-round-cursor" x1="${activeRoundPoint.x.toFixed(2)}" y1="${padding.top}" x2="${activeRoundPoint.x.toFixed(2)}" y2="${axisY}" data-state="active"></line>` : ""}
    <g class="trend-marker-track trend-marker-track-conflict">
      ${roundPoints
        .filter((point) => point.hasConflict)
        .map(
          (point) =>
            `<circle class="trend-round-marker is-conflict${Number(point.roundId) === Number(currentRoundId) ? " is-active" : ""}" cx="${point.x.toFixed(2)}" cy="${conflictTrackY}" r="5" data-round-id="${escapeHtml(point.roundId)}" data-trend-round-id="${escapeHtml(point.roundId)}"><title>该轮存在冲突点</title></circle>`,
        )
        .join("")}
    </g>
    <g class="trend-marker-track trend-marker-track-repair">
      ${roundPoints
        .filter((point) => point.hasRepair)
        .map(
          (point) =>
            `<circle class="trend-round-marker is-repair${Number(point.roundId) === Number(currentRoundId) ? " is-active" : ""}" cx="${point.x.toFixed(2)}" cy="${repairTrackY}" r="5" data-round-id="${escapeHtml(point.roundId)}" data-trend-round-id="${escapeHtml(point.roundId)}"><title>该轮存在修复动作</title></circle>`,
        )
        .join("")}
    </g>
    <g class="trend-marker-track trend-marker-track-blocked">
      ${roundPoints
        .filter((point) => point.hasBlocked)
        .map(
          (point) =>
            `<circle class="trend-round-marker is-blocked" cx="${point.x.toFixed(2)}" cy="${blockedTrackY}" r="5" data-round-id="${escapeHtml(point.roundId)}"><title>该轮存在阻断</title></circle>`,
        )
        .join("")}
    </g>
    <g class="trend-marker-track trend-marker-track-skipped">
      ${roundPoints
        .filter((point) => point.hasSkipped)
        .map(
          (point) =>
            `<circle class="trend-round-marker is-skipped" cx="${point.x.toFixed(2)}" cy="${skippedTrackY}" r="5" data-round-id="${escapeHtml(point.roundId)}"><title>该轮存在跳过</title></circle>`,
        )
        .join("")}
    </g>
    ${roundPoints
      .map(
        (point) =>
          `<g class="trend-axis-round${Number(point.roundId) === Number(currentRoundId) ? " is-active" : ""}" data-state="${Number(point.roundId) === Number(currentRoundId) ? "active" : "idle"}" data-round-id="${escapeHtml(point.roundId)}" data-trend-round-id="${escapeHtml(point.roundId)}">
        <text x="${point.x.toFixed(2)}" y="${height - 10}" text-anchor="middle">#${escapeHtml(point.roundId)}</text>
      </g>`,
      )
      .join("")}
    ${renderTrendTrackLabels({
      left: padding.left - 12,
      conflictY: conflictTrackY,
      repairY: repairTrackY,
      blockedY: blockedTrackY,
      skippedY: skippedTrackY,
    })}
  </svg>`;
}

export function renderTrendReplayControls({
  hasFrames = false,
  isPlaying = false,
  hasStarted = false,
  activeRoundLabel = "",
  activeIndex = 0,
  totalFrames = 0,
} = {}) {
  return `<div class="trend-replay-controls" data-state="${isPlaying ? "playing" : hasStarted ? "paused" : "idle"}">
    ${renderTrendReplayStatus({ hasFrames, isPlaying, hasStarted, activeRoundLabel, activeIndex, totalFrames })}
    <div class="trend-replay-actions">
      <button type="button" class="trend-replay-btn" data-action="start" data-trend-replay-action="start" data-state="${hasStarted ? "ready" : "idle"}" ${hasFrames ? "" : "disabled"}>
        回放
      </button>
      <button type="button" class="trend-replay-btn" data-action="pause" data-trend-replay-action="pause" data-state="${isPlaying ? "active" : "idle"}" ${isPlaying ? "" : "disabled"}>
        暂停
      </button>
      <button type="button" class="trend-replay-btn" data-action="resume" data-trend-replay-action="resume" data-state="${hasStarted && !isPlaying ? "active" : "idle"}" ${hasStarted && !isPlaying ? "" : "disabled"}>
        继续
      </button>
      <button type="button" class="trend-replay-btn" data-action="reset" data-trend-replay-action="reset" data-state="${hasStarted ? "active" : "idle"}" ${hasStarted ? "" : "disabled"}>
        重置
      </button>
    </div>
  </div>`;
}

export function renderTrendFrameStrip(frames = [], { activeRoundId = null } = {}) {
  if (!Array.isArray(frames) || !frames.length) {
    return `<div class="empty-state">暂无回放帧。</div>`;
  }
  return `<div class="trend-frame-strip" role="list">
    ${frames
      .map((frame, index) => {
        const roundId = frame?.roundId ?? frame?.id ?? index + 1;
        const isActive = Number(roundId) === Number(activeRoundId);
        const label = frame?.roundLabel || frame?.label || `第${roundId}轮`;
        const summary = frame?.summary || "";
        const metrics = Array.isArray(frame?.metrics) ? frame.metrics : [];
        const markers = Array.isArray(frame?.markers) ? frame.markers : [];
        return `<button type="button" class="trend-frame${isActive ? " is-active" : ""}" role="listitem" data-round-id="${escapeHtml(roundId)}" data-trend-round-id="${escapeHtml(roundId)}" data-state="${isActive ? "active" : "idle"}">
          <strong>${escapeHtml(label)}</strong>
          ${summary ? `<span class="trend-frame-summary">${escapeHtml(summary)}</span>` : ""}
          ${renderFrameMetrics(metrics)}
          ${
            markers.length
              ? `<span class="trend-frame-markers">${markers
                  .map((marker) => {
                    const track = marker?.track || "default";
                    const markerLabel = marker?.label || "";
                    if (!markerLabel) {
                      return "";
                    }
                    return `<span class="trend-frame-marker track-${escapeHtml(track)}" data-track="${escapeHtml(track)}">${escapeHtml(markerLabel)}</span>`;
                  })
                  .join("")}</span>`
              : ""
          }
        </button>`;
      })
      .join("")}
  </div>`;
}

function renderStageMarkers(markers = []) {
  return markers
    .map(
      (marker) => `<span class="brainflow-marker tone-${escapeHtml(marker.tone)}" title="${escapeHtml(marker.detail || marker.label)}">
        ${escapeHtml(marker.label)}
      </span>`,
    )
    .join("");
}

export function renderBrainflowStages(
  stages = [],
  { conflictMarkers = [], blockedMarkers = [], skippedMarkers = [], repairMarkers = [] } = {},
) {
  if (!stages.length) {
    return `<div class="empty-state">等待脑流数据。</div>`;
  }
  const conflictsByStage = stageMap(conflictMarkers);
  const blockedByStage = stageMap(blockedMarkers);
  const skippedByStage = stageMap(skippedMarkers);
  const repairsByStage = stageMap(repairMarkers);
  return `<div class="brainflow-track" style="--brainflow-stage-count:${escapeHtml(stages.length)}">
    ${stages
      .map((stage, index) => {
        const stageConflicts = conflictsByStage[stage.key] || [];
        const stageBlocks = blockedByStage[stage.key] || [];
        const stageSkips = skippedByStage[stage.key] || [];
        const stageRepairs = repairsByStage[stage.key] || [];
        const tone = stageConflicts.length
          ? "conflict"
          : stageBlocks.length
            ? "blocked"
            : stageSkips.length
              ? "skipped"
              : stageRepairs.length
                ? "repair"
                : "stable";
        return `<div class="brainflow-stage-wrap${stage.key === "output" ? " is-output" : ""}${index === stages.length - 1 ? " is-terminal" : ""}">
          <article class="brainflow-stage is-${escapeHtml(tone)}" data-stage-key="${escapeHtml(stage.key)}">
            <div class="brainflow-stage-head">
              <span class="brainflow-stage-index">${String(index + 1).padStart(2, "0")}</span>
              <strong>${escapeHtml(stage.title)}</strong>
            </div>
            <p class="brainflow-stage-body">${escapeHtml(stage.body)}</p>
            <div class="brainflow-stage-meta">${escapeHtml(stage.meta || "--")}</div>
            ${
              stageConflicts.length || stageBlocks.length || stageSkips.length || stageRepairs.length
                ? `<div class="brainflow-stage-markers">
                    ${renderStageMarkers(stageConflicts)}
                    ${renderStageMarkers(stageBlocks)}
                    ${renderStageMarkers(stageSkips)}
                    ${renderStageMarkers(stageRepairs)}
                  </div>`
                : ""
            }
          </article>
          ${
            index < stages.length - 1
              ? `<div class="brainflow-connector" aria-hidden="true">
                  <span class="brainflow-connector-line"></span>
                  <span class="brainflow-connector-pulse"></span>
                </div>`
              : ""
          }
        </div>`;
      })
      .join("")}
  </div>`;
}

export function renderBrainflowOutput(output = {}) {
  return `<article class="brainflow-output-card">
    <div class="brainflow-output-head">
      <span>输出话术</span>
      <strong>${escapeHtml(output.title || "最终外显表达")}</strong>
    </div>
    <p class="brainflow-output-text">${escapeHtml(output.text || "当前没有可展示的输出。")}</p>
    <div class="brainflow-output-meta">
      <span>${escapeHtml(output.summary || "--")}</span>
      <span>${escapeHtml(`输出路径：${output.route || "--"}`)}</span>
      <span>${escapeHtml(`输出方式：${output.delivery_mode || "--"}`)}</span>
      <span>${escapeHtml(`模型：${output.model || "--"}`)}</span>
    </div>
  </article>`;
}
