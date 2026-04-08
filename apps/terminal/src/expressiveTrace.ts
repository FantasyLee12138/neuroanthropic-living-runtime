function normalizeRecord(value: unknown): Record<string, unknown> | undefined {
  return typeof value === "object" && value !== null && !Array.isArray(value) ? (value as Record<string, unknown>) : undefined;
}

function normalizeRecordList(value: unknown): Array<Record<string, unknown>> {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.filter((entry): entry is Record<string, unknown> => typeof entry === "object" && entry !== null);
}

function readNumber(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === "string" && value.trim() && !Number.isNaN(Number(value))) {
    return Number(value);
  }
  return null;
}

function readText(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

function readBoolean(value: unknown): boolean | null {
  return typeof value === "boolean" ? value : null;
}

function normalizeMonologueFragment(value: unknown): Record<string, unknown> | null {
  const record = normalizeRecord(value);
  if (!record) {
    return null;
  }
  return {
    fragmentId: readText(record.fragment_id ?? record.fragmentId),
    recordedAt: readText(record.recorded_at ?? record.recordedAt),
    category: readText(record.category),
    content: readText(record.content),
    source: readText(record.source),
  };
}

export function extractMonologueStream(value: unknown): Record<string, unknown> | null {
  const trace = normalizeRecord(value);
  const monologueStream = normalizeRecord(trace?.monologue_stream ?? trace?.monologueStream ?? value);
  if (!monologueStream) {
    return null;
  }
  const sampleFragments = normalizeRecordList(monologueStream.sample_fragments ?? monologueStream.sampleFragments)
    .map(normalizeMonologueFragment)
    .filter((entry): entry is Record<string, unknown> => Boolean(entry));
  return {
    active: readBoolean(monologueStream.active),
    hiddenByDefault: readBoolean(monologueStream.hidden_by_default ?? monologueStream.hiddenByDefault),
    freshGenerated: readBoolean(monologueStream.fresh_generated ?? monologueStream.freshGenerated),
    generatedTotal: readNumber(monologueStream.generated_total ?? monologueStream.generatedTotal),
    lastGeneratedAt: readText(monologueStream.last_generated_at ?? monologueStream.lastGeneratedAt),
    recentFragmentCount: readNumber(monologueStream.recent_fragment_count ?? monologueStream.recentFragmentCount),
    sampleFragments,
  };
}

export function normalizeExpressiveTrace(value: unknown): Record<string, unknown> {
  const trace = normalizeRecord(value) ?? {};
  const monologueStream = extractMonologueStream(trace);
  if (!monologueStream) {
    return trace;
  }
  return {
    ...trace,
    monologueStream,
  };
}

export function summarizeMonologueStream(value: unknown): string[] {
  const stream = extractMonologueStream(value);
  if (!stream) {
    return [];
  }
  const fragmentCount = readNumber(stream.recentFragmentCount) ?? 0;
  const generatedTotal = readNumber(stream.generatedTotal);
  const samples = normalizeRecordList(stream.sampleFragments)
    .map((entry) => readText(entry.content))
    .filter((entry): entry is string => Boolean(entry))
    .slice(0, 2);
  if (fragmentCount <= 0 && (generatedTotal == null || generatedTotal <= 0) && samples.length === 0) {
    return [];
  }
  const visibility = stream.hiddenByDefault === false ? "默认可见" : "默认不显示";
  const totalText = generatedTotal != null && generatedTotal > 0 ? `，累计 ${generatedTotal} 条` : "";
  const lines = [`隐藏独白流：最近 ${fragmentCount} 条碎片${totalText}，${visibility}`];
  if (samples.length > 0) {
    lines.push(`样本：${samples.join("；")}`);
  }
  return lines;
}
