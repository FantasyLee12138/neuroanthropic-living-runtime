function normalizeWhitespace(value: string): string {
  return value.replace(/\s+/g, " ").trim();
}

function splitInlineSegments(line: string): string[] {
  return line
    .split(/[；|]/)
    .map((part) => normalizeWhitespace(part))
    .filter((part) => part.length > 0);
}

export function toReadableLines(text: string | null | undefined, options?: { fallback?: string; maxLines?: number }): string[] {
  const fallback = options?.fallback ?? "尚未接入";
  const maxLines = options?.maxLines ?? Number.POSITIVE_INFINITY;
  if (!text || !text.trim()) {
    return [fallback];
  }

  const rows = text
    .split("\n")
    .flatMap((line) => {
      const normalized = normalizeWhitespace(line);
      if (!normalized) {
        return [];
      }
      const segments = splitInlineSegments(normalized);
      return segments.length > 0 ? segments : [normalized];
    })
    .slice(0, maxLines);

  return rows.length > 0 ? rows : [fallback];
}

export function joinReadableLabel(label: string, value: string | null | undefined, options?: { fallback?: string; maxLines?: number }): string[] {
  const lines = toReadableLines(value, options);
  if (lines.length === 0) {
    return [`${label}：${options?.fallback ?? "尚未接入"}`];
  }
  return lines.map((line, index) => (index === 0 ? `${label}：${line}` : `  ${line}`));
}
