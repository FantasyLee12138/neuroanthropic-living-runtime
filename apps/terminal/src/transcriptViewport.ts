export function buildTranscriptWindow<T>(lines: T[], visibleCount: number, offset: number): {
  lines: T[];
  offset: number;
  maxOffset: number;
} {
  const safeVisibleCount = Math.max(1, visibleCount);
  const maxOffset = Math.max(0, lines.length - safeVisibleCount);
  const nextOffset = Math.max(0, Math.min(offset, maxOffset));
  const end = lines.length - nextOffset;
  const start = Math.max(0, end - safeVisibleCount);
  return {
    lines: lines.slice(start, end),
    offset: nextOffset,
    maxOffset
  };
}

export function moveTranscriptOffset(input: {
  currentOffset: number;
  direction: "older" | "newer";
  pageSize: number;
  totalLines: number;
  visibleCount: number;
}): number {
  const step = Math.max(1, input.pageSize);
  const maxOffset = Math.max(0, input.totalLines - Math.max(1, input.visibleCount));
  if (input.direction === "older") {
    return Math.min(maxOffset, input.currentOffset + step);
  }
  return Math.max(0, input.currentOffset - step);
}
