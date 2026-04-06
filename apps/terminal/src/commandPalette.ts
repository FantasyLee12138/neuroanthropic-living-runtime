import type { UiAction } from "./types.js";

export interface PaletteEntry {
  id: string;
  label: string;
  group: "查看" | "控制" | "本地";
  disabled: boolean;
  run: () => void | Promise<void>;
  haystack: string;
}

export type PaletteSectionId = "recommended" | "recent" | "all";
export type PaletteContext = "approval" | "running" | "idle";

export interface PaletteSection {
  id: PaletteSectionId;
  label: "Recommended" | "Recent" | "All";
  entries: PaletteEntry[];
}

type PaletteSourceKind = UiAction["kind"] | "local";

interface PaletteRecord {
  entry: PaletteEntry;
  kind: PaletteSourceKind;
  index: number;
}

export function buildPaletteEntries(
  primary: UiAction[],
  secondary: UiAction[],
  options?: {
    onUiAction?: (action: UiAction) => void | Promise<void>;
    onLocalAction?: (command: "help" | "clear" | "compact" | "exit") => void | Promise<void>;
  },
): PaletteEntry[] {
  return buildPaletteRecords(primary, secondary, options).map((record) => record.entry);
}

export function buildPaletteSections(
  primary: UiAction[],
  secondary: UiAction[],
  options?: {
    context?: PaletteContext;
    recentIds?: string[];
    onUiAction?: (action: UiAction) => void | Promise<void>;
    onLocalAction?: (command: "help" | "clear" | "compact" | "exit") => void | Promise<void>;
  },
): PaletteSection[] {
  const records = buildPaletteRecords(primary, secondary, options);
  const recentIds = options?.recentIds ?? [];
  const context = options?.context ?? "running";

  return [
    {
      id: "recommended",
      label: "Recommended",
      entries: sortPaletteRecords(records, context).map((record) => record.entry),
    },
    {
      id: "recent",
      label: "Recent",
      entries: buildRecentEntries(records, recentIds),
    },
    {
      id: "all",
      label: "All",
      entries: records.map((record) => record.entry),
    },
  ];
}

export function filterPaletteSections(sections: PaletteSection[], query: string): PaletteSection[] {
  const normalized = query.trim().toLowerCase();
  if (!normalized) {
    return sections;
  }

  return sections
    .map((section) => ({
      ...section,
      entries: section.entries.filter((entry) => entry.haystack.includes(normalized)),
    }))
    .filter((section) => section.entries.length > 0);
}

export function filterPaletteEntries(entries: PaletteEntry[], query: string): PaletteEntry[] {
  const normalized = query.trim().toLowerCase();
  if (!normalized) {
    return entries;
  }
  return entries.filter((entry) => entry.haystack.includes(normalized));
}

function buildPaletteRecords(
  primary: UiAction[],
  secondary: UiAction[],
  options?: {
    onUiAction?: (action: UiAction) => void | Promise<void>;
    onLocalAction?: (command: "help" | "clear" | "compact" | "exit") => void | Promise<void>;
  },
): PaletteRecord[] {
  const onUiAction = options?.onUiAction ?? (() => undefined);
  const onLocalAction = options?.onLocalAction ?? (() => undefined);
  const records: PaletteRecord[] = [];

  primary.forEach((action, index) => {
    records.push({
      entry: createPaletteEntry(action, "查看", () => onUiAction(action)),
      kind: action.kind,
      index,
    });
  });

  const secondaryOffset = records.length;
  secondary.forEach((action, offset) => {
    records.push({
      entry: createPaletteEntry(action, "控制", () => onUiAction(action)),
      kind: action.kind,
      index: secondaryOffset + offset,
    });
  });

  const localActions: Array<"help" | "clear" | "compact" | "exit"> = ["help", "clear", "compact", "exit"];
  const localOffset = records.length;
  localActions.forEach((command, offset) => {
    const index = localOffset + offset;
    records.push({
      entry: {
        id: command,
        label: localLabel(command),
        group: "本地",
        disabled: false,
        run: () => onLocalAction(command),
        haystack: `${localLabel(command)} ${command}`.toLowerCase(),
      },
      kind: "local",
      index,
    });
  });

  return records;
}

function createPaletteEntry(action: UiAction, group: PaletteEntry["group"], run: () => void | Promise<void>): PaletteEntry {
  return {
    id: action.id,
    label: action.label,
    group,
    disabled: action.disabled,
    run,
    haystack: `${action.label} ${action.value} ${action.id}`.toLowerCase(),
  };
}

function buildRecentEntries(records: PaletteRecord[], recentIds: string[]): PaletteEntry[] {
  const byId = new Map(records.map((record) => [record.entry.id, record.entry] as const));
  const seen = new Set<string>();
  const entries: PaletteEntry[] = [];

  for (const id of recentIds) {
    if (seen.has(id)) {
      continue;
    }
    const entry = byId.get(id);
    if (!entry) {
      continue;
    }
    seen.add(id);
    entries.push(entry);
  }

  return entries;
}

function sortPaletteRecords(records: PaletteRecord[], context: PaletteContext): PaletteRecord[] {
  return [...records]
    .map((record) => ({ record, weight: getRecommendationWeight(record, context) }))
    .sort((left, right) => {
      if (left.weight !== right.weight) {
        return left.weight - right.weight;
      }
      return left.record.index - right.record.index;
    })
    .map(({ record }) => record);
}

function getRecommendationWeight(record: PaletteRecord, context: PaletteContext): number {
  if (context === "approval") {
    if (isApprovalEntry(record)) {
      return 0;
    }
    return getRunningWeight(record) + 1;
  }

  if (context === "idle") {
    const idleRank = getIdleRank(record);
    if (idleRank !== null) {
      return idleRank;
    }
    return 4 + getRunningWeight(record);
  }

  return getRunningWeight(record);
}

function getRunningWeight(record: PaletteRecord): number {
  if (record.kind === "drawer" || record.kind === "approval" || record.kind === "approval_nav") {
    return 0;
  }
  if (record.kind === "command") {
    return 1;
  }
  return 2;
}

function getIdleRank(record: PaletteRecord): number | null {
  switch (record.entry.id) {
    case "cognition":
      return 0;
    case "meta":
      return 1;
    case "help":
      return 2;
    case "compact":
      return 3;
    default:
      return null;
  }
}

function isApprovalEntry(record: PaletteRecord): boolean {
  return record.kind === "approval" || record.kind === "approval_nav" || record.entry.id === "approvals" || record.entry.haystack.includes("审批") || record.entry.haystack.includes("approval");
}

function localLabel(command: "help" | "clear" | "compact" | "exit"): string {
  switch (command) {
    case "help":
      return "帮助";
    case "clear":
      return "清空";
    case "compact":
      return "紧凑视图";
    case "exit":
      return "退出";
  }
}
