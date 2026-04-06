import type { PaletteSection } from "./commandPalette.js";

const DEFAULT_RECOMMENDED_LIMIT = 6;
const DEFAULT_RECENT_LIMIT = 4;

function dedupeEntries(section: PaletteSection["entries"], seen: Set<string>): PaletteSection["entries"] {
  return section.filter((entry) => {
    if (seen.has(entry.id)) {
      return false;
    }
    seen.add(entry.id);
    return true;
  });
}

function takeRecommended(entries: PaletteSection["entries"], limit: number): PaletteSection["entries"] {
  const enabled = entries.filter((entry) => !entry.disabled);
  if (enabled.length >= limit) {
    return enabled.slice(0, limit);
  }
  return [...enabled, ...entries.filter((entry) => entry.disabled).slice(0, limit - enabled.length)];
}

export function compactPaletteSections(sections: PaletteSection[], query: string): PaletteSection[] {
  if (query.trim().length > 0) {
    return sections;
  }

  const recommended = sections.find((section) => section.id === "recommended");
  const recent = sections.find((section) => section.id === "recent");
  const all = sections.find((section) => section.id === "all");

  if (!recommended || !recent || !all) {
    return sections;
  }

  const seen = new Set<string>();
  const recommendedEntries = dedupeEntries(takeRecommended(recommended.entries, DEFAULT_RECOMMENDED_LIMIT), seen);
  const recentEntries = dedupeEntries(recent.entries.slice(0, DEFAULT_RECENT_LIMIT), seen);
  const allEntries = dedupeEntries(all.entries, seen);

  return sections.map((section) => {
    if (section.id === "recommended") {
      return { ...section, entries: recommendedEntries };
    }
    if (section.id === "recent") {
      return { ...section, entries: recentEntries };
    }
    if (section.id === "all") {
      return { ...section, entries: allEntries };
    }
    return section;
  });
}
