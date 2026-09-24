/**
 * Merges freshly parsed bulletin courses into the committed catalog
 * (v6.7.0). The rules come from the professor (2026-09-24):
 *
 * - The catalog only grows. A course the bulletin no longer lists is marked
 *   `retired` and kept, with its skill links, so a student who took it can
 *   still find and keep it.
 * - A renamed course keeps its code; the old title moves to
 *   `previousTitles` so search finds either name.
 * - A new code carrying a retired course's title is reported as a suspected
 *   renumbering for a person to confirm; it is never linked automatically.
 * - Independent Studies are excluded (decision of 2026-07-28), and courses
 *   whose content varies by section (topics, internships, contemporary
 *   issues) are "freeform": no static skill mapping.
 *
 * Output is sorted so a rerun with no bulletin change produces no diff.
 */

import type { ParsedCourse } from "./bulletin";

export interface CatalogCourse {
  code: string;
  altCodes: string[];
  title: string;
  previousTitles: string[];
  credits: number;
  prereq: string[][];
  prereqUncertain: boolean;
  retired: boolean;
  special?: "freeform";
}

export interface MergeReport {
  added: string[];
  retired: string[];
  retitled: { code: string; from: string; to: string }[];
  suspectedRenumbering: { from: string; to: string; title: string }[];
  prereqChanged: string[];
}

const EXCLUDED = /^independent stud/i;
const FREEFORM = /\b(internship|topics?|contemporary issues|special studies|practicum in)\b/i;

const byCode = (a: { code: string }, b: { code: string }) => a.code.localeCompare(b.code);
const sameTitle = (a: string, b: string) => a.trim().toLowerCase() === b.trim().toLowerCase();

export function mergeCatalog(
  existing: CatalogCourse[],
  parsed: ParsedCourse[],
  inScope: Set<string>,
  /** First run only: the hand-written catalog's titles are not renames. */
  options: { seeding?: boolean } = {},
): { courses: CatalogCourse[]; report: MergeReport } {
  const report: MergeReport = { added: [], retired: [], retitled: [], suspectedRenumbering: [], prereqChanged: [] };
  const fresh = new Map(parsed.filter((p) => !EXCLUDED.test(p.title)).map((p) => [p.code, p]));
  const out = new Map<string, CatalogCourse>();

  for (const old of existing) {
    const p = fresh.get(old.code);
    if (!p) {
      if (!old.retired) report.retired.push(old.code);
      out.set(old.code, { ...old, retired: true });
      continue;
    }
    const previousTitles = [...old.previousTitles];
    if (!sameTitle(old.title, p.title) && !options.seeding) {
      if (!previousTitles.some((t) => sameTitle(t, old.title))) previousTitles.push(old.title);
      report.retitled.push({ code: old.code, from: old.title, to: p.title });
    }
    if (JSON.stringify(old.prereq) !== JSON.stringify(p.prereq) && !options.seeding) report.prereqChanged.push(old.code);
    // An existing course keeps its own flag: the title heuristic below is
    // only for courses new to the catalog (ISA 621 "... Topics I" is not).
    out.set(old.code, build(p, previousTitles, old.special ?? null));
  }

  for (const code of [...inScope].sort()) {
    if (out.has(code)) continue;
    const p = fresh.get(code);
    if (!p) continue;
    out.set(code, build(p, [], undefined));
    report.added.push(code);
    const gone = existing.find((e) => !fresh.has(e.code) && sameTitle(e.title, p.title));
    if (gone) report.suspectedRenumbering.push({ from: gone.code, to: code, title: p.title });
  }

  return { courses: [...out.values()].sort(byCode), report };
}

/** `special` null means "decided already, not freeform"; undefined means "new, detect". */
function build(p: ParsedCourse, previousTitles: string[], special: "freeform" | null | undefined): CatalogCourse {
  const freeform = special === undefined ? (FREEFORM.test(p.title) ? "freeform" : undefined) : (special ?? undefined);
  return {
    code: p.code,
    altCodes: [...p.altCodes].sort(),
    title: p.title,
    previousTitles,
    credits: p.credits,
    prereq: p.prereq,
    prereqUncertain: p.prereqUncertain,
    retired: false,
    ...(freeform ? { special: freeform } : {}),
  };
}
