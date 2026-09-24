/**
 * Catalog collector (v6.7.0): the bulletin and Simple Syllabus to
 * web/catalog/*.json. Run by the catalog-refresh GitHub Action and locally
 * with `npm run catalog:collect`. Read-only on the network; writes only the
 * catalog folder. See docs/superpowers/specs/2026-09-24-fsb-catalog-skills-and-picker-design.md.
 */

import { existsSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { parseCoursePage, parseProgram, type ParsedCourse, type ProgramGroup } from "./bulletin";
import { mergeCatalog, type CatalogCourse, type MergeReport } from "./merge";
import { DOC_URL, extractEvidence, listSections, mergeEvidence, type CourseEvidence } from "./syllabi";
import { COURSES as SEED } from "./seed-courses";

const DIR = path.join(process.cwd(), "catalog");
/** Enough sections to see how a course is taught, few enough to be polite. */
const SECTIONS_PER_COURSE = 4;
const PAUSE_MS = 250;

interface ProgramConfig { key: string; name: string; kind: "core" | "major" | "comajor" | "minor"; url: string }
export interface CatalogProgram extends ProgramConfig { groups: ProgramGroup[] }

const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

const ATTEMPTS = 4;

/**
 * A page's text, retried with backoff. A dropped connection or a body that
 * breaks off mid-read is retried like an HTTP error: the first Action run
 * (2026-09-24) died on "other side closed" from the Bulletin's server.
 */
export async function fetchText(
  url: string,
  fetchImpl: typeof fetch = fetch,
  wait: (ms: number) => Promise<void> = sleep,
): Promise<string> {
  let last = "";
  for (let attempt = 1; attempt <= ATTEMPTS; attempt++) {
    try {
      const res = await fetchImpl(url, { headers: { "user-agent": "ChatISA catalog refresh (Miami University FSB)" } });
      if (res.ok) return await res.text();
      last = `HTTP ${res.status}`;
    } catch (err) {
      last = err instanceof Error ? err.message : String(err);
    }
    if (attempt < ATTEMPTS) await wait(2000 * attempt);
  }
  throw new Error(`${url}: ${last} (after ${ATTEMPTS} attempts)`);
}

const get = (url: string) => fetchText(url);

function readJson<T>(name: string, fallback: T): T {
  const file = path.join(DIR, name);
  return existsSync(file) ? (JSON.parse(readFileSync(file, "utf8")) as T) : fallback;
}

/** Stable JSON: two-space indent, trailing newline. Callers sort their data. */
function writeJson(name: string, data: unknown): void {
  writeFileSync(path.join(DIR, name), JSON.stringify(data, null, 2) + "\n");
}

/** "2026-27" for any date from August 2026 through July 2027. */
export function academicYear(d: Date): string {
  const y = d.getUTCMonth() >= 7 ? d.getUTCFullYear() : d.getUTCFullYear() - 1;
  return `${y}-${String((y + 1) % 100).padStart(2, "0")}`;
}

async function main() {
  const now = new Date();
  const config = readJson<{ programs: ProgramConfig[] }>("programs.config.json", { programs: [] });

  // 1. Programs.
  const programs: CatalogProgram[] = [];
  const unparsed: { program: string; row: string }[] = [];
  for (const p of config.programs) {
    const { groups, unparsed: bad } = parseProgram(await get(p.url));
    programs.push({ ...p, groups });
    for (const row of bad) unparsed.push({ program: p.key, row });
    await sleep(PAUSE_MS);
  }
  const programCodes = new Set(programs.flatMap((p) => p.groups.flatMap((g) => g.items.flatMap((i) => i.codes))));

  // 2. Courses: every prefix any program or the existing catalog uses.
  const seeding = !existsSync(path.join(DIR, "courses.json"));
  const existing: CatalogCourse[] = seeding
    ? SEED.map((c) => ({
        code: c.code, altCodes: c.altCodes, title: c.title, previousTitles: [], credits: c.credits,
        prereq: [], prereqUncertain: false, retired: false, ...(c.special ? { special: c.special } : {}),
      }))
    : readJson<CatalogCourse[]>("courses.json", []);
  const prefixes = [...new Set([...programCodes, ...existing.map((c) => c.code)].map((c) => c.slice(0, 3)))].sort();
  const parsed: ParsedCourse[] = [];
  for (const prefix of prefixes) {
    parsed.push(...parseCoursePage(await get(`https://bulletin.miamioh.edu/courses-instruction/${prefix.toLowerCase()}/`)));
    await sleep(PAUSE_MS);
  }
  const { courses, report } = mergeCatalog(existing, parsed, programCodes, { seeding });
  const inCatalog = new Set(courses.map((c) => c.code));
  // Deliberate exclusions (Independent Studies) are not problems to report.
  const excluded = new Set(parsed.filter((p) => /^independent stud/i.test(p.title)).map((p) => p.code));
  const missingFromBulletin = [...programCodes].filter((c) => !inCatalog.has(c) && !excluded.has(c)).sort();

  const descriptions: Record<string, { description: string; prereqText: string }> = {};
  for (const p of parsed) if (inCatalog.has(p.code)) descriptions[p.code] = { description: p.description, prereqText: p.prereqText };

  // 3. Syllabi: current and future terms, a few sections per course.
  const previous = readJson<Record<string, CourseEvidence>>("syllabi.json", {});
  const alt = new Map<string, string>();
  for (const c of courses) for (const a of c.altCodes) alt.set(a, c.code);
  const fetchJson = async (url: string) => JSON.parse(await get(url));
  const listing = await listSections(fetchJson);
  const syllabi: Record<string, CourseEvidence> = { ...previous };
  let fetched = 0;
  for (const [listed, sections] of [...listing.entries()].sort()) {
    const code = inCatalog.has(listed) ? listed : alt.get(listed);
    if (!code) continue;
    const picked = sections.slice(0, SECTIONS_PER_COURSE);
    const evidence = [];
    for (const s of picked) {
      evidence.push({ term: s.term, evidence: extractEvidence(await fetchJson(DOC_URL + s.code)) });
      fetched++;
      await sleep(PAUSE_MS);
    }
    const prior = previous[code];
    const merged = mergeEvidence(prior
      ? [...evidence, { term: prior.terms[0] ?? "", evidence: prior }]
      : evidence);
    // Section counts describe this pull, not the history.
    syllabi[code] = { ...merged, terms: [...new Set([...(prior?.terms ?? []), ...merged.terms])].filter(Boolean).sort(), sections: picked.length };
  }

  writeJson("programs.json", { bulletinYear: academicYear(now), programs });
  writeJson("courses.json", courses);
  writeJson("descriptions.json", Object.fromEntries(Object.entries(descriptions).sort()));
  writeJson("syllabi.json", Object.fromEntries(Object.entries(syllabi).sort()));
  writeJson("manifest.json", { collectedAt: now.toISOString(), bulletinYear: academicYear(now), syllabusSectionsRead: fetched });
  writeJson(".collect-report.json", {
    seeding, report, unparsed, missingFromBulletin,
    counts: { programs: programs.length, courses: courses.length, withSyllabus: Object.keys(syllabi).length },
  } satisfies { seeding: boolean; report: MergeReport; unparsed: unknown; missingFromBulletin: string[]; counts: unknown });
  console.log(`catalog: ${programs.length} programs, ${courses.length} courses, ${Object.keys(syllabi).length} with syllabus evidence, ${fetched} syllabi read`);
  if (unparsed.length || missingFromBulletin.length) {
    console.log(`needs attention: ${unparsed.length} unparsed rows, ${missingFromBulletin.length} program courses not in the bulletin`);
  }
}

if (process.argv[1] && process.argv[1].endsWith("collect.ts")) {
  main().catch((err) => {
    console.error(err);
    process.exit(1);
  });
}
