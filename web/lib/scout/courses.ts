/**
 * The FSB course catalog Job Scout and the Portfolio Builder use (v6.7.0).
 *
 * Read from catalog/courses.json, which the catalog pipeline builds from the
 * Miami Bulletin: every course in the FSB business core, the BSB majors,
 * the co-majors, and the AI for Business minor, plus every course that was
 * ever in the catalog. The catalog only grows: a course the bulletin drops
 * stays here with `retired: true`, so a student who took it can still find
 * it; a renamed course keeps its old titles in `previousTitles`.
 *
 * `special: "freeform"` courses (internships, topics seminars) have no
 * static skill mapping: their content varies per student, so the profile
 * asks one line about what they worked on and maps it with a model instead.
 * Credits drive match weighting: a 1.5-credit course contributes half the
 * depth of a 3-credit one. Independent Studies are excluded (2026-07-28).
 */

import catalog from "@/catalog/courses.json";

export interface CourseDef {
  /** Primary code as students know it, e.g. "ISA 401". */
  code: string;
  /** Cross-listed and graduate codes that count as the same course. */
  altCodes: string[];
  title: string;
  /** Earlier titles of the same course, newest last. */
  previousTitles: string[];
  credits: number;
  /** AND of OR-groups of course codes; empty when there are none. */
  prereq: string[][];
  /** Instructor permission or standing can substitute: never infer from it. */
  prereqUncertain: boolean;
  /** No longer in the bulletin; still searchable and still counted. */
  retired: boolean;
  special?: "freeform";
}

const isStrings = (x: unknown): x is string[] => Array.isArray(x) && x.every((s) => typeof s === "string");

/**
 * Checked once at load (review fix, v6.7.0), like course-skills.json: a
 * malformed file from a catalog PR fails its integrity step and startup,
 * not a student's first click in the checklist.
 */
export function validateCourses(raw: unknown): CourseDef[] {
  if (!Array.isArray(raw)) throw new Error("catalog/courses.json must be an array");
  return raw.map((c, i) => {
    const x = c as Record<string, unknown>;
    const ok =
      typeof x.code === "string" && typeof x.title === "string" && typeof x.credits === "number" &&
      isStrings(x.altCodes) && isStrings(x.previousTitles) &&
      Array.isArray(x.prereq) && x.prereq.every(isStrings) &&
      typeof x.prereqUncertain === "boolean" && typeof x.retired === "boolean" &&
      (x.special === undefined || x.special === "freeform");
    if (!ok) throw new Error(`catalog/courses.json entry ${i} is malformed`);
    return c as CourseDef;
  });
}

export const COURSES: CourseDef[] = validateCourses(catalog);

const byCode = new Map<string, CourseDef>();
for (const c of COURSES) {
  byCode.set(c.code, c);
  for (const a of c.altCodes) if (!byCode.has(a)) byCode.set(a, c);
}

export function getCourse(code: string): CourseDef | undefined {
  return byCode.get(code);
}

/**
 * Search match on code (with or without the space), cross-listed codes,
 * title, and previous titles, so a renamed course is still found by the
 * name students know.
 */
export function matchesCourse(c: CourseDef, query: string): boolean {
  const q = query.trim().toLowerCase();
  if (!q) return false;
  const hay = [c.code, ...c.altCodes, c.title, ...c.previousTitles].join(" | ").toLowerCase();
  return hay.includes(q) || hay.replace(/\s+/g, "").includes(q.replace(/\s+/g, ""));
}

/**
 * The ISA courses students most often take (instructor's own popularity
 * call, 2026-07-29), offered by the showcase picker when a student has no
 * Job Scout courses yet. A unit test asserts every code here exists in
 * COURSES.
 */
export const POPULAR_CODES: Record<string, string[]> = {
  foundations: ["ISA 125", "ISA 225", "ISA 235"],
  core300: [
    "ISA 301", "ISA 303", "ISA 305", "ISA 321", "ISA 336",
    "ISA 345", "ISA 365", "ISA 381", "ISA 387", "ISA 391",
  ],
  advanced400: [
    "ISA 401", "ISA 403", "ISA 405", "ISA 406", "ISA 414",
    "ISA 419", "ISA 444", "ISA 491", "ISA 495", "ISA 496",
  ],
};
