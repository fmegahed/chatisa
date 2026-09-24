/**
 * Automatic prerequisites (v6.7.0, the professor's rules of 2026-09-24).
 *
 * Registration enforces prerequisites, so a course a student has done or is
 * taking now is good evidence they completed its prerequisites. Those are
 * added as Done, labelled with the course that caused them, and removable.
 *
 * - An "or" group takes the student's own path: an option one of their own
 *   programs requires, then one the business core requires, then one from
 *   their major's department, then the first listed.
 * - A group the student already meets (any option on their record) adds
 *   nothing.
 * - Skipped entirely: a course whose prerequisite allows instructor
 *   permission or depends on class standing (nothing can be inferred), a
 *   group of four or more options (too uncertain to guess), and options
 *   outside the catalog.
 * - A prerequisite the student removed is never added back, and nothing is
 *   inferred through it.
 * - Courses the student set themselves are never changed.
 */

import { getCourse as catalogCourse, type CourseDef } from "./courses";
import { programCodes as catalogProgramCodes } from "./programs";
import type { ProfileCourse } from "./profile-store";

export interface PrereqDeps {
  getCourse: (code: string) => Pick<CourseDef, "code" | "prereq" | "prereqUncertain"> | undefined;
  /** Every course code a program lists. */
  programCodes: (key: string) => string[];
}

const DEFAULT_DEPS: PrereqDeps = { getCourse: catalogCourse, programCodes: catalogProgramCodes };
const CORE = "business-core";
const MAX_OPTIONS = 4;

const prefix = (code: string) => code.slice(0, 3);

/** The department a program is mostly made of ("FIN" for Finance). */
function department(codes: string[]): string | null {
  const counts = new Map<string, number>();
  for (const c of codes) counts.set(prefix(c), (counts.get(prefix(c)) ?? 0) + 1);
  let best: string | null = null;
  for (const [p, n] of counts) if (best === null || n > (counts.get(best) ?? 0)) best = p;
  return best;
}

/**
 * The student's own path through an "or" choice: an option one of their own
 * programs requires, then one the business core requires, then one from
 * their major's department, then the first listed. Options must already be
 * in-catalog primary codes, and there must be at least one.
 */
export function pathChooser(
  programs: string[],
  deps: PrereqDeps = DEFAULT_DEPS,
): (options: string[]) => string {
  const primary = (code: string) => deps.getCourse(code)?.code ?? code;
  const own = programs.filter((k) => k !== CORE);
  const ownRequired = new Set(own.flatMap((k) => deps.programCodes(k)).map(primary));
  const coreRequired = new Set(deps.programCodes(CORE).map(primary));
  const departments = new Set(own.map((k) => department(deps.programCodes(k))).filter((d): d is string => d !== null));
  return (options) =>
    options.find((o) => ownRequired.has(o)) ??
    options.find((o) => coreRequired.has(o)) ??
    options.find((o) => departments.has(prefix(o))) ??
    options[0];
}

export function inferPrereqs(
  state: { programs: string[]; courses: ProfileCourse[]; removedPrereqs: string[] },
  deps: PrereqDeps = DEFAULT_DEPS,
): ProfileCourse[] {
  const primary = (code: string) => deps.getCourse(code)?.code ?? code;
  const result = [...state.courses];
  const have = new Set(state.courses.map((c) => primary(c.code)));
  const removed = new Set(state.removedPrereqs.map(primary));
  const choose = pathChooser(state.programs, deps);

  const visit = (code: string, root: string) => {
    const course = deps.getCourse(code);
    if (!course || course.prereqUncertain) return;
    for (const group of course.prereq) {
      if (group.some((o) => have.has(primary(o)))) continue;
      if (group.length >= MAX_OPTIONS) continue;
      const options = group.filter((o) => deps.getCourse(o)).map(primary);
      if (options.length === 0) continue;
      const pick = choose(options);
      if (removed.has(pick)) continue;
      have.add(pick);
      result.push({ code: pick, status: "done", addedBecause: root });
      visit(pick, root);
    }
  };

  for (const c of state.courses) visit(primary(c.code), c.code);
  return result;
}
