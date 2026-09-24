/**
 * The course checklist's state changes (v6.7.0), shared by Job Scout's
 * profile and the Portfolio Builder's Classes step. Pure functions over the
 * programs, the courses with their status, and the prerequisites a student
 * removed; no React, no storage.
 *
 * Prerequisites are re-derived after every change: courses the student set
 * themselves are kept exactly, and everything added automatically is
 * recomputed from them. So taking a course away takes its prerequisites
 * with it, and a new program choice re-routes "or" picks along the new path.
 */

import { getCourse as catalogCourse } from "./courses";
import { getProgram, programCodes as catalogProgramCodes, type ProgramGroup } from "./programs";
import { inferPrereqs, pathChooser, type PrereqDeps } from "./prereqs";
import { currentTerm, type ProfileCourse } from "./profile-store";

export interface ChecklistState {
  programs: string[];
  courses: ProfileCourse[];
  removedPrereqs: string[];
}

export type RowStatus = "done" | "now" | "none";

export interface ChecklistDeps extends PrereqDeps {
  programGroups: (key: string) => ProgramGroup[];
}

const DEFAULT_DEPS: ChecklistDeps = {
  getCourse: catalogCourse,
  programCodes: catalogProgramCodes,
  programGroups: (key) => getProgram(key)?.groups ?? [],
};

export function statusOf(state: Pick<ChecklistState, "courses">, code: string): RowStatus {
  return state.courses.find((c) => c.code === code)?.status ?? "none";
}

/** Prerequisites added automatically because of `root`, in the order added. */
export function addedFor(state: Pick<ChecklistState, "courses">, root: string): string[] {
  return state.courses.filter((c) => c.addedBecause === root).map((c) => c.code);
}

/** Keep the student's own courses; recompute every automatic one. */
function settle(state: ChecklistState, deps: PrereqDeps): ChecklistState {
  const own = state.courses.filter((c) => !c.addedBecause);
  return { ...state, courses: inferPrereqs({ ...state, courses: own }, deps) };
}

/** Codes present after a change that were not there before, for the live region. */
function newlyAdded(before: ChecklistState, after: ChecklistState): string[] {
  const had = new Set(before.courses.map((c) => c.code));
  return after.courses.filter((c) => !had.has(c.code)).map((c) => c.code);
}

export function setCourseStatus(
  state: ChecklistState,
  code: string,
  status: RowStatus,
  deps: PrereqDeps = DEFAULT_DEPS,
  now: Date = new Date(),
): { state: ChecklistState; added: string[] } {
  const existing = state.courses.find((c) => c.code === code);
  let removedPrereqs = state.removedPrereqs;
  let courses: ProfileCourse[];
  if (status === "none") {
    courses = state.courses.filter((c) => c.code !== code);
    // Taking away a prerequisite that was added for the student is the same
    // as removing it: it must not come straight back.
    if (existing?.addedBecause && !removedPrereqs.includes(code)) removedPrereqs = [...removedPrereqs, code];
  } else {
    const entry: ProfileCourse = status === "now" ? { code, status, term: currentTerm(now) } : { code, status };
    courses =
      existing && !existing.addedBecause
        ? state.courses.map((c) => (c.code === code ? entry : c))
        : [...state.courses.filter((c) => c.code !== code), entry];
    removedPrereqs = removedPrereqs.filter((c) => c !== code);
  }
  let next = settle({ ...state, courses, removedPrereqs }, deps);
  // A course the student set themselves and now says Not yet to would come
  // straight back as another course's prerequisite; their word wins.
  if (status === "none" && next.courses.some((c) => c.code === code)) {
    next = settle({ ...state, courses, removedPrereqs: [...removedPrereqs, code] }, deps);
  }
  return { state: next, added: newlyAdded(state, next).filter((c) => c !== code) };
}

export function removeAddedPrereq(
  state: ChecklistState,
  code: string,
  deps: PrereqDeps = DEFAULT_DEPS,
): ChecklistState {
  return setCourseStatus(state, code, "none", deps).state;
}

/**
 * A group every student must complete (no "select" instruction, and not a
 * capstone choice): the rows the "mark the required courses Done" shortcut
 * may fill in.
 */
export function isAllRequired(group: ProgramGroup): boolean {
  const all = !group.instruction || /^complete/i.test(group.instruction);
  return all && !/capstone/i.test(`${group.title} ${group.subtitle ?? ""}`);
}

/** Mark every all-required row of a program Done, leaving rows already set. */
export function markRequiredDone(
  state: ChecklistState,
  programKey: string,
  deps: ChecklistDeps = DEFAULT_DEPS,
): { state: ChecklistState; added: string[] } {
  const choose = pathChooser(state.programs, deps);
  const have = new Set(state.courses.map((c) => c.code));
  const courses = [...state.courses];
  for (const group of deps.programGroups(programKey).filter(isAllRequired)) {
    for (const item of group.items) {
      if (item.codes.some((c) => have.has(c))) continue;
      const options = item.codes.filter((c) => deps.getCourse(c));
      if (options.length === 0) continue;
      const pick = choose(options);
      have.add(pick);
      courses.push({ code: pick, status: "done" });
    }
  }
  const next = settle({ ...state, courses }, deps);
  return { state: next, added: newlyAdded(state, next) };
}

/** "Finished": Taking-now courses become Done. */
export function finishCourses(
  state: ChecklistState,
  codes: string[],
  deps: PrereqDeps = DEFAULT_DEPS,
): ChecklistState {
  const set = new Set(codes);
  const courses = state.courses.map((c) => (set.has(c.code) ? { code: c.code, status: "done" as const } : c));
  return settle({ ...state, courses }, deps);
}

/** "Still taking": move the courses to the current term so the prompt goes away. */
export function keepTaking(state: ChecklistState, codes: string[], now: Date = new Date()): ChecklistState {
  const set = new Set(codes);
  const term = currentTerm(now);
  return { ...state, courses: state.courses.map((c) => (set.has(c.code) ? { ...c, term } : c)) };
}
