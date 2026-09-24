/**
 * Job Scout's local profile, saved-job, and project-artifact storage.
 * localStorage on purpose: the server never learns which courses a student
 * took, which skills they confirmed, which jobs they saved, or what they
 * built (local-first decision, 2026-07-28). A server record appears only
 * when they hand a job to JobApp Drafter.
 */

import type { PublishedWork } from "@/lib/portfolio/published";
import type { CourseSkillLevel } from "./course-skills";

const PROFILE_KEY = "js-profile-v1";
const SAVED_KEY = "js-saved-v1";
const PROJECTS_KEY = "js-projects-v1";

export interface ProfileExtra {
  skillId: string;
  level: CourseSkillLevel;
  /** Where the student confirmed it from, for honest display. */
  source: "resume" | "freeform" | "manual";
  evidence?: string;
}

/**
 * A student's own correction to a computed skill level (user feedback,
 * 2026-07-29): the courses say Working but they know they are Strong, or
 * the reverse. Stored in the student's words, applied over the noisy-OR.
 */
export interface SkillOverride {
  skillId: string;
  level: "strong" | "working" | "introduced";
}

/**
 * One course on the student's record (v6.7.0). Done or Taking now; a
 * course not on the record is "Not yet". `term` is when it was (or is being)
 * taken, where known; `addedBecause` marks a prerequisite added automatically
 * because of another course, so the picker can label it and offer Remove.
 */
export interface ProfileCourse {
  code: string;
  status: "done" | "now";
  term?: string;
  addedBecause?: string;
}

/**
 * Profile v2 (v6.7.0): the programs the student is in, and courses with a
 * status. v1 stored a plain list of course codes; it migrates on read
 * (every course Done), so nobody loses what they entered. Stored under the
 * same key; the `v` field tells the two apart.
 */
export interface ScoutProfile {
  v: 2;
  /** Program keys from catalog/programs.config.json ("business-analytics"). */
  programs: string[];
  courses: ProfileCourse[];
  /** Prerequisites the student removed; never added back automatically. */
  removedPrereqs: string[];
  extras: ProfileExtra[];
  overrides?: SkillOverride[];
}

/** Every course the student has, done or in progress. */
export function courseCodes(profile: Pick<ScoutProfile, "courses">): string[] {
  return profile.courses.map((c) => c.code);
}

/**
 * The academic term a date falls in: Fall (August to December), Spring
 * (January to May), Summer (June and July). Winter term courses are simply
 * recorded under Spring.
 */
export function currentTerm(date: Date): string {
  const m = date.getMonth();
  const y = date.getFullYear();
  return m >= 7 ? `Fall ${y}` : m <= 4 ? `Spring ${y}` : `Summer ${y}`;
}

/** Taking-now courses recorded in an earlier term: "Did you finish it?" */
export function staleTakingNow(profile: Pick<ScoutProfile, "courses">, date: Date): ProfileCourse[] {
  const term = currentTerm(date);
  return profile.courses.filter((c) => c.status === "now" && c.term !== undefined && c.term !== term);
}

/**
 * v2 keeps a snapshot per save (title/company/apply link), so a saved job
 * outlives its posting's retirement from the weekly feed (user feedback,
 * 2026-07-29: "where do the saved jobs go?").
 */
export interface SavedSnapshot {
  id: string;
  title: string;
  company: string;
  applyUrl: string;
  savedAt: string;
}

export interface SavedState {
  v: 2;
  saved: SavedSnapshot[];
  hiddenIds: string[];
}

/** A generated portfolio project. Scaffold JSON lives in device-files. */
export interface ProjectRecord {
  id: string;
  repoName: string;
  summary: string;
  skillIds: string[];
  createdAt: string;
  /** "polished" = organized from the student's own uploads (the primary
   * path, 2026-07-29); absent or "scaffold" = generated from scratch. */
  mode?: "scaffold" | "polished";
  /** Set when the student pastes their pushed repo. Gates the profile
   * contribution: an unbuilt scaffold never inflates a skill. */
  repoUrl: string | null;
}

export interface ProjectsState {
  v: 1;
  projects: ProjectRecord[];
}

/** Corrupt or foreign JSON degrades to "no profile", never to a crash. */
export function loadProfile(): ScoutProfile | null {
  try {
    const raw = localStorage.getItem(PROFILE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as { v?: number; courses?: unknown[]; programs?: string[]; removedPrereqs?: string[]; extras?: ProfileExtra[]; overrides?: SkillOverride[] };
    if (!parsed || !Array.isArray(parsed.courses)) return null;
    if (parsed.v === 1) {
      return {
        v: 2, programs: [], removedPrereqs: [],
        courses: (parsed.courses as unknown[]).filter((c): c is string => typeof c === "string").map((code) => ({ code, status: "done" as const })),
        extras: parsed.extras ?? [],
        overrides: parsed.overrides ?? [],
      };
    }
    if (parsed.v !== 2) return null;
    return {
      v: 2,
      programs: parsed.programs ?? [],
      removedPrereqs: parsed.removedPrereqs ?? [],
      courses: (parsed.courses as ProfileCourse[]).filter((c) => c && typeof c.code === "string" && (c.status === "done" || c.status === "now")),
      extras: parsed.extras ?? [],
      overrides: parsed.overrides ?? [],
    };
  } catch {
    return null;
  }
}

export function saveProfile(profile: ScoutProfile): void {
  localStorage.setItem(PROFILE_KEY, JSON.stringify(profile));
}

export function clearProfile(): void {
  localStorage.removeItem(PROFILE_KEY);
}

const EMPTY_SAVED: SavedState = { v: 2, saved: [], hiddenIds: [] };

export function loadSaved(): SavedState {
  try {
    const raw = localStorage.getItem(SAVED_KEY);
    if (!raw) return { ...EMPTY_SAVED };
    const parsed = JSON.parse(raw) as
      | SavedState
      | { v: 1; savedIds?: string[]; hiddenIds?: string[] };
    if (parsed?.v === 2) {
      return {
        v: 2,
        saved: parsed.saved ?? [],
        hiddenIds: parsed.hiddenIds ?? [],
      };
    }
    if (parsed?.v === 1) {
      // v1 stored bare ids with no snapshot; carry them forward as
      // placeholder rows rather than dropping a student's saves.
      return {
        v: 2,
        saved: (parsed.savedIds ?? []).map((id) => ({
          id,
          title: "Saved posting",
          company: "",
          applyUrl: "",
          savedAt: "",
        })),
        hiddenIds: parsed.hiddenIds ?? [],
      };
    }
    return { ...EMPTY_SAVED };
  } catch {
    return { ...EMPTY_SAVED };
  }
}

function writeSaved(state: SavedState): SavedState {
  localStorage.setItem(SAVED_KEY, JSON.stringify(state));
  return state;
}

export function toggleSaved(snapshot: Omit<SavedSnapshot, "savedAt">): SavedState {
  const state = loadSaved();
  const existing = state.saved.some((s) => s.id === snapshot.id);
  state.saved = existing
    ? state.saved.filter((s) => s.id !== snapshot.id)
    : [...state.saved, { ...snapshot, savedAt: new Date().toISOString() }];
  return writeSaved(state);
}

export function hidePosting(id: string): SavedState {
  const state = loadSaved();
  if (!state.hiddenIds.includes(id)) state.hiddenIds.push(id);
  state.saved = state.saved.filter((s) => s.id !== id);
  return writeSaved(state);
}

// --------------------------------------------------------------- projects

export function loadProjects(): ProjectsState {
  try {
    const raw = localStorage.getItem(PROJECTS_KEY);
    if (!raw) return { v: 1, projects: [] };
    const parsed = JSON.parse(raw) as ProjectsState;
    if (parsed?.v !== 1) return { v: 1, projects: [] };
    return { v: 1, projects: parsed.projects ?? [] };
  } catch {
    return { v: 1, projects: [] };
  }
}

function writeProjects(state: ProjectsState): ProjectsState {
  localStorage.setItem(PROJECTS_KEY, JSON.stringify(state));
  return state;
}

export function addProject(record: ProjectRecord): ProjectsState {
  const state = loadProjects();
  state.projects = [record, ...state.projects.filter((p) => p.id !== record.id)];
  return writeProjects(state);
}

export function setProjectRepoUrl(id: string, repoUrl: string | null): ProjectsState {
  const state = loadProjects();
  state.projects = state.projects.map((p) =>
    p.id === id ? { ...p, repoUrl } : p,
  );
  return writeProjects(state);
}

export function removeProject(id: string): ProjectsState {
  const state = loadProjects();
  state.projects = state.projects.filter((p) => p.id !== id);
  return writeProjects(state);
}

/**
 * The profile contribution earned by REAL projects: a polished project
 * counts immediately (it was organized from work the student already did),
 * while a from-scratch scaffold counts only once its repo URL exists — an
 * unbuilt scaffold never inflates a skill (user decisions, 2026-07-29).
 */
export function projectExtras(state: ProjectsState): ProfileExtra[] {
  return state.projects
    .filter((p) => p.mode === "polished" || p.repoUrl)
    .flatMap((p) =>
      p.skillIds.map((skillId) => ({
        skillId,
        level: "applied" as const,
        source: "manual" as const,
        evidence: `built ${p.repoName}`,
      })),
    );
}

/** Skills demonstrated by sites published with the Portfolio Builder count
 * like built projects: the repo exists, so the work is real. */
export function publishedExtras(works: PublishedWork[]): ProfileExtra[] {
  return works.flatMap((w) =>
    w.skillIds.map((skillId) => ({
      skillId,
      level: "applied" as const,
      source: "manual" as const,
      evidence: `published ${w.title}`,
    })),
  );
}
