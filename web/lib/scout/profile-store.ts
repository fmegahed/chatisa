/**
 * Job Scout's local profile, saved-job, and project-artifact storage.
 * localStorage on purpose: the server never learns which courses a student
 * took, which skills they confirmed, which jobs they saved, or what they
 * built (local-first decision, 2026-07-28). A server record appears only
 * when they hand a job to JobApp Drafter.
 */

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

export interface ScoutProfile {
  v: 1;
  /** Primary course codes ("ISA 401"), as checked off by the student. */
  courses: string[];
  extras: ProfileExtra[];
  overrides?: SkillOverride[];
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
    const parsed = JSON.parse(raw) as ScoutProfile;
    if (parsed?.v !== 1 || !Array.isArray(parsed.courses)) return null;
    return {
      v: 1,
      courses: parsed.courses,
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

// -------------------------------------------------------------- portfolio

const PORTFOLIO_KEY = "js-portfolio-v1";

/**
 * The published portfolio site's record (v6.3.0). Deliberately NOT a
 * ProjectRecord: the site showcases projects, it is not itself evidence of
 * a skill, so it must never feed projectExtras.
 */
export interface PortfolioRecord {
  v: 1;
  repoName: string;
  repoUrl: string | null;
  pagesUrl: string | null;
  generatedAt: string;
  publishedAt: string | null;
  /** The saved-job ids this generation was tailored to. */
  jobIds: string[];
}

export function loadPortfolio(): PortfolioRecord | null {
  try {
    const raw = localStorage.getItem(PORTFOLIO_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as PortfolioRecord;
    return parsed.v === 1 ? parsed : null;
  } catch {
    return null;
  }
}

export function savePortfolio(record: PortfolioRecord): PortfolioRecord {
  localStorage.setItem(PORTFOLIO_KEY, JSON.stringify(record));
  return record;
}

export function clearPortfolio(): void {
  try {
    localStorage.removeItem(PORTFOLIO_KEY);
  } catch {
    // Best-effort, like the other stores.
  }
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
