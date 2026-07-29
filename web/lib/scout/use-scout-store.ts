"use client";

import { useSyncExternalStore } from "react";
import {
  addProject,
  hidePosting,
  loadProfile,
  loadProjects,
  loadSaved,
  removeProject,
  saveProfile,
  setProjectRepoUrl,
  toggleSaved,
  type ProjectRecord,
  type ProjectsState,
  type SavedSnapshot,
  type SavedState,
  type ScoutProfile,
} from "./profile-store";

/**
 * React bindings for the local profile/saved/project stores.
 * useSyncExternalStore (the PauseDial pattern) rather than read-in-an-
 * effect: the server has no localStorage, so it renders the "unknown"
 * snapshot and the client swaps in the stored value at hydration without a
 * cascading render.
 */

type ProfileSnapshot = ScoutProfile | false | "unknown";

const listeners = new Set<() => void>();
let profileCache: ScoutProfile | false | undefined;
let savedCache: SavedState | undefined;
let projectsCache: ProjectsState | undefined;

function notify() {
  for (const l of listeners) l();
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

function profileSnapshot(): ProfileSnapshot {
  if (profileCache === undefined) profileCache = loadProfile() ?? false;
  return profileCache;
}

const EMPTY_SAVED: SavedState = { v: 2, saved: [], hiddenIds: [] };
const EMPTY_PROJECTS: ProjectsState = { v: 1, projects: [] };

function savedSnapshot(): SavedState {
  if (savedCache === undefined) savedCache = loadSaved();
  return savedCache;
}

function projectsSnapshot(): ProjectsState {
  if (projectsCache === undefined) projectsCache = loadProjects();
  return projectsCache;
}

export function useScoutProfile(): [
  ProfileSnapshot,
  (profile: ScoutProfile) => void,
] {
  const profile = useSyncExternalStore(
    subscribe,
    profileSnapshot,
    () => "unknown" as const,
  );
  return [
    profile,
    (next: ScoutProfile) => {
      saveProfile(next);
      profileCache = next;
      notify();
    },
  ];
}

export function useScoutSaved(): {
  saved: SavedState;
  toggle: (snapshot: Omit<SavedSnapshot, "savedAt">) => void;
  hide: (id: string) => void;
} {
  const saved = useSyncExternalStore(
    subscribe,
    savedSnapshot,
    () => EMPTY_SAVED,
  );
  return {
    saved,
    toggle: (snapshot) => {
      savedCache = toggleSaved(snapshot);
      notify();
    },
    hide: (id: string) => {
      savedCache = hidePosting(id);
      notify();
    },
  };
}

export function useScoutProjects(): {
  projects: ProjectsState;
  add: (record: ProjectRecord) => void;
  setRepoUrl: (id: string, repoUrl: string | null) => void;
  remove: (id: string) => void;
} {
  const projects = useSyncExternalStore(
    subscribe,
    projectsSnapshot,
    () => EMPTY_PROJECTS,
  );
  return {
    projects,
    add: (record) => {
      projectsCache = addProject(record);
      notify();
    },
    setRepoUrl: (id, repoUrl) => {
      projectsCache = setProjectRepoUrl(id, repoUrl);
      notify();
    },
    remove: (id) => {
      projectsCache = removeProject(id);
      notify();
    },
  };
}
