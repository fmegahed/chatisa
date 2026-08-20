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
import {
  clearGithubConnection,
  loadGithubConnection,
  saveGithubConnection,
  type GithubConnection,
} from "./github-store";

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

let githubCache: GithubConnection | null | undefined;

function githubSnapshot(): GithubConnection | null {
  if (githubCache === undefined) githubCache = loadGithubConnection();
  return githubCache;
}

/**
 * The OAuth popup writes localStorage from ANOTHER document, which this
 * document's caches cannot see. Both signals it can produce (a same-origin
 * postMessage, and the storage event when the write comes from another
 * window) invalidate the cache and re-render.
 */
function subscribeGithub(listener: () => void): () => void {
  listeners.add(listener);
  const invalidate = (e: MessageEvent | StorageEvent) => {
    if (e instanceof StorageEvent && e.key !== null && e.key !== "js-github-v1") {
      return;
    }
    if (
      e instanceof MessageEvent &&
      (e.origin !== window.location.origin ||
        (e.data as { type?: string } | null)?.type !== "chatisa:github-connected")
    ) {
      return;
    }
    githubCache = undefined;
    notify();
  };
  window.addEventListener("storage", invalidate);
  window.addEventListener("message", invalidate);
  return () => {
    listeners.delete(listener);
    window.removeEventListener("storage", invalidate);
    window.removeEventListener("message", invalidate);
  };
}

export function useGithubConnection(): {
  connection: GithubConnection | null;
  set: (value: { token: string; login: string }) => void;
  clear: () => void;
} {
  const connection = useSyncExternalStore(
    subscribeGithub,
    githubSnapshot,
    () => null,
  );
  return {
    connection,
    set: (value) => {
      githubCache = saveGithubConnection(value);
      notify();
    },
    clear: () => {
      clearGithubConnection();
      githubCache = null;
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
