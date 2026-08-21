/**
 * Published-work store for Portfolio Builder (2026-08-20). Every published
 * career or showcase site records a lightweight summary here so other parts
 * of the app (Job Scout matching, the dashboard) can reference what a
 * student has shipped without touching the full draft or GitHub. Backed by
 * localStorage "pb-published-v1"; subscribePublished/usePublishedWork keep
 * React views in sync across tabs (storage event) and within a tab (an
 * in-memory listener set, since the storage event does not fire in the
 * writing tab).
 */

import { useSyncExternalStore } from "react";

const KEY = "pb-published-v1";

export interface PublishedWork {
  id: string; kind: "career" | "showcase"; title: string; summary: string;
  skillIds: string[]; repoUrl: string; pagesUrl: string | null; publishedAt: string;
}

const listeners = new Set<() => void>();
let cache: PublishedWork[] | undefined;

function notify() { for (const l of listeners) l(); }

export function loadPublished(): PublishedWork[] {
  try {
    const raw = localStorage.getItem(KEY);
    const parsed = raw ? (JSON.parse(raw) as unknown) : [];
    return Array.isArray(parsed) ? parsed.filter((w): w is PublishedWork => !!w && typeof (w as PublishedWork).repoUrl === "string") : [];
  } catch {
    return [];
  }
}

function save(list: PublishedWork[]): PublishedWork[] {
  try { localStorage.setItem(KEY, JSON.stringify(list)); } catch { /* quota */ }
  cache = list;
  notify();
  return list;
}

export function upsertPublished(work: PublishedWork): PublishedWork[] {
  return save([work, ...loadPublished().filter((w) => w.id !== work.id)]);
}

export function removePublished(id: string): PublishedWork[] {
  return save(loadPublished().filter((w) => w.id !== id));
}

export function subscribePublished(listener: () => void): () => void {
  listeners.add(listener);
  const onStorage = (e: StorageEvent) => {
    if (e.key === null || e.key === KEY) { cache = undefined; notify(); }
  };
  if (typeof window !== "undefined") window.addEventListener("storage", onStorage);
  return () => {
    listeners.delete(listener);
    if (typeof window !== "undefined") window.removeEventListener("storage", onStorage);
  };
}

const EMPTY: PublishedWork[] = [];
function snapshot(): PublishedWork[] {
  if (cache === undefined) cache = loadPublished();
  return cache;
}

export function usePublishedWork(): PublishedWork[] {
  return useSyncExternalStore(subscribePublished, snapshot, () => EMPTY);
}
