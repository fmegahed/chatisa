/**
 * Portfolio Builder device stores (2026-08-20). Site records (metadata: what
 * exists, where it is published) live in localStorage under "pb-sites-v1".
 * Drafts (content plus rendered html plus any files kept for republish) can
 * be large, so they live in IndexedDB via lib/scout/device-files.ts, one
 * record per site id. Nothing here leaves the device; matches the local-first
 * decision behind Job Scout's resume/portfolio storage.
 */

import type { FileRole } from "./files";
import { migrateCareerV1, type SiteContent } from "./content";
import { getItem, putItem, removeItem } from "@/lib/scout/device-files";

const SITES_KEY = "pb-sites-v1";
const LEGACY_PORTFOLIO_KEY = "js-portfolio-v1";
/** Job Scout v6.3.0 kept the portfolio content and html here (IndexedDB). */
const LEGACY_DRAFT_KEY = "portfolio";

export interface SiteRecord {
  v: 1; id: string; kind: "career" | "showcase"; title: string; repoName: string;
  repoUrl: string | null; pagesUrl: string | null; generatedAt: string; publishedAt: string | null;
}
export interface StoredFile {
  projectSlug: string | null; name: string; role: FileRole; publish: boolean;
  bytes: number; text: string | null; base64: string | null;
}
export interface CareerStudent { name: string; links: { label: string; url: string }[]; courses: string[] }
export interface ShowcaseMeta { course: string; semester: string; team: string[] }
export interface SiteDraft {
  v: 1; content: SiteContent; html: string; student: CareerStudent | null;
  showcaseMeta: ShowcaseMeta | null; files: StoredFile[];
  photoBase64: string | null; resumeBase64: string | null; resumeLink: boolean;
  /** Optional so drafts written before Task 12 still parse. */
  readme?: string | null;
  skillIds?: string[];
}

export function newSiteId(): string {
  return crypto.randomUUID();
}

export function loadSites(): SiteRecord[] {
  try {
    const raw = localStorage.getItem(SITES_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as unknown;
    return Array.isArray(parsed)
      ? parsed.filter((s): s is SiteRecord => !!s && typeof s === "object" && (s as SiteRecord).v === 1 && typeof (s as SiteRecord).id === "string")
      : [];
  } catch {
    return [];
  }
}

function saveSites(sites: SiteRecord[]): SiteRecord[] {
  try { localStorage.setItem(SITES_KEY, JSON.stringify(sites)); } catch { /* quota; in-memory result still returned */ }
  return sites;
}

/**
 * A student has one career portfolio: it publishes to one repository, so a
 * second career record would be a second site record fighting over the same
 * repo. Writing a career site therefore replaces any career site already
 * stored; showcase sites accumulate normally.
 */
export function upsertSite(record: SiteRecord): SiteRecord[] {
  const sites = loadSites().filter(
    (s) => s.id !== record.id && !(record.kind === "career" && s.kind === "career"),
  );
  return saveSites([record, ...sites]);
}

export function removeSite(id: string): SiteRecord[] {
  return saveSites(loadSites().filter((s) => s.id !== id));
}

export function careerSite(): SiteRecord | null {
  return loadSites().find((s) => s.kind === "career") ?? null;
}

/**
 * The v6.3.0 draft that went with the record: Job Scout kept the model's
 * content and the rendered html in IndexedDB under "portfolio". Lifting the
 * record without it would leave the student a site they can open but not
 * edit, so the content is migrated to v2 and rewritten as a SiteDraft under
 * the new id. A draft that will not migrate is dropped, not kept: the
 * student still has the record and can regenerate.
 */
async function migrateJobScoutDraft(siteId: string): Promise<void> {
  const record = await getItem<{ content: unknown; html?: string }>(LEGACY_DRAFT_KEY);
  if (!record) return;
  const content = migrateCareerV1(record.content);
  // A draft already written under this id is the student's current work and
  // outranks the legacy copy.
  if (content && !(await getDraft(siteId))) {
    await putDraft(siteId, {
      v: 1, content: { kind: "career", content }, html: record.html ?? "",
      student: null, showcaseMeta: null, files: [],
      photoBase64: null, resumeBase64: null, resumeLink: false,
    });
  }
  await removeItem(LEGACY_DRAFT_KEY);
}

/**
 * One-shot lift of Job Scout's v6.3.0 PortfolioRecord into a SiteRecord,
 * with its draft. Returns the career site id (the one it created, or the one
 * already there) so the caller can open it.
 */
export async function migrateJobScoutPortfolio(): Promise<string | null> {
  let id: string | null = null;
  try {
    const raw = localStorage.getItem(LEGACY_PORTFOLIO_KEY);
    if (!raw) {
      id = careerSite()?.id ?? null;
      if (id) await migrateJobScoutDraft(id);
      return id;
    }
    const old = JSON.parse(raw) as { repoName?: string; repoUrl?: string | null; pagesUrl?: string | null; generatedAt?: string; publishedAt?: string | null };
    const existing = careerSite();
    if (existing) {
      id = existing.id;
    } else {
      id = newSiteId();
      upsertSite({
        v: 1, id, kind: "career", title: "Portfolio",
        repoName: old.repoName || "portfolio", repoUrl: old.repoUrl ?? null, pagesUrl: old.pagesUrl ?? null,
        generatedAt: old.generatedAt || new Date().toISOString(), publishedAt: old.publishedAt ?? null,
      });
    }
    localStorage.removeItem(LEGACY_PORTFOLIO_KEY);
  } catch {
    /* corrupt legacy record: ignore */
  }
  if (id) await migrateJobScoutDraft(id);
  return id;
}

export function putDraft(id: string, draft: SiteDraft): Promise<boolean> { return putItem(`pb-draft:${id}`, draft); }
export function getDraft(id: string): Promise<SiteDraft | null> { return getItem<SiteDraft>(`pb-draft:${id}`); }
export function deleteDraft(id: string): Promise<void> { return removeItem(`pb-draft:${id}`); }
