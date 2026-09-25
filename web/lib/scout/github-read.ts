/**
 * Read-only GitHub access for Job Scout's "Your GitHub" (v6.8.0). Runs in
 * the browser with the student's own token (the v6.3.0 invariant: the
 * token never reaches a ChatISA route). Builds a RepoSummary for
 * /api/scout/repo-skills; never writes to any repository.
 */

import type { GithubConnection } from "./github-store";
import { classifyGithubResponse } from "./github";
import {
  authorshipFrom, clipSummary, pickCodePaths, pickDependencyPaths, stripNotebook,
  SUMMARY_LIMITS, type RepoListing, type RepoSummary,
} from "./github-summary";

export type ReadError =
  | { kind: "auth" }
  | { kind: "not-found" }
  | { kind: "rate-limit"; resetAt: string | null }
  | { kind: "network" }
  | { kind: "github"; status: number };

type Result<T> = ({ ok: true } & T) | { ok: false; error: ReadError };

const API = "https://api.github.com";

async function asError(res: Response): Promise<ReadError> {
  if (res.status === 404) return { kind: "not-found" };
  const e = await classifyGithubResponse(res);
  return e.kind === "auth" || e.kind === "rate-limit" || e.kind === "github" ? e : { kind: "github", status: res.status };
}

function client(conn: GithubConnection, fetchImpl: typeof fetch) {
  const get = (path: string, raw = false) =>
    fetchImpl(`${API}${path}`, {
      headers: {
        authorization: `Bearer ${conn.token}`,
        accept: raw ? "application/vnd.github.raw+json" : "application/vnd.github+json",
        "x-github-api-version": "2022-11-28",
      },
    });
  return { get };
}

const enc = (p: string) => p.split("/").map(encodeURIComponent).join("/");

export async function listOwnRepos(conn: GithubConnection, fetchImpl: typeof fetch = fetch): Promise<Result<{ repos: RepoListing[] }>> {
  try {
    const res = await client(conn, fetchImpl).get("/user/repos?visibility=public&affiliation=owner&sort=pushed&per_page=100");
    if (!res.ok) return { ok: false, error: await asError(res) };
    const rows = (await res.json()) as Record<string, unknown>[];
    const repos = rows
      .filter((r) => !r.fork && !r.archived && !r.private)
      .map((r) => ({
        fullName: String(r.full_name),
        description: (r.description as string | null) ?? null,
        language: (r.language as string | null) ?? null,
        pushedAt: String(r.pushed_at ?? ""),
        defaultBranch: String(r.default_branch ?? "main"),
        htmlUrl: String(r.html_url ?? `https://github.com/${String(r.full_name)}`),
      }));
    return { ok: true, repos };
  } catch {
    return { ok: false, error: { kind: "network" } };
  }
}

export async function readRepo(
  conn: GithubConnection,
  listing: RepoListing,
  fetchImpl: typeof fetch = fetch,
  /** Files over 6 MB the student chose to include ("Read it anyway"). */
  include: string[] = [],
): Promise<Result<{ summary: RepoSummary }>> {
  const { get } = client(conn, fetchImpl);
  const repo = `/repos/${enc(listing.fullName)}`;
  try {
    const meta = await get(repo);
    if (!meta.ok) return { ok: false, error: await asError(meta) };
    const info = (await meta.json()) as { description?: string; topics?: string[]; fork?: boolean; archived?: boolean };

    const langRes = await get(`${repo}/languages`);
    const languages = langRes.ok ? ((await langRes.json()) as Record<string, number>) : {};

    // 204 (no commits yet) and errors both mean "no authorship to show".
    const contribRes = await get(`${repo}/contributors?anon=1&per_page=100`);
    const contributors = contribRes.status === 200 ? ((await contribRes.json()) as { login?: string; type?: string; contributions: number }[]) : [];

    // 409 is GitHub's answer for an empty repository.
    const treeRes = await get(`${repo}/git/trees/${encodeURIComponent(listing.defaultBranch)}?recursive=1`);
    const treeBody = treeRes.ok ? ((await treeRes.json()) as { truncated?: boolean; tree?: { path: string; type: string; size?: number }[] }) : { tree: [] };
    if (!treeRes.ok && treeRes.status !== 409 && treeRes.status !== 404) return { ok: false, error: await asError(treeRes) };
    const tree = (treeBody.tree ?? []).filter((t) => t.type === "blob").map((t) => ({ path: t.path, size: t.size ?? 0 }));

    const readmeRes = await get(`${repo}/readme`, true);
    const readme = readmeRes.ok ? (await readmeRes.text()).slice(0, SUMMARY_LIMITS.readmeChars) : "";

    // Notebooks are read whole so their outputs can be removed before the
    // 45,000-character clip; other files only need their opening text.
    const readFile = async (path: string) => {
      const r = await get(`${repo}/contents/${enc(path)}`, true);
      if (!r.ok) return null;
      const raw = await r.text();
      return { path, text: path.endsWith(".ipynb") ? stripNotebook(raw) : raw.slice(0, SUMMARY_LIMITS.fileChars) };
    };
    const picked = pickCodePaths(tree, include);
    const dependencyFiles = (await Promise.all(pickDependencyPaths(tree).map(readFile))).filter((f): f is { path: string; text: string } => f !== null);
    const codeFiles = (await Promise.all(picked.paths.map(readFile))).filter((f): f is { path: string; text: string } => f !== null);

    return {
      ok: true,
      summary: clipSummary({
        fullName: listing.fullName,
        description: info.description ?? listing.description ?? "",
        topics: info.topics ?? [],
        fork: Boolean(info.fork),
        archived: Boolean(info.archived),
        languages,
        authorship: authorshipFrom(contributors, conn.login),
        tree,
        treeTruncated: Boolean(treeBody.truncated),
        readme,
        dependencyFiles,
        codeFiles,
        skippedLarge: picked.skippedLarge,
      }),
    };
  } catch {
    return { ok: false, error: { kind: "network" } };
  }
}

/** A repository's files (blobs) with sizes, for Import from GitHub (v6.9.0). */
export async function readRepoTree(
  conn: GithubConnection,
  listing: RepoListing,
  fetchImpl: typeof fetch = fetch,
): Promise<Result<{ tree: { path: string; size: number }[]; truncated: boolean }>> {
  try {
    const res = await client(conn, fetchImpl).get(
      `/repos/${enc(listing.fullName)}/git/trees/${encodeURIComponent(listing.defaultBranch)}?recursive=1`,
    );
    // 409 is GitHub's answer for an empty repository.
    if (res.status === 409) return { ok: true, tree: [], truncated: false };
    if (!res.ok) return { ok: false, error: await asError(res) };
    const body = (await res.json()) as { truncated?: boolean; tree?: { path: string; type: string; size?: number }[] };
    return {
      ok: true,
      tree: (body.tree ?? []).filter((t) => t.type === "blob").map((t) => ({ path: t.path, size: t.size ?? 0 })),
      truncated: Boolean(body.truncated),
    };
  } catch {
    return { ok: false, error: { kind: "network" } };
  }
}

/** One file's raw bytes (text or binary), read-only. */
export async function fetchRepoFile(
  conn: GithubConnection,
  fullName: string,
  path: string,
  fetchImpl: typeof fetch = fetch,
): Promise<Result<{ bytes: ArrayBuffer }>> {
  try {
    const res = await client(conn, fetchImpl).get(`/repos/${enc(fullName)}/contents/${enc(path)}`, true);
    if (!res.ok) return { ok: false, error: await asError(res) };
    return { ok: true, bytes: await res.arrayBuffer() };
  } catch {
    return { ok: false, error: { kind: "network" } };
  }
}
