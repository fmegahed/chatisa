import type { GithubConnection } from "./github-store";

/**
 * Browser-side GitHub push engine (v6.3.0). Every call here runs in the
 * student's browser with their own token; nothing passes through a ChatISA
 * server. Derived from the CareerBridge prototype's sync service with its
 * defects fixed: one tree request with inline content instead of a blob
 * round-trip per file, ref polling instead of a fixed 2-second sleep after
 * repo creation, an ours-check before ever updating an existing repo, and
 * typed errors the UI can translate into student-facing copy.
 */

export interface PushFile {
  path: string;
  /** UTF-8 text, or base64 when encoding is "base64" (photos, PDFs). */
  contents: string;
  encoding?: "base64";
}

export type PushError =
  | { kind: "auth" }
  | { kind: "name-taken"; suggestion: string | null }
  | { kind: "rate-limit"; resetAt: string | null }
  | { kind: "too-large" }
  | { kind: "network" }
  | { kind: "cancelled" }
  | { kind: "github"; status: number };

export type PushProgress =
  | { stage: "prepare" }
  | { stage: "upload"; done: number; total: number; path: string }
  | { stage: "commit" };

export interface PushOptions {
  message: string;
  expectedRepoUrl: string | null;
  /** Called as each binary file lands and when the commit starts. */
  onProgress?: (p: PushProgress) => void;
  /** Aborting stops before the next request; the repository is left as it was. */
  signal?: AbortSignal;
  /** Pause before the single retry of a failed request (tests set 0). */
  retryDelayMs?: number;
}

export type PushResult =
  | { ok: true; repoUrl: string; defaultBranch: string }
  | { ok: false; error: PushError };

const MAX_FILES = 60;
// Raised 2026-08-23 from 400 KB / 2 MB: a 1.5 MB deck and a 15 MB notebook are
// ordinary student deliverables. Binaries go through the blobs API (100 MB each
// on GitHub's side); these caps keep one site inside what a browser tab can
// hold as base64 and what Pages serves comfortably.
const MAX_FILE_BYTES = 25 * 1024 * 1024;
const MAX_TOTAL_BYTES = 100 * 1024 * 1024;

export const PUSH_LIMITS = { files: MAX_FILES, fileBytes: MAX_FILE_BYTES, totalBytes: MAX_TOTAL_BYTES } as const;

const API = "https://api.github.com";

function headers(conn: GithubConnection): Record<string, string> {
  return {
    authorization: `Bearer ${conn.token}`,
    accept: "application/vnd.github+json",
    "x-github-api-version": "2022-11-28",
    "content-type": "application/json",
  };
}

/**
 * GitHub has two rate limits. The primary one answers 403 with
 * x-ratelimit-remaining: 0 and a reset epoch; the secondary (abuse) limit,
 * which a burst of blob uploads can trip, answers 403 or 429 with a
 * retry-after in seconds and "secondary rate limit" in the body. Both are
 * "wait, then try again", so both become rate-limit for the UI.
 */
async function classify(res: Response): Promise<PushError> {
  if (res.status === 401) return { kind: "auth" };
  if (res.status === 403 || res.status === 429) {
    const retryAfter = res.headers.get("retry-after");
    if (res.headers.get("x-ratelimit-remaining") === "0") {
      const reset = res.headers.get("x-ratelimit-reset");
      return { kind: "rate-limit", resetAt: reset ? new Date(Number(reset) * 1000).toISOString() : null };
    }
    let secondary = res.status === 429 || retryAfter !== null;
    if (!secondary) {
      try {
        secondary = /secondary rate limit|abuse/i.test(await res.clone().text());
      } catch {
        secondary = false;
      }
    }
    if (secondary) {
      const parsed = retryAfter ? Number(retryAfter) : NaN;
      const seconds = Number.isFinite(parsed) && parsed > 0 ? parsed : 60;
      return { kind: "rate-limit", resetAt: new Date(Date.now() + seconds * 1000).toISOString() };
    }
  }
  return { kind: "github", status: res.status };
}

class CancelledError extends Error {}

const sleep = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));

function isAbort(err: unknown): boolean {
  return err instanceof CancelledError || (err instanceof DOMException && err.name === "AbortError");
}

/**
 * One request, retried once on a network failure or a 5xx: a 25 MB blob
 * upload that drops mid-way should not cost the student the whole publish.
 * 4xx answers are GitHub's verdict and are never retried. An aborted signal
 * stops before the request (or the retry) goes out.
 */
async function gh(
  conn: GithubConnection,
  method: string,
  path: string,
  body?: unknown,
  ctrl: { signal?: AbortSignal; retryDelayMs?: number } = {},
): Promise<Response> {
  const once = () => {
    if (ctrl.signal?.aborted) throw new CancelledError();
    return fetch(`${API}${path}`, {
      method,
      headers: headers(conn),
      body: body === undefined ? undefined : JSON.stringify(body),
      signal: ctrl.signal,
    });
  };
  try {
    const res = await once();
    if (res.status < 500) return res;
  } catch (err) {
    if (isAbort(err)) throw err;
  }
  await sleep(ctrl.retryDelayMs ?? 1500);
  return once();
}

/** Byte length in UTF-8, which is what GitHub counts. */
function utf8Bytes(s: string): number {
  return new TextEncoder().encode(s).length;
}

/** Bytes GitHub will store: decoded length for base64, UTF-8 length otherwise. */
export function pushFileBytes(file: PushFile): number {
  if (file.encoding === "base64") {
    const trimmed = file.contents.replace(/=+$/, "");
    return Math.floor((trimmed.length * 3) / 4);
  }
  return utf8Bytes(file.contents);
}

/**
 * Creates the repository (public, auto-initialised so a parent commit always
 * exists) and waits until its default branch ref answers, replacing the
 * prototype's fixed sleep. Returns the default branch name.
 */
async function createRepo(
  conn: GithubConnection,
  repoName: string,
): Promise<{ ok: true; defaultBranch: string } | { ok: false; error: PushError }> {
  const created = await gh(conn, "POST", "/user/repos", {
    name: repoName,
    private: false,
    auto_init: true,
    has_issues: true,
    has_projects: false,
    has_wiki: false,
  });
  if (created.status === 422) return { ok: false, error: { kind: "name-taken", suggestion: null } };
  if (!created.ok) return { ok: false, error: await classify(created) };
  const defaultBranch =
    ((await created.json()) as { default_branch?: string }).default_branch ?? "main";

  for (let attempt = 0; attempt < 10; attempt++) {
    const ref = await gh(
      conn,
      "GET",
      `/repos/${conn.login}/${repoName}/git/ref/heads/${defaultBranch}`,
    );
    if (ref.ok) return { ok: true, defaultBranch };
    await new Promise((r) => setTimeout(r, 300 * (attempt + 1)));
  }
  return { ok: false, error: { kind: "github", status: 500 } };
}

/** First free repoName-2 .. repoName-9 on the account, or null. */
async function suggestFreeName(
  conn: GithubConnection,
  repoName: string,
): Promise<string | null> {
  for (let n = 2; n <= 9; n++) {
    const candidate = `${repoName}-${n}`;
    const res = await gh(conn, "GET", `/repos/${conn.login}/${candidate}`);
    if (res.status === 404) return candidate;
  }
  return null;
}

export async function pushToRepo(
  conn: GithubConnection,
  repoName: string,
  files: PushFile[],
  opts: PushOptions,
): Promise<PushResult> {
  const ctrl = { signal: opts.signal, retryDelayMs: opts.retryDelayMs };
  const progress = opts.onProgress ?? (() => {});
  if (files.length === 0 || files.length > MAX_FILES) {
    return { ok: false, error: { kind: "too-large" } };
  }
  let total = 0;
  for (const f of files) {
    const bytes = pushFileBytes(f);
    if (bytes > MAX_FILE_BYTES) return { ok: false, error: { kind: "too-large" } };
    total += bytes;
  }
  if (total > MAX_TOTAL_BYTES) return { ok: false, error: { kind: "too-large" } };

  try {
    progress({ stage: "prepare" });
    const repoPath = `/repos/${conn.login}/${repoName}`;
    const existing = await gh(conn, "GET", repoPath, undefined, ctrl);

    let defaultBranch: string;
    if (existing.status === 404) {
      const created = await createRepo(conn, repoName);
      if (!created.ok) return created;
      defaultBranch = created.defaultBranch;
    } else if (existing.ok) {
      const repo = (await existing.json()) as {
        html_url: string;
        default_branch: string;
      };
      const ours =
        opts.expectedRepoUrl !== null &&
        repo.html_url.toLowerCase() === opts.expectedRepoUrl.toLowerCase();
      if (!ours) {
        // A same-named repo we did not create: never overwrite it silently.
        return {
          ok: false,
          error: { kind: "name-taken", suggestion: await suggestFreeName(conn, repoName) },
        };
      }
      defaultBranch = repo.default_branch;
    } else {
      return { ok: false, error: await classify(existing) };
    }

    const refRes = await gh(conn, "GET", `${repoPath}/git/ref/heads/${defaultBranch}`, undefined, ctrl);
    if (!refRes.ok) return { ok: false, error: await classify(refRes) };
    const parentSha = ((await refRes.json()) as { object: { sha: string } }).object.sha;

    const parentCommit = await gh(conn, "GET", `${repoPath}/git/commits/${parentSha}`, undefined, ctrl);
    if (!parentCommit.ok) return { ok: false, error: await classify(parentCommit) };
    const baseTree = ((await parentCommit.json()) as { tree: { sha: string } }).tree.sha;

    const tree: { path: string; mode: "100644"; type: "blob"; content?: string; sha?: string }[] = [];
    const uploads = files.filter((f) => f.encoding === "base64").length;
    let done = 0;
    for (const f of files) {
      if (f.encoding === "base64") {
        const blobRes = await gh(conn, "POST", `${repoPath}/git/blobs`, {
          content: f.contents,
          encoding: "base64",
        }, ctrl);
        if (!blobRes.ok) return { ok: false, error: await classify(blobRes) };
        const sha = ((await blobRes.json()) as { sha: string }).sha;
        tree.push({ path: f.path, mode: "100644", type: "blob", sha });
        done++;
        progress({ stage: "upload", done, total: uploads, path: f.path });
      } else {
        tree.push({ path: f.path, mode: "100644", type: "blob", content: f.contents });
      }
    }
    progress({ stage: "commit" });
    const treeRes = await gh(conn, "POST", `${repoPath}/git/trees`, { base_tree: baseTree, tree }, ctrl);
    if (!treeRes.ok) return { ok: false, error: await classify(treeRes) };
    const treeSha = ((await treeRes.json()) as { sha: string }).sha;

    const commitRes = await gh(conn, "POST", `${repoPath}/git/commits`, {
      message: opts.message,
      tree: treeSha,
      parents: [parentSha],
    }, ctrl);
    if (!commitRes.ok) return { ok: false, error: await classify(commitRes) };
    const commitSha = ((await commitRes.json()) as { sha: string }).sha;

    const patchRes = await gh(conn, "PATCH", `${repoPath}/git/refs/heads/${defaultBranch}`, {
      sha: commitSha,
    }, ctrl);
    if (!patchRes.ok) return { ok: false, error: await classify(patchRes) };

    return {
      ok: true,
      repoUrl: `https://github.com/${conn.login}/${repoName}`,
      defaultBranch,
    };
  } catch (err) {
    if (isAbort(err)) return { ok: false, error: { kind: "cancelled" } };
    return { ok: false, error: { kind: "network" } };
  }
}

/**
 * Best-effort GitHub Pages enablement. GitHub's docs tie the Pages API to
 * the broader `repo` scope, and this app deliberately holds only
 * `public_repo`, so a 403 here is expected on some accounts: the caller
 * shows a one-click manual path to the repo's Pages settings instead.
 */
export async function enablePages(
  conn: GithubConnection,
  repoName: string,
  defaultBranch: string,
): Promise<
  | { ok: true; pagesUrl: string }
  | { ok: false; needsManual: true; settingsUrl: string }
> {
  const pagesUrl = `https://${conn.login.toLowerCase()}.github.io/${repoName}/`;
  try {
    const res = await gh(conn, "POST", `/repos/${conn.login}/${repoName}/pages`, {
      build_type: "legacy",
      source: { branch: defaultBranch, path: "/" },
    });
    if (res.status === 201 || res.status === 409) return { ok: true, pagesUrl };
  } catch {
    // Fall through to the manual path.
  }
  return {
    ok: false,
    needsManual: true,
    settingsUrl: `https://github.com/${conn.login}/${repoName}/settings/pages`,
  };
}

/* File-set assemblers: pure, so the exact pushed contents are unit-testable. */

export function scaffoldFileSet(scaffold: {
  readme: string;
  files: PushFile[];
}): PushFile[] {
  return [{ path: "README.md", contents: scaffold.readme }, ...scaffold.files];
}
