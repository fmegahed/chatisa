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
  contents: string;
}

export type PushError =
  | { kind: "auth" }
  | { kind: "name-taken"; suggestion: string | null }
  | { kind: "rate-limit"; resetAt: string | null }
  | { kind: "too-large" }
  | { kind: "network" }
  | { kind: "github"; status: number };

export type PushResult =
  | { ok: true; repoUrl: string; defaultBranch: string }
  | { ok: false; error: PushError };

const MAX_FILES = 60;
const MAX_FILE_BYTES = 400_000;
const MAX_TOTAL_BYTES = 2_000_000;

const API = "https://api.github.com";

function headers(conn: GithubConnection): Record<string, string> {
  return {
    authorization: `Bearer ${conn.token}`,
    accept: "application/vnd.github+json",
    "x-github-api-version": "2022-11-28",
    "content-type": "application/json",
  };
}

function classify(res: Response): PushError {
  if (res.status === 401) return { kind: "auth" };
  if (res.status === 403 && res.headers.get("x-ratelimit-remaining") === "0") {
    const reset = res.headers.get("x-ratelimit-reset");
    return {
      kind: "rate-limit",
      resetAt: reset ? new Date(Number(reset) * 1000).toISOString() : null,
    };
  }
  return { kind: "github", status: res.status };
}

async function gh(
  conn: GithubConnection,
  method: string,
  path: string,
  body?: unknown,
): Promise<Response> {
  return fetch(`${API}${path}`, {
    method,
    headers: headers(conn),
    body: body === undefined ? undefined : JSON.stringify(body),
  });
}

/** Byte length in UTF-8, which is what GitHub counts. */
function utf8Bytes(s: string): number {
  return new TextEncoder().encode(s).length;
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
  if (!created.ok) return { ok: false, error: classify(created) };
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
  opts: { message: string; expectedRepoUrl: string | null },
): Promise<PushResult> {
  if (files.length === 0 || files.length > MAX_FILES) {
    return { ok: false, error: { kind: "too-large" } };
  }
  let total = 0;
  for (const f of files) {
    const bytes = utf8Bytes(f.contents);
    if (bytes > MAX_FILE_BYTES) return { ok: false, error: { kind: "too-large" } };
    total += bytes;
  }
  if (total > MAX_TOTAL_BYTES) return { ok: false, error: { kind: "too-large" } };

  try {
    const repoPath = `/repos/${conn.login}/${repoName}`;
    const existing = await gh(conn, "GET", repoPath);

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
      return { ok: false, error: classify(existing) };
    }

    const refRes = await gh(conn, "GET", `${repoPath}/git/ref/heads/${defaultBranch}`);
    if (!refRes.ok) return { ok: false, error: classify(refRes) };
    const parentSha = ((await refRes.json()) as { object: { sha: string } }).object.sha;

    const parentCommit = await gh(conn, "GET", `${repoPath}/git/commits/${parentSha}`);
    if (!parentCommit.ok) return { ok: false, error: classify(parentCommit) };
    const baseTree = ((await parentCommit.json()) as { tree: { sha: string } }).tree.sha;

    const treeRes = await gh(conn, "POST", `${repoPath}/git/trees`, {
      base_tree: baseTree,
      tree: files.map((f) => ({
        path: f.path,
        mode: "100644",
        type: "blob",
        content: f.contents,
      })),
    });
    if (!treeRes.ok) return { ok: false, error: classify(treeRes) };
    const treeSha = ((await treeRes.json()) as { sha: string }).sha;

    const commitRes = await gh(conn, "POST", `${repoPath}/git/commits`, {
      message: opts.message,
      tree: treeSha,
      parents: [parentSha],
    });
    if (!commitRes.ok) return { ok: false, error: classify(commitRes) };
    const commitSha = ((await commitRes.json()) as { sha: string }).sha;

    const patchRes = await gh(conn, "PATCH", `${repoPath}/git/refs/heads/${defaultBranch}`, {
      sha: commitSha,
    });
    if (!patchRes.ok) return { ok: false, error: classify(patchRes) };

    return {
      ok: true,
      repoUrl: `https://github.com/${conn.login}/${repoName}`,
      defaultBranch,
    };
  } catch {
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

export function polishFileSet(stored: {
  plan: { readme: string; gitignore: string; extraFiles: PushFile[] };
  textFiles: PushFile[];
  binaryPaths: string[];
}): PushFile[] {
  const files: PushFile[] = [
    { path: "README.md", contents: stored.plan.readme },
    { path: ".gitignore", contents: stored.plan.gitignore },
    ...stored.plan.extraFiles,
    ...stored.textFiles,
  ];
  if (stored.binaryPaths.length > 0) {
    files.push({
      path: "ADD_THESE_FILES.md",
      contents: [
        "# Files to add yourself",
        "",
        "These files were part of your project but were not kept in the browser, so they were not pushed. Add each one on github.com with the \"Add file\" button, then delete this note:",
        "",
        ...stored.binaryPaths.map((p) => `- ${p}`),
        "",
      ].join("\n"),
    });
  }
  return files;
}

export function portfolioFileSet(html: string): PushFile[] {
  return [
    { path: "index.html", contents: html },
    // Pages serves the branch through Jekyll unless told not to; .nojekyll
    // keeps the deploy a plain file copy.
    { path: ".nojekyll", contents: "" },
    {
      path: "README.md",
      contents:
        "# Portfolio\n\nThis site was generated with ChatISA's Job Scout and is published with GitHub Pages. Edit index.html to make it yours.\n",
    },
  ];
}
