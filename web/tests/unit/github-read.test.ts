import { describe, expect, it } from "vitest";
import { listOwnRepos, readRepo } from "@/lib/scout/github-read";

const conn = { v: 1 as const, token: "t", login: "ada", connectedAt: "" };
const json = (status: number, body: unknown, headers: Record<string, string> = {}) =>
  new Response(body === null ? null : JSON.stringify(body), { status, headers: { "content-type": "application/json", ...headers } });
function fake(routes: Record<string, () => Response>) {
  const calls: string[] = [];
  const f = (async (input: string | URL) => {
    const url = new URL(String(input));
    const key = url.pathname + (url.search ? url.search : "");
    calls.push(key);
    const hit = Object.entries(routes).find(([k]) => key.startsWith(k));
    return hit ? hit[1]() : json(404, {});
  }) as typeof fetch;
  return { f, calls };
}
const listing = { fullName: "ada/churn", description: "Churn model", language: "Python", pushedAt: "2026-09-01T00:00:00Z", defaultBranch: "main", htmlUrl: "https://github.com/ada/churn" };

describe("listOwnRepos", () => {
  it("lists the student's public repositories, forks and archived ones left out", async () => {
    const { f } = fake({
      "/user/repos": () => json(200, [
        { full_name: "ada/churn", description: "Churn model", language: "Python", pushed_at: "2026-09-01T00:00:00Z", default_branch: "main", html_url: "https://github.com/ada/churn", fork: false, archived: false, private: false },
        { full_name: "ada/forked", fork: true, archived: false, private: false },
        { full_name: "ada/old", fork: false, archived: true, private: false },
      ]),
    });
    const out = await listOwnRepos(conn, f);
    expect(out).toEqual({ ok: true, repos: [listing] });
  });

  it("reports an expired token as auth", async () => {
    const { f } = fake({ "/user/repos": () => json(401, {}) });
    expect(await listOwnRepos(conn, f)).toEqual({ ok: false, error: { kind: "auth" } });
  });
});

describe("readRepo", () => {
  const full = () => fake({
    "/repos/ada/churn/languages": () => json(200, { Python: 12_000 }),
    "/repos/ada/churn/contributors": () => json(200, [{ login: "ada", type: "User", contributions: 20 }]),
    "/repos/ada/churn/git/trees/main": () => json(200, { truncated: false, tree: [
      { path: "README.md", type: "blob", size: 20 }, { path: "requirements.txt", type: "blob", size: 20 },
      { path: "src/model.py", type: "blob", size: 40 }, { path: "src", type: "tree" },
    ] }),
    "/repos/ada/churn/readme": () => new Response("# Churn\nPredicts churn."),
    "/repos/ada/churn/contents/requirements.txt": () => new Response("pandas\nscikit-learn"),
    "/repos/ada/churn/contents/src/model.py": () => new Response("import pandas as pd\nfrom sklearn.ensemble import GradientBoostingClassifier"),
    "/repos/ada/churn": () => json(200, { description: "Churn model", topics: ["ml"], fork: false, archived: false }),
  });

  it("builds a summary from the repository", async () => {
    const out = await readRepo(conn, listing, full().f);
    expect(out.ok).toBe(true);
    if (!out.ok) return;
    expect(out.summary.authorship).toEqual({ studentCommits: 20, totalCommits: 20 });
    expect(out.summary.readme).toContain("Predicts churn");
    expect(out.summary.dependencyFiles).toEqual([{ path: "requirements.txt", text: "pandas\nscikit-learn" }]);
    expect(out.summary.codeFiles.map((c) => c.path)).toEqual(["src/model.py"]);
    expect(out.summary.tree.map((t) => t.path)).not.toContain("src");
    expect(out.summary.skippedLarge).toEqual([]);
  });

  it("skips a file over 6 MB and lists it, then reads it when the student includes it", async () => {
    const routes = () => fake({
      "/repos/ada/churn/languages": () => json(200, { Python: 1 }),
      "/repos/ada/churn/contributors": () => json(200, [{ login: "ada", type: "User", contributions: 20 }]),
      "/repos/ada/churn/git/trees/main": () => json(200, { truncated: false, tree: [{ path: "big.ipynb", type: "blob", size: 14_000_000 }] }),
      "/repos/ada/churn/readme": () => new Response("# Churn"),
      "/repos/ada/churn/contents/big.ipynb": () => new Response(JSON.stringify({ cells: [{ cell_type: "code", source: ["import pandas as pd"], outputs: [{ data: "x".repeat(1000) }] }] })),
      "/repos/ada/churn": () => json(200, { description: "", topics: [], fork: false, archived: false }),
    });
    const skipped = await readRepo(conn, listing, routes().f);
    expect(skipped.ok && skipped.summary.skippedLarge).toEqual([{ path: "big.ipynb", size: 14_000_000 }]);
    expect(skipped.ok && skipped.summary.codeFiles).toEqual([]);
    const included = await readRepo(conn, listing, routes().f, ["big.ipynb"]);
    expect(included.ok && included.summary.codeFiles).toEqual([{ path: "big.ipynb", text: "import pandas as pd" }]);
    expect(included.ok && included.summary.skippedLarge).toEqual([]);
  });

  it("reads an empty repository as a README-only summary, not an error", async () => {
    const { f } = fake({
      "/repos/ada/churn/languages": () => json(200, {}),
      "/repos/ada/churn/contributors": () => new Response(null, { status: 204 }),
      "/repos/ada/churn/git/trees/main": () => json(409, { message: "Git Repository is empty." }),
      "/repos/ada/churn/readme": () => json(404, {}),
      "/repos/ada/churn": () => json(200, { description: "", topics: [], fork: false, archived: false }),
    });
    const out = await readRepo(conn, listing, f);
    expect(out).toMatchObject({ ok: true, summary: { authorship: { studentCommits: 0, totalCommits: 0 }, codeFiles: [], readme: "" } });
  });

  it("reports a repository that is gone as not-found, and a revoked token as auth", async () => {
    expect(await readRepo(conn, listing, fake({ "/repos/ada/churn": () => json(404, {}) }).f)).toEqual({ ok: false, error: { kind: "not-found" } });
    expect(await readRepo(conn, listing, fake({ "/repos/ada/churn": () => json(401, {}) }).f)).toEqual({ ok: false, error: { kind: "auth" } });
  });

  it("reports GitHub's rate limit with its reset time", async () => {
    const { f } = fake({ "/repos/ada/churn": () => json(403, {}, { "x-ratelimit-remaining": "0", "x-ratelimit-reset": "1790000000" }) });
    expect(await readRepo(conn, listing, f)).toEqual({ ok: false, error: { kind: "rate-limit", resetAt: new Date(1790000000 * 1000).toISOString() } });
  });
});
