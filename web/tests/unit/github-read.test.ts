import { describe, expect, it } from "vitest";
import { fetchRepoFile, listOwnRepos, readRepo, readRepoTree } from "@/lib/scout/github-read";

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
    "/repos/ada/churn/git/trees/HEAD": () => json(200, { truncated: false, tree: [
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
      "/repos/ada/churn/git/trees/HEAD": () => json(200, { truncated: false, tree: [{ path: "big.ipynb", type: "blob", size: 14_000_000 }] }),
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

  it("leaves a Git LFS pointer out of what the model reads", async () => {
    const lfs = `version https://git-lfs.github.com/spec/v1
oid sha256:${"b".repeat(64)}
size 9000000
`;
    const { f } = fake({
      "/repos/ada/churn/languages": () => json(200, { Python: 1 }),
      "/repos/ada/churn/contributors": () => json(200, [{ login: "ada", type: "User", contributions: 20 }]),
      "/repos/ada/churn/git/trees/HEAD": () => json(200, { tree: [
        { path: "src/model.py", type: "blob", size: 40 }, { path: "src/weights.py", type: "blob", size: 130 },
      ] }),
      "/repos/ada/churn/readme": () => json(404, {}),
      "/repos/ada/churn/contents/src/model.py": () => new Response("import pandas as pd"),
      "/repos/ada/churn/contents/src/weights.py": () => new Response(lfs),
      "/repos/ada/churn": () => json(200, { description: "", topics: [], fork: false, archived: false }),
    });
    const out = await readRepo(conn, listing, f);
    expect(out.ok && out.summary.codeFiles.map((c) => c.path)).toEqual(["src/model.py"]);
  });

  it("reads an empty repository as a README-only summary, not an error", async () => {
    const { f } = fake({
      "/repos/ada/churn/languages": () => json(200, {}),
      "/repos/ada/churn/contributors": () => new Response(null, { status: 204 }),
      "/repos/ada/churn/git/trees/HEAD": () => json(409, { message: "Git Repository is empty." }),
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

describe("readRepoTree", () => {
  it("reads the default branch as HEAD, so a branch name with a slash works", async () => {
    const { f, calls } = fake({ "/repos/ada/churn/git/trees/HEAD": () => json(200, { tree: [] }) });
    await readRepoTree(conn, { ...listing, defaultBranch: "release/2026" }, f);
    expect(calls[0]).toBe("/repos/ada/churn/git/trees/HEAD?recursive=1");
  });
  it("lists blobs with sizes on the default branch", async () => {
    const { f, calls } = fake({
      "/repos/ada/churn/git/trees/HEAD": () => json(200, { truncated: true, tree: [{ path: "a.py", type: "blob", size: 5 }, { path: "src", type: "tree" }] }),
    });
    expect(await readRepoTree(conn, listing, f)).toEqual({ ok: true, tree: [{ path: "a.py", size: 5 }], truncated: true });
    expect(calls[0]).toContain("recursive=1");
  });
  it("treats an empty repository as no files, and reports a revoked token", async () => {
    expect(await readRepoTree(conn, listing, fake({ "/repos/ada/churn/git/trees/HEAD": () => json(409, {}) }).f)).toEqual({ ok: true, tree: [], truncated: false });
    expect(await readRepoTree(conn, listing, fake({ "/repos/ada/churn/git/trees/HEAD": () => json(401, {}) }).f)).toEqual({ ok: false, error: { kind: "auth" } });
  });
});

describe("fetchRepoFile", () => {
  it("returns the raw bytes, binary included", async () => {
    const png = new Uint8Array([0x89, 0x50, 0x4e, 0x47]);
    const { f, calls } = fake({ "/repos/ada/churn/contents/figures/roc%20curve.png": () => new Response(png) });
    const out = await fetchRepoFile(conn, listing, "figures/roc curve.png", f);
    expect(out.ok && new Uint8Array(out.bytes)).toEqual(png);
    expect(calls[0]).toBe("/repos/ada/churn/contents/figures/roc%20curve.png");
  });
  it("reports a file that has gone as not-found", async () => {
    expect(await fetchRepoFile(conn, listing, "gone.py", fake({}).f)).toEqual({ ok: false, error: { kind: "not-found" } });
  });

  describe("Git LFS", () => {
    const pointer = (size: number) =>
      `version https://git-lfs.github.com/spec/v1
oid sha256:${"a".repeat(64)}
size ${size}
`;
    const media = "https://media.githubusercontent.com/media/ada/churn/main/data/sales%202026.csv";

    it("follows a pointer to GitHub's LFS host, without sending the token there", async () => {
      const csv = new TextEncoder().encode("region,sales\neast,10\n");
      const seen: { url: string; auth: string | null }[] = [];
      const f = (async (input: string | URL, init?: RequestInit) => {
        const url = String(input);
        seen.push({ url, auth: new Headers(init?.headers).get("authorization") });
        if (url.startsWith("https://api.github.com/")) return new Response(pointer(csv.byteLength));
        if (url === media) return new Response(csv);
        return json(404, {});
      }) as typeof fetch;
      const out = await fetchRepoFile(conn, listing, "data/sales 2026.csv", f);
      expect(out.ok && new TextDecoder().decode(out.bytes)).toBe("region,sales\neast,10\n");
      expect(seen[1]).toEqual({ url: media, auth: null });
    });

    it("refuses a pointer to a file over 25 MB before downloading it", async () => {
      const seen: string[] = [];
      const f = (async (input: string | URL) => {
        seen.push(String(input));
        return new Response(pointer(40 * 1024 * 1024));
      }) as typeof fetch;
      expect(await fetchRepoFile(conn, listing, "data/big.csv", f)).toEqual({ ok: false, error: { kind: "lfs-too-large", bytes: 40 * 1024 * 1024 } });
      expect(seen).toHaveLength(1);
    });

    it("reports an LFS file GitHub will not serve (or serves at the wrong size)", async () => {
      const unavailable = (async (input: string | URL) =>
        String(input).startsWith("https://api.github.com/") ? new Response(pointer(20)) : new Response("quota", { status: 403 })) as typeof fetch;
      expect(await fetchRepoFile(conn, listing, "data/a.csv", unavailable)).toEqual({ ok: false, error: { kind: "lfs" } });
      const short = (async (input: string | URL) =>
        String(input).startsWith("https://api.github.com/") ? new Response(pointer(20)) : new Response("abc")) as typeof fetch;
      expect(await fetchRepoFile(conn, listing, "data/a.csv", short)).toEqual({ ok: false, error: { kind: "lfs" } });
    });

    it("leaves a file that merely mentions LFS alone", async () => {
      const text = `# Notes

${pointer(10)}`;
      const { f } = fake({ "/repos/ada/churn/contents/NOTES.md": () => new Response(text) });
      const out = await fetchRepoFile(conn, listing, "NOTES.md", f);
      expect(out.ok && new TextDecoder().decode(out.bytes)).toBe(text);
    });
  });
});
