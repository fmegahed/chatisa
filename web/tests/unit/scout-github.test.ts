import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  decodeOauthState,
  encodeOauthState,
  safeReturnPath,
} from "@/lib/scout/github-state";
import {
  clearGithubConnection,
  loadGithubConnection,
  saveGithubConnection,
} from "@/lib/scout/github-store";
import {
  enablePages,
  polishFileSet,
  portfolioFileSet,
  pushToRepo,
  scaffoldFileSet,
} from "@/lib/scout/github";
import {
  escapeHtml,
  renderPortfolioHtml,
  type PortfolioContent,
} from "@/lib/scout/portfolio-html";

/**
 * The GitHub push path runs entirely in the student's browser with their
 * own token, so nothing server-side can catch a defect here: these tests
 * are the only gate between a bug and a student's own GitHub account.
 */

function memoryStorage(): Storage {
  const map = new Map<string, string>();
  return {
    getItem: (k: string) => map.get(k) ?? null,
    setItem: (k: string, v: string) => void map.set(k, v),
    removeItem: (k: string) => void map.delete(k),
    clear: () => map.clear(),
    key: (i: number) => [...map.keys()][i] ?? null,
    get length() {
      return map.size;
    },
  } as Storage;
}

beforeEach(() => {
  vi.stubGlobal("localStorage", memoryStorage());
});

describe("oauth state cookie", () => {
  it("round-trips state and return path", () => {
    const encoded = encodeOauthState({
      state: "0123456789abcdef0123",
      returnPath: "/job-scout?tab=projects",
    });
    expect(decodeOauthState(encoded)).toEqual({
      state: "0123456789abcdef0123",
      returnPath: "/job-scout?tab=projects",
    });
  });

  it("decodes tampered or truncated cookies to null, never throws", () => {
    expect(decodeOauthState(undefined)).toBeNull();
    expect(decodeOauthState("not-base64!!")).toBeNull();
    expect(decodeOauthState(Buffer.from("{}").toString("base64url"))).toBeNull();
    // A short state cannot carry enough entropy to resist guessing.
    expect(
      decodeOauthState(
        Buffer.from(JSON.stringify({ state: "short", returnPath: "/x" })).toString(
          "base64url",
        ),
      ),
    ).toBeNull();
  });

  it("refuses off-origin return paths (open-redirect guard)", () => {
    expect(safeReturnPath("/job-scout?tab=projects")).toBe("/job-scout?tab=projects");
    expect(safeReturnPath("//evil.example/phish")).toBe("/job-scout");
    expect(safeReturnPath("https://evil.example")).toBe("/job-scout");
    expect(safeReturnPath("/\\evil")).toBe("/job-scout");
    expect(safeReturnPath(null)).toBe("/job-scout");
  });
});

describe("github connection store", () => {
  it("round-trips a connection", () => {
    saveGithubConnection({ token: "gh-token", login: "student" });
    const loaded = loadGithubConnection();
    expect(loaded?.token).toBe("gh-token");
    expect(loaded?.login).toBe("student");
    clearGithubConnection();
    expect(loadGithubConnection()).toBeNull();
  });

  it("degrades corrupt or incomplete records to null", () => {
    localStorage.setItem("js-github-v1", "{not json");
    expect(loadGithubConnection()).toBeNull();
    localStorage.setItem("js-github-v1", JSON.stringify({ v: 1, token: "" }));
    expect(loadGithubConnection()).toBeNull();
    localStorage.setItem("js-github-v1", JSON.stringify({ v: 2, token: "x", login: "y" }));
    expect(loadGithubConnection()).toBeNull();
  });
});

/** A tiny fake GitHub: routes are matched by "METHOD path", in call order. */
function fakeGithub(routes: Record<string, () => Response>) {
  const calls: { method: string; path: string; body: unknown }[] = [];
  const fetchStub = vi.fn(async (url: RequestInfo | URL, init?: RequestInit) => {
    const path = String(url).replace("https://api.github.com", "");
    const method = init?.method ?? "GET";
    calls.push({
      method,
      path,
      body: init?.body ? JSON.parse(String(init.body)) : undefined,
    });
    const handler = routes[`${method} ${path}`];
    if (!handler) return new Response("{}", { status: 500 });
    return handler();
  });
  vi.stubGlobal("fetch", fetchStub);
  return calls;
}

const json = (status: number, body: unknown, headers?: Record<string, string>) =>
  new Response(JSON.stringify(body), { status, headers });

const CONN = { v: 1 as const, token: "t", login: "student", connectedAt: "" };
const FILES = [{ path: "README.md", contents: "# Hi" }];

describe("pushToRepo", () => {
  it("creates a missing repo, waits for its ref, and commits in sequence", async () => {
    const calls = fakeGithub({
      "GET /repos/student/demo": () => json(404, {}),
      "POST /user/repos": () => json(201, { default_branch: "main" }),
      "GET /repos/student/demo/git/ref/heads/main": () =>
        json(200, { object: { sha: "parent" } }),
      "GET /repos/student/demo/git/commits/parent": () =>
        json(200, { tree: { sha: "base" } }),
      "POST /repos/student/demo/git/trees": () => json(201, { sha: "tree1" }),
      "POST /repos/student/demo/git/commits": () => json(201, { sha: "commit1" }),
      "PATCH /repos/student/demo/git/refs/heads/main": () => json(200, {}),
    });

    const result = await pushToRepo(CONN, "demo", FILES, {
      message: "m",
      expectedRepoUrl: null,
    });
    expect(result).toEqual({
      ok: true,
      repoUrl: "https://github.com/student/demo",
      defaultBranch: "main",
    });
    expect(calls.map((c) => `${c.method} ${c.path.split("/git/")[0]}`)[0]).toBe(
      "GET /repos/student/demo",
    );
    const sequence = calls.map((c) => c.method);
    expect(sequence).toEqual(["GET", "POST", "GET", "GET", "GET", "POST", "POST", "PATCH"]);
    // The tree builds on the parent commit's tree, keeping unrelated files.
    const tree = calls.find((c) => c.path.endsWith("/git/trees"));
    expect((tree?.body as { base_tree: string }).base_tree).toBe("base");
  });

  it("updates only a repo it created; a stranger's repo gets a suggestion", async () => {
    fakeGithub({
      "GET /repos/student/demo": () =>
        json(200, {
          html_url: "https://github.com/student/demo",
          default_branch: "main",
        }),
      "GET /repos/student/demo-2": () => json(404, {}),
    });
    const stranger = await pushToRepo(CONN, "demo", FILES, {
      message: "m",
      expectedRepoUrl: null,
    });
    expect(stranger).toEqual({
      ok: false,
      error: { kind: "name-taken", suggestion: "demo-2" },
    });
  });

  it("pushes updates when the repo is the one it recorded", async () => {
    const calls = fakeGithub({
      "GET /repos/student/demo": () =>
        json(200, {
          html_url: "https://github.com/student/demo",
          default_branch: "main",
        }),
      "GET /repos/student/demo/git/ref/heads/main": () =>
        json(200, { object: { sha: "parent" } }),
      "GET /repos/student/demo/git/commits/parent": () =>
        json(200, { tree: { sha: "base" } }),
      "POST /repos/student/demo/git/trees": () => json(201, { sha: "tree1" }),
      "POST /repos/student/demo/git/commits": () => json(201, { sha: "commit1" }),
      "PATCH /repos/student/demo/git/refs/heads/main": () => json(200, {}),
    });
    const result = await pushToRepo(CONN, "demo", FILES, {
      message: "m",
      expectedRepoUrl: "https://github.com/student/demo",
    });
    expect(result.ok).toBe(true);
    expect(calls.some((c) => c.path === "/user/repos")).toBe(false);
  });

  it("maps provider failures to typed errors", async () => {
    fakeGithub({ "GET /repos/student/demo": () => json(401, {}) });
    expect(
      await pushToRepo(CONN, "demo", FILES, { message: "m", expectedRepoUrl: null }),
    ).toEqual({ ok: false, error: { kind: "auth" } });

    fakeGithub({
      "GET /repos/student/demo": () =>
        json(403, {}, { "x-ratelimit-remaining": "0", "x-ratelimit-reset": "1800000000" }),
    });
    const limited = await pushToRepo(CONN, "demo", FILES, {
      message: "m",
      expectedRepoUrl: null,
    });
    expect(!limited.ok && limited.error.kind).toBe("rate-limit");

    vi.stubGlobal(
      "fetch",
      vi.fn(async () => {
        throw new TypeError("offline");
      }),
    );
    expect(
      await pushToRepo(CONN, "demo", FILES, { message: "m", expectedRepoUrl: null }),
    ).toEqual({ ok: false, error: { kind: "network" } });
  });

  it("refuses oversized pushes before any network call", async () => {
    const fetchStub = vi.fn();
    vi.stubGlobal("fetch", fetchStub);
    const result = await pushToRepo(
      CONN,
      "demo",
      [{ path: "big.csv", contents: "x".repeat(500_000) }],
      { message: "m", expectedRepoUrl: null },
    );
    expect(result).toEqual({ ok: false, error: { kind: "too-large" } });
    expect(fetchStub).not.toHaveBeenCalled();
  });
});

describe("enablePages", () => {
  it("treats created and already-enabled as success", async () => {
    fakeGithub({ "POST /repos/student/portfolio/pages": () => json(201, {}) });
    expect(await enablePages(CONN, "portfolio", "main")).toEqual({
      ok: true,
      pagesUrl: "https://student.github.io/portfolio/",
    });
    fakeGithub({ "POST /repos/student/portfolio/pages": () => json(409, {}) });
    expect((await enablePages(CONN, "portfolio", "main")).ok).toBe(true);
  });

  it("falls back to the manual settings path on 403 (scope ceiling)", async () => {
    fakeGithub({ "POST /repos/student/portfolio/pages": () => json(403, {}) });
    expect(await enablePages(CONN, "portfolio", "main")).toEqual({
      ok: false,
      needsManual: true,
      settingsUrl: "https://github.com/student/portfolio/settings/pages",
    });
  });
});

describe("file-set assemblers", () => {
  it("scaffold: README plus the generated files", () => {
    const files = scaffoldFileSet({
      readme: "# R",
      files: [{ path: "src/a.py", contents: "pass" }],
    });
    expect(files.map((f) => f.path)).toEqual(["README.md", "src/a.py"]);
  });

  it("polish: lists never-persisted binaries instead of silently dropping them", () => {
    const files = polishFileSet({
      plan: {
        readme: "# R",
        gitignore: "data/",
        extraFiles: [{ path: "data/README.md", contents: "where" }],
      },
      textFiles: [{ path: "src/f.R", contents: "x" }],
      binaryPaths: ["docs/report.pdf"],
    });
    const note = files.find((f) => f.path === "ADD_THESE_FILES.md");
    expect(note?.contents).toContain("docs/report.pdf");
    expect(
      polishFileSet({
        plan: { readme: "", gitignore: "", extraFiles: [] },
        textFiles: [],
        binaryPaths: [],
      }).some((f) => f.path === "ADD_THESE_FILES.md"),
    ).toBe(false);
  });

  it("portfolio: index.html plus .nojekyll so Pages serves files verbatim", () => {
    expect(portfolioFileSet("<html/>").map((f) => f.path)).toEqual([
      "index.html",
      ".nojekyll",
      "README.md",
    ]);
  });
});

describe("portfolio site renderer", () => {
  const content: PortfolioContent = {
    siteTitle: "Portfolio",
    headline: 'Analyst <script>alert("x")</script>',
    about: "About & more",
    skillGroups: [{ title: "Data", skills: ["SQL <img onerror=1>"] }],
    projectCards: [
      {
        repoName: "demo",
        title: "Demo",
        blurb: "b",
        skillLabels: ["SQL"],
        repoUrl: "https://github.com/student/demo",
      },
    ],
    courseHighlights: [{ course: "ISA 245", why: "SQL" }],
  };
  const student = {
    name: "Kaitlin",
    links: [
      { label: "LinkedIn", url: "https://linkedin.com/in/k" },
      // A hostile link must render as nothing, not as an executable href.
      { label: "Evil", url: "javascript:alert(1)" },
    ],
  };

  it("escapes every interpolated string, so employer text cannot inject markup", () => {
    const html = renderPortfolioHtml(content, student);
    expect(html).not.toContain("<script>alert");
    expect(html).toContain("&lt;script&gt;");
    expect(html).not.toContain("<img onerror");
    expect(html).not.toContain("javascript:");
    expect(html).toContain("About &amp; more");
  });

  it("is deterministic: same content, same bytes", () => {
    expect(renderPortfolioHtml(content, student)).toBe(
      renderPortfolioHtml(content, student),
    );
  });

  it("escapeHtml covers the five HTML metacharacters", () => {
    expect(escapeHtml(`&<>"'`)).toBe("&amp;&lt;&gt;&quot;&#39;");
  });
});
