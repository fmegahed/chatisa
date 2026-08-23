import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  decodeOauthState,
  encodeOauthState,
  publicOrigin,
  safeReturnPath,
} from "@/lib/scout/github-state";
import {
  clearGithubConnection,
  loadGithubConnection,
  saveGithubConnection,
} from "@/lib/scout/github-store";
import {
  enablePages,
  pushFileBytes,
  pushToRepo,
  scaffoldFileSet,
} from "@/lib/scout/github";

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
    expect(safeReturnPath("//evil.example/phish")).toBe("/portfolio");
    expect(safeReturnPath("https://evil.example")).toBe("/portfolio");
    expect(safeReturnPath("/\\evil")).toBe("/portfolio");
    expect(safeReturnPath(null)).toBe("/portfolio");
  });
});

describe("publicOrigin", () => {
  const internal = (headers: Record<string, string> = {}) =>
    new Request("http://127.0.0.1:3000/api/scout/github/start", { headers });

  it("prefers AUTH_URL: the OAuth app is registered against the public origin", () => {
    expect(
      publicOrigin(internal({ "x-forwarded-host": "other.example" }), {
        AUTH_URL: "https://chatisa.fsb.miamioh.edu/",
      }),
    ).toBe("https://chatisa.fsb.miamioh.edu");
  });

  it("falls back to forwarded headers behind the TLS relay", () => {
    expect(
      publicOrigin(
        internal({
          "x-forwarded-proto": "https",
          "x-forwarded-host": "chatisa.fsb.miamioh.edu",
        }),
        {},
      ),
    ).toBe("https://chatisa.fsb.miamioh.edu");
  });

  it("uses the request origin in plain local development", () => {
    expect(
      publicOrigin(new Request("http://localhost:3000/api/x"), {}),
    ).toBe("http://localhost:3000");
  });

  it("ignores a malformed AUTH_URL and an empty forwarded host", () => {
    expect(
      publicOrigin(internal({ "x-forwarded-proto": "https", "x-forwarded-host": "" }), {
        AUTH_URL: "not a url",
      }),
    ).toBe("http://127.0.0.1:3000");
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

describe("pushToRepo hardening (2026-08-23)", () => {
  const happy = {
    "GET /repos/student/demo": () => json(404, {}),
    "POST /user/repos": () => json(201, { default_branch: "main" }),
    "GET /repos/student/demo/git/ref/heads/main": () => json(200, { object: { sha: "parent" } }),
    "GET /repos/student/demo/git/commits/parent": () => json(200, { tree: { sha: "base" } }),
    "POST /repos/student/demo/git/trees": () => json(201, { sha: "tree1" }),
    "POST /repos/student/demo/git/commits": () => json(201, { sha: "commit1" }),
    "PATCH /repos/student/demo/git/refs/heads/main": () => json(200, {}),
  };
  const binaries = [
    { path: "a.png", contents: "AAAA", encoding: "base64" as const },
    { path: "b.png", contents: "BBBB", encoding: "base64" as const },
    { path: "README.md", contents: "# Hi" },
  ];

  it("retries one blob upload that fails with a 5xx or a network error, then succeeds", async () => {
    // First blob: 502 then ok. Second blob: network throw then ok.
    let blobCalls = 0;
    const calls = fakeGithub({
      ...happy,
      "POST /repos/student/demo/git/blobs": () => {
        blobCalls++;
        if (blobCalls === 1) return json(502, {});
        if (blobCalls === 3) throw new TypeError("Failed to fetch");
        return json(201, { sha: `blob${blobCalls}` });
      },
    });
    const result = await pushToRepo(CONN, "demo", binaries, { message: "m", expectedRepoUrl: null, retryDelayMs: 0 });
    expect(result).toMatchObject({ ok: true });
    // Two blobs, each of which needed one retry: 502 then ok, throw then ok.
    expect(blobCalls).toBe(4);
    expect(calls.filter((c) => c.path.endsWith("/git/blobs"))).toHaveLength(4);
  });

  it("does not retry a 4xx", async () => {
    let blobCalls = 0;
    fakeGithub({
      ...happy,
      "POST /repos/student/demo/git/blobs": () => {
        blobCalls++;
        return json(422, {});
      },
    });
    const result = await pushToRepo(CONN, "demo", binaries, { message: "m", expectedRepoUrl: null, retryDelayMs: 0 });
    expect(result).toEqual({ ok: false, error: { kind: "github", status: 422 } });
    expect(blobCalls).toBe(1);
  });

  it("reads GitHub's secondary rate limit as a rate limit with its retry-after", async () => {
    fakeGithub({
      ...happy,
      "POST /repos/student/demo/git/blobs": () =>
        new Response(JSON.stringify({ message: "You have exceeded a secondary rate limit. Please wait a few minutes before you try again." }), {
          status: 403, headers: { "retry-after": "60" },
        }),
    });
    const result = await pushToRepo(CONN, "demo", binaries, { message: "m", expectedRepoUrl: null, retryDelayMs: 0 });
    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result.error.kind).toBe("rate-limit");
    if (result.error.kind !== "rate-limit") return;
    expect(result.error.resetAt).not.toBeNull();
    expect(new Date(result.error.resetAt as string).getTime()).toBeGreaterThan(Date.now() + 30_000);
  });

  it("reports progress per uploaded file and then the commit", async () => {
    fakeGithub({ ...happy, "POST /repos/student/demo/git/blobs": () => json(201, { sha: "b" }) });
    const seen: string[] = [];
    const result = await pushToRepo(CONN, "demo", binaries, {
      message: "m", expectedRepoUrl: null,
      onProgress: (p) => seen.push(p.stage === "upload" ? `${p.stage} ${p.done}/${p.total} ${p.path}` : p.stage),
    });
    expect(result.ok).toBe(true);
    expect(seen).toEqual(["prepare", "upload 1/2 a.png", "upload 2/2 b.png", "commit"]);
  });

  it("stops at the next request when aborted and says so", async () => {
    const controller = new AbortController();
    let blobCalls = 0;
    fakeGithub({
      ...happy,
      "POST /repos/student/demo/git/blobs": () => {
        blobCalls++;
        controller.abort();
        return json(201, { sha: "b" });
      },
    });
    const result = await pushToRepo(CONN, "demo", binaries, { message: "m", expectedRepoUrl: null, signal: controller.signal });
    expect(result).toEqual({ ok: false, error: { kind: "cancelled" } });
    expect(blobCalls).toBe(1);
  });
});

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
      Array.from({ length: 61 }, (_, i) => ({ path: `f${i}.txt`, contents: "x" })),
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
});

describe("push engine binary files", () => {
  it("counts decoded bytes for base64 files", () => {
    // "aGVsbG8=" is "hello": 5 bytes, not 8.
    expect(pushFileBytes({ path: "a.bin", contents: "aGVsbG8=", encoding: "base64" })).toBe(5);
    expect(pushFileBytes({ path: "a.txt", contents: "héllo" })).toBe(6);
  });

  it("uploads base64 files as blobs and references their sha in the tree", async () => {
    const calls: { method: string; path: string; body: unknown }[] = [];
    const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
      const path = new URL(url).pathname;
      const method = init?.method ?? "GET";
      const body = init?.body ? JSON.parse(String(init.body)) : undefined;
      calls.push({ method, path, body });
      const json = (status: number, data: unknown) =>
        new Response(JSON.stringify(data), { status, headers: { "content-type": "application/json" } });
      if (method === "GET" && path === "/repos/me/site") return json(404, {});
      if (method === "POST" && path === "/user/repos") return json(201, { default_branch: "main" });
      if (path.includes("/git/ref/heads/")) return json(200, { object: { sha: "p" } });
      if (path.includes("/git/commits/p")) return json(200, { tree: { sha: "b" } });
      if (path.endsWith("/git/blobs")) return json(201, { sha: "blob-sha" });
      if (path.endsWith("/git/trees")) return json(201, { sha: "t" });
      if (path.endsWith("/git/commits")) return json(201, { sha: "c" });
      if (path.includes("/git/refs/heads/")) return json(200, {});
      return json(500, { path });
    });
    vi.stubGlobal("fetch", fetchMock);
    try {
      const result = await pushToRepo(
        { v: 1, token: "t", login: "me", connectedAt: "" },
        "site",
        [
          { path: "index.html", contents: "<p>hi</p>" },
          { path: "assets/photo.jpg", contents: "aGVsbG8=", encoding: "base64" },
        ],
        { message: "m", expectedRepoUrl: null },
      );
      expect(result.ok).toBe(true);
      const blob = calls.find((c) => c.path.endsWith("/git/blobs"));
      expect(blob?.body).toEqual({ content: "aGVsbG8=", encoding: "base64" });
      const tree = calls.find((c) => c.path.endsWith("/git/trees"))?.body as {
        tree: { path: string; sha?: string; content?: string }[];
      };
      expect(tree.tree).toEqual([
        { path: "index.html", mode: "100644", type: "blob", content: "<p>hi</p>" },
        { path: "assets/photo.jpg", mode: "100644", type: "blob", sha: "blob-sha" },
      ]);
    } finally {
      vi.unstubAllGlobals();
    }
  });
});
